#include "nano_nccl/mpi_c_api.h"

#include "c_api_internal.h"
#include "nano_nccl/mpi.h"
#include "nano_nccl/types.h"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <mpi.h>

namespace {

using nano_nccl::c_api::begin_call;
using nano_nccl::c_api::fail;
using nano_nccl::c_api::translate_exceptions;

void mpi_check(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(message, static_cast<size_t>(length)));
}

nano_nccl::TransportKind to_cpp_transport(nano_nccl_transport_t transport) {
    switch (transport) {
        case NANO_NCCL_TRANSPORT_AUTO:
            return nano_nccl::TransportKind::Auto;
        case NANO_NCCL_TRANSPORT_SHM:
            return nano_nccl::TransportKind::Shm;
        case NANO_NCCL_TRANSPORT_P2P:
            return nano_nccl::TransportKind::P2p;
        case NANO_NCCL_TRANSPORT_SOCKET:
            return nano_nccl::TransportKind::Socket;
        case NANO_NCCL_TRANSPORT_RDMA:
            return nano_nccl::TransportKind::Rdma;
        default:
            throw std::invalid_argument(
                "MPI subgroup transport must be auto, shm, p2p, socket, or rdma");
    }
}

struct MpiRuntimeState {
    std::mutex mutex;
    std::size_t leases = 0;
    bool initialized_by_adapter = false;
};

MpiRuntimeState& mpi_runtime_state() {
    static MpiRuntimeState state;
    return state;
}

void acquire_mpi() {
    MpiRuntimeState& state = mpi_runtime_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    int initialized = 0;
    mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
    if (initialized == 0) {
        int provided = MPI_THREAD_SINGLE;
        mpi_check(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED,
                                  &provided),
                  "MPI_Init_thread");
        if (provided < MPI_THREAD_FUNNELED) {
            MPI_Finalize();
            throw std::runtime_error(
                "MPI does not provide MPI_THREAD_FUNNELED");
        }
        state.initialized_by_adapter = true;
    } else {
        int finalized = 0;
        mpi_check(MPI_Finalized(&finalized), "MPI_Finalized");
        if (finalized != 0) {
            throw std::runtime_error("MPI has already been finalized");
        }
    }
    ++state.leases;
}

void release_mpi(MPI_Comm communicator, bool has_lease) noexcept {
    int finalized = 0;
    const bool mpi_active =
        MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0;
    if (mpi_active && communicator != MPI_COMM_NULL) {
        MPI_Comm_free(&communicator);
    }
    if (!has_lease) return;

    MpiRuntimeState& state = mpi_runtime_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.leases > 0) --state.leases;
    if (mpi_active && state.initialized_by_adapter && state.leases == 0) {
        MPI_Finalize();
        state.initialized_by_adapter = false;
    }
}

}  // namespace

extern "C" nano_nccl_status_t nano_nccl_create_mpi_subgroup_communicator(
    const nano_nccl_mpi_subgroup_config_t* config,
    nano_nccl_communicator_t** output) {
    begin_call();
    if (output == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "communicator output must be non-null");
    }
    *output = nullptr;
    if (config == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "MPI subgroup config must be non-null");
    }
    if (config->color < 0 || config->key < 0) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "MPI subgroup color and key must be non-negative");
    }

    return translate_exceptions([&] {
        acquire_mpi();
        bool has_lease = true;

        MPI_Comm control_comm = MPI_COMM_NULL;
        try {
            mpi_check(MPI_Comm_split(MPI_COMM_WORLD, config->color, config->key,
                                     &control_comm),
                      "MPI_Comm_split");
            nano_nccl::CommunicatorConfig cpp_config;
            cpp_config.device = config->device;
            cpp_config.transport = to_cpp_transport(config->transport);
            auto owner = std::make_unique<NanoNcclCommunicator>();
            owner->communicator = nano_nccl::create_communicator_from_mpi(
                control_comm, cpp_config);
            owner->cleanup = [control_comm, has_lease] {
                release_mpi(control_comm, has_lease);
            };
            *output = owner.release();
            has_lease = false;
        } catch (...) {
            release_mpi(control_comm, has_lease);
            throw;
        }
    });
}

extern "C" nano_nccl_status_t nano_nccl_mpi_channel_count(
    int* channel_count) {
    begin_call();
    if (channel_count == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "channel_count output must be non-null");
    }
    *channel_count = nano_nccl::kChannels;
    return NANO_NCCL_STATUS_SUCCESS;
}
