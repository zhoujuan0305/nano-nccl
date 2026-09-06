#include "nano_nccl/mpi_c_api.h"

#include "c_api_internal.h"
#include "nano_nccl/mpi.h"
#include "nano_nccl/types.h"

#include <memory>
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
        case NANO_NCCL_TRANSPORT_SOCKET:
            return nano_nccl::TransportKind::Socket;
        case NANO_NCCL_TRANSPORT_RDMA:
            return nano_nccl::TransportKind::Rdma;
        default:
            throw std::invalid_argument(
                "MPI subgroup transport must be auto, socket, or rdma");
    }
}

void release_mpi(MPI_Comm communicator, bool owns_mpi) noexcept {
    if (communicator != MPI_COMM_NULL) MPI_Comm_free(&communicator);
    if (!owns_mpi) return;
    int finalized = 0;
    if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0) MPI_Finalize();
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
        int initialized = 0;
        mpi_check(MPI_Initialized(&initialized), "MPI_Initialized");
        const bool owns_mpi = initialized == 0;
        if (owns_mpi) {
            int provided = MPI_THREAD_SINGLE;
            mpi_check(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED,
                                      &provided),
                      "MPI_Init_thread");
            if (provided < MPI_THREAD_FUNNELED) {
                MPI_Finalize();
                throw std::runtime_error(
                    "MPI does not provide MPI_THREAD_FUNNELED");
            }
        }

        MPI_Comm control_comm = MPI_COMM_NULL;
        try {
            mpi_check(MPI_Comm_split(MPI_COMM_WORLD, config->color, config->key,
                                     &control_comm),
                      "MPI_Comm_split");
            nano_nccl::CommunicatorConfig cpp_config;
            cpp_config.devices = {config->device};
            cpp_config.transport = to_cpp_transport(config->transport);
            auto owner = std::make_unique<NanoNcclCommunicator>();
            owner->communicator = nano_nccl::create_communicator_from_mpi(
                control_comm, cpp_config);
            owner->cleanup = [control_comm, owns_mpi] {
                release_mpi(control_comm, owns_mpi);
            };
            *output = owner.release();
        } catch (...) {
            release_mpi(control_comm, owns_mpi);
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
