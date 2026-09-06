#include "nano_nccl/nano_nccl.h"

#include "nano_nccl/communicator.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

struct NanoNcclCommunicator {
    std::unique_ptr<nano_nccl::Communicator> communicator;
};

namespace {

thread_local char g_last_error[1024]{};

nano_nccl_status_t fail(nano_nccl_status_t status, const char* message) noexcept {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
    return status;
}

nano_nccl_status_t fail(nano_nccl_status_t status,
                        const std::exception& error) noexcept {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", error.what());
    return status;
}

void begin_call() noexcept { g_last_error[0] = '\0'; }

template <typename Function>
nano_nccl_status_t translate_exceptions(Function&& function) noexcept {
    try {
        std::forward<Function>(function)();
        return NANO_NCCL_STATUS_SUCCESS;
    } catch (const std::invalid_argument& error) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT, error);
    } catch (const std::exception& error) {
        return fail(NANO_NCCL_STATUS_ERROR, error);
    } catch (...) {
        return fail(NANO_NCCL_STATUS_ERROR, "unknown C++ exception");
    }
}

bool to_cpp_dtype(nano_nccl_dtype_t value, nano_nccl::DType* dtype) {
    switch (value) {
        case NANO_NCCL_DTYPE_FLOAT:
            *dtype = nano_nccl::DType::Float;
            return true;
        case NANO_NCCL_DTYPE_FLOAT16:
            *dtype = nano_nccl::DType::Float16;
            return true;
        case NANO_NCCL_DTYPE_BFLOAT16:
            *dtype = nano_nccl::DType::BFloat16;
            return true;
        default:
            return false;
    }
}

bool to_cpp_redop(nano_nccl_redop_t value, nano_nccl::RedOp* redop) {
    switch (value) {
        case NANO_NCCL_REDOP_SUM:
            *redop = nano_nccl::RedOp::Sum;
            return true;
        case NANO_NCCL_REDOP_AVG:
            *redop = nano_nccl::RedOp::Avg;
            return true;
        case NANO_NCCL_REDOP_MAX:
            *redop = nano_nccl::RedOp::Max;
            return true;
        case NANO_NCCL_REDOP_MIN:
            *redop = nano_nccl::RedOp::Min;
            return true;
        default:
            return false;
    }
}

bool to_cpp_transport(nano_nccl_transport_t value,
                      nano_nccl::TransportKind* transport) {
    switch (value) {
        case NANO_NCCL_TRANSPORT_AUTO:
            *transport = nano_nccl::TransportKind::Auto;
            return true;
        case NANO_NCCL_TRANSPORT_SHM:
            *transport = nano_nccl::TransportKind::Shm;
            return true;
        case NANO_NCCL_TRANSPORT_P2P:
            *transport = nano_nccl::TransportKind::P2p;
            return true;
        case NANO_NCCL_TRANSPORT_SOCKET:
            *transport = nano_nccl::TransportKind::Socket;
            return true;
        case NANO_NCCL_TRANSPORT_RDMA:
            *transport = nano_nccl::TransportKind::Rdma;
            return true;
        case NANO_NCCL_TRANSPORT_MIXED:
            *transport = nano_nccl::TransportKind::Mixed;
            return true;
        default:
            return false;
    }
}

nano_nccl_transport_t from_cpp_transport(nano_nccl::TransportKind transport) {
    switch (transport) {
        case nano_nccl::TransportKind::Auto:
            return NANO_NCCL_TRANSPORT_AUTO;
        case nano_nccl::TransportKind::Shm:
            return NANO_NCCL_TRANSPORT_SHM;
        case nano_nccl::TransportKind::P2p:
            return NANO_NCCL_TRANSPORT_P2P;
        case nano_nccl::TransportKind::Socket:
            return NANO_NCCL_TRANSPORT_SOCKET;
        case nano_nccl::TransportKind::Rdma:
            return NANO_NCCL_TRANSPORT_RDMA;
        case nano_nccl::TransportKind::Mixed:
            return NANO_NCCL_TRANSPORT_MIXED;
    }
    return NANO_NCCL_TRANSPORT_AUTO;
}

nano_nccl_status_t validate_handle(const nano_nccl_communicator_t* communicator) {
    if (communicator == nullptr || communicator->communicator == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "communicator must be non-null");
    }
    return NANO_NCCL_STATUS_SUCCESS;
}

nano_nccl_status_t validate_buffer_args(
    const nano_nccl_communicator_t* communicator,
    const void* const* send_buffers, void* const* recv_buffers,
    const cudaStream_t* streams, std::size_t count) {
    if (send_buffers == nullptr || recv_buffers == nullptr || streams == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "collective buffer and stream arrays must be non-null");
    }
    if (count == 0) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "collective count must be positive");
    }
    const int rank_count = communicator->communicator->local_rank_count();
    for (int rank = 0; rank < rank_count; ++rank) {
        if (send_buffers[rank] == nullptr || recv_buffers[rank] == nullptr ||
            streams[rank] == nullptr) {
            return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                        "collective buffers and streams must be non-null");
        }
        if (send_buffers[rank] == recv_buffers[rank]) {
            return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                        "in-place collectives are unsupported");
        }
    }
    return NANO_NCCL_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

uint32_t nano_nccl_abi_version(void) { return NANO_NCCL_ABI_VERSION; }

const char* nano_nccl_status_string(nano_nccl_status_t status) {
    switch (status) {
        case NANO_NCCL_STATUS_SUCCESS:
            return "success";
        case NANO_NCCL_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case NANO_NCCL_STATUS_UNSUPPORTED:
            return "unsupported";
        case NANO_NCCL_STATUS_ERROR:
            return "error";
        default:
            return "unknown status";
    }
}

const char* nano_nccl_get_last_error(void) { return g_last_error; }

nano_nccl_status_t nano_nccl_create_communicator(
    const nano_nccl_communicator_config_t* config,
    nano_nccl_communicator_t** communicator) {
    begin_call();
    if (communicator == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "communicator output must be non-null");
    }
    *communicator = nullptr;
    if (config == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT, "config must be non-null");
    }
    if (config->devices == nullptr || config->device_count == 0) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "config devices must be non-null and non-empty");
    }
    nano_nccl::TransportKind transport;
    if (!to_cpp_transport(config->transport, &transport)) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "unsupported transport value");
    }

    return translate_exceptions([&] {
        nano_nccl::CommunicatorConfig cpp_config;
        cpp_config.devices.assign(config->devices,
                                  config->devices + config->device_count);
        cpp_config.transport = transport;
        auto owner = std::make_unique<nano_nccl_communicator_t>();
        owner->communicator = nano_nccl::create_communicator(cpp_config);
        *communicator = owner.release();
    });
}

nano_nccl_status_t nano_nccl_destroy_communicator(
    nano_nccl_communicator_t* communicator) {
    begin_call();
    if (communicator == nullptr) {
        return NANO_NCCL_STATUS_SUCCESS;
    }
    return translate_exceptions([&] { delete communicator; });
}

nano_nccl_status_t nano_nccl_all_reduce(
    nano_nccl_communicator_t* communicator,
    const nano_nccl_all_reduce_args_t* args) {
    begin_call();
    nano_nccl_status_t status = validate_handle(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) return status;
    if (args == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "all_reduce args must be non-null");
    }

    status = validate_buffer_args(communicator, args->send_buffers,
                                  args->recv_buffers, args->streams, args->count);
    if (status != NANO_NCCL_STATUS_SUCCESS) return status;
    nano_nccl::DType dtype;
    nano_nccl::RedOp redop;
    if (!to_cpp_dtype(args->dtype, &dtype)) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT, "unsupported dtype value");
    }
    if (!to_cpp_redop(args->redop, &redop)) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "unsupported reduction operation value");
    }

    return translate_exceptions([&] {
        const int rank_count = communicator->communicator->local_rank_count();
        nano_nccl::CollectiveArgs cpp_args{
            std::vector<const void*>(args->send_buffers,
                                     args->send_buffers + rank_count),
            std::vector<void*>(args->recv_buffers,
                               args->recv_buffers + rank_count),
            std::vector<cudaStream_t>(args->streams, args->streams + rank_count),
            args->count,
            dtype,
            redop,
        };
        communicator->communicator->all_reduce(cpp_args);
    });
}

nano_nccl_status_t nano_nccl_reduce_scatter(
    nano_nccl_communicator_t* communicator,
    const nano_nccl_reduce_scatter_args_t* args) {
    begin_call();
    nano_nccl_status_t status = validate_handle(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) return status;
    if (args == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "reduce_scatter args must be non-null");
    }
    status = validate_buffer_args(communicator, args->send_buffers,
                                  args->recv_buffers, args->streams,
                                  args->recv_count);
    if (status != NANO_NCCL_STATUS_SUCCESS) return status;
    nano_nccl::DType dtype;
    nano_nccl::RedOp redop;
    if (!to_cpp_dtype(args->dtype, &dtype)) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT, "unsupported dtype value");
    }
    if (!to_cpp_redop(args->redop, &redop)) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "unsupported reduction operation value");
    }
    return fail(NANO_NCCL_STATUS_UNSUPPORTED,
                "reduce_scatter is unsupported by the current implementation");
}

nano_nccl_status_t nano_nccl_all_gather(
    nano_nccl_communicator_t* communicator,
    const nano_nccl_all_gather_args_t* args) {
    begin_call();
    nano_nccl_status_t status = validate_handle(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) return status;
    if (args == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "all_gather args must be non-null");
    }
    status = validate_buffer_args(communicator, args->send_buffers,
                                  args->recv_buffers, args->streams,
                                  args->send_count);
    if (status != NANO_NCCL_STATUS_SUCCESS) return status;
    nano_nccl::DType dtype;
    if (!to_cpp_dtype(args->dtype, &dtype)) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT, "unsupported dtype value");
    }
    return fail(NANO_NCCL_STATUS_UNSUPPORTED,
                "all_gather is unsupported by the current implementation");
}

nano_nccl_status_t nano_nccl_check_async_error(
    const nano_nccl_communicator_t* communicator) {
    begin_call();
    nano_nccl_status_t status = validate_handle(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) return status;
    return translate_exceptions(
        [&] { communicator->communicator->check_async_error(); });
}

nano_nccl_status_t nano_nccl_local_rank_count(
    const nano_nccl_communicator_t* communicator, int* rank_count) {
    begin_call();
    nano_nccl_status_t status = validate_handle(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) return status;
    if (rank_count == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "rank_count output must be non-null");
    }
    *rank_count = communicator->communicator->local_rank_count();
    return NANO_NCCL_STATUS_SUCCESS;
}

nano_nccl_status_t nano_nccl_global_rank_count(
    const nano_nccl_communicator_t* communicator, int* rank_count) {
    begin_call();
    nano_nccl_status_t status = validate_handle(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) return status;
    if (rank_count == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "rank_count output must be non-null");
    }
    *rank_count = communicator->communicator->global_rank_count();
    return NANO_NCCL_STATUS_SUCCESS;
}

nano_nccl_status_t nano_nccl_transport(
    const nano_nccl_communicator_t* communicator,
    nano_nccl_transport_t* transport) {
    begin_call();
    nano_nccl_status_t status = validate_handle(communicator);
    if (status != NANO_NCCL_STATUS_SUCCESS) return status;
    if (transport == nullptr) {
        return fail(NANO_NCCL_STATUS_INVALID_ARGUMENT,
                    "transport output must be non-null");
    }
    *transport = from_cpp_transport(communicator->communicator->transport());
    return NANO_NCCL_STATUS_SUCCESS;
}

}  // extern "C"
