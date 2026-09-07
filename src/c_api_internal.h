#pragma once

#include "nano_nccl/communicator.h"
#include "nano_nccl/nano_nccl.h"

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

struct NanoNcclCommunicator {
    std::unique_ptr<nano_nccl::Communicator> communicator;
    std::function<void()> before_destroy;
    std::function<void()> cleanup;

    ~NanoNcclCommunicator() {
        // Distributed bootstrap state must outlive communicator proxy threads.
        communicator.reset();
        if (cleanup) cleanup();
    }
};

namespace nano_nccl::c_api {

extern thread_local char last_error[1024];

nano_nccl_status_t fail(nano_nccl_status_t status,
                        const char* message) noexcept;
nano_nccl_status_t fail(nano_nccl_status_t status,
                        const std::exception& error) noexcept;
void begin_call() noexcept;

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

}  // namespace nano_nccl::c_api
