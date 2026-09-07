#include "transport/shm/mpi_shm.h"

#include "core/buffer.h"
#include "transport/simple/connection.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nano_nccl::transport::shm {

namespace {

constexpr std::size_t kNameBytes = 128;

#if defined(NANO_NCCL_TRANSPORT_TEST_FAULT_INJECTION)
bool inject_failure(const char* point, int global_rank) {
    const char* requested = std::getenv("NANO_NCCL_TEST_TRANSPORT_FAILURE");
    return global_rank == 0 && requested != nullptr &&
           std::strcmp(requested, point) == 0;
}
#endif

void mpi_check(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    throw std::runtime_error(std::string(operation) + ": " +
                             std::string(error, static_cast<std::size_t>(length)));
}

void report_cuda_cleanup(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s failed during SHM cleanup: %s\n", operation,
                 cudaGetErrorString(status));
}

void report_system_cleanup(int status, const char* operation) noexcept {
    if (status == 0) return;
    std::fprintf(stderr, "%s failed during SHM cleanup: %s\n", operation,
                 std::strerror(errno));
}

std::string gather_first_error(MPI_Comm control_comm,
                               const std::array<char, 512>& local_error) {
    std::vector<std::array<char, 512>> errors(kRanks);
    mpi_check(MPI_Allgather(local_error.data(),
                            static_cast<int>(local_error.size()), MPI_CHAR,
                            errors.data(), static_cast<int>(local_error.size()),
                            MPI_CHAR, control_comm),
              "MPI_Allgather(SHM errors)");
    for (const auto& error : errors) {
        if (error[0] != '\0') return error.data();
    }
    return "unknown SHM setup error";
}

class PosixShmOwner final {
public:
    explicit PosixShmOwner(int device) : device_(device) {
        mappings_.reserve(2);
    }

    ~PosixShmOwner() {
        if (!name_.empty()) unlink_region();
        if (!mappings_.empty()) {
            report_cuda_cleanup(cudaSetDevice(device_), "cudaSetDevice");
        }
        for (auto it = mappings_.rbegin(); it != mappings_.rend(); ++it) {
            if (it->registered) {
                report_cuda_cleanup(cudaHostUnregister(it->pointer),
                                    "cudaHostUnregister");
            }
            report_system_cleanup(
                munmap(it->pointer, simple::kConnectionRegionBytes), "munmap");
            if (it->fd >= 0) report_system_cleanup(close(it->fd), "close");
        }
    }

    PosixShmOwner(const PosixShmOwner&) = delete;
    PosixShmOwner& operator=(const PosixShmOwner&) = delete;

    void set_name(std::string name) { name_ = std::move(name); }

    void add_mapping(void* pointer, int fd) {
        mappings_.push_back({pointer, fd, false});
    }

    void mark_registered(void* pointer) {
        for (Mapping& mapping : mappings_) {
            if (mapping.pointer == pointer) {
                mapping.registered = true;
                return;
            }
        }
        throw std::logic_error("SHM registration does not belong to this owner");
    }

    void unlink_region() noexcept {
        if (name_.empty()) return;
        report_system_cleanup(shm_unlink(name_.c_str()), "shm_unlink");
        name_.clear();
    }

private:
    struct Mapping {
        void* pointer;
        int fd;
        bool registered;
    };

    int device_ = 0;
    std::string name_;
    std::vector<Mapping> mappings_;
};

std::pair<std::string, int> create_region_name(int global_rank) {
    static std::atomic<unsigned long> sequence{0};
    for (int attempt = 0; attempt < 32; ++attempt) {
        const std::string name =
            "/nano_nccl_" + std::to_string(static_cast<long long>(getpid())) +
            "_" + std::to_string(global_rank) + "_" +
            std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
        const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC,
                                S_IRUSR | S_IWUSR);
        if (fd >= 0) return {name, fd};
        if (errno != EEXIST) {
            throw std::system_error(errno, std::generic_category(), "shm_open(create)");
        }
    }
    throw std::runtime_error("could not allocate a unique SHM region name");
}

void* map_region(int fd) {
    void* pointer = mmap(nullptr, simple::kConnectionRegionBytes,
                         PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pointer == MAP_FAILED) {
        throw std::system_error(errno, std::generic_category(), "mmap(SHM)");
    }
    return pointer;
}

}  // namespace

ConnectionResources create_mpi_shm_connections(
    MPI_Comm control_comm, int global_rank, int device,
    const std::vector<TransportKind>& edge_kinds) {
    if (std::find(edge_kinds.begin(), edge_kinds.end(), TransportKind::Shm) ==
        edge_kinds.end()) {
        return {};
    }

    const int incoming_edge = (global_rank + kRanks - 1) % kRanks;
    const bool has_incoming = edge_kinds[incoming_edge] == TransportKind::Shm;
    const bool has_outgoing = edge_kinds[global_rank] == TransportKind::Shm;
    auto owner = std::make_unique<PosixShmOwner>(device);
    std::array<char, kNameBytes> local_name{};
    std::array<char, 512> local_error{};
    int local_ok = 1;
    void* incoming_host = nullptr;

    try {
        if (has_incoming) {
            auto [name, fd] = create_region_name(global_rank);
            owner->set_name(name);
            if (ftruncate(fd, static_cast<off_t>(simple::kConnectionRegionBytes)) != 0) {
                const int error = errno;
                close(fd);
                throw std::system_error(error, std::generic_category(),
                                        "ftruncate(SHM)");
            }
            try {
                incoming_host = map_region(fd);
                owner->add_mapping(incoming_host, fd);
            } catch (...) {
                close(fd);
                throw;
            }
#if defined(NANO_NCCL_TRANSPORT_TEST_FAULT_INJECTION)
            if (inject_failure("shm_allocation", global_rank)) {
                throw std::runtime_error("injected SHM allocation failure");
            }
#endif
            std::memset(incoming_host, 0, simple::kConnectionRegionBytes);
            std::snprintf(local_name.data(), local_name.size(), "%s", name.c_str());
        }
    } catch (const std::exception& error) {
        local_ok = 0;
        std::snprintf(local_error.data(), local_error.size(),
                      "SHM incoming edge %d->%d creation failed at global rank %d: %s",
                      incoming_edge, global_rank, global_rank, error.what());
    }

    int all_ok = 0;
    mpi_check(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN,
                            control_comm),
              "MPI_Allreduce(SHM creation)");
    if (all_ok == 0) {
        const std::string error = gather_first_error(control_comm, local_error);
        owner.reset();
        throw std::runtime_error("same-host SHM creation failed: " + error);
    }

    std::vector<std::array<char, kNameBytes>> names(kRanks);
    mpi_check(MPI_Allgather(local_name.data(), static_cast<int>(local_name.size()),
                            MPI_CHAR, names.data(),
                            static_cast<int>(local_name.size()), MPI_CHAR,
                            control_comm),
              "MPI_Allgather(SHM names)");

    local_error.fill('\0');
    void* outgoing_host = nullptr;
    void* incoming_device = nullptr;
    void* outgoing_device = nullptr;
    try {
        if (has_outgoing) {
            const int receiver = (global_rank + 1) % kRanks;
            const char* name = names[receiver].data();
            if (name[0] == '\0') {
                throw std::runtime_error("SHM receiver did not publish its Ring region");
            }
            const int fd = shm_open(name, O_RDWR | O_CLOEXEC, 0);
            if (fd < 0) {
                throw std::system_error(errno, std::generic_category(),
                                        "shm_open(peer)");
            }
            try {
                outgoing_host = map_region(fd);
                owner->add_mapping(outgoing_host, fd);
            } catch (...) {
                close(fd);
                throw;
            }
        }

        CUDA_CHECK_THROW(cudaSetDevice(device));
        for (void* pointer : {incoming_host, outgoing_host}) {
            if (pointer == nullptr) continue;
            CUDA_CHECK_THROW(cudaHostRegister(
                pointer, simple::kConnectionRegionBytes,
                cudaHostRegisterPortable | cudaHostRegisterMapped));
            owner->mark_registered(pointer);
#if defined(NANO_NCCL_TRANSPORT_TEST_FAULT_INJECTION)
            if (inject_failure("shm_registration", global_rank)) {
                throw std::runtime_error("injected SHM registration failure");
            }
#endif
        }
        if (has_incoming) {
            CUDA_CHECK_THROW(cudaHostGetDevicePointer(
                &incoming_device, incoming_host, 0));
        }
        if (has_outgoing) {
            CUDA_CHECK_THROW(cudaHostGetDevicePointer(
                &outgoing_device, outgoing_host, 0));
        }
    } catch (const std::exception& error) {
        local_ok = 0;
        std::snprintf(
            local_error.data(), local_error.size(),
            "SHM edge setup failed at global rank %d "
            "(incoming %d->%d, outgoing %d->%d): %s",
            global_rank, incoming_edge, global_rank, global_rank,
            (global_rank + 1) % kRanks, error.what());
    }

    mpi_check(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN,
                            control_comm),
              "MPI_Allreduce(SHM mapping)");
    if (all_ok == 0) {
        const std::string error = gather_first_error(control_comm, local_error);
        owner.reset();
        throw std::runtime_error("same-host SHM CUDA mapping failed: " + error);
    }

    mpi_check(MPI_Barrier(control_comm), "MPI_Barrier(SHM initialize)");
    owner->unlink_region();

    ConnectionResources resources;
    for (int channel = 0; channel < kChannels; ++channel) {
        if (has_incoming) {
            resources.view.recv_fifo[channel] =
                simple::connection_fifo(incoming_device, channel);
            resources.view.recv_head[channel] =
                simple::connection_head(incoming_device, channel);
            resources.view.recv_tail[channel] =
                simple::connection_tail(incoming_device, channel);
        }
        if (has_outgoing) {
            resources.view.send_fifo[channel] =
                simple::connection_fifo(outgoing_device, channel);
            resources.view.send_head[channel] =
                simple::connection_head(outgoing_device, channel);
            resources.view.send_tail[channel] =
                simple::connection_tail(outgoing_device, channel);
        }
    }
    resources.owners.push_back(erase_connection_owner(std::move(owner)));
    return resources;
}

}  // namespace nano_nccl::transport::shm
