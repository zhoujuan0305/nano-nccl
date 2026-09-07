#include "transport/p2p/mpi_p2p.h"

#include "core/buffer.h"
#include "transport/p2p/p2p_topology.h"
#include "transport/simple/connection.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nano_nccl::transport::p2p {

namespace {

constexpr std::size_t kControlNameBytes = 128;
constexpr std::size_t kPciBusIdBytes = 32;
constexpr std::chrono::seconds kCloseTimeout{2};

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
    std::fprintf(stderr, "%s failed during CUDA IPC cleanup: %s\n", operation,
                 cudaGetErrorString(status));
}

void report_system_cleanup(int status, const char* operation) noexcept {
    if (status == 0) return;
    std::fprintf(stderr, "%s failed during CUDA IPC cleanup: %s\n", operation,
                 std::strerror(errno));
}

std::string gather_first_error(MPI_Comm control_comm,
                               const std::array<char, 512>& local_error) {
    std::vector<std::array<char, 512>> errors(kRanks);
    mpi_check(MPI_Allgather(local_error.data(),
                            static_cast<int>(local_error.size()), MPI_CHAR,
                            errors.data(), static_cast<int>(local_error.size()),
                            MPI_CHAR, control_comm),
              "MPI_Allgather(CUDA IPC errors)");
    for (const auto& error : errors) {
        if (error[0] != '\0') return error.data();
    }
    return "unknown CUDA IPC setup error";
}

class MpiP2pOwner final {
public:
    explicit MpiP2pOwner(int device) : device_(device) {}

    ~MpiP2pOwner() {
        report_cuda_cleanup(cudaSetDevice(device_), "cudaSetDevice");
        if (remote_region_ != nullptr) {
            const cudaError_t status = cudaIpcCloseMemHandle(remote_region_);
            report_cuda_cleanup(status, "cudaIpcCloseMemHandle");
            if (status == cudaSuccess && remote_close_control_ != nullptr) {
                __atomic_store_n(remote_close_control_, 1U, __ATOMIC_RELEASE);
            }
            remote_region_ = nullptr;
        }

        bool sender_closed = !wait_for_close_;
        if (wait_for_close_ && local_close_control_ != nullptr) {
            const auto deadline = std::chrono::steady_clock::now() + kCloseTimeout;
            while (__atomic_load_n(local_close_control_, __ATOMIC_ACQUIRE) == 0U &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            sender_closed =
                __atomic_load_n(local_close_control_, __ATOMIC_ACQUIRE) != 0U;
        }
        if (local_region_ != nullptr && sender_closed) {
            report_cuda_cleanup(cudaFree(local_region_), "cudaFree");
            local_region_ = nullptr;
        } else if (local_region_ != nullptr) {
            std::fprintf(
                stderr,
                "CUDA IPC peer did not close within timeout; leaking exported "
                "region during teardown\n");
        }

        unmap_control(remote_close_control_, remote_control_fd_);
        unmap_control(local_close_control_, local_control_fd_);
        if (!local_control_name_.empty()) {
            report_system_cleanup(shm_unlink(local_control_name_.c_str()),
                                  "shm_unlink");
        }
    }

    MpiP2pOwner(const MpiP2pOwner&) = delete;
    MpiP2pOwner& operator=(const MpiP2pOwner&) = delete;

    void set_local_region(void* pointer) noexcept { local_region_ = pointer; }
    void set_remote_region(void* pointer) noexcept { remote_region_ = pointer; }

    void set_local_control(std::string name, std::uint32_t* pointer, int fd) {
        local_control_name_ = std::move(name);
        local_close_control_ = pointer;
        local_control_fd_ = fd;
    }

    void set_remote_control(std::uint32_t* pointer, int fd) noexcept {
        remote_close_control_ = pointer;
        remote_control_fd_ = fd;
    }

    void arm_local_close_wait(bool wait_for_close = true) noexcept {
        wait_for_close_ = wait_for_close;
    }

    void unlink_local_control() noexcept {
        if (local_control_name_.empty()) return;
        report_system_cleanup(shm_unlink(local_control_name_.c_str()),
                              "shm_unlink");
        local_control_name_.clear();
    }

private:
    static void unmap_control(std::uint32_t*& pointer, int& fd) noexcept {
        if (pointer != nullptr) {
            report_system_cleanup(munmap(pointer, sizeof(std::uint32_t)),
                                  "munmap");
            pointer = nullptr;
        }
        if (fd >= 0) {
            report_system_cleanup(close(fd), "close");
            fd = -1;
        }
    }

    int device_ = 0;
    void* local_region_ = nullptr;
    void* remote_region_ = nullptr;
    std::string local_control_name_;
    std::uint32_t* local_close_control_ = nullptr;
    std::uint32_t* remote_close_control_ = nullptr;
    int local_control_fd_ = -1;
    int remote_control_fd_ = -1;
    bool wait_for_close_ = false;
};

std::tuple<std::string, std::uint32_t*, int> create_close_control(
    int global_rank) {
    static std::atomic<unsigned long> sequence{0};
    for (int attempt = 0; attempt < 32; ++attempt) {
        const std::string name =
            "/nano_nccl_ipc_" +
            std::to_string(static_cast<long long>(getpid())) + "_" +
            std::to_string(global_rank) + "_" +
            std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
        const int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC,
                                S_IRUSR | S_IWUSR);
        if (fd < 0) {
            if (errno == EEXIST) continue;
            throw std::system_error(errno, std::generic_category(),
                                    "shm_open(CUDA IPC control)");
        }
        if (ftruncate(fd, static_cast<off_t>(sizeof(std::uint32_t))) != 0) {
            const int error = errno;
            close(fd);
            shm_unlink(name.c_str());
            throw std::system_error(error, std::generic_category(),
                                    "ftruncate(CUDA IPC control)");
        }
        void* mapping = mmap(nullptr, sizeof(std::uint32_t),
                             PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapping == MAP_FAILED) {
            const int error = errno;
            close(fd);
            shm_unlink(name.c_str());
            throw std::system_error(error, std::generic_category(),
                                    "mmap(CUDA IPC control)");
        }
        auto* control = static_cast<std::uint32_t*>(mapping);
        __atomic_store_n(control, 0U, __ATOMIC_RELAXED);
        return {name, control, fd};
    }
    throw std::runtime_error("could not allocate a unique CUDA IPC control name");
}

std::pair<std::uint32_t*, int> open_close_control(const char* name) {
    const int fd = shm_open(name, O_RDWR | O_CLOEXEC, 0);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category(),
                                "shm_open(CUDA IPC peer control)");
    }
    void* mapping = mmap(nullptr, sizeof(std::uint32_t),
                         PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        const int error = errno;
        close(fd);
        throw std::system_error(error, std::generic_category(),
                                "mmap(CUDA IPC peer control)");
    }
    return {static_cast<std::uint32_t*>(mapping), fd};
}

}  // namespace

std::vector<bool> probe_mpi_p2p_edges(
    MPI_Comm control_comm, int global_rank, int device,
    const std::vector<int>& node_ids) {
    CUDA_CHECK_THROW(cudaSetDevice(device));
    void* local_probe = nullptr;
    cudaIpcMemHandle_t local_handle{};
    std::array<char, kPciBusIdBytes> local_pci_bus_id{};
    int local_handle_ok = 1;
    cudaError_t cuda_status = cudaMalloc(&local_probe, 1);
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaIpcGetMemHandle(&local_handle, local_probe);
    }
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaDeviceGetPCIBusId(
            local_pci_bus_id.data(),
            static_cast<int>(local_pci_bus_id.size()), device);
    }
    if (cuda_status != cudaSuccess) {
        local_handle_ok = 0;
        cudaGetLastError();
    }

    std::vector<cudaIpcMemHandle_t> handles(kRanks);
    std::vector<int> handle_ok(kRanks);
    std::vector<std::array<char, kPciBusIdBytes>> pci_bus_ids(kRanks);
    mpi_check(MPI_Allgather(&local_handle, sizeof(local_handle), MPI_BYTE,
                            handles.data(), sizeof(local_handle), MPI_BYTE,
                            control_comm),
              "MPI_Allgather(CUDA IPC probe handles)");
    mpi_check(MPI_Allgather(&local_handle_ok, 1, MPI_INT, handle_ok.data(), 1,
                            MPI_INT, control_comm),
              "MPI_Allgather(CUDA IPC probe status)");
    mpi_check(MPI_Allgather(
                  local_pci_bus_id.data(),
                  static_cast<int>(local_pci_bus_id.size()), MPI_CHAR,
                  pci_bus_ids.data(),
                  static_cast<int>(local_pci_bus_id.size()), MPI_CHAR,
                  control_comm),
              "MPI_Allgather(CUDA PCI bus ids)");

    const int receiver = (global_rank + 1) % kRanks;
    int outgoing_ok = 0;
    if (node_ids[global_rank] == node_ids[receiver] &&
        local_handle_ok != 0 && handle_ok[receiver] != 0 &&
        has_bidirectional_native_atomics(
            pci_bus_ids[global_rank].data(), pci_bus_ids[receiver].data())) {
        void* remote_probe = nullptr;
        cuda_status = cudaIpcOpenMemHandle(
            &remote_probe, handles[receiver], cudaIpcMemLazyEnablePeerAccess);
        if (cuda_status == cudaSuccess) {
            cuda_status = cudaIpcCloseMemHandle(remote_probe);
            outgoing_ok = cuda_status == cudaSuccess ? 1 : 0;
            if (cuda_status != cudaSuccess) cudaGetLastError();
        } else {
            cudaGetLastError();
        }
    }
    mpi_check(MPI_Barrier(control_comm), "MPI_Barrier(CUDA IPC probe)");
    if (local_probe != nullptr && cudaFree(local_probe) != cudaSuccess) {
        outgoing_ok = 0;
        cudaGetLastError();
    }

    std::vector<int> edge_ok(kRanks);
    mpi_check(MPI_Allgather(&outgoing_ok, 1, MPI_INT, edge_ok.data(), 1,
                            MPI_INT, control_comm),
              "MPI_Allgather(CUDA IPC edge status)");
    std::vector<bool> result(kRanks);
    for (int edge = 0; edge < kRanks; ++edge) result[edge] = edge_ok[edge] != 0;
    return result;
}

ConnectionResources create_mpi_p2p_connections(
    MPI_Comm control_comm, int global_rank, int device,
    const std::vector<TransportKind>& edge_kinds) {
    if (std::find(edge_kinds.begin(), edge_kinds.end(), TransportKind::P2p) ==
        edge_kinds.end()) {
        return {};
    }

    auto owner = std::make_unique<MpiP2pOwner>(device);
    CUDA_CHECK_THROW(cudaSetDevice(device));
    const int incoming_edge = (global_rank + kRanks - 1) % kRanks;
    const bool has_incoming = edge_kinds[incoming_edge] == TransportKind::P2p;
    const bool has_outgoing = edge_kinds[global_rank] == TransportKind::P2p;
    void* incoming_region = nullptr;
    cudaIpcMemHandle_t local_handle{};
    std::array<char, kControlNameBytes> local_control_name{};
    int local_ok = 1;
    std::array<char, 512> local_error{};

    try {
        if (has_incoming) {
            CUDA_CHECK_THROW(cudaMalloc(&incoming_region,
                                        simple::kConnectionRegionBytes));
            owner->set_local_region(incoming_region);
#if defined(NANO_NCCL_TRANSPORT_TEST_FAULT_INJECTION)
            if (inject_failure("p2p_allocation", global_rank)) {
                throw std::runtime_error("injected P2P allocation failure");
            }
#endif
            CUDA_CHECK_THROW(cudaMemset(incoming_region, 0,
                                        simple::kConnectionRegionBytes));
            CUDA_CHECK_THROW(cudaIpcGetMemHandle(&local_handle, incoming_region));
            auto [name, control, fd] = create_close_control(global_rank);
            owner->set_local_control(name, control, fd);
            std::snprintf(local_control_name.data(), local_control_name.size(),
                          "%s", name.c_str());
        }
    } catch (const std::exception& error) {
        local_ok = 0;
        std::snprintf(
            local_error.data(), local_error.size(),
            "P2P incoming edge %d->%d allocation/export failed at global rank %d: %s",
            incoming_edge, global_rank, global_rank, error.what());
    }

    std::vector<cudaIpcMemHandle_t> handles(kRanks);
    std::vector<std::array<char, kControlNameBytes>> control_names(kRanks);
    mpi_check(MPI_Allgather(&local_handle, sizeof(local_handle), MPI_BYTE,
                            handles.data(), sizeof(local_handle), MPI_BYTE,
                            control_comm),
              "MPI_Allgather(CUDA IPC handles)");
    mpi_check(MPI_Allgather(local_control_name.data(),
                            static_cast<int>(local_control_name.size()), MPI_CHAR,
                            control_names.data(),
                            static_cast<int>(local_control_name.size()), MPI_CHAR,
                            control_comm),
              "MPI_Allgather(CUDA IPC close controls)");
    int all_ok = 0;
    mpi_check(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN,
                            control_comm),
              "MPI_Allreduce(CUDA IPC allocation)");
    if (all_ok == 0) {
        const std::string error = gather_first_error(control_comm, local_error);
        owner.reset();
        throw std::runtime_error("CUDA IPC FIFO allocation failed: " + error);
    }

    local_error.fill('\0');
    void* outgoing_region = nullptr;
    try {
        if (has_outgoing) {
            const int receiver = (global_rank + 1) % kRanks;
            const char* control_name = control_names[receiver].data();
            if (control_name[0] == '\0') {
                throw std::runtime_error(
                    "CUDA IPC receiver did not publish close control");
            }
            auto [control, fd] = open_close_control(control_name);
            owner->set_remote_control(control, fd);
            CUDA_CHECK_THROW(cudaIpcOpenMemHandle(
                &outgoing_region, handles[receiver],
                cudaIpcMemLazyEnablePeerAccess));
            owner->set_remote_region(outgoing_region);
#if defined(NANO_NCCL_TRANSPORT_TEST_FAULT_INJECTION)
            if (inject_failure("p2p_open", global_rank)) {
                throw std::runtime_error("injected P2P open failure");
            }
#endif
        }
    } catch (const std::exception& error) {
        local_ok = 0;
        std::snprintf(
            local_error.data(), local_error.size(),
            "P2P outgoing edge %d->%d mapping failed at global rank %d: %s",
            global_rank, (global_rank + 1) % kRanks, global_rank, error.what());
    }

    mpi_check(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN,
                            control_comm),
              "MPI_Allreduce(CUDA IPC mapping)");
    const int opened_outgoing = outgoing_region == nullptr ? 0 : 1;
    std::vector<int> opened_edges(kRanks);
    mpi_check(MPI_Allgather(&opened_outgoing, 1, MPI_INT,
                            opened_edges.data(), 1, MPI_INT, control_comm),
              "MPI_Allgather(CUDA IPC opened edges)");
    if (has_incoming) {
        owner->arm_local_close_wait(opened_edges[incoming_edge] != 0);
    }
    if (all_ok == 0) {
        const std::string error = gather_first_error(control_comm, local_error);
        owner.reset();
        throw std::runtime_error("CUDA IPC FIFO mapping failed: " + error);
    }
    mpi_check(MPI_Barrier(control_comm), "MPI_Barrier(CUDA IPC initialize)");
    owner->unlink_local_control();

    ConnectionResources resources;
    for (int channel = 0; channel < kChannels; ++channel) {
        if (has_incoming) {
            resources.view.recv_fifo[channel] =
                simple::connection_fifo(incoming_region, channel);
            resources.view.recv_head[channel] =
                simple::connection_head(incoming_region, channel);
            resources.view.recv_tail[channel] =
                simple::connection_tail(incoming_region, channel);
        }
        if (has_outgoing) {
            resources.view.send_fifo[channel] =
                simple::connection_fifo(outgoing_region, channel);
            resources.view.send_head[channel] =
                simple::connection_head(outgoing_region, channel);
            resources.view.send_tail[channel] =
                simple::connection_tail(outgoing_region, channel);
        }
    }
    resources.owners.push_back(erase_connection_owner(std::move(owner)));
    return resources;
}

}  // namespace nano_nccl::transport::p2p
