#pragma once

#include <cstddef>
#include <cstdint>

namespace nano_nccl::transport::socket {

struct SocketSliceHeader {
    std::uint32_t payload_bytes_be;
};

static_assert(sizeof(SocketSliceHeader) == sizeof(std::uint32_t));

void set_tcp_nodelay(int fd);
void send_all(int fd, const void* data, std::size_t bytes);
void recv_all(int fd, void* data, std::size_t bytes);
void send_slice(int fd, const void* payload, std::uint32_t payload_bytes);
std::uint32_t recv_slice(int fd, void* payload, std::size_t payload_capacity);

}  // namespace nano_nccl::transport::socket
