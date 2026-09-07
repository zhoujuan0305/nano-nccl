#pragma once

#include "nano_nccl/types.h"
#include "transport/simple/connection.h"

#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nano_nccl::transport {

using ConnectionOwner = std::unique_ptr<void, void (*)(void*)>;

template <typename Owner>
ConnectionOwner erase_connection_owner(std::unique_ptr<Owner> owner) {
    return ConnectionOwner(owner.release(), [](void* pointer) {
        delete static_cast<Owner*>(pointer);
    });
}

struct ConnectionResources {
    simple::SimpleConnectionView view;
    std::vector<ConnectionOwner> owners;
};

inline void merge_connection_resources(ConnectionResources* destination,
                                       ConnectionResources source) {
    for (int channel = 0; channel < kChannels; ++channel) {
        if (source.view.send_fifo[channel] != nullptr) {
            if (destination->view.send_fifo[channel] != nullptr) {
                throw std::logic_error(
                    "multiple transports published one Simple send connection");
            }
            destination->view.send_fifo[channel] = source.view.send_fifo[channel];
            destination->view.send_head[channel] = source.view.send_head[channel];
            destination->view.send_tail[channel] = source.view.send_tail[channel];
        }
        if (source.view.recv_fifo[channel] != nullptr) {
            if (destination->view.recv_fifo[channel] != nullptr) {
                throw std::logic_error(
                    "multiple transports published one Simple receive connection");
            }
            destination->view.recv_fifo[channel] = source.view.recv_fifo[channel];
            destination->view.recv_head[channel] = source.view.recv_head[channel];
            destination->view.recv_tail[channel] = source.view.recv_tail[channel];
        }
    }
    destination->owners.insert(
        destination->owners.end(),
        std::make_move_iterator(source.owners.begin()),
        std::make_move_iterator(source.owners.end()));
}

}  // namespace nano_nccl::transport
