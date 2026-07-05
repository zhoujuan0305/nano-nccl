#include "transport/shm/shm_fifo.h"

namespace nano_nccl::transport::shm {

// 当前所有 FIFO helper 都是 __host__ __device__ inline 或 __device__ inline，
// 定义在头文件中以便 kernel 与 host 共享。此 .cu 仅作为翻译单元纳入构建目标。

}  // namespace nano_nccl::transport::shm
