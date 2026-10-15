# C1：16MiB+4B 的 Ring 循环边界

GAP-ID：`GAP-RANK4-CHUNK-CAP-001`；分类：Different / performance-critical。这是源码推导，实测耗时的因果贡献仍为 **Unknown**。

## 固定输入

本轮float、4 ranks、1 channel、32MiB FIFO/channel，nano按分布式MPI路径执行。nano kernel SHA `0c31cff07ee5fb71f765307777bd84d78de5812351d4e8dbbae762e2c0d17518`，基于commit `cf473a44b9bdc89a1dacb88ccfef612a3e4deba9`加冻结修改。NCCL为v2.30.7-1、commit `73cf112295c33aee2b895f329f592f2a9b4b0f97`；运行日志四rank均报告未找到外部tuner。

## nano 的容量边界

- `src/transport/simple/protocol.h`：FIFO steps=8；`step.h::step_elems<float>()`使32MiB FIFO的每slot容量为4MiB，即1,048,576 floats。
- `src/collective/all_reduce/communicator.cu::ensure_fifo_buffers`（1047–1061）：`topology_.distributed`下slot容量固定为step_elems；当前双机路径确实走此分支。
- `src/collective/all_reduce/ring_simple_geometry.h::cbd_part`（20–43）：C1的 `chunk_count = align_up(ceil(count/4),128 floats)`；`loop_chunk_elems`将其限制到slot容量。
- `src/kernels/ring_simple_kernel.cuh`（1674–1683、1733）：按 `4 × loop_chunk`推进outer loop；最后不足一轮时再按4 floats向量宽度对齐实际chunk。

| 输入 | count（float） | grain对齐后chunk（float） | nano outer loops |
|---|---:|---:|---:|
| 9MiB | 2,359,296 | 589,824 | 1 |
| 12MiB | 3,145,728 | 786,432 | 1 |
| 16MiB | 4,194,304 | 1,048,576 | 1 |
| 16MiB+4B | 4,194,305 | 1,048,704 | 2 |
| 32MiB | 8,388,608 | 2,097,152 | 2 |

`16MiB+4B`首轮消耗4,194,304 floats，第二轮只剩一个有效float。第二轮chunk按向量宽度对齐到4 floats，其他空slice仍按现有逻辑跳过。它增加了尾部通信/同步机会；具体增量WR、CTS和等待时间本轮没有采计数。

## NCCL 的名义chunk与末轮缩小

- `src/include/collectives.h`（19–20）：AllReduce Ring的chunkSteps=4、sliceSteps=2。
- `src/enqueue.cc::calcCollChunking`（2220–2229）：stepSize=32MiB/8=4MiB，名义chunk上限=4MiB×4=16MiB/rank。当前Ring/Simple分支没有其他算法的chunk缩小启发式；未加载外部tuner。
- `src/enqueue.cc`（722–737）与 `src/include/device.h::ncclCollCbdPart`（328–360）：将chunk grain写入并解码到工作描述符。
- `src/device/all_reduce.h::runRing`（14–39）：名义outer-loop stride为4 ranks×16MiB=64MiB；遇末轮时按剩余count缩小实际chunk。上表所有输入均在一次outer loop内，实际每rank处理量随剩余count缩小。

因此nano和NCCL的outer-loop结构在该边界存在确定差异。额外loop能解释一个合理的慢路径候选，但源码公式不能单独证明计时差值的来源，也不能将两种循环数直接当作发送字节量或网络WR数量。本轮保持冻结代码；需要另外的有界几何A/B或低扰动计数才能隔离其性能贡献。
