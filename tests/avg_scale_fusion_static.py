#!/usr/bin/env python3
"""Verify Avg scaling is fused into the final reduce-scatter write."""

from pathlib import Path
import sys


def function_body(source: str, signature: str) -> str:
    start = source.find(signature)
    if start == -1:
        raise AssertionError(f"missing function: {signature}")
    open_brace = source.find("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def require_in_order(body: str, fragments: list[str]) -> None:
    position = 0
    for fragment in fragments:
        position = body.find(fragment, position)
        if position == -1:
            raise AssertionError(f"missing fragment: {fragment}")
        position += len(fragment)


def main() -> int:
    source = Path(sys.argv[1]).read_text()

    kernel = function_body(source, "void ring_simple_kernel(")
    if kernel.count("recv_reduce_copy_send<T, kRedOp>(") != 1:
        raise AssertionError("kernel must have one final reduce-and-broadcast call")
    require_in_order(kernel, [
        "float inverse_nranks = 1.0f;",
        "if constexpr (kRedOp == RedOp::Avg) {",
        "inverse_nranks = 1.0f / static_cast<float>(nranks);",
        "if (!recv_reduce_copy_send<T, kRedOp>(",
        "&recv_tail_cache, &send_head_cache, inverse_nranks,",
        "nworkers, &s_wait_status)) return;",
    ])

    final_copy = function_body(source, "__device__ inline bool recv_reduce_copy_send")
    require_in_order(final_copy, [
        "std::uint64_t* send_head_cache, float inverse_nranks,",
        "int nworkers, int* wait_status)",
        "reduce_broadcast_volatile_worker<T, kRedOp>(",
        "local + slice_offset, recv, out + slice_offset, dst, work,",
        "inverse_nranks, nworkers);",
    ])

    final_reduce = function_body(
        source, "__device__ inline void reduce_broadcast_volatile_worker")
    require_in_order(final_reduce, [
        "std::size_t count, float inverse_nranks, int nworkers)",
        "float4 v = make_float4(",
        "if constexpr (kRedOp == RedOp::Avg) {",
        "v.x = scale_avg(v.x, inverse_nranks);",
        "v.y = scale_avg(v.y, inverse_nranks);",
        "v.z = scale_avg(v.z, inverse_nranks);",
        "v.w = scale_avg(v.w, inverse_nranks);",
        "dst04[i] = v;",
        "dst14[i] = v;",
        "reduce_broadcast_volatile_packed16_worker<T, kRedOp>(",
        "local, recv, dst0, dst1, count, inverse_nranks, nworkers);",
        "T v = RedOpTraits<kRedOp, T>::apply(local[i], recv_v[i]);",
        "if constexpr (kRedOp == RedOp::Avg) {",
        "v = scale_avg(v, inverse_nranks);",
        "dst0[i] = v;",
        "dst1[i] = v;",
    ])

    packed_final_reduce = function_body(
        source, "__device__ inline void reduce_broadcast_volatile_packed16_worker")
    require_in_order(packed_final_reduce, [
        "std::size_t count, float inverse_nranks, int nworkers)",
        "uint4 value = reduce_packed16<T, kRedOp>(",
        "if constexpr (kRedOp == RedOp::Avg) {",
        "value = scale_avg_packed16<T>(value, inverse_nranks);",
        "dst04[i] = value;",
        "dst14[i] = value;",
    ])

    if "scale_avg_worker" in source:
        raise AssertionError("Avg must not use an output-wide scaling worker")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, IndexError) as error:
        print(f"Avg scale fusion static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
