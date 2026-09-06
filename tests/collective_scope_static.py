#!/usr/bin/env python3
"""Keep single-host collective guards and AllGather's copy-only dispatch."""

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
    for collective, args_type in (
        ("all_gather", "AllGatherArgs"),
        ("reduce_scatter", "ReduceScatterArgs"),
    ):
        dispatch = function_body(
            source, f"void {collective}(const {args_type}& args)")
        require_in_order(dispatch, [
            "check_async_error();",
            "if (topology_.distributed)",
            f'"{collective} currently supports single-host communicators only"',
            "validate_args(args);",
        ])

    launcher = function_body(source, "struct AllGatherKernelLauncher")
    if "RedOp" in launcher:
        raise AssertionError("AllGather launcher must not have a redop dimension")
    typed_dispatch = function_body(
        source, "void all_gather_typed(const AllGatherArgs& args)")
    if "RedOp" in typed_dispatch:
        raise AssertionError("AllGather typed dispatch must not fabricate a redop")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, IndexError) as error:
        print(f"Collective scope static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
