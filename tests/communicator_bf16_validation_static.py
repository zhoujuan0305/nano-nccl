#!/usr/bin/env python3
"""Verify BF16 device capability validation is not on the collective hot path."""

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
                return source[open_brace:index + 1]
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
    dispatch = function_body(source, "void all_reduce(const CollectiveArgs& args)")
    case_start = dispatch.find("case DType::BFloat16:")
    case_end = dispatch.find("}", case_start)
    if case_start == -1 or case_end == -1:
        raise AssertionError("missing BF16 dispatch case")
    bf16_case = dispatch[case_start:case_end]
    if "ensure_bf16_devices_validated();" not in bf16_case:
        raise AssertionError("BF16 dispatch must use cached capability validation")
    if "require_bf16_devices(devices_);" in bf16_case:
        raise AssertionError("BF16 dispatch directly validates every collective")

    helper = function_body(source, "void ensure_bf16_devices_validated()")
    require_in_order(helper, [
        "if (bf16_devices_validated_) return;",
        "require_bf16_devices(devices_);",
        "bf16_devices_validated_ = true;",
    ])
    if "bool bf16_devices_validated_ = false;" not in source:
        raise AssertionError("missing BF16 validation cache state")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, IndexError) as error:
        print(f"BF16 capability validation static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
