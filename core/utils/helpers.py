import logging
import platform
from pathlib import Path
from typing import Tuple

import pyopencl as cl
from base58 import b58decode


def check_character(name: str, character: str) -> None:
    try:
        b58decode(character)
    except ValueError as e:
        logging.error(f"{str(e)} in {name}")
        raise SystemExit(1)
    except Exception as e:
        raise e


def need_skip_len44(prefixes, ignore_case=False):
    """
    返回值：
        True  —— 所有前缀在 44 字符长度内都不可能出现，可安全跳过 44 区
        False —— 至少有一个前缀在 44 字符长度内可行，必须保留 44 区
    完全对齐 solana-keygen grind 的 Rust 逻辑
    """
    if not prefixes:
        return False

    for p in prefixes:
        s = p.upper() if ignore_case else p
        probe = s + "1" * (44 - len(s))  # 与 Rust 的 '1' 填充一致
        decoded = b58decode(probe)

        if len(decoded) == 32:           # 至少一个前缀能落在 44 字符区
            return False                 # → 不可剪枝

    return True                          # → 可剪枝

def load_kernel_source(
    starts_with_list: Tuple[str], ends_with: str, is_case_sensitive: bool
) -> Tuple[str, bool]:
    """
    Update OpenCL codes with parameters
    """
    prefixes = (
        [list(prefix.encode()) for prefix in starts_with_list]
        if starts_with_list
        else [[]]
    )

    max_prefix_len = max((len(p) for p in prefixes), default=0)

    for p in prefixes:
        p.extend([0] * (max_prefix_len - len(p)))

    SUFFIX_BYTES = list(ends_with.encode())

    kernel_path = Path(__file__).parent.parent / "opencl" / "kernel.cl"
    if not kernel_path.exists():
        raise FileNotFoundError("Kernel source file not found.")
    with kernel_path.open("r") as f:
        source_lines = f.readlines()

    for i, line in enumerate(source_lines):
        if line.startswith("#define N "):
            source_lines[i] = f"#define N {len(prefixes)}\n"
        elif line.startswith("#define L "):
            source_lines[i] = f"#define L {max_prefix_len}\n"
        elif line.startswith("constant uchar PREFIXES"):
            prefixes_str = "{"
            for prefix in prefixes:
                prefixes_str += "{" + ", ".join(map(str, prefix)) + "}, "
            prefixes_str = prefixes_str.rstrip(", ") + "}"
            source_lines[i] = f"constant uchar PREFIXES[N][L] = {prefixes_str};\n"
        elif line.startswith("constant uchar SUFFIX[]"):
            source_lines[i] = (
                f"constant uchar SUFFIX[] = {{{', '.join(map(str, SUFFIX_BYTES))}}};\n"
            )
        elif line.startswith("constant bool CASE_SENSITIVE"):
            source_lines[i] = (
                f"constant bool CASE_SENSITIVE = {str(is_case_sensitive).lower()};\n"
            )

    source_str = "".join(source_lines)
    if "NVIDIA" in str(cl.get_platforms()) and platform.system() == "Windows":
        source_str = source_str.replace("#define __generic\n", "")
    if cl.get_cl_header_version()[0] != 1 and platform.system() != "Windows":
        source_str = source_str.replace("#define __generic\n", "")
    return source_str, need_skip_len44(starts_with_list, not is_case_sensitive)
