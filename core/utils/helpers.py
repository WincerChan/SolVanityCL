import logging
import platform
from pathlib import Path

import pyopencl as cl
from base58 import b58decode
from typing import List

def check_character(name: str, character: str) -> None:
    try:
        b58decode(character)
    except ValueError as e:
        logging.error(f"{str(e)} in {name}")
        raise SystemExit(1)
    except Exception as e:
        raise e


def load_kernel_source(
    starts_with_list: List[str], ends_with: str, is_case_sensitive: bool
) -> str:
    """
    Update OpenCL codes with parameters
    """
    prefixes = []
    
    if not starts_with_list:
        prefixes.append([])
        max_prefix_len = 0
    else:
        for prefix in starts_with_list:
                prefix_bytes = list(prefix.encode())
                prefixes.append(prefix_bytes)
        max_prefix_len = max(len(p) for p in prefixes)
    
    for i in range(len(prefixes)):
        while len(prefixes[i]) < max_prefix_len:
            prefixes[i].append(0)
    
    SUFFIX_BYTES = list(ends_with.encode())

    kernel_path = Path(__file__).parent.parent / "opencl" / "kernel.cl"
    if not kernel_path.exists():
        raise FileNotFoundError("Kernel source file not found.")
    with kernel_path.open("r") as f:
        source_lines = f.readlines()

    for i, line in enumerate(source_lines):
        if line.startswith("#define N 1"):
            source_lines[i] = f"#define N {len(prefixes)}\n"
        elif line.startswith("#define L 3"):
            source_lines[i] = f"#define L {max_prefix_len}\n"
        elif line.startswith("constant uchar PREFIXES"):
            prefixes_str = "{"
            for prefix in prefixes:
                prefixes_str += "{" + ", ".join(map(str, prefix)) + "}, "
            prefixes_str = prefixes_str.rstrip(", ") + "}"
            source_lines[i] = f"constant uchar PREFIXES[N][L] = {prefixes_str};\n"
        elif line.startswith("constant uchar SUFFIX[]"):
            source_lines[i] = f"constant uchar SUFFIX[] = {{{', '.join(map(str, SUFFIX_BYTES))}}};\n"
        elif line.startswith("constant bool CASE_SENSITIVE"):
            source_lines[i] = f"constant bool CASE_SENSITIVE = {str(is_case_sensitive).lower()};\n"

    source_str = "".join(source_lines)
    if "NVIDIA" in str(cl.get_platforms()) and platform.system() == "Windows":
        source_str = source_str.replace("#define __generic\n", "")
    if cl.get_cl_header_version()[0] != 1 and platform.system() != "Windows":
        source_str = source_str.replace("#define __generic\n", "")
    return source_str
