import logging
import platform
from pathlib import Path

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


def load_kernel_source(
    starts_with: str, ends_with: str, is_case_sensitive: bool
) -> str:
    """
    Update OpenCL codes with parameters
    """
    PREFIX_BYTES = list(starts_with.encode())
    SUFFIX_BYTES = list(ends_with.encode())

    kernel_path = Path(__file__).parent.parent / "opencl" / "kernel.cl"
    if not kernel_path.exists():
        raise FileNotFoundError("Kernel source file not found.")
    with kernel_path.open("r") as f:
        source_lines = f.readlines()

    for i, line in enumerate(source_lines):
        if line.startswith("constant uchar PREFIX[]"):
            source_lines[i] = (
                f"constant uchar PREFIX[] = {{{', '.join(map(str, PREFIX_BYTES))}}};\n"
            )
        if line.startswith("constant uchar SUFFIX[]"):
            source_lines[i] = (
                f"constant uchar SUFFIX[] = {{{', '.join(map(str, SUFFIX_BYTES))}}};\n"
            )
        if line.startswith("constant bool CASE_SENSITIVE"):
            source_lines[i] = (
                f"constant bool CASE_SENSITIVE = {str(is_case_sensitive).lower()};\n"
            )

    source_str = "".join(source_lines)
    if "NVIDIA" in str(cl.get_platforms()) and platform.system() == "Windows":
        source_str = source_str.replace("#define __generic\n", "")
    if cl.get_cl_header_version()[0] != 1 and platform.system() != "Windows":
        source_str = source_str.replace("#define __generic\n", "")
    return source_str
