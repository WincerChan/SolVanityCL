import os
import platform
from pathlib import Path

import click
import pyopencl as cl

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
os.environ["PYOPENCL_NO_CACHE"] = "TRUE"


def get_kernel_source(starts_with: str, ends_with: str, is_case_sensitive: bool) -> str:
    """
    Update OpenCL codes with parameters
    """
    PREFIX_BYTES = list(starts_with.encode())
    SUFFIX_BYTES = list(ends_with.encode())

    kernel_path = Path(__file__).parent / "kernel.cl"
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


def get_all_gpu_devices() -> list:
    devices = [
        device
        for platform_obj in cl.get_platforms()
        for device in platform_obj.get_devices(device_type=cl.device_type.GPU)
    ]
    return [d.int_ptr for d in devices]


def get_selected_gpu_devices(platform_id: int, device_ids: list) -> list:
    platform_obj = cl.get_platforms()[platform_id]
    devices = platform_obj.get_devices()
    return [devices[d_id].int_ptr for d_id in device_ids]


def get_chosen_devices() -> tuple:
    if "CHOOSED_OPENCL_DEVICES" in os.environ:
        platform_str, devices_str = os.environ.get("CHOOSED_OPENCL_DEVICES", "").split(
            ":"
        )
        return int(platform_str), list(map(int, devices_str.split(",")))
    platforms = cl.get_platforms()
    click.echo("Choose platform:")
    for idx, plat in enumerate(platforms):
        click.echo(f"{idx}. {plat.name}")
    platform_id = click.prompt(
        "Choice", default=0, type=click.IntRange(0, len(platforms) - 1)
    )
    click.echo("Choose device(s):")
    all_devices = platforms[platform_id].get_devices()
    for d_idx, device in enumerate(all_devices):
        click.echo(f"{d_idx}. {device.name}")
    device_ids_str = click.prompt("Choice, comma-separated", default="0", type=str)
    devices_list = list(map(int, device_ids_str.split(",")))
    click.echo(
        f"Set environment variable CHOOSED_OPENCL_DEVICES='{platform_id}:{device_ids_str}' to avoid future prompts."
    )
    return platform_id, devices_list
