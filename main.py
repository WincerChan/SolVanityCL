import json
import logging
import multiprocessing
import os
import platform
import secrets
import sys
import time
from math import ceil
from multiprocessing.pool import Pool
from typing import List, Optional, Tuple

import pyopencl as cl

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
os.environ["PYOPENCL_NO_CACHE"] = "TRUE"
from pathlib import Path

import click
import numpy as np
from base58 import b58decode, b58encode
from nacl.signing import SigningKey

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")


class HostSetting:
    def __init__(self, kernel_source: str, iteration_bits: int) -> None:
        self.iteration_bits = iteration_bits
        self.iteration_bytes = np.ubyte(ceil(iteration_bits / 8))
        self.global_work_size = 1 << iteration_bits
        self.local_work_size = 32
        self.key32 = self.generate_key32()

        self.kernel_source = kernel_source

    def generate_key32(self):
        token_bytes = (
            secrets.token_bytes(32 - self.iteration_bytes)
            + b"\x00" * self.iteration_bytes
        )
        key32 = np.array([x for x in token_bytes], dtype=np.ubyte)
        return key32

    def increase_key32(self):
        current_number = int(bytes(self.key32).hex(), base=16)
        next_number = current_number + (1 << self.iteration_bits)
        _number_bytes = next_number.to_bytes(32, "big")
        new_key32 = np.array([x for x in _number_bytes], dtype=np.ubyte)
        carry_index = 0 - int(
            self.iteration_bytes
        )  # for uint8 underflow on windows platform
        if (new_key32[carry_index] < self.key32[carry_index]) and new_key32[
            carry_index
        ] != 0:
            new_key32[carry_index] = 0

        self.key32[:] = new_key32


def check_character(name: str, character: str):
    try:
        b58decode(character)
    except ValueError as e:
        logging.error(f"{str(e)} in {name}")
        sys.exit(1)
    except Exception as e:
        raise e


def get_kernel_source(starts_with: str, ends_with: str, cl, is_case_sensitive: bool):
    PREFIX_BYTES = list(bytes(starts_with.encode()))
    SUFFIX_BYTES = list(bytes(ends_with.encode()))

    with open(Path("opencl/kernel.cl"), "r") as f:
        source_lines = f.readlines()

    for i, s in enumerate(source_lines):
        if s.startswith("constant uchar PREFIX[]"):
            source_lines[i] = (
                f"constant uchar PREFIX[] = {{{', '.join(map(str, PREFIX_BYTES))}}};\n"
            )
        if s.startswith("constant uchar SUFFIX[]"):
            source_lines[i] = (
                f"constant uchar SUFFIX[] = {{{', '.join(map(str, SUFFIX_BYTES))}}};\n"
            )
        if s.startswith("constant bool CASE_SENSITIVE"):
            source_lines[i] = (
                f"constant bool CASE_SENSITIVE = {str(is_case_sensitive).lower()};\n"
            )

    source_str = "".join(source_lines)

    if "NVIDIA" in str(cl.get_platforms()) and platform.system() == "Windows":
        source_str = source_str.replace("#define __generic\n", "")

    if cl.get_cl_header_version()[0] != 1 and platform.system() != "Windows":
        source_str = source_str.replace("#define __generic\n", "")

    return source_str


def get_all_gpu_devices():
    devices = [
        device
        for platform in cl.get_platforms()
        for device in platform.get_devices(device_type=cl.device_type.GPU)
    ]
    return [d.int_ptr for d in devices]


def get_selected_gpu_devices(platform_id, device_ids):
    platform = cl.get_platforms()[platform_id]
    devices, device_ptrs = platform.get_devices(), []
    for d_id in device_ids:
        device_ptrs.append(devices[d_id].int_ptr)
    return device_ptrs


def _choose_platform():
    platforms = cl.get_platforms()
    print("Choose platform:")
    for p_idx, _platform in enumerate(platforms):
        print(f"{p_idx}. {_platform}")
    return len(platforms)


def _choose_devices(platform_id):
    # not sorting, maybe has bug?
    platforms = cl.get_platforms()
    print("Choose device(s):")
    all_devices = platforms[platform_id].get_devices()
    for d_idx, device in enumerate(all_devices):
        print(f"{d_idx}. {device}")


def get_choosed_devices(pool):
    if "CHOOSED_OPENCL_DEVICES" in os.environ:
        (platform_id, device_ids) = os.environ.get("CHOOSED_OPENCL_DEVICES", "").split(
            ":"
        )
        return int(platform_id), list(map(int, device_ids.split(",")))
    platforms_count = pool.apply(_choose_platform)
    platform_id = click.prompt(
        "Choice ", default=0, type=click.IntRange(0, platforms_count)
    )
    pool.apply(_choose_devices, (platform_id,))
    try:
        device_ids = click.prompt(
            "Choice, comma-separated ",
            default="0",
            type=str,
        )
        devices_list = list(map(int, device_ids.split(",")))
    except ValueError:
        print("Input Error.")
        exit(-1)
    print(
        f"Set the environment variable CHOOSED_OPENCL_DEVICES='{platform_id}:{device_ids}' to avoid being asked again."
    )
    return (platform_id, devices_list)


def multi_gpu_init(
    index: int,
    setting: HostSetting,
    gpu_counts,
    stop_flag,
    lock,
    choosed_devices: Optional[Tuple[int, List[int]]] = None,
):
    # get all platforms and devices
    try:
        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=index,
            setting=setting,
            choosed_devices=choosed_devices,
        )
        i = 0
        st = time.time()
        while True:
            result = searcher.find(i == 0, st=time.time())
            if result[0]:
                with lock:
                    if not stop_flag.value:
                        stop_flag.value = 1
                return result
            if time.time() - st > max(gpu_counts, 1):
                i = 0
                st = time.time()
                with lock:
                    if stop_flag.value:
                        return result
            else:
                i += 1
    except Exception as e:
        logging.exception(e)
    return [0]


def save_result(outputs, output_dir):
    result_count = 0
    for output in outputs:
        if not output[0]:
            continue
        result_count += 1
        pv_bytes = bytes(output[1:])
        pv = SigningKey(pv_bytes)
        pb_bytes = bytes(pv.verify_key)
        pubkey = b58encode(pb_bytes).decode()

        logging.info(f"Found: {pubkey}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, f"{pubkey}.json").write_text(
            json.dumps(list(pv_bytes + pb_bytes))
        )
    return result_count


class Searcher:
    def __init__(
        self,
        *,
        kernel_source,
        index: int,
        setting: HostSetting,
        choosed_devices: Optional[Tuple[int, List[int]]] = None,
    ):
        if choosed_devices is None:
            device_ids = get_all_gpu_devices()
        else:
            device_ids = get_selected_gpu_devices(*choosed_devices)
        # context and command queue
        self.context = cl.Context(
            [cl.Device.from_int_ptr(device_ids[index])],
        )
        self.gpu_chunks = len(device_ids)
        self.command_queue = cl.CommandQueue(self.context)

        self.setting = setting
        self.index = index
        self.display_index = index if not choosed_devices else choosed_devices[1][index]
        self.prev_time = None
        self.is_nvidia = (
            "NVIDIA" in cl.Device.from_int_ptr(device_ids[index]).platform.name.upper()
        )

        # build program and kernel
        program = cl.Program(self.context, kernel_source).build()
        self.program = program
        self.kernel = cl.Kernel(program, "generate_pubkey")

        self.memobj_key32 = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            32 * np.ubyte().itemsize,
            hostbuf=self.setting.key32,
        )
        self.memobj_output = cl.Buffer(
            self.context, cl.mem_flags.READ_WRITE, 33 * np.ubyte().itemsize
        )

        self.memobj_occupied_bytes = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.array([self.setting.iteration_bytes]),
        )
        self.memobj_group_offset = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.array([self.index]),
        )
        self.output = np.zeros(33, dtype=np.ubyte)
        self.kernel.set_arg(0, self.memobj_key32)
        self.kernel.set_arg(1, self.memobj_output)
        self.kernel.set_arg(2, self.memobj_occupied_bytes)
        self.kernel.set_arg(3, self.memobj_group_offset)

    def filter_valid_result(self, outputs):
        valid_outputs = []
        for output in outputs:
            if not output[0]:
                continue
            valid_outputs.append(output)
        return valid_outputs

    def find(self, log_stats=True, st=0.0):
        cl.enqueue_copy(self.command_queue, self.memobj_key32, self.setting.key32)
        self.setting.increase_key32()

        global_worker_size = self.setting.global_work_size // self.gpu_chunks
        cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_worker_size,),
            (self.setting.local_work_size,),
        )
        if self.prev_time is not None and self.is_nvidia:
            time.sleep(self.prev_time * 0.98)
        cl._enqueue_read_buffer(
            self.command_queue, self.memobj_output, self.output
        ).wait()
        self.prev_time = time.time() - st
        if log_stats:
            logging.info(
                f"GPU {self.display_index} Speed: {global_worker_size / ((time.time() - st) * 10**6):.2f} MH/s"
            )

        return self.output


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option(
    "--starts-with",
    type=str,
    help="Public key starts with the indicated prefix.",
    default="",
)
@click.option(
    "--ends-with",
    type=str,
    help="Public key ends with the indicated suffix.",
    default="",
)
@click.option(
    "--count",
    type=int,
    help="Count of pubkeys to generate.",
    default=1,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    help="Output directory.",
    default="./",
)
@click.option(
    "--select-device/--no-select-device",
    type=bool,
    help="Select OpenCL device manually",
    default=False,
)
@click.option(
    "--iteration-bits",
    type=int,
    help="Number of the iteration occupied bits. Recommended 24, 26, 28, 30, 32. The larger the bits, the longer it takes to complete an iteration.",
    default=24,
)
@click.option(
    "--is-case-sensitive",
    type=bool,
    help="Whether the search should be case sensitive or not.",
    default=True,
)
@click.pass_context
def search_pubkey(
    ctx,
    starts_with: str,
    ends_with: str,
    count: int,
    output_dir: str,
    select_device: bool,
    iteration_bits: int,
    is_case_sensitive: bool,
):
    """Search Solana vanity pubkey"""

    if not starts_with and not ends_with:
        print("Please provides at least [starts with] or [ends with]\n")
        click.echo(ctx.get_help())
        sys.exit(1)

    check_character("starts_with", starts_with)
    check_character("ends_with", ends_with)

    choosed_devices = None
    with Pool() as pool:
        if select_device:
            choosed_devices = get_choosed_devices(pool)
            gpu_counts = len(choosed_devices[1])
        else:
            gpu_counts = len(pool.apply(get_all_gpu_devices))

    logging.info(
        f"Searching Solana pubkey that starts with '{starts_with}' and ends with '{ends_with} with case sensitivity {'on' if is_case_sensitive else 'off'}"
    )
    logging.info(f"Searching with {gpu_counts} OpenCL devices")

    result_count = 0
    with multiprocessing.Manager() as manager:
        with Pool(processes=gpu_counts) as pool:
            kernel_source = get_kernel_source(
                starts_with, ends_with, cl, is_case_sensitive
            )
            lock = manager.Lock()
            while result_count < count:
                stop_flag = manager.Value("i", 0)
                results = pool.starmap(
                    multi_gpu_init,
                    [
                        (
                            x,
                            HostSetting(kernel_source, iteration_bits),
                            gpu_counts,
                            stop_flag,
                            lock,
                            choosed_devices,
                        )
                        for x in range(gpu_counts)
                    ],
                )
                result_count += save_result(results, output_dir)


@cli.command(context_settings={"show_default": True})
def show_device():
    """Show OpenCL devices"""

    platforms = cl.get_platforms()

    for p_index, platform_ in enumerate(platforms):
        print(f"Platform {p_index}: {platform_.name}")

        devices = platform_.get_devices()

        for d_index, device in enumerate(devices):
            print(f"- Device {d_index}: {device.name}")


if __name__ == "__main__":
    cli()
