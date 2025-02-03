import json
import logging
import os
import platform
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from multiprocessing.pool import Pool
import multiprocessing
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
        carry_index = 0 - int(self.iteration_bytes) # for uint8 underflow on windows platform
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


def get_kernel_source(starts_with: str, ends_with: str, cl):
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


def single_gpu_init(context, setting):
    searcher = Searcher(
        kernel_source=setting.kernel_source,
        index=0,
        setting=setting,
        context=context,
    )
    i = 0
    st = time.time()
    while True:
        result = searcher.find(i == 0)
        if result[0]:
            return [result]
        if time.time() - st > 1:
            i = 0
            st = time.time()
        else:
            i += 1

def multi_gpu_init(index: int, setting: HostSetting, gpu_counts, stop_flag, lock):
    # get all platforms and devices
    try:
        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=index,
            setting=setting,
        )
        i = 0
        st = time.time()
        while True:
            result = searcher.find(i == 0)
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
        self, *, kernel_source, index: int, setting: HostSetting, context=None
    ):
        device_ids = get_all_gpu_devices()
        # context and command queue
        if context:
            self.context = context
            self.gpu_chunks = 1
        else:
            self.context = cl.Context(
                [cl.Device.from_int_ptr(device_ids[index])],
            )
            self.gpu_chunks = len(device_ids)
        self.command_queue = cl.CommandQueue(self.context)

        self.setting = setting
        self.index = index

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

    def find(self, log_stats=True):
        cl.enqueue_copy(self.command_queue, self.memobj_key32, self.setting.key32)
        self.setting.increase_key32()

        st = time.time() if log_stats else 0
        global_worker_size = self.setting.global_work_size // self.gpu_chunks
        cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_worker_size,),
            (self.setting.local_work_size,),
        )
        cl._enqueue_read_buffer(self.command_queue, self.memobj_output, self.output).wait()
        if log_stats:
            logging.info(
                f"GPU {self.index} Speed: {global_worker_size/ ((time.time() - st) * 10**6):.2f} MH/s"
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
@click.pass_context
def search_pubkey(
    ctx,
    starts_with: str,
    ends_with: str,
    count: int,
    output_dir: str,
    select_device: bool,
    iteration_bits: int,
):
    """Search Solana vanity pubkey"""

    if not starts_with and not ends_with:
        print("Please provides at least [starts with] or [ends with]\n")
        click.echo(ctx.get_help())
        sys.exit(1)

    check_character("starts_with", starts_with)
    check_character("ends_with", ends_with)

    logging.info(
        f"Searching Solana pubkey that starts with '{starts_with}' and ends with '{ends_with}'"
    )
    with Pool() as pool:
        gpu_counts = len(pool.apply(get_all_gpu_devices))

    result_count = 0

    logging.info(f"Searching with {gpu_counts} OpenCL devices")
    if select_device:
        with ThreadPoolExecutor(max_workers=1) as executor:
            context = cl.create_some_context()
            kernel_source = get_kernel_source(starts_with, ends_with, cl)
            while result_count < count:
                setting = HostSetting(kernel_source, iteration_bits)
                future = executor.submit(single_gpu_init, context, setting)
                result = future.result()
                result_count += save_result(result, output_dir)
        return

    with multiprocessing.Manager() as manager:
        with Pool(processes=gpu_counts) as pool:
            kernel_source = get_kernel_source(starts_with, ends_with, cl)
            lock = manager.Lock()
            while result_count < count:
                stop_flag = manager.Value("i", 0)
                results = pool.starmap(
                    multi_gpu_init, [
                        (x, HostSetting(kernel_source, iteration_bits), gpu_counts, stop_flag, lock)
                        for x in range(gpu_counts)
                    ]
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
