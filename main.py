import json
import logging
import os
import platform
import secrets
import sys
import time
from math import ceil

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
from pathlib import Path

import click
import numpy as np
import pyopencl as cl
from base58 import b58decode, b58encode
from nacl.signing import SigningKey

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")


def check_character(name: str, character: str):
    try:
        b58decode(character)
    except ValueError as e:
        logging.error(f"{str(e)} in {name}")
        sys.exit(1)
    except Exception as e:
        raise e


def get_kernel_source(starts_with: str, ends_with: str):
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

    if cl.get_cl_header_version()[0] != 1 and platform.system() != "Windows":
        source_str = source_str.replace("#define __generic\n", "")

    return source_str


class Searcher:
    def __init__(self, *, context, kernel_source, iteration_bits: int):
        # context and command queue
        self.context = context
        self.command_queue = cl.CommandQueue(context)

        # build program and kernel
        program = cl.Program(context, kernel_source).build()
        self.kernel = cl.Kernel(program, "generate_pubkey")

        self.iteration_bits = iteration_bits
        self.iteration_bytes = np.ubyte(ceil(iteration_bits / 8))
        self.global_work_size = (1 << iteration_bits,)
        self.local_work_size = (32,)

        self.key32 = self.generate_key32()

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
        carry_index = 0 - self.iteration_bytes
        if (new_key32[carry_index] < self.key32[carry_index]) and new_key32[
            carry_index
        ] != 0:
            new_key32[carry_index] = 0

        self.key32[:] = new_key32

    def find(self):
        memobj_key32 = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            32 * np.ubyte().itemsize,
            hostbuf=self.key32,
        )
        memobj_output = cl.Buffer(
            self.context, cl.mem_flags.READ_WRITE, 33 * np.ubyte().itemsize
        )
        memobj_occupied_bytes = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.array([self.iteration_bytes]),
        )

        output = np.zeros(33, dtype=np.ubyte)
        self.kernel.set_arg(0, memobj_key32)
        self.kernel.set_arg(1, memobj_output)
        self.kernel.set_arg(2, memobj_occupied_bytes)

        st = time.time()
        cl.enqueue_nd_range_kernel(
            self.command_queue, self.kernel, self.global_work_size, self.local_work_size
        )

        cl._enqueue_read_buffer(self.command_queue, memobj_output, output).wait()
        logging.info(
            f"Speed: {self.global_work_size[0] / ((time.time() - st) * 10**6):.2f} MH/s"
        )

        return output


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

    if select_device:
        context = cl.create_some_context()
    else:
        # get all platforms and devices
        devices = [
            device
            for platform in cl.get_platforms()
            for device in platform.get_devices()
        ]
        context = cl.Context(devices)

    logging.info(f"Searching with {len(context.devices)} OpenCL devices")

    kernel_source = get_kernel_source(starts_with, ends_with)

    searcher = Searcher(
        context=context,
        kernel_source=kernel_source,
        iteration_bits=iteration_bits,
    )

    result_count = 0

    while result_count < count:
        output = searcher.find()
        searcher.increase_key32()

        if not output[0]:
            continue

        pv_bytes = bytes(output[1:])
        pv = SigningKey(pv_bytes)
        pb_bytes = bytes(pv.verify_key)
        pubkey = b58encode(pb_bytes).decode()

        logging.info(f"Found: {pubkey}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, f"{pubkey}.json").write_text(
            json.dumps(list(pv_bytes + pb_bytes))
        )

        result_count += 1

        time.sleep(0.1)


@cli.command(context_settings={"show_default": True})
def show_device():
    """Show OpenCL devices"""

    platforms = cl.get_platforms()

    for p_index, platform in enumerate(platforms):
        print(f"Platform {p_index}: {platform.name}")

        devices = platform.get_devices()

        for d_index, device in enumerate(devices):
            print(f"- Device {d_index}: {device.name}")


if __name__ == "__main__":
    cli()
