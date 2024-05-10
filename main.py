import json
import logging
import secrets
import sys
import time
from math import ceil
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

    if cl.get_cl_header_version()[0] != 1:
        source_str = source_str.replace("#define __generic\n", "")

    return source_str


def generate_key32(iteration_bytes: np.ubyte):
    token_bytes = secrets.token_bytes(32 - iteration_bytes) + b"\x00" * iteration_bytes
    key32 = np.array([x for x in token_bytes], dtype=np.ubyte)
    return key32


def increase_key32(
    *, key32: np.ndarray, iteration_bits: int, iteration_bytes: np.ubyte
):
    current_number = int(bytes(key32).hex(), base=16)
    next_number = current_number + (1 << iteration_bits)
    _number_bytes = next_number.to_bytes(32, "big")
    new_key32 = np.array([x for x in _number_bytes], dtype=np.ubyte)
    carry_index = 0 - iteration_bytes
    if (new_key32[carry_index] < key32[carry_index]) and new_key32[carry_index] != 0:
        new_key32[carry_index] = 0

    key32[:] = new_key32
    return key32


def find(
    *,
    context,
    command_queue,
    kernel,
    key32: np.ndarray,
    iteration_bytes: np.ubyte,
    global_work_size,
    local_work_size,
):
    memobj_key32 = cl.Buffer(
        context,
        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        32 * np.ubyte().itemsize,
        hostbuf=key32,
    )
    memobj_output = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, 33 * np.ubyte().itemsize
    )
    memobj_occupied_bytes = cl.Buffer(
        context,
        cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=np.array([iteration_bytes]),
    )

    output = np.zeros(33, dtype=np.ubyte)
    kernel.set_arg(0, memobj_key32)
    kernel.set_arg(1, memobj_output)
    kernel.set_arg(2, memobj_occupied_bytes)

    st = time.time()
    cl.enqueue_nd_range_kernel(command_queue, kernel, global_work_size, local_work_size)

    cl._enqueue_read_buffer(command_queue, memobj_output, output).wait()
    logging.info(
        f"Speed: {global_work_size[0] / ((time.time() - st) * 10**6) :.2f} MH/s"
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
    "--iteration-bits",
    type=int,
    help="Number of the iteration occupied bits. Recommended 24, 26, 28, 30, 32. The larger the bits, the longer it takes to complete an iteration.",
    default=24,
)
@click.option(
    "--count",
    type=int,
    help="Count of pubkeys to generate.",
    default=1,
)
@click.pass_context
def search_pubkey(
    ctx, starts_with: str, ends_with: str, iteration_bits: int, count: int
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

    kernel_source = get_kernel_source(starts_with, ends_with)

    devices = []

    # get platform and device
    for platform in cl.get_platforms():
        devices.extend(platform.get_devices())

    logging.info(f"Searching with {len(devices)} OpenCL devices")

    # context and command queue
    context = cl.Context(devices)
    command_queue = cl.CommandQueue(context)

    # build program and kernel
    program = cl.Program(context, kernel_source).build()
    kernel = cl.Kernel(program, "generate_pubkey")

    iteration_bytes = np.ubyte(ceil(iteration_bits / 8))
    global_work_size = (1 << iteration_bits,)
    local_work_size = (32,)

    key32 = generate_key32(iteration_bytes)

    result_count = 0

    while result_count < count:
        output = find(
            context=context,
            command_queue=command_queue,
            kernel=kernel,
            key32=key32,
            iteration_bytes=iteration_bytes,
            global_work_size=global_work_size,
            local_work_size=local_work_size,
        )
        key32 = increase_key32(
            key32=key32, iteration_bits=iteration_bits, iteration_bytes=iteration_bytes
        )

        if not output[0]:
            continue

        pv_bytes = bytes(output[1:])
        pv = SigningKey(pv_bytes)
        pb_bytes = bytes(pv.verify_key)
        pubkey = b58encode(pb_bytes).decode()

        logging.info(f"Found: {pubkey}")
        Path(f"{pubkey}.json").write_text(json.dumps(list(pv_bytes + pb_bytes)))

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
