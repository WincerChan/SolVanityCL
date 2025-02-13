from src.lib import get_choosed_devices, check_character, get_kernel_source, save_result
from src.searcher import get_all_gpu_devices, multi_gpu_init
from src.host_setting import HostSetting

import logging
import multiprocessing
import os
import platform
import sys
import time
from multiprocessing.pool import Pool
import click
import pyopencl as cl
from pathlib import Path

os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
os.environ["PYOPENCL_NO_CACHE"] = "TRUE"

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")


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
    if select_device:
        choosed_devices = get_choosed_devices()
        gpu_counts = len(choosed_devices[1])
    else:
        gpu_counts = len(get_all_gpu_devices())

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
    # it's important
    multiprocessing.set_start_method("spawn")
    cli()
