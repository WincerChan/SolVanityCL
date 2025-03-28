import logging
import multiprocessing
import sys
from multiprocessing.pool import Pool
from typing import List, Optional, Tuple

import click
import pyopencl as cl

from core.config import DEFAULT_ITERATION_BITS, HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
    get_chosen_devices,
)
from core.searcher import multi_gpu_init, save_result
from core.utils.helpers import check_character, load_kernel_source

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option(
    "--starts-with",
    type=str,
    default=[],
    help="Public key starts with the indicated prefix. Provide multiple arguments to search for multiple prefixes.",
    multiple=True,
)
@click.option(
    "--ends-with",
    type=str,
    default="",
    help="Public key ends with the indicated suffix.",
)
@click.option("--count", type=int, default=1, help="Count of pubkeys to generate.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default="./",
    help="Output directory.",
)
@click.option(
    "--select-device/--no-select-device",
    default=False,
    help="Select OpenCL device manually",
)
@click.option(
    "--iteration-bits",
    type=int,
    default=DEFAULT_ITERATION_BITS,
    help="Iteration bits (e.g., 24, 26, 28, etc.)",
)
@click.option(
    "--is-case-sensitive", type=bool, default=True, help="Case sensitive search flag."
)
def search_pubkey(
    starts_with,
    ends_with,
    count,
    output_dir,
    select_device,
    iteration_bits,
    is_case_sensitive,
):
    """Search for Solana vanity pubkeys."""
    if not starts_with and not ends_with:
        click.echo("Please provide at least one of --starts-with or --ends-with.")
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        sys.exit(1)

    for prefix in starts_with:
        check_character("starts_with", prefix)
    check_character("ends_with", ends_with)

    chosen_devices: Optional[Tuple[int, List[int]]] = None
    if select_device:
        chosen_devices = get_chosen_devices()
        gpu_counts = len(chosen_devices[1])
    else:
        gpu_counts = len(get_all_gpu_devices())

    logging.info(
        "Searching Solana pubkey with starts_with=(%s), ends_with=%s, is_case_sensitive=%s",
        ", ".join(repr(s) for s in starts_with),
        repr(ends_with),
        is_case_sensitive,
    )
    logging.info(f"Using {gpu_counts} OpenCL device(s)")

    result_count = 0
    with multiprocessing.Manager() as manager:
        with Pool(processes=gpu_counts) as pool:
            kernel_source = load_kernel_source(
                starts_with, ends_with, is_case_sensitive
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
                            chosen_devices,
                        )
                        for x in range(gpu_counts)
                    ],
                )
                result_count += save_result(results, output_dir)


@cli.command(context_settings={"show_default": True})
def show_device():
    """Show available OpenCL devices."""
    platforms = cl.get_platforms()
    for p_index, platform in enumerate(platforms):
        click.echo(f"Platform {p_index}: {platform.name}")
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        for d_index, device in enumerate(devices):
            click.echo(f"  - Device {d_index}: {device.name}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    cli()
