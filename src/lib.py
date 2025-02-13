from src.host_setting import HostSetting
from src.searcher import Searcher

import logging
import json
import time
from base58 import b58decode, b58encode
from nacl.signing import SigningKey
from typing import Optional, Tuple, List
import pyopencl as cl
from pathlib import Path

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")


def get_chosen_devices():
    if "CHOOSED_OPENCL_DEVICES" in os.environ:
        (platform_id, device_ids) = os.environ.get("CHOOSED_OPENCL_DEVICES", "").split(
            ":"
        )
        return int(platform_id), list(map(int, device_ids.split(",")))
    platforms = cl.get_platforms()
    print("Choose platform:")
    for p_idx, _platform in enumerate(platforms):
        print(f"{p_idx}. {_platform}")
    platform_id = click.prompt(
        "Choice ", default=0, type=click.IntRange(0, len(platforms))
    )

    print("Choose device(s):")
    all_devices = platforms[platform_id].get_devices()
    # not sorting, may have a bug?
    for d_idx, device in enumerate(all_devices):
        print(f"{d_idx}. {device}")
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
