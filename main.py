import json
import logging
import secrets
import time
from pathlib import Path

import numpy as np
import pyopencl as cl
from base58 import b58encode
from nacl.signing import SigningKey

logging.basicConfig(format="%(asctime)s %(message)s")

global_work_size = (1 << 24,)
local_work_size = (32,)

# load kernel source code
source_str = Path("opencl/kernel.cl").read_text()
if cl.get_cl_header_version()[0] != 1:
    source_str = source_str.replace("#define __generic\n", "")


def generate_randomkey():
    gnumber_p = Path(".global_number.dat")
    if gnumber_p.exists():
        token_bytes = gnumber_p.read_bytes()
    else:
        token_bytes = secrets.token_bytes(32)
    # no existed, generate new one
    key32 = np.array([x for x in token_bytes], dtype=np.ubyte)
    return key32


def increament(key32: np.ndarray):
    # 从 28 位开始，到27 位，26 位
    for x in range(27, -1, -1):
        if key32[x] < 255:
            key32[x] += 1
            break
        else:
            key32[x] = 0
    Path(".global_number.dat").write_bytes(bytes(key32))


# get platform and device
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]

# context and command queue
context = cl.Context([device])
command_queue = cl.CommandQueue(context)


# build program and kernel
program = cl.Program(context, source_str).build()
kernel = cl.Kernel(program, "generate_pubkey")


def loop_find(key32):
    memobj_key32 = cl.Buffer(
        context,
        cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        32 * np.ubyte().itemsize,
        hostbuf=key32,
    )
    memobj_output = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, 33 * np.ubyte().itemsize
    )

    output = np.zeros(33, dtype=np.ubyte)
    kernel.set_arg(0, memobj_key32)
    kernel.set_arg(1, memobj_output)

    st = time.time()
    cl.enqueue_nd_range_kernel(command_queue, kernel, global_work_size, local_work_size)

    cl._enqueue_read_buffer(command_queue, memobj_output, output).wait()
    logging.warning(
        f"speed: {global_work_size[0] / ((time.time() - st) * 10**6) :.2f} MH/s"
    )

    return output


if __name__ == "__main__":
    key32 = generate_randomkey()
    while True:
        output = loop_find(key32)
        increament(key32)
        if not output[0]:
            continue
        pv_bytes = bytes(output[1:])
        pv = SigningKey(pv_bytes)
        pb_bytes = bytes(pv.verify_key)
        sol_addr = b58encode(pb_bytes).decode()
        logging.warning(f"成功: {sol_addr}")
        Path(f"{sol_addr}.json").write_text(json.dumps(list(pv_bytes + pb_bytes)))
        time.sleep(0.1)
