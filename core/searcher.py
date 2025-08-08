import logging
import time
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

from core.config import HostSetting
from core.opencl.manager import (
    get_all_gpu_devices,
    get_selected_gpu_devices,
)


class Searcher:
    def __init__(
        self,
        kernel_source: str,
        index: int,
        setting: HostSetting,
        chosen_devices: Optional[Tuple[int, List[int]]] = None,
    ):
        if chosen_devices is None:
            devices = get_all_gpu_devices()
        else:
            devices = get_selected_gpu_devices(*chosen_devices)
        enabled_device = devices[index]
        self.context = cl.Context([enabled_device])
        self.gpu_chunks = len(devices)
        self.command_queue = cl.CommandQueue(self.context)
        self.setting = setting
        self.index = index
        self.display_index = (
            index if chosen_devices is None else chosen_devices[1][index]
        )
        self.prev_time = None
        self.is_nvidia = "NVIDIA" in enabled_device.platform.name.upper()

        program = cl.Program(self.context, kernel_source).build()
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

    def find(self, log_stats: bool = True) -> np.ndarray:
        start_time = time.time()
        cl.enqueue_copy(self.command_queue, self.memobj_key32, self.setting.key32)
        global_work_size = self.setting.global_work_size // self.gpu_chunks
        local_size = self.setting.local_work_size
        global_size = ((global_work_size + local_size - 1) // local_size) * local_size # align global size and local size
        cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_size,),
            (local_size,),
        )
        self.command_queue.flush()
        self.setting.increase_key32()
        if self.prev_time is not None and self.is_nvidia:
            time.sleep(self.prev_time * 0.98)
        cl.enqueue_copy(self.command_queue, self.output, self.memobj_output).wait()
        self.prev_time = time.time() - start_time
        if log_stats:
            logging.info(
                f"GPU {self.display_index} Speed: {global_work_size / ((time.time() - start_time) * 1e6):.2f} MH/s"
            )
        return self.output


def multi_gpu_init(
    index: int,
    setting: HostSetting,
    gpu_counts: int,
    stop_flag,
    lock,
    chosen_devices: Optional[Tuple[int, List[int]]] = None,
) -> List:
    try:
        searcher = Searcher(
            kernel_source=setting.kernel_source,
            index=index,
            setting=setting,
            chosen_devices=chosen_devices,
        )
        i = 0
        st = time.time()
        while True:
            result = searcher.find(i == 0)
            if result[0]:
                with lock:
                    if not stop_flag.value:
                        stop_flag.value = 1
                return result.tolist()
            if time.time() - st > max(gpu_counts, 1):
                i = 0
                st = time.time()
                with lock:
                    if stop_flag.value:
                        return result.tolist()
            else:
                i += 1
    except Exception as e:
        logging.exception(e)
    return [0]


def save_result(outputs: List, output_dir: str) -> int:
    from core.utils.crypto import save_keypair

    result_count = 0
    for output in outputs:
        if not output[0]:
            continue
        result_count += 1
        pv_bytes = bytes(output[1:])
        save_keypair(pv_bytes, output_dir)
    return result_count
