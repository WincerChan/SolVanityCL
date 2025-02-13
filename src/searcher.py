from src.host_setting import HostSetting

import logging
from typing import List, Optional, Tuple
import pyopencl as cl
import time
import numpy as np

logging.basicConfig(level="INFO", format="[%(levelname)s %(asctime)s] %(message)s")



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

    def find(self, log_stats=True):
        st = time.time()
        cl.enqueue_copy(self.command_queue, self.memobj_key32, self.setting.key32)

        global_worker_size = self.setting.global_work_size // self.gpu_chunks
        cl.enqueue_nd_range_kernel(
            self.command_queue,
            self.kernel,
            (global_worker_size,),
            (self.setting.local_work_size,),
        )
        # This uses a bit of CPU, so we may as well do it while waiting for the GPU to compute.
        self.setting.increase_key32()

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


