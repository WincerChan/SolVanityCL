import secrets
from math import ceil

import numpy as np

DEFAULT_ITERATION_BITS = 24
DEFAULT_LOCAL_WORK_SIZE = 32


class HostSetting:
    def __init__(self, kernel_source: str, iteration_bits: int):
        self.iteration_bits = iteration_bits
        # iteration_bytes 为需要被迭代覆盖的字节数（向上取整）
        self.iteration_bytes = np.ubyte(ceil(iteration_bits / 8))
        self.global_work_size = 1 << iteration_bits
        self.local_work_size = DEFAULT_LOCAL_WORK_SIZE
        self.kernel_source = kernel_source
        self.key32 = self.generate_key32()

    def generate_key32(self) -> np.ndarray:
        token_bytes = secrets.token_bytes(
            32 - int(self.iteration_bytes)
        ) + b"\x00" * int(self.iteration_bytes)
        key32 = np.array(list(token_bytes), dtype=np.ubyte)
        return key32

    def increase_key32(self) -> None:
        current_number = int(bytes(self.key32).hex(), 16)
        next_number = current_number + (1 << self.iteration_bits)
        new_key32 = np.array(list(next_number.to_bytes(32, "big")), dtype=np.ubyte)
        carry_index = -int(self.iteration_bytes)
        if (new_key32[carry_index] < self.key32[carry_index]) and new_key32[
            carry_index
        ] != 0:
            new_key32[carry_index] = 0
        self.key32[:] = new_key32
