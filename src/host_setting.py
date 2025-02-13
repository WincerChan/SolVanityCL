import logging
from math import ceil
import time
import numpy as np
import secrets

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
        carry_index = 0 - int(
            self.iteration_bytes
        )  # for uint8 underflow on windows platform
        if (new_key32[carry_index] < self.key32[carry_index]) and new_key32[
            carry_index
        ] != 0:
            new_key32[carry_index] = 0

        self.key32[:] = new_key32