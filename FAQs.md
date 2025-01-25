## What platforms does this project support?

This project has been tested and works correctly only on the **arm64 Darwin platform**, **Linux platforms with Nvidia GPUs** and **Windows platforms with Nvidia GPUs** . Results may be incorrect on Windows + AMD platforms (Especially the Vega architecture).

## Does this project support Multi-GPU?

Support for NVIDIA Multi-GPU. Tested with 2 GPUs, 4 GPUs, and 8 GPUs. Due to a known issue (busy-wait) when running OpenCL in NVIDIA, each GPU will fully utilize one CPU core. Make sure you have enough CPU cores.

## Why does this project not work on my computer?

1. Make sure your platform is supported.
2. For those with both a CPU and GPU that support OpenCL, please ensure you select the GPU to run the program (use the `--select-device` parameter)!

## Can I request custom features or modifications?

Yes, I offer customization services tailored to your specific needs. Feel free to contact me via [Telegram](https://t.me/Tivsae) for pricing details.
