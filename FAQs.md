## What platforms does this project support?

This project has been tested and works correctly only on the **arm64 Darwin platform**, **Linux platforms with Nvidia GPUs** and **Windows platforms with Nvidia GPUs** . Results may be incorrect on Windows + AMD platforms (Especially the Vega architecture).

## Does this project support Multi-GPU?

Support for NVIDIA Multi-GPU. Tested with 2 GPUs, 4 GPUs, and 8 GPUs. Due to a known issue (busy-wait) when running OpenCL in NVIDIA, each GPU will fully utilize one CPU core. Make sure you have enough CPU cores.

> The busy-waiting problem has been fixed. The program will never fully utilize the CPU cores but still needs enough of them.

## Why does this project not work on my computer?

1. Make sure your [platform is supported](#what-platforms-does-this-project-support).
2. For those with both a CPU and GPU that support OpenCL, please ensure you select the GPU to run the program (use the `--select-device` parameter)!

## How long does this program take to generate a matched address?

In Ethereum, private keys are simple 256-bit numbers, and sequential increments result in predictable public key changes.

Solana’s process is more complex: its 512-bit private key is derived from a seed using SHA-512, with only the first half used for key generation. Due to this hashing step, even if Solana’s private keys are incremented consecutively, the public keys remain non-sequential and appear more unpredictable than Ethereum’s.

Solana addresses with 6/7/8-letter prefixes or suffixes are difficult to mathematically estimate in terms of generation time, and their difficulty varies.

Theoretically, a program would need to perform 58 ** 6 computations to generate any 6-letter prefix/suffix address if the generation process is truly random. A server equipped with 8×4090 GPUs has a hashing speed of 616 MH/s, so the estimated time to generate such an address is (58 ** 6) / (616 × 1E6) ~= 61.7 seconds.

However, based on my testing, generating a 6-letter address actually takes around 20 minutes on an 8×4090 server.

If you require an 8-letter address, it may take days on an 8x4090 server.

## Can I request custom features or modifications?

Yes, I offer customization services tailored to your specific needs. Feel free to [contact me](https://confirmly.itswincer.com/contact) for pricing details.
