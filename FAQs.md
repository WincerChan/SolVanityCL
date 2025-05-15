## What platforms does this project support?

This project has been tested and works correctly only on the **arm64 Darwin platform**, **Linux platforms with Nvidia GPUs** and **Windows platforms with Nvidia GPUs** . Results may be incorrect on Windows + AMD platforms (Especially the Vega architecture).

## Does this project support Multi-GPU?

Support for NVIDIA Multi-GPU. Tested with 2 GPUs, 4 GPUs, and 8 GPUs. Due to a known issue (busy-wait) when running OpenCL in NVIDIA, each GPU will fully utilize one CPU core. Make sure you have enough CPU cores.

> The busy-waiting problem has been fixed. The program will never fully utilize the CPU cores but still needs enough of them.

## Why does this project not work on my computer?

1. Make sure your [platform is supported](#what-platforms-does-this-project-support).
2. For those with both a CPU and GPU that support OpenCL, please ensure you select the GPU to run the program (use the `--select-device` parameter)!

## How long does this program generally take to generate a matched address?

- **Ethereum** keys are simple 256-bit integers, so incrementing the private key leads to sequential, predictable public keys.  
- **Solana** uses a 512-bit seed passed through SHA-512 (with only the first half used), so even consecutive private keys yield non-sequential, hard-to-predict public keys.  

Estimating prefix/suffix search time depends heavily on which Base58 characters you’re targeting. If you treat every character as equally likely — which is only a rough approximation — you’d need about $58^6 \div (616\times10^6)\approx61.7\text{ seconds} $
on an 8×4090-GPU server to find a 6-character match. In practice, we’ve measured closer to **20 minutes** for 6-character prefixes or suffixes.

However, **not all characters are equally likely** in a Solana address. Some letters (like ‘1’) appear much less often, and others (like those in the ‘2–H’ range) much more. For the full breakdown of per-character probabilities and how they affect search time, see our [Final Decision](https://blog.itswincer.com/posts/solana-vanity-prefix-vs-suffix-probability-en/#final-decision) analysis.  

## Can I request custom features or modifications?

Yes, I offer customization services tailored to your specific needs. Feel free to [contact me](https://confirmly.itswincer.com/contact) for pricing details.
