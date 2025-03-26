## Installation

You can install it directly on Windows (not WSL) and on Unix-like systems. For details on supported platforms, check [FAQs.md](./FAQs.md).

```bash
$ python3 -m pip install -r requirements.txt
```

Requires Python 3.6 or higher.

## Docker

Only works on Linux platforms with Nvidia GPUs. [Check this doc](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation).

```bash
$ docker build -t sol_vanity_cl .
$ docker run --rm -it --gpus all sol_vanity_cl
```

You will enter the container. The source code is located in the /app directory in the container, and all dependencies have been installed.

Use the Docker image loerfy/sol_vanity_cl:latest. You can easily use the template I created on [vast.ai](https://cloud.vast.ai/?ref_id=109219&creator_id=109219&name=SolVanityCL) or [runpod.io](https://runpod.io/console/deploy?template=fgllgqsl24&ref=uh5x1hv5) to run this program. Please note:

1. The device’s CUDA version should be greater than 12.0.
2. The source code is located in the /app directory, so you don’t need to download the code from GitHub.

## Usage

```bash
$ python3 main.py

Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  search-pubkey  Search Solana vanity pubkey
  show-device    Show OpenCL devices
```

### Search Pubkey

```bash
$ python3 main.py search-pubkey --help

Usage: main.py search-pubkey [OPTIONS]

  Search Solana vanity pubkey

Options:
  --starts-with TEXT              Public key starts with the indicated prefix. Provide multiple arguments to search for multiple prefixes.
  --ends-with TEXT                Public key ends with the indicated suffix.
  --count INTEGER                 Count of pubkeys to generate.  [default: 1]
  --output-dir DIRECTORY          Output directory.  [default: ./]
  --select-device / --no-select-device
                                  Select OpenCL device manually  [default: no-
                                  select-device]
  --iteration-bits INTEGER        Number of the iteration occupied bits.
                                  Recommended 24, 26, 28, 30, 32. The larger
                                  the bits, the longer it takes to complete an
                                  iteration.  [default: 24]
  --is-case-sensitive BOOLEAN     Whether the search should be case sensitive
                                  or not. [default: True]
  --help                          Show this message and exit.
```

Example:

```bash
$ python3 main.py search-pubkey --starts-with SoL # run
$ solana-keygen pubkey SoLxxxxxxxxxxx.json # you should install solana cli to verify it
```


## FAQs

See [FAQs.md](./FAQs.md).


## Donations

If you find this project helpful, please consider making a donation:

SOLANA: `PRM3ZUA5N2PRLKVBCL3SR3JS934M9TZKUZ7XTLUS223`

EVM: `0x8108003004784434355758338583453734488488`
