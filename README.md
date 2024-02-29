## Installation

python3 -m pip install -r requirements.txt (require python3.6+)

## Run

1. *Edit* `gen_pattern.py`: Change the prefix and suffix to the Solana address you want. This step involves modifying the Python script to specify the desired Solana address pattern.
2. *Generate pattern* `python3 gen_pattern.py`: Update the kernel file.
3. *Execute main script* `python3 main.py`: This command presumably searches for Solana keys matching the pattern specified in gen_pattern.py and then writes them to a file in the current directory.