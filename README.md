## Installation

python3 -m pip install -r requirements.txt

Requires Python 3.6 or higher

## Run

1. *Edit* `gen_pattern.py`: Change the prefix and suffix to the Solana address you want. This step involves modifying the Python script to specify the desired Solana address pattern.
2. *Generate pattern* `python3 gen_pattern.py`: Update the kernel file.
3. Edit main.py: Go to line 14 and adjust the global_work_size variable. Ensure the value does not exceed (1 << 32). Recommended starting points are 1 << 24, 1 << 28, 1 << 30, or 1 << 32. Keep in mind, the larger the global_work_size, the longer it will take to complete one iteration.
4. *Execute main script* `python3 main.py`: This command presumably searches for Solana keys matching the pattern specified in gen_pattern.py and then writes them to a file in the current directory.
