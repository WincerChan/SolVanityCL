## Installation

python3 -m pip install -r requirements.txt

Requires Python 3.6 or higher

## Run

1. **Edit** `gen_pattern.py`: Change the prefix and suffix to the Solana address you want. This step involves modifying the Python script to specify the desired Solana address pattern.
2. **Generate pattern** `python3 gen_pattern.py`: Update the kernel file.
3. **Edit** `main.py`: Locate line 15 and adjust the ITERATION_OCCUPIED_BITS variable. Recommended starting points are 24, 26, 28, 30, 32. Keep in mind, the larger the ITERATION_OCCUPIED_BITS, the longer it will take to complete one iteration.
4. **Execute main script** `python3 main.py`: This command presumably searches for Solana keys matching the pattern specified in gen_pattern.py and then writes them to a file in the current directory.
