from base58 import b58decode

PREFIX = "SoL"
SUFFIX = ""

b58decode(PREFIX)
b58decode(SUFFIX)

PREFIX_BYTES = list(bytes(PREFIX.encode()))
SUFFIX_BYTES = list(bytes(SUFFIX.encode()))

with open("opencl/kernel.cl", "r") as f:
    source_lines = f.readlines()

for i, s in enumerate(source_lines):
    if s.startswith("constant uchar PREFIX[]"):
        source_lines[i] = (
            f"constant uchar PREFIX[] = {{{', '.join(map(str, PREFIX_BYTES))}}};\n"
        )
        print("Succeed update prefix in kernel file.")
    if s.startswith("constant uchar SUFFIX[]"):
        source_lines[i] = (
            f"constant uchar SUFFIX[] = {{{', '.join(map(str, SUFFIX_BYTES))}}};\n"
        )
        print("Succeed update suffix in kernel file.")


with open("opencl/kernel.cl", "w") as f:
    f.writelines(source_lines)
