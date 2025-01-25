FROM nvidia/opencl:latest

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-pyopencl \
    python3-nacl \
    git

# Install specific version of click
RUN pip3 install base58 click pyopencl PyNaCl numpy

# Clone repository
RUN git clone https://github.com/WincerChan/SolVanityCL.git /app

WORKDIR /app

# Install requirements, forcing latest compatible versions

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Keep original entrypoint
ENTRYPOINT ["/bin/bash"]
