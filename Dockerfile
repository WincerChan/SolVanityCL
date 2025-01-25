# Use a more specific base image with Python 3.9 + OpenCL
FROM nvidia/opencl:devel-ubuntu20.04

# Install system dependencies and clean up in one RUN layer
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    python3-dev \
    ocl-icd-opencl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/WincerChan/SolVanityCL.git /app

WORKDIR /app

# Install Python dependencies with explicit versions
RUN python3 -m pip install --upgrade pip && \
    pip3 install \
    "click==8.0.4" \  # Explicitly use latest available version
    pyopencl \        # Use pip version instead of system package
    base58 \
    pynacl \
    "numpy==1.26.4"

# Environment configuration
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/bin/bash"]
