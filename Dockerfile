FROM nvidia/opencl:devel-ubuntu20.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    python3-dev \
    ocl-icd-opencl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone repo
RUN git clone https://github.com/WincerChan/SolVanityCL.git /app

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    pip3 install -r requirements.txt

# Environment
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/bin/bash"]
