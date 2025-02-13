FROM ubuntu:24.04
LABEL maintainer "WincerChan <WincerChan@gmail.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        clinfo && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# for debug
RUN apt-get install -y neovim \
    python3-ipython

# python dependences
RUN apt-get install -y python3-click \
    python3-base58 \
    python3-nacl \
    python3-numpy \
    python3-pyopencl


# source codes
COPY main.py /app
COPY core /app/core
COPY LICENSE /app/LICENSE

# container-runtime
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR /app
ENTRYPOINT ["/bin/bash"]
