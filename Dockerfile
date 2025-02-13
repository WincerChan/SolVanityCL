FROM ubuntu:24.04
LABEL maintainer "WincerChan <WincerChan@gmail.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        clinfo

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# python dependences
RUN apt-get install -y python3-click \
    python3-base58 \
    python3-nacl \
    python3-numpy \
    python3-pyopencl && \
    rm -rf /var/lib/apt/lists/*


# source codes
COPY core /app/core
COPY main.py /app
COPY LICENSE /app

# container-runtime
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR /app
ENTRYPOINT ["/bin/bash"]
