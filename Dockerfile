FROM nvidia/opencl:latest

RUN apt-get update && apt-get install -y \
   python3 \
   python3-pip \
   git \
   python3-pyopencl \
   python3-nacl

RUN git clone https://github.com/WincerChan/SolVanityCL.git /app

WORKDIR /app

RUN pip3 install \
   "click>=8.1.0" \
   pyopencl \
   base58 \
   PyNaCl \
   "numpy==1.26.4"

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENTRYPOINT ["/bin/bash"]
