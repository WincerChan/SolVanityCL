FROM nvidia/opencl:latest

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    python3-pyopencl \
    python3-nacl

# Clone the repository
RUN git clone https://github.com/WincerChan/SolVanityCL.git /app

# Set working directory
WORKDIR /app

# Install Python requirements
RUN pip3 install -r requirements.txt

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Use bash as entrypoint
ENTRYPOINT ["/bin/bash"]
