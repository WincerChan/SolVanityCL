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

# Accept the public key from an environment variable
ENV avain_ssh=pub


# Configure SSH to start on boot and run on port 26177
RUN mkdir /var/run/sshd && \
    echo 'Port 22' >> /etc/ssh/sshd_config && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


# Keep original entrypoint
ENTRYPOINT ["/entrypoint.sh"]

