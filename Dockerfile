FROM nvidia/opencl:latest

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-pyopencl \
    python3-nacl \
    python3-flask

# Install Python dependencies
RUN pip3 install base58 click flask

# Environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Create app and results directories
RUN mkdir -p /app/results

# Copy application files
COPY opencl /app/opencl
COPY main.py /app/
COPY web_server.py /app/

# Set working directory
WORKDIR /app

# Expose port for web access
EXPOSE 8000

# Start both the vanity address generator and web server
CMD python3 main.py search-pubkey --ends-with pump --output-dir /app/results & python3 web_server.py
