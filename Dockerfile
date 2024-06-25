FROM nvidia/opencl:latest

# 更新包列表并安装 Python 3 和 pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN apt-get install -y \
    python3-pyopencl \
    python3-nacl

# 安装 Python 依赖
RUN pip3 install base58 click
# 环境变量
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 将当前目录的内容拷贝到 Docker 容器中的 /app 目录
COPY opencl /app/opencl
COPY main.py /app

# 设置工作目录
WORKDIR /app


# 使用 bash 作为入口点
ENTRYPOINT ["/bin/bash"]