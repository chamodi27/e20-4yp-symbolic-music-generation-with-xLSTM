# Dockerfile — xLSTM Music Generation API
#
# Base image: CUDA 12.4 development image (has nvcc, C++ compiler, CUDA libs)
# This is required for the sLSTM CUDA kernel compilation that happens on
# first model load.
#
# Build:
#   docker build -t xlstm-music-api .
#
# Run:
#   docker run --gpus all -p 5000:5000 \
#     -e MODEL_PATH=/app/model \
#     -v /path/to/checkpoint-46000-last:/app/model \
#     xlstm-music-api

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-distutils \
    cmake \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.4 support first (large, has its own index)
RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install all other Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Set working directory
WORKDIR /app

# Copy application code
COPY app.py generator.py converter.py token_analysis.py ./

# Environment variables with defaults
ENV HOST=0.0.0.0
ENV PORT=5000
ENV CONTEXT_LENGTH=16384
# MODEL_PATH must be provided at runtime (via -e or -v):
# docker run -e MODEL_PATH=/app/model -v /local/checkpoint:/app/model ...
ENV MODEL_PATH=""

# sLSTM CUDA kernels compile to a cache directory — persist it between runs
# by mounting a volume here if desired:
# docker run -v xlstm-kernel-cache:/root/.cache/torch_extensions ...
ENV TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions

EXPOSE ${PORT}

CMD ["python", "app.py"]
