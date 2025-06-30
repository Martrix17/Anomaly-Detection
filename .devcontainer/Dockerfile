FROM mcr.microsoft.com/devcontainers/python:1-3.13

# Install OS packages
RUN apt-get update && apt-get install -y \
    git curl wget libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx \
    ffmpeg && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.6 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# App and utility dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
