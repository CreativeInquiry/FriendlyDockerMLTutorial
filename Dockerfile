FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    curl

# Install Python 3
RUN apt-get install -y \
    python3-dev \
    python3-numpy \
    python3-pip

# Install OpenCV
RUN apt-get install -y \
    libopencv-dev \
    python-opencv

# Cleanup
RUN rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 --no-cache-dir install \
    opencv-python \
    Pillow \
    pyyaml \
    tqdm

# Install project specific dependencies
RUN pip3 --no-cache-dir install \
    tensorflow-gpu==1.8.0 \
    git+git://github.com/JiahuiYu/neuralgym.git@88292adb524186693a32404c0cfdc790426ea441
