FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

SHELL ["/bin/bash", "-c"]

RUN rm /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.6 \
        python3.6-dev && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    pip3 --no-cache-dir install \
        numpy==1.17.4 \
        PyYAML==5.1.2 \
        mkl==2019.0 \
        mkl-include==2019.0 \
        cmake==3.15.3 \
        cffi==1.13.2 \
        typing==3.7.4.1 \
        six==1.13.0 \
        Pillow==6.2.1 \
        scipy==1.4.1 && \
    cd /tmp && \
    git clone https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git checkout v1.3.0 && \
    git submodule update --init --recursive && \
    python3.6 setup.py install && \
    cd /tmp && \
    git clone https://github.com/pytorch/vision.git && \
    cd vision && \
    git checkout v0.4.1 && \
    python3.6 setup.py install && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg && \
    pip3 --no-cache-dir install \
	    opencv-python==4.1.2.30 \
	    albumentations==0.4.3 \
	    tqdm==4.39.0 \
	    timm==0.1.18 \
	    efficientnet-pytorch==0.6.3 \
	    ffmpeg-python==0.2.0 \
	    tensorflow==1.15.2 && \
    cd / && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*
