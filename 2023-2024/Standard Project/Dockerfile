FROM nvcr.io/nvidia/pytorch:22.10-py3
# 21.10 --> 2080ti (11.4)
# 22.10 --> 3060ti

# ENV Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/include/"
ENV CUDA_LAUNCH_BLOCKING=1
ENV TF_FORCE_GPU_ALLOW_GROWTH='true'

COPY . /sp
RUN pip3 install -r /sp/requirements.txt
WORKDIR /sp

ENV GENSIM_DATA_DIR="/sp/gensim-data/"
ENV NUMBA_CACHE_DIR="/sp/numba/"

ENV PYTHONPATH="$PYTHONPATH:/sp/"

# Project-specific dependencies
RUN pip3 install -r requirements.txt