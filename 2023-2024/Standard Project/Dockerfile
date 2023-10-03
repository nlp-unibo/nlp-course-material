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

COPY "./Standard Project" "/Standard Project"
RUN pip3 install -r "/Standard Project/requirements.txt"
WORKDIR "/Standard Project"

ENV GENSIM_DATA_DIR="/Standard Project/gensim-data/"
ENV NUMBA_CACHE_DIR="/Standard Project/numba/"

# Project-specific dependencies
RUN pip3 install -r requirements.txt