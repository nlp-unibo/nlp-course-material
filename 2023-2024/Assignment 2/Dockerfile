FROM nvcr.io/nvidia/pytorch:22.10-py3
# 21.10 --> 2080ti (11.4)
# 22.10 --> 3060ti

# ENV Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/include/"

COPY . /a2
RUN pip3 install -r /a2/requirements.txt
WORKDIR /a2

ENV GENSIM_DATA_DIR="/a2/gensim-data/"
ENV NUMBA_CACHE_DIR="/a2/numba/"

ENV PYTHONPATH="$PYTHONPATH:/a2/"

# Project-specific dependencies
RUN pip3 install -r requirements.txt