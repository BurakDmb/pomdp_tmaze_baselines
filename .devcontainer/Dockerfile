# FROM nvidia/cuda:11.4.0-base-ubuntu20.04
FROM gpuci/miniconda-cuda:11.4-base-ubuntu20.04
# ARG DEBIAN_FRONTEND=noninteractive


ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""

ENV HTTP_PROXY=HTTP_PROXY
ENV HTTPS_PROXY=HTTPS_PROXY

# ENV TZ=Europe/Istanbul
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN apt-get update

# RUN apt-get install -y wget \
#     locales \
#     lsb-release \
#     mesa-utils \
#     git \
#     subversion \
#     nano \
#     xterm \
#     wget \
#     curl \
#     htop \
#     libssl-dev \
#     build-essential \
#     dbus-x11 \
#     software-properties-common && rm -rf /var/lib/apt/lists/*

# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 
# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"

RUN conda --version
RUN conda config --set ssl_verify false

RUN . /root/.bashrc && \
    conda init bash && \
    conda create -n env python=3.8

RUN source activate env && conda info && conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN source activate env && conda info && \
    conda install -c conda-forge gym scikit-learn profilehooks progressbar matplotlib stable-baselines3 tensorboard flake8 -y


RUN conda info && pip install tensorboard-reducer --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir
