FROM nvidia/cuda:11.4.0-base-ubuntu20.04
# ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update

RUN apt-get install -y wget \
    locales \
    lsb-release \
    mesa-utils \
    git \
    subversion \
    nano \
    xterm \
    wget \
    curl \
    htop \
    libssl-dev \
    build-essential \
    dbus-x11 \
    software-properties-common && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN conda --version

RUN conda install python=3.8 pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN conda install gym scikit-learn profilehooks progressbar matplotlib stable-baselines3 tensorboard flake8 -c conda-forge
RUN pip install tensorboard-reducer