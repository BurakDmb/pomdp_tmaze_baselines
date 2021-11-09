# FROM nvidia/cuda:11.4.0-base-ubuntu20.04
FROM gpuci/miniconda-cuda:11.4-base-ubuntu20.04
# ARG DEBIAN_FRONTEND=noninteractive

RUN conda --version
RUN conda config --set ssl_verify false

RUN . /root/.bashrc && \
    conda init bash && \
    conda create -n env python=3.8

RUN source activate env && conda info && conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN source activate env && conda info && \
    conda install -c conda-forge gym scikit-learn profilehooks progressbar matplotlib stable-baselines3 tensorboard numpy pandas flake8 -y


RUN conda info && pip install tensorboard-reducer --no-dependencies --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir

RUN apt update && apt install xauth -y && apt-get clean

RUN echo conda activate env >> ~/.bashrc
