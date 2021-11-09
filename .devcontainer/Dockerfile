FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN conda --version
RUN conda config --set ssl_verify false


RUN conda info && \
    conda install -c conda-forge gym scikit-learn profilehooks progressbar matplotlib stable-baselines3 tensorboard numpy pandas flake8 -y


RUN conda info && pip install tensorboard-reducer --no-dependencies --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir

# Dependencies for glvnd and X11 and ssh.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    xauth \
    libxext6 \
    libx11-6 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev openssh-server mesa-utils nano \
  && rm -rf /var/lib/apt/lists/*
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN mkdir /var/run/sshd
RUN echo 'root:root' |chpasswd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN sed -ri 's/#X11DisplayOffset 10/X11DisplayOffset 10/g' /etc/ssh/sshd_config
RUN sed -ri 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config
RUN mkdir /root/.ssh

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN echo "export PATH=/opt/conda/bin:$$PATH" >> /root/.bashrc



EXPOSE 22
CMD    ["/usr/sbin/sshd", "-D"]


