FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Dependencies for glvnd and X11 and ssh.
RUN apt-get update \
  && apt-get install -y -qq \
    xauth \
    libxext6 \
    libx11-6 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev \
    openssh-server \
    mesa-utils \
    nano \
    git \
    curl \
    wget \
  && rm -rf /var/lib/apt/lists/*

ENV GIT_SSL_NO_VERIFY true
RUN conda --version
RUN conda config --set ssl_verify false

RUN conda info && \
    conda install -c conda-forge gym scikit-learn profilehooks progressbar matplotlib tensorboard numpy pandas flake8 -y

RUN conda info && pip install tensorboard-reducer --no-dependencies --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir

RUN conda info && pip3 install git+https://github.com/DLR-RM/stable-baselines3 --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir && pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/ppo-lstm --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir


# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


ARG UNAME=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

RUN echo "$UNAME:$UNAME" |chpasswd
RUN echo "$UNAME:$UNAME"

RUN mkdir /var/run/sshd

RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^#?PasswordAuthentication\s+.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM no/g' /etc/ssh/sshd_config
RUN sed -ri 's/#X11DisplayOffset 10/X11DisplayOffset 10/g' /etc/ssh/sshd_config
RUN sed -ri 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config
RUN sed -ri 's/#Port 22/Port 222/g' /etc/ssh/sshd_config

RUN ssh-keygen -A

# RUN ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -q -N ""
# RUN ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -q -N ""

RUN ls -al /etc/ssh

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# USER $UNAME
RUN mkdir /home/${UNAME}/.ssh
# RUN ln -s /etc/ssh/ssh_host_ed25519_key.pub /home/${UNAME}/.ssh/authorized_keys

RUN echo "export PATH=/opt/conda/bin:\$PATH" >> /home/${UNAME}/.bashrc
RUN echo "export http_proxy=$http_proxy" >> /home/${UNAME}/.bashrc
RUN echo "export https_proxy=$https_proxy" >> /home/${UNAME}/.bashrc
RUN git config --global http.sslVerify "false"

EXPOSE 222
CMD    ["/usr/sbin/sshd", "-D", "-e"]


