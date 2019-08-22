# base image
FROM  nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# docker build 커맨드로 빌드 할 때 설정 옵션 지정
ARG  PYTHON_VERSION=3.6

# 필요한 라이브러리 설치, 도커 이미지 실행 시킬때 사용 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    vim \
    tmux \
    locales \
    cmake \
    sudo && \
    rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \ 
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy scipy ipython && \
    /opt/conda/bin/conda install -y scikit-learn && \
    /opt/conda/bin/conda install -y pytorch torchvision -c pytorch && \
    /opt/conda/bin/conda clean -ay

ENV PATH /opt/conda/bin:$PATH

# locale 설정
RUN locale-gen ko_KR.UTF-8
ENV LANG=ko.KR.utf8 TZ=Asia/Seoul
ENV LC_ALL=ko_KR.utf8

# python library 설치
RUN pip install --upgrade pip
RUN pip install jupyter
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install tensorflow-gpu

WORKDIR /workspace
VOLUME /backup

# 도커 컨테이너 시작할때 실행할 커맨드
CMD /bin/bash
