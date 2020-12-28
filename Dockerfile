#Adaptation of https://hub.docker.com/r/opensciencegrid/tensorflow-gpu/dockerfile


FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

#SHELL ["/bin/bash", "--login", "-c"]

#LABEL opensciencegrid.name="CUDA/TensorFlow KDTree Operator"
#LABEL opensciencegrid.definition_url="https://github.com/pezzus/anisoEikoNet"

RUN apt-get update && apt-get upgrade -y --allow-unauthenticated

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get upgrade -y --allow-unauthenticated && \
    apt-get install -y --allow-unauthenticated \
        build-essential \
        #cmake \
        curl \
        davix-dev \
        dcap-dev \
        fonts-freefont-ttf \
        g++ \
        gcc \
        gfal2 \
        gfortran \
        git \
        libafterimage-dev \
        libavahi-compat-libdnssd-dev \
        #libcfitsio-dev \
        #libfftw3-dev \
        libfreetype6-dev \
        libftgl-dev \
        libgfal2-dev \
        libgif-dev \
        libgl2ps-dev \
        libglew-dev \
        libglu-dev \
        libgraphviz-dev \
        libgsl-dev \
        libjemalloc-dev \
        libjpeg-dev \
        libkrb5-dev \
        libldap2-dev \
        liblz4-dev \
        liblzma-dev \
        #libmysqlclient-dev \
        libpcre++-dev \
        #libpng-dev \
        #libpq-dev \
        #libpythia8-dev \
        #libqt4-dev \
        libreadline-dev \
        #libsqlite3-dev \
        libssl-dev \
        #libtbb-dev \
        #libtiff-dev \
        #libx11-dev \
        libxext-dev \
        libxft-dev \
        #libxml2-dev \
        libxpm-dev \
        libz-dev \
        libzmq3-dev \
        locales \
        lsb-release \
        make \
        module-init-tools \
        #openjdk-8-jdk \
        pkg-config \
        #r-base \
        #r-cran-rcpp \
        #r-cran-rinside \
        rsync \
        srm-ifce-dev \
        unixodbc-dev \
        unzip \
        vim \
        wget \
        && \
    apt-get clean 


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN sh Miniconda3-latest-Linux-x86_64.sh -b -p ./anaconda

ENV PATH="/anaconda/bin:${PATH}"

RUN conda update -n base -c defaults conda -y

RUN conda create --name tf_nndistance_env python=3.7 pip

#RUN conda create --name tf_nndistance_env python=3.7 pip numpy scipy tensorflow-gpu=2.2 ipython pip -c conda-forge -y

RUN conda run -n tf_nndistance_env pip install tensorflow-gpu==2.2 ipython

RUN conda run -n tf_nndistance_env pip install numpy scipy

RUN conda init bash

RUN echo conda activate ${ENVIRONMENT_NAME} >> /root/.bashrc

ADD . /tf_nearest_neighbor/

RUN cd /tf_nearest_neighbor/src/ && mkdir -p build && cd build

RUN conda run -n tf_nndistance_env /tf_nearest_neighbor/finish_build

#############################
