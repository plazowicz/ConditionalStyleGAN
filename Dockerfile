FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

WORKDIR /home

RUN apt-get update

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
RUN conda --version

RUN git clone https://github.com/plazowicz/ConditionalStyleGAN.git && cd ConditionalStyleGAN && pip install -r requirements.txt

COPY dataset /home/ConditionalStyleGAN/dataset