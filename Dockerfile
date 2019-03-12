FROM nvidia/cuda:9.0-cudnn7-runtime

RUN apt update
# Vanity Software
RUN apt install -y emacs python3 python3-pip colordiff source-highlight 
# TF Requirements
RUN apt install -y libnccl2=2.4.2-1+cuda9.0 libnccl-dev=2.4.2-1+cuda9.0 libcupti-dev
RUN pip3 install numpy scipy tensorflow-gpu ipython pylint coverage gym hyperopt gensim psutil networkx

RUN echo "test -e /mnt/.bashrc_dk  && source /mnt/.bashrc_dk" >> ~/.bashrc
