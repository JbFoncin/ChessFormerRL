FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
USER root

# Install OpenCV with Gstreamer support
WORKDIR /usr/src
RUN apt-get update
RUN apt-get install -y git

RUN pip install scipy && \
    pip install tensorflow && \
    pip install chess && \
    pip install ipykernel

RUN apt-get update && apt-get install -y gnupg2

RUN cd .. && useradd -ms /bin/bash ubuntu

USER ubuntu
ENV HOME /home/ubuntu
WORKDIR /home/ubuntu