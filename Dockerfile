FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
USER root

WORKDIR /usr/src
RUN apt-get update
RUN apt-get install -y git

RUN pip install scipy==1.10.1 && \
    pip install tensorboard==2.11.2 && \
    pip install tensorboard-data-server==0.6.1 && \
    pip install tensorboard-plugin-wit==1.8.1 && \
    pip install tensorflow==2.11.0 && \
    pip install tensorflow-estimator==2.11.0 && \
    pip install tensorflow-io-gcs-filesystem==0.31.0 && \
    pip install chess==1.9.4 && \
    pip install ipykernel==6.21.2

RUN apt-get update && apt-get install -y gnupg2

RUN cd .. && useradd -ms /bin/bash ubuntu

USER ubuntu
ENV HOME /home/ubuntu
WORKDIR /home/ubuntu
