#####################
# BANANA DOCKERFILE #
#####################

# Must use a Cuda version 11+
FROM nvcr.io/nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev ca-certificates g++ python3-numpy gcc make git python3-setuptools python3-wheel python3-pip aria2 && aria2c -q -d /tmp -o cmake-3.21.0-linux-x86_64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz && tar -zxf /tmp/cmake-3.21.0-linux-x86_64.tar.gz --strip=1 -C /usr
WORKDIR /

# Install git & wget
RUN apt-get update && apt-get install -y git wget gfortran libsm6 libblas-dev liblapack-dev ffmpeg python3-pip sudo && \
    wget http://ftp.de.debian.org/debian/pool/main/y/youtube-dl/youtube-dl_2021.02.04.1-1_all.deb
    sudo apt-get install ./youtube-dl_2021.02.04.1-1_all.deb

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker

# Install python packages
RUN pip3 install --upgrade pip
ADD services/banana/banana_requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Clone model
RUN git clone https://github.com/samhita-alla/GeoEstimation.git

ADD services/banana/server.py GeoEstimation/
ADD services/banana/app.py GeoEstimation/
ADD app/post_processing.py GeoEstimation/
ADD app/pre_processing.py GeoEstimation/

WORKDIR /GeoEstimation

RUN mkdir -p resources/s2_cells && \
    wget -nc https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/geo-cells/cells_50_5000.csv -O resources/s2_cells/cells_50_5000.csv && \
    wget -nc https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/geo-cells/cells_50_2000.csv -O resources/s2_cells/cells_50_2000.csv && \
    wget -nc https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/geo-cells/cells_50_1000.csv -O resources/s2_cells/cells_50_1000.csv

# Download model
RUN wget https://huggingface.co/Samhita/geolocator/resolve/main/geolocator.onnx

EXPOSE 8000

CMD python3 -u server.py
