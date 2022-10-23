#####################
# BANANA DOCKERFILE #
#####################

# Must use a Cuda version 11+
FROM nvcr.io/nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

WORKDIR /

# Install git & wget
RUN apt-get update && apt-get install -y git wget gfortran libsm6 libblas-dev liblapack-dev ffmpeg youtube-dl

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
