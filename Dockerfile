#####################
# BANANA DOCKERFILE #
#####################

# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/samhita-alla/GeoEstimation.git

WORKDIR /GeoEstimation

RUN mkdir -p resources/s2_cells \
    && wget -nc https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/geo-cells/cells_50_5000.csv -O resources/s2_cells/cells_50_5000.csv
    && wget -nc https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/geo-cells/cells_50_2000.csv -O resources/s2_cells/cells_50_2000.csv
    && wget -nc https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/geo-cells/cells_50_1000.csv -O resources/s2_cells/cells_50_1000.csv

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install onnxruntime-gpu

RUN wget https://huggingface.co/Samhita/geolocator/resolve/main/geolocator.onnx

# We add the banana boilerplate here
ADD services/banana/server.py .

# Add your custom app code, init() and inference()
ADD services/banana/app.py .

EXPOSE 8000

CMD python3 -u server.py
