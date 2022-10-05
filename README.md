# 📍 GeoLocator

An end-to-end ML application built on top of [GeoEstimation](https://github.com/TIBHannover/GeoEstimation)'s pre-trained ResNet model that identifies the location (latitude, longitude) given an image or video.

Team: [@yiyixuxu](https://github.com/yiyixuxu), [@WinsonTruong](https://github.com/WinsonTruong), [@dayweek](https://github.com/dayweek), [@samhita-alla](https://github.com/samhita-alla)

TODO: include a screencast of the application

## 🥞 Stack

GeoLocator is the result attained by utilizing some best-in-class tools. We've used:

- **PyTorch Lightning** to construct the ResNet model (thanks to GeoEstimation's pre-trained model!)
- **YouTube DL** to download YouTube videos
- **Katna** to capture video frames
- **Geopy** to generate location from latitudes and longitudes
- **Plotly** to plot the maps
- **ONNX** to generate a serialized model
- **ONNXRuntime** to generate predictions
- **BentoML** to prep the model for serving
- **Gradio** to build the user-facing side of the application

## 🛠 Setup

The ``geolocator.ipynb`` notebook contains the relevant code and commands to set up the environment. Here's what it does:

- Clones our forked GeoEstimation repository
- Clones the geolocator repository in case you're running the notebook on Google Colab
- Downloads the model checkpoint and hyperparameters
- Downloads the pre-calculated partitionings
- Validates if the code's working fine by generating predictions
- Generates an ONNX version of the pre-trained model
- Generates predictions using ONNXRuntime to verify if the ONNX conversion was proper
- Creates a Bento ONNX model
- [Optional] Run the Bento service to generate predictions using the Bento model (doesn't work on Google Colab)

## Contents

### Pre- and post-processing logic

The `post_processing.py` and `pre_processing.py` files contain functions that does post- and pre-processing of the data, respectively.
`post_processing.py` contains functions to generate predictions.

...

- ``bentoml build``
- ``bentoml containerize geolocator:latest``
