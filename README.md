# 📍 FSDL 2022: GeoLocator

An end-to-end ML application built on top of [GeoEstimation](https://github.com/TIBHannover/GeoEstimation)'s pre-trained ResNet model that identifies the location (latitude, longitude) given an image or video.

Team: [@yiyixuxu](https://github.com/yiyixuxu), [@WinsonTruong](https://github.com/WinsonTruong), [@dayweek](https://github.com/dayweek), [@samhita-alla](https://github.com/samhita-alla)

TODO: include a screencast of the application

## 🥞 Stack

GeoLocator is the result attained by utilizing some of the best-in-class tools. We've used:

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

## 🗃 Contents

### Pre- and post-processing logic

The `post_processing.py` and `pre_processing.py` files contain functions that does post- and pre-processing of the data, respectively.
`post_processing.py` contains functions to generate our custom solution to generate location for videos.
`pre_processing.py` downloads a YouTube video if given and captures the video frames.

#### Our video prediction logic

The `post_processing.py` contains two technique to predict the location of a video -- DBSCAN clustering and prediction confidence.

DBSCAN Clustering clusters the predicted latitudes and longitudes of video frames, finds the dense cluster, and computes
the mean of the dense cluster's latitudes and longitudes.

Prediction confidence associates a prediction logit to every prediction and returns the latitude and longitude pertaining to the highest prediction logit.

We found that prediction confidence surpassed the performance of DBSCAN clustering, and thus, we zeroed in on prediction confidence as the inference logic.

### BentoML

After generating a Bento model, run the following commands in the `GeoEstimation` directory to generate a Docker image:

- `bentoml build`
- `bentoml containerize geolocator:<version>`
- `docker run -it --rm -p 3000:3000 geolocator:<version> serve --production`
- `python test_api.py` (Test!)

## AWS Lambda

To generate a publicly-accessible API endpoint, deploy the bento to AWS Lambda.
Run the following commands in the `services/bentoml` directory:

- `bentoctl operator install aws-lambda`
- `bentoctl init`
- `bentoctl build -b geolocator:<version> -f deployment_config.yaml`
- `terraform init`
- `terraform apply -var-file=bentoctl.tfvars -auto-approve`

To update the deployment:

- `bentoml build` in `GeoEstimation`
- `bentoctl build -b geolocator:<version> -f deployment_config.yaml` in `bentoml`
- `bentoctl apply -f deployment_config.yaml`

## Gradio

A user may want to input an image, a video file, or a YouTube link.
The Gradio interface has been designed such that either of these can be provided.

## Gantry
