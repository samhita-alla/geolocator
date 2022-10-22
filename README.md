# 📍 FSDL 2022: GeoLocator

An end-to-end ML application built on top of [GeoEstimation](https://github.com/TIBHannover/GeoEstimation)'s pre-trained ResNet model that identifies the location (latitude, longitude) given an image, a video or a YouTube link.

Team: [@samhita-alla](https://github.com/samhita-alla), [@yiyixuxu](https://github.com/yiyixuxu), [@WinsonTruong](https://github.com/WinsonTruong), [@dayweek](https://github.com/dayweek)

![](https://user-images.githubusercontent.com/27777173/194872168-41ab2f35-bc92-414f-b55d-c0fd7ac39477.png)

![](https://user-images.githubusercontent.com/27777173/194872185-e658c3b8-4d55-44b4-9214-bb99a7bb8328.png)

![](https://user-images.githubusercontent.com/27777173/194872209-e7dff2d7-61f9-4ce1-b46d-7fd853b7f5ab.png)

## 🥞 Stack

We built GeoLocator by utilizing some of the best-in-class tools.

- **PyTorch Lightning** to construct the ResNet model (thanks to GeoEstimation's pre-trained model!)
- **YouTube DL** to download YouTube videos
- **Katna** to capture video frames
- **Geopy** to generate location from latitudes and longitudes
- **Plotly** to plot maps
- **ONNX** to serialize the model
- **ONNXRuntime** to generate predictions
- **BentoML** to prep the model for serving / **Banana** for GPU inference
- **Gradio** to build the user-facing side of the application
- **Gantry** to monitor the application/model

## 🛠 Setup

The `geolocator.ipynb` notebook contains the relevant code and commands to set up the environment. Here's what it does:

- Installs the requirements
- Clones the [forked GeoEstimation repository](https://github.com/samhita-alla/GeoEstimation)
- Clones the geolocator repository in case you're running the notebook on Google Colab
- Downloads the model checkpoint and hyperparameters
- Downloads the pre-calculated partitionings
- Validates if the code's working fine by generating predictions
- Generates an ONNX version of the pre-trained model
- Generates predictions using ONNXRuntime to verify if the ONNX conversion was proper
- Creates a Bento ONNX model
- [Optional] Runs the Bento service (doesn't work on Google Colab)

## 🗃 Contents

### Pre- and post-processing logic

The `post_processing.py` and `pre_processing.py` files in the `app` directory contain functions that do post- and pre-processing of the data, respectively.
`post_processing.py` contains the custom inferencing logic to predict the location of a video or YouTube link.
`pre_processing.py` downloads a YouTube video if given and captures the video frames.

#### Our video prediction logic

The `post_processing.py` file contains two techniques to predict the location of a video: DBSCAN clustering and prediction confidence.

DBSCAN Clustering clusters the predicted latitudes and longitudes of video frames, finds the dense cluster, and computes
the mean of the dense cluster's latitudes and longitudes.

Prediction confidence associates a prediction logit to every prediction and returns the latitude and longitude pertaining to the highest prediction logit.

We found that prediction confidence surpassed the performance of DBSCAN clustering, and thus, we zeroed in on prediction confidence as the inference logic.

#### Environment variables

Gradio and Gantry can be accessed only after initializing the following environment variables in a `.env` file:

```
ENDPOINT (AWS EC2 public IP)
GANTRY_APP_NAME
GANTRY_API_KEY
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
MAPBOX_TOKEN
```

> AWS creds are required to store the uploaded images.

## 🤖 BentoML

After generating a Bento model, run the following commands in the `services/bentoml` directory to generate a Docker image:

- `bentoml build`
- `bentoml containerize geolocator:<version>`
- `docker run -it --rm -p 3000:3000 geolocator:<version> serve --production`
- `python test_api.py --url http://127.0.0.1:3000` (Test!)

## ⚙️ AWS EC2

To generate a publicly-accessible API endpoint, deploy the bento to AWS EC2 by following the steps outlined in [ec2_setup.md](services/bentoml/ec2_setup.md).

## 📈 Gradio

Gradio code can be found under the [services/gradio](services/gradio/) directory.

`cd` to the `services/gradio` directory and run `python app.py` to launch the Gradio UI.

## ☑️ Gantry

Code for Gantry flagging can be found under the [services/gantry_callback](services/gantry_callback/) directory.

> Gantry monitoring is currently implemented to flag the outputs of images only.

## 🍌 Banana

[Banana](https://www.banana.dev/) provides inference hosting for ML models on serverless GPUs.
The code is enclosed in the [services/banana](services/banana/) directory.

This repository is connected to banana, so `git push` triggers the build.
