# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

import subprocess

from sanic import Sanic, response

import app as user_src

# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
user_src.init()

# Create the http server app
server = Sanic("geolocator_app")


# Healthchecks verify that the environment is correct on Banana Serverless
@server.route("/healthcheck", methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})


@server.route("/predict-image", methods=["POST"])
def inference_image(request):
    try:
        model_inputs = response.json.loads(request.json)
    except Exception:
        model_inputs = request.json

    output = user_src.inference_image(model_inputs)

    return response.json(output)


@server.route("/predict-video", methods=["POST"])
def inference_video(request):
    try:
        model_inputs = response.json.loads(request.json)
    except Exception:
        model_inputs = request.json

    output = user_src.inference_video(model_inputs)

    return response.json(output)


@server.route("/predict-url", methods=["POST"])
def inference_url(request):
    try:
        model_inputs = response.json.loads(request.json)
    except Exception:
        model_inputs = request.json

    output = user_src.inference_url(model_inputs)

    return response.json(output)


if __name__ == "__main__":
    server.run(host="0.0.0.0", port="8000", workers=1)
