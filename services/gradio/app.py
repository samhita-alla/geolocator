import json
import mimetypes
import os
from typing import Dict, Tuple, Union

import gradio as gr
import pandas as pd
import plotly
import plotly.express as px
import requests
from dotenv import load_dotenv
from gantry_callback.gantry_util import GantryImageToTextLogger
from gantry_callback.s3_util import make_unique_bucket_name

load_dotenv()

URL = os.getenv("ENDPOINT")
GANTRY_APP_NAME = os.getenv("GANTRY_APP_NAME")
GANTRY_KEY = os.getenv("GANTRY_API_KEY")
AWS_KEY = os.getenv("AWS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KET")


def get_plotly_graph(
    latitude: float, longitude: float, location: str
) -> plotly.graph_objects.Figure:
    lat_long_data = [[latitude, longitude, location]]
    map_df = pd.DataFrame(lat_long_data, columns=["latitude", "longitude", "location"])

    px.set_mapbox_access_token(os.getenv("MAPBOX_TOKEN"))
    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        hover_name="location",
        color_discrete_sequence=["fuchsia"],
        zoom=5,
        height=300,
    )
    fig.update_layout(mapbox_style="dark")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def gradio_error():
    raise gr.Error("Unable to detect the location!")


def get_outputs(
    data: Dict[str, Union[str, float, None]]
) -> Tuple[str, plotly.graph_objects.Figure]:
    location = data["location"]
    if location is None:
        gradio_error()

    return data["location"], get_plotly_graph(
        latitude=data["latitude"], longitude=data["longitude"], location=location
    )


def image_gradio(img_file: str) -> Tuple[str, plotly.graph_objects.Figure]:
    data = json.loads(
        requests.post(
            f"{URL}predict-image",
            files={
                "image": (
                    img_file,
                    open(img_file, "rb"),
                    mimetypes.guess_type(img_file)[0],
                )
            },
        ).text
    )

    return get_outputs(data=data)


def video_gradio(video_file: str) -> Tuple[str, plotly.graph_objects.Figure]:
    data = json.loads(
        requests.post(
            f"{URL}predict-video",
            files={
                "video": (
                    video_file,
                    open(video_file, "rb"),
                    "application/octet-stream",
                )
            },
        ).text
    )

    return get_outputs(data=data)


def url_gradio(url: str) -> Tuple[str, plotly.graph_objects.Figure]:
    data = json.loads(
        requests.post(
            f"{URL}predict-url",
            headers={"content-type": "text/plain"},
            data=url,
        ).text
    )

    return get_outputs(data=data)


with gr.Blocks() as demo:
    gr.Markdown("# GeoLocator")
    gr.Markdown(
        "### An app that guesses the location of an image 🌌, a video 📹 or a YouTube link 🔗."
    )
    with gr.Tab("Image"):
        with gr.Row():
            img_input = gr.Image(type="filepath", label="im")
            with gr.Column():
                img_text_output = gr.Textbox(label="Location")
                img_plot = gr.Plot()
        img_text_button = gr.Button("Go locate!")
        with gr.Row():
            # Flag button
            img_flag_button = gr.Button("Flag this output")
    with gr.Tab("Video"):
        with gr.Row():
            video_input = gr.Video(type="filepath", label="video")
            with gr.Column():
                video_text_output = gr.Textbox(label="Location")
                video_plot = gr.Plot()
        video_text_button = gr.Button("Go locate!")
    with gr.Tab("YouTube Link"):
        with gr.Row():
            url_input = gr.Textbox(label="YouTube video link")
            with gr.Column():
                url_text_output = gr.Textbox(label="Location")
                url_plot = gr.Plot()
        url_text_button = gr.Button("Go locate!")

    # Gantry flagging #
    callback = GantryImageToTextLogger(application=GANTRY_APP_NAME, api_key=GANTRY_KEY)

    callback.setup(
        components=[img_input, img_text_output],
        flagging_dir=make_unique_bucket_name(prefix=GANTRY_APP_NAME, seed="420"),
    )

    img_flag_button.click(
        fn=lambda *args: callback.flag(args),
        inputs=[img_input, img_text_output],
        outputs=img_text_output,
        preprocess=False,
    )
    ###################

    img_text_button.click(
        image_gradio, inputs=img_input, outputs=[img_text_output, img_plot]
    )
    video_text_button.click(
        video_gradio, inputs=video_input, outputs=[video_text_output, video_plot]
    )
    url_text_button.click(
        url_gradio, inputs=url_input, outputs=[url_text_output, url_plot]
    )

    examples = gr.Examples(".", inputs=[img_input, video_input, url_input])

    gr.Markdown(
        "Check out the [GitHub repository](https://github.com/samhita-alla/geolocator) that this demo is based off of."
    )

demo.launch()
