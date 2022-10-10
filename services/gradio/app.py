import json
import mimetypes
import os
from typing import Tuple

import gradio as gr
import pandas as pd
import plotly
import plotly.express as px
import requests
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("ENDPOINT")


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

    location = data["location"]
    return data["location"], get_plotly_graph(
        latitude=data["latitude"], longitude=data["longitude"], location=location
    )


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

    location = data["location"]
    return location, get_plotly_graph(
        latitude=data["latitude"], longitude=data["longitude"], location=location
    )


def url_gradio(url: str) -> Tuple[str, plotly.graph_objects.Figure]:
    data = json.loads(
        requests.post(
            f"{URL}predict-url",
            headers={"content-type": "text/plain"},
            data=url,
        ).text
    )

    location = data["location"]
    return location, get_plotly_graph(
        latitude=data["latitude"], longitude=data["longitude"], location=location
    )


with gr.Blocks() as demo:
    gr.Markdown("# GeoLocator")
    gr.Markdown(
        "## An app that guesses the location of an image 🌌, a video 📹 or a YouTube link 🔗."
    )
    gr.Markdown(
        "Find the code powering this application [here](https://github.com/samhita-alla/geolocator)."
    )
    with gr.Tab("Image"):
        with gr.Row():
            img_input = gr.Image(type="filepath", label="im")
            with gr.Column():
                img_text_output = gr.Textbox(label="Location")
                img_plot = gr.Plot()
        img_text_button = gr.Button("Go locate!")
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

demo.launch()
