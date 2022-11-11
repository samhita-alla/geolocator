import base64
import json

# import mimetypes
import os
import sys
from io import BytesIO
from typing import Dict, Tuple, Union

import banana_dev as banana
import geopy.distance
import gradio as gr
import pandas as pd
import plotly
import plotly.express as px

# import requests
from dotenv import load_dotenv

sys.path.append("..")

from gantry_callback.gantry_util import GantryImageToTextLogger  # noqa: E402
from gantry_callback.s3_util import make_unique_bucket_name  # noqa: E402

load_dotenv()

URL = os.getenv("ENDPOINT")
GANTRY_APP_NAME = os.getenv("GANTRY_APP_NAME")
GANTRY_KEY = os.getenv("GANTRY_API_KEY")
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
BANANA_API_KEY = os.getenv("BANANA_API_KEY")
BANANA_MODEL_KEY = os.getenv("BANANA_MODEL_KEY")

examples = json.load(open("examples.json"))


def compute_distance(map_data: Dict[str, Dict[str, Union[str, float, None]]]):
    hierarchy_lat, hierarchy_long = (
        map_data["hierarchy"]["latitude"],
        map_data["hierarchy"]["longitude"],
    )

    coarse_lat, coarse_long = (
        map_data["coarse"]["latitude"],
        map_data["coarse"]["longitude"],
    )

    fine_lat, fine_long = (
        map_data["fine"]["latitude"],
        map_data["fine"]["longitude"],
    )

    hierarchy_to_coarse = geopy.distance.geodesic(
        (hierarchy_lat, hierarchy_long), (coarse_lat, coarse_long)
    ).miles

    hierarchy_to_fine = geopy.distance.geodesic(
        (hierarchy_lat, hierarchy_long), (fine_lat, fine_long)
    ).miles

    return hierarchy_to_coarse, hierarchy_to_fine


def get_plotly_graph(
    map_data: Dict[str, Dict[str, Union[str, float, None]]]
) -> plotly.graph_objects.Figure:

    hierarchy_to_coarse, hierarchy_to_fine = compute_distance(map_data)
    what_to_consider = {"hierarchy"}
    if hierarchy_to_coarse > 30:
        what_to_consider.add("coarse")
    if hierarchy_to_fine > 30:
        what_to_consider.add("fine")

    size_map = {"hierarchy": 3, "fine": 1, "coarse": 1}
    lat_long_data = []
    for subdivision, location_data in map_data.items():
        if subdivision in what_to_consider:
            lat_long_data.append(
                [
                    subdivision,
                    float(location_data["latitude"]),
                    float(location_data["longitude"]),
                    location_data["location"],
                    size_map[subdivision],
                ]
            )

    map_df = pd.DataFrame(
        lat_long_data,
        columns=["subdivision", "latitude", "longitude", "location", "size"],
    )

    px.set_mapbox_access_token(MAPBOX_TOKEN)
    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        hover_name="location",
        hover_data=["latitude", "longitude", "subdivision"],
        color="subdivision",
        color_discrete_map={
            "hierarchy": "fuchsia",
            "coarse": "blue",
            "fine": "blue",
        },
        zoom=3,
        height=500,
        size="size",
    )

    fig.update_layout(mapbox_style="dark")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def gradio_error():
    raise gr.Error("Unable to detect the location!")


def get_outputs(
    data: Dict[str, Dict[str, Union[str, float, None]]]
) -> Tuple[str, str, plotly.graph_objects.Figure]:
    if data is None:
        gradio_error()

    location, latitude, longitude = (
        data["hierarchy"]["location"],
        data["hierarchy"]["latitude"],
        data["hierarchy"]["longitude"],
    )
    if location is None:
        gradio_error()

    return (
        location,
        f"{latitude},{longitude}",
        get_plotly_graph(map_data=data),
    )


def image_gradio(img_file: str) -> Tuple[str, str, plotly.graph_objects.Figure]:
    # data = json.loads(
    #     requests.post(
    #         f"{URL}predict-image",
    #         files={
    #             "image": (
    #                 img_file,
    #                 open(img_file, "rb"),
    #                 mimetypes.guess_type(img_file)[0],
    #             )
    #         },
    #     ).text
    # )
    with open(img_file, "rb") as image_file:
        image_bytes = BytesIO(image_file.read())

    data = json.loads(
        banana.run(
            BANANA_API_KEY,
            BANANA_MODEL_KEY,
            {
                "image": base64.b64encode(image_bytes.getvalue()).decode("utf-8"),
                "filename": os.path.basename(img_file),
            },
        )["modelOutputs"][0]
    )

    return get_outputs(data=data)


# def video_gradio(video_file: str) -> Tuple[str, str, plotly.graph_objects.Figure]:
#     # data = json.loads(
#     #     requests.post(
#     #         f"{URL}predict-video",
#     #         files={
#     #             "video": (
#     #                 video_file,
#     #                 open(video_file, "rb"),
#     #                 "application/octet-stream",
#     #             )
#     #         },
#     #     ).text
#     # )

#     data = json.loads(
#         banana.run(
#             BANANA_API_KEY,
#             BANANA_MODEL_KEY,
#             {
#                 "video": video_file,
#                 "filename": os.path.basename(video_file),
#             },
#         )["modelOutputs"][0]
#     )

#     return get_outputs(data=data)


def url_gradio(url: str) -> Tuple[str, str, plotly.graph_objects.Figure]:
    # data = json.loads(
    #     requests.post(
    #         f"{URL}predict-url",
    #         headers={"content-type": "text/plain"},
    #         data=url,
    #     ).text
    # )
    data = json.loads(
        banana.run(BANANA_API_KEY, BANANA_MODEL_KEY, {"url": url},)[
            "modelOutputs"
        ][0]
    )

    return get_outputs(data=data)


with gr.Blocks() as demo:
    gr.Markdown("# GeoLocator")
    gr.Markdown(
        "### An app that guesses the location of an image 🌌 or a YouTube link 🔗."
    )
    with gr.Tab("Image"):
        with gr.Row():
            img_input = gr.Image(type="filepath", label="Image")
            with gr.Column():
                img_text_output = gr.Textbox(label="Location")
                img_coordinates = gr.Textbox(label="Coordinates")
                img_plot = gr.Plot()
        img_text_button = gr.Button("Go locate!")
        with gr.Row():
            # Flag button
            img_flag_button = gr.Button("Flag this output")
        gr.Examples(examples["images"], inputs=[img_input])
    # with gr.Tab("Video"):
    #     with gr.Row():
    #         video_input = gr.Video(type="filepath", label="Video")
    #         with gr.Column():
    #             video_text_output = gr.Textbox(label="Location")
    #             video_coordinates = gr.Textbox(label="Coordinates")
    #             video_plot = gr.Plot()
    #     video_text_button = gr.Button("Go locate!")
    #     gr.Examples(examples["videos"], inputs=[video_input])
    with gr.Tab("YouTube Link"):
        with gr.Row():
            url_input = gr.Textbox(label="Link")
            with gr.Column():
                url_text_output = gr.Textbox(label="Location")
                url_coordinates = gr.Textbox(label="Coordinates")
                url_plot = gr.Plot()
        url_text_button = gr.Button("Go locate!")
        gr.Examples(examples["video_urls"], inputs=[url_input])

    # Gantry flagging for image #
    callback = GantryImageToTextLogger(application=GANTRY_APP_NAME, api_key=GANTRY_KEY)

    callback.setup(
        components=[img_input, img_text_output],
        flagging_dir=make_unique_bucket_name(prefix=GANTRY_APP_NAME, seed="420"),
    )

    img_flag_button.click(
        fn=lambda *args: callback.flag(args),
        inputs=[img_input, img_text_output, img_coordinates],
        outputs=None,
        preprocess=False,
    )
    ###################

    img_text_button.click(
        image_gradio,
        inputs=img_input,
        outputs=[img_text_output, img_coordinates, img_plot],
    )
    # video_text_button.click(
    #     video_gradio,
    #     inputs=video_input,
    #     outputs=[video_text_output, video_coordinates, video_plot],
    # )
    url_text_button.click(
        url_gradio,
        inputs=url_input,
        outputs=[url_text_output, url_coordinates, url_plot],
    )

    gr.Markdown(
        "Check out the [GitHub repository](https://github.com/samhita-alla/geolocator) that this demo is based off of."
    )

demo.launch()
