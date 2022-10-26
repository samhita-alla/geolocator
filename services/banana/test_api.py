"""
Tests the Bento Service API
"""
import argparse
import base64
import mimetypes
import os
from io import BytesIO

import requests
from test_banana_api import _upload_video_to_s3

image_file_name = "../gradio/data/test/images/greece.jpg"
video_file_name = "../gradio/data/test/videos/newyork.mp4"


def generate_predictions(args):
    url = args.url
    # generate_image_prediction(url)
    generate_video_prediction(url)
    # generate_url_prediction(url)


def generate_image_prediction(url):
    with open(image_file_name, "rb") as image_file:
        image_bytes = BytesIO(image_file.read())

    print(
        requests.post(
            url,
            json={
                "image": base64.b64encode(image_bytes.getvalue()).decode("utf-8"),
                "filename": os.path.basename(image_file_name),
            },
        ).text
    )


def generate_video_prediction(url):
    with open(video_file_name, "rb") as video_file:
        video_b64_string = base64.b64encode(
            BytesIO(video_file.read()).getvalue()
        ).decode("utf8")

    video_mime = mimetypes.guess_type(video_file_name)[0]

    s3_uri = _upload_video_to_s3(f"data:{video_mime};base64," + video_b64_string)

    print(
        requests.post(
            url,
            json={
                "video": s3_uri,
                "filename": os.path.basename(video_file_name),
            },
        ).text
    )


def generate_url_prediction(url):
    print(
        requests.post(
            url,
            headers={"content-type": "text/plain"},
            json={"url": "https://www.youtube.com/watch?v=ADt1LnbL2HI"},
        ).text
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, type=str, help="API endpoint URL")
    args = parser.parse_args()
    generate_predictions(args=args)
