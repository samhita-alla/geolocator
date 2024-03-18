"""
Tests the Bento Service API
"""

import argparse
import mimetypes

import requests

image_file_name = "../gradio/data/test/images/greece.jpg"
video_file_name = "../gradio/data/test/videos/newyork.mp4"


def generate_predictions(args):
    url = args.url
    generate_image_prediction(url)
    generate_video_prediction(url)
    generate_url_prediction(url)


def generate_image_prediction(url):
    print(
        requests.post(
            f"{url}predict-image",
            files={
                "image": (
                    image_file_name,
                    open(image_file_name, "rb"),
                    mimetypes.guess_type(image_file_name)[0],
                )
            },
        ).text
    )


def generate_video_prediction(url):
    print(
        requests.post(
            f"{url}predict-video",
            files={
                "video": (
                    video_file_name,
                    open(video_file_name, "rb"),
                    "application/octet-stream",
                )
            },
        ).text
    )


def generate_url_prediction(url):
    print(
        requests.post(
            f"{url}predict-url",
            headers={"content-type": "text/plain"},
            data="https://www.youtube.com/watch?v=ADt1LnbL2HI",
        ).text
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, type=str, help="API endpoint URL")
    args = parser.parse_args()
    generate_predictions(args=args)
