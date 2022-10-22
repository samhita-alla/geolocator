import base64
import os
from io import BytesIO

import banana_dev as banana
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BANANA_API_KEY")
model_key = os.getenv("BANANA_MODEL_KEY")

image_file_name = "../gradio/data/test/images/greece.jpg"
video_file_name = "../gradio/data/test/videos/newyork.mp4"


def generate_image_prediction():
    with open(image_file_name, "rb") as image_file:
        image_bytes = BytesIO(image_file.read())

    print(
        banana.run(
            api_key,
            model_key,
            {
                "image": base64.b64encode(image_bytes.getvalue()).decode("utf-8"),
                "filename": os.path.basename(image_file_name),
            },
        )
    )


def generate_video_prediction():
    with open(video_file_name, "rb") as video_file:
        video_bytes = BytesIO(video_file.read())

    print(
        banana.run(
            api_key,
            model_key,
            {
                "video": base64.b64encode(video_bytes.getvalue()).decode("utf-8"),
                "filename": os.path.basename(video_file_name),
            },
        )
    )


def generate_url_prediction():
    print(
        banana.run(
            api_key, model_key, {"url": "https://www.youtube.com/watch?v=ADt1LnbL2HI"}
        )
    )
