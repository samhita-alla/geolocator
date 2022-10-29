import base64
import mimetypes
import os
import sys
from io import BytesIO

import banana_dev as banana
from dotenv import load_dotenv
from smart_open import open as smartopen

sys.path.append("..")

from gantry_callback.s3_util import add_access_policy  # noqa: E402
from gantry_callback.s3_util import (  # noqa: E402
    enable_bucket_versioning,
    get_or_create_bucket,
    get_uri_of,
    make_key,
    make_unique_bucket_name,
)
from gantry_callback.string_img_util import read_b64_string  # noqa: E402

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


def _upload_video_to_s3(video_b64_string):
    bucket = get_or_create_bucket(
        make_unique_bucket_name(prefix="geolocator-app", seed="420")
    )
    enable_bucket_versioning(bucket)
    add_access_policy(bucket)

    data_type, video_buffer = read_b64_string(video_b64_string, return_data_type=True)
    video_bytes = video_buffer.read()
    key = make_key(video_bytes, filetype=data_type)

    s3_uri = get_uri_of(bucket, key)

    with smartopen(s3_uri, "wb") as s3_object:
        s3_object.write(video_bytes)

    return s3_uri


def generate_video_prediction():
    with open(video_file_name, "rb") as video_file:
        video_b64_string = base64.b64encode(
            BytesIO(video_file.read()).getvalue()
        ).decode("utf8")

    video_mime = mimetypes.guess_type(video_file_name)[0]

    s3_uri = _upload_video_to_s3(f"data:{video_mime};base64," + video_b64_string)

    print(
        banana.run(
            api_key,
            model_key,
            json={
                "video": s3_uri,
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


if __name__ == "__main__":
    generate_image_prediction()
    generate_video_prediction()
    generate_url_prediction()
