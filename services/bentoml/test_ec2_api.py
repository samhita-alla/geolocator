"""
Tests the Bento Service API
"""
import mimetypes

import requests

image_file_name = "test_data/santorini-island-greece-santorini-island-greece-oia-town-traditional-white-houses-churches-blue-domes-over-caldera-146011399.jpg"
video_file_name = "test_data/newyork_video.mp4"
url = "https://0cclq8rfj5.execute-api.us-east-2.amazonaws.com/"

# image
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

# video
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

# url
print(
    requests.post(
        f"{url}predict-url",
        headers={"content-type": "text/plain"},
        data="https://www.youtube.com/watch?v=ADt1LnbL2HI",
    ).text
)
