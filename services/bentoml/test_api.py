"""
Tests the Bento Service API
"""
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import mimetypes

image_file_name = "test_data/santorini-island-greece-santorini-island-greece-oia-town-traditional-white-houses-churches-blue-domes-over-caldera-146011399.jpg"
video_file_name = "test_data/newyork_video.mp4"

for metadata in ["image", "video", "url"]:
    m = MultipartEncoder(
        fields={
            "url": "https://www.youtube.com/watch?v=ADt1LnbL2HI",
            "image": (
                image_file_name,
                open(image_file_name, "rb"),
                mimetypes.guess_type(image_file_name)[0],
            ),
            "video": (
                video_file_name,
                open(video_file_name, "rb"),
                "application/octet-stream",
            ),
            "metadata": metadata,
        }
    )

    print(
        requests.post(
            "http://127.0.0.1:3000/predict",
            headers={"Content-Type": m.content_type},
            data=m,
        ).text
    )
