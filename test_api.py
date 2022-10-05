"""
Tests the Bento Service API
"""
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

m = MultipartEncoder(
    fields={
        "data": "https://www.youtube.com/watch?v=ADt1LnbL2HI",
        "metadata": "url",
    }
)

print(
    requests.post(
        "http://127.0.0.1:3000/predict",
        headers={"Content-Type": m.content_type},
        data=m,
    ).text
)
