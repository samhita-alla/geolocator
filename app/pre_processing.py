from __future__ import unicode_literals

from typing import Any, Dict, Tuple

import youtube_dl
from Katna.config import Video as VideoConfig

NUMBER_OF_FRAMES = 20
MAX_FILESIZE = 10000000
SELECTED_FRAMES_DIRECTORY = "selected-frames"


def sort_key(key):
    file_size = key["filesize"]
    if file_size:
        return int(key["filesize"])
    return 0


def validate_extension(selected_format: Dict[str, Any]) -> str:
    extension = selected_format.get("ext")
    if extension and extension not in map(
        lambda x: x.replace(".", ""), VideoConfig.video_extensions
    ):
        raise ValueError(f"{extension} isn't supported.")
    return extension


def extract_youtube_video(url: str) -> Tuple[str, Dict[str, Any]]:
    ydl = youtube_dl.YoutubeDL({})

    # extra information about the video
    info_dict = ydl.extract_info(url, download=False)
    formats = info_dict.get("formats", [])

    # sort the formats in descending order w.r.t the file size
    sorted_formats = sorted(formats, key=sort_key, reverse=True)

    # remove "webm" formatted videos
    filtered_sorted_formats = list(filter(lambda x: x["ext"] != "webm", sorted_formats))

    # select the best format -- the nearest big number to MAX_FILESIZE
    selected_format = {}
    for format in filtered_sorted_formats:
        file_size = format["filesize"]
        if file_size and file_size < MAX_FILESIZE and format["vcodec"] != "none":
            selected_format = format
            break

    # verify if the extension is valid
    extension = validate_extension(selected_format)

    # extract YT video
    videos_path = "videos"
    ydl_opts = {
        "max_filesize": MAX_FILESIZE,
        "format": selected_format.get("format_id"),
        "outtmpl": f"{videos_path}/%(id)s.%(ext)s",
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.cache.remove()
        ydl.download([url])
        saved_location = f"videos/{info_dict['id']}.{extension}"

    return saved_location, info_dict
