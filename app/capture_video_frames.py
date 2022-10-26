import argparse
import os
import time

from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter

NUMBER_OF_FRAMES = 20
SELECTED_FRAMES_DIRECTORY = "selected-frames"


def capture_frames(video_file_path: str, frames_directory: str) -> None:
    # create a directory to store video frames
    os.makedirs(frames_directory, exist_ok=True)
    diskwriter = KeyFrameDiskWriter(location=frames_directory)

    vd = Video()
    start = time.time()
    try:
        vd.extract_video_keyframes(
            no_of_frames=NUMBER_OF_FRAMES, file_path=video_file_path, writer=diskwriter
        )
    except Exception as e:
        raise ValueError(f"Error capturing the frames: {e}")
    print(f"end: {time.time() - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Video file")
    parser.add_argument("--dir", type=str, help="Frames directory")
    args = parser.parse_args()
    capture_frames(args.video, args.dir)
