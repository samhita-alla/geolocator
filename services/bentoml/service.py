"""
Bento Service Definition
"""
from __future__ import annotations

import io
import os
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import bentoml
import numpy as np
import pandas as pd
import torch
from bentoml.io import File, Image, Text
from classification.dataset import FiveCropImageDataset
from classification.train_base import MultiPartitioningClassifier
from post_processing import generate_prediction_logit
from pre_processing import capture_frames, extract_youtube_video
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

IMAGE_PARENT_DIR = "geolocator-images"
ONNX_MODEL = "onnx_geolocator"
VERSION = "latest"
BATCH_SIZE = 13
NUM_OF_WORKERS = 0


def create_image_dir(img_data: Image) -> str:
    unique_string = str(uuid.uuid4())
    image_dir = os.path.join(IMAGE_PARENT_DIR, os.path.basename(unique_string))

    os.makedirs(image_dir)
    img_data.save(
        f"{image_dir}/{unique_string}.{(img_data.format).lower()}",
        img_data.format,
    )

    return image_dir


def img_processor(img_data: Image) -> str:
    image_dir = create_image_dir(img_data=img_data)
    return image_dir


def video_helper(video_file: str, info_dict: Dict[str, Any]) -> str:
    frames_directory = capture_frames(video_file_path=video_file, info_dict=info_dict)
    return frames_directory


def video_processor(video_file: io.BytesIO[Any]) -> str:
    video_file_name = os.path.basename(video_file._name)
    with open(video_file_name, "wb") as outfile:
        outfile.write(video_file.read())

    info_dict = {"id": os.path.basename(video_file_name)}
    return video_helper(video_file=video_file_name, info_dict=info_dict)


def url_processor(url: str) -> str:
    video_file, info_dict = extract_youtube_video(url=url)
    return video_helper(video_file=video_file, info_dict=info_dict)


model = MultiPartitioningClassifier(
    hparams={
        "partitionings": {
            "shortnames": ["coarse", "middle", "fine"],
            "files": [
                "resources/s2_cells/cells_50_5000.csv",
                "resources/s2_cells/cells_50_2000.csv",
                "resources/s2_cells/cells_50_1000.csv",
            ],
        },
    },
    build_model=False,
)
model.eval()


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


class GeoLocatorRunnable(bentoml.Runnable):
    """
    Custom BentoML runner to fetch multiple ONNX outputs
    """

    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        super().__init__()

        # load the model instance
        self.model = bentoml.onnx.load_model(f"{ONNX_MODEL}:{VERSION}")

    @bentoml.Runnable.method(batchable=False)
    def fetch_multiple_onnx_outputs(self, images) -> List[np.ndarray]:
        ort_inputs = {self.model.get_inputs()[0].name: to_numpy(images)}
        ort_outs = self.model.run(None, ort_inputs)
        return ort_outs


# initialize custom bentoml runner, svc and input spec
geolocator_runner = bentoml.Runner(
    GeoLocatorRunnable,
    models=[bentoml.onnx.get(f"{ONNX_MODEL}:{VERSION}")],
)
svc = bentoml.Service("geolocator", runners=[geolocator_runner])


def predict_helper(image_dir: str) -> str:
    dataloader = torch.utils.data.DataLoader(
        FiveCropImageDataset(meta_csv=None, image_dir=image_dir),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_OF_WORKERS,
    )

    rows = []
    for batch in tqdm(dataloader):
        images, meta_batch = batch
        cur_batch_size = images.shape[0]
        ncrops = images.shape[1]

        # reshape crop dimension to batch
        images = torch.reshape(images, (cur_batch_size * ncrops, *images.shape[2:]))

        # fetch predictions
        yhats_numpy = geolocator_runner.fetch_multiple_onnx_outputs.run(images)

        # convert numpy arrays to tensors
        yhats = [torch.from_numpy(numpy_array) for numpy_array in yhats_numpy]

        # post-processing logic
        # logits courtesy: @yiyixuxu
        yhats, hierarchy_preds = model._multi_crop_inference_helper(
            cur_batch_size, ncrops, yhats
        )
        (
            pred_classes,
            pred_latitudes,
            pred_longitudes,
            pred_logits,
        ) = model.inference_helper(yhats, hierarchy_preds)

        img_paths = meta_batch["img_path"]

        for p_key in pred_classes.keys():
            for img_path, pred_class, pred_lat, pred_lng, pred_logit in zip(
                img_paths,
                pred_classes[p_key].cpu().numpy(),
                pred_latitudes[p_key].cpu().numpy(),
                pred_longitudes[p_key].cpu().numpy(),
                pred_logits[p_key].cpu().numpy(),
            ):
                rows.append(
                    {
                        "img_id": Path(img_path).stem,
                        "p_key": p_key,
                        "pred_class": pred_class,
                        "pred_lat": pred_lat,
                        "pred_lng": pred_lng,
                        "pred_logit": pred_logit,
                    }
                )

    geolocator_df = pd.DataFrame.from_records(rows)
    geolocator_df.set_index(keys=["img_id", "p_key"], inplace=True)

    # get the location
    location, *_ = generate_prediction_logit(inference_df=geolocator_df)

    # clear up the image directory -- memory optimization
    shutil.rmtree(image_dir, ignore_errors=True)
    return location


@svc.api(input=Image(), output=Text(), route="predict-image")
def predict_image(image: PILImage) -> str:
    image_dir = img_processor(img_data=image)
    return predict_helper(image_dir=image_dir)


@svc.api(input=File(), output=Text(), route="predict-video")
def predict_video(video: io.BytesIO[Any]) -> str:
    image_dir = video_processor(video_file=video)
    return predict_helper(image_dir=image_dir)


@svc.api(input=Text(), output=Text(), route="predict-url")
def predict_url(url: str) -> str:
    image_dir = url_processor(url=url)
    return predict_helper(image_dir=image_dir)
