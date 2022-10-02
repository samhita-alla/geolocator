import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import bentoml
import numpy as np
import pandas as pd
import torch
from bentoml.io import Multipart, Text
from tqdm.auto import tqdm

from classification.dataset import FiveCropImageDataset
from classification.train_base import MultiPartitioningClassifier
from post_processing import generate_prediction
from pre_processing import capture_frames, extract_youtube_video

image_parent_dir = "geolocator-images"
ONNX_MODEL = "onnx_geolocator"
VERSION = "latest"


def create_image_dir(img_file: str) -> str:
    image_dir = os.path.join(image_parent_dir, os.path.basename(img_file).split(".")[0])

    # clear the image directory before filling it up
    shutil.rmtree(image_dir, ignore_errors=True)
    os.makedirs(image_dir)
    shutil.copy(img_file, image_dir)

    return image_dir


def img_processor(img_file: str) -> str:
    image_dir = create_image_dir(img_file=img_file)
    return image_dir


def video_helper(video_file: str, info_dict: Dict[str, Any]) -> str:
    # capture frames
    frames_directory = capture_frames(video_file_path=video_file, info_dict=info_dict)
    return frames_directory


def video_processor(video_file: str) -> str:
    info_dict = {"id": os.path.basename(video_file).split(".")[0]}
    return video_helper(video_file=video_file, info_dict=info_dict)


def url_processor(url: str) -> str:
    video_file, info_dict = extract_youtube_video(url=url)
    return video_helper(video_file=video_file, info_dict=info_dict)


model = MultiPartitioningClassifier(
    hparams={
        "partitionings": {
            "shortnames": ["coarse", "middle", "fine"],
            "files": [
                "../resources/s2_cells/cells_50_5000.csv",
                "../resources/s2_cells/cells_50_2000.csv",
                "../resources/s2_cells/cells_50_1000.csv",
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
input_spec = Multipart(data=Text(), metadata=Text())


@svc.api(input=input_spec, output=Text())
def predict(data: str, metadata: str) -> str:
    if metadata == "image":
        image_dir = img_processor(img_file=data)
    elif metadata == "video":
        image_dir = video_processor(video_file=data)
    elif metadata == "url":
        image_dir = url_processor(url=data)

    dataloader = torch.utils.data.DataLoader(
        FiveCropImageDataset(meta_csv=None, image_dir=image_dir),
        batch_size=13,
        shuffle=False,
        num_workers=0,
    )

    rows = []
    for batch in tqdm(dataloader):
        images, meta_batch = batch
        cur_batch_size = images.shape[0]
        ncrops = images.shape[1]

        # reshape crop dimension to batch
        images = torch.reshape(images, (cur_batch_size * ncrops, *images.shape[2:]))

        print(images.shape)

        # fetch predictions
        yhats_numpy = geolocator_runner.fetch_multiple_onnx_outputs.run(images)

        # convert numpy arrays to tensors
        yhats = [torch.from_numpy(numpy_array) for numpy_array in yhats_numpy]

        # post-processing logic
        yhats, hierarchy_preds = model._multi_crop_inference_helper(
            cur_batch_size, ncrops, yhats
        )
        pred_classes, pred_latitudes, pred_longitudes = model.inference_helper(
            yhats, hierarchy_preds
        )

        img_paths = meta_batch["img_path"]

        for p_key in pred_classes.keys():
            for img_path, pred_class, pred_lat, pred_lng in zip(
                img_paths,
                pred_classes[p_key].cpu().numpy(),
                pred_latitudes[p_key].cpu().numpy(),
                pred_longitudes[p_key].cpu().numpy(),
            ):
                rows.append(
                    {
                        "img_id": Path(img_path).stem,
                        "p_key": p_key,
                        "pred_class": pred_class,
                        "pred_lat": pred_lat,
                        "pred_lng": pred_lng,
                    }
                )

    geolocator_df = pd.DataFrame.from_records(rows)
    geolocator_df.set_index(keys=["img_id", "p_key"], inplace=True)

    # get the location
    return generate_prediction(inference_df=geolocator_df)
