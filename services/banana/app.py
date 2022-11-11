import base64
import glob
import io
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import boto3
import onnxruntime
import pandas as pd
import PIL.Image
import torch
import torchvision

# from dotenv import load_dotenv
from tqdm.auto import tqdm

sys.path.append("../../app/")

from classification.train_base import MultiPartitioningClassifier  # noqa: E402
from post_processing import generate_prediction_logit  # noqa: E402
from pre_processing import extract_youtube_video  # noqa: E402

# load_dotenv()

IMAGE_PARENT_DIR = "geolocator-images"
VERSION = "latest"
VIDEOS_DIRECTORY = "videos"
BATCH_SIZE = 13
NUM_OF_WORKERS = 0
SELECTED_FRAMES_DIRECTORY = "selected-frames"

#######################
# HELPER FUNCTIONS ####
#######################


class FiveCropImageDataset(torch.utils.data.Dataset):
    """
    Data Preprocessor
    """

    def __init__(
        self,
        meta_csv: Union[str, Path, None],
        image_dir: Union[str, Path],
        img_id_col: Union[str, int] = "img_id",
        allowed_extensions: List[str] = ["jpg", "jpeg", "png"],
    ):
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        self.image_dir = image_dir
        self.img_id_col = img_id_col
        self.meta_info = None
        if meta_csv is not None:
            print(f"Read {meta_csv}")
            self.meta_info = pd.read_csv(meta_csv)
            self.meta_info.columns = map(str.lower, self.meta_info.columns)
            # rename column names if necessary to use existing data
            if "lat" in self.meta_info.columns:
                self.meta_info.rename(columns={"lat": "latitude"}, inplace=True)
            if "lon" in self.meta_info.columns:
                self.meta_info.rename(columns={"lon": "longitude"}, inplace=True)
            self.meta_info["img_path"] = self.meta_info[img_id_col].apply(
                lambda img_id: str(self.image_dir / img_id)
            )
        else:
            image_files = []
            for ext in allowed_extensions:
                image_files.extend([str(p) for p in self.image_dir.glob(f"**/*.{ext}")])
            self.meta_info = pd.DataFrame(image_files, columns=["img_path"])
            self.meta_info[self.img_id_col] = self.meta_info["img_path"].apply(
                lambda x: Path(x).stem
            )
        self.tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self):
        return len(self.meta_info.index)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        meta = self.meta_info.iloc[idx]
        meta = meta.to_dict()
        meta["img_id"] = meta[self.img_id_col]

        image = PIL.Image.open(meta["img_path"]).convert("RGB")
        image = torchvision.transforms.Resize(256)(image)
        crops = torchvision.transforms.FiveCrop(224)(image)
        crops_transformed = []
        for crop in crops:
            crops_transformed.append(self.tfm(crop))
        return torch.stack(crops_transformed, dim=0), meta


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def create_image_dir(img_data: str, filename: str) -> str:
    unique_string = str(uuid.uuid4())
    image_dir = os.path.join(IMAGE_PARENT_DIR, os.path.basename(unique_string))

    os.makedirs(image_dir)

    image_bytes = io.BytesIO(base64.b64decode(img_data.encode("utf-8")))
    with open(f"{image_dir}/{unique_string}-{filename}", "wb") as image_file:
        image_file.write(image_bytes.getbuffer())

    return image_dir


def img_processor(img_data: str, filename: str) -> str:
    image_dir = create_image_dir(img_data=img_data, filename=filename)
    return image_dir


def video_helper(video_file: str, info_dict: Dict[str, Any]) -> str:
    frames_directory = f"{SELECTED_FRAMES_DIRECTORY}/{info_dict['id'].split('.')[0]}"
    subprocess.run(
        [
            "python",
            "capture_video_frames.py",
            "--video",
            video_file,
            "--dir",
            frames_directory,
        ],
    )
    return frames_directory


def download_object(bucket, remote_file, local_file):
    s3_client = boto3.client("s3")
    s3_client.download_file(bucket, remote_file, local_file)


def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    sub_part = s3_uri.replace("s3://", "").split("/")
    bucket = sub_part.pop(0)
    key = "/".join(sub_part)

    return bucket, key


def video_processor(video_file: str, filename: str) -> str:
    os.makedirs(VIDEOS_DIRECTORY, exist_ok=True)

    unique_string = str(uuid.uuid4())
    video_file_name = f"{VIDEOS_DIRECTORY}/{unique_string}-{filename}"

    bucket, key = parse_s3_uri(video_file)

    download_object(bucket, key, video_file_name)

    info_dict = {"id": os.path.basename(video_file_name)}

    return video_helper(video_file=video_file_name, info_dict=info_dict)


def url_processor(url: str) -> str:
    video_file, info_dict = extract_youtube_video(url=url)
    return video_helper(video_file=video_file, info_dict=info_dict)


classifier = MultiPartitioningClassifier(
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
classifier.eval()


def delete_object(bucket, remote_file):
    s3_client = boto3.client("s3")
    s3_client.delete_object(Bucket=bucket, Key=remote_file)


def predict_helper(image_dir: str, metadata: str) -> Union[dict, None]:
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
        ort_inputs = {model.get_inputs()[0].name: to_numpy(images)}

        # ONNX Runtime will return a list of outputs
        yhats_numpy = model.run(None, ort_inputs)

        # convert numpy arrays to tensors
        yhats = [torch.from_numpy(numpy_array) for numpy_array in yhats_numpy]

        # post-processing logic
        # logits courtesy: @yiyixuxu
        yhats, hierarchy_preds = classifier._multi_crop_inference_helper(
            cur_batch_size, ncrops, yhats
        )
        (
            pred_classes,
            pred_latitudes,
            pred_longitudes,
            pred_logits,
        ) = classifier.inference_helper(yhats, hierarchy_preds)

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

    # get the location
    result = generate_prediction_logit(inference_df=geolocator_df)

    # clear up the image_dir and downloaded videos
    shutil.rmtree(image_dir, ignore_errors=True)
    if metadata in ["video", "url"]:
        files = glob.glob(
            os.path.join(
                VIDEOS_DIRECTORY,
                image_dir.split(SELECTED_FRAMES_DIRECTORY + "/")[1].split(".")[0]
                + ".*",
            )
        )
        for each_file in files:
            os.remove(each_file)

    return result


###########################
# HELPER FUNCTIONS END ####
###########################

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    device = 0 if torch.cuda.is_available() else -1

    model = onnxruntime.InferenceSession(
        "geolocator.onnx",
        providers=["CUDAExecutionProvider" if device == 0 else "CPUExecutionProvider"],
    )


def inference_image(model_inputs: dict) -> Union[dict, None]:
    # Parse out your arguments
    image = model_inputs.get("image", None)
    if image is None:
        return {"message": "No image provided"}
    filename = model_inputs.get("filename", None)

    image_dir = img_processor(img_data=image, filename=filename)
    return predict_helper(image_dir=image_dir, metadata="image")


def inference_video(model_inputs: dict) -> Union[dict, None]:
    # Parse out your arguments
    video = model_inputs.get("video", None)
    if video is None:
        return {"message": "No video provided"}
    filename = model_inputs.get("filename", None)

    image_dir = video_processor(video_file=video, filename=filename)

    # delete uploaded video
    bucket, key = parse_s3_uri(video)
    delete_object(bucket, key)

    return predict_helper(image_dir=image_dir, metadata="video")


def inference_url(model_inputs: dict) -> Union[dict, None]:
    # Parse out your arguments
    url = model_inputs.get("url", None)
    if url is None:
        return {"message": "No url provided"}

    image_dir = url_processor(url=url)
    return predict_helper(image_dir=image_dir, metadata="url")
