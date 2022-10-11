# GantryImageToTextLogger.py

"""
class to handle flagging in gradio to gantry
"""

import os
from typing import List, Optional, Union

import gantry
import gradio as gr
from gradio.components import Component
from smart_open import open


class GantryImageToTextLogger(gr.FlaggingCallback):
    """
    A FlaggingCallback that logs flagged image-to-text data to Gantry via S3.
    Originally written by FSDL educators https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022/blob/main/app_gradio/flagging.py
    and adjusted for the geolocator project
    """

    def __init__(
        self,
        application: str,
        version: Union[int, str, None] = None,
        api_key: Optional[str] = None,
    ):
        """Logs image-to-text data that was flagged in Gradio to Gantry.

        Images are logged to Amazon Web Services' Simple Storage Service (S3).

        The flagging_dir provided to the Gradio interface is used to set the
        name of the bucket on S3 into which images are logged.

        See the following tutorial by Dan Bader for a quick overview of S3 and the AWS SDK
        for Python, boto3: https://realpython.com/python-boto3-aws-s3/

        See https://gradio.app/docs/#flagging for details on how
        flagging data is handled by Gradio.

        See https://docs.gantry.io for information about logging data to Gantry.

        Parameters
        ----------
        application
            The name of the application on Gantry to which flagged data should be uploaded.
            Gantry validates and monitors data per application.
        version
            The schema version to use during validation by Gantry. If not provided, Gantry
            will use the latest version. A new version will be created if the provided version
            does not exist yet.
        api_key
            Optionally, provide your Gantry API key here. Provided for convenience
            when testing and developing locally or in notebooks. The API key can
            alternatively be provided via the GANTRY_API_KEY environment variable.
        """
        self.application = application
        self.version = version
        gantry.init(api_key=api_key)

    def setup(self, components: List[Component], flagging_dir: str):
        """Sets up the GantryImageToTextLogger by creating or attaching to an S3 Bucket."""
        self._counter = 0
        self.bucket = get_or_create_bucket(flagging_dir)
        enable_bucket_versioning(self.bucket)
        add_access_policy(self.bucket)
        (
            self.image_component_idx,
            self.text_component_idx,
            self.text_component2_idx,
        ) = self._find_image_video_and_text_components(components)

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None) -> int:
        """Sends flagged outputs and feedback to Gantry and image inputs to S3."""

        image = flag_data[self.image_component_idx]
        text = flag_data[self.text_component_idx]
        text2 = flag_data[self.text_component2_idx]

        feedback = {"flag": flag_option}
        if username is not None:
            feedback["user"] = username

        data_type, image_buffer = read_b64_string(image, return_data_type=True)
        image_url = self._to_s3(image_buffer.read(), filetype=data_type)

        self._to_gantry(
            input_image_url=image_url,
            pred_location=text,
            pred_coordinates=text2,
            feedback=feedback,
        )
        self._counter += 1
        display("Flagging response has been succesfully sent to gantry.io!")

        return self._counter

    def _to_gantry(self, input_image_url, pred_location, pred_coordinates, feedback):
        inputs = {"image": input_image_url}
        outputs = {"location": pred_location, "coordinates": pred_coordinates}

        gantry.log_record(
            self.application,
            self.version,
            inputs=inputs,
            outputs=outputs,
            feedback=feedback,
        )

    def _to_s3(self, image_bytes, key=None, filetype=None):
        if key is None:
            key = make_key(image_bytes, filetype=filetype)

        s3_uri = get_uri_of(self.bucket, key)

        with open(s3_uri, "wb") as s3_object:
            s3_object.write(image_bytes)

        return s3_uri

    def _find_image_video_and_text_components(self, components: List[Component]):
        """
        Manual indexing of images and text components
        """

        image_component_idx = 0
        text_component_idx = 1
        text_component2_idx = 2

        return (
            image_component_idx,
            text_component_idx,
            text_component2_idx,
        )


def get_api_key() -> Optional[str]:
    """Convenience method for fetching the Gantry API key."""
    api_key = os.environ.get("GANTRY_API_KEY")
    return api_key
