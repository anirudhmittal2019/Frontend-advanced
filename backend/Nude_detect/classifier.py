# Simple content classifier for inappropriate content detection
# Supports both image and video processing

import os
import argparse
import cv2
import keras
import pydload
import logging
import numpy as np
from PIL import Image as pil_image

from utils.video_utils import get_interest_frames_from_video
from config import cls_model_path

logging.basicConfig(level=logging.DEBUG)

# Basic PIL setup
if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        "nearest": pil_image.NEAREST,
        "bilinear": pil_image.BILINEAR,
        "bicubic": pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, "HAMMING"):
        _PIL_INTERPOLATION_METHODS["hamming"] = pil_image.HAMMING
    if hasattr(pil_image, "BOX"):
        _PIL_INTERPOLATION_METHODS["box"] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, "LANCZOS"):
        _PIL_INTERPOLATION_METHODS["lanczos"] = pil_image.LANCZOS

def load_img(
    path, grayscale=False, color_mode="rgb", target_size=None, interpolation="nearest"
):
    # Basic image loader
    # Takes image path and returns processed PIL image
    if grayscale is True:
        logging.warn(
            "grayscale is deprecated. Please use " 'color_mode = "grayscale"')
        color_mode = "grayscale"
    if pil_image is None:
        raise ImportError(
            "Could not import PIL.Image. " "The use of `load_img` requires PIL."
        )

    if isinstance(path, type("")):
        img = pil_image.open(path)
    else:
        path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
        img = pil_image.fromarray(path)

    if color_mode == "grayscale":
        if img.mode != "L":
            img = img.convert("L")
    elif color_mode == "rgba":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    elif color_mode == "rgb":
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    "Invalid interpolation method {} specified. Supported "
                    "methods are {}".format(
                        interpolation, ", ".join(
                            _PIL_INTERPOLATION_METHODS.keys())
                    )
                )
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img

def load_images(image_paths, image_size, image_names):
    # Batch image processor
    # Returns numpy array of processed images
    loaded_images = []
    loaded_image_paths = []

    for i, img_path in enumerate(image_paths):
        try:
            image = load_img(img_path, target_size=image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(image_names[i])
        except Exception as ex:
            logging.exception("Error reading {} {}".format(
                img_path, ex), exc_info=True)

    return np.asarray(loaded_images), loaded_image_paths

class Classifier(object):
    # Main classifier for content checking
    nsfw_model = None

    def __init__(self, cls_model_path):
        # Load model from path
        model_path = cls_model_path

        if not os.path.exists(model_path):
            raise Exception(
                "Please Downloading the checkpoint before using", model_path)

        self.nsfw_model = keras.models.load_model(model_path)

    def classify_video(
        self,
        video_path,
        batch_size=4,
        image_size=(256, 256),
        categories=["unsafe", "safe"],
    ):
        # Process video frame by frame
        frame_indices = None
        frame_indices, frames, fps, video_length = get_interest_frames_from_video(
            video_path
        )
        logging.debug(
            "VIDEO_PATH: {}, FPS: {}, Important frame indices: {}, Video length: {}".format(
                video_path, fps, frame_indices, video_length)
        )

        frames, frame_names = load_images(
            frames, image_size, image_names=frame_indices)

        if not frame_names:
            return {}

        model_preds = self.nsfw_model.predict(frames, batch_size=batch_size)
        preds = np.argsort(model_preds, axis=1).tolist()

        probs = []
        for i, single_preds in enumerate(preds):
            single_probs = []
            for j, pred in enumerate(single_preds):
                single_probs.append(model_preds[i][pred])
                preds[i][j] = categories[pred]

            probs.append(single_probs)

        return_preds = {
            "metadata": {
                "fps": fps,
                "video_length": video_length,
                "video_path": video_path,
            },
            "preds": {},
        }

        for i, frame_name in enumerate(frame_names):
            return_preds["preds"][frame_name] = {}
            for _ in range(len(preds[i])):
                return_preds["preds"][frame_name][preds[i][_]] = probs[i][_]

        return return_preds

    def classify(
        self,
        image_paths=[],
        batch_size=4,
        image_size=(256, 256),
        categories=["unsafe", "safe"],
    ):
        # Process single or multiple images
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        loaded_images, loaded_image_paths = load_images(
            image_paths, image_size, image_names=image_paths
        )

        if not loaded_image_paths:
            return {}

        model_preds = self.nsfw_model.predict(
            loaded_images, batch_size=batch_size
        )

        preds = np.argsort(model_preds, axis=1).tolist()

        probs = []
        for i, single_preds in enumerate(preds):
            single_probs = []
            for j, pred in enumerate(single_preds):
                single_probs.append(model_preds[i][pred])
                preds[i][j] = categories[pred]

            probs.append(single_probs)

        images_preds = {}

        for i, loaded_image_path in enumerate(loaded_image_paths):
            images_preds[loaded_image_path] = {}
            for _ in range(len(preds[i])):
                images_preds[loaded_image_path][preds[i][_]] = probs[i][_]

        return images_preds

# Quick test functions
def clf_images():
    # Test image classification
    m = Classifier(cls_model_path)
    unsafe_img = "./data/image/nude/0B6FE142-67A9-4451-B977-6E22C0EC12D7.jpg"
    safe_img = "./data/image/normal/030u5wa70z6z.jpg"
    img_path_1 = "./data/image/nude/0D16FBAD-655B-440C-A7E1-32D20408DF40.jpg"  # real

    images_preds = m.classify([unsafe_img, safe_img, img_path_1])

    print("# preds -------")
    for k, v in images_preds.items():
        print(k, v)

def clf_video():
    # Test video classification
    m = Classifier(cls_model_path)
    video_path = "./data/video/123.mp4"

    result = m.classify_video(video_path)
    metadata_info = result["metadata"]
    preds_info = result["preds"]

    print("# metadata -------")
    for k, v in metadata_info.items():
        print(k, v)

    print("# preds -------")
    for k, v in preds_info.items():
        print(k, v)

def clf():
    # Interactive test mode
    m = Classifier(cls_model_path)

    while 1:
        print(
            "\n Enter single image path or multiple images seperated by || (2 pipes) \n"
        )
        images = input().split("||")
        images = [image.strip() for image in images]
        print(m.classify(images), "\n")

# Run tests
if __name__ == "__main__":
    clf_video()