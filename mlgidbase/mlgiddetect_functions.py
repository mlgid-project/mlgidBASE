from mlgiddetect.inference import Inference
from mlgiddetect.configuration import Config
from mlgiddetect.preprocessing import standard_preprocessing
from mlgiddetect.postprocessing import standard_postprocessing
from mlgiddetect.preprocessing import (preprocess_geometry, contrast_correction, add_batch_and_color_channel,
                                        grayscale_to_color)
from mlgiddetect.dataloader import PyGIDDataset
from mlgiddetect.dataloader import ImageContainer
import numpy as np
from .pygid_functions import read_detected_peaks

def load_config(config):
    if isinstance(config, Config):
        config.PREPROCESSING_LINEAR_CONTRAST = True
        return config
    elif isinstance(config, str):
        config = Config(config)
        config.PREPROCESSING_LINEAR_CONTRAST = True
        return config
    elif config is None:
        config = Config()
        return check_valid_config(config)
    else:
        raise TypeError("Invalid config_detect. It should be a string, None or a Config object.")

def load_inference(config):
    try:
        return Inference(config)
    except:
        raise ValueError("Detection failed. Couldn't load the model.")

def check_valid_config(config):
    config.PREPROCESSING_CUDA = False
    config.PREPROCESSING_FLIPHORIZONTAL = False
    config.PREPROCESSING_QUAZIPOLAR = False
    config.PREPROCESSING_LINEAR_CONTRAST = True
    config.PREPROCESSING_NO_CONTRASTCORRECTION = False
    config.PREPROCESSING_POLAR_SHAPE = [512, 1024]
    config.PREPROCESSING_POLAR_CONVERSION = True
    config.PREPROCESSING_LINEAR_PERC_977 = False
    config.MODEL_TYPE == 'faster_rcnn'
    return config

def run_mlgiddetect(img, q_xy_axes,q_z_axes, imp, config_detect):
    try:
        img_container_detect = ImageContainer()
    except:
        raise ValueError("Detection failed. Couldn't create ImageContainer()")

    try:
        img_container_detect.from_pygid(config_detect, np.nan_to_num(img), q_xy_axes[-1], q_z_axes[-1], 0)
    except:
        raise ValueError("Detection failed. Couldn't load data from NeXus")
    # polar conversion
    try:
        img_container_detect.converted_polar_image, img_container_detect.raw_polar_image, img_container_detect.converted_mask = standard_preprocessing(
        config_detect, img_container_detect.raw_reciprocal)
    except:
        raise ValueError("Detection failed. Couldn't run image preprocessing")
    # detection
    try:
        raw_results = imp.infer(img_container_detect)
    except:
        raise ValueError("Detection failed. Couldn't detect any peaks")
    # postprocessing
    try:
        img_container_detect = standard_postprocessing(img_container_detect, raw_results)
    except:
        raise ValueError("Detection failed. Couldn't run image postprocessing")
    img_container_detect.split_polar_images = None
    return img_container_detect


def standard_preprocessing_from_polar(config, raw_polar_img):
    equalized_polar, mask = contrast_correction(config, raw_polar_img)
    equalized_polar = add_batch_and_color_channel(equalized_polar)
    mask = add_batch_and_color_channel(mask)

    # reshape for detr model
    if config.MODEL_TYPE == 'detr':
        equalized_polar = grayscale_to_color(equalized_polar)
        equalized_polar = equalized_polar[:, :, :, :]
        equalized_polar = np.pad(equalized_polar, ((0, 0), (0, 0,), (0, 832), (0, 0)))

    return equalized_polar, raw_polar_img, mask


def run_mlgiddetect_from_polar(img, imp, config_detect):
    img_container_detect = ImageContainer()
    img_container_detect.config = config_detect
    # img_container_detect.q_xy = q_xy
    # img_container_detect.q_z = q_z
    img_container_detect.nr = 0
    img_container_detect.polar_img_shape = img.shape
    (
        img_container_detect.converted_polar_image,
        img_container_detect.raw_polar_image,
        img_container_detect.converted_mask,
    ) = standard_preprocessing_from_polar(img_container_detect.config, img)
    raw_results = imp.infer(img_container_detect)
    img_container_detect = standard_postprocessing(
        img_container_detect, raw_results
    )
    img_container_detect.split_polar_images = None
    return img_container_detect
