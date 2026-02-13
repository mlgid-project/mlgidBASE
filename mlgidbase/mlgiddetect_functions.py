from mlgiddetect.inference import Inference
from mlgiddetect.configuration import Config
from mlgiddetect.preprocessing import standard_preprocessing
from mlgiddetect.postprocessing import standard_postprocessing
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



# def peaks2img_container(filename, entry, frame_num):
#     detected_peaks = read_detected_peaks(filename, entry, frame_num)
#     print("detected_peaks", detected_peaks)
#     img_container_detect = ImageContainer()
#     img_container_detect.radius = detected_peaks['radius']
#     img_container_detect.radius_width = detected_peaks['radius_width']
#     img_container_detect.angle = detected_peaks['angle']
#     img_container_detect.angle_width = detected_peaks['angle_width']
#     img_container_detect.scores = detected_peaks['scores']
#
#     raw_reciprocal, ai, wavelength, converted_polar_image, q_xy, q_z


