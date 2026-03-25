from mlgiddetect.inference import Inference
from mlgiddetect.configuration import Config
from mlgiddetect.preprocessing import standard_preprocessing
from mlgiddetect.postprocessing import standard_postprocessing
from mlgiddetect.preprocessing import (contrast_correction, add_batch_and_color_channel,
                                        grayscale_to_color)
from mlgiddetect.dataloader import ImageContainer
from .pygid_functions import save_detect, read_conversion_from_nexus
import importlib.metadata
import numpy as np
from datetime import datetime


def _run_detection(analysis, entry=None, frame_num=None, config_detect=None, model_type=None):
    """
    Run peak detection on GID data.

    This function applies the mlgiddetect model to either in-memory data or
    NeXus-stored data, depending on the analysis configuration.

    Parameters
    ----------
    analysis : object
        Analysis object containing data, configuration, and state.
    entry : str, optional
        NeXus entry to process. If None, all entries are processed.
    frame_num : int, optional
        Frame index to process. If None, all frames are processed.
    config_detect : str or Config, optional
        Detection configuration or path to configuration file.
    model_type : str, optional
        Model type to use (e.g., 'faster_rcnn', 'detr').
    """
    if config_detect is not None:
        if analysis.config_detect is not None:
            analysis.logger.info(f"config_detect is already set. The previous config is to be used")
        else:
            analysis.config_detect = config_detect
    analysis.config_detect = load_config(config_detect)
    if model_type is not None:
        if analysis.config_detect.MODEL_TYPE != model_type:
            analysis.config_detect.MODEL_TYPE = model_type
            analysis.imp_detect = None
    if analysis.imp_detect is None:
        load_inference(analysis)

    if not analysis.from_nexus:
        if frame_num != 1 and not frame_num is None:
            analysis.logger.warning("frame_num will be ignored.")
        _run_detection_from_memory(analysis)
    else:
        _run_detection_from_file(analysis, entry, frame_num)


def load_inference(analysis):
    """
    Initialize the detection inference model.

    Parameters
    ----------
    analysis : object
        Analysis object containing detection configuration.
    """
    try:
        analysis.imp_detect = Inference(analysis.config_detect)
    except:
        raise ValueError("Detection failed. Couldn't load the model.")


def _run_detection_from_memory(analysis):
    """
    Run detection on in-memory polar images.

    Parameters
    ----------
    analysis : object
        Analysis object containing polar images and conversion data.
    """
    analysis.config_detect.GEO_RECIPROCAL_SHAPE = list(analysis.pygid_conversion.img_gid_q[0].shape)
    analysis.config_detect.GEO_PIXELPERANGSTROEM = analysis.config_detect.GEO_RECIPROCAL_SHAPE[0] / np.nanmax(analysis.q_abs)
    analysis.config_detect.GEO_QMAX = np.nanmax(analysis.q_abs)
    analysis.img_container_detect_list = []

    for i in range(len(analysis.img_pol)):
        img_pol = np.array(analysis.img_pol[i])
        img_container_detect = run_mlgiddetect_from_polar(img_pol,
                                                          analysis.imp_detect, analysis.config_detect)
        img_container_detect.ai = analysis.pygid_conversion.ai_list[i]
        img_container_detect.wavelength = analysis.pygid_conversion.matrix[0].params.wavelength
        img_container_detect.metadata = _set_detection_metadata(analysis)
        analysis.img_container_detect_list.append(img_container_detect)


def _run_detection_from_file(analysis, entry, frame_num):
    """
    Run detection on data stored in a NeXus file.

    Parameters
    ----------
    analysis : object
        Analysis object with file access.
    entry : str or None
        Entry to process. If None, all entries are processed.
    frame_num : int or None
        Frame index to process. If None, all frames are processed.
    """
    if entry is None:
        for entry in analysis.entry_dict:
            _run_detection_single_entry(analysis, entry, frame_num)
        return
    if not entry in analysis.entry_dict:
        raise ValueError("entry not found in the NeXus file")
    _run_detection_single_entry(analysis, entry, frame_num)


def _run_detection_single_entry(analysis, entry, frame_num):
    """
    Run detection for a single NeXus entry.

    Parameters
    ----------
    analysis : object
        Analysis object containing NeXus metadata.
    entry : str
        Entry name to process.
    frame_num : int or None
        Frame index to process. If None, all frames are processed.
    """
    frame_num_all = analysis.entry_dict[entry]['shape'][0]
    if frame_num is None:
        for frame_num in range(frame_num_all):
            _run_detection_single_frame(analysis, entry, frame_num)
        return
    if frame_num >= frame_num_all:
        raise ValueError("frame_num is out of range")
    _run_detection_single_frame(analysis, entry, frame_num)


def _run_detection_single_frame(analysis, entry, frame_num):
    """
    Run detection for a single frame and save results to file.

    Parameters
    ----------
    analysis : object
        Analysis object containing NeXus file and detection model.
    entry : str
        Entry name.
    frame_num : int
        Frame index.
    """
    conversion = read_conversion_from_nexus(analysis.nexus, entry, frame_num)
    img_container_detect = run_mlgiddetect(conversion.img_gid_q[0], conversion.matrix[0].q_xy, conversion.matrix[0].q_z,
                                           analysis.imp_detect, analysis.config_detect)
    img_container_detect.metadata = _set_detection_metadata(analysis)
    save_detect(analysis.filename, entry, frame_num, img_container_detect)
    analysis.logger.info(f"Saved detected peaks to file: {analysis.filename}, entry: {entry}, frame: {frame_num}")


def _set_detection_metadata(analysis):
    """
    Generate metadata dictionary for detection results.

    Parameters
    ----------
    analysis : object
        Analysis object containing detection configuration.

    Returns
    -------
    dict
        Metadata including program name, version, timestamp, and configuration.
    """
    metadata = {'program': 'mlgiddetect',
                'version': importlib.metadata.version("mlgiddetect"),
                'date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'),
                }
    metadata.update(analysis.config_detect.__dict__)
    return metadata

def load_config(config):
    """
    Load and validate detection configuration.

    Parameters
    ----------
    config : str, Config, or None
        Configuration file path, Config object, or None for default.

    Returns
    -------
    Config
        Initialized and validated configuration object.
    """
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



def check_valid_config(config):
    """
    Apply default preprocessing settings to configuration.

    Parameters
    ----------
    config : Config
        Configuration object.

    Returns
    -------
    Config
        Updated configuration with enforced defaults.
    """
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
    """
    Run peak detection on a reciprocal-space image.

    Parameters
    ----------
    img : ndarray
        2D reciprocal-space image.
    q_xy_axes : ndarray
        q_xy axis values.
    q_z_axes : ndarray
        q_z axis values.
    imp : Inference
        Detection model.
    config_detect : Config
        Detection configuration.

    Returns
    -------
    ImageContainer
        Container with detected peaks and metadata.
    """
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
    """
    Apply preprocessing to a polar-transformed image.

    Parameters
    ----------
    config : Config
        Detection configuration.
    raw_polar_img : ndarray
        Input polar image.

    Returns
    -------
    tuple of ndarray
        Preprocessed image, original polar image, and mask.
    """
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
    """
    Run detection directly on a polar image.

    Parameters
    ----------
    img : ndarray
        Polar-transformed image.
    imp : Inference
        Detection model.
    config_detect : Config
        Detection configuration.

    Returns
    -------
    ImageContainer
        Container with detected peaks and metadata.
    """
    img_container_detect = ImageContainer()
    img_container_detect.config = config_detect
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
