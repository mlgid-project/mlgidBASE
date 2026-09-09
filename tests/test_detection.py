"""Tests for the detection step of the pipeline.

This module holds:

* standalone unit tests for :func:`mlgidbase.mlgiddetect_functions.load_config`,
  mirroring the three configuration scenarios from
  ``docs/tutorials/tutorial_02_detection.ipynb``;
* reusable helpers (``detect_*``) that drive ``analysis.run_detection`` and are
  consumed by the end-to-end tests in ``test_mlgidbase.py``.
"""

import yaml

from mlgidbase.mlgiddetect_functions import load_config

from .common import DINO_YAML, FASTER_RCNN_YAML


# ---------------------------------------------------------------------------
# load_config: detection-configuration key handling
#
# Checks that PREPROCESSING_LOG, PREPROCESSING_HISTOGRAMEQUALIZATION and
# MODEL_TYPE end up with the expected values on the resulting Config object.
# ---------------------------------------------------------------------------

def test_load_config_from_yaml_file():
    """Config loaded from a YAML file path, with model_type='dino' (tutorial cell 8).

    dino.yaml sets PREPROCESSING.LOG=True and PREPROCESSING.HISTOGRAMEQUALIZATION=True;
    the explicit model_type argument must win for MODEL_TYPE.
    """
    config = load_config(DINO_YAML, 'dino')

    assert config.PREPROCESSING_LOG is True
    assert config.PREPROCESSING_HISTOGRAMEQUALIZATION is True
    assert config.MODEL_TYPE == 'dino'


def test_load_config_from_dict():
    """Config supplied as a nested dict, no model_type (tutorial cell 10).

    Only the keys present in the dict are overridden; MODEL_TYPE falls back to
    the mlgiddetect Config default ('dino').
    """
    config_dict = {
        'PREPROCESSING': {
            'LOG': True,
            'HISTOGRAMEQUALIZATION': False,
        }
    }

    config = load_config(config_dict, None)

    assert config.PREPROCESSING_LOG is True
    assert config.PREPROCESSING_HISTOGRAMEQUALIZATION is False
    assert config.MODEL_TYPE == 'dino'


def test_load_config_from_yaml_dict_with_override():
    """YAML file read into a dict, then PREPROCESSING.LOG flipped (tutorial cell 12).

    The overridden key must be reflected, the untouched key keeps the YAML value,
    and MODEL_TYPE comes from the dict's MODEL.TYPE entry.
    """
    with open(DINO_YAML, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    config_dict['PREPROCESSING']['LOG'] = False

    config = load_config(config_dict, None)

    assert config.PREPROCESSING_LOG is False
    assert config.PREPROCESSING_HISTOGRAMEQUALIZATION is True
    assert config.MODEL_TYPE == 'dino'


# ---------------------------------------------------------------------------
# Reusable detection helpers for the end-to-end pipeline tests.
# ---------------------------------------------------------------------------

def detect_dino(analysis):
    """Run detection with the built-in 'dino' model on all frames and one frame."""
    analysis.run_detection(config_detect=None, model_type='dino')
    analysis.run_detection(entry='entry_0000', frame_num=0, config_detect=None, model_type='dino')
    assert analysis.config_detect.MODEL_TYPE == 'dino'


def detect_faster(analysis):
    """Run detection with the built-in 'faster_rcnn' model on all frames and one frame."""
    analysis.run_detection(config_detect=None, model_type='faster_rcnn')
    analysis.run_detection(entry='entry_0000', frame_num=0, config_detect=None, model_type='faster_rcnn')
    assert analysis.config_detect.MODEL_TYPE == 'faster_rcnn'


def detect_dino_config(analysis):
    """Run detection using the dino.yaml configuration file."""
    analysis.run_detection(config_detect=DINO_YAML)
    analysis.run_detection(entry='entry_0000', frame_num=0)
    assert analysis.config_detect.MODEL_TYPE == 'dino'


def detect_faster_config(analysis):
    """Run detection using the faster_rcnn.yaml configuration file."""
    analysis.run_detection(config_detect=FASTER_RCNN_YAML)
    analysis.run_detection(entry='entry_0000', frame_num=0)
    assert analysis.config_detect.MODEL_TYPE == 'faster_rcnn'
