"""Shared paths and constants for the mlgidBASE test suite.

The individual pipeline-step modules (``test_detection``, ``test_fitting``,
``test_matching``, ``test_peak_operations``, ``test_data_saver``) and the
end-to-end ``test_mlgidbase`` module all import from here so that the location
of the bundled example data is defined in a single place.
"""

import os

THIS_DIR = os.path.dirname(__file__)

# Absolute path to the example folder shipped with the repository.
EXAMPLE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "example"))

# NeXus file used as the default input for the file-based pipeline tests.
NEXUS_FILE = os.path.join(EXAMPLE_DIR, "BA2PbI4.h5")

# Detection configuration files.
DINO_YAML = os.path.join(EXAMPLE_DIR, "dino.yaml")
FASTER_RCNN_YAML = os.path.join(EXAMPLE_DIR, "faster_rcnn.yaml")
