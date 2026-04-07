=====================
mlgidBASE Documentation
=====================

Welcome to the ``mlgidBASE`` documentation!

``mlgidBASE`` is a Python package for machine learning–driven analysis of grazing-incidence wide-angle X-ray scattering (GIWAXS) data. It provides a full workflow from peak detection to matching with known crystal structures.

.. image:: images/mlgid_logo.png
   :width: 400px
   :align: center
   :alt: mlgid

The package builds on the following components:

- `mlgidDETECT <https://github.com/mlgid-project/mlgidDETECT>`__ — for peak detection
- `pygidFIT <https://github.com/mlgid-project/pygidFIT>`__ — for two-dimensional peak fitting
- `mlgidMATCH <https://github.com/mlgid-project/mlgidMATCH_private>`__ — for matching experimental peaks to known structures


Key Features
============

Initialization
   Can be created from a :class:`pygid.Conversion` object or loaded directly from a
   :doc:`NeXus file <tutorials/tutorial_01_initialization>`.

Methods
   Provides functions for:

   - Peak detection: :doc:`tutorials/tutorial_02_detection`
   - Peak fitting: :doc:`tutorials/tutorial_03_fitting`
   - Peak matching: :doc:`tutorials/tutorial_04_matching`

Visualization
   Supports visualization at all stages of the analysis pipeline:
   :doc:`tutorials/tutorial_06_visualization`.

Peak Adjustment
   Includes functions to add or delete peaks, either interactively or programmatically:
   :doc:`tutorials/tutorial_07_peak_operations`.

Data Access
   Enables retrieving analysis results from the NeXus file for further processing:
   :doc:`tutorials/tutorial_08_get_data`.


.. toctree::
   :maxdepth: 2
   :caption: Outline

   Quick Start
   tutorials_toctree
   File Format