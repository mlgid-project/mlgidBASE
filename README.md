# mlgidBASE
[![Python version](https://img.shields.io/badge/python-3.9%7C3.10%7C3.11%7C3.12%7C3.13%7C3.14-blue.svg)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/Documentation%20%26%20Tutorials-red)](https://mlgidbase.readthedocs.io/en/latest/#)

`mlgidBASE` is a Python package for machine learning–driven analysis of grazing-incidence wide-angle X-ray scattering (GIWAXS) data. It provides a full workflow from peak detection to matching with known crystal structures.

The package builds on the following components:

- [pygid](https://github.com/mlgid-project/pygid) — for detector image conversion
- [mlgidDETECT](https://github.com/mlgid-project/mlgidDETECT) — for peak detection  
- [pygidFIT](https://github.com/mlgid-project/pygidFIT) — for two-dimensional peak fitting  
- [mlgidMATCH](https://github.com/mlgid-project/mlgidMATCH_private) — for matching experimental peaks to known structures  

---

## Key Features

- **Initialization**  
  Can be created from a [`pygid.Conversion`](./docs/tutorials/tutorial_05_from_memory.ipynb) object or loaded directly from a [NeXus file](./docs/tutorials/tutorial_01_initialization.ipynb).

- **Methods**  
  Provides functions for:
  - [Peak detection](./docs/tutorials/tutorial_02_detection.ipynb)  
  - [Peak fitting](./docs/tutorials/tutorial_03_fitting.ipynb)  
  - [Peak matching](./docs/tutorials/tutorial_04_matching.ipynb)

- **Visualization**  
  Supports [visualization](./docs/tutorials/tutorial_06_visualization.ipynb) at all stages of the analysis pipeline.

- **Peak Adjustment**  
  Includes functions to [add or delete peaks](./docs/tutorials/tutorial_07_peak_operations.ipynb), either interactively or programmatically.

- **Data Access**  
  Enables [retrieving analysis results](./docs/tutorials/tutorial_08_get_data.ipynb) from the NeXus file for further processing.

---

## Installation

### Install using pip

```bash
pip install mlgidbase
```

### Install from source

First, clone the repository:

```bash
git clone https://github.com/mlgid-project/mlgidBASE.git
```

Then navigate to the project directory and install it in editable mode:
```bash
cd mlgidBASE
pip install -e .
```
---

## How to Use

For full details, see the dedicated [tutorials](./docs/tutorials).

### Quick Start
```python
from mlgidbase import mlgidBASE

# Initialize analysis from a NeXus file
filename = r'./example/BA2PbI4.h5'
analysis = mlgidBASE(filename=filename)

# Run peak detection
analysis.run_detection()

# Run peak fitting
analysis.run_fitting()

# Run peak matching with preprocessed CIFs
analysis.run_matching(
    cif_prepr='./example/prepr_cifs.pickle'
)
```

### Data Format

The structure of the analysis results saved in the NeXus file is documented in the output file format 
[guide](docs/tutorials/output_file_format.md).

It describes how entries, frames, and peak information are stored for further inspection or processing.
