# mlgidBASE

`mlgidBASE` is a package dedicated to machine learning–based analysis of grazing-incidence wide-angle X-ray scattering (GIWAXS) data.

The package builds upon:

- [mlgidDETECT](https://github.com/mlgid-project/mlgidDETECT) for peak detection  
- [pygidFIT](https://github.com/mlgid-project/pygidFIT) for two-dimensional peak fitting  
- [mlgidMATCH](https://github.com/mlgid-project/mlgidMATCH_private) for matching experimental peaks with known structures  

`mlgidBASE` can be used after data conversion performed with [pygid](https://github.com/mlgid-project/pygid), either:
- before saving (for faster data access), or  
- after saving as a NeXus file.

---

## Installation

### Prerequisites

Before installing `mlgidBASE`, ensure that the following packages are installed:

- `pygid`  
  Install from: https://github.com/mlgid-project/pygid  
  or via PyPI:
  ```bash
  pip install pygid
  ```
- `mlgidDETECT` 
https://github.com/mlgid-project/mlgidDETECT
- `pygidFIT` 
https://github.com/mlgid-project/pygidFIT
- `mlgidMATCH` 
https://github.com/mlgid-project/mlgidMATCH_private

  

After installing all prerequisites, install `mlgidBASE` from source.

First, clone the repository:

```bash
git clone https://github.com/mlgid-project/mlgidBASE.git
```

Then navigate to the project directory and install it in editable mode:
cd mlgidBASE
pip install -e .


#### Required Python version: 3.12

---

## How to Use (Short Version)



### Run the class methods:

Before starting, convert the data using `pygid` and either save it as a NeXus file or pass a `pygid.Conversion` instance after running the `det2q_gid()` conversion.

```python
from mlgidbase import mlgidBASE
import pygid
```
Option 1: Initialize from file

```python
filename = r'example\eiger4m_result.h5'
analysis = mlgidBASE(filename=filename)
```
Option 2: Initialize from pygid.Conversion

```python
conversion = pygid.Conversion(
    matrix=matrix,
    path=filename,
    dataset=dataset,
    frame_num=frame_num
)

analysis = mlgidBASE(pygid_conversion=conversion)
```
### Run analysis methods

```python
# peak detection
analysis.run_detection(
  entry='entry_0000',           # if read from file
  frame_num=0,                  # if read from file
)

# peak fitting
analysis.run_fitting(
  entry='entry_0000',           # if read from file
  frame_num=0,                  # if read from file
  clustering_distance_peaks = 10,
  clustering_distance_rings = 10,
  clustering_extend = 2
)

# peak matching
analysis.run_matching(
  entry='entry_0000',           # if read from file
  frame_num=0,                  # if read from file
  cif_prepr = 'prepr_cifs.pickle',
  probability_threshold = 0.5,
  intensity_threshold = 0,
  peaks_type = 'segments',
  device = 'cpu'
)
```

#### Description

#### 1. `mlgidBASE.run_detection()`

Runs peak detection on the specified entry or frame.

**Parameters:**

- `entry` — Data entry to process. Defaults to `None` (all entries).
- `frame_num` — Frame number for detection. Defaults to `None` (all frames).
- `config_file` — Path to detection configuration file. Defaults to `None` (default parameters).

---

#### 2. `mlgidBASE.run_fitting()`

Performs fitting of detected peaks with clustering.

**Parameters:**

- `entry` — Data entry to fit. Defaults to `None`.
- `frame_num` — Frame number to fit. Defaults to `None`.
- `crit_angle` (`float`) — Critical angle used in fitting. Defaults to `0`.
- `clustering_distance_peaks` (`float`) — Distance threshold for clustering peaks. Defaults to `10`.
- `clustering_distance_rings` (`float`) — Distance threshold for clustering rings. Defaults to `10`.
- `clustering_extend` (`float`) — Cluster extension factor. Defaults to `2`.
- `use_pool` (`bool`) — Enable multiprocessing. Defaults to `False`.
- `debug` (`bool`) — Enable debug output. Defaults to `False`.

---

#### 3. `mlgidBASE.run_matching()`

Matches fitted peaks to CIF patterns.

**Parameters:**

- `entry`  — Data entry for matching. Defaults to `None` - all entries.
- `frame_num`  — Frame number for matching. Defaults to `None` - all frames.
- `cif_prepr` — Preprocessed CIFs object / path to PICKLE file to load. (Required)
- `probability_threshold` (`float`) — Matching threshold for peaks. Defaults to `0.5`.
- `intensity_threshold` - Intensity threshold for fitted peaks to be used in matching. 
- `peaks_type` (`str`) — Type of peaks used for matching (`'segments'` or `'rings'`). Defaults to `'peaks'`.
- `device` (`str`) — Computation device (`'cpu'` or `'cuda'`). Defaults to `'cpu'`.


---
### Plot result
#### `mlgidBASE.plot_analysis_results()`

The result of conversion and analysis can be visualized using `plot_analysis_results` function. 
Users can control the color, size and styles of plotted peaks/rings. 

```python
detected_params = {'line_width': 0.5,
                 'line_style': "--",
                 'line_color': "black",
                 'plot': False}
fitted_params = {'plot_segments': True,
                 'marker': 'o',
                 'marker_size': 50,
                 'marker_facecolor': "none",
                 'marker_edgecolor': "bone",
                 'plot_rings': True,
                 'line_width': 1,
                 'line_style': "--",
                 'line_color': "bone",
                 'plot': False}
matched_params = {'plot_segments': True,
                 'marker': ['s'],
                 'marker_size': [50],
                 'marker_facecolor': ["none"],
                 'marker_edgecolor': ["green", "red"],
                 'plot_rings': True,
                 'line_width': [1],
                 'line_style': ["--"],
                 'line_color': ["green"],
                 'plot': True,
                 'legend': True}
                              
analysis.plot_analysis_results(
          detected_params = detected_params,
          fitted_params = fitted_params,
          matched_params = matched_params,
          frame_num = None, entry = None, # if read from file 
          return_result=False, plot_result=True,
          clims=(50, 1e4), 
          xlim=(None, None), ylim=(None, None),
          save_fig=True, path_to_save_fig="img.png")
```
Examples:
- rings
<p align="center">
  <img src="./example/img_entry_0001_fr_0000_sol_0000.png" width="400" alt="pygidFIT">
</p>

- peaks

<p align="center">
  <img src="./example/img_entry_0001_fr_0000_sol_0001.png" width="400" alt="pygidFIT">
</p>

### Save result

#### `mlgidBASE.save_result()`

If the `mlgidBASE` instance was created from a file, results are saved automatically at each step (for every entry and frame).

If the analysis is performed from a `pygid.Conversion` instance, `save_result()` must be called manually after completing the analysis.


```python
analysis.save_result(
    path_to_save='result.h5',
    save_polar=True
)
```

**Parameters:**

- `path_to_save` (`str`, optional) — Full path including the file name where the data will be saved. The file format must be `.h5`. Default is `'result.h5'`.

- `overwrite_file` (`bool`, optional) — Whether to overwrite an existing HDF5 file if it already exists. Default is `True`.

- `h5_group` (`str`, optional) — Name of the group inside the HDF5 file where the data will be stored. Default is `'entry_0000'`.

- `overwrite_group` (`bool`, optional) — Whether to overwrite an existing group within the HDF5 file. Default is `False`.

- `smpl_metadata` (`pygid.SampleMetadata`, optional) — Instance containing sample-related metadata to be stored in the file. Default is `None`.

- `exp_metadata` (`pygid.ExpMetadata`, optional) — Instance containing experimental metadata to be stored in the file. Default is `None`.

- `save_polar` (`bool`, optional) - Whether save the polar image used in analysis.

The method stores processed images, detected peaks, fitted results, matched data, unit cell parameters, and associated metadata in a structured HDF5 format.

---

The example of usage can be found in the `example` folder. 


