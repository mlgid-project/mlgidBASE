## How to use

Import the package
```python
from mlgidbase import mlgidBASE
```
Initialize analysis from a NeXus file (`pygid`)
```python
filename = r'./example/BA2PbI4.h5'
analysis = mlgidBASE(filename=filename)
```
Run peak detection
```python
analysis.run_detection()
```
Run peak fitting
```python
analysis.run_fitting()
```
Run peak matching with preprocessed CIFs
```python
analysis.run_matching(
    cif_prepr='./example/prepr_cifs.pickle'
)
```
Plot the result of analysis
```python
analysis.plot_analysis_results()
```