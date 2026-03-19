## Installation Procedure for the `mlgid` Project

### 1. Create a directory for the `mlgid` packages
```bash
mkdir mlgid
cd mlgid
```
### 2. Create the personal token on [github](https://github.com/settings/tokens). Save the personal token.


### 3. Clone the packages
```bash
git clone https://github.com/mlgid-project/pygid.git
git clone https://github.com/mlgid-project/pygidFIT.git
git clone https://github.com/mlgid-project/mlgidDETECT.git
git clone https://github.com/mlgid-project/mlgidMATCH_private.git
git clone https://github.com/mlgid-project/mlgidBASE.git
```
### 4. Create and activate the venv
```bash
python -m venv mlgid_venv
source mlgid_venv/bin/activate
```

### 5. Install the packages
```bash
pip install -e ./pygid/.
pip install -e ./pygidFIT/.
pip install -e ./mlgidDETECT/.
pip install -e ./mlgidMATCH_private/.
pip install -e ./mlgidBASE/.
```
### 6. Insall `ipykernel` and create the kernel
```bash
pip install ipykernel
python -m ipykernel install --user --name mlgid_venv --display-name "mlgid_pipeline_venv"
```

### 7. (OPTIONAL) Pull the latest versions from the github:
in the package folder:
```bash
git pull origin main
```


