# M1-compatible Mediapipe-model-maker

[mediapipe-model-maker · PyPI](https://pypi.org/project/mediapipe-model-maker/) will not install on M1 Macs. This repository is a workaround to get it running. (Could also help for Windows users)

## Why is it not working?

The dependency [tensorflow-text · PyPI](https://pypi.org/project/tensorflow-text/) announced they will only support linux from now on, which means any python libs that depend on it will not install on M1. In this repo we have removed it as a dependency. 

# Setup 

## (Optional) Create a virtual environment
```bash
python -m venv env
source env/bin/activate
```

## Install dependencies
```bash
pip install -r requirements.txt
pip install tf-models-official==2.17.0 --no-deps
# now install this repo as a package
pip install .
```

## Test
```bash
cd test
python train_hands_model.py
```