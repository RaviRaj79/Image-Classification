# Image Classification with TensorFlow and Keras

Starter project for training, evaluating, and predicting image classes with TensorFlow/Keras.

## Project layout

```text
image classification/
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|-- src/
|   `-- image_classification/
|       |-- utils/
|       |-- config.py
|       |-- data.py
|       |-- dataset_tools.py
|       |-- evaluate.py
|       |-- model.py
|       |-- predict.py
|       `-- train.py
|-- tests/
|-- requirements.txt
`-- README.md
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## One-command runner

```powershell
.\run.ps1
```

## Simple demo commands

```powershell
.\demo.ps1 open dog
.\demo.ps1 open cat
.\demo.ps1 predict dog
.\demo.ps1 predict cat
```

Useful variants:

```powershell
.\run.ps1 -Action train
.\run.ps1 -Action evaluate
.\run.ps1 -Action predict -Image data\raw\your_class\your_image.jpg
.\run.ps1 -Action predict -Image data\raw\your_class\your_image.jpg -OpenImage
```

## Check dataset summary

```powershell
python -m src.image_classification.dataset_tools --summary
```

## Train model

```powershell
python -m src.image_classification.train
```

## Evaluate trained model

```powershell
python -m src.image_classification.evaluate
```

## Predict a single image

```powershell
python -m src.image_classification.predict --image data\raw\your_class\some_image.jpg
```

## Notes

- The default model is a CIFAR-10-friendly CNN tuned for `32x32` images.
- Training artifacts are saved in `models/`.
- Put your own dataset in `data/raw/<class_name>/image.jpg` format before training.
- Update values in `src/image_classification/config.py` to match your dataset and hardware.
