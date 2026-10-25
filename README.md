# Minimal CIFAR-10

A small PyTorch CNN for training and evaluating on CIFAR-10.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Train

The dataset is downloaded to `data/` on the first run. The default run trains
for one epoch and saves the model as `cifar10.pt`.

```powershell
python cifar10.py --mode train
```

Useful options:

```powershell
python cifar10.py --mode train --epochs 10 --batch-size 64 --learning-rate 0.001
```

## Test

Evaluate a saved checkpoint:

```powershell
python cifar10.py --mode test
```

The script uses CUDA automatically when it is available.

## Model

The CNN contains 545,098 trainable parameters and two convolutional blocks
followed by two fully connected layers.
