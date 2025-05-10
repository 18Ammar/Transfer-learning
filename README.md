
# HENet Transfer Learning for Font Classification

This project implements a HENet-based deep learning model using transfer learning with ResNet backbones to classify Arabic fonts into 200 classes.

## Features
- Uses pretrained ResNet (resnet18 or resnet50) as the backbone
- Implements HEMaxBlock attention to enhance key features
- Label smoothing for improved generalization
- Cosine learning rate scheduler
- Early stopping and model checkpointing

## Dataset
Place your dataset in this structure:

```
data/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── test/
    ├── class_0/
    ├── class_1/
    └── ...
```

## Training

```bash
python train.py
```

Adjust paths and hyperparameters in `train.py` as needed.

## Dependencies
- Python 3.x
- PyTorch
- torchvision
- numpy
- PIL

Install them using:

```bash
pip install torch torchvision
```

## Model Architecture

- Pretrained ResNet Backbone (resnet18 / resnet50)
- 1x1 Conv → HEMaxBlock → AvgPool → FC

## License
MIT License
