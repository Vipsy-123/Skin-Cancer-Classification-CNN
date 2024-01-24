# Skin Cancer Classification ML pipeline using CNN & Pre-Trained Models(Transfer Learning)

## Overview

This project focuses on the binary classification of skin cancer images into "Benign" or "Malignant" categories. The implementation includes a Convolutional Neural Network (CNN) with Transfer Learning using TensorFlow and Keras. The dataset is not provided here, but you can use your own skin cancer image dataset.

## Requirements 

- Python 3.10.11
- TensorFlow 2.11.0
- NumPy
- Matplotlib
- Anaconda Navigator
- Your Skin Cancer Image Dataset (not provided, you can use your own dataset)

## Installation 

### 1. Install Anaconda Navigator:

Follow the steps in [this video](https://www.youtube.com/watch?v=Ejzubp-B83o&t=1165s](https://www.youtube.com/watch?v=BXsgHC8qTac)) to install Anaconda Navigator.


### 2. Install Tensorflow :
Follow the steps in [this video](https://www.youtube.com/watch?v=QUjtDIalh0k&t=137s) to install Tensorflow.

** 2.1 Create a Conda Environment:**
```bash
  $ conda create -n py310 python=3.10
```

** 2.2 Activate Conda Environment:**
```bash
  $ conda activate py310
```

** 2.3 Install CUDA toolkit & CUDNN Library:**
```bash
  $ conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

** 2.4 Install Tensorflow:**
```bash
  $ python -m pip install "tensorflow=2.11"
```

** 2.5 Test GPU**
```bash
  $ Python
  >> import tensorflow as tf
  >> tf.config.list_physical_devices('GPU')

```

## Usage

### _Dataset_:
  Download the Benign and Malignant images dataset from [ISIC Archive](https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%5D)
