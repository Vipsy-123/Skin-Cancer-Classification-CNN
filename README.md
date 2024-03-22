# Skin Cancer Classification ML pipeline using CNN & Pre-Trained Models(Transfer Learning)

## Overview

- This project focuses on the binary classification of skin cancer images into "Benign" or "Malignant" categories. The implementation includes a Convolutional Neural Network (CNN) and/or Transfer Learning using TensorFlow and Keras.
-  The dataset is not provided here, but you can use your own Skin Cancer image dataset.
- I have used 10,000 images for Training .
- 5000 Benign and 5000 Malignant images with Train:Validate:Test ratio 7:2:1

## Dataset Directory Structure
```bash
├───Data
│   ├───Test
│   │   ├───benign # 500
│   │   └───malignant # 500
│   ├───Train
│   │   ├───benign # 3500
│   │   └───malignant # 3500
│   └───Validate
│       ├───benign # 1000
│       └───malignant # 1000
│           └───.ipynb_checkpoints
```

## Requirements 

- Python 3.10.11
- TensorFlow 2.11.0
- NumPy
- Matplotlib
- Anaconda Navigator
- Your Skin Cancer Image Dataset (not provided, you can use your own dataset)

## Installation 

### 1. Install Anaconda Navigator:

Click here to [install]((https://www.anaconda.com/download)) Anaconda Navigator.

### 2. Install Tensorflow :
Follow the steps in [this video](https://www.youtube.com/watch?v=QUjtDIalh0k&t=137s) to install Tensorflow.

**2.1 Create a Conda Environment:**
```bash
  $ conda create -n py310 python=3.10
```

**2.2 Activate Conda Environment:**
```bash
  $ conda activate py310
```

**2.3 Install CUDA toolkit & CUDNN Library:**
```bash
  $ conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

**2.4 Install Tensorflow:**
```bash
  $ python -m pip install "tensorflow<2.11"
```

**2.5 Test GPU**
```bash
  $ Python
  >> import tensorflow as tf
  >> tf.config.list_physical_devices('GPU')

```

## Usage

### Clone this Repository:
```bash
   git clone  https://github.com/Vipsy-123/Skin-Cancer-Classification-using-CNN.git
```

### _Dataset_:
  Download the Benign and Malignant images dataset from [ISIC Archive](https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%5D)

### CNN Usage :
1. Open cnn1.ipynb
2. Run the cells to perform predictions on new skin cancer images

### Transfer Learning Usage :   
1. Open DensNet201.ipynb
2. You need to make 3 changes
    - Import your pre-trained model in Section 1. Installing Dependencies
    - Change input size in Section 2. Data Preparation & Augmentation
    - Change to your Model name in Section 3. Building the Model
       - e.g  base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

## Using models later :
  You can see pre-trained models in models Directory to save Model Change the Model name in Section 4. Training the Model


