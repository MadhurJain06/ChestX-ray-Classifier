# ChestX-ray-Classifier
Multilabel Classification(6 classes) of Diseases using X-ray Images of Chest, using NIH (compressed) dataset.

## Dataset:
https://www.kaggle.com/datasets/mohamedasak/chest-x-ray-6-classes-dataset

## Model/ Architecture:
The model used here is CNN- Resnet Model (Stands for Residual Network) used for fast and deep learning as it allows to skip connection layers and process image faster, also yields high output accuracy.

Resnet50 here used means the model is 50 layered CNN pre-trained on ImageNet
### Why use RESNET50??

- Already knows basic visual features (edges, textures, shapes).
- Saves you from training from scratch (which requires millions of images).
- Faster Training.

I used a pre-trained ResNet50 because even with 15k images, training a deep network from scratch is unnecessary and inefficient.
Transfer learning gives better accuracy in less time.

## Preprocessing:
```
rescale=1./255
horizontal_flip=True
rotation_range=10
zoom_range=0.1
```
Rescale: Rescaling the pixels of image from 0-255 to 0-1, makes training stable.
Rotation: small angle change
horizontal flip: contains differnt X-ray orientations

## Pipeline:
```
flow_from_directory()
```
It is a high level loader from keras
This helps in:
1. Reading Images from folders
2. Resizes to 224x224
3. Label them and folders, so we don't have to manually label each
4. Sends them to model during the training, after preprocessing, batching and shuffling.


