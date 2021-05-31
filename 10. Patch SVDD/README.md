# Patch SVDD codes

This is unofficial Patch SVDD codes. 

It is a little different from official github codes.

Patch SVDD for image anomaly detection & anomaly segmentation.

Paper: https://arxiv.org/abs/2006.16067

Official Code: https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch

<br/>

## 0. Used package versions

torch==1.8.1+cu101

matplotlib==3.2.2

numpy==1.19.5

scikit-image==0.16.2

scikit-learn==0.22.2.post1

tqdm==4.41.1

Pillow==7.1.2

imageio==2.4.1

opencv-python==4.1.2.30

ngt==1.12.2

<br/>

## 1. MVTec AD dataset installation

In this code, use MVTec AD dataset.

So, you need to download this dataset.

[Download link](https://www.mvtec.com/company/research/datasets/mvtec-ad/) 



## 2. Training

**Step 1. you need to set 'DATASET_PATH' variable.**

Set DATASET_PATH to the root path of the dataset.

This code use MVTec AD dataset, but you can use your custom dataset. 

(I also use this model for my custom anomaly detection dataset.)

<br/>

**Step 2. Use main_train.py for training Patch SVDD Encoder and classifier**











