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

<br/>

## 2. Training

**Step 1. you need to set 'DATASET_PATH' variable.**

Set [DATASET_PATH](https://github.com/PeterKim1/paper_code_review/blob/master/10.%20Patch%20SVDD/codes/mvtecad.py#L8) to the root path of the dataset.

This code use MVTec AD dataset, but you can use your custom dataset. 

(I also use this model for my custom anomaly detection dataset.)

<br/>

**Step 2. Use [main_train.py](https://github.com/PeterKim1/paper_code_review/blob/master/10.%20Patch%20SVDD/main_train.py) for training Patch SVDD Encoder and classifier**

```
python main_train.py --obj=bottle --lr=1e-4 --lambda_value=1e-3 --D=64 --epochs=300
```

* `obj` denotes the name of the class out of 15 MVTec AD classes. 
  If you use custom dataset, maybe you need to delete this argument or modify functions which have `obj` argument.
* `lr` denotes the learning rate of Adam optimizer. (default = 1e-4)
* `lambda_value` denotes the value of 'lambda' in Eq. 6 of the paper. (default = 1)
* `D` denotes the number of embedding dimension (default = 64)
* `epochs` denotes the number of training epoch (default = 300)

<br/>

After training, you can get trained encoder weights. 

You can change the path of saving weights ->  [here](https://github.com/PeterKim1/paper_code_review/blob/master/10.%20Patch%20SVDD/codes/networks.py#L172)

Examples of saved weights ->   [here](https://github.com/PeterKim1/paper_code_review/tree/master/10.%20Patch%20SVDD/ckpts/bottle)

I changed the code related to saving weights, becuz I need to experiment several times with changing hyperparameters(lambda, epochs, etc...)

Additionally, evaluation process takes a long time, so I exclude evaluation process [like this.](https://github.com/PeterKim1/paper_code_review/blob/master/10.%20Patch%20SVDD/main_train.py#L110)

<br/>

# 3. Evaluation

```
python main_evaluate.py --obj=bottle --epochs=300 --lambda_value=1e-3 --D=64
```

* `obj` denotes the name of class.
* `epochs` denote the number of epochs when I used to train Patch SVDD Encoder.
* `lambda_value` denote the value of 'lambda' when I used to train Patch SVDD Encoder.
* `D` denote the number of embedding dimension.

<br/>

main_evaluate.py loads the trained encoder saved in ckpts directory.

`enchier.pkl` files in ckpts directory is trained encoder weights given by the author of Patch SVDD.

`300_0.001_64_enchier.pkl` and `600_0.001_64_enchier.pkl` in [ckpts/bottle](https://github.com/PeterKim1/paper_code_review/tree/master/10.%20Patch%20SVDD/ckpts/bottle) directory is trained encoder weights by me. 







