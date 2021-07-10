# 11. PaDiM(a Patch Distribution Modeling Framework for Anomaly Detection and Localization) codes

This is **unofficial** PaDiM(Patch Distribution Modeling Framework for Anomaly Detection and Localization) codes.

I implemented Three pre-trained models(ResNet, Wide ResNet, EfficientNet-B5) in original paper.

Paper: https://arxiv.org/abs/2011.08785

<br/>

## 0. Used package versions

efficientnet_pytorch==0.7.1

numpy==1.19.5

tqdm==4.41.1

scikit-learn==0.22.2.post1

scikit-image==0.16.2

matplotlib==3.2.2

torch==1.9.0+cu102

Pillow==7.1.2

torchvision==0.10.0+cu102

<br/>

## 1. MVTec AD dataset installation

In this code, use MVTec AD dataset.

So, you need to download this dataset.

[Download link](https://www.mvtec.com/company/research/datasets/mvtec-ad/) 

<br/>

## 2. How to run this codes?

<br/>

## 2.1 EfficientNet-B5 model

```python
python main_eff_net.py --data_path='.../MVTec' --save_path='./mvtec_result'
```

* `data_path` denotes location of MVTec AD dataset.
* `save_path` denotes location of saving result.

<br/>

<br/>

## 3. Result

<br/>

## 3.1 EfficientNet-B5 model

* Image-level anomaly detection result (AUROC)

| MVTec AD class                                 | EfficientNet-B5<br />(This code)         | EfficientNet-B5<br />(Original Paper)    |
| ---------------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| Carpet                                         | 1.0                                      | -                                        |
| Grid                                           | 0.985                                    | -                                        |
| Leather                                        | 1.0                                      | -                                        |
| Tile                                           | 0.991                                    | -                                        |
| Wood                                           | 0.976                                    | -                                        |
| **All texture classes**                        | **0.9904**                               | **0.990**                                |
| Bottle                                         | 1.0                                      | -                                        |
| Cable                                          | 0.959                                    | -                                        |
| Capsule                                        | 0.945                                    | -                                        |
| Hazelnut                                       | 0.829                                    | -                                        |
| Metal Nut                                      | 0.944                                    | -                                        |
| Pill                                           | 0.970                                    | -                                        |
| Screw                                          | 0.911                                    | -                                        |
| Toothbrush                                     | 0.994                                    | -                                        |
| Transistor                                     | 0.998                                    | -                                        |
| Zipper                                         | 0.938                                    | -                                        |
| **All object classes**                         | **0.9488**                               | **0.972**                                |
| **<span style="color:red">All classes</span>** | **<span style="color:red">0.963</span>** | **<span style="color:red">0.979</span>** |

<br/>

In Image-level anomaly detection, all textures classes result is similar to original paper.

But all objects classes result is different from original paper, It has 2.32% difference from original paper.

So, All classes result has 1.6% difference from original paper.

<br/>

* Pixel-level anomaly localization result (AUROC)

| MVTec AD class                                 | EfficientNet-B5<br />(This code)         | EfficientNet-B5<br />(Original Paper) |
| ---------------------------------------------- | ---------------------------------------- | ------------------------------------- |
| Carpet                                         | 0.983                                    | -                                     |
| Grid                                           | 0.958                                    | -                                     |
| Leather                                        | 0.983                                    | -                                     |
| Tile                                           | 0.917                                    | -                                     |
| Wood                                           | 0.923                                    | -                                     |
| **All texture classes**                        | **0.9528**                               | -                                     |
| Bottle                                         | 0.973                                    | -                                     |
| Cable                                          | 0.972                                    | -                                     |
| Capsule                                        | 0.981                                    | -                                     |
| Hazelnut                                       | 0.970                                    | -                                     |
| Metal Nut                                      | 0.955                                    | -                                     |
| Pill                                           | 0.953                                    | -                                     |
| Screw                                          | 0.977                                    | -                                     |
| Toothbrush                                     | 0.984                                    | -                                     |
| Transistor                                     | 0.982                                    | -                                     |
| Zipper                                         | 0.963                                    | -                                     |
| **All object classes**                         | **0.9701**                               | -                                     |
| **<span style="color:red">All classes</span>** | **<span style="color:red">0.965</span>** | -                                     |

In the original paper, no Pixel-level anomaly localization results are presented.

So, I can't compare this result with original paper's result.



* ROC Curve

![roc_curve](https://user-images.githubusercontent.com/57930520/125163143-c0d6aa00-e1c6-11eb-9543-4f7068ce9f52.png)



* Localization examples

  - Bottle

  ![bottle_0](https://user-images.githubusercontent.com/57930520/125163270-5c681a80-e1c7-11eb-9ac4-5fc5cfa08e20.png)

  <br/>

  - Cable

  ![cable_2](https://user-images.githubusercontent.com/57930520/125163295-7bff4300-e1c7-11eb-8099-835ab65ccd9d.png)

  <br/>

  - Capsule

  ![capsule_18](https://user-images.githubusercontent.com/57930520/125163336-a6510080-e1c7-11eb-9711-b4f034849770.png)

  <br/>

  - Carpet

  ![carpet_5](https://user-images.githubusercontent.com/57930520/125163452-31ca9180-e1c8-11eb-85ae-e5e1277c0ca0.png)

  <br/>

  - Grid

  ![grid_13](https://user-images.githubusercontent.com/57930520/125163485-532b7d80-e1c8-11eb-8e49-c419d9014f88.png)

  <br/>















