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

## 2.2 ResNet / Wide ResNet model

```python
python main.py --data_path='.../MVTec' --save_path='./mvtec_result' --arch='resnet18'
```

* `data_path` denotes location of MVTec AD dataset.
* `save_path` denotes location of saving result.
* `arch` denotes what kinds of architecture to use.

<br/>

## 3. Result

<br/>

## 3.1 EfficientNet-B5 model

* Image-level anomaly detection result (AUROC)

| MVTec AD class          | EfficientNet-B5<br />(This code) | EfficientNet-B5<br />(Original Paper) |
| ----------------------- | -------------------------------- | ------------------------------------- |
| Carpet                  | 1.0                              | -                                     |
| Grid                    | 0.984                            | -                                     |
| Leather                 | 1.0                              | -                                     |
| Tile                    | 0.992                            | -                                     |
| Wood                    | 0.987                            | -                                     |
| **All texture classes** | **0.9926**                       | **0.990**                             |
| Bottle                  | 1.0                              | -                                     |
| Cable                   | 0.973                            | -                                     |
| Capsule                 | 0.956                            | -                                     |
| Hazelnut                | 0.974                            | -                                     |
| Metal Nut               | 0.980                            | -                                     |
| Pill                    | 0.923                            | -                                     |
| Screw                   | 0.927                            | -                                     |
| Toothbrush              | 0.994                            | -                                     |
| Transistor              | 1.0                              | -                                     |
| Zipper                  | 0.954                            | -                                     |
| **All object classes**  | **0.9681**                       | **0.972**                             |
| **All classes**         | **0.9763**                       | **0.979**                             |

<br/>

<br/>

* Pixel-level anomaly localization result (AUROC)

| MVTec AD class          | EfficientNet-B5<br />(This code) | EfficientNet-B5<br />(Original Paper) |
| ----------------------- | -------------------------------- | ------------------------------------- |
| Carpet                  | 0.943                            | -                                     |
| Grid                    | 0.883                            | -                                     |
| Leather                 | 0.947                            | -                                     |
| Tile                    | 0.894                            | -                                     |
| Wood                    | 0.868                            | -                                     |
| **All texture classes** | **0.907**                        | -                                     |
| Bottle                  | 0.944                            | -                                     |
| Cable                   | 0.936                            | -                                     |
| Capsule                 | 0.977                            | -                                     |
| Hazelnut                | 0.951                            | -                                     |
| Metal Nut               | 0.955                            | -                                     |
| Pill                    | 0.939                            | -                                     |
| Screw                   | 0.962                            | -                                     |
| Toothbrush              | 0.975                            | -                                     |
| Transistor              | 0.974                            | -                                     |
| Zipper                  | 0.934                            | -                                     |
| **All object classes**  | **0.9547**                       | -                                     |
| **All classes**         | **0.9388**                       | -                                     |

In the original paper, no Pixel-level anomaly localization results are presented.

So, I can't compare this result with original paper's result.

Compared with image-level AUROC, pixel-level AUROC is little low.



* ROC Curve(EfficientNet-B5)

![roc_curve (1)](https://user-images.githubusercontent.com/57930520/125718833-e485617b-2527-4ca9-8824-cfe6e06e4356.png)



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



## 3.2 ResNet / Wide ResNet model

* Image-level anomaly detection result (AUROC)

| MvTec AD class          | R18-Rd100<br />(This code) | R18-Rd100<br />(Original Paper) | WR50-Rd550<br />(This code) | WR50-Rd550<br />(Original Paper) |
| ----------------------- | -------------------------- | ------------------------------- | --------------------------- | -------------------------------- |
| Carpet                  | 0.984                      | -                               | 0.999                       | -                                |
| Grid                    | 0.898                      | -                               | 0.957                       | -                                |
| Leather                 | 0.988                      | -                               | 1.0                         | -                                |
| Tile                    | 0.959                      | -                               | 0.974                       | -                                |
| Wood                    | 0.990                      | -                               | 0.988                       | -                                |
| **All texture classes** | **0.964**                  | -                               | **0.984**                   | **0.988**                        |
| Bottle                  | 0.996                      | -                               | 0.998                       | -                                |
| Cable                   | 0.855                      | -                               | 0.922                       | -                                |
| Capsule                 | 0.870                      | -                               | 0.915                       | -                                |
| Hazelnut                | 0.841                      | -                               | 0.933                       | -                                |
| Metal Nut               | 0.974                      | -                               | 0.992                       | -                                |
| Pill                    | 0.869                      | -                               | 0.944                       | -                                |
| Screw                   | 0.745                      | -                               | 0.844                       | -                                |
| Toothbrush              | 0.947                      | -                               | 0.972                       | -                                |
| Transistor              | 0.925                      | -                               | 0.978                       | -                                |
| Zipper                  | 0.741                      | -                               | 0.909                       | -                                |
| **All object classes**  | **0.876**                  | -                               | **0.941**                   | **0.936**                        |
| **All classes**         | **0.905**                  | -                               | **0.955**                   | **0.953**                        |

<br/>

* Pixel-level anomaly localization result (AUROC)

| MvTec AD class          | R18-Rd100<br />(This code) | R18-Rd100<br />(Original Paper) | WR50-Rd550<br />(This code) | WR50-Rd550<br />(Original Paper) |
| ----------------------- | -------------------------- | ------------------------------- | --------------------------- | -------------------------------- |
| Carpet                  | 0.988                      | 0.989                           | 0.990                       | 0.991                            |
| Grid                    | 0.936                      | 0.949                           | 0.965                       | 0.973                            |
| Leather                 | 0.990                      | 0.991                           | 0.989                       | 0.992                            |
| Tile                    | 0.917                      | 0.912                           | 0.939                       | 0.941                            |
| Wood                    | 0.940                      | 0.936                           | 0.941                       | 0.949                            |
| **All texture classes** | **0.953**                  | **0.956**                       | **0.965**                   | **0.969**                        |
| Bottle                  | 0.981                      | 0.981                           | 0.982                       | 0.983                            |
| Cable                   | 0.949                      | 0.958                           | 0.968                       | 0.967                            |
| Capsule                 | 0.982                      | 0.983                           | 0.986                       | 0.985                            |
| Hazelnut                | 0.979                      | 0.977                           | 0.979                       | 0.982                            |
| Metal Nut               | 0.967                      | 0.967                           | 0.971                       | 0.972                            |
| Pill                    | 0.946                      | 0.947                           | 0.961                       | 0.957                            |
| Screw                   | 0.972                      | 0.974                           | 0.983                       | 0.985                            |
| Toothbrush              | 0.986                      | 0.987                           | 0.987                       | 0.988                            |
| Transistor              | 0.968                      | 0.972                           | 0.975                       | 0.975                            |
| Zipper                  | 0.976                      | 0.982                           | 0.984                       | 0.985                            |
| **All object classes**  | **0.971**                  | **0.973**                       | **0.978**                   | **0.978**                        |
| **All classes**         | **0.965**                  | **0.967**                       | **0.973**                   | **0.975**                        |

<br/>

* ROC Curve(R18-Rd100)

![roc_curve_r18](https://user-images.githubusercontent.com/57930520/125166382-63e2f000-e1d6-11eb-835a-0951157dbc88.png)

<br/>

* ROC Curve(WR50-Rd550)

![roc_curve_wr50](https://user-images.githubusercontent.com/57930520/125166427-81b05500-e1d6-11eb-99e5-882dabed71b2.png)

<br/>



# 4. Reference

* https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master

