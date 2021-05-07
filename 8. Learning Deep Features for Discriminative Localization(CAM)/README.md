# CAM Codes

<br/>

* 이번 모델은 Cat vs Dog Dataset을 이용하여 실험을 진행하였습니다.
* 해당 코드를 재현하려면, Cat vs Dog Dataset을 다운받으셔야 합니다.
* 따라서, 먼저 Dataset을 받는 부분부터 소개하고, 그 다음 모델과 관련된 내용을 소개합니다.



![image](https://user-images.githubusercontent.com/57930520/117242459-4741d900-ae70-11eb-95ae-4456073167fb.png)

![image](https://user-images.githubusercontent.com/57930520/117242480-5163d780-ae70-11eb-88c1-ca5ed23b526f.png)

![image](https://user-images.githubusercontent.com/57930520/117242518-5fb1f380-ae70-11eb-8207-761b74c0c909.png)



<br/>

## 1. How to download cat vs dog dataset?

<br/>

### https://www.kaggle.com/c/dogs-vs-cats/data

해당 링크를 들어가서, kaggle 아이디로 접속한 뒤, "train.zip" 파일을 받으셔야 합니다.

![image](https://user-images.githubusercontent.com/57930520/117242312-f5994e80-ae6f-11eb-8128-62d7b7631b31.png)

"test1.zip"의 경우, label이 없으므로 지도학습을 진행할 수 없어 "train.zip"만 사용합니다.

<br/>




# 2. Setting cat vs dog dataset

<br/>

CatvsDog_dataset_making.ipynb를 이용하시면, train dataset과 test dataset을 구성하실 수 있습니다.

기존 CatvsDog train dataset은 고양이 12,500장, 개 12,500장 (총 25,000장)으로 구성되어 있었으며 이를 train과 test 모두에 이용하기 위해 train 고양이 10,000장, 개 10,000장(총 20,000장)으로 구성하고 test 고양이 2,500장, 개 2,500장(총 5,000장)으로 구성하였습니다.

<br/>



# 3. Baseline Model

<br/>

Baseline Model로는 Resnet18을 사용하였습니다.



Resnet에 대한 상세한 사항은 제 [Resnet folder](https://github.com/PeterKim1/paper_code_review/tree/master/6.%20ResNet) 를 참고해주시면 됩니다.



해당 코드에서는 Resnet이 메인이 아니므로, torchvision에서 지원하는 resnet18 모델을 그대로 가져와서 사용하였고, torchvision의 resnet18은 ImageNet을 대상으로 구축된 모델이여서 1000 class classification model입니다.



따라서 이를 Cat과 Dog를 분류하는 binary classification model로 만들고자 마지막 fully connected layer를 2차원으로 만들어 binary classification이 가능하도록 변경하였습니다.



해당 Baseline Model의 training graph는 다음과 같습니다.

Red Line: Test, Blue Line: Train

![image](https://user-images.githubusercontent.com/57930520/117245117-40699500-ae75-11eb-8b3a-296868c49775.png)



# 4. Class Activation Map

<br/>

![image](https://user-images.githubusercontent.com/57930520/117415460-4d5cb600-af53-11eb-867b-2db932ec994c.png)



논문에 나온 대로, GAP을 거친 후(현재 resnet18 기준 512차원) 마지막 의사결정인 binary classification을 할 때 사용한 가중치를 활용하여 Class Activation Map을 생성합니다.

학습된 Baseline model weight를 이용하여 Class Activation Map을 사용할 수 있으며, 이는 Class_Activation_Map.ipynb 파일을 이용해서 생성할 수 있습니다.



Class_Activation_Map.ipynb에서 조절할 수 있는 주요한 hyperparameter는 총 3개 입니다.



1. num_result : 결과의 개수를 결정합니다.
2. result = heatmap * a + img * b: 해당 코드에서는 a가 0.7, b가 0.5이지만 이를 조절하면 결과의 이미지의 모습이 약간 달라집니다. Weighted sum 형태 이므로, heatmap을 더 진하게 표현할지 image를 더 진하게 표현할지에 따라서 조절하시면 됩니다.
3. props = generate_bbox(img, CAMs[0], CAMs[0].max()*a) : a는 Class Activation Map의 최대값에 대해 몇 %보다 높은 값을 갖고 bounding box를 만들지 결정합니다. a가 낮을 경우, image 사이즈와 동일한 bounding box를 만드므로 해당 방법의 의미를 잃게 되고, a가 높을 경우, 과하게 작은 형태의 bounding box를 만들게 되므로 개나 고양이의 매우 작은 부분만 포함하는 bounding box가 만들어질 수 있습니다. 해당 코드에서는 60%를 사용하였습니다.





# 5. Experiment examples



(Original / Class Activation Map / Original + Class Activation Map + patch로 구성하였습니다.)



![img1.png](https://github.com/PeterKim1/paper_code_review/blob/master/8.%20Learning%20Deep%20Features%20for%20Discriminative%20Localization(CAM)/Examples/img1.png?raw=true)

![cam1.png](https://github.com/PeterKim1/paper_code_review/blob/master/8.%20Learning%20Deep%20Features%20for%20Discriminative%20Localization(CAM)/Examples/cam1.png?raw=true)

![Result1.png](https://github.com/PeterKim1/paper_code_review/blob/master/8.%20Learning%20Deep%20Features%20for%20Discriminative%20Localization(CAM)/Examples/Result1.png?raw=true)



![img2.png](https://github.com/PeterKim1/paper_code_review/blob/master/8.%20Learning%20Deep%20Features%20for%20Discriminative%20Localization(CAM)/Examples/img2.png?raw=true)

![cam2.png](https://github.com/PeterKim1/paper_code_review/blob/master/8.%20Learning%20Deep%20Features%20for%20Discriminative%20Localization(CAM)/Examples/cam2.png?raw=true)

![Result2.png](https://github.com/PeterKim1/paper_code_review/blob/master/8.%20Learning%20Deep%20Features%20for%20Discriminative%20Localization(CAM)/Examples/Result2.png?raw=true)



![img3.png](https://github.com/PeterKim1/paper_code_review/blob/master/8.%20Learning%20Deep%20Features%20for%20Discriminative%20Localization(CAM)/Examples/img3.png?raw=true)

![cam3.png](https://github.com/PeterKim1/paper_code_review/blob/master/8.%20Learning%20Deep%20Features%20for%20Discriminative%20Localization(CAM)/Examples/cam3.png?raw=true)

![Result3.png](https://github.com/PeterKim1/paper_code_review/blob/master/8.%20Learning%20Deep%20Features%20for%20Discriminative%20Localization(CAM)/Examples/Result3.png?raw=true)





















