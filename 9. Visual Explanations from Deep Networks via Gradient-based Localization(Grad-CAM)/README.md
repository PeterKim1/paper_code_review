# Grad-CAM Codes

<br/>

<br/>

수정 진행중



# 2. Pre-trained Grad-CAM(just Grad-CAM)

ImageNet Pre-trained resnet18을 사용하여 Grad-CAM을 생성합니다.

Grad-CAM_pretrained.ipynb 파일을 통해 구현하실 수 있습니다.

해당 코드는 image를 입력 받았을 때, Pre-trained resnet18 weight를 이용해 classification을 진행하고 이에 대한 Grad-CAM heatmap 및 Grad-CAM + image를 만들어줍니다.

이번 section에서는 다양한 layer에서 추출된 feature map을 통해서 Grad-CAM을 만들어보고 어떤 차이가 있는지를 확인합니다.



### 2.1 Grad-CAM made by last conv layer

left: Original image / center: Grad-CAM / right: Original image + Grad-CAM

![image](https://user-images.githubusercontent.com/57930520/117650329-899a4b80-b1cb-11eb-9ce8-c3edda776ba6.png)

<br/>

### 2.2 Grad-CAM made by third conv layer

left: Original image / center: Grad-CAM / right: Original image + Grad-CAM

![image](https://user-images.githubusercontent.com/57930520/117650532-ca926000-b1cb-11eb-971c-be7f62bea702.png)

<br/>

### 2.3 Grad-CAM made by second conv layer

left: Original image / center: Grad-CAM / right: Original image + Grad-CAM

![image](https://user-images.githubusercontent.com/57930520/117650572-db42d600-b1cb-11eb-91a8-224a80d67980.png)

<br/>

### 2.4 Grad-CAM made by first conv layer

left: Original image / center: Grad-CAM / right: Original image + Grad-CAM

![image](https://user-images.githubusercontent.com/57930520/117650742-0a594780-b1cc-11eb-9017-fbcbbca497be.png)

<br/>

기존의 CAM의 경우, 마지막 conv layer의 결과만을 이용하여 만들 수 있었으나 Grad-CAM의 경우 마지막 conv layer 외의 layer에도 같은 방법론을 적용해 Grad-CAM을 얻어낼 수 있습니다.

따라서 다른 conv layer를 사용해서 얻어지는 Grad-CAM을 만들어보았습니다.

처음에 가까운 layer 일수록 receptive field가 작으므로 매우 localized region만을 본다는 것을 확인할 수 있으며 가장 마지막 conv layer는 매우 넓은 영역을 conv가 인식하고 있음을 확인할 수 있었습니다.



# 3. Pre-trained Grad-CAM(with Guided Backprop)

ImageNet Pre-trained resnet50을 사용하여 Grad-CAM을 생성합니다.

(resnet18로 결과를 확인해보면 생각보다 깔끔하게 고양이와 개가 분류되지 않아 resnet50을 사용하였습니다. 코드에서 모델의 종류를 resnet18으로 변경하여도 구동에는 문제가 없습니다.)













