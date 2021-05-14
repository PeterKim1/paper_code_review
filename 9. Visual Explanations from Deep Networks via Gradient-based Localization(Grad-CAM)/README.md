# Grad-CAM Codes

<br/>

<br/>

수정 진행중



# 2. Pre-trained Grad-CAM(just Grad-CAM)

ImageNet Pre-trained resnet18을 사용하여 Grad-CAM을 생성합니다.

[Grad-CAM_pretrained.ipynb](https://github.com/PeterKim1/paper_code_review/blob/master/9.%20Visual%20Explanations%20from%20Deep%20Networks%20via%20Gradient-based%20Localization(Grad-CAM)/Grad-CAM_pretrained.ipynb) 파일을 통해 구현하실 수 있습니다.

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

<br/>

<br/>

# 3. Pre-trained Grad-CAM(with Guided Backprop)

ImageNet Pre-trained resnet50을 사용하여 Grad-CAM을 생성합니다.

(resnet18로 결과를 확인해보면 생각보다 깔끔하게 고양이와 개가 분류되지 않아 resnet50을 사용하였습니다. 코드에서 모델의 종류를 resnet18으로 변경하여도 구동에는 문제가 없습니다.)

[Grad-CAM_pretrained-Cat+Dog.ipynb](https://github.com/PeterKim1/paper_code_review/blob/master/9.%20Visual%20Explanations%20from%20Deep%20Networks%20via%20Gradient-based%20Localization(Grad-CAM)/Grad-CAM_pretrained-Cat%2BDog.ipynb) 파일을 통해서 구현하실 수 있습니다.

해당 코드는 논문의 Fig. 1처럼 이미지 내에서 개와 고양이가 동시에 존재할 때, class-discriminative 하면서 high-resolution인 Guided Grad-CAM을 만드는 코드입니다.

이를 통해 논문에서 제시한 방법론이 유효한지를 확인합니다.

<br/>

### 3.1 Cat class

Imagenet class index 기준 281인 Tabby Cat에 대한 예측 score를 기반으로 Grad-CAM을 만들었을 때의 결과를 나타냅니다.

<br/>

![image](https://user-images.githubusercontent.com/57930520/118268363-4526ec80-b4f8-11eb-900d-83db6c05e39c.png)

<br/>

Guided Backpropagation을 이용했을 때는 fine-grained feature들을 잘 잡아낸다는 것을 확인할 수 있습니다.

하지만 이는 class-discriminative하지 않으며, 개와 고양이 모두에 대한 feature를 잡아냅니다.

Grad-CAM은 고양이에 대한 region은 잘 잡아내지만, 단순히 region만을 잡아냅니다.

따라서 Grad-CAM과 Guided Backpropagation을 곱하게 되면 Guided Grad-CAM이 만들어지고 이는 class-discriminative하면서도 high-resolution한 이미지를 만들어냅니다.

<br/>

### 3.2 Dog class

Imagenet class index 기준 254인 pug-dog에 대한 예측 score를 기반으로 Grad-CAM을 만들었을 때의 결과를 나타냅니다.

![image](https://user-images.githubusercontent.com/57930520/118268943-06456680-b4f9-11eb-9cb3-92b385e3f9ec.png)

<br/>









