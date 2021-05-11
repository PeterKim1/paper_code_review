# Grad-CAM Codes

<br/>

<br/>

수정 진행중



# Pre-trained Grad-CAM

ImageNet Pre-trained resnet18을 사용하여 Grad-CAM을 생성합니다.

Grad-CAM_pretrained.ipynb 파일을 통해 구현하실 수 있습니다.



### Grad-CAM made by last conv layer

left: Original image / center: Grad-CAM / right: Original image + Grad-CAM

![image](https://user-images.githubusercontent.com/57930520/117650329-899a4b80-b1cb-11eb-9ce8-c3edda776ba6.png)

<br/>

### Grad-CAM made by third conv layer

left: Original image / center: Grad-CAM / right: Original image + Grad-CAM

![image](https://user-images.githubusercontent.com/57930520/117650532-ca926000-b1cb-11eb-971c-be7f62bea702.png)

<br/>

### Grad-CAM made by second conv layer

left: Original image / center: Grad-CAM / right: Original image + Grad-CAM

![image](https://user-images.githubusercontent.com/57930520/117650572-db42d600-b1cb-11eb-91a8-224a80d67980.png)

<br/>

### Grad-CAM made by first conv layer

left: Original image / center: Grad-CAM / right: Original image + Grad-CAM

![image](https://user-images.githubusercontent.com/57930520/117650742-0a594780-b1cc-11eb-9017-fbcbbca497be.png)

<br/>

기존의 CAM의 경우, 마지막 conv layer의 결과만을 이용하여 만들 수 있었으나 Grad-CAM의 경우 마지막 conv layer 외의 layer에도 같은 방법론을 적용해 Grad-CAM을 얻어낼 수 있습니다.

따라서 다른 conv layer를 사용해서 얻어지는 Grad-CAM을 만들어보았습니다.

처음에 가까운 layer 일수록 receptive field가 작으므로 매우 localized region만을 본다는 것을 확인할 수 있으며 가장 마지막 conv layer는 매우 넓은 영역을 conv가 인식하고 있음을 확인할 수 있었습니다.



### Grad-CAM vs Class Activation Map

![image](https://user-images.githubusercontent.com/57930520/117653530-78ebd480-b1cf-11eb-9cd7-2398e7c2c2a0.png)

같은 그림에 대해서 CAM과 Grad-CAM을 그려보았을 때, 동일한 결과가 도출되는 것을 알 수 있습니다.



