# MobileNetV1 Codes

* 실험은 STL-10 dataset을 가지고 실험하였습니다.
* ImageNet의 resolution인 224x224를 그대로 적용하고자, CIFAR-10이나 MNIST에 비해서 original resolution이 높은 데이터셋인 STL-10 dataset을 사용하였고 이를 224x224로 resize해서 사용하였습니다.



# 1. MobileNetV1 Main architecture

<br/>

## 1.1 Main code

```python
class MobileNetV1(nn.Module):
    def __init__(self, n_classes, alpha):
        super(MobileNetV1, self).__init__()
        
        assert alpha in {0.25, 0.5, 0.75, 1}

        self.alpha = alpha

        def conv_standard(in_channel, out_channel, s):
            return nn.Sequential(
                nn.Conv2d(int(in_channel), int(out_channel), kernel_size = 3, stride = s, padding = 1, bias = False),
                nn.BatchNorm2d(int(out_channel)),
                nn.ReLU(inplace = True)
            )
    
        def conv_mb(in_channel, out_channel, s):
            return nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(int(in_channel), int(in_channel), kernel_size = 3, stride = s, padding = 1, groups = int(in_channel), bias = False),
                nn.BatchNorm2d(int(in_channel)),
                nn.ReLU(inplace = True),

                # Pointwise Convolution
                nn.Conv2d(int(in_channel), int(out_channel), kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(int(out_channel)),
                nn.ReLU(inplace = True),
            )

        self.model = nn.Sequential(
            conv_standard(3, 32 * alpha, 2),
            conv_mb(32 * alpha, 64 * alpha, 1),
            conv_mb(64 * alpha, 128 * alpha, 2),
            conv_mb(128 * alpha, 128 * alpha, 1),
            conv_mb(128 * alpha, 256 * alpha, 2),
            conv_mb(256 * alpha, 256 * alpha, 1),
            conv_mb(256 * alpha, 512 * alpha, 2),
            conv_mb(512 * alpha, 512 * alpha, 1),
            conv_mb(512 * alpha, 512 * alpha, 1),
            conv_mb(512 * alpha, 512 * alpha, 1),
            conv_mb(512 * alpha, 512 * alpha, 1),
            conv_mb(512 * alpha, 512 * alpha, 1),
            conv_mb(512 * alpha, 1024 * alpha, 2),
            conv_mb(1024 * alpha, 1024 * alpha, 1),
            nn.AdaptiveAvgPool2d(1),
            )
        self.fc = nn.Linear(int(1024*alpha), n_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, int(1024*self.alpha))
        x = self.fc(x)
        return x
```



* 논문에서 언급된 Width Multiplier를 alpha라는 변수를 이용하여 적용할 수 있도록 설계하였습니다.
* 그리고 논문에 나온대로 alpha는 1, 0.75, 0.5, 0.25만 사용할 수 있도록 assert으로 지정해 다른 값들이 사용될 수 없도록 하였습니다.
* 대부분의 channel 수가 4의 배수이므로 channel 수가 정수로 떨어질 수 있도록 하기 위해 n/4의 값을 곱해준 것으로 판단됩니다.



# 2. Loss Graphs

## 2.1 1.0 MobileNet-224

![image](https://user-images.githubusercontent.com/57930520/125634226-b3834ca7-9414-48ad-8778-7d412a78f7b9.png)

* Accuracy graph - Train acc(Blue), Test acc(Red)
* Loss graph - Train loss(Red), test loss(Blue)

<br/>

## 2.2 0.75 MobileNet-224

![image](https://user-images.githubusercontent.com/57930520/125634407-4048269a-086e-4ca2-a954-777d5bb7bdf6.png)

* Accuracy graph - Train acc(Blue), Test acc(Red)
* Loss graph - Train loss(Red), test loss(Blue)





# 3. Experimental result

| Model              | Test Accuracy(50 epoch) | The number of parameters |
| ------------------ | ----------------------- | ------------------------ |
| 1.0 MobileNet-224  | 62.2125                 | 3,217,226                |
| 0.75 MobileNet-224 | 59.9875                 | 1,824,250                |
|                    |                         |                          |

