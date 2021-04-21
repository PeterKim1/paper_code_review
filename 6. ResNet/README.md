# ResNet Codes

<br/>

* 모든 실험은 CIFAR-10 Dataset을 활용하여 진행하였습니다.

<br/>

## 1. ResNet Main architecture

<br/>

### 1.1 Main code

<br/>

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```
<br/>

논문에 나온 내용을 토대로, ResNet18과 ResNet34는 BasicBlock을, ResNet50 이상은 Bottleneck 구조를 사용합니다.

ResNet의 마지막 Layer에 softmax가 없는 이유는, loss function인 nn.CrossEntropyLoss()가 log_softmax와 nll_loss를 함께 포함하고 있기 때문입니다. 



<br/> 

### 1.2 ResNet18

![image](https://user-images.githubusercontent.com/57930520/115508940-d53a9300-a2b8-11eb-9eb1-6568b4afa6dc.png)

<br/>

### 1.3 ResNet50

![image](https://user-images.githubusercontent.com/57930520/115548131-11361e00-a2e2-11eb-92da-7729a45681ce.png)




# 2. Loss Graphs



<br/>

### 2.1 ResNet18

![image](https://user-images.githubusercontent.com/57930520/115562987-05525800-a2f2-11eb-839b-c633821bf52c.png)

![image](https://user-images.githubusercontent.com/57930520/115563025-0f745680-a2f2-11eb-8687-c389d9135e67.png)

<br/>

### 2.2 ResNet50

![image](https://user-images.githubusercontent.com/57930520/115563108-261aad80-a2f2-11eb-9b12-5ab633e2b393.png)

![image](https://user-images.githubusercontent.com/57930520/115563139-2f0b7f00-a2f2-11eb-9196-cb9a788d25b1.png)



<br/>



# 3. Experiment Results

<br/>

<br/>

실험에 사용된 각 Architecture와 Top Test Accuracy, 그리고 이를 도달했을 때의 epoch을 나타냅니다.



| Network Architecture | Top Test Accuracy | Epoch when reach to Top Test Accuracy |
| -------------------- | ----------------- | ------------------------------------- |
| ResNet18             | 95.45%            | 181 epoch                             |
| ResNet50             | 95.64%            | 179 epoch                             |

<br/>



