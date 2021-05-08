# WRN Codes

<br/>

* 모든 실험은 CIFAR-10 Dataset을 활용하여 진행하였습니다.

<br/>

## 1. WRN Main architecture

<br/>

### 1.1 Main code

<br/>

```python
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8, 1, 0)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
```
<br/> 

### 1.2 WRN-40-2

![image](https://user-images.githubusercontent.com/57930520/116812848-d5297580-ab8b-11eb-94e6-563e0fdd52f1.png)

<br/>

### 1.3 WRN-16-8

![image](https://user-images.githubusercontent.com/57930520/116812869-fa1de880-ab8b-11eb-8e5a-b7df982fe2fd.png)



### 1.4 WRN-16-10

![image](https://user-images.githubusercontent.com/57930520/117536885-9a668800-b038-11eb-8c4b-48557eb417a0.png)






# 2. Loss Graphs



<br/>

### 2.1 WRN-40-2

![image](https://user-images.githubusercontent.com/57930520/116696198-8e594580-a9fc-11eb-91f9-426e7ed847c2.png)

![image](https://user-images.githubusercontent.com/57930520/116696244-9ca76180-a9fc-11eb-89ee-6b88450c1b2c.png)

<br/>

### 2.2 WRN-16-8

![image](https://user-images.githubusercontent.com/57930520/116812802-93003400-ab8b-11eb-960a-a000e33eec5c.png)



<br/>



### 2.3 WRN-16-10

![image](https://user-images.githubusercontent.com/57930520/117536803-3217a680-b038-11eb-8b47-0acc00455867.png)





# 3. Experiment Results

<br/>

<br/>

실험에 사용된 각 Architecture와 Top Test Accuracy, 그리고 이를 도달했을 때의 epoch을 나타냅니다.



| Network Architecture | Top Test Accuracy | Epoch when reach to Top Test Accuracy |
| -------------------- | ----------------- | ------------------------------------- |
| WRN-40-2             | 95.22%            | 185 epoch                             |
| WRN-16-8             | 95.65%            | 199 epoch                             |
| WRN-16-10            | 95.82%            | 175 epoch                             |

<br/>



