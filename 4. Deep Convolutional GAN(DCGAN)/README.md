# Deep Convolutional GAN (DCGAN) Codes

<br/>

<br/>

## 1. DCGAN Main architecture

<br/>

### 1.1 Generator

<br/>

```python
# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is (batch, latent_size, 1, 1)
            nn.ConvTranspose2d(latent_size, hidden_g * 8, 4, 1, 0, bias=False),
            # (batch, latent_size, 1, 1) => (batch, 512, 4, 4)
            nn.BatchNorm2d(hidden_g * 8),
            nn.ReLU(False),
            nn.ConvTranspose2d(hidden_g * 8, hidden_g * 4, 4, 2, 1, bias=False), 
            # (batch, 512, 4, 4) => (batch, 256, 8, 8)
            nn.BatchNorm2d(hidden_g * 4),
            nn.ReLU(False),
            nn.ConvTranspose2d(hidden_g * 4, hidden_g * 2, 4, 2, 1, bias=False), 
            # (batch, 256, 8, 8) => (batch, 128, 16, 16)
            nn.BatchNorm2d(hidden_g * 2),
            nn.ReLU(False),
            nn.ConvTranspose2d(hidden_g * 2, hidden_g, 4, 2, 1, bias=False), 
            # (batch, 128, 16, 16) => (batch, 64, 32, 32)
            nn.BatchNorm2d(hidden_g),
            nn.ReLU(False),
            nn.ConvTranspose2d(hidden_g,      1, 4, 2, 1, bias=False), 
            # (batch, 64, 32, 32) => (batch, 1, 64, 64)
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output
```
<br/>

<br/>

### 1.2 Discriminator

<br/>

<br/>

```python
# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (batch, 1, 64, 64)
            nn.Conv2d(1, hidden_d, 4, 2, 1, bias=False), 
            # (batch, 1, 64, 64) => (batch, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(hidden_d, hidden_d * 2, 4, 2, 1, bias=False), 
            # (batch, 64, 32, 32) => (batch, 128, 16, 16)
            nn.BatchNorm2d(hidden_d * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(hidden_d * 2, hidden_d * 4, 4, 2, 1, bias=False), 
            # (batch, 128, 16, 16) => (batch, 256, 8, 8)
            nn.BatchNorm2d(hidden_d * 4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(hidden_d * 4, hidden_d * 8, 4, 2, 1, bias=False), 
            # (batch, 256, 8, 8) => (batch, 512, 4, 4)
            nn.BatchNorm2d(hidden_d * 8),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(hidden_d * 8, 1, 4, 1, 0, bias=False), 
            # (batch, 512, 4, 4) => (batch, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
```
<br/>

<br/>



# 2. Loss Graphs



<br/>

Red line : average discriminator loss per epoch

Blue line : average generator loss per epoch

<br/>

![img](https://blog.kakaocdn.net/dn/Xec8o/btq0Y3pMwfe/Sc1gJfWoKSIWuLhOFQu2dk/img.png)



<br/>

<br/>



# 3. Experiment Results

<br/>

<br/>

(16, 100, 1, 1) 차원의 Gaussian noise vector를 만든 후, 학습이 진행됨에 따라 이 Gaussian noise vector를 가지고 만들어지는 결과물을 표현한 gif입니다.

<br/>

## ![result2021-03-25-14_11_short](https://user-images.githubusercontent.com/57930520/112485122-ed081f80-8dbd-11eb-8bd9-29c846461d67.gif)

<br/>

<br/>



# 4. Latent space visualization

## 4.1 Walking in the latent space

<br/>

<br/>

(논문에서 6.1 Walking in the latent space에 해당)



(1, 100) 차원의 Gaussian noise vector 2개를 가지고 두 vector를 interpolate해서 만들어진 vector를 학습이 완료된 generator에 넣었을 때 얻게 되는 결과.

random variable이 어떻게 만들어지는지에 따라 다른 결과를 얻게 됩니다. 

 <br/>

(첫 번째)

![latent2](https://user-images.githubusercontent.com/57930520/112483299-3d7e7d80-8dbc-11eb-8fc8-d5d9a20ab26f.png)





<br/>

<br/>

(두 번째)

![latent1](https://user-images.githubusercontent.com/57930520/112589617-c3dea200-8e44-11eb-9f48-944e50583325.png)

<br/>

<br/>

# 5. 느낀점

* 실험을 일단은 MNIST로 진행했긴 했는데, 해당 논문에는 살짝 부적합하지 않나 싶다.
* 특히 latent space 부분에서, 논문에서 나온 vector arithmetic을 해보기가 워낙 애매해서 일단은 이 부분은 따로 하지 않았다.
* vector arithmetic을 하려면 CelebA 데이터셋을 가져와야 하지 않나 싶다. 다음주에 논문 발표 일정이 있어서 당장은 MNIST로 마무리 짓고 다음주에 여유가 되면 CelebA 버전을 따로 더 돌려서 해볼 예정에 있다.
* 이전에 간단한 GAN으로 돌렸을 때는 약간 지지직 거리게 이미지들이 생성되었고, VAE는 뿌옇게 나오는 현상이 보였는데 DCGAN은 그런게 훨씬 덜한 것 같다. 물론 아직도 정말 natural image에 가까워지기에는 멀었지만.....
* Generative model이라면 이미지가 만들어지는 gif를 만들어야 한다고 생각했어서 이번 기회에 gif을 만들어보았다. (이전 GAN 논문과 VAE 논문에서 나온 결과도 gif로 만드는 코드를 추가해서 추가할까 싶다.)
* 새삼 왜 많은 사람들이 generative model(생성 모델)에 관심을 가지는지 느끼게 되는 것 같다. 단순히 training sample을 memorization하는게 아니라 유사하게 생성한다는 점이 매우 흥미로웠다. 
* 근데 여전히 GAN framework의 한계점인진 모르겠지만 학습의 결과가 일정하지는 않다.... 같은 코드를 돌리는데 어떨때는 Discriminator가 압도적으로 잘 학습되어서 Generator가 학습을 못 하게 되는 상황에 빠지기도 했다. 역시 어렵다..