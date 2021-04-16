# Wasserstein GAN (WGAN) Codes

<br/>

<br/>

## 1. WGAN Main architecture

<br/>

<br/>

### 1.1 Generator (DCGAN architecture)

<br/>

```python
class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
```
<br/>

### 1.2 Discriminator (DCGAN architecture)

<br/>

```python
class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
        )

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
```




# 2. Loss Graphs



<br/>

First: Generator loss

Second: Wasserstein estimate (Discriminator loss)

Third: loss per epoch (Red Line: Generator loss / Blue Line: Wasserstein estimate)

![image](https://user-images.githubusercontent.com/57930520/114997404-c0d25100-9eda-11eb-9449-f683bd9462ea.png)

<br/>

해당 실험에서는 weight clipping의 값을 논문과는 다르게 0.005로 주었는데, 이는 논문의 값인 0.01로 적용하였을 때 Wasserstein estimate가 하향하는 곡선으로 결과가 나오지 않았기 때문입니다.





# 3. Experiment Results

<br/>

<br/>

고정된 (10, 100, 1, 1)의 latent vector를 만든 후, 학습이 진행됨에 따라서 latent vector가 어떤 그림으로 바뀌는지를 gif로 만든 결과입니다.



![result_5](https://user-images.githubusercontent.com/57930520/114997653-0abb3700-9edb-11eb-8d68-5cffccb3bd4a.gif)



<br/>

<br/>

# 4. 느낀점

* 데이터셋이 달라서인지 어떤 이유에서 인지는 모르겠지만 논문에서는 엄청 깔끔한 이미지가 얻어지는 것과는 달리, 내가 했을 때는 그렇게 깔끔한 이미지가 얻어지지 않았다.
* 그리고 분명 논문에서는 학습의 안정성이 좋아졌다고 했는데, 데이터셋을 바꾸거나 혹은 Discriminator와 Generator의 구조만 바꾸더라도 같은 코드에서도 바로 이상한 결과가 나온다. 여전히 학습의 안정성 문제가 있다고 보여진다.
* weight clipping의 값을 0.01로 계속해서 실험해봤는데도 불구하고 결과가 너무 나오지 않아 여러가지 자료를 찾아보니 weight clipping의 값에 따라서 실험 결과에 영향을 크게 미친다는 사실을 알게 되었고 궁금해서 이를 더 줄여보았더니 그래도 멀쩡한 그래프가 나왔다.
* WGAN을 사용할 때는 weight clipping의 값을 변화시키면서 실험해볼 필요성이 있다고 느껴진다.
* 그리고 critic(Discriminator)를 generator에 비해서 얼마나 더 많이 실험할지도 실험결과에 영향을 미치기 때문에, 좋은 결과를 얻기 위해서는 이 또한 튜닝할 필요가 있다고 생각했다.
* 생성 모델이라는 것은 분명 신기하고 재미있는 분야라고 생각되지만, 좋은 결과를 얻기가 쉽지 않다라는 사실을 항상 느낀다. 물론 이건 내가 초보라서 그럴지도...