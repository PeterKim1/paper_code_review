# Deep Convolutional GAN (DCGAN) Codes

<br/>

* 2021-03-31 일부 code 및 결과가 변경되었습니다. README 추가 예정입니다 ㅠㅠㅠ
* It includes 32x32 MNIST / 32x32 FashionMNIST / 64x64 FashionMNIST experiment result.

<br/>

## 1. DCGAN Main architecture

<br/>

### 1.1.1 Generator (when use 32x32 images)

<br/>

```python
# Generator
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

### 1.1.2 Generator (when use 64x64 images, same as DCGAN paper)

<br/>

```python
# Generator
class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256, 128]
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
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx64x64)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
```
<br/>

### 1.2.1 Discriminator (when use 32x32 images)

<br/>

```python
# Discriminator
class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx32x32)
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
            nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
```
<br/>

### 1.2.2 Discriminator (when use 64x64 images, same as DCGAN paper)

<br/>

```python
# Discriminator
class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [128, 256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx64x64)
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x32x32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
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
            nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
```




# 2. Loss Graphs



<br/>

### 2.1 32x32 MNIST Dataset

Left: loss graph per batch

Right: loss graph per epoch

Red Line: Generator loss

Blue Line: Discriminator loss

![image](https://user-images.githubusercontent.com/57930520/113244781-a8661200-92f0-11eb-9734-bc876e5ab00e.png)

<br/>

### 2.2 32x32 Fashion MNIST Dataset

Left: loss graph per batch

Right: loss graph per epoch

Red Line: Generator loss

Blue Line: Discriminator loss

![image](https://user-images.githubusercontent.com/57930520/113245057-262a1d80-92f1-11eb-916f-790bf67b4ae9.png)

<br/>

### 2.3 64x64 Fashion MNIST Dataset

Left: loss graph per batch

Right: loss graph per epoch

Red Line: Generator loss

Blue Line: Discriminator loss

![image](https://user-images.githubusercontent.com/57930520/113245161-4eb21780-92f1-11eb-909b-b26f3b5837ae.png)



<br/>

# 3. Experiment Results

<br/>

<br/>

(추가 예정, 변경 진행중)

<br/>

<br/>

<br/>



# 4. Latent space visualization

## 4.1 Walking in the latent space

<br/>

<br/>

(논문에서 6.1 Walking in the latent space에 해당)



(추가 예정, 변경 진행중)

 <br/>







<br/>

<br/>



<br/>

<br/>

# 5. 느낀점

* (추가 예정, 변경 진행중)