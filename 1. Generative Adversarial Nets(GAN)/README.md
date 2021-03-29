# Generative Adversarial Nets (GAN) Codes

<br/>

<br/>

## 1. GAN Main architecture

<br/>

### 1.1 Discriminator

<br/>

```python
# Discriminator class
class Dis_model(nn.Module):
    def __init__(self, image_size, hidden_space):
        super(Dis_model, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(image_size, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, 1),
            nn.Sigmoid())
    
    def forward(self, input_x):
        x = self.features(input_x)
        return x
```
<br/>

<br/>

### 1.2 Generator

<br/>

<br/>

```python
# Generator class
class Gen_model(nn.Module):
    def __init__(self, latent_space, hidden_space, image_size):
        super(Gen_model, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(latent_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, hidden_space),
            nn.ReLU(),
            nn.Linear(hidden_space, image_size),
            nn.Tanh())
        
    def forward(self, input_x):
        x = self.features(input_x)
        return x
```
<br/>

<br/>

# 2. Loss Graphs

<br/>



Red line: Average Generator loss per epoch

Blue line: Average Discriminator loss per epoch

![image](https://user-images.githubusercontent.com/57930520/112741242-f8766900-8fbe-11eb-9a18-eea10a4820e0.png)





<br/>



# 3. Experiment Results

<br/>

<br/>

Gaussian noise (10, 100)을 고정해두고, 이를 각 epoch 별로 학습된 Generator를 가지고 이미지를 생성하였을 때 만들어지는 이미지로 만든 gif입니다.



![GAN_result_short_2](https://user-images.githubusercontent.com/57930520/112816337-40c78100-90bc-11eb-86ad-8408c2ddd0ac.gif)





