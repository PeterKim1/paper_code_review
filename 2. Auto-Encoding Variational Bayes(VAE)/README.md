# Auto-Encoding Variational Bayes(VAE) Codes



### * VAE Main architecture

<br/>

```python
# VAE model
class VAE(nn.Module):
    def __init__(self, image_size, hidden_size_1, hidden_size_2, latent_size):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(image_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc31 = nn.Linear(hidden_size_2, latent_size)
        self.fc32 = nn.Linear(hidden_size_2, latent_size)

        self.fc4 = nn.Linear(latent_size, hidden_size_2)
        self.fc5 = nn.Linear(hidden_size_2, hidden_size_1)
        self.fc6 = nn.Linear(hidden_size_1, image_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

<br/><br/>



### * Loss Graphs

<br/>

<br/>

1. Train

<br/>

첫 번째: Loss의 KL-Divergence term

두 번째: Loss의 Reconstruction Error term

세 번째: Total Loss

<br/>

<br/>

![image](https://user-images.githubusercontent.com/57930520/111641422-7c528780-8840-11eb-9cf8-4d3ee08af467.png)

<br/>

<br/>

2. Test

<br/>

첫 번째: Loss의 KL-Divergence term

두 번째: Loss의 Reconstruction Error term

세 번째: Total Loss

<br/>

<br/>

![image-20210318232046435](C:\Users\KDH\AppData\Roaming\Typora\typora-user-images\image-20210318232046435.png)

<br/>

<br/>

### * Experiment Results

<br/>

<br/>

50 epoch동안 학습이 완료된 후, 위 : 실제 데이터 / 아래: 실제 데이터가 VAE model의 input으로 들어갔을 때 reconstruction 된 이미지

<br/><br/>

![image](https://user-images.githubusercontent.com/57930520/111147948-65a4fa00-85ce-11eb-8566-f259c77af27b.png)

<br/>

<br/>



50 epoch동안 학습이 완료된 후, (64, 2)차원의 표준 정규 분포 데이터가 VAE model로 들어갔을 때 generated된 이미지 

<br/><br/>



![image](https://user-images.githubusercontent.com/57930520/111148227-b9afde80-85ce-11eb-8591-17ffe6b528c2.png)