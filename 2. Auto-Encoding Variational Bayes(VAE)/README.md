# Auto-Encoding Variational Bayes(VAE) Codes





## 1. VAE Main architecture



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



## 2. Loss Graphs



<br/>

<br/>

### 2.1. Train



<br/>

첫 번째: Loss의 KL-Divergence term

두 번째: Loss의 Reconstruction Error term

세 번째: Total Loss

<br/>

<br/>

![image](https://user-images.githubusercontent.com/57930520/111641422-7c528780-8840-11eb-9cf8-4d3ee08af467.png)

<br/>

<br/>

### 2.2. Test



<br/>

첫 번째: Loss의 KL-Divergence term

두 번째: Loss의 Reconstruction Error term

세 번째: Total Loss

<br/>

<br/>

![image](https://user-images.githubusercontent.com/57930520/111648081-61831180-8846-11eb-98b3-2626ba8b3655.png)

<br/>

<br/>



## 3. Experiment Results



<br/>

<br/>

###     3.1. latent dimension = 2, training epochs = 50 epochs

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



<br/><br/>

### 3.2. latent dimension = 100, training epochs = 100 epochs

<br/><br/>

100 epoch동안 학습이 완료된 후, 위 : 실제 데이터 / 아래: 실제 데이터가 VAE model의 input으로 들어갔을 때 reconstruction 된 이미지

<br/><br/>

![image](https://user-images.githubusercontent.com/57930520/111641814-da7f6a80-8840-11eb-9a2b-18d7a899b0d9.png)



<br/><br/>

100 epoch동안 학습이 완료된 후, (64, 100)차원의 표준 정규 분포 데이터가 VAE model로 들어갔을 때 generated된 이미지 

<br/><br/>

![image](https://user-images.githubusercontent.com/57930520/111641896-eff49480-8840-11eb-8f88-c8959ce27462.png)

<br/><br/>

### 3.3. latent dimension = 200, training epoch = 100 epochs

<br/><br/>

100 epoch동안 학습이 완료된 후, 위 : 실제 데이터 / 아래: 실제 데이터가 VAE model의 input으로 들어갔을 때 reconstruction 된 이미지

<br/><br/>

![image](https://user-images.githubusercontent.com/57930520/111646551-17e5f700-8845-11eb-93f7-d7edd9640f4d.png)



<br/><br/>

100 epoch동안 학습이 완료된 후, (64, 200)차원의 표준 정규 분포 데이터가 VAE model로 들어갔을 때 generated된 이미지 

<br/><br/>

![image](https://user-images.githubusercontent.com/57930520/111646616-26341300-8845-11eb-9a24-8a023abed612.png)



<br/>

<br/>

## 4. Latent space visualization

### 4.1. 2-D visualization 10000 MNIST test image made by trained VAE

<br/>

latent space를 2차원으로 두고 학습된 VAE를 가지고 10000개의 test image를 encoding 했을 때 얻어진 결과를 시각화

<br/>

![image](https://user-images.githubusercontent.com/57930520/111756807-fc7afa80-88dd-11eb-842d-bf8e5005081e.png)

<br/>

### 4.2. From random variable To image made by trained VAE

<br/>

학습된 VAE를 가지고 (-3, -3)부터 (3, 3)까지 균등한 간격으로 숫자를 만들어내서 총 400가지 값을 학습이 완료된 VAE에 투입했을 때 만들어지는(decoding) 이미지들을 시각화

<br/>

![image](https://user-images.githubusercontent.com/57930520/111757649-e457ab00-88de-11eb-8992-9b068b4350f8.png)



## 5. 느낀점

- KL Divergence가 increase되는 현상을 해결하고 싶었는데 구글에서 마땅히 자료를 찾지 못해서 해결을 못함(혹시나 추후 해결방법을 알게 되면 수정할 예정이다. 살다보면 언젠가 알 수 있게 되지 않을까?)
- 일단 Total Loss가 줄어들었으니 수렴한 것 같기는 하다.
- Auto-Encoder 구조에서 핵심이라고 보여지는 latent dimension을 다양하게 바꿔보면서 실험해본 결과, 확실히 latent dimension이 너무 적으면 별로인 것 같다. 충분한 수의 latent dimension을 주는게 중요해 보인다.(Auto-encoder 구조는 고차원의 데이터를 저차원으로 데이터를 꾸겨넣는 것인데 더 낮은 차원으로 꾸겨넣을수록 데이터가 날라가기 때문에 성능의 차이를 가져오는 것인게 아닐까 싶다.)
- latent dimension이 2 일 때와 100일 때, 200일 때를 비교해보면 2보다는 100이 훨씬 좋지만, 100과 200인 경우를 비교해보면 크게 엄청 차이가 난다고 느껴지진 않는다. 즉, latent dimension은 어느 임계점 정도로 올리면 이보다 낮은 값일 때에 비해서 성능이 좋아지지만, 그 이후에는 크게 성능에 영향을 주지 않는 것 같다.
- VAE가 학습한 잠재 공간(latent space)을 시각화하는 코드를 찾으려고 엄청 애쓰다가 코드를 다 짜고 나서 몇 일 후에 발견해 추가했다. (사실 이 코드를 짜고 깃헙에 올리는 시점에 시각화 코드를 찾으려고 애쓰다가 결국 못 찾아서 그냥 올렸는데 추후에 발견한 것이다.)
- Test image 10000개에 대해서 시각화한 결과를 봤을 때는 다른 라벨을 가진 데이터들을 ''완전하게'' 분리되는 공간으로 학습하지 못하는 것을 확인하였다. 
- 즉, 물과 기름이 나눠지듯이 각 클래스에 해당하는 데이터들이 2-d 공간으로 mapping 되었을 때 서로 완전히 구별된다면 좋을탠데 그렇게 하지 못함을 확인하였다.
- (-3, -3)부터 (3, 3)까지 각 축을 기준으로 20개씩 균등하게 뽑아 학습된 VAE에 투입하였을 때 얻어지는 결과를 보니 latent space에 무언가 패턴이 있지 않을까 라는 생각을 할 수 있게 되었다.
- 예를 들면, 시각화 결과에서 좌하단은 기울기가 양수인 선형 직선과 같은 모양인데 이는 투입된 값이 (-3, -3)이며 우하단은 일직선으로 세워진 1의 모양을 하고 있다. 이는 투입된 값이 (-3, 3)인 위치이다. 
- 이 결과만 놓고 유추해보자면, (x, y) 중 y의 값이 커질수록 숫자가 일직선으로 세워진다고 유추해볼 수 있다. 물론 다른 row의 결과를 보면 숫자가 바뀌기도 하지만, 어쨋든 기울어진 모양에서 세워진 모양으로 바뀌는 패턴은 얼추 들어맞는 것 같다.









