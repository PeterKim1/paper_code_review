# Adversarial Autoencoders (AAE) Codes

<br/>

<br/>

## 1. AAE Main architecture

<br/>

### 1.1 Encoder

<br/>

```python
# Encoder
class Q_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)

        self.lin1.weight.data.normal_(0, 0.01)
        self.lin2.weight.data.normal_(0, 0.01)
        self.lin3gauss.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #x = self.lin1(x)
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        #x = self.lin2(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss
```
<br/>

<br/>

### 1.2 Decoder

<br/>

<br/>

```python
# Encoder
class Q_net(nn.Module):  
    def __init__(self,X_dim,N,z_dim):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)

        self.lin1.weight.data.normal_(0, 0.01)
        self.lin2.weight.data.normal_(0, 0.01)
        self.lin3gauss.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #x = self.lin1(x)
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        #x = self.lin2(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss
```
<br/>

<br/>

### 1.3 Discriminator

<br/>

<br/>

```python
# Discriminator
class D_net_gauss(nn.Module):  
    def __init__(self,N,z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

        self.lin1.weight.data.normal_(0, 0.01)
        self.lin2.weight.data.normal_(0, 0.01)
        self.lin3.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return torch.sigmoid(self.lin3(x))    
```
<br/>

<br/>

# 2. Loss Graphs



## 2.1 latent dim = 2



<br/>



![img](https://blog.kakaocdn.net/dn/bics1O/btq0AG9TC6V/loo57kTdJRgZgoO7j79KXK/img.png)





## 2.2 latent dim = 15



<br/>

![img](https://blog.kakaocdn.net/dn/xKjOb/btq0EEXy1UB/PKoWj0oscDEMoy0BZkT6v0/img.png)





# 3. Experiment Results

<br/>

<br/>

## 3.1 latent dimension = 2

<br/>

<br/>

(16, 2) 차원의 Gaussian random vector를 가지고 학습이 완료된 Decoder를 통해 decoding한 결과

(임의의 latent vector로부터 이미지를 decoding 하는 능력을 테스트하는 실험)

<br/>

![img](https://blog.kakaocdn.net/dn/LEEvI/btq0MULQEgZ/q7BHkCIxxFhXAyg8dFEybk/img.png)

<br/>

<br/>

위: 테스트 데이터 셋 8개의 이미지

아래: 학습이 완료된 AAE를 가지고 위의 이미지를 복원해서 얻어지는 복원된 이미지

(본적 없는 테스트 데이터를 잘 복원해낼 수 있는지를 테스트하는 실험)

<br/>

![img](https://blog.kakaocdn.net/dn/bmDPSy/btq0DrRHFBz/igurBKtgILq15q48Dr8gk1/img.png)



## 3.2 latent dimension = 15

<br/>

<br/>

(16, 2) 차원의 Gaussian random vector를 가지고 학습이 완료된 Decoder를 통해 decoding한 결과

(임의의 latent vector로부터 이미지를 decoding 하는 능력을 테스트하는 실험)

<br/>

![img](https://blog.kakaocdn.net/dn/Qs6HW/btq0EDYHMfj/xBkHiaBdQ3ZFl00Nh4LG30/img.png)

<br/>

<br/>

위: 테스트 데이터 셋 8개의 이미지

아래: 학습이 완료된 AAE를 가지고 위의 이미지를 복원해서 얻어지는 복원된 이미지

(본적 없는 테스트 데이터를 잘 복원해낼 수 있는지를 테스트하는 실험)

<br/>

![img](https://blog.kakaocdn.net/dn/H75gX/btq0A25Wvn5/mawG7IKMv5ng5UsWRnnjY1/img.png)





# 4. Latent space visualization

## 4.1 From random variable To image made by trained AAE

<br/>

학습된 AAE를 가지고 (-10, -10)부터 (10, 10)까지 균등한 간격으로 숫자를 만들어내서 총 400가지 값을 학습이 완료된 AAE에 투입했을 때 만들어지는(decoding) 이미지들을 시각화

<br/>

![img](https://blog.kakaocdn.net/dn/vNke6/btq0MVqsCUE/yL7QPw3sY7ZR5Ra8cgAIX1/img.png)



# 5. 느낀점

* 논문에 나오는 깔끔한 결과를 만들기 위해서 엄청나게 고군분투 했지만, 결국 논문에 나온 성능을 낼 수 있는 AAE를 만들지 못한 점에 대해서 너무 아쉽다.
* 구글링을 통해서 Pytorch로 AAE를 올려둔 Github들을 전부 다 돌려보고 했지만, 막상 논문에 나온 수준의 성능이 나오는 코드는 없었다. (물론 지구상에 존재하지만 내가 못 찾은 것일지도 모른다.)
* 계속해서 learning rate 등의 hyperparameter를 조정해봤지만, 딱히 유의미한 성능 향상을 가져오는 결과는 없었다. (분명히 GAN에서는 hyperparameter가 중요하다길래, 이렇게 저렇게 바꿔보면서 수십번은 실험했다.)
* 개인적인 의견으로, 왜 이렇게 논문에 나오는 성능을 내기가 어려운지에 대해서 고민해보았는데 해당 모델의 한계점(?) 혹은 특징 이라고 한다면 Encoder가 Decoder와 쌍을 이루어 Autoencoder의 역할을 해야하면서 동시에 GAN framework에서는 Generator 역할을 해야 한다는 점이다.
* 그렇다면 Encoder는 Reconstruction error based update시에는 "너는 이미지를 잘 복원해야해!" 라고 강요받지만, Generator error based update시에는 "너는 Discriminator를 잘 속여야해!" 라고 강요받는다는 의미가 된다.
* 만약, Reconstruction error를 기준으로 했을 때, 이미지를 더 잘 복원하게 되는 Gradient direction과 Generator error를 기준으로 했을 때 Discriminator를 더 잘 속일 수 있게 되는 Gradient direction이 trade-off 관계에 있을 수도 있지 않을까? 라는 생각을 했다. (우리가 직관적으로 떠올리기 어려운 고차원이니까 이럴 수 있지 않을까..??)
* 예를 들자면, 아빠는 이거 하지마! 라고 하고, 엄마는 이거 해! 라고 하면 아이가 혼란스러워 하듯이, Encoder는 서로가 반대로 코칭하는 엄마와 아빠에 의해서 고통받는 아이일 수도 있지 않을까? 라는 생각을 했다.
* 물론 이건 언제까지나 초보 연구원의 추측이므로 틀릴 가능성이 더 높다.
* 하지만 일단 논문에 나온 대로의 manifold learning은 잘 못했지만 reconstruction을 기준으로 했을 때는 VAE보다 훨씬 더 좋은 성능이 나왔다. 따라서, Anomaly Detection과 같이 reconstruction을 이용하는 상황에서는 쓸 수 있지 않을까?