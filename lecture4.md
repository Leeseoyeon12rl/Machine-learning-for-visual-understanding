### 시각적 이해를 위한 머신러닝 4강: Neural Networks & Backpropagation ###

Nonlinear한 Decision Boundary를 어떻게 결정할까?

## [Issues with Linear Classifiers]

- Visually: Linear classifiers learn one template per class
- Geometrically: Linear classifiers can only draw linear decision boundaries
    
    ### → Feature Space
    
    - We may extract some features to represent the input
    - If the inputs are linearly separable in the feature space, a linear classifier may work well
    
    ### Image Features
    
    Instead of directly mapping input-output (pixel-class) relationship with a linear classfier, we may extract some features to represent the input (image).
    

Examples:

Color histogram, Histogram of oriented gradients(HoG), Bag of words with a pre-defined dictionary (codebook)

⇒ What if we can train end-to-end, such that feature extraction step also takes gradient from the classification loss?

- Feature extraction까지 parameterize. → 데이터로부터 배움.
    - 안 좋은 점이 있을까?
        
        데이터가 매우 많아야 함. 계산량 매우 큼.
        
        → 회의적 시각 존재… 따라서 domain knowledge 활용이 중요함.
        

## [Neural Networks]

### Perceptron

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/57b1b80e-31d5-4dfc-a63a-fa906c0af4ca/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/75ec63c9-fdc6-42e0-bb04-22a0fae470e6/image.png)

위와 같이 다층 퍼셉트론을 쌓음으로써 XOR 문제를 해결할 수 있게 됨.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/de896928-65d4-4112-a384-5c0474a822f8/image.png)

f(x) = Wx와 f(x) = W2(W1x)는 사실 수학적으로 같은 것을 의미한다. → still linear. . .

그럼 어떻게 다층 퍼셉트론은 non-linearlity를 표현할 수 있는가?

→ Activation function !! (활성화 함수)

- 활성화 함수는 뉴런의 입력 신호를 비선형적으로 변환하여 다음 층으로 전달하는 함수.
- 뉴런의 출력을 결정함.
- ex) Sigmoid, ReLU, Softmax, tanh

![loss function vs activation function](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/ffe1c5e3-080b-4043-b022-1bce055b97d4/image.png)

loss function vs activation function

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/d1208648-1f39-4188-b26a-7752b555e790/image.png)

- 매번 gradient descent를 계산할 순 없음 !
- 따라서 back propagation 고안

## [Back Propagation: Computing Gradients]

역전파는 신경망에서 손실 함수의 기울기를 계산해 가중치를 업데이트하는 과정.

1. Forward pass: 입력 데이터를 신경망을 통과시키면서 출력을 계산. 이때 가중치들이 이용됨.
2. Loss 계산: 모델 출력과 실제 출력의 차이를 바탕으로 손실 함수를 계산.
3. Backward pass(역전파): 손실 함수에 대한 기울기를 계산해 가중치를 업데이트. 이때 체인 룰을 통해 미분 수행.

![Local gradient: 순전파에서 입력과 가중치의 곱을 미분한 값, 역전파에서 해당 레이어에서의 미분 값.
Upstream gradient: 손실 함수의 미분 값.
Downstream gradient: 상류 기울기와 국소 기울기의 곱.](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/9d52cd25-c6ba-4cb5-8783-0c603a79c196/image.png)

Local gradient: 순전파에서 입력과 가중치의 곱을 미분한 값, 역전파에서 해당 레이어에서의 미분 값.
Upstream gradient: 손실 함수의 미분 값.
Downstream gradient: 상류 기울기와 국소 기울기의 곱.

Backpropagation 계산 과정

![Backpropagation = Upstream gradient * Local gradient](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/4eb611d3-b098-482b-9ed4-f142eeddd707/image.png)

Backpropagation = Upstream gradient * Local gradient

![(input이 2개이므로 각각 w, x라고 둠.) 만약 input이 3개 이상이면 본인을 제외한 나머지 local gradient를 모두 곱함.](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/a5f523d5-3492-497a-9c7c-c9566c703d00/image.png)

(input이 2개이므로 각각 w, x라고 둠.) 만약 input이 3개 이상이면 본인을 제외한 나머지 local gradient를 모두 곱함.

Back propagation 규칙

![→ PyTorch, Tensorflow는 Backpropagation을 계산해주는 도구임.](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/c565646a-ab3c-4d3c-b5e3-112c0f634516/image.png)

→ PyTorch, Tensorflow는 Backpropagation을 계산해주는 도구임.

## [Backpropagation with Vectors and Matrices]

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/614c9967-1b78-4e57-ac57-71fe5ade3295/image.png)

![- Input 20 dim, output 10 dim인 노드가 있다면:
Upstream gradient: 10 dim, Local gradient: 20*10 dim, Downstream gradient: 20 dim](https://prod-files-secure.s3.us-west-2.amazonaws.com/2ea24a63-8a16-48f6-a443-fa75df032b44/bd260648-72b8-4873-8f1c-0b6f7703c06c/image.png)

- Input 20 dim, output 10 dim인 노드가 있다면:
Upstream gradient: 10 dim, Local gradient: 20*10 dim, Downstream gradient: 20 dim

- 스칼라, 벡터, matrix, tensor든  모든 edge는 forward, backward 모두 같은 크기를 가져야 함.
- Upstream Gradient * Local Gradient = Downstream Gradient
