+++
title = "잡음 제거 오토인코더 실습: 텐서의 모양부터 학습과 평가까지"
description = "잡음 이미지가 8채널 특징맵을 거쳐 복원되는 과정을 따라간다. 전치 합성곱, 파라미터 공유, 이미지별 MSE, split·cat, 학습 루프와 검증을 그림과 코드로 연결하고, 노트북 재실행과 StopIteration의 원인까지 확인한다."
date = 2026-11-03
slug = "denoising-autoencoder-pytorch"

[extra]
katex = true
toc = true
styles = ["denoising-autoencoder-pytorch/article.css"]
+++

잡음이 섞인 손글씨 이미지 한 장을 받아서 깨끗한 이미지에 가까운 결과를 만들고 싶다. 모델은 어느 픽셀이 잡음인지 정답으로 전달받지 않는다. <strong>잡음 이미지를 입력받고, 모델이 만든 이미지와 깨끗한 원본의 차이를 줄이도록 학습한다.</strong> 인코더와 디코더가 함께 이 작업을 맡는다.

[앞의 오토인코더 기초 글](/autoencoder-denoising-basics/)에서는 입력이 잠재 표현과 복원값으로 바뀌는 원리를 살펴봤다. 이번에는 이미지 여러 장을 실제 PyTorch 코드로 처리하면서, 그 원리가 어떤 텐서와 함수 호출로 이어지는지 확인한다. 읽는 동안 다음 질문을 계속 연결해 보자. 지금 다루는 것은 <strong>이미지 데이터인가, 모델의 가중치인가? 어떤 코드가 값의 배열 모양을 바꾸고, 어떤 코드가 오차를 계산하며, 어떤 코드가 가중치를 바꾸는가?</strong>

[전체 Python 실습 파일](/denoising-autoencoder-pytorch/examples-ko.py)과 [Rust 배치 처리 예제](/denoising-autoencoder-pytorch/batching-ko.rs)를 함께 제공한다. Python 파일은 학습 자료의 `data`를 받아 실행한다. 원본 데이터와 `d2_tools.py`의 구현은 첨부 자료에 포함되어 있지 않으므로, 이 글에 실제 학습 MSE나 복원 성공 사례를 임의로 넣지는 않았다. 그림은 계산 구조를 설명하며, 실제 결과를 확인하는 코드는 뒤에서 실행한다. API 설명은 2026년 11월 3일 공식 문서를 기준으로 확인했다.

## 1. 입력과 정답이 어디로 가는지 먼저 연결한다

<strong>오토인코더(Autoencoder)</strong>는 입력에서 중간 표현을 계산하고, 그 표현으로 데이터를 재구성하는 신경망이다. <strong>인코더(Encoder)</strong>는 입력을 중간 특징으로 바꾸는 부분이고, <strong>디코더(Decoder)</strong>는 그 특징으로 원래 데이터 형태의 출력을 계산하는 부분이다. <strong>재구성(Reconstruction)</strong>은 출력값을 다시 계산한다는 뜻이며, 원본을 반드시 정확하게 되살린다는 뜻은 아니다.

<strong>잡음 제거 오토인코더(Denoising Autoencoder)</strong>에서는 입력과 정답을 구분한다. 입력에는 잡음을 넣지만, 출력과 비교하는 <strong>정답 데이터(Target)</strong>는 깨끗한 원본으로 유지한다. 여기에는 숫자가 3인지 7인지 알려주는 분류 레이블이 필요하지 않다. 필요한 것은 같은 원본에서 나온 잡음 입력과 깨끗한 정답의 대응이다.[^dae]

<strong>손실(Loss)</strong>은 출력과 정답 사이의 차이를 측정한 숫자다. <strong>기울기(Gradient)</strong>는 현재 가중치를 조금 바꾸면 손실이 어느 방향으로 얼마나 민감하게 변하는지 나타낸다. <strong>역전파(Backpropagation)</strong>가 이 기울기를 계산하고, <strong>옵티마이저(Optimizer)</strong>가 기울기를 이용해 학습 대상 숫자를 갱신한다. 이 이름들의 역할을 구분한 뒤 전체 그림을 보자.

<figure class="dae-figure">
<img src="/denoising-autoencoder-pytorch/overview-ko.svg" alt="깨끗한 원본에서 잡음 입력을 만든다. 잡음 입력은 인코더와 디코더를 지나 복원 이미지가 되고, 깨끗한 원본은 손실의 정답으로 전달된다. 역전파는 기울기를 계산하고 옵티마이저는 인코더와 디코더의 파라미터를 갱신한다." loading="lazy">
<figcaption>원본은 잡음 입력을 만드는 데 사용되고, 동시에 복원 결과를 비교하는 기준으로 남는다. 모델의 두 부분은 하나의 최종 손실로 함께 학습한다.</figcaption>
</figure>

이번 모델을 다섯 가지 기준으로 나누면 다음과 같다.

| 구분 | 이 실습에서의 대상 | 실행 중 어떻게 달라지는가? |
| --- | --- | --- |
| 입력 | 잡음이 있는 흑백 이미지 | 배치마다 다른 이미지와 잡음을 사용한다. |
| 뉴런·층 구조 | 합성곱 → ReLU → 전치 합성곱 → Sigmoid | 같은 구조를 반복 실행한다. 각 위치의 출력값이 가중합과 활성화 계산으로 만들어진다. |
| 중간값 | 인코더가 만든 8채널 특징맵 | 현재 입력과 현재 가중치로 매번 계산한다. |
| 가중치·편향 | 두 합성곱 층에 등록된 학습 대상 숫자 | 옵티마이저가 갱신한다. |
| 최종 출력 | 입력과 같은 크기의 복원 이미지 | 현재 모델이 입력으로부터 계산한 결과다. |

중간 특징맵이 모델의 가중치인 것은 아니다. 가중치는 모델에 저장되어 여러 이미지에 반복 사용되고, 특징맵은 특정 이미지를 넣어서 얻은 계산 결과다. 또한 인코더와 디코더 사이에 별도의 세 번째 ‘잡음 제거 모델’이 들어 있는 것도 아니다. <strong>잡음 입력을 깨끗한 원본에 가깝게 복원하도록 전체 모델을 학습하는 것</strong>이 잡음 제거 과정이다.

## 2. 딕셔너리에서 텐서를 꺼낸 뒤, 텐서의 위치를 선택한다

실습의 `data`는 이름으로 데이터를 찾아 꺼내는 <strong>딕셔너리(Dictionary)</strong>다. 이름에 해당하는 문자열을 <strong>키(Key)</strong>라고 한다. 여기서 꺼내는 이미지 데이터는 여러 숫자를 정해진 구조로 담은 <strong>텐서(Tensor)</strong>다.

| 코드 | 꺼내는 데이터 | 용도 |
| --- | --- | --- |
| `data["clean_train"]` | 깨끗한 훈련 이미지 | 잡음 입력과 훈련 정답을 만든다. |
| `data["clean"]` | 깨끗한 검증 이미지 | 훈련에 사용하지 않은 원본을 평가 기준으로 삼는다. |
| `data["noisy"]` | 검증 원본에 잡음을 넣은 이미지 | 같은 입력을 반복 사용해 모델의 변화를 비교한다. |

키 이름이 `clean`이라고 실제 데이터 분리가 보장되는 것은 아니다. 검증 원본을 훈련 원본과 분리하는 작업은 데이터를 준비할 때 해야 한다. 또한 `data["noisy"][i]`와 `data["clean"][i]`는 같은 원본 이미지에 대응해야 한다.

### 축 번호, 축의 크기, 인덱스는 서로 다르다

<strong>축(Axis, Dimension)</strong>은 이미지 번호·채널 번호·공간 위치처럼 데이터를 구분하는 기준이다. <strong>shape</strong>는 각 축의 크기를 적은 정보다. <strong>인덱스(Index)</strong>는 그 축에서 특정 원소를 선택하는 위치 번호이며, 0부터 시작한다.

이미지 전체의 개수를 `N`, 한 번의 호출에서 처리하는 <strong>배치(Batch)</strong>의 이미지 수를 `B`로 적겠다. 한 장은 높이 28, 너비 28인 흑백 이미지이며 채널은 1개다. 전체 검증 데이터는 `(N, 1, 28, 28)`, 현재 처리하는 배치는 `(B, 1, 28, 28)`이다.

| 축 번호 | 의미 | 전체 데이터에서의 크기 | 유효 인덱스 |
| ---: | --- | ---: | --- |
| `dim=0` | 이미지 번호 | `N` | `0`부터 `N-1`까지 |
| `dim=1` | 채널 번호 | `1` | `0` |
| `dim=2` | 높이 방향 위치 | `28` | `0`부터 `27`까지 |
| `dim=3` | 너비 방향 위치 | `28` | `0`부터 `27`까지 |

배치 shape가 `(64, 1, 28, 28)`이라면 이미지가 64장 있다는 뜻이다. 마지막 이미지 인덱스는 63이다. 흑백이라는 이유로 채널 축이 사라지는 것도 아니다.

### `data["clean"][0, 0, 5, 10]`은 왜 5차원이 아닐까?

첫 대괄호의 `["clean"]`은 <strong>딕셔너리에서 텐서를 꺼내는 작업</strong>이다. 두 번째 대괄호의 `[0, 0, 5, 10]`은 <strong>그 텐서의 네 축에서 위치를 선택하는 작업</strong>이다. 문자열 키는 텐서의 축이 아니다.

| 선택 | 남는 대상 | shape |
| --- | --- | --- |
| `data["clean"]` | 검증 이미지 전체 | `(N, 1, 28, 28)` |
| `data["clean"][0]` | 0번 이미지 | `(1, 28, 28)` |
| `data["clean"][0, 0]` | 0번 이미지의 흑백 채널 | `(28, 28)` |
| `data["clean"][0, 0, 5]` | 높이 인덱스 5에 놓인 너비 방향 값들 | `(28,)` |
| `data["clean"][0, 0, 5, 10]` | 해당 위치의 픽셀 하나 | `()` |

숫자 하나만 담긴 `shape=()` 텐서를 <strong>스칼라 텐서(Scalar Tensor)</strong>라고 한다. `.item()`을 호출하면 그 값을 Python 숫자로 가져온다. `(1,)`은 크기 1인 축이 있는 구조이고, `()`는 축이 없는 구조다.

```python
import torch
from torch import nn  # torch.nn의 신경망 도구를 nn이라는 이름으로 쓴다.

# data는 원본 학습 자료의 데이터 준비 셀에서 만들어져 있어야 한다.
for name in ("clean_train", "clean", "noisy"):
    images = data[name]  # 문자열 키로 이미지 텐서를 꺼낸다.
    print(name, type(images), tuple(images.shape), images.dtype, images.device)

# 끝값 5000은 제외한다. 이미지 인덱스 0~4999를 선택한다.
clean_train = data["clean_train"][:5000].to(device="cpu", dtype=torch.float32)
clean_val = data["clean"].to(device="cpu", dtype=torch.float32)
noisy_val = data["noisy"].to(device="cpu", dtype=torch.float32)

print(tuple(clean_val[0, 0, 5, 10].shape))  # ()
print(clean_val[0, 0, 5, 10].item())       # 해당 픽셀의 Python 숫자.
```

`dtype`은 숫자 자료형, `device`는 계산 장치를 뜻한다. 이 실습은 CPU의 `float32`를 사용한다. `.to(dtype=torch.float32)`는 자료형을 바꿀 뿐, 0~255 픽셀을 자동으로 0~1로 바꾸지 않는다. 입력이 실제로 0~255라면 그 데이터의 정의를 확인한 후 255로 나누는 등의 정규화가 필요하다. 이미 0~1인 데이터를 다시 255로 나누면 값의 범위가 잘못된다.

전체 실습 파일은 이미지 shape, 빈 데이터, 유한하지 않은 값, 0~1 범위를 검사한다. <strong>shape가 같다는 검사는 이미지의 정답 대응까지 확인해 주지 않는다.</strong> 같은 원본이 같은 인덱스에 놓였는지는 생성 과정이나 ID로 확인해야 한다.

## 3. 모델을 만드는 코드와 이미지를 처리하는 코드를 나눈다

`torch.nn`은 신경망 층과 손실 계산 도구를 제공한다. 그 안의 `nn.Module`은 모델의 하위 층과 파라미터를 관리하는 기본 클래스다. `__init__()`은 모델 객체를 만들 때 실행되고, `forward()`는 입력으로부터 출력을 계산하는 순서를 정의한다.[^module]

이번에는 인코더의 출력 채널 수를 `feature_channels`라는 생성자 인자로 받는다. 기본값은 원본 실습과 같은 8이다. 이 값을 4로 바꾸면 뒤에서 같은 구조의 4채널 모델을 만들 수 있다.

```python
class LabDenoiser(nn.Module):
    def __init__(self, feature_channels=8):
        super().__init__()  # 하위 층과 파라미터를 관리하는 기능을 초기화한다.

        self.encoder = nn.Sequential(
            # in_channels=1, out_channels=feature_channels, kernel_size=3.
            # stride=2: 위치 간격 2, padding=1: 입력 가장자리의 0 패딩 1.
            nn.Conv2d(1, feature_channels, 3, stride=2, padding=1),
            nn.ReLU(),  # 음수는 0으로, 양수는 그대로. shape는 유지한다.
        )
        self.decoder = nn.Sequential(
            # 입력 특징 채널들을 이용해 출력 채널 하나를 계산한다.
            # 4×4 커널, stride=2, padding=1의 전치 합성곱이다.
            nn.ConvTranspose2d(feature_channels, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),  # 출력값을 0~1 범위로 보낸다.
        )

    def forward(self, x):
        features = self.encoder(x)    # 입력으로부터 중간 특징을 계산한다.
        return self.decoder(features) # 중간 특징으로 복원 이미지를 계산한다.

model = LabDenoiser(feature_channels=8).cpu()  # 층과 파라미터를 준비한다.
with torch.no_grad():  # 지금은 출력 shape만 보므로 자동 미분 기록을 끈다.
    reconstructed = model(noisy_val[:2])     # 준비된 모델에 실제 이미지를 넣는다.
print(tuple(reconstructed.shape))            # 검증 이미지가 2장 이상이면 (2,1,28,28).
```

`nn.Conv2d(...)`의 인자에는 채널 수와 커널 설정이 들어간다. 이때 이미지를 전달하는 것은 아니다. 반면 `model(noisy_val[:2])`에는 실제 이미지 텐서를 전달한다. `[:2]`는 끝 인덱스 2를 제외하므로 0번과 1번 이미지를 선택한다.

일반적인 실행에는 `model.forward(x)`를 직접 부르기보다 `model(x)`를 사용한다. `nn.Module`의 호출 절차가 `forward()`와 등록된 부가 기능을 함께 처리하기 때문이다.[^module]

### `nn.Sequential` 안에는 무엇이 들어갈까?

`nn.Sequential`은 층 객체들을 순서대로 등록하고, 호출되면 앞 층의 결과를 다음 층의 입력으로 전달한다. 생성할 때 들어가는 것은 `Conv2d` 객체와 `ReLU` 객체이고, 실제 이미지 `x`는 나중에 `encoder(x)`로 들어간다.[^sequential]

```python
# 아래 두 줄은 model.encoder(x)의 계산 순서를 펼쳐 쓴 것이다.
conv_result = model.encoder[0](noisy_val[:2])  # 등록된 첫 층: Conv2d.
features = model.encoder[1](conv_result)      # 등록된 둘째 층: ReLU.
```

단순히 층을 차례대로 연결할 때는 `Sequential`이 편하다. 입력을 보관했다가 나중에 더하는 잔차 연결처럼 경로가 갈라지면 `forward()`에서 그 흐름을 직접 작성한다. 변수 `features`는 계산 결과의 이름이며, 새로운 층이나 새로운 가중치를 선언한 것이 아니다.

## 4. 이미지 수, 특징맵 수, 가중치의 모양을 따로 읽는다

<strong>특징맵(Feature Map)</strong>은 합성곱의 결과를 공간 위치에 맞춰 저장한 데이터다. 일반적인 `Conv2d`의 <strong>필터(Filter)</strong> 하나는 모든 입력 채널을 이용해 출력 채널 하나를 만든다. 이 모델은 인코더 필터가 8개이므로 이미지마다 특징맵 8개를 계산한다.

<figure class="dae-figure">
<img src="/denoising-autoencoder-pytorch/shapes-ko.svg" alt="배치 B장의 입력은 (B,1,28,28), 인코더 중간 특징은 (B,8,14,14), 복원 출력은 (B,1,28,28)이다. 인코더 가중치는 (8,1,3,3), 디코더 가중치는 (8,1,4,4)이며 각각의 편향은 (8,), (1,)이다." loading="lazy">
<figcaption>모든 단계에서 이미지 수 B는 유지된다. 특징맵은 이미지마다 계산하고, 가중치는 이미지와 공간 위치에 걸쳐 공유한다.</figcaption>
</figure>

| 단계 | 데이터 shape | 값이 뜻하는 것 |
| --- | --- | --- |
| 잡음 입력 | `(B, 1, 28, 28)` | 이미지 B장, 흑백 채널 하나 |
| `Conv2d` | `(B, 8, 14, 14)` | 필터 8개가 계산한 가중합 |
| `ReLU` | `(B, 8, 14, 14)` | 음수 가중합을 0으로 바꾼 중간 특징 |
| `ConvTranspose2d` | `(B, 1, 28, 28)` | 복원 픽셀을 만들기 위한 가중합 |
| `Sigmoid` | `(B, 1, 28, 28)` | 0~1 범위의 복원 픽셀값 |

ReLU와 Sigmoid 같은 <strong>활성화 함수(Activation Function)</strong>는 가중합에 적용하는 함수다. 이 모델의 ReLU와 Sigmoid에는 학습 파라미터가 없지만, 역전파는 이 함수들을 거쳐 전달되는 변화까지 계산해야 한다.

### 공간 크기가 줄면 항상 데이터도 압축될까?

지금 세려는 것은 <strong>이미지 한 장을 표현하는 텐서 원소의 개수</strong>다. 채널 수와 각 채널의 높이·너비를 곱하면 된다. 입력은 채널 1개에 `28×28`이고, 중간 특징은 채널 8개에 `14×14`다.

| 대상 | 이미지 한 장당 원소 수 |
| --- | ---: |
| 입력 | `1 × 28 × 28 = 784` |
| 중간 특징 | `8 × 14 × 14 = 1,568` |

<strong>공간 해상도는 줄었지만 전체 원소 수는 늘었다.</strong> 따라서 이 모델을 ‘반드시 더 적은 숫자로 압축하는 구조’라고 설명하면 맞지 않는다. 채널을 4개로 바꾼 경우에는 `4×14×14=784`개로 입력과 원소 수가 같다. 이 경우에도 계산된 값이 원본과 같다는 뜻은 아니다.

## 5. 전치 합성곱은 겹치는 출력 위치에 계산 결과를 더한다

<strong>전치 합성곱(Transposed Convolution)</strong>은 학습 가능한 커널로 출력 위치의 값들을 계산하는 연산이다. 설정에 따라 출력의 높이와 너비를 늘릴 수 있다. 이 연산은 인코더가 했던 계산을 자동으로 취소하는 역함수가 아니다. 인코더와 디코더의 가중치도 서로 별도로 학습한다.[^transpose]

현재 디코더는 `ConvTranspose2d(8, 1, 4, stride=2, padding=1)`이다. 입력은 8채널의 `14×14` 특징맵이고, 출력은 1채널의 `28×28` 데이터다. 먼저 채널 하나의 입력 위치 하나에서 시작해 보자.

1. 그 위치의 입력값에 해당 채널의 `4×4` 커널 가중치들을 각각 곱한다.
2. 곱셈으로 얻은 값 16개를 출력의 대응하는 `4×4` 위치에 더한다.
3. 이웃한 입력 위치도 같은 커널을 사용한다. `stride=2`이므로 그 커널의 출력 배치 위치는 두 칸 옮겨진다.
4. 두 계산이 같은 출력 위치에 도달하면 그 값을 합산한다.
5. 나머지 입력 채널들도 각자의 커널로 계산하고, 같은 출력 위치에 더한다. 마지막에는 출력 채널의 편향을 더한다.

여기서 ‘더할 값’은 입력값과 커널 가중치를 곱한 결과다. 따로 존재하는 특별한 종류의 숫자가 아니다. 겹친 위치에는 여러 곱셈 결과가 들어오므로 그 결과들을 더한다.

### 왜 커널 길이 4를 한 번만 더할까?

높이 또는 너비 중 한 방향만 살펴보자. 입력 길이는 14이므로 입력 인덱스는 `0~13`이다. 경계를 조정하기 전, 각 입력 위치에 대응하는 커널 시작 위치를 쓰면 다음과 같다.

| 입력 인덱스 | 커널 시작 위치 | 커널이 차지하는 출력 위치 |
| ---: | ---: | --- |
| 0 | 0 | 0, 1, 2, 3 |
| 1 | 2 | 2, 3, 4, 5 |
| 2 | 4 | 4, 5, 6, 7 |
| 12 | 24 | 24, 25, 26, 27 |
| 13 | 26 | 26, 27, 28, 29 |

<figure class="dae-figure">
<img src="/denoising-autoencoder-pytorch/transpose-overlap-ko.svg" alt="길이 4인 커널을 2칸 간격으로 배치하면 첫 커널은 위치 0~3, 다음 커널은 2~5, 마지막 커널은 26~29를 차지한다. 전체 길이 30에서 양쪽 경계 하나씩을 제외하면 출력 길이는 28이다." loading="lazy">
<figcaption>첫 커널은 4칸을 차지한다. 이후 커널은 앞의 커널과 겹치므로 전체 길이를 2칸씩 늘린다. 이 그림은 값이 아니라 위치와 겹침을 보여준다.</figcaption>
</figure>

첫 커널이 차지한 길이는 4다. 남은 입력 위치는 13개이고, 각 위치가 추가될 때 전체 길이는 2씩 늘어난다. 그러므로 경계를 제외하기 전 길이는 `4 + 13×2 = 30`이다. 마지막 커널을 기준으로 세어도 ‘시작 위치 26 + 길이 4’이므로 똑같이 30이다. 첫 커널의 길이를 빼먹은 것이 아니다.

현재 `padding=1` 설정에서는 이 범위의 양쪽에서 한 칸씩 제외한 연결로 이해하면 된다. `30-1-1=28`이다. 이는 연결과 크기를 설명하는 동등한 방법이지, 내부 구현이 반드시 큰 텐서를 만든 뒤 잘라낸다는 뜻은 아니다.

### 위치를 확인한 다음 크기 공식으로 정리한다

`L`은 입력의 한 방향 길이, `k`는 그 방향의 커널 길이, `s`는 stride, `p`는 padding이다. 현재처럼 `dilation=1`, 전치 합성곱의 `output_padding=0`일 때 다음 관계를 쓴다. `floor`는 결과를 넘지 않는 가장 큰 정수로 내리는 연산이다.[^conv][^transpose]

| 연산 | 출력 길이 | 현재 모델에 적용 |
| --- | --- | --- |
| 일반 합성곱 | `floor((L + 2p - k) / s) + 1` | 입력 28, 커널 3, stride 2, padding 1 → 14 |
| 전치 합성곱 | `(L - 1) × s - 2p + k` | 입력 14, 커널 4, stride 2, padding 1 → 28 |

`stride=2`만 보고 ‘항상 두 배가 된다’고 판단하면 안 된다. 커널 크기와 padding도 출력 길이에 영향을 준다. 이 모델도 28×28 입력을 기준으로 설계했으며, 임의의 다른 입력 크기에서 출력이 입력과 같아진다고 보장하지 않는다.

<details>
<summary>dilation과 output_padding이 있는 일반식은 나중에 참고한다</summary>

<strong>팽창률(Dilation)</strong>은 커널 원소 사이의 간격을 뜻하며 기본값은 1이다. 이를 `d`, 전치 합성곱의 `output_padding`을 `o`라고 하면 전치 합성곱 출력 길이는 `(L-1)×s - 2p + d×(k-1) + o + 1`이다.

`output_padding`은 가능한 출력 크기 중 어느 크기를 사용할지 조정하는 설정이다. 출력 뒤에 0인 픽셀을 단순히 덧붙이는 연산으로 이해하면 틀린다. 이 실습은 `d=1`, `o=0`이므로 앞의 식만으로 충분하다. 실무에서는 목표 출력 shape와 공식 문서의 조건을 함께 확인하면 된다.

</details>

### 확대 방법은 전치 합성곱 하나뿐일까?

<strong>업샘플링(Upsampling)</strong>은 높이·너비의 크기를 늘리는 작업이다. <strong>보간(Interpolation)</strong>은 기존 값으로 새로운 위치의 값을 정해진 규칙에 따라 계산하는 방법이다. `nearest` 보간은 가까운 기존 값을 선택한다.

```python
# 원래 모델에 적용하는 코드는 아니다. 다른 디코더 구조를 비교하는 예시다.
alternative_decoder = nn.Sequential(
    # 학습 가중치 없이 높이·너비를 각각 2배로 늘린다.
    nn.Upsample(scale_factor=2, mode="nearest"),  # (B,8,14,14) → (B,8,28,28)
    nn.Conv2d(8, 1, 3, padding=1),                # (B,8,28,28) → (B,1,28,28)
    nn.Sigmoid(),
)
```

전치 합성곱은 학습한 커널로 확대된 위치의 값을 계산한다. 위 구조는 정해진 규칙으로 먼저 확대하고, 그 뒤의 합성곱에서 학습한 가중치를 사용한다. 출력 shape가 같아도 계산법과 출력값은 다르다.[^upsample]

전치 합성곱으로 생성한 이미지에는 격자무늬 같은 <strong>체커보드 아티팩트(Checkerboard Artifact)</strong>가 생길 수 있다. 보간과 합성곱을 분리하는 방식은 이를 줄이기 위한 대안 중 하나다. 커널 4가 stride 2로 나누어떨어지는 현재 설정은 불균일한 겹침이라는 한 원인을 줄이지만, 좋은 복원 품질이나 아티팩트 부재를 보장하지는 않는다.[^checkerboard]

## 6. 파라미터는 209개지만 계산을 209번만 하는 것은 아니다

<strong>파라미터(Parameter)</strong>는 모델에 저장된 학습 대상 숫자다. 이 모델에서는 입력에 곱하는 <strong>가중치(Weight)</strong>와 가중합에 더하는 <strong>편향(Bias)</strong>이 해당한다. 같은 커널을 여러 위치와 여러 이미지에서 반복 사용하는 것을 <strong>가중치 공유(Weight Sharing)</strong>라고 한다.

먼저 이미지 수와 관계없이 필요한 연결부터 세자. 현재 `groups=1`에서는 입력 채널 하나와 출력 채널 하나의 연결마다 공간 커널이 있다. 디코더의 입력 채널이 8개이고 출력 채널이 1개라면 `4×4` 커널은 8개다. 출력 채널이 2개라면 각 출력 채널을 위한 별도 커널 8개씩, 총 16개가 필요하다. 편향은 출력 채널마다 하나씩 둔다.[^conv][^transpose]

| 층 | 가중치 shape | 가중치 수 | 편향 shape·개수 | 합계 |
| --- | --- | ---: | --- | ---: |
| 인코더 `Conv2d` | `(8, 1, 3, 3)` | `8×1×3×3=72` | `(8,)`, 8개 | 80 |
| 디코더 `ConvTranspose2d` | `(8, 1, 4, 4)` | `8×1×4×4=128` | `(1,)`, 1개 | 129 |
| `ReLU`, `Sigmoid` | 없음 | 0 | 없음 | 0 |
| 전체 모델 | 파라미터 텐서 4개 | 200 | 9개 | <strong>209</strong> |

두 가중치 shape가 모두 `(8, 1, …, …)`로 시작해서 같은 축 순서처럼 보일 수 있다. 하지만 이 모델에서는 우연히 숫자가 같을 뿐이다.

| 층 | `groups=1`일 때 weight의 축 순서 |
| --- | --- |
| `Conv2d` | `(출력 채널, 입력 채널, 커널 높이, 커널 너비)` |
| `ConvTranspose2d` | `(입력 채널, 출력 채널, 커널 높이, 커널 너비)` |

이미지 B장이 들어와도 각 이미지에 독립적인 가중치 209개를 새로 만들지 않는다. 같은 209개를 더 많이 사용한다. 따라서 <strong>파라미터 원소 수, 파라미터 텐서 수, 실제 곱셈·덧셈 횟수, 가중치 갱신 횟수</strong>를 분리해서 세야 한다. 파라미터 수만으로 속도나 전체 메모리 사용량을 알 수도 없다. 중간 특징맵과 역전파 계산에도 자원이 필요하다.

### 코드로 세면 `part`는 어디서 생길까?

원문의 짧은 코드는 딕셔너리 컴프리헨션이다. <strong>컴프리헨션(Comprehension)</strong>은 반복하면서 컬렉션을 만드는 문법이다. 먼저 같은 결과를 일반 반복문으로 만들어 보자.

```python
counts = {}

for name, part in [("encoder", model.encoder), ("decoder", model.decoder)]:
    # 첫 반복: name="encoder", part=model.encoder.
    # 둘째 반복: name="decoder", part=model.decoder.
    total = 0
    for parameter in part.parameters():  # 등록된 파라미터 텐서를 차례로 받는다.
        total += parameter.numel()       # numel(): 텐서 안의 숫자 총개수.
    counts[name] = total

print(counts)                 # {'encoder': 80, 'decoder': 129}
print(sum(counts.values()))   # 값 80과 129를 더하므로 209.

# 같은 집계를 컴프리헨션으로 작성한 형태다.
counts = {
    name: sum(p.numel() for p in part.parameters())
    for name, part in [("encoder", model.encoder), ("decoder", model.decoder)]
}
```

`for name, part in ...`가 각 튜플을 꺼내 두 변수에 값을 대입한다. `part`를 미리 선언할 필요가 없다. 안쪽의 `p`도 `for p in part.parameters()`에서 값을 받는다. Python은 `part`와 `Part`를 서로 다른 이름으로 취급한다. 컴프리헨션의 반복 변수는 바깥에 새로 남지 않는다는 점도 일반 `for`문과 다르다.[^comprehension]

`part`라는 단어 자체에 ‘모델 일부’라는 기능이 있는 것은 아니다. 나중에 `for part in noisy.split(256)`이라고 쓰면, 그때의 `part`는 이미지 묶음 텐서다. <strong>변수의 역할은 그 변수에 어떤 값을 대입했는지로 결정된다.</strong>

`parameters()`에는 `requires_grad=False`로 동결한 파라미터도 포함된다. 학습 대상으로 설정된 것만 세려면 `sum(p.numel() for p in model.parameters() if p.requires_grad)`처럼 조건을 붙인다. 현재 모델은 네 파라미터 텐서 모두 학습 대상이다.[^module]

## 7. 이미지는 같은 위치끼리 뺀다. 펼친 다음에만 뺄 수 있는 것은 아니다

복원 이미지를 원본과 비교한다는 것은 대응하는 픽셀값을 비교한다는 뜻이다. 이미지 번호 `i`, 채널 번호 `c`, 높이 인덱스 `h`, 너비 인덱스 `w`가 같아야 한다.

`reconstructed[i, c, h, w] - clean[i, c, h, w]`

예를 들어 `[0, 0, 5, 10]`은 앞에서 확인한 0번 이미지의 한 픽셀이다. 복원의 그 위치와 원본의 그 위치를 뺀다. 다른 이미지의 픽셀과 섞어서 비교하지 않는다.

<strong>원소별 연산(Element-wise Operation)</strong>은 같은 위치의 값끼리 각각 계산하는 방식이다. shape가 같은 두 텐서는 `(B, 1, 28, 28)` 구조 그대로 뺄 수 있다. 각 차이를 제곱해도 shape는 유지된다.

```python
def require_same_shape(prediction, target):
    # 이번 복원 문제에서는 두 텐서의 shape가 정확히 같아야 한다.
    if prediction.shape != target.shape:
        raise ValueError(
            f"예측 shape={tuple(prediction.shape)}, 정답 shape={tuple(target.shape)}"
        )

with torch.no_grad():
    example_clean = clean_val[:2]
    example_recon = model(noisy_val[:2])
    require_same_shape(example_recon, example_clean)

    difference = example_recon - example_clean  # (B,1,28,28): 픽셀별 차이.
    squared_error = difference.square()        # (B,1,28,28): 차이를 각각 제곱.
    mse_all = squared_error.mean()             # (): 모든 원소의 평균.
```

<strong>평균 제곱 오차(MSE, Mean Squared Error)</strong>는 대응하는 값의 차이를 제곱하고 그 결과들을 평균낸 값이다. `nn.MSELoss()`의 기본 설정은 `reduction="mean"`으로, 전달한 텐서의 모든 원소를 평균낸다. 이미지 수로만 나누는 것이 아니라 채널·높이·너비까지 포함한 전체 원소 수로 나눈다.[^mse]

| 설정 | 무엇을 반환하는가? | 이 실습에서의 shape |
| --- | --- | --- |
| `reduction="none"` | 픽셀별 제곱 오차 | `(B, 1, 28, 28)` |
| `reduction="sum"` | 모든 제곱 오차의 합 | `()` |
| `reduction="mean"` | 모든 제곱 오차의 평균 | `()` |

`nn.MSELoss()(prediction, target)`에도 두 단계가 있다. 첫 괄호 `nn.MSELoss()`는 손실 계산 객체를 만들고, 둘째 괄호는 그 객체에 예측과 정답을 전달한다. `loss_fn = nn.MSELoss()` 다음에 `loss_fn(prediction, target)`이라고 쓰는 것과 같다.

### 이미지마다 오차 하나를 남기려면

<strong>flatten</strong>은 지정한 축들을 하나로 합치는 연산이다. `flatten(1)`의 1은 축의 크기가 아니라 <strong>합치기 시작할 축 번호</strong>다. 0번 이미지 축을 남기고 채널·높이·너비를 합친다. 한 이미지에는 `1×28×28=784`개 픽셀이 있으므로 결과는 `(B, 784)`다.[^flatten]

그다음 `mean(1)`은 각 이미지 안의 784개 값을 평균낸다. 최종 결과는 이미지마다 오차 하나가 있는 `(B,)`다. `flatten`은 값을 평균내지 않고, `mean`은 평균을 계산한다. 두 연산의 역할이 다르다.

<figure class="dae-figure">
<img src="/denoising-autoencoder-pytorch/mse-reduction-ko.svg" alt="복원과 정답을 같은 위치끼리 빼고 제곱하면 (B,1,28,28)의 픽셀 오차가 생긴다. 전체 mean은 스칼라를 반환한다. flatten(1)은 (B,784), mean(1)은 (B,)의 이미지별 MSE를 반환한다." loading="lazy">
<figcaption>이미지별 오차를 보려면 이미지 번호 축을 남겨야 한다. 전체 평균과 이미지별 평균은 같은 픽셀 오차를 서로 다른 범위로 모은 결과다.</figcaption>
</figure>

```python
def image_mse(prediction, target):
    require_same_shape(prediction, target)
    pixel_errors = (prediction - target).square()  # (B,1,28,28)
    return pixel_errors.flatten(1).mean(1)        # (B,784) → (B,)

# flatten 없이 채널·높이·너비만 평균내도 이미지별 MSE가 된다.
per_image_a = squared_error.flatten(1).mean(1)
per_image_b = squared_error.mean(dim=(1, 2, 3))

# allclose는 부동소수점 반올림에 따른 작은 차이를 허용하여 비교한다.
assert torch.allclose(per_image_a, per_image_b)
assert torch.allclose(per_image_a.mean(), mse_all)
```

여기서 `assert`는 계산 관계를 확인하는 학습용 검사다. Python 최적화 실행에서는 제거될 수 있으므로, 반드시 유지해야 하는 입력 검사는 앞의 `require_same_shape()`처럼 `if`와 예외로 작성한다. 이 동작은 [Python의 assert 문서](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement)에서도 확인할 수 있다.

모든 이미지의 크기가 같으므로 ‘이미지별 MSE를 구한 뒤 이미지끼리 평균’과 ‘모든 픽셀을 한 번에 평균’은 같다. 각 이미지가 평균에 포함하는 픽셀 수가 같기 때문이다. 이 관계와, 크기가 다른 <strong>배치들의 평균</strong>을 합치는 문제는 뒤에서 구분한다.

`flatten()`처럼 시작 축을 생략하면 이미지 축까지 합쳐 `(B×784,)`가 된다. 숫자가 없어지는 것은 아니지만, 이미지별 구분을 나타내던 축은 사라진다. 이미지별 평가를 하려면 `flatten(1)`처럼 남길 축을 의식해야 한다.

### 오류가 안 났는데도 잘못 비교할 수 있다

<strong>브로드캐스팅(Broadcasting)</strong>은 크기가 다른 텐서도 정해진 규칙에 따라 맞춰 계산하는 기능이다. 뒤쪽 축부터 맞추며, 없는 앞쪽 축은 크기 1로 취급한다.[^broadcast]

| 예측 shape | 정답 shape | 뺄셈 결과 | 해석 |
| --- | --- | --- | --- |
| `(B,1,28,28)` | `(B,1,28,28)` | `(B,1,28,28)` | 의도한 이미지 쌍의 픽셀 비교 |
| `(B,1,28,28)` | `(1,1,28,28)` | `(B,1,28,28)` | 정답 한 장이 모든 예측에 반복 사용됨 |
| `(B,1,28,28)` | `(B,28,28)` | `(B,B,28,28)` | 서로 다른 이미지끼리도 비교되는 구조가 됨 |

마지막 경우의 정답은 앞에 크기 1인 축이 있다고 해석되어 `(1,B,28,28)`처럼 맞춰진다. 예측의 첫 B축과 정답의 둘째 B축이 각각 유지되므로 `(B,B,28,28)`이 된다. <strong>뺄셈이 실행되었다는 사실과 올바른 이미지 대응으로 계산했다는 사실은 다르다.</strong> 그래서 손실 계산 직전에 shape를 확인한다.

## 8. 한 배치 학습에서 실제로 가중치를 바꾸는 줄을 찾는다

<strong>순전파(Forward Pass)</strong>는 입력으로부터 출력을 계산하는 과정이다. 순전파로 복원값을 만든 다음 손실을 계산하고, 역전파로 기울기를 계산한 뒤 옵티마이저로 갱신한다. 기울기는 최적의 이동 거리를 알려주는 값이 아니며, 갱신 방식은 옵티마이저와 <strong>학습률(Learning Rate)</strong> 설정에 따라 정해진다.

```python
def train_step(net, optimizer, noisy_images, clean_images):
    net.train()  # 학습 모드를 설정한다. 이 호출만으로 가중치가 바뀌지는 않는다.
    optimizer.zero_grad(set_to_none=True)  # 이전 .grad를 None으로 초기화한다.

    reconstructed = net(noisy_images)      # 순전파: 잡음 이미지 → 복원 이미지.
    require_same_shape(reconstructed, clean_images)
    error = nn.MSELoss()(reconstructed, clean_images)  # 모든 픽셀의 평균 오차.

    error.backward()  # 기울기를 계산하여 파라미터의 .grad에 누적한다.
    optimizer.step()  # 그 기울기와 옵티마이저 상태로 가중치·편향을 갱신한다.

    return error.item()  # 이미 계산한 손실을 기록용 Python 숫자로 가져온다.
```

| 코드 | 바꾸거나 만드는 대상 | 가중치 자체를 갱신하는가? |
| --- | --- | --- |
| `net.train()` | 모델의 동작 모드 | 아니오 |
| `zero_grad(set_to_none=True)` | 이전에 저장한 기울기 | 아니오 |
| `net(noisy_images)` | 복원 이미지와 역전파에 필요한 계산 기록 | 아니오 |
| `MSELoss` | 출력과 정답을 비교한 손실 | 아니오 |
| `error.backward()` | 파라미터의 기울기 | 아니오 |
| `optimizer.step()` | 모델의 가중치와 편향, 옵티마이저 내부 상태 | <strong>예</strong> |

PyTorch는 기울기를 누적하므로, 현재처럼 배치마다 새 기울기를 쓰려면 이전 기울기를 비운다. `zero_grad()`는 가중치를 0으로 만드는 함수가 아니다. `set_to_none=True`는 기울기 저장값을 `None`으로 초기화하는 설정이다.[^zero][^autograd]

현재 모델의 파라미터 원소는 209개다. 역전파를 마치면 관련 기울기도 각 파라미터와 같은 shape에 저장된다. 하지만 그 숫자들을 계산할 때는 여러 이미지와 여러 공간 위치에서 사용한 가중치의 영향을 모두 반영한다. <strong>같은 가중치를 여러 번 쓴 계산 경로들의 영향을 합쳐, 그 가중치에 대한 하나의 기울기를 얻는다.</strong> 기울기 결과가 209개라고 연산을 209번만 하는 것은 아니다.

함수를 정의하는 것만으로 학습이 시작되지도 않는다. 실제 배치를 넣어 `train_step(model, optimizer, noisy, clean)`을 호출해야 한다. 또한 반환문은 `step()` 뒤에 있지만, `error`는 <strong>갱신 전 순전파에서 계산한 손실</strong>이다. 반환하는 위치가 측정 시점을 바꾸지 않는다.

원문의 `error.detach().item()`도 같은 기록 목적이다. `detach()`는 자동 미분 연결에서 분리한 텐서를, `item()`은 원소 하나인 텐서의 Python 숫자를 반환한다. Python 숫자 하나만 반환하는 이 경우에는 `error.item()`만으로 충분하다.[^item][^detach]

## 9. `split → model → cat`은 이미지를 나누어 처리한 뒤 순서대로 잇는다

검증 이미지를 한꺼번에 처리하면 중간 특징맵도 한꺼번에 계산해야 한다. 이를 작은 묶음으로 나누어 추론하면 한 번의 모델 호출에 필요한 중간 메모리를 줄일 수 있다.

원문의 `torch.cat([model(part) for part in noisy.split(256)])`를 읽으려면 세 연산을 분리하면 된다.

| 연산 | 하는 일 | 반환값 |
| --- | --- | --- |
| `noisy.split(256, dim=0)` | 이미지 번호 축을 최대 256장씩 나눈다. | 텐서 조각들이 들어 있는 튜플 |
| `model(part)` | 현재 이미지 묶음을 복원한다. | 같은 묶음 크기의 복원 텐서 |
| `torch.cat(results, dim=0)` | 결과들을 기존 이미지 번호 축으로 잇는다. | 전체 복원 텐서 |

`split`이 반환하는 조각은 원본 데이터를 공유하는 <strong>뷰(View)</strong>다. 이 코드에서는 조각을 읽어서 모델 입력으로 사용한다. 나누는 축의 길이가 256으로 나누어떨어지지 않으면 마지막 조각만 작아진다.[^split]

그림의 <strong>600장은 실제 검증 데이터 수가 아니라 설명용 장수</strong>다. 실제 개수는 `noisy_val.shape[0]`으로 확인한다.

<figure class="dae-figure">
<img src="/denoising-autoencoder-pytorch/split-cat-ko.svg" alt="설명용 이미지 600장을 256장, 256장, 88장으로 나눈다. 각 묶음을 같은 모델로 복원하고 dim=0으로 연결하여 원래 순서의 (600,1,28,28) 복원 텐서를 얻는다." loading="lazy">
<figcaption>묶음 1의 첫 이미지는 전체 인덱스 256번, 묶음 2의 첫 이미지는 512번이다. 순서대로 처리하고 연결하면 원본과의 대응이 유지된다.</figcaption>
</figure>

```python
def predict_fixed(model, noisy, batch_size=256):
    if noisy.shape[0] == 0:
        raise ValueError("복원할 이미지가 없습니다.")

    model.eval()  # 평가 모드로 바꾼다.
    with torch.no_grad():  # 기울기 계산용 기록을 만들지 않는다.
        results = []
        for part in noisy.split(batch_size, dim=0):
            reconstructed = model(part)
            results.append(reconstructed)
        return torch.cat(results, dim=0)  # 기존 이미지 축을 따라 연결한다.
```

리스트 컴프리헨션은 여기의 반복문과 `append`를 짧게 적은 형태다. `cat`은 픽셀값을 더하거나 평균내지 않는다. 이미지들의 나열을 이어 붙인다. 반면 `stack`은 새 축을 추가하며, 입력 텐서들의 shape도 같아야 한다. 256·256·88장처럼 마지막 묶음이 작은 이번 목적에는 `cat`이 맞는다.[^cat]

원본 입력, 각 묶음의 출력, 연결한 최종 결과를 저장할 메모리는 여전히 필요하다. 전체 결과가 필요 없고 MSE만 필요하다면, 묶음별 제곱 오차의 합과 원소 수만 누적하는 방법도 가능하다.

### `eval()`과 `no_grad()`를 함께 사용하는 이유

`eval()`은 층의 동작 모드를 평가용으로 바꾼다. 일부 값을 무작위로 제외하는 Dropout이나 정규화 통계를 사용하는 BatchNorm처럼 모드에 따라 동작이 달라지는 층에 영향을 준다. `no_grad()`는 자동 미분을 위한 계산 기록을 끈다. 둘은 서로 독립된 기능이다.[^autograd]

| 설정 | 층의 학습·평가 모드 | 기울기 계산용 기록 |
| --- | --- | --- |
| `model.eval()`만 사용 | 평가 모드로 변경 | 따로 끄지 않음 |
| `torch.no_grad()`만 사용 | 기존 모드 유지 | 사용 범위 안에서 비활성화 |
| 둘 다 사용 | 평가 모드 | 비활성화 |

현재 모델에는 Dropout과 BatchNorm이 없어서 `train()`과 `eval()` 전환 자체가 출력 계산법을 바꾸지는 않는다. 그래도 역할을 나누어 작성하면 모델에 다른 층을 추가했을 때도 의도를 유지하기 쉽다.

`predict_fixed`라는 이름 자체가 입력을 고정하지는 않는다. 같은 `noisy_val`을 반복 전달하므로 고정된 입력으로 비교하는 것이다. 함수는 모델을 평가 모드로 남기며, 다음 `train_step()`의 `net.train()`이 학습 모드로 다시 바꾼다.

## 10. 학습을 시작하기 전에 초기 상태를 분명히 만든다

<strong>시드(Seed)</strong>는 난수 생성의 시작 상태를 지정하는 값이다. 같은 환경에서 같은 초기화와 난수 사용 순서를 재현할 때 쓴다. 시드 하나로 서로 다른 PyTorch 버전·장치·플랫폼의 모든 결과가 같아진다고 보장할 수는 없다.[^randomness]

<strong>Adam</strong>은 현재 기울기뿐 아니라 이전 기울기에 관한 내부 상태도 사용하는 옵티마이저다. `lr`은 학습률 인자다. 아래 `0.01`은 원본 실습의 설정이며, 모든 데이터와 모델에 적합하다는 뜻은 아니다.[^adam]

```python
# SEED는 원본 실습에서 정한 정수다.
# 이 셀을 실행하면 새 모델과 새 옵티마이저로 실험을 시작한다.
torch.manual_seed(SEED)
model = LabDenoiser(feature_channels=8).cpu()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 서로 다른 용도의 난수 흐름을 따로 관리한다.
shuffle = torch.Generator().manual_seed(SEED + 4000)
noise_rng = torch.Generator().manual_seed(SEED + 5000)

# 학습 전 모델로 고정된 검증 입력을 복원한다.
initial_recon = predict_fixed(model, noisy_val)
initial_mse = image_mse(initial_recon, clean_val).mean().item()
history = [[0, None, initial_mse]]  # [에포크, 훈련 MSE, 검증 MSE].
```

`model.parameters()`는 방금 만든 모델의 파라미터를 옵티마이저에 연결한다. `SEED+4000`과 `SEED+5000`의 덧셈값에는 오토인코더의 특별한 의미가 없다. 섞기와 잡음 생성에 서로 다른 난수 흐름을 사용하려는 선택이다. 생성기를 사용할 때마다 상태가 진행되므로, 매 호출이 같은 난수를 반환하는 것은 아니다.

`history`는 복원 이미지를 담는 리스트가 아니다. 에포크 번호와 손실 숫자를 담는다. 첫 기록의 `None`은 아직 훈련 오차를 측정하지 않았다는 뜻이며, 오차 0을 뜻하지 않는다. `initial_recon`은 위에서 새로 만든 모델의 학습 전 출력이다. 학습된 모델을 유지한 채 평가 부분만 다시 실행했다면 그 결과를 미학습 모델의 출력이라고 부르면 안 된다.

원문에 있는 `float((initial_recon-clean_val).square().mean())`도 스칼라 텐서를 기록용 Python 실수로 바꾸는 표현이다. 여기의 `.item()`과 같은 목적으로 쓰며, 둘 다 새로운 학습을 실행하지 않는다.

모델의 중간 shape를 확인하고 싶다면 다음처럼 등록한 층을 차례로 실행할 수 있다. 일반적인 추론은 `model(x)` 한 번이면 충분하며, 이 코드는 내부 연결을 확인하는 용도다.

```python
with torch.no_grad():
    h = noisy_val[:2]
    print("input", tuple(h.shape))
    for name, part in [("encoder", model.encoder), ("decoder", model.decoder)]:
        for layer in part:
            h = layer(h)  # 현재 결과를 다음 층의 입력으로 사용한다.
            # :<16은 층 이름을 최소 16칸 너비로 왼쪽 정렬하는 출력 형식이다.
            print(f"{name} {layer.__class__.__name__:<16}", tuple(h.shape))
```

검증 이미지가 두 장 이상이면 `(2,1,28,28) → (2,8,14,14) → (2,8,14,14) → (2,1,28,28) → (2,1,28,28)`을 확인할 수 있다. 이 순전파 검사만으로 가중치가 학습되지는 않는다.

## 11. 5,000장을 섞고 나누어 3에포크 학습한다

<strong>학습 스텝(Training Step)</strong>은 한 배치로 손실과 기울기를 계산하고 파라미터를 한 번 갱신하는 과정이다. <strong>에포크(Epoch)</strong>는 이번 훈련 데이터 전체를 한 번씩 학습에 사용한 과정이다.

실습은 훈련 이미지 5,000장, 훈련 배치 크기 최대 64장, 3에포크를 사용한다. 64장짜리 배치 78개가 4,992장을 처리하고, 마지막 배치가 남은 8장을 처리한다. 따라서 한 에포크는 79스텝, 3에포크는 <strong>237번의 `optimizer.step()`</strong>이다. 검증에서 사용한 256은 추론 묶음 크기이며 훈련 배치 크기와 별개다.

### 먼저 인덱스를 섞고, 그 인덱스로 이미지를 꺼낸다

`torch.randperm(5000)`은 `0~4999`의 인덱스를 중복 없이 섞은 `(5000,)` 텐서를 만든다. 여기에 `.split(64)`를 적용하면 이미지 자체가 아니라 <strong>이미지 인덱스</strong>를 최대 64개씩 나눈다. `clean_train[ids]`가 그 번호에 해당하는 실제 이미지를 선택한다.[^randperm]

| 코드 | 내용 | shape |
| --- | --- | --- |
| `shuffled_ids` | 섞인 이미지 번호 5,000개 | `(5000,)` |
| 일반 배치의 `ids` | 선택할 이미지 번호 64개 | `(64,)` |
| `clean_train[ids]` | 선택한 이미지 64장 | `(64,1,28,28)` |
| 마지막 배치의 `ids` | 남은 이미지 번호 8개 | `(8,)` |
| 마지막 `clean_train[ids]` | 선택한 이미지 8장 | `(8,1,28,28)` |

검증에서는 이미지 텐서를 `split`했고, 여기서는 인덱스 텐서를 `split`한다. `split`의 규칙은 같지만 나누는 대상의 의미가 다르다.

### 잡음은 원본과 같은 shape로 만든다

`torch.randn(clean.shape)`는 주어진 shape에 평균 0, 표준편차 1인 정규분포 난수를 만든다. <strong>표준편차(Standard Deviation)</strong>는 값이 평균 주변에서 퍼지는 정도다. 여기에 0.25를 곱하면 더하기 전 잡음의 표준편차가 0.25가 된다.[^randn]

`clean + noise`는 원본의 각 위치에 해당 위치의 잡음을 더한다. `.clamp(0, 1)`은 0보다 작은 값을 0으로, 1보다 큰 값을 1로 제한한다. 범위 안의 값은 그대로 둔다. 제한 과정이 있으므로 최종 `noisy-clean`의 분포와 표준편차가 원래 잡음과 똑같다고 말할 수는 없다.[^clamp]

```python
for epoch in range(1, 4):  # 실제 반복값 1, 2, 3. 끝값 4는 제외한다.
    sse, count = 0.0, 0
    shuffled_ids = torch.randperm(len(clean_train), generator=shuffle)

    for ids in shuffled_ids.split(64):
        clean = clean_train[ids]  # 보통 (64,1,28,28), 마지막은 (8,1,28,28).
        noise = 0.25 * torch.randn(
            clean.shape,
            generator=noise_rng,  # 잡음 전용 난수 생성기를 사용한다.
            dtype=clean.dtype,    # 원본과 같은 숫자 자료형.
            device=clean.device,  # 원본과 같은 계산 장치.
        )
        noisy = (clean + noise).clamp(0, 1)
        loss = train_step(model, optimizer, noisy, clean)

        # 이 두 줄은 기록용 통계다. 원소 수가 다른 배치들의 평균을 올바르게 모은다.
        sse += loss * clean.numel()
        count += clean.numel()

    train_mse = sse / count
    recon = predict_fixed(model, noisy_val)  # 같은 검증 입력. 가중치 갱신 없음.
    validation_mse = image_mse(recon, clean_val).mean().item()
    history.append([epoch, train_mse, validation_mse])
    print(f"epoch {epoch}: train={train_mse:.4f}, validation={validation_mse:.4f}")
```

같은 원본도 다음 에포크에서는 새로운 잡음과 함께 학습한다. 반면 검증 입력은 계속 같은 `noisy_val`이다. 출력의 `.4f`는 표시할 소수 자릿수만 정한다. `history`에 저장된 실제 숫자를 그 자릿수로 바꾸지는 않는다.

이 실습은 메모리에 있는 텐서를 직접 섞고 나눈다. 데이터가 커지면 <strong>Dataset</strong>으로 데이터 한 개를 읽는 방법을 정의하고, <strong>DataLoader</strong>로 배치 구성·섞기·읽기를 관리할 수 있다. 사용하는 도구가 달라져도 배치 준비, 순전파, 손실, 역전파, 갱신의 순서는 같다.[^dataloader]

## 12. 배치 평균을 모을 때는 그 평균을 만든 원소 수도 필요하다

앞의 `loss`는 현재 배치의 평균 제곱 오차다. `clean.numel()`은 그 평균을 만든 픽셀 원소 수다. <strong>제곱 오차 합계(Sum of Squared Errors, SSE)</strong>를 구하려면 평균에 원소 수를 곱한다.

`배치 MSE = 그 배치의 제곱 오차 합계 / 그 배치의 원소 수`

따라서 `loss * clean.numel()`은 해당 배치의 제곱 오차 합계에 해당한다. 모든 배치의 합계와 원소 수를 각각 모은 뒤 나누면, 처리한 모든 픽셀의 평균 오차가 된다. 실제 부동소수점 계산에는 작은 반올림 차이가 있을 수 있다.

| 배치 | 이미지 수 | 평균을 만든 픽셀 수 |
| --- | ---: | ---: |
| 일반 배치 | 64 | `64×1×28×28 = 50,176` |
| 마지막 배치 | 8 | `8×1×28×28 = 6,272` |

두 배치의 MSE를 같은 비중으로 평균내면, 8장짜리 배치가 64장짜리 배치와 같은 비중을 갖는다. 그러면 마지막 배치의 이미지 한 장이 상대적으로 8배 큰 비중을 갖는다. 우리가 구하는 것은 배치라는 묶음들의 단순 평균이 아니라 <strong>처리한 모든 픽셀 제곱 오차의 평균</strong>이다.

이렇게 원소 수에 맞춰 비중을 적용하는 평균을 <strong>가중 평균(Weighted Mean)</strong>이라고 한다. 모든 이미지 크기가 같으면 이미지 수로 가중해도 같다. 각 이미지의 픽셀 수 784가 분자와 분모에서 공통으로 없어지기 때문이다. 다른 손실 함수에서는 무엇으로 나누어 평균냈는지부터 확인해야 한다.

여기에는 학습과 기록을 구분하는 중요한 경계가 있다. `train_step()` 안에서 이미 `backward()`와 `step()`이 끝났다. `sse`와 `count`는 그때 측정한 숫자를 보고용으로 합친다. <strong>이 집계는 마지막 작은 배치의 파라미터 갱신 비중까지 보정하지 않는다.</strong>

### 훈련 MSE와 검증 MSE는 측정 시점도 다르다

`history`의 각 기록은 `[에포크 번호, 훈련 MSE, 검증 MSE]`다. `record[0]`은 에포크, `record[1]`은 훈련 오차, `record[2]`는 검증 오차다. 시작 기록의 훈련 값은 `None`이므로, 훈련 곡선에는 그 기록을 제외한다.

| 기록 | 어떤 데이터로 측정하는가? | 어떤 모델 상태로 측정하는가? |
| --- | --- | --- |
| 훈련 MSE | 배치마다 새 잡음을 넣은 훈련 이미지 | 에포크 도중 계속 갱신되는 여러 시점의 모델 |
| 검증 MSE | 고정된 검증 잡음 이미지 | 에포크가 끝난 시점의 모델 하나 |

훈련 MSE는 에포크가 끝난 모델로 훈련 데이터를 다시 평가한 값과 다르다. 그래서 검증 MSE가 훈련 MSE보다 낮다는 사실만으로 버그라고 단정할 수 없다. 데이터, 잡음, 측정 시점이 모두 다르다.

```python
import matplotlib.pyplot as plt

# history[1:]은 시작 기록을 제외한다. 시작 기록에는 훈련 MSE가 없다.
plt.plot(
    [record[0] for record in history[1:]],
    [record[1] for record in history[1:]],
    label="train MSE",
)
plt.plot(
    [record[0] for record in history],
    [record[2] for record in history],
    label="validation MSE",
)
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()
```

두 오차가 함께 감소하면 훈련 과정의 오차와 고정된 검증 복원 오차가 함께 줄어든 것이다. 훈련 오차만 줄고 검증 오차가 계속 나빠지면 <strong>과적합(Overfitting)</strong>, 즉 훈련 데이터에 맞추는 동안 새로운 데이터의 성능이 나빠지는 가능성 등을 살핀다. 두 값이 거의 개선되지 않으면 입력·정답 대응, 값 범위, 기울기, 학습률, 모델 구조부터 확인한다. 세 에포크의 짧은 결과만으로 장기 경향을 확정하지는 않는다.

## 13. 학습 전보다 좋아졌는가와 잡음 입력보다 좋아졌는가는 다른 질문이다

처음 모델의 복원 오차가 매우 컸다면 학습으로 오차가 줄어도, 잡음 이미지를 그대로 내보낸 것보다 나쁠 수 있다. 따라서 비교 기준을 하나 더 둔다. <strong>기준선(Baseline)</strong>은 모델이 얼마나 개선했는지 판단할 비교 대상이다. 여기서는 ‘모델을 사용하지 않고 잡음 입력을 그대로 출력했을 때의 MSE’를 기준선으로 삼는다.

두 비교 모두 같은 깨끗한 원본을 사용한다.

| 비교 대상 | 정답 | 알려주는 것 |
| --- | --- | --- |
| 잡음 입력 `noisy_val` | `clean_val` | 원래 입력의 픽셀 오차 |
| 모델 출력 `recon` | 같은 `clean_val` | 모델을 거친 뒤의 픽셀 오차 |

```python
recon = predict_fixed(model, noisy_val)
baseline = image_mse(noisy_val, clean_val).mean().item()
restored = image_mse(recon, clean_val).mean().item()
print(f"입력 MSE={baseline:.4f}, 복원 MSE={restored:.4f}")

# 각 이미지의 입력 오차에서 복원 오차를 뺀다. 결과 shape는 (N,)이다.
delta = image_mse(noisy_val, clean_val) - image_mse(recon, clean_val)
improved = (delta > 0).sum().item()  # True인 이미지 개수를 센다.
worsened = (delta < 0).sum().item()
tied = (delta == 0).sum().item()
print("개선 / 악화 / 정확히 동률:", improved, worsened, tied)
```

원문에서 복원 호출 직전에 둔 `recon = None`은 다음 대입으로 바로 덮어쓴다. 위처럼 바로 `recon = predict_fixed(...)`를 실행하는 정상 흐름에서는 그 초기 대입을 생략해도 된다.

`restored < baseline`이면 전체 평균 픽셀 오차가 줄었다. 하지만 모든 이미지가 개선되었다는 뜻은 아니다. `delta[i]`는 i번 이미지가 얼마나 개선되었는지 구분한다. 양수면 복원 MSE가 더 작고, 음수면 더 크다.

`delta == 0`은 허용오차 없이 정확히 같은지 비교한다. 출력에 보이는 소수 넷째 자리 숫자가 같다는 뜻과는 다르다. 실질적으로 비슷한 결과를 묶어야 한다면 평가 목적에 맞는 <strong>허용오차(Tolerance)</strong>를 정해 `delta > tolerance`, `delta < -tolerance`, `abs(delta) <= tolerance`로 나눌 수 있다. 그 기준값도 결과와 함께 기록해야 한다.

### 같은 이미지의 배경과 숫자 획을 함께 본다

픽셀 MSE 하나로 사람이 판단하는 복원 품질을 모두 설명할 수는 없다. 배경 잡음이 줄었지만 숫자의 가는 획도 지워졌을 수 있고, 획이 흐려졌지만 넓은 배경의 오차가 줄어 전체 MSE는 낮아졌을 수도 있다. 숫자만으로 위치별 변화를 알 수 없으므로 다음을 함께 확인한다.

| 관찰 대상 | 확인할 내용 |
| --- | --- |
| 전체 평균 | 복원 MSE가 입력 MSE보다 낮은가? |
| 이미지별 오차 | 어떤 이미지가 개선되고 어떤 이미지가 악화되었는가? |
| 배경 | 잡음 점과 불필요한 밝기가 줄었는가? |
| 숫자 획 | 가는 획, 굵기, 연결, 원래 형태가 유지되는가? |

한 비교 묶음에는 반드시 <strong>같은 원본의 깨끗한 이미지·잡음 입력·복원 결과</strong>를 놓는다. 그림마다 명암 범위가 달라 생기는 착시를 줄이려면 `imshow(..., cmap="gray", vmin=0, vmax=1)`처럼 공통 범위를 사용한다. `cmap`은 명암 표시 방식이고 `vmin`, `vmax`는 픽셀값을 표시할 공통 범위다.[^imshow]

다운로드 코드의 `plot_triplets(clean, noisy, recon)`은 같은 인덱스의 세 이미지를 같은 열에 놓는다. 기본값은 최대 6장이며, 제목의 `Index`는 전달받은 텐서에서의 위치 번호다. 원래 데이터셋의 고유 ID와 자동으로 같아지는 것은 아니다.

원본 자료의 `tools.draw_triplets(...)`는 사용자 정의 함수다. PyTorch 기본 함수가 아니므로 `val_ids`, `same8`, `OUT`이 무엇을 뜻하는지는 구현을 확인해야 한다. 필요하면 `inspect.getsource(tools.draw_triplets)`로 함수 소스를 읽는다. 다운로드 코드는 그 도우미 파일 없이 그림을 그리지만, <strong>실제 원본 이미지 데이터는 별도로 필요하다.</strong>

결과를 적을 때는 “학습했으므로 잡음이 제거되었다”라고 먼저 결론 내리지 않는다. 입력·복원 MSE의 실제 값, 개선·악화된 이미지, 눈으로 확인한 획과 배경의 변화를 연결해서 적는다.

## 14. 4채널과 8채널은 중간 특징맵의 개수가 다르다

다음 표는 <strong>인코더 출력 채널만 4와 8로 바꾸고 커널·stride·padding·나머지 층을 동일하게 유지한 비교</strong>다. 이론 슬라이드의 원본이 없으므로 슬라이드 모델도 반드시 이 구조였다고 확정하지 않는다.

| 비교 항목 | 중간 4채널 | 중간 8채널 |
| --- | --- | --- |
| 입력 | `(B,1,28,28)` | `(B,1,28,28)` |
| 인코더 필터 수 | 4 | 8 |
| 중간 특징 | `(B,4,14,14)` | `(B,8,14,14)` |
| 이미지 한 장당 중간값 수 | 784 | 1,568 |
| 최종 출력 | `(B,1,28,28)` | `(B,1,28,28)` |
| 인코더 파라미터 | 40 | 80 |
| 디코더 파라미터 | 65 | 129 |
| 전체 파라미터 | <strong>105</strong> | <strong>209</strong> |

4와 8은 이미지 수나 최종 출력의 흑백 채널 수가 아니다. 이미지마다 계산하는 중간 특징맵의 개수다. 8채널 모델은 더 많은 중간 특징을 사용할 수 있지만, 실제 MSE가 더 낮거나 획이 더 잘 보존된다는 보장은 없다.

비교하려면 훈련·검증 원본, 고정 검증 잡음, 학습량, 평가 방법을 같게 두고 각 모델과 옵티마이저를 새로 만들어야 한다. 다운로드 코드의 `run_experiment(data, seed=SEED, feature_channels=4)`와 `feature_channels=8` 호출은 각각 새 모델을 만든다. 섞기와 잡음 생성기는 모델 초기화와 분리했으므로 두 실험에서 같은 데이터 순서와 훈련 잡음을 사용할 수 있다. 같은 시드가 서로 다른 구조의 가중치를 일대일로 같게 만든다는 뜻은 아니다.

차이가 작다면 여러 시드의 결과도 확인해 한 번의 초기화에 따른 차이인지 살핀다. 구조상 차이는 위 표로 확정할 수 있지만, 성능 차이는 실제 실험 결과로 판단해야 한다.

## 15. 노트북의 셀을 다시 실행하면 무엇이 새로 시작될까?

노트북은 앞에서 만든 객체를 계속 보관한다. 학습 루프만 다시 실행하면 화면의 에포크 번호는 다시 1부터 시작할 수 있지만, 모델은 이미 학습한 가중치를 가지고 있을 수 있다. `history`를 비우는 것과 모델을 초기화하는 것은 별개다.

| 다시 실행한 코드 | 새로 시작하는 것 | 그대로 남을 수 있는 것 |
| --- | --- | --- |
| `history = []` | 기록 리스트 | 모델 가중치, Adam 상태 |
| 난수 생성기와 시드 설정 | 이후 난수 흐름 | 이미 만들어진 모델 가중치, Adam 상태 |
| `model = LabDenoiser()` | 새 모델 가중치 | 이전 옵티마이저와 그 상태 |
| 새 모델로 새 Adam 생성 | 새 파라미터 연결과 Adam 상태 | 별도로 보관한 과거 결과 |

새 실험은 <strong>시드 설정 → 새 모델 생성 → 그 모델로 새 옵티마이저 생성 → 난수 생성기와 기록 초기화 → 학습</strong> 순서로 시작한다. 모델만 새로 만들고 기존 옵티마이저를 쓰면, 옵티마이저가 이전 모델의 파라미터를 계속 참조할 수 있다.

클래스 정의를 수정했다고 이미 만들어 둔 `model` 객체가 자동으로 새 구조로 바뀌지도 않는다. 수정한 클래스 정의를 실행하고 모델 객체를 다시 만든다. 제공한 `run_experiment()`는 이 초기화 과정을 매 호출마다 수행한다.

`torch.set_num_threads(4)`의 4는 CPU 연산에 사용할 스레드 수다. 이미지 수, 채널 수, 층 수와는 관계없다. 여러 장치에서 가장 빠른 설정을 보장하는 숫자도 아니다. 통합 코드는 원본 실습의 CPU 설정을 유지한다.[^threads]

## 16. 준비 단계의 `StopIteration`은 어떤 계산에서 발생했을까?

원본 준비 코드에는 다음과 같은 파일 탐색이 있었다.

```python
from pathlib import Path

# 이해를 위한 원래 코드다. 두 후보에 파일이 없으면 StopIteration이 발생한다.
ASSETS = next(
    p.resolve()
    for p in [Path("assets"), Path("../assets")]
    if (p / "d2_tools.py").exists()
)
```

`Path("assets")`는 현재 기준 폴더 아래의 `assets`를 뜻하고, `Path("../assets")`는 상위 폴더 아래의 `assets`를 뜻한다. 두 경로에서 `d2_tools.py`가 있는 후보만 남긴다. `next()`는 그렇게 남은 후보 중 첫 번째를 가져온다.

<strong>직접 원인은 두 후보 모두 조건을 통과하지 않아 가져올 경로가 없는데, 기본값 없는 `next()`를 호출한 것이다.</strong> 모델 학습이나 합성곱이 실패한 상황이 아니다. 다만 왜 후보 경로에서 파일이 발견되지 않았는지는 추가로 확인해야 한다. 파일이 누락되었을 수도 있고, 상대 경로의 기준 폴더가 예상과 달랐을 수도 있다.[^next]

<strong>현재 작업 디렉터리(Current Working Directory)</strong>는 상대 경로를 해석하는 기준 폴더다. 노트북 파일을 저장한 폴더와 항상 같다고 가정하지 말고 `Path.cwd()`로 확인한다. `resolve()`는 경로를 절대 경로로 해석하며, 없던 파일을 만들어 주지 않는다.[^pathlib]

```python
def locate_assets():
    candidates = [Path("assets"), Path("../assets")]
    found = next(
        (
            directory.resolve()
            for directory in candidates
            # is_file()은 이름이 존재하는 것에 더해 실제 파일인지 검사한다.
            if (directory / "d2_tools.py").is_file()
        ),
        None,  # 조건에 맞는 후보가 없으면 None을 반환한다.
    )

    if found is None:
        checked = "\n".join(
            str((directory / "d2_tools.py").resolve())
            for directory in candidates
        )
        raise FileNotFoundError(
            f"현재 작업 디렉터리: {Path.cwd()}\n"
            f"d2_tools.py를 찾지 못했습니다.\n검사한 경로:\n{checked}"
        )
    return found

# 원본 도우미 파일이 실제로 필요한 준비 셀에서 호출한다.
# ASSETS = locate_assets()
```

`next(..., None)`은 파일 누락을 고치는 코드가 아니다. 후보가 없을 때 무엇이 문제인지 더 구체적으로 설명하기 위한 처리다. 출력된 절대 경로와 실제 파일 위치를 비교한 다음 원래의 `assets` 폴더를 복구하거나 경로를 맞춰야 한다. 빈 `d2_tools.py`를 만들어도 필요한 함수가 없으므로 해결되지 않는다.

뒤에 이어지는 `importlib` 코드는 발견한 Python 파일을 모듈로 불러와 실행하기 위한 단계다. 파일을 찾는 단계와 그 파일 안의 코드를 실행하는 단계를 나누면 어디에서 실패했는지 분명해진다.

## 17. 설명을 이해한 뒤 전체 실험을 한 번 실행한다

위의 짧은 코드들은 각 단계의 의미를 설명한다. [전체 Python 실습 파일](/denoising-autoencoder-pytorch/examples-ko.py)은 데이터 검사, 모델 생성, 학습, 평가, 그림 함수를 한곳에 묶었다. 다음 세 조건을 먼저 준비한다.

| 필요한 것 | 준비 조건 |
| --- | --- |
| Python 패키지 | 현재 노트북 커널에서 `torch`, `matplotlib`를 불러올 수 있어야 한다. |
| `data` | `clean_train`, `clean`, `noisy` 텐서가 있어야 한다. 훈련 데이터는 최소 5,000장, 모든 이미지는 `(N,1,28,28)`과 0~1 값 범위다. |
| 원본 대응 | 검증 `clean`과 `noisy`의 같은 인덱스가 같은 원본이어야 하며, 검증 원본은 훈련에서 분리되어 있어야 한다. |

아래는 다운로드한 파일을 노트북의 현재 작업 폴더에 둔 경우다. `runpy.run_path()`는 지정한 Python 파일을 실행하고, 그 파일에 정의된 이름들을 딕셔너리로 반환한다. 이 파일은 정의를 읽는 단계에서 학습하지 않으므로, 함수들을 불러온 후 `run_experiment()`를 호출한다.[^runpy]

```python
import runpy

# examples-ko.py를 현재 작업 폴더에 내려받은 뒤 실행한다.
# 파일 위치가 다르면 실제 파일의 경로로 바꾼다.
examples = runpy.run_path("examples-ko.py")
run_experiment = examples["run_experiment"]
plot_history = examples["plot_history"]
plot_triplets = examples["plot_triplets"]

# data와 SEED는 원본 학습 자료에서 준비한 값이다.
# 호출할 때마다 새 모델·새 옵티마이저로 3에포크를 학습한다.
result = run_experiment(data, seed=SEED, feature_channels=8)

plot_history(result["history"])
plot_triplets(result["clean_val"], result["noisy_val"], result["recon"])

print(result["counts"])    # {'encoder': 80, 'decoder': 129, 'total': 209}
print(result["baseline"])  # 실제 검증 잡음 입력의 MSE.
print(result["restored"])  # 실제 복원 결과의 MSE. 입력보다 작을지는 실행해서 판단한다.
```

`result`에는 모델, 옵티마이저, `history`, 시작·최종 복원 이미지, 검증 입력·정답, 기준선·복원 MSE, 이미지별 `delta`, 파라미터 수가 들어 있다. 4채널을 비교하려면 `feature_channels=4`로 별도 호출하고 결과를 다른 변수에 보관하면 된다.

이 코드는 Python 3.12.14와 PyTorch 2.15.0+cpu 환경에서 합성 데이터로 전체 실행을 확인했다. shape, 파라미터 수, 평균 계산, 배치 순서, 237회 갱신, 그림 생성을 검증했고 같은 시드에서 원본 실습 코드와 복원 텐서도 비교했다. 이는 <strong>실행 흐름의 검증</strong>이다. 원본 수업 이미지의 품질 평가 결과는 아니다.

## 18. Rust에서도 나누기·연결하기·평균 집계의 원리는 같다

Rust의 `Vec<T>`는 같은 자료형의 값들을 순서대로 담는 가변 길이 배열이다. <strong>슬라이스(Slice)</strong>는 이미 존재하는 연속된 데이터 구간을 참조하는 방식이다. `.chunks(256)`은 최대 256개씩의 슬라이스를 제공하고, `extend_from_slice()`는 그 안의 값을 결과 벡터 뒤에 이어 붙인다.[^rust]

| Python 실습의 개념 | Rust 표준 라이브러리에서의 대응 |
| --- | --- |
| 인덱스 범위 `range(600)` | `0..600`, 끝값 제외 |
| 순서대로 묶음 나누기 | `slice.chunks(256)` |
| 처리 결과를 순서대로 이어 붙이기 | `Vec::extend_from_slice()` |
| 평균 × 원소 수를 합친 뒤 전체 원소 수로 나누기 | 같은 가중 평균 계산 |

아래 코드는 실제 이미지 대신 설명용 ID 600개로 순서 보존을 확인한다. 신경망이나 자동 미분을 구현하는 코드는 아니다. 평균 함수의 `Option<f64>`는 값이 있으면 `Some(평균)`, 평균을 계산할 원소가 없으면 `None`을 반환한다는 자료형이다. `match`는 이 두 경우를 나누어 처리한다.

```rust
// 이 예제는 데이터 분할·연결·집계를 확인한다. 오토인코더 구현은 아니다.
fn weighted_epoch_mse(batches: &[(f64, usize)]) -> Option<f64> {
    // batches: (평균 손실, 그 평균을 만든 원소 수) 튜플들의 참조.
    // f64는 실수 자료형, usize는 길이·인덱스 등에 쓰는 정수 자료형이다.
    let mut sse = 0.0;
    let mut count: usize = 0;
    for &(mean_loss, elements) in batches {
        // &패턴으로 참조가 가리키는 튜플을 읽고, 복사 가능한 숫자들을 받는다.
        sse += mean_loss * elements as f64; // as f64: 곱셈을 위해 실수로 변환.
        count += elements;
    }
    if count == 0 { None } else { Some(sse / count as f64) }
}

fn main() {
    // 600은 설명용 ID 개수다. 0..600의 마지막 실제 값은 599다.
    let image_ids: Vec<usize> = (0..600).collect();
    let mut combined: Vec<usize> = Vec::new();
    for part in image_ids.chunks(256) {
        println!("묶음의 이미지 수: {}", part.len()); // 256, 256, 88.
        combined.extend_from_slice(part); // 실제 추론 대신 ID를 순서대로 연결.
    }
    assert_eq!(combined, image_ids); // 연결 전후 순서와 내용이 같은지 확인.

    // 두 손실값은 평균 집계를 위해 정한 예시이며 실제 실험 성능이 아니다.
    // 원소 수는 각각 흑백 28×28 이미지 64장과 8장에서 계산했다.
    let batches = [(0.04, 64 * 28 * 28), (0.01, 8 * 28 * 28)];
    match weighted_epoch_mse(&batches) {
        Some(mean) => println!("예시 가중 평균 MSE: {mean:.6}"),
        None => println!("평균을 계산할 원소가 없습니다."),
    }
}
```

Python과 Rust 모두 반복 대상에서 값을 받아 지역 변수에 넣고, 묶음의 순서를 유지하면서 처리한다. 차이는 이번 Python 코드의 PyTorch Tensor가 다차원 연산과 자동 미분까지 제공한다는 점이다. 위 Rust 코드는 표준 라이브러리로 일반 데이터의 흐름만 확인한다. Rust에서 실제 신경망을 만들려면 해당 기능을 제공하는 별도의 구현이 필요하다.

## 19. 어디까지 익히면 다음 실습으로 넘어가도 될까?

전치 합성곱의 모든 변형과 내부 구현을 외우는 것이 목표는 아니다. 이번에는 아래 코드가 다루는 대상을 설명할 수 있으면 된다.

| 중요도 | 직접 설명하거나 확인할 내용 |
| --- | --- |
| 필수 | 딕셔너리 키로 텐서를 꺼내는 것과 텐서 축을 인덱싱하는 것을 구분한다. |
| 필수 | 입력·중간 특징·복원의 shape와, 별도로 저장된 가중치 shape를 읽는다. |
| 필수 | 같은 이미지의 같은 픽셀을 비교하고, 전체 평균과 이미지별 평균을 구분한다. |
| 필수 | `backward()`는 기울기 계산, `step()`은 파라미터 갱신임을 설명한다. |
| 필수 | `split → model → cat`에서 이미지 순서가 보존되는 이유를 설명한다. |
| 필수 | 마지막 작은 배치를 포함한 평균을 원소 수로 가중해 기록한다. |
| 필수 | 고정 검증 입력의 기준선·복원 MSE와 실제 획·배경을 함께 확인한다. |
| 알아두기 | 4/8채널 비교를 할 때 모델과 Adam을 함께 새로 만들고 조건을 맞춘다. |
| 문제를 만났을 때 참고 | 전치 합성곱 일반식, 다른 업샘플링 구조, 파일 탐색 진단 코드를 확인한다. |

처음의 질문으로 돌아가 보자. 모델은 잡음 이미지를 입력받아 복원 이미지를 계산한다. 같은 원본의 같은 픽셀과 비교한 손실로 기울기를 구하고 가중치를 갱신한다. 평가에서는 고정된 입력을 다시 복원한 뒤, <strong>입력 자체보다 실제로 오차가 줄었는지, 줄어드는 과정에서 필요한 숫자 획을 잃지는 않았는지</strong> 확인한다. 이 연결을 코드에서 따라갈 수 있다면 다음 데이터와 모델에서도 같은 학습 흐름을 적용할 수 있다.

## 참고 자료

본문의 구조·shape·파라미터 수·배치 수는 제시한 모델과 실습 조건으로 직접 계산했다. API 동작과 관련 개념은 아래 공식 문서 및 원문을 확인했다. 확인일은 2026년 11월 3일이다.

[^dae]: Vincent 외, [Extracting and Composing Robust Features with Denoising Autoencoders](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf), ICML 2008. 손상된 입력에서 원본을 복원하는 학습 관계.
[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html). 하위 모듈·파라미터 등록과 모델 호출.
[^sequential]: PyTorch, [Sequential](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sequential.html). 등록된 모듈을 순서대로 연결하는 동작.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html). 인자, 가중치 구조, 출력 크기.
[^transpose]: PyTorch, [ConvTranspose2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ConvTranspose2d.html). 전치 합성곱의 정의·출력 크기·가중치·output_padding.
[^upsample]: PyTorch, [Upsample](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Upsample.html). 보간을 통한 공간 크기 확대.
[^checkerboard]: Odena, Dumoulin, Olah, [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/), Distill 2016. 겹침과 아티팩트, 확대 후 합성곱 대안.
[^comprehension]: Python, [Displays for lists, sets and dictionaries](https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries). 컴프리헨션의 반복과 변수 범위.
[^mse]: PyTorch, [MSELoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MSELoss.html). `none`, `mean`, `sum`의 집계 범위.
[^flatten]: PyTorch, [flatten](https://docs.pytorch.org/docs/2.15/generated/torch.flatten.html) 및 [mean](https://docs.pytorch.org/docs/2.15/generated/torch.mean.html). 축 합치기와 평균 계산.
[^broadcast]: PyTorch, [Broadcasting semantics](https://docs.pytorch.org/docs/2.15/notes/broadcasting.html). 뒤쪽 축부터 크기를 맞추는 규칙.
[^zero]: PyTorch, [Optimizer.zero_grad](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.zero_grad.html). 기울기 초기화와 `set_to_none`.
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html). 계산 기록·역전파·평가 모드와 자동 미분 모드의 구분.
[^item]: PyTorch, [Tensor.item](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.item.html). 원소 하나인 텐서의 Python 숫자 반환.
[^detach]: PyTorch, [Tensor.detach](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.detach.html). 자동 미분 연결에서 분리한 텐서 반환.
[^split]: PyTorch, [split](https://docs.pytorch.org/docs/2.15/generated/torch.split.html). 지정 축 분할과 마지막 조각, 뷰 반환.
[^cat]: PyTorch, [cat](https://docs.pytorch.org/docs/2.15/generated/torch.cat.html) 및 [stack](https://docs.pytorch.org/docs/2.15/generated/torch.stack.html). 기존 축 연결과 새 축 추가.
[^randomness]: PyTorch, [Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html). 시드와 재현성의 범위.
[^adam]: PyTorch, [Adam](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html). 파라미터·학습률과 옵티마이저 상태.
[^randperm]: PyTorch, [randperm](https://docs.pytorch.org/docs/2.15/generated/torch.randperm.html). 정수 범위의 무작위 순열.
[^randn]: PyTorch, [randn](https://docs.pytorch.org/docs/2.15/generated/torch.randn.html). 표준 정규분포 난수 생성.
[^clamp]: PyTorch, [clamp](https://docs.pytorch.org/docs/2.15/generated/torch.clamp.html). 값의 최소·최대 범위 제한.
[^dataloader]: PyTorch, [Dataset and DataLoader](https://docs.pytorch.org/docs/2.15/data.html). 데이터 읽기와 배치 구성.
[^imshow]: Matplotlib, [imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html). 명암과 표시 범위.
[^threads]: PyTorch, [set_num_threads](https://docs.pytorch.org/docs/2.15/generated/torch.set_num_threads.html). CPU 연산의 스레드 수 설정.
[^next]: Python, [next](https://docs.python.org/3/library/functions.html#next). 반복자 소진과 기본값·StopIteration.
[^pathlib]: Python, [pathlib](https://docs.python.org/3/library/pathlib.html). 현재 작업 폴더와 경로 해석·파일 확인.
[^runpy]: Python, [runpy.run_path](https://docs.python.org/3/library/runpy.html#runpy.run_path). 파일 실행과 전역 이름 딕셔너리 반환.
[^rust]: Rust, [slice::chunks](https://doc.rust-lang.org/std/primitive.slice.html#method.chunks) 및 [Vec::extend_from_slice](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.extend_from_slice). 슬라이스 분할과 벡터 연결.
