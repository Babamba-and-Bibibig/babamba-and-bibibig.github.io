+++
title = "잡음 제거 모델의 결과 읽기: 이미지별 MSE와 잡음 세기 비교"
description = "4채널 오토인코더를 구현하고, 한 배치 학습에서 이미지별 MSE까지 연결한다. 리스트·스칼라·stack·unsqueeze의 모양을 확인한 뒤, 같은 모델의 평가와 잡음 세기별 재학습을 구분하고 23%와 32%의 의미를 정확히 읽는다."
date = 2026-11-03T18:04:22+09:00
slug = "denoising-autoencoder-training-evaluation"

[extra]
katex = true
toc = true
styles = ["denoising-autoencoder-training-evaluation/article.css"]
+++

잡음 제거 모델을 실행해서 이미지가 나왔다. 이제는 모델이 무엇을 개선했는지 확인해야 한다. 전체 평균 오차가 줄었는지, 이미지마다 오차가 줄었는지, 숫자의 획이 유지됐는지는 각각 확인하는 대상이 다르다. 잡음을 더 강하게 넣어 비교할 때도 **같은 모델을 다시 평가했는지, 새로운 모델을 처음부터 학습했는지**에 따라 결과의 의미가 달라진다.

[앞의 오토인코더 실습 글](/denoising-autoencoder-pytorch/)에서는 이미지 배치가 모델을 통과하고 학습되는 흐름을 살펴봤다. 이번에는 4채널 모델을 직접 만든 뒤 **모델 구현 → 한 배치 학습 → 이미지별 오차 → 잡음 세기별 비교 → 결과 해석**을 따라간다. 앞 글을 읽지 않았더라도 필요한 입력과 함수의 역할을 여기서 확인할 수 있도록 구성했다.

[전체 Python 실습 파일](/denoising-autoencoder-training-evaluation/examples-ko.py)에는 아래 모델과 계산 함수, 학습·평가 절차가 들어 있다. 원래 실습에서 사용한 보조 함수들의 본문은 제공되지 않았으므로, 실행 파일의 보조 함수는 이 글을 위해 별도로 구현했다. 글에 인용한 평가 로그와 새 코드의 실행 결과는 구분한다. API는 2026년 11월 3일 공식 문서를 확인했고, 핵심 계산은 PyTorch 2.15.0의 CPU 환경에서 검산했다.

## 1. 입력, 정답, 복원 결과를 먼저 연결한다

<strong>잡음 제거 오토인코더(Denoising Autoencoder)</strong>는 잡음이 있는 이미지를 입력받고, 깨끗한 원본에 가까운 이미지를 계산하도록 학습하는 신경망이다. 입력에서 중간 특징을 계산하는 부분을 **인코더(Encoder)**, 그 특징에서 이미지 형태의 출력을 계산하는 부분을 <strong>디코더(Decoder)</strong>라고 한다.

이 글에서는 세 변수의 역할을 끝까지 유지한다.

| 변수 | 담고 있는 데이터 | 사용하는 곳 |
| --- | --- | --- |
| `clean` | 잡음이 없는 원본 이미지 | 출력과 비교할 정답, 즉 타깃(Target) |
| `noisy` | 원본에 잡음을 넣은 이미지 | 모델에 넣는 입력 |
| `recon` | 모델이 계산한 복원 이미지 | `clean`과 비교할 결과 |

모델은 `noisy`를 받는다. `clean`은 모델 출력과 비교하는 기준으로 남겨 둔다. 두 이미지의 차이를 숫자로 측정한 값이 <strong>손실(Loss)</strong>이다. 여기서는 <strong>평균제곱오차(Mean Squared Error, MSE)</strong>를 사용한다. 같은 위치의 두 픽셀을 빼고, 차이를 제곱한 다음 평균내는 계산이다.

모델에 저장된 학습 대상 숫자는 <strong>가중치(Weight)</strong>와 <strong>편향(Bias)</strong>이다. 둘을 합쳐 <strong>파라미터(Parameter)</strong>라고 부른다. <strong>기울기(Gradient)</strong>는 현재 파라미터를 조금 바꾸면 손실이 어느 방향으로 얼마나 민감하게 변하는지 나타낸다. <strong>역전파(Backpropagation)</strong>가 기울기를 계산하고, <strong>옵티마이저(Optimizer)</strong>가 그 기울기를 이용해 파라미터를 갱신한다.

<figure class="dae-eval-figure">
<img src="/denoising-autoencoder-training-evaluation/model-shapes-ko.svg" alt="잡음 이미지 N장, 채널 1개, 공간 28×28이 인코더를 지나 채널 4개와 공간 14×14인 특징으로 바뀐 뒤 디코더에서 채널 1개, 공간 28×28의 복원 이미지가 된다. 원본 clean은 복원 이미지와 MSE를 계산하는 기준으로 연결된다." loading="lazy">
<figcaption>모델의 입력은 noisy, 손실의 정답은 clean이다. 중간 특징은 입력에서 계산한 값이며, 모델에 저장된 가중치와 구별한다.</figcaption>
</figure>

같은 모델을 다섯 가지 대상으로 나누면 다음과 같다.

| 대상 | 이번 모델에서 확인할 것 |
| --- | --- |
| 입력 | 잡음이 있는 흑백 이미지 배치 |
| 뉴런·층 구조 | 합성곱과 ReLU, 전치 합성곱과 Sigmoid |
| 중간값 | 이미지마다 계산되는 4채널 특징맵 |
| 가중치·편향 | 두 합성곱 층에 저장된 총 105개의 학습 대상 숫자 |
| 최종 출력 | 원본과 같은 크기로 계산한 복원 이미지 |

**모델 생성, 학습, 평가는 서로 다른 작업이다.** 모델 생성은 구조와 초기 파라미터를 준비한다. 학습은 입력과 정답으로 파라미터를 수정한다. 평가는 현재 파라미터가 만든 결과를 측정한다. `model(noisy)`라는 호출만으로 파라미터가 학습되는 것은 아니다.[^module]

## 2. 4채널 모델을 만들고 텐서의 모양을 따라간다

<strong>텐서(Tensor)</strong>는 숫자들과 그 숫자들이 배치된 구조를 함께 다루는 자료형이다. **shape**는 각 축의 크기를 순서대로 표시한다. 이미지 배치에서는 `(N, C, H, W)` 순서를 사용한다.

| 축 번호 | 기호 | 의미 | `(8, 1, 28, 28)`의 크기 | 사용할 수 있는 인덱스 |
| ---: | --- | --- | ---: | --- |
| 0 | `N` | 이미지 개수 | 8 | 0~7 |
| 1 | `C` | 이미지 한 장의 채널 수 | 1 | 0 |
| 2 | `H` | 채널의 높이 | 28 | 0~27 |
| 3 | `W` | 채널의 너비 | 28 | 0~27 |

축 번호 1, 그 축의 크기 4, 그 축에서 선택할 인덱스 0~3은 서로 다른 정보다. 이 글에서 채널 수 1은 흑백 입력이라는 데이터 조건이다. 중간 특징의 채널은 4개이고, 배치 예시는 이미지 8장을 사용한다.

`torch.nn`에는 신경망의 층과 손실을 계산하는 클래스가 들어 있다. `from torch import nn`으로 가져오면 `torch.nn.Conv2d`를 `nn.Conv2d`라고 쓸 수 있다. <strong>클래스(Class)</strong>는 객체를 만드는 정의이고, <strong>객체(Object)</strong>는 그 정의로 실제 생성한 대상이다. 모델 클래스의 `__init__()`에서는 층을 만들고, `forward()`에서는 입력을 계산할 순서를 적는다.

```python
import torch
from torch import nn  # torch.nn의 이름을 nn으로 사용한다.


class StudentDAE(nn.Module):
    def __init__(self):
        super().__init__()  # nn.Module이 층과 파라미터를 관리하도록 초기화한다.
        self.encoder = nn.Conv2d(
            in_channels=1,   # 입력 이미지의 채널 수.
            out_channels=4,  # 인코더가 계산할 특징맵의 채널 수.
            kernel_size=3,   # 공간 커널은 3×3.
            stride=2,        # 커널 시작 위치를 2칸씩 이동한다.
            padding=1,       # 입력의 각 경계에 1칸씩 패딩한다.
            bias=True,       # 출력 채널마다 편향을 사용한다.
        )
        self.decoder = nn.ConvTranspose2d(
            in_channels=4,   # 인코더에서 받은 채널 수.
            out_channels=1,  # 복원할 흑백 이미지의 채널 수.
            kernel_size=4,   # 공간 커널은 4×4.
            stride=2,        # 입력 위치별 출력 영역의 시작점 간격.
            padding=1,       # 출력의 경계를 조정하는 설정.
            bias=True,
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, noisy):
        z = self.encoder(noisy)   # (N,1,28,28) → (N,4,14,14).
        z = self.relu(z)          # 값은 바뀌지만 shape는 유지된다.
        recon = self.decoder(z)   # (N,4,14,14) → (N,1,28,28).
        return self.sigmoid(recon)
```

`ReLU`는 음수를 0으로 바꾸고 양수는 유지한다. `Sigmoid`는 이 문제의 원본 픽셀 범위에 맞춰 출력을 0과 1 사이로 제한한다. 둘 다 텐서의 shape는 바꾸지 않는다. 이러한 함수를 <strong>활성화 함수(Activation Function)</strong>라고 한다. 잡음 제거 능력은 전체 모델의 파라미터를 학습하면서 얻는다.[^activations]

두 층을 생성하는 인자는 <strong>생성자(Constructor)</strong>에 전달한 설정이다. 나중에 `model(noisy)`를 호출할 때 전달하는 인자는 실제 이미지 텐서다. 이때 PyTorch가 모델의 `forward()` 계산을 실행한다.

```python
shape_model = StudentDAE()  # 모양 확인용 모델. 아직 학습 전이다.
shape_input = torch.zeros(8, 1, 28, 28)  # 모양 확인용 입력이다.
shape_output = shape_model(shape_input)
print(tuple(shape_output.shape))  # (8, 1, 28, 28)
```

위 출력은 복원 품질을 확인한 결과가 아니다. 입력 8장을 넣었을 때 출력도 같은 이미지 모양으로 나오는지 확인한 것이다.

### 출력 원소 수와 파라미터 수는 따로 센다

이번 설정에서는 중간 특징 한 장이 `4 × 14 × 14 = 784`개의 값을 갖는다. 입력 한 장도 `1 × 28 × 28 = 784`개다. **공간의 높이와 너비는 줄지만, 채널이 늘어서 전체 원소 수는 같다.** 이 중간 출력을 <strong>잠재 표현(Latent Representation)</strong>이라고 부를 수 있지만, 숫자 개수까지 줄인 구조라고 설명하면 맞지 않는다.

파라미터는 이와 다른 숫자다. 커널에 저장된 가중치와 출력 채널별 편향을 센다. 아래는 `groups=1`인 이번 설정의 계산이다.[^conv][^transpose]

| 층 | 가중치 shape와 축 의미 | 가중치 수 | 편향 수 | 합계 |
| --- | --- | ---: | ---: | ---: |
| `Conv2d` | `(4,1,3,3)` = 출력 채널·입력 채널·커널 높이·너비 | `4×1×3×3=36` | 4 | 40 |
| `ConvTranspose2d` | `(4,1,4,4)` = 입력 채널·출력 채널·커널 높이·너비 | `4×1×4×4=64` | 1 | 65 |
| 전체 | 두 층의 학습 대상 숫자 | 100 | 5 | **105** |

가중치 shape 앞부분이 둘 다 `(4,1)`이지만, 두 층에서 그 축의 의미는 다르다. 커널은 여러 공간 위치에서 같은 가중치를 재사용한다. 따라서 이미지 수나 이미지 공간 크기가 바뀌어도 이 모델의 파라미터 수는 105개로 유지된다. 계산해야 하는 출력값과 실제 연산량까지 105개로 고정된다는 뜻은 아니다.

이름은 `nn.ReLU()`와 `nn.Sigmoid()`처럼 대소문자를 정확히 적는다. 반면 `self.encoder`의 `encoder`는 개발자가 정한 속성 이름이다. 정의와 사용이 일치해야 하며, 일반적인 철자를 사용하면 코드를 읽기 쉽다. 두 합성곱의 `bias` 기본값은 `True`지만, 여기서는 조건을 보이기 위해 명시했다.

## 3. 28 → 14 → 28은 커널을 놓는 위치에서 나온다

### 합성곱에서는 놓을 수 있는 시작 위치를 센다

높이 한 방향부터 보자. 원본의 길이는 28이고 양쪽에 패딩 1칸씩을 추가하면 길이 30이 된다. 패딩까지 포함한 인덱스는 0~29다. 길이 3인 커널의 시작점은 최대 27까지 가능하지만, 이번에는 시작점을 2칸씩 옮긴다.

| 시작 위치 | 커널이 사용하는 위치 | 가능 여부 |
| ---: | --- | --- |
| 0 | 0, 1, 2 | 가능 |
| 2 | 2, 3, 4 | 가능 |
| 26 | 26, 27, 28 | 가능 |
| 다음 후보 28 | 28, 29, 30 | 인덱스 30이 없으므로 불가능 |

실제 시작점은 `0, 2, 4, …, 26`의 14곳이다. 높이 방향의 출력은 14개가 되고, 너비에도 같은 계산을 적용해서 `14×14`가 된다.

```python
starts = list(range(0, 28, 2))  # 시작 0, 끝 경계 28은 제외, 간격 2.
print(starts)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
print(len(starts))  # 14
```

이 위치 계산을 식으로 정리하자. `L_in`은 입력 길이, `L_out`은 출력 길이, `k`는 커널 길이, `s`는 시작점 이동 간격, `p`는 한쪽 패딩 길이다. 커널 원소 사이 간격을 정하는 `dilation`은 기본값 1을 사용한다.

<div class="dae-eval-math">
$$
L_{out}=\left\lfloor\frac{L_{in}+2p-k}{s}\right\rfloor+1
$$
</div>

`L_in + 2p - k`는 커널이 들어갈 수 있는 최대 시작 위치다. 이것을 `s`로 나눈 뒤 내림하면 처음 이후 이동할 수 있는 횟수가 된다. 마지막 `+1`은 시작점 0에서 수행한 첫 계산을 포함한다. <strong>내림(Floor)</strong>은 계산값보다 크지 않은 가장 큰 정수를 선택한다. 이번에는 `floor((28+2−3)/2)+1 = 13+1 = 14`다.[^conv]

### 전치 합성곱에서는 출력 영역이 차지하는 길이를 센다

<strong>전치 합성곱(Transposed Convolution)</strong>은 입력값 하나와 커널의 여러 가중치를 곱하고, 그 결과들을 대응하는 출력 위치에 더하는 방식으로 이해할 수 있다. 여기서 입력값의 **기여값**은 해당 입력값과 해당 가중치의 곱을 뜻한다. 역전파의 기울기를 말하는 것이 아니다.

한 방향에 입력값 `A`, `B`가 있고 커널 길이가 4, 시작점 간격이 2라고 하자. 경계 조정 전에는 `A`의 계산이 출력 위치 0~3에 들어가고, `B`의 계산이 2~5에 들어간다.

| 출력 위치 | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| `A`에서 더하는 값 | `A·w₀` | `A·w₁` | `A·w₂` | `A·w₃` | 없음 | 없음 |
| `B`에서 더하는 값 | 없음 | 없음 | `B·w₀` | `B·w₁` | `B·w₂` | `B·w₃` |

`w₀`부터 `w₃`는 커널의 네 가중치다. 출력 위치 2에서는 `A·w₂ + B·w₀`을 계산한다. **곱해서 더할 항은 8개지만, 서로 다른 출력 위치는 6개**다. 두 영역이 2칸 겹치기 때문이다. 같은 커널 길이 4에서 시작점 간격도 4로 바꾸면 영역은 0~3과 4~7에 놓이므로 겹치지 않는다. 겹침은 커널 길이와 이동 간격의 조합에서 생긴다.

2차원에서도 높이와 너비에 같은 배치를 적용한다. 입력 한 채널의 `2×2` 위치에 `A, B, C, D`가 있다면, 각 값은 자기 위치에 대응하는 `4×4` 출력 영역에 계산 결과를 더한다.

<figure class="dae-eval-figure">
<img src="/denoising-autoencoder-training-evaluation/transpose-2d-ko.svg" alt="입력 A B C D의 4×4 출력 영역이 각각 0,0과 0,2와 2,0과 2,2에서 시작해 겹친다. 전체 6×6 공간에서 가장자리를 한 칸씩 제외한 결과는 4×4이다." loading="lazy">
<figcaption>한 입력 채널에서 생기는 공간적 겹침이다. 실제 디코더는 입력 채널 4개의 계산도 같은 출력 채널의 대응 위치에 합산하고 편향을 더한다.</figcaption>
</figure>

이 그림에서 경계 조정 전 `(2,2)` 위치에는 `A·k₂₂ + B·k₂₀ + C·k₀₂ + D·k₀₀`이 더해진다. `k`의 아래 두 인덱스는 커널 내부의 높이·너비 위치다. 입력 채널들의 값을 합친다고 출력 높이와 너비를 채널 수만큼 곱하지는 않는다.

실제 디코더의 한 방향 입력은 14개다. 인덱스 0~13의 각 입력에 대응하는 출력 영역은 시작점 `0, 2, …, 26`에 놓인다. 마지막 영역은 26~29를 차지한다. 따라서 경계 조정 전 길이는 **첫 영역 4칸 + 이후 13번 이동 × 2칸 = 30칸**이다. 13번 이동해서 26칸 전진한 것이다.

`padding=1`은 이 설정에서 양쪽 경계 1칸씩을 제외하는 것과 같은 크기 효과를 낸다. 길이 30에서 2칸을 제외하면 28이 남는다. 이것은 계산을 설명하는 방법이며, PyTorch가 내부에서 반드시 길이 30의 텐서를 만든 뒤 잘라야 한다는 뜻은 아니다.

추가 출력 크기 조정값인 `output_padding`도 기본값 0을 사용하면 관계는 다음과 같다.[^transpose]

<div class="dae-eval-math">
$$
L_{out}=(L_{in}-1)s+k-2p
$$
</div>

`L_in−1`은 첫 위치 이후의 이동 횟수이고, 여기에 `s`를 곱하면 마지막 영역의 시작점이 된다. 커널 길이 `k`를 더해서 전체 길이를 얻고, 양쪽 경계 조정 `2p`를 뺀다. 이번 값은 `(14−1)×2+4−2×1=28`이다.

현재 인코더는 길이 27과 28을 모두 14로 바꾼다. 디코더는 그 14를 28로 바꾼다. 따라서 `27 → 14 → 28`도 가능하다. 전치 합성곱이 입력의 원래 크기와 정보를 항상 정확하게 되찾는 역함수는 아니다. **출력 shape가 맞는지 확인하는 일과 실제 픽셀이 잘 복원됐는지 평가하는 일은 별도로 필요하다.**

## 4. 한 배치 학습에서 실제로 바뀌는 값을 확인한다

<strong>배치(Batch)</strong>는 한 번의 계산에 묶어서 처리할 이미지들이다. 한 배치를 학습한다는 것은 그 이미지들의 입력과 정답으로 기울기를 계산하고 모델 파라미터를 한 번 갱신한다는 뜻이다.

먼저 `nn.MSELoss()`는 손실을 계산할 객체를 만든다. 그 객체에 예측과 정답을 전달해야 실제 손실값이 나온다.

```python
loss_fn = nn.MSELoss()       # 기본값 reduction="mean"인 손실 객체 생성.
```

그다음 실제 복원 `pred`와 정답 `clean`을 준비해서 `loss_fn(pred, clean)`을 호출하면 손실 텐서를 얻는다. `nn.MSELoss()(pred, clean)`은 객체 생성과 계산을 한 줄로 적은 표현이다. 생성자 괄호와 계산을 위한 호출 괄호의 역할이 다르다. 기본 MSE는 배치의 모든 이미지·채널·픽셀에 대응하는 제곱오차를 평균낸다.[^mse]

```python
def train_one_batch(model, optimizer, noisy, clean):
    model.train()  # 학습 모드 설정. 이 줄 자체가 파라미터를 갱신하지는 않는다.
    optimizer.zero_grad(set_to_none=True)  # 이전 .grad를 초기화한다.

    pred = model(noisy)  # 현재 파라미터로 복원 이미지를 계산한다.
    if pred.shape != clean.shape:
        raise ValueError("복원 이미지와 정답의 shape가 다릅니다.")

    loss = nn.MSELoss()(pred, clean)  # 아직 갱신하기 전의 손실.
    loss.backward()                  # 각 파라미터에 대한 기울기 계산.
    optimizer.step()                 # 기울기를 사용해 파라미터 갱신.
    return loss.item()               # 앞서 계산한 손실을 Python 숫자로 반환.
```

함수의 <strong>매개변수(Parameter)</strong>는 `model`, `optimizer`처럼 전달받을 값을 가리키는 이름이다. 모델의 학습 파라미터라는 말과 문맥을 구별하자. 여기서 `optimizer`는 이미 만들어서 전달한 객체다. 그 객체의 기능인 <strong>메서드(Method)</strong>를 실행하므로 `optimizer.step()`이라고 쓴다. `optimizer().step()`처럼 옵티마이저 객체 자체를 함수처럼 호출하지 않는다.

### 기울기 초기화는 학습한 가중치를 지우지 않는다

파라미터의 `.grad`에는 `backward()`가 계산한 기울기가 저장된다. PyTorch는 기존 `.grad`에 기울기를 누적하므로, 이번 배치의 기울기만 사용하려면 이전 기울기를 초기화해야 한다. `set_to_none=True`는 기울기 저장값을 `None`으로 되돌리는 방식이다. 가중치와 편향은 지금까지 학습한 값을 유지한다.[^zero-grad]

| 호출 | 바뀌는 대상 | 그때 아직 하지 않은 일 |
| --- | --- | --- |
| `zero_grad()` | 이전 기울기 저장값 | 새 배치의 기울기 계산 |
| `model(noisy)` | 새로 계산한 복원 텐서 | 파라미터 갱신 |
| `loss.backward()` | 파라미터별 `.grad` | 옵티마이저에 의한 갱신 |
| `optimizer.step()` | 모델의 가중치와 편향, 옵티마이저 상태 | 갱신 후 예측과 손실의 재계산 |

`backward → zero_grad → step` 순서로 실행하면, 갱신에 사용할 기울기를 중간에 지우게 된다. `step` 뒤에 초기화하는 방식도 첫 기울기 상태를 올바르게 관리하면 사용할 수 있지만, 이 글은 매 배치 시작에 초기화하는 한 가지 순서를 따른다.

### `step()` 뒤에 반환해도 손실은 갱신 전 값이다

`loss`는 `pred`와 `clean`으로 이미 계산한 결과다. 그다음 `step()`이 모델의 파라미터를 바꿔도 기존 `pred`와 `loss`를 자동으로 다시 계산하지 않는다. 따라서 **`return loss.item()`은 갱신 전에 측정한 손실을 반환한다.**

갱신 후 손실을 확인하려면 새 파라미터로 순전파와 손실 계산을 다시 해야 한다. 다음은 측정 시점만 확인하는 인공 이미지 세 장의 예시다. `rand()`는 0~1 범위의 균일한 난수를 만들고, `randn_like()`는 입력과 같은 shape의 표준 정규 난수를 만든다. 실제 손글씨 복원 품질을 측정하는 실험은 아니다.

```python
torch.manual_seed(7)
step_model = StudentDAE().cpu()  # 측정 시점 검산을 위한 별도 모델.
step_optimizer = torch.optim.Adam(step_model.parameters(), lr=0.01)
demo_clean = torch.rand(3, 1, 28, 28)  # 인공 원본 세 장.
demo_noisy = (demo_clean + 0.25 * torch.randn_like(demo_clean)).clamp(0, 1)

before = train_one_batch(step_model, step_optimizer, demo_noisy, demo_clean)
with torch.no_grad():  # 비교용 재계산에는 역전파 기록이 필요 없다.
    demo_recon = step_model(demo_noisy)  # 갱신한 가중치로 새로 계산한다.
    after = nn.MSELoss()(demo_recon, demo_clean).item()
print("갱신 전 측정:", before, "갱신 후 재측정:", after)
```

한 번의 갱신으로 손실이 항상 작아진다고 보장하지는 않는다. `item()`은 원소 하나인 텐서에서 Python 숫자를 꺼낸다. 이 손실 텐서의 shape는 `()`이고, 꺼낸 값은 `float`다. 먼저 `loss = loss.item()`으로 바꾼 뒤 `loss.backward()`를 호출하면 Python 숫자에 `backward()`가 없어서 실행할 수 없다.[^item]

## 5. 이미지마다 MSE 하나를 남긴다

전체 평균 하나만 알면 어떤 이미지가 좋아졌고 어떤 이미지가 나빠졌는지 구별하기 어렵다. 이미지별 비교를 하려면 이미지가 `N`장일 때 MSE도 `N`개 남겨야 한다.

먼저 비교할 위치를 정하자. `n`은 이미지 번호, `c`는 채널 번호, `h`와 `w`는 높이·너비 위치다. `recon[n,c,h,w]`와 `clean[n,c,h,w]`를 뺀 값이 그 위치의 오차다. 같은 shape의 텐서는 다차원 상태에서 그대로 뺄 수 있다. 뺄셈을 하기 위해 반드시 `flatten()`부터 해야 하는 것은 아니다.

`(recon − clean).square()`는 각 오차를 제곱하고 입력 shape를 유지한다. 여기까지는 각 픽셀의 제곱오차가 남아 있다. 그 뒤에 **어느 범위를 평균낼지** 정해야 한다.

| 계산할 범위 | 결과에 남는 값 | 결과 shape |
| --- | --- | --- |
| 이미지 한 장의 채널·높이·너비 전체 | 그 이미지의 MSE 하나 | `()` |
| 위 계산을 이미지 N장에 각각 적용 | 이미지별 MSE N개 | `(N,)` |
| 배치의 모든 대응 원소 전체 | 전체 평균 MSE 하나 | `()` |

먼저 반복문으로 한 장씩 처리해 보자. `zip()`은 복원 이미지와 정답 이미지를 같은 순서끼리 묶는다. 두 입력의 길이가 다르면 짧은 쪽에서 끝나므로, 먼저 shape가 일치하는지 확인한다.[^python-builtins]

```python
def per_image_mse(reconstruction, clean):
    if reconstruction.shape != clean.shape:
        raise ValueError("두 배치의 shape가 같아야 합니다.")
    if reconstruction.ndim != 4 or reconstruction.shape[0] == 0:
        raise ValueError("이미지가 있는 (N,C,H,W) 텐서가 필요합니다.")

    losses = []  # Python 리스트. 이미지별 손실 텐서를 하나씩 보관한다.
    for recon_img, clean_img in zip(reconstruction, clean):
        # 한 장은 (C,H,W). 각 이미지 안의 값만 1차원으로 펼친다.
        error = recon_img.flatten() - clean_img.flatten()
        loss = error.square().mean()  # 이 이미지의 MSE. shape는 ().
        losses.append(loss)          # 스칼라 텐서를 리스트 끝에 추가한다.

    return torch.stack(losses)  # 스칼라 N개를 하나의 (N,) 텐서로 결합한다.
```

이 함수의 반환값은 평균 하나가 아니다. 이미지 순서에 맞는 MSE 목록을 하나의 텐서로 만든 것이다. 앞 절의 인공 이미지 세 장을 이어서 사용하면, 이미지별 값 세 개를 받은 다음 전체 평균을 따로 계산할 수 있다.

```python
each = per_image_mse(demo_recon, demo_clean)  # 세 장의 MSE, shape (3,).
overall = each.mean()                       # MSE 세 개의 평균, shape ().
print(tuple(each.shape), tuple(overall.shape))  # (3,) ()
```

`losses`는 Python 리스트라서 `losses.sum()` 메서드가 없다. 더 중요한 점은 리스트의 값을 어떻게 합치더라도, 평균 하나로 반환하면 이미지별 정보를 잃는다는 것이다. 이 함수의 목적에 맞는 반환은 `torch.stack(losses)`다.

## 6. 리스트, 스칼라 텐서, `(N,)`을 따로 읽는다

<strong>스칼라(Scalar)</strong>는 값 하나를 뜻한다. `torch.tensor(0.1)`은 축이 없는 0차원 텐서이고, shape는 `()`다. `torch.tensor([0.1])`은 크기 1인 축이 있으므로 shape가 `(1,)`이다. 둘 다 원소 하나를 담지만 축의 수가 다르다.

다음 숫자는 모양을 설명하기 위한 예시이며 학습 성능값이 아니다.

```python
losses = [torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3)]
print(len(losses))               # 3: 리스트에 들어 있는 텐서 개수.
print(tuple(losses[0].shape))    # (): 첫 번째 텐서는 스칼라.

each = torch.stack(losses)       # 같은 shape의 텐서를 새 축으로 결합.
print(each)                     # tensor([0.1000, 0.2000, 0.3000])
print(tuple(each.shape))        # (3,)

column = each.unsqueeze(1)       # 크기가 1인 새 축을 1번 위치에 넣는다.
print(tuple(column.shape))      # (3, 1)
print(tuple(each.shape))        # (3,): each 자체에 다시 대입하지 않았다.
```

<figure class="dae-eval-figure">
<img src="/denoising-autoencoder-training-evaluation/scalar-stack-ko.svg" alt="shape가 없는 Python 리스트에 shape ()인 스칼라 텐서 세 개가 들어 있다. stack은 (3,) 텐서를 만들고 unsqueeze(1)은 (3,1)로 바꾼다. 이미지별 MSE는 별도로 (8,1,28,28)을 (8,784)로 펼쳐 평균한 (8,) 결과를 만든다." loading="lazy">
<figcaption>리스트의 길이, 리스트 안 텐서의 shape, 결합한 텐서의 shape를 순서대로 확인한다. stack이 추가하는 축의 크기는 결합한 텐서 개수다.</figcaption>
</figure>

`losses` 자체에는 텐서의 `.shape` 속성이 없다. 안에 들어 있는 각 텐서가 `()`이고, 그 텐서들을 `stack`해서 `(3,)`이 된 것이다. 이미지 8장에 적용하면 같은 원리로 `(8,)`이 된다.[^stack]

| 표현 | 뜻 | 인덱스로 선택하는 예 |
| --- | --- | --- |
| `()` | 축이 없는 텐서 | 숫자를 읽을 때 `.item()` |
| `(8,)` | 크기 8인 축 하나 | `each[0]`부터 `each[7]` |
| `(8,1)` | 크기 8, 크기 1인 축 두 개 | `column[0,0]`부터 `column[7,0]` |

`(8,)`의 쉼표는 Python에서 원소 하나인 <strong>튜플(Tuple)</strong>을 표현하는 문법이다. `(8)`은 숫자 8을 괄호로 감싼 것이고, `(8,)`은 숫자 8이 들어 있는 튜플이다. 쉼표가 두 번째 축을 생략했다는 뜻은 아니다. `torch.Size([8])`과 `tuple(each.shape)`로 출력한 `(8,)`은 같은 shape를 표시한다.[^tuples]

스칼라 텐서 `()` 여덟 개를 `stack`하면 `(8,)`이다. 길이 1인 텐서 `(1,)` 여덟 개를 `stack`하면 `(8,1)`이다. 입력 텐서의 shape부터 확인하면 두 결과를 구별할 수 있다.

## 7. `stack`과 `unsqueeze`는 무엇을 새 축에 넣는가?

`stack`은 **여러 텐서**를 받아 새 축으로 결합한다. `unsqueeze`는 **텐서 하나**에 크기 1인 축을 추가한다. 둘 다 입력 텐서보다 결과의 축 수가 하나 많아지지만, 결합하는 대상과 새 축의 크기가 다르다. `dim`은 새 축을 넣을 위치다.[^stack][^unsqueeze]

```python
a = torch.tensor([10, 20, 30])  # shape (3,), 원래 위치는 0~2.
b = torch.tensor([40, 50, 60])  # shape (3,), a와 같은 모양.

x = torch.stack([a, b], dim=0)
print(x.tolist())  # [[10, 20, 30], [40, 50, 60]]
print(tuple(x.shape))  # (2, 3)

y = torch.stack([a, b], dim=1)
print(y.tolist())  # [[10, 40], [20, 50], [30, 60]]
print(tuple(y.shape))  # (3, 2)

u = a.unsqueeze(0)
print(u.tolist())  # [[10, 20, 30]]
print(tuple(u.shape))  # (1, 3)

v = a.unsqueeze(1)
print(v.tolist())  # [[10], [20], [30]]
print(tuple(v.shape))  # (3, 1)
```

`tolist()`는 텐서 값을 Python 리스트 형태로 보여준다. 위 결과에서 `(2,3)`의 첫 축은 `a`와 `b`라는 두 묶음을 구분한다. `(3,2)`의 첫 축은 원래의 세 위치를 구분하고, 두 번째 축은 그 위치에 있는 `a`의 값과 `b`의 값을 구분한다. 예를 들어 `x[1,2]`와 `y[2,1]`은 모두 `b`의 마지막 값 60이다.

`a.unsqueeze(1)`은 `b`를 사용하지 않는다. `a`의 숫자 3개를 유지하면서 크기 1인 축을 추가할 뿐이다. 반환 텐서는 원래 데이터 저장공간을 공유하며, 기존 `a`의 shape를 바꾸려면 `a = a.unsqueeze(1)`처럼 새 결과를 다시 대입해야 한다.

## 8. 이미지 축을 남기면 반복문 없이 같은 MSE를 구한다

반복문에서는 한 장의 픽셀들을 평균내는 작업을 이미지 수만큼 반복했다. 배치 전체를 한 번에 계산할 때도 남길 정보는 같다. **이미지 번호를 구분하는 0번 축은 유지하고, 한 이미지 안의 채널·높이·너비만 평균**내면 된다.

```python
def per_image_mse_fast(reconstruction, clean):
    if reconstruction.shape != clean.shape:
        raise ValueError("두 배치의 shape가 같아야 합니다.")
    if reconstruction.ndim != 4 or reconstruction.shape[0] == 0:
        raise ValueError("이미지가 있는 (N,C,H,W) 텐서가 필요합니다.")

    squared_error = (reconstruction - clean).square()  # (N,C,H,W).
    pixels = squared_error.flatten(start_dim=1)        # (N,C*H*W).
    return pixels.mean(dim=1)                          # (N,).
```

`flatten(start_dim=1)`은 1번 축부터 마지막 축까지 합친다. 이번 배치 `(8,1,28,28)`은 `(8,784)`가 된다. 이미지 0의 제곱오차 784개, 이미지 1의 제곱오차 784개처럼 각 이미지의 묶음이 남는다. 그다음 `mean(dim=1)`이 이미지마다 784개 값을 평균내서 `(8,)`을 만든다. 기본값에서는 평균낸 축을 결과에서 없앤다.[^flatten][^mean]

다음 세 줄의 결과를 나란히 보면 어떤 정보를 유지하는지 분명해진다.

| 코드 | `(8,1,28,28)` 제곱오차에 적용한 결과 | 남는 정보 |
| --- | --- | --- |
| `error.flatten(1).mean(1)` | `(8,)` | 각 이미지의 MSE |
| `error.mean(dim=(1,2,3))` | `(8,)` | 위와 같은 범위의 평균 |
| `error.flatten().mean()` | `()` | 이미지 구분까지 없앤 전체 평균 |

여기서 `error`는 픽셀별 제곱오차 텐서다. 인자 없는 `flatten()`은 0번 이미지 축도 합쳐서 `(6272,)`를 만든다. `6272`는 이미지 8장에 픽셀 784개씩 있는 전체 원소 수다. 이것을 평균내면 이미지별 MSE를 다시 알아낼 수 없다.

같은 크기의 이미지로 만든 배치라면 이미지별 평균의 평균과 전체 원소의 평균은 수학적으로 같다. 컴퓨터에서는 합산 순서에 따른 작은 부동소수점 차이가 생길 수 있으므로, 허용 오차 안에서 같은지 비교하는 `torch.allclose()`를 사용한다.

```python
# 두 함수의 동작 확인용 값이다. 실제 복원 성능을 나타내지 않는다.
clean_check = torch.zeros(8, 1, 28, 28)
recon_check = torch.full_like(clean_check, 0.2)  # 같은 shape를 0.2로 채운다.

loop_each = per_image_mse(recon_check, clean_check)
fast_each = per_image_mse_fast(recon_check, clean_check)
whole = nn.MSELoss()(recon_check, clean_check)

assert torch.allclose(loop_each, fast_each)
assert torch.allclose(fast_each.mean(), whole)
print(tuple(fast_each.shape))  # (8,)
```

각각의 결과가 같은 것은 평균을 낸 범위가 같기 때문이다. `fast_each`에는 여전히 이미지별 값이 남아 있으므로, 이후 이미지마다 개선 여부를 비교할 수 있다.

## 9. 학습한 모델을 입력 기준선과 비교한다

<strong>기준선(Baseline)</strong>은 모델을 썼을 때의 이득을 비교할 기준이다. 잡음 제거에서는 잡음 입력을 아무 처리 없이 그대로 쓰는 경우를 기준선으로 둘 수 있다. 그래서 두 오차를 계산한다.

| 비교 | 계산 | 답하는 질문 |
| --- | --- | --- |
| 입력 오차 | `noisy`와 `clean`의 MSE | 복원하기 전에는 원본과 얼마나 달랐는가? |
| 복원 오차 | `recon`과 같은 `clean`의 MSE | 모델을 통과한 뒤에는 원본과 얼마나 달랐는가? |

여기서는 **이미 학습한 모델 하나를 평가한다.** 이 절의 `model`에는 훈련 데이터로 학습을 마친 객체를 넣는다. 앞의 `shape_model`은 모양 확인용, `step_model`은 갱신 시점 검산용이므로 이 평가의 학습 완료 모델과 구별한다. 실제 데이터를 사용해 모델을 학습하고 평가하는 전체 호출은 17절에 있다. 평가 함수 안에서는 새 모델을 만들거나 `fit()`으로 다시 학습하지 않는다. `predict_fixed()`는 PyTorch 내장 함수가 아니라, 전달한 입력을 학습 없이 처리하도록 별도로 정의할 함수 이름이다.

평가 코드에 자주 함께 나오는 두 기능도 구분해야 한다.[^module][^no-grad]

| 코드 | 역할 | 이것만으로 하지 않는 일 |
| --- | --- | --- |
| `model.eval()` | Dropout, BatchNorm처럼 모드에 따라 달라지는 층의 평가 동작 설정 | 기울기 추적 중단 |
| `torch.no_grad()` | 블록 안 계산의 역전파용 기록 중단 | 모델의 평가 모드 설정 |

이번 모델에는 Dropout이나 BatchNorm이 없다. 따라서 해당 층 때문에 학습·평가 출력이 달라지는 상황은 없지만, 두 기능의 역할은 여전히 다르다. `eval()`이 파라미터를 자동으로 동결하거나 `no_grad()`가 모델 구조를 바꾸는 것은 아니다.

한 번에 처리하기 어려운 이미지 배치는 `split()`으로 나눌 수 있다. 아래 함수의 `split(batch_size, dim=0)`은 이미지 축에서 최대 `batch_size`장씩 나누고, `cat(..., dim=0)`은 결과를 원래 순서대로 다시 잇는다. 함수는 주어진 `noisy`만 사용하며 잡음을 새로 만들지 않는다.

```python
def predict_fixed(model, noisy, batch_size=256):
    if noisy.ndim != 4 or noisy.shape[0] == 0:
        raise ValueError("이미지가 있는 (N,C,H,W) 텐서가 필요합니다.")
    was_training = model.training  # 호출 전에 사용하던 모드를 기억한다.
    model.eval()
    try:
        with torch.no_grad():
            parts = []
            for batch in noisy.split(batch_size, dim=0):
                parts.append(model(batch))
            return torch.cat(parts, dim=0)
    finally:
        model.train(was_training)  # 호출 전 모드로 복구한다. 학습 실행이 아니다.
```

`try` 안에서 결과를 반환하더라도 `finally`에 있는 모드 복구는 실행된다. 모델과 입력은 같은 장치에 있어야 한다. 이 글의 전체 실습 파일은 모델을 CPU에 만들고, 입력도 CPU의 `float32` 텐서인지 검사한다.

다음은 학습한 `model`, 평가용 `noisy`와 그 원본 `clean`이 준비된 뒤 실행하는 코드다. 두 배치는 같은 이미지 순서로 대응해야 한다.

```python
recon = predict_fixed(model, noisy)  # 학습 없이 현재 모델로 복원한다.
base_each = per_image_mse_fast(noisy, clean)  # (N,): 이미지별 입력 오차.
rest_each = per_image_mse_fast(recon, clean)  # (N,): 이미지별 복원 오차.

delta = base_each - rest_each  # 같은 이미지의 입력 MSE - 복원 MSE.
improved = int((delta > 0).sum().item())
worsened = int((delta < 0).sum().item())
tied = int((delta == 0).sum().item())

print("이미지별 MSE 모양:", tuple(rest_each.shape))
print(f"입력 MSE={base_each.mean().item():.4f}, "
      f"복원 MSE={rest_each.mean().item():.4f}")
print("개선 / 악화 / 동률:", improved, worsened, tied)
```

`delta`는 차이를 나타내는 이름이다. `delta[i] > 0`이면 이미지 `i`의 복원 오차가 입력 오차보다 작다. `(delta > 0)`은 이미지마다 참·거짓을 담은 텐서이고, 이것을 `sum()`하면 참인 이미지 수가 나온다. **오차 감소량을 합산한 값이 아니라 개선된 이미지의 개수**다.

`delta == 0`은 계산된 값이 정확히 같은지를 본다. 작은 수치 차이까지 동률로 다룰 목적이라면 별도로 허용 오차를 정해야 한다. `:.4f`는 화면에 소수점 네 자리로 표시하는 형식이며, 텐서의 내부 값을 네 자리로 바꾸지 않는다. `recon`에 계산 결과를 바로 대입할 수 있으므로 앞에서 `recon = None`을 적을 필요도 없다.

## 10. 이미지 10,000장의 평가 로그에서 읽을 수 있는 것

다음은 이 글의 바탕이 된 실습 기록에 담긴 출력이다. 원본 이미지와 기존 학습 모델이 함께 제공된 것은 아니므로, 아래 숫자를 새 실행 파일로 재현했다고 주장하지 않는다. **이미 있는 평가 로그를 해석하는 예시**로 읽는다.

```text
이미지별 MSE 모양: (10000,)
입력 MSE=0.0327, 복원 MSE=0.0078
개선 / 악화 / 동률: 10000 0 0
```

`(10000,)`은 이미지 10,000장 각각에 MSE 하나가 있다는 뜻이다. 입력 평균은 0.0327이고 복원 평균은 0.0078이므로, 이 기록에서는 복원 후 전체 평균 오차가 줄었다. 마지막 줄은 평균 하나뿐 아니라 **평가한 이미지마다 계산한 MSE도 모두 줄었다**는 의미다.

그러나 한 이미지의 MSE는 그 이미지 안의 많은 픽셀을 평균한 값이다. 일부 픽셀의 오차가 커지더라도 다른 픽셀들의 오차가 충분히 줄면 평균은 작아질 수 있다. 따라서 이 로그만으로 모든 픽셀과 모든 획이 개선됐다고 말할 수는 없다.

또한 이 모델이 숫자를 몇 번으로 분류했는지는 여기서 측정하지 않았다. 이미지의 픽셀 오차가 줄었다는 결과와 숫자 인식 정확도가 높아졌다는 결과는 다르다. 숫자 인식까지 확인하려면 분류 결과를 별도로 평가해야 한다.

## 11. 잡음 세기를 바꿀 때는 먼저 실험의 질문을 정한다

<strong>시그마(σ, Sigma)</strong>는 여기서 잡음의 세기를 조절하는 값이다. 가우시안 잡음을 더하는 구현에서는 더할 잡음의 <strong>표준편차(Standard Deviation)</strong>를 조절한다. 표준편차는 값들이 평균 주변에서 얼마나 퍼지는지를 나타낸다. σ가 커지면 더하는 잡음값들의 흔들림도 커진다. 구체적인 생성 방법은 다음 절에서 직접 정의한다.

잡음 세기 0.25와 0.50을 비교하는 방법은 두 가지다.

| 구분 | 실험 A: 같은 모델 평가 | 실험 B: 잡음 세기마다 새 모델 학습 |
| --- | --- | --- |
| 먼저 정할 질문 | 지금 가진 모델은 다른 잡음에도 대응하는가? | 각 잡음 조건에 맞춰 학습하면 어떤 결과가 나오는가? |
| 모델 | 이미 학습한 같은 모델 유지 | 각 조건마다 초기 모델을 새로 생성 |
| 학습 | 비교 루프 안에서 하지 않음 | 각 조건마다 처음부터 실행 |
| 바뀌는 것 | 평가 입력의 잡음 세기 | 학습 잡음, 학습 후 모델, 평가 잡음 |
| 비교 루프의 핵심 | 잡음 생성 → 예측 → 오차 측정 | 새 모델·옵티마이저 → 학습 → 잡음 생성 → 예측 → 오차 측정 |

<figure class="dae-eval-figure">
<img src="/denoising-autoencoder-training-evaluation/experiment-protocols-ko.svg" alt="실험 A는 한 번 학습한 모델 하나를 고정하고 잡음 세기 0.25와 0.50 입력을 평가한다. 실험 B는 각 잡음 세기에서 모델과 옵티마이저를 새로 만들어 학습한 뒤 평가한다." loading="lazy">
<figcaption>이 글의 보완 구현에서 사용하는 비교 조건이다. 두 실험 모두 같은 검증 원본과 같은 기본 잡음 ε를 사용하지만, 실험 B에서는 학습이 각각 새로 진행된다.</figcaption>
</figure>

**반복문 안에 `StudentDAE()`와 `fit()`이 있다면 그 반복문은 단순 평가가 아니다.** 기존에 학습한 모델과 별개로, 각 조건에 사용할 새 모델을 만들고 다시 학습하는 실험이다. 모델 생성과 학습 호출을 먼저 찾으면 코드가 답하려는 질문을 구별할 수 있다.

원래 실습의 잡음 비교는 실험 B에 해당한다. 따라서 그 결과를 같은 모델의 평가 잡음만 바꿔 얻은 결과로 해석하면 안 된다.

## 12. 학습과 검증에서 사용할 잡음을 명확하게 정의한다

`fit`, `predict_fixed`, `make_val_noisy`는 PyTorch의 내장 API가 아니다. 작성자가 정의한 함수 이름이다. 이름만 보고 실제 학습 횟수, 난수 고정 방식, 데이터 분리 여부를 확정할 수 없다. 이 글에서는 전체 실행 파일에 이 함수들의 동작을 직접 정의했다.

먼저 학습에 쓸 깨끗한 이미지 `clean_train`과, 평가에 쓸 깨끗한 이미지 `clean_val`을 서로 겹치지 않게 준비한다. 둘 다 `(N,1,28,28)`의 CPU `float32` 텐서이고, 픽셀값은 0~1 범위를 사용한다. `float32`는 숫자 하나를 32비트 실수로 저장하는 자료형이다. 이름에 `val`을 붙이는 것만으로 데이터가 분리되지는 않는다. 원본 이미지 ID나 데이터 분할 기록으로 확인해야 한다.

### 같은 검증 이미지에는 같은 기본 잡음을 사용한다

`torch.randn()`은 표준 정규분포에서 난수를 만든다. 이 분포의 평균은 0, 표준편차는 1이다. 여기서 만든 텐서를 `epsilon`, 기호로는 `ε`라고 부르자. `sigma * epsilon`이 원본에 더할 잡음이다. 그런 다음 `clamp(0,1)`로 0보다 작은 값은 0, 1보다 큰 값은 1로 제한한다.[^noise]

```python
def make_val_noisy(clean, sigma, seed=9026):
    # 이 글의 CPU·float32 입력을 위한 독립 구현이다.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epsilon = torch.randn(
        clean.shape,        # 원본의 모든 픽셀 위치에 대응하는 잡음.
        dtype=clean.dtype,   # 원본과 같은 실수 자료형.
        generator=generator,
    )
    return (clean + sigma * epsilon).clamp(0.0, 1.0)
```

<strong>난수 생성기(Generator)</strong>는 난수들을 만들어 내는 상태를 관리한다. <strong>시드(Seed)</strong>는 같은 시작 조건에서 난수 순서를 다시 만들기 위한 설정이다. 이 함수는 호출할 때마다 별도 생성기를 같은 시드로 시작하므로, 같은 환경과 shape에서는 같은 `epsilon`을 다시 얻는다. σ만 바꾸면 동일한 기본 잡음에 다른 배율을 적용한다.

σ가 0.5라고 해서 픽셀 절반만 망가지는 것은 아니다. 또한 `clamp`가 범위 밖의 값을 잘라낸 뒤에는 실제 `noisy−clean`의 분포가 달라진다. 따라서 입력 MSE가 반드시 `sigma**2`이거나, 실제 입력 오차의 표준편차가 정확히 σ라고 단정하지 않는다.

### `fit()`은 한 배치 학습을 반복하는 함수다

<strong>에포크(Epoch)</strong>는 훈련 데이터 전체를 한 번 사용하는 단위다. 이미지가 5,000장이고 최대 배치 크기가 128이면, 한 에포크에서 128장짜리 배치 39개와 마지막 8장짜리 배치 하나를 처리한다. 배치마다 갱신하므로 한 에포크의 갱신은 40번이다. 이 조건에서 3에포크는 120번의 갱신이다. 학습을 세 번의 `step()`만 수행한다는 뜻이 아니다.

다음은 앞에서 만든 `train_one_batch()`를 반복하는 핵심 구현이다. 생성한 난수의 흐름을 분명히 하기 위해 이미지 순서를 섞는 생성기와 잡음 생성기를 따로 둔다. 다운로드 파일에는 입력 검사도 포함되어 있다.

```python
def fit(model, optimizer, clean_train, *, sigma,
        epochs=3, batch_size=128, seed=2026, verbose=True):
    order_rng = torch.Generator(device="cpu").manual_seed(seed)
    noise_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    history = []
    n_images = len(clean_train)

    for epoch in range(1, epochs + 1):  # 1부터 epochs까지. 끝 경계는 제외된다.
        # 0~n_images-1의 이미지 인덱스를 무작위 순서로 배열한다.
        order = torch.randperm(n_images, generator=order_rng)
        weighted_loss = 0.0

        for start in range(0, n_images, batch_size):
            ids = order[start:start + batch_size]  # 마지막은 더 작을 수 있다.
            clean = clean_train[ids]
            epsilon = torch.randn(
                clean.shape, dtype=clean.dtype, generator=noise_rng,
            )
            noisy = (clean + sigma * epsilon).clamp(0.0, 1.0)
            loss = train_one_batch(model, optimizer, noisy, clean)
            weighted_loss += loss * len(clean)  # 배치에 들어 있던 이미지 수 반영.

        epoch_loss = weighted_loss / n_images
        history.append(epoch_loss)
        if verbose:  # False여도 위의 학습은 전부 실행된다.
            print(f"epoch={epoch}, train_mse={epoch_loss:.6f}")

    return history
```

매개변수 목록의 `*` 뒤에 있는 설정은 `sigma=0.25`처럼 이름을 붙여 전달한다. `torch.randperm()`은 이미지 인덱스를 중복 없이 섞는다. `order[start:start+batch_size]`는 시작 위치를 포함하고 끝 경계를 제외하며, 끝을 넘어가는 마지막 슬라이스는 남은 원소만 가져온다.

마지막 배치가 작을 수 있으므로 배치 평균을 그대로 평균내지 않고 이미지 수를 반영했다. 모든 이미지의 크기가 같은 이 실습에서는 `배치 평균 × 배치 이미지 수`를 합산한 뒤 전체 이미지 수로 나누면 된다. 다만 이 기록은 **여러 갱신 시점에서 측정한 배치 손실의 평균**이다. 에포크 마지막의 모델 하나를 고정해서 평가한 검증 손실과는 측정 시점이 다르다.

`fit()`은 전달받은 모델의 파라미터를 직접 갱신한다. 그래서 반환된 `history`를 변수에 받지 않아도 모델의 학습은 진행된다. `verbose=False`는 이 구현에서 로그를 숨길 뿐, 학습을 생략하지 않는다. 원래 보조 함수에 있던 동일한 옵션의 정확한 동작은 원래 함수 본문에서 확인해야 한다.

## 13. 같은 모델 평가와 새 모델 재학습을 코드로 나눈다

### 실험 A: 이미 학습한 모델을 그대로 평가한다

다음의 `model`은 훈련 데이터로 학습을 마친 객체다. 반복문 안에서는 새 모델과 옵티마이저를 만들지 않는다. 여기서 쓰는 `SIGMAS`는 비교할 잡음 세기 두 개를 담은 리스트다.

```python
SIGMAS = [0.25, 0.50]

for sigma in SIGMAS:
    val_noisy = make_val_noisy(clean_val, sigma, seed=9026)
    recon = predict_fixed(model, val_noisy)
    base = per_image_mse_fast(val_noisy, clean_val).mean().item()
    rest = per_image_mse_fast(recon, clean_val).mean().item()
    print(sigma, base, rest)
```

이 결과는 **같은 학습 결과를 가진 모델이 서로 다른 잡음 입력에 어떻게 대응하는지** 보여준다. 평가 중 파라미터를 갱신하지 않는다.

### 실험 B: 잡음 세기마다 처음부터 학습한다

이제 목적을 바꾼다. 각 잡음 조건에서 별도로 학습한 모델들의 결과를 비교한다. σ가 바뀔 때마다 새로운 학습을 시작한다는 점을 먼저 확인한 뒤 코드를 읽자.

```python
SEED = 2026  # 이 실행 예제에서 선택한 초기화·훈련 난수 설정.
rows = []

for sigma in SIGMAS:
    torch.manual_seed(SEED)  # 같은 환경에서 같은 초기 가중치를 맞춘다.
    trial = StudentDAE().cpu()  # 새 모델 생성. 기존 model을 가져오지 않는다.
    trial_optimizer = torch.optim.Adam(
        trial.parameters(),  # 이번 trial의 가중치와 편향에 연결한다.
        lr=0.01,             # 학습률: 갱신 크기에 영향을 주는 설정.
    )

    fit(
        trial,
        trial_optimizer,
        clean_train,
        sigma=sigma,      # 왼쪽은 매개변수 이름, 오른쪽은 현재 반복값.
        epochs=3,         # 이 글의 구현에서는 학습 횟수를 직접 명시한다.
        batch_size=128,
        seed=SEED,
        verbose=False,    # 학습은 수행하고 로그만 숨긴다.
    )

    val_noisy = make_val_noisy(clean_val, sigma, seed=9026)
    recon = predict_fixed(trial, val_noisy)  # 방금 학습한 trial을 평가한다.
    base_each = per_image_mse_fast(val_noisy, clean_val)
    rest_each = per_image_mse_fast(recon, clean_val)
    rows.append((
        sigma,
        base_each.mean().item(),
        rest_each.mean().item(),
        int((base_each > rest_each).sum().item()),
    ))
```

이 코드의 네 가지 연결을 구분하자.

1. **`manual_seed()`는 초기 조건을 맞춘다.** 같은 환경에서 같은 시드를 설정한 직후 같은 모델을 만들면 같은 초기 파라미터를 얻도록 맞출 수 있다. 가중치를 고정해서 학습되지 못하게 하는 기능은 아니다.
2. **`StudentDAE().cpu()`는 새 모델을 생성하고 CPU에 둔다.** `.cpu()`는 기존 학습 결과를 불러오는 호출이 아니다. Mac에서도 이 코드는 GPU가 아니라 CPU를 사용한다. 입력 데이터도 CPU에 있어야 한다.
3. **`Adam()`은 이번 모델의 옵티마이저를 만들고, `fit()`이 학습을 실행한다.** Adam은 기울기와 누적 상태를 이용해 갱신하는 옵티마이저다. 생성만으로 학습하지 않는다. 새 모델에 이전 모델의 옵티마이저를 그대로 연결하면 다른 파라미터를 갱신하는 실수가 생길 수 있다.[^adam]
4. **`make_val_noisy()`와 `predict_fixed()`는 학습 뒤의 평가를 준비하고 실행한다.** 여기에는 `backward()`나 `step()`이 없다. 앞에서 전달한 `trial`의 파라미터가 이미 갱신되어 있으므로 그 객체를 그대로 평가한다.

`fit(trial, torch.optim.Adam(...), ...)`처럼 옵티마이저 생성을 함수 인자 안에 적어도 의미는 같다. Python은 안쪽의 `Adam(...)`을 먼저 실행해서 옵티마이저 객체를 만든 뒤, 그 결과를 `fit()`의 인자로 전달한다. 두 줄로 나누면 어떤 모델의 파라미터에 연결했는지 확인하기 쉽다.

원래 실습의 `fit(..., sigma=sigma, verbose=False)` 호출에는 에포크 수가 직접 보이지 않는다. 3에포크라는 설명을 확인하려면 함수 정의나 주변 설정이 필요하다. 위의 보완 구현에서는 `epochs=3`을 명시해 그 조건을 드러냈다. `lr=0.01`, 배치 크기 128, 시드 값도 이 실행 예제의 설정이며 최적값을 보장하지 않는다.

같은 시드 하나로 모든 비교 조건이 자동 통제되지는 않는다. 데이터 분할, 이미지 순서, 잡음 생성 방식도 확인해야 한다. 실행 파일은 초기화·학습 순서·학습 잡음·검증 잡음의 난수 상태를 구분한다. 그래도 PyTorch 버전이나 장치가 달라지면 같은 결과가 항상 보장되는 것은 아니다.[^randomness]

## 14. 23%와 32%는 복원 뒤에 남은 MSE 비율이다

비율을 계산하는 목적은 **입력에 있던 평균 제곱오차를 기준으로 복원 후 오차가 얼마나 남았는지** 확인하는 것이다. 입력 평균 MSE를 `base`, 복원 평균 MSE를 `rest`라고 하자. 남은 비율은 `rest / base`다. 이를 100%에서 빼면 MSE 감소율이 된다.

| 계산 | 의미 |
| --- | --- |
| `ratio = rest / base` | 복원 MSE가 입력 MSE의 몇 배인지 |
| `ratio × 100%` | 입력 대비 남은 MSE 백분율 |
| `(1 − ratio) × 100%` | 입력 대비 MSE 감소율 |

`ratio < 1`이면 평균 MSE가 줄었고, `ratio > 1`이면 오히려 커졌다. `base`가 0이면 0으로 나눌 수 없으므로 이 비율을 정의할 수 없다.

### 기존 평가 로그의 비율을 먼저 읽는다

앞의 기록에서 `base=0.0327`, `rest=0.0078`이다. 이 숫자는 각각 입력 평균 MSE와 복원 평균 MSE이므로, 남은 비율을 계산할 때 다음 순서로 대응시킨다.

`복원 평균 MSE / 입력 평균 MSE = 0.0078 / 0.0327 ≈ 0.2385`

따라서 복원 후 MSE는 입력의 약 <strong>23.85%</strong>가 남았고, MSE 감소율은 약 <strong>76.15%</strong>다. 화면에 소수점 네 자리로 표시된 값을 사용했으므로 근삿값이다. 실제 결과를 보고할 때는 반올림하기 전 `base`와 `rest`로 비율을 계산한 뒤 표시한다.

### 잡음 세기별로 따로 학습한 기록은 별도로 읽는다

다음은 별도 재학습 비교에서 보고된 비율이다. 위의 `0.0327`, `0.0078`과 같은 실험의 절대값으로 합쳐 쓰지 않는다. 이 기록에는 조건별 입력·복원 MSE의 절대값이 포함되어 있지 않다.

| 학습과 평가에 사용한 σ | 보고된 복원/입력 MSE | 대응하는 MSE 감소율 |
| ---: | ---: | ---: |
| 0.25 | 약 23% | 약 77% |
| 0.50 | 약 32% | 약 68% |

두 조건 모두 입력 대비 MSE가 줄었다. 이 비교에서는 강한 잡음 조건의 상대적인 감소율이 더 작았다. `77% − 68% = 9%포인트`이므로 **MSE 감소율이 약 9%포인트 낮았다**고 표현한다. <strong>퍼센트포인트(Percentage Point)</strong>는 두 백분율의 차이를 나타내는 단위다. 단순히 “9% 낮다”라고 쓰면 무엇을 분모로 계산한 변화인지 달라질 수 있다.

“성능이 77% 좋아졌다”보다는 “입력 대비 평균 MSE가 약 77% 감소했다”가 측정 대상을 정확히 드러낸다. 비율만 보고 두 조건의 절대 복원 오차가 얼마나 달랐는지 알아낼 수는 없다. 절대 오차 비교에는 각 조건의 `base`와 `rest`가 필요하다.

```python
# rows는 앞의 재학습 비교에서 실제로 기록한 결과다.
for sigma, base, rest, improved in rows:
    ratio = rest / base if base > 0 else float("nan")
    reduction = (1 - ratio) * 100
    print(
        f"σ={sigma:.2f}, 입력 MSE={base:.4f}, 복원 MSE={rest:.4f}, "
        f"남은 MSE={ratio:.1%}, 감소율={reduction:.1f}%, 개선 장수={improved}"
    )
```

`nan`은 유효한 숫자로 계산할 수 없는 상태를 표시한다. `:.1%`는 비율에 100을 곱한 백분율을 소수점 한 자리로 보여준다. 반면 `reduction`은 이미 100을 곱했으므로 `:.1f` 뒤에 `%` 문자를 붙인다.

여기서 계산하는 것은 **평균 MSE끼리의 비율**이다. `rest_each.mean() / base_each.mean()`과 `(rest_each / base_each).mean()`은 일반적으로 다르다. 앞의 계산에서는 입력 오차가 큰 이미지가 전체 입력 오차에서 더 큰 비중을 차지하지만, 뒤의 계산에서는 이미지마다 구한 비율에 같은 비중을 준다. 뒤의 방식은 입력 MSE가 0인 이미지에 대한 처리도 따로 필요하다. 무엇을 계산했는지 지표 이름과 함께 남겨야 한다.

## 15. 같은 이미지의 배경과 획을 직접 비교한다

MSE 같은 숫자로 측정하는 것을 **정량 평가(Quantitative Evaluation)**, 실제 결과의 모양과 변화를 살펴보는 것을 <strong>정성 평가(Qualitative Evaluation)</strong>라고 한다. 이미지 복원에서는 둘을 함께 확인한다.

비교하는 `clean`, `noisy`, `recon`은 같은 원본 이미지에 대응해야 한다. 데이터가 섞였다면 같은 배열 위치만으로 충분하지 않으므로, 원본 ID도 확인한다. 예를 들어 다른 숫자의 원본과 복원을 나란히 놓으면 획을 비교하는 의미가 없어진다.

| 확인할 부분 | 원본과 비교해 볼 내용 |
| --- | --- |
| 배경 | 잡음 점이 줄었는가? 원래 비어 있던 곳에 새 흔적이 생기지 않았는가? |
| 가는 획 | 끊기거나 사라진 부분이 있는가? |
| 획의 두께와 경계 | 원본보다 지나치게 두껍거나 흐려졌는가? |
| 전체 형태 | 숫자를 구분하는 데 필요한 모양이 유지되는가? |

실행 파일의 `save_triplet()`은 같은 배열 위치의 세 이미지를 저장한다. 표시 밝기도 세 그림 모두 0~1로 맞춘다. 다음 코드는 원본 ID와 배열 순서가 이미 대응함을 확인한 뒤 사용한다.

```python
# save_triplet은 함께 제공한 전체 Python 파일에 정의되어 있다.
# 반환값은 저장한 PNG 하나의 pathlib.Path 객체다.
saved_path = save_triplet(clean, noisy, recon, index=0, out_dir="figures")
print(saved_path.name)      # 예: image_0.png
print(saved_path.exists())  # 해당 경로가 실제로 존재하는지 확인한다.
```

그 함수는 Matplotlib의 `imshow()`로 `images[index,0]`을 표시한다. 첫 인덱스는 이미지 번호이고 두 번째 0은 흑백 채널을 선택한다. 이렇게 선택한 텐서의 shape는 `(28,28)`이다. `detach()`로 역전파 기록과 분리한 뒤 `cpu().numpy()`로 화면에 그릴 배열을 준비하고, `savefig()`로 저장한다. 실제 손글씨 이미지가 제공되지 않았으므로 이 글에는 임의로 만든 복원 성공 그림을 넣지 않았다.[^imshow]

### `PosixPath(...)` 목록이 출력돼도 곧바로 오류는 아니다

다른 시각화 함수가 여러 그림 경로를 반환하면 노트북에 아래처럼 보일 수 있다.

```text
[PosixPath('figures/trained_1.png'), PosixPath('figures/trained_2.png')]
```

`PosixPath`는 Python의 `pathlib`가 Unix 계열 시스템에서 경로를 표현하는 객체다. 대괄호는 경로들이 리스트에 들어 있다는 뜻이다. Jupyter의 Python 실행 환경인 IPython은 셀 마지막 표현식의 반환값을 자동으로 보여줄 수 있다. 그림을 표시하는 함수가 경로 목록도 반환하면 그림 다음에 그 목록이 나올 수 있다.[^pathlib][^ipython]

호출 결과를 `saved_paths = ...`처럼 변수에 받으면 마지막 반환값의 자동 표시를 피할 수 있다. 노트북에서는 호출 끝의 세미콜론도 같은 목적으로 사용할 수 있다. 다만 함수 내부의 `print()`나 명시적인 화면 출력까지 모두 없애는 기능은 아니다.

**경로 객체를 반환했다는 사실과 파일이 실제로 저장됐다는 사실도 구별한다.** `exists()`가 참인지 확인하고, 필요한 그림을 열어 내용까지 확인한다. 원래의 `tools.draw_triplets()`와 이 글의 `save_triplet()`은 서로 다른 함수다. 전자는 원본 구현이 없으므로 인자 이름만 보고 내부의 이미지 선택과 저장 방식을 단정하지 않는다.

## 16. 결과로 확인한 범위를 정확하게 적는다

“말할 수 있는 범위와 확인하지 않은 범위를 구분한다”는 것은 **측정한 사실을 적고, 측정하지 않은 결과까지 확대하지 않는 것**이다. 단순히 조심스러운 문장을 덧붙이는 일이 아니다. 실험에서 무엇을 바꿨고 무엇을 관찰했는지 맞추는 과정이다.

| 관찰한 기록 | 그 기록으로 확인한 것 | 추가로 확인해야 하는 것 |
| --- | --- | --- |
| 10,000장 모두 이미지별 MSE 감소 | 각 이미지의 평균 픽셀 오차 감소 | 모든 픽셀·모든 중요한 획의 개선 여부 |
| 재학습 비교에서 남은 MSE 23%, 32% | 각각의 조건에서 입력 대비 약 77%, 68% 감소 | 같은 모델 하나의 잡음 변화 대응 |
| 한 데이터와 두 잡음 세기의 평가 | 해당 조건에서의 측정 결과 | 실제 사진, 다른 데이터, 다른 잡음 종류의 결과 |
| 한 난수 조건에서 수행한 학습 | 그 초기 조건에서 얻은 결과 | 여러 시드로 반복했을 때의 변동 |
| 픽셀 MSE 측정 | 픽셀값의 제곱오차 | 숫자 분류 정확도나 다른 실제 작업의 성능 |

<strong>일반화(Generalization)</strong>는 학습에 직접 사용하지 않은 데이터에서도 모델이 잘 작동하는 성질이다. 검증 이미지가 훈련에 사용되지 않았는지 확인해야 일반화 성능의 근거로 사용할 수 있다. 데이터 키가 `clean_train`, `clean_val`이라고 되어 있다는 이유만으로 분리가 증명되지는 않는다.

<strong>학습 곡선(Learning Curve)</strong>은 학습 스텝이나 에포크가 진행되는 동안 손실을 기록한 그래프다. 전반적인 감소, 정체, 큰 변동을 확인할 수 있다. 학습 손실이 줄어드는데 별도로 측정한 검증 손실이 지속적으로 증가하면, 훈련 데이터에 맞추면서 새로운 데이터의 성능은 나빠지는 <strong>과적합(Overfitting)</strong>을 점검할 이유가 된다.

이번에 제공된 기록에는 실제 학습 곡선과 같은 ID의 비교 그림이 포함되어 있지 않다. 따라서 곡선이 매끄럽게 감소했다거나 특정 숫자의 획이 보존됐다고 적지 않는다. 전체 실행 파일의 `fit()`은 학습 손실 기록을 반환하지만, 검증 손실 곡선까지 자동으로 생성하는 함수는 아니다.

잡음 세기별 재학습 결과를 한 문단으로 쓴다면 다음처럼 측정 조건과 결론을 붙여 쓸 수 있다.

> 잡음 세기마다 새 모델을 학습한 실습 기록에서, σ=0.25는 입력 대비 평균 MSE가 약 77%, σ=0.50은 약 68% 감소했다. 두 조건 모두 입력보다 복원 오차가 작았고, 강한 잡음 조건에서는 상대적인 감소율이 약 9%포인트 낮았다. 다만 각 조건의 모델을 따로 학습했으므로, 같은 모델 하나가 잡음 변화에 어떻게 대응하는지는 이 결과에서 직접 확인하지 않았다. 다른 데이터와 잡음 종류, 여러 초기 조건, 숫자 인식 정확도에 대한 결론은 별도의 평가가 필요하다.

이 문단은 앞의 단일 모델 로그 `0.0327 → 0.0078`을 다른 실험의 절대값으로 사용하지 않는다. 수치를 읽을 때마다 어느 모델과 어느 데이터에서 측정했는지 함께 남기는 것이 핵심이다.

## 17. 전체 실행 파일에서 두 실험을 직접 구분한다

[Python 실습 파일](/denoising-autoencoder-training-evaluation/examples-ko.py)은 모델, 한 배치 학습, 전체 학습, 이미지별 MSE, 두 실험, 그림 저장을 한 파일에 모았다. PyTorch가 필요하며, 그림 저장 기능을 사용할 때만 Matplotlib도 필요하다. 실습 데이터는 포함되어 있지 않다. 원본 이미지와 분할이 없는 상태에서 앞의 평가 숫자를 재현할 수는 없다.

먼저 데이터 없이 텐서 모양과 함수의 동작을 확인할 수 있다. 내려받은 파일을 노트북과 같은 폴더에 두고 다음을 실행한다. Python의 `runpy.run_path()`는 파일의 코드를 실행하고, 그 파일에서 정의한 이름과 객체를 딕셔너리로 돌려준다. 이 호출에서는 파일의 명령행용 학습 진입점을 실행하지 않는다.[^runpy]

```python
import runpy

lab = runpy.run_path("examples-ko.py")  # 함수와 모델 정의를 읽는다.
lab["self_check"]()  # 인공 텐서로 shape·MSE·학습/평가의 동작을 검산한다.
```

이 검산은 105개 파라미터, `28→14→28`과 `27→14→28`, 반복문과 벡터 연산의 MSE 일치, `stack`과 `unsqueeze`, 손실의 측정 시점, 평가 중 파라미터 보존을 확인한다. 인공 텐서로 검사하므로 손글씨 복원 품질을 측정한 결과가 아니다.

실제 학습에는 깨끗한 훈련 이미지와 별도로 분리한 검증 이미지가 필요하다. 아래에서는 이미 준비한 딕셔너리 `data`에 `"clean_train"`과 `"clean"` 키로 두 텐서가 들어 있다고 가정한다. 검증용 `"clean"`은 훈련에 사용하지 않은 원본이어야 한다. `cpu().float()`는 장치와 자료형을 맞추며, 0~255 데이터를 0~1로 정규화하는 기능은 아니다. 입력이 이미 0~1 범위인지 확인한다.

```python
import torch
import runpy

lab = runpy.run_path("examples-ko.py")
clean_train = data["clean_train"].detach().cpu().float()
clean_val = data["clean"].detach().cpu().float()
lab["check_images"](clean_train, "clean_train")
lab["check_images"](clean_val, "clean_val")

# 실험 A에 사용할 모델을 먼저 한 번 학습한다.
torch.manual_seed(2026)
model = lab["StudentDAE"]().cpu()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
history = lab["fit"](
    model, optimizer, clean_train,
    sigma=0.25, epochs=3, batch_size=128, seed=2026,
)

# A: 방금 학습한 같은 모델의 평가 입력만 바꾼다. 추가 학습은 없다.
rows_a = lab["evaluate_sigmas"](model, clean_val, sigmas=(0.25, 0.50))
lab["print_rows"]("A: 같은 모델 평가", rows_a)

# B: 잡음 세기마다 새 모델을 만들고 처음부터 3에포크씩 학습한다.
rows_b = lab["retrain_sigmas"](
    clean_train, clean_val, sigmas=(0.25, 0.50),
    epochs=3, batch_size=128, verbose=False,
)
lab["print_rows"]("B: 조건별 새 모델 학습", rows_b)
```

이 전체 코드는 A를 위한 모델 한 개와 B를 위한 모델 두 개를 학습한다. **학습 실행은 총 세 번이며, 각 실행에서 3에포크를 처리한다.** A의 평가 루프에서 학습이 일어나는 것은 아니다.

다운로드 파일의 `rows_a`, `rows_b`는 열 이름을 가진 딕셔너리들의 리스트다. 앞에서 계산 과정을 보여주기 위해 만든 튜플 형태의 `rows`와 구별한다. `print_rows()`는 다운로드 파일의 구조에 맞춰 실제 측정된 입력 MSE, 복원 MSE, 남은 비율, 감소율, 개선·악화·동률 수를 출력한다.

숫자를 얻으면 먼저 입력과 복원을 같은 검증 원본에 비교했는지 확인하자. 다음으로 비교한 모델이 하나인지 여러 개인지, 학습이 몇 번 실행됐는지 확인한다. 이 두 조건이 분명해야 결과표의 수치를 올바른 질문에 연결할 수 있다.

## 18. Rust에서도 계산 순서와 남길 축은 같다

Rust의 **tch**는 PyTorch의 C++ 라이브러리인 **LibTorch**를 Rust에서 사용하도록 연결하는 라이브러리다. Python과 함수 이름·인자 표현은 다르지만, 기울기 초기화부터 갱신까지의 순서와 이미지별 MSE를 남기는 방식은 같다.[^tch]

아래는 대응하는 핵심 함수다. `ModuleT`는 학습·평가 여부를 인자로 받아 순전파하는 모델의 공통 기능을 나타내는 <strong>트레이트(Trait)</strong>다. `&Tensor`는 텐서를 빌려 사용한다는 뜻이고, `&mut nn::Optimizer`는 옵티마이저를 수정할 수 있도록 빌린다는 뜻이다.

```rust
use tch::{nn, Kind, Reduction, Tensor};

fn train_one_batch(
    model: &impl nn::ModuleT,
    optimizer: &mut nn::Optimizer,
    noisy: &Tensor,
    clean: &Tensor,
) -> f64 {
    optimizer.zero_grad();               // 이전 기울기 초기화.
    let pred = model.forward_t(noisy, true); // true: 학습 동작 선택.
    assert_eq!(pred.size(), clean.size());
    let loss = pred.mse_loss(clean, Reduction::Mean); // 전체 원소 평균.
    loss.backward();                     // 기울기 계산.
    optimizer.step();                    // 파라미터 갱신.
    loss.double_value(&[])               // 갱신 전에 계산한 스칼라 값.
}

fn per_image_mse(reconstruction: &Tensor, clean: &Tensor) -> Tensor {
    assert_eq!(reconstruction.size(), clean.size());
    (reconstruction - clean)
        .square()
        .flatten(1, -1)   // 이미지 축은 남기고 1번부터 마지막 축까지 합친다.
        .mean_dim(
            &[1_i64][..], // 평균낼 축 목록. i64는 64비트 정수 자료형.
            false,       // 평균낸 축을 크기 1로 남기지 않는다.
            Kind::Float, // 이 예제의 float32 결과 자료형.
        )
}

fn show_loss_shapes() {
    // 실제 학습 성능이 아닌 shape 확인용 숫자다.
    let losses = vec![
        Tensor::from(0.1_f32),
        Tensor::from(0.2_f32),
        Tensor::from(0.3_f32),
    ];
    let each = Tensor::stack(&losses, 0); // 스칼라 세 개 → [3].
    let expanded = each.unsqueeze(1);    // 크기 1인 축 추가 → [3, 1].
    println!("{:?}, {:?}", each.size(), expanded.size());
}
```

`&[1_i64][..]`는 축 번호 1을 담은 목록을 빌린 슬라이스로 전달한다. `double_value(&[])`의 빈 인덱스는 축이 없는 스칼라 텐서의 값을 읽는 표현이다. `forward_t(..., false)`는 평가 동작을 선택하고, 기울기 기록까지 끄려면 `tch::no_grad(...)`를 별도로 사용한다.

| Python | Rust tch | 같은 역할 |
| --- | --- | --- |
| `tensor.shape` | `tensor.size()` | 각 축의 크기 확인 |
| `loss.item()` | `loss.double_value(&[])` | 스칼라 텐서에서 숫자 읽기 |
| `torch.stack(losses, dim=0)` | `Tensor::stack(&losses, 0)` | 텐서 여러 개를 새 축으로 결합 |
| `tensor.unsqueeze(1)` | `tensor.unsqueeze(1)` | 크기 1인 축 추가 |
| `flatten(1).mean(1)` | `flatten(1, -1).mean_dim(...)` | 이미지 축을 남긴 평균 계산 |

[전체 Rust 예제](/denoising-autoencoder-training-evaluation/examples-ko.rs)와 [Cargo.toml](/denoising-autoencoder-training-evaluation/Cargo.toml)에는 모델 생성과 실행 진입점도 포함했다. Rust 코드는 tch 0.26.0 공식 API와 대조했으며, 이 글의 검수 환경에서 컴파일·실행한 코드는 아니다. 해당 버전이 요구하는 LibTorch 2.13.0 환경을 별도로 맞춰야 한다. Python 검산에 사용한 PyTorch 2.15.0을 그대로 연결해 실행할 수 있다고 가정하지 않는다.[^tch-setup]

## 참고 자료

API 확인일은 2026년 11월 3일이다. 평가 로그의 수치는 제공된 실습 기록으로, 아래 공식 문서에서 가져온 성능 수치가 아니다.

[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html). 모델 구성, 호출, `train`, `eval`, `cpu`.
[^activations]: PyTorch, [ReLU](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ReLU.html), [Sigmoid](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sigmoid.html). 활성화 계산.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html). 출력 크기와 가중치·편향의 shape.
[^transpose]: PyTorch, [ConvTranspose2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ConvTranspose2d.html). 출력 크기, 겹침, 역함수와의 구분.
[^mse]: PyTorch, [MSELoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MSELoss.html). 원소별 제곱오차와 평균 범위.
[^zero-grad]: PyTorch, [Optimizer.zero_grad](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.zero_grad.html). 기울기 초기화와 `set_to_none`.
[^item]: PyTorch, [Tensor.item](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.item.html). 원소 하나인 텐서의 숫자 반환.
[^python-builtins]: Python, [zip](https://docs.python.org/3/library/functions.html#zip). 같은 위치의 원소 묶기와 짧은 입력에서의 종료.
[^stack]: PyTorch, [stack](https://docs.pytorch.org/docs/2.15/generated/torch.stack.html). 새 축으로 같은 크기의 텐서 결합.
[^tuples]: Python, [Tuples](https://docs.python.org/3/library/stdtypes.html#tuples). 원소 하나인 튜플의 쉼표 표기.
[^unsqueeze]: PyTorch, [unsqueeze](https://docs.pytorch.org/docs/2.15/generated/torch.unsqueeze.html). 크기 1인 축 추가와 저장공간 공유.
[^flatten]: PyTorch, [flatten](https://docs.pytorch.org/docs/2.15/generated/torch.flatten.html). 지정한 범위의 축 합치기.
[^mean]: PyTorch, [mean](https://docs.pytorch.org/docs/2.15/generated/torch.mean.html). 평균낼 축과 결과 shape.
[^no-grad]: PyTorch, [no_grad](https://docs.pytorch.org/docs/2.15/generated/torch.no_grad.html). 기울기 추적 중단.
[^noise]: PyTorch, [randn](https://docs.pytorch.org/docs/2.15/generated/torch.randn.html), [clamp](https://docs.pytorch.org/docs/2.15/generated/torch.clamp.html). 표준 정규 난수와 값의 범위 제한.
[^adam]: PyTorch, [Adam](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html). 옵티마이저 생성과 갱신 설정.
[^randomness]: PyTorch, [Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html). 난수 시드와 재현성의 범위.
[^imshow]: Matplotlib, [imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html). 이미지 배열과 공통 밝기 범위.
[^pathlib]: Python, [pathlib](https://docs.python.org/3/library/pathlib.html). 경로 객체와 파일 존재 확인.
[^ipython]: IPython, [셀의 실행과 출력](https://ipython.readthedocs.io/en/stable/development/execution.html), [반환값 출력 억제](https://ipython.readthedocs.io/en/stable/config/custommagics.html#silencing-the-magic-output). 마지막 표현식 표시와 세미콜론.
[^runpy]: Python, [runpy.run_path](https://docs.python.org/3/library/runpy.html#runpy.run_path). 파일 실행과 전역 이름의 딕셔너리 반환.
[^tch]: tch 0.26.0, [Tensor](https://docs.rs/tch/0.26.0/tch/struct.Tensor.html), [ModuleT](https://docs.rs/tch/0.26.0/tch/nn/trait.ModuleT.html), [Optimizer](https://docs.rs/tch/0.26.0/tch/nn/struct.Optimizer.html). Rust 대응 API.
[^tch-setup]: tch 0.26.0, [설치 안내](https://docs.rs/crate/tch/0.26.0/source/README.md). LibTorch 버전과 실행 환경 조건.




