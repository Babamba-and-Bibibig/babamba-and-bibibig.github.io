+++
title = "PyTorch로 CNN 이어 만들기: VGG 블록, Flatten·Linear, 잔차 연결"
description = "입력·가중치·출력을 구분하고, 작은 합성곱을 쌓는 이유부터 VGG 블록과 분류기, 잔차 연결까지 구현한다. 채널과 shape, 파라미터 수를 직접 검산하고 eval·no_grad·실제 학습의 차이도 확인한다."
date = 2026-10-28
slug = "cnn-vgg-residual-pytorch"

[extra]
katex = true
toc = true
styles = ["cnn-vgg-residual-pytorch/article.css"]
+++

합성곱 한 층의 계산을 이해했더라도, 여러 층을 연결하기 시작하면 새로운 질문이 생긴다. 첫 번째 합성곱의 출력은 두 번째 합성곱에 어떻게 들어갈까? `3×3` 합성곱을 두 번 쌓으면 왜 입력의 `5×5` 범위까지 연결될까? 마지막 특징 맵을 `Linear`에 전달할 때는 어떤 숫자들을 세어야 할까?

이번에는 이런 연결을 직접 구현한다. 먼저 입력 데이터와 가중치, 출력 데이터를 구분한다. 이어서 합성곱 두 층과 풀링을 묶은 VGG식 블록을 만들고, 두 블록 뒤에 분류기를 붙인다. 마지막에는 원래 입력을 계산 결과에 더하는 잔차 연결과, 더할 두 경로의 크기를 맞추는 방법을 살펴본다.

앞선 [CNN의 채널과 커널](/cnn-convolution-basics/), [합성곱 직접 계산 실습](/cnn-convolution-practice/)에서 다룬 계산이 출발점이다. 다만 이 글만 읽어도 따라갈 수 있도록 지금 필요한 용어와 shape를 다시 설명한다. 예제의 초점은 <strong>어떤 숫자를 계산하는지, 그 숫자가 어느 축에 몇 개 남는지</strong>에 있다.

Python 코드 블록은 위에서 아래로 이어진다. [주석이 포함된 전체 Python 코드](/cnn-vgg-residual-pytorch/examples-ko.py)와 [Rust 개수 검산 코드](/cnn-vgg-residual-pytorch/shape-checks-ko.rs)도 함께 제공한다. Python 실습은 작은 무작위 입력으로 구조와 값을 검사한다. 실제 이미지 분류 성능을 측정하거나 모델을 학습시키는 예제는 아니다. 학습 함수는 뒤에서 정의하지만 호출하지 않는다.

## 1. 모델을 만들 때와 입력을 계산할 때를 나눈다

<strong>CNN(합성곱 신경망)</strong>은 입력의 작은 공간 영역에서 같은 가중치를 반복 사용해 특징을 계산하는 신경망이다. <strong>층 또는 계층(layer)</strong>은 그 안에서 한 단계의 계산을 맡는 구성 요소다.

이 글에서 계속 구별할 대상은 세 가지다.

| 대상 | 무엇을 담는가 | 예 |
| --- | --- | --- |
| 입력 데이터 | 이번 계산에서 처리할 숫자 | 원본 이미지, 앞 층의 출력 특징 맵 |
| 파라미터 | 입력을 계산하는 데 사용하며 학습으로 조정하는 숫자 | 합성곱·Linear의 가중치와 편향 |
| 출력 데이터 | 입력과 현재 파라미터로 계산한 결과 | 새 특징 맵, 클래스별 점수 |

<strong>가중치</strong>는 입력값에 곱하는 숫자이고, <strong>편향</strong>은 곱셈 결과를 더한 뒤 추가하는 숫자다. 모델이 학습하면서 조정하는 이 숫자들을 <strong>파라미터(parameter)</strong>라고 한다.

합성곱 층을 만들면 필요한 가중치와 편향이 준비된다. 하지만 입력을 전달하지 않았다면 그 입력에 대한 출력 특징 맵은 아직 없다. 또 모델이 출력값을 계산했다고 해서 가중치가 자동으로 학습되는 것도 아니다.

| 단계 | 실행하는 일 | 이때 생기는 것 |
| --- | --- | --- |
| 클래스 정의 | 모델이 가질 구성 요소와 계산 방법을 작성한다. | 모델을 만들기 위한 정의 |
| 모델 객체 생성 | 정의에 따라 실제 층을 만들고 등록한다. | 층 객체와 초기 파라미터 |
| 입력 준비 | 이미지 데이터를 읽거나 예제 텐서를 만든다. | 입력 데이터 |
| `model(x)` 호출 | 현재 입력을 정해 둔 순서로 계산한다. | 출력 특징 맵 또는 점수 |
| 구조 검사 | 중간 크기와 계산값을 확인한다. | 검사 결과 |
| 실제 학습 | 손실을 구하고, 기울기를 계산하고, 파라미터를 갱신한다. | 변경된 파라미터 |

여기서 <strong>객체</strong>는 프로그램에서 실제로 만들어 사용하는 대상이다. 텐서도 객체이고, 합성곱 층도 객체다. 클래스와 객체의 차이는 직접 모델을 정의하기 전에 다시 살펴본다.

## 2. 배치, 채널, 한 칸의 숫자는 서로 다른 단위다

<strong>텐서(tensor)</strong>는 PyTorch에서 숫자 배열을 저장하고 계산하는 자료형이다. 배열에서 위치를 구분하는 기준을 <strong>축</strong>이라고 하며, <strong>shape</strong>는 각 축의 크기를 나타낸다.

이미지 여러 장을 한 번에 처리하는 묶음은 <strong>배치(batch)</strong>다. 이 글에서는 이미지 배치의 축을 `[N, C, H, W]` 순서로 읽는다.

| 축 번호 | 의미 | `[1, 4, 7, 7]`에서의 크기 | 유효 인덱스 |
| --- | --- | ---: | --- |
| 0 | 데이터 개수 `N` | 1 | 0 |
| 1 | 데이터 하나의 채널 수 `C` | 4 | 0~3 |
| 2 | 채널 하나의 높이 `H` | 7 | 0~6 |
| 3 | 채널 하나의 너비 `W` | 7 | 0~6 |

<strong>인덱스(index)</strong>는 실제로 선택할 위치의 번호이며 0부터 시작한다. 축 번호 1, 그 축의 크기 4, 선택할 인덱스 2는 서로 다른 정보다.

RGB 이미지라면 세 채널은 빨강·초록·파랑 성분을 구분한다. 합성곱 이후의 채널은 계산으로 만든 <strong>특징 맵(feature map)</strong>이다. 특징 맵은 이 문맥에서 채널 하나의 세로·가로 출력 배열을 말한다. 중간 채널이 16개가 되었다고 원본 사진이 16장으로 늘어난 것은 아니다.

`x`의 shape가 `[1, 4, 7, 7]`일 때 다음 표현은 각각 다른 크기의 대상을 선택한다.

| 표현 | 선택하는 대상 | 결과 shape |
| --- | --- | --- |
| `x[0]` | 첫 데이터 전체 | `[4, 7, 7]` |
| `x[0, 0]` | 첫 데이터의 첫 채널 전체 | `[7, 7]` |
| `x[0, 0, 3, 3]` | 그 채널의 행 3, 열 3에 있는 숫자 하나 | `[]`, 0차원 텐서 |

마지막의 `3`은 네 번째 행과 네 번째 열을 뜻한다. 출력 한 칸을 계산한다는 말은 바로 이런 위치에 들어갈 숫자 하나를 구한다는 뜻이다. 특징 맵 전체를 숫자 하나로 줄인다는 뜻이 아니다.

이번 채널 설명에는 `4→6` 합성곱을 사용한다. 입력 채널 수, 출력 채널 수, 커널 크기를 서로 다른 숫자로 두면 각 숫자의 역할이 드러난다. 뒤에서 만들 분류 모델은 별도로 RGB `3`채널 입력을 사용한다.

## 3. 필터 하나 안에는 입력 채널별 커널이 있다

다음 층의 세 인자를 먼저 읽어 보자.

```python
# 이 한 줄은 층을 만드는 표현이다. 아직 입력 이미지를 계산하지 않는다.
# nn을 가져오는 준비 코드는 다음 절에 있다.
# nn.Conv2d(4, 6, 3)
```

첫 번째 `4`는 입력 채널 수, 두 번째 `6`은 출력 채널 수, 마지막 `3`은 커널의 공간 크기 `3×3`이다. 마지막 인자가 커널의 개수 3을 뜻하는 것은 아니다.

문헌에 따라 <strong>필터(filter)</strong>와 <strong>커널(kernel)</strong>을 같은 뜻으로 쓰기도 한다. 이 글에서는 연결을 구분하기 위해 다음과 같이 부른다.

| 이름 | 이 글에서의 의미 | `Conv2d(4, 6, 3)`의 경우 |
| --- | --- | --- |
| 채널별 커널 | 입력 채널 하나에 대응하는 2차원 가중치 배열 | `[3, 3]` |
| 필터 | 출력 채널 하나를 만드는 전체 가중치 묶음 | `[4, 3, 3]` |
| 층 전체의 가중치 | 필터 6개의 가중치를 모은 텐서 | `[6, 4, 3, 3]` |

아래 설명은 <strong>`groups=1`인 일반적인 합성곱</strong>을 기준으로 한다. 이 설정에서는 필터마다 모든 입력 채널을 사용한다. 채널 연결을 여러 그룹으로 나누는 그룹 합성곱에서는 필터가 받는 채널 범위가 달라진다.[^conv]

### 채널과 커널은 1:1로 대응한다

필터 A에는 A0, A1, A2, A3이라는 채널별 커널이 있다. A0은 입력 채널 0을, A1은 입력 채널 1을 계산한다. 나머지도 같은 방식이다.

| 입력 데이터 | 대응하는 가중치 | 한 출력 위치에서 하는 계산 |
| --- | --- | --- |
| 입력 채널 0의 `3×3` 영역 | 커널 A0 | 같은 위치끼리 곱한 9개 값을 합산 |
| 입력 채널 1의 `3×3` 영역 | 커널 A1 | 같은 위치끼리 곱한 9개 값을 합산 |
| 입력 채널 2의 `3×3` 영역 | 커널 A2 | 같은 위치끼리 곱한 9개 값을 합산 |
| 입력 채널 3의 `3×3` 영역 | 커널 A3 | 같은 위치끼리 곱한 9개 값을 합산 |

그다음 채널별 결과 네 개를 더하고, 편향을 사용한다면 필터 A의 편향 하나를 더한다. 이 결과가 출력 채널 A의 한 칸이다.

입력 채널이 4개이고 커널도 4개라는 이유로 `4×4=16`쌍을 계산하지 않는다. 채널마다 자기 커널이 대응하므로 <strong>4쌍</strong>이다. 필터 B도 같은 입력 채널 네 개를 사용하지만, B0~B3은 A와 별도로 학습되는 가중치다.

<figure class="cnn-vgg-figure">
<img src="/cnn-vgg-residual-pytorch/figure-channel-filter-ko.svg" alt="입력 네 채널이 필터 하나의 채널별 3×3 커널 네 개와 각각 대응한다. 채널별 곱셈 결과를 합해 출력 한 칸을 만들며, 필터 여섯 개는 출력 여섯 채널을 만든다." loading="lazy">
<figcaption>입력 채널은 데이터이고 커널은 가중치다. 둘의 개수가 같아도 같은 대상을 세는 것은 아니다.</figcaption>
</figure>

그림은 편향을 포함한 경우다. 다음 절의 수동 검산에서는 곱셈·합산을 따로 보기 위해 편향을 제외한다.

### 6, 24, 216은 각각 무엇을 세는가

| 세는 대상 | 계산 | 개수 |
| --- | --- | ---: |
| 출력 채널을 만드는 필터 | `out_channels` | 6 |
| 필터 하나의 채널별 커널 | `in_channels` | 4 |
| 채널별 커널 하나의 가중치 | `3×3` | 9 |
| 필터 하나의 가중치 | `4×3×3` | 36 |
| 층 전체의 채널별 커널 | `6×4` | 24 |
| 층 전체의 가중치 | `6×4×3×3` | 216 |
| 편향을 사용할 때의 편향 | 출력 필터마다 하나 | 6 |

입력 채널이 16개인 `3×3` 합성곱이라면 필터 하나의 가중치는 `16×9=144`개다. 이때의 144는 출력 특징 맵 개수가 아니다. 출력 특징 맵 개수는 여전히 필터 개수인 `out_channels`로 정한다.

또 같은 필터는 여러 공간 위치와 여러 이미지에서 반복 사용된다. 배치가 두 배가 되어도 필터가 새로 두 배 생성되지는 않는다. 학습으로 값이 갱신되기 전까지 같은 모델 파라미터를 반복 사용하는 것이다.

## 4. 출력 한 칸을 직접 계산해 Conv2d와 비교한다

먼저 PyTorch 이름과 실행 장치를 준비한다. `torch`는 텐서와 연산을 제공하는 패키지다. `nn`은 층과 모델을 구성하는 `torch.nn`을 짧게 부르는 이름이고, `F`는 함수 형태의 신경망 연산을 제공하는 `torch.nn.functional`의 별칭이다.

```python
import torch
from torch import nn
import torch.nn.functional as F

# 모델과 입력을 같은 장치에 둔다. 이번 작은 예제는 CPU로 실행한다.
device = torch.device("cpu")

# 같은 실행 순서에서 무작위 입력과 초기 가중치를 재현하기 쉽게 한다.
# 모든 버전·장치·연산의 완전한 재현성을 보장하는 설정은 아니다.
torch.manual_seed(16)

conv = nn.Conv2d(
    in_channels=4,
    out_channels=6,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,  # 이번 검산은 가중치와 입력의 곱셈·합산만 확인한다.
).to(device)

# randn은 무작위 실수 텐서를 만든다. 사진을 읽는 함수가 아니다.
x = torch.randn(1, 4, 7, 7, device=device)

print(tuple(conv.weight.shape))        # (6, 4, 3, 3): 전체 필터
print(tuple(conv.weight[0].shape))     # (4, 3, 3): 필터 A
print(tuple(conv.weight[0, 0].shape))  # (3, 3): A의 채널 0용 커널
print(conv.weight.numel())             # 216: 저장된 가중치 숫자 수
```

`numel()`은 텐서의 원소 수를 반환한다. 가중치 텐서의 축은 입력 텐서의 축과 의미가 다르다. `conv.weight`의 첫 축 6은 이미지 여섯 장이 아니라 <strong>출력 필터 여섯 개</strong>를 뜻한다.

현재 설정에서 입력과 출력의 높이·너비는 모두 7이다. 출력 `y[0, 0, 3, 3]`을 만드는 데에는 각 입력 채널의 행 2~4, 열 2~4가 필요하다. `2:5`는 2, 3, 4를 선택하는 슬라이스이며 끝의 5는 포함하지 않는다.

`torch.no_grad()`는 아래 계산을 역전파용 연산 기록에 넣지 않도록 한다. 출력값 자체는 정상적으로 계산한다. 지금은 학습 없이 숫자를 비교하므로 사용하며, 학습·평가 모드와의 차이는 뒤에서 살펴본다.

```python
with torch.no_grad():
    y = conv(x)
    actual_value = y[0, 0, 3, 3]  # 첫 출력 채널의 숫자 하나

    # 모든 입력 채널에서 같은 3×3 공간 영역을 선택한다.
    patch = x[0, :, 2:5, 2:5]    # [4, 3, 3], 아직 입력 데이터다.
    kernels = conv.weight[0]    # [4, 3, 3], 필터 A의 가중치다.

    # *는 같은 위치끼리의 곱셈이다. 행렬곱이 아니다.
    products = patch * kernels  # [4, 3, 3], 곱셈 결과 36개

    # 현재 products의 축 1·2는 높이·너비다. 채널마다 9개를 더한다.
    channel_results = products.sum(dim=(1, 2))  # [4]
    manual_value = channel_results.sum()       # 네 채널 결과의 합

    # 실수 계산의 작은 오차를 허용해 같은 결과인지 검사한다.
    assert torch.allclose(
        manual_value, actual_value, atol=1e-6, rtol=1e-4
    )

print(tuple(y.shape))             # (1, 6, 7, 7)
print(tuple(y[0, 0].shape))       # (7, 7)
print(channel_results)           # 채널별 합계 네 개
print(manual_value.item())       # 직접 구한 출력값
print(actual_value.item())       # Conv2d가 구한 출력값
```

`assert`는 조건이 거짓이면 오류를 발생시키는 검사 문장이다. `allclose`는 대응하는 숫자가 허용 오차 안에서 가까운지 확인한다. `atol`은 절대 오차, `rtol`은 비교값의 크기에 비례하는 상대 오차 기준이다. `item()`은 원소 하나짜리 텐서에서 일반 Python 숫자를 꺼낸다.[^check]

이번 계산에서 크기는 다음 순서로 바뀐다.

| 단계 | shape | 저장된 것 |
| --- | --- | --- |
| 입력 패치 | `[4, 3, 3]` | 채널 네 개에서 읽은 입력값 36개 |
| 원소별 곱셈 | `[4, 3, 3]` | 입력값과 가중치의 곱 36개 |
| 공간 축 합산 | `[4]` | 채널별 결과 네 개 |
| 채널 축 합산 | `[]` | 출력 한 칸의 값 하나 |

필터 A의 가중치는 36개지만 출력 특징 맵 A에는 `7×7=49`개 숫자가 있다. 같은 가중치 36개를 다른 위치에 반복 적용해서 49칸을 만들기 때문이다. 여섯 필터의 출력 전체에는 `1×6×7×7=294`개 숫자가 있다.

이렇게 채널별 결과를 따로 적으면 계산 관계를 이해하기 쉽다. 실제 PyTorch 구현이 모든 중간 배열을 이 코드처럼 별도로 저장하거나 같은 순서로 실행해야 하는 것은 아니다. 여러 위치와 채널을 병렬로 처리할 수 있다.

PyTorch의 `Conv2d`는 수학적 합성곱처럼 커널을 뒤집지 않는 <strong>교차상관</strong>을 사용한다. 딥러닝에서는 이 계산을 일반적으로 합성곱이라고 부른다.[^conv]

## 5. 채널 수와 공간 크기는 각각 확인한다

합성곱을 연결할 때는 <strong>채널</strong>과 <strong>높이·너비</strong>를 나누어 읽으면 된다.

| 인자 | 정하는 것 |
| --- | --- |
| `in_channels` | 이 층이 받을 입력 채널 수 |
| `out_channels` | 만들 출력 채널 수이자 필터 개수 |
| `kernel_size` | 커널의 높이·너비 |
| `stride` | 다음 계산 위치로 이동하는 간격 |
| `padding` | 입력 가장자리에 추가한 것으로 취급하는 영역의 폭 |
| `dilation` | 커널의 이웃 원소가 참조하는 입력 위치 사이의 간격 |

기본 패딩 방식에서는 추가 영역을 0으로 취급한다. 이 글의 `dilation`은 모두 1이므로 커널은 붙어 있는 입력 위치를 읽는다.

### 출력 크기 공식은 적용할 수 있는 시작 위치의 개수다

입력 높이를 `H`, 커널 높이를 `K`, 위·아래 각각의 패딩 폭을 `P`, 이동 간격을 `S`, dilation을 `D`라고 하자. 커널이 차지하는 입력 범위의 높이는 `D(K−1)+1`이다.

패딩을 포함한 높이 `H+2P`에서 그 범위를 빼면 시작 위치가 이동할 수 있는 거리가 남는다. 이 거리를 `S`씩 나누고, 처음 시작 위치 하나를 더하면 출력 높이가 된다.[^conv]

<div class="cnn-vgg-math">
$$
H_{\mathrm{out}}
=\left\lfloor\frac{H+2P-D(K-1)-1}{S}\right\rfloor+1
$$
</div>

`floor`는 해당 값 이하인 가장 큰 정수로 내리는 연산이다. 너비도 같은 방법으로 계산한다. 이 글의 유효한 크기 계산에서는 나머지 공간에 커널을 한 번 더 놓을 수 없으면 버린다고 읽으면 된다.

`K=3, P=1, S=1, D=1`을 넣으면 `H_out=H`, `W_out=W`다. 이것은 <strong>높이·너비를 유지하는 조건</strong>이다. 채널까지 유지하려면 `in_channels`와 `out_channels`도 같아야 한다.

| 층 | 입력 | 출력 |
| --- | --- | --- |
| `Conv2d(16, 16, 3, padding=1)` | `[1,16,14,14]` | `[1,16,14,14]` |
| `Conv2d(16, 32, 3, padding=1)` | `[1,16,14,14]` | `[1,32,14,14]` |
| `Conv2d(16, 32, 3, stride=2, padding=1)` | `[1,16,14,14]` | `[1,32,7,7]` |

마지막 행은 `floor((14+2−2−1)/2)+1 = floor(13/2)+1 = 7`이다. `3×3` 커널과 패딩 1만 보고 전체 shape가 같다고 판단하면 stride와 출력 채널의 변화를 놓친다.

### 두 Conv를 잇는 조건과 두 출력을 더하는 조건

두 Conv를 순서대로 연결하려면 앞 층의 `out_channels`가 뒤 층의 `in_channels`와 맞아야 한다. 공간 크기는 뒤의 커널을 적용할 수 있으면 달라도 된다.

예를 들어 패딩 없는 `3×3` Conv 두 개는 `[1,3,28,28] → [1,8,26,26] → [1,8,24,24]`로 연결할 수 있다. 중간 높이·너비가 줄어드는 것이 연결 오류는 아니다.

반면 뒤에서 만들 잔차 연결은 두 텐서의 같은 위치 값을 더한다. 그때에는 합산 지점의 전체 shape도 맞춰야 한다. <strong>다음 층에 전달하는 것과 두 경로를 더하는 것은 확인할 조건이 다르다.</strong>

Conv 객체를 만들 때 높이·너비를 지정하지 않는 이유도 여기서 알 수 있다. 실제 `conv(x)` 호출에서 전달된 입력의 공간 크기를 사용하기 때문이다. `Conv2d(16,16,3,padding=1)`은 `14×14`도, `20×30`도 처리할 수 있지만 입력 채널 수는 16이어야 한다.

## 6. 클래스, 객체, 함수 호출을 구별한다

VGG 블록을 작성하기 전에 Python 코드의 괄호가 어떤 역할을 하는지 정리하자.

<strong>클래스(class)</strong>는 객체가 가질 속성과 동작을 정의한다. <strong>함수(function)</strong>는 인자를 받아 작업을 수행한다. 객체를 통해 `x.flatten(...)`처럼 호출하는 함수는 <strong>메서드(method)</strong>라고 부른다.

`nn.Module`은 PyTorch에서 층과 모델을 구성할 때 사용하는 기본 클래스다. 가중치가 있는 `Conv2d`뿐 아니라 가중치 없는 `ReLU`와 `MaxPool2d`도 Module이다.[^module]

| 표현 | 모델 코드에서 하는 일 |
| --- | --- |
| `class MyVGGBlock(nn.Module):` | Module을 기반으로 새로운 클래스의 구조를 정의한다. |
| `__init__` | 객체 생성 과정에서 속성과 하위 층을 초기화한다. |
| `super().__init__()` | 부모 Module의 관리 기능을 초기화한다. |
| `self` | 현재 만들어진 객체를 가리킨다. |
| `self.conv1 = nn.Conv2d(...)` | 층 객체를 만들고 현재 모델의 하위 모듈로 등록한다. |
| `forward(self, x)` | 실제 입력을 받았을 때의 계산 순서를 정의한다. |
| `model = MyVGGBlock(...)` | 정의를 사용해 실제 모델 객체를 만든다. |
| `model(x)` | Module의 호출 절차를 거쳐 `forward`를 실행한다. |

하위 모듈로 등록된 층은 `model.parameters()`로 파라미터를 모으거나 `model.to(device)`로 장치를 바꿀 때 함께 관리된다. `model.forward(x)`를 직접 부르면 Module이 제공하는 호출 처리 일부를 거치지 않으므로 일반적인 실행에는 `model(x)`를 쓴다.[^module]

### ReLU를 만드는 괄호와 계산하는 괄호

<strong>ReLU</strong>는 음수는 0으로, 0 이상은 그대로 반환하는 함수다. 예를 들어 `[-2,3,0,-4]`를 `[0,3,0,0]`으로 바꾼다. 원소별 값은 바뀔 수 있지만 shape는 유지하며 학습할 가중치는 없다.[^relu]

가중합만 여러 번 연결하면 하나의 가중합으로 정리되는 경우가 있다. ReLU는 입력값에 따라 음수 구간과 양수 구간의 계산이 달라지는 <strong>비선형 변환</strong>을 추가한다. 이 글에서는 먼저 실제 값이 어떻게 바뀌는지 확인하면 된다.

| 표현 | 뜻 |
| --- | --- |
| `F.relu` | 함수 자체 |
| `F.relu(x)` | 함수를 실행해 얻은 출력 텐서 |
| `F.relu()` | 입력을 주지 않아 발생하는 잘못된 호출 |
| `nn.ReLU` | 클래스 자체 |
| `nn.ReLU()` | ReLU 층 객체 생성 |
| `relu_layer(x)` | 이미 만들어 둔 층 객체 실행 |
| `nn.ReLU()(x)` | 객체를 만들고 바로 입력을 계산 |

```python
relu_input = torch.tensor([-2.0, 3.0, 0.0, -4.0], device=device)

relu_function = F.relu  # 함수 자체를 저장한다. 아직 입력을 계산하지 않는다.
relu_layer = nn.ReLU()  # 층 객체를 만든다. 아직 입력을 계산하지 않는다.

result1 = relu_function(relu_input)
result2 = relu_layer(relu_input)

assert torch.equal(result1, result2)  # shape와 모든 원소값이 같다.
print(result1)  # tensor([0., 3., 0., 0.])
```

`self.relu = F.relu`는 함수 자체를 속성에 저장하므로 사용할 수 있다. `self.relu = F.relu(x)`는 계산 결과인 텐서를 저장하는 코드다. `self.relu = nn.ReLU()`는 층 객체를 저장하고 하위 모듈로 등록한다.

`nn.Sequential`에는 `nn.ReLU()`처럼 모듈 객체를 넣는다. 직접 작성한 `forward` 안에서는 `F.relu(x)`를 호출해도 된다. `nn.relu(x)`는 이 용도로 사용하는 올바른 API가 아니다.

이 관계를 모든 함수형 API에 그대로 확대해서는 안 된다. 예를 들어 함수형 Dropout은 학습·평가 여부를 인자로 전달해 관리해야 한다. 여기서 확인한 것은 기본 ReLU의 계산이다.

## 7. MaxPool은 채널마다 최댓값을 고른다

<strong>최대 풀링(MaxPool)</strong>은 각 채널의 작은 공간 영역에서 가장 큰 숫자 하나를 선택한다. 서로 다른 채널의 값을 합치는 계산이 아니므로 채널 수는 유지된다.

`kernel_size=2`는 최댓값을 검사할 영역이 `2×2`라는 뜻이고, `stride=2`는 다음 시작 위치로 두 칸 이동한다는 뜻이다. 문제에서 “2×2 MaxPool, stride 2”라고 주었다면 이 두 조건을 각각 코드로 옮긴다. stride가 2이므로 kernel도 반드시 2라고 추론하는 것은 아니다.

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)

pool_input = torch.tensor(
    [[[
        [3.0, 7.0, 1.0, 4.0],
        [2.0, 5.0, 8.0, 0.0],
        [6.0, 1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0, 9.0],
    ]]],
    device=device,
)

pool_output = pool(pool_input)
print(pool_output)
# tensor([[[[7., 8.],
#           [6., 9.]]]])
print(tuple(pool_output.shape))  # (1, 1, 2, 2)
```

왼쪽 위 영역은 `3,7,2,5` 중 7을, 오른쪽 위는 `1,4,8,0` 중 8을 남긴다. 아래의 두 영역에서는 각각 6과 9가 남는다. 입력 숫자 16개가 네 영역의 최댓값 네 개로 줄어든다.

Conv의 `kernel_size`는 가중치 배열의 공간 크기지만, MaxPool의 `kernel_size`는 값을 검사할 영역의 크기다. <strong>MaxPool에는 학습할 커널 가중치가 없다.</strong>

현재 `kernel_size=2, stride=2, padding=0, ceil_mode=False`에서는 높이·너비가 각각 `floor(H/2)`, `floor(W/2)`가 된다. `32→16`, `30→15`이며, 홀수 `15→7`에서는 남는 가장자리 한 칸을 별도 출력으로 만들지 않는다.[^pool]

`MaxPool2d(2)`처럼 stride를 생략하면 커널 크기와 같은 2가 사용된다. 하지만 `MaxPool2d(3, stride=2)`처럼 두 값을 다르게 설정할 수도 있다.

## 8. 입력 없이 가중치 개수부터 비교할 수 있다

<strong>`nn.Sequential`</strong>은 여러 층 객체를 정해진 순서로 실행하도록 묶는다. 객체를 만들 때는 층을 등록하고, 나중에 입력을 전달하면 앞 층의 출력을 다음 층에 넘긴다.[^sequential]

다음은 입력·중간·출력 채널을 모두 16으로 두고 두 구조를 비교하는 코드다. 편향을 제외해 가중치 수만 센다.

```python
C = 16

two_small = nn.Sequential(
    nn.Conv2d(C, C, 3, padding=1, bias=False),
    nn.ReLU(),
    nn.Conv2d(C, C, 3, padding=1, bias=False),
).to(device)

one_large = nn.Conv2d(
    C, C, 5, padding=2, bias=False
).to(device)

# parameters()가 제공하는 각 파라미터 텐서의 숫자 개수를 합한다.
small_count = sum(p.numel() for p in two_small.parameters())
large_count = sum(p.numel() for p in one_large.parameters())

print(small_count)  # 4608
print(large_count)  # 6400
```

Conv는 생성 시점에 입력 채널, 출력 채널, 커널 크기를 알고 있다. 따라서 입력 이미지 없이 필요한 가중치를 만들 수 있다. 위 코드는 특징 맵을 계산한 것이 아니라 <strong>층을 만들고 그 안의 파라미터 개수를 센 것</strong>이다.

| 구조 | 편향을 제외한 가중치 계산 | 결과 |
| --- | --- | ---: |
| `3×3` Conv 한 층 | `16×16×3×3` | 2,304 |
| `3×3` Conv 두 층 | `2×16×16×3×3` | 4,608 |
| `5×5` Conv 한 층 | `16×16×5×5` | 6,400 |

`16×16`의 첫 16은 출력 필터 수, 두 번째 16은 필터당 입력 채널별 커널 수다. 두 Conv가 같은 shape의 가중치를 갖는다는 것이 같은 가중치 객체를 공유한다는 뜻은 아니다. 첫 층 2,304개와 두 번째 층의 별도 가중치 2,304개를 합친 값이다.

편향을 넣는다면 작은 커널 두 층에는 `16+16=32`개가, 큰 커널 한 층에는 16개가 추가된다. 입력의 높이·너비나 배치 크기는 파라미터 수에 곱하지 않는다.

또 `parameters()`는 현재 기울기 계산을 끈 파라미터도 제공한다. 학습 가능한 것만 세려면 다음처럼 `requires_grad` 조건을 붙인다.

```python
trainable_count = sum(
    p.numel() for p in two_small.parameters() if p.requires_grad
)
```

## 9. 3×3 두 층은 왜 입력의 5×5 범위까지 연결되는가

앞 절에서는 파라미터 개수만 셌다. 이번에는 실제 입력을 추가하고, 두 번째 층의 출력 한 칸이 원래 입력 어디까지 연결되는지 추적한다.

```python
rf_x = torch.randn(1, 16, 7, 7, device=device)

with torch.no_grad():
    # Sequential의 0번은 첫 Conv, 1번은 ReLU, 2번은 두 번째 Conv다.
    rf_m = two_small[1](two_small[0](rf_x))
    rf_y = two_small[2](rf_m)
    rf_large = one_large(rf_x)

print(tuple(rf_m.shape))  # (1, 16, 7, 7)
print(tuple(rf_y.shape))  # (1, 16, 7, 7)
```

`rf_y[0,0,3,3]` 한 칸을 보자. 두 번째 Conv는 중간 특징 맵 `rf_m`의 각 채널에서 행 2~4, 열 2~4의 `3×3` 영역을 읽는다. 중간 채널 16개에서 읽으므로 입력으로 받는 중간값은 `16×9=144`개다.

이 중간값 하나하나는 첫 번째 Conv가 원래 입력의 `3×3` 영역을 계산해 만든 값이다. 첫 번째 Conv도 입력 채널 16개를 모두 사용한다.

| 두 번째 Conv가 읽는 중간 위치 | 그 중간값을 만들 때 읽은 원래 입력의 행 | 원래 입력의 열 |
| --- | --- | --- |
| `(2,2)` | 1, 2, 3 | 1, 2, 3 |
| `(2,3)` | 1, 2, 3 | 2, 3, 4 |
| `(2,4)` | 1, 2, 3 | 3, 4, 5 |
| `(3,2)` | 2, 3, 4 | 1, 2, 3 |
| `(3,3)` | 2, 3, 4 | 2, 3, 4 |
| `(3,4)` | 2, 3, 4 | 3, 4, 5 |
| `(4,2)` | 3, 4, 5 | 1, 2, 3 |
| `(4,3)` | 3, 4, 5 | 2, 3, 4 |
| `(4,4)` | 3, 4, 5 | 3, 4, 5 |

이 표의 입력 위치들을 중복 없이 모으면 행 1~5와 열 1~5다. 원래 입력의 각 채널에서 `5×5` 영역이 출력 한 칸의 계산에 연결될 수 있다.

<figure class="cnn-vgg-figure">
<img src="/cnn-vgg-residual-pytorch/figure-receptive-field-ko.svg" alt="두 번째 3×3 합성곱이 읽는 중간 위치 2부터 4까지를 원래 입력으로 추적하면 행과 열 1부터 5까지의 5×5 영역이 된다." loading="lazy">
<figcaption>커널 자체는 두 층 모두 3×3이다. 넓어진 것은 특정 출력 한 칸에 연결될 수 있는 원래 입력의 범위다.</figcaption>
</figure>

이처럼 구조상 한 출력값에 영향을 줄 수 있는 입력 범위를 <strong>이론적 수용 영역(receptive field)</strong>이라고 한다. 여기서의 조건은 두 Conv 모두 `stride=1, dilation=1`이고, 중간에 Pool이 없으며, 패딩의 영향을 받지 않는 내부 위치를 살펴본다는 것이다.

ReLU는 같은 위치의 숫자를 바꾸므로 공간 범위를 넓히지 않는다. 실제로는 가중치가 0이거나 ReLU가 0을 출력해 어떤 입력의 영향이 사라질 수도 있다. 수용 영역은 <strong>연결될 수 있는 범위</strong>를 설명한다.

### 같은 범위를 읽어도 계산 과정은 다르다

`5×5` Conv 한 층은 원래 입력의 그 범위를 직접 가중합한다. `3×3` 두 층은 먼저 중간 특징을 계산하고 ReLU를 적용한 뒤, 그 중간값을 다시 가중합한다.

따라서 두 구조의 수용 영역이 같아도 출력값이 같을 필요는 없다. `3×3` 커널이 실행 중에 `5×5` 커널로 바뀌는 것도 아니다.

VGG는 작은 합성곱을 여러 층 쌓는 구성을 사용한다. 앞의 예에서는 `6,400−4,608=1,792`개의 가중치를 줄였고, 큰 커널 한 층을 기준으로 28% 적다. 이 비율은 채널을 모두 16으로 두고 편향을 제외한 이번 조건의 결과다. 파라미터 감소만으로 정확도나 실제 실행 속도가 반드시 개선된다고 결론 내릴 수는 없다.[^vgg]

## 10. Conv 두 번과 Pool 한 번을 VGG식 블록으로 묶는다

<strong>블록(block)</strong>은 여러 층을 하나의 구성 요소로 묶은 것이다. 이번 블록의 목적은 공간 크기를 유지하며 특징을 두 번 계산한 뒤, 마지막 Pool에서 높이·너비를 줄이는 것이다.

계산 순서는 `Conv1 → ReLU → Conv2 → ReLU → MaxPool`이다. 예를 들어 입력 채널이 3, 블록의 출력 채널이 8이면 첫 Conv는 `3→8`, 두 번째 Conv는 `8→8`이어야 한다. 두 번째 Conv가 받는 데이터는 첫 Conv가 계산한 결과이기 때문이다.

이 관계를 생성자에 적으면 다음과 같다.

```python
class MyVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()  # 부모 Module의 관리 기능을 준비한다.

        # 원래 입력 채널을 블록의 출력 채널 수로 바꾼다.
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

        # 첫 Conv가 만든 채널 수를 입력받는다.
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

        # 채널은 유지하고 높이·너비를 줄인다.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        result = self.conv1(x)
        result = F.relu(result)
        result = self.conv2(result)
        result = F.relu(result)
        return self.pool(result)
```

`result`에는 각 단계의 출력 텐서를 순서대로 저장한다. `self.conv1`은 층 객체이고 `self.conv1(x)`는 그 층으로 계산한 텐서다. 층 객체를 다음 층의 입력으로 넘기는 것이 아니다.

두 Conv의 출력 채널 수가 같더라도 같은 계산을 반복하는 것은 아니다. 각 층에는 별도 가중치가 있으며, 두 번째 층은 첫 번째 층이 이미 변환한 데이터를 받는다.

현재의 기본 `bias=True`에 따라 두 Conv에는 편향도 만들어진다. 앞의 가중치 수 비교에서 사용한 `bias=False`와 조건이 다르므로, 나중에 이 모델의 전체 파라미터를 셀 때는 편향을 포함한다.

### 정사각형 입력과 직사각형 입력으로 확인한다

`MyVGGBlock(3,8)` 객체 하나를 만들고 두 입력 크기를 넣어 보자. 앞의 `*input_shape`는 튜플 안의 네 숫자를 각각 위치 인자로 펼쳐 전달하는 Python 문법이다.

```python
vgg_block = MyVGGBlock(3, 8).to(device)

with torch.no_grad():
    for input_shape, expected_shape in [
        ((2, 3, 32, 32), (2, 8, 16, 16)),
        ((1, 3, 30, 26), (1, 8, 15, 13)),
    ]:
        images = torch.randn(*input_shape, device=device)

        # 중간 shape를 확인하려고 단계별로 계산한다.
        after_conv1 = F.relu(vgg_block.conv1(images))
        after_conv2 = F.relu(vgg_block.conv2(after_conv1))
        after_pool = vgg_block.pool(after_conv2)

        # 원래 입력부터 블록의 forward를 별도로 다시 실행한다.
        block_output = vgg_block(images)

        assert tuple(block_output.shape) == expected_shape
        assert torch.allclose(
            block_output, after_pool, atol=1e-6, rtol=1e-4
        )
        print(
            input_shape,
            tuple(after_conv1.shape),
            tuple(after_conv2.shape),
            tuple(after_pool.shape),
        )
```

| 입력 | Conv1과 ReLU 뒤 | Conv2와 ReLU 뒤 | Pool 뒤 |
| --- | --- | --- | --- |
| `[2,3,32,32]` | `[2,8,32,32]` | `[2,8,32,32]` | `[2,8,16,16]` |
| `[1,3,30,26]` | `[1,8,30,26]` | `[1,8,30,26]` | `[1,8,15,13]` |

공간 크기를 줄이는 것은 마지막 Pool이다. 따라서 “블록 안의 Conv가 공간 크기를 유지한다”와 “블록 전체가 공간 크기를 유지한다”는 다른 말이다.

중간 계산 뒤에 `vgg_block(images)`를 다시 부른 이유는 직접 나눈 계산과 `forward`의 결과가 맞는지 확인하기 위해서다. 보통 사용할 때는 블록을 한 번 호출하면 된다. 이전에 계산한 `after_conv2`가 전체 호출에서 자동으로 재사용되는 것은 아니다.

## 11. 이 블록과 실제 VGG-16은 어떻게 다른가

이번 `MyVGGBlock`은 작은 합성곱을 쌓고 Pool을 적용하는 구성을 연습하기 위한 블록이다. 이것을 두 개 붙인 작은 모델을 VGG-16 전체라고 부르지는 않는다.

VGG-16의 대표적인 구성은 합성곱 묶음마다 Conv를 `2, 2, 3, 3, 3`개 사용한다. 합성곱은 총 13개이고, 뒤에 완전연결층 세 개가 이어진다. <strong>완전연결층(fully connected layer, FC)</strong>은 각 출력이 입력 특징값 전체를 사용해 계산되는 층으로, 여기서는 `nn.Linear`로 구현한다.[^vgg]

`224×224` RGB 입력을 기준으로 큰 흐름을 적으면 다음과 같다.

| 구간 | Conv 개수 | 구간의 출력 채널 | Pool 뒤 높이·너비 |
| --- | ---: | ---: | --- |
| 첫 번째 묶음 | 2 | 64 | `112×112` |
| 두 번째 묶음 | 2 | 128 | `56×56` |
| 세 번째 묶음 | 3 | 256 | `28×28` |
| 네 번째 묶음 | 3 | 512 | `14×14` |
| 다섯 번째 묶음 | 3 | 512 | `7×7` |

마지막 특징값 `25,088`개는 `512×7×7`에서 나온다. FC 세 층의 특징값 수는 `25,088→4,096→4,096→1,000`으로 바뀐다. 이름의 16은 이런 구성의 학습 가중치가 있는 Conv 13개와 FC 3개를 센 것이다. ReLU와 Pool까지 모두 세어서 16인 것은 아니다.

TorchVision의 VGG 구현에는 분류기 앞에서 출력을 `7×7`로 맞추는 `AdaptiveAvgPool2d`가 있고, FC 사이에 ReLU와 Dropout이 있다. Dropout은 학습 때 일부 값을 무작위로 0으로 만드는 층으로, 뒤에서 실행 모드와 함께 설명한다. 또한 논문의 구성표에는 최종 Softmax가 있지만, TorchVision의 `forward`는 Softmax를 적용하기 전 점수를 반환한다.[^vgg-code]

우리의 작은 모델은 채널을 8과 16으로 줄이고, Conv 네 개와 Linear 한 개로 계산 연결을 확인한다. 실제 VGG의 거대한 분류기를 모두 만들지 않아도 채널 연결과 Flatten의 원리를 살펴볼 수 있다.

## 12. Flatten에서는 마지막 출력 데이터의 개수를 센다

이제 `32×32` RGB 입력을 다섯 종류로 분류하는 작은 모델을 설계하자. “다섯 종류”는 출력 점수를 다섯 개 만들겠다는 뜻이다. 어떤 종류인지는 실제 데이터와 정답 번호의 대응으로 정해진다.

두 VGG식 블록을 다음처럼 연결한다.

| 단계 | shape | 바뀐 이유 |
| --- | --- | --- |
| 입력 | `[N,3,32,32]` | RGB 이미지 `N`장 |
| `MyVGGBlock(3,8)` | `[N,8,16,16]` | Conv가 채널 8개, Pool이 공간 크기 절반 |
| `MyVGGBlock(8,16)` | `[N,16,8,8]` | Conv가 채널 16개, Pool이 공간 크기 절반 |
| `flatten(start_dim=1)` | `[N,1024]` | 이미지별 `16×8×8`개 값을 한 축으로 모음 |
| `Linear(1024,5)` | `[N,5]` | 이미지마다 점수 다섯 개 계산 |

<figure class="cnn-vgg-figure">
<img src="/cnn-vgg-residual-pytorch/figure-model-shapes-ko.svg" alt="RGB 입력 N,3,32,32가 두 VGG식 블록을 지나 N,16,8,8이 된다. 이미지별 1024개 값을 펼쳐 Linear로 N,5 점수를 만든다." loading="lazy">
<figcaption>N은 이미지 개수로 유지된다. 16×8×8은 마지막 특징 맵에 들어 있는 이미지 한 장당 데이터 개수다.</figcaption>
</figure>

Linear가 입력받는 것은 Conv의 필터나 커널 자체가 아니다. Conv와 Pool이 <strong>계산해서 만든 출력 데이터</strong>다. 마지막 특징 맵은 16개이고, 각 맵에는 `8×8=64`개 숫자가 있다. 따라서 이미지 한 장당 입력 특징값은 `16×64=1,024`개다.

`16×9`를 사용하면 다른 대상을 센 것이다. `3×3=9`는 채널별 커널의 가중치 개수이고, `8×8=64`는 마지막 출력 특징 맵 하나의 데이터 개수다.

### start_dim=1이 배치 축을 남긴다

<strong>Flatten</strong>은 여러 축을 한 축으로 펼치는 연산이다. `start_dim=1`이면 축 0은 유지하고 축 1부터 마지막 축까지 합친다.[^flatten]

| 축 | 원래 역할 | 펼친 뒤 |
| --- | --- | --- |
| 0 | 이미지 `N`장 | `N`으로 유지 |
| 1 | 채널 16개 | 아래 공간 축과 합쳐 1,024개 특징 |
| 2 | 높이 8 | 위 특징 축에 포함 |
| 3 | 너비 8 | 위 특징 축에 포함 |

값을 더하거나 평균 내는 연산이 아니다. 각 이미지 안의 값 순서를 유지하면서 shape를 바꾼다. 메모리도 항상 새로 복사하는 것은 아니며, 입력 상태와 펼치는 범위에 따라 원본·뷰·복사본이 반환될 수 있다. 여기서 뷰는 같은 저장 공간을 다른 shape로 보는 텐서다.[^flatten]

같은 작업을 하는 표현은 다음과 같다.

```python
# features는 앞 블록이 만든 텐서라고 가정한 사용 형식이다.
# flat = features.flatten(start_dim=1)         # Tensor 메서드
# flat = torch.flatten(features, start_dim=1)  # torch 함수
# flatten_layer = nn.Flatten(start_dim=1)      # 층 객체 생성
# flat = flatten_layer(features)              # 생성한 층 실행
```

이때 `F.flatten(...)`을 사용하는 것은 아니다.

### 배치까지 펼치면 N=1에서 실수를 놓치기 쉽다

`features.flatten()`의 기본 시작 축은 0이다. 배치 축까지 합친다.

| 원래 shape | `flatten(start_dim=1)` | `flatten()` |
| --- | --- | --- |
| `[2,16,8,8]` | `[2,1024]` | `[2048]` |
| `[1,16,8,8]` | `[1,1024]` | `[1024]` |

`N=2`에서는 마지막 축이 2,048이 되어 `Linear(1024,5)`와 맞지 않는다. `N=1`에서는 마지막 축이 우연히 1,024이므로 Linear 계산은 가능하지만 결과가 `[5]`가 된다. 우리가 요구한 배치 포함 출력 `[1,5]`와 다르다.

Flatten이 이미지의 의미를 판단하는 것은 아니다. 원래 이미지별 구분을 나타내던 <strong>배치 축이 사라진 것</strong>이다. 그래서 한 장뿐 아니라 두 장을 넣는 검사도 필요하다.

## 13. 두 블록과 Linear를 하나의 모델로 연결한다

전체 모델은 앞에서 정의한 블록 두 개와 Linear 한 개를 속성으로 갖는다. `forward`에는 계산 순서만 적는다.

```python
class MySmallVGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = MyVGGBlock(3, 8)
        self.block2 = MyVGGBlock(8, 16)

        # 마지막 특징 맵의 이미지당 데이터 수는 16×8×8이다.
        self.fc = nn.Linear(16 * 8 * 8, 5)

    def forward(self, x):
        result = self.block1(x)
        result = self.block2(result)
        result = result.flatten(start_dim=1)
        return self.fc(result)  # 아직 확률이 아니라 점수다.


# 클래스 정의와 별도로 실제 객체를 만든다.
torch.manual_seed(16)
small_model = MySmallVGG().to(device)
```

중간 shape와 전체 `forward`의 결과를 함께 검사해 보자. `logits`는 분류기의 원래 점수를 가리키는 이름이며, 바로 다음 절에서 확률과 구분한다.

```python
with torch.no_grad():
    for batch_size in (2, 1):
        images = torch.randn(batch_size, 3, 32, 32, device=device)

        feature1 = small_model.block1(images)
        feature2 = small_model.block2(feature1)
        flat_features = feature2.flatten(start_dim=1)
        manual_logits = small_model.fc(flat_features)

        # images부터 전체 모델을 다시 계산한다.
        logits = small_model(images)

        assert tuple(feature1.shape) == (batch_size, 8, 16, 16)
        assert tuple(feature2.shape) == (batch_size, 16, 8, 8)
        assert tuple(flat_features.shape) == (batch_size, 1024)
        assert tuple(logits.shape) == (batch_size, 5)
        assert torch.allclose(
            logits, manual_logits, atol=1e-6, rtol=1e-4
        )

        print(
            "block1:", tuple(feature1.shape),
            "block2:", tuple(feature2.shape),
            "flatten:", tuple(flat_features.shape),
            "logits:", tuple(logits.shape),
        )
```

고정적으로 확인할 수 있는 shape는 다음과 같다.

```text
block1: (2, 8, 16, 16) block2: (2, 16, 8, 8) flatten: (2, 1024) logits: (2, 5)
block1: (1, 8, 16, 16) block2: (1, 16, 8, 8) flatten: (1, 1024) logits: (1, 5)
```

이미지 개수가 2에서 1로 바뀌어도 한 장의 특징값은 1,024개로 같다. Linear는 같은 가중치를 각 이미지의 특징 벡터에 적용한다. 여기서 <strong>특징 벡터</strong>는 한 이미지의 특징값을 한 축에 나열한 배열이다.

### 입력의 높이·너비를 바꾸면 어떤 조건을 확인할까

이 모델은 `32×32` 입력을 기준으로 Linear를 정했다. 하지만 “32×32가 아닌 모든 입력은 오류”라는 뜻은 아니다. 예를 들어 `33×33`도 두 번의 현재 Pool을 거치면 `16×16`, `8×8`이 된다.

정확한 조건은 <strong>Linear에 도착할 때 이미지마다 특징값이 1,024개여야 한다</strong>는 것이다. 배치 `N`은 바뀔 수 있지만, 높이·너비를 임의로 바꿔도 항상 이 조건이 성립하는 것은 아니다.

설정을 바꾸면 특징값 수를 다시 계산해야 한다. 예를 들어 별도의 흑백 모델을 `[N,1,28,28] → [N,16,14,14] → [N,32,7,7]`로 구성하면 `32×7×7=1,568`개다. 그 모델의 10분류 Linear는 `Linear(1568,10)`이 된다. 이번 RGB 모델의 1,024와는 마지막 출력 크기부터 다르다.

## 14. Linear의 가중치, 로짓, 확률을 나눈다

`Linear(1024,5)`는 입력값 1,024개를 사용해 출력 점수 다섯 개를 만든다. 출력 점수 하나를 계산하려면 입력 1,024개에 각각 가중치를 곱해 더하고, 그 출력의 편향 하나를 더한다. 서로 다른 가중치 묶음 다섯 개가 이 계산을 각각 수행한다.

PyTorch는 Linear의 가중치를 `[out_features, in_features]` 순서로 저장한다. 따라서 이 층의 `weight.shape`는 `[5,1024]`, 편향은 `[5]`다.[^linear]

| 데이터 또는 파라미터 | shape | 역할 |
| --- | --- | --- |
| 펼친 입력 `X` | `[N,1024]` | 이미지마다 특징값 1,024개 |
| `weight` | `[5,1024]` | 출력 다섯 개가 각각 사용할 가중치 |
| `weight.T` | `[1024,5]` | 행렬곱에 맞게 두 축을 바꾼 가중치 |
| `bias` | `[5]` | 각 출력에 더하는 편향 |
| 출력 | `[N,5]` | 이미지마다 점수 다섯 개 |

식으로 적으면 `X @ weight.T + bias`다. `@`는 행렬곱이고, 여기의 `.T`는 2차원 가중치 행렬의 두 축을 바꾼다. 편향 다섯 개는 각 이미지의 점수에 동일하게 더해진다.

```python
with torch.no_grad():
    # 바로 앞 검사의 flat_features를 사용한다.
    direct_scores = (
        flat_features @ small_model.fc.weight.T + small_model.fc.bias
    )
    layer_scores = small_model.fc(flat_features)
    assert torch.allclose(
        direct_scores, layer_scores, atol=1e-6, rtol=1e-4
    )

print(tuple(small_model.fc.weight.shape))  # (5, 1024)
print(small_model.fc.weight.numel())       # 5120
print(small_model.fc.bias.numel())         # 5
```

이전에 직접 행렬을 정의할 때 `W=[입력 수, 출력 수]`로 놓았다면 PyTorch의 저장 순서와 반대다. 행렬곱 식에서 `weight.T`를 사용하는 이유가 여기에 있다.

### Linear는 자동으로 확률을 출력하지 않는다

<strong>로짓(logits)</strong>은 이 분류 모델에서 Linear가 만든 원래 점수다. 음수일 수 있고, 점수를 모두 더한 값이 1일 필요도 없다.

<strong>Softmax</strong>는 한 이미지의 클래스별 점수를 0~1 사이의 값으로 바꾸고 합계가 1이 되게 한다. 모델이 여러 클래스에 부여하는 확률 형태의 값이지만, 학습하지 않은 모델에 적용한다고 유용한 예측이나 신뢰할 수 있는 확률이 생기는 것은 아니다.[^softmax]

```python
small_model.eval()

with torch.no_grad():
    prediction_images = torch.randn(2, 3, 32, 32, device=device)
    logits = small_model(prediction_images)     # [2, 5]

    # 축 1은 각 이미지 안의 클래스 점수 다섯 개다.
    probabilities = torch.softmax(logits, dim=1)

    # 가장 큰 점수가 놓인 클래스 번호를 고른다.
    predicted_classes = logits.argmax(dim=1)   # [2]

print(probabilities.sum(dim=1))  # 약 tensor([1., 1.])
print(predicted_classes)        # 이미지별 클래스 번호, 범위 0~4
```

`argmax`는 지정한 축에서 가장 큰 값의 인덱스를 반환한다. 가장 큰 점수의 클래스만 고른다면 Softmax 없이 로짓에 바로 적용해도 된다. Softmax가 점수의 대소 순서를 유지하기 때문이다.

출력 0번이 어떤 종류를 뜻하는지는 데이터의 정답 번호와 학습 과정으로 연결된다. `Linear(1024,5)`가 스스로 다섯 종류의 이름을 정하는 것은 아니다.

일반적인 정수 정답 분류에서 `nn.CrossEntropyLoss`에는 <strong>Softmax를 미리 적용하지 않은 로짓</strong>을 전달한다. 내부 계산에 필요한 정규화가 포함되기 때문이다. 현재 모델의 입력 형식은 다음과 같다.[^crossentropy]

| 손실 함수에 전달하는 값 | shape | 자료형과 값 |
| --- | --- | --- |
| `logits` | `[N,5]` | 클래스별 실수 점수 |
| `labels` | `[N]` | `torch.long`, 각 값은 0~4 |

### 전체 파라미터는 9,421개다

이번 작은 RGB 모델은 Conv와 Linear의 기본 편향을 포함한다. ReLU, Pool, Flatten에는 학습 파라미터가 없다.

| 층 | 가중치와 편향 계산 | 파라미터 수 |
| --- | --- | ---: |
| 첫 블록 Conv1, `3→8` | `8×3×3×3 + 8` | 224 |
| 첫 블록 Conv2, `8→8` | `8×8×3×3 + 8` | 584 |
| 둘째 블록 Conv1, `8→16` | `16×8×3×3 + 16` | 1,168 |
| 둘째 블록 Conv2, `16→16` | `16×16×3×3 + 16` | 2,320 |
| Linear, `1024→5` | `5×1024 + 5` | 5,125 |
| 합계 | `224+584+1168+2320+5125` | **9,421** |

```python
total_parameters = sum(p.numel() for p in small_model.parameters())
assert total_parameters == 9421
print(total_parameters)
```

이미지 한 장의 Flatten 데이터 수 1,024, 최종 점수 수 5, 모델의 전체 파라미터 수 9,421은 서로 다른 대상을 세는 값이다. 앞서 예로 든 `1→16→32` 흑백 블록 구성과 `Linear(1568,10)`은 기본 편향을 포함하면 전체 파라미터가 32,058개다. 모델의 채널이나 출력 크기를 바꾸면 이 계산도 다시 해야 한다.

## 15. 잔차 연결은 같은 위치의 두 숫자를 더한다

지금까지는 한 층의 출력을 다음 층으로 순서대로 전달했다. <strong>잔차 연결(residual connection)</strong>은 입력이 일부 층을 거쳐 계산되는 경로와, 그 층들을 건너 합산 지점으로 전달되는 경로를 함께 둔다.

Conv 등으로 입력을 변환하는 쪽을 <strong>주 경로</strong>, 일부 층을 건너 전달하는 쪽을 <strong>shortcut</strong>이라고 부른다. 입력을 그대로 전달하는 shortcut에서는 주 경로 결과와 원래 입력을 같은 위치끼리 더한다.

주 경로 계산을 수학적으로 `F(x)`라고 쓰면 이번 블록은 다음과 같다.

<div class="cnn-vgg-math">
$$y=\operatorname{ReLU}(F(x)+x)$$
</div>

이 수학식의 `F`는 주 경로를 나타내는 기호다. Python의 `F.relu`에서 `F`는 `torch.nn.functional`의 별칭이다. 글자는 같아도 가리키는 대상이 다르다.

### 먼저 사람이 정한 숫자로 덧셈을 확인한다

아래의 `residual_f`는 아직 Conv가 계산한 결과가 아니다. 어떤 덧셈인지 눈으로 확인하려고 직접 정한 값이다.

```python
residual_x = torch.tensor(
    [[2.0, -1.0],
     [0.0,  3.0]],
    device=device,
)
residual_f = torch.tensor(
    [[-1.0,  2.0],
     [ 1.0, -4.0]],
    device=device,
)

residual_sum = residual_f + residual_x
residual_y = F.relu(residual_sum)

print(residual_sum)
# tensor([[ 1.,  1.],
#         [ 1., -1.]])

print(residual_y)
# tensor([[1., 1.],
#         [1., 0.]])

print(F.relu(residual_f))  # shortcut이 없는 경우와 비교
# tensor([[0., 2.],
#         [1., 0.]])
```

왼쪽 위는 `-1+2=1`, 오른쪽 아래는 `-4+3=-1`이다. 합산 후 ReLU를 적용하므로 마지막의 -1은 0이 된다. 채널을 이어 붙이거나 행렬곱을 한 것이 아니다.

### 실제 Conv 두 개로 주 경로를 만든다

입력이 `[N,16,14,14]`라면 두 Conv 모두 채널 16, 커널 3, stride 1, padding 1로 둔다. 그러면 주 경로 결과도 `[N,16,14,14]`여서 원래 입력과 대응한다.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        # x는 원래 입력이다. 같은 데이터·채널·행·열의 값끼리 더한다.
        return F.relu(out + x)


block = ResidualBlock(16).to(device)
sample = torch.randn(1, 16, 14, 14, device=device)

with torch.no_grad():
    transformed = block.conv2(F.relu(block.conv1(sample)))
    block_output = block(sample)

    assert transformed.shape == sample.shape
    assert torch.allclose(
        block_output,
        F.relu(transformed + sample),
        atol=1e-6,
        rtol=1e-4,
    )

index = (0, 0, 0, 0)  # 첫 데이터, 첫 채널, 첫 행, 첫 열
print("입력값:", sample[index].item())
print("주 경로 값:", transformed[index].item())
print("합산 후 ReLU:", block_output[index].item())
```

출력 한 칸은 `max(주 경로의 그 칸 값 + 입력의 그 칸 값, 0)`이다. `out`을 계산하는 동안 `x`를 다른 값으로 덮어쓰지 않았으므로 마지막 덧셈에서 원래 입력을 사용할 수 있다.

### 왜 잔차를 학습한다고 부르는가

합산 지점에서 원하는 변환을 `H(x)`라고 하자. 입력 `x`를 그대로 더해 주면 주 경로는 `H(x)−x`에 해당하는 차이를 계산하는 형태로 구성할 수 있다. 이 차이를 <strong>잔차(residual)</strong>라고 부른다.

원래 ResNet 연구는 층을 더 깊게 쌓았을 때 훈련 오차까지 나빠지는 학습의 어려움을 다뤘다. shortcut과 잔차를 더하는 구조는 깊은 모델의 최적화를 돕는다. 다만 모든 학습 문제나 기울기 소실을 완전히 해결한다고 단정할 수는 없다. 현재의 작은 예제는 잔차 덧셈이 구현되었는지를 확인할 뿐, 성능 향상을 측정하지 않는다.[^resnet]

또 `return F.relu(out)`과 `return F.relu(out + x)`는 같은 shape를 반환할 수 있다. 따라서 <strong>shape가 같다는 검사만으로 shortcut이 사용되었는지는 알 수 없다.</strong> `forward`의 실제 연산을 읽고, 위처럼 합산 결과와 비교해야 한다. 일부 입력에서는 두 식의 값도 우연히 같을 수 있다.

## 16. 두 경로의 크기가 다르면 목표 출력에 맞춘다

이번에는 블록의 출력 채널을 늘리면서 높이·너비를 줄이고 싶다고 하자.

`[N,16,14,14] → [N,32,7,7]`

주 경로의 첫 Conv를 `16→32`, `3×3`, `stride=2`, `padding=1`로 두면 이 크기를 만들 수 있다. 다음 Conv는 `32→32`로 두고 공간 크기를 유지한다.

문제는 원래 입력 `x`가 여전히 `[N,16,14,14]`라는 것이다. 주 경로의 `[N,32,7,7]`과 같은 위치끼리 더할 수 없다. shortcut에도 `16→32`, `1×1`, `stride=2`인 Conv를 두면 두 경로를 같은 출력 크기로 맞출 수 있다.

학습 가능한 변환으로 shortcut의 채널과 크기를 맞추는 방식을 <strong>projection shortcut</strong>이라고 한다.[^resnet]

<figure class="cnn-vgg-figure">
<img src="/cnn-vgg-residual-pytorch/figure-residual-paths-ko.svg" alt="크기가 유지되는 잔차 블록은 입력을 그대로 더한다. 채널을 16에서 32로 늘리고 공간을 14에서 7로 줄이는 블록은 shortcut에 stride 2의 1×1 Conv를 적용해 두 경로를 맞춘다." loading="lazy">
<figcaption>합산 지점의 shape를 먼저 정한다. 주 경로와 shortcut이 각각 그 shape의 텐서를 만들도록 구성한다.</figcaption>
</figure>

```python
class ProjectionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1
        )

        # 1×1 Conv로 채널을 바꾸고 stride=2로 공간 크기를 줄인다.
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1, stride=2
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        shortcut_result = self.shortcut(x)
        return F.relu(out + shortcut_result)


projection_block = ProjectionResidualBlock(16, 32).to(device)
projection_input = torch.randn(1, 16, 14, 14, device=device)

with torch.no_grad():
    main_path = projection_block.conv2(
        F.relu(projection_block.conv1(projection_input))
    )
    shortcut_path = projection_block.shortcut(projection_input)
    projection_output = projection_block(projection_input)

    assert tuple(main_path.shape) == (1, 32, 7, 7)
    assert main_path.shape == shortcut_path.shape
    assert torch.allclose(
        projection_output,
        F.relu(main_path + shortcut_path),
        atol=1e-6,
        rtol=1e-4,
    )

print(tuple(main_path.shape))          # (1, 32, 7, 7)
print(tuple(shortcut_path.shape))      # (1, 32, 7, 7)
print(tuple(projection_output.shape))  # (1, 32, 7, 7)
```

주 경로와 shortcut은 서로 다른 Conv 객체를 가지므로 가중치도 별개다. 두 경로는 같은 입력에서 출발하지만 다른 계산을 한다.

### 왜 원래 입력 쪽을 바꾸는가

반드시 입력 쪽만 바꿔야 하는 법칙은 없다. 주 경로 결과의 공간 크기를 `7×7→14×14`로 키우고 채널을 `32→16`으로 바꿔 원래 입력과 더할 수도 있다.

하지만 그렇게 만든 블록의 출력은 `[N,16,14,14]`다. 처음에 목표로 정한 `[N,32,7,7]`과 다른 설계가 된다. 공간 크기를 키운다고 앞에서 사라진 입력 정보가 정확히 복원되는 것도 아니다.

따라서 <strong>다음 층으로 전달할 목표 shape를 먼저 정하고, 합산할 두 경로를 그 shape에 맞춘다.</strong> 크기를 유지할 때는 입력을 그대로 전달하고, 이번처럼 채널 증가와 공간 축소가 필요하면 shortcut도 변환한다.

PyTorch의 덧셈은 일부 다른 shape 사이에서도 <strong>브로드캐스팅</strong>을 허용한다. 크기 1인 축 등을 다른 텐서에 맞춰 반복하는 것처럼 계산하는 규칙이다. 하지만 현재 잔차 블록의 의도는 같은 데이터·채널·위치를 직접 대응하는 것이므로, 덧셈이 오류 없이 실행된다는 사실만 보지 말고 전체 shape가 일치하는지 확인한다.

여기의 두 잔차 블록은 연결을 설명하기 위해 BatchNorm을 생략한 교육용 구현이다. 실제 TorchVision ResNet은 BatchNorm을 포함하고 일부 Conv에서 편향을 사용하지 않는 등 세부 구성이 다르다.[^resnet-code]

## 17. 1×1 Conv에도 여러 채널을 계산하는 가중치가 있다

`1×1`이라는 표현은 <strong>채널별 커널의 공간 크기</strong>다. `Conv2d(16,32,1)`이라는 층 전체에 가중치 하나만 있다는 뜻이 아니다.

| 대상 | 계산 | 파라미터 수 |
| --- | --- | ---: |
| 입력 채널 하나에 대응하는 커널 | `1×1` | 1 |
| 출력 채널 하나를 만드는 필터 | `16×1×1` | 16 |
| 출력 채널 32개를 만드는 전체 층 | `32×16×1×1` | 512 |
| 기본 편향 | 출력 채널마다 하나 | 32 |

가중치 shape는 `[32,16,1,1]`이다. `weight[0]`은 첫 출력 채널을 만드는 가중치 16개를 갖고, `weight[0,0]`은 그중 입력 채널 0에 대응하는 `1×1` 커널이다.

한 위치의 출력값은 같은 공간 위치에 있는 입력 채널 16개의 값을 각각 다른 가중치와 곱해 더하고 편향을 더해서 만든다. <strong>`1×1` Conv도 채널 방향으로는 여러 값을 조합한다.</strong> 주변의 `3×3` 공간을 읽는 계산과 다른 점은 공간 범위다.

<figure class="cnn-vgg-figure">
<img src="/cnn-vgg-residual-pytorch/figure-pointwise-ko.svg" alt="1×1 Conv의 필터 하나는 같은 위치의 입력 채널 16개를 가중치 16개와 곱해 더한다. 이런 필터 32개가 가중치 512개와 편향 32개를 가지며 출력 채널 32개를 만든다." loading="lazy">
<figcaption>공간 크기 1×1과 입력 채널 수 16은 서로 다른 축의 정보다.</figcaption>
</figure>

### stride=2는 어떤 위치를 읽는가

현재 shortcut은 `kernel_size=1, stride=2, padding=0`이다. 높이 14의 입력에서 행 인덱스 `0,2,4,6,8,10,12`를 읽고, 너비도 같은 방식으로 읽는다. 각 방향에 일곱 위치가 있으므로 출력은 `7×7`이다.

1×1 연산이 공간 크기를 항상 유지하는 것은 아니다. stride 1이면 현재 조건에서 유지되고, stride 2이면 계산할 위치를 두 칸 간격으로 선택한다.

```python
print(tuple(projection_block.shortcut.weight.shape))  # (32, 16, 1, 1)
print(projection_block.shortcut.weight.numel())       # 512
print(projection_block.shortcut.bias.numel())         # 32
```

### 가중치는 1로 고정되어 있는가

기본 Conv의 가중치는 초기화된 값으로 시작해 학습 과정에서 조정된다. 커널 크기가 1이라고 가중치 값도 1인 것은 아니다.

만약 모든 가중치를 1, 모든 편향을 0으로 직접 정한다면, 각 출력 채널은 같은 위치의 입력 채널 16개 값을 모두 더하게 된다. 그 경우 출력 채널들이 같은 합계를 만들 뿐, 원래 입력을 그대로 복사하는 연산이 아니다.

반대로 가중치를 모두 0으로 만들면 입력값의 기여가 사라지고 편향만 남는다. 편향도 0이어야 출력 전체가 0이 된다. shortcut이 원래 입력을 그대로 전달하는 경우와, 학습 가능한 1×1 Conv를 거치는 경우를 구분해야 하는 이유다.

## 18. no_grad와 eval은 서로 다른 설정이다

지금까지 계산 검산에 `torch.no_grad()`를 사용했고, 분류 점수를 확인할 때는 `small_model.eval()`도 호출했다. 둘은 같은 기능이 아니다.

먼저 <strong>자동 미분(autograd)</strong>은 수행한 연산의 관계를 이용해 변화율인 <strong>기울기(gradient)</strong>를 계산하는 기능이다. 학습에서는 손실이 각 파라미터의 변화에 얼마나 민감한지를 이 기울기로 구한다.

이를 위해 순전파 중에 어떤 텐서가 어떤 연산으로 만들어졌는지 기록한다. 이 연결된 기록을 <strong>계산 그래프(computational graph)</strong>라고 한다. 순전파는 입력에서 출력을 구하는 계산이고, 역전파는 그 관계를 거슬러 손실의 기울기를 구하는 계산이다.[^autograd]

### no_grad는 해당 계산의 역전파 기록을 끈다

`with torch.no_grad():` 안에서도 합성곱, ReLU, Pool의 출력값은 정상적으로 계산한다. 다만 일반적인 연산을 역전파용 그래프에 기록하지 않는다. 학습에 쓰지 않을 검사나 추론에서 불필요한 기록과 관련 메모리 사용을 줄일 수 있다.[^nograd]

여기서 `with torch.no_grad():`는 블록에 들어갈 때 기울기 기록 설정을 끄고, 나갈 때 이전 상태로 복구하도록 한다. `with` 자체는 이런 진입·종료 처리를 해당 객체에 맡기는 Python 문법이다.

| `no_grad`가 하는 일 | 별도로 관리해야 하는 일 |
| --- | --- |
| 블록 안의 일반적인 역전파용 연산 기록을 끈다. | 기존 파라미터의 `requires_grad`를 영구적으로 바꾸는 것 |
| 현재 입력의 출력값은 계속 계산한다. | 기존 `.grad`를 비우는 것 |
| 블록을 나가면 이전 기록 설정으로 돌아간다. | 모델을 평가 모드로 전환하는 것 |

`requires_grad`는 해당 텐서의 기울기가 필요한지를 나타내는 설정이고, `.grad`는 계산된 기울기를 저장하는 속성이다. 기존 가중치를 지우거나 새로 초기화하는 것도 `no_grad`의 동작이 아니다.

새 `Parameter`를 만드는 일부 함수에는 예외가 있으므로 “블록 안에서 만들어지는 모든 텐서는 무조건 `requires_grad=False`다”라고 외우지 않는다. 여기서 확인하는 것은 이미 만들어진 모델로 수행하는 일반적인 순전파다.[^nograd]

### eval은 층을 평가 방식으로 실행하게 한다

`model.eval()`은 `model.train(False)`와 같은 평가 모드 전환이다. 등록된 하위 모듈에도 적용되며, 다시 `model.train()`을 호출하기 전까지 유지된다. 하지만 자동 미분 기록을 끄지는 않는다.[^module]

모드에 따라 동작이 달라지는 대표적인 층은 다음과 같다.

| 층 | 학습 모드 `train` | 평가 모드 `eval` |
| --- | --- | --- |
| Dropout | 일부 원소를 무작위로 0으로 만들고 남은 값의 크기를 보정한다. | 입력을 그대로 통과시킨다. |
| BatchNorm | 기본 설정에서 현재 배치의 통계로 정규화하고 저장 통계를 갱신한다. | 기본 설정에서 저장된 통계를 사용한다. |

<strong>정규화</strong>는 이 문맥에서 평균과 분산을 이용해 값의 분포를 조정하는 계산이다. BatchNorm의 `track_running_stats=False`처럼 저장 통계를 사용하지 않는 설정에서는 위의 기본 동작과 달라진다.[^dropout][^batchnorm]

현재 작은 모델의 Conv, ReLU, MaxPool, Linear는 자체 계산 방식이 train/eval에 따라 달라지지 않는다. 그래도 평가 때 `eval()`을 명시해 두면 나중에 모드에 의존하는 층을 추가했을 때 실행 목적이 분명해진다.

### 두 설정은 독립적으로 조합된다

| 실행 조합 | 역전파용 기록 | Dropout·기본 BatchNorm 동작 |
| --- | --- | --- |
| `train` + 일반 실행 | 필요 조건에서 기록 | 학습 방식 |
| `train` + `no_grad` | 해당 일반 연산 기록 안 함 | 학습 방식 |
| `eval` + 일반 실행 | 필요 조건에서 기록 | 평가 방식 |
| `eval` + `no_grad` | 해당 일반 연산 기록 안 함 | 평가 방식 |

기울기 추적이 켜져 있고, 기울기를 요구하는 입력이나 파라미터가 미분 가능한 연산에 참여하면 기록이 만들어진다. <strong>입력 이미지의 `requires_grad`가 False여도 모델 가중치가 True이면 출력은 추적 대상이 될 수 있다.</strong>

```python
probe_input = torch.randn(1, 3, 32, 32, device=device)
small_model.eval()

tracked_output = small_model(probe_input)

with torch.no_grad():
    untracked_output = small_model(probe_input)

print(tracked_output.requires_grad)        # True: eval만으로 추적이 꺼지지 않는다.
print(untracked_output.requires_grad)      # False
print(small_model.fc.weight.requires_grad) # True: 기존 가중치 설정은 유지된다.
```

`model.train()` 역시 모드만 바꾼다. 이것만 호출해서 손실을 계산하거나 가중치가 갱신되지는 않는다.

### 텐서를 조작할 때마다 no_grad가 필요한 것은 아니다

판단 기준은 그 연산이 이후 학습에서 미분되어야 하는지다. 예를 들어 학습 중 `features = features * 2`를 계산하고 다음 층으로 넘긴다면, 이 곱셈도 손실에 영향을 주는 계산이므로 보통 기록해야 한다. 이를 `no_grad`로 감싸 새 결과를 만들면 그 연산을 통한 앞 층의 기울기 연결이 끊길 수 있다.

반면 학습 단계 사이에 가중치를 직접 수정하는 작업은 보통 미분 기록에 넣지 않는다.

```python
# 수동 가중치 수정의 사용 형식이다. 본 실습에서는 실행하지 않는다.
# with torch.no_grad():
#     layer.weight.mul_(0.9)
```

`mul_` 끝의 `_`는 기존 텐서를 직접 수정한다는 뜻이다. 하지만 `no_grad`가 어느 시점의 수정이든 안전하게 만들어 주는 것은 아니다. 순전파 후 역전파가 끝나기 전에 필요한 값을 바꾸면 오류나 잘못된 계산의 원인이 될 수 있다.[^autograd]

일반적인 `optimizer.step()`은 파라미터 갱신에 필요한 처리를 하므로 사용자가 매번 `no_grad`로 감쌀 필요는 없다. `print(layer.weight)`나 `tensor.shape`처럼 값을 확인한다는 이유만으로 이 설정이 필요한 것도 아니다.

추론에는 `torch.inference_mode()`도 있다. 추가 관리 비용을 줄이지만 그 안에서 만든 텐서를 이후 자동 미분 계산에 사용하는 데 더 엄격한 제약이 있으므로 `no_grad`와 완전히 같은 기능으로 취급하지 않는다.[^autograd]

## 19. 구조 검사 다음에 실제 학습이 이어진다

무작위 입력으로 shape가 맞는지 확인한 것은 <strong>구조 검사</strong>다. 분류를 잘하는지 확인하려면 실제 데이터와 정답, 학습 과정, 별도의 평가 과정이 필요하다.

지금까지 출력 점수가 나온 이유는 초기 가중치만으로도 순전파 계산을 수행할 수 있기 때문이다. 모델을 생성하거나 `eval()`을 호출했다는 사실이 학습 완료를 뜻하지는 않는다.

### 실제 학습에서 추가되는 작업

<strong>손실(loss)</strong>은 모델의 예측과 정답을 비교해 만든 학습 기준값이다. <strong>옵티마이저(optimizer)</strong>는 계산된 기울기를 사용해 파라미터를 갱신하는 객체다.

| 순서 | 코드 | 역할 |
| --- | --- | --- |
| 1 | `model.train()` | 학습 방식으로 실행할 모드를 선택 |
| 2 | `optimizer.zero_grad(...)` | 이전 배치의 기울기 제거 |
| 3 | `logits = model(images)` | 순전파로 원래 점수 계산 |
| 4 | `loss = criterion(logits, labels)` | 예측과 정답으로 손실 계산 |
| 5 | `loss.backward()` | 파라미터별 손실의 기울기 계산 |
| 6 | `optimizer.step()` | 계산한 기울기로 파라미터 갱신 |

`criterion`은 손실 함수 객체에 붙인 변수 이름이다. 현재 다섯 클래스 분류에서는 `nn.CrossEntropyLoss()`를 사용할 수 있다.[^crossentropy]

아래 함수는 학습 한 배치의 순서를 나타낸다. <strong>이 글과 전체 실행 파일에서는 함수를 정의만 하고 호출하지 않는다.</strong>

```python
def train_one_batch(model, images, labels, optimizer, criterion):
    model.train()

    # 이전 기울기를 제거한다. None으로 설정해 다음 계산을 준비한다.
    optimizer.zero_grad(set_to_none=True)

    # 학습에 필요한 계산은 no_grad로 감싸지 않는다.
    logits = model(images)
    loss = criterion(logits, labels)

    loss.backward()   # 기울기를 계산한다. 아직 파라미터 갱신은 아니다.
    optimizer.step()  # 기울기를 사용해 파라미터를 갱신한다.

    return loss.item()  # 기록용 Python 숫자


# 실제 데이터와 정답을 준비한 뒤 사용할 객체의 생성 예다.
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(small_model.parameters(), lr=0.01)
# loss_value = train_one_batch(
#     small_model, images, labels, optimizer, criterion
# )
```

`images`는 모델과 같은 장치의 `[N,3,32,32]` 텐서, `labels`는 같은 장치의 `[N]` 정수 텐서여야 한다. 정답은 `torch.long` 자료형이며 0~4 범위다. 주석의 `lr`은 파라미터 갱신 크기를 조절하는 <strong>학습률(learning rate)</strong>이다.

PyTorch의 역전파는 기울기를 누적할 수 있으므로 일반적인 배치별 학습에서는 이전 기울기를 먼저 비운다. `set_to_none=True`는 0으로 채운 텐서 대신 `.grad`를 `None`으로 설정하는 방식이다. `None`은 아직 그 값이 없다는 Python의 특별한 값이다.[^optim]

평가나 예측에는 보통 `model.eval()`과 `with torch.no_grad():`를 함께 사용한다. 학습을 다시 시작할 때는 `model.train()`으로 전환하고, 학습 순전파를 `no_grad` 밖에서 수행한다.

### 과제 코드의 ready와 예외 처리는 무엇인가

수업용 코드에 `vgg_ready = False`나 `small_model_ready = False`가 있다면, 검사를 통과했는지 표시하려고 만든 일반 Python 변수다. PyTorch가 자동으로 관리하는 상태가 아니다. `small_model = None`도 아직 실제 모델 객체를 저장하지 않았다는 표시다.

`raise NotImplementedError(...)`는 미구현 상태를 알리기 위해 일부러 예외를 발생시키는 문장이다. `__init__` 안에 남아 있으면 객체 생성이 중단된다. `return` 뒤에 있으면 그 경로에서는 실행되지 않지만 완성된 코드에서는 제거한다.

`try` 블록의 `except NotImplementedError`는 그 예외만 처리한다. shape 불일치로 생기는 `RuntimeError`, 검사 실패의 `AssertionError`, 잘못된 속성 접근의 `AttributeError`까지 모두 잡는 것은 아니다.[^python]

이 글의 독립 실행 파일에서는 이런 과제 진행용 조건 대신 완성된 모델을 만들고 검사한다. 검사에 실패하면 원인을 볼 수 있도록 오류를 남긴다.

### 검사가 보장하는 범위도 확인한다

이번 검산은 출력 크기, 파라미터 수, 수동 계산과의 일치를 확인한다. 실제 데이터의 분류 정확도나 일반화 성능을 증명하지 않는다. 또한 학습용 `assert`는 Python을 최적화 옵션으로 실행하면 생략될 수 있으므로 필수 사용자 입력 검증을 대신하는 장치로 쓰지 않는다.[^check]

중간 층을 따로 실행한 결과와 전체 `forward`를 비교하는 방식은 현재의 Conv·ReLU·Pool 구성에 맞춘 것이다. Dropout이나 학습 중 BatchNorm처럼 상태나 무작위 동작이 개입하는 층을 추가했다면 평가 모드와 반복 실행의 조건도 함께 확인해야 한다.

## 20. Rust에서도 같은 개수와 호출 관계를 확인한다

채널 수, 가중치 개수, 이미지 한 장의 특징값 개수는 프로그래밍 언어가 바뀌어도 같다. 함수 자체를 변수에 저장하는 것과 함수를 호출해 반환값을 저장하는 것도 구분할 수 있다.

아래 Rust 코드는 표준 기능으로 숫자와 shape 계산만 확인한다. PyTorch 모델처럼 CNN 전체나 자동 미분을 실행하지 않는다. 실제 신경망을 실행하려면 그런 기능을 제공하는 라이브러리가 필요하다.

코드를 읽기 전에 사용한 자료형을 정리하자.

| Rust 표현 | 의미 |
| --- | --- |
| `f32` | 32비트 부동소수점 실수 자료형 |
| `usize` | 배열 크기·인덱스 등에 사용하는 부호 없는 정수 자료형 |
| `fn(f32) -> f32` | `f32` 하나를 받아 `f32`를 반환하는 함수 포인터의 자료형 |
| `let` | 변수 바인딩 생성 |
| `assert_eq!(a, b)` | 두 값이 같은지 검사하는 매크로 |
| `println!` | 화면에 값을 출력하는 매크로 |

함수 포인터는 호출할 함수를 값으로 보관하는 방식이다. 아래에서 `relu`만 적으면 함수 자체를 가리키고, `relu_function(-2.0)`처럼 괄호와 인자를 붙이면 실행한 결과를 얻는다.[^rust]

```rust
fn relu(x: f32) -> f32 {
    // 음수이면 0, 아니면 원래 값을 반환한다.
    x.max(0.0)
}

fn main() {
    // Python의 relu_function = F.relu와 같은 구분이다.
    let relu_function: fn(f32) -> f32 = relu;
    let result = relu_function(-2.0);
    assert_eq!(result, 0.0);

    // Conv2d(4, 6, 3): 편향을 제외한 전체 가중치 수.
    let input_channels: usize = 4;
    let output_filters: usize = 6;
    let weights = output_filters * input_channels * 3 * 3;
    assert_eq!(weights, 216);

    // 1×1 Conv도 입력 채널 16개와 출력 필터 32개를 계산한다.
    let projection_weights: usize = 32 * 16 * 1 * 1;
    assert_eq!(projection_weights, 512);

    // Pool 두 번 뒤의 공간 크기로 Linear 입력 수를 계산한다.
    let height: usize = 32 / 2 / 2;
    let width: usize = 32 / 2 / 2;
    let in_features: usize = 16 * height * width;
    assert_eq!(in_features, 1024);

    for batch_size in [2_usize, 1_usize] {
        let flat_shape = [batch_size, in_features];
        let score_shape = [batch_size, 5];
        println!("Flatten: {flat_shape:?}, 점수: {score_shape:?}");
    }

    // 잔차 연결의 한 위치 계산만 확인한다.
    let original: f32 = 2.0;
    let transformed: f32 = -1.0;
    let residual_result = relu(transformed + original);
    assert_eq!(residual_result, 1.0);
}
```

`x.max(0.0)`는 두 값 중 큰 값을 반환한다. 함수 본문의 마지막 식에 세미콜론이 없으므로 그 결과가 반환값이 된다. 출력 문자열의 `{flat_shape:?}`는 배열을 확인하기 쉬운 디버그 형식으로 표시한다.

`32 / 2 / 2`는 여기서 정수 나눗셈이고 결과가 8이다. 마지막 특징값 수는 `16×8×8`로 계산하며, 배치 크기는 별도의 첫 축으로 남긴다. 코드가 만든 `flat_shape`와 `score_shape`는 크기를 기록한 정수 배열이다. 실제 이미지 데이터 텐서를 만든 것은 아니다.

## 21. 전체 코드를 실행하고 무엇을 확인했는지 정리한다

본문의 개별 계산을 한 번에 실행하려면 [전체 Python 코드](/cnn-vgg-residual-pytorch/examples-ko.py)를 사용한다. PyTorch를 사용할 수 있는 Python 환경에서 다음처럼 실행한다.

```bash
python examples-ko.py
```

파일에는 앞의 네 모델 클래스와 `train_one_batch`가 정의되어 있다. 직접 실행하면 `run_checks()`가 작은 입력으로 구조와 계산 결과를 검사한다. 데이터셋을 내려받지 않으며, 학습 함수나 `optimizer.step()`을 호출하지 않는다.

| 검사 | 확인하는 값 또는 관계 |
| --- | --- |
| 합성곱 한 칸 | 패치와 가중치의 곱셈·합산이 Conv 출력과 허용 오차 내 일치 |
| ReLU | 함수형과 층 객체의 결과 일치 |
| MaxPool | `[[7,8],[6,9]]` |
| 작은 커널 두 층과 큰 커널 한 층 | 가중치 수 `4,608`과 `6,400` |
| VGG식 블록 | 정사각형·직사각형 입력의 중간 shape |
| 작은 분류 모델 | 배치 2·1에서 각각 `[2,5]`, `[1,5]` |
| Linear | `X @ weight.T + bias`와 층의 출력 일치 |
| 전체 파라미터 | `9,421` |
| 잔차 블록 | 주 경로와 shortcut을 더한 뒤 ReLU를 적용한 값 |
| Projection | 두 경로 모두 `[1,32,7,7]`, shortcut 가중치 `512`개와 편향 `32`개 |
| 실행 모드 | `eval`만 사용하면 추적 가능, `no_grad` 안에서는 기록 안 함 |

한국어·영어 전체 Python 파일은 Python 3.12.14, PyTorch 2.14.1+cpu에서 실행해 검산을 통과했다. 무작위 가중치의 출력 소수값은 실행 순서나 버전·장치에 따라 달라질 수 있으므로 고정된 출력 문자열보다 shape, 개수, `allclose`의 결과를 확인한다.

Rust의 개수 계산은 [별도 Rust 파일](/cnn-vgg-residual-pytorch/shape-checks-ko.rs)로 실행할 수 있다. Rust 컴파일러가 있는 환경에서 다음 명령을 사용한다.

```bash
rustc --crate-name cnn_shape_checks shape-checks-ko.rs -o shape-checks-ko
./shape-checks-ko
```

한국어·영어 Rust 파일도 rustc 1.99.1에서 컴파일하고 실행해 모든 검산을 통과했다.

마지막으로, 코드를 읽다가 숫자의 뜻이 헷갈리면 계산 대상을 다시 구분하면 된다.

| 읽고 있는 숫자 | 먼저 확인할 것 |
| --- | --- |
| `3×3` | 가중치 커널의 공간 크기인지, 입력에서 고른 패치 크기인지 |
| `16×9` | 입력 채널 16개를 쓰는 필터 하나의 가중치를 세는지 |
| `16×8×8` | 마지막 특징 맵의 이미지 한 장당 데이터를 세는지 |
| `[N,5]` | `N`장의 이미지에 각각 점수 다섯 개가 있는지 |
| `F(x)+x` | 두 경로의 전체 shape와 같은 위치의 덧셈이 맞는지 |
| `model.eval()` | 평가 모드만 전환한 것인지 |
| `loss.backward()` | 기울기만 계산한 것인지 |
| `optimizer.step()` | 실제로 파라미터를 갱신한 것인지 |

합성곱이 만든 출력 데이터는 다음 층의 입력이 된다. 층을 쌓을 때는 채널의 연결과 공간 크기를 확인하고, 분류기로 넘어갈 때는 마지막 데이터의 개수를 센다. 잔차 연결에서는 두 경로가 합산 지점에서 만나는 shape와 덧셈을 확인한다. 그 계산을 실행하는 것과 학습용 기록을 남기는 것, 파라미터를 실제로 바꾸는 것까지 나누어 읽으면 모델 전체의 동작을 따라갈 수 있다.

## 참고 문서

API의 입력 형식과 동작은 아래 공식 문서에서 확인했다. VGG와 ResNet의 구조는 원 논문과 TorchVision 구현을 함께 참고했다. 본문의 작은 모델, 개수 계산, 예제 검산, 그림은 설명을 위해 구성한 것이다.

[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): 입력·출력과 가중치 shape, groups, 출력 크기, 교차상관 계산.
[^check]: PyTorch, [torch.allclose](https://docs.pytorch.org/docs/stable/generated/torch.allclose.html), [torch.numel](https://docs.pytorch.org/docs/stable/generated/torch.numel.html), [Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html); Python, [assert 문](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement).
[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html): 하위 모듈 등록, 호출, train·eval, 파라미터 관리.
[^relu]: PyTorch, [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [functional.relu](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html).
[^pool]: PyTorch, [MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html): 커널·stride·출력 크기와 기본 설정.
[^sequential]: PyTorch, [Sequential](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html).
[^vgg]: Karen Simonyan, Andrew Zisserman, [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/html/1409.1556v6), 2절: 작은 합성곱을 쌓는 구조, 수용 영역, 모델 구성.
[^vgg-code]: TorchVision, [VGG 구현](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vgg.html): 구성 D의 Conv 배열, avgpool, classifier, forward 반환값.
[^flatten]: PyTorch, [torch.flatten](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html): 시작·끝 축, 원소 순서, 뷰와 복사.
[^linear]: PyTorch, [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html): 가중치 저장 순서와 선형 계산.
[^softmax]: PyTorch, [Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html).
[^crossentropy]: PyTorch, [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): 정규화 전 로짓과 클래스 번호 정답의 형식.
[^resnet]: Kaiming He 외, [Deep Residual Learning for Image Recognition](https://arxiv.org/html/1512.03385v1), 3절: 잔차 학습, identity·projection shortcut, 크기 변경.
[^resnet-code]: TorchVision, [ResNet 구현](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html): BasicBlock의 Conv·BatchNorm·합산과 downsample.
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html): 연산 기록, grad mode, 평가 모드, inference mode, 제자리 수정.
[^nograd]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html): 기록 비활성화와 새 파라미터 생성 시의 예외.
[^dropout]: PyTorch, [Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).
[^batchnorm]: PyTorch, [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html).
[^optim]: PyTorch, [Optimizer.zero_grad](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html): 기울기 제거와 set_to_none.
[^python]: Python, [클래스](https://docs.python.org/3/tutorial/classes.html), [오류와 예외](https://docs.python.org/3/tutorial/errors.html).
[^rust]: The Rust Programming Language, [Advanced Functions and Closures](https://doc.rust-lang.org/book/ch20-04-advanced-functions-and-closures.html): 함수 포인터와 함수 자체 전달.
