+++
title = "CNN은 이미지를 어떻게 계산할까: 채널과 커널부터 Linear까지"
description = "RGB 채널이 필터에 연결되는 구조부터 합성곱 한 칸의 실제 계산, 패딩과 출력 크기, 풀링과 Flatten, Linear까지 따라간다. PyTorch 코드로 각 단계의 값과 shape를 확인한다."
date = 2026-10-27T18:12:00+09:00
slug = "cnn-convolution-basics"

[extra]
katex = true
toc = true
styles = ["cnn-convolution-basics/article.css"]
+++

이미지를 분류하는 모델에 사진 한 장을 넣으면, 모델은 사진을 숫자 배열로 받아 계산한다. 처음에는 위치마다 저장된 밝기나 색 성분을 사용하고, 중간에는 가중치를 곱하고 더해 만든 새로운 숫자 배열을 사용한다. 마지막에는 분류할 종류마다 점수를 하나씩 만든다.

<strong>CNN(합성곱 신경망)</strong>은 이미지의 일정한 영역에서 같은 가중치를 반복 사용하며 계산하는 신경망이다. 이 글에서는 <strong>입력의 어느 숫자와 어느 가중치가 계산되는지, 그 결과가 어디에 몇 개 저장되는지</strong>를 따라가 본다.

먼저 RGB 입력과 필터의 연결을 살펴본다. 이어서 합성곱 한 위치를 실제 숫자로 계산하고, 마지막에는 흑백 이미지 두 장을 받아 각각 점수 10개를 출력하는 PyTorch 모델을 실행한다. RGB 예제는 채널 연결을 살펴보기 위한 예제이고, 마지막 모델의 입력은 1채널이다.

코드는 PyTorch가 설치된 Python 환경에서 실행한다. 각 코드 블록은 위에서 아래로 이어지며, [전체 실행 코드](/cnn-convolution-basics/examples.py)도 따로 받을 수 있다.

## 1. 이미지 한 장과 채널 한 개부터 구분한다

<strong>픽셀</strong>은 이미지에서 세로·가로 위치로 구분되는 지점이다. 흑백 이미지라면 한 위치에 밝기값 하나를 저장할 수 있다. RGB 이미지라면 같은 위치에 빨강(R), 초록(G), 파랑(B) 성분을 각각 저장한다. 이 성분을 구분하는 기준이 <strong>채널</strong>이다.

따라서 RGB 이미지 한 장에는 채널이 세 개 있다. 채널 하나만 선택하면 세로·가로로 배열된 숫자들이 남는다. 중간 층에서도 이런 구조를 사용하지만, 그때의 채널이 빨강·초록·파랑을 뜻하는 것은 아니다.

<strong>텐서(Tensor)</strong>는 PyTorch에서 숫자 배열을 저장하고 계산하는 자료형이다. <strong>차원 또는 축</strong>은 배열의 위치를 구분하는 기준이고, <strong>shape</strong>는 각 차원의 크기다. 한 번에 처리하는 이미지 묶음은 <strong>배치(batch)</strong>라고 한다.

이 글의 이미지 배치는 다음 순서로 저장한다. 인덱스는 배열 안의 위치 번호이며 0부터 시작한다.

| 차원 번호 | 의미 | `(2, 3, 28, 28)`에서의 값 |
| --- | --- | --- |
| 0 | 이미지 개수 `N` | 이미지 2장 |
| 1 | 이미지 한 장의 채널 수 `C` | 이미지마다 RGB 3채널 |
| 2 | 채널 하나의 세로 크기 `H` | 세로 28칸 |
| 3 | 채널 하나의 가로 크기 `W` | 가로 28칸 |

숫자 하나를 읽는 순서는 `x[이미지 번호, 채널 번호, 세로 위치, 가로 위치]`다. `x[0, 1, 2, 3]`은 첫 이미지의 두 번째 채널에서 세로 인덱스 2, 가로 인덱스 3에 있는 값이다.

채널이 3개인 것과 텐서가 3차원인 것은 다른 이야기다. 배치까지 포함한 `(2, 3, 28, 28)`은 <strong>4차원 텐서</strong>다. 흑백 이미지 두 장은 같은 순서로 `(2, 1, 28, 28)`이라고 쓴다.

### 전체 모델에서는 무엇이 바뀌는가

<strong>층(layer)</strong>은 모델에서 한 단계의 계산을 담당하는 구성 요소다. <strong>가중치</strong>는 입력값에 곱하는 숫자, <strong>편향</strong>은 곱셈 결과를 합친 뒤 더하는 숫자다. 모델이 학습하면서 바꾸는 가중치와 편향을 <strong>학습 파라미터</strong>라고 한다.

아래 그림은 뒤에서 만들 모델의 전체 흐름이다. 각 연산의 역할을 먼저 간단히 읽고, 구체적인 계산은 차례로 살펴보자.

| 연산 | 하는 일 |
| --- | --- |
| 합성곱 `Conv2d` | 입력의 일정한 영역과 가중치를 곱해 더한다. 위치를 옮기며 반복해 새 채널들을 만든다. |
| `ReLU` | 음수인 값을 0으로 바꾸고 나머지 값은 유지한다. |
| 최대 풀링 `MaxPool2d` | 채널마다 작은 영역에서 최댓값 하나를 선택한다. |
| `Flatten` | 이미지 한 장의 값들을 한 줄로 나열한다. 이미지끼리는 섞지 않는다. |
| `Linear` | 펼친 값들에 가중치를 곱하고 더해서 최종 점수들을 만든다. |

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/cnn-shape-flow.svg" alt="흑백 이미지 두 장 (2,1,28,28)이 합성곱과 풀링 두 번을 거쳐 (2,64,7,7)이 된다. 이미지마다 값을 펼쳐 (2,3136)을 만들고 Linear로 점수 (2,10)을 계산한다.">
<figcaption>앞의 2는 끝까지 이미지 개수다. 중간의 32와 64는 채널 수이고, 마지막의 10은 이미지마다 출력할 점수의 개수다.</figcaption>
</figure>

분류할 정답의 종류를 <strong>클래스</strong>라고 한다. 마지막 점수 10개는 클래스 10개에 대응한다. 어떤 데이터셋을 쓰는지에 따라 클래스의 의미가 달라지므로, 출력이 10개라는 이유만으로 숫자 0~9를 분류한다고 단정할 수는 없다.

## 2. 필터 하나는 입력 채널들을 함께 계산한다

<strong>커널(kernel)</strong>과 <strong>필터(filter)</strong>는 자료마다 같은 뜻으로 쓰기도 한다. 여기서는 연결 구조를 구분하기 위해 다음처럼 부르겠다.

| 이름 | 이 글에서 가리키는 것 | RGB 입력에 3×3 합성곱을 적용할 때 |
| --- | --- | --- |
| 채널별 커널 | 특정 입력 채널에 곱할 2차원 가중치 배열 | `(3, 3)` |
| 필터 | 출력 채널 하나를 만들 가중치 전체 | 채널별 커널 3개를 가진 `(3, 3, 3)` |
| 출력 채널 또는 특징 맵 | 필터로 여러 위치를 계산해서 만든 2차원 결과 | 예를 들어 `(7, 7)` |

<strong>커널에 들어 있는 숫자가 바로 가중치다.</strong> 커널과 가중치가 서로 다른 재료여서 나중에 합쳐지는 것이 아니다. 커널은 가중치들을 공간 모양에 맞춰 배열한 것이다.

이번 연결 설명은 `groups=1`인 일반적인 합성곱을 기준으로 한다. `groups`는 입력·출력 채널의 연결을 몇 그룹으로 나눌지 정하는 설정이며, 1이면 각 필터가 모든 입력 채널을 사용한다.[^conv]

### RGB 입력에 필터 두 개를 적용해 보자

입력은 RGB 이미지 한 장 `(1, 3, 7, 7)`이다. 필터 A와 B를 두고, 각 필터에 R·G·B용 3×3 커널을 하나씩 만든다. 그림의 가장자리 처리는 뒤에서 설명할 `padding=1`을 사용하므로 출력의 세로·가로도 7이다.

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/rgb-filter-connections.svg" alt="입력 R G B 채널이 각각 필터 A의 R G B용 커널과 필터 B의 R G B용 커널에 연결된다. 필터마다 세 채널의 계산을 합치고 자기 편향을 한 번 더해 출력 채널 하나를 만든다." loading="lazy">
<figcaption>필터 A와 필터 B 모두 RGB 전체를 사용한다. 각 필터가 가진 커널의 값은 별개이며, 필터 하나가 출력 채널 하나를 만든다.</figcaption>
</figure>

필터 A가 <strong>출력의 특정 위치 한 칸</strong>을 만드는 과정은 다음과 같다.

1. R 채널에서 3×3 영역을 골라 A의 R용 커널과 같은 위치끼리 곱한다. 곱셈 결과 9개를 더한다.
2. 같은 세로·가로 위치의 G 영역과 A의 G용 커널도 곱해서 더한다.
3. B 영역과 A의 B용 커널도 같은 방식으로 계산한다.
4. 세 채널에서 얻은 결과를 모두 더하고, A의 편향 하나를 한 번 더한다.
5. 그 숫자를 출력 채널 A의 해당 위치에 저장한다.

한 출력값에 사용한 곱셈은 `입력 채널 3개 × 3 × 3 = 27개`다. 이것을 여러 공간 위치에서 반복하면 출력 채널 A 전체가 만들어진다. 필터 B도 같은 입력을 자기 가중치로 계산한다. 이 순서는 계산 관계를 풀어 쓴 것이며, GPU가 반드시 이 순서대로 한 개씩 실행한다는 뜻은 아니다.

### 가중치의 shape와 출력의 shape는 다르다

`C_in`은 입력 채널 수, `C_out`은 출력 채널 수, `K_H`와 `K_W`는 커널의 세로·가로 크기다. `groups=1`일 때 가중치 전체의 shape는 <strong><code>(C_out, C_in, K_H, K_W)</code></strong>다.

| 선택한 대상 | 이번 예제의 shape | 들어 있는 것 |
| --- | --- | --- |
| `weight` | `(2, 3, 3, 3)` | 필터 2개의 전체 가중치 |
| `weight[0]` | `(3, 3, 3)` | 첫 필터의 R·G·B용 커널 |
| `weight[0, 0]` | `(3, 3)` | 첫 필터의 R용 커널 |
| `weight[0, 0, 0, 0]` | `()` | 그 커널의 가중치 숫자 하나 |
| `bias` | `(2,)` | 출력 채널마다 하나씩 있는 편향 |
| 출력 `y` | `(1, 2, 7, 7)` | 이미지 한 장의 출력 채널 2개 |
| `y[0, 0]` | `(7, 7)` | 첫 이미지의 첫 출력 채널 |

shape `()`는 숫자 하나를 담은 0차원 텐서다. `(2,)`는 숫자 두 개를 담은 1차원 텐서다.

이 층의 가중치는 `2×3×3×3=54개`, 편향은 `2개`다. 출력 숫자는 `1×2×7×7=98개`다. <strong>필터 개수, 가중치 개수, 출력 숫자 개수는 각각 다른 개수</strong>다.

특징 맵은 가중치가 아니라 <strong>입력과 현재 가중치로 계산한 결과</strong>다. 다음 층은 이 결과를 입력으로 받는다. 앞 층의 필터 자체를 다음 층의 입력 데이터로 전달하는 것은 아니다.

### 같은 가중치는 어디에서 다시 쓰는가

필터 A는 입력의 다른 위치에서도 자기 가중치를 다시 사용한다. 배치에 이미지가 여러 장 있어도 같은 A를 사용한다. 이것을 <strong>가중치 공유</strong>라고 한다.

필터 B는 A와 별개의 가중치를 갖는다. A 안의 R용 커널과 G용 커널도 별개의 가중치다. 또한 학습으로 갱신하면 그 값은 바뀔 수 있다. 공유한다는 말은 <strong>한 번의 계산에서 같은 모델 파라미터를 여러 위치와 여러 이미지에 사용한다</strong>는 뜻이다.

## 3. `nn.Conv2d`에서는 무엇을 정하는가

`torch`는 PyTorch의 기능을 가져오는 이름이다. `from torch import nn`은 신경망 층을 제공하는 `torch.nn`을 `nn`이라는 이름으로 가져온다. `torch.nn.functional`에는 ReLU 같은 계산 함수가 있고, 아래에서는 짧게 `F`라고 부른다.

`nn.Conv2d`는 합성곱 층을 만드는 <strong>클래스</strong>다. 여기서 Python의 클래스는 객체의 구조와 동작을 정의한 것이며, 앞에서 말한 분류의 정답 종류와는 다른 뜻이다. 클래스를 호출하면 실제로 사용할 <strong>객체</strong>가 생긴다.

```python
import torch  # 텐서 생성과 계산 기능을 가져온다.
from torch import nn  # torch.nn의 층 클래스들을 nn이라는 이름으로 사용한다.
import torch.nn.functional as F  # 신경망 계산 함수들을 F라는 이름으로 사용한다.

conv_example = nn.Conv2d(
    in_channels=3,       # 실제 입력이 RGB이므로 입력 채널을 3으로 맞춘다.
    out_channels=32,     # 필터 32개를 만든다. 출력 채널 수도 32가 된다.
    kernel_size=3,       # 채널별 커널의 세로·가로를 3×3으로 정한다.
    stride=1,           # 계산 영역의 시작 위치를 한 칸씩 옮긴다.
    padding=1,          # 입력의 위·아래·왼쪽·오른쪽에 각각 한 칸을 추가해 계산한다.
    dilation=1,          # 커널 가중치를 이웃한 입력 칸에 적용한다.
    groups=1,           # 각 필터가 모든 입력 채널을 사용한다.
    bias=True,          # 출력 채널마다 학습 가능한 편향 하나를 만든다.
    padding_mode="zeros",  # 추가한 가장자리 칸은 0으로 취급한다.
)
```

`Conv2d`의 `2d`는 세로·가로 두 공간 방향을 처리한다는 뜻이다. 배치와 채널을 포함한 입력 텐서 전체가 2차원이라는 뜻은 아니다.

위 설정은 `nn.Conv2d(3, 32, 3, padding=1)`로도 쓸 수 있다. 처음 세 인자의 순서는 <strong>입력 채널 수, 출력 채널 수, 커널 크기</strong>다. 배치 크기는 이 인자에 포함하지 않는다.

| 구분 | 개발자가 하는 일 |
| --- | --- |
| `in_channels` | 실제로 들어오는 데이터의 채널 수와 맞춘다. |
| `out_channels` | 이 층에서 만들 필터 개수를 선택한다. |
| `kernel_size` | 채널별로 한 번에 읽을 공간 영역의 크기를 선택한다. |
| `weight`, `bias` | 층이 초기값을 생성한다. 이후 학습으로 내부 숫자를 바꾼다. |

`out_channels=32`는 함수 호출의 <strong>설정 인자</strong>이고, `conv_example.weight` 안의 숫자는 <strong>학습 파라미터</strong>다. 일반적인 학습에서는 필터 개수 32를 늘리는 것이 아니라, 32개 필터 안의 숫자를 수정한다.

`nn.Conv2d(...)`를 호출하면 층과 초기 가중치가 생긴다. 만들어진 층에 `conv_example(x)`처럼 입력을 넣어야 실제 합성곱을 계산한다.

### 앞 층의 출력 채널 수를 다음 층의 입력 채널 수에 맞춘다

| 단계 | 입력 shape | 층 설정 | 가중치 shape | 출력 shape |
| --- | --- | --- | --- | --- |
| 첫 합성곱 | `(1, 3, 7, 7)` | `Conv2d(3, 2, 3, padding=1)` | `(2, 3, 3, 3)` | `(1, 2, 7, 7)` |
| 둘째 합성곱 | `(1, 2, 7, 7)` | `Conv2d(2, 4, 3, padding=1)` | `(4, 2, 3, 3)` | `(1, 4, 7, 7)` |

둘째 층은 RGB가 아니라 첫 층이 만든 두 채널을 받는다. 따라서 둘째 층의 필터 하나에는 3×3 커널이 두 개 필요하다. 입력 채널 수와 출력 채널 수가 같을 필요는 없지만, <strong>실제로 전달되는 채널 수와 받는 층의 <code>in_channels</code>는 같아야 한다.</strong>

커널이 반드시 정사각형일 필요도 없다. `kernel_size=(3, 5)`는 세로 3칸, 가로 5칸이다. 같은 `Conv2d` 층의 모든 필터는 지정한 공간 커널 크기를 사용한다.[^conv]

## 4. 출력 크기는 계산할 수 있는 시작 위치의 개수다

먼저 가장자리에 값을 추가하지 않고, 7×7 입력에 3×3 커널을 한 칸씩 옮겨 보자. 이 절의 기본 조건은 <strong><code>stride=1</code>, <code>padding=0</code>, <code>dilation=1</code></strong>이다.

가로 방향의 입력 인덱스는 `0, 1, 2, 3, 4, 5, 6`이다. 3칸짜리 영역을 놓을 수 있는 위치는 다음과 같다.

| 시작 인덱스 | 선택하는 입력 인덱스 | 해당 출력의 가로 인덱스 |
| ---: | --- | ---: |
| 0 | 0, 1, 2 | 0 |
| 1 | 1, 2, 3 | 1 |
| 2 | 2, 3, 4 | 2 |
| 3 | 3, 4, 5 | 3 |
| 4 | 4, 5, 6 | 4 |
| 5 | 5, 6, 7 | 불가능: 입력 인덱스 7이 없다. |

시작 위치가 0부터 4까지 있으므로 계산 위치는 <strong>5개</strong>다. 처음에서 끝까지 이동한 횟수는 4번이지만 처음 위치에서도 계산한다. 세로 방향도 5개이므로 필터 하나의 출력은 5×5다.

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/padding-start-positions.svg" alt="길이 7의 입력에 길이 3의 커널을 놓으면 시작 위치 0부터 4까지 다섯 곳이다. 양쪽에 패딩 한 칸씩을 추가하면 길이가 9가 되고 시작 위치 일곱 곳에서 계산한다." loading="lazy">
<figcaption>커널 크기는 한 번에 읽을 영역의 크기다. 출력 크기는 그 영역을 놓고 계산할 수 있는 위치의 개수다.</figcaption>
</figure>

### `L - K + 1`에서 마지막 `+1`은 왜 필요한가

`L`을 입력 길이, `K`를 커널 길이, `s`를 시작 인덱스라고 하자. 입력의 마지막 인덱스는 `L-1`이고, `s`에서 시작한 커널의 마지막 인덱스는 `s+K-1`이다.

<div class="cnn-math">
$$
\begin{aligned}
s+K-1 &\le L-1 &&\text{커널의 끝이 입력의 끝을 넘지 않아야 한다.}\\
s &\le L-K &&\text{양쪽에서 }K-1\text{을 뺀다.}
\end{aligned}
$$
</div>

시작 인덱스는 `0, 1, …, L-K`다. 양 끝을 포함한 개수는 `(L-K)-0+1`, 즉 <strong><code>L-K+1</code></strong>이다. 여기의 `+1`은 패딩을 추가한다는 뜻이 아니라 0번 위치까지 포함해 세는 항이다.

이제 숫자를 대입하면 `7-3+1=5`, `28-3+1=26`이다. 3×3 커널의 출력이 언제나 5×5인 것은 아니다. 입력 크기가 달라지면 계산 가능한 위치 수도 달라진다.

### 패딩은 계산 영역의 가장자리를 추가한다

<strong>패딩(padding)</strong>은 입력의 가장자리 바깥에도 계산용 값을 두는 것이다. `padding=1`이면 위·아래·왼쪽·오른쪽에 각각 한 칸을 추가해 계산한다. 기본 방식인 `padding_mode="zeros"`에서는 추가 영역을 0으로 취급한다. 반면 기본 인자 `padding=0`은 칸을 추가하지 않는다는 뜻이다.

28칸 길이에 양쪽으로 한 칸씩 추가하면 계산 영역의 길이는 30이다. 여기에 3칸짜리 커널을 놓을 수 있는 위치는 `30-3+1=28개`다.

<div class="cnn-math">
$$28+\underbrace{1+1}_{\text{양쪽 패딩}}-3+\underbrace{1}_{\text{시작 위치 포함}}=28$$
</div>

가로와 세로에 같은 원리를 적용하므로 28×28이 유지된다. <strong>크기가 같다는 뜻이지, 출력 숫자가 입력 숫자와 같다는 뜻은 아니다.</strong> 또한 `Conv2d`가 내부에서 패딩을 처리하므로 원본 입력의 `x.shape`가 30×30으로 바뀌는 것도 아니다.

| 입력 shape | `padding` | 패딩을 포함한 공간 | 출력 shape: 필터 32개 |
| --- | ---: | --- | --- |
| `(2, 1, 7, 7)` | 0 | 7×7 | `(2, 32, 5, 5)` |
| `(2, 1, 7, 7)` | 1 | 9×9 | `(2, 32, 7, 7)` |
| `(2, 1, 28, 28)` | 0 | 28×28 | `(2, 32, 26, 26)` |
| `(2, 1, 28, 28)` | 1 | 30×30 | `(2, 32, 28, 28)` |

네 경우 모두 가중치 shape는 `(32, 1, 3, 3)`, 편향 shape는 `(32,)`다.

### 공간 크기를 유지하려면 패딩을 얼마나 넣어야 할까

`P`를 한쪽 패딩 길이라고 하자. `stride=1`, `dilation=1`이고 양쪽 패딩이 같다면 출력 길이는 `L+2P-K+1`이다. 이 길이가 입력 `L`과 같도록 정리한다.

<div class="cnn-math">
$$
\begin{aligned}
L &= L+2P-K+1 &&\text{입력 길이와 출력 길이를 같게 둔다.}\\
0 &= 2P-K+1 &&\text{양쪽에서 }L\text{을 뺀다.}\\
2P &= K-1 &&\text{양쪽에 }K-1\text{을 더한다.}\\
P &= \frac{K-1}{2} &&\text{양쪽을 2로 나눈다.}
\end{aligned}
$$
</div>

| 커널 크기 | 한쪽에 필요한 패딩 | 설정 |
| --- | --- | --- |
| 3×3 | 세로·가로 각각 1 | `padding=1` |
| 5×5 | 세로·가로 각각 2 | `padding=2` |
| 7×7 | 세로·가로 각각 3 | `padding=3` |
| 3×5 | 위·아래 1, 왼쪽·오른쪽 2 | `padding=(1, 2)` |

`padding="same"`으로 공간 크기가 유지되도록 맡길 수도 있다. PyTorch `Conv2d`에서는 이 설정을 `stride=1`에서 지원한다. 출력 <strong>채널 수</strong>는 여전히 `out_channels`로 정한다.[^conv]

<details>
<summary>stride·dilation이 달라지는 경우의 출력 길이</summary>

<strong>스트라이드(stride)</strong> `S`는 영역의 시작 위치를 옮기는 간격이다. 시작 위치를 `0, S, 2S, …`로 선택하므로, `dilation=1`일 때 출력 길이는 다음과 같다.

<div class="cnn-math">
$$\left\lfloor\frac{L+2P-K}{S}\right\rfloor+1$$
</div>

`floor`, 즉 바닥 기호는 해당 수보다 크지 않은 가장 큰 정수를 뜻한다. 여기서는 유효한 크기 설정에서 나눗셈의 소수 부분을 버리는 것과 같다.

<strong>딜레이션(dilation)</strong> `D`는 커널의 이웃 가중치를 적용할 입력 위치 사이의 간격이다. 가중치 개수는 그대로 두고 적용 간격을 넓힐 수 있다. 커널이 차지하는 실제 범위는 `K_effective=D×(K-1)+1`이므로 위 식의 `K`를 `K_effective`로 바꾸면 된다. 세로와 가로는 각각 계산한다.

짝수 커널에서는 `(K-1)/2`가 정수가 아니므로, 같은 크기를 유지하기 위해 양쪽 패딩이 달라져야 할 수 있다. 정수 나눗셈으로 무조건 버려서 대칭 패딩을 정하면 크기가 어긋날 수 있다.

</details>

## 5. 합성곱 출력 한 칸을 실제로 계산한다

이번에는 입력 한 장, 입력 채널 하나, 필터 32개를 사용한다. 전체 입력 shape는 `(1, 1, 7, 7)`이다. 채널 하나에 1부터 49까지 차례로 넣는다.

<div class="cnn-math">
$$
X=\begin{bmatrix}
1&2&3&4&5&6&7\\
8&9&10&11&12&13&14\\
15&16&17&18&19&20&21\\
22&23&24&25&26&27&28\\
29&30&31&32&33&34&35\\
36&37&38&39&40&41&42\\
43&44&45&46&47&48&49
\end{bmatrix}
$$
</div>

처음 생성한 `Conv2d`는 정해진 초기화 규칙에 따라 무작위 가중치를 갖는다. 손계산과 대조할 수 있도록 <strong>32개 중 첫 필터의 가중치만 직접 지정</strong>하겠다. 이 숫자들은 학습해서 얻은 값이 아니다. 나머지 필터 31개는 생성한 초기값을 사용한다.

### 같은 위치끼리 곱한 뒤, 모두 더하고 편향을 더한다

<strong>패치(patch)</strong>는 현재 계산하려고 선택한 입력의 작은 영역이다. 첫 위치에서는 왼쪽 위 3×3 영역을 고른다. 첫 필터의 커널과 편향은 다음과 같이 정한다.

<div class="cnn-math">
$$
\text{patch}=\begin{bmatrix}1&2&3\\8&9&10\\15&16&17\end{bmatrix},
\qquad
K=\begin{bmatrix}1&2&-1\\-2&1&3\\2&-1&1\end{bmatrix},
\qquad b=2
$$
</div>

`patch * K`는 <strong>원소별 곱셈</strong>이다. 같은 위치의 숫자끼리 곱하므로 결과도 3×3이다. 행렬곱 연산자 `@`와는 다르다.

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/convolution-one-position.svg" alt="입력의 첫 3×3 영역과 커널을 같은 위치끼리 곱하면 1 4 -3, -16 9 30, 30 -16 17이 나온다. 아홉 값을 합친 56에 편향 2를 더하면 첫 출력값 58이다." loading="lazy">
<figcaption>출력 한 칸을 완성하려면 곱셈 결과를 모두 합산해야 한다. 이 그림은 입력 채널이 하나인 경우다. RGB라면 세 채널의 계산 결과까지 합친 뒤 편향을 한 번 더한다.</figcaption>
</figure>

숫자 배열을 만드는 `torch.tensor`는 전달한 값으로 텐서를 만든다. `torch.arange(1, 50)`은 1 이상 50 미만의 정수를 순서대로 만들고, `reshape`는 값의 순서를 유지하면서 지정한 모양으로 배치한다. `dtype=torch.float32`는 각 값을 32비트 실수로 저장하라는 설정이다.

```python
x7 = torch.arange(1, 50, dtype=torch.float32).reshape(1, 1, 7, 7)
# 이미지 1장, 채널 1개, 세로·가로 7칸이다.

kernel3 = torch.tensor(
    [[1, 2, -1], [-2, 1, 3], [2, -1, 1]],
    dtype=torch.float32,  # 합성곱 가중치와 같은 실수 자료형을 사용한다.
)

patch = x7[0, 0, 0:3, 0:3]  # 첫 이미지·첫 채널에서 행 0~2, 열 0~2를 선택한다.
products = patch * kernel3  # 같은 위치끼리 곱한다. 결과 shape는 (3, 3)이다.
```

슬라이스 `0:3`은 0부터 시작해 <strong>끝 번호 3을 제외</strong>한다. 따라서 인덱스 0, 1, 2를 선택한다.

### `products`가 왜 숫자 하나가 아닌가

`products`는 ‘곱셈 결과들’이라는 뜻으로 붙인 변수명이다. PyTorch가 정한 특별한 이름이 아니다. 이 변수에는 곱셈 결과 9개가 들어 있으며, 아직 합산하지 않았다.

```python
print(products.tolist())  # 텐서를 Python 리스트로 바꾸어 값들을 확인한다.
# [[1.0, 4.0, -3.0], [-16.0, 9.0, 30.0], [30.0, -16.0, 17.0]]

print(products.sum(dim=1).tolist())  # 각 행 안의 세 값을 더한다: [2.0, 23.0, 31.0].
print(products.sum().item())  # 아홉 값을 모두 더한다: 56.0.
print((products.sum() + 2).item())  # 편향 2를 더한 최종 출력 한 칸: 58.0.
```

`sum`은 값을 더하는 메서드다. `dim=1`은 `(3, 3)`에서 두 번째 차원을 따라 합산하라는 뜻이다. 첫 행은 `1+4-3=2`, 둘째 행은 `-16+9+30=23`, 셋째 행은 `30-16+17=31`이 된다. `dim`을 지정하지 않으면 모든 원소를 더한다.[^sum]

| 코드 | 실제 결과 | 결과 shape |
| --- | --- | --- |
| `patch * kernel3` | 곱셈 결과 9개 | `(3, 3)` |
| `products.sum(dim=1)` | `[2, 23, 31]` | `(3,)` |
| `products.sum()` | `56` | `()` |
| `products.sum() + 2` | `58` | `()` |

`.tolist()`는 텐서의 값을 Python 리스트 등으로 바꾸고, `.item()`은 원소 하나인 텐서에서 Python 숫자를 꺼낸다. 행별 합을 출력하는 것은 <strong>중간 계산을 읽기 위한 확인</strong>이다. 실제 합성곱이 행별 합 리스트를 최종 출력하는 것은 아니다.

### 가중치에 커널을 복사하는 이유

이번에는 같은 계산을 `Conv2d`에 맡긴다. `copy_`는 기존 텐서의 내용을 전달한 값으로 바꾸고, `fill_`은 선택한 텐서를 한 값으로 채운다. 두 메서드 끝의 `_`는 여기서 기존 텐서 자체를 수정한다는 뜻이다.

`torch.no_grad()`는 해당 구간에서 역전파에 사용할 계산 기록을 만들지 않게 한다. 지금은 손계산과 결과를 맞춰 보는 구간이므로, 직접 지정한 값을 복사하고 순전파 결과를 확인하는 데 사용한다. 가중치를 영구히 고정하는 설정은 아니다.[^no-grad]

```python
conv3 = nn.Conv2d(
    in_channels=1, out_channels=32, kernel_size=3,
    stride=1, padding=0, bias=True,
)  # 입력 채널 1개, 필터 32개인 층을 만든다.

with torch.no_grad():  # 이 구간의 역전파 계산 기록은 만들지 않는다.
    conv3.weight[0, 0].copy_(kernel3)  # 첫 필터의 첫 채널용 가중치를 지정한다.
    conv3.bias[0].fill_(2.0)  # 첫 출력 채널의 편향을 2로 지정한다.
    y3 = conv3(x7)  # 실제 합성곱: 전체 출력 shape는 (1, 32, 5, 5)다.

print(y3[0, 0])  # 첫 이미지의 첫 출력 채널만 확인한다.
```

첫 출력 채널의 전체 결과는 다음과 같다.

<div class="cnn-math">
$$
\begin{bmatrix}
58&64&70&76&82\\
100&106&112&118&124\\
142&148&154&160&166\\
184&190&196&202&208\\
226&232&238&244&250
\end{bmatrix}
$$
</div>

첫 값은 `y3[0, 0, 0, 0]=58`이다. 오른쪽 한 칸에서는 패치가 `[[2,3,4],[9,10,11],[16,17,18]]`로 바뀌고, 같은 가중치를 사용한 결과가 `64`다. 이렇게 25개 위치를 계산해 5×5 배열 하나를 만든다. 나머지 31개 필터도 같은 입력을 각자의 가중치로 처리하므로 <strong>전체 출력은 <code>(1, 32, 5, 5)</code></strong>다.

모든 위치를 직접 계산해 비교할 수도 있다. `empty_like`는 지정한 텐서와 같은 모양·자료형·장치의 저장 공간을 만들되, 초기 숫자는 정하지 않는다. 아래 반복문이 각 칸을 모두 채운다.

```python
manual3 = torch.empty_like(y3[0, 0])  # 첫 출력 채널과 같은 (5, 5) 공간을 만든다.
with torch.no_grad():
    for row in range(y3.shape[2]):  # 출력의 세로 인덱스 0~4를 선택한다.
        for col in range(y3.shape[3]):  # 출력의 가로 인덱스 0~4를 선택한다.
            patch = x7[0, 0, row:row + 3, col:col + 3]  # 현재 위치의 3×3 영역.
            manual3[row, col] = (patch * kernel3).sum() + conv3.bias[0]

assert torch.allclose(manual3, y3[0, 0])  # 작은 실수 오차를 허용해 두 결과를 비교한다.
```

`assert`는 뒤의 조건이 거짓이면 실행을 멈추는 검사문이다. `torch.allclose`는 두 텐서의 값들이 허용한 실수 오차 안에서 같은지를 반환한다.

<details>
<summary>같은 가중치에 padding=1을 적용하면 첫 값은 어떻게 바뀔까?</summary>

왼쪽 위에서 읽는 3×3 영역은 `[[0,0,0],[0,1,2],[0,8,9]]`다. 추가한 패딩 0도 곱셈에 포함한다. 같은 커널을 곱하면 전체 합은 8이고, 편향 2를 더한 출력은 <strong>10</strong>이다. 계산 가능한 위치가 늘어나므로 첫 출력 채널의 크기는 7×7이 된다.

PyTorch의 `Conv2d`는 커널을 뒤집지 않고 위처럼 같은 위치끼리 곱한다. 수학적으로는 <strong>교차상관</strong>이라는 연산이지만, CNN에서는 관례적으로 합성곱이라고 부른다.[^conv]

</details>

## 6. 커널을 키우면 출력 크기와 계산량은 어떻게 바뀌는가

같은 7×7 입력에 5×5 커널을 적용하면 시작 위치는 세로·가로 각각 `0, 1, 2`다. 따라서 출력 채널 하나의 크기는 <strong>3×3</strong>이 된다. 이동 간격은 여전히 `stride=1`이다. 커널 크기가 5라고 자동으로 5칸씩 이동하지 않는다.

다음은 5×5 커널을 사용한 <strong>별도의 층</strong>이다. 앞의 3×3 층 뒤에 연결하는 것이 아니라, 같은 `x7`을 두 설정으로 각각 계산한다.

```python
kernel5 = torch.tensor(
    [[1, -1, 2, 1, -2],
     [2, 1, -1, 2, 1],
     [-1, 2, 1, -2, 2],
     [1, -2, 2, 1, -1],
     [2, 1, -1, 2, 1]],
    dtype=torch.float32,  # 5×5 가중치도 실수로 만든다.
)
conv5 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
with torch.no_grad():
    conv5.weight[0, 0].copy_(kernel5)  # 첫 필터의 커널을 직접 지정한다.
    conv5.bias[0].fill_(2.0)  # 비교를 위해 편향은 같은 2를 사용한다.
    y5 = conv5(x7)  # 같은 원본 x7을 입력한다: (1, 32, 3, 3).
    first5 = (x7[0, 0, :5, :5] * kernel5).sum() + conv5.bias[0]

print(first5.item())  # 263.0: 첫 위치에서 25개 곱을 합친 261에 편향 2를 더한다.
print(y5[0, 0].tolist())
# [[263.0, 277.0, 291.0], [361.0, 375.0, 389.0], [459.0, 473.0, 487.0]]
```

한 위치에서 계산할 곱셈은 늘었지만, 계산할 위치는 줄었다. 그래서 전체 연산량은 두 수를 함께 봐야 한다.

| 무엇을 세는가 | `groups=1`에서의 계산 |
| --- | --- |
| 가중치 개수 | `C_out × C_in × K_H × K_W` |
| 출력 숫자 하나를 만들 곱셈 수 | `C_in × K_H × K_W` |
| 배치 전체의 출력 숫자 개수 | `N × C_out × H_out × W_out` |
| 직접 계산 기준 전체 곱셈 수 | `N × C_out × H_out × W_out × C_in × K_H × K_W` |

다음은 이미지 1장, 입력 채널 1개, 필터 32개, 스트라이드 1로 비교한 결과다. 덧셈은 제외하고 곱셈만 센다.

| 입력 공간 | 커널 | 패딩 | 출력 공간 | 필터당 출력 위치 | 출력 한 칸의 곱셈 | 전체 곱셈 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 7×7 | 3×3 | 0 | 5×5 | 25 | 9 | 7,200 |
| 7×7 | 5×5 | 0 | 3×3 | 9 | 25 | 7,200 |
| 28×28 | 3×3 | 0 | 26×26 | 676 | 9 | 194,688 |
| 28×28 | 5×5 | 0 | 24×24 | 576 | 25 | 460,800 |
| 28×28 | 3×3 | 1 | 28×28 | 784 | 9 | 225,792 |
| 28×28 | 5×5 | 2 | 28×28 | 784 | 25 | 627,200 |

7×7 예제에서 전체 곱셈 수가 같은 것은 `25×9`와 `9×25`가 우연히 같기 때문이다. 일반 규칙은 아니다. <strong>출력 크기와 나머지 조건이 같다면 큰 커널의 곱셈 수가 더 많다.</strong> 패딩을 고정해 출력 크기까지 달라진다면, 새 출력 크기를 넣어 다시 계산해야 한다.

이 표는 같은 위치끼리 직접 곱하는 방식으로 센 값이다. 실제 GPU 실행 시간은 내부 계산 방식과 하드웨어에도 영향을 받으므로, 표의 곱셈 수만으로 시간을 그대로 환산할 수는 없다.

## 7. ReLU와 풀링은 서로 다른 일을 한다

합성곱 결과에 바로 다음 합성곱만 적용하는 대신, 중간값을 바꾸거나 공간 크기를 줄이는 연산을 넣을 수 있다. 이 모델에서는 합성곱 뒤에 ReLU, 그 뒤에 최대 풀링을 적용한다.

### ReLU는 값만 바꾸고 모양은 유지한다

<strong>활성화 함수</strong>는 층에서 계산한 값에 적용하는 변환이다. ReLU는 각 숫자에 `max(0, 값)`을 적용한다. 음수는 0, 양수와 0은 그대로 남는다.[^relu]

<div class="cnn-math">
$$
\begin{bmatrix}-2&3&0\\4&-1&5\end{bmatrix}
\quad\xrightarrow{\text{ReLU}}\quad
\begin{bmatrix}0&3&0\\4&0&5\end{bmatrix}
$$
</div>

음수가 있던 위치 자체를 삭제하지 않으므로 shape는 그대로다. `(2, 32, 28, 28)`에 적용하면 결과도 `(2, 32, 28, 28)`이다.

ReLU는 음수 구간과 양수 구간에 서로 다른 계산을 적용한다. 이런 <strong>비선형 변환</strong>을 넣으면, 고정된 가중치를 한 번 곱하고 편향을 더하는 계산만으로 표현할 수 없는 관계를 모델 안에서 만들 수 있다.

### 최대 풀링은 채널마다 영역의 최댓값을 고른다

`nn.MaxPool2d(kernel_size=2)`는 채널 하나 안에서 2×2 영역의 최댓값을 선택한다. `stride`를 생략하면 커널 크기와 같은 2가 된다. `Conv2d`의 기본 스트라이드 1과 다르다.[^pool]

| 한 채널의 입력 4×4 | 2×2 최대 풀링 결과 |
| --- | --- |
| `[[1,3,2,0], [4,2,5,1], [0,1,7,3], [2,6,4,8]]` | `[[4,5], [6,8]]` |

왼쪽 위 영역 `[[1,3],[4,2]]`에서는 4를 고르고, 오른쪽 위 영역에서는 5를 고른다. 아래 두 영역도 같은 방식으로 처리한다. 다음 채널은 그 채널의 값들끼리 따로 처리한다. <strong>채널끼리 최댓값을 비교하거나 합치는 연산이 아니다.</strong>

따라서 `(2, 32, 28, 28)`은 `(2, 32, 14, 14)`가 된다. 이미지 수 2와 채널 수 32는 유지되고, 세로·가로 크기가 줄어든다.

길이가 7이면 시작 위치 0, 2, 4에서 2칸짜리 영역을 고를 수 있다. 6에서 시작하면 인덱스 7까지 필요하므로 들어가지 않는다. 기본 설정에서는 완전히 들어가는 세 위치만 계산해 <strong><code>7 → 3</code></strong>이 된다. 이 모델은 기본값 `ceil_mode=False`, 패딩 0을 사용한다.

최대 풀링에는 학습할 가중치가 없다. 영역을 선택하고 그 안의 최댓값을 반환하는 계산 규칙이다.

## 8. Flatten은 한 이미지의 값들을 펼친다

마지막 풀링 결과가 `(2, 64, 7, 7)`이라고 하자. 이미지 두 장 각각에 7×7 특징 맵이 64개씩 있다. 이미지 한 장에 들어 있는 숫자는 <strong><code>64×7×7=3136개</code></strong>다.

`flatten(start_dim=1)`은 0번 차원인 배치를 남기고, 1번 차원부터 마지막 차원까지 하나로 합친다. 값을 더하거나 평균 내지 않는다. 같은 값들을 순서대로 나열해 <strong><code>(2, 3136)</code></strong>으로 만든다.[^flatten]

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/pooling-flatten.svg" alt="최대 풀링은 각 채널 안의 2×2 영역에서 최댓값을 선택한다. Flatten은 이미지 두 장을 서로 섞지 않고, 각 이미지의 채널 64개와 7×7 위치 값을 3136개짜리 배열로 펼친다." loading="lazy">
<figcaption>위쪽 숫자는 채널별 풀링을 보여 주는 별도의 예다. 풀링은 값을 골라 공간 크기를 줄이고, Flatten은 남아 있는 값을 그대로 펼친다.</figcaption>
</figure>

| 입력 차원 | Flatten 이후 |
| --- | --- |
| 0번: 배치 2 | 그대로 2 |
| 1번: 채널 64, 2번: 세로 7, 3번: 가로 7 | 하나의 차원으로 합쳐 3136 |

`start_dim=1`의 `1`은 “결과 텐서를 무조건 1차원으로 만든다”는 뜻이 아니다. <strong>펼치기 시작할 차원 번호</strong>다. 배치를 남겼으므로 전체 결과 `(2, 3136)`은 여전히 2차원이다.

반대로 이 입력에 `flatten()`을 인자 없이 호출하면 배치까지 합쳐 `(6272,)`가 된다. 그러면 두 이미지의 구분이 모양에서 사라지므로 현재 모델에는 맞지 않는다.

## 9. `Linear(3136, 10)`의 두 숫자는 어디에서 왔는가

`nn.Linear(in_features, out_features)`는 입력값마다 가중치를 곱하고 더해서 출력값들을 만드는 층이다. `in_features`는 <strong>샘플 하나의 입력값 수</strong>, `out_features`는 <strong>샘플 하나의 출력값 수</strong>다. 이 인자에도 배치 크기는 넣지 않는다.[^linear]

| 숫자 | 정해지는 이유 |
| ---: | --- |
| 3136 | 앞쪽 층에서 이미지 한 장에 남긴 값이 `64×7×7개`이기 때문이다. |
| 10 | 이번 모델이 클래스 10개의 점수를 하나씩 출력하도록 설계했기 때문이다. |

합성곱 부분은 여러 위치에 가중치를 재사용하며 중간 특징값을 만든다. 이 모델의 마지막 Linear는 펼쳐진 <strong>3136개 값을 함께 사용해 이미지 전체의 클래스 점수 10개</strong>를 만든다. Linear는 회귀 전용 층이 아니며, 분류에서도 이런 점수를 만드는 데 사용한다.

### 출력 점수 하나마다 가중치 3136개가 필요하다

이미지 하나를 펼친 값을 `x[0], …, x[3135]`라고 하자. 클래스 번호 `j`는 0부터 9까지다. `W[j,i]`는 입력 `i`를 클래스 `j`의 점수에 반영할 가중치이고, `b[j]`는 해당 점수의 편향이다.

<div class="cnn-math">
$$\text{score}[j]=x[0]W[j,0]+x[1]W[j,1]+\cdots+x[3135]W[j,3135]+b[j]$$
</div>

출력 점수 하나에 가중치가 3136개 필요하고, 점수가 10개이므로 이런 묶음이 10개 필요하다. PyTorch는 이를 <strong><code>weight.shape=(10, 3136)</code></strong>, 편향은 <strong><code>bias.shape=(10,)</code></strong>으로 저장한다.

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/linear-weight-layout.svg" alt="Linear 입력은 (2,3136), 저장 가중치는 (10,3136)이다. 가중치를 전치한 (3136,10)과 행렬곱하고 클래스별 편향을 더하면 (2,10) 점수가 나온다." loading="lazy">
<figcaption>가중치의 한 행에 출력 점수 하나를 만드는 가중치 3136개가 들어 있다. 이미지 두 장은 같은 가중치를 사용한다.</figcaption>
</figure>

<strong>전치</strong>는 2차원 행렬의 행과 열을 바꾸는 것이다. `W.T`라고 쓴다. `@`는 왼쪽의 한 행과 오른쪽의 한 열에서 대응하는 숫자를 곱해 더하는 <strong>행렬곱</strong>이다. 두 행렬의 안쪽 크기가 맞아야 하며, 이 경우 실제 계산은 다음과 같다.

<div class="cnn-math">
$$\underbrace{X}_{(2,3136)}\;@\;\underbrace{W^{\mathsf T}}_{(3136,10)}+\underbrace{b}_{(10,)}=\underbrace{Y}_{(2,10)}$$
</div>

편향 10개는 각 이미지의 점수 10개에 똑같이 더해진다. 직접 가중치를 `(입력 수, 출력 수)`로 만들었다면 `X @ W`로 쓸 수 있다. PyTorch의 `nn.Linear`는 저장 방향이 `(출력 수, 입력 수)`이므로 `X @ W.T + b`와 같은 계산을 한다.

### 마지막 값 10개는 아직 확률이 아니다

<strong>로짓(logits)</strong>은 확률로 바꾸기 전의 클래스별 점수다. 음수여도 되고, 10개를 더한 값이 1일 필요도 없다. 가장 큰 점수의 위치를 고르면 예측 클래스 번호가 된다.

학습에서는 `(N, 10)`인 로짓과, 이미지마다 정답 클래스 번호 하나를 담은 `(N,)` 정답을 비교한다. 이 비교로 얻는 학습용 숫자를 <strong>손실</strong>이라고 한다. <strong>Softmax</strong>는 한 이미지의 클래스 점수들을 합이 1인 확률로 바꾸는 함수다. `CrossEntropyLoss`는 로짓을 직접 받는 분류 손실이므로, 입력하기 전에 Softmax로 확률로 바꾸지 않는다.[^loss]

## 10. 모델 전체의 크기와 파라미터 수를 맞춘다

이제 입력을 <strong>흑백 이미지 2장, 각 28×28</strong>로 정하고 전체 구조를 살펴보자. 합성곱은 3×3 커널과 패딩 1로 공간 크기를 유지하고, 두 번의 최대 풀링이 `28 → 14 → 7`로 줄인다.

| 단계 | 입력 shape | 가중치 shape | 출력 shape |
| --- | --- | --- | --- |
| `Conv2d(1, 32, 3, padding=1)` | `(2, 1, 28, 28)` | `(32, 1, 3, 3)` | `(2, 32, 28, 28)` |
| ReLU | `(2, 32, 28, 28)` | 없음 | `(2, 32, 28, 28)` |
| `MaxPool2d(2)` | `(2, 32, 28, 28)` | 없음 | `(2, 32, 14, 14)` |
| `Conv2d(32, 64, 3, padding=1)` | `(2, 32, 14, 14)` | `(64, 32, 3, 3)` | `(2, 64, 14, 14)` |
| ReLU | `(2, 64, 14, 14)` | 없음 | `(2, 64, 14, 14)` |
| `MaxPool2d(2)` | `(2, 64, 14, 14)` | 없음 | `(2, 64, 7, 7)` |
| `flatten(start_dim=1)` | `(2, 64, 7, 7)` | 없음 | `(2, 3136)` |
| `Linear(3136, 10)` | `(2, 3136)` | `(10, 3136)` | `(2, 10)` |

둘째 합성곱에서 필터 하나의 shape는 <strong><code>(32, 3, 3)</code></strong>이다. 앞 층에서 받은 입력 채널 32개를 모두 사용하기 때문이다. 그런 필터가 64개 있으므로 가중치 전체는 `(64, 32, 3, 3)`이다.

편향까지 포함해 학습할 숫자도 세어 보자.

| 층 | 가중치 수 | 편향 수 | 합계 |
| --- | ---: | ---: | ---: |
| 첫 합성곱 | `32×1×3×3 = 288` | 32 | 320 |
| 둘째 합성곱 | `64×32×3×3 = 18,432` | 64 | 18,496 |
| 마지막 Linear | `10×3136 = 31,360` | 10 | 31,370 |
| 전체 | 50,080 | 106 | <strong>50,186</strong> |

ReLU, 최대 풀링, Flatten에는 이 모델에서 학습할 가중치가 없다. 이미지 2장을 20장으로 바꾸어도 파라미터 50,186개는 그대로다. 같은 파라미터를 더 많은 이미지에 적용하므로 데이터 처리량은 늘어난다.

### `64×7×7`은 학습 전에 알 수 있다

마지막 채널 수 64는 개발자가 `out_channels=64`로 정했다. 세로·가로 7은 입력 크기와 앞쪽 연산에서 나왔다. 모델을 학습해야 알아낼 수 있는 숫자가 아니다.

같은 모델에 다른 크기의 이미지를 넣으면 다음과 같이 달라진다.

| 최초 입력 공간 | 첫 풀링 후 | 둘째 풀링 후 | 이미지당 펼친 값 수 | 기존 `Linear(3136,10)` |
| --- | --- | --- | ---: | --- |
| 28×28 | 14×14 | 7×7 | 3136 | 연결 가능 |
| 7×7 | 3×3 | 1×1 | 64 | 입력 개수 불일치 |
| 32×32 | 16×16 | 8×8 | 4096 | 입력 개수 불일치 |

중간 결과가 7×7이라고 해서 처음에 7×7 이미지를 입력했다는 뜻은 아니다. 또한 기본 풀링은 나머지 영역을 제외하므로, 이 구조에서는 29~31 크기도 두 번 풀링하면 7이 된다. 따라서 `Linear(3136,10)`만 보고 최초 입력이 반드시 28×28이었다고 역산할 수는 없다.

입력 모양이 연결된다는 사실과 그 크기의 이미지로 학습했다는 사실도 별개다.

## 11. PyTorch로 모델을 만들고 중간 shape를 확인한다

`nn.Module`은 PyTorch가 모델의 하위 층과 파라미터를 관리하는 기본 클래스다. 기존 클래스의 기능을 바탕으로 새 클래스를 정의하는 문법을 <strong>상속</strong>이라고 한다. 이를 이용한 `MyCNN` 안에 필요한 층과 실행 순서를 정의한다.[^module]

| 문법 또는 메서드 | 역할 |
| --- | --- |
| `__init__` | `MyCNN()`으로 객체를 만들 때 실행된다. 사용할 층을 준비한다. |
| `self` | 현재 모델 객체를 가리킨다. `self.conv1` 등에 층을 보관한다. |
| `super().__init__()` | 부모 클래스 `nn.Module`의 초기화 작업을 수행한다. |
| `features` | 이 예제에서 직접 정한 메서드 이름이다. 합성곱·ReLU·풀링까지 계산한다. |
| `forward` | 실제 입력을 어떤 순서로 계산해 반환할지 정의한다. |
| `model(x)` | PyTorch의 호출 과정을 거쳐 `forward`를 실행한다. |

`fc`는 입력값 전체를 각 출력의 계산에 연결하는 <strong>완전연결층(fully connected layer)</strong>을 가리키려고 붙인 이름이다. 여기서는 `nn.Linear` 객체를 보관한다.

`__init__`에 층을 적은 순서만으로 데이터가 자동으로 흐르지는 않는다. <strong><code>forward</code>와 그 안에서 호출한 메서드의 실행 순서가 데이터의 계산 순서</strong>다.

아래 블록은 앞의 손계산 예제와 별도로 그대로 실행할 수 있다. `trace=True`를 전달하면 각 단계의 shape를 출력한다. `trace`는 이 예제에서 추가한 확인용 인자다.

```python
import torch  # 텐서와 장치 선택 기능을 사용한다.
from torch import nn  # 신경망 층과 모델 기본 클래스를 사용한다.
import torch.nn.functional as F  # ReLU 계산 함수를 사용한다.


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()  # 하위 층과 파라미터를 관리할 기본 기능을 초기화한다.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 입력 1채널 → 출력 32채널. 28×28 공간 크기는 유지한다.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 입력 32채널 → 출력 64채널. 들어오는 공간 크기를 유지한다.
        self.pool = nn.MaxPool2d(kernel_size=2)  # 기본 stride도 2다.
        self.fc = nn.Linear(64 * 7 * 7, 10)  # 이미지당 값 3136개 → 점수 10개.

    def features(self, x, trace=False):
        if trace:
            print("입력:", tuple(x.shape))  # tuple은 shape를 일반 튜플로 표시한다.
        x = self.conv1(x)
        if trace:
            print("conv1:", tuple(x.shape))
        x = F.relu(x)  # 음수만 0으로 바꾼다. shape는 유지한다.
        if trace:
            print("ReLU 1:", tuple(x.shape))
        x = self.pool(x)  # 28×28 → 14×14.
        if trace:
            print("Pool 1:", tuple(x.shape))
        x = self.conv2(x)
        if trace:
            print("conv2:", tuple(x.shape))
        x = F.relu(x)
        if trace:
            print("ReLU 2:", tuple(x.shape))
        x = self.pool(x)  # 14×14 → 7×7. 같은 풀링 객체를 다시 호출한다.
        if trace:
            print("Pool 2:", tuple(x.shape))
        return x  # 마지막 특징 맵을 반환한다.

    def forward(self, x, trace=False):
        x = self.features(x, trace=trace)  # 합성곱·활성화·풀링을 순서대로 실행한다.
        x = x.flatten(start_dim=1)  # 배치 차원을 남기고 이미지당 값을 펼친다.
        if trace:
            print("Flatten:", tuple(x.shape))
        if x.shape[1] != self.fc.in_features:  # 실제 값 수와 Linear의 요구값을 비교한다.
            raise ValueError(
                f"Linear에 필요한 값은 이미지당 {self.fc.in_features}개, "
                f"현재 전달된 값은 이미지당 {x.shape[1]}개입니다."
            )
        x = self.fc(x)  # 클래스별 로짓 10개를 만든다.
        if trace:
            print("Linear:", tuple(x.shape))
        return x


# CUDA는 NVIDIA GPU, MPS는 Apple GPU를 사용할 때의 장치 이름이다.
if torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU를 사용할 수 있으면 선택한다.
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple GPU를 사용할 수 있으면 선택한다.
else:
    device = torch.device("cpu")  # GPU를 사용할 수 없으면 CPU에서 실행한다.

my_model = MyCNN().to(device)  # 모델을 만들고 파라미터를 선택한 장치로 옮긴다.
my_model.eval()  # 평가 모드로 전환한다. 이 호출이 자동 미분을 끄지는 않는다.
probe = torch.zeros(2, 1, 28, 28, device=device)  # 연결 확인용 0 입력 두 장.

with torch.no_grad():  # 역전파 기록 없이 순전파만 확인한다.
    output = my_model(probe, trace=True)

assert output.shape == (2, 10)  # 이미지 두 장에 점수 10개씩 나왔는지 확인한다.
```

중간 출력은 다음과 같다. shape의 순서가 바뀌는 Flatten 지점까지 함께 확인한다.

```text
입력:    (2, 1, 28, 28)
conv1:   (2, 32, 28, 28)
ReLU 1:  (2, 32, 28, 28)
Pool 1:  (2, 32, 14, 14)
conv2:   (2, 64, 14, 14)
ReLU 2:  (2, 64, 14, 14)
Pool 2:  (2, 64, 7, 7)
Flatten: (2, 3136)
Linear:  (2, 10)
```

`x = self.pool(F.relu(self.conv1(x)))`처럼 한 줄로 쓴다면 안쪽의 `conv1(x)`부터 실행하고, 그 결과에 ReLU, 마지막에 풀링을 적용한다. 위 코드는 같은 계산을 나누어 적어 중간 모양을 확인한 것이다.

`eval()`은 학습·평가 모드에 따라 달라지는 층의 동작을 전환한다. 지금 사용한 층들은 이 전환으로 계산 규칙이 달라지지 않는다. `no_grad()`는 자동 미분 기록을 끄는 별도 기능이며, 서로를 대신하지 않는다.[^module][^no-grad]

현재 모델은 <strong>구조를 만들고 연결만 확인한 상태</strong>다. 입력이 모두 0이어도 편향이 있으므로 출력이 모두 0일 필요는 없다. 출력 모양이 `(2,10)`이라고 해서 이미지를 잘 분류한다는 뜻도 아니다.

### 파라미터 개수와 Linear 계산도 확인한다

`named_parameters()`는 모델의 파라미터 이름과 해당 텐서를 차례로 제공한다. `numel()`은 텐서에 들어 있는 숫자 개수를 반환한다. `parameters()`는 이름 없이 파라미터 텐서들을 제공한다.

```python
for name, parameter in my_model.named_parameters():
    print(name, tuple(parameter.shape), parameter.numel())  # 이름·shape·원소 수.

# 파라미터마다 원소 수를 구하고, Python의 sum으로 모두 더한다.
parameter_count = sum(p.numel() for p in my_model.parameters())
assert parameter_count == 50186  # 손으로 센 전체 파라미터 수와 비교한다.

with torch.no_grad():
    feature_maps = my_model.features(probe)  # (2, 64, 7, 7).
    flattened = feature_maps.flatten(start_dim=1)  # (2, 3136).
    print("Linear 입력 수:", flattened.shape[1])  # 3136: 배치 크기를 제외한 값 수.
    scores_direct = flattened @ my_model.fc.weight.T + my_model.fc.bias
    scores_layer = my_model.fc(flattened)  # 같은 계산을 Linear에 맡긴다.
    assert torch.allclose(scores_direct, scores_layer)  # 두 방식의 점수를 비교한다.
    predicted_classes = scores_layer.argmax(dim=1)  # 이미지마다 최대 점수의 위치.
    print("예측 번호의 shape:", tuple(predicted_classes.shape))  # (2,).
```

`argmax(dim=1)`은 각 이미지의 점수 10개에서 가장 큰 값이 있는 <strong>인덱스</strong>를 반환한다. 점수 10개가 클래스 번호 하나로 바뀌므로 결과 shape는 `(2,)`다. 아직 학습하지 않았으므로 이 번호를 분류 성능의 근거로 사용하지 않는다.

## 12. 학습하려면 정답·손실·역전파·갱신이 더 필요하다

<strong>순전파</strong>는 입력부터 출력까지 계산하는 과정이다. <strong>손실</strong>은 출력과 정답을 비교한 학습용 수치다. <strong>기울기</strong>는 파라미터를 조금 바꿀 때 손실이 어느 방향으로 얼마나 변하는지를 나타낸다. <strong>역전파</strong>는 계산 기록을 거슬러 각 파라미터의 기울기를 구하는 과정이고, <strong>옵티마이저</strong>는 그 기울기로 파라미터를 갱신하는 도구다.[^training]

| 단계 | 코드 | 결과 |
| --- | --- | --- |
| 구조 생성 | `MyCNN()` | 층과 초기 가중치가 생긴다. |
| 학습 모드 선택 | `model.train()` | 학습 모드로 바뀐다. 이것만으로 가중치가 바뀌지는 않는다. |
| 이전 기울기 비우기 | `optimizer.zero_grad(...)` | 이전 배치의 기울기를 비운다. |
| 순전파 | `model(images)` | 이미지마다 클래스 점수가 나온다. |
| 손실 계산 | `F.cross_entropy(logits, labels)` | 점수와 정답으로 평균 손실을 구한다. |
| 역전파 | `loss.backward()` | 파라미터별 기울기를 계산한다. |
| 갱신 | `optimizer.step()` | 기울기를 사용해 가중치와 편향을 바꾼다. |

특징 맵은 입력과 현재 가중치로 매번 새로 계산된다. 학습에서 수정하는 대상은 <strong>그 특징 맵을 만들어 내는 가중치와 편향</strong>이다.

같은 가중치를 여러 위치와 여러 이미지에서 사용했다면, 그 사용들이 손실에 준 영향이 해당 가중치의 기울기로 모인다. 위치마다 별도의 학습 가중치를 새로 만드는 것은 아니다. `weight.grad`는 `weight`와 같은 shape를 갖지만, 그 결과를 얻기 위한 계산에는 여러 위치와 여러 이미지가 기여한다.

아래 함수는 이미지 배치 하나와 그에 대응하는 정답을 받아 학습 한 단계를 실행한다. `images`는 실수 텐서 `(N,1,28,28)`, `labels`는 정답 번호를 담은 정수 텐서 `(N,)`이며 값은 0~9다. 두 텐서의 같은 순번이 같은 이미지에 대응해야 한다.

```python
def train_one_batch(model, images, labels, optimizer):
    model_device = next(model.parameters()).device  # 첫 파라미터가 있는 장치를 읽는다.
    images = images.to(device=model_device, dtype=torch.float32)  # 실수 이미지 입력.
    labels = labels.to(device=model_device, dtype=torch.long)  # 정답 번호는 정수형.
    model.train()  # 학습 모드로 바꾼다.
    optimizer.zero_grad(set_to_none=True)  # 이전 .grad를 None으로 비운다.
    logits = model(images)  # (N, 10): 확률로 바꾸지 않은 점수.
    loss = F.cross_entropy(logits, labels)  # 이미지들의 손실을 기본 설정으로 평균 낸다.
    loss.backward()  # 각 학습 파라미터의 기울기를 계산한다.
    optimizer.step()  # 옵티마이저가 실제 파라미터를 갱신한다.
    return loss.item()  # 숫자 하나인 손실 텐서에서 Python 숫자를 꺼낸다.


optimizer = torch.optim.Adam(
    my_model.parameters(),  # 갱신할 파라미터들을 전달한다.
    lr=0.001,  # 학습률: 파라미터 갱신 크기에 사용하는 설정값.
)

# 실제 이미지와 정답을 준비한 뒤 다음과 같이 호출한다.
# loss_value = train_one_batch(my_model, images, labels, optimizer)
```

`next()`는 전달받은 항목들에서 다음 항목 하나를 꺼낸다. 여기서는 첫 파라미터를 읽어 모델의 장치를 알아낸다. `torch.long`은 64비트 정수 자료형이다. `torch.optim`에는 Adam 등의 옵티마이저가 들어 있다.[^adam]

이 코드 블록은 함수를 <strong>정의</strong>하고 옵티마이저를 준비한다. 마지막 호출은 주석이므로 자동으로 학습하지 않는다. 실제 분류를 학습하려면 올바른 이미지·정답을 준비해 배치를 반복하고, 학습에 사용하지 않은 데이터로 예측을 평가해야 한다.

## 13. 크기가 맞지 않을 때는 어디를 확인하는가

오류가 나면 먼저 <strong>실제 입력 → 각 층의 출력 → 펼친 값 수 → Linear의 요구 입력 수</strong>를 순서대로 확인한다. 층 이름만 보면서 숫자를 바꾸면 앞뒤 연결이 다시 어긋날 수 있다.

| 증상 | 먼저 확인할 내용 |
| --- | --- |
| 입력 채널 수가 맞지 않는다. | 실제 입력의 `shape[1]`과 현재 `Conv2d.in_channels`가 같은지 본다. |
| 합성곱의 출력 공간이 예상보다 작다. | `kernel_size`, `padding`, `stride`를 확인한다. 기본 패딩은 0이다. |
| 풀링 결과가 예상보다 작다. | `MaxPool2d(2)`는 기본 스트라이드도 2라는 점을 확인한다. |
| Linear에서 행렬 크기 오류가 난다. | Flatten 결과의 `shape[1]`과 `fc.in_features`를 비교한다. |
| 이미지 두 장이 한 묶음으로 펼쳐졌다. | `flatten(start_dim=1)`인지 확인한다. |
| CPU와 GPU 관련 오류가 난다. | 입력과 모델 파라미터가 같은 장치에 있는지 확인한다. |

직접 크기를 계산하기 어렵다면 앞에서 사용한 `my_model.features(probe)`의 결과를 펼쳐 `shape[1]`을 읽으면 된다. 이 단계는 학습이 필요 없다.

### `LazyLinear`는 첫 입력에서 필요한 크기를 정한다

`nn.LazyLinear(10)`은 첫 입력의 마지막 차원 크기를 읽어 `in_features`를 정한다. 처음 `(2,3136)`을 받으면 입력 크기가 3136으로 결정된다. 그 뒤 `(2,4096)`을 넣을 때마다 자동으로 새 층을 만드는 것은 아니다. 초기화 이후에는 입력 크기가 맞아야 한다.[^lazy]

### 공간 크기가 달라도 같은 개수로 모으는 설계도 있다

`nn.AdaptiveAvgPool2d((1,1))`은 각 채널의 공간 값들을 평균 내서 채널마다 값 하나로 만든다. <strong>평균 풀링</strong>은 영역의 값들을 더한 뒤 개수로 나누는 연산이다.[^adaptive]

| 단계 | shape |
| --- | --- |
| 합성곱에서 나온 중간 결과 | `(N, 64, H, W)` |
| `AdaptiveAvgPool2d((1,1))` | `(N, 64, 1, 1)` |
| `flatten(start_dim=1)` | `(N, 64)` |
| `Linear(64,10)` | `(N, 10)` |

이 구조에서는 Linear 입력값 수가 공간 크기 `H,W`에 의존하지 않는다. 다만 기존 모델의 값을 그대로 펼치는 대신 평균으로 합치므로 <strong>계산 방식이 바뀌는 설계 변경</strong>이다. 앞쪽 합성곱과 풀링이 처리할 수 있는 입력 크기여야 한다는 조건도 남는다.

## 14. 모델을 읽을 때는 숫자의 단위를 붙인다

코드에 같은 숫자가 반복되어도 그 숫자가 세는 대상은 다를 수 있다. 층을 읽을 때 다음처럼 무엇의 개수인지 함께 적으면 연결 관계를 확인하기 쉽다.

| 표현 | 정확히 무엇을 나타내는가 |
| --- | --- |
| RGB의 3 | 이미지 한 장의 입력 채널 수 |
| 커널의 3×3 | 채널별로 읽는 공간 영역의 세로·가로 크기 |
| `out_channels=32` | 필터 32개, 따라서 출력 채널 32개 |
| 가중치 `(32,1,3,3)` | 필터 32개가 각각 채널별 커널 하나를 가진 구조 |
| 출력 `(2,32,28,28)` | 이미지 두 장 각각에 28×28 결과 배열 32개 |
| `64×7×7=3136` | 마지막 풀링 후 이미지 한 장에 남은 숫자 수 |
| `Linear`의 10 | 이미지 한 장에 출력할 클래스 점수 수 |
| 파라미터 50,186개 | 학습으로 바꾸는 가중치와 편향 숫자의 총개수 |

실제 값을 따라갈 때는 <strong>영역 선택 → 같은 위치끼리 곱셈 → 모든 입력 채널의 결과 합산 → 편향 추가 → 출력의 한 위치에 저장</strong> 순서로 보면 된다. 한 위치의 계산을 여러 위치에서 반복하면 출력 채널이 되고, 여러 필터가 각각 출력 채널을 만든다.

그 뒤에는 채널별로 ReLU와 풀링을 적용하고, 이미지마다 값을 펼쳐 Linear로 점수를 만든다. 이 과정에서 입력 데이터, 가중치, 중간 출력, 정답을 따로 구분하면 각 shape가 왜 필요한지도 함께 확인할 수 있다.

## 덧붙임: 보조 코드와 추가 조건

<details>
<summary>연습 코드의 None·try·except·____는 합성곱과 어떤 관계가 있을까?</summary>

`result = None`은 아직 계산 결과를 저장하지 않았다는 표시로 사용할 수 있다. `None`은 0이나 빈 텐서가 아닌 Python의 특별한 값이며, CNN을 실행하려면 반드시 먼저 넣어야 하는 값은 아니다.

`try`는 오류가 발생할 수 있는 코드를 실행하는 구간이고, `except NameError as error`는 정의되지 않은 이름을 사용했을 때의 오류를 받아 처리한다. `____`는 Python에서 사용할 수 있는 이름이다. 연습 자료에서는 채워 넣을 부분을 표시하는 용도로 쓰기도 한다. 정의하지 않고 그대로 실행하면 `NameError`가 난다.

`if error.name != "____": raise`는 빈칸 이름 때문에 생긴 오류만 따로 안내하고, 다른 이름 때문에 생긴 오류는 다시 발생시키겠다는 뜻이다. `raise`는 이 위치에서 받은 오류를 다시 전달한다. 이 보조 코드가 합성곱이나 학습을 수행하거나, shape 오류까지 해결해 주는 것은 아니다.[^python]

`count_params(model)` 같은 이름도 별도로 만든 함수일 수 있다. 이름만으로 PyTorch 내장 기능이라고 판단하지 말고 함수 정의를 확인하면 된다. 이 글은 `sum(p.numel() for p in model.parameters())`로 파라미터 원소 수를 직접 셌다.

</details>

<details>
<summary>groups가 1이 아닌 합성곱은 어떻게 다른가?</summary>

`groups`가 1보다 크면 입력·출력 채널을 여러 그룹으로 나누고, 각 출력 채널은 자기 그룹에 해당하는 입력 채널들만 사용한다. 따라서 일반 가중치 shape는 `(C_out, C_in/groups, K_H, K_W)`다. 입력 채널 수와 출력 채널 수는 모두 `groups`로 나누어떨어져야 한다.[^conv]

이 글의 “필터 하나가 모든 입력 채널을 사용한다”는 설명과 가중치 수 계산은 `groups=1`을 전제로 한다. 필터를 “출력 채널 하나를 만드는 가중치 묶음”이라고 부르는 기준은 그대로 사용할 수 있다.

필터 수를 늘리면 가중치와 출력 채널이 늘지만, 모든 필터가 서로 다른 특징을 학습한다는 보장은 없다. 또한 모든 CNN이 반드시 Flatten과 Linear로 끝나는 것도 아니다. 앞에서 살펴본 평균 풀링처럼 다른 출력 구조를 선택할 수 있다.

</details>

<details>
<summary>Python과 Rust에서 같은 출력 크기를 계산해 보기</summary>

합성곱의 출력 위치를 세는 원리는 언어와 관계없이 같다. 아래는 <strong>실제 합성곱이나 자동 미분이 아닌, 출력 길이를 계산하는 함수</strong>다. 두 코드 모두 정수 크기와 양쪽 대칭 패딩을 사용하며, 커널이 입력의 계산 영역 안에 들어가는지 먼저 검사한다.

Python의 `//`는 내림 나눗셈이다. 이 함수에서는 분자가 0 이상이고 분모가 양수이므로, 나눈 결과의 소수 부분을 버리는 것과 같다.

```python
def conv_output_length(input_length, kernel, padding=0, stride=1, dilation=1):
    # 모든 인자는 정수 크기다. 입력·커널·이동 간격·적용 간격은 양수여야 한다.
    if input_length <= 0 or kernel <= 0 or stride <= 0 or dilation <= 0:
        raise ValueError("입력·커널·stride·dilation은 양수여야 합니다.")
    if padding < 0:
        raise ValueError("패딩은 0 이상이어야 합니다.")
    effective_kernel = dilation * (kernel - 1) + 1  # 간격을 포함한 커널 적용 범위.
    padded_length = input_length + 2 * padding  # 양쪽 패딩을 더한 입력 길이.
    if padded_length < effective_kernel:
        raise ValueError("커널이 패딩을 포함한 입력보다 큽니다.")
    return (padded_length - effective_kernel) // stride + 1  # 시작 위치의 개수.


assert conv_output_length(7, 3) == 5  # 패딩이 없으면 출력 길이 5.
assert conv_output_length(28, 3, padding=1) == 28  # 패딩 1이면 길이 유지.
```

Rust의 `usize`는 배열 길이나 인덱스에 주로 쓰는 부호 없는 정수형이다. 입력값의 자료형을 명시하며, 정수끼리 `/`를 적용하면 정수 몫을 구한다. 아래처럼 음수가 없는 범위에서는 Python의 `//`와 같은 결과다. 함수의 반환형은 `-> usize`이고, 마지막 식에 세미콜론을 붙이지 않아 그 값을 반환한다.[^rust]

```rust
fn conv_output_length(
    input: usize,     // 원본 입력 길이.
    kernel: usize,    // 커널 가중치 개수.
    padding: usize,   // 한쪽에 추가할 패딩 길이.
    stride: usize,    // 계산 영역을 이동하는 간격.
    dilation: usize,  // 커널 가중치를 적용할 입력 위치 사이의 간격.
) -> usize {
    assert!(input > 0 && kernel > 0 && stride > 0 && dilation > 0);
    let effective_kernel = dilation * (kernel - 1) + 1;
    let padded = input + 2 * padding;
    assert!(padded >= effective_kernel); // 음수가 될 뺄셈을 하기 전에 검사한다.
    (padded - effective_kernel) / stride + 1 // 가능한 시작 위치 수를 반환한다.
}

fn main() {
    assert_eq!(conv_output_length(7, 3, 0, 1, 1), 5);
    assert_eq!(conv_output_length(28, 3, 1, 1, 1), 28);
    let after_pool1 = 28_usize / 2; // 2×2, stride 2인 첫 풀링의 출력 길이: 14.
    let after_pool2 = after_pool1 / 2; // 같은 설정의 둘째 풀링: 7.
    let in_features = 64 * after_pool2 * after_pool2;
    println!("Flatten: {}", in_features); // 이미지당 3136개 값.
}
```

Python은 함수 인자에 기본값을 둘 수 있지만 위 Rust 함수는 다섯 인자를 모두 받는다. 어느 언어든 출력 크기를 계산하는 함수만으로 CNN 모델이 만들어지거나 학습되는 것은 아니다.

</details>

## 참고 자료

예제의 API 동작과 크기 규칙은 2026년 10월 27일 PyTorch 공식 문서를 기준으로 확인했다. 본문 코드의 합성곱 수치, 중간 shape, 파라미터 수는 CPU 환경에서 직접 실행해 비교했다.

[^conv]: PyTorch [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): 채널 연결, 인자, 가중치·입력·출력 shape, 패딩과 초기화.
[^sum]: PyTorch [torch.sum](https://docs.pytorch.org/docs/stable/generated/torch.sum.html): 전체 합과 지정한 차원의 합산. [Tensor.tolist](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html), [Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html): Python 값으로 변환.
[^pool]: PyTorch [MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html): 채널별 최대 풀링, 기본 stride, 출력 크기.
[^relu]: PyTorch [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html): 원소별 변환과 입력·출력 shape.
[^flatten]: PyTorch [torch.flatten](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html): 펼치는 차원의 범위와 원소 순서.
[^linear]: PyTorch [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html): 가중치 저장 방향과 선형 계산.
[^module]: PyTorch [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html): 모델 구성, 호출, 파라미터, 학습·평가 모드.
[^no-grad]: PyTorch [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html): 자동 미분 기록을 끄는 구간.
[^loss]: PyTorch [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): 로짓, 정답 클래스 번호, 평균 손실.
[^training]: PyTorch [Neural Networks](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html): 모델 정의부터 손실·역전파·갱신까지의 순서.
[^adam]: PyTorch [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html): 파라미터와 학습률 설정.
[^lazy]: PyTorch [LazyLinear](https://docs.pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html): 첫 입력으로 입력 크기를 정하는 초기화.
[^adaptive]: PyTorch [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html): 지정한 공간 크기로 평균 풀링.
[^python]: Python [try 문](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement), [NameError](https://docs.python.org/3/library/exceptions.html#NameError), [None](https://docs.python.org/3/library/constants.html#None).
[^rust]: Rust 공식 책 [정수 자료형](https://doc.rust-lang.org/book/ch03-02-data-types.html), [함수와 반환값](https://doc.rust-lang.org/book/ch03-03-how-functions-work.html).
