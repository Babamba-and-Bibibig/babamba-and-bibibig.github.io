+++
title = "CNN 잔차 연결의 크기 맞추기: stride, 패딩, 1×1 Conv와 Identity"
description = "같은 입력에서 출발한 두 경로를 어떻게 더할까? 채널과 공간 크기를 나누어 읽고, 커널·패딩·stride의 관계를 계산한 뒤 Identity와 1×1 합성곱을 선택하는 잔차 블록을 구현한다."
date = 2026-10-29
slug = "cnn-residual-projection"

[extra]
katex = true
toc = true
styles = ["cnn-residual-projection/article.css"]
+++

잔차 연결에서는 합성곱으로 계산한 결과에 입력에서 이어진 다른 경로의 결과를 더한다. 그런데 합성곱을 통과하며 채널 수나 높이·너비가 바뀌었다면, 원래 입력을 그대로 더할 수 없다. 덧셈 지점에서 두 결과의 크기를 맞춰야 한다.

입력 `(2, 8, 16, 16)`을 출력 `(2, 16, 8, 8)`로 바꾸는 블록을 만들어 보자. 이 과정에서 <strong>채널은 8개에서 16개로 늘고, 높이와 너비는 각각 16에서 8로 줄어든다.</strong> 이 두 변화를 나누어 확인하면 `Identity`를 사용할 때와 `1×1 Conv`를 사용할 때를 구분할 수 있다.

설명은 전체 연결 그림에서 시작한다. 이어서 각 축의 의미, 출력 크기를 정하는 인자, 두 경로의 크기가 같아지는 이유를 확인하고 코드를 작성한다. [앞선 VGG·잔차 연결 글](/cnn-vgg-residual-pytorch/)에서 다룬 내용 중, 이번에는 두 경로의 크기를 맞추는 부분을 자세히 살펴본다.

[주석을 포함한 전체 Python 코드](/cnn-residual-projection/examples-ko.py)와 [Rust 조건·크기 검산 코드](/cnn-residual-projection/shape-checks-ko.rs)를 내려받아 실행할 수 있다. Python 예제는 PyTorch가 설치된 환경에서 CPU로 실행하며, 모델의 구조와 계산값을 확인한다.

## 1. 같은 입력이 두 경로로 들어간다

<strong>CNN</strong>은 합성곱을 사용하는 신경망이다. 합성곱은 입력의 일정한 공간 범위에 가중치를 곱하고 더하여 새 값을 계산한다. <strong>가중치</strong>는 입력값에 곱하는 숫자다. <strong>계층</strong>은 한 단계의 계산을 맡는 구성 요소이고, <strong>블록</strong>은 여러 계층과 연결을 묶은 단위다.

<strong>잔차 연결(residual connection)</strong>은 주 경로의 결과에 입력에서 이어진 다른 경로의 결과를 더하는 연결이다. 입력을 더하는 구조는 깊은 신경망의 학습을 돕기 위해 사용한다. 여기서는 그 연결을 실제 텐서로 구현하는 데 집중한다.[^resnet]

그림을 읽는 데 필요한 이름은 다음과 같다.

| 이름 | 이 블록에서 하는 일 |
| --- | --- |
| 주 경로 | 합성곱 두 층을 거쳐 새로운 특징값을 계산한다. |
| Shortcut | 주 경로의 두 합성곱을 거치지 않고 덧셈 지점으로 이어진다. |
| Projection | 이 글에서는 Shortcut에 `1×1` 합성곱을 넣어 채널·공간 크기를 맞추는 변환을 뜻한다. |
| Identity | 입력을 바꾸지 않고 그대로 반환하는 계층이다. |
| ReLU | 음수는 0으로 바꾸고, 0 이상인 값은 유지하는 함수다. shape는 유지한다. |

<strong>shape</strong>는 텐서의 각 축 크기다. 아래 그림의 숫자는 `(데이터 개수, 채널 수, 높이, 너비)` 순서이며, `K`는 커널 크기, `P`는 한쪽 패딩, `S`는 이동 간격인 stride를 나타낸다. 각 인자의 역할은 뒤에서 계산과 함께 확인한다.

<figure class="residual-figure">
<img src="/cnn-residual-projection/residual-paths-ko.svg" alt="입력 (2,8,16,16)이 주 경로와 Shortcut으로 나뉜다. 주 경로는 3×3 합성곱, ReLU, 3×3 합성곱을 거친다. Shortcut은 원래 입력에 1×1 합성곱을 적용한다. 두 결과 (2,16,8,8)을 더한 뒤 ReLU를 적용한다." loading="lazy">
<figcaption>두 경로는 같은 원래 입력 x에서 출발한다. Shortcut에 들어가는 것은 첫 번째 합성곱의 출력이 아니다.</figcaption>
</figure>

주 경로는 `conv1 → ReLU → conv2`를 계산한다. Shortcut은 별도로 원래 입력 `x`를 계산한다. 두 결과가 준비되면 같은 위치의 값끼리 더하고, 합산 결과에 ReLU를 적용한다.

그림에서 경로가 나뉘어 있다고 Python 코드가 두 경로를 반드시 동시에 실행한다는 뜻은 아니다. 그림은 <strong>어느 입력이 어느 계산에 연결되는지</strong>를 나타낸다.

## 2. 채널 수와 공간 크기를 따로 읽는다

<strong>텐서</strong>는 숫자들을 일정한 형태로 저장하고 계산하는 PyTorch의 자료형이다. 숫자의 위치를 구분하는 기준을 <strong>축</strong>이라고 한다. 이미지 배치를 다루는 이 글에서는 축 순서를 `(N, C, H, W)`로 사용한다.

<strong>배치</strong>는 한 번에 처리하는 데이터 묶음이다. <strong>채널</strong>은 데이터 하나 안에서 서로 다른 값의 배열을 구분한다. RGB 이미지에는 색 성분별 세 채널이 있고, 합성곱을 거친 중간 데이터에는 계산으로 만든 여러 채널이 있다. 이 글의 입력 8채널은 그런 중간 데이터를 생각하면 된다.

입력 `x`의 shape가 `(2, 8, 16, 16)`일 때 각 숫자는 다음 뜻이다.

| 축 번호 | 기호 | 의미 | 축의 크기 | 유효 인덱스 |
| --- | --- | --- | ---: | --- |
| 0 | `N` | 한 번에 처리하는 데이터 수 | 2 | 0~1 |
| 1 | `C` | 데이터 하나의 채널 수 | 8 | 0~7 |
| 2 | `H` | 채널 하나의 높이 | 16 | 0~15 |
| 3 | `W` | 채널 하나의 너비 | 16 | 0~15 |

<strong>인덱스</strong>는 선택할 위치의 번호이며 0부터 시작한다. 축 번호 1, 채널 개수 8, 채널 인덱스 0~7은 서로 다른 정보다. `x[0, 3, 4, 6]`은 첫 데이터의 채널 인덱스 3에서 높이 인덱스 4, 너비 인덱스 6에 있는 숫자 하나를 선택한다.

계산으로 얻은 값을 높이·너비 위치에 배치한 배열을 <strong>특징 맵</strong>이라고 한다. 이 글에서 공간 크기는 특징 맵의 `H×W`이며, 채널 개수 `C`는 공간 크기에 포함하지 않는다.

| 구분 | 입력 `(2,8,16,16)` | 출력 `(2,16,8,8)` | 변화 |
| --- | ---: | ---: | --- |
| 데이터 수 `N` | 2 | 2 | 유지 |
| 채널 수 `C` | 8 | 16 | 증가 |
| 높이 `H` | 16 | 8 | 감소 |
| 너비 `W` | 16 | 8 | 감소 |

출력의 유효 인덱스는 `N: 0~1`, `C: 0~15`, `H: 0~7`, `W: 0~7`이다. 채널이 늘었다고 입력 이미지가 16장으로 늘어난 것은 아니다.

## 3. Conv2d의 인자에서 두 가지 정보를 찾는다

`nn.Conv2d`는 2차원 합성곱 계층을 만드는 클래스다. `nn`은 PyTorch의 계층들이 들어 있는 `torch.nn`을 가리키며, 코드에서는 `from torch import nn`으로 가져온다.

다음 호출은 입력 8채널을 출력 16채널로 바꾸는 합성곱 계층을 만든다.

`nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)`

| 인자 | 의미 | 예제의 설정 |
| --- | --- | --- |
| `in_channels` | 이 계층이 받는 입력 채널 수 | 8 |
| `out_channels` | 이 계층이 만드는 출력 채널 수 | 16 |
| `kernel_size`, `K` | 입력 채널별로 한 번에 읽는 공간 범위 | `3×3` |
| `padding`, `P` | 위·아래·왼쪽·오른쪽에 각각 추가하는 칸 수 | 각 방향 1칸 |
| `stride`, `S` | 다음 계산 위치로 이동하는 간격 | 각 방향 2칸 |

기본 패딩 방식은 추가한 영역을 0으로 취급한다. 따라서 `P=1`이면 높이는 위아래 합계 2칸, 너비도 좌우 합계 2칸이 추가된 상태로 계산한다.[^conv]

출력 shape를 읽을 때는 먼저 두 질문을 나눈다.

1. <strong>출력 채널은 몇 개인가?</strong> `out_channels`를 확인한다.
2. <strong>출력 높이·너비는 얼마인가?</strong> 입력 높이·너비와 `K·P·S`를 함께 확인한다.

이 글의 범위는 `dilation=1`, `groups=1`이다. `dilation`은 커널의 이웃 원소가 읽는 입력 위치 사이의 간격이며, 1이면 붙어 있는 위치를 읽는다. `groups=1`은 각 출력 필터가 모든 입력 채널을 사용하는 기본 설정이다. `K·P·S`는 높이와 너비에 같은 정수를 사용한다.

이 전제에서 출력 채널과 공간 크기를 계산해 보자.

## 4. 출력 높이는 커널을 놓을 수 있는 위치의 개수다

입력 높이를 `H`, 커널 높이를 `K`, 위아래 각각의 패딩을 `P`, stride를 `S`라고 하자. 패딩을 포함한 높이는 `H+2P`다.

커널 전체가 들어가려면 시작 위치는 최대 `H+2P−K`까지 이동할 수 있다. 시작 위치 0에서 출발하여 `S`칸씩 이동하므로, 출력 높이는 다음과 같다.[^conv]

<div class="residual-math">
$$
H_{\mathrm{out}}=\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1
$$
</div>

`floor`, 즉 `⌊ ⌋`는 내림이다. 예를 들어 `floor(7.5)=7`이다. 마지막 `+1`은 처음 시작 위치 0에서도 한 번 계산하기 때문에 붙는다. 너비는 `H` 대신 `W`를 넣어 계산한다.

### 높이 16에 3×3, 패딩 1, stride 2를 적용한다

| 계산 단계 | 계산 | 결과 |
| --- | --- | ---: |
| 패딩까지 포함한 높이 | `16+2×1` | 18 |
| 커널이 들어가는 마지막 시작 위치 | `18−3` | 15 |
| stride 2로 실제 사용하는 시작 위치 | `0, 2, 4, 6, 8, 10, 12, 14` | 8곳 |
| 출력 높이 | `floor(15/2)+1` | 8 |

이 표의 시작 위치는 <strong>패딩까지 포함한 배열에서 센 번호</strong>다. 원래 입력의 인덱스와 구분해야 한다. 시작 위치 15에도 커널 자체는 들어가지만, 0부터 2씩 이동하는 계산에서는 15에 도착하지 않는다.

너비도 16이므로 같은 계산으로 8이 된다. `out_channels=16`까지 함께 읽으면 결과는 `(2, 16, 8, 8)`이다.

## 5. 왜 3×3과 1×1에 같은 stride를 쓰는가

주 경로의 첫 합성곱은 `K=3, P=1`이고, Shortcut의 합성곱은 `K=1, P=0`이다. 커널 크기는 다르지만 두 설정에는 공통점이 있다.

| 설정 | `K−2P` 계산 | 결과 |
| --- | --- | ---: |
| `K=3, P=1` | `3−2×1` | 1 |
| `K=1, P=0` | `1−2×0` | 1 |

출력 크기 식의 `H+2P−K`가 두 설정에서 모두 `H−1`이 된다. 따라서 같은 높이의 입력과 같은 stride를 사용하면 출력 높이도 같다.

<div class="residual-math">
$$
K-2P=1
\quad\Longrightarrow\quad
H_{\mathrm{out}}
=\left\lfloor\frac{H-1}{S}\right\rfloor+1
=\left\lceil\frac{H}{S}\right\rceil
$$
</div>

`ceil`, 즉 `⌈ ⌉`는 올림이다. `ceil(7.5)=8`이다. 양의 정수인 입력 크기와 stride에서 위 관계가 성립한다. 예를 들어 `H=15, S=2`라면 `floor(14/2)+1=8`이고, `ceil(15/2)`도 8이다.

<strong>같은 stride를 쓸 수 있는 이유는 두 경로의 커널·패딩 조합이 같은 출력 크기 식을 만들기 때문이다.</strong> stride만 같다는 사실로 판단한 것이 아니다.

<figure class="residual-figure">
<img src="/cnn-residual-projection/kps-positions-ko.svg" alt="입력 높이 16에서 stride 2를 적용한다. 3×3 패딩 1과 1×1 패딩 0은 시작 위치가 각각 8곳이다. 3×3 패딩 0은 시작 위치가 7곳이다. 각 번호는 패딩을 포함한 해당 배열의 시작 위치다." loading="lazy">
<figcaption>출력 높이는 실제로 계산하는 시작 위치의 개수다. stride가 같아도 커널과 패딩의 관계가 달라지면 개수가 달라진다.</figcaption>
</figure>

### 홀수 크기에서도 두 경로가 맞는다

| 입력 공간 크기 | stride | `3×3, P=1` 출력 | `1×1, P=0` 출력 |
| --- | ---: | --- | --- |
| `16×16` | 1 | `16×16` | `16×16` |
| `16×16` | 2 | `8×8` | `8×8` |
| `15×13` | 1 | `15×13` | `15×13` |
| `15×13` | 2 | `8×7` | `8×7` |

`stride=2`에서 높이 15는 7이 아니라 8이 된다. 이 설정에서는 높이와 너비를 각각 2로 나눈 뒤 올림한 크기가 나온다.

홀수 커널에 `P=(K−1)/2`를 사용하면 같은 관계를 만들 수 있다. `K/P`가 `1/0`, `3/1`, `5/2`, `7/3`인 경우가 모두 해당한다. `S=1`이면 공간을 유지하고, `S=2`이면 각 크기를 올림한 절반으로 만든다.

더 일반적으로, 이 글의 전제에서 같은 입력에 적용하는 두 합성곱의 `K−2P`와 `S`가 각각 같으면 출력 공간 크기가 같다. 이는 크기를 맞출 수 있는 충분한 조건이다. 특정 입력에서 출력 크기가 우연히 같았다고 두 설정이 모든 입력에서 같아지는 것은 아니다.

## 6. padding=1을 그대로 옮기면 왜 틀리는가

`3×3 Conv`에 쓰던 패딩 1을 `1×1 Conv`에도 그대로 넣으면 이번 블록의 크기가 맞지 않는다.

| 입력 높이 | `K` | `P` | `S` | 출력 높이 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 1 | 2 | `floor((16+2−3)/2)+1 = 8` |
| 16 | 1 | 0 | 2 | `floor((16−1)/2)+1 = 8` |
| 16 | 1 | 1 | 2 | `floor((16+2−1)/2)+1 = 9` |
| 16 | 3 | 0 | 2 | `floor((16−3)/2)+1 = 7` |

`1×1 Conv`에 패딩 1을 설정하는 것 자체가 금지된 것은 아니다. 다만 이 경우 Shortcut이 `9×9`를 만들기 때문에 주 경로의 `8×8`과 맞지 않는다. 필요한 설정은 <strong>`K=1, P=0, S=2`</strong>다.

마지막 행은 stride만 맞춰서는 부족한 또 다른 예다. `3×3, P=0, S=2`는 `7×7`을 만들므로 `1×1, P=0, S=2`의 `8×8`과 다르다.

## 7. conv2는 첫 번째 합성곱의 크기를 유지한다

Shortcut은 주 경로의 <strong>최종 결과</strong>와 크기가 같아야 한다. 따라서 `conv1`만 확인하고 끝내면 안 된다. 그 뒤에 있는 계층이 크기를 다시 바꾸는지도 확인해야 한다.

| 경로 | 계층 | 채널 변화 | `K, P, S` | 출력 shape |
| --- | --- | --- | --- | --- |
| 주 경로 | `conv1` | `8→16` | `3, 1, 2` | `(2,16,8,8)` |
| 주 경로 | ReLU | `16→16` | 해당 없음 | `(2,16,8,8)` |
| 주 경로 | `conv2` | `16→16` | `3, 1, 1` | `(2,16,8,8)` |
| Shortcut | `1×1 Conv` | `8→16` | `1, 0, 2` | `(2,16,8,8)` |

`conv2`는 `K=3, P=1, S=1`이므로 들어온 높이·너비를 유지한다. ReLU도 shape를 바꾸지 않는다. 그래서 `conv1`과 크기가 맞는 Shortcut을 만들면 주 경로의 최종 결과와도 맞는다.

반대로 `conv2`까지 `stride=2`로 만들면 주 경로는 `16×16 → 8×8 → 4×4`로 바뀐다. 기존 Shortcut은 여전히 `8×8`이므로 두 결과를 더할 수 없다.

코드의 `stride=stride`도 구분해 읽어야 한다. 왼쪽은 `Conv2d`가 받는 인자의 이름이고, 오른쪽은 블록의 `__init__`이 받은 변수다. 블록에 `stride=2`를 전달했다고 모든 합성곱에 그 값이 자동으로 적용되는 것은 아니다. 개발자가 전달한 계층에만 적용된다.

## 8. 1×1 합성곱은 같은 위치의 모든 입력 채널을 계산한다

`1×1`은 <strong>공간 범위</strong>를 나타낸다. 입력 채널이 하나라는 뜻도, 필터의 가중치가 하나라는 뜻도 아니다.

<strong>편향</strong>은 입력값과 가중치의 곱셈 결과들을 합한 뒤 추가하는 숫자다. 학습으로 조정하는 가중치와 편향을 함께 <strong>파라미터</strong>라고 한다. <strong>필터</strong>는 출력 채널 하나를 만드는 가중치 묶음으로 부르겠다. 문헌에서는 커널과 필터를 같은 뜻으로 쓰기도 하지만, 여기서는 공간 크기와 전체 채널 계산을 구분한다.

일반적인 `3×3` 합성곱에서 입력이 8채널이면 필터 하나의 가중치는 `8×3×3=72`개다. 같은 8채널에 `1×1` 합성곱을 쓰면 필터 하나의 가중치는 `8×1×1=8`개다. 공간 범위는 줄었지만 입력 채널 여덟 개를 함께 계산한다는 연결은 유지된다.

<figure class="residual-figure">
<img src="/cnn-residual-projection/pointwise-channels-ko.svg" alt="같은 공간 위치의 입력 8채널 값이 모든 출력 필터에 연결된다. 필터마다 가중치 8개로 곱하고 합한 뒤 편향을 더해 자기 출력 채널의 한 값을 만든다. 필터 16개가 출력 16채널을 만든다." loading="lazy">
<figcaption>입력값 8개를 필터마다 서로 다른 가중치 8개로 계산한다. 출력 채널 수 16은 필터 개수다.</figcaption>
</figure>

### 출력 한 칸을 숫자로 확인한다

한 공간 위치의 입력값과 첫 번째 필터의 가중치를 다음처럼 정해 보자. 이 숫자들은 계산을 확인하기 위해 지정한 예시이며, 학습된 필터의 값이라는 뜻은 아니다.

| 입력 채널 인덱스 | 입력값 | 필터 0의 가중치 | 곱셈 결과 |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0.5 | 0.5 |
| 1 | 2 | −1 | −2 |
| 2 | 0 | 2 | 0 |
| 3 | −1 | 1 | −1 |
| 4 | 3 | 0 | 0 |
| 5 | 1 | 0.25 | 0.25 |
| 6 | 2 | −0.5 | −1 |
| 7 | −2 | 1 | −2 |

곱셈 결과 여덟 개를 더하면 `−5.25`다. 이 필터의 편향이 `0.5`라면 출력 채널 0의 해당 위치 값은 `−4.75`가 된다. 나머지 15개 필터도 같은 입력값 여덟 개를 사용하지만, 각자 가진 가중치와 편향으로 다른 출력값을 계산한다.

### 가중치 shape에는 배치 축이 없다

`Conv2d(8, 16, 1)`의 전체 가중치 shape는 `(16, 8, 1, 1)`이다.[^conv]

| 가중치 축 번호 | 의미 | 크기 | 유효 인덱스 |
| --- | --- | ---: | --- |
| 0 | 출력 필터 수 | 16 | 0~15 |
| 1 | 필터 하나가 받는 입력 채널 수 | 8 | 0~7 |
| 2 | 채널별 커널 높이 | 1 | 0 |
| 3 | 채널별 커널 너비 | 1 | 0 |

이 텐서의 첫 번째 16은 데이터 16개가 아니라 필터 16개다. 전체 가중치는 `16×8×1×1=128`개이고, 기본값인 `bias=True`를 사용하면 편향 16개가 별도로 있다. 가중치와 편향을 합하면 학습으로 조정할 숫자는 144개다.

같은 필터 가중치는 여러 공간 위치와 배치의 각 데이터에 재사용된다. 배치가 1개에서 2개로 바뀌어도 계층의 가중치 개수는 늘지 않는다.

### stride=2이면 어느 위치를 읽는가

`1×1, P=0, S=2`에서는 높이·너비 각각 입력 인덱스 `0, 2, 4, …`를 선택한다. 입력이 `16×16`이면 각 방향에서 `0~14` 사이의 짝수 인덱스 8개를 사용한다. 선택한 각 위치에서는 여전히 모든 입력 채널을 계산한다.

예를 들어 출력의 높이 인덱스 2, 너비 인덱스 3을 계산할 때에는 원래 입력의 높이 인덱스 4, 너비 인덱스 6을 읽는다. 따라서 이 Shortcut은 <strong>채널을 새로 계산하면서 공간 크기도 바꾼다.</strong> 입력값을 복사하거나 텐서의 모양만 바꾸는 연산은 아니다.

## 9. Identity를 고르는 조건은 어디서 나오는가

지금까지 확인한 주 경로의 설정은 고정되어 있다.

- `conv1`: `K=3, P=1`, 블록이 받은 stride 사용.
- `conv2`: `K=3, P=1, S=1`, `conv1`의 공간 크기 유지.
- 최종 채널 수: `out_channels`.

이 블록에서는 다음 두 조건을 함께 만족할 때 Identity를 선택한다. 입력 높이·너비가 달라져도 두 경로의 shape가 같게 유지되는 설정이다.

| 확인할 것 | Identity를 선택하는 조건 |
| --- | --- |
| 채널 수 유지 | `in_channels == out_channels` |
| 공간 크기 유지 | `stride == 1` |

Python의 `and`는 두 조건이 모두 참일 때 참이다. 그러므로 Identity 조건은 `in_channels == out_channels and stride == 1`이다. 반대로 `in_channels != out_channels or stride != 1`이면 이 블록에서는 projection을 사용한다. `or`는 둘 중 하나 이상이 참이면 참이다.

| 입력·출력 채널 | stride | 공간 변화 | 선택 |
| --- | ---: | --- | --- |
| `8→8` | 1 | 유지 | Identity |
| `8→16` | 1 | 유지 | `1×1 Conv`, 채널 변경 |
| `8→8` | 2 | 축소하도록 설정 | `1×1 Conv`, 공간 변경 |
| `8→16` | 2 | 축소하도록 설정 | `1×1 Conv`, 채널·공간 변경 |

<strong>stride가 1이어도 채널 수가 달라지면 projection을 사용한다.</strong> `8→16`처럼 채널을 늘릴 때도, `16→8`처럼 줄일 때도 같다.

이 조건은 위에 정한 주 경로 구조를 전제로 한다. 다른 커널·패딩이나 크기를 바꾸는 계층을 추가했다면 주 경로 전체의 출력 크기를 다시 계산해야 한다.

### 실제 입력 높이·너비를 몰라도 선택할 수 있는 이유

구체적인 `H`, `W`를 몰라도 이 블록의 크기 변화 규칙은 안다. `conv1`의 높이는 `ceil(H/S)`가 되고, `conv2`가 그 높이를 유지한다. Shortcut의 `1×1, P=0` 합성곱도 같은 stride로 `ceil(H/S)`를 만든다. 너비도 같은 관계다.

가중치 값이 학습으로 바뀌어도 커널·패딩·stride 설정이 같으면 이 크기 규칙은 그대로다. 그래서 모델을 만드는 `__init__` 단계에서 계층을 선택할 수 있다. 실제로 실행할 때에는 그 설계가 맞는지 두 경로의 shape를 다시 검사한다.

## 10. Identity를 만드는 것과 입력을 전달하는 것은 다르다

<strong>클래스</strong>는 객체가 가질 구성과 동작을 정의한다. <strong>객체</strong>는 그 정의를 바탕으로 실제로 만들어 사용하는 대상이다. `nn.Identity`는 클래스이고, `nn.Identity()`의 결과는 계층 객체다.

`self`는 현재 블록 객체를 가리킨다. `self.shortcut`은 그 블록 안에서 사용할 계층을 저장하는 속성이다.

| 코드 | 하는 일 | 결과 |
| --- | --- | --- |
| `nn.Identity()` | Identity 계층을 만든다. | 계층 객체 |
| `self.shortcut = nn.Identity()` | 만든 계층을 블록의 속성에 저장한다. | 아직 입력은 계산하지 않음 |
| `skip = self.shortcut(x)` | 저장한 계층에 입력 `x`를 전달한다. | 입력과 같은 텐서 객체 |
| `nn.Identity()(x)` | 계층을 만든 직후 입력을 전달한다. | 입력과 같은 텐서 객체 |
| `nn.Identity(x)` | 생성자에 `x`를 전달한다. | 텐서가 아니라 계층 객체 |

Identity의 생성자는 전달받은 인자를 무시하므로 `nn.Identity(x)`가 반드시 오류를 내는 것은 아니다. 하지만 이 표현으로 텐서를 계산한 것은 아니다. <strong>계층 생성과 계층 호출을 구분해야 한다.</strong>[^identity]

Identity는 입력을 복사하지 않고 같은 객체를 반환한다. 가중치와 편향도 없다. 여기서 반환은 화면에 출력한다는 뜻이 아니라 호출한 코드에 결과를 돌려준다는 뜻이다.

### Shortcut에서 그냥 x를 써도 된다

크기가 같다면 Shortcut 결과로 `skip = x`를 사용해도 된다. Identity를 쓰는 이유는 생성 단계에서 `Identity` 또는 `Conv2d`를 선택해 저장하고, 계산 단계에서는 둘 다 `self.shortcut(x)`로 호출할 수 있기 때문이다.

이때 입력을 그대로 전달하는 것은 <strong>Shortcut 경로</strong>다. 블록 전체는 주 경로를 계산하고 두 결과를 더한 뒤 ReLU를 적용한다. 블록의 `forward`를 `return x`로 바꾸면 그 계산까지 모두 생략한다.

또한 Shortcut에 Identity를 쓴다고 블록의 최종 출력이 항상 원래 입력과 같은 것도 아니다. 주 경로의 값이 더해지고, 마지막 ReLU가 음수를 0으로 바꾸기 때문이다.

## 11. 같은 위치끼리 더한 뒤 ReLU를 적용한다

두 텐서의 shape가 `(N,C,H,W)`로 같으면 각 위치의 값을 다음과 같이 더한다.

`merged[n, c, h, w] = main[n, c, h, w] + skip[n, c, h, w]`

`n, c, h, w`는 각각 데이터·채널·높이·너비의 인덱스다. 같은 위치의 값 두 개가 결과값 하나가 되므로 출력 shape는 여전히 `(N,C,H,W)`다.

<figure class="residual-figure">
<img src="/cnn-residual-projection/elementwise-add-ko.svg" alt="한 데이터의 한 채널에서 2×3 공간 일부를 선택한다. 주 경로 [[1,-2,3],[4,0,-1]]와 Shortcut [[2,1,-4],[-1,5,2]]를 같은 위치끼리 더하면 [[3,-1,-1],[3,5,1]]이다. ReLU 이후에는 [[3,0,0],[3,5,1]]이 된다." loading="lazy">
<figcaption>덧셈도 ReLU도 이 그림의 공간 칸 수를 바꾸지 않는다. 두 경로의 값을 먼저 더한 뒤 음수를 0으로 바꾼다.</figcaption>
</figure>

채널을 이어 붙이는 연산과는 다르다. 예제에서 16채널 텐서 두 개를 더해도 결과는 16채널이다. 또한 두 경로에 먼저 ReLU를 각각 적용한 뒤 더하는 것과도 다르다. 그림의 첫 번째 줄 마지막 칸은 `3+(−4)=−1`이므로 최종값이 0이다. 각 값에 먼저 ReLU를 적용하면 `3+0=3`이 되어 다른 결과가 나온다.[^relu]

PyTorch의 일반 덧셈에는 일부 축 크기가 1인 텐서를 다른 크기에 맞춰 계산하는 <strong>브로드캐스팅</strong> 규칙이 있다. 하지만 여기서는 그 기능을 사용하지 않는다. 잔차 연결을 의도한 형태로 구현했는지 확인하기 위해 두 경로의 shape가 정확히 같은지 검사한다.[^broadcast]

## 12. 블록을 구현하고 다섯 가지 입력으로 확인한다

`nn.Module`은 PyTorch에서 모델과 계층을 구성하는 기본 클래스다. <strong>상속</strong>은 기존 클래스의 기능을 이어받아 새 클래스를 정의하는 것이다. 이 클래스를 상속한 뒤 `super().__init__()`을 호출하면 하위 계층과 파라미터를 관리하는 기능을 사용할 수 있다. `self.conv1`처럼 속성에 넣은 계층은 블록의 하위 계층으로 등록된다.[^module]

클래스 안에 정의한 함수를 <strong>메서드</strong>라고 한다. 코드에서 정의할 두 메서드의 역할은 다음과 같다.

| 메서드 | 호출되는 때 | 할 일 |
| --- | --- | --- |
| `__init__` | 블록 객체를 만들 때 | 사용할 계층을 만들고 저장한다. |
| `forward` | `block(x)`로 입력을 전달할 때 | 준비한 계층들을 사용해 현재 입력을 계산한다. |

아래 블록은 합성곱 두 층과 Shortcut의 관계를 보기 위한 학습용 구현이다. 별도의 정규화 계층은 넣지 않고 편향을 사용한다. TorchVision의 실제 ResNet BasicBlock은 중간 값의 분포를 조정하는 BatchNorm 계층과 `bias=False` 합성곱을 사용한다. 여기서 가져온 것은 `3×3/P=1`과 `1×1/P=0`으로 크기를 맞추는 원리다.[^resnet-code]

실행에 쓰는 이름들도 먼저 확인하자.

| 이름 | 역할 |
| --- | --- |
| `torch` | 텐서 생성과 연산을 제공하는 패키지 |
| `nn` | 계층 클래스가 있는 `torch.nn` |
| `F` | 함수 형태의 신경망 연산이 있는 `torch.nn.functional`의 별칭 |
| `torch.randn(...)` | 지정한 shape의 무작위 실수 텐서 생성 |
| `torch.no_grad()` | 해당 계산을 역전파용 연산 기록에 넣지 않도록 설정 |
| `assert` | 조건이 거짓이면 오류를 발생시키는 검사 문장 |
| `isinstance(...)` | 객체가 지정한 클래스에 속하는지 검사 |
| `is` | 값만 같은 것이 아니라 같은 객체인지 검사 |

`no_grad`를 사용해도 출력값은 계산한다. 지금은 구조를 검사하므로 역전파용 기록을 만들지 않는 것이다. 이 설정 자체가 모델을 학습하거나 평가 모드로 바꾸지는 않는다.[^no-grad]

다음은 한 번에 실행할 수 있는 전체 Python 코드다. 마지막에는 앞서 계산한 `1×1` 합성곱의 출력 `−4.75`와 잘못된 패딩의 출력 크기도 확인한다.

```python
import torch
from torch import nn
import torch.nn.functional as F

# 같은 실행 순서에서 입력과 초기 가중치를 재현하기 쉽게 한다.
# 이번 예제의 모델과 텐서는 모두 기본 장치인 CPU에 둔다.
torch.manual_seed(16)


class MyProjectionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # nn.Module의 계층·파라미터 관리 기능을 먼저 초기화한다.
        super().__init__()

        # 주 경로 1: 채널을 바꾸고, 전달받은 stride로 공간을 조절한다.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,  # 출력 채널마다 편향 하나를 사용한다.
        )

        # 주 경로 2: 첫 합성곱의 채널·공간 크기를 유지한다.
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 이 주 경로의 K·P 설정에서 채널과 공간을 유지하는 조건이다.
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            # 같은 원래 입력을 주 경로의 최종 shape에 맞게 계산한다.
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=True,
            )

    def forward(self, x):
        # F.relu는 음수를 0으로 바꾸며 shape는 유지한다.
        main = F.relu(self.conv1(x))
        main = self.conv2(main)

        # conv1의 출력이 아니라 원래 입력 x를 전달한다.
        skip = self.shortcut(x)

        # 먼저 같은 위치의 값끼리 더하고, 그 결과에 ReLU를 적용한다.
        return F.relu(main + skip)


# 각 항목은 입력채널, 출력채널, stride, 입력shape, 예상출력shape다.
cases = [
    (8, 16, 2, (2, 8, 16, 16), (2, 16, 8, 8)),
    (8, 8, 1, (1, 8, 15, 13), (1, 8, 15, 13)),
    (8, 16, 1, (1, 8, 15, 13), (1, 16, 15, 13)),
    (8, 8, 2, (1, 8, 15, 13), (1, 8, 8, 7)),
    (16, 8, 1, (1, 16, 15, 13), (1, 8, 15, 13)),
]

for in_c, out_c, stride, input_shape, expected_shape in cases:
    block = MyProjectionResidualBlock(in_c, out_c, stride)
    x = torch.randn(input_shape)

    with torch.no_grad():
        # 두 경로를 따로 확인한다. 아직 둘을 더한 결과가 아니다.
        main = block.conv2(F.relu(block.conv1(x)))
        skip = block.shortcut(x)
        assert main.shape == skip.shape, "두 경로의 shape가 다릅니다."

        # block(x)는 PyTorch의 호출 절차를 통해 forward를 실행한다.
        y = block(x)
        assert tuple(y.shape) == expected_shape

        # allclose는 실수의 작은 계산 오차를 허용해 값들을 비교한다.
        # atol은 절대 오차, rtol은 비교값 크기에 비례하는 오차 기준이다.
        assert torch.allclose(y, F.relu(main + skip), atol=1e-6, rtol=1e-5)

    expected_identity = in_c == out_c and stride == 1
    assert isinstance(block.shortcut, nn.Identity) == expected_identity

    if expected_identity:
        # Identity는 복사본이 아니라 같은 입력 객체를 반환한다.
        assert skip is x
        shortcut_name = "Identity"
    else:
        # 가중치의 축은 출력필터, 입력채널, 커널높이, 커널너비다.
        assert tuple(block.shortcut.weight.shape) == (out_c, in_c, 1, 1)
        shortcut_name = "1x1 Conv"

    print(f"{input_shape} -> {tuple(y.shape)} | {shortcut_name}")


# 1×1 Conv의 한 출력 위치를 표의 숫자로 검산한다.
# zeros는 지정한 shape를 0으로 채운 텐서를 만든다.
point_x = torch.zeros(1, 8, 16, 16)
projection = nn.Conv2d(8, 16, kernel_size=1, stride=2, padding=0)

with torch.no_grad():
    # :는 입력 채널 8개 전체를 선택한다.
    # torch.tensor는 주어진 숫자들을 텐서로 만든다.
    point_x[0, :, 4, 6] = torch.tensor([1., 2., 0., -1., 3., 1., 2., -2.])
    projection.weight[0, :, 0, 0] = torch.tensor(
        [0.5, -1., 2., 1., 0., 0.25, -0.5, 1.]
    )
    projection.bias[0] = 0.5

    point_y = projection(point_x)
    # stride=2이므로 출력 위치 (2,3)은 입력 위치 (4,6)을 읽는다.
    actual = point_y[0, 0, 2, 3]
    expected = torch.tensor(-4.75)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

# item은 원소 하나짜리 텐서에서 일반 Python 숫자를 꺼낸다.
print("1x1 Conv의 한 출력값:", actual.item())


# 같은 stride라도 K·P가 다르면 출력 공간이 다를 수 있다.
size_x = torch.zeros(1, 8, 16, 16)
settings = [
    (3, 1, 2, (8, 8)),
    (1, 0, 2, (8, 8)),
    (1, 1, 2, (9, 9)),
    (3, 0, 2, (7, 7)),
]

with torch.no_grad():
    for kernel, padding, stride, expected_hw in settings:
        conv = nn.Conv2d(8, 16, kernel, stride=stride, padding=padding)
        # shape[2:]는 축 2·3, 즉 출력 높이·너비를 선택한다.
        actual_hw = tuple(conv(size_x).shape[2:])
        assert actual_hw == expected_hw
        print(f"K={kernel}, P={padding}, S={stride} -> {actual_hw}")


# None과 False는 이 예제에서 정한 준비 상태의 초기값이다.
projection_block = None
projection_ready = False

projection_block = MyProjectionResidualBlock(8, 16, stride=2)
with torch.no_grad():
    check_y = projection_block(torch.randn(2, 8, 16, 16))

assert tuple(check_y.shape) == (2, 16, 8, 8)
projection_ready = True  # 검사를 통과한 뒤 상태 표시를 바꾼다.
print("블록 구조 확인 완료:", projection_ready)
```

다섯 입력의 출력은 다음과 같다.

```text
(2, 8, 16, 16) -> (2, 16, 8, 8) | 1x1 Conv
(1, 8, 15, 13) -> (1, 8, 15, 13) | Identity
(1, 8, 15, 13) -> (1, 16, 15, 13) | 1x1 Conv
(1, 8, 15, 13) -> (1, 8, 8, 7) | 1x1 Conv
(1, 16, 15, 13) -> (1, 8, 15, 13) | 1x1 Conv
```

이 예제에서 `projection_block=None`은 아직 블록 객체를 저장하지 않았음을 나타내고, `projection_ready=False`는 준비 전 상태를 표시한다. `projection_ready=True`로 바꾸는 것 자체가 구조를 검사하거나 모델을 학습시키지는 않는다. 위 코드에서는 실제 출력 검사에 성공한 뒤 값을 바꾼다.

## 13. 크기가 맞지 않을 때 확인할 순서

먼저 <strong>덧셈 직전의 두 shape</strong>를 확인한다. 예를 들어 주 경로가 `(2,16,8,8)`이고 Shortcut이 `(2,8,16,16)`이라면 데이터 수 `N`은 같지만 `C`, `H`, `W`가 다르다.

그다음 두 경로를 다음 순서로 살펴본다.

1. 두 경로가 같은 원래 입력 `x`에서 출발하는지 확인한다.
2. 주 경로의 마지막 채널 수와 Shortcut의 출력 채널 수를 비교한다.
3. 각 경로의 모든 합성곱에 대해 `K·P·S`와 출력 높이·너비를 계산한다.
4. 중간 계층이 공간 크기를 한 번 더 바꾸지 않는지 확인한다.
5. 두 결과의 shape가 같아진 뒤 덧셈과 ReLU를 확인한다.

| 잘못된 설정이나 코드 | 원인 | 이 블록에서 고칠 부분 |
| --- | --- | --- |
| 채널·공간이 바뀌는데 Identity 사용 | Shortcut이 원래 `(N,C,H,W)`를 그대로 반환한다. | `1×1` projection 선택 |
| `1×1 Conv`에 `padding=1` | 예제의 Shortcut이 `9×9`를 만든다. | `padding=0` 사용 |
| `conv2`도 `stride=2` | 주 경로가 공간을 두 번 줄인다. | `conv2`는 `stride=1` 사용 |
| `self.shortcut(main)` | 원래 입력 대신 이미 계산한 결과를 받는다. | `self.shortcut(x)` 사용 |
| `nn.Identity(x)` | 입력 처리 대신 계층 객체를 만든다. | 만든 객체에 `x`를 전달 |
| `out_channesl`, `shorcut`, `Identify` | 선언한 인자·속성·클래스 이름과 다르다. | `out_channels`, `shortcut`, `Identity` 확인 |

`forward`에서 실제 shape를 비교하는 것은 정상적인 검사다. 다만 shape가 다르다는 이유로 <strong>`forward`를 호출할 때마다 새로운 Conv를 만들면</strong> 가중치도 매번 새로 초기화된다. 이미 만들어 둔 옵티마이저, 즉 파라미터를 갱신하는 객체가 새 파라미터를 자동으로 포함하는 것도 아니다. 이 예제에서는 학습할 계층을 `__init__`에서 한 번 만들고 `forward`에서 재사용한다.[^module]

## 14. 같은 조건과 크기 계산을 Rust로 표현한다

Python과 Rust는 채널 수를 비교하고 출력 크기를 계산하는 원리가 같다. 조건을 쓰는 문법과 정수 자료형 표기가 다르다.

| 의미 | Python | Rust |
| --- | --- | --- |
| 두 조건 모두 참 | `and` | `&&` |
| 하나 이상의 조건이 참 | `or` | `\|\|` |
| 두 값이 같은지 비교 | `==` | `==` |
| 양수 정수를 나누고 몫만 사용 | `//` | 정수끼리 `/` |

아래의 `usize`는 개수와 인덱스 등에 사용하는 부호 없는 정수 타입이고, `bool`은 참·거짓을 나타낸다. `->` 뒤에는 함수의 반환 타입을 적는다. 함수 끝의 세미콜론 없는 식이 반환값이 된다. `assert!`는 조건을, `assert_eq!`는 두 값이 같은지를 검사한다.[^rust]

이 Rust 코드는 계층 선택과 공간 크기 계산만 확인한다. 합성곱이나 PyTorch 모델을 Rust로 구현한 코드는 아니다.

```rust
// 이 글의 주 경로: conv1은 K=3/P=1, conv2는 K=3/P=1/S=1.
fn use_identity(in_channels: usize, out_channels: usize, stride: usize) -> bool {
    in_channels == out_channels && stride == 1
}

// dilation=1, 양쪽에 같은 정수 패딩을 사용하는 한 축의 출력 크기다.
fn output_size(input: usize, kernel: usize, padding: usize, stride: usize) -> usize {
    assert!(input > 0 && kernel > 0 && stride > 0);
    let padded = input + 2 * padding;

    // 커널이 입력에 들어가는지 검사한 뒤 부호 없는 정수를 뺀다.
    assert!(padded >= kernel);

    // 분자가 0 이상이므로 정수 나눗셈의 몫이 floor와 같다.
    (padded - kernel) / stride + 1
}

fn main() {
    assert!(use_identity(8, 8, 1));
    assert!(!use_identity(8, 16, 1));  // !는 참·거짓을 뒤집는다.
    assert!(!use_identity(8, 8, 2));

    // 두 커널·패딩 조합이 짝수와 홀수 입력 모두에서 같은 크기를 만든다.
    for input in [16_usize, 15, 13] {
        let main_size = output_size(input, 3, 1, 2);
        let skip_size = output_size(input, 1, 0, 2);
        assert_eq!(main_size, skip_size);
        println!("입력 {input}: 주 경로 {main_size}, Shortcut {skip_size}");
    }

    assert_eq!(output_size(16, 1, 1, 2), 9);  // 패딩을 1로 바꾼 경우
    assert_eq!(output_size(16, 3, 0, 2), 7);  // 3×3에서 패딩을 뺀 경우
    println!("Identity 조건과 출력 크기 확인 완료");
}
```

이 블록을 직접 바꿔 볼 때에는 먼저 출력 채널 수를 정하고, 주 경로 전체의 공간 크기 변화를 확인하면 된다. 원래 입력과 최종 shape가 같게 유지되는 설정이면 Identity를 사용한다. 크기를 바꾸는 설정이면 주 경로의 최종 결과에 맞도록 Shortcut을 설계하고, 실제 입력으로 두 shape를 확인한다.

## 참고 문서

본문의 API와 구현은 2026년 10월 29일 공식 문서를 기준으로 확인했다. 수식에 대입한 숫자와 표의 결과는 위 예제의 설정으로 계산했다.

[^resnet]: Kaiming He 외, [Deep Residual Learning for Image Recognition](https://arxiv.org/html/1512.03385v1), 3.1~3.3절. 잔차 학습과 identity·projection shortcut을 설명한다.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html). 입력·출력·가중치의 shape, 커널·패딩·stride·dilation·groups·bias의 정의와 출력 크기 식.
[^identity]: PyTorch, [Identity](https://docs.pytorch.org/docs/stable/generated/torch.nn.Identity.html), [Identity 소스](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py). 생성자 인자와 입력을 그대로 반환하는 `forward` 구현.
[^relu]: PyTorch, [functional.relu](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html). 각 원소에 적용하는 ReLU 연산.
[^broadcast]: PyTorch, [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html). 크기가 다른 텐서의 연산을 허용하는 축별 규칙.
[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [Optimizer](https://docs.pytorch.org/docs/stable/optim.html). 하위 계층·파라미터 등록, 모듈 호출, 옵티마이저에 전달하는 파라미터.
[^resnet-code]: TorchVision, [ResNet 구현](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html). `conv3x3`, `conv1x1`, `BasicBlock`, `_make_layer`의 크기 조정과 계층 구성.
[^no-grad]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html). 출력 계산을 유지하면서 해당 영역의 기울기 기록을 비활성화한다.
[^rust]: The Rust Programming Language, [Control Flow](https://doc.rust-lang.org/book/ch03-05-control-flow.html), [Data Types](https://doc.rust-lang.org/book/ch03-02-data-types.html), [Functions](https://doc.rust-lang.org/book/ch03-03-how-functions-work.html). 조건문, 정수 자료형, 함수의 반환값.
