+++
title = "합성곱 직접 계산하기: 인덱싱부터 MNIST 시각화까지"
description = "입력 패치와 커널로 출력 한 칸을 계산하고, stride·padding의 출력 크기를 유도한다. PyTorch의 자동 미분과 detach를 구분한 뒤 MNIST의 라플라시안 응답을 시각화한다."
date = 2026-10-28
slug = "cnn-convolution-practice"

[extra]
katex = true
toc = true
styles = ["cnn-convolution-practice/article.css"]
+++

합성곱 코드는 `out = conv(x)` 한 줄로 실행할 수 있다. 그런데 출력의 첫 숫자가 왜 그 값인지 확인하려면 입력에서 어느 영역을 읽었는지, 커널의 어떤 숫자를 곱했는지, 그 결과를 어디까지 더했는지 알아야 한다. 출력 크기나 시각화 코드도 이 계산에서 이어진다.

이 글에서는 작은 숫자 배열로 출력 한 칸을 직접 계산한 뒤, 같은 과정을 손글씨 이미지에 적용한다. 먼저 입력과 커널을 준비하고, 인덱싱으로 작은 영역을 선택한다. 그다음 같은 위치끼리 곱하고 더한다. 계산 위치를 옮기면서 출력 전체를 만들고, 마지막에는 그 숫자들을 화면에 표시한다.

중간에는 가중치의 변화가 출력에 어떻게 연결되는지 확인하는 자동 미분 실험도 따로 진행한다. 사람이 정한 고정 필터를 적용하는 것과 모델을 학습하는 것은 구분해서 읽으면 된다. 이 글의 예제는 가중치를 수정하는 `optimizer.step()`을 실행하지 않는다.

CNN(합성곱 신경망)의 채널 연결과 전체 모델 구조는 [앞선 글: 채널과 커널부터 Linear까지](/cnn-convolution-basics/)에서 살펴볼 수 있다. 여기서는 지금 사용하는 숫자와 코드를 이해하는 데 필요한 개념부터 설명한다. Python 코드 블록은 위에서 아래로 이어지며, [주석이 포함된 전체 실행 코드](/cnn-convolution-practice/examples-ko.py)도 함께 제공한다.

## 1. 이미지의 숫자를 담고, 축의 의미를 정한다

이미지는 위치마다 숫자가 저장된 데이터로 다룰 수 있다. 흑백 이미지라면 한 위치에 밝기값 하나가 있고, RGB 이미지라면 빨강·초록·파랑 성분을 구분해 저장한다. 이 성분을 구분하는 기준이 <strong>채널</strong>이다.

PyTorch의 <strong>텐서(tensor)</strong>는 숫자들을 저장하고 계산하는 자료형이다. 텐서에서 위치를 지정하는 각 자리를 <strong>축</strong>, 각 축에 원소가 몇 개 있는지 나타내는 정보를 <strong>shape</strong>, 실제로 선택할 위치의 번호를 <strong>인덱스</strong>라고 한다. 인덱스는 0부터 시작한다.

이번 입력은 흑백 이미지 한 장에 해당하는 4행 × 5열의 작은 배열이다. 실제 사진 대신 간단한 숫자를 사용하면 계산을 손으로 확인할 수 있다.

`torch.tensor(...)`는 전달한 숫자들로 텐서를 만드는 함수다. `reshape(...)`는 텐서의 전체 원소 수를 유지하면서 축별 크기를 다시 정하는 메서드다. 메서드는 `x.reshape(...)`처럼 객체 뒤에 붙여 호출하는 기능을 말한다.

아래의 `tuple(...)`은 여러 값을 순서대로 묶는 Python 자료형인 튜플로 바꾼다. 여기서는 shape를 `(4, 5)`처럼 읽기 쉬운 형태로 출력하려고 사용한다.

```python
import torch  # 숫자 배열인 텐서와 계산 기능을 가져온다.
from torch import nn  # 합성곱 층 등의 클래스를 nn이라는 이름으로 사용한다.
import torch.nn.functional as F  # 함수 형태의 신경망 연산을 F로 부른다.

X = torch.tensor([
    [1., 2., 0., 1., 3.],
    [0., 3., 1., 2., 1.],
    [2., 1., 0., 1., 0.],
    [1., 0., 2., 3., 2.],
])

# 4×5개의 값은 유지하고, 이미지와 채널을 선택할 축을 앞에 둔다.
x = X.reshape(1, 1, 4, 5)

print(tuple(X.shape))  # (4, 5)
print(tuple(x.shape))  # (1, 1, 4, 5)
```

`1.`은 `1.0`과 같은 실수 표기다. `X`와 `x`는 서로 다른 변수 이름이며, Python은 대소문자를 구별한다. `F`도 특별한 예약어가 아니라 모듈을 짧게 부르려고 붙인 이름이다.

이미지 여러 장을 한꺼번에 처리할 때 사용하는 데이터 묶음을 <strong>배치(batch)</strong>라고 한다. 이 글에서는 이미지가 한 장이어도 배치 형식인 `(N, C, H, W)`를 사용한다. 순서대로 이미지 수, 채널 수, 높이, 너비다.

| 축 번호 | 이 축에서 선택하는 대상 | `x.shape`에 적힌 크기 | 사용할 수 있는 인덱스 |
| --- | --- | ---: | --- |
| 0 | 이미지 | 1 | 0 |
| 1 | 선택한 이미지의 채널 | 1 | 0 |
| 2 | 행 | 4 | 0, 1, 2, 3 |
| 3 | 열 | 5 | 0, 1, 2, 3, 4 |

shape의 `1`은 원소가 한 개라는 뜻이다. 따라서 그 축에서는 인덱스 `0`만 사용할 수 있다. 이미지와 채널을 나타내는 축이 두 개 있다는 사실을 이미지나 채널이 두 개 있다는 뜻으로 읽으면 안 된다.

`reshape` 전후의 원소 수는 `4×5 = 1×1×4×5 = 20`으로 같다. 여기서는 원래 행·열의 배치도 그대로 유지한다. 이 단계에서 합성곱을 계산한 것은 아니다.

`nn.Conv2d`는 배치 축 없는 `(C, H, W)` 입력도 받는다. 여기서 4차원 형식을 사용하는 이유는 이후 이미지 개수와 채널 개수를 같은 방식으로 구분하기 위해서다.[^conv]

## 2. `x[0, 0, :3, :3]`으로 무엇을 꺼내는가

먼저 `x[0, 0]`을 읽어 보자. 첫 번째 `0`은 이미지 하나 중 첫 이미지를, 두 번째 `0`은 그 이미지의 채널 하나 중 첫 채널을 선택한다. 행과 열은 지정하지 않았으므로 전부 남는다.

```python
image_channel = x[0, 0]  # 이미지 0번과 채널 0번을 선택한다.
print(tuple(image_channel.shape))  # (4, 5)
print(torch.equal(image_channel, X))  # True: 크기와 원소값이 같다.
```

이 결과에는 숫자 두 개가 들어 있는 것이 아니라 원래의 4×5 배열 전체가 들어 있다. 이미지와 채널을 정수 인덱스로 선택했으므로 그 두 축이 결과의 shape에서 빠진 것이다.

현재 예제의 `x[0, 0]`과 `x[0][0]`은 같은 결과다. 뒤의 표기는 이미지를 먼저 선택하고, 그 결과에서 채널을 선택한다. 앞의 표기는 두 축의 인덱스를 한 대괄호에 적는다. Python이 `(0, 0)`이라는 튜플을 전달하면 PyTorch 텐서가 이를 여러 축의 선택 조건으로 해석한다. 일반적인 Python 중첩 리스트는 `data[0][0]`을 사용한다. 복잡한 인덱싱까지 두 표기가 언제나 같다고 확장하지는 않는다.[^index]

이번에는 왼쪽 위 3×3 영역만 선택한다. 이렇게 이미지에서 골라낸 작은 영역을 <strong>패치(patch)</strong>라고 한다.

```python
# :3은 0:3의 생략형이다. 시작 0은 포함하고 끝 3은 제외한다.
patch = x[0, 0, :3, :3]

print(patch)
# tensor([[1., 2., 0.],
#         [0., 3., 1.],
#         [2., 1., 0.]])

print(tuple(patch.shape))  # (3, 3)
```

첫 번째 `:3`은 행 인덱스 `0, 1, 2`, 두 번째 `:3`은 열 인덱스 `0, 1, 2`를 선택한다. 정수로 선택한 이미지·채널 축은 빠지고, 슬라이스로 선택한 행·열 축은 남는다.

패치는 앞으로 계산에 사용할 입력이다. 입력 일부를 읽었을 뿐이므로 아직 합성곱 출력이 만들어지지 않았다. 원본에서 해당 영역이 삭제되는 것도 아니다.

## 3. 패치와 커널로 출력 한 칸을 계산한다

<strong>가중치</strong>는 입력값에 곱하는 숫자다. 그 가중치를 공간 모양에 맞춰 배열한 것을 <strong>커널(kernel)</strong>이라고 부른다. 다음 커널은 왼쪽 열의 값을 더하고 오른쪽 열의 값을 빼도록 직접 정했다. 가운데 열은 0을 곱하므로 합계에 기여하지 않는다.

```python
K = torch.tensor([
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.],
])

products = patch * K  # 같은 위치의 숫자끼리 각각 곱한다.
print(products)
# tensor([[ 1., 0.,  0.],
#         [ 0., 0., -1.],
#         [ 2., 0.,  0.]])
```

`*`는 여기서 <strong>원소별 곱셈</strong>이다. `patch[0, 0]`과 `K[0, 0]`, `patch[0, 1]`과 `K[0, 1]`처럼 같은 위치의 값끼리 계산한다. 곱셈을 아홉 번 했으므로 결과도 3×3으로 남는다. 행렬곱을 하거나, 곱한 결과를 자동으로 더하는 연산은 아니다.

`products`는 여러 곱셈 결과를 담았다는 뜻으로 붙인 변수 이름이다. 출력 문구에 `elementwise products`라고 적혀 있다면 같은 위치의 원소끼리 곱한 결과들이라는 뜻이다.

<figure class="conv-practice-figure">
<img src="/cnn-convolution-practice/patch-products-ko.svg" alt="첫 입력 패치와 3×3 커널을 같은 위치끼리 곱하면 행별 합계가 1, -1, 2가 되고 전체 합계는 2가 된다." loading="lazy">
<figcaption>원소별 곱셈은 3×3 결과를 만들고, 그 아홉 값을 모두 더하면 출력 한 칸의 값이 된다.</figcaption>
</figure>

### `sum(dim=1)`은 어느 숫자들을 더하는가

`products.shape`는 `(3, 3)`이다. 이 텐서의 0번 축은 행, 1번 축은 열이다. `sum(dim=1)`은 각 행에서 열 방향의 값들을 더한다.[^sum]

| 행 | 해당 행의 곱셈 결과 | 열 축의 값들을 더한 결과 |
| --- | --- | ---: |
| 0 | `[1, 0, 0]` | 1 |
| 1 | `[0, 0, -1]` | -1 |
| 2 | `[2, 0, 0]` | 2 |

행마다 합계 하나가 남으므로 결과의 shape는 `(3,)`이다. `dim=1`의 `1`은 두 번째 축의 번호이며, 인덱스 1에 있는 숫자만 더하라는 뜻이 아니다.

```python
row_sums = products.sum(dim=1)  # 행마다 열의 값 세 개를 더한다.
print(row_sums)                # tensor([ 1., -1.,  2.])
print(row_sums.tolist())       # [1.0, -1.0, 2.0]

first_output = products.sum()  # 축을 지정하지 않아 모든 원소를 더한다.
print(first_output)            # tensor(2.)
print(tuple(first_output.shape))  # ()
print(first_output.item())     # 2.0
```

`tolist()`는 텐서 값을 일반 Python 리스트로 바꾼다. 합산을 추가로 수행하지 않는다. `item()`은 원소가 정확히 하나인 텐서에서 일반 Python 숫자를 꺼낸다. 이렇게 꺼낸 숫자는 자동 미분 연결을 유지하지 않으므로 출력이나 기록에 사용한다.[^values]

shape `()`는 숫자 하나를 담은 0차원 텐서다. 원소가 없는 텐서라는 뜻이 아니다. `sum(dim=0)`을 사용했다면 같은 열의 여러 행을 더했을 것이고, `sum()`은 행·열 구분 없이 모든 값을 더한다.

이번 출력값은 `1 + (-1) + 2 = 2`다. 행별 합계를 먼저 적은 것은 계산을 확인하기 위해서다. 실제 합성곱이 반드시 `row_sums`라는 중간 변수를 만들어야 하는 것은 아니다.

### 같은 계산을 `nn.Conv2d`로 실행한다

<strong>층(layer)</strong>은 신경망에서 한 단계의 계산을 맡는 구성 요소다. `nn.Conv2d`는 합성곱 층을 만드는 클래스이고, `nn.Conv2d(...)`를 호출하면 가중치와 계산 설정을 가진 객체가 생긴다. 여기서 객체는 실제로 사용할 층 하나라고 보면 된다.

출력 채널 하나를 만드는 전체 가중치 묶음을 이 글에서는 <strong>필터(filter)</strong>라고 부른다. 지금은 입력 채널도 출력 채널도 하나이므로 필터 하나에 3×3 커널 하나가 들어간다. 자료에 따라 커널과 필터를 같은 뜻으로 쓰기도 하므로, 실제 가중치 shape와 연결 구조를 함께 확인하는 편이 정확하다.

```python
conv = nn.Conv2d(
    in_channels=1,   # 입력 이미지마다 채널이 한 개다.
    out_channels=1,  # 필터 한 개로 출력 채널 한 개를 만든다.
    kernel_size=3,   # 공간 커널의 높이와 너비를 3×3으로 정한다.
    bias=False,      # 곱셈 결과를 합한 뒤 더하는 편향은 사용하지 않는다.
)

print(tuple(conv.weight.shape))  # (1, 1, 3, 3)

# 생성된 가중치에 사람이 정한 K의 값을 복사한다.
with torch.no_grad():
    conv.weight.copy_(K.reshape(1, 1, 3, 3))

out = conv(x)  # 입력 x와 현재 가중치로 실제 합성곱을 실행한다.
print(out[0, 0].tolist())
# [[2.0, 2.0, -3.0], [0.0, -2.0, 0.0]]

print(tuple(out.shape))  # (1, 1, 2, 3)
```

처음 생성한 층에는 무작위 초기 가중치가 들어 있다. `copy_`는 기존 가중치 저장 공간에 `K`의 숫자를 복사한다. 끝의 `_`는 PyTorch에서 보통 기존 텐서를 직접 수정하는 연산에 붙는다. `K`와 `conv.weight`가 같은 객체가 되는 것은 아니다.[^copy]

PyTorch는 나중에 변화율을 계산할 수 있도록 연산 관계를 기록하는 <strong>자동 미분</strong> 기능을 제공한다. 여기의 `with torch.no_grad():`는 들여쓴 복사 작업을 그 기록에 넣지 않도록 한다. 가중치를 직접 지정할 때 쓰는 방식이며, 가중치를 영구적으로 동결하는 명령은 아니다. 자동 미분의 목적과 출력에 기록이 남는 이유는 뒤에서 숫자로 확인한다.[^nograd]

지정하지 않은 설정은 `stride=1`, `padding=0`, `dilation=1`, `groups=1`이다. 커널이 이웃한 입력 칸을 읽고, 시작 위치를 한 칸씩 옮기며, 입력 바깥에 칸을 추가하지 않는다. 각 필터는 모든 입력 채널을 사용한다. 다음 절부터 이동 간격과 가장자리 처리를 바꿔 볼 것이다.

| 표현 | 가리키는 대상 |
| --- | --- |
| `conv` | 가중치와 계산 설정을 가진 층 객체 |
| `conv.weight` | 층 안에 들어 있는 가중치 텐서 |
| `conv(x)` | 현재 입력과 가중치로 계산해 반환한 출력 텐서 |
| `out[0, 0]` | 첫 이미지의 첫 출력 채널 전체, shape `(2, 3)` |
| `out[0, 0, 0, 0]` | 첫 출력 채널의 첫 행·첫 열에 있는 숫자 하나 |

수동으로 구한 첫 출력값과 실제 층의 결과를 비교해 보자. `assert`는 조건이 참이면 진행하고 거짓이면 오류를 발생시킨다.

```python
assert first_output.item() == out[0, 0, 0, 0].item()  # 두 값 모두 2.0이다.
assert torch.equal(
    out[0, 0],
    torch.tensor([[2., 2., -3.], [0., -2., 0.]]),
)
```

가중치 아홉 개를 서로 다른 여섯 위치에서 반복 사용해서 출력 여섯 개를 만들었다. 출력 숫자 개수와 가중치 개수는 다르다. 여러 위치에서 계산해 얻은 이 출력 배열을 <strong>특징 맵(feature map)</strong>이라고도 한다.

`products`를 `conv`에 전달한 것은 아니다. `conv(x)`는 자기 가중치로 계산하고, `patch * K`와 `assert`는 그 계산을 사람이 따라가며 확인하는 코드다. `print`와 `assert`를 지워도 합성곱 자체는 실행된다.

PyTorch의 `Conv2d`는 여기서처럼 커널을 뒤집지 않고 계산한다. 수학적으로는 교차상관이라고 부르는 방식이지만, 딥러닝에서는 보통 합성곱이라는 이름을 사용한다.[^conv]

### 채널이 여러 개일 때의 가중치

입력의 shape와 가중치의 shape는 앞의 두 축부터 의미가 다르다. `groups=1`에서 입력은 `(이미지 수, 입력 채널 수, 높이, 너비)`, 가중치는 `(출력 채널 수, 입력 채널 수, 커널 높이, 커널 너비)`다.

RGB 입력에 출력 필터를 두 개 두면 가중치 shape는 `(2, 3, 3, 3)`이다. 필터 하나에 R·G·B용 커널이 하나씩 들어 있다. 같은 위치의 세 채널을 각각 계산한 후 더해서 출력값 하나를 만든다. 한 출력값에 가중치 `3×3×3=27개`를 사용하고, 두 필터의 전체 가중치는 `2×27=54개`다. 편향을 사용하면 필터마다 하나씩 추가한다.

따라서 `conv.weight[0, 0]`에서 첫 `0`은 이미지 번호가 아니라 출력 필터 번호다. 두 번째 `0`은 그 필터에서 사용할 입력 채널 번호다. 출력 채널은 필터의 계산 결과를 구분하며, 반드시 RGB 색상 성분을 뜻하지는 않는다.

## 4. ReLU를 적용하면 합성곱 출력이 어떻게 달라지는가

<strong>ReLU</strong>는 입력받은 숫자가 음수이면 0으로 만들고, 0이나 양수이면 그대로 두는 함수다. 합성곱 다음에 사용할 수 있는 별도의 계산이다.[^relu]

```python
relu_out = F.relu(out)  # 각 원소에 독립적으로 적용한다.
print(relu_out[0, 0].tolist())
# [[2.0, 2.0, 0.0], [0.0, 0.0, 0.0]]

assert relu_out.shape == out.shape  # 숫자 개수와 축별 크기는 같다.
```

여러 숫자를 합치는 계산이 아니므로 shape가 유지된다. 이 함수 자체에는 학습할 가중치가 없다.

ReLU를 쓰는 이유를 설명할 때 <strong>비선형성</strong>이라는 말이 나온다. 여기서는 값을 더하거나 배율을 적용하는 순서를 바꾸어도 결과가 그대로 유지되는 선형 계산의 성질이 성립하지 않는다는 뜻이다. 작은 숫자로 확인할 수 있다. `ReLU(1) + ReLU(-1)`은 `1 + 0 = 1`이다. 입력을 먼저 더한 `ReLU(1 + (-1))`은 `ReLU(0) = 0`이다. 두 결과가 다르다.

이처럼 입력의 부호에 따라 계산 규칙이 달라지는 단계를 넣으면, 여러 층의 계산을 단순한 선형 계산 하나로만 표현하는 제약에서 벗어날 수 있다. 다만 음수가 언제나 쓸모없는 정보라는 뜻은 아니다. 뒤의 라플라시안 시각화에서는 음수와 양수가 각각 의미를 가지므로 ReLU를 적용하지 않는다.

## 5. 출력 크기는 커널을 놓을 수 있는 시작 위치의 개수다

<strong>Stride(스트라이드)</strong>는 커널의 시작 위치 사이의 간격이다. `stride=2`이면 시작 위치를 `0, 2, 4, …`처럼 고른다. <strong>Padding(패딩)</strong>은 입력 가장자리에 계산용 값을 추가하는 처리다. 여기서는 양쪽에 0을 같은 개수만큼 추가한다.

<strong>Dilation(딜레이션)</strong>은 커널의 이웃한 가중치가 읽는 입력 위치 사이의 간격이다. 이 글에서는 계속 `dilation=1`을 사용하므로 입력의 이웃한 칸들을 읽는다. 높이와 너비의 출력 길이는 각각 같은 방법으로 구할 수 있다.

### 먼저 마지막으로 허용되는 시작 좌표를 구한다

길이가 6인 입력의 인덱스는 `0, 1, 2, 3, 4, 5`다. 여기에 길이 3인 커널을 놓는다. 시작 좌표 3에서는 `3, 4, 5`를 읽으므로 가능하다. 시작 좌표 4에서는 `4, 5, 6`을 읽으려 하지만 인덱스 6이 없으므로 불가능하다.

이 관계를 기호로 적어 보자. `L`은 원래 입력 길이, `p`는 한쪽에 추가하는 패딩 길이, `k`는 커널 길이다. 양쪽에 패딩을 두므로 전체 길이 `T`는 `L+2p`다. `j`는 패딩된 입력에서 커널이 시작하는 좌표다.

커널은 `j`부터 `j+k-1`까지 읽는다. 마지막으로 읽는 칸이 입력의 마지막 인덱스 `T-1`을 넘으면 안 된다.

<div class="conv-practice-math">
$$
\begin{aligned}
j+k-1 &\le T-1 \\
j &\le T-k &&\text{양쪽에서 }k-1\text{을 뺀다.}
\end{aligned}
$$
</div>

마지막 허용 시작 좌표를 `M`이라고 하면 다음과 같다.

<div class="conv-practice-math">
$$M=T-k=L+2p-k$$
</div>

길이 6, 커널 길이 3에서는 `M=6-3=3`이다. 아직 stride는 사용하지 않았다. 우선 커널 전체가 들어갈 수 있는 범위를 구한 것이다.

### `range(0, M + 1, s)`의 `+1`

이제 이동 간격을 `s`라고 하자. Python의 `range(start, stop, step)`는 시작값부터 간격만큼 증가하는 정수들을 표현한다. 종료 경계인 `stop`은 포함하지 않는다. `list(...)`로 감싸면 해당 정수들을 리스트로 모아 볼 수 있다.[^range]

허용 조건은 `j ≤ M`이다. 정수 좌표에서는 이를 `j < M+1`로 쓸 수 있다. 그래서 종료 경계를 `M+1`로 정한다.

```python
L, p, k, s = 4, 1, 3, 2  # 입력 4칸, 양쪽 패딩 1칸, 커널 3칸, 간격 2칸.
T = L + 2 * p             # 패딩을 포함하면 6칸이다.
M = T - k                 # 마지막 허용 시작 좌표는 3이다.
starts = list(range(0, M + 1, s))

print(starts)  # [0, 2]
```

`range(0, 4, 2)`이므로 0과 2를 선택하고 4는 제외한다. 범위상 허용되는 마지막 좌표는 3이지만, 실제로 방문한 마지막 좌표는 2다. stride가 정한 간격에 3이 들어 있지 않기 때문이다. `+1`이 마지막 좌표 3을 강제로 추가하는 것은 아니다.

<figure class="conv-practice-figure">
<img src="/cnn-convolution-practice/stride-positions-ko.svg" alt="길이 6인 입력에서 길이 3인 커널은 시작 좌표 3까지 들어갈 수 있다. stride 2이면 실제 시작은 0과 2이고, 다음 후보 4는 범위를 벗어난다." loading="lazy">
<figcaption>마지막 허용 시작 좌표 3, 실제 마지막 시작 좌표 2, 출력 개수 2는 서로 다른 정보다.</figcaption>
</figure>

### 출력 길이 공식에서 stride로 나누고 다시 1을 더하는 이유

출력의 인덱스를 `q`라고 하자. 첫 출력은 입력 시작 좌표 0에서, 두 번째 출력은 `s`에서, 세 번째 출력은 `2s`에서 계산한다.

| 출력 인덱스 `q` | 입력에서의 시작 좌표 `j` |
| ---: | ---: |
| 0 | 0 |
| 1 | `s` |
| 2 | `2s` |
| 3 | `3s` |

따라서 `j=q×s`다. 시작 좌표가 `M` 이하여야 하므로 `q×s ≤ M`, 양쪽을 양수 `s`로 나누면 `q ≤ M/s`다.

출력 인덱스는 정수여야 한다. 마지막 출력 인덱스는 `M/s`를 넘지 않는 가장 큰 정수다. 이렇게 아래쪽 정수로 내리는 연산을 <strong>floor(내림)</strong>라고 한다. `M=3`, `s=2`이면 `M/s=1.5`이므로 마지막 출력 인덱스는 1이다. 인덱스 0과 1이 있으므로 출력 개수는 2개다.

<div class="conv-practice-math">
$$
L_{\mathrm{out}}
=\left\lfloor\frac{M}{s}\right\rfloor+1
=\left\lfloor\frac{L+2p-k}{s}\right\rfloor+1
$$
</div>

이 식은 `dilation=1`, 양쪽 패딩이 같은 경우에 사용한다. `L`, `k`, `s`는 양의 정수이고 `p`는 0 이상이며, 커널이 패딩된 입력에 들어갈 수 있어야 한다. `M`이 0 이상인 이 조건에서 Python의 정수 나눗셈 `//`로 계산하면 다음과 같다.[^conv]

```python
output_length = M // s + 1  # 3 // 2 + 1 = 1 + 1 = 2.
assert output_length == len(starts) == 2
```

두 식의 `+1`은 역할이 다르다. `range`의 `M+1`은 종료값을 제외하는 규칙에 맞춰 경계를 정한다. 출력 길이의 `floor(M/s)+1`은 0부터 시작하는 마지막 인덱스를 원소 개수로 바꾼다.

`range`의 종료 경계를 stride로 나눈 다음 `step`에도 stride를 넣으면 서로 다른 두 계산을 섞게 된다. 종료 경계는 입력 좌표 기준이고, 출력 길이는 그 좌표 중 선택한 시작 위치의 개수다.

또한 stride가 2여도 출력 길이가 입력 길이의 정확히 절반이 되는 것은 아니다. 시작 위치 0에서도 출력 하나를 만들고, 커널 전체가 들어가는 위치만 포함하기 때문이다. 위 공식의 내림 규칙은 그대로 적용된다.

### 높이와 너비는 따로 세고, 모든 조합을 계산한다

원래의 4×5 입력에 3×3 커널을 `stride=2`, `padding=0`으로 적용해 보자.

행은 0에서 시작하면 `0, 1, 2`를 읽을 수 있다. 다음 후보인 2에서는 `2, 3, 4`가 필요한데 행 인덱스 4는 없다. 따라서 시작 행은 `[0]`이다. 열은 0과 2에서 모두 커널을 놓을 수 있으므로 시작 열은 `[0, 2]`다.

실제 계산 위치는 두 목록의 모든 조합인 `(0, 0)`과 `(0, 2)`다. 각각 출력의 `(0, 0)`과 `(0, 1)`을 만든다. 시작 행 한 개와 시작 열 두 개를 곱하면 출력은 1×2가 된다. 행 목록과 열 목록을 같은 순번끼리만 짝짓는 계산이 아니다.

## 6. 같은 입력과 가중치로 stride·padding을 비교한다

계산 설정의 효과를 보려면 입력과 가중치를 같게 두어야 한다. `nn.Conv2d(...)`로 새 층을 만들면 새 초기 가중치가 생긴다. 기존 층에 넣어 둔 `K`가 자동으로 복사되는 것은 아니다. 반면 이미 만든 `conv`를 다시 호출하는 것은 재초기화가 아니며, 그 층이 가진 가중치를 계속 사용한다.

`F.conv2d`는 입력과 가중치를 직접 전달받아 합성곱을 계산하는 함수다. 다음 코드에서는 같은 `x`와 `conv.weight`를 전달하고 `stride`, `padding`만 바꾼다.[^functional]

```python
# shape 정보의 마지막 두 항목을 꺼낸다. 입력 이미지의 두 행을 고르는 것이 아니다.
H, W = x.shape[-2:]
settings = [(1, 0), (2, 0), (1, 1), (2, 1)]  # (stride, padding) 쌍이다.
shape_results = {}  # 설정마다 출력 shape를 저장할 딕셔너리다.

for stride, padding in settings:
    actual = F.conv2d(
        x,                   # 네 번 모두 같은 입력이다.
        conv.weight,         # 네 번 모두 같은 가중치다.
        stride=stride,
        padding=padding,
    )

    starts_h = list(range(0, H + 2 * padding - 3 + 1, stride))
    starts_w = list(range(0, W + 2 * padding - 3 + 1, stride))
    shape_results[(stride, padding)] = tuple(actual.shape)

    print("설정:", (stride, padding))
    print("시작 행:", starts_h, "시작 열:", starts_w)
    print("출력:", actual[0, 0].tolist())
    assert tuple(actual.shape[-2:]) == (len(starts_h), len(starts_w))
```

`settings`의 항목 하나는 숫자 두 개를 묶은 튜플이다. `for stride, padding in settings`는 그 두 숫자를 각각의 변수로 나누어 받는다. 딕셔너리는 키와 값을 연결하는 자료형이며, `shape_results`에는 `(stride, padding)`을 키로 사용해 출력 shape를 저장했다.

패딩이 있는 경우 시작 좌표는 <strong>패딩된 입력</strong>을 기준으로 한다.

| stride | padding | 시작 행 | 시작 열 | 출력의 높이 × 너비 |
| ---: | ---: | --- | --- | --- |
| 1 | 0 | `[0, 1]` | `[0, 1, 2]` | 2×3 |
| 2 | 0 | `[0]` | `[0, 2]` | 1×2 |
| 1 | 1 | `[0, 1, 2, 3]` | `[0, 1, 2, 3, 4]` | 4×5 |
| 2 | 1 | `[0, 2]` | `[0, 2, 4]` | 2×3 |

실제 출력 숫자도 함께 비교해 보자.

```text
stride=1, padding=0
[[ 2,  2, -3],
 [ 0, -2,  0]]

stride=2, padding=0
[[ 2, -3]]

stride=1, padding=1
[[-5,  0,  2, -3,  3],
 [-6,  2,  2, -3,  4],
 [-4,  0, -2,  0,  6],
 [-1,  1, -3,  0,  4]]

stride=2, padding=1
[[-5,  2,  3],
 [-4, -2,  6]]
```

첫 설정과 마지막 설정은 출력 크기가 모두 2×3인데 값은 다르다. 읽은 입력 영역과 이동 간격이 다르기 때문이다. shape가 같다는 사실만으로 같은 계산이라고 판단할 수 없다.

`padding=1`에서 입력과 같은 4×5 크기가 나온 경우는 커널이 3×3이고 `stride=1`, `dilation=1`인 설정이다. 패딩을 1로 지정하면 언제나 입력 크기가 유지되는 것은 아니다.

또한 `F.conv2d`는 `conv` 객체의 설정을 자동으로 가져오지 않는다. 함수에 전달한 인자와 그 함수의 기본값을 사용한다. 현재 `conv`는 편향이 없고 기본 stride·padding을 사용하므로 `conv(x)`와 `F.conv2d(x, conv.weight)`가 같은 계산이다. 다른 설정을 쓰는 층이라면 해당 값도 맞춰 전달해야 한다.

### 패딩 후 첫 출력값은 왜 2에서 -5가 되는가

`F.pad`로 0 패딩을 직접 만들어 보자. 두 번째 인자 `(1, 1, 1, 1)`은 왼쪽, 오른쪽, 위쪽, 아래쪽에 각각 한 칸을 추가한다는 뜻이다. 마지막 축인 너비의 양쪽부터 지정하고, 그다음 높이의 양쪽을 지정한다.[^pad]

```python
padded_x = F.pad(x, (1, 1, 1, 1))  # 기본 상수 패딩 값은 0이다.
print(tuple(padded_x.shape))       # (1, 1, 6, 7)
print(padded_x[0, 0].tolist())
# [[0., 0., 0., 0., 0., 0., 0.],
#  [0., 1., 2., 0., 1., 3., 0.],
#  [0., 0., 3., 1., 2., 1., 0.],
#  [0., 2., 1., 0., 1., 0., 0.],
#  [0., 1., 0., 2., 3., 2., 0.],
#  [0., 0., 0., 0., 0., 0., 0.]]

padded_patch = padded_x[0, 0, :3, :3]  # 패딩된 배열의 왼쪽 위 3×3이다.
padded_products = padded_patch * K

print(padded_patch.tolist())
# [[0., 0., 0.], [0., 1., 2.], [0., 0., 3.]]
print(padded_products.tolist())
# [[0., 0., 0.], [0., 0., -2.], [0., 0., -3.]]
print(padded_products.sum(dim=1).tolist())  # [0.0, -2.0, -3.0]
print(padded_products.sum().item())        # -5.0
```

커널은 그대로다. 새 첫 패치에는 왼쪽과 위쪽에 추가한 0이 들어 있고, 오른쪽 열에 원본의 2와 3이 들어 있다. 그래서 `0-2-3=-5`다. `F.pad`는 패딩 결과 텐서를 반환하며, 원래 `x`의 shape나 숫자를 덮어쓰지 않는다.

원래 첫 패치는 패딩된 배열의 `(1, 1)`에서 시작한다. 그 위치에서는 이전처럼 결과가 2다.

```python
pad_out = F.conv2d(x, conv.weight, padding=1)
old_patch = padded_x[0, 0, 1:4, 1:4]  # 인덱스 1, 2, 3을 선택한다.

assert torch.equal(old_patch, patch)
assert pad_out[0, 0, 0, 0].item() == -5.0
assert pad_out[0, 0, 1, 1].item() == 2.0

# 직접 패딩한 입력을 쓸 때는 합성곱에서 추가 패딩을 주지 않는다.
manual_pad_out = F.conv2d(padded_x, conv.weight, padding=0)
assert torch.equal(manual_pad_out, pad_out)
```

이미 패딩한 `padded_x`에 다시 `padding=1`을 지정하면 패딩이 중복된다. 추가한 0은 이미지 바깥의 실제 픽셀을 복원한 값도 아니다. 바깥값을 0으로 두고 계산하기로 한 경계 처리다.

## 7. 라플라시안 필터의 `-4`는 어디에서 나오는가

이번에는 중심 밝기와 주변 밝기를 비교하는 커널을 사용한다. 다음 <strong>라플라시안(Laplacian) 필터</strong>는 상하좌우 이웃과 중심의 차이를 더하도록 사람이 정한 가중치다.[^laplacian]

```python
laplace = torch.tensor([
    [0.,  1., 0.],
    [1., -4., 1.],
    [0.,  1., 0.],
])
```

상하좌우 값을 각각 `위`, `아래`, `왼쪽`, `오른쪽`이라고 부르면 계산은 다음과 같다.

<div class="conv-practice-math">
$$
\begin{aligned}
&(\text{위}-\text{중심})+(\text{아래}-\text{중심})\\
&\quad +(\text{왼쪽}-\text{중심})+(\text{오른쪽}-\text{중심})\\
&=\text{위}+\text{아래}+\text{왼쪽}+\text{오른쪽}-4\times\text{중심}
\end{aligned}
$$
</div>

이웃 네 값에서 중심을 한 번씩 빼기 때문에 중심의 가중치가 `-4`다. 대각선은 이번 계산에 사용하지 않으므로 0이다. 같은 식을 `4×(상하좌우 평균-중심)`으로도 쓸 수 있다. 따라서 이 커널에서는 중심이 주변 평균보다 밝으면 음수, 어두우면 양수, 같으면 0이 나온다.

출력이 0이라고 주변 값이 모두 같다는 뜻은 아니다. 중심이 1이고 상하좌우가 `2, 0, 0, 2`이면 차이는 `1, -1, -1, 1`이다. 서로 상쇄되어 합계가 0이 된다.

이 라플라시안은 수학적으로 공간에 따른 2차 미분을 근사하는 계산이다. 이 글에서는 그 용어를 이용해 추가 공식을 전개하지 않고, 방금 적은 네 방향의 차이 합으로 값을 해석한다. 이미지의 어느 위치에서 밝기가 변하는지 계산하는 작업이며, 학습 중 손실이 가중치에 따라 얼마나 변하는지 구하는 자동 미분과는 대상이 다르다.

### 새 5×5 입력에서 직접 확인한다

앞의 4×5 입력과 구분하려고 새 변수 `laplace_x`를 사용한다. `laplace_conv`에는 방금 정한 커널을 복사한다.

```python
laplace_x = torch.tensor([
    [1., 2., 3., 0., 1.],
    [0., 1., 2., 3., 1.],
    [3., 0., 1., 2., 0.],
    [2., 3., 0., 1., 2.],
    [1., 2., 3., 0., 1.],
]).reshape(1, 1, 5, 5)

laplace_conv = nn.Conv2d(1, 1, 3, bias=False)
with torch.no_grad():
    laplace_conv.weight.copy_(laplace.reshape(1, 1, 3, 3))

laplace_out = laplace_conv(laplace_x)
first_laplace_patch = laplace_x[0, 0, :3, :3]

print((first_laplace_patch * laplace).tolist())
# [[0.0, 2.0, 0.0], [0.0, -4.0, 2.0], [0.0, 0.0, 0.0]]
print((first_laplace_patch * laplace).sum().item())  # 0.0
print(laplace_out[0, 0].tolist())
# [[0.0, 0.0, -7.0], [8.0, 0.0, -3.0], [-8.0, 8.0, 0.0]]
print(tuple(laplace_out.shape))  # (1, 1, 3, 3)
```

첫 패치의 중심은 1이다. 위 2, 아래 0, 왼쪽 0, 오른쪽 2이므로 `2+0+0+2-4×1=0`이다. 전체 출력 크기는 높이와 너비 각각 `5-3+1=3`이다.

가중치를 미리 넣었다는 이유만으로 사전학습된 필터라고 부르지는 않는다. <strong>무작위 초기화</strong>는 시작할 값을 정하는 것이고, <strong>직접 설계</strong>는 원하는 계산을 하도록 사람이 값을 정하는 것이다. <strong>사전학습 가중치</strong>는 이전 학습 과정에서 조정된 값을 가져온 것이다. 이 라플라시안은 직접 설계한 경우다.

`nn.Conv2d(1, 1, 3, padding=1)`이나 `nn.Conv2d(1, 1, 3, stride=2)`를 새로 만들면 서로 다른 초기 가중치가 들어간다. 그 상태에서는 5×5 또는 2×2라는 출력 크기를 비교할 수 있지만, 같은 라플라시안의 숫자 비교는 아니다. 동일한 필터로 비교하려면 `F.conv2d(laplace_x, laplace_conv.weight, padding=1)`처럼 기존 가중치를 전달한다.

## 8. 출력은 학습 파라미터가 아닌데 왜 자동 미분에 연결되는가

지금까지는 입력과 가중치로 출력 숫자를 계산했다. 일반적인 학습에서는 그 출력으로 예측을 만들고, 예측과 목표가 얼마나 다른지 <strong>손실</strong>이라는 숫자로 계산한다. 그다음 가중치를 바꾸어 손실을 줄이려고 한다.

이때 필요한 것이 <strong>기울기</strong>다. 다른 조건을 고정한 현재 상태에서, 특정 값을 아주 조금 바꿀 때 손실이 얼마나 변하는지 나타내는 변화율이다. 가중치를 직접 학습하려면 가중치에 대한 손실의 기울기를 구해야 한다.

가중치를 바꾸면 합성곱 출력이 바뀌고, 그 출력이 바뀌면 뒤에서 계산한 손실도 바뀔 수 있다. 따라서 최종적으로 가중치의 기울기를 구하려 해도, 중간 출력과 연결된 계산 관계가 필요하다.

### 한 경로의 변화를 숫자로 연결해 본다

복잡한 합성곱에 앞서 출력이 하나인 아주 작은 예를 보자. 입력을 3으로 고정하고, 가중치를 `w`, 출력을 `z=3w`, 목표를 0으로 두자. 손실은 목표와의 차이를 제곱한 `L=z²`다. `w=2`이면 `z=6`, `L=36`이다.

가중치를 `h`만큼 바꾸면 출력은 `3h`만큼 바뀐다. 따라서 출력의 가중치에 대한 변화율은 3이다. 이번에는 출력 6을 `d`만큼 바꾸었을 때 손실 차이를 계산해 보자.

<div class="conv-practice-math">
$$
\begin{aligned}
(6+d)^2-6^2 &= 12d+d^2 \\
\frac{12d+d^2}{d} &= 12+d \qquad (d\ne 0)
\end{aligned}
$$
</div>

출력 변화 `d`를 아주 작게 만들면 손실의 변화율은 12에 가까워진다. 가중치 변화가 출력에서 3배가 되고, 그 출력 변화가 손실에서 현재 약 12배로 이어지므로 가중치에 대한 손실 변화율은 `12×3=36`이다.

직접 확인하면 `w`를 2에서 2.001로 바꿀 때 출력은 6.003, 손실은 36.036009다. 손실 변화 0.036009를 가중치 변화 0.001로 나누면 36.009로, 방금 구한 36에 가깝다. 기울기 36은 최적 가중치까지 이동할 거리라는 뜻이 아니라, 현재 위치에서의 변화율이다.

이처럼 중간 변화율을 곱해 전체 변화율을 구하는 규칙을 <strong>연쇄법칙</strong>이라고 한다. `∂L/∂w`는 다른 조건을 고정했을 때 가중치 `w`에 대한 손실 `L`의 변화율을 뜻한다. 같은 방식으로 `∂L/∂z`는 출력에 대한 손실 변화율, `∂z/∂w`는 가중치에 대한 출력 변화율이다.

<div class="conv-practice-math">
$$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial z}\times\frac{\partial z}{\partial w}$$
</div>

합성곱에서는 같은 가중치 하나가 여러 출력 위치에서 쓰인다. 출력들을 `z₀, z₁, …`처럼 구분하면, 가중치가 바뀔 때 각 출력을 거쳐 손실로 전달되는 영향을 모두 더해야 한다. `Σ`는 그 여러 항을 더한다는 기호이고, `j`는 출력값을 구분하는 인덱스다.

<div class="conv-practice-math">
$$\frac{\partial L}{\partial w}=\sum_j\left(\frac{\partial L}{\partial z_j}\times\frac{\partial z_j}{\partial w}\right)$$
</div>

출력에 대한 기울기를 중간에서 사용한다고 해서, 출력값을 학습 파라미터로 바꾸는 것은 아니다. 최종적으로 가중치의 기울기를 구하기 위해 계산 경로를 따라가는 것이다.[^autograd]

### 기록, 기울기 계산, 가중치 수정은 각각 언제 일어나는가

PyTorch의 자동 미분과 관련된 속성 세 가지를 구별해 두면 현재 상태를 읽기 쉽다.

| 속성 | 확인하는 정보 |
| --- | --- |
| `requires_grad` | 이 텐서가 자동 미분에 관여하는지 |
| `grad_fn` | 기록된 연산으로 만들어진 텐서의 역전파 연결 정보 |
| `grad` | 역전파로 계산되어 해당 텐서에 저장된 기울기 |

일반적인 기울기 기록 상태에서 `x.requires_grad`는 `False`여도 `conv.weight.requires_grad`가 `True`이면 `conv(x)`의 출력은 자동 미분에 연결된다. 입력 데이터를 바꾸려는 것이 아니라 가중치가 출력에 미치는 영향을 계산해야 하기 때문이다.

계산 결과를 만드는 방향을 <strong>순전파</strong>, 손실에서 앞쪽의 연산을 따라 기울기를 구하는 과정을 <strong>역전파</strong>라고 한다. 순전파에서 계산 관계를 기록했다고 곧바로 기울기가 계산되거나 가중치가 수정되는 것은 아니다. `backward()`가 기울기를 계산하고, 학습에서 사용하는 <strong>옵티마이저</strong>가 `step()`으로 등록된 파라미터를 수정한다.[^autograd]

### 출력 여섯 개와 가중치 아홉 개의 기울기를 확인한다

앞의 4×5 입력과 커널을 사용하되, 실험용 층 `grad_conv`를 따로 만든다. 확인용 손실은 출력 여섯 개를 각각 제곱한 뒤 평균낸 값이다. 출력의 목표를 모두 0으로 둔 평균제곱오차이며, 손글씨 숫자를 분류하는 학습 목표는 아니다.

현재 출력이 `2, 2, -3, 0, -2, 0`이므로 손실은 다음과 같다.

<div class="conv-practice-math">
$$L=\frac{2^2+2^2+(-3)^2+0^2+(-2)^2+0^2}{6}=\frac{21}{6}=3.5$$
</div>

`square()`는 각 숫자를 제곱하고, `mean()`은 전체 평균을 구한다. `retain_grad()`는 중간 출력의 기울기도 `.grad`에 보관해 달라는 요청이다. 출력값 자체를 학습 파라미터로 등록하는 명령은 아니다.

역전파 전후의 가중치를 비교하려면 계산 전 값을 별도로 보관해야 한다. 아래의 `detach()`는 기존 자동 미분 연결이 없는 텐서를 얻고, 이어지는 `clone()`은 숫자까지 독립적으로 복사한다. 두 메서드의 차이는 다음 절에서 다시 확인한다.

```python
grad_conv = nn.Conv2d(1, 1, 3, bias=False)
with torch.no_grad():
    grad_conv.weight.copy_(conv.weight)  # 앞에서 사용한 커널과 값을 맞춘다.

weight_before = grad_conv.weight.detach().clone()  # 비교할 숫자 복사본이다.
grad_out = grad_conv(x)       # 출력 shape: (1, 1, 2, 3).
grad_out.retain_grad()        # 중간 출력의 기울기를 보관하도록 요청한다.
grad_loss = grad_out.square().mean()
grad_loss.backward()         # 손실에서 가중치까지 기울기를 계산한다.

print(grad_loss.item())                 # 3.5
print(tuple(grad_out.grad.shape))      # (1, 1, 2, 3)
print(tuple(grad_conv.weight.grad.shape))  # (1, 1, 3, 3)
print(grad_out.grad[0, 0])
# tensor([[ 0.6667,  0.6667, -1.0000],
#         [ 0.0000, -0.6667,  0.0000]])

# backward는 기울기를 계산했지만 가중치를 수정하지 않았다.
assert torch.equal(grad_conv.weight.detach(), weight_before)
```

출력의 기울기는 출력 여섯 개에 대응한다. 왜 첫 값이 약 0.6667인지도 확인할 수 있다. 출력 하나 `z`의 손실 항은 `z²/6`이다. 이를 `d`만큼 바꾸면 손실 차이는 `(2zd+d²)/6`이고, 변화량 `d`로 나눈 값은 `(2z+d)/6`이다. `d`가 아주 작아지면 기울기는 `2z/6`에 가까워진다. `z=2`에서는 `2×2/6=2/3`, `z=-3`에서는 `2×(-3)/6=-1`이다.

가중치의 기울기는 아홉 가중치에 대응하며, 각 가중치가 쓰인 모든 출력 위치의 영향을 합한다. 예를 들어 커널의 왼쪽 위 가중치는 여섯 패치의 왼쪽 위 입력값 `1, 2, 0, 0, 3, 1`과 곱해졌다. 따라서 그 가중치의 기울기는 다음과 같다.

<div class="conv-practice-math">
$$\frac23\times1+\frac23\times2+(-1)\times0+0\times0+\left(-\frac23\right)\times3+0\times1=0$$
</div>

이 가중치를 바꾸면 개별 출력은 변하지만, 현재 손실에 대한 각 경로의 영향이 상쇄된 것이다. 가중치가 계산에 사용되지 않았다는 뜻이 아니다.

`grad_conv.weight`처럼 기록된 연산 결과가 아닌 출발점의 텐서를 <strong>leaf tensor</strong>라고 한다. 이런 학습 파라미터에는 기본적으로 역전파 후 `.grad`가 저장된다. 중간 출력 `grad_out`은 연산 결과인 non-leaf 텐서여서, 역전파에 필요한 중간 기울기를 사용하더라도 `.grad`에 기본 보관하지는 않는다. 위에서는 직접 확인하려고 `retain_grad()`를 호출했다.[^autograd]

일반적인 학습 반복은 이전 기울기 초기화 → 순전파 → 손실 계산 → `backward()` → `optimizer.step()` 순서다. `backward()`는 기존 `.grad`에 기울기를 누적하므로 반복 학습에서는 초기화 단계가 필요하다. 이 실험은 새 층에서 한 번만 기울기를 계산했고 가중치 수정은 실행하지 않았다. 가중치를 수정하면 다음 순전파에서 그 가중치로 출력이 새로 계산된다.

## 9. `detach()`, `no_grad()`, 가중치 동결을 구분한다

합성곱 결과를 화면에 표시하려고 할 때 `detach()`를 자주 보게 된다. 이 메서드는 기존 자동 미분 연결을 사용하지 않는 텐서를 반환한다. 숫자와 shape는 유지되며, 반환값도 여전히 PyTorch 텐서다.[^detach]

```python
result = laplace_conv(laplace_x)
detached_result = result.detach()

assert torch.equal(result, detached_result)  # 숫자는 같다.
assert result.shape == detached_result.shape
assert result.requires_grad is True
assert detached_result.requires_grad is False
assert laplace_conv.weight.requires_grad is True
```

`result.detach()`가 원래 `result`의 연결을 지우거나 가중치를 수정하지는 않는다. 반환된 `detached_result`가 기존 연결을 사용하지 않는 것이다. 또한 `detach()`는 숫자 저장 공간을 복사하지 않는다. 두 텐서가 저장 공간을 공유하므로 분리된 텐서의 값을 직접 수정하면 원래 값에도 영향을 줄 수 있다.

숫자까지 별도로 보관하려면 복사본을 만드는 `clone()`을 연결한다.

```python
independent_copy = result.detach().clone()  # 연결에서 분리하고 데이터를 복사한다.
```

단순히 `print(result)`로 값을 출력할 때는 `detach()`가 필수가 아니다. 반면 자동 미분을 추적하는 텐서를 일반적인 NumPy 배열로 변환해 Matplotlib에 전달하려면 다음과 같이 처리한다. <strong>NumPy 배열</strong>은 Python에서 수치 배열을 다루는 또 다른 자료형이다.

```python
array_for_plot = result.detach().cpu().numpy()
# detach(): 기존 자동 미분 연결이 없는 텐서를 얻는다.
# cpu(): GPU 등에 있다면 CPU 쪽으로 옮긴다.
# numpy(): NumPy 배열로 변환한다.
```

기본 `numpy()` 변환은 CPU에 있고 기울기를 추적하지 않는 등 변환 조건을 만족하는 텐서에 사용한다. `detach()` 하나가 CPU 이동과 NumPy 변환까지 수행하는 것은 아니다.[^numpy]

결과를 표시하는 용도로만 계산할 예정이라면 처음부터 역전파용 기록을 만들지 않아도 된다.

```python
with torch.no_grad():
    result_without_graph = laplace_conv(laplace_x)

assert result_without_graph.requires_grad is False
assert laplace_conv.weight.requires_grad is True
```

`detach()`는 이미 계산한 결과에서 연결 없는 텐서를 얻고, `no_grad()`는 해당 블록의 계산을 역전파용으로 기록하지 않는다. 두 방식 모두 가중치 자체의 `requires_grad`를 영구적으로 바꾸지는 않는다.[^nograd]

새 층을 고정 필터로만 사용한다면 `laplace_conv.weight.requires_grad_(False)`로 해당 가중치의 새 기울기 추적을 끌 수 있다. 이미 학습하던 모델이라면 남아 있는 `.grad`와 옵티마이저의 수정 대상도 함께 확인해야 한다. `requires_grad=False`로 바꾸는 것을 이미 저장된 모든 기울기와 업데이트 상태까지 지우는 명령으로 생각하면 안 된다.

이 글에서는 옵티마이저로 가중치를 수정하지 않으므로 `requires_grad=True`여도 필터 값은 그대로다. `True`는 이미 학습된 값이라는 뜻도, 지금 자동으로 값이 수정되고 있다는 뜻도 아니다.

반대로 학습용 손실을 계산할 출력에 무심코 `detach()`를 적용하면, 그 출력 경로를 통한 앞쪽 가중치의 기울기가 끊긴다. 손실을 계산할 값은 연결을 유지하고, 화면에 보여 줄 값을 별도로 분리하는 방식으로 목적을 구분한다.

## 10. MNIST 이미지에 같은 라플라시안을 적용한다

이제 작은 숫자 배열에서 확인한 계산을 실제 손글씨 이미지에 적용한다. <strong>MNIST</strong>는 28×28 흑백 손글씨 숫자 이미지와 숫자 정답으로 이루어진 데이터셋이다. 학습용 60,000장과 테스트용 10,000장이 제공된다. 여기서는 학습용 묶음의 첫 이미지를 골라 필터 반응을 확인한다. 정답은 표시할 때만 사용한다.[^mnist]

`torchvision.datasets`는 이미지 데이터셋을 읽는 기능을, `transforms`는 이미지를 변환하는 기능을 제공한다. `datasets.MNIST(...)`는 데이터를 준비하는 객체를 만들고, 그 객체의 `mnist[0]`에서 첫 이미지와 정답을 꺼낸다.

```python
import time  # 경과 시간을 재는 Python 표준 모듈이다.
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # 그림을 표시하는 기능을 가져온다.

DATA_ROOT = "./data"  # 데이터 파일을 읽거나 저장할 위치다.
started = time.perf_counter()  # 경과 시간의 시작점을 기록한다.

mnist = datasets.MNIST(
    DATA_ROOT,
    train=True,                      # 학습용 데이터 묶음을 선택한다.
    download=True,                   # 필요한 데이터가 없으면 다운로드한다.
    transform=transforms.ToTensor(),  # 이미지를 꺼낼 때 적용할 변환이다.
)

img, label = mnist[0]  # (이미지, 정답) 두 반환값을 각각 받는다.
print(len(mnist))      # 60000
print(label)          # 첫 이미지의 정답: 5
print(tuple(img.shape))  # (1, 28, 28)
```

`train=True`는 모델을 학습하라는 명령이 아니다. 어느 데이터 묶음을 선택할지 정하는 인자다. `download=True`도 이미 사용할 데이터가 있으면 그것을 재사용한다.[^mnist]

`transforms.ToTensor()`는 변환 객체를 만든다. 이 객체를 `transform`에 전달하면 이미지를 꺼낼 때 적용한다. 데이터셋을 만드는 이 줄에서 60,000장 모두를 한꺼번에 변환하는 것은 아니다.

MNIST의 8비트 흑백 이미지에서는 `ToTensor`가 0~255 밝기값을 255로 나누어 0~1 실수값으로 바꾼다. shape는 `(채널, 높이, 너비)`, 즉 `(1, 28, 28)`이다. 이 변환이 모든 종류의 입력에 언제나 255 나누기를 하는 것은 아니며, 지금 사용하는 MNIST 이미지에 해당하는 설명이다.[^totensor]

### 한 줄에 연결된 연산을 순서대로 읽는다

이미지 `img` 앞에 배치 축을 추가하려면 `unsqueeze(0)`을 사용한다. <strong>unsqueeze</strong>는 지정한 위치에 크기 1인 축을 넣는 메서드다. 여기의 `0`은 새 축을 삽입할 위치이며 이미지 인덱스 0을 선택하라는 뜻이 아니다.[^unsqueeze]

```python
batch = img.unsqueeze(0)       # (1, 28, 28) → (1, 1, 28, 28).
result = laplace_conv(batch)   # (1, 1, 28, 28) → (1, 1, 26, 26).
detached_result = result.detach()  # 숫자와 shape는 그대로, 연결에서 분리한다.
edge = detached_result[0, 0]   # 이미지와 출력 채널을 선택해 (26, 26)을 얻는다.

print(tuple(edge.shape))  # (26, 26)
elapsed = time.perf_counter() - started
print("데이터 준비·계산 경과 초:", round(elapsed, 3))
```

같은 코드를 `edge = laplace_conv(img.unsqueeze(0)).detach()[0, 0]`처럼 한 줄로 쓸 수도 있다. 실제로 수행한 일을 순서대로 나누면 다음과 같다.

| 단계 | shape | 해당 단계에서 한 일 |
| --- | --- | --- |
| `img` | `(1, 28, 28)` | 채널 하나의 원본 이미지 |
| `img.unsqueeze(0)` | `(1, 1, 28, 28)` | 맨 앞에 이미지 개수를 나타내는 배치 축 추가 |
| `laplace_conv(batch)` | `(1, 1, 26, 26)` | 현재 라플라시안 가중치로 합성곱 계산 |
| `result.detach()` | `(1, 1, 26, 26)` | 자동 미분 연결에서 분리된 텐서 얻기 |
| `[0, 0]` | `(26, 26)` | 이미지 0번의 출력 채널 0번 선택 |

원본 `img.shape`의 첫 `1`은 채널 수다. `batch.shape`의 첫 `1`은 이미지 수, 두 번째 `1`은 채널 수다. `unsqueeze`가 픽셀값이나 새 이미지를 추가한 것은 아니다.

출력의 높이와 너비가 26인 이유는 각각 `(28-3)//1+1=26`이기 때문이다. `detach()`나 `[0, 0]`이 가장자리 두 칸을 잘라 26을 만든 것이 아니다. 공간 크기는 합성곱 단계에서 이미 26이 되었다.

`edge`는 원본 이미지가 아니라 필터의 계산 결과다. 변수 이름에 `edge`를 썼다고 경계를 추출하는 처리가 추가되는 것은 아니다. 어떤 계산인지는 `laplace_conv`에 들어 있는 가중치가 정한다.

`time.perf_counter()`는 경과 시간을 잴 때 사용하는 시각 값을 반환한다. 두 호출의 차이가 그 사이의 소요 시간이다. 위 코드의 측정 구간에는 데이터 준비, 이미지 선택, 합성곱, 중간 출력 등이 포함되므로 순수 다운로드 시간이나 순수 합성곱 시간으로 해석하지 않는다. `round(elapsed, 3)`은 초 단위 결과를 소수점 아래 셋째 자리까지 반올림한다.

## 11. 흑백 이미지의 출력이 파랑·빨강으로 보이는 이유

원본 이미지에는 0~1의 밝기값이 들어 있다. 라플라시안 출력에는 주변과 중심의 차이를 계산한 음수·0·양수가 들어 있다. 두 배열의 숫자가 뜻하는 것이 다르다.

이 커널에서는 음수일 때 중심이 상하좌우 평균보다 밝고, 양수일 때 중심이 그 평균보다 어둡다. 그 부호를 사람이 구분하기 쉽게 서로 다른 색으로 표시할 수 있다. 색은 합성곱이 만든 데이터가 아니라, 계산된 숫자를 화면에 보여 주는 방식이다.

### 먼저 0을 가운데 놓을 색상 범위를 정한다

절댓값은 숫자에서 부호를 제외한 크기다. `edge.abs()`로 모든 원소의 절댓값을 구하고, 그중 최댓값을 사용하면 음수·양수 양쪽을 같은 크기의 범위로 표시할 수 있다.

```python
limit = max(float(edge.abs().max()), 1e-6)
```

안쪽부터 읽으면 각 연산의 역할이 보인다.

1. `edge.abs()`는 각 원소의 절댓값을 담은 새 텐서를 반환한다. 원래 `edge`를 덮어쓰지 않는다.
2. `.max()`는 그 텐서 전체에서 가장 큰 숫자 하나를 골라 텐서로 반환한다.
3. `float(...)`는 그 값을 일반 Python 실수로 꺼낸다. 이 경우 `.item()`으로 꺼내도 된다.
4. 바깥쪽 `max(..., 1e-6)`는 Python 기본 함수로 두 숫자 중 큰 값을 고른다. `1e-6`은 `0.000001`이다.

예를 들어 출력 일부가 `[-5, -2, 0, 3]`이라면 절댓값은 `[5, 2, 0, 3]`, 최댓값은 5다. 따라서 표시 범위를 `-5`부터 `+5`까지 정한다. 실제 출력에 `+5`가 없어도 0을 가운데 두기 위해 양쪽을 맞춘다.

모든 출력이 0이면 최댓값도 0이다. 이때 `max(0.0, 0.000001)`을 사용하면 표시 범위가 `0~0`으로 겹치는 것을 피할 수 있다. 이 코드는 유한한 값으로 이루어진 이번 출력의 색상 범위를 정하는 코드이며, NaN이나 무한대를 정리하는 일반적인 데이터 정제 과정은 아니다.

화면에 그릴 것은 절댓값 텐서가 아니라 <strong>부호가 그대로 있는 원래 `edge`</strong>다. `abs().max()`는 범위의 크기 하나만 구한다.

### 색을 정하는 것은 `imshow`의 색상표와 범위다

`imshow`는 배열의 숫자를 화면에 표시한다. `cmap`은 어떤 색상표를 쓸지, `vmin`과 `vmax`는 색상표의 양 끝에 어떤 숫자를 대응시킬지 정한다. 여기서는 0을 기준으로 파랑과 빨강 계열이 나뉘는 `coolwarm`을 사용한다.[^imshow]

기본적인 선형 대응에서 숫자 `v`의 색상표 위치는 다음처럼 계산된다. 색상표의 시작은 0, 끝은 1이다.[^colors]

<div class="conv-practice-math">
$$\text{색상표 위치}=\frac{v-v_{\min}}{v_{\max}-v_{\min}}$$
</div>

범위가 `-5~+5`라면 -5는 `(-5+5)/10=0`, 0은 `(0+5)/10=0.5`, +5는 `(5+5)/10=1`이 된다. 0이 색상표의 정확한 가운데에 놓인다.

| 숫자 | `coolwarm`에서 표시할 위치 |
| --- | --- |
| `-limit` | 파란 계열의 끝 |
| 0 | 가운데의 밝은 색 |
| `+limit` | 붉은 계열의 끝 |

`coolwarm`이라는 이름만 지정하면 0이 자동으로 가운데가 되는 것은 아니다. 여기서는 `vmin=-limit`, `vmax=limit`이라는 대칭 범위를 함께 지정했기 때문에 가운데가 된다. 이 표시용 대응은 `edge`의 숫자를 수정하지 않는다.

### 원본과 필터 결과를 나란히 표시한다

`plt.subplots(1, 2)`는 표시 영역을 1행 × 2열로 만든다. 반환되는 `fig`는 전체 그림을 관리하는 Figure 객체이고, `axes`는 각각의 그래프 표시 영역인 Axes 객체들을 담은 배열이다. 이 문맥의 Axes는 텐서의 축 번호를 말하는 것이 아니다.

```python
fig, axes = plt.subplots(
    1, 2,                # 그래프 표시 영역 두 개를 나란히 만든다.
    figsize=(8, 3),      # 전체 그림의 너비 8인치, 높이 3인치다.
    constrained_layout=True,  # 제목과 색상 범례가 겹치지 않도록 배치한다.
)

axes[0].imshow(
    img[0].cpu().numpy(),  # 원본의 채널 0번을 골라 28×28 배열로 전달한다.
    cmap="gray",          # 원래 밝기값은 흑백으로 표시한다.
    vmin=0,
    vmax=1,
    interpolation="nearest",  # 칸 사이에 부드러운 색을 보간하지 않는다.
)
axes[0].set_title(f"MNIST label={label}")  # {label}에 실제 정답이 들어간다.

im = axes[1].imshow(
    edge.cpu().numpy(),   # 이미 detach한 26×26 필터 결과다.
    cmap="coolwarm",
    vmin=-limit,          # 음수와 양수를 같은 크기의 범위로 표시한다.
    vmax=limit,
    interpolation="nearest",
)
axes[1].set_title("Fixed Laplacian response")
fig.colorbar(im, ax=axes[1])  # im의 숫자-색상 대응을 범례로 보여 준다.

for ax in axes:
    ax.axis("off")  # 표시 영역의 눈금과 테두리를 숨긴다.

plt.show()  # 완성한 그림을 표시한다.
```

`im`은 `imshow`가 만든 이미지 표시 객체다. 원본 텐서 `img`나 필터 결과 텐서 `edge`와 다르다. `colorbar`는 이 표시 객체에 설정된 숫자와 색의 관계를 보여 준다. `axis("off")`도 화면의 눈금과 테두리를 숨길 뿐, 텐서의 축이나 데이터를 없애지 않는다.

<figure class="conv-practice-figure conv-practice-photo">
<img src="/cnn-convolution-practice/mnist-laplacian.png" alt="MNIST 첫 학습 이미지인 숫자 5의 흑백 원본과, 같은 이미지에 고정 라플라시안을 적용한 파랑·빨강 응답 및 색상 범례." loading="lazy">
<figcaption>실제 MNIST 첫 학습 이미지로 실행한 결과. 원본은 28×28, 라플라시안 출력은 26×26이다. 음수와 양수를 보존했으며 ReLU는 적용하지 않았다.</figcaption>
</figure>

그림을 만든 실행에서 출력 범위는 약 `-1.7686~2.0588`이었다. 표시 범위는 절댓값 최댓값에 맞춰 약 `-2.0588~2.0588`로 두었다. 원본과 출력은 배열 크기가 다르지만 화면에서는 각각의 표시 영역에 맞춰 보인다. 출력의 `(r, c)`는 원본에서 `(r, c)`에 시작하는 3×3 패치의 계산이며, 그 패치의 중심은 원본의 `(r+1, c+1)`이다.

화면은 컬러여도 `edge`는 여전히 26×26 숫자 배열이다. RGB 세 채널이 새로 생긴 것이 아니다. 색상표를 바꾸거나 흑백으로 그려도 이미 계산한 합성곱 값은 같다.

## 12. 전체 예제를 실행하고 계산 결과를 확인한다

본문 코드와 같은 계산을 [전체 실행 파일](/cnn-convolution-practice/examples-ko.py)로 실행할 수 있다. 파일에는 입력과 커널, 네 가지 stride·padding 결과, 라플라시안, 자동 미분, `detach()` 관련 검산이 포함되어 있다. `torch`, `torchvision`, `matplotlib`을 사용할 수 있는 Python 환경을 준비한다.

```bash
# 작은 배열의 계산과 자동 미분 검산만 실행한다. 데이터 다운로드는 하지 않는다.
python examples-ko.py

# MNIST 데이터가 없으면 다운로드하고, 필터 결과를 화면에 표시한다.
python examples-ko.py --mnist

# 화면을 여는 대신 그림 파일로 저장한다.
python examples-ko.py --mnist --save-figure mnist-laplacian.png
```

숫자가 작은 정수로 이루어진 합성곱 예제는 `torch.equal`로 정확히 일치하는지 확인한다. 일반적인 실수 연산은 계산 순서나 장치에 따라 작은 오차가 생길 수 있으므로, 기울기 같은 실수 결과의 검산에는 `torch.testing.assert_close`처럼 허용 오차를 둔 비교를 사용한다.[^check]

파일의 `assert`는 학습용 검산이다. Python을 최적화 옵션으로 실행하면 생략될 수 있으므로 필수 입력 검증을 대신하는 장치로 사용하지 않는다. `print`는 계산 결과를 사람이 읽도록 출력하고, `imshow`는 이미 계산된 숫자를 그림으로 보여 준다. 실제 합성곱을 실행하는 부분은 `conv(x)`나 `F.conv2d(...)`다.

이 예제의 검산은 Python 3.12, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu 환경에서 수행했다. MNIST 그림도 실제 데이터로 생성했다. 실행 시간은 다운로드 여부와 환경에 따라 달라지므로 고정된 결과값으로 제시하지 않는다.

입력의 한 패치에서 시작한 곱셈과 합산은 출력 한 칸을 만든다. 시작 위치를 바꾸어 같은 가중치를 다시 쓰면 출력 전체가 만들어진다. 자동 미분은 그 출력이 가중치와 어떤 계산으로 연결되었는지 추적하고, 시각화는 출력에 저장된 숫자를 사람이 읽을 수 있는 색으로 표현한다. 코드를 읽을 때 각 줄이 이 중 어느 일을 하는지 확인하면, 숫자의 계산과 데이터 모양, 학습과 화면 표시를 차례로 따라갈 수 있다.

## 참고 문서

아래 링크는 API의 입력 형식과 동작을 확인한 공식 문서다. 본문의 작은 배열 계산과 그림은 이 글의 예제로 직접 구성했다.

[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): 입력·가중치 shape, 출력 크기, 교차상관 계산.
[^index]: Python, [Subscriptions](https://docs.python.org/3/reference/expressions.html#subscriptions): 대괄호 인덱싱과 튜플 전달.
[^sum]: PyTorch, [torch.sum](https://docs.pytorch.org/docs/stable/generated/torch.sum.html): 전체 합계와 지정한 축의 합계.
[^values]: PyTorch, [Tensor.tolist](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html), [Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html).
[^copy]: PyTorch, [Tensor.copy_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.copy_.html).
[^nograd]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html).
[^relu]: PyTorch, [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html).
[^range]: Python, [Ranges](https://docs.python.org/3/library/stdtypes.html#ranges).
[^functional]: PyTorch, [functional.conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html).
[^pad]: PyTorch, [functional.pad](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html).
[^laplacian]: OpenCV, [Image Filtering: Laplacian](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html#laplacian).
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html): 연산 기록, 기울기 추적, leaf·non-leaf 텐서.
[^detach]: PyTorch, [Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html): 자동 미분 연결 분리와 저장 공간 공유.
[^numpy]: PyTorch, [Tensor.numpy](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html).
[^mnist]: Torchvision, [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).
[^totensor]: Torchvision, [ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html).
[^unsqueeze]: PyTorch, [torch.unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html).
[^imshow]: Matplotlib, [imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html), [subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html).
[^colors]: Matplotlib, [Colormap normalization](https://matplotlib.org/stable/users/explain/colors/colormapnorms.html).
[^check]: PyTorch, [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close); Python, [assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement).
