+++
title = "PyTorch 첫 학습: 텐서 모양부터 역전파와 옵티마이저까지"
description = "텐서의 모양, 같은 객체와 새 객체, 자동 미분의 계산 기록을 분리해서 익힌다. 각 개념을 작은 코드로 확인한 뒤 회귀 학습과 미니배치 분류, 옵티마이저 비교로 연결한다."
date = 2026-10-23T16:00:00+09:00
slug = "pytorch-training-basics"

[extra]
katex = true
toc = true
styles = ["pytorch-training-basics/article.css"]
+++

입력 숫자 하나를 받아 다른 숫자를 예측하려고 한다. 모델은 입력에 가중치를 곱하고 편향을 더해 예측한다. 그 예측이 정답과 다르면 가중치와 편향을 수정한다. 여기서 알아야 할 것은 세 가지다. **어떤 숫자끼리 계산하는지, 계산 결과를 어디에 저장하는지, 무엇을 근거로 값을 수정하는지**다.

예를 들어 예측 100개와 정답 100개가 있어도, 텐서의 모양을 잘못 맞추면 같은 샘플끼리 비교하는 대신 모든 조합을 비교할 수 있다. 또 같은 숫자가 나오는 뺄셈이어도 기존 텐서의 값을 바꾸는 코드와 새로운 텐서를 만드는 코드는 다르게 동작한다.

아래에서는 회귀용 합성 데이터와 두 초승달 모양의 분류 데이터를 사용한다. 실제 서비스의 성능을 측정한 데이터는 아니다. Python에서 변수 대입, 함수 호출, 기본 반복문을 읽을 수 있고 `torch`, `scikit-learn`, `matplotlib`이 설치된 환경을 전제로 한다. PyTorch의 기능과 이 글에 필요한 추가 문법은 사용하는 위치에서 설명한다. 코드 블록은 위에서 아래로 실행할 수 있으며, 전체 회귀 예제는 데이터 생성부터 다시 포함한다.

## 1. 먼저 학습의 전체 순서를 구분한다

**샘플**은 데이터 한 건이고, **특징**은 그 샘플을 설명하는 입력 항목이다. **가중치**는 입력에 곱하는 숫자이며, **편향**은 곱셈 결과에 더하는 숫자다. 학습으로 수정하는 이런 숫자를 **파라미터**라고 부른다.

| 이름 | 이번 예제에서의 역할 |
| --- | --- |
| 입력 | 예측에 사용하는 데이터다. |
| 정답 | 각 입력에 대응하는 비교 기준이다. |
| 예측 | 현재 가중치와 편향으로 계산한 결과다. |
| 손실 | 예측이 정답에 맞지 않는 정도를 수치로 계산한 것이다. |
| 기울기 | 특정 파라미터를 조금 바꿀 때 손실이 얼마나 변하는지 나타낸다. |
| 옵티마이저 | 기울기를 받아 정해진 규칙으로 파라미터를 수정하는 도구다. |

**순전파**는 입력으로 예측을 계산하는 과정이다. 예측과 정답으로 손실을 계산한 뒤, **역전파**로 각 파라미터의 기울기를 구한다. 마지막에 그 기울기를 이용해 파라미터를 수정한다. 역전파와 파라미터 수정은 별개다.

```text
입력과 정답 준비 → 가중치와 편향 준비
    ↓
예측 계산 → 손실 계산 → 기울기 계산 → 가중치와 편향 수정
    ↑                                           │
    └──────────── 수정한 값으로 다시 계산 ─────────┘
```

함께 처리할 샘플 묶음이 **배치**다. 전체 학습 데이터를 한 차례 처리하면 **에폭** 하나다. 우선 회귀 데이터 전체를 한 배치로 처리하고, 뒤에서 작은 배치로 나누는 방법을 다룬다.

## 2. 텐서의 값, 모양, 차원 수를 따로 확인한다

**텐서(Tensor)**는 숫자들을 일정한 구조로 저장하고 계산하는 PyTorch의 자료형이다. `torch.tensor(...)`는 전달한 숫자로 텐서를 만드는 함수다. 텐서처럼 프로그램에서 실제로 만들어진 대상을 **객체**라고 한다. 변수는 그 객체를 가리키는 이름이다.

`shape`는 각 차원의 크기, `ndim`은 차원의 개수, `numel()`은 전체 원소 개수를 나타낸다. `dtype`은 각 원소를 어떤 자료형으로 저장하는지 나타낸다. `shape`처럼 괄호 없이 읽는 정보는 **속성**, `numel()`처럼 객체에 붙여 호출하는 함수는 **메서드**다.

아래의 `a`, `b`, `c`, `d`는 각각 숫자 하나를 뜻한다.

| 구조 | shape | 차원 수 | 원소 수 |
| --- | --- | ---: | ---: |
| `a` | `()` | 0 | 1 |
| `[a]` | `(1,)` | 1 | 1 |
| `[[a]]` | `(1, 1)` | 2 | 1 |
| `[[a, b], [c, d]]` | `(2, 2)` | 2 | 4 |

`(1,)`의 쉼표는 항목이 하나인 튜플을 나타내는 Python 문법이다. 숫자가 하나라고 반드시 0차원 텐서는 아니다. 먼저 학습 설정 없이 모양만 확인하자.

```python
import torch  # PyTorch의 텐서 기능을 가져온다.

values = torch.tensor([2.0])  # 숫자 하나를 1차원 구조로 저장한다.
print(values.shape)  # torch.Size([1]): shape가 (1,)이다.
print(values.ndim)  # 1: 차원이 하나다.
print(values.numel())  # 1: 원소도 하나다.
```

입력 `X`의 모양이 `(500, 2)`라면, 이 글에서는 **행이 샘플**, **열이 특징**이다. 샘플 500개에 각각 특징이 2개 있으며, 텐서 자체는 2차원이다.

모델 설명의 ‘입력 특징이 2차원’은 특징 두 개를 뜻하기도 한다. 그 샘플 하나는 `(2,)`로, 여러 샘플은 `(N, 2)`로 표현할 수 있다. `N`은 샘플 수다. **특징 개수와 텐서 차원 수는 서로 다른 기준이다.**

## 3. 입력과 정답을 같은 행끼리 대응시킨다

**회귀**는 숫자 값을 예측하는 문제다. 입력 `x`에 대해 `2*x + 1`에 작은 잡음을 더한 값을 정답으로 만들자. **잡음**은 규칙에서 조금씩 벗어나도록 더하는 값이다. `2`와 `1`은 정답을 생성하는 데 사용한 계수이지, 앞으로 모델이 학습할 가중치와 편향의 초기값이 아니다.

### 3.1. 입력 모양부터 만든다

`torch.linspace(시작, 끝, 개수)`는 양 끝값을 포함해 같은 간격의 숫자를 만든다. `unsqueeze(dim)`은 지정한 위치에 크기가 1인 차원을 추가한 텐서를 반환한다. 차원 위치는 0부터 시작한다. 원소 값과 개수는 바뀌지 않는다.[^shape]

<figure class="pt-figure">
<img src="/pytorch-training-basics/unsqueeze.svg" alt="(N,)에 unsqueeze(0)을 적용하면 (1,N), unsqueeze(1)을 적용하면 (N,1)이 된다.">
<figcaption>샘플 N개에 특징 하나씩 필요하므로 (N, 1)을 사용한다. (1, N)은 샘플 하나에 특징 N개를 놓는 모양이다.</figcaption>
</figure>

```python
x = torch.linspace(0, 1, 100)  # 0과 1을 포함한 숫자 100개: (100,).
x2 = x.unsqueeze(1)  # 샘플 100개 × 특징 1개: (100, 1).
```

`unsqueeze(1)`의 `1`은 추가할 원소 수가 아니라 **새 차원의 위치**다. `x`는 여전히 `(100,)`이고, 반환받은 `x2`는 `(100, 1)`이다. `x.unsqueeze(1)`만 호출하고 반환값을 사용하지 않으면 `x`의 모양은 바뀌지 않는다.

### 3.2. 난수로 잡음을 더한다

**난수**는 정해진 생성 방식으로 뽑은 무작위 숫자다. `torch.manual_seed(42)`는 난수 생성의 시작 상태를 정한다. 같은 환경에서 같은 순서로 실행한 실험을 재현할 때 사용하며, `42` 자체에 특별한 의미는 없다.

`torch.randn(100, 1)`은 평균 0, 표준편차 1인 **표준정규분포**에서 숫자를 뽑아 `(100, 1)` 텐서를 만든다. 표준편차는 값이 평균 주변에 얼마나 퍼지는지 나타내는 수치다. 여기에 `0.05`를 곱하면 잡음 분포의 표준편차가 `0.05`가 된다. 모든 값이 `-0.05`와 `0.05` 사이에 제한된다는 뜻은 아니다.[^random]

```python
torch.manual_seed(42)  # 이어지는 난수 생성의 시작 상태를 정한다.
noise = 0.05 * torch.randn(100, 1)  # 샘플마다 더할 잡음: (100, 1).
y2 = 2 * x2 + 1 + noise  # 각 입력에 대응하는 정답: (100, 1).
```

```text
입력의 첫 행  → 첫 번째 잡음을 더한 정답의 첫 행
입력의 둘째 행 → 두 번째 잡음을 더한 정답의 둘째 행
...
x2 (100, 1)                    y2 (100, 1)
```

`linspace`는 난수를 사용하지 않으므로 시드 설정의 영향을 받지 않는다. 난수 100개의 실제 평균이 정확히 0일 필요도 없다. 장치나 PyTorch 버전까지 달라지면 같은 시드만으로 완전히 같은 결과를 보장하지는 않는다.

<details>
<summary>추가 확인: unsqueeze는 숫자를 복사하는가?</summary>

`x2`는 `x`와 숫자가 저장된 메모리를 공유한다. 같은 저장 내용을 다른 모양으로 보는 이런 텐서를 **뷰(view)**라고 한다. 두 변수의 모양은 달라도 숫자를 저장한 부분은 공유하므로, 한쪽의 원소 값을 직접 수정하면 다른 쪽에서도 그 수정이 보인다. 이 글에서는 입력값을 수정하지 않고 모양만 맞추어 사용한다.

</details>

## 4. 오류 없이 실행되어도 샘플 대응이 틀릴 수 있다

원하는 결과를 먼저 정하자. 샘플 100개에 예측값 하나씩 필요하므로 예측 모양은 정답과 같은 **`(100, 1)`**이다.

`*`는 원소별 곱셈이고, `@`는 행렬곱이다. 행렬곱은 왼쪽의 한 행과 오른쪽의 한 열에서 대응하는 값들을 곱해 더한다. 2차원끼리는 `(행 수, 중간 크기) @ (중간 크기, 열 수)`에서 중간 크기가 맞아야 한다. 결과는 `(행 수, 열 수)`다. 다만 오른쪽이 1차원 벡터라면 이번 예제의 결과는 1차원이 된다.[^shape]

여기서는 모양만 검사하므로 가중치와 편향에도 학습 설정을 넣지 않는다. `torch.zeros(1)`은 0 하나를 담은 `(1,)` 텐서를 만든다.

```python
w = torch.randn(1)  # 가중치 숫자 하나: (1,).
b = torch.zeros(1)  # 편향 숫자 하나: (1,).
wrong_pred = x2 @ w + b  # (100, 1) @ (1,) → (100,).
print(wrong_pred.shape)  # torch.Size([100]): 원하는 (100, 1)이 아니다.
```

### 4.1. 뺄셈에서 브로드캐스팅이 적용되는 과정

**브로드캐스팅**은 크기가 다른 텐서를 원소별로 계산할 때 사용하는 모양 맞춤 규칙이다. 오른쪽 차원부터 비교해서 크기가 같거나 둘 중 하나가 1이면 계산할 수 있다. 부족한 앞쪽 차원은 크기가 1인 것처럼 취급한다. 같은 값을 여러 위치에 사용한 것처럼 계산하지만 실제 데이터를 복사해야 한다는 뜻은 아니다.

```text
예측: (100,)     → 앞쪽 차원이 부족하므로 (1, 100)처럼 취급
정답: (100, 1)   → 각 행에 정답 하나
결과: (100, 100) → 정답 하나가 예측 100개 전체와 비교됨
```

<figure class="pt-figure">
<img src="/pytorch-training-basics/broadcasting.svg" alt="예측 한 행과 정답 한 열은 모든 조합의 오차를 만든다. 두 텐서가 모두 한 열이면 같은 행끼리 비교한다.">
<figcaption>세 샘플만 표시했다. 필요한 것은 아래쪽의 일대일 비교이지, 위쪽의 모든 조합 비교가 아니다.</figcaption>
</figure>

```python
wrong_error = wrong_pred - y2  # 예측 100개와 정답 100개의 모든 조합을 비교한다.
print(wrong_error.shape)  # torch.Size([100, 100]): 오차가 10,000개다.
```

### 4.2. 원하는 모양으로 바꾸고 검사한다

가중치를 `(1, 1)`로 계산에 넣으면 출력이 `(100, 1)`로 유지된다. 아래의 `assert`는 조건이 거짓이면 실행을 멈추는 검사문이다. `torch.allclose`는 작은 실수 계산 오차를 허용하며 두 텐서의 값이 같은지 확인한다.

```python
y_hat = x2 @ w.unsqueeze(1) + b  # (100, 1) @ (1, 1) → (100, 1).
assert y_hat.shape == y2.shape  # 예측과 정답의 모양을 검사한다.
same_pred = x2 * w + b  # 입력 특징이 하나인 이번 문제에서는 같은 예측이다.
assert torch.allclose(y_hat, same_pred)  # 두 계산 결과의 값도 비교한다.
```

특징이 여러 개면 `*`와 `@`는 일반적으로 다른 계산이다. 모양이 같아도 샘플 순서가 다르면 비교가 틀린다. **원하는 출력 모양 → 실제 결과 모양 → 같은 샘플끼리의 대응**을 차례로 확인해야 한다.

## 5. 값을 변환하는 함수는 계산 단위부터 구분한다

**활성화 함수**는 신경망의 중간값 등에 적용하는 변환 함수다. **클래스**는 분류할 범주이고, **로짓(logits)**은 확률로 바꾸기 전의 클래스별 점수다. 점수는 음수일 수도 있다.

| 함수 | 어떤 값끼리 계산하는가 | 결과의 특징 |
| --- | --- | --- |
| Sigmoid | 각 원소를 독립적으로 변환한다. | 각 값이 0과 1 사이가 된다. 전체 합은 1일 필요가 없다. |
| ReLU | 각 원소를 독립적으로 변환한다. | 음수만 0으로 바꾸고 나머지는 유지한다. |
| Softmax | 한 샘플의 클래스 점수들을 함께 계산한다. | 해당 샘플의 확률 합이 1이 된다. |

### 5.1. Sigmoid와 ReLU를 먼저 확인한다

`torch.sigmoid`와 `torch.relu`는 입력과 같은 모양의 텐서를 반환한다. 다음 입력은 `(2, 3)`이므로 두 결과도 `(2, 3)`이다.[^activation]

```python
z = torch.tensor([[-2.0, 0.0, 3.0], [1.0, -4.0, 2.0]])  # 비교할 값: (2, 3).
sigmoid_out = torch.sigmoid(z)  # 원소마다 Sigmoid를 적용한다.
relu_out = torch.relu(z)  # 결과: [[0, 0, 3], [1, 0, 2]].
```

직접 계산식을 쓰려면 두 함수를 더 알아야 한다. `torch.exp(z)`는 각 원소에 `e`의 거듭제곱을 적용한다. `e`는 약 `2.71828`인 수학 상수다. `torch.where(조건, 참일 때 값, 거짓일 때 값)`는 각 위치에서 조건에 맞는 값을 선택한다.

```python
sigmoid_manual = 1 / (1 + torch.exp(-z))  # Sigmoid의 계산식이다.
relu_manual = torch.where(z > 0, z, 0.0)  # 양수 위치는 원래 값, 그 밖은 0이다.
assert torch.allclose(sigmoid_out, sigmoid_manual)  # 위 입력에서 결과를 확인한다.
assert torch.allclose(relu_out, relu_manual)  # 값의 변환 결과를 확인한다.
```

ReLU는 1보다 큰 값도 그대로 둔다. Sigmoid는 수학적으로 유한한 입력을 0보다 크고 1보다 작은 값으로 바꾸지만, 컴퓨터의 제한된 정밀도에서는 극단적인 결과가 0 또는 1로 표시될 수 있다. 직접 쓴 지수 계산식도 큰 입력에서 문제가 생길 수 있으므로 실제 모델에는 내장 함수를 사용한다. 위 비교는 출력값을 확인한 것이며 모든 지점에서 미분 처리까지 동일하다는 주장은 아니다.

### 5.2. Softmax의 dim은 어느 값들을 함께 계산할지 정한다

`logits2`의 행은 샘플, 열은 클래스다. 샘플 두 개에 클래스 세 개의 점수를 놓는다. `torch.softmax(logits2, dim=1)`에서 `dim=1`은 열에 놓인 클래스 점수들끼리 계산하라는 뜻이다. 결과 모양은 `(2, 3)` 그대로다.

```python
logits2 = torch.tensor([[2.0, 0.5, 0.3], [0.2, 2.5, 0.1]])  # 점수: (2, 3).
probs = torch.softmax(logits2, dim=1)  # 각 행을 클래스 확률 세 개로 바꾼다.
```

계산은 **각 점수에 지수 적용 → 같은 행의 값들을 합산 → 각 값을 자기 행의 합계로 나눔**이다. `.sum(dim=1, keepdim=True)`는 열을 합산하되 그 차원을 크기 1로 남긴다. 따라서 합계 모양은 `(2,)`가 아니라 `(2, 1)`이다.

<figure class="pt-figure">
<img src="/pytorch-training-basics/softmax.svg" alt="각 행의 지수값을 자기 행의 합계로 나눈다. 두 행의 합계는 (2,1)로 저장한다.">
<figcaption>a부터 f는 지수 계산 결과다. 첫 행은 s₁, 둘째 행은 s₂로 나누므로 샘플끼리 섞이지 않는다.</figcaption>
</figure>

```python
exp_logits = torch.exp(logits2)  # 작은 점수로 계산 원리를 확인한다: (2, 3).
row_sums = exp_logits.sum(dim=1, keepdim=True)  # 각 행 합계: (2, 1).
probs_manual = exp_logits / row_sums  # 자기 행 합계로 나눔: (2, 3).
assert torch.allclose(probs, probs_manual)  # 내장 함수의 결과와 비교한다.
```

`dim=-1`은 마지막 차원을 뜻하므로 이 2차원 입력에서는 `dim=1`과 같다. 클래스 점수만 담은 1차원 입력에서는 `dim=0` 또는 `dim=-1`을 사용한다.

<details>
<summary>추가 확인: 큰 점수에서도 직접 계산하려면</summary>

위 수동 코드는 작은 점수로 계산 순서를 확인하기 위한 것이다. 큰 양수에 지수 계산을 적용하면 컴퓨터가 표현할 수 있는 범위를 넘을 수 있다. 각 행의 최댓값을 먼저 빼면 지수에 들어가는 값이 0 이하가 된다.

`.amax(dim=1, keepdim=True)`는 행마다 최댓값 하나를 구해 `(2, 1)`로 남긴다. 같은 행의 모든 점수에서 같은 값을 빼면 지수값들에 같은 비율이 적용되고, 분자와 분모에서 그 비율이 상쇄되어 최종 확률은 같다.

```python
row_max = logits2.amax(dim=1, keepdim=True)  # 행마다 최댓값: (2, 1).
shifted = logits2 - row_max  # 각 행의 점수에서 해당 행의 최댓값을 뺀다.
exp_shifted = torch.exp(shifted)  # 너무 큰 양수의 지수 계산을 피한다.
shifted_sums = exp_shifted.sum(dim=1, keepdim=True)  # 행별 합계다.
probs_stable = exp_shifted / shifted_sums  # 최종 확률을 계산한다.
assert torch.allclose(probs, probs_stable)  # 원래 Softmax와 비교한다.
```

</details>

## 6. 손실의 계산 원리와 계산 도구의 사용법을 나눈다

### 6.1. MSE는 오차를 제곱한 뒤 평균낸 값이다

**MSE(평균제곱오차)**는 예측과 정답을 같은 위치끼리 빼고, 제곱하고, 평균낸 값이다. `.square()`는 각 원소를 제곱하고 `.mean()`은 원소 전체의 평균을 계산한다.

```text
예측: [2.5, 0.0, 2.0]       정답: [3.0, -0.5, 2.0]
오차: [-0.5, 0.5, 0.0]
제곱: [0.25, 0.25, 0.0]
평균: (0.25 + 0.25 + 0.0) / 3 = 0.166666...
```

```python
pred = torch.tensor([2.5, 0.0, 2.0])  # 예측 세 개다.
target = torch.tensor([3.0, -0.5, 2.0])  # 같은 순서의 정답 세 개다.
manual_loss = (pred - target).square().mean()  # shape ()인 손실 텐서를 만든다.
```

기본 MSE는 **모든 출력 원소**를 평균낸다. 샘플마다 출력이 여러 개면 샘플 수만으로 나누는 것이 아니다. 메서드 이름은 `.squared()`가 아니라 `.square()`다.[^loss]

### 6.2. 손실 객체 생성과 실제 계산을 분리한다

여기서 Python의 **클래스**는 객체를 만드는 정의다. 앞서 분류에서 말한 ‘클래스=범주’와 다른 뜻이다. `nn.MSELoss`는 손실 계산 객체를 만드는 클래스다. `nn.MSELoss()`로 객체를 만든 뒤, 그 객체에 예측과 정답을 전달하면 손실을 계산한다.

```text
nn.MSELoss()       → 손실 계산 객체 생성
loss_fn           → 그 객체를 가리키는 변수
loss_fn(pred, target) → 실제 예측과 정답으로 손실 계산
```

```python
from torch import nn  # 층과 손실 계산 클래스를 제공하는 부분을 가져온다.

loss_fn = nn.MSELoss()  # 계산 도구를 준비한다. 아직 데이터는 비교하지 않는다.
loss = loss_fn(pred, target)  # 이 호출에서 실제 손실을 계산한다.
assert torch.allclose(loss, manual_loss)  # 직접 계산한 MSE와 같다.
```

`nn.MSELoss(pred, target)`처럼 객체 생성 자리에 데이터를 넣지 않는다. `nn.MSELoss()(pred, target)`는 생성과 호출을 붙여 쓴 표현이지만, 두 줄로 나누면 어느 단계인지 더 분명하다.

### 6.3. 분류 손실에는 원래 점수와 정답 번호를 넣는다

`nn.CrossEntropyLoss`는 분류용 손실 계산 객체를 만드는 클래스다. 여기서는 샘플마다 **정답 클래스 번호 하나**를 전달한다. 정답 클래스에 부여한 확률이 낮을수록 손실이 커지는 방식이다. 이 도구에는 필요한 계산이 포함되어 있으므로 **Softmax를 먼저 적용하지 않은 로짓**을 전달한다.[^loss]

`N`은 샘플 수, `C`는 클래스 수다. `torch.long`은 클래스 번호 같은 정수를 저장하는 64비트 정수 자료형이며 `torch.int64`와 같다.

| 전달값 | 모양 | 의미 |
| --- | --- | --- |
| `logits` | `(N, C)` | 각 샘플의 클래스별 실수 점수 |
| `labels` | `(N,)` | 각 샘플의 정답 번호. 0부터 C−1까지 |
| 기본 결과 | `()` | 샘플별 손실을 평균낸 숫자 하나 |

```python
labels = torch.tensor([0, 1], dtype=torch.long)  # 첫 샘플은 0번, 둘째는 1번 클래스다.
ce_fn = nn.CrossEntropyLoss()  # 손실 계산 객체를 만든다.
ce = ce_fn(logits2, labels)  # 앞 절의 점수를 넣는다. probs를 넣지 않는다.
print(ce)  # tensor(0.2577)
```

`labels.long()`은 이미 있는 텐서를 이 정수 자료형으로 변환하는 메서드다. 위처럼 생성할 때 지정했다면 추가 변환은 필요 없다. 확률들을 정답으로 전달하는 다른 방식도 지원하지만, 이 글은 정수 번호 방식을 사용한다.

<details>
<summary>추가 확인: CrossEntropyLoss의 내부 계산 이름</summary>

정답 클래스의 확률을 `p`라고 하면 이 설정의 샘플별 손실은 `-log(p)`다. 여기서 `log`는 자연로그이며, 앞에서 사용한 지수 함수의 역함수다. 즉 `exp(a)=p`일 때 `log(p)=a`다. 확률이 1이면 손실은 0이고, 확률이 0에 가까워질수록 손실은 커진다.

`LogSoftmax`는 클래스 확률의 자연로그를 계산하는 함수다. `NLLLoss`는 그 결과에서 정답 클래스 위치의 값을 선택하고 부호를 바꾸어 손실을 구한다. 정수 라벨 방식의 CrossEntropyLoss는 이 두 작업을 연결한 계산과 동등하다. 실제 코드에서는 이 함수들을 따로 조립하지 않고 `nn.CrossEntropyLoss()`를 사용한다.

</details>

## 7. 자동 미분은 기록한 계산으로 기울기를 구한다

앞에서는 예측과 손실의 값만 계산했다. 이제 **어느 가중치를 어떻게 바꿀지 판단할 기울기**가 필요하다. PyTorch가 계산 관계를 이용해 기울기를 구하는 기능이 **자동 미분**이다.

자동 미분은 두 단계로 나뉜다. 먼저 예측과 손실을 계산할 때 어떤 텐서에 어떤 연산을 적용했는지 기록한다. 그다음 손실에서 출발하여 기록을 거슬러 기울기를 계산한다. 아직 기울기를 구하지 않았다는 것과 계산 기록이 없다는 것은 다르다.

| 기능 | 언제 필요한가 | 무엇을 하는가 |
| --- | --- | --- |
| `requires_grad=True` | 학습할 텐서를 준비할 때 | 이후 이 텐서의 기울기를 구하도록 설정한다. |
| `loss.backward()` | 손실을 계산한 뒤 | 기록된 관계를 사용해 기울기를 계산한다. |
| `w.grad` | 기울기를 계산한 뒤 | `w`의 각 원소에 대응하는 기울기를 보관한다. |

`True`는 참을 나타내는 값이다. `None`은 현재 값이 없음을 나타낸다. `.grad`가 `None`인 것은 가중치나 기울기가 0이라는 뜻이 아니다.

```python
w = torch.randn(1, requires_grad=True)  # 이제 가중치의 기울기 계산을 설정한다.
b = torch.zeros(1, requires_grad=True)  # 편향도 학습 대상이다.
y_hat = x2 @ w.unsqueeze(1) + b  # 예측: (100, 1). 계산 관계도 기록한다.
loss = (y_hat - y2).square().mean()  # 평균 손실: ().
print(w.grad)  # None: 아직 backward를 실행하지 않았다.
loss.backward()  # 현재 손실의 가중치·편향 기울기를 계산한다.
print(w.grad.shape, b.grad.shape)  # 둘 다 torch.Size([1])이다.
```

```text
순전파: w, b → 예측 → 손실
역전파: 손실 → 예측을 만든 계산 → w의 기울기, b의 기울기
```

손실은 숫자 하나여도 각 파라미터가 그 손실에 미치는 영향은 다르다. 따라서 파라미터마다 기울기가 생긴다. 중간 계산이 뒤의 계산에 미치는 영향을 연결하는 미분 규칙을 **연쇄법칙**이라고 한다. 여기서 `backward()`는 그 계산을 수행하며, `w`와 `b`의 값 자체는 수정하지 않는다.[^autograd]

다음 순전파와 역전파를 수행하면 새 기울기는 기존 `.grad`에 더해진다. 배치마다 별도로 갱신하려면 이전 기울기를 비운다. `w.grad = None`은 **기울기 저장값만 비우며 가중치는 유지**한다.

<details>
<summary>추가 확인: 기울기는 어느 텐서에 저장되는가?</summary>

위의 `w`, `b`는 다른 미분 연산의 결과가 아니라 직접 만든 학습 시작 텐서다. 이런 텐서를 **리프 텐서**라고 한다. `requires_grad=True`인 리프 텐서에는 기본적으로 `.grad`가 저장된다.

`w.unsqueeze(1)`은 `w`로부터 계산한 중간 텐서다. 그 결과로 예측해도 연결을 따라 기울기가 원본 `w`까지 전달된다. 중간 텐서의 `.grad`를 모두 보관하는 것은 아니다. 따라서 원본 변수 `w`는 유지하고, 필요한 위치에서 `w.unsqueeze(1)`의 결과를 사용한다.

</details>

## 8. 같은 값을 계산해도 같은 텐서를 수정하는 것은 아니다

이 절에서는 학습이나 미분 없이 **텐서의 값과 변수의 연결**만 확인한다. 비교할 코드는 `w -= 0.5`와 `w = w - 0.5`다. 시작값이 같으면 뺄셈 결과의 숫자는 같다. 차이는 기존 텐서가 바뀌는지에 있다.

먼저 `same_object = w`는 텐서 복사가 아니다. 두 이름이 같은 텐서 하나를 가리키게 한다. `is`는 두 이름이 **같은 객체**를 가리키는지 검사하는 Python 연산자다.

### 8.1. -=는 기존 텐서의 값을 바꾼다

```python
w = torch.tensor([2.0])  # 숫자 2.0이 들어 있는 텐서를 만든다.
same_object = w  # 새 텐서를 만들지 않고 같은 텐서에 이름을 하나 더 붙인다.
w -= 0.5  # 기존 텐서의 값을 2.0에서 1.5로 바꾼다.
print(w)  # tensor([1.5000])
print(same_object)  # tensor([1.5000]): 같은 텐서를 보므로 변경된 값이 보인다.
print(w is same_object)  # True: 여전히 같은 객체다.
```

```text
수정 전: w ───────────┐       수정 후: w ───────────┐
                     ├→ 텐서 [2.0]                ├→ 같은 텐서 [1.5]
         same_object ┘                same_object ┘
```

### 8.2. w = w - ...는 새 텐서에 w를 연결한다

공정하게 비교하려고 시작값을 다시 `2.0`으로 준비한다.

```python
w = torch.tensor([2.0])  # 두 번째 비교도 2.0에서 시작한다.
same_object = w  # 두 이름이 같은 텐서를 가리킨다.
w = w - 0.5  # 뺄셈 결과인 새 텐서를 만들고 w만 그 텐서에 연결한다.
print(w)  # tensor([1.5000]): 새 텐서의 값이다.
print(same_object)  # tensor([2.]): 기존 텐서는 수정하지 않았다.
print(w is same_object)  # False: 이제 서로 다른 객체다.
```

```text
수정 후: w ───────────→ 새 텐서 [1.5]
         same_object ─→ 기존 텐서 [2.0]
```

두 코드 모두 `w`를 읽으면 `1.5`가 나온다. 그러나 기존 텐서를 함께 가리키던 `same_object`를 읽으면 차이가 드러난다. **숫자가 같은지와 객체가 같은지는 별개의 질문이다.** 이 동작은 PyTorch 텐서의 뺄셈 구현에 관한 설명이다. Python의 일반 숫자에 `-=`를 사용한다고 기존 숫자 객체가 수정되는 것은 아니다.[^assignment]

기호 순서를 뒤집은 `w =- 0.5`는 `w = -0.5`다. ‘현재 값에서 0.5를 빼기’가 아니라 ‘음수 0.5를 대입하기’이므로, 갱신 코드를 쓰려다 그렇게 입력했다면 오타다.

## 9. 계산 기록을 끄는 기능을 익힌 뒤 학습에 연결한다

### 9.1. 계산 실행과 계산 기록은 서로 다른 작업이다

7절에서 자동 미분은 순전파의 계산 관계를 기록한다고 했다. **숫자 계산은 실행하되 역전파를 위한 기록은 남기지 않을 때** 사용하는 것이 `torch.no_grad()`다. 텐서 수정과 새 텐서 대입을 구별하는 기능이 아니다.

`with`는 정해진 범위에 들어갈 때 설정을 적용하고, 그 범위를 벗어날 때 정리하는 Python 문법이다. 여기서는 `with torch.no_grad():` 다음에 **들여쓴 줄들 동안** 기록을 끈다. 들여쓰기가 끝나면 블록 전에 사용하던 기록 설정으로 돌아간다.[^no_grad]

먼저 가중치를 수정하지 않고, 같은 곱셈을 세 번 실행해 차이만 확인하자.

```python
source = torch.tensor([2.0], requires_grad=True)  # 기울기를 구할 시작 텐서다.
recorded = source * 2  # 기록이 켜진 상태에서 곱셈을 한다.

with torch.no_grad():
    unrecorded = source * 2  # 곱셈은 실행하지만 이 곱셈의 미분 기록은 만들지 않는다.

recorded_again = source * 2  # 블록 밖이므로 이전 기록 설정이 복구되어 있다.
print(recorded.requires_grad)  # True
print(unrecorded.requires_grad)  # False
print(recorded_again.requires_grad)  # True
print(source.requires_grad)  # True: 원본 텐서의 설정은 바꾸지 않았다.
```

세 곱셈 결과의 숫자는 모두 `4`다. 다른 것은 **해당 결과를 만들 때 미분 계산 관계를 기록했는지**다. `no_grad`는 이미 구한 `.grad`를 지우지 않으며, 원본의 `requires_grad` 설정을 영구히 끄지도 않는다.

### 9.2. 가중치 수정 구간에서 사용하는 이유

이번 학습은 ‘현재 가중치로 만든 손실의 기울기’를 구하고, 그 기울기로 가중치를 수정하는 방식이다. **가중치를 수정하는 연산 자체까지 다음 역전파의 계산에 포함하려는 것은 아니다.** 그래서 수정할 때만 기록을 끈다. PyTorch는 기록이 켜진 상태에서 이런 학습 시작 텐서를 직접 수정하려고 하면 오류를 발생시킨다.[^autograd]

```text
기록을 켠 상태: 예측 계산 → 손실 계산 → backward로 기울기 계산
기록을 끈 구간: 이미 구한 기울기로 기존 가중치와 편향의 값만 수정
구간 종료:      기울기를 비우고, 다음 순전파의 계산 관계를 새로 기록
```

여기서 두 선택은 역할이 다르다. **`-=`는 기존 텐서를 유지하고 값을 바꾸는 선택**이다. **`no_grad`는 수정 연산을 미분 기록에 넣지 않는 선택**이다. 어느 하나가 다른 하나의 뜻은 아니다.

`no_grad` 안에서 `w = w - ...`로 새 뺄셈 결과를 만들면 그 결과에는 기울기 추적이 설정되지 않는다. 그 새 결과로 `w`를 교체하면 원래 학습 대상과 달라진다. 아래에서는 학습 대상인 기존 `w`, `b`를 유지하며 갱신한다. 이것은 역전파가 끝난 뒤의 갱신 방법이지, 순전파 중 필요한 중간값을 임의로 덮어써도 된다는 뜻은 아니다.

### 9.3. 전체 회귀 학습을 연결한다

**학습률** `lr`은 기울기를 이용한 수정량의 크기를 조절한다. 이번 기본 갱신식은 `새 값 = 현재 값 - lr × 기울기`다. 기울기는 최적값까지의 거리가 아니므로 학습률이 지나치게 크면 손실이 커질 수 있다.

기록용 코드도 구분하자. `loss.item()`은 원소 하나짜리 텐서에서 Python 숫자를 꺼낸다. `losses = []`는 빈 리스트를 만들고, `.append(...)`는 그 끝에 값을 추가한다. 꺼낸 Python 숫자는 기록용이며 역전파에는 원래 `loss` 텐서를 사용한다. `range(500)`은 반복문을 500번 실행하게 한다.

```python
import torch  # 이 블록만 실행해도 회귀 실험을 시작할 수 있다.

torch.manual_seed(42)  # 난수 생성의 시작 상태를 정한다.
x2 = torch.linspace(0, 1, 100).unsqueeze(1)  # 입력: (100, 1).
y2 = 2 * x2 + 1 + 0.05 * torch.randn(100, 1)  # 정답: (100, 1).
w = torch.randn(1, requires_grad=True)  # 모든 샘플이 공유하는 가중치다.
b = torch.zeros(1, requires_grad=True)  # 모든 샘플이 공유하는 편향이다.
lr = 0.3  # 이번 실험의 학습률이다.
losses = []  # 손실 숫자들을 저장한다.

for step in range(500):
    y_hat = x2 @ w.unsqueeze(1) + b  # 현재 파라미터로 예측한다: (100, 1).
    assert y_hat.shape == y2.shape  # 의도하지 않은 모양의 비교를 막는다.
    loss = (y_hat - y2).square().mean()  # 평균 손실: ().
    losses.append(loss.item())  # 갱신 전에 계산한 손실을 기록한다.
    loss.backward()  # 현재 손실에 대한 기울기를 계산한다.

    with torch.no_grad():
        w -= lr * w.grad  # 미분 기록 없이 기존 가중치의 값을 수정한다.
        b -= lr * b.grad  # 기존 편향도 같은 방식으로 수정한다.

    w.grad = None  # 다음 반복에 이전 기울기가 더해지지 않도록 비운다.
    b.grad = None  # 편향의 기울기도 비운다.

print("w:", w.item(), "b:", b.item())  # 학습된 숫자를 확인한다.
```

매번 전체 100개를 사용하므로 한 반복이 전체 배치 학습 한 번이다. 가중치와 편향은 하나씩이지만, 100개 샘플 모두가 손실과 기울기 계산에 기여한다. **파라미터 개수와 계산량은 같은 것이 아니다.** 정답에 잡음이 있으므로 학습된 값이 정확히 `2`, `1`일 필요는 없다.

## 10. 텐서를 계산할 장치를 선택한다

**device**는 텐서를 저장하고 계산할 장치다. `cpu`는 CPU, 여기서 `cuda`는 NVIDIA GPU를 사용하는 경로, `mps`는 지원되는 Mac의 Apple GPU를 사용하는 경로다. `is_available()`은 현재 해당 경로를 사용할 수 있는지 참·거짓으로 반환한다.[^device]

**`.to(device)`는 텐서를 지정한 장치로 옮긴 결과를 반환하는 메서드**다. 기존 변수 이름을 자동으로 바꾸지 않으므로 반환값을 받아야 한다. 이미 장치와 자료형이 같다면 원래 텐서를 반환할 수 있다.

```python
if torch.cuda.is_available():
    device = "cuda"  # CUDA를 사용할 수 있으면 선택한다.
elif torch.backends.mps.is_available():
    device = "mps"  # CUDA가 없고 MPS가 있으면 선택한다.
else:
    device = "cpu"  # 두 GPU 경로가 없으면 CPU에서 실행한다.

t = torch.randn(2, 2)  # 기본 설정에서는 CPU에 텐서를 만든다.
t_on_device = t.to(device)  # 반환된 텐서를 새 변수로 받는다.
print(t_on_device.device)  # 실제 장치를 확인한다.
```

장치만 옮기는 이 코드에서는 모양과 값의 의미가 유지된다. 이후 모델을 사용할 때도 **모델의 파라미터, 현재 입력, 현재 정답을 같은 계산 장치에 준비**한다. 모델을 옮기는 코드는 모델 객체를 만드는 방법을 설명한 뒤에 연결한다.

## 11. 이미 정의된 층을 쓰고, 그다음 모델 클래스로 묶는다

### 11.1. nn.Linear 하나로 예측한다

**뉴런**은 입력마다 가중치를 곱하고 더한 뒤 편향을 더하는 계산 단위다. **층(layer)**은 그런 계산 단위들을 모은 단계다. `nn.Linear(입력 특징 수, 출력 뉴런 수)`는 그 계산을 하는 층을 만든다. 두 인자는 정수이며 `(1,)` 같은 모양 튜플이 아니다.[^linear]

`nn.Linear(1, 1)`은 특징 하나를 받아 값 하나를 출력한다. 내부에 필요한 가중치와 편향도 생성한다. 이 숫자들은 학습 파라미터로 등록된 특별한 텐서인 **Parameter**로 관리된다.

```python
linear = nn.Linear(1, 1)  # 가중치와 편향을 가진 층 객체를 만든다.
linear_prediction = linear(x2)  # 입력 (100, 1) → 출력 (100, 1).
print(linear.weight.shape)  # torch.Size([1, 1]): 가중치 숫자 하나다.
print(linear.bias.shape)  # torch.Size([1]): 편향 숫자 하나다.
```

층을 **생성한 것**, 그 층으로 **예측한 것**, 손실과 기울기를 구해 **학습한 것**은 다르다. 위 코드는 생성과 예측까지만 수행했다.

### 11.2. 필요한 계산 순서를 모델 클래스에 적는다

여러 층을 한 모델로 관리하려면 `nn.Module`을 기본으로 모델 클래스를 정의한다. 기존 클래스의 기능을 새 클래스에서도 사용하는 방식을 **상속**이라고 한다. 여기서는 파라미터와 하위 층을 관리하는 기능을 가져온다.

| 코드에 나오는 이름 | 역할 |
| --- | --- |
| `class LinearRegression(nn.Module)` | `nn.Module`의 기능을 사용하는 새 모델 클래스를 정의한다. |
| `__init__` | 모델 객체를 생성할 때 준비할 일을 적는 메서드다. |
| `self` | 현재 생성하거나 사용하는 모델 객체다. |
| `super().__init__()` | 먼저 `nn.Module`의 기본 관리 기능을 초기화한다. |
| `self.linear` | 현재 모델에 연결해 보관하는 선형층이다. |
| `forward` | 입력을 어떤 순서로 계산해 반환할지 적는 메서드다. |

`def`는 함수나 메서드의 정의를 시작하고, `return`은 계산 결과를 호출한 곳으로 돌려준다. `model(x2)`로 모델을 호출하면 PyTorch가 이 `forward` 계산을 실행한다.

```python
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()  # 하위 층과 파라미터를 관리할 준비를 한다.
        self.linear = nn.Linear(1, 1)  # 입력 하나 → 출력 하나인 층을 보관한다.

    def forward(self, x):
        return self.linear(x)  # 보관한 층으로 예측하고 결과를 반환한다.

model = LinearRegression()  # 준비 과정인 __init__이 실행된다.
prediction = model(x2)  # 계산 과정인 forward가 실행된다.
```

`self.linear`에 층을 연결하면 모델이 그 층의 파라미터도 찾을 수 있다. 일반 텐서를 아무 속성에 저장한다고 모두 학습 파라미터로 등록되는 것은 아니다.

### 11.3. 파라미터를 꺼내 확인하고 장치로 옮긴다

`model.parameters()`는 학습 파라미터 텐서를 차례로 제공하는 **이터레이터**를 반환한다. 이터레이터는 항목을 하나씩 꺼낼 수 있는 객체다. `named_parameters()`는 각 텐서의 이름도 함께 제공한다. 아래 반복문은 그 두 항목을 `name`, `p`로 각각 받는다.

```python
for name, p in model.named_parameters():
    print(name, p.shape)  # linear.weight (1, 1), linear.bias (1,)를 확인한다.

model = model.to(device)  # 모델 내부의 파라미터 등을 선택한 장치로 옮긴다.
```

텐서의 `.to()`와 달리 모델의 `.to()`는 모델 내부에서 등록된 파라미터 등을 변환하고 모델을 반환한다. 준비 순서는 **모델 생성 → 장치 이동 → 해당 모델의 파라미터로 옵티마이저 생성**이다. 옵티마이저 사용법은 14절에서 연결한다.

## 12. 2 → 16 → 2 모델에서 각 숫자가 세는 것을 확인한다

이제 회귀에서 분류로 문제를 바꾼다. 새 문제의 입력은 좌표 두 개이고, 정답은 클래스 0 또는 1이다. **MLP**는 선형층과 활성화 함수를 연결한 신경망이다. **은닉층**은 입력과 최종 출력 사이에서 중간값을 계산하는 층이다.

이번 모델에는 은닉층 하나와 출력층 하나가 있다. 은닉 뉴런 16개는 같은 입력 특징 두 개를 각각 자기 가중치로 계산한다. 출력 뉴런 두 개는 클래스별 점수를 하나씩 만든다. `fc1`, `fc2`는 첫째·둘째 선형층에 붙인 이름이다.

```text
입력 특징 2개 → 은닉 뉴런 16개 → 클래스 점수를 만드는 출력 뉴런 2개
```

`B`는 현재 배치의 샘플 수다. `z1`은 첫 선형층 결과, `h1`은 그 값에 ReLU를 적용한 결과다. 입출력 모양을 먼저 놓으면 다음과 같다.

```text
x (B, 2) → fc1 → z1 (B, 16) → ReLU → h1 (B, 16) → fc2 → logits (B, 2)
```

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 모델의 파라미터 관리 기능을 준비한다.
        self.fc1 = nn.Linear(2, 16)  # 특징 2개로 중간값 16개를 만든다.
        self.fc2 = nn.Linear(16, 2)  # 중간값 16개로 클래스 점수 2개를 만든다.

    def forward(self, x):
        z1 = self.fc1(x)  # (B, 2) → (B, 16).
        h1 = torch.relu(z1)  # 음수만 0으로 바꾼다. 모양은 (B, 16)이다.
        logits = self.fc2(h1)  # (B, 16) → (B, 2).
        return logits  # 분류 손실에 넣을 원래 점수를 반환한다.
```

‘16차원 은닉 표현’이라고 할 때 16은 샘플당 중간 특징 수다. `h1` 텐서 자체는 `(B, 16)`인 2차원이다. 학습으로 가중치가 바뀌면 이 중간 특징을 계산하는 방식도 바뀐다. 이런 특징을 학습과 함께 만드는 것이 **표현 학습**이다. 원래 값을 복사해서 열만 늘리는 것이 아니다.

### 12.1. 데이터 행과 가중치 행은 의미가 다르다

`nn.Linear`는 가중치를 **`(출력 뉴런 수, 입력 특징 수)`**로 저장한다. 첫 층의 `weight`는 `(16, 2)`다. 가중치 한 행은 뉴런 하나가 사용할 두 가중치를 담는다.

**전치**는 2차원 행렬의 행과 열을 바꾸는 작업이며 `.T`로 나타낸다. 실제 계산은 `X @ weight.T + bias`에 해당한다.[^linear]

<figure class="pt-figure">
<img src="/pytorch-training-basics/linear-weight.svg" alt="저장된 weight는 (16,2)이고 전치하면 (2,16)이다. 입력 (B,2)와 곱하면 (B,16)이 된다.">
<figcaption>입력의 행은 샘플이고, 저장된 weight의 행은 뉴런 하나의 가중치 묶음이다. 행이라는 위치만 같다고 같은 뜻은 아니다.</figcaption>
</figure>

`i`는 샘플 위치, `j`는 출력 뉴런 위치이며 모두 0부터 센다. `X[i, 0]`은 샘플 `i`의 첫 특징, `weight[j, 0]`은 뉴런 `j`가 그 특징에 곱할 가중치다.

```text
z1[i, j] = X[i, 0] * weight[j, 0]
         + X[i, 1] * weight[j, 1]
         + bias[j]

모양: (B, 2) @ (2, 16) + (16,) → (B, 16)
```

직접 `X @ W`를 작성한다면 `W`를 `(입력 특징 수, 출력 뉴런 수)`로 놓는다. `nn.Linear.weight`는 반대 방향으로 저장하고 전치해 계산한다. 계산 원리가 달라진 것이 아니다.

### 12.2. 층 수, 파라미터 텐서 수, 숫자 수를 나눈다

| 파라미터 | 모양 | 숫자 개수 |
| --- | --- | ---: |
| `fc1.weight` | `(16, 2)` | 32 |
| `fc1.bias` | `(16,)` | 16 |
| `fc2.weight` | `(2, 16)` | 32 |
| `fc2.bias` | `(2,)` | 2 |
| 합계 | 텐서 4개 | 82 |

학습 가능한 선형층은 2개, 파라미터 텐서는 4개, 그 안의 숫자는 82개다. 역전파 후 이 파라미터들에 저장되는 기울기도 총 82개 원소다. 샘플을 더 넣어도 파라미터 수는 그대로지만, 각 샘플이 계산에 참여하므로 연산량은 늘어난다.

ReLU가 필요한 이유도 구분하자. 선형층만 이어 붙이면 ‘곱하고 더하는 계산’을 정리해 하나의 선형층으로 표현할 수 있다. ReLU는 입력값이 음수인지에 따라 일부 값을 0으로 바꾸므로, 그런 단일 계산으로 합쳐지지 않는 **비선형 관계**를 만들 수 있다. `torch.relu`는 `forward` 안에서 부른 함수이므로 `print(model)`의 하위 층 목록에는 나타나지 않는다. 목록에 없다고 계산이 빠진 것은 아니다.

## 13. 입력과 정답을 묶은 뒤 배치로 나눈다

`make_moons`는 두 초승달 모양으로 배치된 좌표와 정답 번호를 생성하는 함수다. 반환값은 NumPy 배열 두 개다. NumPy 배열은 숫자를 모양에 맞추어 저장하는 자료형이며, 여기서는 반환값을 바로 PyTorch 텐서로 변환한다.

`n_samples`는 샘플 수, `noise`는 좌표에 더할 잡음의 크기, `random_state`는 이 함수의 난수 설정이다. 입력에는 32비트 실수 자료형인 `torch.float32`, 정답에는 앞서 사용한 정수 자료형 `torch.long`을 지정한다.

```python
from sklearn.datasets import make_moons  # 좌표와 클래스 번호를 만드는 함수다.
from torch.utils.data import TensorDataset, DataLoader  # 쌍 묶기와 배치 구성을 사용한다.

X_np, y_np = make_moons(n_samples=500, noise=0.2, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)  # 입력: (500, 2).
y = torch.tensor(y_np, dtype=torch.long)  # 정답 클래스 번호: (500,).
```

`TensorDataset(X, y)`는 같은 위치의 `X[i]`와 `y[i]`를 묶는 객체를 만든다. 두 텐서의 첫 차원, 즉 샘플 수가 같아야 한다. `DataLoader`는 그 객체에서 배치를 구성해 차례로 제공한다.[^data]

| 설정 | 의미 |
| --- | --- |
| `batch_size=64` | 배치 하나에 최대 64개 샘플을 넣는다. |
| `shuffle=True` | 매 에폭의 샘플 순서를 섞는다. 입력·정답의 짝은 유지한다. |
| `drop_last=False` | 마지막에 남은 작은 배치도 사용한다. 기본값도 False다. |

```python
dataset = TensorDataset(X, y)  # 입력과 정답의 쌍을 관리한다.
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
```

<figure class="pt-figure">
<img src="/pytorch-training-basics/batches.svg" alt="500개 샘플을 64개씩 나누면 배치 일곱 개와 52개짜리 마지막 배치가 생긴다.">
<figcaption>64 × 7 + 52 = 500. 이 글의 학습 방식에서는 한 배치마다 갱신 한 번을 수행한다.</figcaption>
</figure>

앞의 일곱 배치는 입력 `(64, 2)`와 정답 `(64,)`, 마지막 배치는 입력 `(52, 2)`와 정답 `(52,)`다. 한 배치를 `model(xb)`에 넣으면 그 배치의 샘플들을 함께 계산한다. 샘플 64개라고 모델을 64번 호출하는 것이 아니다.

여기서 **스텝**은 파라미터 갱신 한 번을 뜻한다. 한 에폭에 배치 8개이므로 갱신 8번이다. 100에폭이면 갱신 800번이다. 전체 데이터를 한 배치로 사용했던 9절은 한 에폭에 갱신 한 번이었다.

## 14. 파라미터 수정은 옵티마이저에 맡긴다

### 14.1. 객체 생성과 메서드의 역할부터 확인한다

앞에서는 `w`, `b`를 직접 수정했다. 모델의 파라미터가 많아지면 옵티마이저에 **수정할 파라미터들과 수정 규칙**을 지정한다.

`torch.optim.Adam`은 옵티마이저를 만드는 클래스다. Adam은 현재 기울기뿐 아니라 이전 기울기의 기록도 이용해 파라미터별 수정량을 정하는 방법이다. 이번 절에서는 생성과 호출 순서를 다루고, 수정량 계산식은 17절에서 분리해 설명한다.[^optim]

| 코드 | 하는 일 |
| --- | --- |
| `model.parameters()` | 이 모델의 학습 파라미터 텐서를 제공한다. |
| `Adam(..., lr=0.01)` | 그 파라미터를 수정할 Adam 객체를 만든다. 아직 수정하지 않는다. |
| `zero_grad(set_to_none=True)` | 이전 파라미터 기울기를 `None`으로 비운다. |
| `loss.backward()` | 현재 손실에 대한 기울기를 계산한다. |
| `optimizer.step()` | 저장된 기울기와 수정 규칙으로 파라미터를 갱신한다. |

기본 설정의 옵티마이저는 갱신 연산을 미분 기록에 넣지 않도록 처리하므로, 아래 반복문에서는 갱신 구간을 직접 감쌀 필요가 없다. `zero_grad`는 가중치나 Adam의 과거 기록까지 초기화하는 것이 아니라 **파라미터의 `.grad`만 비운다**.

### 14.2. 에폭 평균을 기록하는 방법을 먼저 정한다

기본 CrossEntropyLoss는 배치의 평균을 반환한다. **배치 크기가 다르면 배치 평균들을 똑같은 비중으로 평균내면 안 된다.** 현재 배치의 샘플 수를 `B`, 샘플별 손실을 `l₁`부터 `l_B`라고 하자.

```text
배치 평균 = (l₁ + ... + l_B) / B
배치 합계 = 배치 평균 × B
에폭 평균 = 모든 배치 합계의 합 / 실제 처리한 샘플 수
```

각 배치 평균을 `L₁`부터 `L₈`이라고 하면 이번 에폭의 평균은 다음과 같다.

```text
(64*L₁ + 64*L₂ + ... + 64*L₇ + 52*L₈) / 500
```

`(L₁ + ... + L₈) / 8`은 52개짜리 배치와 64개짜리 배치를 같은 비중으로 취급하므로 일반적으로 다르다. **배치 평균 → 배치 합계 → 전체 합계 → 전체 평균**으로 계산한다.

`len(xb)`는 현재 배치의 샘플 수다. `epoch_loss_sum`에는 손실 합계, `sample_count`에는 실제 처리한 샘플 수를 더해 간다. 숫자에 사용하는 `+=`는 기존 값에 오른쪽 값을 더해 다시 그 변수에 대입한다. 이 두 변수는 기록용 숫자이며 역전파에 쓰는 `loss` 텐서와 다르다.

### 14.3. 한 배치 학습을 에폭 안에서 반복한다

<figure class="pt-figure">
<img src="/pytorch-training-basics/workflow.svg" alt="현재 배치의 기울기를 비우고 예측, 손실, backward, optimizer.step을 차례로 실행한다.">
<figcaption>지금까지 살펴본 기능을 한 배치의 실행 순서로 연결했다. 정답 yb는 손실 계산에 사용한다.</figcaption>
</figure>

바깥 반복은 에폭, 안쪽 반복은 배치다. `for xb, yb in loader`는 배치의 입력과 정답을 각각 두 변수로 받는다. 전체 정답 `y`가 아니라 **현재 배치의 정답 `yb`**를 손실 계산에 넣는다.

```python
torch.manual_seed(42)  # 모델 초기값을 만드는 난수 설정이다.
model = MLP().to(device)  # 먼저 모델을 생성하고 장치로 옮긴다.
criterion = nn.CrossEntropyLoss()  # 정수 클래스 번호 방식의 평균 손실이다.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 갱신 도구를 준비한다.
epoch_losses = []  # 에폭별 손실 숫자를 저장한다.

for epoch in range(100):
    epoch_loss_sum = 0.0  # 이번 에폭의 합계를 새로 시작한다.
    sample_count = 0  # 이번 에폭에서 실제 처리한 샘플 수다.

    for xb, yb in loader:
        xb = xb.to(device)  # 현재 배치 입력: (B, 2).
        yb = yb.to(device)  # 현재 배치 정답: (B,).
        optimizer.zero_grad(set_to_none=True)  # 이전 기울기를 비운다.
        logits = model(xb)  # 클래스 점수: (B, 2).
        loss = criterion(logits, yb)  # 현재 배치의 정답과 비교한다.
        loss.backward()  # 기울기를 계산한다.
        optimizer.step()  # 파라미터를 수정한다.

        batch_size = len(xb)  # 마지막 배치도 실제 크기인 52를 사용한다.
        epoch_loss_sum += loss.item() * batch_size  # 평균을 합계로 되돌려 더한다.
        sample_count += batch_size  # 처리한 샘플 수를 더한다.

    epoch_losses.append(epoch_loss_sum / sample_count)  # 배치를 모두 처리한 뒤 한 번 저장한다.
```

손실 합계는 안쪽 반복에서 누적하고, 에폭 평균은 그 반복이 끝난 뒤 저장한다. 마지막 `append`의 들여쓰기가 배치 반복문보다 한 단계 바깥인 이유다.

이 기록은 학습 중 각 배치에서 측정한 손실을 모은 것이다. 배치마다 가중치가 수정되므로, 에폭 종료 후의 모델로 전체 데이터를 다시 평가한 손실과 다를 수 있다. `step()` 뒤에 `loss.item()`을 읽어도 이미 계산한 손실 값은 바뀌지 않는다.

<details>
<summary>추가 확인: 평균 계산의 적용 범위</summary>

위 계산은 모든 샘플을 같은 비중으로 평가하는 기본 CrossEntropyLoss 설정이다. 실제 처리한 수를 세므로 마지막 작은 배치도 반영한다.

어떤 클래스를 더 크게 평가하는 **클래스 가중치**를 지정하거나, 특정 정답 번호를 평가에서 제외하는 `ignore_index`를 지정하면 손실 평균의 분모가 달라질 수 있다. 그 설정을 사용할 때는 손실 문서의 평균 규칙부터 확인한다. 현재 예제에는 두 설정을 사용하지 않는다.[^loss]

</details>

## 15. 함수 전달을 먼저 확인한 뒤 lambda로 줄인다

### 15.1. 함수 이름을 전달하는 것과 함수를 실행하는 것은 다르다

함수도 변수에 넣고 다른 함수에 전달할 수 있는 객체다. 괄호 없는 `double`은 함수 자체이고, `double(3)`은 함수를 실행해 반환값을 받는 표현이다.

다음 `apply`는 **함수 하나와 값 하나를 받아 그 함수를 호출**한다. `operation`은 전달받은 함수를 부를 이름이며, `value`는 그 함수에 전달할 값이다.

```python
def double(p):
    return p * 2  # 함수를 호출했을 때 전달받은 p로 계산한다.

def apply(operation, value):
    return operation(value)  # 이 줄에서 실제 함수를 호출한다.

answer = apply(double, 3)  # double 함수와 숫자 3을 각각 전달한다.
print(answer)  # 6
```

```text
apply(double, 3)
    → operation은 double 함수를 받음, value는 3을 받음
    → operation(value)는 double(3)을 실행
    → double의 p에 3이 전달됨
    → 3 * 2의 결과인 6이 호출한 곳으로 반환됨
```

**함수 정의를 전달했다고 함수 본문이 즉시 실행되는 것은 아니다.** 본문은 `operation(value)`처럼 실제 호출하는 위치에서 실행된다.

### 15.2. lambda도 함수 정의다

`lambda p: p * 2`는 위의 `double`과 같은 계산을 하는 함수를 만든다. 콜론 앞의 `p`는 전달받은 값의 이름이고, 뒤의 식은 호출 시 계산하여 반환할 결과다. `lambda`는 함수를 정의하는 표현식이지, 자기 자신을 호출하는 문법이 아니다.[^python]

```python
answer = apply(lambda p: p * 2, 3)  # 만든 함수를 전달한다. apply 안에서 호출된다.
print(answer)  # 6
```

### 15.3. 같은 방식을 옵티마이저 생성에 적용한다

`torch.optim.SGD`는 기본적으로 현재 기울기에 학습률을 곱해 빼는 옵티마이저를 만든다. 다음 함수는 파라미터들을 받아 SGD 객체를 반환한다.

```python
def make_sgd(p):
    return torch.optim.SGD(p, lr=0.01)  # 받은 파라미터들을 수정할 객체를 만든다.

optimizer_fn = make_sgd  # 함수 자체를 변수에 저장한다. SGD 객체는 아직 만들지 않는다.
```

같은 함수를 `lambda p: torch.optim.SGD(p, lr=0.01)`로 쓸 수 있다. 여기서 `p`는 **함수의 매개변수**, 즉 전달받은 값을 부를 이름이다. 그 값으로 전달되는 것은 **모델의 파라미터 텐서들을 제공하는 이터레이터**다. `p`가 SGD 자신이거나 가중치 하나라는 뜻이 아니다.

<figure class="pt-figure">
<img src="/pytorch-training-basics/lambda.svg" alt="함수를 먼저 전달하고, 새 모델을 만든 뒤 model.parameters()를 인자로 호출하여 SGD 객체를 받는다.">
<figcaption>optimizer_fn(model.parameters())가 p에 값을 전달하는 호출이다. 함수 정의와 호출 시점을 구별하면 순환이나 재귀가 없음을 확인할 수 있다.</figcaption>
</figure>

모델을 공통 학습 함수 안에서 새로 만들 예정이므로, 바깥에서는 그 모델의 파라미터를 아직 전달할 수 없다. 대신 **파라미터를 나중에 받으면 옵티마이저를 만들어 줄 함수**를 전달한다. 다음 절에서 그 호출 위치를 확인하자.

## 16. 공통 학습 함수로 옵티마이저를 비교한다

`train_with`는 옵티마이저 생성 함수와 에폭 수를 받아 **새 MLP를 학습하고 손실 숫자 리스트를 반환**한다. `epochs=200`은 호출할 때 에폭 수를 생략하면 사용할 기본값이다. `_`는 반복 번호를 사용하지 않는다는 관례적인 변수 이름이다.

이번에는 500개 전체를 한 번에 사용한다. 14절의 미니배치 학습과 달리 에폭마다 갱신 한 번이다. 함수 안에서 시드를 다시 설정해 같은 환경에서 같은 초기 모델로 실험을 시작한다.

```python
X_dev = X.to(device)  # 전체 입력: (500, 2).
y_dev = y.to(device)  # 전체 정답: (500,).

def train_with(optimizer_fn, epochs=200):
    torch.manual_seed(0)  # 실험마다 초기 모델 생성 조건을 맞춘다.
    model = MLP().to(device)  # 호출마다 새 모델을 만든다.
    criterion = nn.CrossEntropyLoss()  # 손실 계산 객체를 만든다.
    optimizer = optimizer_fn(model.parameters())  # 여기서 함수의 p에 값을 전달한다.
    losses = []  # 이번 실험의 손실 기록이다.

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)  # 이전 기울기를 비운다.
        logits = model(X_dev)  # 전체 입력으로 클래스 점수를 계산한다.
        loss = criterion(logits, y_dev)  # 전체 배치의 평균 손실을 구한다.
        loss.backward()  # 기울기를 계산한다.
        optimizer.step()  # 선택한 규칙으로 파라미터를 수정한다.
        losses.append(loss.item())  # 이미 계산된 손실 숫자를 기록한다.

    return losses  # 호출한 곳으로 손실 리스트를 반환한다.
```

`results = {}`는 이름과 값을 연결하는 **딕셔너리**를 만든다. 아래에서는 실험 이름에 손실 리스트를 연결한다. 처음 호출 한 줄을 실행할 때는 **lambda 함수 생성 → train_with 호출 → 새 모델 생성 → lambda 호출 → SGD 객체 반환 → 학습 반복** 순서다.

```python
results = {}  # 실험 이름과 손실 기록을 묶는다.
results["SGD lr=0.01"] = train_with(lambda p: torch.optim.SGD(p, lr=0.01))
results["SGD lr=0.5"] = train_with(lambda p: torch.optim.SGD(p, lr=0.5))
results["SGD lr=10.0"] = train_with(lambda p: torch.optim.SGD(p, lr=10.0))
results["Adam lr=0.01"] = train_with(lambda p: torch.optim.Adam(p, lr=0.01))
```

각 값에는 손실 200개가 들어 있다. `results.items()`는 이름과 값의 쌍을 차례로 제공한다. `plt.plot(losses)`는 손실 리스트를 선으로 그린다. `label`은 선의 이름, `legend()`는 그 이름들을 보여 주는 범례다.

```python
import matplotlib.pyplot as plt  # 그래프를 그리는 기능을 plt라는 이름으로 사용한다.

plt.figure(figsize=(8, 5))  # 가로 8인치, 세로 5인치인 그림 하나를 만든다.
for name, losses in results.items():
    plt.plot(losses, label=name)  # 실험별 손실 리스트를 그린다.
plt.xlabel("epoch index")  # 기록 번호는 0부터 시작한다.
plt.ylabel("training loss")  # 훈련 데이터의 손실이다.
plt.legend()  # 선마다 실험 이름을 표시한다.
plt.show()  # 전체 표시 범위로 결과를 확인한다.
```

학습률이 작으면 같은 횟수 동안 손실이 천천히 줄 수 있고, 너무 크면 손실이 튀거나 진동할 수 있다. 그렇다고 `lr=10`이 언제나 발산하거나 Adam이 언제나 가장 좋은 것은 아니다. 데이터, 초기값, 반복 횟수에 따라 달라진다.

`plt.ylim(0, 2)`처럼 표시 범위를 제한하면 2보다 큰 손실이 화면에서 잘린다. 불안정한 학습을 확인하려면 전체 범위부터 본다. 이 그래프는 **훈련 손실**이므로, 새로운 데이터에서도 성능이 좋은지는 따로 남겨 둔 **검증 데이터**로 평가해야 한다.

## 17. 옵티마이저는 어떤 기록으로 수정량을 정하는가

학습 코드의 호출 순서와 옵티마이저 내부 계산은 구분하자. 여기서는 방금 사용한 `step()`이 수정량을 정하는 규칙을 비교한다. **기본 SGD**는 momentum 옵션을 추가하지 않은 경우를 뜻한다. PyTorch의 SGD도 momentum을 설정할 수 있다.[^optim]

| 방법 | 사용하는 정보 |
| --- | --- |
| 기본 SGD | 현재 기울기 |
| Momentum | 현재 기울기와 이전 기울기의 누적값 |
| RMSProp | 현재 기울기와 기울기 제곱의 평균 기록 |
| Adam | 기울기 평균과 기울기 제곱 평균의 두 기록 |

수식에서 `t`는 갱신 순번이고 첫 갱신은 `t=1`이다. `w_(t-1)`은 갱신 전 가중치, `w_t`는 갱신 후 가중치, `g_t`는 이번 손실에 대한 해당 가중치의 기울기다. `lr`은 학습률이다. 아래는 가중치 하나의 기본 계산이며, 다른 가중치와 편향에도 각각 적용한다.

### 17.1. 기본 SGD: 현재 기울기를 사용한다

<div class="pt-math">
$$w_t=w_{t-1}-lr\,g_t$$
</div>

현재 기울기에 학습률을 곱하고, 그 값을 현재 가중치에서 뺀다. 9절에서 직접 작성한 갱신과 같은 기본 규칙이다.

### 17.2. Momentum: 이전 기울기의 영향을 남긴다

`u_t`는 기울기를 누적한 기록이다. `μ`는 이전 기록을 얼마나 유지할지 정하는 비율이며 보통 0 이상 1 미만이다. `u_0=0`으로 시작한다.

<div class="pt-math">
$$u_t=\mu u_{t-1}+g_t$$
$$w_t=w_{t-1}-lr\,u_t$$
</div>

먼저 이전 기록의 일부와 이번 기울기를 더하고, 그 누적값으로 수정한다. 같은 부호가 이어지면 누적되고, 부호가 바뀌면 이전 값과 일부 상쇄될 수 있다. 부호만이 아니라 크기까지 누적한다.

### 17.3. RMSProp: 기울기 크기의 기록으로 조절한다

**이동평균**은 새 값을 받을 때 기존 기록과 새 값을 일정 비율로 섞어 갱신하는 평균이다. 여기서는 오래된 값의 영향이 갱신할수록 작아진다.

`v_t`는 기울기 제곱의 이동평균, `β`는 과거 기록 유지 비율이며 0 이상 1 미만이다. `ε`는 분모가 0이 되지 않도록 더하는 작은 양수다. `√`는 제곱근이다. 기울기를 제곱해서 크기를 모은 뒤 제곱근을 취해 원래 기울기와 대응하는 크기 기준으로 사용한다. `v_0=0`으로 시작한다.

<div class="pt-math">
$$v_t=\beta v_{t-1}+(1-\beta)g_t^2$$
$$w_t=w_{t-1}-lr\frac{g_t}{\sqrt{v_t}+\varepsilon}$$
</div>

첫 식은 제곱 기록을 갱신하고, 둘째 식은 그 기록을 이용해 이번 기울기의 수정량을 조절한다. 기울기 크기가 지속적으로 크면 분모도 커진다. 이 기록은 모든 가중치가 공유하지 않고 **가중치마다 따로 유지**한다.

### 17.4. Adam: 두 종류의 평균 기록을 사용한다

`m_t`는 기울기 자체의 이동평균이고, `v_t`는 기울기 제곱의 이동평균이다. `β₁`, `β₂`는 각 기록의 과거 값 유지 비율이다. 둘 다 0 이상 1 미만이며, `m_0=0`, `v_0=0`으로 시작한다.

<div class="pt-math">
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$
</div>

0에서 시작한 기록은 초기에 작아지는 영향을 받으므로 보정한다. 첫 갱신에서는 `m_0=0`이어서 `m_1=(1-β₁)g_1`이다. 이를 `1-β₁`로 나누면 초기 0 때문에 작아진 배율을 없앨 수 있다. 일반적인 갱신 순번 `t`에서는 다음 보정식을 사용한다. `β₁^t`, `β₂^t`는 각각 `t`제곱이며, 모자 표시가 붙은 `m̂`, `v̂`는 보정한 기록이다.

<div class="pt-math">
$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}$$
$$w_t=w_{t-1}-lr\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}$$
</div>

분자는 최근 기울기의 부호와 크기를 반영하고, 분모는 제곱 기록으로 수정 크기를 조절한다. Adam이 Momentum과 RMSProp의 핵심 아이디어를 함께 사용한다는 뜻이지, 두 옵티마이저를 차례로 호출한다는 뜻은 아니다.

어떤 규칙을 사용하든 **`backward()`는 기울기를 계산하고 `step()`은 그 기울기로 수정량을 정해 적용한다.** 위 수식은 추가 옵션 없는 기본 형태다.

## 덧붙임: 조건으로 값을 고르는 Python 문법

**조건부 표현식**은 조건에 따라 값 하나를 선택하는 문법이다. `참일 때 값 if 조건 else 거짓일 때 값` 순서다. 리스트 컴프리헨션은 반복과 값 계산을 묶어 새 리스트를 만드는 문법이다. 두 기능을 구분해서 보면 `if`의 위치도 이해할 수 있다.[^python]

`%`는 나눗셈의 나머지다. `x % 2 == 0`은 2로 나눈 나머지가 0인지 검사한다.

```python
numbers = [1, 2, 3, 4, 5, 6]  # 처리할 숫자들이다.

# 뒤의 if는 조건에 맞는 원소만 남긴다.
even_numbers = [x for x in numbers if x % 2 == 0]  # [2, 4, 6]

# 앞의 if else는 각 원소를 어떤 값으로 바꿀지 고른다.
parity = ["짝수" if x % 2 == 0 else "홀수" for x in numbers]
# ['홀수', '짝수', '홀수', '짝수', '홀수', '짝수']

# else 뒤에서 한 번 더 조건을 판단할 수 있다.
sizes = ["작음" if x < 3 else "보통" if x < 5 else "큼" for x in numbers]
# ['작음', '작음', '보통', '보통', '큼', '큼']
```

실행 순서는 원소 하나 가져오기 → 조건 판단 → 선택한 값을 결과에 넣기다. 조건이 길어지면 일반 `for`문과 `if/elif/else`로 나누어 쓰는 편이 읽기 쉽다.

[^shape]: PyTorch 공식 문서: [unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html), [matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html), [브로드캐스팅 규칙](https://docs.pytorch.org/docs/stable/notes/broadcasting.html).
[^random]: PyTorch 공식 문서: [재현성](https://docs.pytorch.org/docs/stable/notes/randomness.html), [linspace](https://docs.pytorch.org/docs/stable/generated/torch.linspace.html), [randn](https://docs.pytorch.org/docs/stable/generated/torch.randn.html).
[^activation]: PyTorch 공식 문서: [Sigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html), [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html).
[^loss]: PyTorch 공식 문서: [MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
[^autograd]: PyTorch 공식 문서: [자동 미분과 제자리 연산](https://docs.pytorch.org/docs/stable/notes/autograd.html).
[^assignment]: Python 공식 문서: [대입과 증강 대입](https://docs.python.org/3/reference/simple_stmts.html).
[^no_grad]: PyTorch 공식 문서: [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html). Python 공식 문서: [with 문](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement).
[^device]: PyTorch 공식 문서: [Tensor.to](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html), [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).
[^linear]: PyTorch 공식 문서: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).
[^data]: [PyTorch 데이터셋과 DataLoader](https://docs.pytorch.org/docs/stable/data.html), [scikit-learn make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).
[^python]: Python 공식 문서: [표현식](https://docs.python.org/3/reference/expressions.html), [lambda](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions).
[^optim]: PyTorch 공식 문서: [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html), [RMSprop](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html), [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).
