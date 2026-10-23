+++
title = "PyTorch 첫 학습: 텐서 모양부터 역전파와 옵티마이저까지"
description = "입력과 정답의 모양을 맞추는 일부터 직접 가중치를 수정하는 회귀, 미니배치 분류까지 연결한다. 브로드캐스팅, nn.Linear의 가중치 방향, 배치 손실, lambda 함수 전달을 그림과 실행 코드로 살펴본다."
date = 2026-10-23T16:00:00+09:00
slug = "pytorch-training-basics"

[extra]
katex = true
toc = true
styles = ["pytorch-training-basics/article.css"]
+++

입력 숫자 하나를 받아 다른 숫자를 예측하는 모델을 만든다고 하자. 데이터마다 예측값 하나를 만들고, 그 데이터의 정답과 비교한 뒤, 예측에 사용한 가중치와 편향을 수정해야 한다. 코드가 오류 없이 실행되는 것만으로는 충분하지 않다. 예측 100개와 정답 100개를 잘못 배치하면, 서로 짝이 아닌 값까지 비교하는 계산이 실행될 수 있다.

이 글에서는 **직접 만든 회귀 데이터**로 텐서의 모양과 학습 한 번을 확인한 다음, **두 초승달 모양의 분류 데이터**를 작은 신경망으로 학습한다. 두 데이터 모두 계산을 살펴보기 위한 합성 데이터다. 실제 서비스의 예측 성능을 측정한 자료는 아니다.

먼저 전체 흐름을 보고, 코드가 각 단계에서 무엇을 만드는지 따라가 보자. 예제에는 `torch`, `scikit-learn`, `matplotlib`이 설치된 Python 환경이 필요하다.

## 1. 학습은 무엇을 계산하고 무엇을 바꾸는가

**샘플**은 데이터 한 건이고, **특징**은 샘플을 설명하는 입력 항목이다. **파라미터**는 학습 과정에서 수정하는 숫자다. 여기서는 입력에 곱하는 **가중치**와 계산 결과에 더하는 **편향**이 파라미터다.

| 이름 | 역할 | 이번 학습에서 바꾸는 대상인가 |
| --- | --- | --- |
| 입력 | 모델이 예측에 사용하는 데이터다. | 아니다. |
| 정답 | 예측과 비교할 기준 데이터다. | 아니다. |
| 예측 | 현재 파라미터로 계산한 결과다. | 파라미터가 달라지면 새로 계산한 결과도 달라진다. |
| 손실 | 예측이 정답과 얼마나 맞지 않는지 나타내는 값이다. | 직접 대입해 줄이지 않는다. |
| 기울기 | 특정 파라미터를 조금 바꿀 때 손실이 얼마나 변하는지 나타내는 값이다. | 역전파로 계산한다. |
| 옵티마이저 | 기울기와 갱신 규칙으로 파라미터를 수정하는 객체다. | 가중치와 편향을 수정한다. |

**순전파**는 입력에서 예측과 손실을 계산하는 방향이고, **역전파**는 손실에서 계산 관계를 거슬러 올라가 기울기를 구하는 과정이다. 역전파 자체가 가중치를 수정하는 것은 아니다.

<figure class="pt-figure">
<img src="/pytorch-training-basics/workflow.svg" alt="배치마다 이전 기울기를 비우고 예측, 손실, 역전파, 파라미터 수정을 순서대로 실행한다. 정답은 손실 계산에 들어간다.">
<figcaption>한 배치의 학습 흐름. backward는 기울기를 계산하고, step은 파라미터를 수정한다. 다음 배치는 수정된 파라미터를 사용한다.</figcaption>
</figure>

함께 처리하는 데이터 묶음을 **배치**, 전체 학습 데이터를 한 차례 처리하는 단위를 **에폭**이라고 한다. 이 글의 기본 학습 반복문은 배치 하나마다 역전파와 파라미터 수정을 각각 한 번 실행한다. 데이터 수, 모델 호출 수, 파라미터 수는 서로 다른 개수다.

## 2. 숫자 개수와 텐서 차원 수를 구분한다

**텐서(Tensor)**는 숫자를 일정한 구조로 저장하고 계산하는 PyTorch 자료형이다. `shape`는 각 차원의 크기, `ndim`은 차원의 개수, `numel()`은 전체 원소 개수를 뜻한다. `dtype`은 각 원소의 자료형이다.

아래에서 `a`, `b`, `c`, `d`는 각각 숫자 하나를 나타낸다.

| 숫자를 배치한 구조 | shape | 차원 수 | 원소 개수 |
| --- | --- | ---: | ---: |
| `a` | `()` | 0 | 1 |
| `[a]` | `(1,)` | 1 | 1 |
| `[[a]]` | `(1, 1)` | 2 | 1 |
| `[[a, b], [c, d]]` | `(2, 2)` | 2 | 4 |

`(1,)`의 쉼표는 원소 하나짜리 튜플을 나타내는 Python 문법이다. 숫자가 하나라고 반드시 0차원 텐서인 것은 아니다.

```python
import torch  # 텐서와 자동 미분 기능을 사용한다.
from torch import nn  # 모델 층과 손실 함수 클래스를 사용한다.

w = torch.randn(1, requires_grad=True)  # 원소 1개를 담은 1차원 학습 텐서다.
print(w.shape, w.ndim, w.numel())  # torch.Size([1]) 1 1
```

`requires_grad=True`는 이후 계산에서 이 텐서에 대한 기울기를 구하도록 설정한다. 텐서의 모양을 바꾸거나 기울기를 즉시 계산하는 설정은 아니다.

이제 입력 `X`의 모양이 `(500, 2)`라고 하자. 이 글에서는 **행 하나가 샘플 하나**, **열 하나가 특징 하나**다. 따라서 샘플 500개에 특징이 각각 2개 있다. `X` 자체는 2차원 텐서다.

반면 모델 설명에서 ‘입력 2차원’이라고 하면 특징 두 개를 뜻할 수도 있다. 특징 두 개짜리 샘플 하나를 배치 없이 쓰면 `(2,)`, 여러 샘플을 모으면 `(N, 2)`다. `N`은 샘플 수다. **특징 수와 텐서의 차원 수를 같은 뜻으로 읽지 말자.**

## 3. 회귀 데이터를 만들면서 unsqueeze를 이해한다

**회귀**는 수량처럼 숫자 값을 예측하는 문제다. 여기서는 입력 `x`에 대해 `2*x + 1`에 작은 잡음을 더한 값을 정답으로 만든다. 모델에는 정답을 만드는 계수를 알려 주지 않고, 입력과 정답의 쌍을 사용해 가중치와 편향을 학습시킨다.

먼저 사용할 함수를 구분하자.

| 코드 | 하는 일 | 이 예제의 반환 모양 |
| --- | --- | --- |
| `torch.manual_seed(42)` | PyTorch 난수 생성기의 초기 상태를 설정한다. | 아래 난수 실험의 재현을 위한 설정이다. |
| `torch.linspace(0, 1, 100)` | 양 끝값을 포함해 같은 간격의 숫자 100개를 만든다. | `(100,)` |
| `x.unsqueeze(1)` | 1번 위치에 크기가 1인 차원을 추가한다. | `(100,) → (100, 1)` |
| `torch.randn(100, 1)` | 표준정규분포에서 난수를 뽑는다. | `(100, 1)` |

차원 번호는 0부터 시작한다. `unsqueeze(1)`의 `1`은 추가할 숫자의 개수가 아니라 **새 차원이 들어갈 위치**다. 원소의 값과 개수는 바뀌지 않는다.[^shape]

<figure class="pt-figure">
<img src="/pytorch-training-basics/unsqueeze.svg" alt="길이 N인 1차원 텐서에 unsqueeze 0을 적용하면 1행 N열, unsqueeze 1을 적용하면 N행 1열이 된다.">
<figcaption>이 회귀 문제에는 샘플마다 특징 하나가 필요하다. 따라서 N개의 값을 한 행에 놓는 (1, N)이 아니라 N행 1열인 (N, 1)을 사용한다.</figcaption>
</figure>

```python
torch.manual_seed(42)  # 같은 환경에서 난수를 같은 순서로 만들기 위한 설정이다.

x = torch.linspace(0, 1, 100)  # 입력 숫자 100개. 간격은 1/99다.
x2 = x.unsqueeze(1)  # 샘플 100개 × 특징 1개인 (100, 1)로 표현한다.
noise = 0.05 * torch.randn(100, 1)  # 각 샘플의 정답에 더할 잡음이다.
y2 = 2 * x2 + 1 + noise  # 학습에 사용할 정답. 예측값이 아니다.
```

표준정규분포는 평균이 0이고 표준편차가 1인 분포다. **표준편차**는 값이 평균 주변에 얼마나 퍼져 있는지 나타내는 수치다. 난수에 `0.05`를 곱하면 잡음 분포의 표준편차가 `0.05`가 된다. 모든 잡음이 `-0.05`부터 `0.05` 사이에 있다는 뜻은 아니다. 실제로 뽑은 100개의 평균도 정확히 0일 필요는 없다.

`manual_seed`는 난수 생성에 관한 설정이므로 일정한 간격의 값을 만드는 `linspace`에는 영향을 주지 않는다. 같은 시드라도 PyTorch 버전·장치·연산 방식까지 달라지면 완전히 같은 결과를 보장하지는 않는다.[^random]

`unsqueeze`가 반환한 텐서는 원본과 숫자가 저장된 메모리를 공유한다. 이런 결과를 **뷰(view)**라고 한다. `x.unsqueeze(1)`만 실행하고 반환값을 사용하지 않으면 기존 변수 `x`는 여전히 `(100,)`이다. 위 코드에서는 반환값에 `x2`라는 이름을 붙였다.

## 4. 계산이 된다는 것과 원하는 계산이라는 것은 다르다

이 회귀 모델의 예측은 ‘입력에 가중치를 곱하고 편향을 더한 값’이다. 샘플마다 숫자 하나를 예측하므로 **원하는 예측 모양은 정답과 같은 `(100, 1)`**이다.

`*`는 대응하는 원소를 곱하고, `@`는 텐서 차원에 맞는 행렬곱 규칙을 적용한다. 두 기호를 모양만 보고 바꾸어 쓰면 안 된다.

```python
w = torch.randn(1, requires_grad=True)  # 가중치 원소 1개, shape (1,).
b = torch.zeros(1, requires_grad=True)  # 편향 원소 1개, shape (1,).

wrong_pred = x2 @ w + b  # (100, 1) @ (1,) → (100,). 계산 자체는 가능하다.
wrong_error = wrong_pred - y2  # (100,) - (100, 1) → (100, 100).
print(wrong_error.shape)  # torch.Size([100, 100])
```

2차원 행렬과 1차원 벡터를 `@`로 곱하면 이 경우 결과는 1차원이다. `(k, m) @ (m, n) → (k, n)`이라는 **2차원끼리의 규칙만** 적용하면 이 결과를 놓치게 된다.[^shape]

그다음 뺄셈에는 **브로드캐스팅**이 적용된다. 모양이 다른 텐서의 차원 크기를 오른쪽부터 비교하고, 크기가 같거나 둘 중 하나가 1이면 맞추어 계산한다. 부족한 앞쪽 차원은 크기가 1인 것처럼 취급한다. 데이터를 실제로 복사한다는 뜻은 아니다.

예측 `(100,)`은 뺄셈에서 `(1, 100)`처럼 취급된다. 정답 `(100, 1)`의 각 행이 예측 100개 전체와 비교되므로 결과가 `(100, 100)`이 된다. 필요한 것은 같은 샘플끼리의 오차 100개인데, 실제로는 모든 조합 10,000개를 계산한 것이다.

<figure class="pt-figure">
<img src="/pytorch-training-basics/broadcasting.svg" alt="예측 p1 p2 p3을 한 행에, 정답 t1 t2 t3을 한 열에 놓으면 모든 조합의 오차 3행 3열을 만든다. 둘 다 한 열이면 같은 행끼리의 오차 세 개만 만든다.">
<figcaption>그림은 세 샘플만 표시한다. 위쪽은 모든 조합을 비교하고, 아래쪽은 같은 샘플의 예측과 정답을 일대일로 비교한다.</figcaption>
</figure>

가중치를 `(1, 1)`로 보아 계산하면 원하는 출력 모양을 유지할 수 있다.

```python
y_hat = x2 @ w.unsqueeze(1) + b  # (100, 1) @ (1, 1) → (100, 1).
assert y_hat.shape == y2.shape  # 다르면 실행을 멈춰 잘못된 손실 계산을 막는다.

same_pred = x2 * w + b  # 입력 특징과 가중치가 하나인 이 문제에서는 같은 계산이다.
assert torch.allclose(y_hat, same_pred)  # 작은 실수 오차를 허용해 값을 비교한다.
```

`assert`는 조건이 거짓이면 오류를 발생시키는 검사문이고, `allclose`는 작은 실수 계산 오차를 허용하며 값을 비교하는 함수다. 특징이 여러 개일 때는 `*`와 `@`가 일반적으로 같은 계산이 아니다. 또한 모양이 같아도 샘플 순서가 어긋나면 잘못된 비교다. **모양과 각 원소의 대응을 함께 확인해야 한다.**

## 5. Sigmoid, ReLU, Softmax는 무엇을 바꾸는가

**활성화 함수**는 신경망의 중간값 등에 적용해 값을 변환하는 함수다. 아래 세 함수는 입력과 출력의 모양을 유지하지만, 값을 변환하는 단위가 다르다.[^activation]

| 함수 | 계산 단위 | 결과 |
| --- | --- | --- |
| Sigmoid | 원소 하나씩 독립적으로 계산한다. | 각 값을 0과 1 사이로 변환한다. 전체 합이 1일 필요는 없다. |
| ReLU | 원소 하나씩 독립적으로 계산한다. | 음수는 0, 나머지는 그대로 둔다. 1보다 큰 결과도 가능하다. |
| Softmax | 지정한 차원의 값들을 함께 사용한다. | 여기서는 샘플별 클래스 점수를 합이 1인 확률들로 변환한다. |

```python
z = torch.tensor([[-2.0, 0.0, 3.0], [1.0, -4.0, 2.0]])  # shape (2, 3).

sigmoid_manual = 1 / (1 + torch.exp(-z))  # 원소별 Sigmoid 계산식이다.
sigmoid_builtin = torch.sigmoid(z)  # 실제 사용할 수 있는 내장 함수다.

relu_manual = torch.where(z > 0, z, 0.0)  # 조건이 참인 위치는 z, 아니면 0이다.
relu_builtin = torch.relu(z)  # [[0, 0, 3], [1, 0, 2]]를 반환한다.
```

`exp(z)`는 수학 상수 `e`의 `z`제곱이다. `e`는 약 `2.71828`이다. `torch.exp`는 텐서의 각 원소에 이 계산을 적용한다. `torch.where(조건, 참일 때 값, 거짓일 때 값)`는 각 위치에서 사용할 값을 고른다.

수학적으로 Sigmoid는 유한한 입력을 0보다 크고 1보다 작은 값으로 바꾼다. 컴퓨터의 제한된 정밀도에서는 극단적인 입력의 결과가 0 또는 1로 표현될 수 있다. 직접 쓴 식의 `exp`도 너무 커질 수 있으므로 실제 모델에는 내장 함수를 사용한다. `torch.clamp(z, min=0)`도 음수를 0으로 제한하지만, 직접 구성한 식과 내장 ReLU의 미분 규칙까지 언제나 같다고 일반화하지는 않는다.

### Softmax는 같은 샘플의 클래스 점수끼리 계산한다

**클래스**는 분류할 범주이고, **로짓(logits)**은 확률로 바꾸기 전의 클래스별 점수다. 로짓은 음수여도 된다. `logits2`의 각 행은 샘플 하나, 각 열은 클래스 하나의 점수다.

```python
logits2 = torch.tensor([[2.0, 0.5, 0.3], [0.2, 2.5, 0.1]])  # (2, 3).

row_max = logits2.amax(dim=1, keepdim=True)  # 행마다 최댓값 하나: (2, 1).
shifted = logits2 - row_max  # 같은 행의 점수에서 같은 최댓값을 뺀다: (2, 3).
exp_logits = torch.exp(shifted)  # 원소별 지수 계산: (2, 3).
row_sums = exp_logits.sum(dim=1, keepdim=True)  # 행마다 합계 하나: (2, 1).
probs_manual = exp_logits / row_sums  # 각 행을 자기 행의 합계로 나눈다: (2, 3).

probs = torch.softmax(logits2, dim=1)  # 같은 계산을 내장 함수로 수행한다.
assert torch.allclose(probs_manual, probs)  # 두 결과가 같은지 확인한다.
```

`amax`는 최댓값을 구하는 메서드다. `dim=1`은 이 표에서 열로 놓인 클래스 점수들을 대상으로 계산한다는 뜻이다. `keepdim=True`는 계산한 차원을 없애지 않고 크기 1로 남긴다. 따라서 `(2, 3)`의 열 크기가 3에서 1로 줄어 `(2, 1)`이 된다.

<figure class="pt-figure">
<img src="/pytorch-training-basics/softmax.svg" alt="2행 3열 지수값의 각 행 합계를 2행 1열로 보관하고 원래 행을 각각 자기 행의 합계로 나눈다.">
<figcaption>a부터 f까지는 지수 계산을 마친 값이다. 첫 행은 s1, 둘째 행은 s2로 나눈다. 서로 다른 샘플의 점수를 한꺼번에 합하지 않는다.</figcaption>
</figure>

행의 최댓값을 먼저 빼면 지수 계산에 큰 양수가 들어가는 문제를 줄일 수 있다. 같은 행에서 동일한 값을 빼므로 Softmax 확률은 변하지 않는다. 현재처럼 `(샘플 수, 클래스 수)`이면 `dim=1`이고, 마지막 차원을 뜻하는 `dim=-1`도 같다. 클래스 점수만 있는 1차원 텐서에서는 `dim=0` 또는 `dim=-1`을 사용한다.

## 6. 손실 함수는 만들 때와 계산할 때가 다르다

`nn.MSELoss`와 `nn.CrossEntropyLoss`는 **클래스**, 즉 객체를 만들기 위한 정의다. 클래스 이름 뒤의 괄호로 손실 계산 객체를 만들고, 그 객체를 예측과 정답으로 호출해 실제 손실을 구한다.

```python
loss_fn = nn.MSELoss()  # 손실 계산 객체를 만든다. 아직 예측과 정답을 비교하지 않는다.
pred = torch.tensor([2.5, 0.0, 2.0])  # 예측값 세 개다.
target = torch.tensor([3.0, -0.5, 2.0])  # 각각에 대응하는 정답이다.

loss = loss_fn(pred, target)  # 만든 객체를 호출해 손실 텐서를 얻는다.
manual_loss = (pred - target).square().mean()  # 같은 계산을 직접 작성한다.
assert torch.allclose(loss, manual_loss)  # 결과는 약 0.1667이다.
```

**MSE(평균제곱오차)**는 대응하는 값끼리 오차를 구하고, 제곱한 뒤, 전체 원소의 평균을 구한다. `.square()`는 각 원소를 제곱하는 메서드다. 이름은 `.squared()`가 아니다.

```text
예측 - 정답   [2.5-3.0, 0.0-(-0.5), 2.0-2.0]
오차          [-0.5, 0.5, 0.0]
제곱          [0.25, 0.25, 0.0]
평균          (0.25 + 0.25 + 0.0) / 3 = 0.166666...
```

`nn.MSELoss(pred, target)`는 객체를 만드는 자리에 데이터를 넣은 것이므로 이 용법이 아니다. `nn.MSELoss()(pred, target)`처럼 생성과 호출을 연달아 쓸 수 있지만, 처음에는 두 줄로 나누는 편이 구분하기 쉽다.

기본 MSE의 평균은 **모든 출력 원소의 평균**이다. 샘플마다 출력이 여러 개라면 샘플 수만으로 나누는 계산과 다르다.[^loss]

### 클래스 번호로 정답을 주는 CrossEntropyLoss

분류에서는 샘플마다 클래스별 점수와 정답 클래스 번호를 비교한다. `N`은 샘플 수, `C`는 클래스 수다.

| 대상 | 모양 | 자료형과 내용 |
| --- | --- | --- |
| `logits` | `(N, C)` | 실수형 클래스 점수다. Softmax를 미리 적용하지 않는다. |
| `labels` | `(N,)` | `torch.long` 정수다. 값은 `0`부터 `C-1`까지다. |
| 기본 평균 손실 | `()` | 숫자 하나를 담은 0차원 텐서다. |

```python
labels = torch.tensor([0, 1], dtype=torch.long)  # 첫 샘플은 클래스 0, 둘째는 클래스 1.
ce_fn = nn.CrossEntropyLoss()  # 분류 손실 계산 객체를 만든다.
ce = ce_fn(logits2, labels)  # 앞 절의 원래 점수를 넣는다. probs가 아니다.
print(ce)  # tensor(0.2577)
```

이 설정에서는 정답 클래스에 부여한 확률이 낮을수록 손실이 커진다. 정수 라벨 방식의 계산은 `LogSoftmax`와 `NLLLoss`를 연결한 것과 동등하며, 정답 클래스 확률의 음의 로그를 사용하는 계산이다. 기본 설정은 샘플별 손실의 평균을 반환한다.[^loss]

`labels.long()`은 자료형을 `torch.int64`로 변환한다. 이미 `torch.long`이면 다시 변환할 필요가 없다. CrossEntropyLoss가 정수 라벨만 지원하는 것은 아니다. 클래스별 확률을 정답으로 받는 방식도 있지만, 이 글은 `(N,)` 정수 라벨 방식을 사용한다.

## 7. backward는 계산하고, no_grad 안의 갱신은 값을 바꾼다

자동 미분은 순전파 중 기록한 계산 관계를 사용한다. 가중치와 편향으로 예측을 만들고, 예측으로 손실을 만들었으므로, 손실이 각 파라미터에 얼마나 의존하는지 그 관계를 거슬러 계산할 수 있다. 중간값이 뒤의 결과에 미친 영향을 연결하는 미분 규칙을 **연쇄법칙**이라고 한다.[^autograd]

```text
순전파:    w, b  →  예측  →  손실 L
역전파:    손실 L에서 출발  →  예측 계산의 관계  →  w.grad, b.grad
수정:      계산한 기울기를 사용해 기존 w와 b의 값을 바꾼다.
```

| 코드 | 하는 일 | 하지 않는 일 |
| --- | --- | --- |
| `loss.backward()` | 파라미터 기울기를 계산해 `.grad`에 누적한다. | 파라미터 값을 갱신하지 않는다. |
| `w.grad` | 현재 손실을 `w`의 각 원소에 대해 미분한 값을 담는다. | 최적 가중치까지의 거리가 아니다. |
| `with torch.no_grad():` | 블록 안 연산을 역전파용 기록에 남기지 않는다. | 이미 저장된 `.grad`를 지우지 않는다. |
| `w.grad = None` | 저장된 기울기를 비운다. | `w`의 값을 0으로 만들지 않는다. |

손실이 숫자 하나여도 가중치마다 손실에 미치는 영향이 다르다. 따라서 가중치마다 기울기가 생긴다. `w.grad`의 모양은 `w`와 같다.

직접 생성한 `w`, `b`처럼 다른 미분 계산의 결과가 아닌 시작 텐서를 **리프 텐서**라고 한다. `requires_grad=True`인 리프 텐서에는 기본적으로 `.grad`가 저장된다. `w.unsqueeze(1)`은 원본 `w`와 연결된 중간 결과이므로, 그것을 사용해 예측해도 기울기는 원본 `w`까지 전달된다. 학습할 변수 자체를 뷰로 덮어쓰기보다 계산하는 자리에서 뷰를 사용하면 이 구분을 유지하기 쉽다.

새 순전파 뒤에 `backward()`를 반복하면 기울기가 기존 `.grad`에 더해진다. 이번 실습처럼 배치별 기울기로 각각 갱신하려면 배치 사이에서 기울기를 비워야 한다.

## 8. w -=와 w =-는 전혀 다른 코드다

Python 변수는 객체를 가리키는 이름이다. 같은 숫자가 들어 있는 텐서 두 개와, 이름 두 개가 같은 텐서를 가리키는 상황은 다르다.

| 코드 | PyTorch 텐서에서의 의미 |
| --- | --- |
| `w = w - lr * w.grad` | 계산 결과인 새 텐서에 `w`라는 이름을 붙인다. |
| `w -= lr * w.grad` | 기존 `w` 텐서의 값을 직접 수정한다. |
| `w =- lr * w.grad` | `w = -(lr * w.grad)`다. 기존 가중치를 뺄셈의 시작값으로 사용하지 않는다. |

`-=`는 뺄셈 대입 연산자다. `=-`는 그런 연산자가 아니라 대입 기호 `=` 다음에 음수 부호 `-`가 온 것이다.

또한 `no_grad` 안에서 `w = w - ...`를 실행하면 뺄셈 결과는 기울기를 추적하지 않는 새 텐서다. 그 텐서로 기존 학습 변수를 교체하면 다음 학습의 연결이 달라진다. 반면 같은 위치에서 `w -= ...`를 실행하면 기존 객체의 `requires_grad=True` 설정을 유지하면서 값만 수정한다.

```python
original = torch.tensor([2.0], requires_grad=True)  # 학습할 텐서 객체다.
same_object = original  # 복사하지 않고 같은 객체를 가리킨다.

with torch.no_grad():
    original -= 0.1  # 기존 텐서의 값만 수정한다.

assert original is same_object  # is는 같은 객체인지 검사한다.
assert original.requires_grad  # 기울기 추적 설정은 유지된다.

with torch.no_grad():
    replacement = original - 0.1  # 뺄셈 결과인 새 텐서를 만든다.

assert replacement is not original  # 기존 텐서와 다른 객체다.
assert not replacement.requires_grad  # no_grad 안에서 만든 연산 결과다.
```

이 설명은 **역전파가 끝난 뒤 파라미터를 갱신하는 상황**에 관한 것이다. 순전파 중 필요한 중간값을 임의로 덮어쓰면 역전파에 문제가 생길 수 있다. 또한 `-=`의 동작은 자료형에 달려 있다. Python의 일반 숫자에도 텐서와 같은 객체 수정 규칙을 적용하면 안 된다.[^assignment]

## 9. 직접 가중치를 수정하는 회귀 학습

아래는 앞 절의 회귀 과정을 한 번에 실행할 수 있는 코드다. **학습률** `lr`은 기울기를 이용한 수정량의 전체 크기를 조절한다. 기본 갱신은 ‘현재 가중치에서 학습률과 기울기의 곱을 뺀다’이다.

```python
import torch  # 텐서와 자동 미분을 사용한다.

torch.manual_seed(42)  # 난수 생성 순서를 재현하기 위한 설정이다.
x2 = torch.linspace(0, 1, 100).unsqueeze(1)  # 입력: (100, 1).
y2 = 2 * x2 + 1 + 0.05 * torch.randn(100, 1)  # 정답: (100, 1).

w = torch.randn(1, requires_grad=True)  # 모든 샘플이 함께 사용하는 가중치다.
b = torch.zeros(1, requires_grad=True)  # 모든 샘플이 함께 사용하는 편향이다.
lr = 0.3  # 이번 예제에서 사용할 학습률이다.
losses = []  # 손실 기록용 Python 리스트다.

for step in range(500):
    y_hat = x2 @ w.unsqueeze(1) + b  # 순전파: (100, 1).
    assert y_hat.shape == y2.shape  # 같은 샘플끼리 비교할 모양인지 검사한다.
    loss = (y_hat - y2).square().mean()  # 손실: shape ()인 텐서다.
    losses.append(loss.item())  # 그래프 기록을 위한 Python 숫자를 보관한다.

    loss.backward()  # 현재 손실에 대한 w와 b의 기울기를 계산한다.
    with torch.no_grad():
        w -= lr * w.grad  # 기존 가중치의 값을 수정한다.
        b -= lr * b.grad  # 기존 편향의 값을 수정한다.

    w.grad = None  # 다음 반복에 이전 기울기가 누적되지 않게 비운다.
    b.grad = None  # 편향 기울기도 비운다.

print(f"w={w.item():.3f}, b={b.item():.3f}")  # 학습된 파라미터를 확인한다.
```

`.item()`은 원소 하나짜리 텐서에서 Python 숫자를 꺼내는 메서드다. 숫자를 기록하는 데는 적합하지만, 그 Python 숫자로 역전파할 수는 없다. 역전파에는 원래 `loss` 텐서를 사용한다.

여기서는 100개 전체를 매번 사용하므로 반복 한 번이 전체 배치 학습 한 번이다. 가중치는 하나, 편향도 하나지만 100개 샘플 모두가 손실과 기울기 계산에 기여한다. **파라미터 개수가 계산량을 뜻하는 것은 아니다.**

잡음을 더한 정답이므로 학습된 값이 정확히 `w=2`, `b=1`일 필요는 없다. 기울기는 최적점까지의 거리를 알려 주지 않으므로 학습률을 지나치게 크게 잡으면 손실이 오히려 커질 수 있다.

## 10. CPU, CUDA, MPS에서 무엇을 같은 곳에 두어야 할까

**device**는 텐서를 저장하고 계산할 장치다. `cpu`는 CPU, 이 예제의 `cuda`는 NVIDIA GPU를 사용하는 경로, `mps`는 지원되는 Mac의 Apple GPU를 사용하는 경로다. `is_available()`은 현재 사용할 수 있는지를 참·거짓으로 반환한다.[^device]

```python
if torch.cuda.is_available():
    device = "cuda"  # 사용할 수 있는 CUDA 장치를 선택한다.
elif torch.backends.mps.is_available():
    device = "mps"  # CUDA가 없고 MPS를 사용할 수 있으면 선택한다.
else:
    device = "cpu"  # GPU 경로를 사용할 수 없으면 CPU에서 실행한다.

t = torch.randn(2, 2)  # 기본 장치에 텐서를 만든다.
t_on_device = t.to(device)  # 지정한 장치의 텐서를 반환받는다.
print(t_on_device.device)  # 반환된 텐서의 실제 장치를 확인한다.
```

`Tensor.to(device)`는 반환값을 제공하며, 기존 변수 이름을 자동으로 교체하지 않는다. 이미 장치와 자료형이 같으면 원래 텐서를 반환할 수도 있다. 위 코드에서는 장치만 바꾸므로 모양과 값의 의미는 유지된다.

모델의 `.to(device)`는 등록된 파라미터 등을 모델 내부에서 변환하는 메서드다. 텐서의 `.to()`와 완전히 같은 동작으로 생각하지 말자. 학습 준비 순서는 **모델 생성 → 장치 이동 → 그 모델의 파라미터로 옵티마이저 생성**이다. 실제 계산에서는 입력뿐 아니라 손실 계산에 쓰는 정답도 같은 장치로 옮긴다.

## 11. nn.Module과 nn.Linear는 학습을 대신하는가

`nn.Module`은 모델과 층을 만들 때 사용하는 기본 클래스다. `__init__`은 객체를 만들 때 층과 파라미터를 준비하고, `forward`에는 입력에서 출력을 계산하는 순서를 작성한다. `self`는 현재 객체를 가리킨다. `super().__init__()`은 `nn.Module`의 기본 초기화를 실행한다.

```python
from torch import nn  # 모델을 구성하는 기본 클래스를 가져온다.

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()  # 하위 층과 파라미터를 관리할 기반을 초기화한다.
        self.linear = nn.Linear(1, 1)  # 입력 특징 1개 → 출력 값 1개.

    def forward(self, x):
        return self.linear(x)  # 준비된 층으로 예측값을 계산한다.

model = LinearRegression()  # 모델 객체 생성. 아직 학습하지 않았다.
prediction = model(x2)  # 순전파 실행. 이 호출만으로 파라미터를 갱신하지 않는다.

for name, p in model.named_parameters():
    print(name, tuple(p.shape), type(p).__name__)  # 이름·모양·자료형을 확인한다.
```

이 모델의 파라미터는 `linear.weight`의 `(1, 1)` 텐서와 `linear.bias`의 `(1,)` 텐서다. `nn.Linear(입력 특징 수, 출력 뉴런 수)`의 두 인자는 정수다. `(1,)` 같은 모양 튜플을 넣지 않는다.[^linear]

`nn.Linear`는 가중치와 편향을 만들고 **Parameter**로 관리한다. Parameter는 모델의 학습 파라미터로 등록되는 특별한 텐서다. 층을 `self.linear`에 연결하면 모델이 그 층의 파라미터도 찾을 수 있다. 일반 텐서를 속성에 넣었다고 모두 학습 파라미터가 되는 것은 아니다.

`named_parameters()`는 이름과 파라미터 텐서를 함께 제공하고, `parameters()`는 파라미터 텐서만 차례로 제공한다.

| 직접 작성한 코드 | 모델과 옵티마이저를 사용하는 방식 |
| --- | --- |
| 직접 만든 `w`, `b` | 층이 관리하는 `weight`, `bias` |
| `(pred - target).square().mean()` | `nn.MSELoss()`로 만든 객체 호출 |
| `w -= lr * w.grad` | 기본 SGD 옵티마이저의 `step()` |
| `w.grad = None` | `zero_grad(set_to_none=True)` |

## 12. 2 → 16 → 2 모델의 가중치는 왜 (16, 2)일까

이제 입력 특징 두 개로 클래스 0 또는 1을 분류하는 모델로 바꾼다. **MLP**는 여러 선형층과 활성화 함수를 연결한 신경망이다. 여기서는 은닉층 하나와 출력층 하나를 사용한다. ‘2층’은 파라미터가 있는 선형층이 두 개라는 뜻이고, ‘16’은 은닉층 개수가 아니라 그 층의 뉴런 수다.

**뉴런**은 입력 특징들에 각각 가중치를 곱하고 더한 뒤 편향을 더하는 계산 단위다. 첫 층의 뉴런 16개는 같은 입력 특징 두 개를 받아 각자 다른 가중치와 편향으로 값 하나씩 계산한다.

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 모델의 기본 관리를 초기화한다.
        self.fc1 = nn.Linear(2, 16)  # 특징 2개 → 은닉 값 16개.
        self.fc2 = nn.Linear(16, 2)  # 은닉 값 16개 → 클래스 점수 2개.

    def forward(self, x):
        z1 = self.fc1(x)  # (B, 2) → (B, 16). B는 현재 샘플 수다.
        h1 = torch.relu(z1)  # (B, 16) → (B, 16). 음수를 0으로 바꾼다.
        logits = self.fc2(h1)  # (B, 16) → (B, 2). 원래 점수를 반환한다.
        return logits
```

`z1`은 첫 선형층의 출력이고, `h1`은 여기에 ReLU를 적용한 결과다. `h1`을 ‘16차원 은닉 표현’이라고 해도 현재 텐서 자체는 `(B, 16)`인 2차원이다. 열 16개는 원래 특징을 여러 방식으로 계산한 새로운 특징이다. 학습되는 가중치와 함께 이런 특징을 만들어 가는 것을 **표현 학습**이라고 한다. 값을 복사해서 열만 늘리는 작업이 아니다.

`nn.Linear`의 가중치는 **`(출력 뉴런 수, 입력 특징 수)` 순서로 저장**된다. 따라서 첫 층은 `(16, 2)`다. 가중치의 한 행에 뉴런 하나가 사용할 가중치들이 들어 있다.

<figure class="pt-figure">
<img src="/pytorch-training-basics/linear-weight.svg" alt="입력 X는 B행 2열, 저장된 weight는 16행 2열이다. weight를 전치한 2행 16열과 X를 곱하고 bias를 더하면 B행 16열이 된다.">
<figcaption>입력 행은 샘플 하나를 나타내지만, 저장된 weight의 행은 출력 뉴런 하나의 가중치 묶음을 나타낸다. 같은 ‘행’이라도 텐서마다 의미가 다르다.</figcaption>
</figure>

계산은 `X @ weight.T + bias`에 해당한다. `.T`는 2차원 행렬의 행과 열을 바꾸는 **전치**다.[^linear]

```text
X:            (B, 2)
weight:       (16, 2)
weight.T:     (2, 16)
bias:         (16,)
결과:         (B, 2) @ (2, 16) + (16,) → (B, 16)

샘플 i와 뉴런 j의 계산:
z1[i, j] = X[i, 0] * weight[j, 0]
         + X[i, 1] * weight[j, 1]
         + bias[j]
```

직접 `X @ W`로 계산하려면 `W`를 `(입력 특징 수, 출력 뉴런 수)`로 놓는다. `nn.Linear.weight`는 반대 방향으로 저장한 뒤 전치해 사용한다. 수학이 달라진 것이 아니라 저장 방향이 다르다.

| 파라미터 텐서 | 모양 | 숫자 개수 |
| --- | --- | ---: |
| `fc1.weight` | `(16, 2)` | 32 |
| `fc1.bias` | `(16,)` | 16 |
| `fc2.weight` | `(2, 16)` | 32 |
| `fc2.bias` | `(2,)` | 2 |
| 합계 | 텐서 4개 | 82 |

역전파 후 이 파라미터들에 저장되는 기울기도 텐서 4개, 원소 총 82개다. 층이 두 개라서 기울기가 두 개인 것이 아니다. 중간값의 기울기도 역전파에 사용되지만 기본적으로 전부 `.grad`에 보관하지는 않는다. 배치가 커져도 파라미터 수는 그대로이며, 더 많은 샘플이 계산에 참여하므로 연산량은 늘어난다.

ReLU 없이 선형층만 이어 붙이면 하나의 선형층 계산으로 합칠 수 있다. 중간에 ReLU를 넣으면 입력에 따라 값이 꺾이는 비선형 관계를 표현할 수 있다. `torch.relu`는 `forward`에서 호출한 함수이므로 `print(model)`의 등록된 하위 모듈 목록에는 표시되지 않는다. 목록에 없다고 계산에서 빠진 것은 아니다.

## 13. TensorDataset과 DataLoader가 나누는 것

분류 데이터는 `make_moons`로 만든다. 이 함수는 두 초승달 모양으로 배치된 좌표와 정답 클래스 번호를 NumPy 배열로 반환한다. 아래에서는 PyTorch 텐서로 변환한다. `noise`와 `random_state`는 이번 합성 데이터의 설정이며, 값이 다르면 학습 결과도 달라질 수 있다.

```python
from sklearn.datasets import make_moons  # 합성 분류 데이터를 만드는 함수다.
from torch.utils.data import TensorDataset, DataLoader  # 짝 묶기와 배치 구성을 사용한다.

X_np, y_np = make_moons(n_samples=500, noise=0.2, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)  # (500, 2). 각 행은 좌표 두 개다.
y = torch.tensor(y_np, dtype=torch.long)  # (500,). 각 샘플의 클래스 0 또는 1이다.

dataset = TensorDataset(X, y)  # 같은 위치의 X[i]와 y[i]를 한 쌍으로 묶는다.
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
```

`TensorDataset`에 전달한 텐서들은 첫 차원 크기, 즉 샘플 수가 같아야 한다. `dataset[i]`는 `(X[i], y[i])`를 제공한다. `DataLoader`에는 사용할 `dataset`을 전달해야 한다.[^data]

`shuffle=True`는 에폭을 순회할 때 샘플 순서를 섞되 입력과 정답의 쌍은 유지한다. `drop_last=False`는 마지막 작은 배치도 사용한다는 뜻이며 기본값이다.

<figure class="pt-figure">
<img src="/pytorch-training-basics/batches.svg" alt="500개 샘플을 배치 크기 64로 나누면 64개짜리 배치 일곱 개와 52개짜리 배치 하나가 된다. 모두 처리하면 한 에폭이다.">
<figcaption>64 × 7 + 52 = 500. 이 학습 방식에서는 배치 하나마다 갱신 한 번, 에폭 하나마다 갱신 여덟 번이다.</figcaption>
</figure>

앞의 일곱 배치는 입력 `xb`가 `(64, 2)`, 정답 `yb`가 `(64,)`다. 마지막 배치는 각각 `(52, 2)`, `(52,)`다. 배치에 샘플이 64개라고 `model(xb)`를 64번 호출하는 것이 아니다. 한 번 호출해 샘플 64개의 결과를 함께 얻는다.

이 글에서 **스텝**은 `optimizer.step()`을 한 번 실행한 단위다. 100에폭이면 갱신 800회다. 전체 500개를 한 번에 넣는 전체 배치 방식에서는 에폭마다 갱신 한 번이다. 반복 변수 이름이 `step`인지 `epoch`인지보다 실제 처리 범위를 확인해야 한다.

## 14. 미니배치 학습과 에폭 손실의 계산

이제 앞에서 정의한 `MLP`, `device`, `dataset`, `loader`를 연결한다. **모델을 먼저 장치로 옮기고**, 그 모델의 파라미터로 옵티마이저를 만든다.

```python
torch.manual_seed(42)  # 모델 초기값 생성에 사용할 난수 상태를 정한다.
model = MLP().to(device)  # 모델을 생성하고 계산 장치로 옮긴다.
criterion = nn.CrossEntropyLoss()  # 클래스 번호 방식의 기본 평균 손실이다.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 갱신 대상과 규칙을 정한다.
epoch_losses = []  # 에폭마다 기록할 평균 손실 숫자들을 모은다.

for epoch in range(100):
    epoch_loss_sum = 0.0  # 이번 에폭의 기록용 손실 합계다.
    sample_count = 0  # 실제로 처리한 샘플 수다.

    for xb, yb in loader:
        xb = xb.to(device)  # 현재 배치의 입력을 옮긴다.
        yb = yb.to(device)  # 현재 배치의 정답도 옮긴다.
        optimizer.zero_grad(set_to_none=True)  # 이전 배치의 기울기를 비운다.

        logits = model(xb)  # 현재 배치만 예측한다: (B, 2).
        loss = criterion(logits, yb)  # 전체 정답 y가 아닌 현재 정답 yb와 비교한다.
        loss.backward()  # 현재 손실의 파라미터별 기울기를 계산한다.
        optimizer.step()  # 해당 기울기와 Adam의 규칙으로 파라미터를 수정한다.

        batch_size = len(xb)  # 마지막 배치는 64가 아니라 52다.
        epoch_loss_sum += loss.item() * batch_size  # 배치 평균을 합계로 바꾸어 더한다.
        sample_count += batch_size  # 실제 처리한 개수를 센다.

    epoch_losses.append(epoch_loss_sum / sample_count)  # 모든 배치 처리 후 한 번 기록한다.
```

`zero_grad`가 비우는 것은 파라미터의 기울기다. 이미 학습한 가중치나 Adam이 따로 보관하는 과거 기울기 기록까지 초기화하는 것은 아니다. `step()`은 앞서 직접 작성한 갱신 작업에 해당하지만, **Adam의 계산식은 기본 SGD의 계산식과 다르다.**

### 이미 평균인데 왜 곱했다가 다시 나누는가

현재 설정은 클래스 가중치와 무시할 라벨이 없는 기본 CrossEntropyLoss다. 배치의 샘플 수를 `B`, 각 샘플 손실을 `l₁`부터 `l_B`라고 하자.

```text
현재 배치 평균 loss = (l₁ + ... + l_B) / B
현재 배치 손실 합계 = loss × B
에폭 손실 합계      = 각 배치 손실 합계를 모두 더한 값
에폭 평균 손실      = 에폭 손실 합계 / 실제 처리한 샘플 수
```

배치 평균을 `L₁`부터 `L₈`이라고 하면 이번 에폭의 올바른 평균은 다음과 같다.

```text
(64*L₁ + 64*L₂ + ... + 64*L₇ + 52*L₈) / 500
```

`(L₁ + ... + L₈) / 8`은 샘플 52개짜리 배치와 64개짜리 배치를 같은 비중으로 취급하므로 일반적으로 같은 결과가 아니다. 이미 평균인 값을 무작정 또 나누는 것이 아니라 **배치 평균 → 배치 합계 → 전체 합계 → 전체 평균**으로 계산하는 것이다.

`loss`는 역전파에 사용하는 텐서다. `epoch_loss_sum`과 `epoch_losses`는 관찰과 기록용 Python 숫자이며 파라미터 갱신에 사용하지 않는다. 들여쓰기도 중요하다. 배치 합계는 안쪽 반복문에서 누적하고, 에폭 평균은 안쪽 반복문이 끝난 뒤 한 번 저장한다.

이 기록은 배치마다 서로 다른 시점의 파라미터로 계산한 손실을 모은 것이다. 에폭이 끝난 모델로 전체 데이터를 다시 평가한 손실과는 다를 수 있다. `step()` 뒤에 `loss.item()`을 호출하더라도 이미 계산한 손실 값이 새 파라미터의 손실로 바뀌지는 않는다.

위 코드는 실제 처리 개수를 센다. 현재처럼 전체 데이터를 한 번씩 사용하면 `len(dataset)`과 같지만, 일부 데이터를 버리거나 중복 추출하면 달라질 수 있다. 클래스 가중치나 `ignore_index`를 사용하면 손실 평균의 분모도 따로 확인해야 한다.[^loss]

## 15. lambda p의 p에는 누가 값을 넣는가

옵티마이저를 바꾸어 비교할 때마다 학습 반복문 전체를 복사할 필요는 없다. 공통 학습 함수는 그대로 두고 **옵티마이저를 만드는 함수**를 전달할 수 있다.

```python
def make_sgd(p):
    return torch.optim.SGD(p, lr=0.01)  # 전달받은 모델 파라미터로 SGD 객체를 만든다.

optimizer_fn = make_sgd  # 괄호 없이 함수를 전달한다. 아직 SGD 객체를 만들지 않는다.
```

다음 `lambda`도 같은 역할이다. **lambda는 이름 없이 함수를 정의하는 표현식**이다. 콜론 앞의 `p`는 매개변수이고, 콜론 뒤는 함수를 호출했을 때 계산해 반환할 식이다.[^python]

```python
optimizer_fn = lambda p: torch.optim.SGD(p, lr=0.01)  # 함수를 만든다. 본문은 아직 실행하지 않는다.
```

함수의 매개변수 `p`와 모델의 학습 파라미터는 서로 다른 의미다. `p`는 전달받은 값을 부를 이름이고, 모델의 파라미터는 실제 가중치와 편향 텐서다.

<figure class="pt-figure">
<img src="/pytorch-training-basics/lambda.svg" alt="lambda로 만든 함수가 train_with의 optimizer_fn으로 전달되고, train_with 안에서 model.parameters를 인자로 호출할 때 p가 값을 받는다. 이후 SGD 객체를 만들어 반환한다.">
<figcaption>p는 SGD 함수 자신이 아니다. optimizer_fn(model.parameters())라는 호출이 p에 실제 값을 전달한다. 자기 자신을 다시 호출하는 재귀 과정은 없다.</figcaption>
</figure>

`model.parameters()`는 파라미터 텐서를 차례로 제공하는 **이터레이터**를 반환한다. 이 MLP에서는 텐서 네 개를 제공한다. `p`는 가중치 하나를 뜻하지 않는다.

실행 순서를 풀면 다음과 같다.

```text
1. lambda 식을 평가해 함수 객체를 만든다.
2. train_with에 그 함수를 전달한다.
3. train_with 안에서 새 모델을 만든다.
4. model.parameters()로 그 모델의 파라미터를 제공할 객체를 얻는다.
5. optimizer_fn(그 객체)를 호출한다. 이때 p가 그 객체를 받는다.
6. 함수 본문 torch.optim.SGD(p, lr=0.01)을 실행한다.
7. 생성한 SGD 객체를 반환하여 optimizer에 저장한다.
```

모델이 `train_with` 안에서 생성되기 때문에, 바깥에서 다른 모델의 옵티마이저를 미리 만드는 대신 ‘새 모델을 받으면 옵티마이저를 만들어 주는 함수’를 전달하는 것이다.

## 16. 같은 초기 모델에서 옵티마이저를 바꾸어 본다

이번에는 500개 전체를 한 번에 사용하는 전체 배치 학습이다. 앞서 만든 `X`, `y`, `MLP`, `device`를 사용한다. `train_with`는 호출할 때마다 새 모델을 만들고 손실 숫자들의 리스트를 반환한다. 같은 환경에서 같은 초기값으로 설정을 비교하기 위해 함수 안에서 시드를 다시 설정한다.

```python
X_dev = X.to(device)  # 전체 입력을 한 번에 사용할 장치로 옮긴다.
y_dev = y.to(device)  # 전체 정답도 옮긴다.

def train_with(optimizer_fn, epochs=200):
    torch.manual_seed(0)  # 각 실험의 모델 초기값 생성 조건을 맞춘다.
    model = MLP().to(device)  # 호출마다 새 모델을 만든다.
    criterion = nn.CrossEntropyLoss()  # 클래스 점수와 정답 번호를 비교한다.
    optimizer = optimizer_fn(model.parameters())  # 이 호출이 lambda의 p에 값을 전달한다.
    losses = []  # 이번 실험의 손실 기록이다.

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)  # 이전 기울기를 비운다.
        logits = model(X_dev)  # 전체 500개의 클래스 점수를 계산한다.
        loss = criterion(logits, y_dev)  # 이번 전체 배치의 평균 손실이다.
        loss.backward()  # 기울기를 계산한다.
        optimizer.step()  # 선택한 옵티마이저로 파라미터를 수정한다.
        losses.append(loss.item())  # 갱신 전에 계산한 손실을 기록한다.

    return losses  # 에폭 수만큼의 Python 숫자를 반환한다.

results = {}  # 실험 이름과 손실 리스트를 연결할 딕셔너리다.
results["SGD lr=0.01"] = train_with(lambda p: torch.optim.SGD(p, lr=0.01))
results["SGD lr=0.5"] = train_with(lambda p: torch.optim.SGD(p, lr=0.5))
results["SGD lr=10.0"] = train_with(lambda p: torch.optim.SGD(p, lr=10.0))
results["Adam lr=0.01"] = train_with(lambda p: torch.optim.Adam(p, lr=0.01))
```

`_`는 반복 번호를 사용하지 않겠다는 관례적인 변수 이름이다. `results`의 각 값에는 손실 200개가 들어 있다. 전체 배치 방식이므로 여기서는 한 에폭에 갱신 한 번이다.

```python
import matplotlib.pyplot as plt  # 손실 변화를 그린다.

plt.figure(figsize=(8, 5))  # 그래프 하나를 만든다.
for name, losses in results.items():
    plt.plot(losses, label=name)  # 각 실험의 손실 기록을 그린다.
plt.xlabel("epoch")  # 가로축은 에폭 기록 순서다.
plt.ylabel("loss")  # 세로축은 훈련 손실이다.
plt.legend()  # 각 선의 실험 이름을 표시한다.
plt.show()  # 세로축을 임의로 잘라내지 않고 먼저 전체 결과를 본다.
```

학습률이 작으면 같은 횟수 안에서 손실이 천천히 줄어들 수 있다. 너무 크면 손실이 튀거나 진동하거나 발산할 수 있다. 하지만 `lr=10`이면 반드시 발산한다거나 Adam이 항상 가장 좋다고 단정할 수는 없다. 초기값, 데이터, 반복 횟수 등이 결과에 영향을 준다.

`plt.ylim(0, 2)`처럼 표시 범위를 제한하면 2보다 큰 손실은 화면에서 사라진다. 큰 학습률의 불안정성을 관찰할 때는 먼저 전체 범위를 살펴보자. 또한 이 그래프는 **훈련 데이터의 손실**이다. 새 데이터에서도 잘 맞는지 판단하려면 별도의 검증 데이터가 필요하다.

## 17. SGD, Momentum, RMSProp, Adam의 차이

모든 방법은 기울기로 파라미터를 수정한다. 차이는 **이번 수정량을 만들 때 어떤 기록을 사용하는가**에 있다. 아래는 가중치 감쇠 등 추가 옵션이 없는 기본 형태다. 여기서 ‘기본 SGD’는 momentum을 지정하지 않은 경우다. PyTorch의 SGD에 momentum을 설정할 수도 있다.[^optim]

| 방법 | 수정량을 정할 때 사용하는 정보 |
| --- | --- |
| 기본 SGD | 현재 기울기 |
| Momentum | 현재 기울기와 이전 기울기의 누적값 |
| RMSProp | 현재 기울기와 최근 기울기 제곱의 기록 |
| Adam | 기울기의 평균 기록과 기울기 제곱의 평균 기록, 초기값 보정 |

### 수식에 사용할 기호

`t`는 파라미터를 수정하는 순번이며 첫 수정은 `t=1`이다. `w_(t-1)`은 수정 전의 특정 가중치, `g_t`는 현재 배치 손실을 그 가중치에 대해 미분한 값이다. `lr`은 학습률이다. 아래 식은 가중치 하나에 대한 계산이며, 다른 가중치와 편향에도 각각 적용한다.

**기본 SGD**는 현재 기울기에 학습률을 곱해 뺀다.

<div class="pt-math">
$$w_t = w_{t-1} - lr\,g_t$$
</div>

**Momentum**의 `u_t`는 이전 기울기까지 누적한 값이고, `μ`는 이전 누적값을 유지하는 비율이다. 보통 0 이상 1 미만으로 둔다. `u_0=0`에서 시작하는 기본 형태는 다음과 같다.

<div class="pt-math">
$$u_t = \mu u_{t-1} + g_t,\qquad w_t = w_{t-1} - lr\,u_t$$
</div>

같은 부호의 기울기가 이어지면 누적될 수 있고, 부호가 바뀌면 이전 값과 일부 상쇄될 수 있다. 부호만 보관하는 것이 아니라 부호와 크기를 가진 기울기 값을 누적한다.

**이동평균**은 새 값이 들어올 때 기존 기록과 새 값을 섞어 갱신하는 평균이다. 여기서는 오래된 값의 영향이 갱신을 거듭할수록 작아진다. **RMSProp**은 기울기 제곱의 이동평균 `v_t`를 가중치마다 따로 관리한다. `β`는 과거 기록을 유지하는 비율이며 0 이상 1 미만이다. `ε`는 0으로 나누는 문제 등을 줄이는 작은 양수다. `v_0=0`으로 시작한다.

<div class="pt-math">
$$v_t=\beta v_{t-1}+(1-\beta)g_t^2$$
$$w_t=w_{t-1}-lr\frac{g_t}{\sqrt{v_t}+\varepsilon}$$
</div>

제곱근을 뜻하는 `√`가 분모에 있다. 기울기가 지속적으로 큰 가중치는 제곱의 기록도 커져 수정량이 조절된다. 모든 가중치가 하나의 기록을 함께 사용하는 것이 아니다.

**Adam**은 기울기 자체의 이동평균 `m_t`와 제곱의 이동평균 `v_t`를 함께 사용한다. `β₁`, `β₂`는 각 기록의 과거 값 유지 비율이고, 두 기록은 0에서 시작한다.

<div class="pt-math">
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$
</div>

0에서 시작한 기록은 초기에 작아지는 영향을 받는다. `β₁^t`, `β₂^t`는 각각 `t`제곱이며, 다음 보정에 사용한다.

<div class="pt-math">
$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}$$
$$w_t=w_{t-1}-lr\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}$$
</div>

분자는 최근 기울기의 부호와 크기를 반영하고, 분모는 최근 기울기 크기의 기록으로 수정량을 조절한다. Adam은 Momentum과 RMSProp의 핵심 아이디어를 함께 사용하지만 두 옵티마이저를 차례로 실행하는 알고리즘은 아니다. 어떤 방법을 쓰든 `backward()`가 계산하는 기울기와 `step()`이 적용하는 수정량은 구분해야 한다.

## 덧붙임: 조건으로 값을 고르는 Python 문법

장치를 선택할 때 사용하는 `"cuda" if 조건 else "cpu"`는 **조건부 표현식**이다. 조건에 따라 값 하나를 고른다. 리스트 컴프리헨션 안에서도 같은 문법을 사용할 수 있다. 리스트 컴프리헨션은 반복과 값 계산을 묶어 새 리스트를 만드는 문법이다.[^python]

```python
numbers = [1, 2, 3, 4, 5, 6]  # 처리할 숫자들이다.

# 뒤의 if는 원소를 걸러낸다. %는 나머지를 계산한다.
even_numbers = [x for x in numbers if x % 2 == 0]  # [2, 4, 6]

# 앞의 if else는 각 원소를 어떤 값으로 바꿀지 고른다.
parity = ["짝수" if x % 2 == 0 else "홀수" for x in numbers]
# ['홀수', '짝수', '홀수', '짝수', '홀수', '짝수']

# else 뒤에 또 조건부 표현식을 연결할 수도 있다.
sizes = ["작음" if x < 3 else "보통" if x < 5 else "큼" for x in numbers]
# ['작음', '작음', '보통', '보통', '큼', '큼']
```

원소 하나를 가져오고, 조건을 판단하고, 선택한 값을 결과에 추가한다. 조건이 복잡해지면 일반 `for`문과 `if/elif/else`로 나누는 편이 읽기 쉽다.

[^shape]: PyTorch 공식 문서: [unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html), [matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html), [브로드캐스팅 규칙](https://docs.pytorch.org/docs/stable/notes/broadcasting.html).
[^random]: PyTorch 공식 문서: [재현성](https://docs.pytorch.org/docs/stable/notes/randomness.html), [linspace](https://docs.pytorch.org/docs/stable/generated/torch.linspace.html), [randn](https://docs.pytorch.org/docs/stable/generated/torch.randn.html).
[^activation]: PyTorch 공식 문서: [Sigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html), [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html).
[^loss]: PyTorch 공식 문서: [MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
[^autograd]: PyTorch 공식 문서: [자동 미분](https://docs.pytorch.org/docs/stable/notes/autograd.html), [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html).
[^assignment]: Python 공식 문서: [대입과 증강 대입](https://docs.python.org/3/reference/simple_stmts.html). 텐서의 자동 미분 관련 제한은 PyTorch의 [제자리 연산 설명](https://docs.pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd)을 함께 참고한다.
[^device]: PyTorch 공식 문서: [Tensor.to](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html), [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).
[^linear]: PyTorch 공식 문서: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).
[^data]: PyTorch 공식 문서: [TensorDataset과 DataLoader](https://docs.pytorch.org/docs/stable/data.html).
[^python]: Python 공식 문서: [조건부 표현식과 컴프리헨션](https://docs.python.org/3/reference/expressions.html), [lambda 표현식](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions).
[^optim]: PyTorch 공식 문서: [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html), [RMSprop](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html), [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).
