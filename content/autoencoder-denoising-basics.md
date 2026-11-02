+++
title = "오토인코더와 잡음 제거: 잠재 표현에서 복원·학습까지"
description = "입력 하나가 잠재 표현과 복원값으로 바뀌는 과정을 작은 계산부터 따라간다. 샘플과 특징, MSE, 역전파, 정보 소실을 구분하고, PyTorch로 오토인코더와 잡음 제거 모델을 학습·평가한다."
date = 2026-11-02
slug = "autoencoder-denoising-basics"

[extra]
katex = true
toc = true
styles = ["autoencoder-denoising-basics/article.css"]
+++

이미지에서 숫자 종류를 맞히는 모델은 이미지 한 장을 받아 “3” 같은 분류 결과를 낸다. 이번에는 이미지 한 장을 받아 그 이미지의 픽셀값들을 다시 만드는 모델을 살펴보자. <strong>오토인코더(Autoencoder, AE)</strong>는 입력에서 중간 표현을 계산하고, 그 표현을 이용해 입력을 재구성하도록 학습하는 신경망이다.

원본을 이미 가지고 있는데 왜 다시 만들까? 중간에 사용할 값의 개수를 제한하거나 입력에 잡음을 추가하면, 모델은 그 조건에서 원본을 잘 복원하는 방법을 배워야 한다. 이때 계산한 중간 표현은 특징 추출에, 복원 결과는 잡음 제거 등에 사용할 수 있다. 다만 재구성이 잘된다는 사실만으로 모든 작업에 유용한 특징을 배웠다고 단정할 수는 없다.[^ae]

먼저 특징값 두 개를 잠재값 하나로 바꾸고 다시 두 개로 복원하는 작은 모델을 계산한다. 이어서 양쪽 가중치가 어떻게 학습되는지 확인한 뒤, 이미지의 잡음을 줄이는 모델로 연결한다. 작은 수치 모델, 두 가중치 조건의 비교, MNIST 배열 설명, digits 학습은 각각 별개의 예제다.

[전체 Python 예제](/autoencoder-denoising-basics/examples-ko.py)와 [Rust 순전파 예제](/autoencoder-denoising-basics/forward-ko.rs)를 내려받을 수 있다. 본문에는 설명에 필요한 코드를 순서대로 싣고, 다운로드 파일에는 실행 순서를 묶어 두었다.

## 1. 입력에서 잠재 표현을 거쳐 복원값을 만든다

먼저 그림에 나오는 이름을 구분하자.

| 이름 | 의미 | 무엇을 받거나 만드는가? |
| --- | --- | --- |
| 인코더(Encoder) | 입력을 중간 표현으로 바꾸는 신경망 부분 | 입력 `x`를 받아 `z`를 계산한다. |
| 잠재 표현(Latent Representation) | 인코더가 계산한 중간 데이터 | 샘플마다 계산되는 숫자들의 묶음이다. |
| 디코더(Decoder) | 중간 표현에서 원본 형태의 값을 만드는 신경망 부분 | `z`를 받아 복원값 `x_hat`을 계산한다. |
| 재구성(Reconstruction) | 입력의 형태로 다시 계산한 결과 | `x_hat` 또는 수식의 `x̂`로 적는다. |
| 타깃(Target) | 출력과 비교할 목표 데이터 | 기본 AE에서는 원본 `x`다. |
| 손실(Loss) | 출력과 타깃의 차이를 측정한 숫자 | 이 글에서는 평균제곱오차를 사용한다. |

그림에서 `N`은 한 번에 처리하는 샘플 수, `F`는 샘플 하나의 입력 특징 수, `K`는 샘플 하나의 잠재 특징 수다. 괄호는 실제 데이터 값이 아니라 각 축의 크기를 나타내는 <strong>shape</strong>다. `θe`, `θd`는 각각 인코더와 디코더의 매개변수들을 묶어 가리키는 기호다.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/overview-ko.svg" alt="기본 AE는 원본 입력을 인코더, 잠재 표현, 디코더로 전달하여 재구성하고 같은 원본과 비교한다. 입력과 출력은 (N,F), 잠재 표현은 (N,K)다. DAE는 모델 입력에 잡음을 추가하고 타깃은 깨끗한 원본으로 유지한다." loading="lazy">
<figcaption>모델을 통과하는 경로와 손실의 비교 기준으로 들어가는 경로를 함께 보자. 기본 AE에서는 같은 원본이 입력과 타깃 두 역할을 맡는다.</figcaption>
</figure>

<strong>가중치(Weight)</strong>는 입력값에 곱하는 학습 대상 숫자다. <strong>편향(Bias)</strong>은 가중합에 더하는 학습 대상 숫자다. 둘을 포함해 학습으로 조정하는 값을 <strong>매개변수(Parameter)</strong>라고 한다. 여러 샘플은 같은 모델의 매개변수를 사용한다.

반면 `z`는 특정 샘플을 넣어서 얻은 계산 결과다. 일반적인 AE에서는 각 샘플의 `z`를 독립적인 모델 매개변수로 등록해 직접 수정하지 않는다. 인코더의 가중치가 바뀌면 다음 계산에서 만들어지는 `z`가 달라진다.

| 신경망을 읽는 기준 | 이 글의 작은 모델 |
| --- | --- |
| 입력 | 샘플 하나에 특징값 두 개가 있다. |
| 뉴런·구조 | 인코더의 출력값 하나가 디코더의 두 출력 계산에 연결된다. |
| 중간값 | 잠재값 `z` 하나가 계산된다. |
| 가중치 | 인코더 두 개, 디코더 두 개를 사용한다. |
| 최종 출력 | 원본의 두 특징에 대응하는 복원값 두 개가 나온다. |

인코더와 디코더를 함께 학습하는 이유도 이 연결에서 찾을 수 있다. 디코더만 바꾸면 주어진 `z`를 복원값으로 바꾸는 방법만 조정할 수 있다. 인코더까지 바꾸면 디코더가 사용할 `z`를 만드는 방법도 조정할 수 있다. 하나의 최종 재구성 손실을 이용해 두 부분을 함께 학습한다.

### 정답 레이블이 없는데 무엇과 비교할까?

손글씨 분류에서는 이미지와 숫자 종류 레이블을 연결한다. 기본 AE에서는 이미지 자체를 타깃으로 사용한다. 그래서 별도의 분류 레이블 없이도 손실을 계산할 수 있다.

기본 AE는 전통적으로 <strong>비지도학습(Unsupervised Learning)</strong>의 표현 학습 방법으로 다룬다. 데이터 자체에서 학습 목표를 마련한다는 점에서는 <strong>자기지도학습(Self-supervised Learning)</strong>으로도 설명할 수 있다. 손실 함수가 있다는 사실만으로 사람이 붙인 정답 레이블을 사용하는 학습이라고 판단하면 안 된다.

## 2. 샘플 수와 특징값 수를 먼저 나눈다

<strong>샘플(Sample)</strong>은 데이터 사례 하나이고, <strong>특징(Feature)</strong>은 그 샘플을 표현하는 입력 항목이다. `B=[2,1]`이라는 샘플이 있다면, 이름이 B인 샘플 하나에 첫 특징값 2와 둘째 특징값 1이 들어 있다. 여기서 B는 배치 크기 기호가 아니다. 이 글에서는 배치 크기를 `N`으로 적는다.

<strong>배치(Batch)</strong>는 한 번의 모델 호출에서 함께 처리하는 샘플 묶음이다. <strong>텐서(Tensor)</strong>는 PyTorch가 숫자와 숫자 배열을 표현하는 자료형이다. <strong>축(Dimension, Axis)</strong>은 값의 위치를 구분하는 기준이며, shape는 축마다 값이 몇 개 있는지를 나타낸다.

| 표현 | 실제 값 | shape | 해석 |
| --- | --- | --- | --- |
| 배치 축 없는 샘플 B | `[2,1]` | `(2,)` | 특징값 두 개 |
| 배치 축을 붙인 샘플 B | `[[2,1]]` | `(1,2)` | 샘플 한 개, 샘플당 특징 두 개 |
| 샘플 A와 B의 배치 | `[[1,2],[2,1]]` | `(2,2)` | 샘플 두 개, 샘플당 특징 두 개 |

실제 데이터 `[2,1]`과 shape `(2,1)`은 뜻이 다르다. 전자는 저장된 값이 2와 1이라는 뜻이다. 후자는 첫 축의 크기가 2, 둘째 축의 크기가 1이라는 뜻이다.

배치 `X=[[1,2],[2,1]]`의 축과 인덱스는 다음과 같다. <strong>인덱스(Index)</strong>는 위치 번호이며 0부터 시작한다.

| 축 번호 | 의미 | 축의 크기 | 유효 인덱스 | 선택 예 |
| --- | --- | ---: | --- | --- |
| 0 | 샘플 | 2 | 0, 1 | `X[0]`은 A, `X[1]`은 B |
| 1 | 샘플 안의 특징 | 2 | 0, 1 | `X[1,0]`은 B의 첫 특징값 2 |

`2 → 1 → 2` 구조는 <strong>샘플당 특징값 두 개를 잠재값 하나로 바꾼 뒤 복원값 두 개를 만든다</strong>는 뜻이다. 샘플 두 개를 하나로 합친다는 뜻이 아니다.

| 단계 | 입력 | 잠재 표현 | 복원 |
| --- | --- | --- | --- |
| 배치 shape | `(N,2)` | `(N,1)` | `(N,2)` |
| 샘플 수 | `N` | `N` | `N` |
| 샘플당 값의 수 | 2 | 1 | 2 |

이 모델은 각 샘플의 특징을 변환하도록 정의했기 때문에 샘플 수를 유지한다. 개발자가 샘플 축까지 합산하는 연산을 직접 쓰면 그 축이 사라질 수 있다. 따라서 코드에서 어느 축을 계산하는지 확인해야 한다.

## 3. 특징값 두 개가 복원값 두 개로 돌아오는 과정

다음 계산에서는 <strong>편향과 활성화 함수를 사용하지 않는다.</strong> 활성화 함수(Activation Function)는 가중합에 적용하는 함수이며, 뒤의 이미지 모델에서는 ReLU와 Sigmoid를 사용한다. 지금은 가중치를 곱하고 더하는 연결 자체를 확인한다.

샘플은 `A=[1,2]`, 인코더 가중치는 `e=[0.5,0.5]`, 디코더 가중치는 `d=[0.8,1.2]`다. 학습 대상 숫자는 `e₁, e₂, d₁, d₂` 네 개다. 그림에서는 복원값 `x_hat₁`, `x_hat₂`를 짧게 `r₁`, `r₂`라고도 적었다.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/tiny-forward-ko.svg" alt="샘플 A의 두 특징 1과 2에 인코더 가중치 0.5를 각각 곱해 합하면 z=1.5다. 디코더는 같은 z에 0.8과 1.2를 각각 곱하여 1.2와 1.8을 만든다. 원본과 비교한 MSE는 0.04다." loading="lazy">
<figcaption>인코더에서는 두 곱셈 결과를 합산한다. 디코더에서는 두 곱셈 결과를 서로 더하지 않고 복원 벡터의 각 성분으로 둔다.</figcaption>
</figure>

### 인코더: 무엇을 곱하고 무엇을 합할까?

구하려는 값은 잠재값 `z`다. `x₁, x₂`는 입력의 두 특징값이고, `e₁, e₂`는 각 특징에 곱할 인코더 가중치다. <strong>가중합(Weighted Sum)</strong>은 각각의 입력에 해당 가중치를 곱한 뒤 더한 값이다.

<div class="ae-math">
$$z=e_1x_1+e_2x_2$$
</div>

| 연결 | 입력값 | 인코더 가중치 | 곱셈 결과 |
| --- | ---: | ---: | ---: |
| 첫 특징 → 잠재값 | `x₁=1` | `e₁=0.5` | 0.5 |
| 둘째 특징 → 잠재값 | `x₂=2` | `e₂=0.5` | 1.0 |

두 결과를 더하면 `z=0.5+1.0=1.5`다. 입력의 두 특징을 사용해 잠재값 하나를 계산했다.

### 디코더: 잠재값 하나로 어떻게 두 값을 만들까?

구하려는 값은 원본의 두 특징에 대응하는 `x_hat₁, x_hat₂`다. 디코더가 받는 입력은 원본 `[1,2]`가 아니라 방금 계산한 `z=1.5`다.

<div class="ae-math">
$$\hat{x}_1=d_1z,\qquad \hat{x}_2=d_2z$$
</div>

| 출력 | 사용하는 잠재값 | 디코더 가중치 | 복원값 |
| --- | ---: | ---: | ---: |
| 첫 특징의 복원 | 1.5 | `d₁=0.8` | `0.8×1.5=1.2` |
| 둘째 특징의 복원 | 1.5 | `d₂=1.2` | `1.2×1.5=1.8` |

따라서 복원 벡터는 `[1.2,1.8]`이다. <strong>성분(Component)</strong>은 벡터를 구성하는 개별 값을 말한다. `reconstruction`이라는 변수 이름은 하나지만, 그 안에는 복원 성분 두 개가 들어 있다.

### 같은 계산을 Tensor로 옮긴다

`torch.tensor()`는 데이터로 텐서를 만드는 함수다. `dtype=torch.float64`는 각 값을 64비트 부동소수점으로 표현하겠다는 설정이다. <strong>부동소수점(Floating Point)</strong>은 유한한 비트로 실수를 근사하는 방식이므로, 계산 결과를 출력하면 작은 소수점 오차가 보일 수 있다.

```python
import torch

def tiny_ae(x, e, d):
    # 이 함수는 배치가 없는 샘플 하나만 받는다.
    # x, e, d는 각각 값 두 개를 가진 shape=(2,) 텐서다.
    if x.shape != (2,) or e.shape != (2,) or d.shape != (2,):
        raise ValueError("x, e, d의 shape는 각각 (2,)여야 합니다.")

    # 원소별 곱 [e1*x1, e2*x2]을 합하여 잠재값 하나를 만든다.
    z = (e * x).sum()                  # shape=(): 스칼라 텐서.

    # 동일한 z에 d1과 d2를 각각 곱한다. 두 출력을 합산하지 않는다.
    reconstruction = d * z            # shape=(2,).

    # 대응하는 두 성분의 제곱오차를 평균한다.
    loss = ((reconstruction - x) ** 2).mean()
    return z, reconstruction, loss

A = torch.tensor([1.0, 2.0], dtype=torch.float64)
e = torch.tensor([0.5, 0.5], dtype=torch.float64)
d = torch.tensor([0.8, 1.2], dtype=torch.float64)
z, reconstruction, loss = tiny_ae(A, e, d)

print(z.item())                    # 원소 하나인 텐서 → Python 숫자: 약 1.5.
print(reconstruction.tolist())    # 텐서 → Python 리스트: 약 [1.2, 1.8].
print(loss.item())                 # 약 0.04. 계산 과정은 다음 절에서 확인한다.
```

<strong>스칼라(Scalar)</strong>는 숫자 하나다. 위의 `z.shape`는 축이 없는 `()`다. `(1,)`도 원소는 하나지만 길이 1인 축이 있는 벡터다. `(1,1)`은 두 축이 있다. 값의 개수와 축 구조를 따로 읽어야 한다.

`.item()`과 `.tolist()`는 출력·기록용 변환이다. Python 숫자나 리스트로 바꾼 결과를 통해 PyTorch의 역전파 연결이 이어지는 것은 아니다.

## 4. MSE는 대응하는 성분들의 제곱오차를 평균한다

<strong>평균제곱오차(Mean Squared Error, MSE)</strong>는 복원과 타깃의 같은 위치끼리 차이를 구하고, 그 차이를 제곱한 뒤 평균한 값이다. 순서는 <strong>대응 위치 확인 → 차이 계산 → 각 차이 제곱 → 평균</strong>이다.

앞 절의 타깃은 A의 원본 `[1,2]`이고 복원은 `[1.2,1.8]`이다.

| 계산 | 첫 성분 | 둘째 성분 |
| --- | ---: | ---: |
| 타깃 | 1 | 2 |
| 복원 | 1.2 | 1.8 |
| 복원 − 타깃 | +0.2 | −0.2 |
| 제곱오차 | 0.04 | 0.04 |

평균할 값이 두 개이므로 `(0.04+0.04)/2=0.04`다. <strong>여기서 분모 2는 샘플 수가 아니라 비교한 성분 수</strong>다. 차이만 더하면 양수와 음수가 상쇄될 수 있지만, 제곱한 뒤에는 그런 상쇄가 일어나지 않는다.

두 성분의 원본을 `x₁, x₂`, 복원을 `x_hat₁, x_hat₂`, 손실을 `L`이라고 쓰면 다음 계산이다.

<div class="ae-math">
$$L=\frac{(\hat{x}_1-x_1)^2+(\hat{x}_2-x_2)^2}{2}$$
</div>

제곱하므로 `원본−복원`과 `복원−원본` 중 어느 순서로 빼도 MSE 값은 같다. 이후 기울기를 설명할 때에는 식을 일관되게 쓰기 위해 `복원−타깃`으로 적는다.

### 합계를 낸 다음 `.mean()`을 호출하면?

두 제곱오차를 먼저 더하면 값 하나가 된다. Python 실수에는 `.mean()` 메서드가 없다. PyTorch 스칼라 텐서라면 `.mean()`은 원소 하나의 평균을 구하므로 그 값이 그대로 나온다. 자동으로 2를 나누어 주지 않는다.

```python
# reconstruction과 target은 같은 shape의 텐서다.
target = A
squared_errors = (reconstruction - target).square()  # [0.04, 0.04].
correct = squared_errors.mean()                      # 약 0.04.
sum_then_mean = squared_errors.sum().mean()           # 약 0.08.

# 두 스칼라 텐서가 이미 있다면 stack으로 새 축에 묶을 수 있다.
error1 = squared_errors[0]
error2 = squared_errors[1]
also_correct = torch.stack([error1, error2]).mean()
```

`square()`는 원소별 제곱, `sum()`은 합산, `mean()`은 평균을 계산하는 메서드다. `torch.stack()`은 텐서를 새 축으로 묶는 함수다. 여기서는 스칼라 두 개를 `(2,)` 텐서로 묶는다.[^stack]

모델 계산에서 얻은 `error1`, `error2`를 `torch.tensor([error1,error2])`로 다시 만드는 방식은 학습용 손실을 구성할 때 사용하지 않는다. `torch.tensor()`는 값을 복사한 새 텐서를 만들며 기존 계산과의 자동 미분 연결을 끊는다. `requires_grad=True`를 새 텐서에 붙여도 원래 가중치까지의 연결이 복구되지는 않는다.[^tensor]

### 배치 전체의 평균과 샘플별 평균

실무에서는 `torch.nn.MSELoss()`로 손실 계산 객체를 만든 뒤 `loss_fn(reconstruction,target)`처럼 호출한다. 기본 설정 `reduction="mean"`은 두 텐서에서 대응시킨 <strong>모든 원소의 제곱오차</strong>를 평균한다.[^mse]

| 입력·타깃 shape | 평균에 들어가는 제곱오차 수 | 기본 결과 shape |
| --- | --- | --- |
| `(N,F)` | `N×F` | `()` |
| `(N,C,H,W)` | `N×C×H×W` | `()` |

이미지마다 오차를 따로 보고 싶다면 샘플 축을 남긴다. `(N,C,H,W)`에서 `flatten(start_dim=1)`은 축 0을 유지하고 나머지 축을 묶어 `(N,C×H×W)`로 만든다. 여기에 `mean(dim=1)`을 적용하면 샘플별 오차가 든 `(N,)`이 된다.

손실을 계산하기 전에는 `reconstruction.shape == target.shape`를 확인하자. shape가 다르면 <strong>브로드캐스팅(Broadcasting)</strong>, 즉 호환되는 축의 크기에 맞춰 값을 대응시키는 규칙 때문에 의도하지 않은 비교가 이루어질 수 있다. 또한 픽셀 범위나 평균 기준이 다른 MSE끼리는 숫자만 직접 비교하지 않는다.

## 5. 하나의 손실에서 두 부분의 기울기를 구한다

손실은 현재 복원이 얼마나 틀렸는지를 말해 준다. 가중치를 어느 방향으로 바꿀지는 <strong>기울기(Gradient)</strong>로 계산한다. `∂L/∂w`는 손실 `L`을 가중치 `w`로 나누는 식이 아니다. 다른 독립 가중치와 입력을 고정하고 `w`를 조금 바꿀 때 손실이 얼마나 민감하게 변하는지를 나타내는 <strong>편미분(Partial Derivative)</strong>이다. 이런 값을 여러 변수에 대해 모으면 기울기가 된다.

현재 위치에서 기울기가 양수라면 `w`를 조금 늘릴 때 손실이 증가하는 방향이다. 음수라면 `w`를 조금 늘릴 때 손실이 감소하는 방향이다. 기울기 크기는 현재 변화율이며, 정답 가중치까지의 거리나 최적 이동량은 아니다.

<strong>순전파(Forward Pass)</strong>에서는 입력에서 복원과 손실까지 계산한다. <strong>역전파(Backpropagation)</strong>에서는 손실부터 연결을 거꾸로 따라가며 앞쪽 변수의 기울기를 구한다. 앞쪽 가중치가 손실에 미친 영향을 계산하려면, 그 가중치가 바꾼 중간값이 이후 계산을 통해 손실에 미친 영향을 알아야 하기 때문이다.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/backward-ko.svg" alt="출력 기울기 0.2와 -0.2에서 디코더 기울기 0.3과 -0.3을 계산한다. 잠재값으로 돌아오는 두 경로는 0.16과 -0.24이며 합은 -0.08이다. 여기에 입력 1과 2를 각각 곱해 인코더 기울기 -0.08과 -0.16을 구한다." loading="lazy">
<figcaption>가중치에 저장할 기울기와 앞쪽 계산으로 전달할 중간값의 기울기를 구분한다. z의 기울기는 인코더까지 계산을 이어 가는 데 사용한다.</figcaption>
</figure>

### 먼저 복원값에 대한 손실의 변화율을 구한다

식이 길어지는 것을 줄이기 위해 복원값 `x_hat₁, x_hat₂`를 각각 `r₁, r₂`라고 적자. 앞 절에서 `r₁=1.2`, `r₂=1.8`, 타깃은 `x₁=1`, `x₂=2`였다.

원소 `m`개의 평균제곱오차에서 출력 하나에 대한 기울기는 `2/m × (복원−타깃)`이다. 이번에는 평균할 원소가 두 개라서 `2/m=1`이다. 따라서 `∂L/∂r₁=+0.2`, `∂L/∂r₂=−0.2`다. 모든 MSE에서 차이 자체가 기울기가 되는 것은 아니다.[^mse]

이 변화율을 직접 확인할 수도 있다. 다른 값을 고정하고 첫 복원값만 `1.2`에서 `1.21`로 바꾸면, 손실은 `0.04`에서 `0.04205`로 바뀐다. 손실 변화 `0.00205`를 출력 변화 `0.01`로 나누면 `0.205`다. 출력 변화량을 `0.001`로 줄이면 이 비율은 `0.2005`가 된다. 변화량을 작게 할수록 기울기 `0.2`에 가까워진다.

### 디코더 가중치와 잠재값은 서로 다른 계산 대상이다

<strong>연쇄법칙(Chain Rule)</strong>은 연결된 계산의 변화율을 곱해 앞쪽 변수의 변화율을 구하는 규칙이다. 한 변수가 여러 경로를 통해 손실에 영향을 주면 각 경로에서 계산한 값을 더한다.

디코더의 관계는 `r₁=d₁z`, `r₂=d₂z`다. `d₁`을 조금 바꾸면 `r₁`은 그 변화의 `z`배만큼 바뀐다. 그러므로 `d₁`의 기울기는 첫 출력 기울기에 `z`를 곱해 구한다.

| 구하려는 대상 | 연결에 따른 계산 | 사용한 값 | 결과 |
| --- | --- | --- | ---: |
| `d₁`의 기울기 | `(∂L/∂r₁)×z` | `+0.2×1.5` | +0.30 |
| `d₂`의 기울기 | `(∂L/∂r₂)×z` | `−0.2×1.5` | −0.30 |

다음은 `z`의 기울기다. `z` 하나를 바꾸면 두 복원값이 모두 바뀐다. 첫 복원값을 거치는 영향과 둘째 복원값을 거치는 영향을 함께 계산해야 한다.

| 경로 | 출력으로 전달되는 배율 | 그 출력의 손실 기울기 | z의 손실 변화율에 더할 값 |
| --- | ---: | ---: | ---: |
| `z → r₁ → L` | `d₁=0.8` | +0.2 | `+0.2×0.8=+0.16` |
| `z → r₂ → L` | `d₂=1.2` | −0.2 | `−0.2×1.2=−0.24` |

따라서 `∂L/∂z=+0.16−0.24=−0.08`이다. “두 경로의 영향을 합한다”는 말은 <strong>같은 z가 두 출력 계산에 사용되므로, z의 작은 변화가 전체 손실에 미치는 두 영향을 더한다</strong>는 뜻이다. 두 가중치의 기울기를 무작정 더한 것이 아니다.

### 잠재값의 기울기를 인코더 가중치까지 전달한다

인코더의 관계는 `z=e₁x₁+e₂x₂`다. `e₁`의 변화가 `z`로 전달되는 배율은 첫 입력 `x₁`, `e₂`의 변화가 전달되는 배율은 둘째 입력 `x₂`다.

| 구하려는 대상 | 연결에 따른 계산 | 사용한 값 | 결과 |
| --- | --- | --- | ---: |
| `e₁`의 기울기 | `(∂L/∂z)×x₁` | `−0.08×1` | −0.08 |
| `e₂`의 기울기 | `(∂L/∂z)×x₂` | `−0.08×2` | −0.16 |

가중치 네 개에 대응하는 최종 기울기는 `e.grad=[−0.08,−0.16]`, `d.grad=[+0.30,−0.30]`이다. 이 개수는 실제 계산을 네 번만 한다는 뜻이 아니다. 앞에서 본 출력·중간값의 기울기와 곱셈·합산을 거쳐 얻은 결과다.

PyTorch는 이런 연결을 <strong>자동 미분(Automatic Differentiation, Autograd)</strong>으로 처리한다. 직접 만든 학습 가중치처럼 계산의 시작에 있는 텐서를 <strong>리프 텐서(Leaf Tensor)</strong>라고 부른다. 기본적으로 기울기는 학습 대상 리프 텐서의 `.grad`에 누적된다. 중간값 `z`의 기울기는 계산에 사용되어도 `.grad`에 따로 남지 않을 수 있다. 직접 관찰하려면 역전파 전에 `z.retain_grad()`를 호출한다.[^autograd]

## 6. 기울기를 구하는 일과 가중치를 바꾸는 일

<strong>경사하강법(Gradient Descent)</strong>은 현재 기울기의 반대 방향으로 가중치를 조금 이동시키는 방법이다. <strong>학습률(Learning Rate)</strong>은 그 이동 크기를 조절하는 양의 배율이다.

왜 반대 방향일까? 가중치의 작은 변화량을 `Δw`라고 하면, 근처에서의 손실 변화는 대략 `(∂L/∂w)×Δw`다. 기울기와 반대 부호로 `Δw`를 정하면 이 곱이 음수가 되어 손실을 줄이는 방향이 된다. 너무 크게 움직이면 이 근처의 변화율만으로 결과를 예측할 수 없으므로, 실제 손실 감소가 보장되지는 않는다.

기본 갱신식은 `새 가중치 = 기존 가중치 − 학습률×기울기`다. 앞 절에서 계산한 기울기에 학습률 `0.1`을 적용해 보자.

| 가중치 | 기존 값 | A 손실의 기울기 | 갱신 계산 | 새 값 |
| --- | ---: | ---: | --- | ---: |
| `e₁` | 0.5 | −0.08 | `0.5−0.1×(−0.08)` | 0.508 |
| `e₂` | 0.5 | −0.16 | `0.5−0.1×(−0.16)` | 0.516 |
| `d₁` | 0.8 | +0.30 | `0.8−0.1×0.30` | 0.770 |
| `d₂` | 1.2 | −0.30 | `1.2−0.1×(−0.30)` | 1.230 |

이 갱신은 다음 코드로 확인할 수 있다. 앞 절의 `tiny_ae()`를 사용하되, 가중치를 처음부터 다시 만든다. `requires_grad=True`는 이 가중치에 대한 기울기를 계산하도록 설정한다.

```python
# tiny_ae()와 import torch를 먼저 실행한다.
A = torch.tensor([1.0, 2.0], dtype=torch.float64)
B_input = torch.tensor([2.0, 1.0], dtype=torch.float64)
e = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
d = torch.tensor([0.8, 1.2], dtype=torch.float64, requires_grad=True)

# 이 손실은 가중치까지 미분 연결이 있는 상태로 계산한다.
z_before, recon_before, loss_before = tiny_ae(A, e, d)

# B는 지금 학습에 쓰지 않고 갱신 전 결과만 관찰한다.
with torch.no_grad():
    z_B0, recon_B0, loss_B0 = tiny_ae(B_input, e, d)

loss_before.backward()       # 기울기를 계산한다. 가중치는 아직 그대로다.
grad_e = e.grad.clone()       # 갱신 전 상태에서 구한 기울기를 복사한다.
grad_d = d.grad.clone()

# 갱신 연산 자체의 자동 미분 기록은 만들지 않는다.
with torch.no_grad():
    e -= 0.1 * grad_e         # 인코더 가중치를 실제로 바꾼다.
    d -= 0.1 * grad_d         # 디코더도 같은 갱신 전 상태의 기울기를 쓴다.

# 새 가중치의 결과는 새로 순전파하여 확인한다.
with torch.no_grad():
    for name, x in [("A", A), ("B", B_input)]:
        z, recon, loss = tiny_ae(x, e, d)
        print(name, z.item(), recon.tolist(), loss.item())
```

두 갱신 줄은 차례대로 실행되지만, 둘 다 <strong>같은 갱신 전 모델에서 구한 기울기</strong>를 사용한다. 인코더를 바꾼 뒤 그 새 인코더로 디코더 기울기를 다시 계산한 것이 아니다. 실제 반복 학습에서는 이 갱신을 <strong>옵티마이저(Optimizer)</strong>가 수행한다. Adam처럼 기울기의 이력도 이용하는 옵티마이저는 위의 단순 갱신식과 동작이 다르다.

### A로 학습한 뒤 B는 어떻게 바뀔까?

| 관찰 대상 | 갱신 전 | 갱신 후 |
| --- | --- | --- |
| A의 잠재값 | 1.5 | 1.54 |
| A의 복원 | `[1.2,1.8]` | `[1.1858,1.8942]` |
| A의 MSE | 0.04 | 0.02285764 |
| B의 잠재값 | 1.5 | 1.532 |
| B의 복원 | `[1.2,1.8]` | `[1.17964,1.88436]` |
| B의 MSE | 0.64 | 0.7275415696 |

<strong>A의 손실은 줄었지만 B의 손실은 늘었다.</strong> 가중치를 바꿀 때 사용한 것은 A의 손실 기울기였기 때문이다. 그 방향이 B의 손실도 줄이는 방향이라는 보장은 없다.

이 한 번의 결과만으로 <strong>과적합(Overfitting)</strong>, 즉 학습 데이터에는 잘 맞지만 새로운 데이터에 잘 맞지 않는 상태를 확정하지는 않는다. 여기서 확인한 것은 한 샘플의 개선이 다른 샘플의 개선까지 보장하지 않는다는 사실이다. 실제 학습의 성능은 학습에 쓰지 않은 여러 데이터로 확인해야 한다.

## 7. `no_grad`, `backward`, `step`을 구분한다

PyTorch에서 “모델을 실행했다”, “기울기를 계산했다”, “가중치를 바꿨다”는 서로 다른 동작이다.

| 코드 | 하는 일 | 가중치를 직접 바꾸는가? |
| --- | --- | --- |
| `model(x)` | 현재 가중치로 출력을 계산한다. | 일반적인 순전파에서는 아니다. |
| `loss.backward()` | 손실의 기울기를 관련 `.grad`에 누적한다. | 아니다. |
| `optimizer.step()` | 관리하는 매개변수를 갱신한다. | 그렇다. |
| `optimizer.zero_grad(set_to_none=True)` | 기존 기울기 저장값을 `None`으로 비운다. | 아니다. |
| `with torch.no_grad():` | 블록 안의 일반적인 연산에 역전파용 기록을 남기지 않는다. | 갱신 명령을 직접 쓰면 바꿀 수 있다. |

`no_grad()`는 <strong>자동 미분 기록을 생략하는 문맥 관리자(Context Manager)</strong>다. `with` 블록에 들어가 있는 동안 해당 설정을 적용하고, 블록이 끝나면 이전 상태로 돌아간다. 평가에 필요 없는 기록과 메모리 사용을 줄이는 데 쓴다.[^no-grad]

앞의 B 관찰 코드에서 `no_grad()`를 지워도 B가 자동으로 학습에 포함되지는 않는다. A와 B의 연산을 모두 기록했더라도 `loss_A.backward()`만 호출하면 A 손실에서 연결된 기울기를 계산한다. 두 손실을 함께 사용하려면 `(loss_A+loss_B)/2`처럼 연결한 손실에서 역전파해야 한다.

`no_grad()`는 기존 `.grad`를 비우지 않고, 매개변수의 `requires_grad`를 영구히 바꾸지 않으며, 가중치 수정을 금지하지도 않는다. 그래서 앞의 수동 갱신을 `no_grad()` 안에서 수행할 수 있었다.

### `model.eval()`은 무엇이 다른가?

`model.train()`과 `model.eval()`은 특정 계층의 학습·평가 동작을 설정한다. <strong>Dropout</strong>은 학습 중 일부 값을 무작위로 0으로 만드는 계층이고, <strong>BatchNorm</strong>은 값을 정규화할 때 학습과 평가에서 사용하는 통계가 달라질 수 있는 계층이다. 이런 계층은 모드에 따라 동작이 달라진다.

`model.eval()`은 자동 미분 기록을 끄는 명령이 아니다. 평가에서는 보통 `model.eval()`을 호출하고 `with torch.no_grad():` 안에서 모델을 실행한다. 이 글의 Linear·ReLU·Sigmoid 모델에는 학습·평가에 따라 바뀌는 계층이 없지만, 평가 코드는 같은 형태로 작성한다.[^autograd]

### `detach()`와 `clone()`은 어느 부분을 분리할까?

| 표현 | 기존 미분 연결 | 원본과 저장 공간 | 사용 예 |
| --- | --- | --- | --- |
| `tensor.detach()` | 끊는다. | 공유한다. | 연결을 끊고 값을 읽는다. |
| `tensor.clone()` | 추적 대상이면 유지한다. | 독립 복사한다. | 값의 사본을 만든다. |
| `tensor.detach().clone()` | 끊는다. | 독립 복사한다. | 초기 가중치 상태를 보관한다. |

예를 들어 `e0=e.detach().clone()`으로 복사하면 이후 `e`를 수정해도 `e0`는 그 저장 공간을 공유하지 않는다. `fixed` 같은 변수 이름이나 `detach()`가 값을 변경 불가능하게 만드는 것은 아니다.[^detach][^clone]

반복 학습에서는 매번 <strong>기울기 초기화 → 새 순전파 → 새 손실 → 역전파 → 갱신</strong>을 수행한다. `backward()`는 기존 기울기에 더할 수 있으므로 의도적인 누적 학습이 아니라면 초기화해야 한다. 이전 `loss.backward()` 줄만 반복하는 것은 다음 학습 스텝이 아니다.

## 8. 배치로 계산할 때에는 합산할 축을 지정한다

앞의 `tiny_ae()`는 `(2,)` 샘플 하나를 받도록 만들었다. 그래서 `(e*x).sum()`으로 두 특징을 모두 더해도 샘플끼리 섞이지 않았다.

이번에는 A와 B를 묶은 `X.shape=(2,2)`를 넣는다. 축 0은 샘플, 축 1은 특징이다. <strong>각 샘플 안에서만 특징을 합산하려면 `dim=1`을 지정해야 한다.</strong>

```python
import torch

X = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
e = torch.tensor([0.5, 0.5], dtype=torch.float64)
d = torch.tensor([0.8, 1.2], dtype=torch.float64)

products = X * e                          # [[0.5,1.0],[1.0,0.5]].
Z = products.sum(dim=1, keepdim=True)     # [[1.5],[1.5]], shape=(2,1).
X_hat = Z * d                            # [[1.2,1.8],[1.2,1.8]], (2,2).

print(products.tolist())
print(Z.tolist(), tuple(Z.shape))
print(X_hat.tolist(), tuple(X_hat.shape))
```

`dim=1`은 축 번호 1을 따라 합산하라는 뜻이고, `keepdim=True`는 합산한 축을 크기 1로 남기라는 뜻이다. `(2,1)`의 각 잠재값은 브로드캐스팅에 따라 디코더 가중치 두 개와 각각 곱해진다. 다른 샘플의 잠재값과 합쳐지는 것은 아니다.

`products.sum()`처럼 축을 생략하면 네 원소를 전부 합하여 스칼라 `3.0`을 만든다. 그 결과는 두 샘플의 잠재 표현을 각각 계산한 것이 아니다.

### `nn.Linear`에서는 가중치 모양이 어떻게 될까?

`from torch import nn`은 `torch.nn`을 `nn`이라는 이름으로 가져온다. `nn.Linear`는 입력 특징의 가중합을 계산하는 계층 클래스다. <strong>생성자(Constructor)</strong>는 계층을 만들고, 만든 계층을 호출하면 실제 데이터를 계산한다.[^linear]

| 구분 | 인코더 | 디코더 |
| --- | --- | --- |
| 생성 | `nn.Linear(2,1,bias=False)` | `nn.Linear(1,2,bias=False)` |
| 입력 → 출력 | `(N,2) → (N,1)` | `(N,1) → (N,2)` |
| 가중치 shape | `(1,2)` | `(2,1)` |

생성자의 첫 인자 `in_features`는 샘플당 입력 특징 수, 둘째 인자 `out_features`는 샘플당 출력 특징 수다. `bias=False`는 편향을 사용하지 않는다는 뜻이다. 배치 크기 `N`은 생성자 인자가 아니다. 만들어진 `encoder`에 `encoder(X)`를 호출할 때 실제 배치가 들어간다.

PyTorch Linear의 가중치 shape는 <strong>`(출력 특징 수, 입력 특징 수)`</strong>다. 앞에서 사용한 `e.shape=(2,)`, `d.shape=(2,)`는 같은 계산을 직접 적기 위한 표현이다. Linear를 만들기만 하면 가중치는 초기화 규칙으로 정해지므로, 앞의 수치를 그대로 재현하려면 해당 값들을 가중치에 복사해야 한다.

## 9. 값을 줄이는 일과 정보를 보존하는 일

<strong>차원 축소(Dimensionality Reduction)</strong>는 샘플을 표현하는 특징값의 개수를 줄이는 일이다. 입력보다 작은 중간 표현을 거치도록 제한한 부분을 <strong>병목(Bottleneck)</strong>이라고 한다. 작은 병목은 AE를 구성하는 한 방법이며 모든 AE에 필수인 조건은 아니다.[^ae]

값의 개수가 줄어도 원본을 복원할 수 있는지는 데이터의 규칙과 남겨진 정보에 달려 있다. 이를 확인하기 위해 학습하지 않는 손규칙을 하나 만들어 보자.

입력 `[a,b,c,d]`에서 앞의 두 값 `[a,b]`만 남긴다. 복원할 때에는 그 두 값을 반복해 `[a,b,a,b]`를 만든다.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/information-loss-ko.svg" alt="서로 다른 원본 [0.2,0.8,0.2,0.8]과 [0.2,0.8,0.7,0.1]에서 앞의 두 값만 남기면 같은 코드 [0.2,0.8]이 된다. 반복 규칙은 같은 복원 [0.2,0.8,0.2,0.8]을 만들어 첫 원본은 완전 복원하고 둘째 원본은 MSE 0.185가 된다." loading="lazy">
<figcaption>학습 모델이 아닌 고정 규칙이다. 같은 코드로 합쳐진 두 입력을 이 디코더가 다시 구분할 수 있는지 살펴보는 예제다.</figcaption>
</figure>

| 원본 | 남긴 코드 | 복원 | MSE |
| --- | --- | --- | ---: |
| `[0.2,0.8,0.2,0.8]` | `[0.2,0.8]` | `[0.2,0.8,0.2,0.8]` | 0 |
| `[0.2,0.8,0.7,0.1]` | `[0.2,0.8]` | `[0.2,0.8,0.2,0.8]` | 0.185 |

첫 원본은 뒤의 두 값이 앞의 두 값과 같아서 반복 규칙으로 정확히 복원된다. 둘째 원본의 뒤쪽 값은 `0.7,0.1`이지만 그 정보가 코드에 남지 않았다. 제곱오차는 `[0,0,0.25,0.49]`이고 평균은 `0.185`다.

<strong>결정론적(Deterministic)</strong>이라는 말은 같은 입력과 같은 모델 상태에서 같은 결과를 낸다는 뜻이다. 같은 코드만 받은 고정된 결정론적 디코더는 위의 두 원본을 구분해 둘 다 정확하게 복원할 수 없다.

```python
import torch

def repeat_first_half(x):
    # x는 shape=(4,)다. 슬라이싱 끝값 2는 제외하여 인덱스 0,1만 선택한다.
    code = x[:2].clone()

    # cat은 기존 축으로 이어 붙인다: (2,) 두 개 → (4,).
    reconstruction = torch.cat([code, code])
    mse = (reconstruction - x).square().mean()
    return code, reconstruction, mse

x = torch.tensor([0.2, 0.8, 0.7, 0.1], dtype=torch.float64)
code, reconstruction, mse = repeat_first_half(x)
print(code.tolist(), reconstruction.tolist(), mse.item())  # MSE 약 0.185.
```

`torch.cat()`은 기존 축의 길이를 늘리고, `torch.stack()`은 새 축을 만든다. 둘 다 텐서를 모으지만 결과 shape를 정하는 방식이 다르다.

### 정보 소실과 재구성 오차를 나누어 보자

<strong>정보 소실(Information Loss)</strong>은 서로 다른 원본을 구별할 정보가 표현에 남지 않는 현상이다. <strong>재구성 오차(Reconstruction Error)</strong>는 실제 복원과 원본의 차이다. <strong>손실 함수(Loss Function)</strong>는 그 차이를 숫자로 측정하는 계산 규칙이다.

코드에 필요한 정보가 남아 있어도 디코더가 충분히 학습되지 않았거나 표현 능력이 부족하면 재구성 오차가 생긴다. 반대로 위의 첫 원본처럼 데이터에 반복 규칙이 있으면 적은 값으로도 정확히 복원할 수 있다. 모든 오차를 인코더의 정보 소실로 설명하거나, 값을 줄이면 어떤 데이터도 정확히 복원할 수 없다고 일반화하면 안 된다.

### 잠재값이 줄면 파일도 같은 비율로 작아질까?

RGB `32×32` 이미지는 `3×32×32=3,072`개 값을 가진다. 이를 잠재값 128개로 바꾸었다면 값의 개수는 줄었다. 실제 저장량은 원소당 비트 수, 저장 형식, 부가 정보, 복원 모델을 전달해야 하는지에 따라 달라진다.

실제 학습 기반 압축에서는 잠재 표현을 정해진 값들로 바꾸는 <strong>양자화(Quantization)</strong>와, 값의 발생 확률 등을 이용해 비트열을 만드는 <strong>엔트로피 부호화(Entropy Coding)</strong> 같은 단계가 추가된다. 저장·전송한 비트열을 해독한 뒤 디코더가 복원한다.[^compression]

복원 품질과 필요한 비트 수를 함께 고려하는 관계를 <strong>율–왜곡 절충(Rate–Distortion Trade-off)</strong>이라고 한다. AE의 특징값 개수만 보고 실제 파일 압축률을 결론 내리지는 않는다.

## 10. 잠재공간에서 점과 차원은 무엇일까?

<strong>잠재공간(Latent Space)</strong>은 잠재 표현을 좌표로 놓는 공간이다. 샘플당 잠재값이 하나면 좌표 하나, 두 개면 좌표 두 개, `K`개면 `K`차원 좌표를 사용한다.

`Z.shape=(100,16)`이라면 샘플 100개가 각각 16개 잠재 특징으로 표현되어 있다. <strong>점은 100개이고, 각 점의 좌표 성분은 16개</strong>다. 샘플 수 100이 잠재공간의 차원이라는 뜻은 아니다.

앞에서 A로 한 번 갱신한 인코더 `e=[0.508,0.516]`을 그대로 두고 A와 B를 넣으면 다음 결과가 나온다. 표의 복원 MSE는 같은 갱신 후 디코더 `d=[0.770,1.230]`까지 사용해 계산했다.

| 입력 | 같은 인코더로 계산한 잠재값 | 복원 MSE |
| --- | ---: | ---: |
| A=`[1,2]` | 1.540 | 0.02285764 |
| B=`[2,1]` | 1.532 | 0.7275415696 |

좌표 간격은 `1.540−1.532=0.008`이다. 이 값은 <strong>같은 인코더가 두 입력을 어디에 배치하는지</strong> 보여준다. 잠재값이 작을수록 복원이 잘된다는 뜻도, 좌표가 가깝기만 하면 의미가 비슷하다는 뜻도 아니다.

### 잠재 표현과 임베딩의 관계

<strong>임베딩(Embedding)</strong>은 데이터를 숫자 벡터로 표현한 것이다. AE의 잠재 표현도 넓은 의미에서 임베딩으로 볼 수 있다. 다만 원본 재구성에 맞춰 학습한 표현과, 관련 문장을 찾아내는 의미 검색에 맞춰 학습한 표현은 학습 목표가 다르다.

<strong>코사인 유사도(Cosine Similarity)</strong>는 두 벡터의 방향이 얼마나 비슷한지를 측정한다. 같은 방향이면 1, 직각이면 0, 반대 방향이면 −1이다. 이 측정 함수를 계산하는 것만으로 좋은 임베딩이 학습되지는 않는다. 먼저 모델이 벡터를 만들고, 그 벡터들을 비교한다.[^embedding]

위의 `1.540`과 `1.532`는 모두 양수인 1차원 값이다. 둘 다 같은 양의 방향이므로 수학적인 코사인 유사도는 1이다. 여기서는 수직선 위의 좌표 간격을 보고 있는 것이다. 값이 0인 영벡터는 방향이 없으므로 수학적인 코사인 유사도가 정의되지 않는다.

<details>
<summary>잠재값 두 개를 그래프로 표시하는 코드와 인자</summary>

`Figure`는 그림 전체를 관리하는 객체이고, `Axes`는 그 안의 개별 그래프 영역이다. `plt.subplots()`는 둘을 만든다. 아래의 `y=0`은 점을 한 수평선에 놓기 위한 표시 좌표이며 두 번째 잠재 특징이 아니다.

```python
import matplotlib.pyplot as plt

values = [("A", 1.540), ("B", 1.532)]
fig, ax = plt.subplots(figsize=(7, 2.4))  # 그림 너비·높이, 단위는 인치.
ax.axhline(y=0, color="gray")           # 데이터 좌표 y=0에 수평선을 그린다.
ax.scatter([value for _, value in values], [0, 0])

for name, value in values:             # 튜플에서 이름과 값을 각각 꺼낸다.
    ax.annotate(
        f"{name}: z={value:.3f}",      # 소수점 아래 세 자리로 표시한다.
        xy=(value, 0),                # 설명할 점의 실제 데이터 좌표.
        xytext=(0, 18 if name == "A" else -28),
        textcoords="offset points", # 점에서 글자를 이동할 거리의 단위.
        ha="center",                 # 글자를 수평 중앙 정렬한다.
    )

ax.set_xlim(1.525, 1.547)               # 표시 범위만 정한다. z를 바꾸지 않는다.
ax.set_ylim(-0.45, 0.45)
ax.set_yticks([])                      # 의미 없는 세로 눈금은 숨긴다.
ax.set_xlabel("Latent value z")
fig.tight_layout()                    # 제목·눈금 사이 여백을 조정한다.
plt.show()
```

리스트 컴프리헨션 `[value for _,value in values]`는 `[1.540,1.532]`를 만든다. `_`는 이번 계산에서 사용하지 않을 이름 부분을 받는 관례적 변수다. `annotate()`는 설명 글자를 배치하며, `offset points`의 포인트는 `1/72`인치 단위다. `18`은 잠재 좌표를 18만큼 바꾼다는 뜻이 아니다.[^plot]

`axhline()`의 `y`는 실제 데이터 좌표지만 `xmin`, `xmax`는 그래프 너비에 대한 0~1 비율이다. `set_title`, `set_xticks`, `tick_params`, `spines`는 제목·눈금·글자·테두리 표시를 조정한다. 이런 설정은 값을 학습시키지 않는다. 별도의 색상 사전 이름을 쓴다면 그 사전도 코드 안에서 직접 정의해야 한다.

</details>

## 11. 잡음 제거 AE에서는 입력과 타깃이 달라진다

<strong>잡음 제거 오토인코더(Denoising Autoencoder, DAE)</strong>는 손상된 입력에서 손상 전 원본을 재구성하도록 학습한다. 원본을 `clean`, 원본에 잡음을 추가한 버전을 `noisy`라고 부르자. 여기서 clean은 인위적인 잡음을 더하기 전의 기준 데이터라는 뜻이다.[^dae]

| 구분 | 기본 AE | DAE |
| --- | --- | --- |
| 모델 입력 | `clean` | `noisy` |
| 비교 타깃 | `clean` | `clean` |
| 인코더의 직접 출력 | 잠재 표현 `z` | 잠재 표현 `z` |
| 디코더의 직접 출력 | 재구성 결과 | 재구성 결과 |
| 최종 학습 목표 | 입력 원본에 가깝게 복원 | 잡음 추가 전 원본에 가깝게 복원 |

핵심 차이는 모델에 무엇을 넣고 무엇과 비교하는지에 있다. 같은 구조를 사용하더라도 다음처럼 학습 데이터를 연결하면 과제가 달라진다.[^keras]

```python
# model은 이미 생성한 모델, loss_fn은 nn.MSELoss()다.

# 기본 AE: 원본을 입력받아 원본과 비교한다.
reconstruction = model(clean_batch)
loss = loss_fn(reconstruction, clean_batch)

# DAE: 잡음 입력을 받아 잡음 추가 전 원본과 비교한다.
reconstruction = model(noisy_batch)
loss = loss_fn(reconstruction, clean_batch)
```

DAE도 인코더 하나와 디코더 하나로 만들 수 있다. 인코더가 깨끗한 이미지를 먼저 완성한 뒤 디코더가 한 번 더 복원해야 하는 구조는 아니다. 인코더의 직접 출력은 중간 특징이고, 디코더가 원본 형태의 출력값을 계산한다.

예를 들어 이미지의 784개 픽셀을 잠재 특징 64개로 바꾸었다면, 그 64개 값은 완성된 784픽셀 이미지가 아니다. 기본 DAE는 보통 이 중간값에 별도의 정답 `z`를 주지 않고, 최종 출력과 clean의 손실로 두 부분을 함께 학습한다.

<strong>강건성(Robustness)</strong>은 입력이 흔들리거나 잡음이 있어도 필요한 정보를 안정적으로 활용하는 성질이다. DAE의 학습은 잡음에 강한 표현을 배우도록 유도할 수 있지만 잠재 표현에서 잡음이 완전히 사라진다고 보장하지는 않는다. 잡음 제거 능력은 최종 출력에서 평가한다. 작은 병목 역시 DAE의 필수 조건은 아니다.

## 12. 이미지에서는 샘플·채널·높이·너비를 구분한다

MNIST는 `28×28` 흑백 손글씨 이미지 데이터다. 여기서는 이 크기의 흑백 이미지 한 장을 생각하자. <strong>픽셀(Pixel)</strong>은 이미지의 공간 위치 하나이며, 흑백 이미지는 각 위치에 밝기를 나타내는 값 하나를 가진다. <strong>채널(Channel)</strong>은 이미지의 성분을 나누는 축으로, 이 흑백 이미지의 채널 수는 1이다.

PyTorch 이미지 배치를 `(N,C,H,W)` 순서로 표현하면 `N`은 샘플 수, `C`는 채널 수, `H`는 높이, `W`는 너비다.

| 축 번호 | 의미 | 예제 크기 | 유효 인덱스 |
| --- | --- | --- | --- |
| 0 | 샘플 `N` | `N` | `0`부터 `N−1` |
| 1 | 채널 `C` | 1 | 0 |
| 2 | 높이 방향의 위치 `H` | 28 | 0~27 |
| 3 | 너비 방향의 위치 `W` | 28 | 0~27 |

이미지 한 장을 배치에 넣으면 `(1,1,28,28)`이다. 배치 축 없는 한 장의 채널·공간 표현은 `(1,28,28)`이다. 다만 <strong>shape `(1,28,28)`만 보고 첫 축이 채널이라고 확정할 수는 없다.</strong> 파일 작성자가 채널을 생략한 `(N,H,W)`로 저장했을 수도 있기 때문이다. 배열을 만든 코드와 규약을 확인해야 한다.

단일 흑백 이미지의 `(C,H,W)`로 저장한 배열이라는 것을 확인했다면 `torch.from_numpy(clean).unsqueeze(0)`으로 배치 축을 붙일 수 있다. `from_numpy()`는 NumPy 배열과 메모리를 공유하는 텐서를 만들고, `unsqueeze(0)`는 축 번호 0 위치에 크기 1인 축을 추가한다. 새 이미지를 만들어 내는 연산은 아니다.[^tensor]

### 잡음은 어떻게 추가할까?

<strong>표준정규분포(Standard Normal Distribution)</strong>는 평균 0, 표준편차 1인 정규분포다. <strong>표준편차(Standard Deviation)</strong>는 값들이 평균 주변에서 퍼지는 정도를 나타낸다. 다음 코드는 clean과 같은 shape의 잡음을 만들고, 잡음 크기를 조절한 뒤 픽셀 범위를 0~1로 제한한다.

```python
# clean_batch는 값 범위가 0~1인 (N,1,28,28) 텐서다.
noise = torch.randn_like(clean_batch)          # 같은 shape의 표준정규 잡음.
noisy_batch = (clean_batch + 0.25 * noise).clamp(0, 1)
```

`randn_like()`는 기준 텐서와 같은 shape의 난수 텐서를 만든다. `0.25`를 곱하면 범위 제한 전 잡음의 표준편차가 `0.25`가 된다. `clamp(0,1)`은 0보다 작으면 0, 1보다 크면 1로 바꾼다. 이런 범위 제한을 <strong>클리핑(Clipping)</strong>이라고 한다.

클리핑 후 실제 변화량 `noisy_batch−clean_batch`가 그대로 표준편차 0.25인 정규분포를 따르는 것은 아니다. 경계를 넘은 값이 잘렸기 때문이다.

### 같은 픽셀을 비교한다는 뜻

`(1,28,28)` 배열에서 `image[0,14,14]`를 선택하면 첫 축 인덱스 0, 높이 인덱스 14, 너비 인덱스 14의 값이다. 인덱스는 0부터 시작하므로 화면의 15번째 행과 15번째 열에 해당한다.

같은 위치의 clean 값이 약 `0.7921569`, noisy 값이 약 `0.89783067`인 쌍을 예로 들 수 있다. DAE는 noisy 이미지 전체를 받아 복원 이미지 전체를 계산하고, 복원의 이 위치를 clean의 같은 위치와 비교한다. 숫자 종류 레이블을 맞히는 과제가 아니다. 전체 손실은 해당 픽셀 하나만이 아니라 모든 픽셀을 비교한다.

| 이름 | 언제 준비되는가? | 역할 |
| --- | --- | --- |
| `noisy` | 모델 실행 전 | 입력 |
| `clean` | 모델 실행 전 | 학습·평가의 비교 타깃 |
| `reconstruction` | 모델 실행 후 | 실제 예측 출력 |

화면에 clean을 그렸다고 모델이 복원한 것은 아니다. 실제 사용 단계의 <strong>추론(Inference)</strong>에서는 보통 clean을 모르는 상태에서 noisy를 받아 복원을 만든다. clean은 학습·평가 때 결과를 비교하는 기준이다.

## 13. clean/noisy 네 칸 그림은 무엇을 확인하는가?

배열 파일을 읽어 그림으로 확인하는 단계는 모델 학습 전에도 필요하다. 입력과 타깃이 같은 원본에서 왔는지, 위치가 어긋나지 않았는지, 값 범위가 맞는지 확인하는 과정이다.

<strong>NPZ</strong>는 이름이 있는 여러 NumPy 배열을 담을 수 있는 파일 형식이다. 확장자만으로 내부 배열이 반드시 압축 저장되었다고 단정하지 않는다. 다음 코드는 `clean`, `noisy`라는 이름의 숫자 배열을 가진 파일을 읽는다. 파일 자체는 별도로 준비해야 하며, 뒤의 digits 실습에는 이 파일이 필요 없다.

```python
from pathlib import Path
import numpy as np

# 실제 파일이 있는 경로로 지정한다.
pair_path = Path("mnist_day1_pair.npz")
if not pair_path.is_file():
    raise FileNotFoundError(f"파일 위치를 확인하세요: {pair_path}")

# allow_pickle=False: Python 객체 복원 기능 없이 숫자 배열을 읽는다.
# with 블록을 벗어나면 열린 NPZ 파일을 닫는다.
with np.load(pair_path, allow_pickle=False) as pair:
    clean = pair["clean"].copy()      # 독립적인 배열 사본을 보관한다.
    noisy = pair["noisy"].copy()

assert clean.shape == noisy.shape == (1, 28, 28)
for name, image in [("clean", clean), ("noisy", noisy)]:
    print(name, image.shape, image.dtype, image.min(), image.max())
```

`np.load()`가 NPZ를 읽으면 이름으로 배열을 꺼낼 수 있는 객체를 반환한다. `dtype`은 원소의 자료형이고, `min()`과 `max()`는 배열 안의 실제 최솟값과 최댓값이다. `float32`는 원소당 32비트 부동소수점이라는 뜻이다.[^numpy]

clean의 최댓값이 `1.0`보다 약간 작은 `0.996...`이라고 해도 그 자체로 범위 조정 오류는 아니다. 0~1 범위에 있다는 조건과 반드시 1인 픽셀이 있다는 조건은 다르다. 파일 로딩이 실패했다면 경로를 고쳐 다시 읽어야 한다. 노트북의 이전 셀에 남은 배열이나 이전 그림을 새 파일의 결과로 해석하지 않는다.

### 전체 두 칸과 확대 두 칸

`fig, axes=plt.subplots(2,2)`는 그림 하나와 표시 영역 네 개를 만든다. `axes.shape=(2,2)`는 표시 영역 객체들의 배열 모양이다. 모델의 입력 shape와 관계없다.

| 표시 영역 | 표시하는 데이터 | 배열 크기 |
| --- | --- | --- |
| `axes[0,0]` | noisy 전체 | `28×28` |
| `axes[0,1]` | clean 전체 | `28×28` |
| `axes[1,0]` | noisy의 같은 위치 확대 | `7×7` |
| `axes[1,1]` | clean의 같은 위치 확대 | `7×7` |

이는 서로 다른 샘플 네 개가 아니라 <strong>noisy 한 장과 clean 한 장을 전체·확대 형태로 각각 표시한 것</strong>이다. 모델 호출이나 역전파가 없으므로 잡음 제거 결과를 보여주는 그림도 아니다.

`axes[0,1]`은 화면의 오른쪽 위 표시 영역을 고른다. `image[0,14,14]`는 데이터 안의 픽셀을 고른다. 두 인덱스가 선택하는 대상을 구분하자.

### 확대해도 원래 픽셀 번호를 유지한다

`image[0,11:18,11:18]`은 높이·너비 인덱스 각각 `11,12,13,14,15,16,17`을 선택한다. <strong>끝값 18은 포함하지 않는다.</strong> 각 방향이 `18−11=7`이므로 잘라 낸 배열은 `(7,7)`이다. 원래 인덱스 14는 잘라 낸 배열 안에서 `14−11=3`이다.

이 작은 배열을 기본 좌표로 표시하면 0~6으로 보인다. 원래 좌표 11~17을 유지하려면 `extent=(10.5,17.5,17.5,10.5)`를 지정한다. 순서는 `(왼쪽,오른쪽,아래쪽,위쪽)`이다. 픽셀 중심이 11부터 17까지이고 각 픽셀의 경계는 중심에서 0.5씩 떨어져 있으므로 전체 경계가 10.5~17.5가 된다.[^imshow]

```python
# clean/noisy 파일을 읽은 뒤 실행하는 표시 코드다. 모델을 학습하지 않는다.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for col, (name, image) in enumerate([("noisy", noisy), ("clean", clean)]):
    full = axes[0, col]                  # 전체 이미지를 표시할 영역.
    zoom = axes[1, col]                  # 같은 이미지의 일부를 표시할 영역.
    full.imshow(image[0], cmap="gray", vmin=0, vmax=1,
                interpolation="nearest", origin="upper")
    zoom.imshow(image[0, 11:18, 11:18], cmap="gray", vmin=0, vmax=1,
                interpolation="nearest", origin="upper",
                extent=(10.5, 17.5, 17.5, 10.5))
    full.set_title(f"{name}: full")
    zoom.set_title(f"{name}: rows/cols 11–17")

    # 픽셀 중심 (열14, 행14)의 경계는 각 방향 13.5~14.5다.
    # Rectangle은 표시용 테두리다. 배열의 픽셀값은 바꾸지 않는다.
    zoom.add_patch(Rectangle((13.5, 13.5), 1, 1,
                             fill=False, edgecolor="red", linewidth=2))

fig.tight_layout()
plt.show()
```

`cmap="gray"`는 숫자를 흑백 밝기로 표시하는 색상표다. `vmin=0,vmax=1`을 두 이미지에 동일하게 적용하면 같은 숫자가 같은 밝기로 보인다. `interpolation="nearest"`는 표시할 때 픽셀 경계를 구분하기 쉽게 하고, `origin="upper"`는 배열의 첫 행을 위에 둔다. 이 설정들은 픽셀을 예측하거나 잡음을 제거하지 않는다.

`Rectangle`의 `fill=False`는 내부를 채우지 않는다는 뜻이다. 검정·흰색처럼 대비되는 두 테두리를 겹쳐 그릴 수도 있지만, 그것 역시 표시를 위한 설정이다. 좌표와 밝기 범위는 정확히 이해하고, 글자 크기나 테두리 두께는 필요한 때 찾아보면 된다.

## 14. 실제 모델을 만들고 이미지 shape를 연결한다

이제 수동 가중치 대신 PyTorch 계층을 묶어 모델을 만든다. <strong>클래스(Class)</strong>는 객체의 구성과 동작을 정의하고, `__init__`은 객체를 만들 때 실행되는 초기화 메서드다. `nn.Module`은 신경망 모델의 기본 클래스이고, `nn.Sequential`은 등록한 계층을 순서대로 실행하는 컨테이너다.

이번 모델의 생성자는 `in_features`, `hidden_features`, `latent_features`를 받는다. 각각 샘플당 입력 특징 수, 중간 은닉 특징 수, 잠재 특징 수다. 실제 입력 텐서는 생성자가 아니라 `model(x)` 호출에 들어간다. 그 호출이 모델의 `forward()`에 정의한 계산을 실행한다.

<strong>ReLU</strong>는 음수를 0으로 바꾸고 0 이상은 유지하는 활성화 함수다. <strong>Sigmoid</strong>는 출력을 0~1 범위로 바꾼다. 여기서는 타깃을 0~1로 맞추기 때문에 마지막에 Sigmoid를 사용한다. 모든 AE의 출력층에 Sigmoid가 필수인 것은 아니다.

```python
import torch
from torch import nn

class DenseAutoencoder(nn.Module):
    def __init__(self, in_features, hidden_features, latent_features):
        super().__init__()  # nn.Module의 초기화를 실행한다.

        # 각 Linear는 기본 bias=True이므로 가중치와 편향을 학습한다.
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, latent_features),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N,F), z: (N,K), reconstruction: (N,F).
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

model = DenseAutoencoder(64, 32, 16)  # 계층과 매개변수를 만든다.
```

이 모델은 `(N,64) → (N,32) → (N,16) → (N,32) → (N,64)`를 계산한다. `model.parameters()`는 인코더와 디코더에 등록된 가중치·편향을 모두 제공한다. 따라서 옵티마이저 하나가 두 부분을 함께 관리할 수 있다.

### MNIST 크기의 이미지에 연결하기

한 채널 `28×28` 이미지의 값은 `1×28×28=784`개다. 이 경우 `DenseAutoencoder(784,128,64)`를 만들고 배치의 이미지 축을 특징 축으로 묶는다.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/image-shapes-ko.svg" alt="noisy 이미지 배치 (N,1,28,28)를 샘플별로 펼치면 (N,784)다. 인코더가 (N,64)를 만들고 디코더가 (N,784)를 계산한다. 이를 (N,1,28,28)로 다시 배치하여 clean 타깃과 비교한다." loading="lazy">
<figcaption>flatten은 픽셀의 배치를 바꾸고, 신경망은 새 픽셀값을 계산한다. 마지막 reshape는 계산된 값들을 이미지 형태로 배치한다. 샘플 축 N은 유지된다.</figcaption>
</figure>

```python
# noisy_batch와 clean_batch는 같은 원본에 대응하는 (N,1,28,28) 텐서다.
mnist_model = DenseAutoencoder(784, 128, 64)
loss_fn = nn.MSELoss()

flat_input = noisy_batch.flatten(start_dim=1)          # (N,784).
flat_reconstruction = mnist_model(flat_input)          # (N,784).
reconstruction = flat_reconstruction.reshape_as(noisy_batch)

assert reconstruction.shape == clean_batch.shape       # (N,1,28,28).
loss = loss_fn(reconstruction, clean_batch)
```

`flatten(start_dim=1)`은 샘플 축 0을 남기고 나머지 축을 합친다. `reshape_as(noisy_batch)`는 원소 순서를 유지하면서 기준 텐서와 같은 shape로 바꾼다. 복원 픽셀값은 이미 디코더가 계산했다. reshape가 새로운 복원을 예측하는 것은 아니다.

## 15. 별도 이미지 파일 없이 학습하고 평가한다

실행 예제에는 scikit-learn에 포함된 <strong>digits</strong> 데이터를 사용한다. <strong>이 데이터는 MNIST와 다르다.</strong> 이미지가 `8×8`이고, 원래 픽셀값의 범위는 0~16이다. 총 1,797장을 `16.0`으로 나누어 0~1 범위로 맞추면 이미지당 특징은 64개다.[^digits]

| 항목 | 앞 절의 MNIST 크기 설명 | 실행 예제의 digits |
| --- | --- | --- |
| 이미지 크기 | `28×28` | `8×8` |
| 펼친 특징 수 | 784 | 64 |
| 모델 설정 | `DenseAutoencoder(784,128,64)` | `DenseAutoencoder(64,32,16)` |
| 외부 NPZ 파일 | 파일 읽기 설명에서만 사용 | 필요 없음 |

실행 순서는 <strong>원본 데이터 분리 → 학습 배치 구성 → 입력에만 필요한 잡음 추가 → 복원과 clean 비교 → 역전파·갱신 → 보관해 둔 데이터 평가</strong>다. 같은 원본의 clean과 noisy가 학습용·평가용 양쪽에 섞이지 않도록 원본을 먼저 분리한다.

<strong>데이터셋(Dataset)</strong>은 샘플을 꺼낼 수 있게 묶은 객체다. `TensorDataset`은 텐서의 첫 축 인덱스가 같은 값들을 한 샘플로 묶는다. <strong>DataLoader</strong>는 데이터셋에서 배치를 만든다. `batch_size=64`는 배치당 최대 64개 샘플, `shuffle=True`는 각 순회에서 샘플 순서를 섞는 설정이다.

<strong>에포크(Epoch)</strong>는 학습 데이터를 한 차례 순회하는 단위다. 배치 하나를 처리하고 가중치를 갱신하는 <strong>학습 스텝(Training Step)</strong>과 다르다. 이 예제는 학습 샘플 1,437개를 최대 64개씩 묶으므로 에포크당 23개 배치를 처리한다. 마지막 배치는 29개 샘플이다.

<details>
<summary>모델 생성부터 평가까지의 전체 학습 함수</summary>

앞 절의 `DenseAutoencoder` 정의 다음에 붙여 실행할 수 있다. 함수는 기본 AE와 DAE에 같은 모델 구조를 사용한다. `denoise=True`일 때만 입력에 잡음을 추가하며, 타깃은 항상 clean이다.

```python
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader

def run_digits(denoise=False, epochs=20):
    # seed: 난수의 시작 상태. 같은 환경에서 실험을 다시 비교하기 위해 고정한다.
    torch.manual_seed(42)

    # (1797,8,8) → (1797,64). 원래 픽셀 0~16을 0~1로 맞춘다.
    X = torch.tensor(load_digits().images, dtype=torch.float32)
    X = X.flatten(start_dim=1) / 16.0

    # randperm은 0부터 len(X)-1까지의 인덱스를 무작위 순서로 만든다.
    indices = torch.randperm(len(X))
    split = int(len(X) * 0.8)               # 학습 1437개, 평가 360개.
    train_x = X[indices[:split]]             # 끝값 split은 제외한다.
    test_x = X[indices[split:]]

    loader = DataLoader(TensorDataset(train_x), batch_size=64, shuffle=True)
    model = DenseAutoencoder(64, 32, 16)     # 함수 호출마다 새 모델을 만든다.
    loss_fn = nn.MSELoss()                   # 모든 대응 원소의 제곱오차 평균.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):              # 0부터 epochs-1까지 반복한다.
        model.train()
        total_loss = 0.0

        # TensorDataset에 텐서 하나를 넣었으므로 요소 하나인 묶음이 나온다.
        # (clean_batch,)는 그 묶음에서 실제 배치 텐서를 꺼낸다.
        for (clean_batch,) in loader:
            if denoise:
                input_batch = (
                    clean_batch + 0.25 * torch.randn_like(clean_batch)
                ).clamp(0, 1)
            else:
                input_batch = clean_batch

            optimizer.zero_grad(set_to_none=True)     # 이전 기울기 비우기.
            reconstruction = model(input_batch)       # 새 순전파.
            loss = loss_fn(reconstruction, clean_batch)
            loss.backward()                            # 기울기 계산.
            optimizer.step()                           # 가중치·편향 갱신.

            # 샘플마다 특징 수가 같으므로 배치 샘플 수로 가중해 기록한다.
            # 마지막 작은 배치를 64개짜리 배치와 같은 비중으로 취급하지 않는다.
            total_loss += loss.item() * len(clean_batch)

        if (epoch + 1) % 5 == 0:
            print(f"epoch={epoch + 1}, train_MSE={total_loss / len(train_x):.6f}")

    # 평가용 잡음은 별도 Generator로 고정한다. 학습용 난수 흐름과 분리한다.
    generator = torch.Generator().manual_seed(123)
    if denoise:
        test_input = (
            test_x + 0.25 * torch.randn(test_x.shape, generator=generator)
        ).clamp(0, 1)
    else:
        test_input = test_x

    model.eval()
    with torch.no_grad():
        z = model.encoder(test_input)                  # (360,16).
        reconstruction = model.decoder(z)              # (360,64).
        model_mse = loss_fn(reconstruction, test_x).item()
        baseline_mse = loss_fn(test_input, test_x).item()

    print("입력 / 잠재 / 복원:", test_input.shape, z.shape, reconstruction.shape)
    print("모델 MSE:", model_mse)
    if denoise:
        print("무처리 noisy MSE:", baseline_mse)
        print("MSE 개선량:", baseline_mse - model_mse)

    return model, test_input, reconstruction, test_x

# 둘 중 원하는 과제를 호출한다. 위 함수 정의만으로 학습이 실행되지는 않는다.
ae_model, ae_input, ae_output, ae_target = run_digits(denoise=False, epochs=20)
# dae_model, dae_input, dae_output, dae_target = run_digits(denoise=True, epochs=20)
```

</details>

`loss.item()`으로 Python 숫자를 꺼내는 것은 기록용이다. 역전파는 변환 전 텐서인 `loss`에서 수행한다. 학습 중 출력한 에포크 MSE는 배치마다 갱신되는 여러 모델 상태의 손실을 모은 값이다. 에포크가 끝난 최종 모델을 학습 데이터 전체에 다시 적용한 손실과 항상 같지는 않다.

### 학습했다는 사실만으로 잡음 제거가 성공하지는 않는다

<strong>기준선(Baseline)</strong>은 모델의 성능을 판단하기 위해 비교하는 기본 결과다. DAE에서는 noisy 입력을 그대로 출력했을 때의 오차를 기준선으로 둘 수 있다.

| 비교 대상 | 계산 | 의미 |
| --- | --- | --- |
| 무처리 기준선 | `MSE(noisy,clean)` | 잡음 입력을 그대로 두었을 때의 오차 |
| 모델 결과 | `MSE(reconstruction,clean)` | 모델을 거친 출력의 오차 |
| 개선량 | `기준선 MSE−모델 MSE` | 양수면 이 평가에서 평균 오차가 감소 |

다운로드 코드의 기본 설정을 <strong>CPU에서 20에포크 실행한 결과</strong>는 다음과 같다. 학습용 1,437장과 평가용 360장을 분리했고, 모델 생성·학습 시드 42, 평가용 잡음 시드 123, 잡음 배율 0.25를 사용했다.

| 과제 | 모델의 평가 MSE | 무처리 noisy 기준선 MSE | 해석 |
| --- | ---: | ---: | --- |
| 기본 AE | 0.03209316 | 해당 없음 | clean 입력의 재구성 오차 |
| DAE | 0.03996541 | 0.03788888 | 이 설정에서는 모델 출력의 평균 오차가 더 크다. |

DAE의 개선량은 약 `−0.00207653`이다. <strong>이 실험에서는 잡음 제거 성공이라고 결론 내릴 수 없다.</strong> 학습이 실행되고 손실이 계산되었다는 사실과 기준선보다 나아졌다는 사실은 따로 확인해야 한다. AE와 DAE의 입력 조건도 다르므로 두 모델의 수치를 같은 과제의 순위처럼 읽지 않는다.

검증 환경은 Python 3.12.14, PyTorch 2.15.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0, Matplotlib 3.10.8이며 CPU 계산 스레드는 1개로 설정했다. 환경에 따라 마지막 소수점이나 학습 결과가 달라질 수 있다. 표는 이 실행에서 관찰한 결과이며 모든 실행의 보장값은 아니다.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/digits-dae-ko.svg" alt="실제 학습한 DAE의 평가 샘플 첫 네 개. 각 행은 같은 8×8 digits 원본이며 noisy 입력, 실제 모델 복원, clean 타깃을 나란히 표시한다. 전체 360장에서는 모델 MSE가 무처리 입력의 MSE보다 높다." loading="lazy">
<figcaption>좋게 나온 샘플을 고르지 않고 평가 배열의 첫 네 장을 표시했다. 세 열은 같은 밝기 범위 0~1을 사용한다. 이 그림은 MNIST 파일의 확인 화면과 별개인 실제 digits 모델 실행 결과다.</figcaption>
</figure>

### 코드를 실행할 때 확인할 것

다운로드한 `examples-ko.py`는 PyTorch가 설치된 Python 환경에서 실행한다. digits 실습에는 NumPy와 scikit-learn도 필요하며 그림을 저장하려면 Matplotlib이 필요하다. GPU를 요구하지 않는 CPU 예제다.

| 실행할 명령 | 하는 일 |
| --- | --- |
| `python examples-ko.py --mode tiny` | 작은 순전파, 한 번의 갱신, 배치 계산, Q1·Q2를 확인한다. |
| `python examples-ko.py --mode ae --epochs 20` | 기본 AE를 새로 만들어 학습·평가한다. |
| `python examples-ko.py --mode dae --epochs 20 --output-dir figures` | DAE를 학습·평가하고 결과 그림을 지정 폴더에 저장한다. |
| `python examples-ko.py --pair mnist_day1_pair.npz --output-dir figures` | 별도로 준비한 NPZ의 입력·타깃 표시도 실행한다. |

`--mode`는 과제 선택, `--epochs`는 학습 순회 횟수, `--output-dir`는 결과 저장 폴더, `--pair`는 선택적인 NPZ 경로다. 명령 이름이 `python3`인 환경이라면 같은 인터프리터 환경을 가리키는 `python3`로 실행한다. 이 표의 명령은 Bash와 zsh에서 같은 형태로 사용할 수 있다.

평가 결과를 보면서 모델 크기나 학습 횟수를 반복해서 고르면 그 데이터는 <strong>검증 데이터(Validation Data)</strong>로 사용한 것이다. 최종 성능을 보고할 때는 그 선택 과정에 쓰지 않은 <strong>테스트 데이터(Test Data)</strong>를 따로 둔다. 한 쌍의 이미지가 검증 관찰용으로 분리되어 있다면 그것을 그대로 학습에 쓰는 순간 원래의 검증 역할도 바뀐다.

## 16. 두 가중치 조건의 결과를 비교한다

이제 입력은 같고 가중치가 다른 두 조건을 계산해 보자. 이름을 Q1과 Q2라고 하겠다. 앞의 A로 한 번 갱신한 실험과는 별개의 조건이다.

공통 입력과 타깃은 `B=[2,1]`이다. 샘플 하나에 특징 두 개가 있고 구조는 `2→1→2`다. 편향·활성화 함수 없이 두 복원 성분의 MSE를 계산한다.

| 조건 | 인코더 가중치 `e` | 디코더 가중치 `d` |
| --- | --- | --- |
| Q1 | `[0.2,0.4]` | `[1.0,1.5]` |
| Q2 | `[0.38,0.49]` | `[1.096,1.484]` |

<strong>Q2에서는 인코더와 디코더가 모두 다르다.</strong> Q2의 잠재값을 구한 뒤 Q1의 디코더 `[1.0,1.5]`를 적용하면 주어진 Q2를 계산한 것이 아니다.

계산 순서는 앞과 같다. 먼저 두 입력 특징에 인코더 가중치를 각각 곱해 더한다. 그 잠재값에 디코더 가중치를 각각 곱해 두 복원 성분을 만든다. 마지막으로 타깃의 같은 성분과 비교한다.

| 계산 단계 | Q1 | Q2 |
| --- | --- | --- |
| 잠재값 `z=e₁x₁+e₂x₂` | `0.2×2+0.4×1=0.8` | `0.38×2+0.49×1=1.25` |
| 첫 복원 `d₁z` | `1.0×0.8=0.8` | `1.096×1.25=1.37` |
| 둘째 복원 `d₂z` | `1.5×0.8=1.2` | `1.484×1.25=1.855` |
| 첫 성분 제곱오차, 타깃 2 | `(0.8−2)²=1.44` | `(1.37−2)²=0.3969` |
| 둘째 성분 제곱오차, 타깃 1 | `(1.2−1)²=0.04` | `(1.855−1)²=0.731025` |
| 두 성분 평균 MSE | 0.74 | 0.5639625 |

첫 복원은 `0.8`에서 `1.37`로 바뀌어 타깃 2에 가까워졌다. 둘째 복원은 `1.2`에서 `1.855`로 바뀌어 타깃 1에서 멀어졌다. 첫 성분의 제곱오차 감소가 둘째 성분의 증가보다 커서 평균은 줄었다.

따라서 결론은 <strong>같은 입력 B에서 Q2 조건의 평균 재구성 오차가 더 작다</strong>는 것이다. 모든 출력 성분이 개선되었다거나, 다른 입력에서도 Q2가 더 좋다거나, 잠재값이 클수록 좋은 모델이라고 결론 내릴 수는 없다.

또한 가중치가 달라져도 특정 입력에서 변화가 상쇄되어 같은 출력이 나올 수 있다. 가중치가 바뀌었다는 사실만으로 끝내지 않고 실제 복원과 오차를 계산해야 한다. 여기서는 제공된 두 조건의 순전파를 비교했다. 그 가중치가 어떤 학습 과정에서 만들어졌는지는 이 계산만으로 알 수 없다.

## 17. Rust에서도 같은 순전파를 계산한다

Python과 Rust는 같은 인코더·디코더 관계와 MSE 정의를 사용할 수 있다. 차이는 아래 예제의 구현 방식이다. Python의 PyTorch 텐서는 자동 미분을 이용했고, 아래 Rust 코드는 표준 라이브러리의 고정 길이 배열과 실수 연산만 사용한다. 따라서 <strong>이 Rust 예제는 순전파를 계산하며 자동 미분이나 학습은 수행하지 않는다.</strong> Rust에서도 자동 미분을 제공하는 라이브러리를 선택하면 학습을 구현할 수 있다.

`[f64;2]`는 64비트 실수 두 개를 담는 배열이다. 함수 반환형 `(f64,[f64;2],f64)`는 잠재값, 복원 배열, MSE를 하나의 <strong>튜플(Tuple)</strong>로 반환한다는 뜻이다. `.powi(2)`는 정수 지수 2로 제곱한다.

```rust
// 별도 파일로 실행하는 Rust 순전파 예제다. 외부 라이브러리는 사용하지 않는다.
fn tiny_ae(
    x: [f64; 2],
    e: [f64; 2],
    d: [f64; 2],
) -> (f64, [f64; 2], f64) {
    // 특징값 두 개에 인코더 가중치를 곱하고 더한다.
    let z = e[0] * x[0] + e[1] * x[1];

    // 같은 z로 복원 성분 두 개를 각각 계산한다.
    let reconstruction = [d[0] * z, d[1] * z];

    // 대응하는 두 성분의 제곱오차를 평균한다.
    let mse = (
        (reconstruction[0] - x[0]).powi(2)
        + (reconstruction[1] - x[1]).powi(2)
    ) / 2.0;

    (z, reconstruction, mse)
}

fn main() {
    let x = [2.0, 1.0];
    let conditions = [
        ("Q1", [0.2, 0.4], [1.0, 1.5]),
        ("Q2", [0.38, 0.49], [1.096, 1.484]),
    ];

    for (name, e, d) in conditions {
        let (z, reconstruction, mse) = tiny_ae(x, e, d);
        // :.3과 :.7은 표시할 소수점 자릿수이며 원래 계산값을 바꾸지 않는다.
        println!("{name}: z={z:.3}, reconstruction={reconstruction:?}, MSE={mse:.7}");
    }
}
```

Q1의 MSE는 `0.7400000`, Q2의 MSE는 `0.5639625`로 Python과 같다. 복원 배열을 출력할 때는 부동소수점 표현 때문에 끝에 작은 오차가 보일 수 있다. 다운로드한 Rust 파일에는 이 결과를 확인하는 검산도 포함되어 있다.

## 18. 실제 적용할 때 확인할 것

모델을 바꾸기 전에 먼저 데이터와 계산의 연결을 확인하자.

| 확인할 질문 | 코드에서 확인할 위치 |
| --- | --- |
| 입력의 첫 축은 샘플인가, 채널인가? | 데이터를 만든 코드, `shape`, 실제 인덱싱 |
| 인코더 출력과 디코더 입력이 연결되는가? | `z=encoder(x)`, `decoder(z)` |
| DAE의 손실 타깃이 clean인가? | `loss_fn(reconstruction,clean_batch)` |
| 계산 그래프를 중간에 끊지 않았는가? | 손실 계산 중 `tensor(...)`, `.item()`, `detach()` 사용 위치 |
| 기울기 계산과 갱신을 모두 수행했는가? | `zero_grad`, `backward`, `step`의 순서 |
| 이전 결과를 새 모델 결과로 오해하지 않았는가? | 갱신 후 새 순전파, 파일 로딩 성공 여부 |
| 학습에 쓰지 않은 데이터에서 나아졌는가? | 분할 기준, 모델 MSE와 noisy 기준선 |
| 그림이 실제 모델 출력인가? | clean 배열 표시와 `model(input)` 결과의 구분 |

작은 모델에서는 입력·잠재값·복원값·손실을 직접 따라가고, 실제 모델에서는 그 연결을 배치와 평가 코드로 유지하면 된다. 다음 모델을 시도할 때도 가장 먼저 확인할 것은 입력과 타깃의 대응, 손실이 연결된 경로, 그리고 기준선과 비교한 결과다.

## 참고 문서

계산 예제는 본문의 입력·가중치 조건으로 검산했으며, API와 개념 설명은 아래 원문을 확인했다. 문서 확인일은 2026-11-02다.

[^ae]: Goodfellow, Bengio, Courville, [Deep Learning — Chapter 14: Autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html). 인코더·디코더, 표현 학습, 작은 병목과 다른 제약 방식.
[^dae]: Vincent 외, [Stacked Denoising Autoencoders](https://www.jmlr.org/papers/v11/vincent10a.html), JMLR, 2010. 손상된 입력의 재구성을 학습 목표로 사용하는 원 연구.
[^keras]: Keras, [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder/). noisy 입력과 clean 타깃을 연결한 학습 예제.
[^linear]: PyTorch, [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html). 생성자, 입력·출력 shape, 가중치의 축 순서.
[^mse]: PyTorch, [MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html). 원소별 제곱오차와 기본 평균 범위.
[^tensor]: PyTorch, [torch.tensor](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html). 복사 생성과 자동 미분 기록, NumPy 공유 텐서에 대한 안내.
[^stack]: PyTorch, [torch.stack](https://docs.pytorch.org/docs/stable/generated/torch.stack.html). 새 축을 추가해 텐서들을 묶는 연산.
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html). 연산 기록, 기울기 누적, 평가 모드와 미분 모드.
[^no-grad]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html). 기울기 계산을 위한 기록을 생략하는 문맥 관리자.
[^detach]: PyTorch, [Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html). 자동 미분 연결 분리와 저장 공간 공유.
[^clone]: PyTorch, [torch.clone](https://docs.pytorch.org/docs/stable/generated/torch.clone.html). 미분 가능한 복사 연산.
[^numpy]: NumPy, [numpy.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html). NPZ 읽기, 파일 닫기, `allow_pickle` 설정.
[^imshow]: Matplotlib, [Axes.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html). 밝기 범위, 원점, 표시 좌표와 보간.
[^plot]: Matplotlib, [Axes.axhline](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhline.html), [Axes.annotate](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html). 데이터 좌표와 표시 좌표의 구분.
[^digits]: scikit-learn, [load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). 샘플 수, 이미지 크기, 픽셀 범위.
[^embedding]: Sentence Transformers, [Semantic Textual Similarity](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html). 임베딩 생성과 유사도 계산의 연결.
[^compression]: TensorFlow Compression, [Learned data compression](https://tensorflow.github.io/compression/). 학습 기반 압축의 비트열과 엔트로피 부호화.
