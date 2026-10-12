+++
title = "자동차 연비 예측으로 이해하는 전처리·교차검증·Ridge"
description = "자동차의 무게와 생산 지역으로 연비를 예측하며 fit·transform·predict, 데이터 누수, 교차검증, Grid Search와 refit을 연결한다. 표준화의 의미부터 손실의 미분과 L1·L2 규제까지 단계별로 정리한다."
date = 2026-10-12T19:22:35+09:00
slug = "car-mpg-pipeline"

[extra]
katex = true
toc = true
styles = ["car-mpg-pipeline/article.css"]
+++

자동차의 무게와 생산 지역을 알고 있을 때, 그 자동차의 연비를 예측하려고 한다. 과거 자동차의 정보를 모델에 넣어 학습하고 새 자동차를 예측하면 될 것 같지만, 그 사이에 확인해야 할 일이 있다. 무게를 변환하는 기준은 어디서 구할까? 검증 자료도 변환해야 할까? 여러 설정을 비교한 뒤에는 어떤 모델로 예측할까?

이 글에서는 <strong>학습 자료에서 전처리 기준과 예측 계수를 구하고, 평가 자료와 새 입력에는 그 기준과 계수를 적용한다</strong>는 흐름을 중심으로 과정을 연결한다. 먼저 전체 작업을 살펴보고, 그 안에서 반복되는 표의 변환과 평가 방법을 설명한다. 뒤에서는 스케일링과 규제가 계수 학습에 영향을 주는 이유를 수식으로 확인한다.

바탕이 된 학습 기록에는 수업 결과와 수업용 함수가 포함되어 있다. 그 기록을 해석하는 부분과, 글 끝의 <strong>가상 자동차 자료로 직접 실행하는 예제</strong>를 구분했다. 원본 수업 데이터와 함수 본문을 확인하지 못한 부분은 별도로 표시한다.

## 1. 자동차 한 대의 정보가 예측값이 되기까지

<strong>샘플(sample)</strong>은 데이터 한 건이다. 여기서는 한 행이 자동차 한 대를 나타낸다. <strong>특성(feature)</strong>은 모델에 입력하는 정보이며, <strong>정답 또는 목표값(target)</strong>은 모델이 맞히도록 하는 실제 값이다.

| 이름 | 이 문제에서의 의미 | 용도 |
| --- | --- | --- |
| `weight` | 자동차의 무게 | 숫자의 크기와 차이를 입력으로 사용한다. |
| `origin` | 생산 지역을 구별하는 코드 | 지역을 구별하는 입력으로 사용한다. |
| `mpg` | 연료 1갤런당 주행한 마일 수로 나타낸 연비 | 모델이 예측할 정답이다. |
| 행 `index` 또는 `row_id` | 어느 자동차의 기록인지 확인하는 표식 | 입력과 정답의 대응을 확인한다. 이 예제의 특성에는 넣지 않는다. |

무게의 단위와 지역 코드의 실제 지명은 데이터 설명에서 확인해야 한다. 학습 기록에 없는 단위나 지명을 임의로 붙이지 않는다. 글 끝의 독립 실행 예제에서는 무게를 <strong>lb(파운드)</strong>로 정하고, 지역은 이름 대신 코드 1·2·3으로 구분한다.

입력 표는 `X`, 실제 연비는 `y`, 모델이 계산한 연비는 `y_pred` 또는 수식의 $\hat y$로 쓴다. <strong>실제 값 `y`와 예측값 `y_pred`는 서로 다른 값</strong>이다. 두 값의 차이를 확인해야 예측이 얼마나 틀렸는지 알 수 있다.

예측할 때 아직 모르는 `mpg`를 입력 `X`에 넣으면, 모델이 실제 사용 시점에 받을 수 없는 정보를 학습에 사용하게 된다. 따라서 `weight`와 `origin`은 입력으로, `mpg`는 정답으로 분리한다.

### 학습 자료와 평가 자료의 역할을 먼저 고정한다

<strong>학습</strong>은 입력과 정답의 관계를 구하는 과정이고, <strong>검증</strong>은 그 모델의 학습에 쓰지 않은 자료에서 예측 오차를 확인하는 과정이다. <strong>교차검증(cross-validation, CV)</strong>은 학습 자료 안에서 훈련 부분과 검증 부분을 바꾸어 이 과정을 반복한다. 여러 설정 중 하나를 선택한 뒤 전체 학습 자료로 다시 학습하는 것을 <strong>최종 재학습(refit)</strong>이라고 한다.

아래에서는 이름을 다음과 같이 사용한다.

| 이름 | 포함하는 내용 | 사용하는 단계 |
| --- | --- | --- |
| `train_frame`, `y_train` | 모델을 만들기 위한 입력과 정답 | 교차검증, 설정 선택, 최종 재학습 |
| 교차검증의 훈련 부분 | `train_frame` 안에서 이번 회차의 학습에 사용하는 일부 | 이번 회차의 전처리 기준과 계수 계산 |
| 교차검증의 검증 부분 | `train_frame` 안에서 이번 회차의 평가에 남긴 일부 | 이번 회차의 예측 오차 계산 |
| `valid_frame`, `y_valid` | 처음부터 따로 남긴 입력과 정답 | 설정을 선택한 뒤의 별도 최종 평가 |
| `sample_request` | 실제 연비가 아직 없는 새 자동차의 입력 | 학습된 모델로 예측 |

이 글에서 `valid_frame`은 이름에 `valid`가 들어 있지만 <strong>교차검증 내부의 검증 묶음과는 다른 자료</strong>다. 이 구분을 끝까지 유지해야 점수를 혼동하지 않는다.

다음 흐름도에서 `transform`은 저장된 전처리 기준을 입력에 적용하는 단계이고, `predict`는 학습한 계수로 예측값을 계산하는 단계다. <strong>MSE(평균제곱오차)</strong>는 예측값과 실제 값의 차이를 제곱해 평균낸 값이다. 같은 평가 자료와 같은 목표값 단위로 비교할 때 작을수록 예측 오차가 작다.

<figure class="mpg-figure">
  <img src="/car-mpg-pipeline/workflow-ko.svg" alt="과거 자동차 자료를 학습 자료와 최종 평가 자료로 나눈다. 학습 자료 안에서 후보별 교차검증으로 설정을 선택하고 전체 학습 자료로 다시 학습한다. 최종 평가 자료와 새 자동차는 저장된 전처리 기준과 계수로 예측한다." loading="lazy" />
  <figcaption>최종 평가 자료는 설정 선택과 재학습에 들어가지 않는다. 실제 정답이 있는 자료에서만 예측 오차를 계산할 수 있다.</figcaption>
</figure>

## 2. fit은 기준을 구하고, transform은 그 기준을 적용한다

<strong>전처리</strong>는 모델에 넣을 입력의 표현을 준비하는 작업이다. 여기서는 무게의 숫자 크기를 조정하고, 지역 코드를 계산에 사용할 여러 컬럼으로 바꾼다. 숫자의 기준이나 크기를 바꾸는 전처리를 <strong>스케일링(scaling)</strong>이라고 한다.

`StandardScaler`는 스케일링을 수행하는 객체를 만드는 <strong>클래스</strong>다. `StandardScaler()`를 호출하면 객체가 만들어지지만, 아직 어떤 자동차 자료에서도 평균을 구하지 않았다.

객체에서 호출하는 기능을 <strong>메서드</strong>라고 한다. `fit`, `transform`, `predict`는 어떤 객체에 호출하는지에 따라 아래와 같이 동작한다.[^scaler][^pipeline]

| 호출 | 하는 일 | 이 예제에서의 반환값 |
| --- | --- | --- |
| `scaler.fit(X_train)` | 학습 입력에서 평균·분산 등의 변환 기준을 구해 저장한다. | 기준이 저장된 스케일러 객체 자신 |
| `scaler.transform(X)` | 저장한 기준으로 입력을 변환한다. | 변환된 배열 |
| `scaler.fit_transform(X_train)` | 기준을 구한 뒤 학습 입력도 그 기준으로 변환한다. | 변환된 학습 배열 |
| `model.fit(X_train, y_train)` | 입력과 정답의 관계를 나타내는 계수를 학습한다. | 학습된 모델 객체 자신 |
| `model.predict(X_new)` | 저장된 계수를 사용해 예측값을 계산한다. | 자동차마다 연비 하나가 담긴 배열 |

따라서 `train_scaled = scaler.fit(X_train)`이라고 쓰면 `train_scaled`에는 변환된 입력이 들어가지 않는다. 스케일러 객체가 들어간다. <strong>변환된 값이 필요하면 `transform`이나 `fit_transform`의 반환값을 받아야 한다.</strong>

### 검증 자료도 변환한다

학습 입력을 표준화한 뒤 모델을 학습했다면, 검증 입력도 표준화해서 전달해야 한다. 차이는 <strong>기준을 어디에서 구하느냐</strong>에 있다.

1. 스케일러가 학습 자료에서 기준을 구한다.
2. 스케일러가 그 기준으로 학습 자료를 변환한다.
3. 모델이 변환된 학습 자료와 정답으로 계수를 구한다.
4. 같은 스케일러가 저장된 기준으로 검증 자료를 변환한다.
5. 모델이 저장된 계수로 검증 자료를 예측한다.

검증 자료에서 평균과 표준편차를 새로 구하면, 같은 원래 무게가 학습 때와 다른 숫자로 표현될 수 있다. 그런데 모델의 계수는 학습 때의 표현을 기준으로 구한 값이다. <strong>입력의 기준만 바꾸고 계수를 그대로 쓰면, 학습한 입력과 출력의 관계가 맞지 않게 된다.</strong>[^pitfalls]

또한 학습 자료와 검증 자료를 합쳐 전처리 기준을 구하면, 검증 자료의 정보가 학습 입력의 변환에 영향을 준다. 이렇게 평가에 남겨 두어야 할 정보가 학습에 들어가는 문제를 <strong>데이터 누수(data leakage)</strong>라고 한다. 누수가 있으면 점수가 반드시 좋아진다는 뜻은 아니다. 평가하려고 정한 조건을 지키지 못했다는 것이 문제다.[^pitfalls]

## 3. 표준화된 무게는 무엇을 뜻하는가

표준화는 <strong>학습 평균과의 차이를 학습 표준편차를 기준으로 표현하는 변환</strong>이다. 이 문장을 이해하려면 평균·분산·표준편차가 각각 무엇을 계산하는지 먼저 확인해야 한다.

### 평균에서 얼마나 떨어져 있는지 계산한다

이 절에서 $x_i$는 $i$번째 학습 자동차의 무게, $n$은 학습 자동차 수다. $\mu$는 학습 평균, $\sigma^2$는 학습 분산, $\sigma$는 학습 표준편차를 뜻한다. $\sum_{i=1}^{n}$은 $i=1$부터 $n$까지의 항을 모두 더하라는 기호다.

<div class="mpg-math">
$$
\mu=\frac{1}{n}\sum_{i=1}^{n}x_i
$$
$$
\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2,
\qquad \sigma=\sqrt{\sigma^2}
$$
</div>

평균은 무게를 모두 더해 자동차 수로 나눈 값이다. 분산을 구할 때는 각 무게에서 평균을 빼고, 그 차이를 제곱한 뒤 평균낸다. 차이를 그대로 더하면 평균보다 큰 값의 양수와 작은 값의 음수가 상쇄되기 때문이다.

분산에 제곱근을 취한 값이 표준편차다. 제곱근은 제곱했을 때 해당 값이 되는 수를 구하는 연산이다. 무게의 단위가 lb라면 분산의 단위는 lb²이고, 표준편차의 단위는 다시 lb다. 제곱근을 취하면 퍼짐을 원래 무게와 같은 단위로 표현할 수 있다. 퍼짐을 표현하는 방법이 이것 하나뿐이라는 뜻은 아니다.

### 표준편차로 나누면 상대적인 위치를 알 수 있다

변환할 무게를 $x$, 표준화한 결과를 $z$라고 쓰겠다. 학습 표준편차가 양수일 때 변환식은 다음과 같다.[^scaler]

<div class="mpg-math">
$$
z=\frac{x-\mu}{\sigma}
\quad\Longleftrightarrow\quad
x-\mu=z\sigma
\quad\Longleftrightarrow\quad
x=\mu+z\sigma
$$
</div>

첫 번째 식의 양변에 $\sigma$를 곱하면 두 번째 식이 된다. 두 번째 식의 양변에 $\mu$를 더하면 세 번째 식이 된다. 따라서 $z$는 <strong>원래 무게가 평균에서 표준편차의 몇 배만큼 위나 아래에 있는지</strong> 나타낸다.

| 결과 | 의미 |
| --- | --- |
| $z=0$ | 무게가 학습 평균과 같다. |
| $z=1$ | 무게가 학습 평균보다 학습 표준편차 하나만큼 크다. |
| $z=-2$ | 무게가 학습 평균보다 학습 표준편차 두 개만큼 작다. |

$x-\mu$는 음수가 될 수 있는 <strong>방향이 있는 차이</strong>다. 양수인 $\sigma$로 나누어도 그 방향은 유지된다. 같은 무게 차이라도 표준편차가 크면 $|z|$는 작고, 표준편차가 작으면 $|z|$는 크다. 여기서 $|z|$는 부호를 제외한 크기, 즉 절댓값이다.

이 계산에서는 lb를 lb로 나누므로 결과에 무게 단위가 남지 않는다. 다만 모든 나눗셈에서 단위가 없어지는 것은 아니다. 같은 종류의 단위를 가진 양끼리 나누었기 때문에 가능한 해석이다.

### 표준화는 값을 특정 범위에 가두지 않는다

`StandardScaler`의 기본 설정에서, 분산이 0이 아닌 학습 컬럼을 그 컬럼의 기준으로 변환하면 평균은 0, 표준편차는 1이 된다. 실제 계산에서는 소수의 반올림 오차가 있을 수 있다.

하지만 값이 모두 $-1$과 $1$ 사이에 들어가는 것은 아니다. 검증 자료는 학습 기준을 적용받을 뿐이므로, <strong>검증 자료 자체의 평균과 표준편차가 0과 1이 될 필요도 없다.</strong> 표준화가 자료의 분포 모양을 정규분포로 바꾸는 것도 아니다. 정규분포는 가운데 값 부근에 자료가 많이 모이는 특정한 확률분포이며, 평균을 빼고 일정한 수로 나누는 것만으로 모든 자료가 그 분포가 되지는 않는다.[^preprocessing]

`StandardScaler`가 분산을 구할 때 사용하는 분모는 위 식처럼 $n$이다. NumPy 표기로는 `ddof=0`이다. `ddof`는 분모에서 빼는 수를 뜻한다. 예를 들어 `ddof=1`이면 분모가 $n-1$이다. 표본으로 전체 모집단의 분산을 추정할 때 쓰는 $n-1$ 기준과, 지금 입력의 변환 기준을 계산하는 방식을 섞어 읽지 않는다.[^scaler][^std]

모든 학습 값이 같은 상수 컬럼은 분산이 0이다. 이 경우 `StandardScaler`는 변환에서 나누는 값인 `scale_`을 1로 둔다. 기본 설정대로 평균까지 빼면 해당 학습 컬럼의 값은 모두 0이 된다.[^scaler]

### MinMax는 학습 최솟값과 최댓값을 기준으로 바꾼다

`MinMaxScaler`도 변환 객체를 만드는 클래스다. 기본 설정은 학습 최솟값을 0, 학습 최댓값을 1로 표현한다. 학습 최솟값을 $m_{\mathrm{train}}$, 최댓값을 $M_{\mathrm{train}}$이라고 하면, 두 값이 다를 때 다음 식을 사용한다.[^minmax]

<div class="mpg-math">
$$
x_{\mathrm{scaled}}=
\frac{x-m_{\mathrm{train}}}{M_{\mathrm{train}}-m_{\mathrm{train}}}
$$
</div>

새 무게가 학습 최댓값보다 크면 결과가 1을 넘을 수 있다. 학습 최솟값보다 작으면 0보다 작을 수 있다. 기본값인 `clip=False`는 이런 값을 강제로 0과 1 사이에 자르지 않는다. 학습 기준을 그대로 적용한 결과다.

어느 방법을 사용해도 변환 전후의 행은 같은 자동차다. 예를 들어 학습 기록에서 `row_id=199`를 전후에 비교했다면, 그 값은 같은 자동차의 무게를 서로 다른 기준으로 표현한 것이다. 변환된 무게는 연비 예측값이나 손실값이 아니다.

## 4. 지역 코드를 변환하고, 같은 자동차의 정보를 연결한다

`origin`의 1·2·3은 지역을 구별하는 이름 역할을 한다. 코드 3이 코드 1보다 세 배 큰 지역이라는 뜻은 아니다. 따라서 각 지역에 해당하는지를 0과 1로 표시하는 컬럼을 만든다. 이를 <strong>원-핫 인코딩(one-hot encoding)</strong>이라고 한다.[^onehot]

| 원래 `origin` | `origin_1` | `origin_2` | `origin_3` |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 |
| 2 | 0 | 1 | 0 |
| 3 | 0 | 0 | 1 |

지역 코드는 하나였지만, 계산에 넣을 컬럼은 세 개가 되었다. 자동차 행을 합계나 평균으로 집계한 것은 아니다. <strong>같은 자동차 한 행에 지역 표시 컬럼을 추가한 것</strong>이다.

### Series, DataFrame, 배열의 모양을 구분한다

Pandas의 <strong>Series</strong>는 값 한 열과 그 값의 행 이름인 `index`를 가진다. <strong>DataFrame</strong>은 행 `index`와 컬럼 이름을 가진 표다. NumPy의 일반적인 숫자 <strong>배열</strong>은 값과 모양을 가지지만, Pandas의 행 이름과 컬럼 이름을 함께 보관하지는 않는다.

`shape`는 각 방향으로 값이 몇 개 있는지 나타내는 속성이다. 자동차 수를 $n$이라고 하면 다음 선택은 서로 다르다.[^structures]

| 표현 | 결과 자료형 | `shape` |
| --- | --- | --- |
| `train_frame["weight"]` | Series | `(n,)` |
| `train_frame[["weight"]]` | DataFrame | `(n, 1)` |
| `train_frame[["weight"]].to_numpy(dtype=float)` | 실수형 NumPy 배열 | `(n, 1)` |
| `train_frame[["weight", "origin"]]` | DataFrame | `(n, 2)` |

`["weight"]`라는 <strong>컬럼 이름 목록</strong>을 선택에 전달하면 컬럼이 하나여도 표로 남는다. 그래서 대괄호가 두 겹처럼 보인다. `to_numpy`는 배열로 바꾸는 메서드이고, `dtype=float`는 원소를 실수형으로 표현한다는 설정이다.

학습 코드에서 `X_train_num`은 무게 같은 숫자형 입력, `train_cat`은 지역 같은 범주형 입력을 가리킬 수 있다. `valid_row_ids`는 별도 평가 행의 원래 `index`를 보관하는 용도다. 변수 이름이 비슷해도 목적이 다르다.

### category 지정은 원-핫 인코딩과 다른 작업이다

`pd.CategoricalDtype`은 가능한 범주와 범주의 순서 여부를 정하는 클래스다. `categories=[1, 2, 3]`은 허용된 범주 목록, `ordered=False`는 범주 사이에 대소 순서를 부여하지 않는다는 뜻이다.

이 자료형을 `astype` 메서드로 `origin`에 적용해도 아직 컬럼은 하나다. 그다음 <strong>함수</strong>인 `pd.get_dummies`를 호출해야 위 표처럼 범주별 컬럼이 만들어진다. `columns=["origin"]`은 변환할 컬럼, `dtype=int`는 표시 값을 정수 0과 1로 만들겠다는 설정이다.[^categorical][^dummies]

미리 같은 범주 목록을 지정하면, 평가 자료에 지역 3이 없어도 `origin_3` 컬럼을 만들 수 있다. 학습과 평가의 컬럼 구성을 맞추는 데 도움이 된다. 여기서 1·2·3은 <strong>문제에서 미리 정한 코드 목록</strong>이라고 가정한다. 평가 자료를 살펴본 뒤 처리 규칙을 고르는 과정과 구분해야 한다.

범주형으로 바꾸는 작업이 원-핫 인코딩의 필수 단계는 아니다. 정수형 컬럼도 `get_dummies`의 `columns`에 직접 지정할 수 있다. 반대로 지정하지 않은 값은 범주형 변환에서 결측값이 될 수 있으며, `get_dummies`의 기본 설정에서는 그 행의 범주 표시가 모두 0이 될 수 있다. 변환 전에 허용한 코드인지 확인해야 한다.[^categorical][^dummies]

### concat은 행의 위치보다 index를 기준으로 맞춘다

표준화한 무게 배열에 원래 `index`를 붙여 `train_weight_frame`이라는 DataFrame을 만들고, 지역 표 `train_onehot`과 옆으로 결합한다고 하자.

`pd.concat`은 여러 표를 결합하는 함수다. `axis=1`에서는 <strong>같은 행 index를 맞추고 컬럼을 옆으로 붙인다.</strong> 이때 행을 평균내거나 줄이지 않는다.[^concat]

| 결합 전 무게 표 | 결합 전 지역 표 | 올바르게 결합한 결과 |
| --- | --- | --- |
| 자동차 $n$대 × 무게 1열 | 같은 자동차 $n$대 × 지역 3열 | 같은 자동차 $n$대 × 입력 4열 |
| `(n, 1)` | `(n, 3)` | `(n, 4)` |

배열로 DataFrame을 새로 만들 때 `index`를 생략하면 0부터 시작하는 기본 행 번호가 붙는다. 원래 지역 표의 행 이름이 101·205·309라면, Pandas는 0번째 행과 101번 행을 같은 자동차로 보지 않는다. 그 결과 행이 늘거나 결측값이 생길 수 있다. 행 이름을 잘못 붙인 경우에는 다른 자동차의 정보가 연결될 수도 있다.

그래서 <strong>변환한 배열이 원래 행 순서를 유지하는지 확인한 다음, 그 순서에 맞는 원래 index를 붙인다.</strong> 모델 입력과 `y_train`도 같은 순서인지 확인한다. Pandas가 표를 결합할 때 정렬해 준다는 사실과, scikit-learn이 입력과 정답을 자동차 이름으로 자동 재연결해 준다는 주장은 다르다. 이 예제의 학습에서는 입력의 각 행과 정답의 각 원소가 같은 위치에서 대응해야 한다.[^validation]

글 끝의 수동 전처리 코드는 이 모양과 대응을 확인하는 실습이다. <strong>이렇게 전체 학습 자료로 미리 변환한 `X_train_ready`를 교차검증에 전달하지 않는다.</strong> 다음 절에서는 각 회차 안에서 전처리를 새로 학습하도록 구성한다.

## 5. Pipeline으로 전처리와 모델을 함께 학습한다

<strong>선형회귀</strong>는 각 입력에 계수를 곱한 값을 더하고 절편을 더해 수량을 예측하는 모델이다. 입력이 하나이면 예측식은 $\hat y=wx+b$다. $w$는 입력에 곱하는 <strong>계수</strong>, $b$는 마지막에 더하는 <strong>절편</strong>이다. 학습은 이 값을 데이터에 맞게 정하는 과정이다.[^linear]

여기서 사용할 <strong>Ridge</strong>는 계수의 크기에 추가 비용을 주는 선형회귀다. 예측 오차만 줄이는 대신 계수가 커지는 비용도 함께 고려한다. 이 비용을 추가하는 방법을 <strong>규제 또는 규제화(regularization)</strong>라고 하며, `Ridge(alpha=1.0)`의 `alpha`는 그 강도를 정한다. `alpha`는 경사하강법의 학습률이 아니다. 규제의 계산 원리는 뒤에서 미분을 설명한 다음 연결한다.[^ridge]

<strong>Pipeline</strong>은 전처리와 모델을 정해진 순서로 실행하도록 묶는 클래스다. <strong>ColumnTransformer</strong>는 서로 다른 컬럼에 각기 다른 변환을 적용한 뒤 결과를 결합하는 클래스다. <strong>OneHotEncoder</strong>는 학습 시 정해진 범주 구성을 이후 변환에도 적용하는 클래스다.[^pipeline][^column-transformer][^onehot]

이번 Pipeline은 다음 구성을 사용한다.

| 원래 입력 | 담당 객체 | 변환 결과 |
| --- | --- | --- |
| `weight` | `StandardScaler` | 표준화한 무게 1열 |
| `origin` | `OneHotEncoder` | 지역 표시 3열 |
| 두 변환 결과 | `ColumnTransformer` | 같은 자동차마다 입력 4열 |
| 변환된 입력과 정답 | `Ridge` | 입력별 계수와 절편을 학습 |

`cv_model.fit(train_frame, y_train)`을 호출하면 전처리가 학습 자료에서 기준을 구하고, 같은 자료를 변환한 뒤, Ridge가 변환된 입력과 정답으로 계수를 학습한다.

`cv_model.predict(sample_request)`를 호출하면 저장된 전처리 기준으로 새 입력을 변환하고, 저장된 계수로 예측한다. <strong>예측할 때 전처리 기준과 모델 계수를 다시 학습하지 않는다.</strong>[^pipeline]

### 구성한 코드와 학습을 실행한 코드는 다르다

`Pipeline([...])`을 실행한 것은 처리 구성을 만든 것이다. 아직 데이터를 사용해 학습한 것은 아니다. 전처리는 Ridge가 알아서 추가하는 것이 아니라, Pipeline을 구성한 코드에서 정한다.

학습 기록에 등장하는 `make_model(Ridge(alpha=1.0), include_origin=True)`는 수업에서 제공한 함수 호출이다. 기록의 설명에 따르면 이 함수가 전처리와 모델을 묶는다. 다만 실제 함수 본문은 제공되지 않았으므로, 내부 단계의 이름이나 모든 옵션을 확인했다고 말할 수는 없다.

이 글의 전체 코드는 `make_model`을 재현한 원문이 아니라 <strong>처리 구성을 직접 드러낸 별도 구현</strong>이다. `preprocess`, `weight`, `origin`, `model`이라는 단계 이름도 이 구현에서 정한다. 이 이름을 알아야 나중에 저장된 스케일러와 계수를 찾아볼 수 있다.

`OneHotEncoder`에는 사전에 정한 `categories=[[1, 2, 3]]`을 전달한다. 바깥 목록은 입력 컬럼별 목록이고, 안쪽 목록은 `origin` 한 컬럼의 범주다. `sparse_output=False`는 여기서 일반적인 2차원 배열을 반환하도록 하고, `handle_unknown="error"`는 정한 범주 밖의 값이 오면 오류를 내도록 한다.[^onehot]

## 6. 교차검증에서는 회차마다 전처리부터 다시 학습한다

어떤 모델이 좋은지 판단하려고 한 번만 자료를 나누면, 그 나누기에 따라 결과가 달라질 수 있다. <strong>교차검증(cross-validation, CV)</strong>은 학습 자료 안에서 훈련과 검증에 사용할 부분을 바꾸며 학습·평가를 반복하는 방법이다.[^cv-guide]

학습 기록에서는 자동차 318대를 A·B·C 세 묶음으로 나눴다. 각 묶음이 106대라면 회차별 역할은 다음과 같다. A·B·C는 역할을 설명하기 위한 이름이다.

| 회차 | 전처리와 모델을 학습하는 자동차 | 해당 회차에서 예측을 평가하는 자동차 |
| --- | --- | --- |
| 1회차 | A + B, 212대 | C, 106대 |
| 2회차 | A + C, 212대 | B, 106대 |
| 3회차 | B + C, 212대 | A, 106대 |

<strong>앞 회차의 학습된 모델을 이어 쓰지 않는다.</strong> 각 회차에서 같은 설정의 새 모델을 학습한다. C가 1회차에서 검증 자료이고 2회차에서 훈련 자료여도 괜찮은 이유다. 1회차의 모델은 C로 학습하지 않았고, 2회차의 모델은 별도로 학습한다.

1회차의 스케일러도 A+B에서만 평균과 표준편차를 구해야 한다. 처음부터 A+B+C 전체로 평균을 구해 두면, C의 정보가 1회차의 훈련 입력에 들어간다. 그래서 <strong>전처리까지 포함한 Pipeline 전체</strong>를 교차검증에 전달한다.[^pitfalls][^cv]

`cross_validate`는 이 반복을 실행하는 <strong>함수</strong>다. 입력으로 모델 구성, 원래 입력 표, 정답, 분할 규칙과 평가 방법을 받는다. 여기서는 결과를 회차별 점수와 시간이 담긴 <strong>딕셔너리</strong>로 반환한다. 딕셔너리는 이름인 키와 그 이름에 해당하는 값을 연결해 보관하는 자료구조다.[^cv]

| 결과의 키 | 저장된 내용 | 세 회차일 때의 모양 |
| --- | --- | --- |
| `train_score` | 각 회차의 훈련 점수. `return_train_score=True`로 요청한다. | `(3,)` |
| `test_score` | 각 회차의 검증 점수 | `(3,)` |
| `fit_time` | 각 회차의 학습 시간 | `(3,)` |
| `score_time` | 각 회차의 검증 점수 계산 시간 | `(3,)` |

여기서 `test_score`의 `test`는 교차검증 내부의 검증 부분을 뜻한다. 처음부터 별도로 남긴 `valid_frame`을 평가한 값이 아니다. 또한 `cross_validate`는 복제한 모델들로 회차별 실험을 수행한다. 호출 후 원래 `cv_model` 자체가 전체 학습 자료로 최종 학습되어 있다고 가정하면 안 된다.[^cv][^validation]

전체 코드에서는 `KFold`로 세 분할을 만들고 `cv_splits`에 목록으로 저장한다. `KFold`는 각 묶음을 한 번씩 검증에 쓰도록 분할하는 클래스다. `shuffle=True`는 나누기 전에 순서를 섞고, 정수 `random_state=42`는 같은 입력에서 분할을 재현하도록 한다. 42라는 수 자체에 특별한 성능상의 의미는 없다. 목록을 재사용하므로 후보마다 같은 자동차들로 비교한다.[^kfold]

이 예제는 자동차 행을 개별 관측으로 취급한다. 시간순 기록이나 같은 대상의 반복 측정처럼 분할에 따로 고려할 구조가 있는 데이터는 그 구조에 맞게 나누어야 한다. Pipeline이 잘못된 분할이나 Pipeline 밖에서 이미 발생한 누수까지 자동으로 해결하지는 않는다.[^cv-guide]

## 7. MSE와 음수 점수를 같은 기준으로 읽는다

<strong>MSE(평균제곱오차)</strong>는 각 예측의 오차를 제곱한 뒤 평균낸 값이다. $i$번째 자동차의 정답을 $y_i$, 예측값을 $\hat y_i$, 오차를 $e_i$라고 하면 다음과 같다.[^mse]

<div class="mpg-math">
$$
e_i=\hat y_i-y_i,
\qquad
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}e_i^2
$$
</div>

예를 들어 <strong>설명을 위해 만든</strong> 두 예측을 비교해 보자. 첫 자동차의 실제 연비가 20 mpg이고 예측이 18 mpg라면 오차는 $18-20=-2$다. 두 번째 자동차의 실제 연비가 25 mpg이고 예측이 26 mpg라면 오차는 $26-25=1$이다.

| 자동차 | 실제 연비 | 예측 연비 | 오차 | 오차의 제곱 |
| --- | ---: | ---: | ---: | ---: |
| A | 20 | 18 | −2 | 4 |
| B | 25 | 26 | 1 | 1 |

오차 제곱의 평균은 $(4+1)/2=2.5$다. 이것이 MSE다. 단위는 mpg²이므로, 평균적으로 2.5 mpg 틀렸다는 뜻은 아니다. 같은 정답과 같은 평가 자료에서 비교할 때 MSE가 작을수록 제곱오차가 작다.

<strong>학습 MSE</strong>는 학습한 자료에서 계산한 값이고, <strong>검증 MSE</strong>는 해당 모델의 학습에 사용하지 않은 부분에서 계산한 값이다. 학습 자료에는 잘 맞지만 새 자료에는 잘 맞지 않는 현상을 <strong>과적합(overfitting)</strong>이라고 한다. 학습 MSE는 매우 작은데 검증 MSE가 크면 과적합을 의심할 수 있지만, 점수 차이만으로 원인을 확정할 수는 없다.[^cv-guide]

### neg_mean_squared_error는 오차가 아니라 점수의 부호를 바꾼다

scikit-learn의 모델 비교용 <strong>scorer</strong>는 큰 점수를 좋은 점수로 취급한다. MSE는 작을수록 좋으므로, `scoring="neg_mean_squared_error"`는 MSE에 마이너스를 붙인 점수를 반환한다.[^scoring]

| 후보 | 실제 MSE | 비교용 점수 `−MSE` | 판단 |
| --- | ---: | ---: | --- |
| A | 20 | −20 | B보다 예측 오차가 크다. |
| B | 10 | −10 | 더 작은 MSE, 더 큰 비교용 점수다. |

위 표 역시 부호를 설명하기 위한 예다. $-10$은 $-20$보다 큰 수이므로, scikit-learn도 B를 좋은 후보로 판단한다. <strong>예측 오차 자체를 줄인 것이 아니라, 비교 규칙에 맞게 점수 표현만 바꾼 것</strong>이다.

`cv_result["test_score"]`에 마이너스를 다시 붙이면 각 원소가 실제 MSE로 돌아온다. 따라서 `cv_table`에 담은 `-cv_result["train_score"]`와 `-cv_result["test_score"]`는 다시 <strong>작을수록 좋다</strong>고 읽는다. 모든 평가 점수에 마이너스를 붙이는 것이 아니라, 지금 음수 MSE를 요청했기 때문에 되돌리는 것이다.

### Baseline은 비교의 출발점을 정하는 모델이다

<strong>Baseline</strong>은 이후 모델이 얼마나 개선됐는지 판단하기 위해 정한 기준 모델이다. 학습 기록에서는 무게 하나로 예측하는 `LinearRegression`을 기준으로 삼았다. `baseline`이라는 변수 이름 자체에 특별한 기능은 없다.

입력 특성이나 모델 설정을 바꾼 뒤에도 같은 분할과 같은 지표로 평가해야 비교가 가능하다. 아래 전체 코드도 무게만 사용하는 기준 모델과, 무게·지역을 사용하는 Ridge를 비교한다. 이 비교에서는 입력과 모델이 함께 바뀌므로 <strong>성능 차이를 규제 하나의 효과로 해석하지 않는다.</strong>

기록할 내용은 점수 하나뿐이 아니다. 데이터 분할, 입력 컬럼, 전처리, 모델 설정, 평가 지표를 함께 남겨야 결과를 다시 확인할 수 있다. 이후 같은 예측을 재사용하려면 전처리와 모델을 함께 보관한다. 성능 비교를 위해 항상 학습된 객체까지 저장해야 한다는 뜻은 아니다.

## 8. Grid Search로 설정을 고르고 refit으로 최종 모델을 만든다

모델이 학습하는 값과 사람이 후보로 지정하는 값을 구분해야 한다. <strong>모델 파라미터</strong>는 데이터로 학습하는 계수 $w$와 절편 $b$다. <strong>하이퍼파라미터</strong>는 개별 학습을 시작하기 전에 정하는 설정이며, Ridge의 `alpha`가 여기에 해당한다.

<strong>Grid Search</strong>는 미리 지정한 여러 설정을 각각 평가해 좋은 설정을 선택하는 방법이다. `GridSearchCV`는 각 후보를 교차검증으로 평가하는 탐색 객체를 만드는 클래스다.[^grid]

여기서는 `alpha` 후보를 `[0.01, 0.1, 1, 10, 100]`으로 정한다. 각 후보를 같은 세 분할에서 평가하고, 평균 검증 MSE가 가장 작은 후보를 선택한다. 음수 MSE 점수로 표현하면 평균 점수가 가장 큰 후보를 고르는 일이다.

### model__alpha는 어느 단계의 설정을 바꾸는지 나타낸다

Pipeline의 마지막 단계 이름을 `model`로 정했기 때문에 `model__alpha`는 <strong>그 단계 안의 `alpha`</strong>를 가리킨다. 밑줄 두 개 `__`가 단계 이름과 설정 이름을 구분한다. 앞부분의 이름을 다르게 정했다면 그 이름을 사용해야 한다.[^pipeline]

| 이름 | 의미 |
| --- | --- |
| `param_grid` | 비교할 설정과 후보 값을 담은 딕셔너리 |
| `cv_results_` | 후보별 점수와 설정 등이 담긴 결과 딕셔너리 |
| `param_model__alpha` | 결과표에 기록된 각 후보의 `alpha` |
| `mean_test_score` | 해당 후보의 평균 교차검증 점수. 여기서는 음수 MSE |
| `best_params_` | 선택한 설정을 담은 딕셔너리 |
| `best_score_` | 선택한 후보의 평균 교차검증 점수 |
| `best_estimator_` | `refit=True`인 이 예제에서 전체 학습 자료로 다시 학습한 최종 Pipeline |

`search.cv_results_`, `search.best_params_`, `search.best_score_`, `search.best_estimator_`는 객체에 저장된 <strong>속성</strong>이다. 메서드처럼 괄호를 붙여 호출하지 않고 값을 읽는다. 반면 `param_model__alpha`와 `mean_test_score`는 `cv_results_` 딕셔너리의 <strong>키</strong>이며, 결과를 DataFrame으로 바꾸면 컬럼 이름이 된다. `param_grid`는 탐색 객체를 만들 때 전달하는 설정이다. scikit-learn은 학습 후 생기는 속성 이름에 마지막 밑줄 `_`을 사용하는 경우가 많다.[^grid]

### refit은 선택에 사용한 모델 하나를 그냥 꺼내는 일이 아니다

<strong>refit</strong>은 선택한 설정으로 `search.fit`에 전달했던 전체 학습 자료를 다시 학습하는 단계다. `refit=True`로 구성하고 `search.fit(train_frame, y_train)`을 실행하면, 후보 평가가 끝난 뒤 이 재학습까지 수행한다.[^grid]

| 시점 | 스케일러가 기준을 구하는 자료 | 모델이 계수를 학습하는 자료 |
| --- | --- | --- |
| 교차검증 한 회차 | 그 회차의 훈련 부분 | 그 회차의 훈련 부분 |
| 설정 선택 후 refit | `train_frame` 전체 | `train_frame` 전체 |
| 최종 평가와 새 예측 | 기준을 다시 구하지 않는다. | 계수를 다시 학습하지 않는다. |

학습 기록의 수를 적용하면, 교차검증 중에는 회차별 훈련 자동차 212대로 전처리와 모델을 학습한다. 설정을 선택한 뒤에는 `train_frame`의 318대 전체로 다시 학습한다. 이후 `valid_frame`과 새 자동차에는 <strong>318대에서 구한 최종 기준과 계수</strong>를 적용한다.

후보 5개를 세 회차로 평가하면 Pipeline 학습은 $5\times3=15$회다. 모든 후보가 정상적으로 실행되고 마지막 refit 한 회가 추가되면 `search.fit` 한 호출에서 합계 16회가 된다. 이것은 계수를 16번 갱신했다는 뜻이 아니라, <strong>Pipeline 전체를 학습하는 작업이 16번 실행되었다는 뜻</strong>이다. 앞에서 따로 실행한 `cross_validate`의 학습은 이 횟수에 포함하지 않는다.

`selected_model = search.best_estimator_`는 이미 재학습된 객체를 참조하는 대입이다. 이 한 줄에서 새 학습을 실행하지 않는다. `selected_model.predict(valid_frame)`은 그 객체로 예측하며, `search.predict(valid_frame)`도 재학습된 최종 모델을 사용한다.[^grid]

### 저장된 값은 그 값을 학습한 객체에서 확인한다

`selected_model.named_steps["preprocess"]`는 학습된 Pipeline에서 전처리 단계를 찾는다. 여기서 `named_transformers_["weight"]`로 무게 변환기를 찾으면 `mean_`과 `scale_`을 읽을 수 있다. 마지막 `model` 단계에서는 `coef_`와 `intercept_`를 읽는다.

`mean_`은 저장된 평균, `scale_`은 변환할 때 나누는 값, `coef_`는 입력별 계수, `intercept_`는 절편이다. 이 예제의 Ridge에는 변환된 입력 4열의 계수 4개와 절편 하나가 있다. <strong>처음 구성할 때 만들었던 객체를 추측해서 읽기보다, 최종 선택된 객체에서 실제 저장값을 확인한다.</strong>[^scaler][^ridge][^column-transformer]

### 학습 기록에 있는 수치를 해석한다

다음 값은 제공된 학습 기록의 출력이다. 원본 수업 데이터와 함수 본문을 다시 실행해서 계산한 결과는 아니다.

| 기록된 결과 | 읽을 수 있는 의미 |
| --- | --- |
| 선택된 설정 `{"model__alpha": 10}` | 지정한 후보들과 분할 조건에서 이 설정이 선택되었다. |
| 평균 검증 MSE 약 `19.9297` | 설정 선택에 사용한 교차검증 평균 MSE다. |
| `weight=2800`, `origin=2`의 예측 약 `25.6714 mpg` | 최종 모델이 해당 입력에 대해 계산한 예측값이다. |

`19.9297`은 별도 평가 자료의 MSE가 아니다. 별도 평가 MSE를 알려면 최종 모델의 `valid_frame` 예측과 실제 `y_valid`를 비교해야 한다. `25.6714 mpg`도 실제 연비가 없으면 정확도를 계산할 수 없는 예측값이다.

선택된 `alpha=10`은 지정한 후보 가운데 선택한 값이다. 가능한 모든 실수 값 중 절대적으로 가장 좋은 `alpha`를 찾았다는 뜻은 아니다. 또한 최종 평가 점수를 반복해서 확인하며 설정을 고치면, 그 자료도 선택 과정에 영향을 준다. 그때는 처음 의도한 독립적인 최종 평가 자료의 역할을 유지했다고 볼 수 없다.[^cv-guide]

## 9. 손실의 미분값은 예측 직선의 계수와 다르다

지금까지는 라이브러리에 학습을 맡겼다. 이제 입력을 스케일링하는 일이 계수 학습에 영향을 주는 이유를 살펴보자. 설명을 위해 <strong>입력이 무게 하나인 선형회귀</strong>로 범위를 줄인다. 지역 컬럼까지 포함한 전체 모델의 내부 계산을 그대로 펼치는 것은 아니다.

<strong>함수</strong>는 입력에 따라 출력을 정하는 규칙이고, <strong>함수값</strong>은 특정 입력을 넣어 계산한 결과다. 예측식 $\hat y=wx+b$는 무게 $x$를 예측 연비로 바꾸는 규칙이다. 손실함수는 현재 계수가 얼마나 나쁜 예측을 만드는지를 숫자로 계산하는 규칙이다.

미분은 한 값을 아주 조금 바꾸었을 때 다른 값이 얼마나 빠르게 변하는지를 구한다. 여기서는 입력 무게가 아니라 <strong>학습 중 바꾸는 계수에 따른 손실의 순간 변화율</strong>을 구하려고 한다. 다른 값을 고정하고 한 변수의 영향만 구하는 미분을 <strong>편미분</strong>이라고 쓰며, 기호는 $\partial$다.

이 절에서는 자동차 한 대의 손실을 $\ell$이라고 쓰고, 여러 자동차의 평균 손실을 $J_{\mathrm{data}}$라고 쓴다. $\ell$은 소문자 L에 해당하는 손실 기호다.

<div class="mpg-math">
$$
\hat y=wx+b,\qquad e=wx+b-y,\qquad \ell(e)=\frac{e^2}{2}
$$
$$
e_i=wx_i+b-y_i,\qquad
J_{\mathrm{data}}=\frac{1}{n}\sum_{i=1}^{n}\ell(e_i)
=\frac{1}{2n}\sum_{i=1}^{n}e_i^2
$$
</div>

<strong>$J_{\mathrm{data}}$는 MSE의 절반</strong>이다. 뒤의 미분에서 제곱 때문에 생기는 2와 분모의 2가 맞물리도록 이 기준을 사용한다. 결과표에서 읽는 MSE와 학습 원리를 설명하는 $J_{\mathrm{data}}$를 혼동하지 않는다.

| 기호 | 여기서 뜻하는 것 |
| --- | --- |
| $w$ | 입력 $x$에 곱하는 예측 계수 |
| $dw=\partial J_{\mathrm{data}}/\partial w$ | 현재 $w$에서, $w$에 대한 평균 손실의 미분값을 담는 코드 변수 이름 |
| $db=\partial J_{\mathrm{data}}/\partial b$ | 현재 $b$에서, $b$에 대한 평균 손실의 미분값을 담는 코드 변수 이름 |
| $\eta$ | 계수를 갱신할 때 미분값에 곱하는 양수인 학습률 |

여기서 코드 이름 `dw`를 아주 작은 $w$의 변화량이라는 뜻으로 사용하지 않는다. 또한 $d\ell/de$는 $\ell$을 $e$로 나누라는 명령이 아니라, <strong>$e$에 대한 손실의 미분값을 나타내는 표기 전체</strong>다.

### w가 변하면 오차 e는 왜 x만큼의 비율로 변할까

목적은 손실 $\ell=(wx+b-y)^2/2$를 $w$에 대해 미분하는 것이다. 그 안에서 $w$가 먼저 오차 $e$를 바꾸므로, 이 관계부터 구한다. 이 계산 동안에는 $x$, $b$, $y$를 고정한다. $b$를 학습 내내 고정한다는 뜻은 아니며, $b$의 영향은 따로 계산한다.

$h$를 <strong>계수 $w$에 더하는 변화량</strong>이라고 정의하자. 변화량으로 나누는 동안에는 $h\ne0$이다.

<div class="mpg-math">
$$
e(w)=wx+b-y,
\qquad e(w+h)=(w+h)x+b-y
$$
$$
\begin{aligned}
e(w+h)-e(w)
&=(w+h)x+b-y-(wx+b-y)\\
&=wx+hx+b-y-wx-b+y\\
&=hx
\end{aligned}
$$
</div>

두 번째 줄에서는 곱셈을 펼치고, 괄호 앞의 마이너스를 괄호 안의 모든 항에 적용했다. 세 번째 줄에서는 $wx$와 $-wx$, $b$와 $-b$, $-y$와 $y$가 각각 상쇄되어 $hx$가 남는다.

오차의 변화량을 $w$의 변화량 $h$로 나누면 다음과 같다.

<div class="mpg-math">
$$
\frac{e(w+h)-e(w)}{h}=\frac{hx}{h}=x
$$
$$
\frac{\partial e}{\partial w}
=\lim_{h\to0}\frac{e(w+h)-e(w)}{h}=x
$$
</div>

$\lim_{h\to0}$은 $h$를 0에 가까이 줄였을 때 식이 어떤 값에 가까워지는지를 뜻한다. 여기서는 $h$를 줄여도 비율이 계속 $x$이므로 미분값도 $x$다. 분모에 곧바로 0을 대입해서 나눈 것이 아니다.

$b$에 더하는 변화량을 $k$라고 하면 같은 방식으로 계산할 수 있다. 이번에는 $w$, $x$, $y$를 고정하고 $k\ne0$으로 둔다.

<div class="mpg-math">
$$
e(b+k)-e(b)
=wx+(b+k)-y-(wx+b-y)=k
$$
$$
\frac{\partial e}{\partial b}
=\lim_{k\to0}\frac{k}{k}=1
$$
</div>

절편에 더한 만큼 오차에도 그대로 더해지므로 변화율이 1이다.

### 오차가 변하면 손실은 왜 e의 비율로 변할까

이번에는 손실을 오차 하나의 함수 $\ell(e)=e^2/2$로 본다. $t$를 <strong>오차 $e$에 더하는 변화량</strong>이라고 정의하고, 나누는 동안 $t\ne0$으로 둔다.

<div class="mpg-math">
$$
\begin{aligned}
\frac{\ell(e+t)-\ell(e)}{t}
&=\frac{(e+t)^2/2-e^2/2}{t}\\
&=\frac{e^2/2+et+t^2/2-e^2/2}{t}\\
&=\frac{et+t^2/2}{t}\\
&=e+\frac{t}{2}
\end{aligned}
$$
</div>

첫 줄에는 변화 전후의 오차를 손실식에 넣었다. 둘째 줄은 $(e+t)^2=e^2+2et+t^2$를 펼친 결과다. 셋째 줄에서는 $e^2/2$끼리 상쇄했고, 마지막 줄에서는 남은 두 항을 각각 $t$로 나눴다.

<div class="mpg-math">
$$
\frac{d\ell}{de}
=\lim_{t\to0}\left(e+\frac{t}{2}\right)=e
$$
</div>

마지막에는 $t/2$가 0에 가까워져 $e$가 남는다. 분모의 2를 이유 없이 없앤 것이 아니다.

### 두 변화율을 연결하면 손실의 미분값이 된다

계수 $w$의 변화는 <strong>$w\rightarrow e\rightarrow\ell$</strong> 순서로 손실에 영향을 준다. 변화량을 나타내는 기호 $\Delta$를 사용하면, $\Delta w$와 $\Delta e$가 0이 아닌 경우 다음 등식이 성립한다.

<div class="mpg-math">
$$
\frac{\Delta\ell}{\Delta w}
=\frac{\Delta\ell}{\Delta e}\times\frac{\Delta e}{\Delta w}
$$
</div>

오른쪽에서 같은 변화량 $\Delta e$가 분자와 분모에 있기 때문에 왼쪽과 같다. 이처럼 연결된 관계의 순간 변화율을 곱해 구하는 규칙이 <strong>연쇄법칙(chain rule)</strong>이다. 방금 구한 두 미분값을 연결하면 다음 결과를 얻는다.

<div class="mpg-math">
$$
\frac{\partial\ell}{\partial w}
=\frac{d\ell}{de}\frac{\partial e}{\partial w}
=e\times x
$$
$$
\frac{\partial\ell}{\partial b}
=\frac{d\ell}{de}\frac{\partial e}{\partial b}
=e\times1=e
$$
</div>

이 설명은 미분 기호의 $d$나 $\partial$를 보통 숫자처럼 약분한다는 뜻이 아니다. $x=0$이라서 $e$가 $w$에 따라 변하지 않는 경우에는 손실도 $w$에 따라 변하지 않는다. 이때의 미분값 0 역시 최종식 $ex$와 일치한다.

### 자동차가 여러 대이면 각 손실의 미분값을 평균낸다

$J_{\mathrm{data}}$는 자동차별 손실을 모두 더해 $n$으로 나눈 함수다. $w$를 바꿀 때 각 자동차의 손실도 각각 바뀌므로, <strong>각 손실의 변화량을 더하고 같은 $n$으로 나누면 평균 손실의 변화량</strong>이 된다. 이를 같은 계수 변화량으로 나누고 변화량을 0에 가까이 줄이면, 다음처럼 각 미분값의 평균이 된다. $n$은 이 미분 동안 고정된 자동차 수다.

<div class="mpg-math">
$$
\begin{aligned}
dw=\frac{\partial J_{\mathrm{data}}}{\partial w}
&=\frac{1}{n}\sum_{i=1}^{n}\frac{\partial\ell(e_i)}{\partial w}\\
&=\frac{1}{n}\sum_{i=1}^{n}e_ix_i
\end{aligned}
$$
$$
\begin{aligned}
db=\frac{\partial J_{\mathrm{data}}}{\partial b}
&=\frac{1}{n}\sum_{i=1}^{n}\frac{\partial\ell(e_i)}{\partial b}\\
&=\frac{1}{n}\sum_{i=1}^{n}e_i
\end{aligned}
$$
</div>

이 평균은 <strong>여러 자동차의 손실을 평균낸 함수의 현재 지점 미분값</strong>이다. 서로 멀리 떨어진 두 계수 값 사이의 평균 변화율을 구한 것이 아니다. MSE 자체는 $2J_{\mathrm{data}}$이므로, MSE 자체를 미분하면 위 결과의 두 배가 된다.

### 경사하강법은 미분값의 반대 방향으로 계수를 바꾼다

<strong>경사하강법(gradient descent)</strong>은 현재 계수에서 손실의 미분값을 구한 뒤 그 반대 방향으로 계수를 갱신하는 방법이다. 이 글에서 학습률은 $\eta$로 표기한다.

<div class="mpg-math">
$$
w_{\mathrm{new}}=w-\eta dw,
\qquad b_{\mathrm{new}}=b-\eta db
$$
</div>

$dw>0$이면 현재 지점 부근에서 $w$가 증가할 때 손실도 증가하는 방향이므로 $w$를 줄인다. $dw<0$이면 $w$가 증가하는 쪽에서 손실이 감소하는 방향이므로 $w$를 늘린다. 한 번에 움직이는 폭은 학습률과 미분값의 크기에 따라 결정된다.

미분값은 현재 지점 부근의 정보다. 학습률을 너무 크게 정하면 그 방향으로 움직여도 새 손실이 더 커질 수 있다. 따라서 반대 부호로 갱신한다는 사실만으로 모든 학습률에서 손실 감소가 보장되는 것은 아니다.

한 번의 반복에서는 현재 계수로 예측하고, 오차와 손실을 계산한 다음, 미분값을 구해 계수를 갱신한다. 예측이나 손실을 계산하는 단계 자체가 미분은 아니다. 실제 코드는 미리 유도해 둔 미분 공식에 현재 값을 넣어 계산하고, 새 계수로 이 과정을 반복한다.

### 스케일링과 발산 기록을 연결한다

$dw$에는 $e_ix_i$가 들어간다. 입력의 숫자 표현이 바뀌면 이 계산과 현재 오차가 달라지고, 같은 학습률에서의 갱신 폭과 안정성도 달라질 수 있다. 다만 샘플별 항이 상쇄될 수도 있으므로, 입력이 커지면 전체 $dw$가 언제나 커진다고 단정하면 안 된다.

학습 기록에는 학습률 0.1을 사용한 다음 출력이 있다. 이 표의 수치와 중단 조건은 <strong>기록에 제공된 결과와 설명</strong>이며, 이 글의 가상 데이터 실행 결과가 아니다.

| 입력 표현 | `last_step` | 기록된 학습 MSE | 상태와 의미 |
| --- | ---: | ---: | --- |
| 원본 무게 `Raw` | 3 | 약 $3.261805\times10^{38}$ | `mse_guard_stopped`: 정한 MSE 한계 $10^{30}$을 넘어 중단했다. |
| `MinMax` | 2000 | 약 19.78192 | `iteration_budget_completed`: 지정한 최대 갱신 횟수를 사용했다. |
| 표준화 | 2000 | 약 19.78192 | `iteration_budget_completed`: 지정한 최대 갱신 횟수를 사용했다. |

원본 입력은 처음부터 세 번만 반복하도록 지정한 것이 아니라, 손실이 급격히 커져 중단된 것이다. 나머지 두 방법도 2000회를 완료했다고 해서 정확한 최솟값에 도달했다고 말할 수는 없다. 마지막 학습 MSE가 비슷하다고 수렴 속도나 별도 평가 성능까지 같다는 뜻도 아니다.

기록에는 `run_gd`가 MSE/2의 미분값으로 갱신하고 결과표에는 MSE를 표시한다고 설명되어 있다. 수업 함수 본문은 확인하지 못했으므로, 이 설명과 실제 구현을 대조하려면 해당 함수의 소스를 확인해야 한다.

## 10. L1·L2는 계수를 제곱해서 덮어쓰는 연산이 아니다

이제 <strong>규제가 계수 학습에 추가하는 것이 무엇인지</strong> 살펴보자. 예측 오차를 줄이는 과정에서 계수의 크기에도 비용을 주려면, 계수 크기를 계산한 값을 학습 목표에 더한다. 모델은 그 합을 줄이도록 계수를 정한다.[^linear]

입력이 하나일 때 예측식은 여전히 $\hat y=wx+b$다. 오차로 구한 평균 손실은 앞에서 정의한 $J_{\mathrm{data}}$이고, $\lambda$는 추가 비용의 강도를 조절하는 0 이상의 값이다.

<div class="mpg-math">
$$
J_{\mathrm{L1}}=J_{\mathrm{data}}+\lambda|w|
$$
$$
J_{\mathrm{L2}}=J_{\mathrm{data}}+\lambda w^2
$$
</div>

<strong>L1</strong>은 계수의 절댓값에 비용을 주고, <strong>L2</strong>는 계수의 제곱에 비용을 준다. 예측식을 $\hat y=w^2x+b$로 바꾼 것도 아니고, 저장된 $w$를 $w^2$으로 덮어쓴 것도 아니다. <strong>최소화할 목표에 새 항을 추가한 것</strong>이다.

계수가 $p$개라면 각 계수에 대한 비용을 더한다. 여기서 $p$는 변환을 마친 모델 입력 컬럼 수이고, $j$는 그 컬럼의 번호다.

<div class="mpg-math">
$$
P_{\mathrm{L1}}=\sum_{j=1}^{p}|w_j|,
\qquad
P_{\mathrm{L2}}=\sum_{j=1}^{p}w_j^2
$$
</div>

L2는 <strong>각 계수를 제곱한 뒤 더한다.</strong> 계수를 먼저 모두 더한 뒤 제곱하는 계산과 다르다. 이 규제식에서는 절편 $b$를 패널티에서 제외한다.

### L2 항이 어떻게 계수 갱신에 들어가는가

계수 하나의 L2 비용 $\lambda w^2$를 $w$에 대해 미분해 보자. 앞 절처럼 $h$를 $w$의 변화량으로 두고, $\lambda$는 고정한다.

<div class="mpg-math">
$$
\begin{aligned}
\frac{d(\lambda w^2)}{dw}
&=\lim_{h\to0}\frac{\lambda(w+h)^2-\lambda w^2}{h}\\
&=\lim_{h\to0}\frac{2\lambda wh+\lambda h^2}{h}\\
&=\lim_{h\to0}(2\lambda w+\lambda h)\\
&=2\lambda w
\end{aligned}
$$
</div>

제곱을 펼치면 $\lambda w^2$끼리 상쇄되고, 남은 항을 $h$로 나누면 $2\lambda w+\lambda h$다. $h$가 0에 가까워지면 뒤의 항은 0에 가까워진다. 따라서 오차항의 미분값에 <strong>$2\lambda w$가 추가</strong>된다.

<div class="mpg-math">
$$
\frac{\partial J_{\mathrm{L2}}}{\partial w}
=\frac{\partial J_{\mathrm{data}}}{\partial w}+2\lambda w
$$
$$
\begin{aligned}
w_{\mathrm{new}}
&=w-\eta\left(\frac{\partial J_{\mathrm{data}}}{\partial w}+2\lambda w\right)\\
&=w-2\eta\lambda w-\eta\frac{\partial J_{\mathrm{data}}}{\partial w}\\
&=(1-2\eta\lambda)w-\eta\frac{\partial J_{\mathrm{data}}}{\partial w}
\end{aligned}
$$
</div>

첫 줄의 괄호에 $-\eta$를 곱한 뒤, $w$를 포함한 두 항을 묶었다. L2가 추가하는 방향만 보면 $w>0$일 때는 양수를 빼고, $w<0$일 때는 음수를 빼므로 $w$를 0 쪽으로 움직이게 한다.

L2 항만 따로 본 한 번의 갱신에서 $0<2\eta\lambda<1$이면, $w$에 0보다 크고 1보다 작은 수를 곱하므로 부호를 유지하면서 절댓값이 줄어든다. 그러나 실제 갱신에는 오차항의 미분값도 들어 있다. <strong>모든 반복에서 모든 계수의 절댓값이 반드시 줄어드는 것은 아니다.</strong> 학습률이 너무 크면 규제를 추가했어도 학습이 불안정해질 수 있다.

이 식은 경사하강법으로 L2의 작용을 설명한 것이다. `Ridge`가 반드시 이 반복식으로 학습한다는 뜻은 아니다. Ridge는 설정한 계산 방법에 따라 다른 수치적 방법으로도 계수를 구한다.[^ridge]

### L1과 L2가 다른 계수를 선택할 수 있는 이유

| 방법 | 추가 비용 | 학습 결과에서 볼 수 있는 특징 |
| --- | --- | --- |
| L1 | 계수의 절댓값을 합한다. | 일부 계수를 정확히 0으로 만들 수 있다. |
| L2 | 계수의 제곱을 합한다. | 계수 크기를 억제하면서 여러 계수를 남기는 경향이 있다. |

L1에서 계수가 0이 되면 그 입력 항은 해당 선형 예측식에 기여하지 않는다. 다만 선택된 특성이 현실의 원인이라는 뜻은 아니다. L2도 어떤 계수가 0이 될 가능성을 금지하는 것은 아니다.[^linear]

$|w|$는 $w=0$에서 왼쪽과 오른쪽의 변화율이 달라 일반적인 미분값이 존재하지 않는다. 따라서 L1에서 정확한 0이 나오는 과정을 “보통 경사하강법으로 움직이면 무조건 0이 된다”고 설명하면 안 된다. Lasso는 L1 목표에 맞는 계산 방법으로 계수를 구한다.[^lasso]

### LinearRegression, Ridge, Lasso의 예측식과 학습 목표

같은 입력을 사용하면 세 모델은 계수를 곱해 더하는 선형 예측식의 형태를 공유한다. 차이는 학습할 계수를 결정하는 목표에 있다.[^linear][^ridge][^lasso]

| 클래스 | 학습 목표 |
| --- | --- |
| `LinearRegression` | 별도의 L1·L2 패널티 없이 오차 제곱의 합을 줄인다. |
| `Ridge` | 오차 제곱의 합에 L2 패널티를 더한 목표를 줄인다. |
| `Lasso` | 평균을 반영한 오차항에 L1 패널티를 더한 목표를 줄인다. |

`Ridge`는 L2 비용만 계산하는 객체가 아니다. 계수 학습, 예측, 설정과 계산 방법 관리까지 담당하는 모델 객체다. 규제도 Ridge에서만 사용할 수 있는 개념이 아니다. 다른 모델이나 직접 작성한 학습 목표에도 추가할 수 있다.

또한 <strong>학습 목표와 평가 점수는 구분해야 한다.</strong> Ridge가 학습에서 줄이는 값에는 L2 비용이 포함되지만, 이 글의 `mean_squared_error`와 교차검증 MSE는 예측 오차만 계산한다. 출력된 MSE에 L2 비용이 포함된 것은 아니다.[^ridge][^mse]

### 수식의 λ와 Ridge의 alpha는 크기 기준도 확인해야 한다

앞에서 정의한 $J_{\mathrm{data}}$는 오차 제곱의 합을 $2n$으로 나눈 값이다. 반면 scikit-learn Ridge의 공식 목표는 다음과 같은 크기 기준을 사용한다.[^ridge]

<div class="mpg-math">
$$
\sum_{i=1}^{n}e_i^2
+\texttt{alpha}\sum_{j=1}^{p}w_j^2
$$
</div>

이 전체를 양수인 $2n$으로 나누어도 어느 계수가 목표를 최소로 만드는지는 바뀌지 않는다. 같은 상수로 모든 후보의 목표값을 나누기 때문이다. 나눈 식은 다음과 같다.

<div class="mpg-math">
$$
J_{\mathrm{data}}
+\frac{\texttt{alpha}}{2n}\sum_{j=1}^{p}w_j^2
$$
</div>

따라서 <strong>이 글의 L2 수식과 Ridge를 대응시킬 때는 $\lambda=\texttt{alpha}/(2n)$</strong>이다. 원문의 일반식에 있는 $\lambda$와 코드의 `alpha`를 숫자까지 같은 값으로 바로 취급하면 안 된다.

Lasso의 공식 목표는 $J_{\mathrm{data}}+\texttt{alpha}\sum_j|w_j|$이므로, 이 기준에서는 $\lambda=\texttt{alpha}$다. 패널티의 모양도 다르고 오차항의 크기 기준도 다르므로, Ridge와 Lasso에 같은 숫자의 `alpha`를 넣었다고 같은 규제 효과가 되는 것은 아니다.[^lasso]

### 스케일링은 입력을, 규제는 학습 목표를 바꾼다

스케일링은 입력 $x$의 숫자 표현을 바꾼다. 규제는 계수 $w$의 크기를 고려하도록 학습 목표를 바꾼다. 입력 단위가 달라지면 같은 예측을 표현하는 데 필요한 계수의 크기도 달라지므로, 큰 계수가 언제나 나쁘다는 뜻은 아니다.

여러 숫자형 특성의 단위와 크기가 크게 다르면 규제의 작용도 그 표현에 영향을 받는다. 그래서 규제 선형모델에서는 입력의 스케일을 함께 고려한다. 단, 모든 모델에 스케일링이 필수인 것은 아니며, 규제를 강하게 할수록 검증 성능이 무조건 좋아지는 것도 아니다.[^preprocessing][^linear]

## 11. 입력 개수, 계수 개수, 그래프의 축은 서로 다르다

<strong>표현력</strong>은 모델이 나타낼 수 있는 입력과 출력의 관계 범위다. <strong>학습</strong>은 그 범위 안에서 데이터에 맞는 관계를 정하는 일이고, <strong>예측</strong>은 정해진 관계에 입력을 넣어 함수값을 계산하는 일이다.

입력이 하나인 $\hat y=wx+b$는 계수를 어떻게 정해도 직선이다. 여기에 $x^2$이라는 입력 항을 허용한 $\hat y=b+w_1x+w_2x^2$는 $w_2\ne0$일 때 2차 곡선을 표현할 수 있다. 직선만 허용한 모델을 오래 학습한다고 코드에 없던 $x^2$ 항이 자동으로 생기지는 않는다.[^polynomial]

의사결정나무 회귀는 입력 조건에 따라 서로 다른 예측값을 내놓을 수 있다. 신경망은 정해진 구조의 연산들을 연결해 관계를 표현한다. 모든 모델이 2차 곡선을 만드는 것은 아니며, 함수가 반드시 짧은 다항식으로 적혀 있어야 하는 것도 아니다.[^tree][^mlp]

### 계수가 네 개라고 입력이 네 개라는 뜻은 아니다

| 예측식 | 원래 입력변수 | 학습하는 계수와 절편 | 예측 관계를 그릴 때 |
| --- | --- | --- | --- |
| $\hat y=b+w_1x+w_2x^2+w_3x^3$ | $x$ 하나 | $w_1,w_2,w_3,b$ 네 개 | 가로축 $x$, 세로축 $\hat y$로 그릴 수 있다. |
| $\hat y=b+w_1x_1+w_2x_2+w_3x_3$ | $x_1,x_2,x_3$ 세 개 | $w_1,w_2,w_3,b$ 네 개 | 모든 입력과 출력을 동시에 좌표로 나타내려면 네 좌표가 필요하다. |

첫 식의 $x^2$과 $x^3$은 같은 $x$에서 계산한 값이다. 반면 두 번째 식은 $x_1$, $x_2$, $x_3$를 각각 별도 입력 좌표로 받는다. 실제 데이터에서 이 세 입력의 값이 서로 관련될 수는 있다. 첫 식을 모델에 넣을 표로 준비하면 변환된 입력 컬럼은 세 개지만, 그 값을 만드는 원래 변수는 하나다.

이번 자동차 모델에서도 원래 의미상 특성은 무게와 지역 두 가지다. 전처리 뒤에는 표준화한 무게 1열과 지역 표시 3열이 생겨 입력이 4열이 된다. 여기에 대응하는 계수는 4개이며 절편은 별도로 하나다. <strong>원래 특성 수, 변환 후 컬럼 수, 계수 수를 각각 세어야 한다.</strong>

### 어떤 그래프인지 먼저 축을 확인한다

| 그래프 | 바꾸거나 비교하는 값 | 세로축 | 답하려는 질문 |
| --- | --- | --- | --- |
| 예측 관계 | 입력 $x$ | 예측값 $\hat y$ | 무게가 달라지면 연비를 얼마로 예측하는가? |
| 손실함수 | 계수 $w$. 다른 계수는 고정한다. | 손실 $J$ | 계수를 바꾸면 손실이 어떻게 달라지는가? |
| 학습 진행 | 갱신 횟수 | 손실 또는 MSE | 학습을 반복하면서 오차가 어떻게 바뀌는가? |

$w$와 $b$를 함께 바꾸는 손실 $J(w,b)$는 두 변수의 함수다. $w$축, $b$축, 손실축을 쓰면 곡면으로 나타낼 수 있다. 예측 관계가 직선이어도 손실함수까지 직선일 필요는 없다. 손실함수가 2차식이어도 갱신 횟수에 따른 학습 진행 그래프가 포물선이라는 뜻은 아니다.

## 12. 수업 함수와 라이브러리 기능을 구분한다

함수를 현재 노트북에서 직접 정의하지 않았어도, 다른 파일에서 `import`하면 사용할 수 있다. 학습 기록의 `run_gd`는 `course_helpers`, `make_model`은 `cv_support`에서 가져온 수업용 함수다. `prepare_inputs`, `check_scaling` 같은 이름도 수업에서 따로 제공한 기능이며 Python이나 scikit-learn의 고정된 기본 기능이 아니다.

| 구분 | 이 글의 예 | 읽는 방법 |
| --- | --- | --- |
| 클래스 | `StandardScaler`, `Ridge`, `Pipeline`, `GridSearchCV`, `pd.DataFrame` | 호출하면 해당 역할의 객체를 만든다. |
| 함수 | `cross_validate`, `mean_squared_error`, `pd.get_dummies`, `pd.concat` | 필요한 인자를 전달해 작업을 실행하고 반환값을 받는다. |
| 메서드 | `fit`, `transform`, `predict`, `astype` | 특정 객체에서 호출하는 기능이다. |
| 속성 | `shape`, `mean_`, `scale_`, `coef_`, `best_score_` | 객체에 저장된 정보를 읽는다. |

함수 내부를 매번 전부 읽을 필요는 없다. 하지만 입력과 반환값, 기본 설정, 중단 조건은 알아야 결과를 해석할 수 있다. Python의 `inspect.getfile`은 정의된 파일 위치를, `inspect.getsource`는 접근 가능한 Python 소스 본문을 문자열로 돌려주는 함수다. 소스가 제공되지 않는 형태의 객체에서는 소스 확인이 실패할 수 있다.[^inspect]

처음의 긴 준비 코드도 역할을 나누어 읽으면 된다. `Path`는 파일 경로를 다루고, ZIP 처리는 압축한 자료를 푼다. `RESOURCE_MANIFEST`는 제공 파일의 목록과 관련 정보를 기록한 수업 쪽 자료이며, SHA-256은 파일 내용에서 계산한 값을 비교해 파일이 달라졌는지 검사하는 데 사용한다. `sys.path`에 폴더를 추가하면 Python이 그 폴더의 모듈도 찾을 수 있다. 이 작업들은 모델의 학습식이 아니라 실행에 필요한 파일과 모듈을 준비하는 과정이다.[^python-files]

그래프와 출력 코드도 학습과 구분한다. `np.linspace`는 지정한 구간에 일정 간격의 숫자를 만드는 함수다. 200개를 만든다면 선을 그릴 입력 위치가 200개인 것이지, 자동차 측정 기록을 200대 추가한 것은 아니다. `scatter`는 값을 점으로, `plot`은 값들을 선으로 표시할 때 사용한다. `head`는 표의 앞부분을 보고, `round`는 반올림한 결과를 만든다. 결과표를 반올림했다고 모델의 계수가 다시 학습되는 것은 아니다.[^plotting]

## 13. 가상 자동차 자료로 처음부터 실행하는 Python 코드

다음 코드는 <strong>수업 함수 없이 실행하는 별도 예제</strong>다. 실제 측정값 대신 처리 순서를 확인하기 위해 만든 가상 자동차 16대를 사용한다. 12대를 학습 자료, 4대를 별도 평가 자료로 나누므로 세 회차 교차검증에서는 매번 8대로 학습하고 4대로 검증한다. 앞서 해석한 수업 기록의 318대와는 다른 자료다.

필요한 라이브러리는 NumPy, Pandas, scikit-learn이다. `sparse_output`을 사용하는 코드이므로 scikit-learn 1.2 이상을 기준으로 한다. 이 글을 작성하면서 NumPy 2.3.5, Pandas 2.2.3, scikit-learn 1.8.0 환경에서 전체 코드를 실행해 확인했다. 본문의 라이브러리 동작은 2026년 10월 12일에 공식 문서로 확인했다.

`assert`는 조건이 맞는지 확인하고, 맞지 않으면 실행을 멈추는 문장이다. 여기서는 예측 점수보다 먼저 <strong>자료의 모양과 자동차의 대응이 유지되는지</strong> 확인하는 데 사용한다. `train_test_split`은 입력과 정답을 함께 나누는 함수다. `test_size=4`는 평가 자료의 비율이 아니라 네 건을 남긴다는 뜻이다.[^split]

전체 코드는 아래 순서로 이어진다. 수동 전처리 결과는 구조를 살펴보는 데만 쓰고, 교차검증부터는 원래 입력 표와 Pipeline을 사용한다.

1. 가상 자료를 만들고 입력·정답을 함께 나눈다.
2. 수동 전처리로 `shape`, 범주 컬럼, 행 대응을 확인한다.
3. 같은 분할에서 기준 모델과 Ridge를 교차검증한다.
4. `GridSearchCV`로 `alpha`를 선택하고 전체 학습 자료로 재학습한다.
5. 별도 평가 자료와 새 자동차를 예측하고 저장된 기준을 확인한다.
6. 마지막에는 Rust와 비교할 간단한 계산을 따로 실행한다.

[Python 전체 코드 내려받기](/car-mpg-pipeline/examples-ko.py)

```python
"""가상 자동차로 실행하는 전처리·교차검증·Ridge 복습 예제.

이 코드는 수업 함수의 원문을 복원하거나 수업 결과를 재현하지 않습니다.
16건의 값은 처리 순서를 확인하기 위해 직접 만든 가상 자료입니다.
weight의 단위는 lb(파운드), mpg는 갤런당 주행 마일 수입니다.
origin의 1·2·3은 가상 생산 지역 코드이며 크기나 순서를 뜻하지 않습니다.
mpg를 포함한 모든 값은 실제 자동차의 측정값이 아닙니다.
필요한 라이브러리: pandas, numpy, scikit-learn(1.2 이상).
"""

import inspect

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# 1. 한 행은 가상 자동차 한 대입니다. 정답 mpg도 예제용으로 정한 값입니다.
car_frame = pd.DataFrame(
    [
        (1900, 1, 34.0), (2050, 2, 35.0), (2200, 3, 34.5), (2300, 1, 29.8),
        (2450, 2, 30.8), (2550, 3, 30.2), (2700, 1, 25.9), (2800, 2, 28.0),
        (2900, 3, 27.5), (3050, 1, 22.8), (3150, 2, 24.5), (3300, 3, 23.4),
        (3450, 1, 19.5), (3600, 2, 20.5), (3750, 3, 19.8), (3900, 1, 16.6),
    ],
    columns=["weight", "origin", "mpg"],
    # index는 각 자동차를 확인하는 이름입니다. 모델의 입력에는 넣지 않습니다.
    index=pd.Index(range(101, 117), name="row_id"),
)
X = car_frame[["weight", "origin"]]  # 입력은 DataFrame이며 shape은 (16, 2)입니다.
y = car_frame["mpg"]  # 정답은 Series이며 shape은 (16,)입니다.

# train_test_split 함수는 입력과 정답을 같은 자동차 순서로 함께 나눕니다.
# test_size=4는 별도 평가용 4건, random_state=42는 이 분할의 재현용 설정입니다.
train_frame, valid_frame, y_train, y_valid = train_test_split(
    X, y, test_size=4, random_state=42
)
assert train_frame.shape == (12, 2) and valid_frame.shape == (4, 2)
assert train_frame.index.equals(y_train.index)
assert valid_frame.index.equals(y_valid.index)
assert train_frame.index.intersection(valid_frame.index).empty
print("가상 자료의 학습/별도 평가 shape:", train_frame.shape, valid_frame.shape)
print("입력과 정답의 index 대응을 확인했습니다.")

# 대괄호 하나는 Series, 컬럼명 목록을 전달한 이중 대괄호는 DataFrame입니다.
X_train_num = train_frame[["weight"]].to_numpy(dtype=float)
valid_row_ids = valid_frame.index.to_numpy()  # 원래 행을 확인하기 위한 정보입니다.
print("무게 Series / 표 / 배열의 shape:",
      train_frame["weight"].shape, train_frame[["weight"]].shape, X_train_num.shape)
print("별도 평가 자동차의 row_id:", valid_row_ids)

# 2. 수동 전처리는 변환 결과의 구조를 확인하는 실습입니다.
# 여기서 전체 학습 자료로 만든 X_train_ready를 아래 교차검증에 넣지 않습니다.
# StandardScaler 클래스는 평균과 분산을 저장할 스케일러 객체를 만듭니다.
scaler = StandardScaler()
fit_result = scaler.fit(train_frame[["weight"]])
assert fit_result is scaler  # fit 메서드의 반환값은 변환 배열이 아니라 객체 자신입니다.
train_weight = scaler.transform(train_frame[["weight"]])
valid_weight = scaler.transform(valid_frame[["weight"]])  # 같은 학습 기준을 씁니다.
print("표준화한 학습/별도 평가 배열 shape:", train_weight.shape, valid_weight.shape)

# MinMaxScaler도 클래스입니다. 기본 설정에서는 학습 최솟값을 0, 최댓값을 1로 만듭니다.
minmax_scaler = MinMaxScaler()
train_weight_minmax = minmax_scaler.fit_transform(train_frame[["weight"]])
valid_weight_minmax = minmax_scaler.transform(valid_frame[["weight"]])
print("학습 무게의 MinMax 범위:", train_weight_minmax.min(), train_weight_minmax.max())
print("별도 평가 무게의 MinMax 범위:", valid_weight_minmax.min(), valid_weight_minmax.max())

# CategoricalDtype 클래스에 사전에 정한 허용 코드 목록을 전달합니다.
# 이 목록은 검증 자료를 조사해 얻은 것이 아닙니다. ordered=False는 대소 순서가 없다는 뜻입니다.
origin_dtype = pd.CategoricalDtype(categories=[1, 2, 3], ordered=False)
train_cat = train_frame[["origin"]].astype({"origin": origin_dtype})
valid_cat = valid_frame[["origin"]].astype({"origin": origin_dtype})
assert not train_cat.isna().any().any() and not valid_cat.isna().any().any()

# get_dummies 함수는 지역별 0/1 컬럼을 만듭니다. dtype=int는 정수로 출력합니다.
# 어느 지역의 행이 없어도, 위에 지정한 세 범주에 대응하는 컬럼을 유지합니다.
train_onehot = pd.get_dummies(train_cat, columns=["origin"], dtype=int)
valid_onehot = pd.get_dummies(valid_cat, columns=["origin"], dtype=int)
assert train_onehot.columns.equals(valid_onehot.columns)

# 배열에는 원래 행 index가 없으므로 새 표를 만들 때 해당 자동차의 index를 붙입니다.
train_weight_frame = pd.DataFrame(
    train_weight, index=train_frame.index, columns=["weight_scaled"]
)
valid_weight_frame = pd.DataFrame(
    valid_weight, index=valid_frame.index, columns=["weight_scaled"]
)
# concat 함수의 axis=1은 같은 행 index를 맞추며 컬럼을 옆으로 붙입니다.
X_train_ready = pd.concat([train_weight_frame, train_onehot], axis=1)
X_valid_ready = pd.concat([valid_weight_frame, valid_onehot], axis=1)
assert X_train_ready.index.equals(y_train.index)
assert X_valid_ready.index.equals(y_valid.index)
assert X_train_ready.columns.equals(X_valid_ready.columns)
assert X_train_ready.shape == (12, 4) and X_valid_ready.shape == (4, 4)
print("수동 전처리 결과 shape:", X_train_ready.shape, X_valid_ready.shape)
print(X_train_ready.head())  # head 메서드는 앞부분을 보여 줍니다.

# 3. 실제 교차검증에는 원래 train_frame을 받는 Pipeline을 전달합니다.
# ColumnTransformer 클래스는 (이름, 변환 객체, 대상 컬럼)별 결과를 결합합니다.
preprocess = ColumnTransformer([
    ("weight", StandardScaler(), ["weight"]),
    ("origin", OneHotEncoder(
        categories=[[1, 2, 3]],  # origin 한 컬럼의 사전 허용 코드 목록입니다.
        handle_unknown="error",  # 허용 목록 밖의 코드가 오면 오류로 알립니다.
        sparse_output=False,  # 일반적인 2차원 NumPy 배열을 출력합니다.
    ), ["origin"]),
])
# Pipeline 클래스는 이름을 붙인 전처리와 모델을 순서대로 실행하는 객체를 만듭니다.
# "preprocess", "model", "weight"는 이 예제에서 직접 정한 이름입니다.
cv_model = Pipeline([
    ("preprocess", preprocess),
    ("model", Ridge(alpha=1.0)),  # alpha는 규제 강도입니다. 학습률이 아닙니다.
])
# baseline은 비교 기준입니다. 여기서는 무게 하나만 쓰는 선형회귀입니다.
baseline = Pipeline([
    ("preprocess", ColumnTransformer([
        ("weight", StandardScaler(), ["weight"]),
    ])),
    ("model", LinearRegression()),
])

# KFold 클래스는 3회 분할을 정합니다. 각 회차는 훈련 8건과 검증 4건을 사용합니다.
# shuffle=True는 분할 전 섞기, random_state=42는 같은 분할을 재현하기 위한 설정입니다.
# list는 분할 결과를 보관해 모든 후보에 같은 회차 구성을 적용하게 합니다.
cv_splits = list(KFold(3, shuffle=True, random_state=42).split(train_frame, y_train))

# cross_validate 함수는 회차별 점수와 시간을 담은 딕셔너리를 반환합니다.
# 매 회차 새 Pipeline을 학습하므로 전처리 기준도 해당 회차의 훈련 자료에서만 구합니다.
baseline_result = cross_validate(
    baseline, train_frame, y_train, cv=cv_splits,
    scoring="neg_mean_squared_error", return_train_score=True,
)
cv_result = cross_validate(
    cv_model, train_frame, y_train, cv=cv_splits,
    scoring="neg_mean_squared_error", return_train_score=True,
)
# scorer는 클수록 좋은 점수를 반환합니다. 음수 MSE에 -를 붙이면 실제 MSE가 됩니다.
# test_score는 각 회차의 검증 점수이며 별도 valid_frame의 점수가 아닙니다.
cv_table = pd.DataFrame({
    "학습 MSE": -cv_result["train_score"],
    "검증 MSE": -cv_result["test_score"],
    "기준 모델 검증 MSE": -baseline_result["test_score"],
})
print("교차검증 결과(가상 자료):")
print(cv_table)
print("무게만 쓰는 기준 모델의 평균 검증 MSE:", cv_table["기준 모델 검증 MSE"].mean())
print("무게와 지역을 쓰는 Ridge의 평균 검증 MSE:", cv_table["검증 MSE"].mean())
# 두 모델은 입력 특성도 다릅니다. 차이를 규제 하나의 효과로 해석하지 않습니다.

# 4. GridSearchCV 클래스는 후보별 교차검증과 좋은 후보의 최종 재학습을 구성합니다.
# model__alpha는 Pipeline의 "model" 단계에 있는 alpha 설정을 가리킵니다.
search = GridSearchCV(
    estimator=cv_model,
    param_grid={"model__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=cv_splits, scoring="neg_mean_squared_error", refit=True,
)
search.fit(train_frame, y_train)  # 5후보×3회 학습 후, 선택된 설정으로 12건을 다시 학습합니다.
results = pd.DataFrame(search.cv_results_)  # cv_results_ 속성은 후보별 결과 딕셔너리입니다.
comparison = pd.DataFrame({
    "alpha": results["param_model__alpha"],
    "평균 검증 MSE": -results["mean_test_score"],
})
print("설정 비교(가상 자료):")
print(comparison.round(4).to_string(index=False))  # 출력만 반올림하며 모델은 바뀌지 않습니다.
print("선택된 설정:", search.best_params_)
print("선택된 후보의 평균 검증 MSE:", -search.best_score_)

# 5. best_estimator_는 refit=True에 따라 이미 전체 학습 자료로 다시 학습된 Pipeline입니다.
selected_model = search.best_estimator_  # 이 대입은 추가 학습을 실행하지 않습니다.
final_prediction = selected_model.predict(valid_frame)  # 반환 배열의 shape은 (4,)입니다.
final_mse = mean_squared_error(y_valid, final_prediction)  # 단일 출력 회귀의 MSE 숫자입니다.
print("별도 평가 MSE(가상 자료):", final_mse)
sample_request = pd.DataFrame({"weight": [2800], "origin": [2]})
sample_prediction = selected_model.predict(sample_request)  # 학습된 기준과 계수로 계산합니다.
print("가상 모델의 새 요청 예측(mpg):", sample_prediction)

# named_steps와 named_transformers_ 속성으로 이름에 해당하는 학습된 객체를 찾습니다.
fitted_preprocess = selected_model.named_steps["preprocess"]
fitted_scaler = fitted_preprocess.named_transformers_["weight"]
fitted_ridge = selected_model.named_steps["model"]
print("최종 학습에서 저장된 평균:", fitted_scaler.mean_)
print("최종 학습에서 저장된 변환용 표준편차:", fitted_scaler.scale_)
print("학습된 계수:", fitted_ridge.coef_)
print("학습된 절편:", fitted_ridge.intercept_)

# 아래 두 줄은 수업에서 run_gd를 import한 환경에서만 주석을 해제합니다.
# print(inspect.getfile(run_gd))  # 함수가 정의된 파일 위치를 확인합니다.
# print(inspect.getsource(run_gd))  # 접근 가능한 Python 소스가 있을 때 본문을 확인합니다.

# 6. Rust와 비교할 별도 계산입니다. 다음 값은 위 모델의 학습값이 아닌 임의의 저장값입니다.
# 무게 하나의 표준화와 예측만 확인하며, origin 인코딩과 Ridge 학습은 포함하지 않습니다.
saved_mean, saved_scale, saved_w, saved_b = 3000.0, 500.0, -4.0, 25.0
demo_weight = 2800.0
demo_weight_scaled = (demo_weight - saved_mean) / saved_scale
demo_prediction = saved_w * demo_weight_scaled + saved_b
assert np.isclose(demo_prediction, 26.6)
print("임의의 저장값으로 확인한 예측(mpg):", demo_prediction)
```

가상 예제에서는 선택된 `alpha`가 `0.01`, 평균 교차검증 MSE가 약 `0.5831`, 별도 평가 MSE가 약 `0.2756`으로 계산되었다. 새 요청의 예측은 약 `27.7189 mpg`다. 이 값들은 위 가상 자료와 코드의 실행 결과이며, 실제 자동차의 예측 성능이나 수업 기록의 재현 결과를 뜻하지 않는다.

또한 이 예제의 별도 평가 무게에는 학습 최솟값보다 작은 값이 들어 있다. 그래서 기본 MinMax 변환 결과의 최솟값이 약 `-0.1765`가 된다. 이는 앞에서 설명한 것처럼 학습 기준을 새 자료에도 유지한 결과다.

## 14. Rust에서도 저장한 기준으로 새 입력을 예측한다

Python의 Pipeline이 담당했던 핵심은 학습에서 정한 기준과 계수를 보관하고, 예측에서 그 값을 읽어 사용한다는 점이다. Rust에서도 필요한 값을 <strong>`struct`</strong>, 즉 여러 값을 필드 이름으로 묶는 구조체에 보관할 수 있다.

아래 코드는 무게 하나의 표준화와 예측만 보여준다. `f64`는 소수를 표현하는 64비트 부동소수점 자료형이고, `impl` 블록은 구조체에서 사용할 메서드를 정의한다. `&self`는 현재 모델을 읽기 위해 빌린 참조다. 이 메서드는 저장된 평균이나 계수를 바꾸지 않는다.[^rust]

여기서는 실행 흐름을 확인하기 위해 평균 3000, 나눗값 500, 계수 −4, 절편 25를 임의로 정했다. 위 Ridge의 학습값을 옮긴 것이 아니다. 무게 2800을 넣으면 표준화 값은 $(2800-3000)/500=-0.4$이고, 예측값은 $(-4)\times(-0.4)+25=26.6$이다. Python 전체 코드의 마지막 계산과 같은 값이다.

| 공통점 | 이 예제에서의 차이 |
| --- | --- |
| 저장된 평균과 나눗값으로 새 무게를 변환한다. | Python 전체 예제는 라이브러리의 Pipeline을 사용하고, Rust 예제는 구조체에서 직접 계산한다. |
| 저장된 계수와 절편으로 예측한다. | Rust 예제는 필드의 자료형과 읽기용 참조를 코드에 명시한다. |
| 예측마다 기준을 다시 계산하지 않는다. | Rust 예제에는 계수 학습이나 `origin` 인코딩을 구현하지 않았다. |

[Rust 복습 코드 내려받기](/car-mpg-pipeline/saved-weight-model-ko.rs)

```rust
// 학습된 상태를 읽어 예측하는 흐름을 확인하는 독립 실행 예제입니다.
// 아래 저장값은 실제 학습 결과가 아닌 임의의 숫자입니다.
// origin 처리와 Ridge 학습은 포함하지 않습니다. 무게 단위는 lb입니다.
struct WeightModel {
    mean: f64,  // 학습에서 저장하는 무게의 평균입니다.
    scale: f64, // 학습에서 저장하는 변환용 나눗값입니다.
    w: f64,     // 변환된 무게에 곱하는 계수입니다.
    b: f64,     // 예측식에서 마지막에 더하는 절편입니다.
}

impl WeightModel {
    // &self는 모델을 읽기 위해 빌린 참조입니다. 저장값을 바꾸지 않습니다.
    // weight는 새 자동차의 무게이며 반환값은 예측 연비입니다.
    fn predict(&self, weight: f64) -> f64 {
        assert!(self.scale.is_finite() && self.scale > 0.0);
        let weight_scaled = (weight - self.mean) / self.scale;
        self.w * weight_scaled + self.b
    }
}

fn main() {
    // Python 마지막 예제와 같은 임의의 저장값으로 객체를 만듭니다.
    let model = WeightModel {
        mean: 3000.0,
        scale: 500.0,
        w: -4.0,
        b: 25.0,
    };
    let prediction = model.predict(2800.0);
    // 실수 계산의 작은 표현 오차를 허용하며 기대한 26.6과 비교합니다.
    assert!((prediction - 26.6).abs() < 1e-12);
    println!("임의의 저장값으로 확인한 예측(mpg): {prediction:.1}");
}
```

## 15. 다시 확인할 핵심 구분

| 자주 섞이는 부분 | 구분할 기준 |
| --- | --- |
| `fit`과 `transform` | `fit`은 기준을 구하고, `transform`은 그 기준을 적용한다. |
| 검증 자료의 전처리 | 검증 자료도 변환한다. 기준 계산에는 사용하지 않는다. |
| 교차검증의 반복 | 매 회차 훈련 부분만으로 전처리와 모델을 새로 학습한다. |
| 최종 스케일러 | refit에서 전체 학습 자료로 다시 구한 기준을 사용한다. |
| Pipeline의 구성 | 코드가 전처리와 모델을 지정한다. Ridge가 스케일러를 자동 선택하지 않는다. |
| category와 원-핫 | 하나는 범주를 정의하고, 다른 하나는 범주별 입력 컬럼을 만든다. |
| 표의 결합 | `concat`은 행 index를 맞춘다. 같은 자동차가 연결되어야 한다. |
| MSE와 음수 MSE | MSE는 작을수록, 음수 MSE 점수는 클수록 좋다. |
| `best_score_` | 후보 선택 때의 평균 CV 점수이며 별도 최종 평가 점수가 아니다. |
| Ridge의 `alpha` | 규제 강도다. 경사하강법의 학습률이 아니다. |
| L2의 제곱 | 계수를 제곱해서 덮어쓰지 않고 학습 목표에 비용을 더한다. |
| 미분을 사용하는 시점 | 손실을 계산한 다음 현재 계수에서의 미분값으로 갱신한다. |
| $w$와 `dw` | 하나는 예측 계수이고, 다른 하나는 손실의 미분값을 저장하는 이름이다. |
| 표준화의 의미 | 학습 평균과의 차이를 학습 표준편차 단위로 표현한다. 고정 범위로 제한하지 않는다. |
| 표현력과 학습 | 표현력은 가능한 관계의 범위이고, 학습은 그 안에서 관계를 정한다. |
| 차원과 개수 | 원래 입력 수, 변환 후 컬럼 수, 계수 수, 그래프의 축을 따로 확인한다. |

## 참고 자료

수업의 샘플 수, 출력값과 수업용 함수에 대한 설명은 제공된 학습 기록을 바탕으로 정리했다. 아래 공식 문서는 라이브러리 동작을 확인하는 근거다. 가상 데이터 예제의 출력은 함께 제공한 코드로 계산했다.

[^pitfalls]: scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html). 전처리 일관성, 학습·평가 분리와 데이터 누수.
[^scaler]: scikit-learn, [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). `fit`, `transform`, `mean_`, `scale_`, `ddof=0`과 상수 컬럼 처리.
[^pipeline]: scikit-learn, [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). 단계별 실행, 예측 시 변환, 중첩 설정 이름.
[^preprocessing]: scikit-learn, [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html). 표준화와 스케일링의 목적 및 적용 범위.
[^std]: NumPy, [numpy.std](https://numpy.org/doc/stable/reference/generated/numpy.std.html). 표준편차와 `ddof`의 의미.
[^minmax]: scikit-learn, [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html). 학습 범위와 `clip` 설정.
[^onehot]: scikit-learn, [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). 범주별 변환, `categories`, `sparse_output`, `handle_unknown`.
[^structures]: Pandas, [Intro to data structures](https://pandas.pydata.org/docs/user_guide/dsintro.html), [DataFrame.to_numpy](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html). 행·컬럼 이름과 배열 변환.
[^categorical]: Pandas, [Categorical data](https://pandas.pydata.org/docs/user_guide/categorical.html), [Categorical](https://pandas.pydata.org/docs/reference/api/pandas.Categorical.html), [DataFrame.astype](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html). 범주 정의와 자료형 변환.
[^dummies]: Pandas, [get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html). 범주별 표시 컬럼과 결측값의 기본 처리.
[^concat]: Pandas, [concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html). 결합 방향과 다른 축의 index 정렬.
[^linear]: scikit-learn, [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html). 최소제곱 선형회귀, Ridge와 Lasso.
[^column-transformer]: scikit-learn, [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html). 컬럼별 전처리와 학습된 변환기 접근.
[^ridge]: scikit-learn, [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). L2 목적함수, 규제 강도, 학습 계수와 계산 방법.
[^cv-guide]: scikit-learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html). 교차검증, 최종 평가 자료와 분할 방식.
[^cv]: scikit-learn, [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html). 입력과 반환 딕셔너리, 회차별 점수 및 시간.
[^validation]: scikit-learn, [모델 선택·검증 공식 구현](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_validation.py). 회차별 모델 복제와 입력·정답의 위치 기반 분할.
[^kfold]: scikit-learn, [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html). 분할 횟수, 셔플과 재현 설정.
[^mse]: scikit-learn, [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html). 평균제곱오차 계산.
[^scoring]: scikit-learn, [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html). 큰 점수를 좋은 값으로 보는 scorer 규칙과 음수 손실 점수.
[^grid]: scikit-learn, [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). 후보 탐색, 결과 속성과 `refit`.
[^lasso]: scikit-learn, [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html). L1 목적함수와 계수 학습.
[^polynomial]: scikit-learn, [Polynomial regression](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions). 원래 입력으로 다항 특성을 만드는 방법.
[^tree]: scikit-learn, [Decision Trees](https://scikit-learn.org/stable/modules/tree.html). 입력 조건에 따른 분기와 회귀 예측.
[^mlp]: scikit-learn, [Neural network models](https://scikit-learn.org/stable/modules/neural_networks_supervised.html). 다층 모델의 구조와 표현.
[^inspect]: Python, [inspect](https://docs.python.org/3/library/inspect.html). `getfile`과 `getsource`의 동작 및 제약.
[^python-files]: Python, [pathlib](https://docs.python.org/3/library/pathlib.html), [zipfile](https://docs.python.org/3/library/zipfile.html), [hashlib](https://docs.python.org/3/library/hashlib.html), [모듈 검색 경로](https://docs.python.org/3/tutorial/modules.html#the-module-search-path). 파일 준비와 모듈 불러오기.
[^plotting]: NumPy, [linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html); Matplotlib, [scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html), [plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html); Pandas, [head](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html), [round](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.round.html). 시각화 입력과 출력 확인.
[^split]: scikit-learn, [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). 입력과 정답의 대응을 유지한 분할.
[^rust]: Rust 공식 문서, [Method Syntax](https://doc.rust-lang.org/book/ch05-03-method-syntax.html). 구조체의 메서드와 `&self`.
