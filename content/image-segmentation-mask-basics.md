+++
title = "이미지 세그먼테이션 기초: 배열의 축부터 마스크 변환과 증강까지"
description = "RGB 이미지와 클래스 마스크의 차이에서 시작해 axis, 불리언 인덱싱, 연결 요소, 원-핫과 argmax를 연결한다. 마스크를 망가뜨리지 않는 증강과 PyTorch 학습 입력 형식도 함께 다룬다."
date = 2026-11-25
slug = "image-segmentation-mask-basics"

[extra]
toc = true
styles = ["image-segmentation-mask-basics/article.css"]
+++

이미지 세그먼테이션 코드를 읽다 보면 같은 이미지를 나타내는데도 배열의 모양이 계속 달라진다. 어떤 변수는 `(H, W, 3)`이고, 어떤 변수는 `(H, W)`, 또 다른 변수는 `(C, H, W)`다. 이 차이를 보지 않고 코드를 따라 쓰면 `axis=-1`이 필요한 이유나 `mask[condition] = 2`가 바꾸는 대상을 놓치기 쉽다.

먼저 확인할 것은 <strong>픽셀 하나에 무엇을 몇 개 저장했는가</strong>다. 색상 세 성분인지, 클래스 번호 하나인지, 클래스마다 소속 여부를 저장했는지에 따라 필요한 연산이 달라진다. 원하는 출력 구조를 먼저 정하면 어떤 축을 유지하고 어떤 축을 줄여야 하는지도 판단할 수 있다.

이 글에서는 하늘·땅·공으로 이루어진 작은 이미지를 직접 만든다. 같은 예시를 끝까지 사용해 배열 선택, 마스크 변환, 시각화와 증강을 연결한다. 마지막에는 이 마스크를 PyTorch 학습에 전달할 때 필요한 형식까지 확인한다. 실행 파일은 [examples.py](/image-segmentation-mask-basics/examples.py)에서 받을 수 있다.

## 01. 이미지와 마스크는 같은 픽셀에 다른 정보를 저장한다

<strong>세그먼테이션(Segmentation)</strong>은 이미지의 각 픽셀이 어떤 영역에 속하는지 구분하는 작업이다. <strong>클래스(Class)</strong>는 구분할 종류이고, <strong>마스크(Mask)</strong>는 픽셀별 클래스 번호나 선택 여부 등을 담은 배열이다. 마스크라는 이름만으로 자료형과 구조가 하나로 정해지는 것은 아니다.

이번 예시에서는 클래스 번호를 `0=하늘`, `1=땅`, `2=공`으로 정한다. 번호의 크기는 종류의 크기나 중요도를 뜻하지 않는다. `0`이 항상 배경인 것도 아니다. 이 글의 시맨틱 마스크에서는 하늘을 뜻하며, 다른 데이터셋에서는 별도의 규칙을 확인해야 한다.

`H`는 이미지 높이, `W`는 이미지 너비다. <strong>shape</strong>는 각 축의 크기를 순서대로 나타내고, <strong>축(axis)</strong>은 배열을 구분하는 방향이다. 아래 표에서 `C`는 RGB 성분 수가 아니라 전체 클래스 수다.

<div class="seg-table">

| 변수 | shape | 자료형 | 픽셀 하나에 저장하는 정보 |
| --- | --- | --- | --- |
| `toy` | `(H, W, 3)` | `uint8` | R·G·B 색상 값 3개 |
| `semantic` | `(H, W)` | `uint8` | 클래스 번호 하나 |
| `ball` | `(H, W)` | `bool` | 공이면 `True`, 아니면 `False` |
| `onehot` | `(C, H, W)` | `uint8` | 클래스마다 소속 여부 0 또는 1 |

</div>

`uint8`은 0부터 255까지의 정수를 저장하는 자료형이고, `bool`은 참·거짓을 나타내는 `True`와 `False`를 저장한다. 이 예시의 마스크는 클래스가 세 개뿐이므로 `uint8`로 충분하다.

RGB 이미지의 `toy[y, x]`에는 색상 값 세 개가 있다. 같은 픽셀의 `semantic[y, x]`에는 클래스 번호 하나가 있다. 여기서 `y`는 행 인덱스, `x`는 열 인덱스다. 두 배열을 연결하는 조건은 단순히 높이와 너비가 같다는 것에 그치지 않는다. <strong>같은 행·열 인덱스가 같은 이미지 픽셀을 가리켜야 한다.</strong> 이미지와 마스크를 따로 뒤집으면 shape가 같아도 이 대응은 깨진다.

<strong>시맨틱 세그먼테이션(Semantic Segmentation)</strong>은 같은 종류의 모든 공에 같은 클래스 번호를 준다. <strong>인스턴스 세그먼테이션(Instance Segmentation)</strong>은 종류뿐 아니라 개별 공도 구분한다. 시맨틱 마스크의 `2`는 공이라는 종류이고, 인스턴스 번호는 개별 영역의 식별자다.

<figure class="seg-figure">
  <img src="/image-segmentation-mask-basics/representations-ko.svg" alt="같은 공 두 개를 RGB 이미지, 클래스 마스크, 연결 영역 번호로 각각 표현한 비교" loading="lazy" />
  <figcaption>색상, 클래스 번호, 개별 영역 번호는 서로 다른 정보다. 마스크를 화면에 표시할 때 붙인 색도 저장된 번호 자체와 구분한다.</figcaption>
</figure>

## 02. 원본 배열을 만든 뒤 축과 인덱스를 확인한다

외부 파일 없이 실행할 수 있도록 단색 이미지를 만든다. NumPy는 배열을 다루는 라이브러리이며, `np`는 코드에서 사용하는 짧은 이름이다. OpenCV는 이미지 처리 라이브러리이고 Python에서는 `cv2`라는 이름으로 가져온다.

`np.zeros`는 지정한 shape를 0으로 채운 배열을 만든다. 색상 값은 아직 따로 없으므로 전체를 하늘색으로 채우고, 아래쪽 행을 초록색으로 바꾼 다음 공 두 개를 그린다.

~~~python
import numpy as np
import cv2

H, W = 120, 200
CLASS_NAMES = ["sky", "ground", "ball"]
C = len(CLASS_NAMES)  # 리스트 인덱스와 클래스 번호를 같은 규칙으로 정했다.
BALL_RGB = (230, 60, 50)

toy = np.zeros((H, W, 3), dtype=np.uint8)
toy[:] = (150, 200, 255)
toy[90:] = (90, 160, 70)

# 원의 중심은 (x, y), -1은 내부까지 채우라는 뜻이다.
cv2.circle(toy, (55, 60), 25, BALL_RGB, -1)
cv2.circle(toy, (145, 60), 20, BALL_RGB, -1)

print(toy.shape, toy.dtype, toy.size)
# (120, 200, 3) uint8 72000
~~~

`toy.size`는 전체 원소 수다. RGB 성분도 각각 원소로 세므로 픽셀 수인 `120 × 200 = 24,000`과 다르다. 이미지와 마스크의 공간 크기만 비교할 때는 `toy.shape[:2]`와 `semantic.shape`를 비교한다.

아래 표현은 모두 같은 `toy`에서 출발한다. <strong>정수 인덱스는 하나를 선택해 그 축을 없애고, 슬라이싱은 선택 범위를 정하면서 축을 유지한다.</strong> 뒤쪽 축을 생략하면 그 축 전체를 선택한다.[^indexing]

<div class="seg-table">

| 표현 | 선택하는 대상 | 결과 shape |
| --- | --- | --- |
| `toy[0]` | 첫 번째 행의 모든 픽셀 | `(200, 3)` |
| `toy[0:1]` | 첫 번째 행을 길이 1인 행 축으로 유지 | `(1, 200, 3)` |
| `toy[:, 0]` | 첫 번째 열의 모든 픽셀 | `(120, 3)` |
| `toy[:, :, 0]` | 모든 픽셀의 R 성분 | `(120, 200)` |
| `toy[60, 55]` | 행 60, 열 55인 픽셀의 RGB | `(3,)` |
| `toy[90:]` | 행 인덱스 90부터 마지막 행까지 | `(30, 200, 3)` |
| `toy[:, 90:]` | 열 인덱스 90부터 마지막 열까지 | `(120, 110, 3)` |

</div>

`start:stop`은 시작 인덱스를 포함하고 종료 인덱스를 제외한다. `toy[90:]`는 첫 번째 축에만 범위를 지정한 것이므로 행을 선택한다. `toy[:, 90:]`는 행 전체를 유지하고 두 번째 축인 열을 선택한다.

NumPy의 픽셀 선택은 `toy[y, x]`, OpenCV의 원 중심 지정은 `(x, y)`다. 두 순서를 혼동하지 않는다. 여기서는 직접 만든 배열을 RGB로 해석한다. `cv2.circle`은 전달한 순서대로 성분 값을 쓰므로 이 규칙을 유지할 수 있다. 반면 `cv2.imread`로 읽은 일반적인 컬러 이미지는 BGR 순서이므로 RGB로 사용할 때 변환이 필요하다.[^opencv-image]

또 하나의 실무 차이가 있다. 기본 슬라이싱으로 얻은 배열은 원본 데이터를 공유하는 <strong>뷰(view)</strong>다. 잘라낸 부분만 따로 편집하려면 `crop = toy[20:80, 30:100].copy()`처럼 복사한다. `.copy()` 없이 `crop`을 수정하면 원본의 해당 영역도 바뀔 수 있다.[^indexing]

## 03. RGB 세 값을 비교하면 판단 결과도 세 개다

이번 이미지는 공에 정확히 같은 RGB 값을 사용했다. 공을 찾으려면 각 픽셀의 세 성분이 그 값과 모두 일치하는지 확인하면 된다. 실제 사진에서는 조명과 그림자 때문에 같은 물체의 색도 달라지므로, 이 방법은 지금 만든 단색 예시에만 사용한다.

먼저 성분별 비교를 수행한다.

~~~python
result = toy == BALL_RGB

print(result.shape, result.dtype)
# (120, 200, 3) bool
~~~

`==`는 같은 인덱스에 대응하는 원소들을 비교한다. R·G·B를 한 덩어리로 판단하는 연산이 아니다. 픽셀 값이 `(230, 60, 50)`이면 비교 결과가 `[True, True, True]`, 마지막 성분만 다르면 `[True, True, False]`가 된다.

그런데 이미지 전체에는 원소가 많이 있고, `BALL_RGB`에는 세 값만 있다. 이때 적용되는 규칙이 <strong>브로드캐스팅(Broadcasting)</strong>이다. 서로 다른 shape의 배열을 연산할 때, 크기가 1인 축의 값을 다른 쪽 축에 반복 적용할 수 있도록 대응시킨다. 실제로 거대한 복사 배열을 먼저 만들 필요는 없다.[^broadcasting]

shape를 오른쪽부터 비교한다. 축 크기가 같거나 둘 중 하나가 1이면 그 축은 대응할 수 있고, 없는 앞쪽 축은 크기가 1인 것처럼 취급한다. 이 예시에서는 다음과 같이 판단한다.

<div class="seg-table">

| 대상 | 행 축 | 열 축 | 성분 축 |
| --- | --- | --- | --- |
| `toy` | 120 | 200 | 3 |
| RGB 세 값의 대응 방식 | 1 | 1 | 3 |

</div>

성분 축의 크기는 같고, 행·열 축에서는 RGB 세 값을 반복해서 사용할 수 있다. 그래서 모든 픽셀의 R은 230, G는 60, B는 50과 각각 비교된다. `toy[:] = (150, 200, 255)`로 전체 픽셀을 채울 수 있었던 이유도 같다.

이 규칙은 이미지라는 의미에 의존하지 않는다. NumPy는 축의 크기로 판단한다. 이미지에 다른 배열을 더하거나 곱할 때도 두 shape를 오른쪽 축부터 비교하면 연산 가능한 구조인지 판단할 수 있다.

## 04. axis는 무엇끼리 묶어 계산할지를 정한다

지금 `result`에는 픽셀마다 판단 결과가 세 개씩 있다. 원하는 `ball`에는 픽셀마다 결과가 하나씩 있어야 한다. 서로 다른 픽셀의 판단을 섞으면 안 되고, <strong>같은 행·열 인덱스에 있는 세 성분만 묶어야 한다.</strong>

`np.all`은 전달한 값들이 모두 참인지 검사한다. `axis=-1`을 지정하면 마지막 축의 값들을 묶어 검사한다. 마지막 축은 RGB 성분에 대응하므로 이 목적과 맞는다.[^all]

~~~python
ball = np.all(result, axis=-1)

print(ball.shape, ball.dtype)
# (120, 200) bool
~~~

이때 행·열 인덱스는 유지된다. 각 행·열 조합에서 성분 축만 따라가며 세 값을 검사하고, 그 조합마다 결과 하나를 만든다. 따라서 `ball` 전체가 참·거짓 하나로 바뀌는 것이 아니다. 모든 픽셀의 판단을 담은 2차원 배열이 된다.

<figure class="seg-figure">
  <img src="/image-segmentation-mask-basics/axis-all-ko.svg" alt="같은 행과 열을 유지하면서 RGB 성분별 비교 결과 세 개를 all로 하나의 픽셀 판단으로 합치는 과정" loading="lazy" />
  <figcaption>줄이는 것은 성분 축이다. 행·열 축을 유지해야 원본의 어느 픽셀에 대한 결과인지 연결할 수 있다.</figcaption>
</figure>

이처럼 여러 원소를 합계나 최댓값 같은 하나의 결과로 만드는 작업을 <strong>축소 연산(Reduction)</strong>이라고 한다. `sum`, `mean`, `max`, `all` 등이 여기에 해당한다. 기본 동작에서는 계산한 축이 사라지고 나머지 축이 유지된다. `argmax`도 지정한 축을 비교해 그 축 안의 인덱스 하나를 남기지만, 값 자체를 남기는 `max`와 반환값의 의미가 다르다.

`result`의 shape가 `(120, 200, 3)`일 때 결과 구조는 다음처럼 예측할 수 있다.

<div class="seg-table">

| 연산 | 한 번에 묶는 원소 | 남는 축과 shape |
| --- | --- | --- |
| `result.sum(axis=0)` | 같은 열·성분에 있는 모든 행 | 열·성분, `(200, 3)` |
| `result.sum(axis=1)` | 같은 행·성분에 있는 모든 열 | 행·성분, `(120, 3)` |
| `result.all(axis=2)` | 같은 행·열에 있는 모든 성분 | 행·열, `(120, 200)` |
| `result.all()` | 배열 전체 | 결과 하나 |

</div>

`axis`는 배열의 일부 범위만 선택하는 인자가 아니다. 다른 축의 인덱스 조합마다 어떤 원소끼리 묶을지를 지정한다. 선택 범위를 제한하려면 인덱싱이 먼저 필요하다.

<details>
<summary>축이 네 개일 때도 같은 방법으로 판단하기</summary>

`a`를 shape가 `(3, 4, 30, 3)`인 배열이라고 하자. `a.sum(axis=2)`는 2번 축의 30개 원소를 합산한다. 0·1·3번 축의 인덱스는 각각 고정한 채 2번 축의 인덱스만 바꿔서 한 번의 합산을 수행한다.

그중 하나는 `a[0, 0, :, 0]`에 있는 30개 원소의 합이다. 나머지 축에도 다른 인덱스 조합이 있으므로 같은 작업이 각 조합에서 따로 수행된다. 총 결과 수는 `3 × 4 × 3 = 36`, 결과 shape는 `(3, 4, 3)`이다.

`a[0, 0, :, 0].sum()`은 이 중 한 조합만 선택해 결과 하나를 만든다. `a.sum(axis=2)`는 모든 조합에서 결과를 만든다. 어떤 값끼리 같은지는 묶는 기준이 아니며, 축과 인덱스가 기준이다.

</details>

축을 없애지 않고 크기 1로 남기려면 `keepdims=True`를 사용한다. 예를 들어 `result.all(axis=-1, keepdims=True)`의 shape는 `(120, 200, 1)`이다. 뒤에서 같은 픽셀의 세 성분에 다시 연산을 적용할 때 브로드캐스팅에 활용할 수 있다.

## 05. 조건 배열을 만드는 일과 값을 바꾸는 일을 나눈다

공에 해당하는 픽셀을 찾았으니 클래스 번호를 저장할 차례다. `semantic`을 0으로 채우면 일단 모든 픽셀이 하늘이다. 땅의 행을 1로 바꾸고, 공에 해당하는 픽셀을 2로 바꾼다.

~~~python
semantic = np.zeros((H, W), dtype=np.uint8)
semantic[90:] = 1
semantic[ball] = 2

print(np.unique(semantic))  # 저장된 서로 다른 값 확인
# [0 1 2]
~~~

`semantic[ball] = 2`는 두 배열의 값을 비교하지 않는다. `ball`에서 참인 행·열 인덱스를 골라, `semantic`의 같은 인덱스에 2를 쓴다. 선택되지 않은 값은 유지된다. 참·거짓 배열로 원소를 고르는 방법을 <strong>불리언 인덱싱(Boolean Indexing)</strong>이라고 한다.[^indexing]

<div class="seg-table">

| 표현 | 하는 일 | 결과 또는 변경 |
| --- | --- | --- |
| `semantic == 2` | 공인지 판단 | `(H, W)` 불리언 배열 |
| `semantic[semantic == 2]` | 공인 픽셀의 원래 값을 추출 | 2들만 담은 1차원 배열 |
| `semantic[semantic == 2] = 1` | 공인 픽셀의 값을 바꿈 | 원본 `semantic`이 변경됨 |

</div>

마지막 행은 세 표현의 차이를 보이기 위한 예시다. 이후 코드에서는 공의 클래스 번호를 계속 2로 유지한다.

`semantic == 2`에는 `np.all`이 필요 없다. `semantic`은 픽셀마다 값이 하나이므로 비교 직후부터 픽셀마다 판단 결과가 하나다. 반면 `toy == BALL_RGB`에는 픽셀마다 결과가 세 개여서 이를 합치는 단계가 필요했다.

숫자 0과 1로 된 정수 배열은 불리언 배열과 다르다. 정수 배열을 `[]` 안에 넣으면 정수들을 인덱스로 사용한다. <strong>0은 제외하고 0이 아닌 값은 모두 선택한다는 규칙일 때만</strong> `binary.astype(bool)`로 변환한다. 클래스 번호 마스크에서 공만 찾으려면 `semantic == 2`처럼 원하는 조건을 명시한다.

## 06. 연결된 영역을 나누면 공 두 개를 구분할 수 있다

현재 `ball`은 모든 공을 참으로 표시한다. 각각의 공을 따로 다루려면 이 배열에서 서로 이어진 픽셀들을 묶어 별도의 번호를 붙일 수 있다. 이것을 <strong>연결 요소(Connected Component)</strong>라고 한다.

`cv2.connectedComponents`에는 8비트 단일 채널 배열을 전달한다. 불리언 배열을 `uint8`로 바꾸면 `False`는 0, `True`는 1이 된다. 여기서는 대각선으로 닿은 픽셀도 연결된 것으로 판단하도록 `connectivity=8`을 명시한다. `4`를 사용하면 위·아래·왼쪽·오른쪽으로 닿은 픽셀만 연결한다.[^components]

~~~python
num_labels, labeled = cv2.connectedComponents(
    ball.astype(np.uint8), connectivity=8
)

print(num_labels)       # 3: 배경 번호를 포함한다.
print(labeled.shape)    # (120, 200)
print(labeled.dtype)    # int32

instances = [labeled == i for i in range(1, num_labels)]
print(len(instances))   # 2

# 공이 없는 이미지에서는 빈 리스트를 stack에 전달하지 않는다.
if instances:
    instance_masks = np.stack(instances, axis=0)
else:
    instance_masks = np.zeros((0, H, W), dtype=bool)

print(instance_masks.shape)  # (2, 120, 200)
~~~

`num_labels`는 배경 번호 0을 포함한 라벨 개수다. `labeled`에는 픽셀별 영역 번호가 저장된다. 여기서 배경은 <strong>현재 입력 `ball`에서 0인 모든 픽셀</strong>이므로 하늘과 땅을 모두 포함한다. `semantic`의 클래스 0인 하늘과 범위가 다르다.

`range(1, num_labels)`는 0을 제외하고 마지막 번호까지 선택한다. `instances`는 `(H, W)` 불리언 배열들을 담은 Python 리스트다. `np.stack`은 같은 shape의 배열들을 새 축으로 묶는다. 이때 생긴 첫 축의 크기는 클래스 수가 아니라 개별 영역 수다. 영역이 없을 때는 shape를 `(0, H, W)`로 명시해 두면 이후 처리에서도 구조를 유지할 수 있다.[^stack]

원래 시맨틱 마스크로 되돌릴 때는 개별 공에 모두 같은 클래스 번호를 쓰면 된다.

~~~python
rebuilt = np.zeros((H, W), dtype=np.uint8)
rebuilt[90:] = 1
for instance_mask in instances:
    rebuilt[instance_mask] = 2

assert np.array_equal(rebuilt, semantic)
~~~

`assert`는 조건이 참인지 검사하고, 거짓이면 실행을 멈추는 검증문이다. `np.array_equal`은 두 배열의 shape와 값이 모두 같은지 확인한다.

이 방법이 일반적인 인스턴스 세그먼테이션을 해결해 주는 것은 아니다. 두 공이 붙으면 하나의 연결 요소가 되고, 공 하나가 마스크 안에서 끊어지면 여러 요소가 될 수 있다. 공이 아닌 잡음도 별도 요소가 될 수 있다. 연결 번호 자체에 물체 종류나 일정한 좌우 순서가 보장되는 것도 아니다. <strong>서로 떨어진 공 하나가 연결 영역 하나에 대응한다는 이번 예시의 조건</strong>에서만 개수를 같은 것으로 읽는다.

## 07. 원-핫은 클래스 번호를 클래스별 소속 여부로 바꾼다

현재 `semantic[y, x]`는 클래스 번호 하나다. 어떤 작업에서는 클래스별 마스크를 따로 사용하는 편이 편하다. <strong>원-핫 인코딩(One-hot Encoding)</strong>은 픽셀마다 해당 클래스만 1, 나머지는 0으로 표현하는 방식이다.

클래스 세 개에 사용할 변환표를 먼저 만든다. `np.eye(C)`는 대각선만 1이고 나머지는 0인 `(C, C)` 배열을 만든다. 행 인덱스가 클래스 번호와 맞도록 사용할 수 있다.[^eye]

~~~python
class_vectors = np.eye(C, dtype=np.uint8)
print(class_vectors)
# [[1 0 0]
#  [0 1 0]
#  [0 0 1]]

# 실제 데이터에서는 변환표로 조회하기 전에 유효한 번호인지 확인한다.
assert np.issubdtype(semantic.dtype, np.integer)
assert semantic.min() >= 0 and semantic.max() < C

onehot_hwc = class_vectors[semantic]       # (H, W, C)
onehot = onehot_hwc.transpose(2, 0, 1)    # (C, H, W)
~~~

`class_vectors[semantic]`은 정수 배열 인덱싱이다. `semantic`에 저장된 정수를 변환표의 행 인덱스로 사용한다. 원래 픽셀에는 번호 하나가 있었지만, 조회 결과에는 그 번호에 대응하는 길이 `C`의 배열이 들어간다. 그래서 `(H, W)`가 `(H, W, C)`로 바뀐다.

`transpose(2, 0, 1)`은 기존 2번 축, 0번 축, 1번 축 순서로 축을 배치한다. 값을 섞거나 클래스 번호를 새로 만드는 작업이 아니다. 이후 `onehot[2]`는 공 클래스의 `(H, W)` 마스크가 된다.

<figure class="seg-figure">
  <img src="/image-segmentation-mask-basics/onehot-roundtrip-ko.svg" alt="3행 4열의 별도 예시 마스크를 클래스별 원-핫 마스크로 바꾸고 argmax로 복원하는 구조" loading="lazy" />
  <figcaption>전체 값을 읽을 수 있도록 별도로 만든 3×4 마스크다. 클래스별 마스크로 나눠도 같은 행·열 인덱스끼리 대응하며, 클래스 축에서 argmax를 구하면 원래 번호를 복원할 수 있다.</figcaption>
</figure>

원-핫으로 표현하려면 각 픽셀이 정확히 한 클래스에 속하고, 클래스 번호가 0부터 `C-1`까지의 정수라는 조건이 필요하다. 실제 데이터에 미지정 영역을 뜻하는 255나 -1이 있다면 먼저 별도 처리 규칙을 정해야 한다. 특히 -1은 NumPy에서 마지막 행을 가리키므로 검증 없이 조회하면 오류 없이 잘못된 클래스로 바뀔 수 있다.

## 08. argmax가 돌려주는 것은 최댓값의 인덱스다

원-핫에서 원래 번호로 돌아가려면 픽셀마다 어느 클래스에 1이 있는지 찾는다. 클래스 인덱스가 원래 클래스 번호와 같도록 표를 만들었으므로, 가장 큰 값이 있는 클래스 인덱스를 가져오면 된다.

`max`는 가장 큰 <strong>값</strong>, `argmax`는 그 값이 저장된 <strong>인덱스</strong>를 돌려준다. `onehot`의 클래스 축은 0번이므로 그 축을 비교한다.[^argmax]

~~~python
back = onehot.argmax(axis=0)

print(back.shape)  # (120, 200)
print(np.array_equal(back, semantic))  # True

# 이번 onehot은 np.eye로 만들었으므로 원소가 0 또는 1임이 보장된다.
assert np.all(onehot.sum(axis=0) == 1)
~~~

특정 픽셀에서 `onehot[:, y, x]`를 꺼냈을 때 `[0, 0, 1]`이라면 최대값은 1, 그 인덱스는 2다. 전체 배열에서도 행·열 인덱스를 각각 유지한 채 클래스 축에서 같은 작업을 한다. 결과 shape가 `(H, W)`인 이유다.

`argmax`에는 이 데이터가 올바른 원-핫인지 검증하는 기능이 없다. 모두 0이거나 최대값이 여러 개면 가장 먼저 나오는 최대값의 인덱스를 반환한다. 원-핫 복원에 사용하려면 표현 규칙을 먼저 확인해야 한다. 모델이 출력한 클래스별 점수에 사용할 때는 가장 높은 점수의 클래스를 선택하는 역할을 하며, 그 예측이 맞는지는 정답과 따로 비교해야 한다.

## 09. 클래스 번호를 공통 기준으로 픽셀 수를 센다

클래스별 픽셀 수는 데이터 구성을 확인할 때 바로 쓸 수 있다. 어떤 클래스가 거의 없거나 예상보다 너무 많으면 마스크 변환이나 데이터 선택이 잘못되었는지 살펴볼 단서가 된다. 여기서 세는 것은 <strong>객체 수가 아니라 픽셀 수</strong>다.

공 클래스의 번호는 `CLASS_NAMES`에서 찾을 수 있다. 앞에서 리스트 인덱스와 실제 클래스 번호를 같은 규칙으로 만들었기 때문에 가능한 조회다.

~~~python
ball_id = CLASS_NAMES.index("ball")  # 2

count_from_mask = int((semantic == ball_id).sum())
count_from_onehot = int(onehot[ball_id].sum())

# ravel은 값을 유지하면서 1차원으로 펼친다.
counts = np.bincount(semantic.ravel(), minlength=C)
count_from_all = int(counts[ball_id])

assert count_from_mask == count_from_onehot == count_from_all
assert int(counts.sum()) == H * W

ratio = count_from_mask / semantic.size * 100
print(counts)
print(f"ball: {count_from_mask} pixels, {ratio:.2f}%")
# [14782  6000  3218]
# ball: 3218 pixels, 13.41%
~~~

불리언 배열을 합산하면 `True`가 1, `False`가 0으로 계산된다. 원-핫의 공 마스크도 공인 픽셀만 1이므로 같은 수를 얻는다. `np.bincount`는 0 이상의 정수가 각각 몇 번 나오는지 세며, 결과 배열의 인덱스가 조사한 정수다.[^bincount]

`minlength=C`는 결과 길이를 <strong>최소한</strong> `C`로 만든다. 길이를 정확히 `C`로 제한하거나 잘못된 라벨을 제거해 주지는 않는다. 마스크에 255가 있으면 그 인덱스까지 결과가 만들어질 수 있으므로 07절의 클래스 범위 검증이 먼저다.

## 10. 저장된 번호와 화면에 표시되는 색을 구분한다

마스크를 컬러로 보여 주려면 클래스 번호와 RGB 값을 연결하는 <strong>팔레트(Palette)</strong>를 만든다. 이 색은 화면에서 종류를 구분하기 위한 것이며, 학습 정답의 번호를 바꾸지 않는다.

~~~python
PALETTE = np.array([
    [222, 231, 240],  # 0: sky
    [73, 128, 93],    # 1: ground
    [185, 43, 62],    # 2: ball
], dtype=np.uint8)

color = PALETTE[semantic]
print(color.shape, color.dtype)
# (120, 200, 3) uint8
~~~

07절에서 클래스 번호로 원-핫 배열을 가져왔던 것과 같은 원리다. 여기서는 표의 한 행에 RGB 값 세 개가 있으므로 픽셀마다 세 값이 붙는다. 원본 `semantic`은 여전히 `(H, W)` 정수 배열이다.

Matplotlib로 직접 표시할 수도 있다. <strong>컬러맵(Colormap)</strong>은 숫자를 표시 색에 대응시키는 규칙이다. `vmin`과 `vmax`를 고정하면 어떤 이미지에 특정 클래스가 없더라도 같은 번호가 같은 색으로 표시된다.[^imshow]

~~~python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(PALETTE / 255.0)
plt.imshow(
    semantic, cmap=cmap, vmin=0, vmax=C - 1,
    interpolation="nearest"
)
plt.axis("off")
plt.show()
~~~

화면 표시용 `interpolation="nearest"`는 경계가 부드럽게 섞여 보이지 않게 한다. 원본 배열의 shape나 클래스 번호를 바꾸는 resize 연산과는 다르다. 불리언 마스크는 `plt.imshow(ball, cmap="gray", vmin=0, vmax=1)`로 표시할 수 있다.

## 11. 오버레이는 같은 픽셀의 색상을 섞는 연산이다

<strong>오버레이(Overlay)</strong>는 원본 이미지 위에 마스크 색을 겹쳐 표시하는 방법이다. 이미지와 정답이 맞게 연결됐는지, 변환 후 경계가 어긋나지 않았는지 확인하는 데 유용하다.

반투명하게 보이도록 하려면 같은 픽셀의 R·G·B 성분끼리 원본 색과 표시 색을 일정 비율로 섞는다. `alpha`는 마스크 색의 반영 비율이다. 0이면 원본만, 1이면 마스크 색만 남는다.

~~~python
# 경계 확인을 위해 원본 공 색과 다른 표시 색을 사용한다.
overlay_palette = PALETTE.copy()
overlay_palette[2] = (255, 220, 0)

alpha = 0.45
color_float = overlay_palette[semantic].astype(np.float32)
image_float = toy.astype(np.float32)
mixed = image_float * (1 - alpha) + color_float * alpha

# 이번에는 공 픽셀에만 표시를 적용한다.
overlay = toy.copy()
overlay[ball] = np.clip(mixed[ball], 0, 255).astype(np.uint8)
~~~

`np.float32`는 소수 값을 저장하는 자료형이다. 색상을 섞는 동안 소수 계산을 수행하고, `np.clip`으로 0~255 범위에 제한한 뒤 표시용 정수로 바꾼다. 최종 이미지에서는 선택한 공 이외의 픽셀을 그대로 유지한다.

여기서 `toy`는 3차원이고 `ball`은 2차원이다. 앞쪽 두 축이 일치하므로 `toy[ball]`은 선택된 픽셀들의 RGB 세 성분을 함께 가져온다. 선택된 픽셀 수를 `P`라고 하면 결과 shape는 `(P, 3)`이다. 픽셀을 골랐을 뿐 RGB 성분 축을 줄인 것은 아니다.

## 12. 증강할 때는 이미지와 정답을 같은 방식으로 이동시킨다

<strong>데이터 증강(Data Augmentation)</strong>은 기존 학습 데이터를 변형해 모델이 다양한 입력을 학습하게 만드는 방법이다. 세그먼테이션에서는 변환 뒤에도 <strong>새 이미지의 각 픽셀과 정답 마스크의 같은 인덱스가 대응해야 한다.</strong>

좌우 반전은 열의 순서를 뒤집는다. 이미지에서는 RGB 성분 축을 유지하고, 마스크에서는 클래스 번호를 그대로 옮긴다.

~~~python
image_flip = toy[:, ::-1, :].copy()
mask_flip = semantic[:, ::-1].copy()

assert np.array_equal(
    mask_flip == 2,
    np.all(image_flip == BALL_RGB, axis=-1)
)
~~~

`::-1`은 해당 축의 인덱스를 역순으로 선택한다. `.copy()`는 독립된 배열을 만들고 역방향으로 데이터를 참조하는 구조를 없앤다. 이렇게 하면 이후 PyTorch 텐서로 바꿀 때도 다루기 쉽다.

위 검증은 직접 만든 단색 이미지에서만 색상으로 정답을 재확인할 수 있어서 사용했다. 실제 사진에서는 이미지와 마스크의 공통 변환 규칙을 확인하고 오버레이로 경계를 점검한다.

<figure class="seg-figure">
  <img src="/image-segmentation-mask-basics/paired-flip-ko.svg" alt="원본 이미지와 마스크, 둘 다 좌우 반전한 결과, 이미지만 반전해서 정답이 어긋난 결과를 비교" loading="lazy" />
  <figcaption>이미지와 마스크는 shape가 같아도 대응이 어긋날 수 있다. 두 공의 크기를 다르게 만든 이유도 이런 어긋남을 눈으로 확인하기 위해서다.</figcaption>
</figure>

무작위로 반전 여부를 정한다면 그 결정을 한 번만 내려야 한다. 이미지와 마스크에서 각각 난수를 뽑으면 한쪽만 반전될 수 있다.

~~~python
rng = np.random.default_rng(42)  # 재현 가능한 난수 생성기
do_flip = rng.random() < 0.5     # 한 번 정한 결과를 두 배열에 함께 적용

if do_flip:
    image_aug = toy[:, ::-1, :].copy()
    mask_aug = semantic[:, ::-1].copy()
else:
    image_aug = toy.copy()
    mask_aug = semantic.copy()
~~~

자르기와 회전도 같은 원리다. 자를 행·열 범위나 회전 각도를 한 번 정해 두 배열에 함께 적용한다. 반면 밝기·색상 변환은 픽셀 대응과 정답 의미가 유지되는 범위에서 이미지에만 적용한다. 마스크의 클래스 번호를 밝게 하거나 정규화해서는 안 된다.

일반적인 학습에서는 데이터를 먼저 학습용과 검증용으로 나누고, 무작위 증강은 학습용에 적용한다. 검증용에서도 모델 입력 크기를 맞추는 일정한 resize는 필요할 수 있다. 학습용 증강과 검증용 전처리를 구분해야 한다.

## 13. 마스크 resize에서는 클래스 번호를 평균내지 않는다

이미지 크기를 바꿀 때 새 픽셀 값을 정하는 방법을 <strong>보간(Interpolation)</strong>이라고 한다. 이미지의 색상은 주변 값을 섞어 중간 색을 만들 수 있다. 하지만 클래스 번호는 종류를 나타내는 이름 역할을 하므로 평균값에 의미가 없다.

예를 들어 클래스 0과 클래스 2의 경계를 섞어서 1이 생기면 땅 클래스가 아닌 곳에 땅 번호가 나타날 수 있다. 출력이 정수 자료형이어도 이런 오류가 없어지는 것은 아니다. 소수 결과를 정수로 바꾸는 것만으로 원래 의미가 복구되지는 않는다.

<strong>최근접 이웃 보간(Nearest Neighbor)</strong>은 원본 픽셀 하나를 선택해 값을 가져온다. 클래스 번호 마스크를 resize할 때는 이 방식으로 번호가 섞이지 않게 한다.[^resize]

~~~python
new_H, new_W = 60, 100

image_small = cv2.resize(
    toy, (new_W, new_H), interpolation=cv2.INTER_AREA
)
mask_small = cv2.resize(
    semantic, (new_W, new_H), interpolation=cv2.INTER_NEAREST
)

assert image_small.shape[:2] == mask_small.shape
assert np.isin(np.unique(mask_small), np.unique(semantic)).all()
~~~

`cv2.resize`의 크기 인자는 `(너비, 높이)` 순서다. 이미지와 마스크는 같은 목표 크기로 바꾸되, 이미지 축소에는 `INTER_AREA`, 클래스 번호 마스크에는 `INTER_NEAREST`를 사용했다.

이 검사는 resize 후에 새 클래스 번호가 생기지 않았는지 확인한다. 원래 있던 모든 클래스가 남아 있어야 한다는 뜻은 아니다. 아주 작은 영역은 축소하면서 사라질 수 있다. 또한 허용된 번호만 남았다는 사실만으로 경계 대응까지 맞다고 보장할 수는 없으므로 시각 확인도 함께 한다.

## 14. 실제 마스크를 읽을 때는 파일 형식과 라벨 규칙부터 확인한다

지금까지는 모든 배열을 직접 만들었으므로 값의 의미를 알고 있었다. 실제 데이터셋에서는 파일을 읽기 전에 이미지 파일, 마스크 파일, 클래스 이름·번호를 연결하는 메타데이터를 나눠 확인해야 한다. 파일 목록 두 개를 각각 정렬했다고 해서 같은 인덱스끼리 한 쌍이라고 단정하면 안 된다. 데이터셋이 제공하는 파일 이름이나 샘플 ID 규칙을 확인한다.

<div class="seg-table">

| 확인할 정보 | 확인하는 이유 |
| --- | --- |
| 이미지와 마스크의 연결 규칙 | 다른 이미지의 정답을 붙이는 오류 방지 |
| 마스크의 shape와 자료형 | 번호 배열인지 RGB 색상 배열인지 구분 |
| `np.unique(mask)`와 클래스 표 | 실제 저장 번호와 학습용 번호의 대응 확인 |
| 255·-1 같은 특수 값 | 무시할 영역인지 실제 클래스인지 확인 |
| 이미지와 마스크의 높이·너비 | 공간 크기와 픽셀 대응 점검 |

</div>

PNG 마스크라고 해서 저장 방식이 모두 같지는 않다. 픽셀에 클래스 번호를 직접 저장한 파일도 있고, 색상으로 클래스를 구분한 RGB 파일도 있다. 팔레트 PNG는 픽셀 값에 팔레트 인덱스를 저장할 수 있다. 화면에서 컬러로 보인다는 사실만으로 배열이 RGB라고 판단하지 않는다.[^pillow]

Pillow의 `Image.open`은 이미지 파일을 읽는 함수다. 마스크를 곧바로 RGB나 흑백으로 바꾸지 말고 원래 모드와 저장 값을 먼저 확인한다.

~~~python
from PIL import Image

# mask_path는 해당 데이터셋의 마스크 파일 경로다.
with Image.open(mask_path) as mask_image:
    print(mask_image.mode, mask_image.size)
    raw_mask = np.array(mask_image)

print(raw_mask.shape, raw_mask.dtype)
if raw_mask.ndim == 2:
    print(np.unique(raw_mask))
~~~

이 코드는 실제 `mask_path`를 지정한 뒤 사용하는 파일 검사 예시다. Pillow의 `size`는 `(너비, 높이)`이고, NumPy의 2차원 shape는 `(높이, 너비)`다. `raw_mask`가 RGB라면 각 픽셀의 세 성분을 데이터셋의 색상표와 대조해 번호로 변환해야 한다. `np.unique(raw_mask)`만 호출하면 색상 조합이 아니라 개별 성분 값들을 모으므로 클래스 목록으로 해석하면 안 된다.

정답 번호는 PNG처럼 손실 없이 저장할 수 있는 형식으로 보관한다. JPEG는 압축 과정에서 값이 달라질 수 있어 정확한 클래스 번호를 보존할 마스크 저장 형식으로 사용하지 않는다.

## 15. PyTorch 학습에서는 입력·정답·출력을 따로 맞춘다

앞의 배열 연산을 딥러닝 코드로 연결해 보자. <strong>텐서(Tensor)</strong>는 PyTorch에서 다차원 데이터를 다루는 자료형이고, <strong>배치(Batch)</strong>는 한 번에 처리할 이미지 묶음이다. 아래의 `N`은 배치에 포함한 이미지 수다.

픽셀마다 하나의 클래스를 선택하는 문제에서 `CrossEntropyLoss`에 클래스 번호 정답을 전달하는 경우의 형식은 다음과 같다. <strong>손실(Loss)</strong>은 모델의 예측 점수와 정답의 어긋남을 계산한 값이다.[^crossentropy]

<div class="seg-table">

| 대상 | shape | 자료형과 의미 |
| --- | --- | --- |
| 입력 `images` | `(N, 3, H, W)` | `float32`, RGB 이미지 |
| 정답 `targets` | `(N, H, W)` | `long`, 픽셀별 클래스 번호 |
| 모델 출력 `logits` | `(N, C, H, W)` | 실수, 픽셀별 클래스 점수 |
| 예측 번호 `predictions` | `(N, H, W)` | 클래스 축에서 최고 점수인 인덱스 |

</div>

<strong>로짓(Logit)</strong>은 확률로 바꾸기 전의 모델 출력 점수다. `CrossEntropyLoss`의 입력에는 이 점수를 그대로 전달한다. 지금처럼 정답을 클래스 번호로 전달할 때는 원-핫으로 바꿀 필요가 없다. 정답의 `long`은 PyTorch의 64비트 정수 자료형이다.

먼저 NumPy 배열에서 텐서를 만든다. `torch.from_numpy`는 NumPy 배열의 데이터를 공유하는 CPU 텐서를 만든다. `.float()`와 `.long()`은 자료형을 바꾸고, `.unsqueeze(0)`은 맨 앞에 크기 1인 배치 축을 추가한다. 변환 이후의 텐서를 수정할 때 원본 공유 여부가 문제가 될 수 있으므로 원본 보존이 필요하면 복사한다.[^fromnumpy]

~~~python
import torch
from torch import nn

# 이미지: RGB 성분 축을 앞으로 옮기고 0~1 실수로 변환한다.
image_chw = np.ascontiguousarray(toy.transpose(2, 0, 1))
images = torch.from_numpy(image_chw).float().unsqueeze(0) / 255.0

# 정답: 클래스 번호를 그대로 유지한다. 255로 나누지 않는다.
targets = torch.from_numpy(semantic.copy()).long().unsqueeze(0)

print(tuple(images.shape), images.dtype)
# (1, 3, 120, 200) torch.float32
print(tuple(targets.shape), targets.dtype)
# (1, 120, 200) torch.int64
~~~

`np.ascontiguousarray`는 축을 바꾼 뒤 데이터를 순서대로 이어진 메모리 배열로 준비한다. 여기서는 변환 결과를 다루기 쉽게 만들기 위해 사용했다. 원래 RGB가 0~255라는 조건에서만 위의 `/ 255.0`을 적용한다. 실제 사전학습 모델에서는 그 모델이 요구하는 입력 정규화도 확인해야 한다.

이제 입력부터 손실까지 실제로 연결해 본다. 다음 모델은 합성곱 층 하나로 RGB 성분을 받아 클래스별 점수 `C`개를 출력한다. 합성곱 층의 가중치는 주변 픽셀 성분들을 얼마만큼 반영할지 정하는 학습 대상 값이다. `kernel_size=3`은 3×3 영역을 사용하고, `padding=1`은 기본 이동 간격 1일 때 높이와 너비가 줄지 않도록 가장자리에 한 칸을 더한다.[^conv2d]

~~~python
model = nn.Conv2d(3, C, kernel_size=3, padding=1)
criterion = nn.CrossEntropyLoss()

logits = model(images)
assert logits.shape[0] == targets.shape[0]
assert logits.shape[1] == C
assert logits.shape[2:] == targets.shape[1:]

loss = criterion(logits, targets)
predictions = logits.argmax(dim=1)

print(tuple(logits.shape))       # (1, 3, 120, 200)
print(tuple(predictions.shape))  # (1, 120, 200)
print(loss.ndim)                 # 0: 손실 값 하나
~~~

이 모델은 입력·출력 연결을 확인하는 작은 예시다. 실제 사진을 잘 분할하는 모델을 학습했다는 의미는 아니다. 아직 가중치를 갱신하지 않았으며, 임의로 초기화한 점수에서 예측 번호를 꺼낸 상태다. 가중치를 학습하려면 손실에서 기울기를 계산하고 옵티마이저로 갱신하는 단계가 추가로 필요하다.

NumPy에서는 `axis`, PyTorch에서는 위 연산의 축 인자를 `dim`이라고 쓴다. 클래스 축이 1번이므로 `argmax(dim=1)`이고, 배치·높이·너비 축은 그대로 남는다. 예측 번호만 필요하다면 가장 큰 점수의 인덱스를 바로 선택할 수 있다.

학습용 손실에는 `logits`를 전달한다. `argmax`로 만든 클래스 번호를 전달하면 클래스별 점수와 학습에 필요한 연결이 사라진다. 또 이 글의 마스크처럼 정수 클래스 번호를 담은 배열에 이미지용 `ToTensor`를 그대로 적용하면 일부 입력 형식에서 255로 나눠질 수 있으므로, 정답은 위처럼 정수로 직접 변환한다.[^totensor]

검증과 추론에서는 가중치를 갱신하지 않는다. 실제 모델을 사용할 때는 `model.eval()`로 평가 동작을 선택하고 `torch.no_grad()` 안에서 출력을 만든다. 정답이 있으면 예측과 비교해 평가할 수 있고, 정답이 없는 추론에서는 예측을 생성하고 표시할 수 있다.

## 16. 실행 결과를 확인하고 다음 데이터에 적용한다

본문의 배열 예제는 [examples.py](/image-segmentation-mask-basics/examples.py)에 모았다. 외부 데이터나 학습된 모델을 내려받지 않고 실행할 수 있다. 기본 실행은 이미지와 마스크 생성, 연결 영역, 원-핫 복원, 픽셀 수, 오버레이, 같은 좌우 반전과 최근접 resize를 검증한다.

macOS의 zsh 또는 Linux의 bash·zsh에서, 파일을 저장한 폴더에서 다음과 같이 실행한다. 가상환경은 이 예제의 패키지를 다른 작업과 분리해 설치하는 폴더다.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy opencv-python-headless
python examples.py
~~~

`opencv-python-headless`는 OpenCV의 창 표시 기능을 제외한 패키지다. 이 예제는 `cv2.imshow`를 사용하지 않으므로 충분하다. 기존 환경에 `opencv-python`이 있다면 두 패키지를 함께 추가할 필요는 없다. Matplotlib 시각화 코드를 실행하려면 `matplotlib`, 파일 검사 코드를 실행하려면 `pillow`가 추가로 필요하다.

PyTorch가 설치된 환경에서는 선택 검증도 실행할 수 있다. 장치별 GPU 설치는 별도 환경에 따르며, 이 작은 형식 검증은 CPU에서 실행한다.

~~~bash
python examples.py --torch-check
~~~

실행 환경과 실제 출력은 [verification.txt](/image-segmentation-mask-basics/verification.txt)에 기록했다. 예제의 합격 기준은 이미지가 그럴듯해 보이는 데서 끝나지 않는다.

<div class="seg-table">

| 검증 | 확인하는 관계 |
| --- | --- |
| 이미지 공간 크기와 마스크 shape 비교 | 같은 픽셀 격자를 사용하는가 |
| 클래스 번호 범위 검사 | 변환표의 유효한 행을 조회하는가 |
| 원-핫 합과 `argmax` 역변환 | 표현을 바꿔도 픽셀별 클래스가 유지되는가 |
| 클래스별 픽셀 수의 합 | 전체 픽셀을 빠짐없이 세었는가 |
| 연결 영역 마스크에서 시맨틱 재구성 | 영역을 나눠도 원래 클래스 마스크를 복원하는가 |
| 이미지·마스크의 같은 반전 | 픽셀과 정답의 대응이 유지되는가 |
| 최근접 resize 뒤 번호 검사 | 클래스 번호가 섞이지 않았는가 |

</div>

실무에서 먼저 익힐 것은 이미지·마스크 구조 조사, 인덱싱과 `axis`, 클래스 번호 검증, 함께 적용하는 공간 변환, 모델 입출력의 shape다. 원-핫·팔레트·연결 요소는 필요한 표현이나 처리 목적에 맞춰 꺼내 쓰면 된다. 실제 데이터에서 연결 요소가 곧 객체라는 가정이나, 모든 손실 함수가 같은 정답 형식을 받는다는 가정은 하지 않는다.

다음 데이터셋을 만났을 때 <strong>각 변수에 저장된 정보, 축의 의미, 배열끼리 연결되는 규칙, 바꿔도 유지해야 하는 관계</strong>를 설명하고 검증할 수 있으면 이번 기초의 목표에 도달한 것이다.

## 참고 자료

[^indexing]: NumPy, [Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html). 기본 슬라이싱, 정수 배열 인덱싱, 불리언 인덱싱과 뷰.
[^opencv-image]: OpenCV, [Image file reading and writing](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html). 이미지 읽기와 기본 컬러 채널 순서.
[^broadcasting]: NumPy, [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). 오른쪽 축부터 적용하는 크기 대응 규칙.
[^all]: NumPy, [numpy.all](https://numpy.org/doc/stable/reference/generated/numpy.all.html). 축 지정과 `keepdims`.
[^components]: OpenCV, [Structural Analysis and Shape Descriptors](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html). 연결 요소 입력·출력, 배경 라벨, 4·8 연결성.
[^stack]: NumPy, [numpy.stack](https://numpy.org/doc/stable/reference/generated/numpy.stack.html). 같은 shape의 배열을 새 축으로 결합.
[^eye]: NumPy, [numpy.eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html). 대각선에 1을 둔 변환표 생성.
[^argmax]: NumPy, [numpy.argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html). 최댓값의 인덱스와 동점 처리.
[^bincount]: NumPy, [numpy.bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html). 정수별 등장 횟수와 최소 출력 길이.
[^imshow]: Matplotlib, [matplotlib.pyplot.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html). 컬러맵, 값 범위와 표시 보간.
[^resize]: OpenCV, [Geometric Image Transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html). resize 크기와 보간 방식.
[^pillow]: Pillow, [Concepts — Modes](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes). `P`, `L`, `RGB` 등 이미지 모드.
[^crossentropy]: PyTorch, [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). 원시 출력 점수와 클래스 번호 정답의 shape·자료형.
[^fromnumpy]: PyTorch, [torch.from_numpy](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html). NumPy 배열과 텐서의 메모리 공유.
[^conv2d]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html). 입력·출력 채널, 커널과 패딩.
[^totensor]: TorchVision, [ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html). 일부 이미지 입력의 값 범위 변환과 정답 마스크 사용 주의.
