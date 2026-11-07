+++
title = "문서 복원 오토인코더 실습: 데이터 준비부터 K-Fold와 Kaggle 제출까지"
description = "손상된 문서와 깨끗한 정답을 짝짓고, 패딩과 텐서 모양을 확인한 뒤 오토인코더를 학습한다. 교차 검증, 이미지별 RMSE와 전체 픽셀 RMSE, PSNR, 원본 크기 복구, Kaggle 제출을 하나의 흐름으로 연결한다."
date = 2026-11-07
slug = "document-denoising-kfold-kaggle"

[extra]
katex = true
toc = true
styles = ["document-denoising-kfold-kaggle/article.css"]
+++

문서의 얼룩을 지우는 모델을 만들 때는 이미지를 깨끗하게 출력하는 것만 확인해서는 부족하다. 입력과 정답이 같은 문서인지, 검증 이미지가 학습에 들어가지는 않았는지, 원본에 없던 여백까지 점수에 포함했는지를 함께 살펴야 한다. 이 연결 중 하나가 어긋나면 학습은 정상적으로 실행되더라도 결과를 잘못 해석할 수 있다.

이번에는 Kaggle의 **Denoising Dirty Documents** 데이터를 예로 들어, 파일을 읽는 단계부터 제출 파일을 만드는 단계까지 따라간다. [앞의 오토인코더 글](/denoising-autoencoder-pytorch/)이 모델 한 배치의 학습을 다뤘다면, 여기서는 데이터 준비와 검증을 포함한 전체 실험을 구성한다.

[전체 Python 실행 예제](/document-denoising-kfold-kaggle/examples-ko.py)도 함께 제공한다. 본문 코드는 각 단계의 역할을 설명하는 예시이고, 다운로드 파일은 데이터 읽기·교차 검증·최종 학습·제출 파일 생성을 연결한 프로그램이다. 데이터 수와 과거 점수는 실습 기록에 따른 값이다. 새로 작성한 예제의 동작 확인과 과거 실험의 성능 재현은 구분한다.

## 01. 입력과 정답은 어디에서 만나는가

목표는 손상된 문서를 넣었을 때 깨끗한 문서에 가까운 이미지를 얻는 것이다. <strong>오토인코더(Autoencoder)</strong>는 입력에서 중간 표현을 계산하는 인코더와, 그 표현에서 출력 형태를 계산하는 디코더를 연결한 신경망이다. 이 글에서는 손상 이미지와 깨끗한 정답의 쌍을 사용해 복원하도록 학습한다.

| 이름 | 내용 | 사용하는 곳 |
| --- | --- | --- |
| `X`, `x_batch` | 얼룩이나 잡음이 있는 문서 | 모델의 입력 |
| `z` | 입력에서 계산한 중간 특징 | 디코더의 입력 |
| `pred` | 모델이 출력한 복원 이미지 | 정답과 비교할 예측 |
| `Y`, `y_batch` | 같은 문서의 깨끗한 이미지 | 손실과 평가의 기준 |

**정답 이미지는 디코더에 넣지 않는다.** 디코더는 인코더가 계산한 `z`를 받는다. 깨끗한 정답 `Y`는 모델 출력 `pred`와 비교하는 데 사용한다. 분류 문제에서 정답이 클래스 번호였다면, 이번 문제에서는 이미지 전체가 정답이다.

<figure class="document-figure">
<img src="/document-denoising-kfold-kaggle/overview-ko.svg" alt="손상 입력 X가 인코더와 디코더를 지나 예측 이미지가 된다. 별도로 준비한 깨끗한 정답 Y는 예측과 손실을 계산하는 곳에서 만나며, 학습에서는 손실을 이용해 모델의 파라미터를 수정한다." loading="lazy">
<figcaption>입력·중간 특징·출력은 모델 계산으로 연결되고, 정답은 출력의 오차를 계산하는 기준으로 연결된다.</figcaption>
</figure>

<strong>파라미터(Parameter)</strong>는 학습 중 조절하는 가중치와 편향이다. <strong>손실(Loss)</strong>은 예측과 정답의 차이를 학습 목적에 맞게 요약한 값이다. <strong>역전파(Backpropagation)</strong>는 파라미터를 조금 바꿀 때 손실이 어떻게 달라지는지를 나타내는 기울기를 계산한다. <strong>옵티마이저(Optimizer)</strong>는 그 기울기로 파라미터를 갱신한다.

같은 모델 호출도 목적에 따라 다음처럼 구분한다.

| 작업 | 정답 사용 | 파라미터 변경 | 확인하는 것 |
| --- | --- | --- | --- |
| 학습(Training) | 손실 계산에 사용 | 변경함 | 학습 이미지에서 오차를 줄이는 과정 |
| 검증(Validation) | 점수 계산에 사용 | 변경하지 않음 | 학습에 쓰지 않은 이미지에서의 성능 |
| 추론(Inference) | 없어도 됨 | 변경하지 않음 | 주어진 입력의 예측 |

`model(x)` 한 줄은 예측을 계산한다. 그 호출만으로 모델이 학습되지는 않는다.

## 02. 파일 수, 텐서의 축, 인덱스를 나누어 본다

실습에서 확인한 데이터 구성은 다음과 같다.

| 폴더 | 이미지 수 | 역할 |
| --- | ---: | --- |
| `train` | 144장 | 손상된 학습 입력 |
| `train_cleaned` | 144장 | 같은 파일명의 깨끗한 정답 |
| `test` | 72장 | 정답이 공개되지 않은 제출용 입력 |

학습 입력 144장 중 96장은 너비 540·높이 420, 48장은 너비 540·높이 258이었다. 한 번에 묶어 모델에 넣으려면 같은 배치 안의 텐서 모양이 맞아야 한다. 여기서는 작은 이미지의 위아래에 흰 여백을 붙여 높이 420으로 맞춘다.

<strong>텐서(Tensor)</strong>는 숫자와 그 숫자가 놓인 축 구조를 함께 다루는 자료형이다. <strong>모양(shape)</strong>에는 각 축의 크기가 순서대로 적힌다. 같은 이미지라도 라이브러리에 따라 표기 순서가 다르다.

| 상태 | 모양의 순서 | 작은 문서의 예 |
| --- | --- | --- |
| Pillow 이미지의 `img.size` | 너비·높이 | `(540, 258)` |
| 흑백 NumPy 배열 | 높이·너비 | `(258, 540)` |
| 이미지 한 장의 PyTorch 텐서 | 채널·높이·너비 | `(1, 258, 540)` |
| 패딩한 이미지 24장의 배치 | 이미지·채널·높이·너비 | `(24, 1, 420, 540)` |

배치의 모양을 `(B, C, H, W)`라고 쓰겠다. `B`는 이번에 넣는 이미지 수, `C`는 채널 수, `H`와 `W`는 높이와 너비다. 전체 데이터 수를 나타낼 때는 `N`을 사용한다.

| 축 번호 | 의미 | `(24, 1, 420, 540)`에서 크기 | 유효한 인덱스 |
| ---: | --- | ---: | --- |
| 0 | 이미지 | 24 | 0~23 |
| 1 | 채널 | 1 | 0 |
| 2 | 높이 | 420 | 0~419 |
| 3 | 너비 | 540 | 0~539 |

`dim=2`는 높이 축을 가리킨다. 그 축의 크기가 2라는 뜻은 아니다. `x[2]`는 첫 번째 축에서 인덱스 2에 해당하는 이미지 한 장을 선택한다. 축을 지정하는 번호와 원소를 선택하는 인덱스도 구분해야 한다.

`x[i]`는 `(1,H,W)`이고, `x[i,0]`는 채널까지 선택한 `(H,W)`다. 이미지 한 장을 선택하면서 배치 축을 남기려면 `x[i:i+1]`을 사용한다. 이 결과는 `(1,1,H,W)`다.

Pillow의 `img.size`는 속성이고, 텐서의 `x.size()`는 호출하는 메서드다. 텐서는 `x.shape`로도 모양을 확인할 수 있다. `x.size == (...)`처럼 메서드 자체와 튜플을 비교하면 의도한 검사가 되지 않는다.

## 03. 정렬보다 먼저 확인할 것은 파일명 대응이다

파일 경로는 이미지가 저장된 위치다. 경로 목록을 만들었다고 이미지 픽셀을 읽은 것은 아니다. Python 표준 라이브러리 `pathlib`의 `Path`를 사용하면 폴더와 파일 경로를 다루기 편하다.[^pathlib]

아래 코드에서 `data_dir`은 `train`, `train_cleaned`, `test` 폴더를 담고 있는 위치다. 노트북의 현재 작업 위치가 항상 노트북 파일의 저장 위치와 같지는 않으므로 먼저 출력해서 확인한다.

```python
from pathlib import Path

data_dir = Path.cwd()
train_paths = sorted((data_dir / "train").glob("*.png"))
clean_paths = sorted((data_dir / "train_cleaned").glob("*.png"))

print(data_dir)
print(len(train_paths), len(clean_paths))

assert train_paths, "train 폴더에서 PNG 파일을 찾지 못했습니다."
assert {p.name for p in train_paths} == {p.name for p in clean_paths}
```

`data_dir / "train"`의 `/`는 경로를 이어 붙인다. 폴더를 생성하지는 않는다. `glob("*.png")`는 바로 아래 폴더에서 PNG 경로를 찾는다. `sorted`는 경로를 정렬하지만 숫자 크기 순서를 자동으로 해석하지는 않는다. 파일명이 `1.png`, `10.png`, `2.png` 순서여도 파일명으로 정답을 찾으면 대응에는 문제가 없다.

손상 파일 `train/12.png`의 정답은 `train_cleaned/12.png`다. 입력 목록과 정답 목록을 별개로 가공한 뒤 순서가 같기를 기대하기보다, 입력의 이름으로 정답 경로를 만드는 편이 분명하다.

```python
from PIL import Image

noisy_imgs, clean_imgs, names = [], [], []

for noisy_path in train_paths:
    clean_path = data_dir / "train_cleaned" / noisy_path.name

    # 변환된 이미지 객체를 얻은 다음 원래 파일은 닫는다.
    with Image.open(noisy_path) as opened:
        noisy = opened.convert("L")
    with Image.open(clean_path) as opened:
        clean = opened.convert("L")

    assert noisy.size == clean.size
    noisy_imgs.append(noisy)
    clean_imgs.append(clean)
    names.append(noisy_path.name)

assert len(noisy_imgs) == len(clean_imgs) == len(names) > 0
```

`convert("L")`은 검정 0부터 흰색 255까지 표현하는 8비트 흑백 이미지로 바꾼다. `noisy_imgs`와 `clean_imgs`는 아직 Pillow 이미지를 담은 리스트다.[^pillow]

여기서 `clean_paths.append(clean)`이라고 쓰면 파일 경로를 담던 리스트에 이미지 객체를 섞게 된다. 경로·이미지·텐서를 구분해서 이름을 붙이는 이유가 있다.

두 리스트를 `zip`으로 묶을 때도 길이를 확인한다. 기본 `zip`은 짧은 쪽이 끝나면 종료한다. 한쪽 리스트가 비어 있으면 반복문이 한 번도 실행되지 않으므로, 반복문 뒤의 “검사 완료” 출력만으로 전체 데이터가 검사됐다고 판단할 수 없다.[^builtins]

크기별 개수만 확인하려면 다음으로 충분하다.

```python
counts = {}
for image in noisy_imgs:
    size = image.size
    counts[size] = counts.get(size, 0) + 1

print(counts)               # 실습 기록: {(540, 420): 96, (540, 258): 48}
print(sum(counts.values())) # 144
```

딕셔너리의 값은 이미지 수다. 이 코드가 실제 이미지들을 크기별로 나누어 보관한 것은 아니다.

압축 파일을 사용한다면 먼저 내부 경로를 확인한다. `extractall(data_dir)`는 압축 파일 이름과 같은 폴더를 자동으로 만드는 명령이 아니다. 내부 경로가 `train/12.png`이면 그 경로가 유지되고, `12.png`만 있으면 바로 그 파일이 풀린다. 이미 압축을 풀어 놓았다면 이 과정을 반복할 필요는 없다.

## 04. 텐서 변환, 정규화, 패딩은 서로 다른 작업이다

픽셀을 신경망에 넣기 위해 먼저 실수형 텐서로 바꾼다. 그다음 픽셀 밝기를 0~1 범위로 맞춘다. 마지막으로 이미지 크기를 맞춘다. 세 작업은 목적이 다르다.

```python
import numpy as np
import torch
import torch.nn.functional as F

image = noisy_imgs[0]

x = torch.from_numpy(
    np.array(image, dtype=np.float32)
).unsqueeze(0) / 255.0
```

이 한 줄은 아래 순서로 실행된다.

| 처리 | 결과 | 필요한 이유 |
| --- | --- | --- |
| `np.array(..., dtype=np.float32)` | 실수 배열 `(H,W)` | 픽셀을 실수 계산에 사용 |
| `torch.from_numpy(...)` | 텐서 `(H,W)` | PyTorch 연산에 사용 |
| `unsqueeze(0)` | 텐서 `(1,H,W)` | 흑백 채널 축 추가 |
| `/ 255.0` | 같은 모양, 값은 0~1 | 출력과 정답의 밝기 범위 맞춤 |

`unsqueeze(0)`은 이 단계에서 <strong>채널 축</strong>을 추가한다. 아직 이미지 여러 장을 묶지 않았으므로 배치 축이 아니다. 새 축을 만들 뿐 기존 픽셀을 복제하거나 보간하지 않는다.[^tensor-ops]

<strong>패딩(Padding)</strong>은 이미지 둘레에 영역을 추가하는 작업이다. 원본 글자를 늘이거나 줄이지 않고 크기를 맞추려는 목적이다. 높이 258인 문서는 위아래에 각각 81행을 붙이면 높이 420이 된다.

```python
if x.shape == (1, 258, 540):
    x = F.pad(x, (0, 0, 81, 81), value=1.0)
```

`F.pad`에서 마지막 두 공간 축에 대한 순서는 <strong>왼쪽·오른쪽·위·아래</strong>다. 이미 밝기를 0~1로 바꿨으므로 흰색은 `1.0`이다. `value=1.0`은 새로 붙이는 여백을 채우며, 원본의 글자 픽셀을 흰색으로 덮어쓰지 않는다.[^pad]

<figure class="document-figure">
<img src="/document-denoising-kfold-kaggle/padding-ko.svg" alt="높이 258, 너비 540인 문서의 위아래에 81행씩 흰색 패딩을 추가해 높이 420으로 맞춘다. 원본 영역은 높이 인덱스 81부터 338까지이며 슬라이스는 81:339이다." loading="lazy">
<figcaption>원본 크기와 여백 위치를 기억해 두면 평가와 제출 때 같은 영역을 정확히 되찾을 수 있다.</figcaption>
</figure>

흰 여백을 추가할 때 합성곱 층을 만들 필요는 없다. `F.pad`는 영역을 추가하고, `Conv2d`는 가중치로 특징값을 계산한다. 단순한 크기 맞추기에 학습 가능한 층을 넣으면 불필요한 계산과 설정이 늘어난다.

입력과 정답, 원본 높이를 함께 준비하면 순서가 어긋나는 것을 줄일 수 있다.

```python
def to_tensor(image):
    # Pillow의 (W,H)를 텐서의 (C,H,W)로 바꾼다.
    return torch.from_numpy(
        np.array(image, dtype=np.float32)
    ).unsqueeze(0) / 255.0


noisy_tensors, clean_tensors, heights = [], [], []

for noisy, clean in zip(noisy_imgs, clean_imgs):
    width, height = noisy.size
    if (width, height) not in {(540, 420), (540, 258)}:
        raise ValueError(f"예상하지 못한 크기: {noisy.size}")

    x, y = to_tensor(noisy), to_tensor(clean)
    top = (420 - height) // 2
    bottom = 420 - height - top

    noisy_tensors.append(F.pad(x, (0, 0, top, bottom), value=1.0))
    clean_tensors.append(F.pad(y, (0, 0, top, bottom), value=1.0))
    heights.append(height)  # 파일명·입력·정답과 같은 순서로 저장한다.

train_x = torch.stack(noisy_tensors, dim=0)
train_y = torch.stack(clean_tensors, dim=0)
heights = torch.tensor(heights, dtype=torch.int64)

assert train_x.shape == train_y.shape
print(tuple(train_x.shape))  # 이 데이터 전체라면 (144, 1, 420, 540)
```

이 글의 코드는 파일 순서를 유지한다. 크기별 리스트를 따로 만든 뒤 합치는 방법도 가능하지만, 그 경우 파일명과 원본 높이도 정확히 같은 순서로 합쳐야 한다. 이미지의 모양만 같다고 메타데이터까지 맞는 것은 아니다.

## 05. stack, cat, flatten은 무엇을 바꾸는가

이름이 비슷해도 서로 바꾸는 대상이 다르다. 먼저 현재 자료가 Python 리스트인지 텐서인지 확인한다.

| 연산 | 하는 일 | 이 실습의 예 |
| --- | --- | --- |
| 리스트 `a + b` | 두 리스트를 이어 붙임 | 이미지 텐서를 담은 목록 연결 |
| `torch.stack` | 새 축을 만들어 묶음 | 한 장씩 `(1,H,W)` → `(N,1,H,W)` |
| `torch.cat` | 기존 축을 따라 이어 붙임 | 여러 예측 배치 → 전체 예측 배치 |
| `flatten(1)` | 축 1부터 뒤쪽 축을 하나로 펼침 | `(B,C,H,W)` → `(B,C×H×W)` |
| `mean` | 지정한 범위의 평균 계산 | 이미지마다 오차 하나 계산 |

이미지 한 장씩은 아직 배치 축이 없다. 이들을 `stack`으로 묶을 때 새로 생기는 축이 이미지 축이다. 반대로 배치 추론의 결과는 이미 배치 축을 갖는다. 그런 결과를 모을 때는 `cat(..., dim=0)`으로 기존 이미지 축을 이어 붙인다.[^tensor-ops]

```python
# 각각 (1,420,540)인 한 장짜리 텐서들을 묶는다.
batch = torch.stack(noisy_tensors[:3], dim=0)
print(tuple(batch.shape))  # (3,1,420,540)

# 이미 배치 축이 있는 결과를 이어 붙인다.
joined = torch.cat([batch[:2], batch[2:]], dim=0)
print(tuple(joined.shape)) # (3,1,420,540)

# 모양을 펼칠 뿐 평균은 아직 계산하지 않는다.
flat = batch.flatten(1)
print(tuple(flat.shape))   # (3,226800)
```

전체 데이터 `train_x`의 첫 축이 144라고 해서 학습 배치 크기도 144인 것은 아니다. 전체에서 일부를 꺼내 24장씩 묶으면 그때 모델에 들어가는 `B`가 24다.

## 06. 모델 구조와 실제 이미지 입력을 구분한다

모델을 정의할 때 필요한 것은 사용할 층의 종류와 연결 순서다. 모델을 호출할 때 필요한 것은 실제 입력 텐서다.

`torch.nn`에는 신경망 층과 손실 함수가 들어 있다. `from torch import nn`으로 가져오면 `torch.nn.Conv2d`를 `nn.Conv2d`라고 쓸 수 있다. <strong>클래스(Class)</strong>는 객체를 만드는 정의다. 아래 `CustomModel` 클래스의 `__init__`은 사용할 층을 준비하고, `forward`는 입력이 통과할 계산 순서를 정한다.

```python
from torch import nn

class CustomModel(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

`nn.Sequential`은 안에 넣은 층을 순서대로 실행한다. `ReLU`는 음수를 0으로 바꾸고 양수는 유지한다. `Sigmoid`는 출력 밝기를 정답과 같은 0~1 범위에 맞춘다. 두 활성화 함수는 모양을 바꾸지 않는다.[^activations]

`CustomModel(channels=16)`의 16은 <strong>생성자에 전달하는 구조 설정</strong>이다. `model(x_batch)`의 `x_batch`는 <strong>실제 계산할 이미지</strong>다. 배치 크기 24는 `Conv2d`의 생성자 인자가 아니다.

| 단계 | 출력 모양 | 값의 의미 |
| --- | --- | --- |
| 입력 | `(B,1,420,540)` | 손상된 문서의 밝기 |
| 합성곱·ReLU | `(B,16,210,270)` | 입력에서 계산한 16채널 특징 |
| 전치 합성곱·Sigmoid | `(B,1,420,540)` | 복원한 문서의 밝기 |

<strong>특징맵(Feature Map)</strong>은 입력에서 계산한 공간별 특징값이다. 합성곱 필터 하나는 모든 입력 채널을 함께 처리해 출력 채널 하나를 만든다. 이 모델의 첫 층은 흑백 입력 1채널에서 출력 특징 16채널을 계산한다. 필터의 가중치는 이미지 위치마다 새로 만드는 것이 아니라 여러 위치에서 공유한다.[^conv]

중간 텐서 `z`는 <strong>잠재 표현(Latent Representation)</strong>이라고도 부른다. 다만 공간 크기가 절반이 됐다고 전체 숫자 개수가 줄었다고 말할 수는 없다. 높이와 너비는 각각 절반이지만 채널이 1개에서 16개로 늘었다. 이 설정의 중간 원소 수는 입력의 4배다. 여기서 목표는 문서 복원이며, 저장 용량을 줄이는 압축에 성공했다는 주장은 하지 않는다.

파라미터 수는 중간값 개수와도 다르다.

| 층 | 가중치 모양과 축 의미 | 편향 포함 파라미터 수 |
| --- | --- | ---: |
| `Conv2d` | `(16,1,3,3)`: 출력 채널·입력 채널·커널 높이·너비 | `16×1×3×3 + 16 = 160` |
| `ConvTranspose2d` | `(16,1,4,4)`: 입력 채널·출력 채널·커널 높이·너비 | `16×1×4×4 + 1 = 257` |
| 합계 | 두 층에 저장된 학습 대상 숫자 | **417** |

이 가중치 모양은 `groups=1`인 현재 설정 기준이다. 두 층의 앞부분이 모두 `(16,1)`처럼 보여도 축의 의미는 다르다.[^transpose]

## 07. 높이가 줄었다가 다시 커지는 이유

합성곱의 출력 위치 수는 커널을 적용할 수 있는 위치 수로 결정된다. 패딩은 계산에 사용할 경계 공간을 추가한다. <strong>이동 간격(stride)</strong>을 키우면 적용 위치 사이가 멀어져 출력 위치 수가 줄어든다.

<strong>전치 합성곱(Transposed Convolution)</strong>은 입력 특징과 별도의 가중치로 출력 특징맵을 계산한다. 인코더에서 잃은 정보를 자동으로 되찾는 역함수가 아니다. 어떤 값을 출력해야 할지는 깨끗한 정답과 비교하는 학습을 통해 조절된다.[^transpose]

출력 모양을 확인할 때 사용할 기호를 먼저 정하자. `L_in`과 `L_out`은 한 방향의 입력·출력 길이, `k`는 그 방향의 커널 크기, `s`는 이동 간격, `p`는 해당 층의 `padding` 설정값이다. 합성곱에서는 경계에 추가하는 패딩 크기지만, 전치 합성곱에서는 출력 영역을 조정하는 설정이므로 같은 방식의 입력 여백 추가로 해석하면 안 된다. 아래 식은 커널 내부 간격인 `dilation=1`, 전치 합성곱의 출력 크기 조정값인 `output_padding=0`인 경우다.

<div class="document-math">
$$
L_{\mathrm{out,conv}}
=
\left\lfloor\frac{L_{\mathrm{in}}+2p-k}{s}\right\rfloor+1
$$
</div>

<div class="document-math">
$$
L_{\mathrm{out,transpose}}
=
(L_{\mathrm{in}}-1)s-2p+k
$$
</div>

`floor`는 소수점 아래를 내린다는 뜻이다. 높이와 너비에 각각 적용한다. 현재 인코더의 `k=3,s=2,p=1`은 높이 420을 210, 너비 540을 270으로 만든다. 디코더의 `k=4,s=2,p=1`은 그 결과를 다시 420과 540으로 만든다.

디코더에 들어가는 높이는 원본의 420이 아니라 인코더가 계산한 210이다. 원본 높이 420을 전치 합성곱에 바로 넣는다고 생각하면 출력 크기를 잘못 예상하게 된다.

줄이는 층을 두 개 쌓으면 홀수 크기를 만나는지도 확인해야 한다. 같은 종류의 설정을 두 번 적용할 때 높이 258은 `258 → 129 → 65 → 130 → 260`으로 바뀔 수 있다. 층을 늘린 뒤에도 출력이 정답과 같은 크기인지 실제로 확인해야 하는 이유다. `output_padding`도 출력 배열에 단순히 0을 덧붙이는 명령으로 해석하면 안 된다.

```python
shape_model = CustomModel(channels=16)
with torch.no_grad():
    sample = torch.zeros(2, 1, 420, 540)
    result = shape_model(sample)

assert result.shape == sample.shape
print(tuple(result.shape))  # (2,1,420,540)
print(sum(p.numel() for p in shape_model.parameters()))  # 417
```

이 검사는 출력 모양과 파라미터 수를 확인한다. 아직 학습 전이므로 복원 성능을 확인한 결과는 아니다.

## 08. 비교 기준을 먼저 정한다

<strong>기준선(Baseline)</strong>은 새 모델의 결과를 판단하기 위한 비교 대상이다. 반드시 “아무것도 하지 않기”나 “가장 작은 모델”을 뜻하지는 않는다.

| 비교 대상 | 만드는 방법 | 답하려는 질문 |
| --- | --- | --- |
| 무처리 기준선 | 손상 입력을 그대로 예측으로 사용 | 모델을 적용한 것이 입력을 그대로 두는 것보다 나은가? |
| 출발점 모델 | 먼저 정한 구조와 설정으로 실제 학습 | 구조나 학습 방법 변경이 도움이 됐는가? |
| 개선 모델 | 한 조건을 바꿔 새로 학습 | 바꾼 조건의 효과가 있는가? |

무처리 기준선은 `baseline_pred = x_val`이면 된다. 학습이 필요하지 않다. 반면 인코더 두 층과 디코더 두 층으로 구성한 `1 → 32 → 64 → 32 → 1` 모델은 실제 학습하는 비교 모델이다. 두 대상을 모두 “기준선”이라고 부를 수 있으므로 무엇을 비교했는지 적어야 한다.

비교할 때는 <strong>같은 검증 이미지, 같은 원본 영역, 같은 지표 집계 방식</strong>을 사용한다. 한 모델은 패딩을 포함하고 다른 모델은 패딩을 빼거나, 한쪽은 이미지별 평균이고 다른 쪽은 전체 픽셀 점수이면 결과의 차이에 여러 원인이 섞인다.

## 09. K-Fold는 모델 하나의 검증 대상을 계속 바꾸는 방법이 아니다

<strong>K겹 교차 검증(K-Fold Cross-Validation)</strong>은 데이터를 K개의 검증 묶음으로 나누고, 각 묶음을 한 번씩 검증에 사용하는 절차다. 한 회차에서는 나머지 묶음으로 모델을 학습한다. 다음 회차에서는 모델을 새로 만들어 다른 학습·검증 구성으로 실험한다.[^kfold]

실습의 144쌍을 6개 묶음으로 나누면 묶음마다 24쌍이다. 한 회차에서 120쌍으로 학습하고, 남은 24쌍으로 검증한다.

<figure class="document-figure">
<img src="/document-denoising-kfold-kaggle/kfold-ko.svg" alt="A부터 F까지 여섯 데이터 그룹에서 매 회차 하나를 검증으로 사용한다. 각 회차는 새 모델과 새 옵티마이저로 시작하며 학습 120쌍, 검증 24쌍을 사용한다." loading="lazy">
<figcaption>같은 이미지가 여러 회차의 학습에 사용될 수 있다. 그 이미지의 검증 예측은 해당 이미지를 학습하지 않은 회차의 모델이 만든다.</figcaption>
</figure>

한 회차의 모델을 학습하는 동안에는 학습 이미지와 검증 이미지의 구성원을 고정한다. 학습을 여러 번 반복한다고 검증 이미지를 학습 쪽으로 옮기지 않는다.

| 단위 | 의미 | 현재 설정 |
| --- | --- | --- |
| 배치(Batch) | 한 번에 모델에 넣는 묶음 | 24쌍 |
| 에포크(Epoch) | 현재 학습 데이터 전체를 한 번 사용 | 120쌍을 배치 5개로 처리 |
| Fold 회차 | 하나의 고정된 분할로 새 모델을 학습·검증 | 6회 반복 |

이번 코드처럼 배치마다 한 번 갱신한다면 30에포크 동안 한 모델의 파라미터를 150번 갱신한다. 검증 24쌍은 그 과정에 포함되지 않는다. 6-Fold와 30에포크는 서로 다른 설정이다.

144가 6으로 나누어떨어진다는 것은 계산을 보기 편하게 만든다. 이것이 6-Fold가 통계적으로 최선이라는 근거는 아니다. 5-Fold에서는 검증 이미지가 28장 또는 29장이 될 수 있고, 마지막 배치가 더 작아도 학습할 수 있다.

### KFold가 반환하는 것은 이미지가 아니라 번호다

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=6, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_x), start=1):
    x_train = train_x[train_idx]
    y_train = train_y[train_idx]
    x_val = train_x[val_idx]
    y_val = train_y[val_idx]
    print(fold, len(train_idx), len(val_idx))
```

`train_idx`와 `val_idx`는 선택할 표본 번호의 배열이다. 입력과 정답에 **같은 번호 배열을 사용하기 때문에** 짝이 유지된다. `KFold`가 파일명이나 문서 내용을 보고 정답을 찾아 주는 것은 아니다.

일반 `KFold`는 정답의 종류에 따라 분할하지 않으므로 여기서는 `train_y`를 `split`에 넣을 필요가 없다. `random_state`는 분할을 재현하기 위한 설정이며, 모델의 초기 가중치나 배치 순서까지 함께 고정하지는 않는다.

`enumerate`는 반환된 두 배열 앞에 표시용 회차 번호를 붙인다. 그래서 `fold, (train_idx, val_idx)`처럼 중첩 구조를 풀어 받는다. `fold`는 1부터 시작하는 회차 번호이며 원본 이미지 번호가 아니다.

`next`로 한 회차씩 꺼낼 수도 있다.

```python
folds = kf.split(train_x)
first_train, first_val = next(folds)
second_train, second_val = next(folds)
```

반대로 `next(kf.split(train_x))`를 매번 새로 호출하면 반복자를 새로 만든다. 현재처럼 정수 시드가 고정된 설정에서는 첫 번째 분할을 반복해서 가져온다.


## 10. 같은 문서가 학습과 검증 양쪽에 들어가면 안 된다

<strong>데이터 누수(Data Leakage)</strong>는 처음 보는 대상에서의 성능을 확인하려는데 그 대상의 정보가 학습 과정에 들어간 상태를 뜻한다. 파일명이 서로 다르더라도 같은 원본 문서에서 만들어진 변형 이미지일 수 있다. 이들을 학습과 검증에 나누면 새로운 파일에 대한 평가는 되지만, 새로운 문서 내용에 대한 평가라고 보기 어려울 수 있다.

일반 `KFold`는 이런 관계를 알아내지 않는다. 원본 문서의 관계를 확인할 수 있다면 같은 문서에서 나온 표본에 같은 그룹 번호를 주고 <strong>그룹별 교차 검증(GroupKFold)</strong>을 사용할 수 있다.[^groupfold]

```python
from sklearn.model_selection import GroupKFold

# groups[i]는 train_x[i]가 속한 실제 원본 문서의 그룹 번호다.
# 실제 관계를 조사해서 준비한 배열이 있어야 이 코드를 실행할 수 있다.
splitter = GroupKFold(n_splits=6)
# for train_idx, val_idx in splitter.split(train_x, groups=groups):
#     ... 같은 문서 그룹이 학습·검증 양쪽에 들어가지 않게 학습한다.
```

그룹 번호는 이미지 크기나 임의의 순번으로 대신 만들면 안 된다. 6-Fold를 사용하려면 서로 다른 그룹도 적어도 6개가 필요하다. 그룹 크기에 따라 검증 이미지 수는 회차마다 달라질 수 있다.

이미지의 작은 영역인 <strong>패치(Patch)</strong>를 잘라 학습하더라도 먼저 원본 문서를 분할한다. 그다음 학습 문서에서 패치를 추출한다. 문서에서 만든 패치들을 먼저 섞어 분할하면 같은 문서의 일부가 양쪽에 들어갈 수 있다.

이 글의 실행 예제는 일반 `KFold`를 사용한다. 실제 문서 간 관계를 확인해 그룹 분할을 적용한 결과로 해석해서는 안 된다.

## 11. Dataset은 짝을 꺼내고, DataLoader는 배치를 만든다

`TensorDataset`은 같은 인덱스의 데이터를 함께 꺼내는 객체다. `DataLoader`는 그 데이터셋에서 표본을 뽑아 섞고 배치로 묶어 주는 객체다. 둘은 `torch.utils.data`에 들어 있다.[^data]

```python
from torch.utils.data import TensorDataset, DataLoader

train_idx, val_idx = next(kf.split(train_x))
x_val = train_x[val_idx]
y_val = train_y[val_idx]
val_heights = heights[val_idx]  # 같은 분할의 원본 높이를 선택한다.

dataset = TensorDataset(train_x[train_idx], train_y[train_idx])
loader = DataLoader(dataset, batch_size=24, shuffle=True)

one_x, one_y = dataset[0]
x_batch, y_batch = next(iter(loader))

print(tuple(one_x.shape))   # (1,420,540)
print(tuple(x_batch.shape)) # (24,1,420,540)
```

`TensorDataset`이 두 이미지를 하나로 합친 것은 아니다. `dataset[0]`은 손상 이미지 한 장과 그에 대응하는 정답 한 장을 튜플로 반환한다. 전달한 텐서들은 첫 축의 크기가 같아야 하지만, 뒤쪽 모양이 항상 같아야 하는 것은 아니다. 이 복원 문제에서는 입력과 정답 이미지의 모양도 맞아야 한다.

`shuffle=True`는 데이터셋의 표본을 꺼내는 순서를 섞는다. 한 표본 안의 입력과 정답을 서로 따로 섞지 않고, 검증 데이터를 학습 데이터로 가져오지도 않는다. 기본 `drop_last=False`에서는 마지막에 남은 표본 수가 배치 크기보다 작아도 버리지 않는다.

`list(zip(x,y))`도 이런 데이터에서는 데이터셋처럼 사용할 수 있다. `TensorDataset`만이 유일한 방법은 아니다. 이미 텐서를 준비한 상황에서 인덱스로 쌍을 꺼내기 편한 도구라고 이해하면 된다.

### 장치를 선택하는 것과 데이터를 옮기는 것은 별개다

GPU를 쓰려면 모델과 계산할 데이터를 같은 장치로 옮겨야 한다. NVIDIA GPU에서는 CUDA, Apple Silicon Mac에서는 MPS를 사용할 수 있다. 다음 코드는 사용할 수 있는 장치를 선택하고, 둘 다 없으면 CPU를 사용한다.[^mps]

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = CustomModel(channels=16).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

장치 선택만으로 `train_x` 전체가 GPU로 옮겨지지는 않는다. 전체 데이터는 CPU에 두고 사용할 배치만 옮겨도 된다. 텐서의 `to`는 반환값을 사용해야 한다.

```python
x_batch = x_batch.to(device)
y_batch = y_batch.to(device)

optimizer.zero_grad()         # 이전 배치의 기울기를 비운다.
pred = model(x_batch)         # 현재 파라미터로 예측한다.
assert pred.shape == y_batch.shape
loss = loss_fn(pred, y_batch) # 깨끗한 정답과 비교한다.
loss.backward()              # 파라미터별 기울기를 계산한다.
optimizer.step()             # 파라미터를 갱신한다.
```

`zero_grad`는 파라미터를 초기화하는 명령이 아니다. `backward`도 계산한 기울기로 파라미터를 바꾸는 단계까지 수행하지 않는다. 실제 갱신은 `step`에서 일어난다.[^training]

새 Fold에서는 모델과 옵티마이저를 함께 새로 만든다. 새 모델을 만들고 이전 옵티마이저를 재사용하면 옵티마이저가 이전 모델의 파라미터를 참조할 수 있다. 반대로 배치마다 모델을 새로 만들면 이전 배치에서 배운 내용을 이어 갈 수 없다.

| 코드 위치 | 해야 하는 일 |
| --- | --- |
| Fold 반복 안, 에포크 반복 밖 | 새 모델·새 옵티마이저 준비 |
| 에포크 반복 안 | 같은 모델로 같은 학습 구성 반복 |
| 배치 반복 안 | 입력 이동 → 예측 → 손실 → 역전파 → 갱신 |

`model.train()`은 모드에 따라 달라지는 층을 학습 모드로 놓는다. 그 자체로 학습을 실행하지 않는다. `model.eval()`도 자동 미분을 끄는 명령은 아니다. 검증에서는 평가 모드와 미분 기록 중단을 함께 사용한다.[^module][^nograd]

```python
model.eval()
with torch.no_grad():
    pred = model(x_batch)
```

현재 모델에는 Dropout이나 BatchNorm이 없지만, 나중에 그런 층을 추가하더라도 학습·검증 구분을 유지할 수 있도록 같은 패턴을 사용한다.

## 12. MSE는 어느 범위의 평균인가

<strong>평균제곱오차(Mean Squared Error, MSE)</strong>는 같은 위치의 예측과 정답을 빼고, 그 차이를 제곱한 뒤 정해진 범위에서 평균한 값이다.

차이를 그대로 더하면 밝게 틀린 오차와 어둡게 틀린 오차가 서로 취소될 수 있다. 제곱하면 두 방향의 오차가 모두 양수로 반영된다. 큰 오차가 작은 오차보다 상대적으로 더 강하게 반영된다는 특징도 있다. 밝기 오차가 1보다 작으면 제곱한 숫자 자체는 더 작아질 수 있으므로 “제곱하면 무조건 값이 커진다”는 설명은 정확하지 않다.

먼저 차이를 제곱하기만 하면 모양은 유지된다.

```python
error = (pred - y_batch) ** 2
print(tuple(error.shape))  # (B,1,420,540)
```

`nn.MSELoss`의 `reduction`은 결과를 어떻게 모을지 정한다. 객체를 만들 때 지정하며, 기본값은 문자열 `"mean"`이다.[^mse]

| 설정 | 결과 | 모양 |
| --- | --- | --- |
| `reduction="none"` | 각 위치의 제곱오차 | `(B,C,H,W)` |
| `reduction="mean"` | 모든 원소의 제곱오차 평균 | `()` |
| `reduction="sum"` | 모든 원소의 제곱오차 합 | `()` |

`()`는 축이 없는 스칼라 텐서다. 크기 1인 축을 가진 `(1,)`과 구분한다.

```python
loss_fn_each = nn.MSELoss(reduction="none")
error = loss_fn_each(pred, y_batch)
mse_each = error.mean(dim=(1, 2, 3))

print(tuple(mse_each.shape)) # (B,)
```

`mean(dim=(1,2,3))`은 채널·높이·너비를 평균하고 이미지 축 0을 남긴다. 이미지 한 장 안의 오차를 점수 하나로 모으되, 서로 다른 이미지의 점수는 아직 합치지 않는 것이다. `B`장이 들어왔으면 `B`개의 점수가 나온다.

`error.flatten(1).mean(dim=1)`도 같은 범위를 평균한다. `flatten`은 이미지별 픽셀들을 한 축으로 펼치고, 그 뒤의 `mean`이 평균을 계산한다.

`nn.MSELoss()(pred - y_batch, reduction="none")`처럼 호출하면 안 된다. 비교할 예측과 정답을 따로 전달하고, `reduction`은 손실 함수 객체를 생성할 때 지정한다.

학습에서 기본 `MSELoss`를 사용하면 배치 전체의 손실이 숫자 하나가 된다. 자동 미분은 그 숫자가 어떤 연산을 거쳐 만들어졌는지 추적한다. 손실이 하나라는 이유로 파라미터별 기울기를 계산하지 못하는 것은 아니다.

## 13. RMSE와 평균의 순서를 바꾸면 왜 값이 달라지는가

<strong>평균제곱근오차(Root Mean Squared Error, RMSE)</strong>는 MSE에 제곱근을 적용한 값이다. 낮을수록 예측과 정답의 차이가 작다.

제곱오차를 더하고 평균해도 그 값은 여전히 제곱된 밝기 오차의 척도다. 마지막에 제곱근을 취하면 원래 밝기 오차와 같은 척도로 읽을 수 있다. 다만 이미 여러 오차를 하나로 모았으므로 각 픽셀의 원래 차이를 복원하는 것은 아니다. **척도를 되돌리는 것과 개별 값을 되찾는 것은 다르다.**

```python
rmse_each = mse_each.sqrt()  # 이미지마다 RMSE 하나
mean_image_rmse = rmse_each.mean().item()
```

여기서 중요한 것은 제곱근을 언제 적용했는지다.

### 같은 크기 이미지의 MSE를 다시 평균하는 경우

각 이미지의 픽셀 수가 같다고 하자. 한 이미지 안의 픽셀들은 그 이미지 평균에 같은 비중으로 반영된다. 이미지 평균들도 최종 평균에 같은 비중으로 들어간다. 이미지마다 픽셀 수가 같으므로 결국 모든 픽셀이 같은 비중을 갖는다.

이를 보존되는 값으로도 확인할 수 있다. 한 이미지의 모든 제곱오차를 그 이미지의 평균값으로 바꾸어 놓아도, 그 이미지의 합계와 원소 수는 달라지지 않는다. 모든 이미지에 이 처리를 해도 전체 합계와 전체 원소 수는 유지된다. 이미지 크기까지 같다면 각 이미지 평균이 같은 횟수만큼 반복되므로, 이미지 평균들만 동등하게 평균해도 같은 값이 나온다.

이 조건에서는 <strong>전체 픽셀 MSE = 이미지별 MSE의 평균</strong>이다. 이미지 크기가 다르면 큰 이미지가 가진 픽셀 수를 고려해야 하므로 단순 평균으로 바꿀 수 없다.

### 제곱근은 평균과 자유롭게 순서를 바꿀 수 없다

이미지마다 MSE에 제곱근을 적용하면, 오차가 큰 이미지와 작은 이미지가 각각 변환된 점수로 합쳐진다. 전체 MSE를 먼저 구하면 큰 제곱오차가 합산에 먼저 반영되고, 마지막에 한 번만 제곱근을 적용한다.

제곱근은 모든 입력을 똑같은 비율로 줄이는 연산이 아니다. 변환 전 큰 값과 작은 값 사이의 관계가 변환 후에도 같은 비율로 유지되지 않는다. 이런 변환을 이미지별로 먼저 적용하는 것과, 값을 합친 뒤 적용하는 것은 다른 계산이다.

<figure class="document-figure">
<img src="/document-denoising-kfold-kaggle/metrics-ko.svg" alt="원본 영역의 제곱오차에서 이미지별 평균 후 각 이미지의 제곱근을 평균하는 경로와, 모든 제곱오차 합을 전체 픽셀 수로 나눈 뒤 제곱근을 한 번 적용하는 경로를 비교한다." loading="lazy">
<figcaption>두 점수는 모두 정의할 수 있다. 평가 대상과 계산 순서를 이름에 드러내고 같은 방식끼리 비교한다.</figcaption>
</figure>

같은 크기 이미지라면 코드의 차이는 다음과 같다.

```python
# 이미지별 점수를 먼저 완성한 뒤 이미지마다 같은 비중으로 평균한다.
score_a = ((pred - y_batch) ** 2).mean(dim=(1, 2, 3)).sqrt().mean()

# 모든 픽셀의 제곱오차를 합친 뒤 제곱근을 한 번 적용한다.
score_b = nn.MSELoss()(pred, y_batch).sqrt()
```

이 조건에서는 `score_b`가 `score_a`보다 크거나 같고, 이미지별 MSE가 모두 같을 때 두 값이 같아진다. 원본 크기가 서로 다르면 전체 픽셀 점수에는 큰 이미지가 더 큰 비중으로 들어가는 차이도 추가된다.

계산 차이를 확인할 작은 예를 하나 보자. 같은 크기인 두 이미지의 MSE가 각각 0.01과 0.09라고 가정한다. 실습에서 측정한 값이 아니라 집계 순서를 확인하기 위해 정한 값이다. 각 이미지의 RMSE는 0.1과 0.3이므로 그 평균은 0.2다. 반면 두 MSE의 평균 0.05에 제곱근을 적용하면 약 0.223607이다.

```python
example_mse = torch.tensor([0.01, 0.09])
print(example_mse.sqrt().mean().item()) # 약 0.200000
print(example_mse.mean().sqrt().item()) # 약 0.223607
```

### Fold 평균도 무엇을 평균했는지 확인한다

검증 이미지가 매번 24장이고 각 이미지가 검증에 한 번씩 등장하면, 여섯 Fold의 <strong>이미지별 RMSE 평균</strong>을 다시 평균한 값은 전체 144개의 이미지별 RMSE를 모아 평균한 값과 같다. 모든 그룹의 크기가 같기 때문이다.

하지만 이것은 하나의 모델이 144장을 평가한 점수가 아니다. 서로 다른 여섯 모델이 각자의 검증 이미지에 만든 예측을 모은 점수다. 검증 이미지 수가 다른 분할까지 다루려면 이미지별 결과를 모두 모아 평균하는 편이 안전하다. 이 글의 전체 실행 예제도 이 방법을 사용한다.

## 14. PSNR은 MSE를 밝기 범위와 비교한 점수다

<strong>최대 신호 대 잡음비(Peak Signal-to-Noise Ratio, PSNR)</strong>는 정해진 밝기 범위와 복원 오차의 비율을 로그로 표현한다. 같은 밝기 범위에서는 MSE가 줄어들수록 PSNR이 커진다. 글자 획이나 모양에 대한 별도의 오류를 찾아내는 지표는 아니다.[^psnr]

<strong>데시벨(Decibel, dB)</strong>은 비율을 로그로 나타내는 척도다. 정확도 백분율과는 다르다. 로그는 어떤 밑을 몇 번 거듭제곱해야 주어진 값이 되는지를 나타내는 연산이다. PSNR에서는 밑이 10인 로그를 사용한다.

공식에서 `R`은 평가에 사용하는 밝기 범위의 폭이다. 현재 0~1 데이터에서는 1, 0~255 데이터에서는 255다. 각 이미지에서 실제 관찰된 최댓값으로 매번 바꾸는 값이 아니다.

<div class="document-math">
$$
\mathrm{PSNR}=10\log_{10}\left(\frac{R^2}{\mathrm{MSE}}\right)
$$
</div>

MSE가 제곱된 오차이므로 비교하는 밝기 범위도 제곱한다. 현재 `R=1`을 넣으면 `10 log10(1/MSE)`다. 로그에서 나눗셈은 두 로그의 차이이고, `log10(1)=0`이므로 다음 계산을 사용한다.

<div class="document-math">
$$
10\log_{10}(1/\mathrm{MSE})
=10\{\log_{10}(1)-\log_{10}(\mathrm{MSE})\}
=-10\log_{10}(\mathrm{MSE})
$$
</div>

```python
psnr_each = -10 * torch.log10(mse_each)
mean_image_psnr = psnr_each.mean().item()
```

`torch.log`는 자연로그다. PSNR의 밑 10 로그는 `torch.log10`이다. `mse_each`가 `(B,)`이면 `psnr_each`도 `(B,)`이고, 평균한 뒤에야 원소 하나를 `item()`으로 꺼낼 수 있다.

MSE가 정확히 0이면 PSNR은 양의 무한대 `inf`다. 이미지 중 하나가 완전히 일치하면 단순 평균 PSNR도 무한대가 될 수 있다. `clamp_min(1e-12)`로 이를 막으면 현재 범위에서 120dB의 상한을 둔 점수가 된다. 이는 수치 처리 규칙을 추가한 것이므로 명시해야 한다. 실행 예제는 무한대를 임의로 지우지 않는다.

한 이미지에서는 RMSE가 낮을수록 PSNR이 높다. 여러 이미지의 <strong>RMSE 평균</strong>과 <strong>PSNR 평균</strong>은 각각 다른 변환을 거친 점수의 평균이므로 모델 순위가 언제나 같은 방향으로 대응한다고 보장할 수 없다. 평균 RMSE를 PSNR로 바꾼 값을 이미지별 PSNR의 평균이라고 부르면 안 된다.

## 15. 평가에서는 원본에 없던 흰 여백을 제거한다

실제 문서의 흰 배경은 복원할 대상이다. 크기를 맞추려고 추가한 흰 패딩은 원본 문서에 없던 영역이다. 둘은 구분해야 한다.

손상 입력과 정답에 같은 흰색 패딩을 붙였다면, 무처리 기준선의 패딩 영역 오차는 0이다. 이 0들을 평균에 포함하면 실제 문서 부분의 오차보다 점수가 낮아진다. 모델은 패딩을 완전히 흰색으로 출력하지 않을 수도 있으므로, 모델 점수가 항상 같은 방향으로 변한다고 단정할 수는 없다. 원본에 없던 영역이 점수에 영향을 준다는 것이 문제다.

높이가 258이던 이미지의 원본 영역은 `81:339`다. Python 슬라이스는 시작 위치를 포함하고 끝 위치는 제외한다. 위쪽 81행을 건너뛰면 원본의 첫 행은 인덱스 81이고, 원본 258행이 끝나는 다음 위치가 339다. 마지막 포함 인덱스는 338이다.

다음 함수는 예측·정답·원본 높이를 받고 <strong>이미지별 MSE가 담긴 `(B,)` 텐서</strong>를 반환한다. 여기서 사용하는 전처리처럼 가로 패딩이 없고, 높이 방향으로 중앙 정렬했을 때의 함수다.

```python
def per_image_mse(pred, target, original_heights):
    assert pred.shape == target.shape
    assert len(pred) == len(original_heights)
    values = []

    for i, height in enumerate(original_heights):
        height = int(height)
        assert height in (258, 420)
        top = (pred.shape[-2] - height) // 2
        stop = top + height

        restored = pred[i, :, top:stop, :]
        clean = target[i, :, top:stop, :]
        values.append(((restored - clean) ** 2).mean())

    return torch.stack(values)
```

`values`에는 이미지마다 스칼라 텐서 하나가 들어간다. `stack`은 여기에 이미지 축을 새로 만들어 `(B,)`로 묶는다. 무처리 기준선도 `per_image_mse(x_val,y_val,val_heights)`로 같은 영역을 평가한다. 전체 데이터의 높이 목록이 아니라 현재 검증 이미지와 같은 번호로 선택한 목록을 전달한다.

원본 크기가 다르면 전체 픽셀 RMSE는 제곱오차의 <strong>합계와 실제 원소 수</strong>를 모아 계산해야 한다.

```python
def pooled_pixel_rmse(pred, target, original_heights):
    assert pred.shape == target.shape
    assert len(pred) == len(original_heights)
    squared_error_sum = 0.0
    pixel_count = 0

    for i, height in enumerate(original_heights):
        height = int(height)
        top = (pred.shape[-2] - height) // 2
        error = (
            pred[i, :, top:top + height, :]
            - target[i, :, top:top + height, :]
        ) ** 2

        squared_error_sum += error.sum().item()
        pixel_count += error.numel()

    return (squared_error_sum / pixel_count) ** 0.5
```

`numel()`은 텐서의 전체 원소 수다. 크기가 다른 이미지의 MSE를 동등하게 평균하면 이미지마다 같은 비중을 준다. 위 함수는 픽셀마다 같은 비중을 주므로 두 계산을 구분한다.

이번 학습 예제의 손실은 단순한 `MSELoss`라서 패딩을 포함한다. 평가는 원본 영역만 사용한다. 학습에서도 패딩을 빼려면 원본 위치만 포함하는 <strong>마스크(Mask)</strong>를 사용하거나 같은 크롭 범위에서 손실을 계산할 수 있다. 이는 실제 문서의 흰 배경을 모두 무시하는 것과 다르다.

**패딩 포함 학습 MSE와 패딩 제외 검증 RMSE를 직접 비교해 과적합이라고 판단하면 안 된다.** 과적합은 학습 데이터에 비해 새 데이터에서 성능이 떨어지는 현상을 확인해야 한다. 이를 비교하려면 학습 데이터도 평가 모드에서 다시 예측하고, 검증 데이터와 같은 영역·같은 지표로 측정해야 한다.


## 16. 학습과 교차 검증을 실제 코드로 연결한다

이제 앞에서 만든 `train_x`, `train_y`, `heights`, `CustomModel`, `per_image_mse`를 연결한다. 먼저 한 모델의 학습을 함수로 묶는다.

`train_model`은 학습용 입력·정답과 사용할 장치를 받는다. 함수 안에서 배치를 만들고, **새 모델과 새 옵티마이저를 생성한 뒤**, 정한 에포크만큼 학습한다. 반환값은 학습된 모델과 마지막 에포크의 평균 학습 손실이다. 이렇게 묶으면 Fold가 바뀔 때마다 같은 절차를 새로 시작하기 쉽다.

```python
def train_model(x, y, device, *, channels=16, epochs=30,
                batch_size=24, lr=0.001, seed=42):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=0,
    )

    model = CustomModel(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # 학습 손실에는 패딩이 포함된다.

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            assert pred.shape == y_batch.shape
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            # 마지막 배치가 작아도 이미지 수를 반영한 평균을 만든다.
            loss_sum += loss.item() * len(x_batch)

        train_mse = loss_sum / len(dataset)

    return model, train_mse
```

함수 정의 자체는 학습을 수행하지 않는다. 아래 반복문에서 `train_model(...)`을 호출할 때 해당 Fold의 학습이 실행된다. `epochs`에는 1 이상의 정수를 전달한다.

분할 시드는 어떤 이미지를 어느 Fold에 넣을지 정한다. 모델 초기화와 DataLoader의 시드는 초기 파라미터와 배치 순서를 다룬다. 시드를 기록하면 비교가 쉬워지지만, 같은 시드만으로 장치나 라이브러리 버전이 달라도 완전히 같은 결과가 나온다고 보장할 수는 없다.[^randomness]

검증 결과는 원래 이미지 번호 위치에 저장한다. 이 방식이면 Fold마다 검증 이미지 수가 달라도 마지막에 각 이미지를 한 번씩 모아 평균할 수 있다.

```python
kf = KFold(n_splits=6, shuffle=True, random_state=42)
batch_size = 24
epochs = 30

# CPU에 이미지별 검증 결과를 보관한다.
oof_model_mse = torch.empty(len(train_x), dtype=torch.float64)
oof_base_mse = torch.empty(len(train_x), dtype=torch.float64)
seen = torch.zeros(len(train_x), dtype=torch.int64)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_x), start=1):
    model, train_mse = train_model(
        train_x[train_idx], train_y[train_idx], device,
        channels=16, epochs=epochs, batch_size=batch_size, seed=42 + fold,
    )

    model.eval()
    with torch.no_grad():
        for start in range(0, len(val_idx), batch_size):
            idx = val_idx[start:start + batch_size]
            x_val = train_x[idx].to(device)
            y_val = train_y[idx].to(device)
            val_heights = [int(heights[int(i)]) for i in idx]

            pred = model(x_val)
            model_mse = per_image_mse(pred, y_val, val_heights)
            base_mse = per_image_mse(x_val, y_val, val_heights)

            oof_model_mse[idx] = model_mse.cpu().double()
            oof_base_mse[idx] = base_mse.cpu().double()
            seen[idx] += 1

    fold_score = oof_model_mse[val_idx].sqrt().mean().item()
    print(f"Fold {fold} | 학습 MSE {train_mse:.6f} | 검증 RMSE {fold_score:.6f}")

assert bool((seen == 1).all())  # 모든 이미지가 정확히 한 번 검증됐는지 확인한다.

print("이미지별 모델 RMSE 평균:", oof_model_mse.sqrt().mean().item())
print("이미지별 무처리 RMSE 평균:", oof_base_mse.sqrt().mean().item())
print("이미지별 모델 PSNR 평균:", (-10 * torch.log10(oof_model_mse)).mean().item())

# 원본 문서의 픽셀 수로 가중해야 전체 원본 픽셀 RMSE가 된다.
pixel_counts = torch.as_tensor(heights, dtype=torch.float64) * train_x.shape[-1]
pooled_rmse = (
    (oof_model_mse * pixel_counts).sum() / pixel_counts.sum()
).sqrt().item()
print("전체 원본 픽셀 모델 RMSE:", pooled_rmse)
```

`OOF(Out-of-Fold)`는 각 표본을 학습에 사용하지 않은 Fold 모델로 얻은 검증 결과를 뜻한다. 코드의 `oof_model_mse`는 그런 이미지별 MSE를 원래 순서에 맞춰 보관한다. 다운로드 파일은 이미지별 제곱오차 합과 픽셀 수를 직접 저장하고, 모델과 무처리의 세 지표를 함께 계산한다.

위 학습 로그의 `train_mse`는 에포크 동안 배치별로 기록한 손실의 평균이다. 배치마다 파라미터가 갱신되므로 모든 배치를 완전히 동일한 최종 파라미터로 평가한 값도 아니다. 모델 간 성능 비교에는 별도로 계산한 검증 지표를 사용한다.

반복문이 끝나면 `model`에는 <strong>마지막 Fold의 모델</strong>이 남는다. 앞선 모델들이 점수 배열에 저장된 것은 아니다. 학습된 모델을 보관하려면 가중치와 그 모델의 분할 정보를 함께 저장해야 한다. 다운로드 파일은 각 Fold의 체크포인트와 파일명 목록을 저장한다.

## 17. 이미지를 그릴 때 장치와 축을 확인한다

점수가 좋아졌더라도 글자 획이 흐려지거나 작은 글자가 사라질 수 있다. 입력·복원·정답을 같은 밝기 범위로 나란히 보면 이런 실패를 확인하기 쉽다.

표시 전에 자주 쓰는 세 메서드는 서로 다른 일을 한다.

| 메서드 | 하는 일 | 하지 않는 일 |
| --- | --- | --- |
| `detach()` | 자동 미분 연결에서 분리한 텐서 반환 | CPU로 옮기지 않음 |
| `cpu()` | CPU 텐서 반환 | Python 숫자로 바꾸지 않음 |
| `item()` | 원소가 하나인 텐서에서 Python 숫자 꺼내기 | 여러 점수를 자동으로 평균하지 않음 |

일반적인 NumPy 변환에는 CPU에 있는, 기울기 추적이 필요 없는 텐서를 사용한다. 표시할 때는 `tensor.detach().cpu().numpy()` 패턴이 편하다. `detach`는 원본과 저장 공간을 공유할 수 있으므로 독립 복사와 같지는 않다.[^numpy]

다음 예는 <strong>직전 코드의 마지막 Fold 모델과 그 Fold의 검증 번호</strong>를 사용한다. 전체 6-Fold 결과를 한 모델로 다시 평가하는 코드가 아니다.

<strong>Figure</strong>는 전체 그림이고, <strong>Axes</strong>는 이미지를 그릴 개별 공간이다. `plt.subplots`는 이 둘을 함께 만든다. `axes[row,col]`의 행·열은 그림 배치 위치이며, 이미지 안의 픽셀 좌표와는 다르다.[^subplots]

```python
import random
import matplotlib.pyplot as plt

n_show = min(5, len(val_idx))
selected = random.Random(42).sample(list(map(int, val_idx)), n_show)

x_show = train_x[selected].to(device)
y_show = train_y[selected].to(device)
show_heights = [int(heights[i]) for i in selected]

model.eval()
with torch.no_grad():
    pred_show = model(x_show)

fig, axes = plt.subplots(
    n_show, 3, figsize=(15, 3 * n_show), squeeze=False,
)

for row, image_id in enumerate(selected):
    height = show_heights[row]
    top = (x_show.shape[-2] - height) // 2
    stop = top + height

    images = [
        x_show[row, 0, top:stop, :],
        pred_show[row, 0, top:stop, :],
        y_show[row, 0, top:stop, :],
    ]

    for col, (image, title) in enumerate(
        zip(images, ["Noisy", "Restored", "Clean"])
    ):
        axes[row, col].imshow(
            image.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1,
        )
        axes[row, col].set_title(f"{title} — {names[image_id]}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
```

`row`는 표시할 그림의 행 번호다. `image_id`는 전체 데이터에서의 이미지 번호다. 둘을 구분해서, 선택한 작은 배치에서는 `row`로 접근하고 파일명에서는 원래 번호로 접근한다.

`x_show[row,0]`으로 흑백 채널을 선택해야 `imshow`에 넣을 `(H,W)` 배열이 된다. `vmin=0,vmax=1`은 세 이미지의 밝기 표시 기준을 고정한다. 픽셀값 자체를 다시 정규화하는 옵션은 아니다. 기준을 생략하면 이미지마다 표시 범위가 달라져 흐릿한 결과가 더 선명해 보일 수 있다.[^imshow]

`squeeze=False`를 사용하면 한 장만 표시하더라도 `axes[행,열]` 형태를 유지한다. 표시할 이미지를 늘릴 때는 `n_show`와 그림의 행 수를 함께 바꿔야 한다.

이미지별 점수의 변화를 보고 싶다면 전체 OOF 결과를 그릴 수도 있다.

```python
model_psnr = -10 * torch.log10(oof_model_mse)
base_psnr = -10 * torch.log10(oof_base_mse)

plt.figure(figsize=(12, 4))
plt.plot(model_psnr.cpu().numpy(), label="Model")
plt.plot(base_psnr.cpu().numpy(), label="No-op")
plt.xlabel("Image index in the stored file order")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.tight_layout()
plt.show()
```

가로축은 보관한 파일 순서의 인덱스다. 시간 흐름이나 파일명의 숫자 크기를 의미하지 않는다. `inf`가 있는 경우 그래프가 해당 점수를 유한한 위치에 표시하지 못할 수 있으므로, 그래프만 보지 말고 그 이미지의 MSE가 0인지 따로 확인한다.

## 18. 개선할 때는 한 조건씩 바꾼다

문서가 흐리게 복원됐다고 곧바로 채널·층·손실·학습률을 모두 바꾸면 어떤 변경이 효과가 있었는지 알기 어렵다. 먼저 입력과 정답의 짝, 출력 모양, 학습 손실의 감소, 무처리 대비 점수, 실제 글자 보존 상태를 확인한다. 한 배치의 손실도 충분히 낮출 수 없다면 데이터나 학습 연결 문제를 먼저 살펴볼 수 있다. 그 검사 하나만으로 원인을 단정하지는 않는다.

변경할 수 있는 대상도 구분한다.

| 변경 | 바꾸는 것 | 확인할 점 |
| --- | --- | --- |
| 채널 증가 | 같은 단계에서 표현할 특징의 용량 | 연산량·메모리 증가와 검증 개선 여부 |
| 층 추가 | 특징을 여러 단계로 가공하는 구조 | 출력 크기, 학습 안정성, 주변 영역의 활용 |
| 스킵 연결 | 앞쪽 특징을 뒤쪽 계산으로 전달하는 경로 | 글자 위치 정보와 잡음이 함께 전달될 가능성 |
| 손실 변경 | 어떤 오류를 얼마나 크게 줄일지 정하는 기준 | 대회 지표 및 글자 보존 목표와의 관계 |
| 패치 학습 | 한 번에 볼 문서 영역 | 학습·정답 좌표 일치와 원본 문서 단위 분할 |

<strong>스킵 연결(Skip Connection)</strong>은 앞쪽 특징을 뒤쪽 층으로 직접 전달하는 연결이다. 더하기로 결합하면 두 텐서의 모양을 맞춰야 한다. 채널 방향으로 이어 붙이면 배치·높이·너비가 같아야 하며, 다음 층의 입력 채널 수가 달라진다. 연결을 추가했다는 사실만으로 복원 성능이 좋아지지는 않는다.

층을 더 쌓는다고 항상 공간 크기를 줄여야 하는 것도 아니다. 이동 간격과 패딩을 어떻게 정하는지에 따라 공간 크기를 유지하면서 특징을 더 가공할 수 있다.

흐릿한 회색 출력은 여러 원인으로 생길 수 있다. 모델이 글자와 배경을 충분히 구분하지 못하는 상황에서는 중간 밝기를 내는 것이 제곱오차를 줄이는 선택이 될 수 있다. 하지만 실제 원인이 MSE인지, 학습 부족인지, 모델 용량인지, 정보 손실인지, 잘못 연결된 정답인지는 실험으로 확인해야 한다.

<strong>절대오차 손실(L1 Loss)</strong>은 차이의 절댓값을 사용하고, 가중 손실은 특정 영역의 오차에 더 큰 비중을 준다. <strong>구조적 유사도(Structural Similarity, SSIM)</strong>는 국소적인 밝기·대비·구조를 비교하는 지표다. 이런 기준을 손실에 결합하는 실험도 가능하지만, 현재 코드가 올바르게 학습하고 평가하는지 확인한 뒤 진행하는 편이 효율적이다.[^l1][^psnr]

깨끗한 문서의 배경은 실제로 흰색이다. 배경을 흰색으로 복원했다는 이유로 벌점을 주면 복원 목표와 어긋난다. 글자가 지워지는 문제를 다루려면 글자 영역의 보존과 배경 잡음 제거를 함께 평가해야 한다.

`128×128` 패치를 사용한다면 잘라낼 좌표를 한 번 정해서 손상 이미지와 정답에 똑같이 적용한다. 독립적으로 무작위 크롭을 하면 서로 다른 위치를 입력과 정답으로 학습하게 된다. 패치는 원본 픽셀 크기를 유지하면서 배치 크기를 맞출 수 있지만, 모든 글자나 전체 문맥을 한 번에 보게 해 주지는 않는다.

실험에서는 분할과 평가 방식을 고정하고 한 조건씩 바꾼다. 모델을 바꿨다면 <strong>Fold 안에서 실제로 생성하는 클래스</strong>도 바뀌었는지 확인한다. 바깥에서 새 모델을 만들었어도 Fold 안에서 이전 클래스를 다시 생성하면 이전 모델을 학습하게 된다.

## 19. 교차 검증이 끝나면 제출용 모델을 따로 학습한다

교차 검증은 설정을 비교하는 과정이다. 마지막 Fold의 모델은 120쌍으로 학습했으며, 여섯 Fold를 모두 돌렸다고 그 모델이 144쌍 전체를 학습한 것은 아니다.

모델 구조·에포크·학습률 등을 선택했다면 새 모델로 전체 학습 데이터를 사용해 최종 학습할 수 있다. 앞에서 정의한 `train_model`은 호출마다 모델을 새로 생성하므로 다음과 같이 연결한다.

```python
final_model, final_train_mse = train_model(
    train_x, train_y, device,
    channels=16, epochs=30, batch_size=24, lr=0.001, seed=42,
)

torch.save(
    {
        "state_dict": final_model.state_dict(),
        "channels": 16,
        "epochs": 30,
        "batch_size": 24,
        "learning_rate": 0.001,
        "seed": 42,
        "preprocessing": "L / 255; white vertical padding to (420,540)",
        "train_names": names,
    },
    "final_model.pt",
)
```

`state_dict`는 모델의 파라미터와 등록된 버퍼 등을 담는다. 구조·전처리·학습 조건도 함께 기록해야 같은 계산을 다시 구성할 수 있다. 위 파일은 학습된 모델을 추론에 다시 쓰기 위한 저장이다. 중단된 학습을 같은 옵티마이저 상태로 이어 가려면 옵티마이저 상태 등도 추가로 저장해야 한다.[^saving]

`test`에는 깨끗한 정답이 없다. 복원 출력과 손상 입력의 차이는 “입력을 얼마나 바꿨는지”를 보여 줄 뿐, 깨끗한 정답에 얼마나 가까워졌는지를 알려 주지는 않는다. 로컬에서 복원 품질을 수치로 비교하는 대상은 정답이 있는 검증 데이터다.

최종 모델을 전체 데이터로 학습한 뒤에는 그 데이터에 대한 점수를 독립적인 검증 점수로 보고하지 않는다. 이미 모델이 그 이미지를 학습했기 때문이다.


## 20. 테스트 추론에서도 파일 순서와 원본 크기를 보관한다

테스트 이미지는 학습 때와 같은 흑백 변환·정규화·패딩을 적용한다. 정답이 없다는 점만 다르다. 원본 크기와 파일명은 제출할 때 필요하므로 예측과 같은 순서로 보관한다.

```python
test_paths = sorted((data_dir / "test").glob("*.png"))
assert test_paths, "test 폴더에서 PNG 파일을 찾지 못했습니다."
test_tensors, test_sizes = [], []

for path in test_paths:
    with Image.open(path) as opened:
        image = opened.convert("L")

    width, height = image.size
    if (width, height) not in {(540, 420), (540, 258)}:
        raise ValueError(f"예상하지 못한 크기: {image.size}")

    x = to_tensor(image)
    top = (420 - height) // 2
    x = F.pad(x, (0, 0, top, top), value=1.0)

    test_tensors.append(x)
    test_sizes.append((width, height))

test_x = torch.stack(test_tensors)
print(tuple(test_x.shape))  # 실습 데이터 전체라면 (72,1,420,540)
```

추론도 배치 단위로 실행하면 GPU에 한 번에 올리는 데이터를 줄일 수 있다.

```python
final_model.eval()
predictions = []

with torch.no_grad():
    for start in range(0, len(test_x), batch_size):
        batch = test_x[start:start + batch_size].to(device)
        pred_batch = final_model(batch)
        predictions.append(pred_batch.cpu())

test_pred = torch.cat(predictions, dim=0)
assert test_pred.shape == test_x.shape
```

각 `pred_batch`에 이미 배치 축이 있으므로 `cat`을 사용한다. 결과를 CPU에 모았기 때문에 이후 제출 파일을 만들 때 GPU의 모든 예측을 계속 붙잡아 둘 필요는 없다. 다운로드 실행 예제는 더 나아가 CPU로 옮긴 배치부터 압축 파일에 기록한다.

## 21. Kaggle에는 점수가 아니라 픽셀 예측값을 보낸다

Kaggle은 제출한 픽셀 밝기를 서버가 가진 정답과 비교해 RMSE를 계산한다. 로컬에서 구한 RMSE나 PSNR 숫자를 제출하는 것이 아니다. 대회의 제출 형식은 `id,value`이고, `id`는 <strong>이미지 번호_행 번호_열 번호</strong>다. 밝기는 검정 0부터 흰색 1까지다.[^kaggle]

```csv
id,value
1_1_1,1.0
1_1_2,0.98
```

위 숫자는 형식만 보여 주는 예다. 제출 이미지 번호는 배열의 순번이 아니라 원본 파일명에서 가져온다. `test/12.png`이면 이미지 번호는 12다. Python 텐서 인덱스는 0부터 시작하지만 제출 행·열 번호는 1부터 시작한다.

`test_pred`는 높이 420으로 맞춘 예측이므로 원본 높이가 258인 이미지에서는 위아래 패딩을 먼저 제거한다. 아래 코드는 앞에서 만든 `test_paths`, `test_sizes`, `test_pred`를 사용한다.

```python
import csv
import io
from zipfile import ZipFile, ZIP_DEFLATED

assert len(test_paths) == len(test_sizes) == len(test_pred)
row_count = 0

with ZipFile("submission.csv.zip", "w", compression=ZIP_DEFLATED) as archive:
    with archive.open("submission.csv", "w") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "value"])

            for i, path in enumerate(test_paths):
                width, height = test_sizes[i]
                image = test_pred[i, 0].detach().cpu().numpy()

                top = (image.shape[0] - height) // 2
                image = image[top:top + height, :]

                assert image.shape == (height, width)
                assert np.isfinite(image).all()
                assert ((image >= 0) & (image <= 1)).all()

                for row, pixels in enumerate(image, start=1):
                    for col, value in enumerate(pixels, start=1):
                        writer.writerow([
                            f"{path.stem}_{row}_{col}", float(value)
                        ])
                        row_count += 1

expected_rows = sum(width * height for width, height in test_sizes)
assert row_count == expected_rows
print("기록한 픽셀 수:", row_count)
```

`row_count`는 헤더를 제외한 예측 행 수다. 전체 원본 이미지의 픽셀 수와 같아야 한다. 같은 수의 행을 썼다는 것만으로 좌표 대응이 모두 옳다고 보장되지는 않으므로 파일명·행·열의 첫 값과 마지막 값도 확인한다.

`NaN`이나 범위 밖 값을 무조건 잘라서 감추지 않는다. 현재 Sigmoid 모델에서 정상적인 유한 출력은 0~1 범위다. 검사를 통과하지 못했다면 모델 출력이나 전처리부터 확인한다.

### 터미널과 Jupyter 명령은 앞부분이 다르다

<strong>명령줄 도구(Command-Line Interface, CLI)</strong>는 터미널에서 명령으로 프로그램을 사용하는 방식이다. 2026년 11월 7일 확인한 Kaggle 공식 문서에서는 다음 제출·조회 명령을 제공한다.[^kaggle-cli]

macOS의 zsh나 Linux의 bash·zsh에서는 그대로 실행한다. 파일 경로는 위 코드에서 만든 파일을 기준으로 했다.

```bash
kaggle competitions submit denoising-dirty-documents -f submission.csv.zip -m "Document restoration"
kaggle competitions submissions denoising-dirty-documents
```

특정 제출을 조회하려면 `kaggle competitions submission` 뒤에 새로 발급된 제출 번호를 붙인다. 과거의 제출 번호를 그대로 사용하면 새 실험이 아니라 그 과거 기록을 조회하게 된다.

Jupyter에서 셸 명령을 실행할 때는 맨 앞에 `!`를 붙인다. 현재 커널에 패키지를 설치할 때 사용하는 `%pip`와도 구분한다.

```python
# 다음 두 줄은 Python 스크립트가 아니라 Jupyter 셀용 명령이다.
# %pip install -U kaggle
# !kaggle competitions submissions denoising-dirty-documents
```

인증이 필요하면 터미널에서 `kaggle auth login`을 실행하고, `kaggle quota` 같은 실제 API 요청이 성공하는지도 확인한다. `quota`는 GPU·TPU 사용 할당량 조회 명령이다. 로그인 안내문이 나왔다는 사실과 실제 요청이 인증됐다는 사실은 다르다.[^kaggle-auth]

실습 중에는 기존 `kaggle.json`과 OAuth 인증 정보가 함께 있는 환경에서 “이미 로그인됨” 안내와 인증 오류가 동시에 나타났다. 기존 `kaggle.json`을 다른 이름으로 잠시 보관한 뒤 요청이 성공했다는 기록이 있다. 여기서 확인된 것은 그 파일을 제외했을 때 해당 환경의 요청이 성공했다는 점이다. 모든 인증 오류의 원인이 같다고 일반화하거나 기존 키가 무효였던 이유까지 단정할 수는 없다. 인증 파일의 실제 내용은 노트북 출력이나 공개 저장소에 넣지 않는다.

업로드 성공 메시지는 채점 완료와 다르다. 제출 상태가 `COMPLETE`인지 확인하고, `ERROR`이면 오류 내용을 확인한다. 대회의 참여 조건이나 현재 제출 가능 여부도 제출 시점의 안내를 따른다.

## 22. 실습 점수에서 말할 수 있는 범위를 지킨다

아래 값은 기존 실습 기록이다. 이 글의 예제를 원본 데이터로 다시 학습해 재현한 값은 아니다.

| 기록 | 점수 | 해석할 때 확인할 조건 |
| --- | ---: | --- |
| 초기 모델, 1에포크 | 로컬 평균 RMSE 약 0.45 | 당시 구조·분할·평가 영역 |
| 초기 모델, 30에포크 | 로컬 평균 RMSE 약 0.22694 | 앞 실험과 설정이 같았는지 |
| 무처리 기준선 | 로컬 평균 RMSE 약 0.1437 | 모델과 같은 검증 이미지·집계인지 |
| Kaggle 제출 `58090430` | Public·Private 모두 0.09623, COMPLETE | 제출 모델·테스트 데이터·픽셀 집계 |

로컬에서 **동일한 평가 조건으로** 비교했을 때 모델의 RMSE가 무처리보다 컸다면, 그 실험에서는 모델을 적용한 결과가 더 나빴다고 말할 수 있다.

반면 Kaggle의 0.09623이 로컬 무처리의 0.1437보다 작다는 사실만으로 무처리 대비 개선을 입증할 수는 없다. 데이터가 다르고, 패딩 포함 여부나 이미지별 평균·전체 픽셀 집계가 다를 수 있다. 모델 변경 여부도 확인해야 한다.

RMSE 0.09623을 “정확도 90.377%”로 바꾸는 것도 잘못이다. RMSE는 밝기 오차를 요약한 지표이며 맞힌 비율이 아니다.

실험을 다시 비교하려면 모델 구조, 분할 파일명, 에포크·배치·학습률, 시드, 전처리, 평가 영역, 집계 방식, 모델 파일과 제출 번호를 함께 기록한다. 대회 점수를 반복해서 보며 설정을 바꾸면 그 점수도 모델 선택에 사용하게 된다. 모델 선택은 가능한 한 별도로 정한 로컬 검증을 중심으로 진행한다.[^cross-validation]

## 23. 실행 파일 사용과 자주 헷갈리는 코드

[한국어 Python 실행 파일](/document-denoising-kfold-kaggle/examples-ko.py)을 내려받아 PyTorch가 설치된 가상환경에서 실행한다. 이 파일은 `torch`, `numpy`, `Pillow`, `scikit-learn`을 사용한다. 본문의 시각화 코드를 실행하려면 `matplotlib`도 필요하다. PyTorch 설치 방식은 CPU·CUDA·MPS 환경에 맞춰 선택한다.

데이터 경로가 `./data`라면 그 안에 `train`, `train_cleaned`, `test` 폴더가 있어야 한다. 다음 명령은 macOS와 Linux의 터미널용이며, `python`이 해당 가상환경의 Python을 가리키는 상태를 전제로 한다.

```bash
python examples-ko.py --mode check
python examples-ko.py --mode cv --data-dir ./data --output-dir ./runs
python examples-ko.py --mode final --data-dir ./data --output-dir ./runs
```

| 실행 모드 | 하는 일 | 만들어지는 결과 |
| --- | --- | --- |
| `check` | 합성 데이터로 핵심 연산과 제출 형식 검사 | 검사 통과 여부 |
| `cv` | Fold마다 새 모델을 학습하고 원본 영역으로 검증 | `fold-1.pt` 등, `cv-summary.csv`, `oof.csv` |
| `final` | 전체 학습 데이터로 새 모델을 학습하고 테스트 추론 | `final.pt`, `submission.csv.zip` |

기본 설정은 6-Fold, 30에포크, 배치 24, 채널 16, 학습률 0.001, 시드 42다. `--epochs`, `--batch-size`, `--channels`, `--folds`, `--lr`, `--seed`로 바꿀 수 있다. 장치는 자동 선택하며 `--device cpu`, `--device mps`, `--device cuda`로 지정할 수도 있다. 출력 폴더를 같은 이름으로 재사용하면 같은 이름의 결과 파일을 덮어쓰므로 실험마다 폴더를 나누어 보관한다.

`final`은 제출 파일을 만든다. 실제 Kaggle 업로드는 별도 CLI 명령으로 수행한다. 위 출력 경로를 썼다면 제출 파일은 `./runs/submission.csv.zip`이다.

실습 중 자주 헷갈렸던 부분은 다음처럼 정리할 수 있다.

| 의도 | 확인할 코드 |
| --- | --- |
| NumPy 실수 자료형 | `np.float32` |
| 리스트에 원소 추가 | `.append(value)` |
| 교차 검증 분할 수 | `n_splits` |
| Adam 옵티마이저 | `torch.optim.Adam` |
| 정해진 횟수 반복 | `for epoch in range(epochs)` |
| MPS 사용 가능 여부 | `torch.backends.mps.is_available()` |
| 텐서 모양 확인 | `x.shape` 또는 `x.size()` |
| MSE의 전체 평균 | `nn.MSELoss(reduction="mean")` |
| 여러 점수를 숫자 하나로 출력 | 먼저 집계한 뒤 `.item()` |
| Python 리스트의 평균 | `sum(values) / len(values)` |
| 밑이 10인 로그 | `torch.log10` |
| 여러 그림 공간 생성 | `plt.subplots` |
| 이미지 번호를 중복 없이 선택 | `random.sample(range(n), k)` |
| GPU 결과를 일반 배열로 표시 | `tensor.detach().cpu().numpy()` |

오타뿐 아니라 <strong>현재 변수에 무엇이 들어 있는지</strong>도 확인해야 한다. `scores`에는 점수가 들어 있고 모델은 들어 있지 않다. `clean_paths`에는 경로가 들어 있고 정답 픽셀은 들어 있지 않다. `val_idx`를 다른 반복문에서 덮어쓰면 지금 가진 모델의 검증 분할과 맞지 않을 수 있다.

### Rust에서는 어떤 개념이 대응하는가

이번 신경망 학습은 Python·PyTorch로 구현했다. 파일 경로와 목록을 관리하는 기본 작업은 Rust에서도 대응시킬 수 있다.

| Python | Rust | 공통 역할 |
| --- | --- | --- |
| `Path`의 `/` | `PathBuf::join` | 경로 연결 |
| `list.append` | `Vec::push` | 목록에 값 추가 |
| `dict` | `HashMap` | 키별 값 보관 |
| `zip` | `Iterator::zip` | 두 순회의 원소를 순서대로 묶음 |
| NumPy `float32` | `f32` | 32비트 실수 표현 |

```rust
use std::{collections::HashMap, env, io};

fn main() -> io::Result<()> {
    // 경로를 만들 뿐, 폴더를 생성하거나 이미지를 읽지는 않는다.
    let train_dir = env::current_dir()?.join("train");
    println!("{}", train_dir.display());

    // 키는 (너비, 높이), 값은 해당 크기의 이미지 수다.
    let sizes = [(540, 420), (540, 258), (540, 420)];
    let mut counts = HashMap::new();

    for size in sizes {
        *counts.entry(size).or_insert(0usize) += 1;
    }

    // HashMap의 순회 순서 대신 특정 키로 조회해 결과를 확인한다.
    println!("420행: {}", counts[&(540, 420)]); // 2
    println!("258행: {}", counts[&(540, 258)]); // 1
    Ok(())
}
```

이 Rust 코드는 경로와 크기 집계만 보여 준다. `Vec` 자체가 자동 미분 텐서인 것은 아니며, 신경망 학습까지 수행하는 예제는 아니다. `zip`이 기본적으로 짧은 쪽에 맞춰 종료한다는 점도 두 언어에서 주의한다.[^rust]

## 24. 지금 익혀야 할 것과 검증한 범위

이 실습에서 먼저 익힐 것은 복잡한 모델 구조보다 <strong>데이터의 대응을 끝까지 유지하고 같은 기준으로 성능을 확인하는 능력</strong>이다.

| 우선순위 | 이번에 익힐 내용 |
| --- | --- |
| 실무 필수 | 파일명 짝짓기, 축·모양 확인, Fold별 새 모델, 장치 일치, 원본 영역 평가, 무처리 비교, 제출 좌표·파일 순서 |
| 원리 이해 | 채널과 층의 차이, 전치 합성곱의 역할, 평균의 조건, RMSE·PSNR의 변환과 집계 |
| 이후 확장 | 그룹 관계 조사, 패치 추출 개선, 스킵 연결, 손실 변경, 여러 Fold 모델의 예측 결합 |

실제 데이터에서 같은 원본 문서의 파생 관계가 확인됐다면 그룹 분할은 선택적인 후속 최적화가 아니라 평가를 바로잡기 위해 먼저 해야 하는 작업이다.

설명을 보지 않고 다음 질문에 답하면서 코드를 실행할 수 있으면, 이 단계의 학습 목적을 달성한 것이다.

- 정답 이미지는 모델의 어느 계산에 사용되는가?
- 한 모델의 학습·검증 구성은 언제 정해지고 언제 바뀌는가?
- `mean`이 없애는 축과 남기는 축은 무엇인가?
- 현재 점수는 이미지마다 같은 비중인가, 픽셀마다 같은 비중인가?
- 패딩을 평가와 제출에서 어느 위치까지 제외하는가?
- 지금 테스트에 쓰는 모델은 마지막 Fold 모델인가, 전체 데이터로 새로 학습한 모델인가?

그다음에는 작은 변경 하나를 구현하고, 동일한 분할에서 실행하고, 수치와 복원 이미지를 함께 비교한다. 세부 수학을 계속 확장하는 것보다 이 실험을 직접 완성하는 편이 현재 목표에 더 도움이 된다.

이 글의 실행 파일은 <strong>PyTorch 2.15.0+cpu</strong>에서 핵심 연산을 확인했다. 합성 PNG 7쌍을 사용해 파일 읽기, 크기가 서로 다른 검증 Fold 3개, 모델 저장, 전체 데이터 재학습, 테스트 이미지 2장의 제출 파일 생성까지 실행했다. 제출 파일의 366,120개 픽셀 행, 원본 크기 복구, 파일명과 좌표, 저장 모델 재로딩 후 예측 일치를 확인했다. 이는 코드의 연결과 집계가 의도대로 동작하는지 확인한 검사다. 원본 Kaggle 데이터의 복원 품질이나 앞서 기록한 대회 점수를 재현한 실험은 아니다.

## 참고 자료

공식 API와 Kaggle CLI 문서는 2026년 11월 7일 확인했다. 데이터 장수와 과거 점수는 제공된 실습 기록을 기준으로 적었다.

[^pathlib]: Python, [pathlib](https://docs.python.org/3/library/pathlib.html) — 경로 연결·파일 검색.
[^pillow]: Pillow, [Image.open과 Image.convert](https://pillow.readthedocs.io/en/stable/reference/Image.html) — 이미지 읽기와 흑백 모드.
[^builtins]: Python, [내장 함수](https://docs.python.org/3/library/functions.html#zip) — zip·enumerate·next.
[^tensor-ops]: PyTorch, [unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html), [stack](https://docs.pytorch.org/docs/stable/generated/torch.stack.html), [cat](https://docs.pytorch.org/docs/stable/generated/torch.cat.html), [flatten](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html) — 축 추가·연결·펼치기.
[^pad]: PyTorch, [functional.pad](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html) — 패딩 순서와 채움 값.
[^activations]: PyTorch, [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [Sigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html) — 활성화 함수.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — 인자·출력 모양·가중치 모양.
[^transpose]: PyTorch, [ConvTranspose2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) — 전치 합성곱의 의미와 출력 모양.
[^kfold]: scikit-learn, [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) — 분할 인덱스와 Fold 크기.
[^groupfold]: scikit-learn, [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) — 그룹이 겹치지 않는 분할.
[^data]: PyTorch, [Dataset과 DataLoader](https://docs.pytorch.org/docs/stable/data.html) — 표본·배치·셔플.
[^mps]: PyTorch, [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html), [CUDA](https://docs.pytorch.org/docs/stable/notes/cuda.html) — 장치 사용.
[^training]: PyTorch, [학습 루프](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — 손실·역전파·파라미터 갱신.
[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) — 모델 구성과 학습·평가 모드.
[^nograd]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) — 자동 미분 기록 중단.
[^mse]: PyTorch, [MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) — 원소별 제곱오차와 집계.
[^psnr]: scikit-image, [PSNR과 SSIM](https://scikit-image.org/docs/stable/api/skimage.metrics.html) — 밝기 범위와 이미지 품질 지표.
[^randomness]: PyTorch, [재현성](https://docs.pytorch.org/docs/stable/notes/randomness.html) — 시드와 환경별 차이.
[^numpy]: PyTorch, [Tensor.numpy](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html), [Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html) — 배열 변환과 미분 연결 분리.
[^subplots]: Matplotlib, [subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) — Figure와 Axes 생성.
[^imshow]: Matplotlib, [imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html) — 이미지 표시와 밝기 범위.
[^l1]: PyTorch, [L1Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) — 절대오차 손실.
[^saving]: PyTorch, [모델 저장과 불러오기](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) — state_dict와 체크포인트.
[^kaggle]: Kaggle, [Denoising Dirty Documents](https://www.kaggle.com/c/denoising-dirty-documents) — 평가와 픽셀별 제출 형식.
[^kaggle-cli]: Kaggle, [대회 CLI 명령](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) — 제출·상태·점수 조회.
[^kaggle-auth]: Kaggle, [CLI 인증 안내](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md), [CLI 소스](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/cli.py) — OAuth 로그인과 quota.
[^cross-validation]: scikit-learn, [교차 검증](https://scikit-learn.org/stable/modules/cross_validation.html) — 모델 선택과 검증의 역할.
[^rust]: Rust, [Path](https://doc.rust-lang.org/std/path/struct.Path.html), [HashMap](https://doc.rust-lang.org/std/collections/struct.HashMap.html), [Iterator::zip](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.zip) — 경로·집계·반복.




