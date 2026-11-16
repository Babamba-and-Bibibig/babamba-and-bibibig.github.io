+++
title = "이미지 분류 평가: 데이터 인덱스부터 오분류 이미지까지"
description = "ImageFolder·Subset·DataLoader의 연결을 조사하고, 정답과 예측을 모아 정확도·정밀도·재현율·F1을 계산한다. 평가 결과에서 원본 파일을 찾는 조건과 검증의 한계까지 살펴본다."
date = 2026-11-16T20:39:00+09:00
slug = "image-classification-evaluation"

[extra]
toc = true
styles = ["image-classification-evaluation/article.css"]
+++

이미지 분류에서 정확도를 출력하는 코드는 짧다. 그런데 틀린 이미지를 직접 찾으려면, 그 숫자를 계산하기까지 어떤 샘플을 어떤 순서로 처리했는지 알아야 한다. `y_true[0]`의 0과 `classes[0]`의 0은 같은 숫자지만 가리키는 대상이 다르다. 이 차이를 놓치면 지표는 맞게 계산하고도 엉뚱한 이미지를 화면에 띄울 수 있다.

이번 글은 파일 경로와 정답이 모델의 입력, 예측, 평가 결과로 이어지는 과정을 다룬다. [흉부 X-ray 분류 실습](/chest-xray-classification-pytorch/)에서 전체 학습과 모델 구성을 살펴봤다면, 여기서는 각 객체의 정보와 번호가 어떻게 연결되는지 더 자세히 확인한다. 앞 글을 읽지 않았어도 따라올 수 있도록 필요한 자료구조부터 설명한다.

본문의 `NORMAL`과 `PNEUMONIA`는 두 클래스의 이름으로 사용한다. 파일 수와 혼동 행렬에 붙인 예시 숫자는 설명과 계산을 위한 값이다. 실제 모델의 실행 여부와 성능은 별도로 검증해야 한다. 실행 흐름을 확인할 수 있는 [전체 Python 예제](/image-classification-evaluation/examples-ko.py)와 그 검증 범위는 마지막 절에 안내한다.

## 01. 어떤 정보를 끝까지 연결해야 하는가

목표는 이미지 한 장의 클래스를 예측하고, 틀렸다면 어떤 파일을 어떤 클래스로 잘못 예측했는지 확인하는 것이다. <strong>클래스(Class)</strong>는 구분할 종류, <strong>라벨(Label)</strong>은 각 이미지에 붙은 정답이다. 이미지 한 장과 정답 하나를 묶어 <strong>샘플(Sample)</strong>이라고 부른다.

먼저 전체 작업에서 필요한 입력과 출력을 나눈다.

<div class="eval-table">

| 단계 | 입력 | 처리 후 남아야 할 정보 |
| --- | --- | --- |
| 원본 조사 | 클래스별 이미지 폴더 | 파일 경로, 클래스 이름과 번호, 샘플 개수 |
| 분할 | 전체 샘플 인덱스와 정답 | 학습용·검증용 원본 인덱스 목록 |
| 전처리와 배치 구성 | 선택한 이미지·라벨 | 이미지 텐서와 같은 순서의 정답 텐서 |
| 모델 계산 | 이미지 텐서 | 이미지마다 클래스별 점수 |
| 학습 | 점수와 정답 | 손실과 수정된 모델 가중치 |
| 검증 | 현재 모델의 점수와 정답 | 정답 목록과 예측 목록 |
| 결과 분석 | 두 목록과 원본 연결 정보 | 지표, 혼동 행렬, 오분류 파일과 두 클래스 이름 |

</div>

<strong>샘플 인덱스(Sample Index)</strong>는 어떤 샘플을 꺼낼지 지정하는 정수다. <strong>클래스 번호(Class ID)</strong>는 그 샘플이 어느 종류에 속하는지 나타내는 정수다. 인덱스는 0부터 시작한다. 샘플이 100개라는 것은 샘플 인덱스가 0부터 99까지 있다는 뜻이지, 클래스가 100개라는 뜻은 아니다.

코드를 쓰기 전에는 세 가지를 묻는다. 지금 필요한 것이 픽셀인지, 파일 경로인지, 정답인지 확인한다. 그 정보가 어느 객체에 있는지 찾는다. 마지막으로 다른 정보와 묶을 때 같은 샘플이라는 근거가 있는지 확인한다.

## 02. ImageFolder의 시작 경로는 클래스 폴더를 기준으로 정한다

`ImageFolder`는 클래스별 폴더에서 샘플을 제공하는 torchvision의 데이터셋 클래스다. <strong>데이터셋(Dataset)</strong>은 여기서 샘플 개수를 알려 주고, 샘플 인덱스를 받으면 해당 이미지와 정답을 제공하는 객체를 뜻한다.

기본 `ImageFolder`는 `root` 바로 아래의 하위 폴더 이름을 클래스 이름으로 사용한다. 각 클래스 폴더 안에서는 더 아래의 폴더도 탐색한다. `train`이라는 이름의 뜻을 이해해서 학습용 폴더를 자동으로 찾아주는 것은 아니다.[^folder]

<div class="eval-table">

| 실제 파일 경로의 형태 | 클래스 폴더의 부모인 `root` |
| --- | --- |
| `animals/cat/a.png`, `animals/dog/b.png` | `animals` |
| `animals/images/cat/a.png` | `animals/images` |
| `chest_xray/train/NORMAL/a.jpeg` | `chest_xray/train` |
| `images/a.png`와 별도 `labels.csv` | 기본 ImageFolder만으로 CSV 정답을 연결할 수 없음 |

</div>

`chest_xray` 바로 아래에 `train`과 `test`가 있다면, 그 폴더를 `root`로 지정했을 때 두 이름이 클래스가 된다. 정상과 폐렴을 구분하려던 목적과 다른 정답 체계가 만들어지는 것이다.

`Path`는 경로를 다루는 Python 클래스다. 경로 사이의 `/`는 경로를 이어 붙이며, 폴더를 생성하지 않는다. `Path.cwd()`는 현재 작업 폴더, `Path.home()`은 사용자 홈 폴더를 알려준다. `iterdir()`는 바로 아래의 항목을 순회한다.[^path]

~~~python
from pathlib import Path
from torchvision.datasets import ImageFolder

print("현재 작업 폴더:", Path.cwd())

# 실제로 압축을 푼 경로에 맞게 수정한다.
data_dir = Path.home() / "Downloads" / "chest_xray"
root = data_dir / "train"

if not root.is_dir():
    raise FileNotFoundError(f"클래스 폴더의 부모 경로를 확인하세요: {root}")

print("바로 아래 폴더:", [p.name for p in root.iterdir() if p.is_dir()])
dataset = ImageFolder(root=root)
print("클래스 번호:", dataset.class_to_idx)
~~~

상대 경로 `Path("chest_xray/train")`는 현재 작업 폴더를 기준으로 해석된다. 경로를 만들었다는 사실과 해당 파일이 실제로 존재한다는 사실을 구분해야 한다.

## 03. 데이터셋 전체와 그 안의 목록을 구분한다

`ImageFolder`를 만들면 파일 경로와 라벨 정보를 조사한다. 모든 이미지의 픽셀을 미리 읽어 쌓아 두지는 않는다. 실제 픽셀은 샘플을 요청할 때 읽는다.[^folder]

<div class="eval-table">

| 표현 | 자료형 | 저장 내용·역할 |
| --- | --- | --- |
| `dataset` | `ImageFolder` 객체 | 샘플을 읽는 규칙과 아래 정보 전체 |
| `dataset.classes` | 문자열 리스트 `list[str]` | 번호 순서의 클래스 이름 |
| `dataset.class_to_idx` | 딕셔너리 `dict[str, int]` | 클래스 이름 → 클래스 번호 |
| `dataset.samples` | 튜플 리스트 `list[tuple[str, int]]` | 샘플별 `(파일 경로, 정답 번호)` |
| `dataset.targets` | 정수 리스트 `list[int]` | samples와 같은 순서의 정답 번호 |
| `len(dataset)` | 정수 `int` | 전체 샘플 개수 |

</div>

리스트는 순서가 있는 목록이고, 튜플은 여기서 경로와 라벨 두 값을 한 쌍으로 묶는다. 딕셔너리는 이름처럼 정한 키로 값을 찾는 자료구조다. `targets`는 데이터셋 전체가 아니라 그 안의 정답 목록이다.

설명용으로 정상 2장, 폐렴 2장이 있다고 하자. 기본 클래스 정렬에서 `classes`가 `["NORMAL", "PNEUMONIA"]`라면 번호는 각각 0과 1이다. `samples`의 라벨이 순서대로 0, 0, 1, 1이면 `targets`도 `[0, 0, 1, 1]`이다. 기본 라벨을 변경하지 않은 경우 다음 두 연결이 성립한다.

~~~python
# i는 원본 샘플 인덱스다. 여기서는 첫 샘플을 조사한다.
i = 0
img_path, label = dataset.samples[i]

assert label == dataset.targets[i]
print("파일 경로:", img_path)
print("정답 번호:", label)
print("정답 이름:", dataset.classes[label])
~~~

`assert`는 조건이 거짓이면 오류를 내고 멈추는 검사 문법이다. 위 검사는 목록 사이의 대응을 확인한다. 폴더 안에 이미지가 잘못 분류되어 있는지는 알 수 없다. `ImageFolder`는 이미지 내용을 판단하는 자동 라벨링 도구가 아니라, 폴더에 표현된 정답을 번호로 연결하는 도구다.

### 개수를 셀 때는 라벨만 읽으면 된다

클래스별 이미지 개수를 구하려면 픽셀은 필요 없다. 각 클래스의 개수를 0으로 준비하고, 라벨을 읽을 때마다 해당 클래스의 개수를 하나씩 늘리면 된다.

~~~python
# 키는 클래스 이름, 값은 누적한 이미지 개수다.
class_count = {name: 0 for name in dataset.classes}

for label in dataset.targets:
    class_name = dataset.classes[label]
    class_count[class_name] += 1

print(class_count)
assert sum(class_count.values()) == len(dataset)
~~~

`class_count.values()`는 딕셔너리의 개수 값들을 제공하고, `sum()`은 그 합을 구한다. `for image, label in dataset`으로 순회하면 필요하지 않은 파일 읽기와 전처리까지 실행한다. 어떤 종류가 몇 개인지 셀 때는 종류를 구분하는 정보만 사용하면 된다.

## 04. 파일 경로를 꺼내는 일과 이미지를 읽는 일은 다르다

같은 샘플 인덱스를 사용해도 조회하는 대상에 따라 반환값이 달라진다.

<div class="eval-table">

| 표현 | 수행하는 일 | 반환값 |
| --- | --- | --- |
| `dataset.samples[i]` | 저장된 목록 조회 | 경로 문자열과 정답 정수 |
| `dataset[i]` | 파일 읽기와 설정된 변환 실행 | 이미지 객체 또는 텐서와 정답 |

</div>

변환을 지정하지 않은 기본 PIL 로더에서는 <strong>Pillow 이미지</strong>를 받는다. Pillow는 이미지 파일을 읽고 다루는 라이브러리다. `type(img)`는 객체의 종류를 알려주고, `img.size`는 너비와 높이를 담은 튜플이다.

~~~python
img, label = dataset[0]
img_path, stored_label = dataset.samples[0]

print("파일명:", Path(img_path).name)
print("객체 종류:", type(img))
print("색상 모드:", img.mode)
print("너비, 높이:", img.size)
assert label == stored_label
img.close()
~~~

파일명은 `img.name`을 추측해서 찾지 않고, `samples`의 경로에서 가져온다. 기본 PIL 로더는 읽은 이미지를 RGB로 변환한다. 반환된 `img.mode`가 RGB라고 해서 디스크의 원본도 RGB였다고 결론 내릴 수는 없다. 원본 모드가 필요하면 `Image.open(img_path)`로 파일 자체를 조사해야 한다.[^folder]

<strong>텐서(Tensor)</strong>는 PyTorch에서 여러 축의 숫자를 계산하는 자료형이다. 이후 전처리가 이미지를 텐서로 바꾸면 조사하는 방법도 달라진다.

<div class="eval-table">

| 확인할 것 | Pillow 이미지 | PyTorch 텐서 |
| --- | --- | --- |
| 객체 종류 | `type(img)` | `type(img)` |
| 크기 | `img.size`: `(너비, 높이)` | `img.shape`: 이미지 한 장이면 보통 `(채널, 높이, 너비)` |
| 숫자의 자료형 | 필요하면 `np.asarray(img).dtype` | `img.dtype` |
| 색상 모드 | `img.mode` | 같은 `mode` 속성은 없음 |

</div>

여기서 `np`는 `import numpy as np`로 불러온 NumPy다. `asarray`는 이미지를 숫자 배열로 바꿔 조사할 때 사용한다. `dtype`은 객체 종류가 아니라 내부 숫자의 자료형을 뜻한다. 텐서의 `img.size()`는 호출하는 메서드이고, Pillow의 `img.size`는 저장된 속성이라는 차이도 있다.

## 05. 전처리는 모델이 요구하는 입력 구조에 맞춘다

<strong>전처리(Preprocessing)</strong>는 원본 데이터를 모델에 전달할 수 있는 형태로 바꾸는 작업이다. 변환 함수를 고르기 전에 필요한 채널 수와 높이·너비부터 정한다. 여기서는 RGB 3채널, 높이 224, 너비 224인 실수형 텐서를 예로 든다.

이미지 한 장의 구조 `(C, H, W)`에서 `C`는 채널 개수, `H`는 높이, `W`는 너비다. 축 번호 0, 1, 2가 각각 이 세 의미에 대응한다. `Resize((224, 224))`에는 <strong>높이와 너비를 담은 튜플 하나</strong>를 전달한다. `Resize(224, 224)`와는 다르다.

`Compose`는 변환을 나열한 순서대로 실행한다. `ToTensor`는 일반적인 8비트 RGB·흑백 이미지를 채널이 앞에 오는 실수형 텐서로 만들고 픽셀을 255로 나눈다. 모든 입력 자료형에 같은 범위 변환을 적용하는 것은 아니다.[^transforms]

~~~python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

tensor_dataset = ImageFolder(root=root, transform=val_transform)
image_tensor, label = tensor_dataset[0]

print(image_tensor.shape)  # 이 전처리에서는 (3, 224, 224)
print(image_tensor.dtype)
print(image_tensor.min().item(), image_tensor.max().item())
~~~

`min()`과 `max()`는 가장 작은 값과 큰 값을 구한다. `.item()`은 값 하나를 담은 텐서에서 Python 숫자를 꺼낸다. 위 전처리의 일반적인 8비트 입력은 0~1 범위가 되므로 다시 `/255`하지 않는다. 크기를 강제로 맞추면 원본 비율이 바뀔 수 있다. 이 글에서는 구조를 통일하는 기본 예시로 사용하며, 실제 작업에서는 자르기나 패딩이 더 적절한지도 확인한다.

`Grayscale(num_output_channels=1)`을 사용하면 이미지 텐서의 채널 수는 1이 된다. 모델도 1채널 입력을 받도록 맞춰야 한다. 뒤에서 설명할 Normalize에서도 RGB용 평균·표준편차 세 개를 1채널 텐서에 그대로 적용하지 않는다.

<strong>데이터 증강(Data Augmentation)</strong>은 정답이 유지되는 범위에서 학습 이미지에 변형을 주는 작업이다. `RandomHorizontalFlip(p=0.5)`는 읽을 때 절반의 확률로 좌우 반전하지만, 파일을 수정하거나 샘플 개수를 늘리지 않는다. 영상의 방향 자체가 의미를 가진다면 반전을 자동으로 넣지 않는다. 검증 전처리에는 여기서 무작위 증강을 사용하지 않는다.

### Normalize는 전달받은 기준값을 사용한다

<strong>표준화(Standardization)</strong>는 평균을 빼고 표준편차로 나누어 수치의 기준을 바꾸는 작업이다. 평균은 값들의 중심, 표준편차는 평균 주변에서 값들이 퍼진 정도다. 평균을 빼면 그 중심보다 큰지 작은지가 나타나고, 표준편차로 나누면 그 차이를 일정한 변화 폭 기준으로 나타낼 수 있다.

`Normalize(mean=..., std=...)`는 채널마다 전달된 고정 평균과 표준편차를 사용한다. 이미지 한 장마다 통계를 새로 구하지 않는다. 변환 뒤 값은 0~1을 벗어날 수 있고, 다른 데이터에서 얻은 기준을 썼다고 내 이미지의 평균이 0이 되지도 않는다. 직접 기준을 계산한다면 학습 데이터로 구하고 검증·테스트에 재사용한다.[^normalize]

<strong>사전학습(Pretraining)</strong>은 다른 데이터로 가중치를 미리 학습하는 것, <strong>전이학습(Transfer Learning)</strong>은 그 모델을 새 작업에 사용하는 것이다. 기존 가중치를 사용할 때는 그 가중치가 기대하는 전처리를 먼저 확인한다.

~~~python
from torchvision.models import ResNet18_Weights

# 사용할 가중치 버전을 가리키는 설정 객체다.
weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()
print(preprocess)
~~~

`weights` 자체가 학습된 숫자 텐서라는 뜻은 아니다. 가중치 파일과 전처리 정보가 연결된 설정 객체다. `weights.transforms()`는 등록된 평가·추론용 전처리를 만든다. 해당 ResNet18 설정에는 크기 조정, 중앙 자르기, 범위 변환과 ImageNet 채널 통계 적용이 포함된다. 여기에 `ToTensor`와 `Normalize`를 무조건 다시 붙이면 안 된다.[^weights]

전처리 객체를 만드는 것과 `resnet18(weights=weights)`로 모델과 가중치를 불러오는 것은 별개다. 후자도 새 작업의 출력 클래스 수를 자동으로 알아내지는 않는다. 모델의 출력층 변경은 [앞 글의 모델 구성](/chest-xray-classification-pytorch/)에서 다룬다.

## 06. 샘플 선택과 전처리 설정을 따로 설계한다

학습은 정답을 이용해 모델의 가중치를 바꾸는 과정이다. 검증은 현재 모델을 평가하며 설정을 선택하는 데 사용하고, 테스트는 선택을 마친 모델을 따로 남긴 데이터에서 최종 평가한다. 검증과 테스트에서는 가중치를 갱신하지 않는다.

먼저 원본 샘플 인덱스를 학습용과 검증용으로 나눈다. 이미지 픽셀을 읽어서 분할할 필요는 없다. 어떤 샘플이 어느 클래스인지 알려주는 `targets`만 있으면 클래스 비율을 고려할 수 있다.

<strong>층화 분할(Stratified Split)</strong>은 기존 클래스 비율을 두 분할에 가능한 한 유지하는 방식이다. 원본 비율이 20:80이면 이를 비슷하게 유지하려는 것이지 50:50으로 균형을 맞추는 작업이 아니다.[^split]

~~~python
from sklearn.model_selection import train_test_split

# range의 끝값은 포함하지 않는다. 전체 원본 샘플 인덱스를 만든다.
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=dataset.targets,
    random_state=42,
)

assert len(train_idx) + len(val_idx) == len(dataset)
assert set(train_idx).isdisjoint(val_idx)
~~~

`test_size=0.2`는 두 번째 반환 목록에 약 20%를 배정한다. 이름에 test가 들어 있지만 여기서는 검증용으로 사용한다. `stratify`에 샘플별 정답을 전달하며, `random_state`는 같은 입력과 조건에서 분할을 재현하기 위한 난수 초기값이다. `set(...).isdisjoint(...)`는 두 집합에 공통 원소가 없는지 확인한다. 클래스별 샘플이 너무 적거나 분할 크기가 부족하면 층화 분할이 실패할 수 있다.

이제 <strong>Subset</strong>으로 선택한 인덱스만 제공한다. Subset은 원본 데이터셋과 인덱스 목록을 연결하며, 이미지 파일을 복사하지 않는다. `ImageFolder`에는 이 목적의 `indices` 인자가 없고, 기본 `dataset[train_idx]`도 정수 리스트 전체를 선택하는 방법이 아니다.[^data]

학습과 검증의 전처리를 독립적으로 설정하려면 원본을 읽는 `ImageFolder` 객체를 따로 만든다. 두 객체가 같은 샘플 목록과 클래스 번호를 사용하는지 확인한 뒤 같은 기준의 분할 인덱스를 적용한다.

~~~python
from torch.utils.data import Subset

train_base = ImageFolder(root=root, transform=train_transform)
val_base = ImageFolder(root=root, transform=val_transform)

# 번호만 같아서는 부족하다. 그 번호가 가리키는 경로·라벨·순서가 같아야 한다.
assert dataset.samples == train_base.samples == val_base.samples
assert dataset.class_to_idx == train_base.class_to_idx == val_base.class_to_idx

train_dataset = Subset(train_base, train_idx)
val_dataset = Subset(val_base, val_idx)
~~~

`train_base`와 `val_base`는 둘 다 같은 전체 샘플에 접근한다. 각 Subset이 그중 사용할 샘플을 정한다. 두 Subset이 같은 ImageFolder 객체를 참조하면 그 객체의 `transform`도 공유한다. Subset에 임의로 `transform` 속성을 붙여도 기본 Subset이 이를 실행해 주지는 않는다.

인덱스 교집합이 없다는 검사는 동일 인덱스의 중복만 막는다. 복사된 파일이나 같은 환자의 여러 영상이 양쪽에 섞였는지는 별도로 조사해야 한다. 환자 같은 묶음을 분리하려면 신뢰할 수 있는 그룹 정보와 그에 맞는 분할이 필요하다. PyTorch의 `random_split`은 무작위로 나눈 Subset을 반환하지만 `stratify` 인자는 없다.[^data][^split]

## 07. Subset을 거치면 인덱스의 기준이 바뀐다

원본이 네 샘플이고 설명용 선택 목록이 `[3, 1]`이라면, Subset의 첫 샘플은 원본의 네 번째 샘플이다. 요청할 때 쓰는 인덱스는 각각 0과 3으로 다르다. 선택 목록의 첫 원소에 원본 인덱스 3이 들어 있기 때문에 연결할 수 있다.

<div class="eval-table">

| Subset 내부 샘플 인덱스 | 선택 목록에서 꺼낸 원본 샘플 인덱스 | 실제 요청되는 샘플 |
| ---: | ---: | --- |
| 0 | 3 | `val_base[3]` |
| 1 | 1 | `val_base[1]` |

</div>

이 예에서는 `val_dataset[0]`을 요청하면 `val_dataset.indices[0]`을 읽고, 그 결과 3으로 `val_base[3]`을 요청한다. 샘플을 조회하는 번호의 기준은 달라지지만, 연결된 원본 파일과 기본 클래스 번호는 유지된다. 무작위 변환을 사용한다면 같은 파일을 다시 읽어도 변환된 픽셀은 달라질 수 있다.

<figure class="eval-figure">
<img src="/image-classification-evaluation/data-identity-ko.svg" alt="설명용 선택 목록 [3, 1]에서 검증 샘플 인덱스 0은 원본 샘플 인덱스 3으로 연결된다. samples에서 파일과 라벨을 찾고, 클래스 번호 1은 classes의 PNEUMONIA 이름으로 연결된다." loading="lazy">
<figcaption>파일을 찾는 샘플 인덱스와 이름을 찾는 클래스 번호를 따로 따라간다. 그림의 선택 목록은 원리를 보여 주는 예시다.</figcaption>
</figure>

이후 배치를 제공할 `DataLoader`까지 연결하면 객체는 다음과 같다. <strong>DataLoader</strong>는 데이터셋에서 샘플을 꺼내 여러 샘플의 묶음인 <strong>배치(Batch)</strong>로 제공하는 객체다.

<div class="eval-table">

| 변수 | 실제 종류 | 포함 범위·역할 |
| --- | --- | --- |
| `val_base` | ImageFolder | 원본 전체의 경로·라벨·클래스 정보와 검증 전처리 |
| `val_dataset` | Subset | 원본 중 검증용으로 선택한 샘플 |
| `val_loader` | DataLoader | 선택된 검증 샘플을 배치로 제공 |

</div>

`val_dataset.dataset is val_base`는 두 표현이 같은 객체를 가리키는지 검사한다. `is`는 값이 비슷한지가 아니라 객체가 동일한지를 비교한다. 아래에서 만들 `val_loader.dataset`은 바로 `val_dataset`이다. DataLoader에 ImageFolder를 직접 넣었다면 그 속성도 ImageFolder가 된다. `.dataset`이라는 이름만으로 내부 종류를 단정하지 않는다.

## 08. 배치 개수, 샘플 개수, 순회 상태를 구분한다

학습에서는 같은 샘플 집합을 여러 차례 사용하므로 읽는 순서를 섞을 수 있다. 검증에서는 결과를 원본과 쉽게 연결하도록 기본 순서를 유지하고, 마지막에 남은 작은 배치도 포함한다.

~~~python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    drop_last=False, num_workers=0,
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False,
    drop_last=False, num_workers=0,
)

assert val_loader.dataset is val_dataset
assert val_dataset.dataset is val_base

images, labels = next(iter(val_loader))
print(images.shape, labels.shape)
print(images.dtype, labels.dtype)
assert images.shape[0] == labels.shape[0]
~~~

`batch_size`는 한 배치에 넣을 최대 샘플 개수다. `shuffle`은 그 집합을 읽는 순서를 섞으며 이미지와 정답을 따로 섞지 않는다. `drop_last`는 마지막의 작은 배치를 버릴지 정한다. `num_workers=0`은 별도의 로딩 작업 프로세스를 만들지 않는 설정으로, 여기서는 흐름을 단순하게 확인하기 위해 사용한다.[^data]

분할은 사용할 샘플 집합을 나누는 일이고, shuffle은 그 집합 안의 순서를 바꾸는 일이다. DataLoader를 만들었다고 학습이 일어나는 것도 아니다. 이후 어떤 계산을 실행하느냐에 따라 학습이나 검증이 된다.

`B`를 현재 배치의 샘플 개수라고 하면 이미지 배치의 구조는 `(B, C, H, W)`, 라벨 구조는 `(B,)`다. 배치의 축 번호 0은 샘플, 1은 채널, 2는 높이, 3은 너비를 뜻한다. 기본 방식으로 배치를 묶으려면 이미지 텐서들의 크기가 같아야 한다. ImageFolder 자체가 모든 이미지 크기를 통일해 주지는 않는다.

검증 샘플이 100개이고 배치 크기가 32인 <strong>개수 예시</strong>를 보자.

<div class="eval-table">

| 확인 대상 | 값 | 무엇을 센 것인가 |
| --- | ---: | --- |
| `len(val_dataset)` | 100 | 검증 샘플 개수 |
| `len(val_loader.dataset)` | 100 | DataLoader에 전달한 데이터셋의 샘플 개수 |
| `len(val_loader)` | 4 | 배치 개수 |
| 반복 중 `len(labels)` | 32, 32, 32, 4 | 현재 배치의 실제 샘플 개수 |

</div>

전체 처리 개수를 누적할 때는 `total += len(labels)`를 사용한다. `len(val_loader)`를 배치마다 더하면 배치 개수를 반복해서 더하게 된다. 이 예시에서 `drop_last=True`라면 마지막 4개를 제외한 96개만 처리한다.

### next는 반복자에서 다음 배치를 꺼낸다

<strong>반복 가능한 객체(Iterable)</strong>는 순회를 시작할 수 있는 객체이고, <strong>반복자(Iterator)</strong>는 그 순회가 어디까지 진행됐는지 관리하는 객체다. DataLoader로 `iter()`를 호출하면 배치를 꺼낼 반복자를 얻는다. `next()`는 그 반복자에서 다음 배치 하나를 꺼낸다.[^iteration]

~~~python
batch_iter = iter(val_loader)
images1, labels1 = next(batch_iter)

# 두 번째 배치가 존재하는 경우에만 실행한다.
if len(val_loader) >= 2:
    images2, labels2 = next(batch_iter)
~~~

`next(iter(val_loader))`를 매번 새로 실행하면 같은 반복자의 다음 배치로 이어지는 것이 아니다. 매번 새 순회를 시작해 첫 배치를 받는다. 기본 순차 설정에서는 같은 첫 배치를 받으며, shuffle 설정이나 무작위 변환에 따라 새 순회의 내용은 달라질 수 있다. 소진한 반복자에 `next()`를 더 호출하면 `StopIteration`이 발생한다. 전체를 처리할 때는 `for images, labels in val_loader`를 사용하면 반복자 관리와 종료 처리를 맡길 수 있다.

## 09. 모델의 점수, 활성함수, 손실함수는 역할이 다르다

이미지마다 정답 클래스가 하나인 분류를 생각하자. 모델은 이미지의 숫자와 학습 가능한 <strong>가중치(Weight)</strong>를 이용해 중간값을 계산하고, 마지막에 클래스별 점수를 만든다. <strong>활성함수(Activation Function)</strong>는 중간값을 변환해 다음 계산에 전달한다. <strong>손실함수(Loss Function)</strong>는 최종 점수와 실제 정답을 비교해 개선할 기준을 숫자로 만든다.

예를 들어 ReLU는 음수를 0으로 바꾸고 양수는 그대로 둔다. 가중치를 곱하고 더하는 계산만 이어 붙이면 하나의 같은 형태의 계산으로 합쳐질 수 있다. 중간에 ReLU처럼 입력에 따라 반응이 달라지는 변환을 넣으면 더 복잡한 관계를 표현할 수 있다. 활성함수는 실제 정답을 입력으로 받지 않지만, 분류 손실은 정답과 예측을 비교해야 한다.

모델의 확률 변환 전 점수를 <strong>로짓(Logits)</strong>이라고 한다. 음수가 될 수 있으며 합이 1일 필요도 없다. 클래스 개수를 `K`라고 하면 다음 구조를 사용한다.

<div class="eval-table">

| 변수 | 자료형·구조 | 축의 의미 |
| --- | --- | --- |
| `images` | 실수형 텐서 `(B, C, H, W)` | 샘플, 채널, 높이, 너비 |
| `logits` | 실수형 텐서 `(B, K)` | 샘플, 클래스별 점수 |
| `labels` | `torch.int64` 텐서 `(B,)` | 샘플별 정답 클래스 번호 |
| `preds` | 정수형 텐서 `(B,)` | 샘플별 예측 클래스 번호 |
| `loss` | 기본 평균 설정에서 shape `()`인 텐서 | 배치 손실을 나타내는 값 하나 |

</div>

모델은 `C`채널 입력을 받고 `K`개 점수를 출력하도록 준비되어 있어야 한다. ImageNet의 기본 1,000클래스 출력을 그대로 두고 두 클래스에 맞는 모델을 완성했다고 볼 수는 없다.

`CrossEntropyLoss`는 이 경우 로짓과 정수 정답을 받는다. 정답 클래스에 다른 클래스보다 충분히 큰 점수를 주었는지 평가하며, 필요한 로그 확률 계산을 내부에서 처리한다. 앞에서 Softmax를 따로 적용하지 않는다.[^loss]

<strong>Softmax</strong>는 한 샘플의 점수들을 합이 1인 확률 형태로 변환한다. <strong>argmax</strong>는 가장 큰 값 자체가 아니라 그 값의 인덱스를 반환한다. `logits`의 축 번호 1은 클래스 축이므로 `argmax(dim=1)`의 결과는 샘플별 클래스 번호다. Softmax가 점수의 대소관계를 유지하므로, 예측 번호만 필요할 때는 먼저 Softmax를 계산할 이유가 없다.

~~~python
from torch import nn

# model은 입력 채널과 출력 클래스 개수가 맞도록 준비된 모델이다.
# model, images, labels는 같은 계산 장치에 있다고 놓는다.
loss_fn = nn.CrossEntropyLoss()
logits = model(images)
loss = loss_fn(logits, labels)
preds = logits.argmax(dim=1)
~~~

`images`의 `dim=1`은 채널이고 `logits`의 `dim=1`은 클래스다. 숫자 1의 의미는 텐서 전체 구조에 따라 달라진다. 확률 형태로 출력했다고 실제 정답 가능성을 정확히 나타낸다는 뜻도 아니다.

다른 손실도 있지만 출력·정답 구조를 함께 맞춰야 한다. `BCEWithLogitsLoss`는 점수 하나의 이진 분류나 독립적인 여러 이진 판단에 사용하며, `MSELoss`·`L1Loss`는 연속적인 숫자의 오차를 다룰 때 사용한다. 현재 코드에서 함수 이름만 바꾸어 교체하지 않는다.

## 10. 학습 모드 전환과 가중치 갱신을 구분한다

<strong>기울기(Gradient)</strong>는 손실이 가중치 변화에 어떻게 반응하는지 나타낸다. <strong>역전파(Backpropagation)</strong>는 손실에서 출발해 가중치별 기울기를 계산하는 과정이고, <strong>옵티마이저(Optimizer)</strong>는 그 기울기로 가중치를 수정하는 도구다.

학습은 이전 기울기 초기화, 점수 계산, 손실 계산, 기울기 계산, 가중치 갱신 순서로 진행한다. 모델의 각 층과 뉴런을 구성하는 일, 데이터를 넣어 가중치를 실제로 학습하는 일은 별개다.

<div class="eval-table">

| 표현 | 실제로 하는 일 | 혼동하기 쉬운 부분 |
| --- | --- | --- |
| `model.train()` | 학습 모드로 전환 | 호출만으로 가중치가 갱신되지 않음 |
| `optimizer.zero_grad()` | 이전 기울기 초기화 | 가중치를 0으로 만들지 않음 |
| `loss.backward()` | 기울기 계산 | 가중치 갱신은 아직 하지 않음 |
| `optimizer.step()` | 기울기를 이용한 가중치 갱신 | 학습에서 파라미터를 수정하는 단계 |
| `model.eval()` | 평가 모드로 전환 | 기울기 기록을 자동으로 끄지는 않음 |
| `with torch.no_grad():` | 해당 범위의 역전파용 기록 비활성화 | 평가 모드로 자동 전환하지 않음 |

</div>

학습 모드와 평가 모드의 차이는 Dropout, BatchNorm처럼 모드에 따라 동작이 달라지는 층에 영향을 준다. <strong>Dropout</strong>은 학습 중 일부 중간 출력을 무작위로 제외하는 층이고, <strong>BatchNorm</strong>은 학습·평가 때 사용하는 통계 처리 방식이 다른 층이다. 현재는 두 기능의 내부 구현보다 `eval()`과 `no_grad()`가 서로 다른 설정이라는 점을 확인하면 된다.[^autograd]

모델과 입력·정답 텐서는 계산 가능한 같은 장치에 둔다. 일반적인 단일 장치 모델에서는 첫 번째 파라미터 텐서의 `.device`로 모델 장치를 확인할 수 있다. 여기서 `next(model.parameters())`는 첫 파라미터를 꺼내는 것이며, 앞 절의 배치 반복자와 꺼내는 대상이 다르다.

~~~python
import torch

# model을 사용할 장치에 옮긴 뒤 optimizer를 준비한다.
# lr은 한 번에 가중치를 수정하는 크기를 조절하는 학습률이다.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = next(model.parameters()).device

model.train()
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
~~~

일반적인 배치 학습에서는 새 배치의 기울기로 갱신하기 위해 이전 기울기를 지운다. PyTorch가 `backward()`에서 기울기를 누적하기 때문이다. 지금처럼 전체 학습 샘플을 한 번씩 사용하는 순회를 <strong>에포크(Epoch)</strong> 한 번이라고 한다.

Mac의 Apple GPU는 `mps`, NVIDIA GPU는 `cuda`, 일반 CPU는 `cpu` 장치를 사용한다. 실제 지원 여부를 확인해 선택해야 하며, 본문 코드 조각은 적절한 모델이 준비되었다는 전제다. 모델 생성까지 포함한 실행 예제는 마지막 절에서 제공한다.

## 11. 검증 결과는 샘플마다 같은 순서로 모은다

전체 정답 개수와 전체 샘플 개수만 알면 정확도는 구할 수 있다. 그러나 어떤 클래스를 어떤 클래스로 틀렸는지는 복원할 수 없다. 그래서 실제 클래스 번호를 담을 `y_true`와 예측 번호를 담을 `y_pred`를 함께 만든다. 둘 다 `list[int]`이며 같은 인덱스가 같은 이미지의 결과를 가리켜야 한다.

배치별 결과를 하나의 긴 목록으로 붙일 때는 `extend()`를 쓴다. `append([0, 1])`는 리스트 하나를 원소로 추가해 `[[0, 1]]`이 되지만, `extend([0, 1])`는 정수들을 이어 붙여 `[0, 1]`이 된다.[^lists]

다음 코드는 검증 샘플을 전부 한 번씩 처리하고, 개수·손실·정답·예측을 함께 모은다. `CrossEntropyLoss()`의 기본 설정인 샘플 가중치 없는 평균 손실을 사용하며, 무시할 정답 값은 없다고 놓는다.

~~~python
model.eval()
correct = 0
total = 0
loss_sum = 0.0
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        assert logits.shape == (len(labels), len(val_base.classes))
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)

        batch_count = len(labels)
        correct += (preds == labels).sum().item()
        total += batch_count
        loss_sum += loss.item() * batch_count

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

assert total > 0
assert len(y_true) == len(y_pred) == total == len(val_dataset)

accuracy = correct / total
mean_loss = loss_sum / total
print(f"정확도: {accuracy:.2%}, 평균 손실: {mean_loss:.4f}")
~~~

`preds == labels`는 샘플별로 맞으면 True, 틀리면 False인 텐서를 만든다. `.sum()`은 True를 1로 세어 정답 개수를 더하고, `.item()`은 그 결과를 Python 숫자로 바꾼다. `.cpu()`는 CPU 텐서를 얻고 `.tolist()`는 숫자를 Python 리스트로 바꾼다. `tolist()` 자체도 필요한 CPU 이동을 처리하므로 앞의 `.cpu()`가 반드시 필요한 것은 아니다.[^tolist]

Dataset과 DataLoader는 텐서가 아니므로 `val_loader.dataset.cpu()`처럼 옮기지 않는다. `.cpu()` 호출이 원래 변수까지 덮어쓰는 것도 아니다. 기울기를 추적하는 텐서를 NumPy로 조사할 때 자주 쓰는 `tensor.detach().cpu().numpy()`에서는 `detach()`가 추적을 분리하고, `cpu()`가 장치를 맞추고, `numpy()`가 배열로 변환한다. 서로 다른 작업이다.

### 왜 배치 평균을 다시 단순 평균하지 않는가

각 배치 평균 안에서는 그 배치의 이미지들이 같은 비중을 가진다. 그런데 배치 크기가 다르면, 배치 평균을 같은 비중으로 다시 평균하는 순간 작은 배치의 이미지 한 장이 더 큰 영향을 받는다. 모든 이미지가 동일하게 한 번씩 반영되게 하려면 배치 평균에 실제 샘플 개수를 곱해 합계로 되돌린 뒤, 전체 샘플 개수로 나눠야 한다.

<figure class="eval-figure">
<img src="/image-classification-evaluation/batch-mean-ko.svg" alt="샘플 100개가 32, 32, 32, 4개 배치로 나뉜 예시. 배치 평균을 동일 비중으로 평균하면 마지막 4개 샘플의 영향이 커진다. 배치 평균에 샘플 개수를 곱해 합친 뒤 100으로 나누면 샘플마다 같은 비중을 갖는다." loading="lazy">
<figcaption>같은 대상을 같은 비중으로 평가하려는지 먼저 정한다. 여기서는 샘플 한 장이 기준이다.</figcaption>
</figure>

정확도도 전체 정답 개수를 전체 처리 개수로 나눈다. 배치 크기가 모두 같다면 배치 정확도를 단순 평균해도 같지만, 마지막 배치가 작으면 일반적으로 같지 않다. 클래스 가중치를 적용한 손실이나 무시한 라벨이 있다면 기본 평균의 분모가 달라질 수 있으므로, 위 `loss × batch_count` 규칙을 그대로 일반화하지 않는다.[^loss]

## 12. 정확도·정밀도·재현율은 기준으로 삼는 샘플이 다르다

지표를 외우기 전에 무엇을 분모로 삼는지 생각해 보자. <strong>정확도(Accuracy)</strong>는 평가한 전체 샘플을 기준으로 예측을 맞힌 비율이다. <strong>정밀도(Precision)</strong>와 <strong>재현율(Recall)</strong>은 클래스 하나를 정해 서로 다른 집합을 기준으로 평가한다.

PNEUMONIA를 찾는 대상으로 정했다면 이를 <strong>양성(Positive)</strong>, 나머지를 <strong>음성(Negative)</strong>이라고 부른다. 양성은 좋은 결과라는 뜻이 아니라, 지금 찾아내려는 대상이라는 뜻이다.

<div class="eval-table">

| 지표 | 먼저 모으는 샘플 | 그중 확인하는 것 |
| --- | --- | --- |
| 정확도 | 평가한 모든 샘플 | 실제 클래스와 예측 클래스가 같은가 |
| PNEUMONIA 정밀도 | 모델이 PNEUMONIA라고 예측한 샘플 | 실제 PNEUMONIA인가 |
| PNEUMONIA 재현율 | 실제 PNEUMONIA인 샘플 | 모델이 PNEUMONIA로 찾아냈는가 |
| PNEUMONIA support | 실제 PNEUMONIA인 샘플 | 비율이 아니라 이 샘플들의 개수 |

</div>

정밀도와 재현율의 분자는 같다. 해당 클래스를 맞힌 개수다. 다른 것은 비교 대상 전체다. 정밀도는 “찾았다고 한 것들이 얼마나 맞는가”, 재현율은 “실제로 있는 것들을 얼마나 찾았는가”를 묻는다.[^metrics]

실제 폐렴 이미지를 모두 찾았다면 재현율은 100%다. 이 상태에서 정상 이미지를 폐렴으로 잘못 부르는 일이 늘어도 실제 폐렴의 개수와 찾아낸 폐렴 개수는 그대로다. 재현율은 변하지 않는다. 대신 폐렴이라고 부른 샘플에 틀린 결과가 더 섞이므로 정밀도가 낮아진다. 두 지표가 다른 실수를 드러내는 이유다.

<strong>클래스 불균형(Class Imbalance)</strong>은 클래스별 샘플 개수가 서로 다른 상태다. 계산 예시로 정상 990개와 불량 10개를 전부 정상이라고 예측하면 정확도는 99%지만 불량 재현율은 0%다. 정확도는 샘플마다 한 번씩 반영하므로 샘플이 많은 클래스가 전체 수치를 더 크게 좌우한다.

`support`는 지금 평가에 들어간 실제 정답 개수다. 자동으로 원본 데이터셋 전체 개수를 뜻하지 않으며, 이미지 777장을 환자 777명으로 바꾸어 해석해서도 안 된다. 환자마다 영상이 한 장이라는 별도 확인이 필요하다.

어느 지표를 우선할지는 예측 이후의 행동과 연결해 정한다. 놓쳤을 때 손해가 크면 재현율이 중요하고, 잘못 잡아서 차단·삭제했을 때 손해가 크면 정밀도가 중요할 수 있다. 분야 이름만으로 우선순위를 고정하지 않는다. 문서 검색에서도 가져온 문서 중 관련 문서의 비율은 정밀도이고, 실제 관련 문서 중 가져온 비율은 재현율이다. 단, 무엇을 관련 문서로 볼지 먼저 정해야 한다.

## 13. 혼동 행렬은 틀린 방향까지 보존한다

<strong>혼동 행렬(Confusion Matrix)</strong>은 실제 클래스와 예측 클래스의 조합별 샘플 개수를 센 표다. scikit-learn에서는 행이 실제 정답, 열이 예측이다. 한 가로줄은 실제 클래스 하나를, 한 세로줄은 예측 클래스 하나를 담당한다.[^cm]

다음 값은 지표를 이해하기 위한 <strong>계산 예시</strong>다. 이번 글에서 실제 모델을 학습해 얻은 성능이 아니다.

<div class="eval-table">

| 실제 정답 ↓ / 예측 → | NORMAL | PNEUMONIA | 행 합계 |
| --- | ---: | ---: | ---: |
| NORMAL | 222 | 48 | 270 |
| PNEUMONIA | 12 | 765 | 777 |
| 열 합계 | 234 | 813 | 1,047 |

</div>

222개는 정상을 정상으로 맞힌 샘플이고, 48개는 정상을 폐렴으로 잘못 예측한 샘플이다. 12개는 폐렴을 정상으로 놓친 샘플이며, 765개는 폐렴을 폐렴으로 맞힌 샘플이다.

폐렴을 양성으로 정했을 때 이 네 칸을 부르는 이름도 함께 정리한다.

<div class="eval-table">

| 이름 | 뜻 | 예시 개수 |
| --- | --- | ---: |
| 참음성(True Negative, TN) | 실제 음성을 음성으로 맞힘 | 222 |
| 거짓양성(False Positive, FP) | 실제 음성을 양성으로 잘못 예측 | 48 |
| 거짓음성(False Negative, FN) | 실제 양성을 음성으로 놓침 | 12 |
| 참양성(True Positive, TP) | 실제 양성을 양성으로 맞힘 | 765 |

</div>

<figure class="eval-figure">
<img src="/image-classification-evaluation/confusion-ko.svg" alt="계산 예시 혼동 행렬의 행은 실제 클래스이고 열은 예측 클래스다. 폐렴 정밀도의 분모는 폐렴 예측 열의 합 813, 재현율의 분모는 실제 폐렴 행의 합 777이다." loading="lazy">
<figcaption>같은 765개를 맞혔어도, 예측한 전체와 실제 존재하는 전체 중 어느 쪽을 기준으로 삼느냐에 따라 지표가 달라진다.</figcaption>
</figure>

행과 열의 클래스 순서가 같으므로 대각선은 정답을 맞힌 개수다. 해당 클래스의 실제 전체 개수는 대각선이 아니라 행의 합이다. 열의 합은 그 클래스로 예측한 개수다.

대상을 확인한 뒤 실제로 계산하면 다음과 같다.

<div class="eval-table">

| 구하려는 값 | 이 예시에서의 계산 | 결과 |
| --- | --- | ---: |
| 전체 정확도 | 맞힌 222개와 765개를 합쳐 전체 1,047개로 나눔 | 약 94.27% |
| NORMAL 정밀도 | 정상으로 맞힌 222개 / 정상 예측 234개 | 약 94.87% |
| NORMAL 재현율 | 정상으로 맞힌 222개 / 실제 정상 270개 | 약 82.22% |
| PNEUMONIA 정밀도 | 폐렴으로 맞힌 765개 / 폐렴 예측 813개 | 약 94.10% |
| PNEUMONIA 재현율 | 폐렴으로 맞힌 765개 / 실제 폐렴 777개 | 약 98.46% |

</div>

전체 정확도가 높다고 두 클래스의 재현율이 비슷한 것은 아니다. 이 표에서는 정상 재현율과 폐렴 재현율 사이에 차이가 있다. 어떤 종류의 실수가 많았는지까지 보려면 전체 정확도와 클래스별 지표를 함께 읽어야 한다.

## 14. labels는 클래스 값과 순서를, target_names는 표시 이름을 정한다

혼동 행렬을 계산하는 최소 입력은 같은 샘플 순서의 `y_true`와 `y_pred`다. 파일 경로나 모델 객체는 계산에 필요하지 않다. 클래스 이름은 사람이 결과를 읽을 때 사용하는 정보다.

`confusion_matrix(y_true, y_pred)`는 기본적으로 두 목록 중 한쪽에라도 등장한 클래스 값을 모아 정렬한다. 모든 클래스가 항상 등장하는 것은 아니므로 출력 표의 범위를 명시할 필요가 있다.[^cm]

~~~python
from sklearn.metrics import confusion_matrix

# 실제 검증 결과와 섞이지 않도록 demo라는 별도 이름을 사용한다.
demo_true = [1, 1, 1]
demo_pred = [1, 1, 1]

print(confusion_matrix(demo_true, demo_pred))
# [[3]]: 등장한 클래스 1만 포함한다.

print(confusion_matrix(demo_true, demo_pred, labels=[0, 1]))
# [[0 0]
#  [0 3]]: 빈 클래스 0도 같은 순서로 포함한다.
~~~

`labels`는 정답 번호를 새로 부여하지 않는다. 이미 사용 중인 클래스 값 가운데 무엇을 어떤 순서로 포함할지 지정한다. ImageFolder의 기본 번호는 클래스가 `K`개일 때 0부터 `K-1`까지이므로 다음처럼 전체 번호 목록을 만들 수 있다.

~~~python
class_names = val_base.classes
class_ids = list(range(len(class_names)))

cm = confusion_matrix(y_true, y_pred, labels=class_ids)
~~~

`len(class_names)`는 클래스 개수다. `range()`는 0부터 그 개수 직전까지의 번호를 제공하고, `list()`가 이를 목록으로 만든다. 두 클래스면 `[0, 1]`, 세 클래스면 `[0, 1, 2]`다. `class_ids`를 만들기만 하고 함수에 전달하지 않으면 표에는 아무 영향도 없다.

`target_names`는 보고서에 표시할 이름이다. `labels=[1, 0]`이라면 이름도 그 순서에 맞게 `["PNEUMONIA", "NORMAL"]`이어야 한다. 이름을 달았다고 함수가 번호와 의미를 자동으로 맞춰주지는 않는다. 원래 정답 값이 문자열인 다른 문제에서는 `labels`에도 그 문자열을 사용할 수 있다.

## 15. 행렬에서 지표를 직접 계산하면 축의 의미가 드러난다

`cm`은 NumPy의 2차원 배열이다. 여기서는 모든 클래스를 포함하며 정규화나 샘플 가중치를 적용하지 않은 <strong>개수 행렬</strong>이다. 그 조건에서 전체 합은 평가한 샘플 개수와 같아야 한다.

~~~python
assert len(y_true) > 0
assert cm.sum() == len(y_true) == len(y_pred) == len(val_dataset)

accuracy_from_cm = cm.diagonal().sum() / cm.sum()
print(f"혼동 행렬에서 계산한 정확도: {accuracy_from_cm:.2%}")

class_recall = {}
for row_idx, row in enumerate(cm):
    class_name = class_names[row_idx]
    support = row.sum()

    # 실제 샘플이 없으면 '전부 틀림'과 구분해 표시한다.
    if support == 0:
        class_recall[class_name] = None
    else:
        class_recall[class_name] = row[row_idx] / support

print(class_recall)
~~~

`cm.diagonal()`은 대각선 원소를 담은 배열이고, 그 합이 전체 정답 개수다. `enumerate(cm)`에서는 행 하나씩을 꺼내므로 `row_idx`는 행 인덱스, `row`는 그 행의 1차원 배열이다.

그런데 `row[row_idx]`에서는 행 인덱스와 같은 숫자를 그 행의 열 인덱스로 사용한다. 행과 열의 클래스 순서가 같으므로 같은 번호를 선택하면 대각선의 정답 개수를 얻는다. 변수명을 `col`이라고 적더라도 반복에서 만들어진 번호의 출처는 행 순회다. 그 출처를 드러내는 `row_idx`가 읽기 쉽다.

`None`은 여기서 측정할 수 없음을 표시하려고 정한 값이다. 실제 샘플이 있는데 모두 틀린 재현율 0과, 실제 샘플이 없어 계산할 수 없는 경우는 다르다. 또한 `labels`에 일부 클래스만 넣거나 `normalize`로 비율 행렬을 만들면 위의 전체 개수 검사를 그대로 적용할 수 없다.

## 16. F1과 분류 보고서는 클래스별로 읽는다

<strong>F1 점수(F1 Score)</strong>는 정밀도와 재현율을 함께 평가하는 지표다. 한쪽만 높아서는 높은 점수를 얻기 어렵도록 두 값의 <strong>조화평균(Harmonic Mean)</strong>을 사용한다. 여기서는 수학적 유도보다 “찾았다고 한 결과가 정확한가”와 “실제 대상을 놓치지 않았는가”를 함께 본다는 목적이 먼저다.

정밀도를 `P`, 재현율을 `R`로 쓰면 F1은 `2 × P × R / (P + R)`이다. 두 값은 0~1 비율로 넣는다. 계산 예시로 정밀도 100%·재현율 20%이면 F1은 약 33.3%이고, 두 지표가 모두 80%이면 F1도 80%다. 두 값이 같다는 것만으로 좋은 것은 아니다. 모두 10%라면 F1도 10%다.

0이 나오는 경우에는 계산식의 모양만 보고 판단하지 않는다. 실제로 있거나 예측한 클래스인데 맞힌 것이 하나도 없다면 F1은 0이다. 실제 정답에도 없고 예측에도 없는 클래스는 F1을 평가할 근거가 없으며 `zero_division` 설정에 따라 처리한다. 앞 절의 개수로 쓰는 `2 × TP / (2 × TP + FP + FN)`에서는 이 차이가 드러난다.[^f1]

`classification_report`는 클래스별 정밀도·재현율·F1·support를 한 번에 정리한다. 기본 반환값은 출력용 문자열이다.[^report]

~~~python
from sklearn.metrics import classification_report

report = classification_report(
    y_true,
    y_pred,
    labels=class_ids,
    target_names=class_names,
    digits=4,
    zero_division=0,
)
print(report)
~~~

`digits=4`는 표시할 소수점 자릿수다. `zero_division=0`은 정의되지 않는 지표를 0으로 표시하겠다는 정책이다. 샘플이나 예측이 없다는 문제를 해결하거나, 모두 실제로 측정된 0으로 바꾸는 설정은 아니다. `support`와 예측 개수도 함께 봐야 한다.

<div class="eval-table">

| 보고서 항목 | 무엇을 평균하는가 | 같은 비중을 갖는 대상 |
| --- | --- | --- |
| `accuracy` | 전체에서 맞힌 샘플의 비율 | 각 평가 샘플 |
| `macro avg` | 클래스별 지표를 같은 비중으로 평균 | 각 클래스 |
| `weighted avg` | 클래스별 지표를 support에 비례해 평균 | 해당 지표에 대한 클래스 비중이 실제 샘플 수를 따름 |

</div>

macro F1은 <strong>클래스별 F1의 평균</strong>이다. macro precision과 macro recall을 먼저 구한 뒤 둘로 F1을 계산하는 방식과 일반적으로 같지 않다. 평균을 무엇에 적용하는지와 연산 순서를 함께 확인해야 한다.

다섯 샘플로 계산을 확인하려면 실제 결과를 덮어쓰지 않고 다음 예시를 사용할 수 있다.

~~~python
demo_true = [0, 1, 1, 0, 1]
demo_pred = [0, 1, 0, 0, 1]
demo_names = ["NORMAL", "PNEUMONIA"]

print(confusion_matrix(demo_true, demo_pred, labels=[0, 1]))
print(classification_report(
    demo_true, demo_pred, labels=[0, 1],
    target_names=demo_names, zero_division=0,
))
~~~

예상 혼동 행렬은 `[[2, 0], [1, 2]]`다. 전체 다섯 샘플 중 네 개를 맞혔으므로 정확도는 80%다.

<div class="eval-table">

| 클래스 | 정밀도 | 재현율 | F1 | support |
| --- | ---: | ---: | ---: | ---: |
| NORMAL | 0.67 | 1.00 | 0.80 | 2 |
| PNEUMONIA | 1.00 | 0.67 | 0.80 | 3 |

</div>

동일한 결과 목록과 클래스 순서를 사용했다면 보고서의 재현율은 혼동 행렬에서 직접 계산한 재현율과 같아야 한다. 표시할 때 반올림된 차이만 허용한다. F1 하나가 높다는 이유만으로 업무에서 가장 좋은 모델이라고 결론 내리지 말고, 놓침과 잘못된 탐지의 비용에 맞춰 개별 지표도 확인한다.

## 17. 평가가 맞는 것과 원본 파일 연결이 맞는 것은 별개다

정확도와 혼동 행렬에는 같은 샘플끼리 짝지어진 정답·예측 목록이 필요하다. 목록 전체의 순서를 두 목록에 동일하게 적용해 바꾸어도 지표는 같다. 그런데 결과 인덱스로 원본 파일까지 찾으려면, 결과의 순서가 Subset의 순서와도 대응해야 한다.

현재 코드가 그 대응을 유지하는 조건은 다음과 같다.

- `val_dataset`이 `val_base`를 직접 감싼 한 단계의 Subset이다.
- `val_loader`가 그 Subset을 사용하며 기본 순차 샘플 선택과 배치 순서를 유지한다.
- `shuffle=False`, `drop_last=False`로 전체 샘플을 한 번씩 처리한다.
- 정답과 예측을 처리한 순서대로 함께 이어 붙인다.
- 수집한 뒤 두 목록이나 인덱스 목록을 따로 정렬·필터링하지 않는다.

이 조건에서 결과 목록의 `i`는 `val_dataset[i]`에 대응한다. 그 샘플의 원본 인덱스는 `val_dataset.indices[i]`에 저장되어 있고, 그 인덱스로 `val_base.samples`를 조회하면 파일 경로와 원본 라벨을 얻는다.

<div class="eval-table">

| 변수 | 자료형 | 어디에서 만들어졌는가 | 조회할 대상 |
| --- | --- | --- | --- |
| `i` | 정수 | 결과 목록을 순회할 때 붙인 인덱스 | `val_dataset.indices[i]` |
| `origin_idx` | 정수 | 선택 목록에 저장된 원본 샘플 인덱스 | `val_base.samples[origin_idx]` |
| `true_label` | 정수 | ImageFolder 라벨이 배치를 거쳐 `y_true`에 저장됨 | `val_base.classes[true_label]` |
| `pred_label` | 정수 | 클래스별 점수의 argmax 결과 | `val_base.classes[pred_label]` |

</div>

예측 번호를 클래스 이름으로 바꾸려면 모델의 0번 출력이 학습 데이터의 클래스 0을 뜻해야 한다. 학습과 검증의 `class_to_idx`가 같아야 하는 이유다. `true_label`이라는 변수 이름 때문에 조회가 되는 것이 아니라, 저장된 값의 출처와 번호 체계가 맞기 때문에 조회할 수 있다.

검증 순서를 섞어도 정답과 예측을 함께 모았다면 지표는 계산할 수 있다. 하지만 그 결과의 `i`를 Subset 내부 인덱스로 바로 사용할 수는 없다. 순서를 바꾸는 설계가 필요하다면, 예측할 때 파일 경로나 원본 인덱스도 함께 반환하고 저장하는 방식으로 바꾼다.[^data]

목록 길이가 같다는 검사는 개수만 확인한다. 원본 라벨과 수집한 라벨이 같다는 검사도 같은 클래스의 서로 다른 이미지가 뒤바뀐 경우를 발견하지 못한다. 파일의 동일성은 샘플 순서가 유지되는 구조 또는 함께 보존한 파일 식별 정보로 확인해야 한다.

## 18. 오분류 이미지는 실제 이름과 예측 이름을 함께 표시한다

<strong>오분류 분석(Error Analysis)</strong>은 틀린 샘플을 살펴보며 어떤 종류의 오류가 있는지 조사하는 작업이다. 순서는 정답과 예측 비교, 오답 선택, 원본 인덱스 조회, 파일 경로 조회, 이미지 읽기, 실제·예측 이름 표시로 나눈다.

`zip(y_true, y_pred)`는 같은 목록 인덱스의 두 값을 한 쌍으로 꺼낸다. `enumerate()`는 그 쌍에 결과 인덱스 `i`를 붙인다. `zip`은 짧은 쪽까지만 순회하므로 먼저 길이를 확인한다.

Pillow의 `Image.open()`은 이미지 파일을 연다. `with ... as img`는 블록이 끝날 때 열린 파일을 정리하도록 쓰는 형태다. Matplotlib의 `imshow()`는 그림에 이미지를 올리고, `title()`은 제목, `show()`는 표시, `close()`는 그림 정리를 담당한다. 이미지 파일과 Matplotlib 그림은 서로 다른 자원이다.[^image]

~~~python
from PIL import Image
import matplotlib.pyplot as plt

assert len(y_true) == len(y_pred) == len(val_dataset)
found_error = False

for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
    if true_label != pred_label:
        origin_idx = val_dataset.indices[i]
        img_path, original_label = val_base.samples[origin_idx]

        # 보조 검사다. 같은 클래스 안의 파일 순서까지 증명하지는 않는다.
        assert true_label == original_label

        true_name = val_base.classes[true_label]
        pred_name = val_base.classes[pred_label]
        print("파일:", img_path)
        print("실제:", true_label, true_name, "/ 예측:", pred_label, pred_name)

        with Image.open(img_path) as img:
            plt.figure()
            # 원본이 흑백이어도 표시 과정에서 색상 지도로 오해하지 않도록 한다.
            plt.imshow(img.convert("RGB"))
            plt.title(f"True: {true_name} | Pred: {pred_name}")
            plt.axis("off")
            plt.show()
            plt.close()

        found_error = True
        break

if not found_error:
    print("이번 평가 목록에는 오분류 샘플이 없습니다.")
~~~

`found_error`는 오분류를 찾았는지 저장하는 불리언 값이다. `break`는 첫 오분류를 처리한 뒤 반복문을 끝낸다. 오분류가 없다면 그림이 나오지 않는 것이 정상이며, 이를 메시지로 알려준다.

`plt.title(true_name)` 다음에 `plt.title(pred_name)`을 호출하면 같은 제목을 두 번째 값으로 바꾼다. 실제·예측을 함께 보고 싶다면 하나의 문자열로 만들어 한 번에 지정해야 한다. 위 코드의 영문 제목은 별도 한글 폰트 설정 없이도 쓸 수 있도록 선택했다. 한글 폰트가 준비되어 있다면 `실제: ... | 예측: ...`으로 바꾸어도 된다.

지금 표시한 것은 <strong>원본 파일에서 읽은 이미지</strong>다. 모델에는 Resize나 Normalize를 거친 텐서가 들어갔으므로 같은 모습이라고 단정하지 않는다. 원본을 보고 오류 원인의 후보를 생각할 수는 있지만, 한 장만 보고 원인을 확정하지 않는다. 다음에는 실제 모델 입력도 함께 확인해야 한다.

## 19. 문법보다 먼저 객체 종류와 검사의 범위를 확인한다

오류가 났을 때는 함수 이름을 바꾸기 전에 현재 변수의 종류와 필요한 작업을 확인한다.

<div class="eval-table">

| 목적 | 맞는 표현 | 구분할 것 |
| --- | --- | --- |
| 선택한 인덱스 조회 | `val_dataset.indices[i]` | 리스트 조회는 `[]`, 함수 호출은 `()` |
| 혼동 행렬 한 칸 조회 | `cm[row_idx, col_idx]` | 배열은 함수처럼 `cm(...)`로 호출하지 않음 |
| Python 리스트 합 | `sum(values)` | 리스트에 무조건 `.sum()`을 붙이지 않음 |
| NumPy 배열·텐서 합 | `values.sum()` | 이 객체에는 합계 메서드가 있음 |
| 클래스 이름 조회 | `val_base.classes[label]` | Subset 인덱스가 아니라 클래스 번호를 사용 |
| 이전 기울기 초기화 | `optimizer.zero_grad()` | `optimizer.no_grad()`라는 용도가 아님 |
| 기울기 기록 끄기 | `with torch.no_grad():` | 초기화·평가 모드 전환과 구분 |
| 조건 위반 시 중단 | `assert cm.sum() == len(y_true)` | 비교식만 적는 것과 검사 실패 시 중단하는 것은 다름 |

</div>

여기서 `values`는 합을 구할 목록·배열·텐서를 가리키는 예시 이름이고, `col_idx`는 열 인덱스다. 속성 이름도 `transform`, `drop_last`, `plt.imshow`, `classification_report`처럼 정확히 써야 한다. `val_loader`와 `val_laoder`는 서로 다른 변수 이름이다.

검사가 통과했다고 확인하지 않은 의미까지 참이 되는 것은 아니다.

<div class="eval-table">

| 통과한 검사 | 확인한 것 | 아직 확인하지 못한 것 |
| --- | --- | --- |
| 클래스별 개수 합이 데이터셋 길이와 같음 | 집계한 총수 | 폴더 정답의 실제 정확성 |
| 학습·검증 인덱스가 겹치지 않음 | 동일 원본 인덱스의 분리 | 복제 파일·같은 환자에 의한 누수 |
| 두 `.samples` 목록이 같음 | 경로·라벨·순서 일치 | 이미지 내용과 정답의 타당성 |
| 이미지와 라벨의 배치 크기가 같음 | 개수 일치 | 모든 이미지와 정답의 올바른 연결 |
| 정답·예측 목록의 길이가 같음 | 결과 개수 일치 | 결과 순서와 원본 파일의 동일성 |
| 전체 클래스 개수 행렬의 합이 결과 수와 같음 | 결과가 전부 집계됨 | 새로운 데이터에서의 성능 |

</div>

`assert`는 실습과 개발 검증에 유용하다. Python을 최적화 옵션으로 실행하면 생략될 수 있으므로, 배포 프로그램의 필수 입력 검사까지 assert에만 맡기지 않는다.[^assert]

<details>
<summary>Rust에서도 같은 인덱스 연결을 확인해 보기</summary>

언어가 달라도 원본 목록과 선택 목록을 따로 두는 관계는 같다. 아래는 신경망 구현이 아니라 선택 목록에서 원본을 찾고 라벨로 이름을 조회하는 부분만 비교한 예시다. Rust의 `usize`는 배열 인덱스에 쓰는 정수형이고, `let`은 변수를 만든다. `&str`는 여기서 문자열을 참조하는 자료형이다.

~~~rust
fn main() {
    let classes = ["NORMAL", "PNEUMONIA"];
    let samples: [(&str, usize); 4] = [
        ("normal01.jpeg", 0),
        ("normal02.jpeg", 0),
        ("pneumonia01.jpeg", 1),
        ("pneumonia02.jpeg", 1),
    ];
    let selected_indices: [usize; 2] = [3, 1];

    let subset_index: usize = 0; // 선택 목록의 첫 샘플
    let original_index = selected_indices[subset_index];
    let (path, label) = samples[original_index];

    // 파일은 원본 인덱스로, 이름은 클래스 번호로 조회한다.
    println!("{} | {}", path, classes[label]);
}
~~~

출력은 `pneumonia02.jpeg | PNEUMONIA`다. Python의 `samples[indices[i]]`와 같은 순서로 조회한다. Python 예제는 리스트의 길이가 바뀔 수 있고, 이 Rust 예제의 배열은 선언할 때 길이가 정해진다는 차이가 있다. 중요한 연결 원리는 두 언어에서 동일하다.

</details>

## 20. 스스로 설명할 수 있어야 하는 범위를 정한다

이번 글에서 필요한 실무 핵심은 함수 이름을 더 외우는 것이 아니라, 데이터를 조사한 뒤 평가 결과를 원본 파일까지 연결하는 것이다. 아래 질문에 코드 없이 먼저 답해 보고, 필요하면 작은 예제로 검증한다.

1. 클래스별 개수를 셀 때 픽셀을 읽지 않아도 되는 이유는 무엇인가?
2. `val_dataset.indices[i]`와 `val_base.classes[true_label]`은 각각 어떤 번호를 입력받는가?
3. 검증 순서를 섞어도 지표를 계산할 수 있는데, 원본 파일 조회에는 문제가 생기는 이유는 무엇인가?
4. 배치 크기가 다를 때 배치 평균을 다시 평균하면 어떤 샘플의 비중이 커지는가?
5. 재현율이 100%인데 정밀도가 낮을 수 있는 이유는 무엇인가?

<details>
<summary>생각한 뒤 확인할 답</summary>

1. 필요한 것은 이미지 내용이 아니라 샘플별 클래스 번호다. `targets`만으로 각 클래스의 개수를 셀 수 있다.
2. 첫 번째는 검증 Subset 안의 샘플 인덱스를 받아 원본 샘플 인덱스를 반환한다. 두 번째는 실제 클래스 번호를 받아 클래스 이름을 반환한다.
3. 지표에는 정답과 예측이 같은 샘플끼리 짝지어지면 된다. 원본 조회에는 결과 인덱스가 Subset 인덱스와도 같아야 하는데, 읽는 순서를 섞으면 그 조건이 깨진다.
4. 작은 배치의 샘플 하나가 더 큰 비중을 갖는다. 각 배치 평균을 동일하게 반영하면서 작은 배치 안에서는 더 적은 샘플이 그 비중을 나누기 때문이다.
5. 실제 대상은 모두 찾았지만, 대상이 아닌 것도 많이 포함했을 수 있다. 실제 대상을 기준으로 하는 재현율과 예측한 대상을 기준으로 하는 정밀도의 분모가 다르다.

</details>

<div class="eval-table">

| 학습 구분 | 지금 할 일 |
| --- | --- |
| 실무 필수 | 원본 구조 조사, 클래스 대응 확인, 분할·전처리·배치 구성, 검증 결과 수집, 지표와 오분류 파일 연결 |
| 원리만 이해 | 손실과 가중치 갱신의 역할, 배치 평균의 가중 방식, 정밀도와 재현율의 기준 집합, F1과 클래스 평균의 의미 |
| 필요할 때 참고 | F1의 수학적 유도, 복잡한 중첩 Subset, 병렬 로딩 설정, 그룹 분할의 세부 구현 |

</div>

종료 기준은 빈 노트북에서 데이터 구조를 조사해 `ImageFolder → Subset → DataLoader`를 연결하고, 정답·예측을 모아 지표와 오분류 파일을 구할 수 있는 것이다. 사용한 각 숫자가 샘플 개수인지, 배치 개수인지, 원본 인덱스인지, 클래스 번호인지 설명할 수 있어야 한다.

이후 실제 프로젝트에서는 모델 입력 텐서 시각화, 여러 에포크의 학습·검증 기록, 최적 모델 저장·불러오기, 독립 테스트와 새 이미지 추론을 이어서 구현한다. 저장할 때는 모델 가중치뿐 아니라 클래스 번호 대응과 전처리 설정도 보존해야 한다. 실제 데이터의 라벨 타당성과 분할의 독립성도 따로 확인한다.

## 21. 전체 예제를 실행하고 확인 범위를 구분한다

본문은 작업별 판단을 설명하기 위해 코드가 나뉘어 있다. 한 파일로 실행하려면 [주석이 있는 전체 Python 예제](/image-classification-evaluation/examples-ko.py)를 사용한다. 한국어판과 [영어판 예제](/image-classification-evaluation/examples-en.py)는 같은 처리를 수행한다.

실행 파일은 별도의 데이터 없이도 연결을 점검하도록 파란 이미지 24장과 빨간 이미지 24장을 생성한다. 기본 분할은 학습 36장·검증 12장, 배치 크기는 5이므로 검증 배치의 샘플 수는 5, 5, 2가 된다. 이는 합성 이미지 예시이며 X-ray 성능 실험이 아니다.

빠르게 실행할 수 있도록 다운로드 예제의 입력 크기는 64×64, 검증 비율은 25%로 정했다. 본문의 224×224와 20%는 각 개념을 설명하는 별도 설정이다. 실제 모델은 자신의 입력 구조와 분할 조건에 맞춰 값을 선택해야 한다.

예제의 작은 CNN은 RGB 입력 `(B, 3, 64, 64)`를 받고, 합성곱과 ReLU로 중간 특징을 계산한 뒤 클래스 수 `K`만큼 점수를 출력한다. <strong>합성곱(Convolution)</strong>은 같은 커널 가중치로 이미지의 여러 부분을 계산하는 연산이다. 중간 채널은 8개와 16개이며, 두 번의 풀링으로 높이·너비를 16×16까지 줄인 뒤 샘플마다 펼쳐 출력층에 전달한다. <strong>풀링(Pooling)</strong>은 여기서 작은 구역의 대표값을 취해 공간 크기를 줄이는 작업이다. 생성자에는 클래스 수를, `forward`에는 실제 이미지 배치를 전달한다. 목적은 평가 흐름의 실행 확인이며, 이 작은 모델이 실제 작업에 가장 적합하다는 뜻은 아니다.

필요한 패키지는 `torch`, `torchvision`, `scikit-learn`, `numpy`, `Pillow`, `matplotlib`이다. 사용 중인 Python 환경에 설치하고, PyTorch와 torchvision은 운영체제·GPU에 맞는 호환 조합을 사용한다. GPU 설치 명령은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 환경에 맞게 선택한다.

아래 명령은 예제 파일을 내려받은 폴더의 <strong>macOS zsh와 Linux bash·zsh</strong>에서 사용할 수 있다. 현재 활성화한 Python 환경의 `python`으로 실행한다.

~~~bash
python examples-ko.py --demo --device cpu --output-dir runs/demo-01
~~~

`--demo`는 합성 데이터 실행, `--device cpu`는 CPU 선택, `--output-dir`은 결과를 남길 새 폴더다. 같은 폴더를 덮어쓰지 않으므로 다시 실행할 때는 새 출력 폴더 이름을 사용한다. 실제 이미지 폴더를 사용하는 형식은 다음과 같다. 경로는 자신의 클래스 폴더 부모 경로로 바꾼다.

~~~bash
python examples-ko.py --data-root ./chest_xray/train --epochs 3 --device auto --output-dir runs/real-01
~~~

`--epochs`는 학습 에포크 수다. `--device auto`는 사용 가능 여부를 확인해 CUDA, MPS, CPU 순으로 선택한다. 이 예제의 기본 학습 횟수는 1이며 `--epochs 0`으로 실행하면 미학습 가중치로 평가·파일 연결 과정만 확인한다.

`--output-dir`을 지정했을 때 아래 파일을 저장한다. 출력 폴더는 `data-root` 바깥에 있는 새 폴더여야 한다. 클래스 부모 폴더 안에 결과 폴더를 만들면 다음 조사에서 새 클래스로 오인할 수 있기 때문이다.

<div class="eval-table">

| 저장 파일 | 내용 |
| --- | --- |
| `summary.json` | 클래스 대응, 전처리·분할 설정, 학습 기록, 검증 지표 |
| `classification-report.txt` | 클래스별 정밀도·재현율·F1·support |
| `predictions.csv` | 결과 인덱스·원본 인덱스·파일 경로·실제와 예측 |
| `errors.csv` | 그중 오분류 샘플만 추린 기록 |
| `first-error.png` | 오분류가 있을 때 실제·예측 이름을 함께 표시한 원본 이미지 |

</div>

합성 데이터를 출력 폴더에 보존하는 경우 `demo-data/`도 생성된다. 출력 폴더를 지정하지 않은 합성 실행에서는 임시 데이터를 실행 뒤 정리한다. 222·48·12·765는 별도 계산 예시로 출력하며, 실행한 모델의 성능과 구분한다.

게시 전 CPU에서 한국어판의 합성 데이터 학습·평가와 영어판의 미학습 가중치 평가를 실행했다. 배치 크기가 다른 경우의 집계, 비정렬 Subset의 원본 파일 연결, 순서를 섞은 검증 로더에 대한 연결 거부, 실제·예측에서 빠진 클래스의 보고서 유지도 확인했다. 실행 검증 환경은 PyTorch 2.15.0+cpu와 torchvision 0.29.2+cpu였다.

이 점검이 확인한 것은 프로그램의 처리와 대응 관계다. 실제 X-ray 학습 성능이나 MPS·CUDA 실행 성능을 측정한 것은 아니다. 예제에는 최적 가중치 저장과 독립 테스트 평가까지 포함되어 있지 않으므로, 실제 작업에서는 앞 절의 남은 구현 항목을 이어서 확인한다.

## 참고 자료

[^folder]: [torchvision ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) · [폴더 조사·라벨·이미지 로딩 구현](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html).
[^path]: [Python pathlib](https://docs.python.org/3/library/pathlib.html).
[^transforms]: [torchvision 이미지 변환](https://docs.pytorch.org/vision/stable/transforms.html) · [ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html).
[^normalize]: [torchvision Normalize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html).
[^weights]: [ResNet18과 가중치별 전처리](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) · [가중치 설정 객체 구현](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/_api.html).
[^split]: [scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) · [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
[^data]: [PyTorch Dataset·Subset·DataLoader](https://docs.pytorch.org/docs/stable/data.html).
[^iteration]: [Python 반복자](https://docs.python.org/3/tutorial/classes.html#iterators) · [iter·next](https://docs.python.org/3/library/functions.html#iter).
[^loss]: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) · [학습·검증 루프](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
[^autograd]: [PyTorch 기울기 기록과 평가 모드](https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation).
[^lists]: [Python 리스트의 append·extend](https://docs.python.org/3/tutorial/datastructures.html).
[^tolist]: [Tensor.tolist](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html) · [Tensor.numpy](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html).
[^metrics]: [scikit-learn 분류 평가 지표](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
[^cm]: [scikit-learn confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).
[^f1]: [scikit-learn f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
[^report]: [scikit-learn classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
[^image]: [Pillow 이미지 파일 관리](https://pillow.readthedocs.io/en/stable/reference/open_files.html) · [Matplotlib title](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html) · [close](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.close.html).
[^assert]: [Python assert 문](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement).
