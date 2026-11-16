+++
title = "흉부 X-ray 분류 실습: 데이터 조사부터 CNN과 전이 학습까지"
description = "ImageFolder의 파일 경로와 라벨을 조사하고, 같은 샘플 대응을 유지하며 분할·전처리·학습·검증을 연결한다. 직접 만든 CNN과 ResNet18을 비교하면서 손실 집계, 클래스별 평가, BatchNorm과 가중치 동결의 차이를 살펴본다."
date = 2026-11-16
slug = "chest-xray-classification-pytorch"

[extra]
toc = true
styles = ["chest-xray-classification-pytorch/article.css"]
+++

이미지 분류 코드를 작성할 때 모델부터 만들면, 정작 모델에 무엇을 넣고 있는지 놓치기 쉽다. 파일 경로를 이미지로 착각하거나, 샘플을 고르는 인덱스와 정답 번호를 혼동할 수 있다. 학습이 끝난 뒤에도 전체 정확도 하나만 보면 어느 클래스를 틀리고 있는지 알기 어렵다.

이번에는 정상과 폐렴으로 나뉜 흉부 X-ray 이미지로 분류 실험을 구성한다. 파일 목록을 조사하는 일부터 시작해, 이미지 한 장이 모델의 점수와 평가 결과로 이어지는 과정을 따라간다. 앞의 [CNN 기초 글](/cnn-convolution-basics/)에서 합성곱을 살펴봤다면, 여기서는 데이터 준비와 실험 비교까지 연결한다.

본문 코드는 각 단계의 판단을 설명하는 예시다. 처음부터 실행할 때는 [전체 Python 예제](/chest-xray-classification-pytorch/examples-ko.py)를 사용할 수 있다. 파일 개수와 모델 성능 표는 기존 실습 기록에 따른 값이며, 새 실행 예제의 점검 범위는 마지막 부분에 따로 적었다.

## 01. 이미지, 정답, 모델 출력의 역할부터 정한다

만들려는 프로그램은 X-ray 이미지 한 장을 받아 정상 또는 폐렴 중 하나를 예측한다. <strong>클래스(Class)</strong>는 구분할 정답의 종류이고, <strong>라벨(Label)</strong>은 이미지마다 붙어 있는 정답이다.

이 실습에서 확인한 클래스 번호는 다음과 같다.

| 클래스 이름 | 라벨 번호 | 의미 |
| --- | ---: | --- |
| NORMAL | 0 | 정상으로 분류된 이미지 |
| PNEUMONIA | 1 | 폐렴으로 분류된 이미지 |

이 번호는 모든 이미지 데이터셋의 공통 규칙이 아니다. 데이터를 읽은 뒤 실제 대응을 확인하고 사용해야 한다.

입력 이미지와 정답 라벨을 묶어 <strong>샘플(Sample)</strong> 하나라고 부르겠다. 모델은 이미지를 받아 클래스별 점수를 계산한다. 정답 라벨은 그 점수를 평가하고 학습하는 기준으로 사용한다. 이미지 안에서 정답 번호를 읽어내는 구조가 아니다.

전체 작업은 다음 세 갈래를 구분해야 한다.

| 데이터 | 사용하는 목적 | 가중치를 바꾸는가 |
| --- | --- | --- |
| 학습 데이터 | 예측과 정답을 비교해 모델을 학습 | 바꿈 |
| 검증 데이터 | 모델 구조와 학습 설정을 비교하고 선택 | 바꾸지 않음 |
| 테스트 데이터 | 선택을 마친 모델의 최종 성능 측정 | 바꾸지 않음 |

<strong>가중치(Weight)</strong>와 편향은 모델이 계산에 사용하는 값이다. 이런 학습 대상 값을 <strong>파라미터(Parameter)</strong>라고 한다. 학습률이나 학습 횟수처럼 사람이 정하는 설정은 <strong>하이퍼파라미터(Hyperparameter)</strong>다.

테스트 결과를 보고 설정을 계속 바꾸면, 테스트 데이터도 모델 선택에 사용하게 된다. 최종 평가의 역할을 남겨 두려면 모델 선택과 테스트를 구분해야 한다.[^cv]

<figure class="xray-figure">
<img src="/chest-xray-classification-pytorch/workflow-ko.svg" alt="기존 train과 val을 합친 5232장을 학습 4185장과 검증 1047장으로 나눈다. 학습은 가중치 갱신, 검증은 모델 선택에 사용하고 test 624장은 선택이 끝난 뒤 평가한다." loading="lazy">
<figcaption>원본 폴더 이름과 새로 정한 학습·검증 역할을 구분한다. 테스트는 모델을 선택한 뒤 사용한다.</figcaption>
</figure>

## 02. 원본 폴더와 이미지 개수를 먼저 조사한다

사용하는 데이터는 [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)다. 데이터가 저장된 최상위 폴더 아래에 `train`, `val`, `test`가 있고, 각 폴더 안에 클래스별 폴더가 있다고 놓고 시작한다. 실제 압축을 푼 경로는 먼저 확인해야 한다.

| 경로 예 | 담긴 내용 |
| --- | --- |
| `chest_xray/train/NORMAL/` | 기존 학습 폴더의 정상 이미지 |
| `chest_xray/train/PNEUMONIA/` | 기존 학습 폴더의 폐렴 이미지 |
| `chest_xray/val/NORMAL/` | 기존 검증 폴더의 정상 이미지 |
| `chest_xray/test/PNEUMONIA/` | 기존 테스트 폴더의 폐렴 이미지 |

실습 기록에서 확인한 파일 수는 다음과 같다. 다른 데이터 사본을 사용한다면 같은 숫자라고 가정하지 말고 다시 센다.

| 원본 구분 | NORMAL | PNEUMONIA | 합계 |
| --- | ---: | ---: | ---: |
| train | 1,341 | 3,875 | 5,216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |
| train + val | 1,349 | 3,883 | 5,232 |

검증 이미지가 16장이라면 한 장의 정답 여부만 바뀌어도 정확도가 6.25%포인트 달라진다. 작은 성능 차이로 모델을 선택하기에는 판단이 흔들리기 쉽다. 이 실습에서는 기존 `train`과 `val`을 합친 다음 새 학습·검증 세트를 만들었다. 기존 `test`는 남겨 두었다.

이 선택은 실습을 위한 분할 변경이다. 공식 평가가 정해 둔 분할을 요구하거나, 원본 분할에 환자·기관 등의 구분이 담겨 있다면 그 조건부터 확인해야 한다. 검증 이미지가 적다는 이유만으로 원본 분할을 무조건 바꾸는 규칙은 아니다.

## 03. ImageFolder에 저장된 정보와 반환되는 이미지를 구분한다

<strong>데이터셋(Dataset)</strong>은 여기서 “샘플이 몇 개인가”, “샘플 인덱스를 주면 어떤 데이터를 가져오는가”를 정하는 객체다. <strong>샘플 인덱스</strong>는 데이터를 선택하는 번호다. 정상·폐렴을 나타내는 정답 번호와는 역할이 다르다.

`ImageFolder`는 클래스별 폴더를 읽는 torchvision의 Dataset 클래스다. 하위 폴더 이름을 클래스 이름으로 사용하므로 파일 경로와 라벨 목록을 직접 수집하는 코드를 줄일 수 있다.[^folder]

아래에서 `data_dir`은 실제로 `train`, `val`, `test`가 들어 있는 폴더다. `Path`는 파일 경로를 다루는 Python 클래스이고, `/` 연산자는 경로를 이어 붙인다. 폴더를 새로 만드는 연산은 아니다.

~~~python
from pathlib import Path
from torchvision import datasets

data_dir = Path("chest_xray")
train_dir = data_dir / "train"
val_dir = data_dir / "val"

original_train = datasets.ImageFolder(train_dir)
original_val = datasets.ImageFolder(val_dir)

print(original_train.class_to_idx)
print(len(original_train), len(original_val))

# 같은 정수가 두 데이터셋에서 같은 클래스를 뜻하는지 확인한다.
assert original_train.class_to_idx == original_val.class_to_idx
~~~

`assert`는 조건이 거짓일 때 오류를 내고 진행을 멈추는 검사 문법이다. 위 검사는 두 폴더의 이미지 수가 같은지 확인하는 것이 아니다.

`original_train` 전체와 그 안에 저장된 일부 정보를 구분해 보면 다음과 같다.

| 표현 | 자료형·내용 | 용도 |
| --- | --- | --- |
| `original_train` | ImageFolder 객체 | 샘플을 읽는 전체 규칙 |
| `.classes` | 클래스 이름을 담은 문자열 리스트 | 이름 확인 |
| `.class_to_idx` | 이름과 정수 번호를 연결한 딕셔너리 | 라벨 의미 확인 |
| `.samples` | `(경로 문자열, 라벨 정수)` 튜플들의 리스트 | 파일 목록 조사 |
| `.targets` | samples와 같은 순서의 라벨 정수 리스트 | 클래스별 개수 계산 |
| `original_train[i]` | 실제 이미지와 라벨 한 쌍 | 샘플 읽기 |

여기서 `i`는 샘플 인덱스다. `original_train.samples[0]`은 파일 경로와 라벨을 꺼낸다. `original_train[0]`은 그 경로의 파일을 열고, 설정된 이미지 변환을 실행한 뒤 이미지와 라벨을 반환한다.

전처리를 지정하지 않았다면 기본 PIL 로더에서는 Pillow 이미지가 나온다. <strong>Pillow 이미지</strong>는 이미지 파일을 다루는 Python 객체다. 이후 숫자 배열로 변환하면 <strong>텐서(Tensor)</strong>, 즉 여러 축을 가진 숫자 데이터를 계산하는 PyTorch 자료형을 받게 된다.

직접 Dataset을 만들 때도 역할은 같다. `__init__`에서 경로·라벨·변환 설정을 보관하고, `__len__`에서 개수를 반환하며, `__getitem__`에서 요청받은 샘플을 만들어 반환한다. 색상 사각형을 즉석에서 만드는 Dataset과 파일을 읽는 ImageFolder는 샘플을 얻는 방법이 다를 뿐이다.[^data]

## 04. 데이터셋을 연결해도 원본 객체의 타입은 바뀌지 않는다

기존 `train`과 `val`을 함께 사용하려고 이미지 파일을 옮길 필요는 없다. 두 Dataset을 순서대로 연결하고, 연결된 샘플 인덱스로 접근하면 된다.

`ConcatDataset`은 여러 Dataset을 순서대로 연결한다. `Subset`은 원본 Dataset에서 선택한 인덱스만 사용하도록 연결한다. 둘 다 파일 복사 도구는 아니다.[^data]

~~~python
from torch.utils.data import ConcatDataset, Subset

merged_dataset = ConcatDataset([original_train, original_val])

print(type(merged_dataset))             # ConcatDataset
print(type(merged_dataset.datasets[0])) # ImageFolder
print(len(merged_dataset))              # 실습 기록: 5232
~~~

`merged_dataset.datasets`는 연결한 원본 객체들을 담은 리스트다. 연결된 객체는 ConcatDataset이지만, 그 안에 있는 두 객체는 여전히 ImageFolder다.

기존 train은 샘플 인덱스 0부터 5,215까지 사용한다. 그 뒤에 val의 16장이 이어진다. 따라서 연결된 인덱스 5,220을 요청하면, 앞쪽 5,216장을 지난 뒤 val의 인덱스 4를 요청한다.

`Subset`에는 한 단계가 더 생긴다. 설명을 위해 선택 목록을 `[5220, 8, 10]`으로 만들면, Subset의 인덱스 0은 원본의 인덱스 5,220을 가리킨다. 아래 선택 목록은 연결 관계를 보여 주는 예시다.

~~~python
example_subset = Subset(merged_dataset, [5220, 8, 10])

# example_subset[0]을 요청하면 다음 접근이 이어진다.
# merged_dataset[5220] → original_val[4]
print(example_subset.indices[0]) # 5220
print(example_subset.dataset is merged_dataset) # True
~~~

<figure class="xray-figure">
<img src="/chest-xray-classification-pytorch/index-map-ko.svg" alt="예시 Subset의 인덱스 0이 연결 데이터셋의 5220을 선택하고, 이 번호는 기존 val의 인덱스 4로 연결된다. samples 조회와 이미지 읽기는 서로 다른 작업이다." loading="lazy">
<figcaption>각 인덱스가 어느 Dataset을 기준으로 만들어졌는지 함께 추적한다.</figcaption>
</figure>

표준 ConcatDataset과 Subset이 ImageFolder의 `.targets`, `.samples`, `.class_to_idx`를 자동으로 합쳐 주지는 않는다. 필요한 정보는 원본 객체에서 가져오거나 별도로 만든다. `train_dataset`이라는 변수 이름만 보고 타입을 판단하지 말고, 무엇을 대입했는지 확인해야 한다.

## 05. 클래스별 개수는 이미지 대신 라벨을 센다

클래스별 이미지 수를 알고 싶다면 필요한 정보는 각 이미지의 라벨이다. 픽셀을 읽거나 텐서로 바꿀 이유가 없다. 이미 저장된 라벨 목록을 사용하면 된다.

아래의 `all_targets`는 합친 데이터 전체의 라벨 리스트다. 나중에 DataLoader가 반환할 한 배치의 `labels`와 구분해서 이름을 붙인다.

~~~python
from collections import Counter

# ConcatDataset에 넣은 순서와 동일하게 연결한다.
all_targets = original_train.targets + original_val.targets
counts = Counter(all_targets)
total = len(all_targets)

for class_name, class_id in original_train.class_to_idx.items():
    count = counts[class_id]
    print(f"{class_name}: {count}장, {count / total:.2%}")
~~~

`Counter`는 같은 값이 몇 번 나오는지 세는 도구다. `class_name`은 클래스 이름 문자열, `class_id`는 해당 클래스의 정수 번호, `count`는 그 번호가 나온 횟수다. 실습 기록의 라벨 목록을 사용하면 정상은 1,349장으로 약 25.78%, 폐렴은 3,883장으로 약 74.22%다.

`.class_to_idx`는 딕셔너리 속성이고, `.items()`는 이름과 번호를 함께 순회하는 메서드다. `.samples`도 저장된 리스트이므로 `.samples()`처럼 호출하지 않는다.

`Counter(merged_dataset)`은 목적이 다르다. Dataset을 순회하면 라벨만 나오는 것이 아니라 파일을 읽은 이미지와 라벨이 함께 나온다. 개수만 필요할 때 이미지까지 읽는 일을 추가한 셈이다.

파일명 몇 개를 조사할 때도 같은 원리로 필요한 정보만 사용한다.

~~~python
normal_id = original_train.class_to_idx["NORMAL"]
filenames = []

for path, target in original_train.samples:
    if target == normal_id:
        filenames.append(Path(path).name)

print(filenames[:5])
~~~

`path`와 `target`은 samples의 튜플에서 꺼낸 경로 문자열과 라벨 정수다. `Path(path).name`은 경로의 마지막 파일명 부분을 가져온다. 이미지를 읽지는 않는다.

<details>
<summary>같은 선택을 리스트 컴프리헨션으로 줄이면</summary>

<strong>리스트 컴프리헨션(List Comprehension)</strong>은 반복, 조건, 결과 수집을 묶어 리스트를 만드는 표현이다.

~~~python
filenames = [
    Path(path).name
    for path, target in original_train.samples
    if target == normal_id
]
~~~

생각할 순서는 바뀌지 않는다. samples를 순회하고, 정상 라벨인 항목을 고르고, 그 파일명을 결과에 넣는다. 처음부터 짧은 문법을 떠올릴 필요는 없다.

</details>

## 06. 층화 분할은 클래스 비율을 유지하는 작업이다

학습과 검증을 나눌 때 바꾸려는 것은 이미지의 내용이 아니다. 어느 샘플을 어느 용도로 사용할지를 정하려는 것이다. Dataset은 인덱스로 샘플을 읽을 수 있으므로, 선택한 인덱스 목록만 보관하면 된다.

<strong>층화 분할(Stratified Split)</strong>은 각 클래스의 구성 비율을 나뉜 데이터에도 가능한 한 유지하는 방법이다. 정상에서 일정 비율을 학습용으로 선택하고, 폐렴에서도 같은 비율을 선택한다. 어느 한 클래스만 더 많이 학습 쪽으로 보내지 않으므로 새 학습·검증의 구성이 원본과 비슷하게 유지된다.

정상 약 26%, 폐렴 약 74%인 데이터를 50:50으로 만드는 작업은 아니다. 이미지 수는 정수이므로 비율이 완전히 같지는 않을 수 있다.[^split]

~~~python
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    range(len(merged_dataset)),
    test_size=0.2,
    stratify=all_targets,
    random_state=42,
)

train_set = Subset(merged_dataset, train_idx)
val_set = Subset(merged_dataset, val_idx)
~~~

| 전달값 | 여기서 뜻하는 것 |
| --- | --- |
| `range(len(merged_dataset))` | 0부터 전체 길이 직전까지의 샘플 인덱스 |
| `test_size=0.2` | 두 번째 반환 결과에 전체의 20%를 배정 |
| `stratify=all_targets` | 각 인덱스에 대응하는 라벨을 참고해 비율 유지 |
| `random_state=42` | 같은 입력·설정에서 분할을 재현하는 난수 설정 |

함수 인자 이름이 `test_size`이지만 여기서는 두 번째 결과를 **검증용**으로 사용한다. 기존 `test` 폴더를 건드리는 코드가 아니다. 또한 이 함수가 항상 인덱스를 반환하는 것도 아니다. 이번에는 인덱스를 입력했으므로 나뉜 인덱스 목록이 나온다.

중요한 조건은 `all_targets[i]`와 `merged_dataset[i]`가 같은 샘플의 라벨이어야 한다는 것이다. 두 원본 Dataset을 연결한 순서와 라벨 리스트를 연결한 순서를 맞춘 이유다.

실습에 사용한 개수와 설정에서는 다음 결과가 나온다.

| 새 구분 | NORMAL | PNEUMONIA | 합계 |
| --- | ---: | ---: | ---: |
| 학습 | 1,079 | 3,106 | 4,185 |
| 검증 | 270 | 777 | 1,047 |

분할 후에는 길이, 중복, 클래스별 개수를 확인한다.

~~~python
assert len(all_targets) == len(merged_dataset)
assert len(train_idx) + len(val_idx) == len(merged_dataset)
assert set(train_idx).isdisjoint(val_idx)
assert set(train_idx) | set(val_idx) == set(range(len(merged_dataset)))

print(Counter(all_targets[i] for i in train_idx))
print(Counter(all_targets[i] for i in val_idx))
~~~

`set`은 중복 없이 값을 모은 집합이다. `isdisjoint`는 두 집합에 공통 원소가 없는지 검사한다. `|`는 두 집합의 원소를 합친다. 위 코드는 학습과 검증이 겹치지 않고 전체 인덱스를 빠짐없이 나눴는지 확인한다.

`val_idx`는 merged_dataset을 기준으로 정한 번호다. `Subset(original_val, val_idx)`처럼 다른 Dataset에 적용하면 같은 번호라도 다른 의미가 되거나 범위를 벗어난다.

PyTorch의 `random_split`은 Dataset을 무작위로 나눠 Subset을 반환하지만 클래스 비율을 의도적으로 유지하지는 않는다. “무작위로 나눈다”와 “라벨 구성을 고려해 나눈다”를 구분해야 한다.

## 07. 이미지가 겹치지 않는 것과 환자가 겹치지 않는 것은 다르다

새로운 환자의 이미지에서도 잘 작동하는지 평가하려면, 같은 환자의 다른 사진이 학습과 검증에 동시에 들어가는지 확인해야 한다. 이미 본 환자의 다른 이미지로 평가하면 확인하려던 조건과 달라질 수 있다.[^cv]

층화 분할이 묻는 것은 “클래스 비율을 유지했는가”다. 환자 그룹 분할이 묻는 것은 “같은 환자의 이미지들을 한쪽에 함께 배정했는가”다. 두 조건은 별도로 확인해야 한다.

가장 좋은 근거는 배포자가 제공한 환자 ID와 이미지의 대응표다. 파일명을 이용하려면 파일명 속 어떤 부분이 환자 ID인지 검증된 규칙이 있어야 한다.

예를 들어 `person1000_bacteria_2931.jpeg`와 `person1000_virus_1681.jpeg`에 같은 문자열이 들어 있다는 사실은 조사할 단서가 된다. 그러나 파일명 일부가 비슷하다는 이유만으로 동일 환자라고 확정하면 안 된다. 다른 종류의 이미지에 대한 파일명 규칙을 X-ray에 그대로 적용해서도 안 된다.

이 실습 기록에서는 신뢰할 수 있는 환자 ID를 확보하지 못해 이미지 단위 층화 분할을 사용했다. 인덱스가 겹치지 않는 것은 검사했지만, 환자가 겹치지 않는다는 조건까지 확인한 결과는 아니다.

따라서 아래 성능은 이 이미지 분할에서의 예비 검증 결과로 해석한다. 새로운 환자에 대한 성능을 주장하려면 환자 정보를 조사하고 분할부터 다시 검증해야 한다. 이 목적에서는 환자 분할이 단순한 성능 개선 옵션이 아니라 평가의 전제다.

## 08. 전처리는 설정한 순간이 아니라 샘플을 읽을 때 실행된다

이미지마다 크기와 자료형이 다르면 배치로 묶어 모델에 넣기 어렵다. 먼저 크기를 맞추고, 텐서로 바꾸고, 계산할 실수 자료형과 값 범위를 정한다.

기본 CNN에서는 다음 전처리를 사용한다.[^transforms]

~~~python
import torch
from torchvision.transforms import v2

def make_basic_transform():
    return v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
~~~

`Compose`는 여러 변환을 적힌 순서로 실행하는 객체다. 위 함수는 변환 설정을 만들어 반환한다. 함수를 호출했다고 모든 이미지가 즉시 변환되는 것은 아니다.

| 변환 | 바뀌는 것 | 주의할 점 |
| --- | --- | --- |
| `Resize((224, 224))` | 높이와 너비를 224로 맞춤 | 원본 종횡비를 보존하는 설정은 아님 |
| `ToImage()` | 이미지를 이미지 텐서로 표현 | 이것만으로 값이 0~1이 되지는 않음 |
| `ToDtype(torch.float32, scale=True)` | 실수 자료형과 값 범위 | 현재 8비트 입력에서는 0~255를 0~1로 변환 |

기본 PIL 로더를 사용하는 ImageFolder는 이미지를 RGB로 변환한다. 흑백처럼 보이는 X-ray도 반환된 텐서는 3채널일 수 있다. 같은 흑백 정보를 RGB로 표현했다고 서로 다른 검사 정보가 세 개 생기는 것은 아니다.[^folder]

변환을 호출하는 것은 ImageFolder다. 표준 Subset에 `train_set.transform = ...`을 대입해도 Subset의 샘플 읽기 동작이 그 속성을 사용하지 않는다. 오류 없이 속성이 붙었다는 사실과 실제 변환이 적용됐다는 사실은 다르다.

앞에서 만든 train_set과 val_set은 같은 merged_dataset을 공유한다. 원본 ImageFolder의 transform을 바꾸면 양쪽에 영향을 준다. 나중에 학습에만 무작위 변환을 넣으려면, 같은 파일을 읽되 서로 다른 전처리를 가진 객체를 만들어야 한다.

이를 위해 아래 함수는 원본 폴더 두 개를 다시 읽는 ImageFolder 객체들을 만들고 연결한다. 파일 순서가 기존 메타데이터와 일치하는지도 검사한다.

~~~python
def make_merged_dataset(transform):
    first = datasets.ImageFolder(train_dir, transform=transform)
    second = datasets.ImageFolder(val_dir, transform=transform)

    # 같은 인덱스가 같은 파일·라벨을 선택하는지 확인한다.
    assert first.samples == original_train.samples
    assert second.samples == original_val.samples
    assert first.class_to_idx == second.class_to_idx

    return ConcatDataset([first, second])

train_source = make_merged_dataset(make_basic_transform())
val_source = make_merged_dataset(make_basic_transform())

train_set = Subset(train_source, train_idx)
val_set = Subset(val_source, val_idx)
~~~

지금은 학습과 검증에 같은 종류의 변환을 썼지만, ImageFolder 객체와 변환 설정은 분리했다. 파일을 복제한 것이 아니라 같은 파일을 읽는 방법을 각각 준비한 것이다.

기존 train 폴더에는 학습용 변환, 기존 val 폴더에는 검증용 변환을 넣는 방법으로는 새 분할을 제대로 표현할 수 없다. 새 학습과 새 검증 모두 기존 두 폴더에서 선택한 이미지를 포함하기 때문이다.

설정이 맞는지는 실제로 한 장을 읽어 확인한다.

~~~python
image, target = train_set[0]

print(type(image), image.shape, image.dtype)
print(image.min().item(), image.max().item(), target)

assert isinstance(image, torch.Tensor)
assert image.shape == (3, 224, 224)
assert image.dtype == torch.float32
assert 0.0 <= image.min().item() <= image.max().item() <= 1.0
~~~

`isinstance`는 해당 타입이나 그 하위 타입인지 검사한다. v2의 이미지 텐서는 Tensor의 하위 타입일 수 있다. `.item()`은 원소 하나를 가진 텐서에서 Python 숫자를 꺼낸다.

이 값 범위 검사는 지금의 기본 전처리에 대한 조건이다. 뒤에서 ImageNet 정규화를 적용하면 음수나 1보다 큰 값도 정상적으로 나올 수 있다.

## 09. DataLoader는 샘플을 배치로 묶는다

Dataset이 샘플 하나를 어떻게 가져올지 정한다면, <strong>데이터로더(DataLoader)</strong>는 샘플을 꺼낼 순서와 묶을 개수를 정한다. <strong>배치(Batch)</strong>는 모델에 한 번에 넣는 샘플 묶음이다.[^data]

~~~python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_set, batch_size=32, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_set, batch_size=32, shuffle=False, num_workers=0
)
~~~

`batch_size=32`는 한 배치에 최대 32장을 넣는 설정이다. `num_workers=0`은 현재 Python 프로세스에서 데이터를 읽는 단순한 시작 설정이다. 환경별 병렬 로딩 문제를 먼저 섞지 않고 동작을 확인하기 좋다.

DataLoader를 만들었다고 학습이 실행되지는 않는다. 실제로 순회할 때 Dataset에 샘플을 요청한다. 일반적인 이 설정에서 학습 데이터 전체를 한 번 순회하는 단위를 <strong>에포크(Epoch)</strong>라고 부른다.

| 대상 | 이미지 수 | 배치 32장의 구성 | 배치 수 |
| --- | ---: | --- | ---: |
| 학습 | 4,185 | 32장 배치 130개 + 25장 배치 1개 | 131 |
| 검증 | 1,047 | 32장 배치 32개 + 23장 배치 1개 | 33 |

`len(train_set)`은 이미지 수, `len(train_loader)`는 현재 설정에서 배치 수다. 마지막 배치를 버리는 `drop_last`의 기본값이 False이므로 남은 이미지도 사용한다.

`shuffle=True`는 기본 샘플 선택 방식에서 전체 샘플 인덱스의 순서를 섞는다. 그 순서대로 배치를 만든다. 매 배치마다 전체 이미지에서 중복을 허용해 새로 32장을 뽑는 설정은 아니다. 이미지와 라벨은 한 쌍으로 함께 이동한다.

학습은 앞 배치에서 가중치를 바꾼 뒤 다음 배치를 계산하므로 순서가 결과에 영향을 줄 수 있다. 검증은 같은 모델 상태와 같은 전처리로 전체 샘플을 평가한다. 현재처럼 이미지별 결과를 독립적으로 계산한다면 순서를 바꿔도 전체 정답 개수는 같아야 한다. 검증의 `shuffle=False`는 결과를 추적하기 쉽게 만드는 선택이다. 무작위 전처리나 배치에 의존하는 계산이 있다면 이 조건부터 다시 살펴야 한다.

<details>
<summary>iter와 next는 각각 무엇을 하는가</summary>

<strong>이터레이터(Iterator)</strong>는 순회가 어디까지 진행됐는지 관리하는 객체다. `iter(train_loader)`로 새 순회를 만들고 `next`로 다음 배치를 꺼낸다.

~~~python
iterator = iter(train_loader)
imgs, labels = next(iterator) # 첫 배치
imgs, labels = next(iterator) # 같은 순회의 다음 배치
~~~

`next`가 매번 새 이터레이터를 만드는 것은 아니다. 반대로 `next(iter(train_loader))`를 매번 실행하면 매번 새 순회를 시작한다. 이전에 꺼낸 배치의 다음으로 자동 이어지지 않는다.

</details>

## 10. 텐서의 축과 이미지·라벨의 대응을 함께 본다

한 배치를 꺼내면 `imgs`는 이미지 텐서, `labels`는 정답 번호 텐서다. <strong>모양(shape)</strong>은 각 축의 크기를 순서대로 적은 것이다.

이미지 배치 모양을 `(B, C, H, W)`로 쓰겠다. B는 현재 배치의 이미지 수, C는 채널 수, H는 높이, W는 너비다.

| 축 번호 | 의미 | 32장 배치의 크기 | 유효한 인덱스 |
| ---: | --- | ---: | --- |
| 0 | 이미지 | 32 | 0~31 |
| 1 | 채널 | 3 | 0~2 |
| 2 | 높이 | 224 | 0~223 |
| 3 | 너비 | 224 | 0~223 |

`imgs`의 모양은 `(32, 3, 224, 224)`, `labels`는 `(32,)`다. 정답에는 각 이미지의 클래스 번호 하나씩만 필요하므로 이미지와 같은 모양이 아니다.

`imgs[i]`는 현재 배치의 이미지 한 장이고, `labels[i]`는 같은 이미지의 정답이다. 이 대응은 “인덱스가 같으니 같을 것”이라고 가정해서 얻은 것이 아니다.

1. Dataset이 이미지와 라벨을 한 쌍으로 반환한다.
2. DataLoader가 샘플 쌍을 같은 순서로 배치에 넣는다.
3. 현재 CNN은 배치의 이미지 순서를 바꾸지 않고 이미지마다 점수를 계산한다.
4. 각 이미지의 점수에서 예측 클래스를 선택한다.

이 처리 순서를 유지하므로 이미지·정답·예측의 같은 배치 인덱스가 연결된다. 라벨만 따로 섞거나 예측만 정렬하면 이 관계가 깨진다.

화면에 RGB 이미지를 표시할 때는 축 순서도 확인한다. PyTorch의 이미지 한 장은 `(채널, 높이, 너비)`이고, Matplotlib은 `(높이, 너비, 채널)`을 사용한다. `permute(1, 2, 0)`은 기존 1번 축, 2번 축, 0번 축 순서로 바꾼다. 픽셀을 평균내는 연산은 아니다.[^tensor]

~~~python
import matplotlib.pyplot as plt

imgs, labels = next(iter(val_loader))
image = imgs[0]
class_name = original_train.classes[labels[0].item()]

plt.imshow(image.permute(1, 2, 0).cpu())
plt.title(class_name)
plt.axis("off")
plt.show()
~~~

위 코드는 기본 전처리의 0~1 이미지에 대한 표시 방법이다. ImageNet 정규화가 적용된 텐서는 표시 전에 정규화를 되돌려야 원래 밝기에 가까워진다.

이미지들을 별도 리스트에 `images.append(image)`로 저장했다면 그 리스트에는 이미지 텐서만 있다. 나중에 `image, label = images[0]`처럼 이미지와 라벨 쌍으로 꺼낼 수 없다. 무엇을 넣었는지에 따라 꺼낼 때의 자료 구조도 달라진다.

## 11. CNN은 필요한 출력에서 거꾸로 설계한다

모델 구조를 정하기 전에 최종적으로 필요한 값을 생각해 보자. 이미지마다 정상 점수와 폐렴 점수가 하나씩 필요하다. 입력이 B장이면 출력 모양은 `(B, 2)`가 되어야 한다.

그 점수를 계산하려면 먼저 이미지에서 특징을 계산하고, 분류층이 받을 형태로 정리해야 한다. <strong>합성곱 신경망(Convolutional Neural Network, CNN)</strong>은 작은 이미지 영역에 같은 가중치를 반복 적용하는 합성곱 층을 사용한다. 계산한 반응값을 공간 구조로 보관한 텐서를 <strong>특징맵(Feature Map)</strong>이라고 한다.

여기서는 합성곱을 두 번 적용하고, 각각의 뒤에서 공간 크기를 줄인다. 마지막에는 모든 특징값을 펼쳐 클래스 점수 두 개를 계산한다.

| 처리 | 역할 | 출력 모양 |
| --- | --- | --- |
| 입력 | 전처리된 이미지 배치 | `(B, 3, 224, 224)` |
| Conv1 + ReLU | 특징 채널 16개 계산 | `(B, 16, 224, 224)` |
| MaxPool | 높이·너비 축소 | `(B, 16, 112, 112)` |
| Conv2 + ReLU | 특징 채널 32개 계산 | `(B, 32, 112, 112)` |
| MaxPool | 높이·너비 축소 | `(B, 32, 56, 56)` |
| Flatten | 이미지별 특징을 한 축으로 펼침 | `(B, 100352)` |
| Linear | 클래스별 점수 계산 | `(B, 2)` |

`Conv2d`는 입력 채널의 작은 영역들을 가중치로 조합한다. 한 출력 채널을 계산할 때 입력 채널별 계산 결과를 합친다. 출력 채널 16개는 이미지가 16장이라는 뜻이 아니다. 각 이미지에서 서로 다른 가중치로 계산한 특징 16종이다.[^conv]

`ReLU`는 음수를 0으로 만들고 양수를 유지하는 활성화 함수다. 이를 끼워 넣으면 여러 층이 단순한 선형 계산의 반복에 머물지 않고 더 다양한 관계를 표현할 수 있다.

`MaxPool2d(2)`는 채널마다 2×2 영역에서 큰 값 하나를 고른다. 여기서는 이동 간격도 2이므로 높이와 너비를 절반으로 줄인다. 이 말은 현재의 짝수 크기와 설정에서 성립한다. 풀링에는 학습할 가중치가 없다.[^pool]

합성곱에서 같은 가중치를 여러 이미지 좌표에 사용한다는 것은, 학습 전후로 가중치가 영원히 고정된다는 뜻은 아니다. 한 번의 계산에서 공유한 가중치는 역전파와 업데이트를 거쳐 바뀐다.

<figure class="xray-figure">
<img src="/chest-xray-classification-pytorch/model-shapes-ko.svg" alt="CustomCNN에서 배치 B를 유지하며 3채널 224×224 입력이 16채널, 32채널 특징맵을 거쳐 100352개 특징과 클래스 점수 두 개로 변한다." loading="lazy">
<figcaption>배치 축 B는 유지한다. 채널 수, 공간 크기, 분류층 입력 개수는 서로 다른 설계 항목이다.</figcaption>
</figure>

### 펼칠 것인가, 채널별로 평균낼 것인가

현재 마지막 특징맵은 `(B, 32, 56, 56)`다. <strong>Flatten</strong>은 채널·높이·너비 축을 하나로 펼친다. 특징값을 평균내거나 버리지 않는다. 배치 축을 유지해야 서로 다른 이미지의 특징이 섞이지 않는다.

아래 비교에서는 `features`에 아직 펼치지 않은 특징맵 `(B, 32, 56, 56)`이 들어 있다고 놓는다.

~~~python
flat_features = torch.flatten(features, start_dim=1)
# (B, 32, 56, 56) → (B, 100352)
~~~

한 이미지의 특징값은 채널 32개마다 56×56개씩 있으므로 100,352개다. `start_dim=1`은 1번 축부터 펼친다는 뜻이다. 4차원 텐서의 축 번호는 0~3이므로 `flatten(x, 2, 4)`처럼 존재하지 않는 축을 지정하면 안 된다. Flatten의 `end_dim`은 포함되는 마지막 축 번호다.[^tensor]

다른 방법으로 <strong>전역 평균 풀링(Global Average Pooling, GAP)</strong>이 있다. 이미지별·채널별로 공간 전체를 평균내 채널마다 값 하나를 남긴다.

~~~python
gap = torch.nn.AdaptiveAvgPool2d((1, 1))
summary = gap(features) # (B, 32, 1, 1)
summary = torch.flatten(summary, 1) # (B, 32)
~~~

| 방법 | 분류층에 남기는 정보 | 대가 |
| --- | --- | --- |
| Flatten | 각 채널의 각 좌표에 대응하는 특징값 | 분류층 입력과 가중치 수가 커짐 |
| GAP | 채널마다 공간 반응의 평균 | 어느 좌표에서 반응했는지 직접 남기지 않음 |

GAP 뒤에 Flatten을 사용할 수도 있다. 둘을 반드시 하나만 골라야 하는 것은 아니다. 또한 `AdaptiveAvgPool2d((7, 7))`는 채널당 49개 값을 남기므로, 채널당 하나를 남기는 GAP와 다르다.[^gap]

이번 모델은 Flatten 뒤에 출력 2개인 Linear를 사용한다. 마지막 층의 가중치는 출력 하나당 100,352개이고, 출력마다 편향 하나가 추가되어 총 200,706개다. 모델 전체는 205,794개다. 이 규모를 확인한 뒤 판단해야지, Flatten을 썼다는 이유만으로 무조건 너무 큰 모델이라고 결론 내릴 수는 없다.

## 12. 층을 만드는 코드와 실제 계산 순서를 구분한다

PyTorch 모델은 `nn.Module`을 바탕으로 만든다. `__init__`에서는 사용할 층을 만들고 보관한다. `forward`에서는 입력 텐서에 그 층들을 어떤 순서로 적용할지 정한다. 층을 선언한 순서만으로 계산 순서가 자동 결정되지는 않는다.

아래 모델의 생성자 인자 `num_classes`는 출력 점수 개수다. 반면 `forward`가 받는 `x`는 실제 이미지 배치 텐서다.

~~~python
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
~~~

`self`는 현재 모델 객체를 가리킨다. `self.fc`는 그 객체에 보관한 분류층이다. `self.pool = nn.MaxPool2d(2)`에서 2는 풀링 영역의 크기 설정이고, `self.pool(x)`의 x는 실제 처리할 텐서다. `self.pool(2)`는 크기를 다시 정하는 호출이 아니다.

모델 연결을 확인할 때는 가짜 입력을 넣어 출력 모양부터 검사한다. `torch.randn`은 지정한 모양으로 임의의 실수 텐서를 만든다. 아래 `torch.no_grad()`는 이 확인 과정에서 역전파용 연산 기록을 만들지 않게 한다.

~~~python
shape_model = CustomCNN(num_classes=2)

with torch.no_grad():
    output = shape_model(torch.randn(4, 3, 224, 224))

assert output.shape == (4, 2)
print(sum(p.numel() for p in shape_model.parameters())) # 205794
~~~

`parameters()`는 모델의 파라미터들을 제공하고, `numel()`은 텐서 원소 수를 반환한다. 이 검사는 층 연결과 모양을 확인한다. 실제 이미지의 라벨 대응이나 분류 성능까지 확인한 것은 아니다.

<details>
<summary>자주 생기는 호출·이름 오류</summary>

| 잘못된 표현 | 올바른 표현 | 이유 |
| --- | --- | --- |
| `super.__init__()` | `super().__init__()` | super를 호출해 부모 초기화 사용 |
| `F.ReLU(x)` | `F.relu(x)` | functional의 함수 이름은 소문자 |
| `nn.Fc(...)` | `nn.Linear(...)` | 완전연결층 클래스 이름 |
| `self.pool(2)` | `self.pool(x)` | 설정값이 아니라 텐서 전달 |
| `fc(x)` | `self.fc(x)` | 모델에 보관한 층 사용 |

</details>

`model = CustomCNN(2)`를 다시 실행하면 새 초기 가중치를 가진 모델이 생긴다. 이미 학습한 모델을 유지하려면 확인용 모델의 이름을 따로 쓰거나, 어느 셀을 다시 실행하는지 확인해야 한다.

## 13. 점수, 정답 번호, 예측 번호는 서로 다른 값이다

모델이 계산하는 원시 점수를 <strong>로짓(Logits)</strong>이라고 한다. 현재 모델은 이미지마다 클래스 점수 두 개를 반환한다.

| 변수 | 모양 | 자료형·내용 |
| --- | --- | --- |
| `imgs` | `(B, 3, 224, 224)` | 입력 이미지 실수 텐서 |
| `logits` | `(B, 2)` | 정상·폐렴 점수 실수 텐서 |
| `labels` | `(B,)` | 실제 정답 번호 정수 텐서 |
| `preds` | `(B,)` | 점수를 비교해 고른 예측 번호 |

`logits[i, 0]`은 배치 인덱스 i인 이미지의 정상 점수다. 첫 번째 인덱스는 이미지를 고르고, 두 번째 인덱스는 클래스를 고른다.

앞에서 입출력을 확인한 `shape_model`과 DataLoader에서 꺼낸 같은 배치 `imgs`, `labels`로 구조를 확인할 수 있다. 이 모델은 아직 학습하지 않았다.

~~~python
logits = shape_model(imgs)
preds = logits.argmax(dim=1)
~~~

`argmax`는 가장 큰 값의 인덱스를 반환한다. `dim=1`은 클래스 점수 축을 뜻한다. 점수가 `[1.5, 3.2]`라면 결과는 큰 값 3.2가 아니라 클래스 번호 1이다.

로짓에는 음수나 1보다 큰 값도 나올 수 있다. 확률이 필요하면 <strong>소프트맥스(Softmax)</strong>로 클래스 점수를 양수이며 합이 1인 값으로 변환한다. 정답 클래스를 선택하는 데에는 순서가 유지되므로 로짓에서 바로 argmax를 적용해도 된다. 점수 전체에 같은 수를 더해도 상대적 차이는 같으므로, 정답 점수 하나의 절댓값만 보고 확신을 판단하면 안 된다.

### 손실 함수는 출력과 정답의 의미에 맞춰 고른다

<strong>손실(Loss)</strong>은 예측이 정답에 얼마나 맞지 않는지 학습 목적에 맞게 계산한 값이다. 이번 출력은 `(B, 2)`의 점수이고, 정답은 `(B,)`의 클래스 번호다. 이 조합에는 `CrossEntropyLoss`, 즉 <strong>교차 엔트로피 손실</strong>을 사용할 수 있다.[^ce]

이 함수는 점수와 정답을 같은 모양으로 만들어 원소별로 빼는 함수가 아니다. 이미지마다 클래스 점수들을 받고, 정답 번호가 가리키는 클래스에 얼마나 높은 확률을 부여했는지 평가한다. 정답 번호 하나가 두 점수 중 어떤 것이 정답인지 알려주므로 두 입력의 모양이 달라도 된다.

~~~python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
~~~

현재의 정수 라벨 방식에서는 다음 조건을 맞춘다.

- logits는 Softmax를 적용하기 전의 실수 점수다.
- labels는 `torch.long` 정수 텐서이며 값은 0 또는 1이다.
- 두 텐서의 이미지 순서, 배치 개수, 연산 장치가 같아야 한다.

학습 손실에는 argmax로 고른 `preds`를 넣지 않는다. 클래스 번호만 남기면 두 클래스 사이의 점수 차이에 관한 정보가 사라진다. CrossEntropyLoss 앞에 Softmax를 따로 적용하지도 않는다. 이 손실 함수가 필요한 계산을 수치적으로 안정적인 형태로 처리한다.

기본 `reduction="mean"`은 이미지별 손실을 배치 평균 하나로 반환한다. 손실의 모양은 `()`인 0차원 텐서다. `reduction="none"`이면 이미지별 손실 `(B,)`를, `"sum"`이면 합계 하나를 얻는다.

클래스가 두 개일 때 다른 출력 설계도 가능하다. 예를 들어 점수 하나로 폐렴 여부를 표현하면 `BCEWithLogitsLoss`를 사용할 수 있다. 이때 정답도 출력과 같은 모양의 실수 0·1로 맞춰야 한다. 현재의 점수 두 개 구조에서 손실 함수 이름만 바꾸는 것은 올바른 변경이 아니다.[^bce]

## 14. train, eval, no_grad는 서로 다른 기능이다

학습 여부를 이해하려면 모델의 동작 모드, 기울기 계산, 실제 파라미터 변경을 나누어 보아야 한다.

<strong>기울기(Gradient)</strong>는 파라미터를 조금 바꿀 때 손실이 어느 방향으로 얼마나 변하는지 나타내는 정보다. <strong>역전파(Backpropagation)</strong>는 모델 계산의 연결을 따라 이 기울기를 구한다. <strong>옵티마이저(Optimizer)</strong>는 기울기와 자체 규칙을 이용해 파라미터를 변경한다.

| 코드·설정 | 하는 일 | 이것만으로 하지 않는 일 |
| --- | --- | --- |
| `model.train()` | 모드에 따라 달라지는 층을 학습 방식으로 설정 | 학습 반복문 실행 |
| `model.eval()` | 그런 층을 평가 방식으로 설정 | 기울기 기록 중단 |
| `torch.no_grad()` | 블록 안의 역전파용 연산 기록 중단 | eval 모드로 변경 |
| `param.requires_grad` | 파라미터에 대한 기울기가 필요한지 지정 | 파라미터 업데이트 |
| `loss.backward()` | 기울기 계산 | 가중치 직접 변경 |
| `optimizer.step()` | 관리 중인 파라미터 업데이트 | 다른 모델의 파라미터 업데이트 |

**eval 상태에서도 기울기를 기록하고 역전파할 수 있다.** 반대로 train을 호출해도 실제 학습 반복문을 실행하지 않으면 가중치를 학습한 것이 아니다.[^autograd]

현재 CustomCNN에는 학습·평가 모드에 따라 계산이 달라지는 BatchNorm이나 Dropout이 없다. 그렇더라도 학습에서는 train, 검증에서는 eval을 명시하는 습관이 필요하다. 뒤에서 ResNet18로 바꾸면 이 구분이 실제 동작에 영향을 준다.

`@torch.no_grad()`를 함수 바로 위에 적으면 그 함수 실행 전체에서 기울기 기록을 끈다. 이런 표기를 <strong>데코레이터(Decorator)</strong>라고 한다. 평가 함수에 사용할 수 있지만 일반적인 학습 함수 전체에 붙이면 역전파에 필요한 기록이 만들어지지 않는다.

## 15. 한 배치를 학습하는 순서를 만든다

한 배치 학습에는 현재 가중치로 점수를 계산하고, 정답과 비교해 손실을 계산하고, 기울기를 구한 뒤, 가중치를 바꾸는 작업이 필요하다. 이전 배치의 기울기가 남아 있다면 이번 계산에 누적되므로 먼저 비운다.

모델과 텐서는 같은 <strong>연산 장치(Device)</strong>에 있어야 한다. NVIDIA GPU 환경에서는 CUDA, Apple Silicon에서는 사용 가능한 MPS, 그 외에는 CPU를 선택할 수 있다. 아래 코드는 현재 Python 환경에서 실제 사용 가능한 장치를 확인한다.[^device]

~~~python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = CustomCNN(2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
~~~

`.to(device)`는 모델이나 텐서를 지정 장치로 옮긴다. 모델을 옮긴 뒤 그 모델의 파라미터로 옵티마이저를 만든다. Adam은 옵티마이저의 한 종류이고, `lr`은 업데이트 크기에 영향을 주는 <strong>학습률(Learning Rate)</strong>이다. 0.001은 시작 설정이며 모든 문제의 최적값은 아니다.

모델과 옵티마이저는 한 실험을 시작할 때 만든다. 매 에포크마다 모델을 새로 만들면 앞에서 학습한 가중치가 사라진다.

~~~python
model.train()
total_loss = 0.0
total_samples = 0

for imgs, labels in train_loader:
    imgs = imgs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad() # 이전 배치의 기울기만 비운다.
    logits = model(imgs)
    loss = loss_fn(logits, labels)

    loss.backward()      # 파라미터에 대한 손실의 기울기를 계산한다.
    optimizer.step()     # 그 기울기로 실제 파라미터를 바꾼다.

    total_loss += loss.item() * len(labels)
    total_samples += len(labels)

print("평균 학습 손실:", total_loss / total_samples)
~~~

`zero_grad`는 가중치를 초기화하지 않는다. `backward`는 기울기를 계산하고, `step`이 실제 업데이트를 한다. `loss.item()`은 기록할 Python 숫자를 꺼내는 것이므로, `loss = loss.item()`으로 바꾼 뒤 그 숫자로 역전파하려고 하면 안 된다.

이 학습 루프에는 모델을 새로 만드는 줄이 없다. 다음 배치는 바로 앞 배치에서 갱신한 가중치로 계산한다.

## 16. 배치 평균을 다시 이미지 수와 곱하는 이유

우리가 기록하려는 것은 모든 이미지가 같은 비중으로 반영된 평균 손실이다. 손실 함수가 바로 주는 값은 각 배치의 평균이다. 이 둘은 배치 크기가 다를 때 같은 방식으로 합칠 수 없다.

큰 배치와 작은 배치의 평균을 똑같이 한 번씩 더하고 배치 수로 나누면, 작은 배치도 큰 배치와 같은 영향력을 갖는다. 그러면 작은 배치에 들어간 이미지 한 장의 영향력이 상대적으로 커진다.

동등하게 취급할 대상이 이미지 한 장이라면, 배치 평균을 먼저 그 배치의 손실 합계로 되돌려야 한다. 모든 합계를 모은 뒤 실제 처리한 이미지 수로 나누면 각 이미지가 같은 비중을 갖는다.

~~~python
total_loss += loss.item() * len(labels)
total_samples += len(labels)
epoch_loss = total_loss / total_samples
~~~

여기서 `len(labels)`는 현재 배치에 들어 있는 이미지 수다. 전체 데이터 수나 배치 수가 아니다. 마지막 학습 배치는 25장이므로 그 배치에서는 25를 곱한다.

이 집계가 맞는 조건도 확인해야 한다. 현재처럼 클래스 가중치와 제외 대상 없이 기본 CrossEntropyLoss의 평균을 사용하면, 배치 평균에 이미지 수를 곱해 손실 합계를 얻는다. 클래스 가중치나 `ignore_index`를 사용하면 평균의 분모가 달라질 수 있으므로 집계 방식도 다시 정해야 한다.[^ce]

또한 학습 손실은 배치마다 가중치가 바뀌는 도중에 기록한 값이다. 마지막 모델 하나로 전체 학습 데이터를 다시 평가한 손실과는 다르다. 손실이 0.3166이라고 해서 오답률이 31.66%라는 뜻도 아니다.

## 17. 검증에서는 가중치를 유지하며 결과를 모은다

검증은 학습에 쓰지 않은 이미지에서 현재 모델을 평가하는 작업이다. 모델을 eval 모드로 바꾸고, 기울기 기록을 끈 뒤, 각 배치의 정답과 예측을 같은 순서로 모은다.

~~~python
all_labels = []
all_preds = []

model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        logits = model(imgs.to(device))
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())

assert len(all_labels) == len(all_preds) == len(val_set)
~~~

이 코드에서 labels는 CPU에 그대로 두었다. 모델 계산에 넣는 이미지는 device로 옮기고, 그 장치에서 나온 예측은 CPU로 가져와 리스트로 바꾼다. 검증 손실도 계산하려면 정답 텐서를 logits와 같은 장치로 옮겨 손실 함수에 전달한다.

`all_labels`와 `all_preds`는 이미지마다 정수 하나가 들어 있는 전체 검증 목록이다. 같은 리스트 인덱스끼리 같은 이미지의 정답과 예측이어야 한다.

`append`는 전달한 객체 전체를 원소 하나로 추가하고, `extend`는 그 안의 원소들을 각각 이어 붙인다.

| 시작 리스트 | 실행 | 결과 |
| --- | --- | --- |
| `[0, 1]` | `append([1, 0])` | `[0, 1, [1, 0]]` |
| `[0, 1]` | `extend([1, 0])` | `[0, 1, 1, 0]` |

배치별 리스트를 한 목록으로 모으려는 목적이므로 extend를 쓴다. 두 메서드는 기존 리스트를 변경한다. 반환값을 다시 리스트 변수에 대입하는 방식으로 사용하지 않는다. 전체 결과 목록의 초기화도 배치 반복문 밖에 있어야 이전 배치가 보존된다.

### 클래스별로 직접 세어 보면 지표의 의미가 드러난다

실제 정상 이미지 중 정상으로 맞힌 개수를 알고 싶다면 두 조건이 필요하다. 정답이 정상인지 확인하고, 그중 예측이 맞았는지 확인한다.

조건을 각 원소에 대해 참·거짓으로 표시한 텐서를 <strong>불리언 마스크(Boolean Mask)</strong>라고 한다. 아래에서는 같은 검증 배치의 labels를 preds와 같은 장치로 맞춘 뒤 비교한다.

~~~python
labels = labels.to(preds.device) # 비교할 두 텐서의 장치를 맞춘다.
normal_mask = labels == 0
correct_mask = preds == labels

normal_total = normal_mask.sum().item()
normal_correct = (normal_mask & correct_mask).sum().item()
~~~

`labels == 0`은 라벨을 바꾸지 않는다. 각 정답이 정상인지 표시한다. `&`는 같은 인덱스의 두 조건이 모두 참인지 계산한다. `sum`은 True를 1, False를 0으로 세므로 이미지 개수를 얻는다.

전체 검증에서 누적하려면 클래스별 누적 개수를 반복문 밖에서 만들고 배치마다 더해야 한다. GPU에서 이미지 한 장씩 Python의 `if`와 `.item()`을 반복하기보다, 이런 텐서 단위 계산으로 묶으면 불필요한 장치 간 동기화를 줄일 수 있다.

## 18. 혼동 행렬에서 지표의 분모를 확인한다

<strong>혼동 행렬(Confusion Matrix)</strong>은 실제 정답과 예측의 조합별 개수를 담은 표다. scikit-learn에서는 행이 실제 정답이고, 열이 모델 예측이다.[^metrics]

~~~python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

print(cm)
print("정확도:", accuracy_score(all_labels, all_preds))
print(classification_report(
    all_labels,
    all_preds,
    labels=[0, 1],
    target_names=["NORMAL", "PNEUMONIA"],
    digits=4,
    zero_division=0,
))
~~~

`labels=[0, 1]`은 표의 행·열에 배치할 클래스 순서다. `target_names`도 그 순서에 맞춘 출력 이름이다. 라벨 번호 자체를 다시 부여하는 코드가 아니다.

아래는 뒤에서 비교할 Partial 모델의 검증 기록이다. 수치의 기준을 보기 위해 먼저 혼동 행렬로 펼쳐 본다.

| 실제 정답 ＼ 모델 예측 | NORMAL | PNEUMONIA | 실제 개수 |
| --- | ---: | ---: | ---: |
| NORMAL | 252 | 18 | 270 |
| PNEUMONIA | 16 | 761 | 777 |
| 예측 개수 | 268 | 779 | 1,047 |

<figure class="xray-figure">
<img src="/chest-xray-classification-pytorch/metrics-ko.svg" alt="Partial 검증 혼동 행렬에서 정상 정답 252장, 폐렴 정답 761장을 기준으로 정확도와 클래스별 재현율·정밀도의 분모를 구분한다." loading="lazy">
<figcaption>재현율은 실제 정답을 기준으로, 정밀도는 모델이 예측한 클래스를 기준으로 센다.</figcaption>
</figure>

| 지표 | 먼저 묻는 질문 | 위 기록에서의 계산 |
| --- | --- | --- |
| 정확도(Accuracy) | 전체 이미지 중 몇 장을 맞혔는가 | 1,013 / 1,047 |
| 정상 재현율(Recall) | 실제 정상 중 몇 장을 찾았는가 | 252 / 270 |
| 폐렴 재현율(Recall) | 실제 폐렴 중 몇 장을 찾았는가 | 761 / 777 |
| 폐렴 정밀도(Precision) | 폐렴이라고 예측한 것 중 실제 폐렴은 몇 장인가 | 761 / 779 |

재현율과 정밀도는 분자에 같은 개수를 사용할 수 있지만 분모가 다르다. 먼저 어느 집단을 기준으로 판단하는지 정해야 한다.

<strong>F1 점수</strong>는 정밀도와 재현율을 함께 반영한다. 둘 중 하나가 낮으면 높은 점수를 얻기 어렵다. 보고서의 `macro avg`는 클래스마다 같은 비중으로 평균을 내고, `weighted avg`는 실제 클래스 이미지 수에 비례해 평균을 낸다. `support`는 실제 정답 개수이며 모델이 그 클래스로 예측한 개수가 아니다.

`zero_division=0`은 분모가 0이어서 계산할 수 없는 지표를 0으로 표시한다. 이런 상황이 발생하지 않았다는 뜻은 아니므로 어떤 클래스를 한 번도 예측하지 않았는지 함께 확인해야 한다.[^report]

이 검증 세트의 폐렴 비율은 약 74%다. 전부 폐렴이라고만 예측해도 정확도는 약 74%지만 정상 재현율은 0%다. 정확도와 클래스별 재현율·정밀도를 함께 보는 이유다. 층화 분할은 이런 편향을 자동으로 없애 주지 않는다.

두 클래스가 모두 있는 데이터에서 정상 재현율 1.0, 폐렴 재현율 0.0이 나왔다면 모든 이미지를 정상으로 예측한 것이다. 이것만으로 손실 함수나 지표 코드가 틀렸다고 단정할 수 없다. 실습 기록에서도 실제 학습 전에 새 모델을 평가한 경우가 있었다. 먼저 학습 루프를 실행했는지, 학습 뒤 모델을 새로 만들지는 않았는지, 옵티마이저가 맞는 모델을 관리하는지 확인한다.

## 19. ResNet18에서는 가중치와 전처리를 함께 가져온다

직접 만든 CNN은 무작위 초기 가중치에서 시작했다. 다른 데이터에서 먼저 학습한 가중치를 출발점으로 사용할 수도 있다. 먼저 가중치를 학습하는 것을 <strong>사전 학습(Pretraining)</strong>, 그 가중치를 새 문제에 활용하는 것을 <strong>전이 학습(Transfer Learning)</strong>이라고 한다. 가져온 가중치를 새 데이터로 추가 학습하는 과정은 <strong>미세 조정(Fine-Tuning)</strong>이다.

ResNet18은 층의 입력을 뒤쪽 계산에 더하는 잔차 연결을 가진 CNN이다. 잔차 연결 자체는 [이전 글](/cnn-residual-projection/)에서 설명했다. 여기서는 내부 구조를 다시 구현하지 않고 torchvision이 제공하는 모델과 ImageNet 사전 학습 가중치를 사용한다.

~~~python
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
resnet_model = resnet18(weights=weights)

print(resnet_model.fc)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
~~~

`fc`는 마지막 분류층이다. 원래는 특징값 512개를 받아 1,000개 클래스 점수를 계산한다. 현재 필요한 점수는 2개이므로 분류층을 교체한다. `in_features`는 그 층이 받는 특징값 개수다.

새 Linear는 새 초기 가중치를 가진다. 사전 학습 모델을 가져왔다고 정상·폐렴을 구분하는 이 분류층까지 이미 학습된 것은 아니다.[^resnet]

### 입력값의 기준도 맞춘다

사전 학습된 가중치는 특정한 전처리를 거친 입력으로 학습됐다. 같은 픽셀이라도 입력의 값 범위와 기준을 바꾸면 뒤쪽 특징 계산이 달라질 수 있다. 가중치와 함께 제공되는 전처리부터 확인한다.

~~~python
resnet_transform = weights.transforms()
print(resnet_transform)
~~~

확인한 ResNet18 기본 가중치의 평가 전처리는 다음 순서다.

| 순서 | 처리 | 결과 |
| ---: | --- | --- |
| 1 | 짧은 변을 256으로 조정 | 종횡비를 유지한 크기 조정 |
| 2 | 중앙 224×224를 자름 | 모델 입력 공간 크기 고정 |
| 3 | 실수형 변환과 값 범위 조정 | 채널별 정규화 전의 0~1 값 |
| 4 | 채널 평균을 빼고 표준편차로 나눔 | 사전 학습 입력 기준에 맞춘 값 |

마지막 처리를 <strong>정규화(Normalization)</strong>라고 한다. 이 설정의 채널 평균은 `[0.485, 0.456, 0.406]`, 표준편차는 `[0.229, 0.224, 0.225]`다. 새 X-ray 검증 데이터에서 계산한 수치가 아니라 제공된 가중치의 전처리 설정이다. 정규화 뒤에는 음수나 1보다 큰 값도 나올 수 있다.[^resnet]

`weights.transforms()`는 공식 평가용 전처리다. 빠른 기준 실험에서는 학습에도 사용할 수 있지만, 무작위 증강까지 포함한 유일한 학습 방법은 아니다. 중앙을 자르면 바깥 이미지 정보가 사라질 수 있으므로 X-ray에 적합한지는 별도로 검증해야 한다.

모델 비교에서는 샘플을 다시 무작위로 나누지 않고 기존 인덱스를 재사용한다. 앞의 `make_merged_dataset`은 파일 순서가 원본과 같은지 검사하므로 전처리를 바꾸면서 샘플 대응은 유지할 수 있다.

~~~python
res_train_source = make_merged_dataset(weights.transforms())
res_val_source = make_merged_dataset(weights.transforms())

res_train_set = Subset(res_train_source, train_idx)
res_val_set = Subset(res_val_source, val_idx)

res_train_loader = DataLoader(
    res_train_set, batch_size=32, shuffle=True, num_workers=0
)
res_val_loader = DataLoader(
    res_val_set, batch_size=32, shuffle=False, num_workers=0
)
~~~

인덱스가 같아도 원본 파일 순서가 달라지면 다른 샘플을 고른다. 파일 목록과 라벨 대응을 함께 검사하는 이유다.

## 20. Frozen, Partial, Full은 학습할 가중치 범위를 정한다

전이 학습에서 어디까지 바꿀지는 실험 조건이다. 이 글에서는 다음 세 가지를 비교한다.

| 이름 | 학습할 가중치 | 고정할 가중치 |
| --- | --- | --- |
| Frozen | 새 분류층 fc | 기존 특징 추출층 전체 |
| Partial | 마지막 특징 블록 layer4와 fc | 앞쪽 특징 추출층 |
| Full | 특징 추출층 전체와 fc | 없음 |

Full도 사전 학습 가중치에서 출발한다. 무작위 가중치로 처음부터 학습한다는 뜻이 아니다. Partial과 Full 모두 미세 조정이다.

가중치의 기울기가 필요한지는 `requires_grad`로 설정한다. Partial을 예로 들면 이름이 `layer4.` 또는 `fc.`로 시작하는 파라미터만 선택한다.

~~~python
partial_model = resnet18(weights=weights)
partial_model.fc = nn.Linear(partial_model.fc.in_features, 2)

for name, param in partial_model.named_parameters():
    param.requires_grad = (
        name.startswith("layer4.") or name.startswith("fc.")
    )

partial_model = partial_model.to(device)
trainable_params = [
    param
    for param in partial_model.parameters()
    if param.requires_grad
]
partial_optimizer = torch.optim.Adam(trainable_params, lr=0.0001)
~~~

`named_parameters()`는 이름 문자열과 실제 파라미터 객체를 함께 제공한다. 이름에는 `conv1.weight`, `layer4.0.conv1.weight`, `fc.bias` 같은 값이 들어간다.

`startswith`는 문자열이 특정 문자로 시작하는지 검사한다. 오른쪽 조건의 True 또는 False가 `requires_grad`에 대입된다. 그중 True인 파라미터만 리스트에 모아 현재 모델의 옵티마이저에 전달했다.

여기서 중요한 연결은 “지금 점수를 계산한 모델”과 “지금 옵티마이저가 관리하는 모델”이 같다는 것이다. `partial_model(imgs)`로 손실을 구하면서 CustomCNN용 `optimizer.step()`을 실행해도 원하는 ResNet18 파라미터를 학습시키는 코드가 되지 않는다.

기록에 사용한 시작 학습률은 Frozen 0.001, Partial 0.0001, Full 0.00001이다. 이 수치는 학습 범위별 정답이 아니다. 전체 모델의 학습률을 아주 작게 잡으면 새 분류층도 천천히 학습할 수 있다. 같은 한 에포크라도 학습 범위 외의 조건이 성능에 영향을 준다.

독립적으로 비교하려면 각 모델을 같은 사전 학습 가중치에서 새로 시작한다. Frozen을 학습한 뒤 같은 모델을 Partial로 바꾸어 이어서 학습하는 것도 가능하지만, 그것은 단계적으로 학습 범위를 넓히는 별도의 실험이다.

## 21. 가중치 동결과 BatchNorm 통계 고정을 구분한다

ResNet18에는 <strong>배치 정규화(Batch Normalization, BatchNorm)</strong> 층이 들어 있다. 이 층은 특징값의 분포를 조정한다. 여기서는 세부 수식보다 두 종류의 상태를 구분하는 것이 먼저다.[^batchnorm]

| 상태 | 무엇인가 | 어떻게 바뀌는가 |
| --- | --- | --- |
| 학습 파라미터 | 값을 조정하는 가중치·편향 | 기울기와 옵티마이저로 갱신 |
| 저장 통계값 | 학습 중 관측한 평균·분산을 반영한 값 | 학습 모드의 순전파 중 갱신 |

파라미터에 `requires_grad=False`를 설정해도, 학습 모드에서는 저장 평균·분산이 바뀔 수 있다. 저장 통계값은 일반적인 가중치처럼 옵티마이저가 바꾸는 대상이 아니기 때문이다.

Frozen에서 기존 특징 추출 동작까지 유지하려면 가중치와 통계값을 함께 고정해야 한다. Partial에서도 어떤 통계를 바꿀지 정해야 한다.

### 선택 A: 저장 통계값을 모두 고정한다

~~~python
partial_model.eval()

for imgs, labels in res_train_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    partial_optimizer.zero_grad()
    loss = loss_fn(partial_model(imgs), labels)
    loss.backward()
    partial_optimizer.step()
~~~

이 코드는 eval 상태지만 학습한다. 기울기 기록을 끄지 않았고, layer4와 fc는 학습 가능하며, 그 파라미터를 관리하는 옵티마이저로 업데이트하기 때문이다. BatchNorm의 저장 통계는 고정하고 선택한 가중치만 학습하는 조건이다.[^autograd]

### 선택 B: 학습할 마지막 블록의 통계도 바꾼다

~~~python
partial_model.eval()
partial_model.layer4.train()
partial_model.fc.train()
~~~

이 모드 설정 뒤에 같은 학습 루프를 실행하면 앞쪽 블록은 평가 동작을 유지하고 layer4는 학습 동작을 사용한다. 표준 ResNet18의 fc는 Linear이므로 그 층 자체의 계산은 train·eval에 따라 달라지지 않는다. 가중치의 학습 가능 여부는 별도로 설정되어 있다.

Full에서도 전체 train으로 저장 통계를 갱신할 수 있고, 전체 eval 상태에서 기울기 계산을 유지해 통계를 고정할 수 있다. 어느 쪽이 항상 더 좋은지 정해져 있지는 않다. 어떤 정책을 사용했는지 실험 기록에 남겨야 한다.

검증 함수가 `model.eval()`을 호출하면 검증이 끝난 뒤에도 그 상태가 유지된다. 다음 에포크를 시작할 때 각 실험의 모드 정책을 다시 적용한다. 모든 모델에 무조건 `model.train()`을 호출하면 고정하려던 BatchNorm 통계까지 바뀔 수 있다.

## 22. 기록된 결과에서 확인할 수 있는 범위를 정한다

기존 예비 실험은 정상 270장, 폐렴 777장으로 구성된 검증 이미지 1,047장을 사용했다. 아래 수치는 그 기록을 정리하고 계산한 값이다. Full과 최종 test의 결과는 기록에 제공되지 않았다.

혼동 행렬의 행은 실제 정답, 열은 예측이며 순서는 NORMAL, PNEUMONIA다.

| 모델 | 혼동 행렬 | 정확도 | 정상 재현율 | 폐렴 재현율 | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| CustomCNN | `[[250,20],[24,753]]` | 95.80% | 92.59% | 96.91% | 0.9454 |
| Frozen | `[[233,37],[24,753]]` | 94.17% | 86.30% | 96.91% | 0.9227 |
| Partial | `[[252,18],[16,761]]` | 96.75% | 93.33% | 97.94% | 0.9575 |
| Full | 기록 없음 | — | — | — | — |

CustomCNN의 혼동 행렬은 같은 검증 샘플에서 기록한 정상 정답 250장, 폐렴 정답 753장을 바탕으로 재구성했다. Frozen과 Partial의 혼동 행렬은 기록에 직접 제공된 값이다.

관측한 결과에서는 Partial이 가장 좋았다. Frozen과 비교하면 정상 정답이 19장, 폐렴 정답이 8장 늘었고 총 오답은 61장에서 34장으로 줄었다. 이것은 이미지 개수다. 환자 ID를 확인하지 않았으므로 환자 수로 바꿔 말할 수 없다.

이 결과에 대한 가능한 설명은 있다. Frozen은 기존 특징 가중치를 유지했고, Partial은 마지막 특징 블록을 X-ray에 맞춰 바꿀 수 있었다. 그러나 이 차이가 성능 개선의 유일한 원인이라고 확인한 것은 아니다.

같은 샘플로 비교하더라도 전처리, 학습률, 새 분류층의 초기 가중치, 배치 순서, BatchNorm 정책, 학습량이 다르면 결과에 영향을 줄 수 있다. 이 기록만으로 Partial이 항상 우수하다거나, 전이 학습이 항상 직접 만든 CNN보다 좋다고 일반화할 수 없다.

후속 비교에서는 최소한 다음 조건을 함께 적는다.

| 기록 항목 | 확인하려는 것 |
| --- | --- |
| 파일 목록과 분할 인덱스 | 같은 이미지로 평가했는가 |
| 클래스 번호 대응 | 같은 점수 축과 정답 의미인가 |
| 전처리 | 입력 크기·자르기·정규화가 어떻게 다른가 |
| 초기 가중치와 학습 범위 | 같은 출발점에서 무엇을 바꿨는가 |
| 학습률·에포크·배치 크기 | 어느 정도 학습했는가 |
| BatchNorm 정책 | 저장 통계도 바뀌었는가 |
| 목표 지표 | 어떤 기준으로 모델을 선택했는가 |

이 단계에서는 한 에포크의 예비 결과를 이해하고 비교 절차를 완성하는 것이 우선이다. 충분한 반복 실험이나 환자 독립 검증을 수행한 결과처럼 해석하지 않는다.

## 23. 여러 에포크와 저장은 서로 다른 상태를 관리하는 일이다

여러 에포크는 같은 모델과 옵티마이저로 학습 데이터 순회를 반복하는 것이다. 매번 검증하고, 정해 둔 지표가 좋아졌을 때 모델 상태를 보관한다.

어느 범위의 정보를 언제 초기화할지 구분해야 한다.

| 대상 | 준비·초기화하는 시점 | 유지할 범위 |
| --- | --- | --- |
| 모델·옵티마이저 | 독립 실험 시작 | 여러 에포크 |
| 에포크 손실·샘플 수 | 에포크 시작 | 해당 에포크의 모든 배치 |
| 전체 검증 정답·예측 | 검증 시작 | 해당 검증의 모든 배치 |
| 기울기 | 일반적인 현재 설정에서 배치마다 | 이번 역전파·업데이트 |

`state_dict`는 모델 파라미터와 BatchNorm 통계 같은 저장 상태를 담은 딕셔너리다. 이를 파일로 저장할 수 있다.[^saving]

~~~python
torch.save(model.state_dict(), "custom_cnn_weights.pt")
~~~

이 파일만으로 모델 구조, 전처리, 클래스 번호, train·eval 모드, requires_grad 설정이 자동 복원되지는 않는다. 다시 사용할 때 필요한 설정도 함께 보관해야 한다.

최고 성능 모델을 메모리에 남기려면 단순 대입도 조심해야 한다.

~~~python
import copy

best_weights = copy.deepcopy(model.state_dict())
~~~

`deepcopy`는 독립적인 복사본을 만든다. `best_weights = model.state_dict()`만 사용하면 이후 학습에 따라 그 안에서 참조한 값이 바뀔 수 있다. 검증 지표가 개선될 때 바로 파일로 저장하는 방법도 있다.

학습을 다시 이어갈 목적이라면 옵티마이저 상태, 완료한 에포크, 난수 상태도 필요하다. 분할 인덱스만 저장하지 말고 그 인덱스의 기준인 원본 파일 목록과 클래스 대응도 보관한다. 파일 순서가 달라지면 같은 인덱스로 같은 실험을 재현할 수 없다.

`random_state`로 분할을 고정한 것과 `torch.manual_seed`로 모델 초기화 등의 난수를 설정한 것은 별개다. 시드를 고정해도 라이브러리 버전과 연산 장치가 다르면 모든 계산 결과가 완전히 같다고 보장할 수는 없다.[^randomness]

### 실행 환경에서 문제가 생기면 상태를 나누어 확인한다

GPU를 선택했다고 모델과 텐서가 자동으로 GPU로 이동하지 않는다. 다음 값으로 실제 장치를 확인할 수 있다.

~~~python
print(device)
print(next(model.parameters()).device)
print(imgs.device)
~~~

데이터 처리에는 파일 읽기, 이미지 해석, 전처리, 배치 구성, 장치 전송, 모델 계산이 포함된다. 학습이 느리다면 어느 단계가 오래 걸리는지 분리해서 확인한다. 네트워크 드라이브의 작은 파일 읽기도 원인 후보지만, 느리다는 사실만으로 원인을 확정할 수는 없다.

노트북에서 학습 루프만 다시 실행하면 현재 가중치에서 이어갈 수 있다. 모델 생성까지 다시 실행하면 초기 가중치에서 시작하고, 옵티마이저만 다시 만들면 Adam의 내부 상태도 새로 시작한다.

Colab 같은 환경에서 세션이 초기화되면 코드 파일이 남아 있어도 메모리의 모델은 사라질 수 있다. `FileNotFoundError`가 발생하면 먼저 드라이브 연결과 현재 경로를 확인한다. 저장하지 않은 가중치는 노트북에 남은 점수만으로 복원할 수 없다.

## 24. 실행 예제로 전체 처리를 확인한다

[한국어 Python 실행 예제](/chest-xray-classification-pytorch/examples-ko.py)와 [영어 주석 실행 예제](/chest-xray-classification-pytorch/examples-en.py)는 같은 처리 구조를 사용한다. PyTorch, torchvision, scikit-learn, NumPy, Pillow가 설치된 Python 환경에서 실행한다.

명령의 `--data-dir`에는 `train`, `val`, `test`가 들어 있는 실제 폴더를 지정한다. 아래는 macOS·Linux의 zsh·bash에서 실행하는 형태다. `python`은 필요한 패키지가 설치된 환경의 Python을 가리켜야 한다.

먼저 실제 X-ray 없이 코드 연결을 확인한다.

~~~bash
python examples-ko.py --self-check
~~~

이 옵션은 임시 이미지 폴더로 데이터 연결과 모델 동작을 점검한다. 분류 성능을 측정하는 실험은 아니다.

직접 만든 CNN을 한 에포크 실행하려면 다음과 같이 지정한다.

~~~bash
python examples-ko.py --data-dir ./chest_xray --model custom --epochs 1 --output runs/custom-best.pt
~~~

Partial 전이 학습에서 BatchNorm 저장 통계를 고정하는 예는 다음과 같다.

~~~bash
python examples-ko.py --data-dir ./chest_xray --model partial --epochs 3 --bn-policy fixed --output runs/partial-best.pt
~~~

| 옵션 | 의미 |
| --- | --- |
| `--model custom` | 직접 만든 CNN을 새 초기 가중치로 학습 |
| `--model frozen` | ResNet18의 새 분류층만 학습 |
| `--model partial` | ResNet18의 layer4와 fc 학습 |
| `--model full` | ResNet18 전체 미세 조정 |
| `--bn-policy fixed` | 저장 평균·분산 고정, 선택한 가중치는 학습 |
| `--bn-policy adapt` | Partial은 layer4, Full은 전체의 저장 통계 갱신 |
| `--device auto` | 사용 가능한 CUDA, MPS, CPU 순서로 선택 |
| `--output` | 선택한 모델을 저장할 새 파일 경로 |

Frozen에서는 기존 특징 추출부의 저장 통계를 고정한다. 기본 학습 명령은 test 데이터를 읽지 않는다. 검증 Macro F1이 가장 높은 에포크의 가중치를 독립 사본으로 보관하고, 전체 학습이 끝나면 파일로 저장한다. 따라서 학습 도중 세션이 종료되면 그 실행의 체크포인트 파일은 남지 않는다. 같은 출력 파일을 덮어쓰지 않으므로 다른 실험에는 새 경로를 지정한다.

학습 설정과 모델 선택을 끝낸 다음에는 저장한 모델로 최종 테스트를 실행한다.

~~~bash
python examples-ko.py --data-dir ./chest_xray --evaluate-test runs/partial-best.pt
~~~

이 명령은 선택한 체크포인트를 불러와 test를 평가한다. 새로 학습하지 않는다. 테스트 결과를 보고 계속 모델을 바꾸면 test를 다시 모델 선택에 사용하게 되므로, 어떤 용도로 평가했는지도 기록해야 한다.

<strong>체크포인트(Checkpoint)</strong>는 모델을 다시 불러오거나 학습 상태를 보관하기 위한 저장 파일이다. 예제에는 모델·옵티마이저 상태, 클래스 번호, 상대 파일 목록, 분할 인덱스, 전처리, 학습 설정, 평가 기록이 들어간다. 이 예제의 명령행 기능은 학습과 저장 모델의 테스트 평가이며, 중단 지점부터 정확히 학습을 재개하는 기능까지 제공하지는 않는다.

### 실행 예제에서 확인한 범위

PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, scikit-learn 1.8.0 환경에서 합성 RGB 이미지 폴더로 점검했다. 데이터 분할과 라벨 대응, 독립적인 전처리 객체, 마지막 작은 배치, CNN 출력과 실제 가중치 변경, 평가 손실 집계, ResNet18의 학습 범위와 BatchNorm 통계 변화를 확인했다.

추가로 test 폴더가 없는 상태에서 두 에포크를 학습하고, 최고 검증 모델을 저장한 뒤 다시 불러오는 흐름을 실행했다. 그다음 별도로 만든 test 폴더를 저장 모델로 평가했다.

ResNet 구조 점검에는 사전 학습 가중치를 다운로드하지 않는 `weights=None`을 사용했다. 실제 전이 학습 명령은 지정한 사전 학습 가중치를 가져온다. 이번 점검은 프로그램의 연결과 상태 변경에 대한 검사이며, 실제 X-ray에서 앞의 성능 수치를 재현한 실험은 아니다.

## 25. 처음 보는 데이터셋에도 남길 판단 기준

이 실습에서 반복한 질문은 “지금 가진 정보가 무엇이고, 무엇을 바꾸려는가”였다. 파일 경로와 이미지를 구분해야 불필요한 읽기를 줄일 수 있고, 인덱스의 기준을 알아야 같은 샘플을 계속 추적할 수 있다. 이미지 한 장을 동등하게 평가하려면 평균을 합치는 분모도 확인해야 한다.

| 학습 범위 | 지금 익힐 내용 |
| --- | --- |
| 실무 필수 | 폴더·라벨 조사, 분할 기준, 전처리 검증, 입력·출력 모양, 학습·검증 분리, 클래스별 평가, 저장 |
| 원리 이해 | Flatten과 GAP의 정보 차이, 평균의 분모, 가중치 동결과 BatchNorm 통계 고정 |
| 이후 확장 | 반복 실험, 학습률 탐색, 증강 비교, 로딩 최적화, 혼합 정밀도 |

환자 독립 평가가 목표라면 환자 정보를 확인하는 작업은 뒤로 미룰 최적화 항목이 아니다. 목표가 요구하는 평가 조건부터 갖춰야 한다.

다음 질문을 설명하고 작은 코드로 확인할 수 있으면, 이 단계에서 필요한 기본 구조를 이해한 것이다.

1. 클래스 개수만 알고 싶을 때 `.targets`를 쓰는 이유는 무엇인가?
2. Subset의 인덱스 0과 원본 Dataset의 인덱스 0이 항상 같은 샘플인가?
3. 학습과 검증이 같은 ImageFolder를 공유하면 transform 변경은 어디에 영향을 주는가?
4. 마지막 배치가 작을 때 배치 평균을 그대로 평균내면 어느 이미지의 영향력이 커지는가?
5. eval 상태에서 layer4 가중치를 학습할 수 있는 조건은 무엇인가?
6. 높은 전체 정확도와 낮은 정상 재현율이 동시에 나올 수 있는 이유는 무엇인가?

<details>
<summary>답을 확인하며 자신의 설명과 비교하기</summary>

1. 필요한 것은 이미지 픽셀이 아니라 각 샘플의 라벨이다. targets를 세면 파일 읽기를 추가하지 않는다.
2. Subset의 인덱스는 선택 목록을 기준으로 한다. 그 목록의 첫 값이 원본에서 선택할 인덱스다.
3. 같은 원본을 통해 읽는 양쪽에 영향을 준다. 서로 다른 변환이 필요하면 샘플 대응을 유지한 별도 객체를 준비한다.
4. 작은 배치도 큰 배치와 같은 비중을 가지므로 작은 배치의 이미지 한 장이 더 크게 반영된다.
5. 해당 가중치의 requires_grad가 True이고, 기울기를 기록하며, backward와 그 모델의 옵티마이저 step을 실행해야 한다.
6. 많은 클래스만 잘 맞혀도 전체 정확도는 높을 수 있다. 전체 이미지가 기준인 정확도와 실제 정상만 기준인 재현율은 분모가 다르다.

</details>

다음 실험에서는 한 조건을 바꾸고, 같은 검증 샘플에서 결과를 비교해 보자. 모델을 더 복잡하게 만드는 일보다 데이터의 대응, 업데이트되는 상태, 평가 기준을 스스로 설명하며 끝까지 실행하는 일이 먼저다.

## 참고 자료

공식 API 문서는 2026년 11월 16일 확인했다. 이미지 개수와 예비 실험의 성능 수치는 기존 학습 기록에 따르며, 원본 X-ray 파일과 과거 모델 가중치를 이번 게시 작업에서 다시 평가한 것은 아니다.

[^folder]: torchvision, [ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) · [구현 소스](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html) — 클래스 정렬·메타데이터·RGB 로더·샘플 읽기.
[^data]: PyTorch, [Dataset·ConcatDataset·Subset·DataLoader](https://docs.pytorch.org/docs/stable/data.html) — 데이터 연결·샘플 선택·배치·셔플.
[^split]: scikit-learn, [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — 층화 분할과 반환 결과.
[^cv]: scikit-learn, [교차 검증과 그룹 분할](https://scikit-learn.org/stable/modules/cross_validation.html) — 모델 선택·최종 평가·환자 그룹 분리.
[^transforms]: torchvision, [v2 이미지 변환](https://docs.pytorch.org/vision/stable/transforms.html) · [ToDtype](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html) — 텐서 표현·자료형·값 범위.
[^tensor]: PyTorch, [flatten](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html) · [permute](https://docs.pytorch.org/docs/stable/generated/torch.permute.html) — 축 펼치기와 순서 변경.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — 채널·가중치·출력 모양.
[^pool]: PyTorch, [MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) — 영역과 이동 간격.
[^gap]: PyTorch, [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html) — 출력 공간 크기 지정.
[^ce]: PyTorch, [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — 로짓·정답 표현·평균의 분모.
[^bce]: PyTorch, [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) — 같은 모양의 출력과 실수 정답.
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) — requires_grad·no_grad·평가 모드의 차이.
[^device]: PyTorch, [CUDA](https://docs.pytorch.org/docs/stable/notes/cuda.html) · [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) — 장치별 텐서 계산.
[^metrics]: scikit-learn, [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) — 실제 정답 행과 예측 열.
[^report]: scikit-learn, [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) — 클래스별 지표·평균·분모가 0인 경우.
[^resnet]: torchvision, [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) — 사전 학습 가중치와 평가 전처리.
[^batchnorm]: PyTorch, [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) — 학습 파라미터와 저장 통계.
[^saving]: PyTorch, [모델 저장과 불러오기](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) — state_dict·독립 복사·체크포인트.
[^randomness]: PyTorch, [재현성](https://docs.pytorch.org/docs/stable/notes/randomness.html) — 난수 설정과 환경에 따른 한계.

