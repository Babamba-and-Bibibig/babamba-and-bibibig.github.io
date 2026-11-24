+++
title = "SSD 객체 탐지 평가: 예측 구조부터 AP·mAP와 시각화까지"
description = "이미지·배치·박스의 단위를 구분하고, 신뢰도와 IoU에서 TP·FP·FN, COCO AP와 전체 mAP까지 연결한다. Oxford-IIIT Pet 실습의 검증 결과와 테스트 추론, 저장 조건을 함께 정리한다."
date = 2026-11-24T08:45:00+09:00
slug = "ssd-detection-evaluation"

[extra]
toc = true
styles = ["ssd-detection-evaluation/article.css"]
+++

SSD로 강아지와 고양이를 탐지한 결과를 출력했더니, 이미지 한 장에서 박스가 200개 나왔다. 동물이 그렇게 많이 있는 사진은 아니었다. 박스 옆의 신뢰도와 검증에서 출력한 mAP도 처음에는 무엇을 기준으로 읽어야 하는지 헷갈렸다.

이때 먼저 구분해야 했던 것은 숫자의 크기보다 <strong>그 숫자가 무엇을 세거나 비교하는가</strong>였다. 이미지 개수와 예측 박스 개수는 다르고, 잘못된 예측을 세는 것과 놓친 정답을 세는 것도 다르다. 같은 0~1 범위의 숫자라도 모델의 신뢰도와 박스의 겹침 정도는 다른 정보다.

이 글은 이미 학습한 객체 탐지 모델의 출력을 읽고, 검증 데이터를 평가한 뒤 개별 이미지를 확인하는 과정을 정리한다. [이미지 분류 평가](/image-classification-evaluation/)가 이미지마다 클래스 하나를 비교했다면, 여기서는 이미지마다 개수가 달라질 수 있는 박스와 클래스를 함께 비교한다. 필요한 자료구조부터 설명하므로 앞 글을 읽지 않아도 따라갈 수 있다.

본문의 성능 수치는 실습 기록에서 가져왔다. 아래의 작은 AP 계산 예제는 원리를 확인하기 위한 별도 예시다. 학습한 모델과 원본 노트북을 새로 실행한 결과로 섞어 쓰지 않는다. 실행 가능한 [평가·시각화 함수와 계산 예제](/ssd-detection-evaluation/examples.py)는 마지막 절에 안내한다.

## 01. 무엇을 찾는 모델인지 정답부터 확인한다

<strong>객체 탐지(Object Detection)</strong>는 이미지 안에서 찾으려는 대상의 종류와 그 대상을 감싸는 박스 좌표를 함께 예측하는 작업이다. <strong>바운딩 박스(Bounding Box)</strong>는 대상 영역을 사각형으로 나타낸 것이다. 이번 실습은 SSD라는 객체 탐지 모델로 강아지와 고양이를 구분한다. SSD의 전체 이름은 Single Shot MultiBox Detector다.

중요한 조건이 하나 있다. Oxford-IIIT Pet 데이터셋의 박스 정답은 동물의 <strong>머리 영역</strong>을 나타낸다. 머리를 감싼 예측을 몸 전체와 비교하면, 모델이 학습한 정답과 다른 기준으로 판단하게 된다.[^pets]

원본 데이터셋 전체와 이번 코드에서 사용하는 정보를 먼저 나눈다.

<div class="ssd-table">

| 원본 정보 | 들어 있는 내용 | 이번 객체 탐지에서 하는 일 |
| --- | --- | --- |
| JPG 이미지 | 실제 픽셀 | 모델 입력과 화면 표시 |
| TXT 목록 | 이미지 이름, 종·품종 정보, 분할 목록 | 사용할 파일 이름 확인 |
| XML 주석 | 객체 이름과 머리 박스 좌표 | 이미지별 탐지 정답 구성 |
| 분할 마스크 | 픽셀별 영역 구분 | 이번 박스 평가에서는 사용하지 않음 |

</div>

이미지와 XML은 확장자를 제외한 파일 이름으로 연결한다. 같은 이름이라는 대응을 확인한 다음 이미지와 정답을 한 쌍으로 구성해야 한다. JPG 목록과 XML 목록을 각각 정렬했다는 사실만으로 두 목록의 같은 인덱스가 같은 이미지를 뜻한다고 가정하면 안 된다.

실습에서는 XML에서 읽은 이름을 다음 번호로 바꿨다. <strong>클래스(Class)</strong>는 구분할 종류, <strong>라벨(Label)</strong>은 각 객체에 붙인 클래스 번호를 뜻한다.

~~~python
# XML의 객체 이름을 이 모델에서 사용할 정수 라벨로 바꾼다.
class_map = {"dog": 1, "cat": 2}

# 모델의 정수 라벨을 화면에 표시할 이름으로 되돌린다.
class_names = {number: name for name, number in class_map.items()}
~~~

두 변수는 모두 딕셔너리다. `class_map`은 문자열을 정수로, `class_names`는 정수를 문자열로 연결한다. 현재 `class_names`의 내용은 `{1: "dog", 2: "cat"}`이다.

이 SSD에서는 0이 배경 클래스이므로, 모델의 출력 클래스 수는 배경을 포함해 3개다. 반면 원본 TXT의 `SPECIES` 번호는 `1=cat`, `2=dog`다. 원본 번호를 그대로 사용한 것이 아니라 XML 이름을 다시 매핑했기 때문에 생기는 차이다. <strong>번호는 그 번호를 만든 규칙과 함께 읽어야 한다.</strong>

학습·검증·테스트의 역할도 구분한다.

<div class="ssd-table">

| 단계 | 입력 | 하는 일 | 가중치 갱신 |
| --- | --- | --- | --- |
| 학습 | 이미지와 박스 정답 | 손실 계산 후 모델 수정 | 함 |
| 검증 | 이미지와 박스 정답 | 예측을 만들고 정답과 비교 | 안 함 |
| 이번 테스트 추론 | 별도로 남긴 이미지 | 예측 생성과 사례 확인 | 안 함 |

</div>

이번 실습에서 확인한 테스트 파일에는 탐지 평가에 필요한 XML 박스 정답이 없었다. 종·품종 정보까지 없다는 뜻은 아니다. 이 상태에서는 예측을 그릴 수 있지만, 박스 정답과 비교해야 하는 탐지 mAP는 계산할 수 없다.

## 02. 이미지 한 장, 배치 한 개, 박스 한 개를 나눈다

<strong>텐서(Tensor)</strong>는 PyTorch에서 여러 축으로 배열된 숫자를 다루는 자료형이다. <strong>shape</strong>는 각 축의 크기를 순서대로 나타낸다. 이번 입력 이미지 한 장은 `float32` 텐서이며, shape는 `(3, H, W)`다. `H`는 이미지 높이, `W`는 이미지 너비다. 0번 축은 RGB 채널 3개, 1번 축은 세로 방향, 2번 축은 가로 방향이다.

이미지 한 장을 가져오면 이미지 텐서와 정답 딕셔너리를 받는다. 여기서 `val_dataset`은 검증 샘플을 제공하는 데이터셋 객체다.

~~~python
image, target = val_dataset[0]

print(image.shape)             # (3, H, W)
print(target["boxes"].shape)   # (N, 4)
print(target["labels"].shape)  # (N,)
~~~

`N`은 이 이미지의 <strong>정답 박스 개수</strong>다. `boxes`의 한 행은 박스 하나의 `[x1, y1, x2, y2]` 좌표 네 개이고, `labels`의 같은 인덱스에는 그 박스의 클래스 번호가 있다. 좌표는 왼쪽 위와 오른쪽 아래의 경계를 나타내며, 너비는 `x2-x1`, 높이는 `y2-y1`로 계산한다. 픽셀을 고르는 정수 인덱스와 경계 좌표를 혼동하지 않는다.[^ssd]

<strong>배치(Batch)</strong>는 한 번에 처리할 이미지들의 묶음이다. <strong>DataLoader</strong>는 데이터셋에서 샘플을 꺼내 배치로 묶어 주는 객체다. 이번 탐지용 배치에서는 이미지 텐서와 정답 딕셔너리를 각각 튜플로 묶는다. 튜플은 여러 값을 순서대로 담는 Python 자료구조다.

~~~python
# 이 배치 결합 함수를 DataLoader의 collate_fn 인자로 사용한다.
def detection_collate(batch):
    return tuple(zip(*batch))
~~~

`batch`는 `(image, target)` 쌍들의 목록이다. `*batch`는 각 쌍을 `zip`의 개별 입력으로 펼치고, `zip`은 각 쌍의 첫 원소끼리, 둘째 원소끼리 묶는다. 결과는 `(이미지 튜플, 정답 튜플)`이다. 이미지 크기나 박스 개수가 서로 달라도 이 목록 구조를 유지할 수 있다.

<div class="ssd-table">

| 표현 | 꺼내거나 세는 대상 |
| --- | --- |
| `val_dataset[0]` | 첫 검증 이미지와 정답 |
| `next(iter(val_loader))` | 로더가 제공하는 첫 배치 |
| `len(val_dataset)` | 이미지 수 |
| `len(val_loader)` | 배치 수 |
| `len(target["boxes"])` | 이미지 한 장의 정답 박스 수 |

</div>

일반적인 DataLoader는 `val_loader[0]`처럼 인덱싱하지 않는다. `iter`는 배치를 차례로 꺼낼 반복자를 만들고, `next`는 그 반복자의 다음 항목을 꺼낸다. `next(iter(val_loader))`를 매번 새로 호출하면 반복자도 매번 새로 만들어진다. 같은 반복자에서 계속 다음 배치를 꺼내는 것과 다르며, 로더가 순서를 섞도록 설정되었다면 새 반복의 첫 배치 내용도 달라질 수 있다.

실습 기록의 검증 이미지 735장과 배치 크기 4를 기준으로, 마지막 배치를 버리지 않으면 4장짜리 배치 183개와 3장짜리 배치 1개가 생긴다. 총 184배치다. 이것은 특정 실습의 분할 크기이며 데이터셋 전체 이미지 수가 아니다.

### 모델 호출 뒤의 `[0]`은 첫 이미지 결과다

평가 모드의 TorchVision SSD에는 이미지 텐서들의 리스트를 전달한다. 반환값도 입력 이미지마다 예측 딕셔너리 하나를 담은 리스트다.[^ssd]

아래에서는 `model.eval()`로 예측을 반환하는 동작으로 바꾸고, `torch.no_grad()`로 학습을 위한 계산 기록을 만들지 않도록 한다. 두 설정의 차이는 04절에서 더 자세히 살펴본다.

~~~python
import torch

# model과 device는 학습에 사용한 모델과 계산 장치다.
model_images = [image.to(device)]
model.eval()
with torch.no_grad():
    predictions = model(model_images)
pred = predictions[0]
~~~

`image.to(device)`는 모델과 같은 계산 장치에 있는 텐서를 준비한다. 대괄호는 텐서 하나를 담은 Python 리스트를 만든다. shape가 `(1, 3, H, W)`인 텐서를 새로 만드는 연산은 아니다. `.to(device)`가 반환한 값을 사용했을 뿐, 원래 `image` 변수에 결과를 자동으로 다시 대입하지도 않는다.

입력 이미지가 `B`장이면 `predictions`의 길이도 `B`다. 그중 첫 이미지의 예측 개수를 `M`이라고 하면 구조는 다음과 같다.

<div class="ssd-table">

| 표현 | 자료형·shape | 의미 |
| --- | --- | --- |
| `predictions` | 딕셔너리들의 리스트, 길이 `B` | 이미지별 예측 전체 |
| `predictions[0]` | 딕셔너리 | 첫 이미지의 모든 예측 |
| `pred["boxes"]` | 실수 텐서 `(M, 4)` | 그 이미지의 예측 박스 좌표 |
| `pred["labels"]` | 정수 텐서 `(M,)` | 예측 박스별 클래스 번호 |
| `pred["scores"]` | 실수 텐서 `(M,)` | 예측 박스별 클래스 점수 |
| `pred["boxes"][0]` | 실수 텐서 `(4,)` | 첫 예측 박스의 좌표 네 개 |

</div>

따라서 `model([image.to(device)])[0]`은 첫 박스만 남기는 코드가 아니다. 첫 이미지의 결과를 꺼낸 것이고, 그 안에는 여러 박스가 있다. 정답 박스 수 `N`과 예측 박스 수 `M`도 같을 필요가 없다.

<figure class="ssd-figure">
<img src="/ssd-detection-evaluation/prediction-structure-ko.svg" alt="입력 이미지별 예측 딕셔너리 안에서 같은 j 인덱스의 박스, 라벨, 점수가 하나의 예측으로 연결되는 구조" loading="lazy">
<figcaption>이미지를 고르는 인덱스와, 그 이미지 안의 예측을 고르는 인덱스는 서로 다른 대상을 가리킨다.</figcaption>
</figure>

## 03. 변환과 반복문에서도 같은 예측의 연결을 유지한다

파일 읽기와 텐서 변환은 별개의 작업이다. Pillow는 이미지 파일을 읽는 라이브러리이고, `Image.open`은 파일 경로를 받아 이미지 객체를 만든다. `.convert("RGB")`는 세 채널 이미지로 맞춘다. `to_tensor`는 이번과 같은 일반적인 RGB JPG를 `(3, H, W)` 구조와 0~1 범위의 픽셀 값으로 변환한다.[^tensor]

~~~python
from PIL import Image
from torchvision.transforms.functional import to_tensor

# image_path는 실제 이미지 파일의 경로다.
with Image.open(image_path) as source:
    pil_image = source.convert("RGB")

image_tensor = to_tensor(pil_image)
~~~

Pillow 이미지에 바로 `.to(device)`를 호출할 수는 없다. 반대로 데이터셋에서 이미 텐서를 받았다면 그 텐서를 `Image.open`으로 다시 열 필요도 없다. 파일 경로, 이미지 객체, 텐서 중 무엇을 가지고 있는지 먼저 확인해야 한다.

예측을 순회할 때 지켜야 할 관계는 단순하다. <strong>같은 인덱스의 박스·라벨·점수가 한 예측이다.</strong> `j`가 이미지 안의 예측 인덱스라면 `boxes[j]`, `labels[j]`, `scores[j]`를 함께 다뤄야 한다.

~~~python
# assert는 조건이 거짓이면 오류를 내는 검사 문법이다.
assert len(pred["boxes"]) == len(pred["labels"]) == len(pred["scores"])

for box, label, score in zip(
    pred["boxes"], pred["labels"], pred["scores"]
):
    x1, y1, x2, y2 = box.tolist()
    name = class_names[label.item()]
    score_value = score.item()
~~~

`zip`은 세 입력의 같은 인덱스에 있는 값을 한 묶음으로 제공한다. 반복문 왼쪽의 `box`, `label`, `score`에 각 값이 나뉘어 들어간다. 세 값이 `box` 하나에 모두 담기는 것은 아니다. 길이가 다르면 가장 짧은 입력까지만 순회하므로, 앞의 길이 검사가 대응 오류를 찾는 데 도움이 된다.

한 번의 반복에서 `box`는 shape `(4,)`인 텐서다. `.tolist()`가 좌표 네 개의 Python 리스트를 반환하므로 네 변수에 나누어 담을 수 있다. 이미지에 박스가 하나뿐이라는 가정은 필요 없다. 여러 박스를 한 번에 하나씩 처리하는 것이다.

`.item()`은 원소 하나짜리 텐서에서 Python 숫자를 꺼낸다. `label.item()`은 클래스 이름 조회에 사용할 정수, `score.item()`은 비교나 출력에 사용할 실수다. 값 하나를 다루는 코드와 여러 원소를 가진 텐서 전체를 다루는 코드를 구분해야 한다.

## 04. 평가 모드와 기울기 기록 중단은 역할이 다르다

학습에서는 모델이 낸 결과와 정답의 차이를 <strong>손실(Loss)</strong>로 계산하고, 그 손실을 줄이는 방향으로 가중치를 바꾼다. <strong>기울기(Gradient)</strong>는 가중치를 바꿀 때 손실이 어떻게 변하는지 나타내는 값이다. PyTorch는 학습 때 계산 과정을 기록해 두었다가 기울기를 계산한다.

`model.train()`과 `model.eval()`은 모델의 동작 모드를 바꾼다. `torch.no_grad()`는 해당 구간에서 기울기 계산을 위한 연산 기록을 만들지 않도록 한다. `eval()`만 호출했다고 기울기 기록이 자동으로 꺼지는 것은 아니다.[^autograd]

<div class="ssd-table">

| 목적 | 모드와 호출 | TorchVision SSD의 반환값 |
| --- | --- | --- |
| 학습 | `model.train()` 후 `model(images, targets)` | 분류·박스 회귀 손실 딕셔너리 |
| 평가·추론 | `model.eval()` 후 `model(images)` | 이미지별 `boxes`, `labels`, `scores` 딕셔너리 리스트 |

</div>

여기서 <strong>박스 회귀(Box Regression)</strong>는 박스 좌표를 정답에 맞게 조정하는 예측을 뜻한다. 평가 모드에서 예측을 얻었다고 검증 손실까지 함께 반환되는 것은 아니다.[^ssd]

~~~python
import torch

model.eval()
with torch.no_grad():
    pred = model([image.to(device)])[0]
~~~

다음 학습을 시작할 때는 다시 `model.train()`을 호출한다. 한편 학습 손실을 기록하는 `loss.item()`과 역전파에 쓰는 `loss`도 구분한다. 역전파는 손실에서 가중치의 기울기를 계산하는 과정이므로 원래 손실 텐서가 필요하다. 기록용 누적값에는 Python 숫자를 넣으면 계산 그래프와 불필요하게 연결되지 않는다.

`loss_sum / len(train_loader)`는 <strong>배치마다 반환된 손실값들의 평균</strong>이다. 모든 이미지의 개별 손실 평균과 같다고 바로 말할 수는 없다. 마지막 배치 크기가 다를 수 있고, SSD 손실 자체도 대응된 객체 후보 수에 따른 정규화를 사용한다.[^ssdsource]

각 배치 손실이 이미지별 손실의 단순평균이고 배치 크기까지 같다면, 배치 평균을 다시 평균해도 모든 이미지가 같은 비중을 갖는다. 그 조건을 확인하지 않고 평균의 대상만 바꾸어 설명하면 안 된다.

## 05. 신뢰도, IoU, NMS를 구분한다

<strong>신뢰도(Confidence Score)</strong>는 모델이 예측 클래스에 부여한 점수다. SSD는 이미지에서 특징을 계산하고, 여러 객체 후보에 대해 클래스별 출력과 박스 조정값을 만든다. 클래스 출력에는 <strong>소프트맥스(Softmax)</strong>를 적용해 한 후보의 클래스 점수 합이 1이 되도록 변환한다. 최종 예측에 해당하는 클래스 점수가 `scores`에 담긴다.[^ssdsource]

이 점수는 정답 박스와 비교해서 만든 값이 아니다. 그래서 정답이 없는 새 이미지에서도 출력된다. `score=0.9`를 “이 박스의 실제 정답 확률이 정확히 90%로 검증되었다”는 뜻으로 읽을 수는 없다.

<strong>교집합 대 합집합 비율(IoU, Intersection over Union)</strong>은 두 박스의 겹침 정도다. 두 박스가 차지하는 전체 영역 중 둘 다 포함하는 영역의 비중을 구한다. 겹치는 부분을 두 번 세지 않도록, 전체 영역은 두 박스 면적의 합에서 겹친 면적을 한 번 뺀다.

두 박스를 모두 감싸는 큰 사각형의 면적을 분모로 쓰는 것이 아니다. 그 큰 사각형 안에는 어느 박스에도 들어 있지 않은 영역이 있을 수 있다. 겹침이 없으면 IoU는 0이고, 두 박스가 완전히 일치하면 1이다.

<div class="ssd-table">

| 기준 | 무엇으로 계산하는가 | 무엇을 판단하는가 |
| --- | --- | --- |
| 신뢰도 | 모델의 클래스 출력 | 어떤 예측을 우선 사용하거나 표시할지 |
| 평가의 IoU | 예측 박스와 정답 박스 | 박스가 정답 영역과 충분히 겹치는지 |
| 중복 제거의 IoU | 예측 박스와 다른 예측 박스 | 비슷한 예측을 여러 개 남길지 |

</div>

신뢰도 0.39인 박스가 정답과는 IoU 0.9로 잘 겹칠 수 있다. 두 값은 같은 대상을 측정하지 않으므로, 신뢰도를 IoU 기준과 직접 비교하면 안 된다.

### 박스 200개는 후보를 정리한 뒤 남은 예측 수다

SSD는 실제 객체 개수를 미리 알고 그 수만큼 박스를 만드는 모델이 아니다. 여러 후보를 만든 뒤 낮은 점수를 제외하고, 비슷한 후보를 정리하며, 최종 반환 개수를 제한한다.

<strong>비최대 억제(NMS, Non-Maximum Suppression)</strong>는 많이 겹치는 예측들 중 높은 점수의 박스를 우선 남기는 처리다. TorchVision SSD는 클래스별로 이 처리를 적용한다. 이때 정답 박스는 필요하지 않다.[^ssdsource]

2026년 11월 24일 확인한 공식 SSD 구현의 기본값은 신뢰도 기준 `0.01`, NMS IoU 기준 `0.45`, 이미지당 최종 반환 상한 `200`이다. 모델을 만들 때 바꿀 수 있으므로 실제 객체를 조회하는 것이 정확하다.

~~~python
print("신뢰도 기준:", model.score_thresh)
print("NMS IoU 기준:", model.nms_thresh)
print("이미지당 반환 상한:", model.detections_per_img)
~~~

실습에서는 예측 200개와 앞 5개 라벨 `[2, 2, 2, 2, 2]`, 점수 `[0.9987, 0.1008, 0.0680, 0.0614, 0.0612]`를 확인했다. 여기서 알 수 있는 것은 첫 다섯 예측이 현재 매핑의 `cat`이라는 점이다. 200개 전체가 같은 클래스인지, 정답을 맞혔는지는 이 출력만으로 알 수 없다. 200은 항상 반환해야 하는 개수도 아니다.

## 06. FP는 예측을 세고, FN은 정답을 센다

탐지가 맞았는지 판단하기 전에 같은 이미지인지, 같은 클래스인지, IoU 기준을 충족하는지 확인해야 한다. 일반 객체를 평가하는 이번 설명에서는 <strong>정답 하나를 여러 예측이 중복으로 맞힌 것으로 세지 않는다</strong>. 군중 객체나 무시하도록 표시된 정답의 별도 처리 규칙은 여기서 다루지 않는다.[^coco]

같은 이미지·클래스의 예측은 신뢰도가 높은 순서로 처리한다. 각 예측은 아직 대응되지 않은 정답 중 IoU 기준을 만족하면서 가장 잘 겹치는 정답에 연결한다. 따라서 같은 정답을 노린 중복 예측 중 어느 것이 먼저 대응될지도 점수 순서의 영향을 받는다.

<div class="ssd-table">

| 이름 | 세는 대상 | 판정 |
| --- | --- | --- |
| 올바른 탐지(TP, True Positive) | 남긴 예측 | 같은 이미지·클래스의 아직 대응되지 않은 정답과 IoU 기준을 만족 |
| 잘못된 탐지(FP, False Positive) | 남긴 예측 | 대응 가능한 정답을 찾지 못함 |
| 놓친 정답(FN, False Negative) | 실제 정답 객체 | 올바른 예측이 대응되지 않음 |

</div>

같은 정답을 감싼 중복 박스도 FP가 될 수 있다. 클래스가 틀렸다면 예측한 클래스에서 FP가 되고, 실제 클래스의 정답을 다른 올바른 예측이 찾지 못했다면 그 정답은 FN으로 남는다. 이미지 한 장에 FP와 FN이 동시에 생길 수 있다.

### 신뢰도 0.39인 예측을 지우면 FN이 되는가

제거된 예측 자체가 FN으로 바뀌는 것은 아니다. 먼저 실제 정답이 있었는지, 그 정답을 찾는 다른 예측이 남았는지 확인해야 한다. 아래는 신뢰도 기준을 0.4로 높여 <strong>평가 입력에서</strong> 해당 예측을 제외하는 경우다.

<div class="ssd-table">

| 상황 | 제거 전 | 제거 후 |
| --- | --- | --- |
| 실제 강아지가 없는데 잘못 예측함 | 해당 예측이 FP | 그 FP가 사라짐. 놓친 강아지는 없음 |
| 강아지는 있지만 0.39 박스는 엉뚱한 곳이며 다른 올바른 예측이 없음 | FP와 FN이 함께 있음 | FP는 사라지고 FN은 그대로 |
| 0.39 박스가 강아지를 찾은 유일한 올바른 예측임 | TP | 그 정답이 대응되지 않아 FN으로 남음 |
| 다른 올바른 예측이 정답을 계속 찾고 있음 | 정답이 대응됨 | 제거만으로 그 정답이 FN이 되지는 않음 |

</div>

반대로 `if score < 0.4: continue`가 <strong>그림을 그리는 반복문 안에만</strong> 있다면 화면에서 박스를 생략할 뿐이다. 원래 예측 딕셔너리, 이미 지표에 전달한 예측, 계산을 끝낸 mAP는 바뀌지 않는다. 무엇을 제외하는 코드인지 적용 지점을 함께 봐야 한다.

## 07. 정밀도와 재현율에서 신뢰도가 필요한 이유

<strong>정밀도(Precision)</strong>는 사용한 예측 중 올바른 탐지가 차지하는 비율이다. <strong>재현율(Recall)</strong>은 실제 정답 중 찾아낸 객체가 차지하는 비율이다. 정밀도는 예측을 기준으로, 재현율은 정답을 기준으로 질문한다.

- 정밀도: TP를 `TP + FP`로 나눈다.
- 재현율: TP를 `TP + FN`, 즉 전체 정답 수로 나눈다.

예측이 하나도 없으면 정밀도의 분모가 0이고, 정답이 없으면 재현율의 분모가 0이다. 이런 경우에는 평가 구현의 처리 규칙이 필요하다. 뒤의 계산 예제는 정답과 예측이 존재하는 조건에서 진행한다.

예측 집합이 이미 정해져 있다면 두 비율은 TP·FP·FN만으로 계산된다. 그런데 어떤 예측을 사용할지는 아직 정해야 한다. 높은 신뢰도만 남기면 일부 예측이 제외되고, 기준을 낮추면 낮은 점수의 예측도 포함된다. 그 결과 TP와 FP가 달라지고 두 비율도 달라진다.

<strong>평균 정밀도(AP, Average Precision)</strong>는 이처럼 사용하는 예측의 범위를 바꾸어 가며 얻은 정밀도와 재현율을 하나의 점수로 요약한다. 신뢰도는 비율 공식에 직접 더해지지는 않지만, <strong>어떤 예측부터 포함할지 정하는 순서</strong>에 쓰인다.

“예측을 하나씩 추가한다”는 말은 모델을 다시 실행한다는 뜻이 아니다. 이미 얻은 예측을 점수순으로 정렬한 뒤 첫 예측까지만 사용한 경우, 첫 두 개까지 사용한 경우를 차례로 계산한다는 뜻이다. 점수가 서로 다르면 신뢰도 기준을 점점 낮추는 과정과 연결된다. 추가하는 단위는 이미지나 배치가 아니라 예측 박스다.

## 08. COCO 방식의 AP를 끝까지 계산해 본다

먼저 조건을 고정한다. 강아지 클래스 하나, IoU 기준 0.5, 실제 정답 강아지 2개를 평가한다. 같은 이미지와 클래스의 정답에 대응시키고 중복 여부까지 확인한 뒤, 신뢰도순 판정이 `TP → FP → TP`였다고 하자. 예측의 점수는 차례로 0.9, 0.8, 0.7이다.

<div class="ssd-table">

| 사용하는 예측 | 누적 TP | 누적 FP | 정밀도 | 재현율 |
| --- | ---: | ---: | ---: | ---: |
| 첫 예측까지 | 1 | 0 | 1 | 1/2 |
| 첫 두 예측까지 | 1 | 1 | 1/2 | 1/2 |
| 세 예측 모두 | 2 | 1 | 2/3 | 1 |

</div>

FP를 추가한 두 번째 단계에서는 찾아낸 정답 수가 그대로라 재현율이 늘지 않는다. 사용한 예측 수는 늘었으므로 정밀도는 내려간다. 세 번째 단계에서는 남은 정답을 찾아 재현율이 1이 된다. 중간 결과가 세 쌍 생겼지만 AP가 세 개 생긴 것은 아니다.

현재 TorchMetrics의 기본 COCO 평가에서는 다음 규칙으로 AP 하나를 만든다.[^coco]

1. 재현율 기준을 0.00부터 1.00까지 0.01 간격으로 정한다. 양 끝을 포함해 101개다.
2. 각 기준에 대해 그 이상의 재현율에 도달한 중간 결과를 찾는다.
3. 그 결과들 중 가장 높은 정밀도를 고른다. 도달한 결과가 없으면 0을 넣는다.
4. 이렇게 얻은 101개 정밀도를 평균한다.

이 규칙으로 측정 지점 사이의 정밀도를 정하는 것을 여기서는 <strong>보간(Interpolation)</strong>이라고 부른다. 인접한 두 점을 직선으로 잇고 그 값을 사용하는 규칙과는 다르다.

이 예제에서는 재현율 0.00~0.50의 51개 기준에 정밀도 1을 사용할 수 있다. 재현율 0.51~1.00의 50개 기준에는 재현율 1에 도달한 결과의 정밀도 2/3을 사용한다. 따라서 AP는 `(51 × 1 + 50 × 2/3) / 101`, 약 <strong>0.8349835</strong>다.

이 값은 중간 정밀도 세 개를 단순평균한 `(1 + 1/2 + 2/3) / 3`과 다르다. 정밀도와 재현율을 더한 평균도, 신뢰도 세 개의 평균도 아니다.

<figure class="ssd-figure">
<img src="/ssd-detection-evaluation/ap-ranking-ko.svg" alt="TP, FP, TP를 점수순으로 포함하며 정밀도와 재현율을 계산하고 101개 재현율 기준의 정밀도를 평균하는 예제" loading="lazy">
<figcaption>설명용 예제다. TP·FP 판정, 중간 정밀도·재현율, 마지막 AP를 서로 다른 계산 단계로 나누었다.</figcaption>
</figure>

### 왜 재현율을 기준으로 정밀도를 모으는가

실제 정답 집합을 고정하면 재현율은 “찾아야 할 대상 중 얼마만큼 찾았는가”라는 공통 기준이 된다. 조금 찾은 상태부터 거의 전부 찾은 상태까지 정밀도를 비교할 수 있다. 도달하지 못한 재현율 구간에 0이 들어가기 때문에, 일부만 정확하게 찾고 나머지는 놓치는 모델도 점수에 반영된다.

예측의 마지막 TP·FP 개수가 같더라도 순서가 달라지면 중간 결과가 달라진다. 같은 정답 2개에 `FP → TP → TP` 순서라면 AP는 2/3, 약 0.667이다. 최종 정밀도와 재현율은 앞 예제와 같지만, 높은 신뢰도부터 사용할 때의 결과가 다르다.

이 원리의 범위도 알아야 한다. 정답이 1개이고 예측 순서가 `TP → FP`라면 첫 예측만으로 정밀도와 재현율이 모두 1에 도달한다. COCO 보간 AP는 이 경우 1이 될 수 있다. 따라서 <strong>AP가 1이라는 말은 반환된 모든 박스가 정답이라는 뜻이 아니다.</strong> 낮은 표시 기준에서 잘못된 박스가 보이는 현상과 높은 AP가 함께 나타날 수 있다.

## 09. 배치별 AP의 평균이 전체 AP와 다른 이유

앞의 예측과 정답을 두 배치에 나누어 보자. 설명에서는 클래스와 IoU 기준이 각각 하나이므로 AP와 mAP가 같다.

<div class="ssd-table">

| 평가 범위 | 정답 개수 | 점수순 예측 | AP |
| --- | ---: | --- | ---: |
| 배치 1 | 1 | TP(0.9) | 1 |
| 배치 2 | 1 | FP(0.8), TP(0.7) | 0.5 |
| 전체 | 2 | TP(0.9), FP(0.8), TP(0.7) | 약 0.8349835 |

</div>

배치별 AP를 평균하면 `(1 + 0.5) / 2 = 0.75`다. 전체를 모아 계산한 약 0.835와 다르다. 예측이나 정답을 바꾸지 않았고, 무작위성 때문에 생긴 차이도 아니다.

일반적인 평균에서는 같은 개수의 값들을 묶어 평균한 뒤 다시 평균하면 각 원소의 영향력이 같게 유지된다. 하지만 AP는 이미지마다 이미 정해진 점수를 평균하는 연산이 아니다. 전체 정답 수를 분모로 재현율을 구하고, 같은 클래스의 예측을 전체 점수순으로 정렬한 뒤 보간한다. 배치를 합치면 <strong>순위와 재현율의 기준 집합</strong>이 달라진다.

그래서 검증에서는 배치마다 `metric.update(predictions, targets)`로 예측과 정답을 모은다. 모든 배치가 끝난 뒤 `metric.compute()`를 한 번 호출한다. 배치는 모델에 입력을 나누어 전달하는 단위이고, 최종 평가 범위는 전체 검증 데이터다.

이때 서로 다른 이미지의 박스끼리 정답 대응을 찾지는 않는다. 대응은 같은 이미지 안에서 찾고, AP용 순위는 같은 클래스의 여러 이미지 예측을 모아 구성한다. 평가 범위를 합친다는 말과 이미지 경계를 없앤다는 말을 구분해야 한다.

## 10. mAP가 평균하는 축과 반환 항목을 확인한다

지금까지는 클래스 하나와 IoU 기준 하나에 대한 AP를 구했다. <strong>mAP(mean Average Precision)</strong>는 여러 클래스의 AP를 평균한 값이다. 이번 기본 설정은 클래스마다 같은 비중으로 평균하며, 이미지가 많은 클래스에 자동으로 더 큰 비중을 주지 않는다.

`mAP@0.50`은 IoU 0.50에서 클래스별 AP를 평균한 값이다. `mAP@0.75`도 같은 방식이다. 여러 IoU를 지정한 TorchMetrics의 `results["map"]`은 평가 가능한 클래스와 <strong>지정한 IoU 기준들</strong>에 대해 평균한다.[^torchmetrics]

기본 IoU 목록은 `0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95`의 10개다. `0.75`가 마지막 기준은 아니다. 별도로 6개만 지정했다면 `map`은 그 6개를 기준으로 계산되므로, 점수와 목록을 함께 남겨야 한다.

### 내가 정한 변수 이름과 라이브러리 인자 이름은 다르다

`iou_values`는 내가 정한 Python 변수다. `MeanAveragePrecision`이 받아들이는 공식 인자 이름은 `iou_thresholds`다. 평가 객체를 만드는 생성자에 `iou_thresholds=iou_values`로 전달하면 된다. 생성자에는 계산 규칙을, 이후 `update` 메서드에는 실제 예측과 정답을 전달한다. 메서드는 객체에 속한 함수다.

<div class="ssd-table">

| 생성자 설정 | 정하는 내용 |
| --- | --- |
| `box_format="xyxy"` | 좌표를 `[x1, y1, x2, y2]`로 해석 |
| `iou_type="bbox"` | 픽셀 마스크가 아닌 바운딩 박스 평가 |
| `iou_thresholds=iou_values` | 정답 대응에 사용할 IoU 목록 |
| `max_detection_thresholds=[1, 10, 100]` | 평가할 때 사용할 탐지 수 상한들 |
| `class_metrics=True` | 클래스별 결과도 반환 |
| `extended_summary=True` | IoU별 결과를 추출할 수 있는 상세 배열도 반환 |

</div>

IoU 0.95를 목록에 넣었다고 `results["map_95"]`라는 키가 자동으로 생기지는 않는다. 기본 요약에는 `map`, `map_50`, `map_75`가 있다. <strong>계산 가능한 조건과 반환 딕셔너리의 항목 이름은 별개</strong>다. `map_95`가 없다는 이유로 0.95 평가를 지원하지 않는다고 결론 내리면 안 된다.[^torchmetrics]

<div class="ssd-table">

| 결과 항목 | 의미 |
| --- | --- |
| `map` | 지정한 IoU 목록과 클래스를 평균한, 크기 제한 없는 점수 |
| `map_50`, `map_75` | 해당 IoU에서 계산한 mAP |
| `map_small`, `map_medium`, `map_large` | 객체 면적 구간별 mAP |
| `mar_1`, `mar_10`, `mar_100` | 탐지 수 상한을 각각 적용한 평균 재현율 |
| `mar_small`, `mar_medium`, `mar_large` | 객체 면적 구간별 평균 재현율 |
| `map_per_class` | 각 클래스에서 지정 IoU 목록을 평균한 AP |
| `mar_100_per_class` | 각 클래스의 탐지 상한 100 기준 평균 재현율 |
| `classes` | 클래스별 결과와 연결할 클래스 번호 |

</div>

<strong>평균 재현율(mAR, mean Average Recall)</strong>은 정답을 얼마나 찾았는지를 여러 IoU와 클래스에 대해 평균한 것이다. 정밀도 평균이 아니다. 다만 높은 점수의 잘못된 예측이 탐지 상한을 차지하면 올바른 예측이 밀려나 재현율에도 영향을 줄 수 있다.

크기 구간은 이미지 전체 크기가 아니라 객체의 면적에 적용한다. COCO는 `32²`, `96²` 픽셀 면적을 구분 경계로 쓴다. 가로·세로 길이가 각각 32 또는 96이어야 한다는 뜻은 아니다. 별도 `area` 정보가 없다면 박스 면적을 사용한다. `map`의 전체 크기 평가는 작은·중간·큰 객체 점수 세 개를 다시 평균한 값이 아니다.[^coco]

### 0점과 -1은 다르게 다룬다

0은 계산 가능한 항목에서 얻은 유효한 점수다. -1은 해당 항목을 계산하지 않았거나 정의할 수 없다는 표시다. 예를 들어 클래스별 출력을 켜지 않았거나, 해당 크기의 정답이 없거나, 해당 요약에 필요한 IoU가 목록에 없는 경우를 확인해야 한다.

따라서 평균을 구할 때 -1은 제외하지만 0은 유지한다. `classes`도 모델이 알고 있는 모든 클래스의 목록이라고 가정하지 않는다. 평가 입력에서 관찰된 번호와 결과 배열의 대응을 사용한다. TorchMetrics가 모든 입력에서 0을 자동으로 배경 취급하는 것은 아니며, 배경 처리는 현재 SSD와 라벨 구성의 규칙이다.

### SSD의 200개와 평가의 100개는 범위가 다르다

SSD의 기본 반환 상한 200은 한 이미지의 전체 클래스 예측에 적용한다. 기본 COCO 클래스별 평가의 탐지 상한 100은 이미지·클래스별로 적용된다. 그래서 200개가 반환되었다고 평가에서 정확히 100개를 버렸다고 단정할 수 없다.

탐지가 많다는 경고는 계산 실패와 같지 않다. 경고만 없애려고 `max_detection_thresholds`를 바꾸면 평가 조건도 달라진다. 다른 결과와 비교하려면 탐지 상한도 맞추어야 한다.[^torchmetrics][^coco]

## 11. 전체 검증을 누적하고 원하는 IoU별 점수를 꺼낸다

이제 처리 순서를 정한다. 학습된 `model`, 그 모델이 올라간 `device`, 검증용 `val_loader`가 입력이다. 각 배치에서 예측을 만들고, 이미지 순서를 보존한 채 예측·정답을 CPU로 옮겨 평가 객체에 누적한다. 마지막에 전체 결과를 계산한다.

아래 코드는 새 모델을 학습하거나 XML을 읽어 데이터셋을 만드는 코드가 아니다. `val_loader`는 02절의 구조로 이미지와 정답을 제공해야 한다. 이미지는 0~1 범위의 RGB 실수 텐서이며, 정답의 `boxes`는 `(N, 4)` 실수 텐서, `labels`는 `(N,)`의 `torch.int64` 텐서다. 이미지 변환을 했다면 정답 좌표도 그 이미지와 같은 좌표계를 사용해야 한다.

`MeanAveragePrecision`을 불러오지 못하는 Jupyter 환경이라면 다음 설치 셀을 사용한다. `%pip`는 노트북 셀 문법이며 터미널 명령과 다르다. PyTorch와 torchvision은 이미 해당 환경에서 호환되는 조합으로 설치되어 있어야 한다.

~~~python
%pip install torchmetrics pycocotools
~~~

`torchmetrics`는 지표를 누적·조회하는 인터페이스를 제공한다. `pycocotools`는 여기서 선택한 COCO 평가 계산 구현이다. 일반 터미널에서는 같은 Python 환경의 `python -m pip install torchmetrics pycocotools`를 사용한다.

### 먼저 전체 검증 결과를 계산한다

실습에서 따로 확인한 6개 IoU를 그대로 지정한다. 기본 10개 평균이 목적이라면 목록을 기본 10개로 바꾸어야 한다.

~~~python
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

iou_values = [0.50, 0.75, 0.80, 0.85, 0.90, 0.95]
class_names = {number: name for name, number in class_map.items()}

# 실행마다 새 객체를 만들어 이전 검증 데이터가 섞이지 않도록 한다.
metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    iou_thresholds=iou_values,
    max_detection_thresholds=[1, 10, 100],
    class_metrics=True,
    extended_summary=True,
    backend="pycocotools",
)

image_count = 0
model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        assert len(images) == len(targets)
        predictions = model([image.to(device) for image in images])
        assert len(predictions) == len(images)

        # 모델 자체의 후처리 결과를 사용한다. 표시용 필터는 넣지 않는다.
        predictions_cpu = [
            {key: pred[key].cpu() for key in ("boxes", "labels", "scores")}
            for pred in predictions
        ]

        # 다른 데이터셋에 area·iscrowd가 있다면 그 평가 의미도 보존한다.
        targets_cpu = [
            {
                key: target[key].cpu()
                for key in ("boxes", "labels", "area", "iscrowd")
                if key in target
            }
            for target in targets
        ]
        metric.update(predictions_cpu, targets_cpu)
        image_count += len(images)

if image_count == 0:
    raise ValueError("검증 로더가 이미지를 제공하지 않았습니다.")

results = metric.compute()
print("검증 이미지 수:", image_count)
print("사용한 IoU 기준:", iou_values)
print(f"지정한 {len(iou_values)}개 IoU의 mAP: {results['map'].item():.4f}")
~~~

`predictions_cpu`와 `targets_cpu`는 같은 순서의 이미지별 딕셔너리 리스트다. `area`는 객체 면적, `iscrowd`는 군중 객체인지 나타내는 0/1 정보다. 현재 자료에서는 `boxes`, `labels`를 사용하지만, 다른 데이터에 이 선택 필드가 있다면 임의로 버리지 않아야 한다.[^torchmetrics]

CPU로 옮겨도 누적 메모리가 없어지지는 않는다. 예측과 정답을 보관하므로 CPU 메모리가 필요하고 상세 요약도 추가 비용이 든다. 같은 평가 객체를 다음 검증에서 재사용한다면 시작 전에 `metric.reset()`으로 비운다. `compute()`는 파일 저장이나 초기화를 자동으로 하지 않는다.

### 상세 배열의 다섯 축을 먼저 읽는다

`extended_summary=True`이면 `results["precision"]`에 보간된 정밀도 배열이 들어 있다. shape는 `(T, R, K, A, M)`이다. 여기서의 `M`은 앞에서 예측 박스 개수에 붙인 이름과 달리, <strong>이 라이브러리 출력에서 탐지 상한 선택지의 개수</strong>를 뜻한다. 같은 기호라도 정의한 범위가 다르다는 점을 구분한다.[^torchmetrics]

<div class="ssd-table">

| 축 번호 | 기호 | 의미 | 이번 설정의 크기 |
| --- | --- | --- | --- |
| 0 | T | IoU 기준 | 6 |
| 1 | R | 재현율 기준 | 101 |
| 2 | K | 평가 클래스 | 입력에서 결정 |
| 3 | A | 객체 면적 구간 | 4: all, small, medium, large |
| 4 | M | 탐지 수 상한 선택지 | 3: 1, 10, 100 |

</div>

`precision[i, :, :, 0, -1]`은 i번째 IoU, 모든 재현율과 클래스, 전체 크기 구간, 마지막 탐지 상한을 선택한다. `:`는 해당 축 전체, 인덱스 `-1`은 마지막 항목이다. 여기의 인덱스 -1과 미정의를 나타내는 <strong>배열 값</strong> -1은 다른 뜻이다.

선택 후에는 `(R, K)` 배열이 남는다. 여기서 미정의 값만 제외하고 평균하면 해당 IoU의 mAP를 얻는다.

~~~python
map_by_iou = {}

for i, iou in enumerate(iou_values):
    values = results["precision"][i, :, :, 0, -1]
    valid = values[values >= 0]  # 0점은 남기고 미정의 -1만 제외한다.
    score = valid.mean().item() if valid.numel() > 0 else -1.0
    map_by_iou[iou] = score
    print(f"mAP@{iou:.2f}: {score:.4f}")

# classes의 번호를 이름과 연결한다. 클래스가 하나여도 1차원으로 맞춘다.
class_ids = results["classes"].reshape(-1).tolist()
class_ap = results["map_per_class"].reshape(-1).tolist()
class_ar = results["mar_100_per_class"].reshape(-1).tolist()

for class_id, ap, ar in zip(class_ids, class_ap, class_ar):
    print(f"{class_names[class_id]}: AP={ap:.4f}, AR@100={ar:.4f}")
~~~

`enumerate`는 목록의 인덱스와 값을 함께 제공한다. `numel()`은 텐서 원소 개수, `mean()`은 평균이다. `reshape(-1)`은 전체 원소를 한 축으로 펼치며 그 축의 길이를 자동 계산한다. 여기서는 클래스가 하나일 때도 순회할 수 있는 목록으로 맞춘다. 클래스별 출력이 -1이라면 10절의 미정의 규칙을 적용한다.

검산은 새로운 지표를 더 만드는 것보다, 이미 얻은 두 경로의 값이 일치하는지 확인하는 데서 시작한다. 현재 IoU 목록에는 0.50과 0.75가 있으므로 추출한 두 값은 공식 요약과 작은 수치 오차 범위 안에서 같아야 한다.

~~~python
import math

assert math.isclose(map_by_iou[0.50], results["map_50"].item(), abs_tol=1e-6)
assert math.isclose(map_by_iou[0.75], results["map_75"].item(), abs_tol=1e-6)
~~~

`math.isclose`는 두 숫자가 허용 오차 안에서 가까운지 확인한다. `1e-6`은 0.000001이다. 이 검사는 배열에서 꺼낸 축과 요약값의 대응을 확인한다. 데이터 분할이나 정답 자체가 올바른지까지 보장하지는 않는다.

## 12. 기록된 0.5555와 0.3357을 같은 점수로 비교하지 않는다

처음 실습 기록에는 `mAP ≈ 0.5555`, `mAP@0.50 ≈ 0.9610`이 있었다. 첫 값이 기본 설정에서 계산된 값이라면 10개 IoU 기준의 평균이다. 이후 따로 확인한 IoU별 점수는 다음과 같다.

<div class="ssd-table">

| 평가 IoU 기준 | 기록된 mAP |
| --- | ---: |
| 0.50 | 0.9610 |
| 0.75 | 0.6274 |
| 0.80 | 0.3219 |
| 0.85 | 0.0961 |
| 0.90 | 0.0075 |
| 0.95 | 0.0002 |

</div>

이 여섯 값의 합을 6으로 나누면 약 <strong>0.3357</strong>이다. 따라서 위 6개 기준을 지정한 뒤 `map`이 약 0.33으로 나온 결과는 이 기록과 맞는다. 기본 10개 평균과 선택한 6개 평균은 평가 조건이 다르다. 두 숫자의 차이를 곧바로 모델이 나빠졌다는 근거로 사용할 수 없다.

<figure class="ssd-figure">
<img src="/ssd-detection-evaluation/iou-results-ko.svg" alt="실습에 기록된 여섯 IoU 기준의 mAP. 0.50에서 0.9610, 0.75에서 0.6274, 0.80에서 0.3219, 0.85에서 0.0961, 0.90에서 0.0075, 0.95에서 0.0002" loading="lazy">
<figcaption>기존 실습 기록을 그린 그림이다. 0.55·0.60·0.65·0.70의 개별 결과는 제공되지 않아 채우지 않았다. x축 간격은 실제 IoU 차이를 따른다.</figcaption>
</figure>

`mAP@0.50 = 0.9610`을 “이미지의 96.1%를 맞혔다”고 읽는 것도 잘못이다. mAP는 이미지별 정답률이 아니라 클래스별 예측 순위와 정답 대응으로 계산한 AP의 평균이다.

같은 데이터와 예측을 사용했다면, 이 기록에서 확인할 수 있는 사실은 엄격한 IoU 조건에서 점수가 낮아졌다는 것이다. 박스가 어느 방향으로 어긋났는지, 좌표 처리에 오류가 있는지, 특정 클래스에서만 실패하는지는 이 숫자만으로 확정할 수 없다.

나는 이 결과를 다음 조사 순서로 연결한다. 먼저 데이터·클래스 매핑·IoU 목록·탐지 상한이 같은지 확인한다. 그다음 클래스별·객체 크기별 결과를 살펴보고, 차이가 큰 사례의 정답과 예측을 함께 그린다. 특정 원인을 주장하려면 그 원인에 해당하는 실제 사례가 필요하다.

실무에 충분한지도 숫자 하나로 결정할 수 없다. 놓친 객체와 잘못된 탐지 중 어떤 오류가 더 문제가 되는지, 얼마나 정확한 박스가 필요한지에 따라 사용할 신뢰도 기준과 필요한 평가 조건이 달라진다. 기준을 정할 때는 검증 데이터를 사용하고, 독립 테스트는 선택을 마친 뒤 확인해야 한다.

## 13. 정답과 예측을 같은 이미지에 그린다

전체 점수는 여러 사례를 요약한다. 그 값만으로는 클래스 이름을 반대로 표시했는지, 다른 이미지의 예측을 붙였는지, 박스가 정답의 어느 경계에서 어긋났는지 알 수 없다. <strong>시각화(Visualization)</strong>는 이런 구체적인 사례를 조사하는 작업이다. 한두 장의 보기 좋은 그림으로 전체 성능을 증명하는 작업은 아니다.

화면에 그리기 전에 두 가지 변환을 이해해야 한다. Matplotlib은 컬러 이미지를 `(H, W, C)` 축 순서로 받으므로, 현재 `(C, H, W)` 텐서에는 `permute(1, 2, 0)`으로 축 순서를 바꾼다. 숫자 1, 2, 0은 기존 축 번호다. 이미지 픽셀을 새로 추론하거나 박스 좌표를 바꾸는 연산이 아니다.

박스를 그리는 `Rectangle`은 시작 좌표와 너비·높이를 받는다. 저장된 네 경계 좌표 중 시작 좌표는 `(x1, y1)`이고, 너비와 높이는 각각 `x2-x1`, `y2-y1`이다.[^rectangle] `y2-y2`를 넣으면 계산은 실행되어도 높이가 0인 사각형을 만들게 된다.

### 표시 조건은 말로 먼저 정한다

“신뢰도 0.4 이상인 예측을 표시한다”면 건너뛸 대상은 0.4 <strong>미만</strong>이다. `continue`는 현재 반복의 나머지 코드를 건너뛰고 다음 예측으로 이동한다.

다음 두 줄은 예측을 순회하는 `for`문 내부의 조건만 발췌한 것이다. `display_threshold`는 최소 표시 점수를 담는 실수이며, 여기서는 0.4로 정한다. 이 발췌문을 독립 셀로 실행하지 않고, 바로 아래의 전체 함수에서 사용한다.

~~~python
if score.item() < display_threshold:
    continue
~~~

반대로 `if display_threshold < score.item(): continue`라고 쓰면 높은 점수의 예측을 건너뛴다. `if 0.3 < 0: continue`는 예측 점수와 관계없는 항상 거짓인 조건이다. 비교 기호를 고르기 전에 남길 예측과 건너뛸 예측을 말로 구분하면 이런 오류를 찾기 쉽다.

0.4는 여기서 화면 표시를 위한 예시 선택이다. 최적 기준이라는 뜻은 아니다. 낮은 기준에서도 박스가 몇 개만 보인다는 사실은 그 기준을 통과한 예측이 적다는 뜻일 뿐, 정답을 잘 찾았다는 증거는 아니다.

### 반복되는 예측 표시를 함수로 묶는다

아래 함수는 그릴 영역 `ax`, 한 이미지의 예측 딕셔너리 `pred`, 번호를 이름으로 바꾸는 `class_names`, 표시 기준 `display_threshold`를 받는다. 입력 예측을 수정하지 않고 그린 박스 수를 정수로 반환한다.

~~~python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_predictions(ax, pred, class_names, display_threshold):
    count = 0
    assert len(pred["boxes"]) == len(pred["labels"]) == len(pred["scores"])

    for box, label, score in zip(
        pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu()
    ):
        score_value = score.item()
        if score_value < display_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="red", linewidth=2,
        ))
        ax.text(
            x1, y1, f"{class_names[label.item()]} {score_value:.2f}",
            color="red",
        )
        count += 1

    return count
~~~

`ax.add_patch`는 사각형을 그릴 영역에 추가한다. `fill=False`는 안을 채우지 않는다는 뜻이고, `ax.text`는 해당 좌표에 클래스 이름과 점수를 적는다. 함수의 `count`는 <strong>표시한 예측 수</strong>다. 모델이 반환한 수나 실제 정답 객체 수와 같다고 가정하지 않는다.

이제 검증 데이터셋에서 이미지와 정답을 함께 가져오고, <strong>바로 그 이미지</strong>로 추론한 결과를 표시한다. `fig`는 전체 그림, `ax`는 이미지 하나를 그릴 영역이다.

박스 옆에 적을 `GT`는 정답인 Ground Truth의 줄임말이다.

~~~python
image, target = val_dataset[0]
model.eval()
with torch.no_grad():
    pred = model([image.to(device)])[0]

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image.permute(1, 2, 0).cpu())

# 정답은 초록색, 예측은 빨간색으로 구분한다.
for box, label in zip(target["boxes"], target["labels"]):
    x1, y1, x2, y2 = box.tolist()
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor="lime", linewidth=2,
    ))
    ax.text(x1, y1, f"GT: {class_names[label.item()]}", color="lime")

display_threshold = 0.4
shown = draw_predictions(ax, pred, class_names, display_threshold)
ax.set_title(f"Returned: {len(pred['boxes'])}; shown: {shown}")
ax.axis("off")
plt.show()
~~~

`axis("off")`는 축 눈금과 테두리를 숨기며 이미지나 박스의 좌표를 바꾸지 않는다. 현재 이미지는 `to_tensor`로 만든 0~1 범위다. 다른 코드에서 추가 표준화를 적용했다면 표시 전에 그 변환을 되돌려야 할 수 있다.

이 그림에서는 우선 데이터와 표시의 연결을 점검한다. 정답이 머리를 감싸는지, `class_names`가 실제 매핑과 같은지, 이미지와 `pred`가 같은 호출에서 나온 것인지 확인한다. 그다음 겹침 정도를 살펴야 좌표 오류와 모델의 예측 오류를 혼동하지 않을 수 있다.

## 14. 테스트 추론에서도 파일과 예측을 함께 이동시킨다

테스트 목록은 파일 이름으로 구성한다. `test_ids`는 테스트 TXT에서 읽은 이름들의 집합이고, `jpg_ids`는 실제 JPG의 확장자를 뺀 이름들의 집합이다. 집합의 `&`는 두 집합 모두에 있는 원소를 고른다. `sorted`는 그 결과를 정렬된 리스트로 바꾼다.

~~~python
test_names = sorted(test_ids & jpg_ids)

assert len(test_names) > 0
assert len(set(test_names) & set(train_names)) == 0
assert len(set(test_names) & set(val_names)) == 0
~~~

`train_names`, `val_names`는 학습·검증에 사용한 파일 이름 목록이다. 위 검사는 이름 기준의 분할 중복을 찾는다. 내용이 같은 이미지가 다른 파일명으로 저장된 경우까지 찾아주는 검사는 아니다. 이번 테스트에 XML 박스 정답이 없으므로, 테스트 목록을 만들 때 `xml_ids`까지 교집합에 넣지 않는다.

여러 이미지를 그릴 때는 이미지 읽기, 텐서 변환, 추론, 표시를 같은 반복 안에서 마친다. 그러면 이전 이미지의 예측을 새 이미지 위에 그리는 실수를 줄일 수 있다.

~~~python
import random
from PIL import Image
from torchvision.transforms.functional import to_tensor

# 정렬된 앞부분이 같은 품종에 몰릴 수 있어 중복 없이 최대 6장을 고른다.
# 42는 같은 목록에서 같은 사례를 다시 고르기 위한 값이다.
sample_names = random.Random(42).sample(test_names, min(6, len(test_names)))
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax in axes.flat:
    ax.axis("off")

model.eval()
with torch.no_grad():
    for ax, name in zip(axes.flat, sample_names):
        # image_dir는 JPG 폴더를 가리키는 pathlib.Path 객체다.
        with Image.open(image_dir / f"{name}.jpg") as source:
            pil_image = source.convert("RGB")
        image_tensor = to_tensor(pil_image)
        pred = model([image_tensor.to(device)])[0]

        ax.imshow(pil_image)
        shown = draw_predictions(ax, pred, class_names, display_threshold=0.4)
        ax.set_title(f"{name}\nshown: {shown}")

plt.tight_layout()
plt.show()
~~~

`plt.subplots(2, 3)`은 2행 3열의 그림 영역을 만들고, `axes.flat`은 각 영역을 하나씩 순회하게 한다. `zip`은 그 영역과 파일 이름을 같은 순서로 묶는다. 먼저 모든 축 표시를 숨기므로 이미지가 6장보다 적어도 남은 영역에 눈금이 보이지 않는다. `tight_layout`은 여러 그림의 간격을 조정한다.

반복문이 끝난 뒤 `pred`에는 마지막 이미지의 결과가 남는다. 그 값을 첫 이미지에 다시 그리면 잘못된 조합이 된다. 표시한 여섯 사례는 전체 테스트 성능도 아니다. 정답 박스가 없기 때문에 여기서 확인한 것은 파일을 읽고 예측을 생성해 해당 이미지에 표시하는 과정이다.

## 15. 가중치와 점수의 해석 조건을 함께 저장한다

<strong>체크포인트(Checkpoint)</strong>는 나중에 작업을 이어가거나 추론할 수 있도록 저장한 상태다. `model.state_dict()`는 모델의 가중치와 관련 상태를 담은 딕셔너리다. <strong>옵티마이저(Optimizer)</strong>는 기울기를 이용해 가중치를 갱신하는 객체이며, `optimizer.state_dict()`에는 학습을 이어갈 때 필요한 내부 상태가 들어 있다.[^checkpoint]

가중치만 저장하면 클래스 1이 강아지였는지 고양이였는지, mAP가 어떤 IoU 목록의 평균이었는지 알 수 없다. 점수와 그 해석에 필요한 조건을 같이 보관해야 한다.

아래 예제는 앞의 `results`, `map_by_iou`, `iou_values`와 학습에서 사용한 `optimizer`가 있는 상태에서 실행한다. `Path.cwd()`는 현재 작업 폴더를 반환한다. 특정 Colab 경로에 의존하지 않도록 출력 경로를 직접 정하고, 기존 파일을 실수로 덮어쓰지 않게 확인한다.

후처리 설정 중 `topk_candidates`는 NMS에 앞서 클래스별로 남길 후보 수의 상한이다. 최종 반환 상한인 `detections_per_img`와 함께 보관해 추론 조건을 유지한다.

~~~python
from pathlib import Path

output_path = Path.cwd() / "pet_ssd_evaluated.pt"
if output_path.exists():
    raise FileExistsError("이미 같은 이름의 파일이 있습니다. 새 저장 이름을 정하세요.")

summary_keys = [
    "map", "map_50", "map_75",
    "map_small", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100",
    "map_per_class", "mar_100_per_class", "classes",
]

torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "class_map": class_map,
    "iou_thresholds": iou_values,
    "max_detection_thresholds": [1, 10, 100],
    "postprocess": {
        "score_thresh": float(model.score_thresh),
        "nms_thresh": float(model.nms_thresh),
        "detections_per_img": int(model.detections_per_img),
        "topk_candidates": int(model.topk_candidates),
    },
    "validation_names": list(val_names),
    "metrics": {key: results[key] for key in summary_keys},
    "map_by_iou": map_by_iou,
}, output_path)
~~~

이 저장 예제는 해당 실습의 평가 조건을 추가한 것이다. 완전히 같은 학습을 이어가기 위한 모든 상태를 담았다는 뜻은 아니다. 정확한 재현이 필요하면 모델 생성 코드, 전처리·데이터 버전, 패키지 버전, 학습 단계와 무작위 상태 등도 함께 관리해야 한다. `metric.update`나 `loss_sum += loss.item()`이 이런 기록 파일을 자동으로 만들어 주지는 않는다.

복원할 때는 <strong>같은 구조의 모델을 먼저 생성</strong>해야 한다. 아래의 `model`은 배경 포함 3개 클래스를 출력하며 저장 때와 같은 구조로 준비되어 있어야 한다. 이 글에는 원본 모델 생성 코드가 없으므로 다른 SSD 구성으로 임의 대체하지 않는다.

~~~python
checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
model.to(device)

# 이 설정값들은 가중치 텐서가 아니므로 함께 복원한다.
for name, value in checkpoint["postprocess"].items():
    setattr(model, name, value)

class_map = checkpoint["class_map"]
class_names = {number: name for name, number in class_map.items()}
model.eval()
~~~

`map_location="cpu"`는 저장된 텐서를 우선 CPU에 읽도록 한다. 이후 모델을 원하는 `device`로 이동한다. `setattr`는 객체의 해당 이름 속성에 저장한 값을 다시 설정한다. 추론만 할 때는 옵티마이저 상태를 복원하지 않아도 된다. 학습을 이어가려면 같은 설정의 옵티마이저를 준비하고 그 상태도 불러와야 한다.

저장 호출이 끝났다는 사실만으로 복원 검증이 끝나지는 않는다. 저장 전후에 같은 이미지와 전처리를 사용해 박스·라벨·점수가 재현되는지 확인해야 한다. 실제 실습 가중치의 복원은 이번 글을 준비하며 실행한 항목에 포함하지 않았다.

## 16. 실행 예제에서 확인한 것과 남은 것을 구분한다

본문 코드는 판단 순서에 맞춰 나누었다. [examples.py](/ssd-detection-evaluation/examples.py)에는 전체 평가와 그림 표시를 함수로 묶었고, 모델 없이 실행할 수 있는 AP 계산 예제도 넣었다. 파일에는 한국어 주석과 영문 함수 설명이 함께 들어 있다.

다운로드한 파일이 있는 폴더에서 다음 명령을 실행한다. <strong>macOS의 zsh와 Linux의 bash·zsh</strong>에서 사용 중인 Python 환경의 `python`으로 실행할 수 있다.

~~~bash
python examples.py --demo
~~~

`--demo`는 Python 표준라이브러리만 사용하며 다음 값을 확인한다.

~~~text
TP, FP, TP:       AP = 0.8349835
FP, TP, TP:       AP = 0.6666667
Mean batch AP:         0.7500000
Global AP:             0.8349835
TP, FP (one GT):  AP = 1.0000000
~~~

이 함수는 이미 신뢰도순으로 정렬하고 TP·FP를 판정한 결과를 입력받는다. 박스 대응, 같은 점수의 정렬, 군중 객체 등의 처리까지 직접 구현한 COCO 평가기라는 뜻은 아니다. 실제 박스 평가는 TorchMetrics를 사용한다.

`--verify-metric`은 설명용 박스 좌표를 만들고, 실제 TorchMetrics와 `pycocotools` 결과가 위 계산과 맞는지 확인한다. 이 실행에는 `torch`, `torchvision`, `torchmetrics`, `pycocotools`가 필요하다. 그림 함수에는 `Pillow`와 `matplotlib`도 필요하다.

~~~bash
python examples.py --verify-metric
~~~

자신의 학습된 모델에 적용할 때는 `examples.py`를 노트북과 같은 폴더에 두고 다음처럼 함수를 불러온다. 함수의 입력은 본문의 검증 코드와 같고, 반환값은 전체 결과 딕셔너리와 IoU별 mAP 딕셔너리다.

~~~python
from examples import evaluate_detector, print_evaluation

results, map_by_iou = evaluate_detector(
    model, device, val_loader, class_map,
    iou_thresholds=[0.50, 0.75, 0.80, 0.85, 0.90, 0.95],
)
print_evaluation(results, map_by_iou, class_map)
~~~

이 함수에서 `iou_thresholds`를 생략하면 기본 10개 기준을 사용한다. 함수가 끝난 뒤 모델은 평가 모드이므로 다시 학습할 때 `model.train()`을 호출한다. 검증 이미지 표시 함수는 전체 그림, 해당 이미지의 예측, 표시 개수를 반환한다.

~~~python
import matplotlib.pyplot as plt
from examples import show_validation_image, show_test_images

fig, pred, shown = show_validation_image(
    model, device, val_dataset, class_map,
    index=0, display_threshold=0.4,
)
plt.show()

# test_names의 실제 파일 존재와 학습·검증 분리는 14절에서 먼저 확인한다.
fig, selected_names = show_test_images(
    model, device, image_dir, test_names, class_map,
    display_threshold=0.4, seed=42,
)
plt.show()
~~~

게시 전 CPU 환경에서 AP 계산, 두 배치의 누적 평가, 순서에 따른 AP 차이, IoU별 추출과 `map_50`·`map_75`의 일치를 확인했다. 클래스가 하나인 경우와 6개 IoU 목록도 점검했다. 합성 이미지로 검증 그림과 테스트 JPG 읽기·표시를 실행했고, 표시 기준을 적용해도 원래 예측이 바뀌지 않는 것을 확인했다.

검증 환경은 Python 3.12.14, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, TorchMetrics 1.9.0, pycocotools 2.0.11이다. 자세한 실행 범위는 [검증 기록](/ssd-detection-evaluation/verification.txt)에 남겼다. 패키지 버전은 실행 환경의 기록이며 모든 독자가 반드시 같은 버전을 설치해야 한다는 뜻은 아니다. PyTorch와 torchvision 설치는 [공식 설치 안내](https://pytorch.org/get-started/locally/)에서 자신의 운영체제와 장치에 맞는 조합을 확인한다.

이 점검은 평가 코드의 계산과 정보 연결을 확인한 것이다. 실제 학습한 Pet SSD의 성능 재현, CUDA·MPS 실행, 실제 가중치 복원은 실행하지 않았다. 본문에 기록한 0.9610 등의 점수와 위 합성 예제의 실행 결과를 구분해야 한다.

## 17. 다음 데이터에서도 같은 질문으로 시작한다

이번에 필요한 것은 AP 계산 구현을 통째로 외우는 일이 아니다. 새로운 예측 결과를 받았을 때 그 값의 대상, 대응 관계, 평가 조건을 확인할 수 있으면 된다.

<div class="ssd-table">

| 학습 구분 | 확인할 범위 |
| --- | --- |
| 실무 필수 | 이미지·배치·박스 구분, 클래스 매핑, 입력·출력 구조, 전체 누적 평가, 표시와 평가의 분리, 조건과 가중치 저장 |
| 원리만 이해 | 신뢰도 순서가 AP에 영향을 주는 이유, 재현율 기준의 평균, 배치별 AP 평균과 전체 AP의 차이 |
| 필요할 때 참고 | NMS의 세부 구현, 대규모 평가의 메모리 최적화, 군중·무시 정답 처리, 신뢰도 기준 자동 선택 |

</div>

코드를 보기 전에 다음 질문에 답해 보면 이번 학습의 핵심을 점검할 수 있다.

1. `model([image.to(device)])[0]`의 `[0]`은 무엇을 선택하는가?
2. 신뢰도 0.39인 예측을 제거했을 때, 어떤 조건에서 실제 정답이 FN으로 남는가?
3. 배치마다 이미지 수가 같아도 배치별 AP 평균과 전체 AP가 달라질 수 있는 이유는 무엇인가?
4. `map_95`가 없을 때, 계산 미지원과 결과 조회 방식을 어떻게 구분하는가?
5. 그림에 보이는 박스를 줄였는데 이미 계산한 mAP는 그대로인 이유는 무엇인가?

<details>
<summary>생각한 뒤 확인할 답</summary>

1. 입력한 첫 이미지의 예측 딕셔너리를 선택한다. 그 안에는 해당 이미지의 여러 박스가 들어 있다.
2. 실제 정답이 존재하고 그 정답에 대응하는 다른 올바른 예측이 없을 때다. 제거한 예측 자체가 FN으로 바뀌는 것은 아니다.
3. AP는 고정된 이미지별 점수의 산술평균이 아니다. 전체를 합치면 클래스별 예측 순위와 재현율의 분모인 정답 수가 바뀐다.
4. 생성자의 IoU 목록과 반환 딕셔너리의 키를 따로 확인한다. 0.95를 계산 목록에 넣고 상세 `precision` 배열에서 해당 IoU를 선택할 수 있다.
5. 그림 반복문의 필터가 그릴 대상만 고르고 평가 객체에 전달한 예측은 바꾸지 않았기 때문이다.

</details>

이 단계의 종료 기준은 새 이미지와 정답을 받았을 때, 무엇을 한 쌍으로 연결하고 무엇을 셀지 정한 뒤 전체 지표와 개별 사례를 함께 확인하는 것이다. 결과를 기록할 때는 점수만 적지 않고 사용한 데이터와 평가 기준도 남긴다. 그 조건이 유지되어야 다음 모델과의 비교도 같은 질문에 대한 비교가 된다.

## 참고 자료

[^pets]: [Oxford-IIIT Pet 데이터셋 공식 안내](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 머리 영역 박스와 원본 주석의 정의.
[^ssd]: [TorchVision SSD300 공식 API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html) — 입력·정답·예측 구조와 모드별 반환값.
[^ssdsource]: [TorchVision SSD 구현](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/detection/ssd.html) — 손실 정규화, 클래스 점수, NMS와 반환 상한.
[^tensor]: [TorchVision to_tensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.to_tensor.html) · [ToTensor의 입력과 값 범위](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html).
[^autograd]: [PyTorch 기울기 기록과 평가 모드](https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation).
[^coco]: [COCO 공식 평가 구현](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) — 이미지·클래스별 대응, 101개 재현율 기준, 면적·탐지 상한.
[^torchmetrics]: [TorchMetrics MeanAveragePrecision](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html) — 생성자 설정, 출력 항목과 상세 배열의 축.
[^rectangle]: [Matplotlib Rectangle](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html) — 시작 좌표와 너비·높이 인자.
[^checkpoint]: [PyTorch 모델 저장과 불러오기](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
