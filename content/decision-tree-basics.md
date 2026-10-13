+++
title = "결정트리는 어떻게 배우고 예측할까: 지니 불순도부터 가지치기까지"
description = "펭귄 분류 실습으로 모델 생성·학습·예측·평가를 연결한다. 작은 트리의 경로를 읽고, 6개 샘플로 지니·가중평균·엔트로피를 계산하며 깊이 제한, 가지치기와 특성 중요도를 정리한다."
date = 2026-10-13T18:00:00+09:00
slug = "decision-tree-basics"

[extra]
katex = true
toc = true
styles = ["decision-tree-basics/article.css"]
+++

부리 길이와 몸무게 같은 측정값으로 펭귄의 종류를 구분하려고 한다. 모델을 만들고 정확도를 출력하는 코드는 짧다. 하지만 그 코드를 읽으려면 몇 가지 질문을 구분해야 한다. 모델은 무엇을 학습했을까? 정확도는 무엇을 비교한 값일까? 트리에 표시된 숫자는 클래스일까, 노드 번호일까?

결정트리는 **입력의 어느 특징을 어떤 기준값과 비교할지 학습하고, 예측할 때는 그 조건을 따라가 마지막 노드에서 클래스를 정한다.** 이 흐름을 먼저 연결한 뒤, 조건을 고르는 계산과 트리의 크기를 조절하는 방법을 살펴보자.

이 글은 2026년 10월 13일 학습 내용을 정리한 것이다. 사용하는 예제 세 가지는 서로 다른 자료다.

| 예제 | 주어진 정보 | 사용할 목적 |
| --- | --- | --- |
| 펭귄 실습 | 학습 204개, 검증 69개, 특징 4개와 설정별 실험표 | 학습·평가·설정 비교 |
| 작은 트리 `tiny_model` | `x1`을 0.5, 1.5와 비교하는 구조와 출력 | 노드·리프·예측 경로 해석 |
| 가상 자료 A01 | 특징 두 개와 정답이 주어진 샘플 6개 | 지니·엔트로피·분할 후보 계산 |

펭귄 실험표와 작은 트리는 당시 확인한 결과를 사용한다. 마지막의 전체 코드는 **본문에 모두 제시한 A01 자료로 독립 실행**되며, 이 코드의 결과를 펭귄 실험이나 `tiny_model`의 결과로 해석하지 않는다.

## 1. 입력과 정답을 나누고, 학습과 검증을 나눈다

**샘플(sample)**은 관측한 대상 한 개다. 펭귄 표에서는 한 행이 펭귄 한 마리를 나타낸다. **특성 또는 특징(feature)**은 예측에 사용할 입력 정보이고, **클래스(class)**는 분류하려는 정답 종류다. **라벨(label)**은 각 샘플에 붙인 정답 표시다.

라벨을 0·1·2로 기록했다고 해서 0은 오답이고 1은 정답이라는 뜻이 아니다. 각 숫자가 어떤 종류를 나타내는지는 데이터의 라벨 정의로 확인해야 한다.

입력은 `X`, 실제 정답은 `y`로 구분한다. 여기에 자료를 사용하는 목적에 따라 `train`과 `val`을 붙인다.

| 이름 | 어떤 대상의 무엇인가? | 펭귄 실습의 형태 |
| --- | --- | --- |
| `X_train` | 학습할 펭귄들의 입력 특징 | `(204, 4)` |
| `y_train` | 바로 그 펭귄들의 실제 정답 | `(204,)` |
| `X_val` | 검증할 펭귄들의 입력 특징 | `(69, 4)` |
| `y_val` | 바로 그 검증 펭귄들의 실제 정답 | `(69,)` |

**`shape`는 차원별 크기**다. `(204, 4)`는 204행과 4개 컬럼을 가진 2차원 입력이고, `(204,)`는 정답 204개가 담긴 1차원 배열이다. 마지막 쉼표는 두 번째 차원의 크기를 생략했다는 뜻이 아니다.

`X_train`과 `y_train`은 학습용과 검증용으로 나눈 두 자료가 아니다. **같은 학습 대상의 입력과 정답**이다. 모델은 입력값으로 데이터를 나누고, 해당 정답들을 확인해 그 분할이 클래스를 잘 구분하는지 판단한다. 그래서 `fit(X_train, y_train)`에는 둘 다 필요하다.

입력의 첫 번째 행은 정답의 첫 번째 값과 같은 대상을 가리켜야 한다. 개수만 같아서는 부족하다. 정답 배열만 따로 정렬하면 다른 펭귄의 정답을 연결해서 학습할 수 있다. `np.asarray(y_train)`으로 배열을 만드는 작업도 이미 어긋난 대응을 복구해 주지는 않는다.

### 전체 작업은 이 순서로 이어진다

**학습**은 예측에 필요한 값이나 구조를 데이터로 결정하는 과정이다. **검증**은 후보 모델을 비교하기 위한 평가이고, **테스트**는 설정 선택이 끝난 뒤 따로 남겨 둔 자료로 하는 최종 평가다.

```text
예측할 대상과 클래스의 의미를 정한다.
    ↓
X와 y를 준비하고 같은 대상끼리 대응하는지 확인한다.
    ↓
학습용·검증용·최종 테스트용 자료의 역할을 정한다.
    ↓
모델을 생성하고 설정을 지정한다.
    ↓
fit으로 학습 자료에서 분할 조건과 트리 구조를 구한다.
    ↓
predict로 검증 대상마다 예측 클래스를 구한다.
    ↓
accuracy_score로 예측과 검증 정답을 비교한다.
    ↓
후보들의 검증 점수와 실제 트리 크기를 비교한다.
    ↓
한 입력의 예측 경로와 특징별 중요도를 확인한다.
    ↓
설정 선택을 마친 뒤 별도 테스트 자료로 최종 평가한다.
```

검증 입력에는 학습 때와 같은 특징을 같은 컬럼 순서로 넣는다. 전처리가 있다면 학습 자료에서 정한 기준을 검증에도 적용한다. 검증·테스트 자료를 사용해 전처리 기준을 미리 구하면 평가용 정보가 학습 과정에 들어간다.[^pitfalls]

## 2. 모델을 생성하는 것과 학습하는 것은 다르다

`DecisionTreeClassifier`는 분류용 결정트리 객체를 만드는 **Python 클래스**다. 객체는 실제로 생성해서 사용하는 개별 대상이고, 객체에 속한 함수는 **메서드**, 객체가 제공하는 값은 **속성**이라고 부른다. 여기서 Python 클래스는 앞에서 말한 ‘분류할 정답 종류’와는 다른 뜻이다.

| 코드 | 하는 일 | 반환하는 것 |
| --- | --- | --- |
| `DecisionTreeClassifier(...)` | 설정을 가진 모델 객체를 만든다. | 아직 학습하지 않은 모델 |
| `model.fit(X_train, y_train)` | 학습 자료로 트리를 만든다. | 학습한 모델 객체 자신 |
| `model.predict(X_val)` | 검증 입력에 조건을 적용한다. | 샘플별 예측 라벨 배열 |
| `accuracy_score(y_val, val_pred)` | 정답과 예측을 비교한다. | 기본 설정에서는 정확도 숫자 |
| `model.get_depth()` | 실제 최대 깊이를 조회한다. | 정수 |
| `model.get_n_leaves()` | 리프 개수를 조회한다. | 정수 |
| `model.feature_importances_` | 특징별 중요도를 읽는다. | 특징 수만큼의 값이 담긴 배열 |

`model = DecisionTreeClassifier(...)`를 실행했다는 사실만으로 학습이 끝난 것은 아니다. `fit`을 호출해야 학습 자료를 사용한다. `row = model.fit(...)`이라고 쓰면 `row`에도 모델이 들어간다. 평가 결과표의 한 행이 들어가는 것이 아니다.[^classifier]

`feature_importances_`는 값을 읽는 속성이므로 뒤에 `()`를 붙이지 않는다. 이 모델의 예측은 `model(X_val)`이 아니라 `model.predict(X_val)`로 호출한다.

### 학습이 반드시 경사하강법인 것은 아니다

**경사하강법**은 손실함수의 기울기를 이용해 학습할 숫자를 조금씩 갱신하는 방법이다. 이번 결정트리는 경사하강법으로 분할 조건을 학습하지 않는다.

결정트리의 `fit`은 컬럼과 기준값의 조합을 비교하고, 데이터를 나눌 조건을 선택한다. 그 결과 검사할 특징 번호, 숫자 기준값, 자식 노드의 연결 관계, 각 노드의 클래스 구성 등이 저장된다.

“결정트리가 질문을 학습한다”는 말은 한국어 문장을 생성한다는 뜻이 아니다. **어느 특징을 어느 값과 비교하고, 결과에 따라 어느 노드를 사용할지 정한다는 뜻**이다.[^tree]

`random_state=41`은 학습 중 무작위 선택을 재현하기 위한 설정이다. 같은 데이터와 나머지 조건을 유지했을 때 결과를 다시 확인하는 데 사용한다. 41번 학습하거나 정확도를 41로 정하는 숫자는 아니다. 아래 실습에서는 이를 41로 고정하고, 기본 분할 기준인 `criterion="gini"`를 사용한다.

## 3. 모델 하나의 결과와 샘플 하나의 결과를 구분한다

`measure`, `show`, `candidate`, `row`는 scikit-learn이 정해 놓은 특별한 문법이 아니다. `measure`와 `show`는 수업에서 만든 함수이고, 나머지는 코드 작성자가 붙인 변수 이름이다.

`measure(model, name)`은 전달받은 모델을 학습·평가한 뒤 결과를 **딕셔너리**로 반환한다. 딕셔너리는 `"항목 이름": 값` 형태로 결과를 묶는 자료형이다. `show(rows)`는 여러 결과를 표로 표시한다.

```text
candidate 또는 review_model
    모델 객체를 가리킨다.
          ↓ measure가 모델을 학습하고 평가한다.
row 또는 review_row
    모델 하나의 실험 결과를 담은 딕셔너리다.
          ↓ rows.append(row)로 결과 하나를 추가한다.
rows
    여러 실험 결과를 담은 리스트다.
          ↓ show(rows)가 표를 표시한다.
결과표의 한 행
    펭귄 한 마리가 아니라 모델 설정 하나의 실험 결과다.
```

`"depth=2"`라는 문자열은 실험 이름이다. 실제 깊이 설정은 모델에 전달한 `max_depth=2`가 담당한다. 이름만 `"depth=4"`로 바꾸어도 모델 설정은 바뀌지 않는다.

`measure`가 자료를 어디서 받는지도 확인해야 한다. 함수 밖의 `X_train` 등을 사용하도록 정의했다면 바로 그 변수들을 읽는 것이다. 라이브러리가 학습 데이터를 자동으로 찾아오는 것이 아니다. 이 글의 전체 코드에서는 네 자료를 `compare_settings`에 전달하고, 그 함수 안의 `measure`가 이를 사용하도록 정의한다.

또한 `measure` 안에서 이미 `fit`을 호출하므로 바로 앞에서 같은 모델을 한 번 더 학습할 필요는 없다. 이 결정트리에 다시 `fit`을 호출하면 이전 트리에 학습을 이어 붙이는 것이 아니라 새로 학습한다.

### 정확도 함수는 모델을 몰라도 계산할 수 있다

`accuracy_score(y_val, val_pred)`는 **실제 정답과 이미 만들어진 예측값**을 받는 함수다. 모델 객체를 받지 않는다. 기본 설정의 정확도는 다음 비율이다.[^accuracy]

$$
\text{정확도}=\frac{\text{맞힌 개수}}{\text{평가한 전체 개수}}
$$

분모로 나누는 이유는 맞힌 개수를 전체에 대한 비율로 바꾸기 위해서다. 검증 자료 69개 중 67개를 맞혔다면 다음과 같다.

$$
\frac{67}{69}\approx0.971014\approx97.10\%
$$

코드에서 정답 배열과 예측 배열을 `==`로 비교하면 같은 위치마다 `True` 또는 `False`가 나온다. 이를 합산할 때 `True`는 1, `False`는 0으로 계산되므로 맞힌 개수를 얻는다.

학습 정확도는 학습했던 입력을 다시 예측한 점수이고, 검증 정확도는 해당 `fit`에 사용하지 않은 검증 입력을 예측한 점수다. `gap = train_accuracy - validation_accuracy`로 차이를 기록할 수 있다. `gap=0.05`는 **5%포인트 차이**이지 상대적으로 5% 감소했다는 뜻이 아니다.

차이가 있다는 사실 하나만으로 과적합의 원인을 확정하지는 않는다. 반대로 두 정확도가 모두 낮다면 차이가 작아도 좋은 모델이라고 할 수 없다. 또한 `accuracy_score`가 모델 종류를 알아내 지니나 MSE로 자동 변경하는 것은 아니다. 평가 지표는 예측할 대상과 평가 목적에 맞게 선택해야 한다.

## 4. 이진트리의 ‘이진’은 클래스 수가 아니라 자식 수다

**노드(node)**는 조건이나 클래스 분포를 가진 트리의 구성 요소다. 시작 노드는 **루트(root)**, 조건을 검사해 다음 노드를 선택하는 곳은 **분할 노드**, 자식이 없어 예측을 마치는 곳은 **리프(leaf)**라고 한다.

**이진트리**는 각 노드가 최대 두 자식과 연결되는 구조다. 이번 결정트리에서는 분할 노드에 자식이 둘 있고, 리프에는 없다. 정답 종류가 두 개인 ‘이진 분류’와는 다른 개념이다. 정답이 세 종류 이상이어도 이진트리로 분류할 수 있다.

작은 트리 `tiny_model`의 구조를 보자.

```text
노드 0: x1 <= 0.5 ?                           깊이 0
│
├─ 참   → 노드 1: 클래스 0을 예측 [리프]        깊이 1
│
└─ 거짓 → 노드 2: x1 <= 1.5 ?                 깊이 1
             │
             ├─ 참   → 노드 3: 클래스 0 [리프]  깊이 2
             │
             └─ 거짓 → 노드 4: 클래스 1 [리프]  깊이 2
```

노드 1에서 노드 3과 4로 나누는 구조가 아니다. 노드 1은 이미 리프이고, 노드 2가 다시 나뉜다. 리프는 노드 1·3·4로 총 세 개이며, 같은 깊이에 있을 필요도 없다.

| 구분 | 이 트리의 값 | 뜻 |
| --- | --- | --- |
| 노드 ID | 0·1·2·3·4 | 노드를 구별하는 번호 |
| 예측 클래스 | 0·1 | 분류할 정답 종류 |
| 리프 수 | 3 | 자식 없는 최종 노드의 개수 |
| 실제 최대 깊이 | 2 | 루트부터 가장 깊은 리프까지의 연결 개수 |
| 학습 샘플 수 | 그림만으로 알 수 없다. | 학습에 사용한 대상의 수 |

서로 다른 리프가 모두 클래스 0을 예측할 수 있다. 따라서 **리프 수는 클래스 수도, 데이터 수도 아니다.** `leaf 3`은 ‘깊이 3’이 아니라 ‘노드 ID가 3인 리프’라는 뜻이다.

### 깊이는 루트의 0부터 센다

깊이는 루트에서 해당 노드까지 거친 **연결의 개수**다. 루트에서는 아직 연결을 건너지 않았으므로 깊이가 0이다.

`max_depth=2`는 루트까지 포함하면 최대 세 층, `max_depth=3`은 최대 네 층을 허용한다. 깊이를 한 단계 늘릴 때 각 노드가 최대 둘로 나뉠 수 있으므로 가능한 최대 리프 수는 `1 → 2 → 4 → 8`처럼 증가한다. 최대 깊이가 $d$라면 리프 수의 상한은 $2^d$다.

`max_depth`는 **허용 상한**이고, `get_depth()`는 **실제 결과**다. 상한을 4로 정했다고 반드시 깊이 4까지 만들어지는 것은 아니다. `max_depth=None`도 끝없이 나누라는 뜻이 아니다. 깊이 상한을 지정하지 않을 뿐, 순수한 노드나 유효한 분할의 부재, 최소 샘플 수 같은 중단 조건은 남는다.[^classifier]

노드 깊이가 0부터라는 사실과 설정값의 허용 범위도 다르다. `max_depth`에는 양의 정수 또는 `None`을 사용한다. 루트만 있는 트리의 실제 깊이는 0일 수 있지만, 그것이 `max_depth=0`을 지정한다는 뜻은 아니다.

## 5. 예측 클래스와 예측 경로를 따로 읽는다

`tiny_model`에 `x1=1`인 입력을 넣어 보자. 당시 이 입력은 `[[1, 1]]`이었고, 아래 경로에서 실제로 검사하는 특징은 `x1`이다.

| 순서 | 적용하는 조건 | 판정과 다음 작업 |
| --- | --- | --- |
| 1 | 노드 0에서 `1 <= 0.5` | 거짓이므로 오른쪽 노드 2를 사용한다. |
| 2 | 노드 2에서 `1 <= 1.5` | 참이므로 왼쪽 노드 3을 사용한다. |
| 3 | 노드 3은 리프다. | 저장된 클래스 분포로 예측을 끝낸다. |

‘오른쪽으로 간다’는 말은 원본 행을 다른 파일로 옮긴다는 뜻이 아니다. **다음 계산에 사용할 노드를 선택한다는 뜻**이다. 왼쪽은 클래스 0, 오른쪽은 클래스 1이라는 고정 규칙도 없다. 방향은 조건의 결과이고, 예측 클래스는 도착한 리프의 학습 정보로 정한다.

이 입력의 마지막 기록은 다음과 같았다.

```text
leaf 3, class distribution {0: 0.5, 1: 0.5}, prediction 0
```

노드 3의 학습 자료에서 두 클래스의 비중이 각각 50%였다는 뜻이다. **이 숫자는 검증 정확도 50%가 아니다.** 비율만으로 리프에 학습 샘플이 몇 개 있었는지도 알 수 없다. 동률이면 이 분류기는 `classes_`에 저장된 순서에서 앞쪽 클래스를 선택하므로, 여기서는 0을 예측한다. 실제 정답이 1이면 이 예측은 틀린 것이다.[^classifier]

조건과 입력만 있으면 도착할 리프는 찾을 수 있다. 하지만 그 리프의 예측 정보까지 주어지지 않았다면 최종 클래스는 확정할 수 없다.

### 조회 기능마다 반환하는 것이 다르다

| 기능 | 반환하거나 보여 주는 것 |
| --- | --- |
| `export_text(model, ...)` | 트리 전체의 조건과 리프 예측을 표현한 문자열 |
| `model.predict(X)` | 각 입력의 예측 클래스 |
| `model.apply(X)` | 각 입력이 도착한 리프의 노드 ID |
| `model.decision_path(X)` | 각 입력이 어떤 노드를 사용했는지 나타내는 행렬 |
| 직접 만든 `path_for(...)` | 입력 한 행에 적용된 조건과 리프 정보를 정리한 기록 |

`export_text`의 기본 출력은 특정 입력의 실행 기록이나 학습 샘플 개수표가 아니다. `apply`가 반환한 `[3]`도 클래스 3을 예측했다는 뜻이 아니다.[^structure]

`path_for`는 파일 경로를 찾는 표준 함수가 아니라 경로 설명을 위해 만든 함수다. 모델이 저장한 특징 번호에 이름을 붙이도록 `feature_names`를 전달한다. 이름 목록은 학습 컬럼 순서와 같아야 한다. 출력할 이름을 전달한다고 모델의 분할 조건이 바뀌지는 않는다.

원래 조건 `1 <= 0.5`는 거짓인데 기록에 `1 > 0.5`가 나오는 이유도 출력 방식에 있다. `relation = "<=" if value <= threshold else ">"`가 **실제로 선택한 방향**을 표시하기 때문이다. 모델이 기준값을 바꾼 것이 아니다.

이 설명은 결측값이 없는 입력을 기준으로 한다. `NaN`은 결측 등을 나타내는 특수 값인데, 단순한 `<=` 비교만으로 모델의 실제 결측값 분기를 설명해서는 안 된다. 표시 숫자를 반올림했다면 기준값 바로 근처에서는 표시값만으로 판단하지 않아야 한다. 마지막 코드의 `path_for`는 이런 혼동을 막기 위해 결측값 없는 한 행만 받고, 계산한 리프가 `apply`의 결과와 일치하는지도 확인한다.

### 한 행을 고를 때는 위치와 차원도 확인한다

전체 예측 배열 `val_pred`가 있다면 같은 위치의 값을 선택해 한 대상의 예측을 볼 수 있다. 모델을 재학습하거나 검증 행의 순서를 바꾸지 않았다면 다시 예측할 필요는 없다.

| 선택 | 반환 형태 | 뜻 |
| --- | --- | --- |
| `X_val.iloc[1]` | Series, `(4,)` | 두 번째 행의 값들 |
| `X_val.iloc[1:2]` | DataFrame, `(1, 4)` | 두 번째 행 하나를 2차원으로 유지 |
| `val_pred[1:2]` | NumPy 배열, `(1,)` | 같은 대상의 예측값 하나를 배열로 유지 |

`.iloc`은 정수 **위치**로 선택한다. 위치는 0부터 세고, 슬라이스는 시작을 포함하되 끝은 제외한다. 그래서 `[1:2]`가 두 번째 한 행이다. 이 실습의 `predict`는 `(샘플 수, 특징 수)` 형태의 2차원 입력을 받으므로 한 행도 `(1, 4)`로 유지한다.[^indexing]

정수 인덱스 라벨을 가진 pandas Series의 `y_val[1]`은 라벨이 1인 값을 고른다. 두 번째 위치가 필요하면 `y_val.iloc[1]`을 사용한다. NumPy 배열의 `[1]`은 두 번째 위치다. 원본 행 번호와 현재 배열의 위치가 반드시 같은 것은 아니다.

### decision_path의 행은 샘플이고, 열은 노드 ID다

사용한 노드는 1, 사용하지 않은 노드는 0이다. 방금 입력이 `0 → 2 → 3`을 사용했으므로 한 행은 다음과 같다.

```text
노드 ID       0  1  2  3  4
사용 여부    [1, 0, 1, 1, 0]
행렬 shape   (1, 5)
```

0이 많은 행렬을 **희소 행렬**이라고 한다. `decision_path`는 0이 아닌 값과 위치를 중심으로 저장하는 CSR 형식을 반환한다. 한 샘플만 조회했다면 `.indices`에서 그 행에 기록된 노드 ID들을 볼 수 있다. 여러 샘플의 `.indices` 전체를 한 입력의 경로로 읽으면 안 된다. 희소 행렬의 열 위치 목록 자체가 일반적인 시간순 실행 로그인 것도 아니다.[^csr]

`.toarray()`는 일반 배열로 펼치는 메서드다. 작은 예제를 확인할 때는 편하지만 큰 행렬을 무작정 펼치면 메모리를 많이 사용할 수 있다. 자식 연결은 `model.tree_.children_left[node_id]`와 `model.tree_.children_right[node_id]`로 조회한다. 자식 ID가 `-1`이면 그쪽 자식이 없다. `children_right`라는 독립 변수를 만들지 않았다면 `children_right[0]`만 쓰는 코드는 실행할 수 없다.

## 6. 지니 불순도는 클래스가 섞인 정도를 계산한다

모델이 분할 조건을 고르려면 나눈 결과를 비교할 기준이 필요하다. **지니 불순도(Gini impurity)**는 한 노드의 실제 정답이 한 종류에 모였는지, 여러 종류로 섞였는지를 나타낸다. 예측과 실제 정답을 비교한 오답률이 아니다.

기호의 범위를 먼저 정하자. $K$는 클래스 수, $k$는 0부터 $K-1$까지의 클래스 번호다. $n_k$는 현재 노드의 클래스 $k$ 샘플 수이고, $n$은 그 노드의 전체 샘플 수다. 여기서는 $n>0$이고 샘플·클래스 가중치를 따로 지정하지 않는다.

클래스 비율 $p_k$와 지니 $G$는 다음과 같다. $\sum$는 지정된 항들을 더하라는 기호다.

$$
p_k=\frac{n_k}{n},\qquad
G=1-\sum_{k=0}^{K-1}p_k^2
$$

**더하는 단위는 샘플 한 개가 아니라 클래스 한 종류**다. 클래스가 둘이면 두 항, 셋이면 세 항을 더한다. 이진 분류에만 쓰는 공식이 아니다.[^tree]

### 비율을 그냥 더하지 않고 제곱하는 이유

클래스 비율을 그냥 더하면 항상 1이다. 한 종류가 대부분인 경우와 반씩 섞인 경우를 구별할 수 없다.

제곱은 자기 자신을 곱하는 연산이다. $0<p<1$에서는 $p^2<p$이므로 확률 자체는 더 작아진다. $9/10=90/100$인데 이를 제곱하면 $81/100$이다. 분자뿐 아니라 분모도 달라진다는 점을 같이 봐야 한다.

하지만 서로 다른 양수의 **상대적인 크기비**는 제곱 뒤 더 벌어진다. $a>b>0$이면 다음이 성립한다.

$$
\frac{a^2}{b^2}=\left(\frac{a}{b}\right)^2>\frac{a}{b}
$$

$a/b$가 1보다 크기 때문에 이를 제곱한 비율은 원래 비율보다 크다. 이는 두 값의 절대적인 차이가 언제나 커진다는 주장과는 다르다.

비율의 합이 1로 고정된 상황에서 비교해 보자.

| 클래스 비율 | 비율의 제곱합 | 1에서 뺀 지니 |
| --- | --- | --- |
| `[0.9, 0.1]` | $0.81+0.01=0.82$ | $0.18$ |
| `[0.5, 0.5]` | $0.25+0.25=0.50$ | $0.50$ |

한 종류에 집중된 분포에서 제곱합이 더 크다. 이를 1에서 빼면 혼합이 적을수록 작은 점수가 나온다. 모든 샘플이 같은 클래스이면 지니는 0이다. 정해진 $K$개 클래스의 비율이 모두 같을 때 최대가 되며, 그 값은 $1-K(1/K)^2=1-1/K$다. **0.5는 두 클래스일 때의 최대값이지 모든 분류 문제의 최대값이 아니다.**

**MSE(평균제곱오차)**는 예측 오차를 제곱해 평균낸 값이다. MSE와 지니는 모두 제곱의 성질을 사용하지만 제곱하는 대상이 다르다. MSE는 예측 오차를, 지니는 클래스 비율을 제곱한다. 불순도를 표현하는 방법이 반드시 제곱 하나뿐인 것도 아니다. 뒤에서 볼 엔트로피는 같은 목적에 사용하는 다른 기준이다.

### 가상 샘플 6개로 계산한다

A01에서 `x1`, `x2`는 숫자 특징 이름이고, `label`의 0과 1은 두 정답 종류다. 별도의 물리적 단위나 펭귄 종류를 부여한 자료는 아니다.

| 샘플 번호 | x1 | x2 | 실제 클래스 |
| --- | --- | --- | --- |
| 1 | 1 | 1 | 0 |
| 2 | 2 | 2 | 0 |
| 3 | 3 | 4 | 0 |
| 4 | 4 | 3 | 1 |
| 5 | 5 | 5 | 1 |
| 6 | 6 | 6 | 1 |

조건 `x2 <= 2.5`를 적용하면 왼쪽에는 샘플 1·2, 오른쪽에는 샘플 3·4·5·6이 들어간다. 클래스 0과 1의 개수를 순서대로 적으면 왼쪽은 `[2, 0]`, 오른쪽은 `[1, 3]`이다.

오른쪽 전체 개수인 4와 클래스별 개수 `[1, 3]`은 다른 정보다. 지니를 계산하려면 각 클래스의 개수를 알아야 한다.

$$
G_R
=1-\left[\left(\frac14\right)^2+\left(\frac34\right)^2\right]
=1-\left(\frac1{16}+\frac9{16}\right)
=\frac6{16}
=0.375
$$

$1/4$과 $3/4$은 클래스 0과 1의 비율이지 맞을 확률과 틀릴 확률로 정의한 값이 아니다. 클래스 1 샘플이 세 개라는 정보는 이미 $3/4$의 분자에 들어 있다. $(3/4)^2$을 세 번 더하면 같은 클래스 비율을 중복 계산한다.

왼쪽은 두 샘플 모두 클래스 0이므로 다음과 같다.

$$
G_L=1-\left[\left(\frac22\right)^2+\left(\frac02\right)^2\right]=0
$$

## 7. 자식의 지니를 합칠 때는 샘플 수만큼 반영한다

왼쪽과 오른쪽의 크기가 다른데 지니 두 개를 더해 2로 나누면, 샘플 두 개가 있는 쪽과 네 개가 있는 쪽에 같은 비중을 준다. **가중평균**은 각 값이 차지해야 할 비중을 반영하는 평균이다.

$n_L$, $n_R$를 왼쪽과 오른쪽 샘플 수, $n=n_L+n_R$를 부모 샘플 수라고 하자. 자식 지니는 $G_L$, $G_R$다. 각 지니를 그쪽 샘플 수만큼 반영한 뒤 전체 샘플 수로 나누면 다음 식이 된다.

$$
\text{가중 지니}
=\frac{n_LG_L+n_RG_R}{n}
=\frac{n_L}{n}G_L+\frac{n_R}{n}G_R
$$

나눗셈은 부모에서 각 자식이 차지하는 비중을 구한다. 곱셈은 그 비중만큼 자식의 지니를 반영한다. 덧셈은 두 자식의 기여를 합친다.

A01에서는 왼쪽 2개와 오른쪽 4개이므로 다음과 같다.

$$
\text{가중 지니}
=\frac26\times0+\frac46\times0.375
=0.25
$$

계산 순서는 **클래스별 개수 → 클래스별 비율 → 자식별 지니 → 자식의 데이터 비중 → 가중평균**이다. 오른쪽 항에 $(1/16+9/16)$을 그대로 넣으면 지니가 아니라 비율의 제곱합을 넣은 것이다. 먼저 $1-(1/16+9/16)$을 계산해야 한다.

비교 대상으로 `x2 <= 3.5`를 사용하면 왼쪽은 `[2, 1]`, 오른쪽은 `[1, 2]`다. 양쪽 지니는 모두 $1-(4/9+1/9)=4/9$이고 각 자식에 세 샘플씩 있으므로 가중평균도 $4/9\approx0.4444$다.

같은 부모에서 비교하면 **0.25인 `x2 <= 2.5`가 0.4444인 후보보다 혼합을 더 많이 줄인다.** 이것은 학습 중 분할 기준의 비교이지 검증 정확도를 비교한 결과가 아니다.

## 8. 무한한 실수 대신, 서로 다른 분할을 비교한다

숫자가 연속적이라면 2.1, 2.11, 2.111처럼 기준값을 계속 바꿀 수 있다. 결정트리는 이런 모든 실수를 시험하지 않는다. **현재 학습 자료를 서로 다르게 나누는 경우**를 비교한다.

여기서는 수치형 특징, 결측값 없는 자료, 기본 `splitter="best"`를 설명한다. A01의 `x2`를 정렬하면 `[1, 2, 3, 4, 5, 6]`이다. `x2 <= 2.1`과 `x2 <= 2.9`는 모두 값 1·2를 왼쪽에 넣으므로 같은 학습 분할과 같은 불순도를 만든다.

서로 다른 인접값이 $a<b$일 때, 조건 $x\le t$에서 $a\le t<b$인 기준값들은 같은 학습 분할을 만든다. 보통 $(a+b)/2$를 대표로 사용한다. 서로 다른 값이 $m$개라면 양쪽이 비지 않는 수치 분할의 경계는 최대 $m-1$개다. 숫자의 최솟값과 최댓값 사이가 얼마나 넓은지가 후보 수를 정하지 않는다.

단, **학습 분할이 같다고 모든 미래 입력의 예측까지 같은 것은 아니다.** 2.1과 2.9 사이에 새 입력값이 들어오면 기준값에 따라 다른 쪽으로 갈 수 있다.

### 정렬할 때 정답도 같은 샘플을 따라가야 한다

A01을 `x2`로 정렬하면 샘플 순서는 `1, 2, 4, 3, 5, 6`이다. 정답도 이에 맞춰 `[0, 0, 1, 0, 1, 1]`이 된다. 숫자 컬럼만 정렬하고 정답을 원래 순서에 두면 다른 자료를 계산하는 셈이다.

처음에는 왼쪽 `[0, 0]`, 오른쪽 `[3, 3]`으로 시작한다. 경계를 오른쪽으로 옮길 때 새로 왼쪽에 들어오는 샘플 하나의 클래스 개수만 갱신하면 다음 결과를 얻는다.

| 기준값 | 왼쪽 클래스별 개수 | 오른쪽 클래스별 개수 | 가중 지니 |
| --- | --- | --- | --- |
| 1.5 | `[1, 0]` | `[2, 3]` | 0.4000 |
| 2.5 | `[2, 0]` | `[1, 3]` | 0.2500 |
| 3.5 | `[2, 1]` | `[1, 2]` | 0.4444 |
| 4.5 | `[3, 1]` | `[0, 2]` | 0.2500 |
| 5.5 | `[3, 2]` | `[0, 1]` | 0.4000 |

후보마다 전체 행의 정답을 처음부터 다시 세지 않아도 되므로 계산을 줄인다. 클래스별 점수를 계산하는 비용까지 사라지는 것은 아니다. 실제 구현에는 추가 후보 제외 조건과 계산 최적화가 있고, 정렬도 학습 전체에서 단 한 번만 수행한다는 뜻은 아니다.[^implementation]

위 표는 **`x2` 하나의 후보 비교**다. 전체 모델은 검토한 ‘컬럼 + 기준값’ 조합을 비교한다. 실제로 A01에는 `x1 <= 3.5`라는 조건도 있다. 이 조건은 `[3, 0]`과 `[0, 3]`으로 나누므로 가중 지니가 0이다. 따라서 `x2`에서 찾은 0.25를 전체 컬럼 중 최선이라고 말해서는 안 된다.

현재 노드에서 조건을 고른 뒤 자식에서도 같은 과정을 반복한다. 미래의 모든 트리 구조를 전부 비교해 전체 최적해를 보장하는 방식은 아니다.[^tree]

## 9. 엔트로피는 지니를 다시 평가하는 단계가 아니다

**엔트로피(entropy)**도 클래스 구성의 불확실성을 측정하는 지표다. `criterion="gini"`이면 지니를 사용하고, `criterion="entropy"`이면 엔트로피 감소를 기준으로 분할을 비교한다. 지니를 계산한 뒤 엔트로피로 채점하는 두 단계가 아니다.

**로그**는 지수의 역연산이다. $\log_2(p)=r$는 $2^r=p$라는 뜻이다. 여기서는 확률 $p$인 결과의 **정보량**을 다음과 같이 정의한다.

$$
I(p)=-\log_2(p),\qquad 0<p\le1
$$

확실한 결과는 $p=1$이므로 정보량이 0이다. 드문 결과일수록 정보량이 커진다. $0<p<1$에서는 로그가 음수이므로 마이너스를 붙여 음이 아닌 값으로 바꾼다. 독립적인 결과들의 결합 확률을 곱으로 표현할 때, 로그는 그 정보량을 합으로 표현하게 해 준다.

$$
-\log_2(pq)=-\log_2(p)-\log_2(q)
$$

엔트로피 $H$는 각 클래스의 정보량을 그 클래스 비율로 **가중평균**한 값이다. 앞에서 정의한 $p_k$와 클래스 번호를 그대로 사용하면 다음과 같다.[^entropy]

$$
H=\sum_{k=0}^{K-1}p_kI(p_k)
=-\sum_{k=0}^{K-1}p_k\log_2(p_k)
$$

로그는 확률을 정보량으로 바꾸고, 비율을 곱해 더하는 연산은 그 정보량의 평균을 구한다. $p_k=0$인 항의 기여는 0으로 처리하며 `log2(0)`을 그대로 계산하지 않는다. 엔트로피는 단순히 ‘지니의 제곱을 로그로 바꾼 식’이 아니다.

### 정보 이득은 분할 전후의 차이다

**정보 이득(information gain)**은 부모 엔트로피에서 자식들의 가중평균 엔트로피를 뺀 값이다. 분할 후 불확실성이 많이 줄수록 크다.

A01의 부모는 `[3, 3]`이다. 두 클래스 비율이 $1/2$이고 $\log_2(1/2)=-1$이므로 부모 엔트로피는 1이다. `x2 <= 2.5`의 왼쪽 `[2, 0]`은 엔트로피가 0이고, 오른쪽 `[1, 3]`은 다음과 같다.

$$
H_R=-\left[\frac14\log_2\left(\frac14\right)+\frac34\log_2\left(\frac34\right)\right]
\approx0.811278
$$

가중 엔트로피는 $(2/6)\times0+(4/6)\times0.811278\approx0.540852$이고, 정보 이득은 $1-0.540852\approx0.459148$이다.

| 조건 | 가중 지니 | 가중 엔트로피 | 정보 이득 |
| --- | --- | --- | --- |
| `x2 <= 3.5` | 0.4444 | 0.9183 | 0.0817 |
| `x2 <= 2.5` | 0.2500 | 0.5409 | 0.4591 |

분할 후 지니와 엔트로피는 작을수록, 정보 이득은 클수록 좋다. 이번에는 두 기준이 같은 후보를 선호한다. 다른 자료에서도 항상 같은 분할을 선택한다는 뜻은 아니다.

## 10. 깊이 제한과 가지치기는 트리를 줄이는 방식이 다르다

**하이퍼파라미터**는 학습 방식과 제한 조건을 지정하는 설정이다. `max_depth`와 `ccp_alpha`가 여기에 해당한다. 반면 실제 분할 특징, 기준값, 자식 연결과 클래스 분포는 `fit`이 데이터로 결정한다.

`max_depth`는 트리를 만들 때 허용할 깊이를 제한한다. **가지치기(pruning)**는 어떤 분할 노드 아래의 구조를 제거하고 그 노드를 리프로 만드는 작업이다. 원본 학습 데이터를 삭제하는 작업이 아니다.

`ccp_alpha`는 **비용-복잡도 가지치기**에 사용하는 설정이다. 학습 자료의 불순도뿐 아니라 리프 수에도 비용을 부과한다. $T$는 비교할 트리, $R(T)$는 리프 불순도를 전체 학습 자료에서 차지하는 비중으로 가중 합산한 값, $L(T)$는 리프 수다. $\alpha$는 리프 하나당 더하는 복잡도 비용이다.

$$
\text{비용}(T)=R(T)+\alpha L(T)
$$

리프마다 같은 비용 $\alpha$를 더하므로 $\alpha L(T)$라는 곱셈이 생긴다. 여기에 학습 자료의 불순도 비용을 더해 트리를 비교한다. 같은 시작 트리에서 $\alpha$를 키우면 리프가 많은 구조에 더 큰 비용이 붙으므로 더 단순한 트리를 선호하게 된다.[^pruning]

`ccp_alpha=0`은 기본적으로 이 방식의 가지치기를 수행하지 않는다. `0.01`은 가지를 1% 자르거나 정확도를 1% 낮추라는 뜻도, 학습률도 아니다. 구조는 특정 경계값에서 바뀌므로 알파를 조금 늘려도 결과가 그대로일 수 있다.

**`fit` 안에서 검증 정확도를 보고 자를 가지를 선택하는 것이 아니다.** 가지치기는 위 비용 기준으로 수행하고, 어떤 알파가 나은지 검증 자료로 비교하는 작업은 모델 바깥의 실험이다.

## 11. 실험표에서 무엇을 얻고 무엇을 잃었는지 읽는다

반복문에서 `[1, 2, 4, None]`은 전체 후보 목록이고, `depth`는 이번 반복에서 꺼낸 후보 하나다. 따라서 `max_depth=depth`를 전달한다. 리스트 전체를 `max_depth`에 넣는 코드가 아니다.

반복마다 새 모델을 만든 뒤 `measure`로 결과를 얻고 `rows.append(row)`로 저장한다. `append`는 기존 리스트를 바꾸고 반환값은 `None`이므로 `rows = rows.append(row)`로 쓰면 안 된다. 결과 목록의 원소 수는 실험 수이지 검증 샘플 수가 아니다.

### 깊이 상한을 바꾼 펭귄 실험

다음은 같은 학습·검증 자료와 `random_state=41`을 사용한 당시 결과다.

| 깊이 상한 | 실제 깊이 | 리프 수 | 검증 정확도 |
| --- | --- | --- | --- |
| 1 | 1 | 2 | 0.782609 |
| 2 | 2 | 4 | 0.913043 |
| 4 | 4 | 9 | 0.971014 |
| `None` | 4 | 9 | 0.971014 |

시험한 후보 중 상한 4와 `None`이 가장 높은 검증 정확도를 보였다. 깊이 제한을 없애도 실제 트리는 깊이 4에서 멈췄다. **이 자료에서는 더 큰 깊이를 허용해도 더 깊어지지 않았다는 관찰**이지, 모든 데이터에서 깊이 4를 넘는 것이 무의미하다는 결론이 아니다.

기본 비가중 단일 정답 분류에서 학습 정확도가 정확히 100%라면 각 리프에 속한 학습 샘플들을 하나의 클래스로 모두 맞혔다는 뜻이다. 이 경우 리프 안에 클래스가 섞여 있지 않다. 다만 모델이 ‘검증 정확도가 약 97%니까 충분하다’고 판단해서 멈춘 것은 아니다. 중단 여부는 노드의 학습 자료와 학습 조건으로 결정한다.

### 가지치기 알파를 바꾼 펭귄 실험

| 알파 | 실제 깊이 | 리프 수 | 학습 정확도 | 검증 정확도 |
| --- | --- | --- | --- | --- |
| 0 | 4 | 9 | 1.000000 | 0.971014 |
| 0.005 | 4 | 9 | 1.000000 | 0.971014 |
| 0.01 | 4 | 6 | 0.980392 | 0.927536 |
| 0.02 | 2 | 4 | 0.965686 | 0.913043 |
| 0.05 | 2 | 3 | 0.955882 | 0.913043 |

알파를 0에서 0.01로 바꾸면 리프가 9개에서 6개로 줄었다. 검증 자료는 69개이므로 정확도도 개수로 다시 확인할 수 있다.

$$
0.971014\approx\frac{67}{69},\qquad
0.927536\approx\frac{64}{69}
$$

**리프는 3개 줄었지만 검증에서 맞힌 샘플도 3개 줄었다.** 정확도 차이는 $(3/69)\times100\approx4.35$%포인트다. ‘가지치기가 실패했다’고만 쓰기보다, 구조는 단순해졌지만 이번 검증 정확도는 내려갔다고 설명하는 편이 정확하다.

알파 0과 0.005는 이 표에서 공동으로 가장 높은 검증 정확도를 보였다. 0만 단독 최적이라고 할 수 없다. 알파 0.02와 0.05는 검증 정확도가 같은데 0.05의 리프가 더 적다. 같은 검증 성능에서 단순한 모델을 선호할 근거가 된다.

실제 깊이·리프 수·정확도가 같더라도 모든 분할 조건까지 같다고 증명된 것은 아니다. 결과표의 요약값과 트리 전체 구조를 구분해서 확인한다.

### 좋은 설정을 고를 때는 결론의 범위를 적는다

**과적합**은 학습 자료에 맞춘 세부 규칙이 새 자료에서는 잘 통하지 않는 문제다. 학습 정확도 100%가 새 자료에서도 100%를 보장하지 않는 이유다.

검증 샘플이 69개라면 한 개를 더 맞히는 것만으로도 $(1/69)\times100\approx1.45$%포인트가 달라진다. 한 번의 점수를 고정된 실력처럼 취급하지 않아야 한다.

**교차검증**은 데이터를 나누는 방법을 정하고 학습 부분과 평가 부분을 바꾸어 가며 반복 평가하는 방법이다. `GridSearchCV`는 후보 설정의 조합을 이런 방식으로 비교하는 도구다. 시간 순서가 중요하거나 같은 대상의 반복 기록이 있는 자료는 실제 사용 상황에 맞게 분리해야 한다. 테스트 점수를 보며 설정을 계속 바꾸면 테스트 자료도 설정 선택에 사용한 셈이다.[^selection]

`cost_complexity_pruning_path(X_train, y_train)`는 학습 자료와 설정에 맞는 가지치기 경계값 후보들을 계산한다. 반환되는 **Bunch 객체**는 여러 이름의 결과를 담는 객체이며, 그 안의 `ccp_alphas`는 알파 후보 배열이다. 검증 정확도가 가장 높은 알파를 직접 반환하지는 않는다. 후보들을 별도로 학습·평가해야 하고, 깊이 제한 등 다른 설정이 바뀌면 후보도 달라질 수 있다.[^pruning]

이번 실험의 결론은 다음 범위까지다.

> 현재 검증 정확도를 우선하면 시험한 후보 중 `ccp_alpha=0` 또는 `0.005`가 우선 후보다. `0.01`은 리프를 6개로 줄였지만 검증 정확도가 약 92.75%로 낮아졌다. 모든 가능한 알파 중 최적이라고 확정한 것은 아니며, 다른 적절한 분할에서도 확인할 필요가 있다.

## 12. 특성 중요도는 현재 트리의 기여도이지 원인의 크기가 아니다

`feature_importances_`는 이번 실습에서 **불순도 감소를 기준으로 계산한 특징별 중요도**다. 각 분할이 줄인 불순도에 그 노드의 학습 샘플 비중을 반영하고, 같은 특징을 사용한 분할들의 기여를 합친다. 그 뒤 특징별 기여의 합으로 나누어 상대적인 비중으로 만든다.

불순도 감소가 전혀 없다면 중요도가 모두 0일 수도 있다. 합이 언제나 무조건 1이라고 가정하지 않는다. 이 값은 검증 정확도도, 실제 인과적 영향력도 아니다.[^importance]

예를 들어 `flipper_length_mm`이 가장 높게 나왔다면 “현재 학습된 트리에서 날개지느러미 길이를 사용한 분할이 가중 불순도 감소에 가장 크게 기여했다”고 읽는다. 여기서 `mm`는 길이의 밀리미터 단위다.

| 해석 | 그렇게 읽을 수 없는 이유 |
| --- | --- |
| 중요도 0.55이므로 이 특징 하나로 정답을 55% 맞힌다. | 그 특징만 사용한 검증 정확도가 아니다. |
| 특징을 삭제하면 정확도가 55% 떨어진다. | 삭제 후 재학습·평가한 결과가 아니다. |
| 특징값이 커지면 클래스 1이 된다. | 감소량을 합친 값에는 예측 변화의 방향이 담겨 있지 않다. |
| 날개지느러미 길이가 펭귄 종류의 원인이다. | 특징을 변화시키는 실험으로 인과효과를 확인한 것이 아니다. |

모델의 조건과 입력별 예측을 보면 모델이 어떻게 반응하는지는 확인할 수 있다. 그 반응도 현실의 인과효과와는 다르다. 인과관계를 조사하는 분석 자체가 없다는 뜻이 아니라, **이 중요도만으로 인과관계를 입증하지 못한다는 뜻**이다.

중요도 0은 ‘현재 모델에서 그 특징에 배정된 불순도 감소 기여가 0’이라는 뜻이다. 보통 사용하지 않은 특징일 수 있지만 불순도 감소가 0인 분할에 사용될 가능성도 있으므로, 실제 사용 여부는 `model.tree_.feature`의 특징 번호로 확인한다.

이 중요도는 값의 종류가 많은 특징에 치우칠 수 있다. 비슷한 정보를 가진 특징끼리 기여가 나뉘거나 한쪽에 집중될 수도 있다. 검증 성능에 도움이 되는지 궁금하다면 해당 컬럼을 제외하고 재학습하거나, 검증 자료에서 그 컬럼을 섞었을 때 점수가 어떻게 달라지는지 별도로 확인한다. 이런 검사도 인과관계를 자동으로 입증하지는 않는다.[^importance]

### zip으로 연결하고 sorted로 정렬한다

`zip(feature_names, importance)`는 같은 위치의 이름과 값을 **튜플**로 묶어 꺼내는 **반복자**를 반환한다. 튜플은 `(특징 이름, 중요도)`처럼 값을 묶은 것이고, 반복자는 항목을 차례로 꺼내 사용할 수 있는 객체다. 완성된 리스트를 얻으려면 `list(...)`로 모은다.

`zip`은 이름의 의미를 이해해서 연결하지 않는다. 이름 목록은 반드시 학습 컬럼 순서와 같아야 한다. Python 3.10 이상에서 `strict=True`를 넣으면 길이가 다를 때 오류로 알려 준다. 기본 `zip`은 짧은 쪽이 끝나면 멈춘다.[^python]

`sorted(pairs, key=lambda pair: pair[1], reverse=True)`는 중요도가 큰 순서의 **새 리스트**를 반환한다. `pair`는 튜플 하나, `pair[0]`은 이름, `pair[1]`은 중요도다. `lambda pair: pair[1]`은 각 튜플에서 정렬 기준인 중요도를 꺼내는 짧은 함수이고, `reverse=True`는 큰 값부터 정렬하라는 설정이다.

정렬된 결과가 원래 학습 컬럼 순서인 것은 아니다. 또한 `zip` 반복자는 소비되므로 같은 반복자에 `list(...)`를 두 번 적용하면 두 번째에는 이미 꺼낸 항목이 남아 있지 않다. 출력의 `np.float64(...)`는 NumPy 숫자 자료형 표시이지 또 다른 평가 지표가 아니다.

## 13. 전체 Python 코드로 계산과 조회를 연결한다

아래 코드는 A01의 데이터와 함수 정의를 모두 포함한다. 먼저 지니·엔트로피와 후보별 점수를 계산한 뒤, A01 전용 결정트리를 학습하고 예측 경로와 중요도를 조회한다. **A01의 같은 6개 샘플로 계산한 정확도는 학습 정확도이며 검증 성능이 아니다.**

실행하면 `x1 <= 3.5`가 두 클래스를 완전히 나누어 실제 깊이가 1, 리프가 2가 된다. `max_depth=2`는 상한이므로 이 결과와 모순되지 않는다. 두 번째 샘플의 경로는 `0 → 1`이며, 앞에서 읽었던 `tiny_model`의 `0 → 2 → 3`과 다른 경로다.

직접 정의한 함수들의 역할은 다음과 같다.

| 함수 | 입력과 반환값 |
| --- | --- |
| `proportions_from_counts` | 클래스별 개수를 받아 같은 길이의 비율 배열을 반환한다. |
| `gini_from_counts`, `entropy_from_counts` | 클래스별 개수를 받아 불순도 숫자를 반환한다. |
| `path_for` | 학습된 모델과 숫자 입력 한 행, 특징 이름을 받아 경로 문자열 목록을 반환한다. |
| `compare_settings` | 준비한 학습·검증 입력과 정답을 받아 결과표와 학습된 모델 모음을 반환한다. |

`path_for`는 이 글에서 정의한 별도 구현이다. 리프의 비율은 공개 메서드 `predict_proba`로 얻는다. 이 메서드는 `(샘플 수, 클래스 수)` 확률 배열을 반환하며, 마지막 `[0]`으로 첫 번째 샘플의 클래스별 확률을 고른다. `tree_.value`를 버전 확인 없이 개수로 가정하지 않는다.

`compare_settings`는 함수 정의만으로 학습하지 않는다. 실제 펭귄 자료를 준비한 뒤 맨 아래 주석의 호출을 실행하면 그 자료로 실험한다. 당시 원본 행과 데이터 분할이 제공되지 않았으므로, 아래 A01 결과가 펭귄 실험표를 재현한다고 주장하지 않는다.

<details class="dt-code">
<summary>Python 전체 코드 펼치기</summary>

```python
# Python 3.10 이상. numpy, pandas, scikit-learn이 설치된 환경에서 실행한다.
# A01의 가상 샘플 6개로 계산·학습·경로 조회를 실행한다.
# 펭귄의 당시 검증 점수나 기록의 tiny_model을 재현하는 코드는 아니다.
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text


def proportions_from_counts(counts):
    """클래스별 개수 → 같은 길이의 1차원 비율 배열."""
    counts = np.asarray(counts, dtype=float)  # 순서 유지, 실수 배열로 변환.
    # ndim은 차원 수다. isfinite는 NaN·무한대가 아닌지 검사한다.
    if counts.ndim != 1 or not np.all(np.isfinite(counts)):
        raise ValueError("클래스별 개수는 유한한 1차원 숫자 목록이어야 합니다.")
    total = counts.sum()
    if np.any(counts < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("개수는 음수가 아니고, 합은 유한한 양수여야 합니다.")
    return counts / total


def gini_from_counts(counts) -> float:
    """클래스별 비율을 한 번씩 제곱하고 합을 1에서 뺀다."""
    proportions = proportions_from_counts(counts)
    return float(1 - np.sum(proportions**2))


def entropy_from_counts(counts) -> float:
    """클래스 비율로 정보량을 가중평균한다. 로그의 밑은 2다."""
    proportions = proportions_from_counts(counts)
    positive = proportions[proportions > 0]  # log2(0)을 계산하지 않는다.
    return float(-np.sum(positive * np.log2(positive)))


def path_for(model, one_row, *, feature_names):
    """숫자 입력 한 행의 경로를 문자열 목록으로 반환하는 이 글의 함수."""
    # * 뒤의 인자는 feature_names=...처럼 이름을 붙여 전달한다.
    # 이 구현은 단일 정답 분류와 결측값 없는 입력만 다룬다.
    prediction = model.predict(one_row)[0]  # 학습 여부와 입력 형식도 검사된다.
    if model.n_outputs_ != 1:
        raise ValueError("단일 정답 분류 모델만 지원합니다.")
    # 트리의 내부 입력 형식과 맞춘다. float32는 32비트 실수 자료형이다.
    values = np.asarray(one_row, dtype=np.float32)
    if values.shape != (1, model.n_features_in_):
        raise ValueError("학습 때와 같은 특징을 가진 한 행을 전달하세요.")
    if not np.all(np.isfinite(values)):
        raise ValueError("이 경로 설명 함수에는 NaN·무한대를 넣지 마세요.")
    if len(feature_names) != model.n_features_in_:
        raise ValueError("특징 이름의 개수와 학습 특징 수가 다릅니다.")

    node_id = 0  # 루트 노드의 ID. 깊이와 다른 값이다.
    trace = []
    tree = model.tree_
    while tree.children_left[node_id] != -1:  # -1은 자식이 없다는 표시다.
        feature_id = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        value = float(values[0, feature_id])
        go_left = value <= threshold
        relation = "<=" if go_left else ">"
        trace.append(
            f"node {node_id}: {feature_names[feature_id]} "
            f"{value:.9g} {relation} {threshold:.9g}"
        )
        node_id = int(
            tree.children_left[node_id] if go_left
            else tree.children_right[node_id]
        )

    # predict_proba는 샘플별·클래스별 확률의 2차원 배열을 반환한다.
    probabilities = model.predict_proba(one_row)[0]
    # tree_.value를 개수로 가정하지 않고 공개 메서드로 리프 비율을 얻는다.
    distribution = dict(zip(
        model.classes_.tolist(), probabilities.tolist(), strict=True
    ))
    trace.append(
        f"leaf {node_id}, distribution {distribution}, prediction {prediction}"
    )
    # assert는 조건이 거짓이면 멈춘다. 실제 리프와 수동 경로를 대조한다.
    assert node_id == int(model.apply(one_row)[0])
    return trace


def compare_settings(X_train, y_train, X_val, y_val):
    """준비한 학습·검증 자료로 깊이와 알파 후보를 비교한다."""
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_val, pd.DataFrame):
        raise TypeError("입력에는 pandas DataFrame을 전달하세요.")
    y_train_array, y_val_array = np.asarray(y_train), np.asarray(y_val)
    if y_train_array.ndim != 1 or y_val_array.ndim != 1:
        raise ValueError("정답은 샘플마다 라벨 하나인 1차원이어야 합니다.")
    if len(X_train) != len(y_train_array) or len(X_train) == 0:
        raise ValueError("학습 입력과 정답의 개수를 확인하세요.")
    if len(X_val) != len(y_val_array) or len(X_val) == 0:
        raise ValueError("검증 입력과 정답의 개수를 확인하세요.")
    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("학습·검증의 컬럼과 순서가 같아야 합니다.")
    # 크기 검사는 대상의 실제 대응이나 학습·검증 분리까지 증명하지 않는다.

    def measure(model, name):
        # 바깥 함수에 전달된 자료를 사용한다. 라이브러리가 찾는 것이 아니다.
        model.fit(X_train, y_train)
        train_accuracy = float(accuracy_score(y_train, model.predict(X_train)))
        validation_accuracy = float(accuracy_score(y_val, model.predict(X_val)))
        return {
            "setting": name,
            "depth": int(model.get_depth()),
            "leaves": int(model.get_n_leaves()),
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "gap": train_accuracy - validation_accuracy,
        }

    rows, models = [], {}
    for depth in [1, 2, 4, None]:
        name = f"depth={depth}"  # 문자열은 표의 이름일 뿐이다.
        candidate = DecisionTreeClassifier(max_depth=depth, random_state=41)
        rows.append(measure(candidate, name))  # 반환값을 rows에 대입하지 않는다.
        models[name] = candidate  # 결과표와 별도로 학습된 모델도 보관한다.
    for alpha in [0, 0.005, 0.01, 0.02, 0.05]:
        name = f"alpha={alpha}"
        candidate = DecisionTreeClassifier(ccp_alpha=alpha, random_state=41)
        rows.append(measure(candidate, name))
        models[name] = candidate
    return pd.DataFrame(rows), models


# 1. 본문의 A01 표를 만든다. 한 행은 가상 샘플 하나다.
a01 = pd.DataFrame({
    "sample_id": [1, 2, 3, 4, 5, 6],
    "x1": [1, 2, 3, 4, 5, 6],
    "x2": [1, 2, 4, 3, 5, 6],
    "label": [0, 0, 0, 1, 1, 1],
})
left_gini = gini_from_counts([2, 0])
right_gini = gini_from_counts([1, 3])
weighted_gini = (2 / 6) * left_gini + (4 / 6) * right_gini
print("left / right / weighted Gini:", left_gini, right_gini, weighted_gini)
# isclose는 실수 계산 결과를 허용 오차 안에서 비교한다.
assert np.isclose(left_gini, 0)
assert np.isclose(right_gini, 0.375)
assert np.isclose(weighted_gini, 0.25)

# 2. x2의 모든 서로 다른 인접값 경계를 비교한다.
# sort_values는 행 전체를 정렬하므로 특징과 정답의 대응이 유지된다.
sorted_a01 = a01.sort_values("x2")
x2_sorted = sorted_a01["x2"].to_numpy()  # Series → 1차원 배열.
y_sorted = sorted_a01["label"].to_numpy()
left_counts, right_counts = [0, 0], [3, 3]
total = len(a01)
parent_entropy = entropy_from_counts(right_counts)
threshold_rows = []
for i in range(total - 1):
    label = int(y_sorted[i])
    left_counts[label] += 1
    right_counts[label] -= 1
    left_count, right_count = i + 1, total - i - 1
    threshold = (x2_sorted[i] + x2_sorted[i + 1]) / 2
    weighted_gini = (
        left_count / total * gini_from_counts(left_counts)
        + right_count / total * gini_from_counts(right_counts)
    )
    weighted_entropy = (
        left_count / total * entropy_from_counts(left_counts)
        + right_count / total * entropy_from_counts(right_counts)
    )
    threshold_rows.append({
        "threshold": threshold,
        # copy는 이후 반복에서 개수가 바뀌어도 이번 기록을 보존한다.
        "left_counts": left_counts.copy(),
        "right_counts": right_counts.copy(),
        "weighted_gini": weighted_gini,
        "weighted_entropy": weighted_entropy,
        "information_gain": parent_entropy - weighted_entropy,
    })
print(pd.DataFrame(threshold_rows).to_string(index=False))
assert np.allclose(
    [row["weighted_gini"] for row in threshold_rows],
    [0.4, 0.25, 4 / 9, 0.25, 0.4],
)

# 3. 이 6개 샘플로 A01 전용 모델을 만든다. tiny_model과 다른 모델이다.
X = a01[["x1", "x2"]]  # sample_id와 정답은 입력 특징에서 제외한다.
y = a01["label"]
model = DecisionTreeClassifier(max_depth=2, random_state=41)
model.fit(X, y)
pred = model.predict(X)
# 같은 학습 자료로 계산했으므로 아래 값은 검증 정확도가 아니다.
print("training accuracy:", accuracy_score(y, pred))
print("actual depth / leaves:", model.get_depth(), model.get_n_leaves())
print(export_text(model, feature_names=list(X.columns)))

# 4. 두 번째 샘플을 한 행짜리 2차원 입력으로 선택한다.
one_row = X.iloc[1:2]
print("input / prediction shape:", one_row.shape, pred[1:2].shape)
for line in path_for(model, one_row, feature_names=list(X.columns)):
    print(line)
indicator = model.decision_path(one_row)
print("node-use matrix:", indicator.toarray())  # 작은 한 행만 펼쳐 본다.
print("leaf ID:", model.apply(one_row))

# 5. 중요도는 학습 컬럼 순서로 이름과 연결한 뒤 정렬한다.
pairs = list(zip(X.columns, model.feature_importances_, strict=True))
ordered_importance = sorted(pairs, key=lambda pair: pair[1], reverse=True)
print("feature importance:", ordered_importance)
for position, name in enumerate(X.columns):  # 위치는 0부터 시작한다.
    print(name, "used:", bool(np.any(model.tree_.feature == position)))
probe = DecisionTreeClassifier(random_state=41)
pruning_path = probe.cost_complexity_pruning_path(X, y)
print("candidate alphas:", pruning_path.ccp_alphas)

# 6. 실제 펭귄 학습·검증 자료를 준비한 경우에만 다음 호출을 사용한다.
# table, fitted_models = compare_settings(X_train, y_train, X_val, y_val)
# print(table.to_string(index=False))
# compare_settings의 정의만 실행한 상태에서는 펭귄을 학습하지 않는다.
```

</details>

A01에서 확인할 핵심 출력은 다음과 같다. 로컬 검증에는 scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3을 사용했다. 공식 API 설명은 글 작성일의 문서를 확인했으며, 이 버전 표시는 독자의 설치 버전이나 펭귄 수업의 실행 버전을 뜻하지 않는다.

```text
left / right / weighted Gini: 0.0 0.375 0.25
training accuracy: 1.0
actual depth / leaves: 1 2
node 0: x1 2 <= 3.5
leaf 1, distribution {0: 1.0, 1: 0.0}, prediction 0
node-use matrix: [[1 1 0]]
leaf ID: [1]
candidate alphas: [0.  0.5]
```

`candidate alphas`는 가지치기 경계값이지 최적 검증 점수가 아니다. 위 학습 정확도 1.0을 새 데이터의 성능으로 설명해서도 안 된다.

## 14. Rust에서도 같은 지니를 계산한다

클래스별 개수에서 비율을 구하고, 비율의 제곱합을 1에서 뺀 뒤 자식의 비중으로 가중평균한다는 수학은 Python과 Rust에서 같다. 다른 것은 자료형과 연산 표기다.

Python의 정수끼리 `/`는 실수 나눗셈이다. Rust의 정수끼리 `/`는 정수 나눗셈이므로 아래에서는 **64비트 실수형 `f64`**와 `2.0` 같은 실수 값을 사용한다. Python의 `**2`에 해당하는 계산은 `f64`의 `.powi(2)`로 표현한다.[^rust]

`&[f64]`는 실수 값들의 연속 구간을 빌려 읽는 슬라이스 참조다. 배열 전체를 복사하지 않는다. `iter()`는 항목을 차례로 읽고, `map`은 각 항목을 계산 결과로 바꾸며, `sum`은 그 결과를 합친다. `|&count|`는 참조로 받은 항목에서 계산할 숫자를 꺼내는 표기다.

아래 코드는 결정트리 라이브러리로 모델을 학습하는 예제가 아니라, **A01의 같은 지니 계산을 Rust로 확인하는 독립 프로그램**이다.

<details class="dt-code">
<summary>Rust 전체 코드 펼치기</summary>

```rust
fn gini(counts: &[f64]) -> f64 {
    // 전체 개수가 양수이고 유한한지 먼저 확인한다.
    let total: f64 = counts.iter().sum();
    assert!(total.is_finite() && total > 0.0);
    // all은 모든 항목이 조건을 만족하는지 확인한다.
    assert!(counts.iter().all(|&count| count.is_finite() && count >= 0.0));

    // 클래스마다 비율을 한 번씩 제곱한다. powi(2)는 정수 지수 2의 제곱이다.
    let squared_sum: f64 = counts
        .iter()
        .map(|&count| (count / total).powi(2))
        .sum();
    // 마지막 식에 세미콜론이 없으므로 이 값이 반환된다.
    1.0 - squared_sum
}

fn main() {
    // Python과 같은 클래스별 개수를 사용한다.
    let left_gini = gini(&[2.0, 0.0]);
    let right_gini = gini(&[1.0, 3.0]);
    // 실수끼리 나누므로 2/6, 4/6의 비율이 유지된다.
    let weighted_gini =
        (2.0 / 6.0) * left_gini + (4.0 / 6.0) * right_gini;

    // abs는 절댓값이다. 실수 계산 결과를 작은 허용 오차로 확인한다.
    assert!((left_gini - 0.0).abs() < 1e-12);
    assert!((right_gini - 0.375).abs() < 1e-12);
    assert!((weighted_gini - 0.25).abs() < 1e-12);
    println!("가중 지니: {weighted_gini}"); // 0.25
}
```

</details>

## 15. 복습할 때는 무엇을 입력받고 무엇을 반환하는지 묻는다

| 헷갈리기 쉬운 표현 | 다시 확인할 기준 |
| --- | --- |
| `fit`에 학습용과 검증용을 넣는다. | 학습 입력과 그 학습 대상의 정답을 넣는다. |
| `fit`의 반환값이 평가표다. | 모델 자신을 반환한다. 평가표는 별도 함수가 만든다. |
| `max_depth=3`은 전체 세 층이다. | 루트 깊이가 0이므로 최대 네 층이다. |
| 리프가 7개면 클래스나 데이터가 7개다. | 자식 없는 최종 노드가 7개다. |
| 왼쪽이면 클래스 0이다. | 방향은 조건 결과이고 예측은 리프 정보로 결정한다. |
| 클래스 1이 세 개라서 $(3/4)^2$을 세 번 더한다. | 개수 3은 이미 비율에 들어 있다. 클래스별로 한 번씩 더한다. |
| 가지치기로 단순해졌으므로 더 좋아졌다. | 복잡도 감소와 검증 성능 변화를 따로 확인한다. |
| `None`은 항상 채워야 할 빈칸이다. | `row=None`의 초기 표시와 `max_depth=None`의 설정값은 역할이 다르다. |
| `model is not None`이면 학습됐다. | `None`이 아닌지만 검사한다. 학습 완료 검사가 아니다. |
| `hasattr(model, "tree_")`이면 이번 셀에서 학습했다. | 이전 학습으로 속성이 남아 있을 수 있다. 이번 실행의 증거는 아니다. |

결정트리에서 함께 출력되는 숫자들은 서로 다른 질문에 답한다. **지니는 학습 중 분할 조건을 비교하고, 검증 정확도는 예측이 실제 정답과 얼마나 일치했는지 평가한다. 경로는 한 입력에 적용한 조건을 보여 주고, 중요도는 현재 트리의 불순도 감소 기여를 요약한다.**

코드를 읽을 때도 입력의 자료형과 모양을 확인하고, 어떤 연산을 하는지, 무엇을 반환하는지, 어느 변수에 저장되는지 순서대로 따라가면 된다. 그 다음에야 결과로 어디까지 말할 수 있는지 판단할 수 있다.

## 참고 자료

펭귄의 설정별 수치와 작은 트리의 구조·분포는 2026년 10월 13일 학습 기록에서 가져왔다. A01 계산과 본문의 Python 코드는 별도로 실행해 확인했다. API와 계산 기준은 다음 공식 자료를 참고했다.

[^classifier]: scikit-learn, [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). 생성·학습·예측, 설정, 동률 처리와 조회 메서드.
[^tree]: scikit-learn, [Decision Trees](https://scikit-learn.org/stable/modules/tree.html). 트리 학습, 분할 기준, 지니와 엔트로피.
[^accuracy]: scikit-learn, [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html). 기본 정확도의 입력과 반환값.
[^structure]: scikit-learn, [Understanding the decision tree structure](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html). 노드 구조, 리프, `apply`와 `decision_path`.
[^indexing]: pandas, [Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html). 위치 선택과 라벨 선택.
[^csr]: SciPy, [csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html). 행 단위 희소 저장 형식.
[^implementation]: scikit-learn, [분할 탐색 구현](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_splitter.pyx)과 [클래스 통계·기준 계산 구현](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx). 본문의 반복문은 원리 확인용이며 내부 코드를 그대로 옮긴 것은 아니다.
[^entropy]: SciPy, [entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html). 정보량과 엔트로피의 정의.
[^pruning]: scikit-learn, [Post pruning decision trees with cost complexity pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)과 [Minimal Cost-Complexity Pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning).
[^selection]: scikit-learn, [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html). 후보 설정과 교차검증.
[^pitfalls]: scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html). 전처리 일관성과 데이터 누수.
[^importance]: scikit-learn, [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html). 불순도 기반 중요도의 편향과 중요도 해석의 범위.
[^python]: Python, [Built-in functions](https://docs.python.org/3/library/functions.html). `zip`, `sorted`, `enumerate`.
[^rust]: Rust, [f64](https://doc.rust-lang.org/std/primitive.f64.html)와 [Iterator](https://doc.rust-lang.org/std/iter/trait.Iterator.html). 실수 제곱과 반복자 계산.
