+++
title = "타이타닉으로 배우는 머신러닝: 데이터 준비부터 검증과 앙상블까지"
description = "승객 한 명의 정보를 생존 예측으로 바꾸는 과정을 따라가며 EDA, 결측치와 파생 특성, 데이터 누수, Pipeline, 교차 검증, 학습 곡선, 튜닝과 소프트 투표를 이해한다. 전처리를 언제 학습해야 하는지부터 최종 제출까지 개념과 핵심 코드로 설명한다."
date = 2026-10-19T21:33:00+09:00
slug = "titanic-machine-learning-workflow"

[extra]
toc = true
styles = ["titanic-machine-learning-workflow/article.css"]
+++

타이타닉 승객의 나이, 성별, 승차권 등급을 보고 생존 여부를 예측하려고 한다. 이미 생존 여부가 알려진 승객의 기록으로 규칙을 학습한 뒤, 정답이 없는 승객에게 그 규칙을 적용하는 문제다.

이때 모델을 학습시키는 것만큼 중요한 일이 있다. <strong>모델이 처음 보는 사람에게도 잘 예측하는지 확인하는 일</strong>이다. 학습에 사용한 기록을 잘 맞혔다는 사실만으로는 새로운 기록도 잘 맞힐지 알 수 없다. 평가할 데이터의 정보가 전처리에 섞여도 성능을 실제보다 좋게 판단할 수 있다.[^pitfalls]

데이터의 의미를 읽고, 필요한 특성을 만들고, 여러 모델을 비교하는 모든 작업은 이 목적에 연결된다. 먼저 전체 흐름을 살펴보자. 아래에서 <strong>개발용 데이터</strong>는 모델을 만들고 고르는 데 사용하는 기록이고, <strong>최종 평가용 데이터</strong>는 선택을 끝낸 뒤 성능을 확인하려고 따로 보관하는 기록이다.

<figure class="titanic-figure">
<img src="/titanic-machine-learning-workflow/workflow-ko.svg" alt="정답이 있는 승객 기록을 개발용과 최종 평가용으로 나눈다. 개발용 안에서 EDA, 전처리와 모델의 교차 검증, 특성과 설정의 수정을 반복한다. 선택을 끝내고 최종 평가용으로 평가한 다음 전체 정답 데이터로 재학습하여 캐글 test.csv를 예측한다.">
<figcaption>전처리 기준도 학습 결과다. 평가에 사용할 행은 그 기준을 계산하는 데 참여하지 않는다.</figcaption>
</figure>

## 1. 승객 한 명의 입력과 정답을 구분한다

머신러닝에서 <strong>샘플(sample)</strong>은 학습이나 예측의 대상 하나다. 타이타닉 표에서는 승객 한 명이 샘플 하나이고, 한 행에 그 승객의 기록이 들어 있다. 예측에 사용하는 입력 정보를 <strong>특성(feature)</strong>, 모델이 맞히려는 값을 <strong>목표값(target) 또는 정답</strong>이라고 한다.

캐글 타이타닉 데이터의 주요 컬럼은 다음과 같다.[^titanic]

| 컬럼 | 기록의 의미 |
| --- | --- |
| `Survived` | 생존 여부. `0`은 사망, `1`은 생존이다. |
| `PassengerId` | 승객 기록을 구분하는 식별 번호다. |
| `Pclass` | 승객의 승차권 등급이다. `1`, `2`, `3`은 각각 1·2·3등급이다. |
| `Sex`, `Age` | 기록된 성별과 나이다. 나이의 단위는 년이다. |
| `SibSp` | 함께 탑승한 형제자매와 배우자의 수다. |
| `Parch` | 함께 탑승한 부모와 자녀의 수다. |
| `Fare` | 기록된 승선 요금이다. |
| `Ticket` | 승차권 번호다. |
| `Cabin` | 기록된 객실 번호다. |
| `Embarked` | 탑승 항구다. `C`, `Q`, `S`는 각각 Cherbourg, Queenstown, Southampton이다. |
| `Name` | 기록된 이름이다. |

입력 표에는 관례적으로 `X`, 정답에는 `y`라는 이름을 붙인다. `X`에서 `Survived`를 빼고, `y`에는 `Survived`만 담는다. 예측할 때 알 수 없는 생존 여부를 입력으로 주면 모델의 예측 능력을 평가할 수 없기 때문이다.

`PassengerId`도 이 글의 모델 입력에서 제외한다. 이 번호는 제출할 예측을 원래 승객과 연결하는 데 보관한다. 번호를 삭제하면서 제출에 필요한 연결 정보까지 잃지 않도록 역할을 나누는 것이다.

생존과 사망처럼 <strong>정해진 범주 중 하나를 예측하는 문제를 분류(classification)</strong>라고 한다. 모델은 먼저 생존 확률을 계산할 수 있다. 예를 들어 한 승객에게 생존 확률 `0.7`을 예측했다면, 이 값은 정답 `1`과 다른 종류의 값이다. `0.7`은 모델이 계산한 확률이고, `1`은 실제로 생존했다는 기록이다.

예측 대상이 세 명이라면 모델을 한 번 호출해도 결과는 세 명분 나온다. <strong>호출 횟수와 예측값의 개수는 다르다.</strong>

| 호출 | 이진 분류에서 반환하는 값 |
| --- | --- |
| `model.predict(X)` | 승객마다 `0` 또는 `1` 하나씩 반환한다. 세 명이면 값 세 개다. |
| `model.predict_proba(X)` | 승객마다 두 범주의 확률을 반환한다. 세 명이면 3행 × 2열이다. |

여기서 `model`은 학습된 모델 객체다. 객체에 속한 함수를 <strong>메서드(method)</strong>라고 하며, `predict`와 `predict_proba`는 메서드다. 확률의 열 순서는 모델의 `classes_`에 저장된다. `classes_`가 `[0, 1]`이면 첫 열은 사망 확률, 두 번째 열은 생존 확률이다.[^classifier]

## 2. 데이터를 나누는 목적부터 정한다

캐글의 `train.csv`에는 정답이 있고, `test.csv`에는 제출자가 사용할 정답이 없다. 따라서 캐글의 테스트 파일만으로는 로컬에서 정확도를 계산할 수 없다. <strong>정답이 있는 파일 안에서 학습과 평가에 사용할 기록을 다시 나눠야 한다.</strong>[^titanic]

이 글에서는 정답이 있는 기록의 80%를 개발용, 20%를 최종 평가용으로 나눈다. 80:20은 하나의 선택이며 항상 정해진 정답은 아니다. 중요한 것은 어떤 기록을 어떤 용도로 사용하는지다.

| 구분 | 사용하는 시점과 목적 |
| --- | --- |
| 개발용 데이터 | 데이터를 탐색하고, 특성을 만들고, 모델과 설정을 비교한다. |
| 개발용 내부의 훈련 부분 | 한 번의 비교에서 전처리 기준과 모델을 학습한다. |
| 개발용 내부의 검증 부분 | 그 비교에서 학습하지 않은 행으로 성능을 계산한다. |
| 최종 평가용 데이터 | 모델과 설정을 결정한 뒤 마지막 성능을 확인한다. |
| 캐글 `test.csv` | 제출할 승객의 예측을 생성한다. 로컬에서는 정답을 비교할 수 없다. |

개발용 내부의 훈련·검증 부분은 뒤에서 설명할 교차 검증에 따라 여러 번 바뀐다. 반면 최종 평가용 데이터는 개발 과정 전체에서 따로 보관한다. 이처럼 별도로 남겨 두는 평가 데이터를 <strong>홀드아웃(holdout)</strong>이라고 부른다.

`train_test_split`은 여러 자료를 같은 행 위치에 맞춰 나누는 <strong>함수</strong>다. `X`와 `y`를 함께 전달하면 개발용 입력·평가용 입력·개발용 정답·평가용 정답을 반환한다. 입력과 정답을 각각 제멋대로 섞으면 서로 다른 승객의 정보와 정답이 연결될 수 있다.[^split]

`test_size=0.2`는 20%를 따로 남기는 설정이다. `stratify=y`는 생존·사망 비율이 두 부분에서 가능한 한 비슷하도록 나눈다. `random_state=42`는 같은 자료와 실행 조건에서 분할을 재현하기 위한 난수 설정이다. 숫자 42 자체가 더 좋은 분할이나 성능을 보장하지는 않는다.

## 3. EDA에서는 컬럼의 의미와 실제 값을 함께 확인한다

<strong>EDA는 탐색적 데이터 분석(Exploratory Data Analysis)</strong>의 약자다. 표의 구조와 값의 분포를 살펴보고, 기록의 문제나 예측에 쓸 만한 관계를 찾는 작업이다.

파일을 읽고 컬럼명, 자료형, 정답 컬럼의 유무를 확인하는 것은 데이터를 이해하기 위한 기본 점검이다. 그 뒤 특성을 선택하거나 변환 방법을 결정하는 분석은 개발용 데이터를 기준으로 진행한다. 최종 평가용의 생존율과 오답을 계속 들여다보며 모델을 바꾸면 그 데이터도 개발에 사용한 셈이 된다.[^selection]

### 빈칸은 0과 의미가 다르다

<strong>결측치(missing value)</strong>는 기록에 값이 없다는 뜻이다. 나이가 비어 있다는 것은 나이가 0세라는 뜻이 아니다. 객실 번호가 없다고 해서 객실에 탑승하지 않았다고 단정할 수도 없다.

결측치를 발견하면 먼저 어느 컬럼에서 얼마나 발생했는지 확인한다. 그다음 해당 컬럼을 유지할지, 다른 값으로 채울지, 값이 없었다는 사실을 별도 특성으로 남길지 결정한다. 값이 많이 비었다는 이유만으로 모든 행이나 컬럼을 곧바로 버리지는 않는다.

### 같은 값이 있다고 같은 승객은 아니다

나이, 성별, 승차권 등급이 같은 승객은 여러 명일 수 있다. 이런 특성의 반복은 곧 기록 오류가 아니다. 반대로 같은 승객의 기록이 실수로 두 번 들어갔다면 학습과 평가에 동시에 포함되지 않도록 처리해야 한다.

중복을 검사할 때는 <strong>무엇이 같으면 같은 기록이라고 판단할지</strong>부터 정한다. 모든 행에서 서로 다른 식별 번호를 비교에 넣으면, 나머지 정보가 같은 행도 중복으로 잡히지 않을 수 있다. 식별 번호를 제외하고 찾은 후보는 원래 기록의 의미를 다시 확인해야 한다.

### 히스토그램은 구간마다 몇 명이 있는지 보여 준다

`Fare`의 히스토그램에서 가로축은 요금 구간, 세로축은 그 구간에 들어간 승객 기록의 개수다. `bins=30`은 값을 30개 구간으로 나눠 세겠다는 뜻이다. 막대 하나가 승객 한 명을 뜻하지 않는다.

낮은 요금에 기록이 몰리고 일부 큰 요금이 멀리까지 이어지면, 분포의 오른쪽 꼬리가 길다고 말한다. 이것이 <strong>오른쪽으로 치우친 분포</strong>다. 값이 크다는 사실만으로 오류는 아니다. 실제 가능한 요금인지, 기록 단위가 같은지부터 확인한다.

생존율을 비교할 때도 비율과 인원수를 함께 본다. 설명용 가상 집단에서 `1명 중 1명 생존`과 `100명 중 100명 생존`은 둘 다 100%지만, 판단의 근거가 되는 기록 수는 크게 다르다. 작은 집단에서 관측한 비율을 모든 승객의 규칙으로 바로 일반화하지 않는다.

## 4. 기존 기록에서 예측에 쓸 특성을 만든다

원래 컬럼을 계산하거나 조합해 새 입력을 만드는 일을 <strong>피처 엔지니어링(feature engineering), 또는 특성 만들기</strong>라고 한다. 이름이 붙은 새 컬럼을 많이 만드는 것보다, 그 컬럼의 값이 무엇을 뜻하는지 설명할 수 있어야 한다.

### 함께 탑승한 가족 수를 합친다

`SibSp`는 형제자매·배우자 수이고, `Parch`는 부모·자녀 수다. 두 값에 승객 본인 한 명을 더하면 다음과 같은 특성을 만들 수 있다.

<p class="titanic-formula"><code>FamilySize = SibSp + Parch + 1</code></p>

`SibSp=1`, `Parch=2`인 승객이라면 `1+2+1=4`다. 마지막 `1`은 본인이다. 따라서 `FamilySize=1`은 이 두 컬럼에 함께 탄 가족이 기록되지 않았다는 뜻이다. 친구까지 포함한 전체 동행인 수를 알았다는 뜻은 아니다.

`SibSp`와 `Parch`를 그대로 사용할지, `FamilySize`도 추가할지는 검증으로 비교한다. 새 특성이 원래 두 값을 요약해 줄 수 있지만, 같은 정보를 반복해서 주는 것만으로 성능이 반드시 높아지지는 않는다.

### 객실 번호가 기록되어 있었는지 남긴다

`Has_Cabin`을 `객실 번호가 있으면 1, 비어 있으면 0`으로 만들 수 있다. 원래 문자열을 직접 입력하지 않아도 기록의 유무를 모델이 참고하게 된다.

이때 `0`이 나타내는 것은 <strong>객실 정보가 기록되지 않았다는 사실</strong>뿐이다. 이를 곧바로 3등급 승객이나 특정 계층이라고 해석하면 기록에 없는 내용을 덧붙이게 된다. 승차권 등급과의 관계는 개발용 데이터에서 확인해야 하고, 예측에 도움이 되는지도 검증해야 한다.

### 이름 길이는 정제 규칙에 따라 달라진다

이름에서 성과 호칭을 제외하고 남은 글자 수를 특성으로 사용할 수도 있다. 그러나 공백을 포함할지, 괄호 안 이름을 포함할지, `Mr.`와 다른 호칭을 어떻게 처리할지에 따라 값이 달라진다.

예를 들어 설명용 가상 문자열 `" Lee"`는 앞 공백까지 세면 네 글자이고, 앞뒤 공백을 제거한 `"Lee"`는 세 글자다. 따라서 `name_cnt`라는 이름만으로는 계산 방법을 알 수 없다. <strong>어떤 문자열을 남기고 무엇을 세었는지</strong>를 먼저 정해야 같은 의미의 특성을 만들 수 있다.

이런 계산은 한 승객의 행만으로 수행한다. 다른 승객들의 평균이나 빈도를 알아야 하는 계산과는 구분된다. 이 차이가 다음 전처리 단계에서 중요해진다.

## 5. 빈칸을 채우는 기준은 훈련 데이터에서 구한다

결측치를 다른 값으로 채우는 일을 <strong>결측치 대체(imputation)</strong>라고 한다. 나이에 사용할 수 있는 기준 중 하나는 중앙값이다. <strong>중앙값은 값을 작은 순서로 정렬했을 때 가운데에 있는 값</strong>이다. 값의 개수가 짝수이면 가운데 두 값의 평균을 사용한다.

설명용으로 훈련 부분의 나이가 `20, 30, 40`세라고 하자. 중앙값은 30세다. 여기에 검증 부분의 나이 `80, 90`세를 섞으면 전체 다섯 값의 중앙값은 40세가 된다. 빈칸을 채우는 값이 검증 부분의 영향을 받은 것이다.

<strong>모델 학습이나 모델을 만드는 과정에 평가용 정보가 들어가는 문제를 데이터 누수(data leakage)</strong>라고 한다. 정답을 직접 넣었을 때만 생기는 문제가 아니다. 검증 행의 입력값으로 전처리 기준을 정해도 검증 정보가 학습 과정에 참여한다.[^pitfalls]

따라서 훈련 부분에서 중앙값 30을 구했다면, 그 비교에 사용되는 훈련·검증 양쪽의 빈 나이를 같은 30으로 채운다. 검증 데이터에서는 중앙값을 다시 계산하지 않는다. 이렇게 해야 새 승객을 예측할 때와 같은 조건으로 평가할 수 있다.

### 승차권 등급마다 나이의 중앙값을 따로 구할 수도 있다

전체 훈련 행에서 중앙값 하나를 구하는 대신, `Pclass`별 중앙값을 구할 수도 있다. 다음은 계산 구조를 설명하기 위한 <strong>가상의 결과</strong>다.

| 훈련 데이터의 승차권 등급 | 그 등급에서 나이가 기록된 승객들의 중앙값 |
| --- | ---: |
| 1등급 | 40세 |
| 2등급 | 30세 |
| 3등급 | 20세 |

검증 승객이 2등급이고 나이가 비어 있다면 저장한 표에서 2등급의 값 30을 찾아 채운다. 검증 승객들이 몇 살인지는 이 기준을 바꾸지 않는다.

pandas는 Python에서 표를 다루는 도구 모음이다. 행과 열로 된 표를 <strong>DataFrame</strong>, 값마다 인덱스라는 식별값이 연결된 자료를 <strong>Series</strong>라고 부른다.

훈련 표에서 `groupby("Pclass")["Age"].median()`을 실행하면, <strong>승차권 등급을 인덱스로, 각 등급의 나이 중앙값을 값으로 가진 Series</strong>가 나온다. 검증 표의 `Pclass` 컬럼에 `map(중앙값표)` 메서드를 적용하면 각 승객의 등급에 맞는 중앙값을 찾는다. 결과는 승객 수와 같은 길이의 Series이며, 원래 행 인덱스를 유지한다. 이 결과를 `Age` 컬럼의 `fillna` 메서드에 전달하면 원래 나이가 있는 칸은 유지하고 빈칸만 채울 수 있다.[^pandas-map]

어떤 등급의 훈련 나이가 전부 비어 있으면 그 등급의 중앙값도 구할 수 없다. 그때는 훈련 부분 전체의 나이 중앙값을 사용하는 등 대체 규칙을 미리 정해야 한다. 전체 나이도 모두 비었다면 중앙값이라는 기준 자체가 없으므로, 해당 특성을 제외하거나 별도의 고정값을 쓰는 정책이 필요하다.

### 항구와 요금의 빈칸도 처리 기준이 필요하다

항구처럼 범주를 나타내는 값에는 <strong>최빈값</strong>을 사용할 수 있다. 최빈값은 가장 자주 등장한 값이다. 훈련 부분에서 `S`가 가장 많았다면 빈 항구를 `S`로 채운다. 이것은 그 승객의 실제 항구를 알아낸 결과가 아니라 모델에 입력할 대체값이다.

`Fare`에 빈칸이 있다면 로그 계산만으로 채워지지 않는다. 요금의 훈련 중앙값으로 대체한 뒤 로그 변환을 적용하는 식으로 순서를 정할 수 있다. 여러 모델을 비교할 때에는 각 모델이 결측치를 받을 수 있는지도 확인해야 한다. 예를 들어 로지스틱 회귀에는 결측치를 정리한 숫자 입력을 준비하는 반면, 현재 랜덤 포레스트와 XGBoost에는 결측치를 자체 처리하는 기능이 있다.[^imputer][^forest][^xgb-api]

### 티켓 빈도도 훈련 데이터에서 배우는 값이다

`Ticket`이 같은 기록의 개수를 특성으로 만들 수도 있다. `groupby("Ticket").size()`는 <strong>티켓 번호를 인덱스로, 해당 번호의 행 개수를 값으로 가진 Series</strong>를 반환한다.[^pandas-size]

설명용 훈련 표에서 티켓 A가 두 번, B가 한 번 나왔다고 하자. 이때 저장되는 기준은 `A → 2`, `B → 1`이다. 검증 표에서 A를 가진 승객에게도 2를 넣는다. 검증 승객을 더해서 3을 넣는 계산이 아니다.

따라서 이 특성의 정확한 뜻은 <strong>참조한 훈련 표에서 같은 티켓 번호가 나온 횟수</strong>다. 전체 탑승객의 실제 동행인 수와 같다고 볼 수 없다. 훈련 행은 자기 자신도 횟수에 포함되지만, 검증 행은 참조 훈련 표의 횟수만 받는다.

처음 보는 티켓에 1을 넣기로 정했다면 그 1은 확인한 인원수가 아니라 기본값이다. `0=훈련에서 본 적 없음`으로 표현하는 방법도 가능하다. 어느 정책을 사용하든 특성의 뜻을 일정하게 유지하고, 같은 검증 절차에서 효과를 비교한다.

## 6. 로그 변환과 스케일링은 서로 다른 일을 한다

### 로그 변환은 큰 값 사이의 간격을 줄인다

요금의 범위가 넓을 때는 로그 변환을 후보로 검토할 수 있다. <strong>로그는 어떤 수를 얻으려면 기준이 되는 수를 몇 제곱해야 하는지를 나타내는 계산</strong>이다. 예를 들어 10을 기준으로 한 로그에서 10, 100, 1000은 각각 1, 2, 3이 된다. 열 배씩 증가한 값이 일정한 간격으로 바뀐다.

NumPy의 `np.log1p`는 <strong>입력값에 1을 더한 뒤 자연로그를 구하는 함수</strong>다. 자연로그는 약 2.718인 수 `e`를 기준으로 한다. 배열을 전달하면 각 원소를 변환하고 같은 형태의 배열을 반환한다. 1을 먼저 더하므로 요금이 0이면 `log(1)=0`으로 계산할 수 있다.[^log1p]

| 설명용 요금 값 | `log1p` 변환값, 소수 셋째 자리까지 |
| ---: | ---: |
| 0 | 0.000 |
| 10 | 2.398 |
| 100 | 4.615 |
| 1000 | 6.909 |

10과 100의 원래 차이는 90이고, 100과 1000의 차이는 900이다. 변환 뒤에는 각각 약 2.217과 2.294 차이가 난다. 큰 값 사이의 간격이 상당히 줄었다.

그렇다고 분포가 반드시 좌우 대칭인 종 모양의 <strong>정규분포</strong>가 되는 것은 아니다. 로지스틱 회귀도 입력 컬럼이 정규분포여야 한다는 조건은 없다. 로그 변환을 사용하면 모델에 주는 요금의 표현이 바뀌므로, 검증 결과로 사용 여부를 결정한다.[^logistic-model]

랜덤 포레스트와 XGBoost의 트리는 주로 값의 대소 관계에 따라 행을 나눈다. 따라서 숫자 범위를 줄였다는 이유만으로 예측이 좋아지지는 않는다. 트리 모델에 로그 변환이나 스케일링을 필수 단계로 넣을 필요는 없으며, 구현과 수치 처리에 따라 결과 차이가 생길 수는 있다.[^scaling]

### 스케일링은 컬럼들의 숫자 범위를 조정한다

나이는 수십, 가족 수는 몇 명, 요금은 더 넓은 범위로 나타날 수 있다. <strong>스케일링(scaling)</strong>은 이런 수치 표현의 범위를 조정하는 전처리다. 로지스틱 회귀처럼 각 입력값에 곱할 수인 <strong>계수</strong>를 배워 예측하는 모델에서는, 컬럼의 수치 범위를 조정하면 학습 계산이 더 잘 진행될 수 있다.[^scaling]

대표적인 표준화는 각 값에서 해당 컬럼의 훈련 평균을 빼고, 훈련 표준편차로 나눈다. <strong>평균</strong>은 값들을 더해 개수로 나눈 값이고, <strong>표준편차</strong>는 값들이 평균 주변에서 얼마나 퍼져 있는지를 나타내는 수치다.

scikit-learn은 Python에서 전처리·모델 학습·평가에 사용하는 도구 모음이다. 이 도구들은 새 객체를 만들 때 사용할 기능과 설정을 <strong>클래스(class)</strong>로 정의한다. `StandardScaler`는 표준화 도구를 만드는 클래스이며, 만들어진 객체가 평균·표준편차를 저장하고 변환을 수행한다. 평균과 표준편차는 훈련 행에서만 구하며, 검증 행에는 저장한 기준으로 계산한다. 표준화도 컬럼을 자동으로 정규분포로 만드는 작업은 아니다.[^scaler]

## 7. 문자는 같은 의미의 숫자 칸으로 바꾼다

이 글의 핵심 코드에서는 모델에 숫자 표를 전달한다. `Embarked`의 `C`, `Q`, `S`처럼 <strong>서로 다른 종류를 구분하는 값을 범주형 값</strong>이라고 한다.

문자를 단순히 `C=0`, `Q=1`, `S=2`로 바꾸면, 숫자의 간격을 이용하는 모델은 Q가 C와 S의 중간에 있다고 처리할 수 있다. 항구에는 그런 수치적 간격이 정의되어 있지 않다.

<strong>원-핫 인코딩(one-hot encoding)</strong>은 범주별로 칸을 만들고, 해당하는 칸에 1, 나머지 칸에 0을 넣는 방법이다.[^encoder]

| 탑승 항구 | 항구_C | 항구_Q | 항구_S |
| --- | ---: | ---: | ---: |
| C | 1 | 0 | 0 |
| Q | 0 | 1 | 0 |
| S | 0 | 0 | 1 |

행 수는 승객 수 그대로이고, 항구를 표현하는 컬럼 수가 늘어난다. `Pclass`도 숫자로 기록되어 있지만 승차권 등급을 뜻한다. 등급에는 순서가 있으나 등급 간 차이를 나이나 요금처럼 같은 크기의 차이라고 가정할 필요는 없다. 아래 핵심 코드에서는 `Pclass`도 범주로 인코딩한다.

### 훈련과 검증의 컬럼은 이름과 순서가 같아야 한다

훈련 데이터에 C·Q·S가 모두 있고 검증에는 C·S만 있다고 하자. 각 표에 `pd.get_dummies`를 따로 적용하면 훈련에는 세 컬럼, 검증에는 두 컬럼이 생길 수 있다. 컬럼 개수만 억지로 같게 맞추어도 순서가 다르면 모델이 다른 의미로 값을 읽는다.[^dummies]

`OneHotEncoder`는 훈련에서 범주 목록을 저장한 뒤 다른 데이터에도 같은 구조로 적용하는 클래스다. `handle_unknown="ignore"`를 지정하면 훈련에서 보지 못한 항구가 들어왔을 때 항구 관련 칸들을 모두 0으로 둔다. 새로운 항구의 뜻을 학습했다는 의미는 아니다. 정해진 입력 구조를 유지하는 처리 규칙이다.[^encoder]

각 교차 검증에서 학습한 범주 목록은 서로 다를 수 있다. 필요한 것은 서로 다른 모델끼리 컬럼 수를 강제로 통일하는 일이 아니라, <strong>각 모델이 자기 훈련 때 사용한 구조로 자기 검증 데이터를 받는 것</strong>이다.

## 8. Pipeline은 전처리와 모델을 함께 학습하게 한다

지금까지의 계산은 다음 두 종류로 나뉜다.

| 현재 승객의 행만으로 계산하는 일 | 다른 훈련 행들을 보고 기준을 정하는 일 |
| --- | --- |
| 가족 수 합산, 객실 기록 여부 확인 | 나이·요금의 중앙값 계산 |
| 미리 정한 이름 정제 규칙 적용 | 항구의 최빈값 계산 |
| 미리 정한 요금 로그 변환 적용 | 티켓별 빈도, 평균·표준편차, 인코딩 범주 목록 저장 |

왼쪽 계산 자체는 다른 검증 행의 정보를 참조하지 않는다. 오른쪽 계산은 어떤 행을 기준으로 삼았는지가 결과에 영향을 준다. 다만 로그 변환을 쓸지 같은 <strong>방법의 선택</strong>을 데이터를 보고 결정한다면, 그 선택도 개발 과정에 포함된다.

전처리 도구는 이 기준을 배우는 일과 적용하는 일을 구분한다.[^imputer][^pipeline]

| 동작 | 실제로 하는 일 |
| --- | --- |
| 전처리의 `fit` | 훈련 데이터에서 중앙값·범주 목록 등의 기준을 구해 저장한다. |
| 전처리의 `transform` | 이미 저장한 기준으로 입력을 변환한다. |
| 전처리의 `fit_transform` | 기준을 학습한 뒤 그 기준으로 입력도 변환한다. |
| 모델의 `fit` | 변환된 입력과 정답으로 예측 규칙을 학습한다. |
| 모델의 `predict` | 학습한 규칙으로 각 행의 범주를 예측한다. |

scikit-learn의 <strong>`Pipeline`은 여러 처리 단계를 순서대로 묶는 클래스</strong>다. 전처리와 모델을 한 객체로 묶으면, 이 객체에 학습이나 예측을 요청할 때 필요한 단계가 함께 실행된다.

<strong>`Pipeline(...)`으로 객체를 만드는 순간에는 아직 중앙값이나 모델의 규칙을 학습하지 않는다.</strong> 이때는 사용할 도구와 순서를 정한다. 실제 학습은 `pipeline.fit(X_train, y_train)`을 호출할 때 시작한다.

이 호출은 먼저 전처리의 기준을 훈련 입력에서 구하고, 훈련 입력을 변환한 다음, 변환 결과와 정답으로 모델을 학습한다. 반대로 `pipeline.predict(X_valid)`는 저장된 기준으로 검증 입력을 변환하고 예측한다. 검증 입력으로 전처리의 기준을 다시 배우지 않는다.[^pipeline]

교차 검증에서 전처리 기준도 매번 새로 배우게 하려면, <strong>중앙값과 범주 목록을 구하는 단계까지 Pipeline 안에 넣어야 한다.</strong>

## 9. 교차 검증에서는 전처리부터 매번 새로 학습한다

데이터를 한 번 나누어 얻은 점수는 어떤 승객이 훈련이나 검증에 들어갔는지에 따라 달라질 수 있다. <strong>교차 검증(cross-validation, CV)</strong>은 여러 분할에서 모델을 학습하고 평가하는 방법이다.[^cv]

5회 교차 검증에서는 개발용 데이터를 다섯 묶음으로 나눈다. 이 묶음을 <strong>폴드(fold)</strong>라고 한다. 설명을 위해 묶음에 A부터 E까지 이름을 붙이면 다음과 같다.

| 비교 차례 | 훈련에 사용하는 묶음 | 검증에 사용하는 묶음 |
| --- | --- | --- |
| 1 | B, C, D, E | A |
| 2 | A, C, D, E | B |
| 3 | A, B, D, E | C |
| 4 | A, B, C, E | D |
| 5 | A, B, C, D | E |

각 행은 <strong>전처리와 모델을 새로 학습하는 별도의 비교</strong>다. 첫 번째 모델을 두 번째 비교에서 이어 학습하지 않는다. 그렇게 하면 첫 번째에 훈련했던 B 묶음을 두 번째에서 처음 보는 검증 데이터처럼 평가할 수 없기 때문이다.

`StratifiedKFold`는 정답 범주의 비율을 고려해 묶음을 나누는 클래스다. `n_splits=5`는 다섯 묶음, `shuffle=True`는 분할 전에 섞는 설정이다. 생존·사망 비율을 각 묶음에 가능한 한 비슷하게 유지하지만, 사람 수는 정수이므로 비율이 완전히 같지 않을 수 있다.[^stratified]

이 객체의 `split(X, y)` 메서드는 훈련·검증에 사용할 <strong>행 위치 번호 두 배열</strong>을 차례로 내놓는다. 나누는 위치를 정할 뿐, 모델을 학습하지 않는다. pandas 표에서 이 위치로 행을 고를 때는 위치 기반 선택인 `iloc`를 사용한다. 행의 인덱스 이름으로 고르는 `loc`와는 구분한다.

### 먼저 전처리한 뒤 다섯 묶음으로 나누면 순서가 뒤바뀐다

개발용 데이터 전체의 나이 중앙값을 구한 다음 A~E로 나누었다고 하자. 첫 번째 비교에서는 A가 검증용이지만, A의 나이가 이미 중앙값 계산에 들어갔다. 바깥의 최종 평가용 20%를 잘 분리했더라도 <strong>개발용 내부의 검증 정보는 전처리에 섞인 상태</strong>다.

올바른 첫 번째 비교에서는 B·C·D·E에서 중앙값과 범주 목록을 구한다. 이 기준으로 B·C·D·E를 변환하고 모델을 학습한 다음, A에는 같은 기준을 적용해 예측한다. 두 번째 비교에서는 A·C·D·E를 기준으로 전처리와 모델을 다시 학습한다.

그래서 교차 검증 도구에는 <strong>전처리가 끝난 표와 모델만 전달하는 대신, 원본 입력과 전처리·모델을 묶은 Pipeline을 전달</strong>한다. 학습해야 할 단계 전체가 폴드의 훈련 부분 안에서 실행되도록 만드는 것이다.[^compose]

### 비율을 맞췄다고 모든 검증 문제가 해결되지는 않는다

같은 가족이나 티켓의 승객이 양쪽에 들어갈 수 있다. 이것이 적절한지는 예측하려는 상황에 달려 있다. 지금 가진 기록과 같은 조건에서 모인 다른 승객을 예측하려는지, 아예 처음 보는 가족·집단을 예측하려는지에 따라 필요한 평가가 다르다. 후자를 평가하려면 같은 그룹이 양쪽에 걸치지 않도록 그룹 단위로 나누는 방법을 검토한다.[^cv]

교차 검증은 전처리 누수, 잘못된 기록, 데이터를 모은 조건의 차이까지 자동으로 없애는 도구는 아니다. <strong>앞으로 어떤 데이터를 예측할 것인지에 맞춰 분할 방법을 정해야 한다.</strong>

## 10. 정확도와 손실은 서로 다른 부분을 평가한다

모델을 고르려면 무엇을 잘한다고 판단할지 정해야 한다. 그 기준이 <strong>평가 지표(metric)</strong>다.

### 정확도는 맞힌 승객의 비율이다

설명용 평가 기록 100명 중 80명의 생존 여부를 맞혔다면 정확도는 `80÷100=0.8`, 즉 80%다. `accuracy_score`는 실제 정답과 예측한 범주를 받아 이 비율을 반환하는 함수다.[^accuracy]

교차 검증 점수가 설명용으로 `0.78, 0.82, 0.80, 0.79, 0.81`이라면 다섯 점수의 평균은 `4.00÷5=0.80`이다. 평균과 함께 각 묶음의 점수도 확인한다. 평균이 비슷해도 특정 묶음에서 크게 떨어지는 모델이 있을 수 있다.

성능을 비교하기 전에는 단순한 기준도 필요하다. 이를 <strong>기준 모델(baseline)</strong>이라고 한다. 설명용 훈련 부분에서 사망한 승객이 더 많았다고 하자. 이 기준 모델은 그때부터 모든 새 승객을 사망으로 예측한다. 평가 데이터에서도 실제로 사망한 승객이 70%라면 이 단순한 모델의 정확도는 70%다. 따라서 학습한 모델의 정확도는 같은 평가 데이터에서 얻은 기준 모델의 점수와 비교해야 한다.

scikit-learn의 `DummyClassifier`는 이런 단순한 비교 기준을 만드는 클래스다. 기준 모델도 같은 훈련·검증 분할에서 평가해야 비교의 의미가 있다.[^dummy]

### 로그 손실은 정답에 얼마나 높은 확률을 주었는지 평가한다

생존자를 생존 확률 0.6으로 예측한 모델과 0.9로 예측한 모델은 둘 다 생존이라고 판단할 수 있다. 정확도에서는 둘 다 정답이지만, 두 모델이 계산한 확률은 다르다.

<strong>손실(loss)</strong>은 예측이 정답에서 얼마나 벗어났는지를 수치로 계산한 값이며, 작을수록 좋도록 정의한다. 분류의 <strong>로그 손실(log loss)</strong>은 실제 정답에 부여한 확률을 사용한다.[^logloss]

먼저 `p`를 모델의 생존 확률, `q`를 실제로 일어난 범주에 부여한 확률이라고 정하자.

- 실제로 생존했다면 `q=p`다.
- 실제로 사망했다면 `q=1-p`다.

한 승객의 로그 손실은 `−log(q)`다. 여기서 `log`는 앞서 설명한 자연로그다. 0보다 크고 1 이하인 확률에 자연로그를 취하면 0 이하의 값이 나온다. 앞에 마이너스를 붙이면 정답에 주는 확률이 1에 가까울수록 손실이 0에 가까워지고, 그 확률이 작을수록 손실이 커진다.

| 실제 기록 | 생존 확률 `p` | 실제 범주의 확률 `q` | 한 승객의 로그 손실, 근삿값 |
| --- | ---: | ---: | ---: |
| 생존 | 0.60 | 0.60 | 0.511 |
| 생존 | 0.90 | 0.90 | 0.105 |
| 사망 | 0.60 | 0.40 | 0.916 |
| 사망 | 0.99 | 0.01 | 4.605 |

모두 계산을 설명하기 위한 가상 예측이다. 마지막 행은 사망한 승객을 거의 확실한 생존자로 예측했으므로 손실이 크다. 전체 로그 손실은 각 승객의 손실을 더한 뒤 승객 수로 나눈 평균이다. 실제 구현에서는 로그에 0이 들어가지 않도록 수치적으로 처리한다.

`log_loss` 함수에는 정답과 <strong>예측 확률</strong>을 전달한다. `0` 또는 `1`로 바꾼 최종 판단만 주면 원래 확률 정보를 잃는다. 정확도와 로그 손실은 서로 다른 특성을 평가하므로, 정확도가 더 높은 모델의 로그 손실이 반드시 더 작지는 않다.

## 11. 세 모델이 학습하는 규칙을 구분한다

로지스틱 회귀, 랜덤 포레스트, XGBoost는 모두 이 생존 분류 문제에 사용할 수 있다. 그러나 입력과 정답을 이용해 만드는 규칙이 다르다.

### 로지스틱 회귀는 특성별 계수를 학습한다

<strong>계수</strong>는 각 입력값에 곱하는 수다. 로지스틱 회귀는 입력값에 계수를 곱해 더하고 시작점을 조정하는 값인 <strong>절편</strong>을 더해 점수를 계산한다. 이 점수를 0과 1 사이의 생존 확률로 바꾸고, 훈련 정답에 높은 확률을 주도록 계수와 절편을 조정한다.[^logistic-model]

이 모델에는 결정트리가 없다. `max_iter`는 계산 알고리즘이 계수를 조정하는 최대 반복 횟수에 관한 설정이다. 트리 개수나 승객 수를 뜻하지 않는다.

### 랜덤 포레스트는 여러 트리의 확률을 평균 낸다

<strong>결정트리</strong>는 입력 조건으로 행을 나누고 마지막 지점에서 예측하는 모델이다. 트리 안에서 조건을 검사하거나 결과를 출력하는 구성 요소를 <strong>노드</strong>라고 하며, 더 나뉘지 않는 마지막 노드를 <strong>리프</strong>라고 한다.

랜덤 포레스트는 훈련 행을 중복 허용하여 뽑고, 분할에서 고려할 특성에도 무작위성을 주어 여러 트리를 학습한다. 각 트리는 생존 여부를 예측하는 완성된 모델이다. scikit-learn의 분류 랜덤 포레스트는 <strong>각 트리가 계산한 범주별 확률을 평균</strong> 내어 최종 예측에 사용한다.[^forest]

트리 수를 늘리면 무작위 구성에서 생기는 예측의 흔들림을 줄이는 데 도움이 될 수 있다. 다만 추가 효과는 작아질 수 있고, 계산량과 모델 크기는 커진다. 트리 수만 늘리면 검증 점수가 계속 오르거나 모든 과적합이 사라지는 것은 아니다.

### XGBoost는 현재 예측을 고치는 트리를 추가한다

XGBoost는 <strong>부스팅(boosting)</strong> 방식의 모델을 제공하는 라이브러리다. 이 글에서 다루는 트리 부스팅은 현재 전체 예측의 손실을 줄이는 방향으로 새 트리를 학습하고, 그 출력을 전체 점수에 더하는 과정을 반복한다.[^xgb-model]

이진 분류에서는 현재 생존 확률과 실제 정답을 바탕으로, 어떤 방향으로 점수를 바꾸면 손실이 줄어드는지 계산한다. 새 트리는 이 정보를 이용해 전체 예측에 더할 점수 변화량을 학습한다. 이때 사용하는 것은 최종 범주로 바꾸기 전의 점수와 확률이다.

새 트리를 추가하는 한 차례를 <strong>부스팅 단계</strong>라고 한다. 한 단계에서도 새 트리 내부의 여러 노드가 학습된다. 새 트리의 출력을 반영하는 비중은 <strong>학습률(learning rate)</strong>로 조절한다. 이 비중이 작으면 한 번의 보정이 작아지므로, 필요한 단계 수와 함께 비교해야 한다.

| 단위 | 가리키는 대상 |
| --- | --- |
| 노드 하나 | 한 트리 안의 분기 또는 출력 지점 하나 |
| 트리 하나 | 자기 분기와 리프를 모두 가진 개별 모델 하나 |
| 랜덤 포레스트 하나 | 여러 분류트리와 확률을 평균 내는 규칙 |
| XGBoost 분류 모델 하나 | 여러 단계에서 추가한 트리와 출력을 합치는 규칙 |
| 모델의 `fit` 한 번 | 해당 모델 전체를 학습하는 호출 한 번 |

예를 들어 랜덤 포레스트를 `fit` 한 번으로 학습해도 그 안에서는 설정한 수의 트리들을 만든다. 트리마다 `fit`을 사용자가 따로 호출해야 한다는 뜻은 아니다. 트리의 내부 동작은 [결정트리 기초](/decision-tree-basics/), 여러 트리의 결합은 [랜덤 포레스트와 그래디언트 부스팅](/random-forest-gradient-boosting/)에서 더 자세히 살펴볼 수 있다.

## 12. 학습 곡선은 가로축부터 읽는다

그래프에 손실이나 정확도가 그려져 있어도, 가로축이 무엇인지에 따라 해석은 달라진다. 흔히 학습 곡선이라고 부르는 그림에는 다음처럼 서로 다른 비교가 포함된다.[^curves]

| 가로축 | 무엇을 바꾸는가 | 살펴볼 내용 |
| --- | --- | --- |
| 훈련 데이터 수 | 학습에 사용하는 승객 수 | 데이터가 늘 때 훈련·검증 성능이 어떻게 달라지는가 |
| 트리 깊이 같은 설정값 | 모델이 표현할 수 있는 규칙의 범위 | 어떤 설정에서 검증 성능이 좋아지는가 |
| 최적화 반복 또는 부스팅 단계 | 한 모델의 학습 진행 정도 | 학습이 진행되면서 손실이 어떻게 달라지는가 |

scikit-learn의 `learning_curve` 함수는 첫 번째처럼 <strong>훈련 데이터 수</strong>를 바꿔 훈련·검증 점수를 구한다. `validation_curve` 함수는 두 번째처럼 <strong>지정한 설정값</strong>을 바꿔 점수를 구한다. 함수 이름과 그래프의 실제 가로축을 함께 확인해야 한다.

### 훈련 곡선과 검증 곡선은 함께 본다

<strong>과소적합(underfitting)</strong>은 모델이 필요한 규칙을 충분히 학습하지 못한 상태다. <strong>과적합(overfitting)</strong>은 훈련 기록에만 있는 세부적인 변화까지 따라가서 새로운 기록에서는 성능이 떨어지는 상태다.

| 관찰한 모습 | 먼저 검토할 가능성 |
| --- | --- |
| 훈련·검증 정확도가 모두 낮거나 손실이 모두 높다. | 모델이 너무 단순하거나, 필요한 입력이 부족하거나, 학습이 충분히 진행되지 않았을 수 있다. |
| 훈련 정확도는 높은데 검증 정확도는 낮다. | 훈련에만 잘 맞는 규칙을 학습했을 가능성이 있다. 분할과 데이터 차이도 살핀다. |
| 학습을 더 하자 훈련 손실은 줄지만 검증 손실은 지속해서 커진다. | 추가 학습이 새 데이터 예측에 도움이 되지 않는지 살핀다. |

손실은 작을수록, 정확도는 클수록 좋다. 이 방향을 바꾸어 읽으면 같은 곡선도 반대로 해석하게 된다. 또 한 번의 작은 흔들림만으로 과적합이라고 단정하지 않고 여러 폴드에서 비슷한 경향이 있는지 확인한다.

### 모델마다 ‘더 학습한다’의 뜻이 다르다

로지스틱 회귀에서 `max_iter=10`은 한 번의 `fit`에서 최적화 반복을 최대 10회 허용한다는 뜻이다. <strong>반드시 10회를 수행한다는 뜻은 아니다.</strong> `warm_start=True`는 이전에 구한 계수를 다음 학습의 시작값으로 재사용하는 설정이다.[^logistic-api]

이 상태에서 `fit`을 30번 호출해도, 하나의 최적화 과정을 정확히 300회 연속 실행했다고 볼 수 없다. 중간에 계산이 충분히 안정되었다고 판단해 일찍 멈출 수도 있고, 호출마다 계산 알고리즘의 내부 진행 상태가 다시 시작될 수도 있다. 따라서 호출 횟수만 세어 가로축을 `10, 20, …, 300회`로 표시하면 실제 측정한 값과 어긋날 수 있다.

계수가 충분히 안정되는 것을 <strong>수렴(convergence)</strong>이라고 한다. 수렴 경고가 나오면 입력 스케일, 반복 한도, 학습 설정을 확인한다. 경고를 전부 숨기면 학습이 제대로 끝났는지 판단할 단서도 사라진다.

랜덤 포레스트에서 트리 수를 늘리는 것은 같은 트리의 학습을 더 진행하는 일이 아니다. <strong>별도의 완성된 트리를 더 만들어 평균에 포함하는 일</strong>이다. 부스팅에서 단계를 늘리는 것은 현재 예측을 고치는 트리를 추가하는 일이다. 따라서 트리 개수 그래프를 로지스틱 회귀의 최적화 반복 그래프와 같은 뜻으로 읽을 수 없다.

부스팅에서는 검증 성능이 더 좋아지지 않을 때 추가 단계를 멈추는 <strong>조기 종료(early stopping)</strong>도 사용할 수 있다. 멈출 시점을 정하는 데 사용한 데이터 역시 모델 선택에 참여했으므로, 그 데이터는 마지막 독립 평가용과 구분한다.[^early-stopping]

## 13. 하이퍼파라미터 튜닝은 설정 후보를 같은 방법으로 비교한다

모델이 데이터에서 배우는 값과, 학습 전에 사람이 정하는 설정은 다르다. 로지스틱 회귀의 계수나 트리의 분할 조건은 데이터로 학습한다. 반면 트리를 몇 개 만들지, 깊이를 어디까지 허용할지 같은 설정은 <strong>하이퍼파라미터(hyperparameter)</strong>다. 그 후보들을 비교해 고르는 일을 <strong>튜닝(tuning)</strong>이라고 한다.

| 설정 | 실제로 조절하는 대상 |
| --- | --- |
| `n_estimators` | 랜덤 포레스트에서는 트리 수다. 이 글의 기본 이진 분류 XGBoost 설정에서는 부스팅 단계 수와 추가 트리 수가 대응한다. |
| `max_depth` | 트리 하나의 깊이 상한이다. 첫 노드인 루트의 깊이는 0이다. `max_depth=3`은 깊이 0·1·2·3의 최대 네 층을 허용하며, 실제 트리는 더 얕을 수 있다. |
| `min_samples_split` | 랜덤 포레스트의 노드에서 분할을 시도하기 위해 필요한 최소 샘플 수다. |
| `min_samples_leaf` | 랜덤 포레스트에서 각 리프에 남아 있어야 하는 최소 샘플 수다. |
| `learning_rate` | 부스팅에서 새 트리의 보정을 반영하는 비중이다. |
| `subsample` | XGBoost의 각 부스팅 단계에서 학습에 사용할 행의 비율이다. |
| `colsample_bytree` | XGBoost에서 트리 하나를 만들 때 선택할 특성의 비율이다. |

깊이를 제한하거나 리프에 필요한 샘플 수를 늘리면 지나치게 세밀한 규칙을 만들기 어려워진다. 이런 식으로 모델이 복잡해지는 것을 제한하는 방법을 <strong>규제 또는 정규화(regularization)</strong>라고 한다. 제한을 강하게 걸수록 무조건 좋은 것은 아니다. 필요한 규칙도 배우지 못하면 과소적합이 생길 수 있다.[^forest][^xgb-params]

### GridSearchCV는 지정한 조합을 모두 비교한다

`GridSearchCV`는 모델과 설정 후보를 받아 교차 검증으로 비교하는 클래스다. 후보값의 모든 조합을 살펴보므로, 후보를 늘리면 계산량도 늘어난다.[^grid]

예를 들어 XGBoost의 트리 수를 3가지, 깊이를 3가지, 학습률을 2가지, 행 비율을 2가지, 특성 비율을 2가지로 정했다면 다음처럼 계산한다.

<p class="titanic-formula"><code>설정 조합 수 = 3 × 3 × 2 × 2 × 2 = 72</code><br><code>5회 교차 검증의 모델 학습 수 = 72 × 5 = 360</code></p>

여기서 학습 360회는 <strong>XGBoost 모델 전체를 360번 학습</strong>한다는 뜻이다. 트리나 노드의 총개수가 360이라는 뜻은 아니다.

랜덤 포레스트에서 트리 수 3가지, 깊이 4가지, 최소 분할 샘플 수 3가지를 비교하면 `3×4×3=36`개 조합이고, 5회 교차 검증에는 `36×5=180`번의 모델 학습이 필요하다. 처음에는 작은 후보 범위에서 비교하고, 어떤 설정이 결과를 바꾸는지 확인한 뒤 범위를 넓힐 수 있다.

`RandomizedSearchCV`는 지정한 범위나 분포에서 설정 조합을 뽑아 정해진 횟수만큼 비교하는 클래스다. 후보가 많을 때 탐색량을 제한할 수 있지만, 모든 조합을 확인했다고 볼 수는 없다.[^random-search]

### 최고 점수와 최종 학습된 모델을 구분한다

`GridSearchCV`의 결과에는 다음처럼 의미가 다른 속성이 있다. <strong>속성(attribute)</strong>은 객체에 저장된 값이다.

| 속성 또는 설정 | 뜻 |
| --- | --- |
| `best_params_` | 비교한 후보 중 선택된 설정값이다. |
| `best_score_` | 선택된 설정의 평균 교차 검증 점수다. |
| `refit=True` | 설정을 선택한 뒤, 전달받은 개발용 데이터 전체로 다시 학습한다. |
| `best_estimator_` | 그 재학습까지 끝낸 모델이다. Pipeline을 전달했다면 전처리도 포함한다. |

따라서 앞의 XGBoost 예시는 교차 검증 360회에 개발용 전체 재학습 1회를 더한다. 폴드별 모델 다섯 개를 이어 붙이거나 자동으로 투표시킨 결과가 아니다.[^grid]

`best_score_`는 이미 수많은 후보를 고르는 데 사용한 점수다. 후보 중 우연히 그 검증 데이터에 잘 맞은 설정이 선택될 수도 있다. 이 값을 새로운 데이터에서의 확정적인 정확도로 해석하지 않고, 별도로 남겨 둔 최종 평가로 확인한다.[^selection]

## 14. 특성 중요도는 다음 분석의 단서로 사용한다

<strong>특성 중요도(feature importance)</strong>는 학습한 모델이 어떤 입력 특성을 활용했는지 살펴보는 수치다. 어떤 방식으로 계산했는지에 따라 의미가 달라진다.

랜덤 포레스트에서 흔히 사용하는 중요도는 분할을 통해 범주가 얼마나 더 잘 구분되었는지를 집계한다. XGBoost의 `gain` 방식은 해당 특성을 사용한 분할이 학습 목적을 얼마나 개선했는지에 기반한다. 다른 방식으로 중요도를 계산하면 순위도 달라질 수 있다.[^forest-importance][^xgb-api]

`feature_importances_`는 모델에 실제로 입력된 특성마다 중요도를 반환하는 속성이다. 원-핫 인코딩을 했다면 `Embarked` 한 칸이 여러 칸으로 바뀌었으므로, 중요도와 연결할 이름도 <strong>변환 뒤의 컬럼 이름</strong>이어야 한다. 원본 컬럼명을 그대로 붙이면 값과 이름이 맞지 않을 수 있다.

중요도가 높다고 해서 그 특성이 생존의 직접적인 원인임을 증명하지는 않는다. 예를 들어 객실 정보 유무와 생존 예측이 연결되어도, 그것이 승차권 등급이나 다른 기록의 차이를 함께 반영했을 수 있다. 예측에 유용한 관계와 현실의 원인·결과는 구분해야 한다.[^causal]

### 값을 섞었을 때 성능이 얼마나 떨어지는지도 볼 수 있다

<strong>순열 중요도(permutation importance)</strong>는 모델을 고정한 상태에서 한 컬럼의 값을 행 사이에 섞고, 평가 점수가 얼마나 달라지는지 비교한다. 승객과 그 컬럼 값의 연결을 흐트러뜨렸을 때 예측이 나빠지면, 그 모델이 해당 연결을 이용하고 있었다는 단서가 된다.[^permutation]

다만 비슷한 정보를 가진 다른 컬럼이 있으면 하나를 섞어도 모델이 나머지 컬럼으로 예측할 수 있다. 이때 중요도가 작다는 이유만으로 그 정보 전체가 쓸모없다고 결론 내릴 수는 없다.

이 분석은 새로운 특성을 만드는 데 연결된다. 예를 들어 가족 관련 정보가 유용해 보인다면 가족 수가 많은 집단과 적은 집단의 기록 수·생존율·오답을 다시 살펴볼 수 있다. 관계를 확인하고 새 특성을 제안한 뒤, 추가 전후를 같은 교차 검증으로 비교한다.

<strong>데이터 탐색 → 특성 후보 → 모델 검증 → 중요도·오답 분석 → 특성 수정</strong>의 반복이 개발용 데이터 안에서 이루어지는 것이다. 최종 평가용의 오답을 보고 계속 수정했다면, 그 점수를 더 이상 개발과 독립된 최종 평가라고 부를 수 없다.

## 15. 소프트 투표는 같은 승객의 예측 확률을 합친다

<strong>앙상블(ensemble)</strong>은 여러 모델의 예측을 결합하는 방법이다. 랜덤 포레스트 자체도 여러 트리를 결합한 앙상블이지만, 완성된 랜덤 포레스트 모델과 XGBoost 모델을 다시 결합할 수도 있다.

<strong>하드 투표(hard voting)</strong>는 각 모델이 고른 최종 범주를 모아 다수결을 한다. <strong>소프트 투표(soft voting)</strong>는 모델들의 범주별 확률을 평균 내고, 평균 확률이 가장 큰 범주를 선택한다.[^voting]

다음은 같은 승객 한 명에 대한 <strong>설명용 가상 예측</strong>이다.

| 모델 | 사망 확률 | 생존 확률 |
| --- | ---: | ---: |
| 랜덤 포레스트 | 0.70 | 0.30 |
| XGBoost | 0.10 | 0.90 |
| 같은 비중으로 평균 | `(0.70+0.10)÷2=0.40` | `(0.30+0.90)÷2=0.60` |

각 모델의 최종 판단은 사망과 생존으로 갈리지만, 확률을 평균 내면 생존 확률 0.60이 더 크다. 따라서 소프트 투표는 생존으로 예측한다. 이진 분류에서는 생존 확률이 0.5보다 큰지 보는 것과 연결되며, 정확히 같은 확률이면 구현의 동률 처리 규칙을 따른다.

여기서 평균을 낸 대상은 <strong>같은 승객에 대한 서로 다른 모델의 확률</strong>이다. 서로 다른 승객들의 확률을 한데 평균 내어 모두에게 같은 예측을 주는 계산이 아니다.

모델마다 다른 비중을 줄 수도 있다. 예를 들어 랜덤 포레스트에 1, XGBoost에 2의 비중을 주면 생존 확률은 `(1×0.30+2×0.90)÷(1+2)=0.70`이다. 이런 비중도 개발용 데이터에서 비교해 정할 설정이다.

### 확률을 합쳐도 성능이 좋아지는지는 별도로 확인한다

두 모델이 다른 부분에서 실수한다면 결합이 도움이 될 수 있다. 반대로 둘 다 같은 승객에게 비슷하게 틀리거나, 한 모델이 틀린 예측에 지나치게 높은 확률을 주면 평균이 불리해질 수 있다. 그래서 단일 모델과 앙상블을 같은 개발용 검증 절차로 비교한다. 소프트 투표에는 정확도뿐 아니라 예측 확률의 품질도 중요하다.[^voting-guide]

`VotingClassifier`는 이런 투표 모델을 만드는 클래스다. `voting="soft"`로 설정하고 `fit`을 호출하면 전달받은 모델들의 설정을 복사해 <strong>새 모델들을 다시 학습</strong>한다. 이미 학습한 모델을 전달해도 그 학습 결과를 그대로 보관해 묶는 동작은 아니다.[^voting]

전처리가 필요한 모델들을 교차 검증하려면, 각 모델에 전처리를 포함한 Pipeline을 넣어 투표 모델 전체를 검증할 수 있다. 최종 투표 모델 하나 안에 랜덤 포레스트 하나와 XGBoost 하나가 있고, 각각의 모델 안에 다시 여러 트리가 들어간다는 구조를 구분하면 된다.

## 16. 최종 평가를 마친 뒤 제출용 모델을 다시 학습한다

개발용 데이터로 특성, 전처리, 모델, 설정을 선택했다면 개발용 전체로 학습한 모델을 준비한다. `GridSearchCV`를 `refit=True`로 실행했다면 이 단계까지 완료된 모델을 얻는다.

그다음 처음에 남겨 둔 최종 평가용 입력을 예측하고, 보관한 정답과 비교한다. 여기서 전처리를 다시 학습해서는 안 된다. 개발용에서 저장한 중앙값, 범주 목록과 모델을 그대로 적용한다.

<strong>이 점수는 선택한 구성을 새로운 기록에서 확인한 결과</strong>다. 평가 표가 유한하므로 실제 미래 성능과 정확히 일치한다고 보장되지는 않는다. 이 결과를 보고 설정을 계속 바꾸면 해당 평가 표도 선택에 사용되므로, 이후 점수는 같은 의미의 독립 평가가 아니다.[^cv]

모델의 구성을 확정하고 제출 파일을 만들 때는 `train.csv`의 <strong>정답이 있는 전체 기록</strong>으로 전처리와 모델을 새로 학습할 수 있다. 최종 평가에 남겨 두었던 20%도 이제 제출용 학습에 포함한다. 더 많은 정답 기록으로 최종 예측 모델을 만드는 단계다.

이렇게 다시 학습한 모델은 앞에서 평가한 모델과 학습 데이터가 다르다. 앞의 홀드아웃 점수를 전체 재학습 모델을 독립적으로 평가한 점수라고 바꾸어 부르지는 않는다.

제출용 모델에는 캐글 `test.csv`를 입력한다. 전처리는 전체 정답 데이터에서 새로 구한 기준을 사용하며, 테스트 파일에서 중앙값이나 범주 목록을 다시 학습하지 않는다. 마지막으로 승객마다 예측한 `0` 또는 `1`을 원래 `PassengerId`와 연결한다.[^titanic]

`submission.csv`에는 `PassengerId`, `Survived` 두 컬럼을 저장한다. 행 수는 예측한 테스트 승객 수와 같아야 하며, 각 식별 번호와 예측의 대응이 유지되어야 한다. pandas의 `to_csv(..., index=False)`는 DataFrame의 행 인덱스를 별도 컬럼으로 쓰지 않도록 하는 설정이다.[^csv]

## 17. 핵심 코드: 전처리부터 제출까지 같은 순서를 유지한다

다음 코드는 <strong>전처리 → 교차 검증으로 설정 선택 → 최종 평가 → 전체 재학습 → 제출 파일 저장</strong>을 실행한다. 모델은 랜덤 포레스트 하나를 사용하고, 파생 특성은 `FamilySize`와 `Has_Cabin`을 포함한다.

나이와 요금은 각각 해당 훈련 부분의 중앙값으로 채운다. 앞서 살펴본 등급별 나이 대체, 티켓 빈도, 이름 길이와 로그 변환은 추가로 비교할 특성 후보다. 이 코드의 기본 모델에는 포함하지 않았다. 앙상블도 단일 모델과의 검증 비교를 거쳐 선택하는 별도 후보다.

코드를 읽는 데 필요한 도구는 세 가지다. 모두 앞에서 설명한 처리 순서를 코드로 연결하기 위한 것이다.

| 도구 | 역할과 반환값 |
| --- | --- |
| `FunctionTransformer` 클래스 | 직접 만든 변환 함수를 Pipeline 단계로 감싼다. 아래 함수는 입력 DataFrame을 복사하고 두 특성을 추가한 DataFrame을 반환한다. |
| `ColumnTransformer` 클래스 | 수치형 컬럼과 범주형 컬럼에 서로 다른 전처리를 적용한 뒤, 결과를 컬럼 방향으로 붙여 숫자 표를 만든다. |
| `clone` 함수 | 모델의 설정을 복사한 학습 전 객체를 만든다. 전체 데이터로 새로 학습할 제출용 Pipeline을 준비한다. |

`ColumnTransformer`에는 사용할 컬럼을 명시한다. 나머지 원본 컬럼은 기본 동작에 따라 제외되므로 `PassengerId`, `Name`, `Ticket`, 원래 `Cabin` 문자열은 모델에 들어가지 않는다.[^column-transformer][^function-transformer][^clone]

`model__max_depth`에서 `model`은 Pipeline의 마지막 단계에 붙인 이름이고, 두 개의 밑줄 `__` 뒤는 그 모델의 설정 이름이다. 아래에서는 깊이 3가지와 리프 최소 인원 2가지를 비교하므로 총 6개 조합, 5회 교차 검증에서 30번 학습한 뒤 선택된 설정으로 한 번 더 학습한다.

실행에는 pandas와 scikit-learn, 캐글의 `data/train.csv`와 `data/test.csv`가 필요하다. 코드는 두 CSV의 원래 컬럼 이름을 사용한다. `make_features`는 다른 행의 통계를 구하지 않으며, `validate=False`는 이 함수에 컬럼 이름이 있는 DataFrame을 그대로 전달하게 하는 설정이다.

<details class="titanic-code">
<summary>핵심 Python 코드 펼치기</summary>

```python
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def make_features(frame):
    # 현재 행의 정보만 계산한다. 원본 표를 바꾸지 않도록 복사한다.
    result = frame.copy()
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["Has_Cabin"] = result["Cabin"].notna().astype(int)
    return result  # 승객 행의 순서와 개수는 유지한다.


train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
X = train_df.drop(columns="Survived")  # 입력에서 정답을 제외한다.
y = train_df["Survived"]             # 입력 행과 같은 순서의 정답이다.

# 최종 평가용 20%는 이후 설정 탐색에 사용하지 않는다.
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_columns = ["Age", "Fare", "FamilySize", "Has_Cabin"]
category_columns = ["Pclass", "Sex", "Embarked"]

# 아래 객체들은 전처리 구성을 만든다. 아직 통계를 학습하지 않는다.
category_steps = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocess = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_columns),
    ("category", category_steps, category_columns),
])
pipeline = Pipeline([
    ("features", FunctionTransformer(make_features, validate=False)),
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=1
    )),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline,
    param_grid={"model__max_depth": [3, 5, 7],
                "model__min_samples_leaf": [1, 3]},
    cv=cv, scoring="accuracy", n_jobs=-1, refit=True, error_score="raise",
)

# 각 폴드의 훈련 행으로 전처리부터 새로 학습한다.
# 선택 후에는 개발용 전체로 Pipeline을 다시 학습한다.
search.fit(X_dev, y_dev)
print("선택한 설정:", search.best_params_)
print("설정 선택에 사용한 CV 평균 정확도:", search.best_score_)

# 확정한 구성으로 최종 평가한다. 여기서는 predict만 수행한다.
holdout_pred = search.predict(X_holdout)
holdout_proba = search.predict_proba(X_holdout)
print("최종 평가 정확도:", accuracy_score(y_holdout, holdout_pred))
print("최종 평가 로그 손실:", log_loss(y_holdout, holdout_proba))

# 제출용 객체를 만들어 전체 정답 데이터로 전처리부터 새로 학습한다.
submission_model = clone(search.best_estimator_)
submission_model.fit(X, y)
test_pred = submission_model.predict(test_df)

# 테스트 승객의 순서에 맞춰 식별 번호와 예측을 함께 저장한다.
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"].to_numpy(),
    "Survived": test_pred.astype(int),
})
submission.to_csv("submission.csv", index=False)
```

</details>

`n_jobs=-1`은 탐색의 여러 학습 작업을 병렬로 실행하는 설정이다. 모델 내부는 `n_jobs=1`로 두어 두 계층에서 동시에 작업 수를 늘리지 않았다. `error_score="raise"`는 학습 오류를 숨기지 않고 즉시 확인하도록 한다. `sparse_output=False`는 원-핫 결과를 일반적인 2차원 숫자 배열로 반환하게 한다.

이 코드에서는 캐글의 실제 점수나 최적 설정을 미리 제시하지 않는다. 숫자 계산 예시는 개념 설명용이며, 위 코드는 별도로 만든 가상 표에서 전처리와 예측 흐름을 확인했다. 실제 정확도와 어떤 특성이 유용한지는 사용하는 CSV와 검증 결과로 확인해야 한다.

[^titanic]: Kaggle, [Titanic — 데이터 설명과 제출 과제](https://www.kaggle.com/competitions/titanic/data).
[^pitfalls]: scikit-learn, [Common pitfalls and recommended practices — Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
[^classifier]: scikit-learn, [RandomForestClassifier — predict, predict_proba, classes_](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
[^split]: scikit-learn, [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
[^selection]: scikit-learn, [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html).
[^pandas-map]: pandas, [Series.map](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html), [SeriesGroupBy.median](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.median.html).
[^imputer]: scikit-learn, [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).
[^pandas-size]: pandas, [DataFrameGroupBy.size](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.size.html).
[^log1p]: NumPy, [numpy.log1p](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html).
[^logistic-model]: scikit-learn, [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
[^scaling]: scikit-learn, [Importance of Feature Scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html).
[^scaler]: scikit-learn, [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
[^encoder]: scikit-learn, [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
[^dummies]: pandas, [get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html).
[^pipeline]: scikit-learn, [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
[^compose]: scikit-learn, [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
[^cv]: scikit-learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html).
[^stratified]: scikit-learn, [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).
[^accuracy]: scikit-learn, [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).
[^dummy]: scikit-learn, [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html).
[^logloss]: scikit-learn, [log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).
[^forest]: scikit-learn, [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
[^xgb-model]: XGBoost, [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).
[^curves]: scikit-learn, [Validation curves and learning curves](https://scikit-learn.org/stable/modules/learning_curve.html).
[^logistic-api]: scikit-learn, [LogisticRegression — max_iter, warm_start](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
[^early-stopping]: XGBoost, [Using the Scikit-Learn Estimator Interface — Early Stopping](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#early-stopping).
[^xgb-params]: XGBoost, [Parameters for Tree Booster](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster).
[^grid]: scikit-learn, [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
[^random-search]: scikit-learn, [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
[^forest-importance]: scikit-learn, [Feature importances with a forest of trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html).
[^xgb-api]: XGBoost, [Python API — feature_importances_, n_jobs](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
[^causal]: scikit-learn, [Failure of Machine Learning to infer causal effects](https://scikit-learn.org/stable/auto_examples/inspection/plot_causal_interpretation.html).
[^permutation]: scikit-learn, [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
[^voting]: scikit-learn, [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html).
[^voting-guide]: scikit-learn, [Voting Classifier — Soft voting](https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting).
[^csv]: pandas, [DataFrame.to_csv](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html).
[^column-transformer]: scikit-learn, [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).
[^function-transformer]: scikit-learn, [FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html).
[^clone]: scikit-learn, [clone](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html).
