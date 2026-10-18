"""집 4채로 결정트리, 배깅, 랜덤 포레스트, 회귀 부스팅의 계산을 확인한다.

필요한 패키지: numpy, scikit-learn.
데이터는 계산 과정을 설명하기 위한 가상 자료다. 집값의 단위는 억 원이다.
모든 행의 가중치는 같고, 회귀트리는 제곱오차를 기준으로 학습한다.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

# 한 행은 집 한 채다. 컬럼 순서는 [역세권, 리모델링, 주차장]이다.
# 각 입력의 0은 아니오, 1은 예를 뜻한다.
X = np.array([
    [0, 0, 1],  # 집 A
    [0, 1, 0],  # 집 B
    [1, 0, 0],  # 집 C
    [1, 1, 1],  # 집 D
])
y = np.array([5.0, 7.0, 11.0, 17.0])  # 같은 순서의 실제 집값
new_X = np.array([[1, 0, 1]])  # 새 집 E: 1행, 3컬럼. 실제 집값은 모른다.

print("입력과 정답의 크기:", X.shape, y.shape)  # (4, 3), (4,)

# 1. 집값 자체를 학습하는 회귀트리 하나를 만든다.
single_tree = DecisionTreeRegressor(
    criterion="squared_error",  # 분할 조건을 제곱오차로 평가한다.
    max_depth=1,                # 루트 깊이 0에서 깊이 1까지만 허용한다.
    random_state=42,            # 무작위 선택을 재현하기 위한 설정이다.
)
fit_result = single_tree.fit(X, y)  # 트리 전체의 조건과 리프 값을 학습한다.
assert fit_result is single_tree   # fit의 반환값은 모델 객체 자신이다.
print("집값을 직접 학습한 트리:", single_tree.predict(X))  # [6, 6, 14, 14]

# 2. 서로 다른 학습 표본으로 모델 세 개를 학습하는 배깅이다.
bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=1),
    n_estimators=3,
    bootstrap=True,    # 학습 행을 복원추출한다.
    max_features=1.0,   # 각 모델에 전체 컬럼을 전달한다.
    random_state=42,
)
bagging.fit(X, y)       # 각 모델은 추출된 집의 원래 집값을 학습한다.
print("배깅의 모델 수:", len(bagging.estimators_))  # 3

# 3. 랜덤 포레스트의 max_features는 노드에서 검토할 후보 컬럼 수다.
forest = RandomForestRegressor(
    n_estimators=3,
    max_depth=1,
    bootstrap=True,
    max_features=2,    # 이 예제에서는 각 분할에서 후보 컬럼 두 개를 선택한다.
    random_state=42,
)
forest.fit(X, y)

# 각 트리는 같은 새 집에 대한 집값을 예측한다. 결과들을 평균 내면 최종 예측이다.
forest_outputs = np.array([tree.predict(new_X) for tree in forest.estimators_])
print("포레스트의 개별 집값 예측:", forest_outputs)
print("포레스트의 평균 집값 예측:", forest.predict(new_X))
np.testing.assert_allclose(forest.predict(new_X), forest_outputs.mean(axis=0))

# 4. 부스팅의 바깥 반복을 직접 작성한다. 앞의 모델들과 별도로 학습한다.
initial_value = y.mean()  # 학습 정답의 평균 10.0을 계산한다.
learning_rate = 1.0      # 이번 계산에서는 새 보정값을 전부 반영한다.
current_prediction = np.full(y.shape, initial_value)  # [10, 10, 10, 10]
trees = []              # 학습한 트리를 보관할 리스트다.

print("초기 전체 예측:", current_prediction)
print("초기 MSE:", np.mean((y - current_prediction) ** 2))  # 21.0

# range(1, 4)는 1, 2, 3을 차례로 제공한다. 마지막 4는 포함하지 않는다.
for step in range(1, 4):
    residual = y - current_prediction  # 각 집의 현재 잔차를 계산한다.

    # 반복할 때마다 별도의 새 트리를 만든다.
    tree = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=1,
        random_state=42,
    )
    tree.fit(X, residual)       # 이 새 트리 전체를 잔차에 맞춰 학습한다.
    correction = tree.predict(X)  # 각 집의 보정값 네 개를 계산한다.

    # 새 트리의 예측을 기존 전체 예측에 더한다.
    current_prediction = current_prediction + learning_rate * correction
    trees.append(tree)         # 완성된 트리를 저장한다. 이전 트리는 수정하지 않는다.
    mse = np.mean((y - current_prediction) ** 2)

    print(f"\n부스팅 {step}단계")
    print("학습 목표인 잔차:", residual)
    print("새 트리의 보정 예측:", correction)
    print("갱신된 전체 집값 예측:", current_prediction)
    print("전체 MSE:", mse)    # 단계별로 5.0, 1.0, 0.0

np.testing.assert_allclose(current_prediction, y)

# 5. 학습한 트리 내부를 확인한다. 인덱스 0은 첫 번째 원소를 뜻한다.
first_tree = trees[0]
print("첫 트리의 실제 깊이:", first_tree.get_depth())       # 1
print("첫 트리의 노드 수:", first_tree.tree_.node_count)     # 3
print("집별 리프 노드 번호:", first_tree.apply(X))           # [1, 1, 2, 2]

# 6. 새 집을 예측한다. fit이나 실제 잔차 계산은 하지 않는다.
new_prediction = np.full(new_X.shape[0], initial_value)  # 집 한 채의 시작값 [10]
for tree in trees:
    new_prediction = new_prediction + learning_rate * tree.predict(new_X)

print("\n새 집 E의 부스팅 예측:", new_prediction)  # [13.]
np.testing.assert_allclose(new_prediction, [13.0])

# 7. 같은 설정으로 라이브러리의 전체 부스팅 모델을 학습한다.
boosting = GradientBoostingRegressor(
    loss="squared_error",  # 전체 모델이 줄일 손실을 지정한다.
    n_estimators=3,        # 보정 트리 세 개를 순서대로 학습한다.
    max_depth=1,           # 개별 트리마다 적용하는 깊이 상한이다.
    learning_rate=1.0,
    random_state=42,
)
boosting.fit(X, y)  # 실제 집값을 전달한다. 라이브러리가 내부 잔차를 계산한다.

print("라이브러리의 새 집 예측:", boosting.predict(new_X))  # [13.]
print("내부 트리 배열의 크기:", boosting.estimators_.shape)  # (3, 1)

# staged_predict는 각 단계까지 누적한 전체 예측 배열을 차례로 제공한다.
for stage_prediction in boosting.staged_predict(X):
    print("라이브러리의 단계별 전체 예측:", stage_prediction)

# 두 구현이 이 데이터와 설정에서 같은 결과를 내는지 검사한다.
# assert_allclose는 값이 허용 오차 안에서 같으면 통과하고, 다르면 오류를 낸다.
np.testing.assert_allclose(boosting.predict(X), current_prediction)
np.testing.assert_allclose(boosting.predict(new_X), new_prediction)
