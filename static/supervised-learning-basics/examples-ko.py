"""지도 학습 기본 개념 및 기초 코딩: 배송비 예제부터 실제 데이터까지.

필요한 패키지: numpy, matplotlib, scikit-learn
Jupyter에서 설치할 때는 별도 셀에서 %pip install numpy matplotlib scikit-learn
아래 코드는 위에서 아래로 한 번에 실행한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=6, suppress=True)

# 배송비를 예측하는 가상의 단순화된 자료다.
# x_one: 기본 배송 구간을 넘긴 추가 거리(km), y_one: 배송비(천 원).
# b는 기본요금, w는 추가 거리 1km당 요금이며, 처음에는 b=0, w=1로 둔다.
x_one = np.array([0.0, 1.0, 2.0])
y_one = np.array([1.0, 3.0, 5.0])
b_initial, w_initial = 0.0, 1.0
pred_one = b_initial + w_initial * x_one
e_one = pred_one - y_one
sse = np.sum(e_one ** 2)
mse = np.mean(e_one ** 2)
j_before = mse / 2
mae = np.mean(np.abs(e_one))
rmse = np.sqrt(mse)
print("초기 배송비 예측:", pred_one)
print("오차 = 예측값 - 정답:", e_one)
print("SSE, MSE, J, MAE, RMSE:", sse, mse, j_before, mae, rmse)
assert np.allclose(e_one, [-1.0, -2.0, -3.0], atol=1e-12, rtol=0)
assert np.isclose(j_before, 7 / 3, atol=1e-12, rtol=0)

# 기본요금만 0.1천 원 올려 본다. 원래의 거리당 요금은 그대로다.
b_trial = b_initial + 0.1
pred_b_trial = b_trial + w_initial * x_one
e_b_trial = pred_b_trial - y_one
j_b_trial = np.mean(e_b_trial ** 2) / 2
print("b만 0.1 올린 예측 / 오차 / J:", pred_b_trial, e_b_trial, j_b_trial)
assert np.allclose(pred_b_trial, [0.1, 1.1, 2.1], atol=1e-12, rtol=0)
assert np.allclose(e_b_trial, [-0.9, -1.9, -2.9], atol=1e-12, rtol=0)
assert np.isclose(j_b_trial, 1283 / 600, atol=1e-12, rtol=0)

# 이번에는 다시 원래 b=0, w=1에서 시작해 거리당 요금만 0.1 올린다.
# 위 실험에서 바꾼 b_trial을 이어서 사용하지 않는다.
w_trial = w_initial + 0.1
pred_w_trial = b_initial + w_trial * x_one
e_w_trial = pred_w_trial - y_one
j_w_trial = np.mean(e_w_trial ** 2) / 2
print("w만 0.1 올린 예측 / 오차 / J:", pred_w_trial, e_w_trial, j_w_trial)
assert np.allclose(pred_w_trial, [0.0, 1.1, 2.2], atol=1e-12, rtol=0)
assert np.allclose(e_w_trial, [-1.0, -1.9, -2.8], atol=1e-12, rtol=0)
assert np.isclose(j_w_trial, 83 / 40, atol=1e-12, rtol=0)

# 미분값을 이용해 기본요금과 거리당 요금을 동시에 한 번 갱신한다.
# 두 미분값을 모두 원래 b=0, w=1의 오차에서 먼저 구한다.
db = np.mean(e_one)
dw = np.mean(e_one * x_one)
lr_one = 0.1
b = b_initial - lr_one * db
w = w_initial - lr_one * dw
pred_after = b + w * x_one
mse_after = np.mean((pred_after - y_one) ** 2)
j_after = mse_after / 2
print("갱신 전 db, dw:", db, dw)
print("한 번 갱신 후 b, w, J:", b, w, j_after)
assert np.allclose([db, dw], [-2.0, -8 / 3], atol=1e-12, rtol=0)
assert np.allclose([b, w], [0.2, 19 / 15], atol=1e-12, rtol=0)
assert np.isclose(j_after, 1829 / 1350, atol=1e-12, rtol=0)

# 같은 배송 자료 세 행을 행렬로 써도 예측과 미분값은 같다.
# 앞의 1열은 기본요금 b를 각 배송 건에 한 번씩 더하기 위한 것이다.
Xb_one = np.column_stack([np.ones(len(y_one)), x_one])
theta_one = np.array([b_initial, w_initial])
pred_matrix_one = Xb_one @ theta_one
gradient_one = Xb_one.T @ (pred_matrix_one - y_one) / len(y_one)
theta_after_matrix = theta_one - lr_one * gradient_one
print("세 행의 배송 자료 Xb:\n", Xb_one)
print("행렬로 구한 예측 / [db, dw]:", pred_matrix_one, gradient_one)
print("행렬로 한 번 갱신한 [b, w]:", theta_after_matrix)
assert np.array_equal(Xb_one, [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
assert np.allclose(pred_matrix_one, pred_one, atol=1e-12, rtol=0)
assert np.allclose(gradient_one, [db, dw], atol=1e-12, rtol=0)
assert np.allclose(theta_after_matrix, [b, w], atol=1e-12, rtol=0)

# 배송 건이나 거리 열을 골라 보고, 배열의 행·열 모양도 함께 확인한다.
print("두 번째 배송 행:", Xb_one[1], "shape =", Xb_one[1].shape)
print("두 번째 배송을 한 행으로 유지:", Xb_one[[1]], "shape =", Xb_one[[1]].shape)
print("거리 열:", Xb_one[:, 1], "shape =", Xb_one[:, 1].shape)
print("거리 열을 한 열로 유지:", Xb_one[:, [1]], "shape =", Xb_one[:, [1]].shape)

# 예측과 정답은 같은 배송 건끼리 빼야 한다.
# 한쪽만 세로 한 열로 바꾸면, 의도와 다른 3×3 차이가 만들어진다.
print("예측과 정답의 shape:", pred_one.shape, y_one.shape)
print("정답을 한 열로 바꾼 shape:", y_one[:, None].shape)
print("올바른 같은 배송 건의 오차:", pred_one - y_one)
print("의도와 다른 3×3 결과:\n", pred_one - y_one[:, None])
assert pred_one.shape == y_one.shape

# 같은 배송 자료에서 w=1을 고정하고 기본요금만 갱신해 학습률을 비교한다.
# 이때 db는 b-2다. 각 실험은 반드시 같은 b=0에서 새로 시작한다.
learning_rate_results = []
for rate in [0.2, 0.8, 1.5, 2.2]:
    b_rate = 0.0
    db_rate = b_rate - 2.0
    b_rate = b_rate - rate * db_rate
    pred_rate = b_rate + 1.0 * x_one
    j_rate = np.mean((pred_rate - y_one) ** 2) / 2
    learning_rate_results.append([rate, b_rate, j_rate])
    print("학습률 / 첫 갱신 b / J:", rate, b_rate, j_rate)
assert np.allclose(np.array(learning_rate_results)[:, 1], [0.4, 1.6, 3.0, 4.4],
                   atol=1e-12, rtol=0)
assert np.allclose(np.array(learning_rate_results)[:, 2],
                   [121 / 75, 31 / 75, 5 / 6, 241 / 75], atol=1e-12, rtol=0)

# 입력 두 개로 확장한다: u는 추가 거리(km), v는 특수 포장 여부(없음 0, 있음 1).
# 네 배송 건의 요금은 여전히 천 원 단위다.
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = np.array([1.0, 3.0, 4.0, 6.0])
n = len(y)
Xb = np.column_stack([np.ones(n), X])
theta = np.zeros(3)              # [기본요금 b, 거리당 요금 w_u, 특수 포장 요금 w_v]
e0 = Xb @ theta - y
g0 = Xb.T @ e0 / n
j0 = np.mean(e0 ** 2) / 2
print("네 행의 배송 자료 Xb:\n", Xb)
print("초기 J, gradient:", j0, g0)

# 실제 학습: 요금 세 개를 모두 0에서 시작해 학습률 0.2로 2000회 고친다.
lr = 0.2
n_steps = 2000
history = []
for _ in range(n_steps):
    pred = Xb @ theta
    e = pred - y
    history.append(float(np.mean(e ** 2) / 2))  # 이번 갱신을 하기 직전의 손실
    gradient = Xb.T @ e / n
    theta = theta - lr * gradient

theta_gd = theta.copy()
pred_gd = Xb @ theta_gd
final_j = float(np.mean((pred_gd - y) ** 2) / 2)
history_complete = np.array(history + [final_j])
print("GD 계수 [b, w_u, w_v]:", theta_gd)
print("GD 배송비 예측:", pred_gd)
print("첫 손실 / 마지막 저장 손실 / 최종 손실:", history[0], history[-1], final_j)
print("2000회 갱신 뒤 손실까지 포함한 길이:", len(history_complete))

fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
shown = min(81, len(history_complete))  # 시작점과 처음 80회 갱신까지의 손실을 표시한다.
ax_loss.plot(np.arange(shown), history_complete[:shown])
ax_loss.set(xlabel="Completed updates", ylabel="J = MSE / 2",
            title="Gradient descent on four delivery samples")
ax_loss.grid(alpha=0.25)
fig_loss.tight_layout()

# 최소제곱 풀이와 scikit-learn도 같은 네 배송 건에 맞춰 본다.
# 최소제곱 결과의 첫 항목이 우리가 필요한 계수 배열이다.
result = np.linalg.lstsq(Xb, y)
theta_ls = result[0]
pred_ls = Xb @ theta_ls
model_small = LinearRegression()
fit_return = model_small.fit(X, y)  # 기본요금은 모델이 처리하므로 앞에 1을 붙이지 않은 X를 쓴다.
pred_sk = model_small.predict(X)
print("최소제곱 계수:", theta_ls)
print("fit이 같은 모델을 반환하는가:", fit_return is model_small)
print("sklearn 기본요금 / 거리·포장 요금:", model_small.intercept_, model_small.coef_)
print("최소제곱 / GD / sklearn 예측을 열로 비교:\n", np.column_stack([pred_ls, pred_gd, pred_sk]))
new_input = np.array([1.0, 3.0, 1.0])  # [기본요금용 1, 추가 거리 3km, 특수 포장 있음]
new_prediction = new_input @ theta_gd
print("추가 거리 3km, 특수 포장 있음의 배송비(천 원):", new_prediction)

# 세 방식이 같은 배송비를 예측하는지, 같은 모양의 배열끼리 비교한다.
assert theta_gd.shape == (3,)
assert pred_gd.shape == y.shape
assert np.isfinite(theta_gd).all()
assert np.isfinite(history_complete).all()
assert np.isclose(j0, 7.75, atol=1e-12, rtol=0)
assert np.allclose(g0, [-3.5, -2.25, -2.5], atol=1e-12, rtol=0)
assert np.allclose(theta_gd, [1.0, 2.0, 3.0], atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_ls, atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_sk, atol=1e-8, rtol=0)
assert np.isclose(new_prediction, 10.0, atol=1e-8, rtol=0)
assert final_j < history_complete[0]
print("구현 검증 통과. GD/최소제곱 최대 예측 차이:", np.max(np.abs(pred_gd - pred_ls)))

# 실제 자료로 확장한다. 로더의 추가 스케일링 없이 저장된 특성값을 가져온다.
X_real, y_real = load_diabetes(return_X_y=True, scaled=False)
# 같은 행 번호를 입력과 정답에 함께 적용해 짝을 유지한다.
all_ids = np.arange(len(y_real))
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
X_train, X_test = X_real[train_ids], X_real[test_ids]
y_train, y_test = y_real[train_ids], y_real[test_ids]
assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]
print("전체 / 학습 / 테스트 입력 shape:", X_real.shape, X_train.shape, X_test.shape)

# 모든 테스트 행에 학습 정답의 평균을 예측하는 기준선을 만든다.
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_test)

# BMI 하나만 쓰는 모델과 열 개 특성을 모두 쓰는 모델을 각각 학습한다.
model_bmi = LinearRegression()
model_bmi.fit(X_train[:, [2]], y_train)
pred_bmi = model_bmi.predict(X_test[:, [2]])
model_all = LinearRegression()
model_all.fit(X_train, y_train)
pred_all = model_all.predict(X_test)
predictions = {"baseline": pred_baseline, "bmi": pred_bmi, "all": pred_all}
scores = {}
for name, prediction in predictions.items():
    assert prediction.shape == y_test.shape
    assert np.isfinite(prediction).all()
    scores[name] = mean_squared_error(y_test, prediction)
    print(name, "테스트 MSE:", scores[name])
print("기준선에 쓰인 학습 정답 평균:", y_train.mean())

# 잔차는 실제값 - 예측값이므로, 앞에서 쓴 오차 e와 부호가 반대다.
residual = y_test - pred_all
print("첫 테스트 원본 행 번호:", test_ids[0])
print("첫 테스트 BMI:", X_test[0, 2])
print("첫 테스트 정답 / 전체 모델 예측 / 잔차:", y_test[0], pred_all[0], residual[0])
assert test_ids[0] == 287
assert np.isclose(X_test[0, 2], 25.8, atol=1e-12, rtol=0)
assert y_test[0] == 219.0
fig_residual, ax_residual = plt.subplots(figsize=(7, 4))
ax_residual.scatter(pred_all, residual, alpha=0.65, label="Test samples")
ax_residual.scatter([pred_all[0]], [residual[0]], color="crimson",
                    s=90, label="First test sample")
ax_residual.axhline(0, color="black", linewidth=1)
ax_residual.set(xlabel="Prediction", ylabel="Residual = target - prediction",
                title="Residuals on 89 held-out samples")
ax_residual.legend()
fig_residual.tight_layout()
plt.show()
