"""지도 학습 기본 개념 및 기초 코딩: 처음부터 실행하는 Python 예제.

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

# 화면에 보이는 소수 자릿수만 바꾼다. 계산 정밀도를 바꾸지는 않는다.
np.set_printoptions(precision=6, suppress=True)

# 1. 오차, 제곱합, 평균제곱오차, 학습에 쓸 손실 J
y_demo = np.array([10.0, 20.0, 30.0])
pred_demo = np.array([9.0, 18.0, 27.0])
e_demo = pred_demo - y_demo
sse = np.sum(e_demo ** 2)       # ** 2: 각 원소를 제곱
mse = np.mean(e_demo ** 2)      # mean: 원소의 합을 원소 수로 나눔
j_demo = mse / 2                # 1/2은 MSE 자체의 정의에 포함되지 않음
mae = np.mean(np.abs(e_demo))    # abs: 부호를 없앤 크기
rmse = np.sqrt(mse)             # sqrt: 제곱근
print("오차:", e_demo)
print("SSE, MSE, J, MAE, RMSE:", sse, mse, j_demo, mae, rmse)

# 2. 인덱싱: 행과 열을 고르면서 결과의 shape도 확인
A = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
row_ids = [0, 2]
index_examples = {
    "A[1]": A[1],
    "A[[1]]": A[[1]],
    "A[:, 2]": A[:, 2],
    "A[:, [2]]": A[:, [2]],
    "A[row_ids][1]": A[row_ids][1],
    "A[row_ids][:, [2]]": A[row_ids][:, [2]],
    "A[[0, 2], [1, 2]]": A[[0, 2], [1, 2]],
}
# items(): 사전에서 이름과 결과를 한 쌍씩 꺼냄
for label, values in index_examples.items():
    print(label, "=", values, "shape =", values.shape)
print("A[1][2] =", A[1][2])

# 3. 축 추가, 모양 변경, broadcasting으로 생길 수 있는 실수
v = np.array([1.0, 3.0, 5.0])
p = np.array([0.0, 2.0, 4.0])
print("v, v.T의 shape:", v.shape, v.T.shape)
print("v[:, None]:", v[:, None], "shape =", v[:, None].shape)
print("v[None, :]:", v[None, :], "shape =", v[None, :].shape)
print("reshape(-1, 1):", v.reshape(-1, 1))
print("올바른 같은 위치의 오차:", p - v)
print("의도와 다른 3×3 결과:\n", p - v[:, None])
assert p.shape == v.shape

# 4. 미분을 이용해 입력 하나인 모델을 한 번 갱신
x_one = np.array([0.0, 1.0, 2.0])
y_one = np.array([1.0, 3.0, 5.0])
w, b = 1.0, 0.0
lr_one = 0.1
pred_one = b + w * x_one
e_one = pred_one - y_one
j_before = np.mean(e_one ** 2) / 2
dw = np.mean(e_one * x_one)      # J를 w로 미분한 현재 값
db = np.mean(e_one)              # J를 b로 미분한 현재 값
# 두 기울기를 모두 기존 w, b에서 구한 뒤 갱신한다.
w = w - lr_one * dw
b = b - lr_one * db
pred_after = b + w * x_one
mse_after = np.mean((pred_after - y_one) ** 2)
print("한 번 갱신 전 J, dw, db:", j_before, dw, db)
print("갱신 후 w, b, MSE:", w, b, mse_after)
assert np.isclose(j_before, 7 / 3, atol=1e-12, rtol=0)
assert np.isclose(mse_after, 1829 / 675, atol=1e-12, rtol=0)

# 5. 입력 두 개를 쓰는 별도의 4행 데이터
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = np.array([1.0, 3.0, 4.0, 6.0])
n = len(y)
# ones(n): 1을 n개 생성. c_: 배열들을 열 방향으로 나란히 연결.
Xb = np.c_[np.ones(n), X]
theta = np.zeros(3)              # 계수 순서: [b, w1, w2]
e0 = Xb @ theta - y              # @: 각 행과 계수 벡터의 행렬곱
g0 = Xb.T @ e0 / n              # .T: 이 2차원 배열의 행과 열을 바꿈
j0 = np.mean(e0 ** 2) / 2
trial_theta = theta - 0.1 * g0   # 결과를 다른 이름에 저장; theta는 아직 0
print("Xb:\n", Xb)
print("초기 J, gradient:", j0, g0)
print("학습률 0.1의 시범 갱신:", trial_theta)
print("시범 계산 후 원래 theta:", theta)

# 6. 실제 반복은 학습률 0.2로, 다시 0인 계수에서 시작
lr = 0.2
n_steps = 2000
history = []                     # 갱신 직전의 손실을 순서대로 담는 리스트
for _ in range(n_steps):         # range(2000): 0부터 1999까지, 총 2000번
    pred = Xb @ theta
    e = pred - y
    history.append(float(np.mean(e ** 2) / 2))
    gradient = Xb.T @ e / n
    theta -= lr * gradient      # 기존 배열 내용을 수정하는 제자리 갱신

theta_gd = theta.copy()          # 이후 theta가 바뀌어도 결과를 보존할 독립 복사
pred_gd = Xb @ theta_gd
final_j = float(np.mean((pred_gd - y) ** 2) / 2)
history_complete = np.array(history + [final_j])
print("GD 계수:", theta_gd)
print("GD 예측:", pred_gd)
print("첫 손실 / 마지막 저장 손실 / 최종 손실:", history[0], history[-1], final_j)
print("2000회 갱신 뒤 손실까지 포함한 길이:", len(history_complete))

# subplots는 그림 전체 fig와 그래프를 그릴 영역 ax를 돌려준다.
fig_loss, ax_loss = plt.subplots(figsize=(7, 4))  # figsize의 단위는 인치
shown = min(81, len(history_complete))           # 완료한 갱신 0~80회
ax_loss.plot(np.arange(shown), history_complete[:shown])
ax_loss.set(xlabel="Completed updates", ylabel="J = MSE / 2",
            title="Gradient descent on four samples")
ax_loss.grid(alpha=0.25)           # alpha: 보조선의 불투명도
fig_loss.tight_layout()           # 제목과 축 이름이 겹치지 않도록 여백 조정

# 7. 별도 함수 J(t)=t²/2에서 학습률의 영향 비교
# 여기서의 t는 원래 회귀 모델의 theta와 별개인 숫자 하나다.
for rate in [0.2, 0.8, 1.5, 2.2]:
    t = 3.0
    t_history = [t]              # 갱신 0회 상태부터 기록
    for _ in range(8):
        t -= rate * t           # 이 함수의 미분값은 t
        t_history.append(t)
    losses = np.array(t_history) ** 2 / 2
    print("학습률", rate, "첫 갱신 t/J:", t_history[1], losses[1],
          "8회 뒤 J:", losses[-1])

# 8. 같은 작은 데이터에 최소제곱 풀이와 scikit-learn 적용
# 반환값 네 개: 계수, 잔차 제곱합 배열, rank, singular values.
theta_ls, residual_sums, rank, singular_values = np.linalg.lstsq(
    Xb, y, rcond=None
)
pred_ls = Xb @ theta_ls
model_small = LinearRegression()  # 기본값 fit_intercept=True
fit_return = model_small.fit(X, y)  # 이때는 1열을 붙이기 전 X를 사용
pred_sk = model_small.predict(X)
print("최소제곱 계수:", theta_ls)
print("lstsq의 나머지 반환값:", residual_sums, rank, singular_values)
print("fit이 같은 모델을 반환하는가:", fit_return is model_small)
print("sklearn 절편 / 가중치:", model_small.intercept_, model_small.coef_)
print("최소제곱 / GD / sklearn 예측을 열로 비교:\n", np.c_[pred_ls, pred_gd, pred_sk])
new_input = np.array([1.0, 3.0, 4.0])  # [절편용 1, x1, x2]
print("x1=3, x2=4의 GD 예측:", new_input @ theta_gd)

# 9. 구현 검증: shape를 먼저 확인하고 허용 오차 안에서 수치를 비교
assert theta_gd.shape == (3,)
assert pred_gd.shape == y.shape
assert np.isfinite(theta_gd).all()  # 모든 계수가 NaN·무한대가 아닌지 확인
assert np.isfinite(history_complete).all()
assert np.isclose(j0, 7.75, atol=1e-12, rtol=0)
assert np.allclose(g0, [-3.5, -2.25, -2.5], atol=1e-12, rtol=0)
assert np.allclose(theta_gd, [1.0, 2.0, 3.0], atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_ls, atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_sk, atol=1e-8, rtol=0)
assert final_j < history_complete[0]
print("구현 검증 통과. GD/최소제곱 최대 예측 차이:", np.max(np.abs(pred_gd - pred_ls)))

# 10. 실제 데이터: 로더가 제공하는 추가 스케일링을 생략한 저장 특성값
X_real, y_real = load_diabetes(return_X_y=True, scaled=False)
# 각 행의 번호를 나눠 두 배열에 똑같이 적용한다.
all_ids = np.arange(len(y_real))
train_idx, test_idx = train_test_split(
    all_ids, test_size=0.2, random_state=42
)
X_train, X_test = X_real[train_idx], X_real[test_idx]
y_train, y_test = y_real[train_idx], y_real[test_idx]
assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]
print("전체 / 학습 / 테스트 입력 shape:", X_real.shape, X_train.shape, X_test.shape)

# 11. 기준선: 학습 정답의 평균을 모든 테스트 행에 예측
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_test)

# 12. BMI 한 특성만 사용하는 회귀. [2]로 골라 열 차원을 유지한다.
model_bmi = LinearRegression()
model_bmi.fit(X_train[:, [2]], y_train)
pred_bmi = model_bmi.predict(X_test[:, [2]])

# 13. 전체 열 개 특성의 회귀. 위 BMI 모델과 별도 객체다.
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

# 14. 잔차는 실제값 - 예측값. 학습에서 쓴 e와 부호가 반대다.
residual = y_test - pred_all
print("첫 테스트 원본 행 번호:", test_idx[0])
print("첫 테스트 실제값 / 예측 / 잔차:", y_test[0], pred_all[0], residual[0])
fig_residual, ax_residual = plt.subplots(figsize=(7, 4))
ax_residual.scatter(pred_all, residual, alpha=0.65, label="Test samples")
ax_residual.scatter([pred_all[0]], [residual[0]], color="crimson",
                    s=90, label="First test sample")  # s: 점의 면적
ax_residual.axhline(0, color="black", linewidth=1)  # 잔차 0 기준선
ax_residual.set(xlabel="Prediction", ylabel="Residual = target - prediction",
                title="Residuals on 89 held-out samples")
ax_residual.legend()
fig_residual.tight_layout()
plt.show()                       # 준비한 두 그림을 표시
