# 행렬 계산으로 읽는 신경망: XOR 순전파와 보조 예제
# 필요한 외부 라이브러리는 NumPy 하나다.
# 가중치는 계산 구조를 보여 주기 위해 사람이 정했다.
# 가중치 갱신이나 역전파가 없으므로 이 코드는 실제 학습을 수행하지 않는다.

import numpy as np  # 배열 연산 라이브러리를 np라는 이름으로 사용한다.

np.set_printoptions(precision=4, suppress=True)  # 표시만 소수점 4자리로 정리한다.

# np.array()는 주어진 값으로 배열을 만든다. dtype=float는 실수로 저장한다.
# 한 행은 샘플 하나이고, 두 열은 입력 특징 x1과 x2다.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)  # (4, 2): 샘플 4개, 특징 2개.
y = np.array([0, 1, 1, 0])  # (4,): 같은 행의 실제 정답 클래스 번호.
class_names = np.array(["같다", "다르다"])  # 열 0과 열 1의 의미를 정한다.

# W1의 행은 입력 특징, 열은 도착할 은닉 뉴런을 구분한다.
# 첫 뉴런은 x1 - x2, 둘째 뉴런은 -x1 + x2를 계산한다.
W1 = np.array([
    [1, -1],
    [-1, 1],
], dtype=float)  # (2, 2): 입력 특징 2개, 은닉 뉴런 2개.
b1 = np.zeros(2)  # np.zeros()는 0으로 채운 배열을 만든다. 편향 2개, (2,).

# @는 행렬곱이다. 입력의 행과 W1의 열을 대응시켜 곱한 값들을 더한다.
# 편향 두 개는 모든 샘플 행에 똑같이 더해진다.
Z1 = X @ W1 + b1  # (4, 2) @ (2, 2) + (2,) -> (4, 2).
# np.maximum()은 원소마다 큰 값을 고른다. 음수만 0으로 바꾸는 ReLU다.
H = np.maximum(0, Z1)  # (4, 2) -> (4, 2). 원본 X는 그대로 남는다.

# 출력층의 두 뉴런 모두 h1과 h2를 입력받는다.
# 첫 열은 '같다' 점수, 둘째 열은 '다르다' 점수를 계산한다.
W2 = np.array([
    [-1, 1],
    [-1, 1],
], dtype=float)  # (2, 2): 은닉 출력 2개, 출력 뉴런 2개.
b2 = np.array([1, 0], dtype=float)  # (2,): 첫 점수에 1, 둘째 점수에 0을 더한다.
S = H @ W2 + b2  # (4, 2). 클래스별 점수이며 확률은 아니다.

# argmax()는 큰 값 자체가 아니라 그 값의 인덱스를 반환한다.
# axis=1은 각 샘플의 클래스 점수들을 비교한다. (4, 2) -> (4,).
pred = S.argmax(axis=1)
pred_names = class_names[pred]  # 예측 인덱스에 해당하는 이름 4개를 가져온다.

# max()는 클래스 점수 2개를 최댓값 하나로 줄인다.
# keepdims=True는 줄어든 클래스 축을 크기 1로 남긴다.
row_max = S.max(axis=1, keepdims=True)  # (4, 2) -> (4, 1).
shifted = S - row_max  # (4, 2) - (4, 1) -> (4, 2). 각 행의 최댓값을 뺀다.
# np.exp()는 모든 원소에 e**값을 적용한다. exp(0)=1, exp(-1)≈0.3679.
exp_scores = np.exp(shifted)  # (4, 2). 학습 가중치가 아니라 중간 계산값이다.
# sum(axis=1)은 각 행의 클래스 값을 합친다. keepdims=True는 합계의 열을 남긴다.
row_sums = exp_scores.sum(axis=1, keepdims=True)  # (4, 2) -> (4, 1).
probabilities = exp_scores / row_sums  # 각 행의 합계로 나눈다. 결과는 (4, 2).

# len(y)는 정답 개수 4다. np.arange(4)는 정수 [0, 1, 2, 3]을 만든다.
row_indices = np.arange(len(y))  # (4,): 샘플의 행 번호.
# 같은 위치끼리 짝지어 [0,0], [1,1], [2,1], [3,0]을 선택한다.
# 모델의 예측 pred가 아니라 실제 정답 y가 가리키는 확률을 가져온다.
correct_p = probabilities[row_indices, y]  # (4, 2)에서 하나씩 선택 -> (4,).
losses = -np.log(correct_p)  # np.log()는 원소별 자연로그다. 샘플별 손실 (4,).
mean_ce = losses.mean()  # mean()은 손실 4개의 산술평균을 구한다. 숫자 하나.

# 아주 작은 확률이 0으로 저장되어 log(0)이 되는 문제를 줄이는 계산이다.
# 확률을 먼저 만들고 log를 적용하는 대신 점수에서 로그 확률을 계산한다.
log_probabilities = shifted - np.log(row_sums)  # (4, 2) - (4, 1) -> (4, 2).
stable_losses = -log_probabilities[row_indices, y]  # 정답의 로그 확률을 선택, (4,).

# .size는 배열의 숫자 개수다. 각 배열의 개수를 sum()으로 더한다.
parameter_count = sum(a.size for a in [W1, b1, W2, b2])  # 4 + 2 + 4 + 2 = 12.

# assert는 조건이 거짓이면 멈춘다. allclose()는 작은 실수 오차를 허용한다.
# array_equal()은 배열의 shape와 원소가 같은지, isclose()는 값 두 개가 가까운지 검사한다.
assert X.shape == (4, 2) and y.shape == (4,)
assert H.shape == (4, 2) and S.shape == (4, 2)
assert row_max.shape == (4, 1) and row_sums.shape == (4, 1)
assert np.array_equal(Z1, [[0, 0], [-1, 1], [1, -1], [0, 0]])
assert np.array_equal(H, [[0, 0], [0, 1], [1, 0], [0, 0]])
assert np.array_equal(S, [[1, 0], [0, 1], [0, 1], [1, 0]])
assert np.allclose(probabilities.sum(axis=1), 1.0)  # 각 샘플의 확률 합은 1이다.
assert np.array_equal(pred, y)  # 네 샘플의 예측이 각각 정답과 같아야 한다.
assert np.allclose(stable_losses, losses)
assert np.isclose(mean_ce, 0.3132616875182228) and parameter_count == 12

# 이름과 배열을 한 쌍씩 꺼내어 .shape(차원별 길이)와 실제 값을 출력한다.
for name, values in [
    ("X", X), ("Z1", Z1), ("H", H), ("S", S),
    ("row_max", row_max), ("shifted", shifted), ("exp_scores", exp_scores),
    ("row_sums", row_sums), ("probabilities", probabilities),
    ("correct_p", correct_p), ("losses", losses),
    ("log_probabilities", log_probabilities), ("stable_losses", stable_losses),
]:
    print(name, "shape=", values.shape, "\n", values)
print("예측:", pred, pred_names)
print("평균 교차 엔트로피:", mean_ce, "파라미터 개수:", parameter_count)
print("XOR의 모든 검증이 통과했습니다.")

# === Additional shape and API examples / 보조 shape·API 예제 ===
# 아래 예제들은 앞의 XOR 결과를 바꾸지 않는 별도의 계산이다.

# 1. max는 점수, argmax는 열 번호를 반환한다. 클래스 이름은 별도로 연결한다.
scores3 = np.array([[2.1, 0.4, 5.0], [1.3, 4.2, 0.7]])  # (2, 3).
names3 = np.array(["고양이", "강아지", "토끼"])
print("큰 점수:", scores3.max(axis=1))  # [5.0, 4.2].
print("큰 점수의 열:", scores3.argmax(axis=1))  # [2, 1].
print("예측 이름:", names3[scores3.argmax(axis=1)])  # ['토끼', '강아지'].
assert np.array_equal(scores3.argmax(axis=1), [2, 1])

# 2. 원래 점수 [1000, 1002]에서 같은 최댓값을 빼면 Softmax 확률을 보존한다.
# 점수를 나누거나 0~1로 바꾸면 점수 차이가 바뀌어 확률도 달라진다.
comparison_scores = np.array([
    [-2.0, 0.0],  # 각 점수에서 1002를 뺀 결과.
    [1.0, 1.002],  # 각 점수를 1000으로 나눈 결과.
    [0.0, 1.0],  # 최솟값을 0, 최댓값을 1로 바꾼 결과.
])
comparison_exp = np.exp(comparison_scores)
comparison_p = comparison_exp / comparison_exp.sum(axis=1, keepdims=True)
print("변환 방식별 Softmax:\n", comparison_p)
# [[0.1192, 0.8808], [0.4995, 0.5005], [0.2689, 0.7311]].
assert np.allclose(comparison_p.sum(axis=1), 1.0)

# 3. 입력 특징 4개 -> 은닉 뉴런 5개 -> 은닉 뉴런 6개 -> 출력 뉴런 3개.
# 두 샘플은 모든 층에서 같은 가중치를 사용한다. 이 가중치도 사람이 지정했다.
X_demo = np.array([[2, 3, 5, 7], [11, 13, 17, 19]], dtype=float)  # (2, 4).
W1_demo = np.array([
    [0.2, 0.4, -0.3, 0.8, 0.1],
    [-0.5, 0.2, 0.6, 0.3, -0.2],
    [0.7, -0.1, 0.5, 0.4, 0.6],
    [0.1, 0.9, -0.4, 0.2, 0.5],
])  # (4, 5): 각 열은 첫 은닉 뉴런 하나의 가중치 4개다.
b1_demo = np.zeros(5)
W2_demo = np.array([
    [0.1, -0.2, 0.3, 0.4, -0.1, 0.2],
    [0.2, 0.3, -0.1, 0.1, 0.4, -0.2],
    [-0.3, 0.1, 0.2, -0.2, 0.1, 0.3],
    [0.4, -0.1, 0.1, 0.2, 0.3, 0.1],
    [0.1, 0.2, 0.3, -0.1, -0.2, 0.4],
])  # (5, 6): 은닉 출력 5개를 다음 뉴런 6개에 모두 연결한다.
b2_demo = np.zeros(6)
W3_demo = np.array([
    [0.1, -0.2, 0.3],
    [0.2, 0.1, -0.1],
    [-0.1, 0.3, 0.2],
    [0.3, -0.1, 0.1],
    [0.1, 0.2, 0.4],
    [-0.2, 0.4, 0.1],
])  # (6, 3): 은닉 출력 6개를 출력 뉴런 3개에 모두 연결한다.
b3_demo = np.zeros(3)

H1_demo = np.maximum(0, X_demo @ W1_demo + b1_demo)  # (2, 4) @ (4, 5) -> (2, 5).
H2_demo = np.maximum(0, H1_demo @ W2_demo + b2_demo)  # (2, 5) @ (5, 6) -> (2, 6).
S_demo = H2_demo @ W3_demo + b3_demo  # (2, 6) @ (6, 3) -> (2, 3).

# 첫 샘플의 세 번째 은닉 뉴런(열 번호 2)에 들어가는 계산을 펼친다.
z_demo = (
    X_demo[0, 0] * W1_demo[0, 2]
    + X_demo[0, 1] * W1_demo[1, 2]
    + X_demo[0, 2] * W1_demo[2, 2]
    + X_demo[0, 3] * W1_demo[3, 2]
    + b1_demo[2]
)  # 2*(-0.3) + 3*0.6 + 5*0.5 + 7*(-0.4) + 0 = 0.9.
# H1의 열 2는 다음 W2의 행 2에 대응하여 다음 층의 모든 뉴런에 전달된다.
# W2_demo[2, :]의 ':'는 해당 행의 열 전체를 선택한다. 결과 shape는 (6,)이다.
next_layer_contributions = H1_demo[0, 2] * W2_demo[2, :]
demo_parameter_count = sum(
    a.size for a in [W1_demo, b1_demo, W2_demo, b2_demo, W3_demo, b3_demo]
)  # (4*5 + 5) + (5*6 + 6) + (6*3 + 3) = 82.

assert H1_demo.shape == (2, 5) and H2_demo.shape == (2, 6)
assert S_demo.shape == (2, 3) and demo_parameter_count == 82
assert np.isclose(H1_demo[0, 2], max(0, z_demo))  # isclose()는 값 두 개를 비교한다.
assert np.isclose(H1_demo[0, 2], 0.9)
for name, values in [("H1_demo", H1_demo), ("H2_demo", H2_demo), ("S_demo", S_demo)]:
    print(name, "shape=", values.shape, "\n", values)
print("H1_demo[0, 2]:", H1_demo[0, 2])
print("이 값이 다음 뉴런 6개에 더하는 항:", next_layer_contributions)
print("추가 모델의 파라미터 개수:", demo_parameter_count)
print("모든 예제의 검증이 통과했습니다.")
