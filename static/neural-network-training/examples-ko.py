"""신경망 학습: 순전파, 손실, 역전파, 파라미터 갱신을 직접 확인한다.

필요한 외부 패키지는 PyTorch뿐이다. CPU에서 실행하며 GPU가 필요하지 않다.
가상 데이터: [작은 상자 수, 큰 상자 수] → 포장 시간(분).
A=[2, 1]의 정답은 6분, B=[1, 1]의 정답은 4분이다.
두 사례는 계산 과정을 보여주기 위한 예시이며 실무 예측 성능을 검증하지 않는다.
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# float64는 64비트 실수형이다. 작은 예제의 수치 비교에 사용한다.
DTYPE = torch.float64


class StudyMLP(nn.Module):
    """입력 특징 2개 → 은닉 뉴런 2개 → 출력 뉴런 1개."""

    def __init__(self):
        super().__init__()  # nn.Module이 파라미터를 관리하도록 초기화한다.
        # Parameter는 학습할 텐서(숫자 배열)를 모델에 등록하는 클래스다.
        # 이 예제의 행렬은 행=입력 특징, 열=출력 뉴런으로 배치한다.
        # W_hidden = [[w1, w3], [w2, w4]]: (2, 2).
        self.W_hidden = nn.Parameter(
            torch.tensor([[1.0, -1.0], [2.0, 1.0]], dtype=DTYPE)
        )
        self.b_hidden = nn.Parameter(torch.tensor([0.0, 2.0], dtype=DTYPE))
        self.W_out = nn.Parameter(torch.tensor([[2.0], [-1.0]], dtype=DTYPE))
        self.b_out = nn.Parameter(torch.tensor([1.0], dtype=DTYPE))

    def forward(self, X):
        # @는 행렬 곱이다. 편향은 각 샘플의 해당 뉴런에 더해진다.
        Z = X @ self.W_hidden + self.b_hidden  # (N, 2).
        H = torch.relu(Z)                     # 원소별 max(0, z), (N, 2).
        return H @ self.W_out + self.b_out    # (N, 1).


def check_central_differences(model, X, y, expected_grads):
    """파라미터 하나씩만 바꾸어 수치 기울기와 역전파 결과를 대조한다."""
    eps = 1e-5  # 두 비교 위치는 현재 값 ±0.00001이다.
    checked = 0
    # no_grad는 새로운 미분 기록만 끈다. 숫자 계산은 그대로 수행한다.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            numeric = torch.empty_like(parameter)  # 같은 shape의 결과 보관 배열.
            # view(-1)은 같은 저장 공간을 1차원으로 바라본다. 여기서는 복사하지 않는다.
            flat_parameter = parameter.view(-1)
            flat_numeric = numeric.view(-1)
            for index in range(parameter.numel()):
                original = flat_parameter[index].item()
                try:
                    flat_parameter[index] = original + eps
                    plus = (0.5 * (model(X) - y).square()).mean().item()
                    flat_parameter[index] = original - eps
                    minus = (0.5 * (model(X) - y).square()).mean().item()
                finally:
                    flat_parameter[index] = original  # 비교가 끝나면 원래 값으로 되돌린다.
                # 분모는 비교한 두 위치의 차이다: (w+eps)-(w-eps).
                flat_numeric[index] = (plus - minus) / (2 * eps)
                checked += 1
            # assert_close는 허용 오차 안에서 같은지 검사한다. 다르면 예외를 낸다.
            # rtol은 상대 오차, atol은 절대 오차의 허용 범위를 정한다.
            torch.testing.assert_close(
                numeric, expected_grads[name], rtol=1e-6, atol=1e-8
            )
    assert checked == 9
    print("중앙차분과 역전파를 대조한 파라미터 수:", checked)


def inspect_one_update(X, y):
    """같은 초기 상태에서 수동 계산, 자동미분, SGD 한 번의 갱신을 비교한다."""
    model = StudyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # zero_grad는 이전 .grad를 비운다. 파라미터 자체는 변경하지 않는다.
    optimizer.zero_grad(set_to_none=True)
    # detach는 미분 연결을 끊고, clone은 별도 저장 공간에 값을 복사한다.
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    # 1. forward의 내용을 펼쳐 중간값도 확인한다.
    Z = X @ model.W_hidden + model.b_hidden  # [[4, 1], [3, 2]].
    H = torch.relu(Z)                       # 이번에는 모두 양수여서 Z와 같다.
    pred = H @ model.W_out + model.b_out    # [[8], [5]].
    # 중간 텐서의 .grad는 보통 보관하지 않는다. 학습 확인용으로 보관한다.
    for intermediate in (Z, H, pred):
        intermediate.retain_grad()

    # 2. square는 원소별 제곱, mean은 전체 원소의 평균을 반환하는 메서드다.
    assert pred.shape == y.shape == (2, 1)
    per_loss = 0.5 * (pred - y).square()  # [[2], [0.5]], shape=(2, 1).
    loss = per_loss.mean()               # 1.25, shape=(): 숫자 하나.
    print("\n[순전파와 손실]")
    for name, value in [("Z", Z), ("H", H), ("예측", pred), ("개별 손실", per_loss)]:
        print(name, value.detach().tolist())  # tolist는 텐서를 Python 목록으로 바꾼다.
    print("평균 손실:", loss.item())  # item은 원소 하나를 Python 숫자로 꺼낸다.
    torch.testing.assert_close(pred, torch.tensor([[8.0], [5.0]], dtype=DTYPE))
    assert loss.item() == 1.25

    # 3. 수동 역전파. g_는 '평균 손실의 해당 값에 대한 기울기'를 나타낸다.
    N = X.shape[0]  # 이번 배치에 실제로 들어 있는 샘플 수: 2.
    with torch.no_grad():
        g_pred = (pred - y) / N             # (2, 1): [[1], [0.5]].
        g_W_out = H.T @ g_pred              # (2, 1): [[5.5], [2]].
        g_b_out = g_pred.sum(dim=0)         # (2, 1) → (1,): [1.5].
        g_H = g_pred @ model.W_out.T        # (2, 2): 은닉층으로 전달할 기울기.
        g_Z = g_H * (Z > 0)                 # ReLU의 양수 구간은 1, 나머지는 0.
        g_W_hidden = X.T @ g_Z              # (2, 2): 샘플별 기여를 합산.
        g_b_hidden = g_Z.sum(dim=0)          # (2, 2) → (2,): [3, -1.5].
    # .T는 행과 열을 바꾼다. sum(dim=0)은 샘플별 기여를 합쳐 샘플 차원을 없앤다.
    # /N을 g_pred에 반영했으므로 이후 다시 N으로 나누면 안 된다.
    # ReLU는 0에서 보통의 미분값이 정해지지 않으며 PyTorch는 여기서 0을 쓴다.
    manual = {"W_hidden": g_W_hidden, "b_hidden": g_b_hidden,
              "W_out": g_W_out, "b_out": g_b_out}
    expected = {"W_hidden": [[5.0, -2.5], [3.0, -1.5]],
                "b_hidden": [3.0, -1.5], "W_out": [[5.5], [2.0]],
                "b_out": [1.5]}

    # 4. 자동 역전파. 반환값은 None이며 결과는 파라미터의 .grad에 누적된다.
    result = loss.backward()
    assert result is None
    torch.testing.assert_close(pred.grad, g_pred)
    torch.testing.assert_close(H.grad, g_H)
    torch.testing.assert_close(Z.grad, g_Z)
    print("\n[파라미터별 기울기]")
    for name, parameter in model.named_parameters():
        # backward가 파라미터의 값을 바꾸지 않았는지도 확인한다.
        torch.testing.assert_close(parameter.detach(), before[name])
        torch.testing.assert_close(parameter.grad, manual[name])
        torch.testing.assert_close(parameter.grad, torch.tensor(expected[name], dtype=DTYPE))
        print(name, "기울기:", parameter.grad.tolist(), "shape:", tuple(parameter.grad.shape))
    # numel은 배열 안의 숫자 개수다. 텐서 4개에 파라미터 숫자 9개가 있다.
    assert sum(p.numel() for p in model.parameters()) == 9
    assert sum(p.grad.numel() for p in model.parameters()) == 9
    check_central_differences(model, X, y, manual)

    # 5. 이때 학습 규칙에 따라 파라미터를 실제로 갱신한다.
    optimizer.step()  # 새 값 = 기존 값 - 0.01 * 기울기.
    changed = 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name] - 0.01 * manual[name])
        torch.testing.assert_close(parameter.grad, manual[name])  # 여전히 이전 위치의 기울기.
        changed += (parameter.detach() != before[name]).sum().item()
    assert changed == 9  # 이번에는 기울기 9개가 모두 0이 아니므로 값도 모두 바뀐다.
    with torch.no_grad():
        pred_after = model(X)  # 새 파라미터로 순전파를 다시 실행한다.
        loss_after = (0.5 * (pred_after - y).square()).mean()
    print("\n갱신 후 예측:", pred_after.tolist())
    print("갱신 후 평균 손실:", loss_after.item())
    # isclose는 소수 저장의 작은 오차를 허용하여 숫자 두 개가 가까운지 비교한다.
    assert math.isclose(loss_after.item(), 0.522123460625, rel_tol=1e-12)
    assert loss.item() == 1.25  # 이미 계산한 이전 loss가 자동으로 바뀌지는 않는다.


def inspect_four_layers(X, y):
    """별도 모델의 Linear 4개, 텐서 8개, 파라미터 숫자 21개를 구분한다."""
    # Linear(D, K)는 입력 D개로 출력 K개를 계산하는 층이다.
    # Sequential은 등록한 모듈을 순서대로 실행한다.
    model = nn.Sequential(
        nn.Linear(2, 2, dtype=DTYPE), nn.ReLU(),
        nn.Linear(2, 2, dtype=DTYPE), nn.ReLU(),
        nn.Linear(2, 2, dtype=DTYPE), nn.ReLU(),
        nn.Linear(2, 1, dtype=DTYPE),
    )
    # Linear는 weight를 (출력 수, 입력 수)로 보관하고 X @ weight.T + bias를 계산한다.
    # 자체 초기값을 사용하므로 앞 모델과 기울기의 숫자까지 같지는 않다.
    loss = 0.5 * (model(X) - y).square().mean()
    loss.backward()
    parameters = list(model.parameters())
    assert len(parameters) == 8
    assert sum(p.numel() for p in parameters) == 21
    assert sum(p.grad.numel() for p in parameters) == 21
    print("\n[4층 모델: 값의 shape / 기울기의 shape]")
    for name, p in model.named_parameters():
        assert p.shape == p.grad.shape
        print(name, tuple(p.shape), tuple(p.grad.shape))
    # 0,2,4,6은 모듈 위치다. 1,3,5의 ReLU에는 학습 파라미터가 없다.
    # 기울기 21개는 결과의 개수이며 곱셈·덧셈 21회라는 뜻이 아니다.


def train_three_epochs(X, y):
    """같은 샘플 2개로 일반적인 배치 학습과 추론의 순서를 확인한다."""
    model = StudyMLP()  # 앞에서 갱신한 모델과 별개로 초기값에서 다시 시작한다.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # TensorDataset은 같은 행의 입력과 정답을 하나의 샘플로 묶는 클래스다.
    # DataLoader는 그 샘플을 배치 단위로 꺼내 주는 클래스다.
    loader = DataLoader(TensorDataset(X, y), batch_size=2, shuffle=True, drop_last=False)
    # shuffle은 입력과 정답을 함께 섞는다. drop_last=False는 작은 마지막 배치도 쓴다.
    total_steps = 0
    print("\n[3에포크 학습]")
    for epoch in range(3):
        model.train()  # 학습 모드 전환이다. 이것만으로 학습 계산을 하지는 않는다.
        loss_total, sample_total = 0.0, 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)  # 이전 기울기를 비운다.
            batch_pred = model(batch_X)            # 순전파.
            assert batch_pred.shape == batch_y.shape
            batch_loss = 0.5 * (batch_pred - batch_y).square().mean()
            batch_loss.backward()                  # 현재 위치의 기울기 계산.
            optimizer.step()                       # 파라미터 갱신.
            total_steps += 1
            # 배치 평균×실제 샘플 수를 합하여 에포크 평균을 기록한다.
            count = batch_X.shape[0]
            loss_total += batch_loss.item() * count
            sample_total += count
        print(f"epoch={epoch + 1}, 학습 중 평균 손실={loss_total / sample_total:.6f}")
    assert total_steps == 3  # 샘플 2개÷배치 크기 2×에포크 3회.
    # 위 기록은 각 배치를 처리할 당시의 손실이다. 최종 모델의 재평가와 다르다.
    model.eval()  # Dropout·BatchNorm 등을 평가 모드로 바꾼다. 이번에는 둘 다 없다.
    with torch.no_grad():  # eval만으로 미분 기록이 꺼지지는 않으므로 별도로 지정한다.
        prediction = model(X)
    print("학습 후 추론:", prediction.tolist())


def compare_learning_rates():
    """별도 함수 J(u)=(u-3)^2에서 학습률을 제외한 조건을 같게 둔다."""
    histories = {}  # 학습률이 키, 각 시점의 기록 목록이 값인 딕셔너리.
    expected_u = {0.05: [0.0, 0.3, 0.57, 0.813],
                  0.25: [0.0, 1.5, 2.25, 2.625],
                  1.1: [0.0, 6.6, -1.32, 8.184]}
    print("\n[J(u)=(u-3)^2의 학습률 비교]")
    for learning_rate in [0.05, 0.25, 1.1]:
        u = 0.0  # 매 실험을 같은 위치에서 시작한다.
        records = [{"step": 0, "u": u, "loss": (u - 3) ** 2}]
        for step in range(1, 4):  # 1,2,3의 세 번. 끝의 4는 포함하지 않는다.
            gradient = 2 * (u - 3)  # 매번 새 위치에서 다시 계산한다.
            u = u - learning_rate * gradient
            records.append({"step": step, "u": u, "loss": (u - 3) ** 2})
        # append는 목록에 추가하는 메서드다. 반환값 None을 records에 대입하지 않는다.
        histories[learning_rate] = records
        # zip은 두 목록에서 같은 위치의 항목을 짝지어 꺼낸다.
        for record, expected in zip(records, expected_u[learning_rate]):
            assert math.isclose(record["u"], expected, abs_tol=1e-12)
        print("학습률:", learning_rate)
        print("u:", [round(row["u"], 6) for row in records])
        print("J:", [round(row["loss"], 6) for row in records])
    # 실험마다 갱신 3회, 시작 상태를 포함한 기록 4개다. 두 개수를 구분한다.
    # 이 함수에서는 (u_new-3)=(1-2*learning_rate)*(u-3)이 성립한다.
    # 0.25는 비교한 세 값 중 빠를 뿐, 모든 함수에서 최적인 학습률은 아니다.
    return histories


def main():
    # manual_seed는 난수를 재현하기 쉽게 한다. 직접 설정한 파라미터 9개는 고정값이다.
    torch.manual_seed(0)
    # tensor는 숫자 배열을 만드는 함수다. 입력은 (샘플 수, 특징 수)=(2, 2).
    X = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=DTYPE)
    # 정답과 예측을 모두 (N, 1)로 맞춘다. (N,)과 빼서 잘못 확장되는 일을 막는다.
    y = torch.tensor([[6.0], [4.0]], dtype=DTYPE)
    inspect_one_update(X, y)
    inspect_four_layers(X, y)
    train_three_epochs(X, y)
    compare_learning_rates()
    print("\n모든 수치 검증을 통과했습니다.")


if __name__ == "__main__":
    main()  # 이 파일을 직접 실행했을 때 위의 작업을 순서대로 호출한다.
