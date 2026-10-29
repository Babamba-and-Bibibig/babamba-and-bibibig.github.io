# RNN 시퀀스·은닉 상태 실습: CPU / float64 / 순전파만 실행
import torch
from torch import nn

# B: 배치 크기, T: 시점 수, F: 입력 특징 수, H: 은닉 뉴런 수
B, T, F, H = 2, 5, 4, 3

# 수치 계산을 확인하기 위해 64비트 실수형을 사용한다.
DTYPE = torch.float64

# 하나의 셀 객체를 만든다.
# bias=True: 두 종류의 편향을 사용한다.
# nonlinearity="tanh": 합산 결과에 tanh를 적용한다.
cell = nn.RNNCell(
    input_size=F,
    hidden_size=H,
    bias=True,
    nonlinearity="tanh",
).double()

# 모델 생성 후 우리가 정한 가중치와 편향으로 바꾼다.
# 이 수정 과정은 자동 미분 기록에 넣지 않는다.
with torch.no_grad():
    cell.weight_ih.copy_(torch.tensor([
        [ 0.11, -0.07,  0.04,  0.02],  # 현재 뉴런 A
        [-0.09,  0.13,  0.06, -0.03],  # 현재 뉴런 B
        [ 0.05,  0.08, -0.12,  0.15],  # 현재 뉴런 C
    ], dtype=DTYPE))

    cell.weight_hh.copy_(torch.tensor([
        [ 0.21, -0.14,  0.07],  # 이전 A, B, C → 현재 A
        [-0.16,  0.25,  0.12],  # 이전 A, B, C → 현재 B
        [ 0.19, -0.06,  0.31],  # 이전 A, B, C → 현재 C
    ], dtype=DTYPE))

    cell.bias_ih.copy_(torch.tensor(
        [0.015, -0.025, 0.035], dtype=DTYPE
    ))

    cell.bias_hh.copy_(torch.tensor(
        [-0.008, 0.018, -0.028], dtype=DTYPE
    ))

# shape = (B, T, F) = (2, 5, 4)
x = torch.tensor([
    [
        [2., 5., 7., 11.],
        [3., 8., 12., 6.],
        [4., 6., 9., 13.],
        [8., 3., 5., 10.],
        [7., 9., 2., 14.],
    ],
    [
        [6., 4., 8., 3.],
        [9., 5., 10., 7.],
        [12., 11., 6., 4.],
        [5., 14., 3., 8.],
        [11., 7., 13., 2.],
    ],
], dtype=DTYPE)

# 파라미터 객체의 식별값을 기록한다.
ids_before = tuple(id(p) for p in cell.parameters())

# 파라미터 값도 별도로 복사한다.
# detach(): 자동 미분 기록과 분리한다.
# clone(): 별도 데이터 복사본을 만든다.
values_before = [p.detach().clone() for p in cell.parameters()]

# 두 시퀀스가 각각 은닉값 3개를 가진다.
# shape = (B, H) = (2, 3)
state = torch.zeros(B, H, dtype=DTYPE)

# 매 시점의 (B, H) 결과 텐서를 저장한다.
trace = []

with torch.no_grad():
    for t in range(T):
        # t는 0, 1, 2, 3, 4다.
        # 두 시퀀스에서 같은 순서의 시점을 선택한다.
        # shape: (B, F) = (2, 4)
        xt = x[:, t, :]

        # 이번 시점의 모든 뉴런이 사용할 이전 상태다.
        # 이 텐서를 뉴런별 계산 도중 직접 수정하지 않는다.
        previous_state = state

        # @는 행렬곱이다.
        # .T는 여기서 2차원 가중치 행렬을 전치한다.
        #
        # 입력:       (B, F) = (2, 4)
        # weight_ih:  (H, F) = (3, 4)
        # 전치 결과:  (F, H) = (4, 3)
        # 곱셈 결과:  (B, H) = (2, 3)
        input_part = xt @ cell.weight_ih.T

        # 이전 상태:  (B, H) = (2, 3)
        # weight_hh.T:(H, H) = (3, 3)
        # 곱셈 결과:  (B, H) = (2, 3)
        #
        # 정사각 행렬이어도 전치하면 값의 위치가 바뀐다.
        previous_part = previous_state @ cell.weight_hh.T

        # 편향 shape는 (H,)다.
        # 각 시퀀스의 동일한 뉴런 위치에 같은 편향이 더해진다.
        manual_state = torch.tanh(
            input_part
            + cell.bias_ih
            + previous_part
            + cell.bias_hh
        )

        # 같은 계산을 PyTorch RNNCell로 실행한다.
        # 반환 shape: (B, H) = (2, 3)
        state = cell(xt, previous_state)

        # allclose: 모든 대응 원소가 허용 오차 안에서 같은지 검사한다.
        # rtol: 값의 크기에 비례하는 상대 허용 오차
        # atol: 절대 허용 오차
        assert torch.allclose(
            state, manual_state, rtol=1e-10, atol=1e-12
        )

        trace.append(state)

        # 첫 번째 시퀀스의 새 은닉 상태만 출력한다.
        # state[0]의 shape는 (3,)다.
        print(f"시점 {t + 1}: {state[0].tolist()}")

    # (2, 3) 텐서 5개를 시간축으로 모은다.
    # 결과 shape: (B, T, H) = (2, 5, 3)
    output = torch.stack(trace, dim=1)

print("전체 입력:", tuple(x.shape))
print("전체 출력:", tuple(output.shape))
print("마지막 상태:", tuple(state.shape))

# 첫 번째 시퀀스의 출력: 소수점 여섯 자리 반올림
#
# 시점 1: [0.360099,  0.502765, 0.866036]
# 시점 2: [0.416009,  0.900527, 0.510735]
# 시점 3: [0.567644,  0.654344, 0.940266]
# 시점 4: [0.824405, -0.150259, 0.956839]
# 시점 5: [0.645847,  0.176484, 0.997772]


# -------------------------------------------------------------------
# 하나의 시퀀스를 실행하는 함수
# 입력: (T, F), 반환: (T, H)
# -------------------------------------------------------------------

def run_given_cell(values: torch.Tensor) -> torch.Tensor:
    # ndim은 텐서의 축 개수다.
    # 여기서는 배치 축 없는 시퀀스 (T, F)를 받는다.
    if values.ndim != 2 or values.shape[1] != F or values.shape[0] == 0:
        raise ValueError("values는 비어 있지 않은 (T, F) 텐서여야 합니다.")

    # 모델과 입력의 자료형 및 계산 장치가 맞는지 검사한다.
    if (
        values.dtype != cell.weight_ih.dtype
        or values.device != cell.weight_ih.device
    ):
        raise ValueError("values와 cell의 자료형 및 장치가 같아야 합니다.")

    # 함수 호출마다 초기 상태를 새로 만든다.
    # 따라서 앞선 함수 호출의 마지막 상태를 이어받지 않는다.
    # 배치 축을 생략했으므로 shape는 (H,)다.
    state = torch.zeros(
        H,
        dtype=values.dtype,
        device=values.device,
    )

    trace = []

    with torch.no_grad():
        # 2차원 values를 반복하면 각 시점의 (F,) 텐서를 받는다.
        for xt in values:
            # 이전 state를 입력으로 사용한 뒤 새 state를 받는다.
            state = cell(xt, state)
            trace.append(state)

    # (H,) 텐서 T개를 모아 (T, H)로 만든다.
    return torch.stack(trace, dim=0)


# -------------------------------------------------------------------
# 실험 A: 입력 순서를 바꾸면?
# -------------------------------------------------------------------

original = run_given_cell(x[0])

# 첫 번째 시퀀스의 시점 순서만 바꾼다.
# [2, 0, 1, 3, 4]는 사용할 시점 인덱스의 순서다.
reordered = run_given_cell(x[0, [2, 0, 1, 3, 4], :])

print("원래 순서의 마지막 상태:", original[-1].tolist())
print("바꾼 순서의 마지막 상태:", reordered[-1].tolist())

# 원래 순서:
# [0.645847, 0.176484, 0.997772]
#
# 바꾼 순서:
# [0.638208, 0.188234, 0.997711]

# 이 예제에서는 순서 변경으로 마지막 상태가 달라진다.
assert not torch.allclose(original[-1], reordered[-1])

# 이유:
# 앞선 시점의 입력 순서가 바뀌면 이전 은닉 상태가 달라지고,
# 그 상태가 다음 시점의 계산에 사용되기 때문이다.
#
# 단, 모든 가중치와 입력에서 순서를 바꾸면 반드시 서로 다른
# 결과가 나온다는 보장은 아니다. 여기서는 실제 차이를 확인한다.


# -------------------------------------------------------------------
# 실험 B: 시점 수를 줄이면?
# -------------------------------------------------------------------

# 처음 3개 시점만 처리한다.
short = run_given_cell(x[0, :3, :])

# 5개 시점 전체를 처리했을 때의 앞 3개 결과와 비교한다.
assert torch.allclose(
    short, original[:3], rtol=1e-10, atol=1e-12
)

# 이유:
# 초기 상태, 앞부분 입력, 가중치가 같기 때문이다.
# 단방향 RNN에서 나중 시점의 입력은 이미 계산한 앞 시점의
# 은닉 상태를 거꾸로 바꾸지 않는다.


# -------------------------------------------------------------------
# 실험 C: 파라미터 객체와 값은 그대로인가?
# -------------------------------------------------------------------

# 객체 식별값 비교
assert ids_before == tuple(id(p) for p in cell.parameters())

# 실제 숫자 값 비교
# zip: 현재 파라미터와 저장한 복사본을 순서대로 짝지음
# torch.equal: shape와 각 원소 값이 같은지 검사
# all: 모든 비교가 True인지 검사
assert all(
    torch.equal(p, before)
    for p, before in zip(cell.parameters(), values_before)
)

print("파라미터 텐서 개수:", len(tuple(cell.parameters())))
print("학습 가능한 숫자 개수:", sum(p.numel() for p in cell.parameters()))

# 출력:
# 파라미터 텐서 개수: 4
# 학습 가능한 숫자 개수: 27


# -------------------------------------------------------------------
# 실험 D: nn.RNN에 전체 시퀀스를 넣어도 같은가?
# -------------------------------------------------------------------

rnn = nn.RNN(
    input_size=F,
    hidden_size=H,
    num_layers=1,          # RNN 층 하나
    nonlinearity="tanh",
    bias=True,
    batch_first=True,     # 입력·output: (B, T, 특징)
    bidirectional=False,  # 정방향만 처리
).double()

with torch.no_grad():
    # 공정하게 비교하기 위해 같은 파라미터 값을 넣는다.
    # _l0는 첫 번째 RNN 층, 즉 층 인덱스 0을 의미한다.
    rnn.weight_ih_l0.copy_(cell.weight_ih)
    rnn.weight_hh_l0.copy_(cell.weight_hh)
    rnn.bias_ih_l0.copy_(cell.bias_ih)
    rnn.bias_hh_l0.copy_(cell.bias_hh)

    # 전체 시퀀스를 한 번에 전달한다.
    # 초기 은닉 상태를 생략했으므로 0에서 시작한다.
    rnn_output, h_n = rnn(x)

# RNNCell을 반복 호출한 결과와 비교한다.
assert torch.allclose(
    rnn_output, output, rtol=1e-10, atol=1e-12
)

# 마지막 시점의 상태 비교
assert torch.allclose(
    h_n[0], output[:, -1, :], rtol=1e-10, atol=1e-12
)

# 배치로 계산한 첫 번째 시퀀스와
# 그 시퀀스를 따로 계산한 결과도 비교한다.
assert torch.allclose(
    output[0], original, rtol=1e-10, atol=1e-12
)

print("nn.RNN output:", tuple(rnn_output.shape))  # (2, 5, 3)
print("nn.RNN h_n:", tuple(h_n.shape))            # (1, 2, 3)
print("모든 검증을 통과했습니다.")
