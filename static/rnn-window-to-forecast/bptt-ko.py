"""참고용: 스칼라 RNN의 시간 역전파를 자동미분·유한차분과 비교한다.

샘플·입력 특징·은닉값이 각각 하나인 특수 검산이다.
일반적인 모델 학습에서는 수동 반복문 대신 loss.backward()를 사용한다.
"""

import json
import math

import torch
from torch import nn


def verify_gradient():
    inputs = [1.0, 2.0, 3.0]  # 검산용 입력 세 시점.
    input_weight = 0.5       # 현재 입력에 곱하는 가중치.
    recurrent_weight = 0.5   # 이전 은닉값에 곱하는 별도 가중치.
    target = 0.5             # 마지막 은닉값과 비교하는 목표값.
    loss_factor = 0.5        # 제곱오차를 절반으로 만드는 손실 정의의 계수.
    # 같은 0.5를 사용해도 네 값은 서로 다른 역할이다.
    model = nn.RNN(1, 1, bias=False, batch_first=True).double()
    with torch.no_grad():
        model.weight_ih_l0.fill_(input_weight)
        model.weight_hh_l0.fill_(recurrent_weight)
    x = torch.tensor(inputs, dtype=torch.float64).reshape(1, len(inputs), 1)
    # x: (B=1,T=3,F=1). double과 float64로 작은 수치 차이를 줄인다.
    output, _ = model(x)
    loss = loss_factor * (output[0, -1, 0] - target).square()
    model.zero_grad(set_to_none=True)
    loss.backward()
    autograd_whh = model.weight_hh_l0.grad.item()
    autograd_wih = model.weight_ih_l0.grad.item()

    # 이 스칼라 예제에서만 flatten 결과를 [h1,h2,h3]으로 읽을 수 있다.
    # detach는 원본 output의 연결을 수정하지 않고 출력용 텐서를 반환한다.
    hs = [0.0] + output.detach().flatten().tolist()
    adj_h = hs[-1] - target  # dL/dh3: 마지막 은닉값에 대한 손실 기울기.
    pieces = [0.0] * len(inputs)
    input_pieces = [0.0] * len(inputs)
    backward_rows = []
    for t in range(len(inputs), 0, -1):  # 3,2,1: 끝값 0은 제외한다.
        # dL/da_t = (dL/dh_t) * (dh_t/da_t). tanh를 제거하는 계산이 아니다.
        adj_a = adj_h * (1.0 - hs[t] ** 2)
        # a_t의 recurrent_weight * h_(t-1)에서 가중치를 미분한다.
        pieces[t - 1] = adj_a * hs[t - 1]
        input_pieces[t - 1] = adj_a * inputs[t - 1]
        # 같은 곱셈을 이전 은닉값으로 미분하면 가중치를 곱한다.
        adj_h_prev = adj_a * recurrent_weight
        backward_rows.append({
            "time": t, "hidden": hs[t], "previous_hidden": hs[t - 1],
            "loss_gradient_hidden": adj_h, "tanh_derivative": 1.0 - hs[t] ** 2,
            "loss_gradient_preactivation": adj_a,
            "recurrent_weight_piece": pieces[t - 1],
            "loss_gradient_previous_hidden": adj_h_prev,
        })
        adj_h = adj_h_prev  # 다음 반복에서 이전 시점을 만든 연산을 미분한다.

    def loss_with_recurrent_weight(weight):
        state = 0.0
        for value in inputs:
            state = math.tanh(input_weight * value + weight * state)
        return loss_factor * (state - target) ** 2

    epsilon = 1e-6  # 유한차분에서 가중치를 변화시키는 작은 간격.
    finite_difference = (
        loss_with_recurrent_weight(recurrent_weight + epsilon)
        - loss_with_recurrent_weight(recurrent_weight - epsilon)
    ) / (2.0 * epsilon)
    # 시점별 합은 하나의 공유 가중치에 대한 최종 기울기다.
    assert abs(sum(pieces) - autograd_whh) < 1e-12
    assert abs(sum(input_pieces) - autograd_wih) < 1e-12
    assert abs(finite_difference - autograd_whh) < 1e-8
    assert pieces[0] == 0.0  # h0=0이므로 첫 W_hh 사용분만 0이다.
    # 그 시점의 W_ih 사용분은 0이 아니다. 첫 시점 전체가 배제되지 않는다.
    assert input_pieces[0] != 0.0
    return {
        "hidden_history": hs, "loss": loss.item(),
        "recurrent_weight_pieces": pieces,
        "input_weight_pieces": input_pieces,
        "manual_recurrent_gradient": sum(pieces),
        "autograd_recurrent_gradient": autograd_whh,
        "autograd_input_gradient": autograd_wih,
        "finite_difference_recurrent_gradient": finite_difference,
        "backward_rows": backward_rows,
    }


if __name__ == "__main__":
    print(json.dumps(verify_gradient(), ensure_ascii=False, indent=2))
