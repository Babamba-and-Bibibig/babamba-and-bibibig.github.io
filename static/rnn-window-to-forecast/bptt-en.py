"""Optional reference: compare scalar RNN BPTT with autograd and finite differences.

This is a special check with one sample, one input feature, and one hidden value.
For ordinary model training, use loss.backward() instead of implementing this loop.
"""

import json
import math

import torch
from torch import nn


def verify_gradient():
    inputs = [1.0, 2.0, 3.0]  # Three input time steps chosen for this check.
    input_weight = 0.5       # Weight multiplied by the current input.
    recurrent_weight = 0.5   # A separate weight multiplied by the previous hidden value.
    target = 0.5             # Desired value for the last hidden output.
    loss_factor = 0.5        # Coefficient defining half the squared error.
    # These four values have different roles even though all equal 0.5.
    model = nn.RNN(1, 1, bias=False, batch_first=True).double()
    with torch.no_grad():
        model.weight_ih_l0.fill_(input_weight)
        model.weight_hh_l0.fill_(recurrent_weight)
    x = torch.tensor(inputs, dtype=torch.float64).reshape(1, len(inputs), 1)
    # x: (B=1,T=3,F=1). double/float64 reduces numerical differences in the check.
    output, _ = model(x)
    loss = loss_factor * (output[0, -1, 0] - target).square()
    model.zero_grad(set_to_none=True)
    loss.backward()
    autograd_whh = model.weight_hh_l0.grad.item()
    autograd_wih = model.weight_ih_l0.grad.item()

    # Only in this scalar setup can flatten be read directly as [h1,h2,h3].
    # detach returns a display-only tensor without changing the original output connection.
    hs = [0.0] + output.detach().flatten().tolist()
    adj_h = hs[-1] - target  # dL/dh3: loss gradient with respect to the last hidden value.
    pieces = [0.0] * len(inputs)
    input_pieces = [0.0] * len(inputs)
    backward_rows = []
    for t in range(len(inputs), 0, -1):  # 3,2,1; the stop value zero is excluded.
        # dL/da_t = (dL/dh_t) * (dh_t/da_t). This is not an operation that removes tanh.
        adj_a = adj_h * (1.0 - hs[t] ** 2)
        # Differentiate recurrent_weight * h_(t-1) with respect to the weight.
        pieces[t - 1] = adj_a * hs[t - 1]
        input_pieces[t - 1] = adj_a * inputs[t - 1]
        # Differentiating the same product by the previous hidden value gives the weight.
        adj_h_prev = adj_a * recurrent_weight
        backward_rows.append({
            "time": t, "hidden": hs[t], "previous_hidden": hs[t - 1],
            "loss_gradient_hidden": adj_h, "tanh_derivative": 1.0 - hs[t] ** 2,
            "loss_gradient_preactivation": adj_a,
            "recurrent_weight_piece": pieces[t - 1],
            "loss_gradient_previous_hidden": adj_h_prev,
        })
        adj_h = adj_h_prev  # Next, differentiate operations that produced the previous state.

    def loss_with_recurrent_weight(weight):
        state = 0.0
        for value in inputs:
            state = math.tanh(input_weight * value + weight * state)
        return loss_factor * (state - target) ** 2

    epsilon = 1e-6  # Small weight perturbation used for finite differences.
    finite_difference = (
        loss_with_recurrent_weight(recurrent_weight + epsilon)
        - loss_with_recurrent_weight(recurrent_weight - epsilon)
    ) / (2.0 * epsilon)
    # Sum the time-specific terms to obtain the gradient of one shared weight.
    assert abs(sum(pieces) - autograd_whh) < 1e-12
    assert abs(sum(input_pieces) - autograd_wih) < 1e-12
    assert abs(finite_difference - autograd_whh) < 1e-8
    assert pieces[0] == 0.0  # h0=0 makes only the first use of W_hh contribute zero.
    # The first W_ih term is nonzero; the first time step is not excluded from learning.
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
