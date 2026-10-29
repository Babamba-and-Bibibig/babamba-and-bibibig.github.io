# RNN sequences and hidden states: CPU / float64 / forward computation only
import torch
from torch import nn

# B: batch size, T: time steps, F: input features, H: hidden neurons
B, T, F, H = 2, 5, 4, 3

# Use 64-bit floating-point numbers to inspect the numerical calculation.
DTYPE = torch.float64

# Create one cell object.
# bias=True: use both bias vectors.
# nonlinearity="tanh": apply tanh to the summed inputs and biases.
cell = nn.RNNCell(
    input_size=F,
    hidden_size=H,
    bias=True,
    nonlinearity="tanh",
).double()

# Replace the initial parameters with the weights and biases from the article.
# Do not record these assignments for automatic differentiation.
with torch.no_grad():
    cell.weight_ih.copy_(torch.tensor([
        [ 0.11, -0.07,  0.04,  0.02],  # Current neuron A
        [-0.09,  0.13,  0.06, -0.03],  # Current neuron B
        [ 0.05,  0.08, -0.12,  0.15],  # Current neuron C
    ], dtype=DTYPE))

    cell.weight_hh.copy_(torch.tensor([
        [ 0.21, -0.14,  0.07],  # Previous A, B, C -> current A
        [-0.16,  0.25,  0.12],  # Previous A, B, C -> current B
        [ 0.19, -0.06,  0.31],  # Previous A, B, C -> current C
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

# Record the identity of each parameter object.
ids_before = tuple(id(p) for p in cell.parameters())

# Make separate copies of the parameter values as well.
# detach(): disconnect from the automatic differentiation history.
# clone(): create a separate copy of the data.
values_before = [p.detach().clone() for p in cell.parameters()]

# Each of the two sequences has three hidden values.
# shape = (B, H) = (2, 3)
state = torch.zeros(B, H, dtype=DTYPE)

# Store the (B, H) result tensor from each time step.
trace = []

with torch.no_grad():
    for t in range(T):
        # t takes the values 0, 1, 2, 3, and 4.
        # Select the same time-step index from both sequences.
        # shape: (B, F) = (2, 4)
        xt = x[:, t, :]

        # Every current neuron reads this same previous state.
        # Do not modify this tensor while computing individual neurons.
        previous_state = state

        # @ performs matrix multiplication.
        # .T transposes the two-dimensional weight matrix here.
        #
        # Input:        (B, F) = (2, 4)
        # weight_ih:    (H, F) = (3, 4)
        # Transposed:   (F, H) = (4, 3)
        # Product:      (B, H) = (2, 3)
        input_part = xt @ cell.weight_ih.T

        # Previous state: (B, H) = (2, 3)
        # weight_hh.T:    (H, H) = (3, 3)
        # Product:        (B, H) = (2, 3)
        #
        # Transposing a square matrix still changes the positions of its values.
        previous_part = previous_state @ cell.weight_hh.T

        # Each bias vector has shape (H,).
        # The same bias is added to the matching neuron in every sequence.
        manual_state = torch.tanh(
            input_part
            + cell.bias_ih
            + previous_part
            + cell.bias_hh
        )

        # Perform the same calculation using PyTorch's RNNCell.
        # Returned shape: (B, H) = (2, 3)
        state = cell(xt, previous_state)

        # allclose: compare corresponding elements within a numerical tolerance.
        # rtol: relative tolerance, scaled by the reference value's magnitude.
        # atol: absolute tolerance.
        assert torch.allclose(
            state, manual_state, rtol=1e-10, atol=1e-12
        )

        trace.append(state)

        # Print only the new hidden state of the first sequence.
        # state[0] has shape (3,).
        print(f"Time step {t + 1}: {state[0].tolist()}")

    # Combine five (2, 3) tensors by adding a time axis.
    # Result shape: (B, T, H) = (2, 5, 3)
    output = torch.stack(trace, dim=1)

print("Full input:", tuple(x.shape))
print("Full output:", tuple(output.shape))
print("Final state:", tuple(state.shape))

# Outputs for the first sequence, rounded to six decimal places:
#
# Time step 1: [0.360099,  0.502765, 0.866036]
# Time step 2: [0.416009,  0.900527, 0.510735]
# Time step 3: [0.567644,  0.654344, 0.940266]
# Time step 4: [0.824405, -0.150259, 0.956839]
# Time step 5: [0.645847,  0.176484, 0.997772]


# -------------------------------------------------------------------
# A function that processes one sequence
# Input: (T, F), return value: (T, H)
# -------------------------------------------------------------------

def run_given_cell(values: torch.Tensor) -> torch.Tensor:
    # ndim is the number of tensor axes.
    # This function receives a sequence (T, F), without a batch axis.
    if values.ndim != 2 or values.shape[1] != F or values.shape[0] == 0:
        raise ValueError("values must be a nonempty (T, F) tensor.")

    # Check that the model and input use the same data type and device.
    if (
        values.dtype != cell.weight_ih.dtype
        or values.device != cell.weight_ih.device
    ):
        raise ValueError("values and cell must have the same data type and device.")

    # Create a fresh initial state for every function call.
    # The final state of an earlier call is therefore not carried over.
    # The batch axis is omitted, so the shape is (H,).
    state = torch.zeros(
        H,
        dtype=values.dtype,
        device=values.device,
    )

    trace = []

    with torch.no_grad():
        # Iterating over 2D values yields a (F,) tensor for each time step.
        for xt in values:
            # Use the previous state as input, then receive the new state.
            state = cell(xt, state)
            trace.append(state)

    # Combine T tensors of shape (H,) into a tensor of shape (T, H).
    return torch.stack(trace, dim=0)


# -------------------------------------------------------------------
# Experiment A: What changes when the inputs are reordered?
# -------------------------------------------------------------------

original = run_given_cell(x[0])

# Change only the time-step order in the first sequence.
# [2, 0, 1, 3, 4] specifies the order of time-step indices to use.
reordered = run_given_cell(x[0, [2, 0, 1, 3, 4], :])

print("Final state in the original order:", original[-1].tolist())
print("Final state after reordering:", reordered[-1].tolist())

# Original order:
# [0.645847, 0.176484, 0.997772]
#
# Reordered inputs:
# [0.638208, 0.188234, 0.997711]

# Reordering the inputs changes the final state in this example.
assert not torch.allclose(original[-1], reordered[-1])

# Reason:
# Reordering earlier inputs changes the previous hidden state,
# and that state is used in the calculation at the next time step.
#
# A changed order is not guaranteed to produce a different result for
# every possible choice of weights and inputs. This example does differ.


# -------------------------------------------------------------------
# Experiment B: What changes when the sequence is shortened?
# -------------------------------------------------------------------

# Process only the first three time steps.
short = run_given_cell(x[0, :3, :])

# Compare with the first three results from the complete five-step sequence.
assert torch.allclose(
    short, original[:3], rtol=1e-10, atol=1e-12
)

# Reason:
# The initial state, input prefix, and weights are the same.
# In a unidirectional RNN, later inputs do not retroactively change
# the hidden states that were computed at earlier time steps.


# -------------------------------------------------------------------
# Experiment C: Are the parameter objects and their values unchanged?
# -------------------------------------------------------------------

# Compare object identities.
assert ids_before == tuple(id(p) for p in cell.parameters())

# Compare the numerical values separately.
# zip: pair each current parameter with its saved copy in order.
# torch.equal: check that shapes and element values match.
# all: check that every comparison is True.
assert all(
    torch.equal(p, before)
    for p, before in zip(cell.parameters(), values_before)
)

print("Parameter tensor count:", len(tuple(cell.parameters())))
print("Trainable scalar count:", sum(p.numel() for p in cell.parameters()))

# Output:
# Parameter tensor count: 4
# Trainable scalar count: 27


# -------------------------------------------------------------------
# Experiment D: Does nn.RNN produce the same sequence of states?
# -------------------------------------------------------------------

rnn = nn.RNN(
    input_size=F,
    hidden_size=H,
    num_layers=1,          # One RNN layer.
    nonlinearity="tanh",
    bias=True,
    batch_first=True,     # Input and output: (B, T, features).
    bidirectional=False,  # Process only in the forward direction.
).double()

with torch.no_grad():
    # Use identical parameter values for the comparison.
    # _l0 refers to the first RNN layer, whose layer index is 0.
    rnn.weight_ih_l0.copy_(cell.weight_ih)
    rnn.weight_hh_l0.copy_(cell.weight_hh)
    rnn.bias_ih_l0.copy_(cell.bias_ih)
    rnn.bias_hh_l0.copy_(cell.bias_hh)

    # Pass the entire sequence batch in one call.
    # The omitted initial hidden state defaults to zeros.
    rnn_output, h_n = rnn(x)

# Compare with the repeated RNNCell calls.
assert torch.allclose(
    rnn_output, output, rtol=1e-10, atol=1e-12
)

# Compare the final hidden states.
assert torch.allclose(
    h_n[0], output[:, -1, :], rtol=1e-10, atol=1e-12
)

# Compare the first sequence processed in the batch with the same
# sequence processed on its own.
assert torch.allclose(
    output[0], original, rtol=1e-10, atol=1e-12
)

print("nn.RNN output:", tuple(rnn_output.shape))  # (2, 5, 3)
print("nn.RNN h_n:", tuple(h_n.shape))            # (1, 2, 3)
print("All checks passed.")
