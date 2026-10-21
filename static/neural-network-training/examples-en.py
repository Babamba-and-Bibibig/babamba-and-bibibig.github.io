"""Inspect neural-network training: forward pass, loss, gradients, and updates.

PyTorch is the only external package required. The script runs on the CPU.
Fictional data: [small-box count, large-box count] -> packing time in minutes.
The target for A=[2, 1] is 6 minutes; the target for B=[1, 1] is 4 minutes.
These two examples illustrate calculations; they do not establish predictive quality.
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# float64 stores 64-bit floating-point numbers for these small numerical checks.
DTYPE = torch.float64


class StudyMLP(nn.Module):
    """Two input features -> two hidden neurons -> one output neuron."""

    def __init__(self):
        super().__init__()  # Initialize nn.Module's parameter management.
        # Parameter registers a trainable tensor (an array of numbers) in the model.
        # Our matrices use rows for input features and columns for output neurons.
        # W_hidden = [[w1, w3], [w2, w4]]: (2, 2).
        self.W_hidden = nn.Parameter(
            torch.tensor([[1.0, -1.0], [2.0, 1.0]], dtype=DTYPE)
        )
        self.b_hidden = nn.Parameter(torch.tensor([0.0, 2.0], dtype=DTYPE))
        self.W_out = nn.Parameter(torch.tensor([[2.0], [-1.0]], dtype=DTYPE))
        self.b_out = nn.Parameter(torch.tensor([1.0], dtype=DTYPE))

    def forward(self, X):
        # @ is matrix multiplication. Each sample receives the same neuron biases.
        Z = X @ self.W_hidden + self.b_hidden  # (N, 2).
        H = torch.relu(Z)                     # Elementwise max(0, z), (N, 2).
        return H @ self.W_out + self.b_out    # (N, 1).


def check_central_differences(model, X, y, expected_grads):
    """Change one parameter at a time and compare numerical and analytic gradients."""
    eps = 1e-5  # The two comparison positions are current value +/-0.00001.
    checked = 0
    # no_grad disables new differentiation records, not the numerical calculations.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            numeric = torch.empty_like(parameter)  # Result array with the same shape.
            # view(-1) views the same storage as one dimension; it does not copy here.
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
                    flat_parameter[index] = original  # Restore after each comparison.
                # The denominator is the distance between positions: (w+eps)-(w-eps).
                flat_numeric[index] = (plus - minus) / (2 * eps)
                checked += 1
            # assert_close checks agreement within a tolerance and raises if it fails.
            # rtol sets relative tolerance; atol sets absolute tolerance.
            torch.testing.assert_close(
                numeric, expected_grads[name], rtol=1e-6, atol=1e-8
            )
    assert checked == 9
    print("Parameters checked by central differences and backpropagation:", checked)


def inspect_one_update(X, y):
    """Compare manual gradients, autograd, and one SGD update from the same state."""
    model = StudyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # zero_grad clears previous .grad values; it does not change the parameters.
    optimizer.zero_grad(set_to_none=True)
    # detach removes the differentiation connection; clone copies to separate storage.
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    # 1. Expand forward's operations so we can inspect intermediate values.
    Z = X @ model.W_hidden + model.b_hidden  # [[4, 1], [3, 2]].
    H = torch.relu(Z)                       # All entries are positive, so H equals Z.
    pred = H @ model.W_out + model.b_out    # [[8], [5]].
    # Intermediate .grad values are normally not retained. Keep them for inspection.
    for intermediate in (Z, H, pred):
        intermediate.retain_grad()

    # 2. square returns elementwise squares; mean averages all tensor elements.
    assert pred.shape == y.shape == (2, 1)
    per_loss = 0.5 * (pred - y).square()  # [[2], [0.5]], shape=(2, 1).
    loss = per_loss.mean()               # 1.25, shape=(): a single number.
    print("\n[Forward pass and loss]")
    for name, value in [("Z", Z), ("H", H), ("Prediction", pred), ("Per-sample loss", per_loss)]:
        print(name, value.detach().tolist())  # tolist converts a tensor to Python lists.
    print("Mean loss:", loss.item())  # item extracts a one-element tensor as a Python number.
    torch.testing.assert_close(pred, torch.tensor([[8.0], [5.0]], dtype=DTYPE))
    assert loss.item() == 1.25

    # 3. Manual backpropagation. Each g_ is the mean loss's gradient for that value.
    N = X.shape[0]  # Actual number of samples in this batch: 2.
    with torch.no_grad():
        g_pred = (pred - y) / N             # (2, 1): [[1], [0.5]].
        g_W_out = H.T @ g_pred              # (2, 1): [[5.5], [2]].
        g_b_out = g_pred.sum(dim=0)         # (2, 1) -> (1,): [1.5].
        g_H = g_pred @ model.W_out.T        # (2, 2): gradient passed to the hidden layer.
        g_Z = g_H * (Z > 0)                 # ReLU uses derivative 1 above zero, else 0.
        g_W_hidden = X.T @ g_Z              # (2, 2): sum the sample contributions.
        g_b_hidden = g_Z.sum(dim=0)          # (2, 2) -> (2,): [3, -1.5].
    # .T swaps rows and columns. sum(dim=0) sums samples and removes that dimension.
    # g_pred already includes /N; do not divide by N again later.
    # ReLU has no ordinary derivative at zero; PyTorch uses zero at that point.
    manual = {"W_hidden": g_W_hidden, "b_hidden": g_b_hidden,
              "W_out": g_W_out, "b_out": g_b_out}
    expected = {"W_hidden": [[5.0, -2.5], [3.0, -1.5]],
                "b_hidden": [3.0, -1.5], "W_out": [[5.5], [2.0]],
                "b_out": [1.5]}

    # 4. Automatic backpropagation returns None and accumulates results into .grad.
    result = loss.backward()
    assert result is None
    torch.testing.assert_close(pred.grad, g_pred)
    torch.testing.assert_close(H.grad, g_H)
    torch.testing.assert_close(Z.grad, g_Z)
    print("\n[Gradients for each parameter]")
    for name, parameter in model.named_parameters():
        # Also verify that backward did not change the parameter values.
        torch.testing.assert_close(parameter.detach(), before[name])
        torch.testing.assert_close(parameter.grad, manual[name])
        torch.testing.assert_close(parameter.grad, torch.tensor(expected[name], dtype=DTYPE))
        print(name, "gradient:", parameter.grad.tolist(), "shape:", tuple(parameter.grad.shape))
    # numel counts numerical elements: these four tensors contain nine parameters.
    assert sum(p.numel() for p in model.parameters()) == 9
    assert sum(p.grad.numel() for p in model.parameters()) == 9
    check_central_differences(model, X, y, manual)

    # 5. Only now do we actually update parameters according to the learning rule.
    optimizer.step()  # New value = old value - 0.01 * gradient.
    changed = 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name] - 0.01 * manual[name])
        torch.testing.assert_close(parameter.grad, manual[name])  # Still the old-state gradient.
        changed += (parameter.detach() != before[name]).sum().item()
    assert changed == 9  # All nine gradients are nonzero in this particular example.
    with torch.no_grad():
        pred_after = model(X)  # Run a new forward pass using the new parameters.
        loss_after = (0.5 * (pred_after - y).square()).mean()
    print("\nPredictions after update:", pred_after.tolist())
    print("Mean loss after update:", loss_after.item())
    # isclose compares numbers while allowing small floating-point representation errors.
    assert math.isclose(loss_after.item(), 0.522123460625, rel_tol=1e-12)
    assert loss.item() == 1.25  # The previously computed loss does not change itself.


def inspect_four_layers(X, y):
    """Distinguish four Linear layers, eight tensors, and 21 parameter numbers."""
    # Linear(D, K) computes K outputs from D inputs.
    # Sequential runs its registered modules in order.
    model = nn.Sequential(
        nn.Linear(2, 2, dtype=DTYPE), nn.ReLU(),
        nn.Linear(2, 2, dtype=DTYPE), nn.ReLU(),
        nn.Linear(2, 2, dtype=DTYPE), nn.ReLU(),
        nn.Linear(2, 1, dtype=DTYPE),
    )
    # Linear stores weight as (outputs, inputs) and computes X @ weight.T + bias.
    # This model has its own initial values, so its gradient values may differ.
    loss = 0.5 * (model(X) - y).square().mean()
    loss.backward()
    parameters = list(model.parameters())
    assert len(parameters) == 8
    assert sum(p.numel() for p in parameters) == 21
    assert sum(p.grad.numel() for p in parameters) == 21
    print("\n[Four-layer model: value shape / gradient shape]")
    for name, p in model.named_parameters():
        assert p.shape == p.grad.shape
        print(name, tuple(p.shape), tuple(p.grad.shape))
    # 0,2,4,6 are module positions; the ReLUs at 1,3,5 have no learned parameters.
    # The 21 gradients count results, not the number of multiplications or additions.


def train_three_epochs(X, y):
    """Inspect a standard batch training loop and inference using the same two samples."""
    model = StudyMLP()  # Start from the initial values in a separate model.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # TensorDataset pairs input and target rows to form samples.
    # DataLoader is the class that retrieves those samples in batches.
    loader = DataLoader(TensorDataset(X, y), batch_size=2, shuffle=True, drop_last=False)
    # shuffle moves inputs and targets together; drop_last=False keeps a smaller final batch.
    total_steps = 0
    print("\n[Three training epochs]")
    for epoch in range(3):
        model.train()  # Switch modes; this call alone performs no training computation.
        loss_total, sample_total = 0.0, 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)  # Clear previous gradients.
            batch_pred = model(batch_X)            # Forward pass.
            assert batch_pred.shape == batch_y.shape
            batch_loss = 0.5 * (batch_pred - batch_y).square().mean()
            batch_loss.backward()                  # Gradients at the current state.
            optimizer.step()                       # Update the parameters.
            total_steps += 1
            # Weight each batch mean by its actual sample count for the epoch record.
            count = batch_X.shape[0]
            loss_total += batch_loss.item() * count
            sample_total += count
        print(f"epoch={epoch + 1}, mean loss observed during training={loss_total / sample_total:.6f}")
    assert total_steps == 3  # Two samples / batch size two * three epochs.
    # Those records use losses at batch-processing time, not a final-model reevaluation.
    model.eval()  # Set Dropout/BatchNorm to evaluation mode; this model has neither.
    with torch.no_grad():  # eval does not disable differentiation records on its own.
        prediction = model(X)
    print("Inference after training:", prediction.tolist())


def compare_learning_rates():
    """Keep other conditions fixed for the separate function J(u)=(u-3)^2."""
    histories = {}  # A dictionary mapping each learning rate to its record list.
    expected_u = {0.05: [0.0, 0.3, 0.57, 0.813],
                  0.25: [0.0, 1.5, 2.25, 2.625],
                  1.1: [0.0, 6.6, -1.32, 8.184]}
    print("\n[Learning-rate comparison for J(u)=(u-3)^2]")
    for learning_rate in [0.05, 0.25, 1.1]:
        u = 0.0  # Begin every experiment at the same position.
        records = [{"step": 0, "u": u, "loss": (u - 3) ** 2}]
        for step in range(1, 4):  # Three values: 1,2,3. The stop value 4 is excluded.
            gradient = 2 * (u - 3)  # Recompute at the new position on every iteration.
            u = u - learning_rate * gradient
            records.append({"step": step, "u": u, "loss": (u - 3) ** 2})
        # append adds to a list and returns None; do not assign that result to records.
        histories[learning_rate] = records
        # zip retrieves items at the same position from the two lists as pairs.
        for record, expected in zip(records, expected_u[learning_rate]):
            assert math.isclose(record["u"], expected, abs_tol=1e-12)
        print("Learning rate:", learning_rate)
        print("u:", [round(row["u"], 6) for row in records])
        print("J:", [round(row["loss"], 6) for row in records])
    # Each experiment has three updates and four records, including the initial state.
    # This function satisfies (u_new-3)=(1-2*learning_rate)*(u-3).
    # 0.25 is faster among these three choices, not optimal for every loss function.
    return histories


def main():
    # manual_seed makes random choices reproducible; the nine manual parameters are fixed.
    torch.manual_seed(0)
    # tensor creates an array of numbers. Input shape is (samples, features)=(2, 2).
    X = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=DTYPE)
    # Match target and prediction shapes at (N, 1) to avoid accidental broadcasting with (N,).
    y = torch.tensor([[6.0], [4.0]], dtype=DTYPE)
    inspect_one_update(X, y)
    inspect_four_layers(X, y)
    train_three_epochs(X, y)
    compare_learning_rates()
    print("\nAll numerical checks passed.")


if __name__ == "__main__":
    main()  # Run the operations above in order when this file is executed directly.
