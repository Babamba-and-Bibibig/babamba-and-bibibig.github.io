"""Forecast next-day visitors with an RNN: complete CPU example.

Run python examples-en.py in an environment with PyTorch installed.
This synthetic experiment does not establish accuracy for a real service.
"""

import argparse
import copy
import json
import platform
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# nn is a short name for torch.nn. nn.Module is the base class for networks.
# TensorDataset pairs X and y at the same position into one sample.
# DataLoader groups those samples into minibatches.
VALUE_SCALE = 100.0  # Fixed divisor for visitor counts; this does not create probabilities.
SEQ_LEN = 14        # Number of time steps T in one input window.
HIDDEN_SIZE = 4     # Chosen hidden width H; it is not calculated from the 14 days.
DATA_SEED = 1       # Random starting configuration for the synthetic data.
TRAIN_SEED = 7      # Reproduce model initialization and training-sample order.


def make_windows(values: torch.Tensor, seq_len: int):
    """Turn a 1D record into inputs (N,T,1) and next-day targets (N,1)."""
    if values.ndim != 1 or not 1 <= seq_len < len(values):
        raise ValueError("Expected a 1D record and a positive window length leaving a next-day target.")
    windows = []
    # range(n) excludes n. i is the starting index of an input window.
    for i in range(len(values) - seq_len):
        windows.append(values[i:i + seq_len])  # The stop index is excluded.
    # stack creates a sample dimension; unsqueeze adds the feature dimension.
    X = torch.stack(windows, dim=0).unsqueeze(-1)
    y = values[seq_len:].unsqueeze(-1)  # The value immediately after each window.
    return X, y


def make_data():
    torch.manual_seed(DATA_SEED)
    weekly = torch.tensor(
        [62, 55, 58, 60, 75, 118, 105], dtype=torch.float32,
    )
    # Repeat the entire seven-item pattern 12 times to create 84 synthetic days.
    repeated = weekly.repeat(12)
    # randn samples a mean-zero, unit-standard-deviation distribution; 3 scales the variation.
    visitors = (repeated + torch.randn(len(repeated)) * 3.0) / VALUE_SCALE
    X, y = make_windows(visitors, SEQ_LEN)
    # Target positions in the original record: indices 14 through 83.
    target_positions = torch.arange(SEQ_LEN, len(visitors))
    return visitors, X, y, target_positions


class VisitorRNN(nn.Module):
    """One unidirectional RNN layer and a prediction layer; each call starts at zero."""

    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,           # One observed feature per time step: visitors.
            hidden_size=hidden_size,
            num_layers=1,           # One RNN layer.
            nonlinearity="tanh",    # Apply tanh to each element of the weighted sum.
            batch_first=True,       # Input and output dimension order: (B,T,features).
        )
        self.head = nn.Linear(
            in_features=hidden_size,  # Receive the H values of the last state.
            out_features=1,           # Produce one next-day value per sample.
        )

    def forward(self, x):
        # x: (B,T,1). Omitting h0 starts each call from a zero initial state.
        output, _ = self.rnn(x)  # output: (B,T,H). _ holds the unused h_n.
        last = output[:, -1, :]  # Last time step for every sample: (B,H).
        return self.head(last)   # Final prediction: (B,1).


def verify_shapes_and_state(visitors, X, y):
    """Verify windows, time iteration, and individual versus batched calls."""
    demo = torch.tensor([10., 20., 30., 40., 50., 60., 70.])
    demo_X, demo_y = make_windows(demo, 3)
    assert demo_X[:, :, 0].tolist() == [
        [10., 20., 30.], [20., 30., 40.],
        [30., 40., 50.], [40., 50., 60.],
    ]
    assert demo_y[:, 0].tolist() == [40., 50., 60., 70.]
    assert X.shape == (70, 14, 1) and y.shape == (70, 1)
    assert torch.equal(X[0, :, 0], visitors[0:14])
    assert torch.equal(y[0, 0], visitors[14])
    assert torch.equal(X[69, :, 0], visitors[69:83])
    assert torch.equal(y[69, 0], visitors[83])

    torch.manual_seed(TRAIN_SEED)
    rnn = nn.RNN(1, HIDDEN_SIZE, batch_first=True)
    cell = nn.RNNCell(1, HIDDEN_SIZE)
    # Separately constructed objects have different random weights; copy values for comparison.
    with torch.no_grad():
        cell.weight_ih.copy_(rnn.weight_ih_l0)
        cell.weight_hh.copy_(rnn.weight_hh_l0)
        cell.bias_ih.copy_(rnn.bias_ih_l0)
        cell.bias_hh.copy_(rnn.bias_hh_l0)
    before = {name: p.detach().clone() for name, p in cell.named_parameters()}
    xb = X[:2]  # B=2, T=14, F=1; distinguish this from the full dataset N=70.
    state = xb.new_zeros(2, HIDDEN_SIZE)  # Same dtype/device as the input; shape (B,H).
    history = []
    # unbind(dim=1) returns 14 tensors of shape (B,F), removing the time dimension.
    for xt in xb.unbind(dim=1):
        state = cell(xt, state)  # Compute a new state; this does not update weights.
        history.append(state)   # Do not detach: preserve the connection through time.
    cell_output = torch.stack(history, dim=1)  # (2,14,4).
    output, h_n = rnn(xb)
    assert output.shape == (2, 14, 4) and h_n.shape == (1, 2, 4)
    # allclose permits small floating-point differences.
    assert torch.allclose(cell_output, output, atol=1e-6, rtol=1e-6)
    assert torch.allclose(output[:, -1, :], h_n[0])
    assert all(torch.equal(before[n], p) for n, p in cell.named_parameters())

    model = VisitorRNN()
    model.eval()  # Select evaluation behavior; this does not disable autograd.
    with torch.no_grad():  # Do not record operations for backward inside this block.
        together = model(X)
        individual = torch.cat(
            [model(row.unsqueeze(0)) for row in X], dim=0,
        )
    assert torch.allclose(individual, together, atol=1e-6, rtol=1e-6)

    # Also verify a general multifeature shape: B2,T5,F4,H3.
    multi = torch.randn(2, 5, 4)
    multi_rnn = nn.RNN(4, 3, batch_first=True)
    multi_output, multi_hn = multi_rnn(multi)
    assert multi_output.shape == (2, 5, 3) and multi_hn.shape == (1, 2, 3)
    return {
        "X_shape": list(X.shape), "y_shape": list(y.shape),
        "demo_windows": demo_X[:, :, 0].tolist(),
        "demo_targets": demo_y[:, 0].tolist(),
        "output_shape": list(output.shape), "h_n_shape": list(h_n.shape),
        "cell_rnn_max_difference": (cell_output - output).abs().max().item(),
        "individual_batch_max_difference": (individual - together).abs().max().item(),
        "state_loop_keeps_weights": True,
        "multifeature_output_shape": list(multi_output.shape),
    }


def verify_learning_step(X, y):
    """Use a separate model to verify forward, backward, and parameter updates."""
    torch.manual_seed(TRAIN_SEED)
    model = VisitorRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)  # Clear previous .grad values, not weights.
    prediction = model(X[:2])
    loss = nn.MSELoss()(prediction, y[:2])  # Mean squared error across the two samples.
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    loss.backward()  # Compute weight gradients through all connected time steps.
    assert all(p.grad is not None and p.grad.shape == p.shape for p in model.parameters())
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    recurrent_gradient_shape = list(model.rnn.weight_hh_l0.grad.shape)
    optimizer.step()  # Update weights using the gradients that were computed.
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters())
    return {
        "forward_keeps_weights": True, "backward_keeps_weights": True,
        "step_changes_weights": True,
        "recurrent_gradient_shape": recurrent_gradient_shape,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def train_and_evaluate(visitors, X, y, target_positions):
    train_mask = target_positions < 56
    valid_mask = (target_positions >= 56) & (target_positions < 70)
    test_mask = target_positions >= 70
    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    assert len(X_train) == 42 and len(X_valid) == len(X_test) == 14

    # Shuffle complete windows; retain the date order inside every window.
    generator = torch.Generator().manual_seed(TRAIN_SEED)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=16,    # Pass up to 16 windows in one call.
        shuffle=True,    # Shuffle training windows that start independently.
        drop_last=False, # Include the last ten samples in training.
        generator=generator,  # Separate random generator for sample shuffling.
    )
    torch.manual_seed(TRAIN_SEED)
    device = torch.device("cpu")  # The article reports CPU reproduction results.
    model = VisitorRNN().to(device)
    loss_fn = nn.MSELoss()  # Default reduction="mean": mean of elementwise squared errors.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    batch_sizes = []
    first_epoch_loss = None
    last_epoch_loss = None

    # Use validation to select a checkpoint. Do not use the test set in this loop.
    for epoch in range(1, 301):  # Epochs 1 through 300; 301 is excluded.
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            assert prediction.shape == yb.shape
            loss = loss_fn(prediction, yb)
            loss.backward()
            # If large gradients need limiting, put clipping between backward and step.
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # Each batch loss is a mean; multiply by its sample count before summing.
            loss_sum += loss.item() * len(xb)
            sample_count += len(xb)
            if epoch == 1:
                batch_sizes.append(len(xb))
        train_loss = loss_sum / sample_count
        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(model(X_valid.to(device)), y_valid.to(device)).item()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            # Copy the state independently so later training does not alter the saved values.
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1:
            first_epoch_loss = train_loss
        last_epoch_loss = train_loss
        if epoch in (1, 50, 100, 150, 200, 250, 300):
            print(f"epoch={epoch:3d} train_mse={train_loss:.6f} valid_mse={valid_loss:.6f}")

    assert batch_sizes == [16, 16, 10]
    model.load_state_dict(best_state)  # Restore the weights with the best validation loss.
    model.eval()
    with torch.no_grad():
        test_prediction = model(X_test.to(device)).cpu()
        # Use the final 14 days to forecast day 85, for which no target is available.
        next_input = visitors[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        next_prediction = model(next_input.to(device)).item() * VALUE_SCALE
    test_positions = target_positions[test_mask]
    yesterday_prediction = visitors[test_positions - 1].unsqueeze(-1)
    weekly_prediction = visitors[test_positions - 7].unsqueeze(-1)

    def mae_in_visitors(prediction):
        # MAE is mean absolute error; undo /100 to report the original visitor unit.
        return (prediction - y_test).abs().mean().item() * VALUE_SCALE

    return {
        "split_samples": {"train": 42, "validation": 14, "test": 14},
        "target_index_ranges": {"train": [14, 55], "validation": [56, 69], "test": [70, 83]},
        "batch_sizes": batch_sizes, "steps_per_epoch": len(train_loader),
        "epochs_run": 300, "optimizer_steps": 300 * len(train_loader),
        "best_epoch": best_epoch, "best_validation_mse": best_valid_loss,
        "first_epoch_training_mse": first_epoch_loss,
        "last_epoch_training_mse": last_epoch_loss,
        "test_mae_visitors": {
            "rnn": mae_in_visitors(test_prediction),
            "yesterday": mae_in_visitors(yesterday_prediction),
            "previous_week": mae_in_visitors(weekly_prediction),
        },
        "test_predictions": (test_prediction[:, 0] * VALUE_SCALE).tolist(),
        "test_targets": (y_test[:, 0] * VALUE_SCALE).tolist(),
        "day_85_prediction": next_prediction,
        "evaluation_protocol": "Rolling one-step-ahead with observed history; not a 14-day-ahead forecast.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Optional path for a JSON results file")
    args = parser.parse_args()
    torch.set_num_threads(1)  # Reduce overhead and variability for this small CPU example.
    visitors, X, y, target_positions = make_data()
    results = {
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "device": "cpu", "dtype": "float32", "data_seed": DATA_SEED,
        "training_seed": TRAIN_SEED, "sequence_length": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "shape_checks": verify_shapes_and_state(visitors, X, y),
        "learning_step_checks": verify_learning_step(X, y),
        "training": train_and_evaluate(visitors, X, y, target_positions),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.results is not None:
        args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
