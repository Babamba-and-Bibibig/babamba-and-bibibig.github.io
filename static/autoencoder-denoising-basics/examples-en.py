"""Autoencoders: forward pass, one update, and AE/DAE training.

Run this file from a Python environment with PyTorch, NumPy, scikit-learn,
and Matplotlib installed. The default mode only runs the small example.
No dataset download or separate MNIST file is needed for the digits modes.

Examples:
    python examples-en.py
    python examples-en.py --mode ae --epochs 20
    python examples-en.py --mode dae --epochs 20 --output-dir figures
    python examples-en.py --pair assets/mnist_day1_pair.npz --output-dir figures

The optional pair is an input/target inspection, not a model prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


def tiny_ae(x: torch.Tensor, e: torch.Tensor, d: torch.Tensor):
    """Return latent scalar, reconstructed vector, and scalar MSE."""
    # Each argument contains two values for ONE sample, not a batch.
    if x.shape != (2,) or e.shape != (2,) or d.shape != (2,):
        raise ValueError("x, e, and d must each have shape (2,).")
    # Elementwise multiplication, then sum: (2,) -> scalar shape ().
    z = (e * x).sum()
    # The same scalar z is used by both decoder weights: (2,).
    reconstruction = d * z
    # Subtract matching components; square them; average TWO elements.
    loss = (reconstruction - x).square().mean()
    return z, reconstruction, loss


def run_tiny():
    """Check the article's forward values, gradients, update, and Q1/Q2."""
    # float64 makes this small numerical comparison easier to inspect.
    x_a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    x_b = torch.tensor([2.0, 1.0], dtype=torch.float64)
    # requires_grad records operations that depend on these weights.
    e = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([0.8, 1.2], dtype=torch.float64, requires_grad=True)
    # detach removes the differentiation connection; clone copies storage.
    e_initial = e.detach().clone()
    d_initial = d.detach().clone()

    z_a0, r_a0, loss_a0 = tiny_ae(x_a, e, d)
    with torch.no_grad():
        # B is observed only; its loss is not included in this update.
        z_b0, r_b0, loss_b0 = tiny_ae(x_b, e, d)

    print("Before update")
    for name, z, r, loss in [
        ("A", z_a0, r_a0, loss_a0),
        ("B", z_b0, r_b0, loss_b0),
    ]:
        # item extracts one Python number; tolist extracts a Python list.
        print(f"{name}: z={z.item():.6f}, reconstruction={r.tolist()}, MSE={loss.item():.10f}")

    errors = (r_a0 - x_a).square()
    # A scalar already contains only ONE value; mean cannot recover a count.
    wrong_mean = errors.sum().mean()
    correct_mean = errors.mean()
    print("mean(sum(errors)):", wrong_mean.item())  # 0.08, not the MSE.
    print("mean(errors):", correct_mean.item())     # 0.04.
    # stack adds a new axis and preserves the differentiation connection.
    loss_by_stack = torch.stack([errors[0], errors[1]]).mean()
    assert loss_by_stack.requires_grad and loss_by_stack.grad_fn is not None
    torch.testing.assert_close(loss_by_stack, loss_a0, rtol=0, atol=1e-12)
    # Do not replace stack with torch.tensor([errors[0], errors[1]]).
    # That constructor copies values and breaks their autograd connection.

    # backward computes gradients; it does NOT change e or d.
    loss_a0.backward()
    assert e.grad is not None and d.grad is not None
    expected_e_grad = torch.tensor([-0.08, -0.16], dtype=torch.float64)
    expected_d_grad = torch.tensor([0.3, -0.3], dtype=torch.float64)
    torch.testing.assert_close(e.grad, expected_e_grad, rtol=0, atol=1e-12)
    torch.testing.assert_close(d.grad, expected_d_grad, rtol=0, atol=1e-12)
    torch.testing.assert_close(e.detach(), e_initial, rtol=0, atol=0)
    torch.testing.assert_close(d.detach(), d_initial, rtol=0, atol=0)
    print("e.grad:", e.grad.tolist())
    print("d.grad:", d.grad.tolist())

    # Both updates use gradients computed at the SAME pre-update weights.
    grad_e = e.grad.clone()
    grad_d = d.grad.clone()
    learning_rate = 0.1
    with torch.no_grad():
        # no_grad omits update tracking; it does not forbid changing values.
        e -= learning_rate * grad_e
        d -= learning_rate * grad_d
        # A new forward pass is necessary to obtain the new outputs.
        after_a = tiny_ae(x_a, e, d)
        after_b = tiny_ae(x_b, e, d)
    print("After one update using A")
    print("e:", e.tolist(), "d:", d.tolist())
    for name, result, expected_loss in [
        ("A", after_a, 0.02285764),
        ("B", after_b, 0.7275415696),
    ]:
        z, r, loss = result
        torch.testing.assert_close(
            loss, torch.tensor(expected_loss, dtype=torch.float64),
            rtol=0, atol=1e-12,
        )
        print(f"{name}: z={z.item():.6f}, reconstruction={r.tolist()}, MSE={loss.item():.10f}")

    # Batch axis 0 has size 2; feature axis 1 has size 2.
    x_batch = torch.stack([x_a, x_b])  # (2, 2).
    # dim=1 sums each sample's features; keepdim keeps that axis at size 1.
    z_batch = (x_batch * e_initial).sum(dim=1, keepdim=True)
    r_batch = z_batch * d_initial     # (2, 1) times (2,) -> (2, 2).
    assert z_batch.shape == (2, 1) and r_batch.shape == (2, 2)
    print("Batch Z:", z_batch.tolist(), "shape:", tuple(z_batch.shape))
    print("Batch reconstruction:", r_batch.tolist())
    print("Batch MSE:", (r_batch - x_batch).square().mean().item())  # 0.34.

    # Q1/Q2 are separate conditions, not the previous A-update experiment.
    conditions = [
        ("Q1", [0.2, 0.4], [1.0, 1.5], 0.8, [0.8, 1.2], 0.74),
        ("Q2", [0.38, 0.49], [1.096, 1.484], 1.25, [1.37, 1.855], 0.5639625),
    ]
    print("Q1 / Q2: same input B, different encoder AND decoder weights")
    for name, e_values, d_values, expected_z, expected_r, expected_loss in conditions:
        qe = torch.tensor(e_values, dtype=torch.float64)
        qd = torch.tensor(d_values, dtype=torch.float64)
        z, r, loss = tiny_ae(x_b, qe, qd)
        for actual, expected in [(z, expected_z), (r, expected_r), (loss, expected_loss)]:
            torch.testing.assert_close(
                actual, torch.tensor(expected, dtype=torch.float64),
                rtol=0, atol=1e-12,
            )
        print(f"{name}: z={z.item():.6f}, reconstruction={r.tolist()}, MSE={loss.item():.7f}")
    print("All small-example checks passed.")


class DenseAutoencoder(nn.Module):
    """A fully connected encoder and decoder for inputs shaped (N, F)."""

    def __init__(self, in_features: int, hidden_features: int, latent_features: int):
        # Module registers child layers and their trainable parameters.
        super().__init__()
        # Constructor arguments are FEATURE counts; batch size is not here.
        # Sequential calls its layers in order. Linear includes a bias by default.
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),  # Negative values become zero; shape is unchanged.
            nn.Linear(hidden_features, latent_features),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features),
            nn.Sigmoid(),  # Suitable here because the targets are in [0, 1].
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # model(x) calls this method with an actual batch; it creates no new layers.
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction


def run_digits(denoise: bool = False, epochs: int = 20, output_dir: Path | None = None):
    """Train on 1,437 digits images; evaluate on 360 separate original images."""
    from sklearn.datasets import load_digits
    from torch.utils.data import DataLoader, TensorDataset

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    # A fresh seed AND a fresh model for each run. Exact values may vary by version.
    torch.manual_seed(42)
    # digits is an 8x8 dataset, NOT MNIST. Original pixel values range from 0 to 16.
    # flatten(start_dim=1) preserves sample axis 0 and combines each image's pixels.
    x = torch.tensor(load_digits().images, dtype=torch.float32)
    x = x.flatten(start_dim=1) / 16.0  # (1797, 8, 8) -> (1797, 64), range [0, 1].

    # Split ORIGINAL images before generating noisy versions to avoid overlap.
    indices = torch.randperm(len(x))  # A permutation of indices 0 through 1796.
    split = int(len(x) * 0.8)         # 1437.
    train_x = x[indices[:split]]      # Indices before split; endpoint excluded.
    test_x = x[indices[split:]]       # Remaining 360 images, never used for updates.
    assert train_x.shape == (1437, 64) and test_x.shape == (360, 64)
    # TensorDataset returns one-element tuples here; batch_size is sample count.
    loader = DataLoader(TensorDataset(train_x), batch_size=64, shuffle=True)

    model = DenseAutoencoder(64, 32, 16)  # (N,64)->(N,32)->(N,16)->(N,32)->(N,64).
    loss_fn = nn.MSELoss()                # Default: mean over ALL output elements.
    # parameters includes both encoder and decoder weights and biases.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Mode:", "DAE" if denoise else "AE")
    print("Training shape:", tuple(train_x.shape), "held-out shape:", tuple(test_x.shape))

    for epoch in range(epochs):
        model.train()  # Training behavior for layers such as Dropout/BatchNorm.
        total_loss = 0.0
        # The trailing comma unpacks a tuple containing one clean tensor batch.
        for (clean_batch,) in loader:
            if denoise:
                # randn_like: same shape, mean-zero unit-standard-deviation noise.
                # 0.25 is the noise std BEFORE clamp; clamp clips values to [0, 1].
                input_batch = (
                    clean_batch + 0.25 * torch.randn_like(clean_batch)
                ).clamp(0, 1)
            else:
                input_batch = clean_batch

            # Clear prior gradients; a new forward pass creates a new graph.
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(input_batch)
            assert reconstruction.shape == clean_batch.shape
            # The target stays clean for BOTH AE and DAE.
            loss = loss_fn(reconstruction, clean_batch)
            loss.backward()   # Calculate gradients and accumulate them in .grad.
            optimizer.step()  # Update registered weights using those gradients.

            # Weight batch mean by sample count because the final batch is smaller.
            # This is correct here because EVERY sample has the same 64 features.
            total_loss += loss.item() * len(clean_batch)
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"epoch={epoch + 1}, train_MSE={total_loss / len(train_x):.6f}")

    # A dedicated generator makes evaluation noise independent of training RNG.
    generator = torch.Generator().manual_seed(123)
    if denoise:
        test_input = (
            test_x + 0.25 * torch.randn(test_x.shape, generator=generator)
        ).clamp(0, 1)
    else:
        test_input = test_x

    # eval changes applicable layer behavior; no_grad separately disables recording.
    model.eval()
    with torch.no_grad():
        z = model.encoder(test_input)
        reconstruction = model.decoder(z)
        model_mse = loss_fn(reconstruction, test_x).item()
        baseline_mse = loss_fn(test_input, test_x).item()
    assert z.shape == (360, 16) and reconstruction.shape == (360, 64)
    assert torch.isfinite(reconstruction).all()
    print("Evaluation input shape:", tuple(test_input.shape))
    print("Evaluation latent shape:", tuple(z.shape))
    print("Evaluation reconstruction shape:", tuple(reconstruction.shape))
    print(f"Held-out model MSE: {model_mse:.8f}")
    if denoise:
        print(f"Unprocessed noisy baseline MSE: {baseline_mse:.8f}")
        print(f"MSE improvement: {baseline_mse - model_mse:.8f}")
        print("Model beats noisy baseline:", model_mse < baseline_mse)
    else:
        # A clean input copied unchanged has zero MSE; it is not a learned AE.
        print("AE input is clean. Reconstruction MSE measures the bottleneck model.")

    if output_dir is not None:
        _plot_digits(test_input, reconstruction, test_x, output_dir, denoise)
    # Reusing this held-out set for model selection makes it a validation set.
    # Use a separate untouched test set for a final performance claim.
    return model, test_input, reconstruction, test_x


def _plot_digits(inputs, reconstruction, targets, output_dir: Path, denoise: bool):
    """Save input, model output, and target for six held-out images."""
    import matplotlib
    matplotlib.use("Agg")  # Save files without needing a desktop display.
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    # Each row is the SAME original sample. Columns are different roles.
    fig, axes = plt.subplots(6, 3, figsize=(7, 11), squeeze=False)
    columns = [
        ("Noisy input" if denoise else "Clean input", inputs),
        ("Model reconstruction", reconstruction),
        ("Clean target", targets),
    ]
    for col, (title, values) in enumerate(columns):
        for row in range(6):
            ax = axes[row, col]
            # detach/cpu/numpy prepare plotting data; no learning happens here.
            image = values[row].detach().cpu().reshape(8, 8).numpy()
            ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"held-out {row}")
    fig.suptitle("digits 8x8 — DAE" if denoise else "digits 8x8 — AE")
    fig.tight_layout()
    path = output_dir / ("digits-dae.png" if denoise else "digits-ae.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved figure:", path)


def plot_pair(pair_path: Path, output_dir: Path):
    """Inspect an actual clean/noisy NPZ pair without running a model."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not pair_path.is_file():
        raise FileNotFoundError(f"Pair file not found: {pair_path}")
    # np.load returns an archive-like object; with closes the file after reading.
    # allow_pickle=False permits numerical arrays without loading pickled objects.
    with np.load(pair_path, allow_pickle=False) as pair:
        clean = pair["clean"].copy()
        noisy = pair["noisy"].copy()
    if clean.shape != (1, 28, 28) or noisy.shape != clean.shape:
        raise ValueError("Expected clean and noisy with shape (1, 28, 28).")
    if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ValueError("The arrays contain non-finite values.")
    # The first axis's meaning comes from the data producer, not shape alone.
    print("Pair inspection only; no model has been run.")
    for name, values in [("clean", clean), ("noisy", noisy)]:
        print(name, values.shape, values.dtype, values.min(), values.max())
        print(f"{name}[0, 14, 14] = {values[0, 14, 14]}")
    print("These are the values in the supplied file, not assumed reference values.")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for col, (name, values) in enumerate([("Noisy input", noisy), ("Clean target", clean)]):
        full = axes[0, col]  # Display-area index, not a pixel-array index.
        full.imshow(values[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest", origin="upper")
        full.set_title(f"{name}: full 28x28")
        # The 7x7 region has centers at original row/column indices 11 through 17.
        full.add_patch(Rectangle((10.5, 10.5), 7, 7, fill=False, edgecolor="#ff7043", linewidth=2))
        zoom = axes[1, col]
        patch = values[0, 11:18, 11:18]  # End index 18 excluded; shape is (7, 7).
        zoom.imshow(
            patch, cmap="gray", vmin=0, vmax=1, interpolation="nearest",
            origin="upper", extent=(10.5, 17.5, 17.5, 10.5),
        )
        # extent is (left, right, bottom, top), preserving original pixel centers.
        zoom.set_xticks(range(11, 18))
        zoom.set_yticks(range(11, 18))
        zoom.set_title(f"{name}: 7x7 crop")
        # Center (column 14, row 14) is bounded by 13.5 and 14.5 on each axis.
        zoom.add_patch(Rectangle((13.5, 13.5), 1, 1, fill=False, edgecolor="#ff7043", linewidth=2))
        for ax in (full, zoom):
            ax.set_xlabel("column index")
            ax.set_ylabel("row index")
    fig.suptitle("Input/target inspection — not a model prediction")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pair-inspection.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved figure:", path)


def main():
    parser = argparse.ArgumentParser(description="AE/DAE numerical and training examples.")
    # choices rejects misspelled modes; default runs no training loop.
    parser.add_argument("--mode", choices=["tiny", "ae", "dae"], default="tiny", help="Example to run.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for ae/dae.")
    parser.add_argument("--pair", type=Path, help="Optional existing clean/noisy MNIST NPZ file.")
    parser.add_argument("--output-dir", type=Path, help="Directory for optional figures.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.pair is not None and not args.pair.is_file():
        parser.error(f"Pair file not found: {args.pair}")

    print("PyTorch:", torch.__version__)
    if args.mode == "tiny":
        run_tiny()
    else:
        run_digits(denoise=args.mode == "dae", epochs=args.epochs, output_dir=args.output_dir)
    if args.pair is not None:
        plot_pair(args.pair, args.output_dir or Path("ae-figures"))


# Importing this file defines functions; executing it runs main.
if __name__ == "__main__":
    main()
