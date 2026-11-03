"""Four-channel denoising autoencoder: implementation, one training step, two comparisons.

This independent implementation makes the workflow explained in the article executable.
It does not reproduce the unavailable original fit/predict_fixed/make_val_noisy helpers.
Requires torch. Only the optional image-saving function also requires matplotlib.
Uses CPU float32 tensors of shape (N, 1, 28, 28), with pixels in [0, 1].

Check shapes and calculations without a dataset:
    python examples-en.py --self-check

Prepare actual data, then run:
    torch.save({"clean_train": clean_train, "clean": clean_val}, "data.pt")
    python examples-en.py --data data.pt --protocol both

Prepare clean_train and clean as non-overlapping training and validation images.
The dictionary key names alone do not prove that the datasets are separate.
Three epochs, batch_size=128, and lr=0.01 are example settings, not guaranteed optima.
No performance values are preset. This does not claim to reproduce the reported source results.
"""

import argparse
import math
from pathlib import Path

import torch
from torch import nn


def check_images(images, name="images"):
    """Validate the image shape, dtype, device, and range required by this CPU example."""
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name}: torch.Tensor required")
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError(f"{name}: expected (N, 1, 28, 28), got {tuple(images.shape)}")
    if images.shape[0] == 0:
        raise ValueError(f"{name}: empty batch")
    if images.device.type != "cpu" or images.dtype != torch.float32:
        raise ValueError(f"{name}: CPU float32 required; use images.detach().cpu().float()")
    if not torch.isfinite(images).all().item():
        raise ValueError(f"{name}: NaN or infinity")
    if images.min().item() < 0 or images.max().item() > 1:
        raise ValueError(f"{name}: pixel values must lie in [0, 1]")


def check_pair(reconstruction, clean):
    """Reject mismatched shapes and empty image batches before per-image comparisons."""
    if reconstruction.shape != clean.shape:
        raise ValueError("reconstruction and clean must have identical shapes")
    if reconstruction.ndim != 4 or any(size == 0 for size in reconstruction.shape):
        raise ValueError("nonempty (N, C, H, W) tensors required")


def check_sigma(sigma):
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")


class StudentDAE(nn.Module):
    """(N,1,28,28) → (N,4,14,14) → (N,1,28,28)."""

    def __init__(self):
        super().__init__()
        # Constructors receive layer settings; actual images are passed later.
        self.encoder = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1, bias=True)
        self.decoder = nn.ConvTranspose2d(
            4, 1, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.relu = nn.ReLU()  # Replace negative values with zero, without changing shape.
        self.sigmoid = nn.Sigmoid()  # Produce outputs to compare with pixel targets in [0, 1].

    def forward(self, noisy):
        # noisy is the actual input passed when the model is called.
        z = self.relu(self.encoder(noisy))
        return self.sigmoid(self.decoder(z))


def require_optimizer_matches(model, optimizer):
    """Prevent accidentally reusing an optimizer connected to a different model."""
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if model_ids != optimizer_ids:
        raise ValueError("optimizer must belong to this model's trainable parameters")


def train_one_batch(model, optimizer, noisy, clean):
    """Update parameters once and return the loss measured before that update."""
    model.train()  # Set training mode; this call alone does not train the model.
    optimizer.zero_grad(set_to_none=True)  # Clear previous gradients, not the learned weights.
    pred = model(noisy)
    check_pair(pred, clean)  # Prevent unintended broadcasting.
    loss = nn.MSELoss()(pred, clean)  # Mean squared error over every image, channel, and pixel.
    loss.backward()  # Compute .grad for each trainable parameter.
    optimizer.step()  # Update parameters using the computed gradients.
    return loss.item()  # The existing loss is not automatically recomputed after step.


def per_image_mse(reconstruction, clean):
    """Calculate one error per image with a loop. Return shape: (N,)."""
    check_pair(reconstruction, clean)
    losses = []  # A Python list does not have the tensor.shape attribute.
    for recon_img, clean_img in zip(reconstruction, clean):
        loss = (recon_img.flatten() - clean_img.flatten()).square().mean()
        losses.append(loss)  # Each loss is a scalar tensor with no axes: shape ().
    return torch.stack(losses)  # Stack N scalar tensors along a new axis: (N,).


def per_image_mse_fast(reconstruction, clean):
    """Apply the same calculation to the whole batch while retaining image axis 0."""
    check_pair(reconstruction, clean)
    squared_error = (reconstruction - clean).square()  # (N,C,H,W).
    pixels_per_image = squared_error.flatten(start_dim=1)  # (N,C*H*W).
    return pixels_per_image.mean(dim=1)  # One mean for each image: (N,).


def add_noise(clean, sigma, generator):
    """Scale standard normal noise by sigma, then constrain the pixel range."""
    check_sigma(sigma)
    epsilon = torch.randn(clean.shape, dtype=clean.dtype, generator=generator)
    # After clipping, the actual (noisy-clean) distribution is no longer the original Gaussian.
    return (clean + sigma * epsilon).clamp(0.0, 1.0)


def make_val_noisy(clean, sigma, seed=9026):
    """The same images and seed reuse the same epsilon when creating validation inputs."""
    check_images(clean, "clean")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return add_noise(clean, sigma, generator)


def fit(model, optimizer, clean_train, *, sigma, epochs=3, batch_size=128,
        seed=2026, verbose=True):
    """Train the supplied model in place. History contains batch-size-weighted epoch means."""
    check_images(clean_train, "clean_train")
    check_sigma(sigma)
    require_optimizer_matches(model, optimizer)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    # Keep model initialization, example order, and noise random generators separate.
    order_rng = torch.Generator(device="cpu").manual_seed(seed)
    noise_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    history = []
    n_images = len(clean_train)

    for epoch in range(1, epochs + 1):
        order = torch.randperm(n_images, generator=order_rng)
        weighted_loss = 0.0
        for start in range(0, n_images, batch_size):
            clean = clean_train[order[start:start + batch_size]]
            noisy = add_noise(clean, sigma, noise_rng)
            loss_before_update = train_one_batch(model, optimizer, noisy, clean)
            weighted_loss += loss_before_update * len(clean)
        # Give each image equal weight even when the last batch is smaller.
        epoch_loss = weighted_loss / n_images
        history.append(epoch_loss)
        if verbose:  # False hides these logs; all training above still runs.
            print(f"epoch={epoch}, train_mse={epoch_loss:.6f}")
    # This averages losses from different update times, not validation loss of the final model.
    return history


def predict_fixed(model, noisy, batch_size=256):
    """Evaluate the supplied inputs unchanged. Do not resample noise or train."""
    check_images(noisy, "noisy")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()  # Select evaluation behavior for layers that distinguish modes.
    try:
        with torch.no_grad():  # Do not record a graph for backpropagation.
            parts = [model(batch) for batch in noisy.split(batch_size, dim=0)]
            return torch.cat(parts, dim=0)  # Preserve the original image order.
    finally:
        model.train(was_training)  # Restore the mode that was active before this call.


def evaluate(model, noisy, clean):
    """Compare input and reconstruction errors against the same clean images."""
    check_images(clean, "clean")
    check_pair(noisy, clean)
    recon = predict_fixed(model, noisy)
    base_each = per_image_mse_fast(noisy, clean)
    rest_each = per_image_mse_fast(recon, clean)
    delta = base_each - rest_each
    base = base_each.mean().item()
    rest = rest_each.mean().item()
    ratio = rest / base if base > 0 else float("nan")
    return {
        "base_mse": base, "rest_mse": rest, "remaining_ratio": ratio,
        "reduction_ratio": 1 - ratio,
        "improved": int((delta > 0).sum().item()),
        "worsened": int((delta < 0).sum().item()),
        "tied": int((delta == 0).sum().item()),
        "count": len(clean), "base_each": base_each, "rest_each": rest_each,
        "recon": recon,
    }


def evaluate_sigmas(model, clean_val, sigmas=(0.25, 0.50), *, val_seed=9026):
    """Experiment A: retain one trained model and change only evaluation noise strength."""
    rows = []
    for sigma in sigmas:
        noisy = make_val_noisy(clean_val, sigma, seed=val_seed)
        rows.append({"sigma": sigma, **evaluate(model, noisy, clean_val)})
    return rows


def retrain_sigmas(clean_train, clean_val, sigmas=(0.25, 0.50), *, seed=2026,
                   val_seed=9026, epochs=3, batch_size=128, verbose=False):
    """Experiment B: train a fresh model with a fresh optimizer for each noise strength."""
    rows = []
    for sigma in sigmas:
        torch.manual_seed(seed)  # Start with identical initial weights in the same environment.
        trial = StudentDAE().cpu()  # Create a new model, rather than load trained weights.
        trial_optimizer = torch.optim.Adam(trial.parameters(), lr=0.01)
        history = fit(trial, trial_optimizer, clean_train, sigma=sigma,
                      epochs=epochs, batch_size=batch_size, seed=seed, verbose=verbose)
        noisy = make_val_noisy(clean_val, sigma, seed=val_seed)
        rows.append({"sigma": sigma, "model": trial, "history": history,
                     **evaluate(trial, noisy, clean_val)})
    return rows


def save_triplet(clean, noisy, recon, index, out_dir):
    """Save clean/noisy/reconstructed images at the same index and return the saved path."""
    import matplotlib.pyplot as plt  # Required only when saving a figure.

    check_pair(noisy, clean)
    check_pair(recon, clean)
    if not 0 <= index < len(clean):
        raise IndexError("image index outside the batch")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, images, title in zip(axes, (clean, noisy, recon), ("Clean", "Noisy", "Recon")):
        ax.imshow(images[index, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)  # Use the same brightness range for all three images.
        ax.axis("off")
    fig.suptitle(f"Image index {index}")
    fig.tight_layout()
    path = out_dir / f"image_{index}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def print_rows(label, rows):
    print(label)
    for row in rows:
        print(
            f"sigma={row['sigma']:.2f}, input_mse={row['base_mse']:.6f}, "
            f"recon_mse={row['rest_mse']:.6f}, remaining={row['remaining_ratio']:.1%}, "
            f"reduction={row['reduction_ratio']:.1%}, "
            f"improved/worsened/tied={row['improved']}/{row['worsened']}/{row['tied']}"
        )


def self_check():
    """Check shapes and calculations with synthetic tensors, not handwriting performance."""
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = StudentDAE().cpu()
    clean = torch.rand(3, 1, 28, 28)
    noisy = make_val_noisy(clean, 0.25)
    assert tuple(model.encoder.weight.shape) == (4, 1, 3, 3)
    assert tuple(model.decoder.weight.shape) == (4, 1, 4, 4)
    assert sum(p.numel() for p in model.parameters()) == 105
    with torch.no_grad():
        assert tuple(model.encoder(clean).shape) == (3, 4, 14, 14)
        assert tuple(model(clean).shape) == (3, 1, 28, 28)
        assert tuple(model(torch.zeros(2, 1, 27, 27)).shape) == (2, 1, 28, 28)

    each = per_image_mse(noisy, clean)
    assert tuple(each.shape) == (3,)
    assert torch.allclose(each, per_image_mse_fast(noisy, clean))
    assert torch.allclose(each.mean(), nn.MSELoss()(noisy, clean))
    scalars = [torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3)]
    assert scalars[0].shape == torch.Size([])
    assert tuple(torch.stack(scalars).shape) == (3,)
    assert tuple(torch.stack(scalars).unsqueeze(1).shape) == (3, 1)
    a, b = torch.tensor([10, 20, 30]), torch.tensor([40, 50, 60])
    assert torch.stack([a, b], dim=0).tolist() == [[10, 20, 30], [40, 50, 60]]
    assert torch.stack([a, b], dim=1).tolist() == [[10, 40], [20, 50], [30, 60]]
    assert tuple(a.unsqueeze(0).shape) == (1, 3) and tuple(a.unsqueeze(1).shape) == (3, 1)
    assert tuple(a.shape) == (3,)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    require_optimizer_matches(model, optimizer)
    before_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        loss_before = nn.MSELoss()(model(noisy), clean).item()
    returned_loss = train_one_batch(model, optimizer, noisy, clean)
    with torch.no_grad():
        loss_after = nn.MSELoss()(model(noisy), clean).item()
    assert math.isclose(returned_loss, loss_before, rel_tol=1e-7)
    assert not math.isclose(returned_loss, loss_after, rel_tol=1e-7)
    assert any(not torch.equal(before_parameters[name], p) for name, p in model.named_parameters())

    saved_state = {name: p.detach().clone() for name, p in model.state_dict().items()}
    model.train()
    prediction = predict_fixed(model, noisy, batch_size=2)
    assert model.training and not prediction.requires_grad
    assert all(torch.equal(saved_state[name], p) for name, p in model.state_dict().items())
    rows_a = evaluate_sigmas(model, clean)
    assert len(rows_a) == 2
    assert all(torch.equal(saved_state[name], p) for name, p in model.state_dict().items())
    assert torch.equal(noisy, make_val_noisy(clean, 0.25))
    middle = torch.full((2, 1, 28, 28), 0.5)
    eps = torch.randn(middle.shape, generator=torch.Generator().manual_seed(77))
    for sigma in (0.25, 0.50):
        assert torch.equal(make_val_noisy(middle, sigma, 77), (middle + sigma * eps).clamp(0, 1))

    other_model = StudentDAE().cpu()
    try:
        fit(other_model, optimizer, clean, sigma=0.25, epochs=1, verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("optimizer for a different model was accepted")
    rows_b = retrain_sigmas(clean, middle, sigmas=(0.25, 0.25), epochs=1, batch_size=2)
    assert rows_b[0]["model"] is not rows_b[1]["model"]
    assert rows_b[0]["model"] is not model
    assert all(torch.equal(a, b) for a, b in zip(rows_b[0]["model"].parameters(), rows_b[1]["model"].parameters()))
    assert len(rows_b[0]["history"]) == 1
    print(f"SELF-CHECK PASS (synthetic tensors only; torch {torch.__version__})")
    print(f"parameters=105, scalar=(), per_image=(3,), unsqueeze=(3,1)")
    print(f"loss_before={loss_before:.8f}, returned={returned_loss:.8f}, loss_after={loss_after:.8f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--protocol", choices=("evaluate", "retrain", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.data is None:
        parser.error("use --self-check, or provide --data data.pt")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    # Read a tensor dictionary; do not import unavailable source helpers.
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    clean_train = data["clean_train"].detach().cpu().float()
    clean_val = data["clean"].detach().cpu().float()
    check_images(clean_train, "clean_train")
    check_images(clean_val, "clean_val")
    seed = 2026

    if args.protocol in ("evaluate", "both"):
        # For standalone execution, first train one model at sigma=0.25.
        print("TRAIN ONE MODEL for experiment A (sigma=0.25)")
        torch.manual_seed(seed)
        model = StudentDAE().cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        fit(model, optimizer, clean_train, sigma=0.25, epochs=args.epochs,
            batch_size=args.batch_size, seed=seed)
        print_rows("A: SAME MODEL, CHANGE EVALUATION NOISE ONLY", evaluate_sigmas(model, clean_val))

    if args.protocol in ("retrain", "both"):
        # This branch starts training again for each of the two conditions.
        rows = retrain_sigmas(clean_train, clean_val, seed=seed, epochs=args.epochs,
                             batch_size=args.batch_size, verbose=True)
        print_rows("B: NEW MODEL AND TRAINING FOR EACH NOISE LEVEL", rows)


if __name__ == "__main__":
    main()
