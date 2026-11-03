"""Denoising autoencoder: a complete example using the prepared exercise data.

Required external packages: torch, matplotlib.
This file does not include the original images or the course helper d2_tools.py.
Reconstruction metrics are computed from the supplied data, not hard-coded.

Import these functions in your notebook, then run:
    result = run_experiment(data, seed=SEED)
    plot_history(result["history"])
    plot_triplets(result["clean_val"], result["noisy_val"], result["recon"])

Each call to run_experiment creates a new model and optimizer.
Reading this file or defining its functions does not train a model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def check_images(images, name):
    """Check the shape and value range of (N, 1, 28, 28) images for the CPU exercise."""
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor.")
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError(
            f"{name}: expected shape (N, 1, 28, 28), "
            f"got {tuple(images.shape)}"
        )
    if images.shape[0] == 0:
        raise ValueError(f"{name}: no images were provided.")
    if not torch.isfinite(images).all().item():
        raise ValueError(f"{name}: contains NaN or infinite values.")
    if images.min().item() < 0 or images.max().item() > 1:
        raise ValueError(f"{name}: pixel values must be in the range 0 to 1.")


def require_same_shape(prediction, target):
    """Prevent unintended broadcasting in the MSE calculation."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {tuple(prediction.shape)}, "
            f"target {tuple(target.shape)}"
        )


class LabDenoiser(nn.Module):
    """Map one grayscale channel to multiple feature channels, then reconstruct it."""

    def __init__(self, feature_channels=8):
        super().__init__()
        if not isinstance(feature_channels, int) or feature_channels <= 0:
            raise ValueError("feature_channels must be a positive integer.")

        # Pass layer settings to the constructor; supply images when calling the layer.
        self.encoder = nn.Sequential(
            # One input channel, feature_channels output channels, and a 3-by-3 kernel.
            # stride=2: move the kernel by 2. padding=1: add one border position.
            nn.Conv2d(1, feature_channels, 3, stride=2, padding=1),
            nn.ReLU(),  # Replace negative values with 0; add no dimensions or parameters.
        )
        self.decoder = nn.Sequential(
            # Transposed convolution: compute a 28-by-28 output from 14-by-14 features.
            nn.ConvTranspose2d(feature_channels, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),  # Produce output values for comparison with targets in the range 0 to 1.
        )

    def forward(self, x):
        # x: (N, 1, 28, 28), features: (N, feature_channels, 14, 14).
        features = self.encoder(x)
        return self.decoder(features)  # Returned shape: (N, 1, 28, 28).


def parameter_counts(model):
    """Count scalar parameter elements per module, not parameter tensors."""
    counts = {}
    for name, part in [("encoder", model.encoder), ("decoder", model.decoder)]:
        counts[name] = sum(parameter.numel() for parameter in part.parameters())
    counts["total"] = sum(counts.values())
    return counts


def image_mse(prediction, target):
    """Return a tensor of shape (N,) containing one mean squared error per image."""
    require_same_shape(prediction, target)
    # Subtract matching positions and square the results; preserve the shape.
    pixel_errors = (prediction - target).square()
    # Keep image axis 0, flatten from axis 1, then average within each image.
    return pixel_errors.flatten(start_dim=1).mean(dim=1)


def train_step(net, optimizer, noisy_images, clean_images):
    """Compute one batch loss and its gradients, then update the parameters once."""
    net.train()  # Set training mode; this line does not update the weights.
    optimizer.zero_grad(set_to_none=True)  # Clear the previous gradients.
    reconstructed = net(noisy_images)
    require_same_shape(reconstructed, clean_images)
    error = nn.MSELoss()(reconstructed, clean_images)
    if not torch.isfinite(error).item():
        raise ValueError("The training loss contains NaN or an infinite value.")
    error.backward()  # Compute gradients for the parameters.
    optimizer.step()  # Update parameters using gradients and optimizer state.
    # This loss was computed before the update. item() returns a Python number.
    return error.detach().item()


def predict_fixed(model, noisy, batch_size=256):
    """Reconstruct input chunks in order and return the full tensor in that order."""
    if noisy.shape[0] == 0:
        raise ValueError("No images were provided for reconstruction.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    model.eval()  # Evaluation mode and autograd recording are separate settings.
    with torch.no_grad():
        results = []
        for part in noisy.split(batch_size, dim=0):
            results.append(model(part))
        # Join along the existing image axis; do not add or average pixel values.
        return torch.cat(results, dim=0)


def run_experiment(data, seed, feature_channels=8):
    """Train a new model for 3 epochs on the exercise data and return its evaluation.

    Data requirements:
      clean_train: clean training images in [0, 1]; use the first 5,000.
      clean:       clean validation images in [0, 1].
      noisy:       fixed noisy images matching the originals and order in clean.
      All image tensors must have shape (N, 1, 28, 28).

    Shape checks cannot verify matching original IDs or train/validation separation.
    Verify these conditions separately when preparing the data.
    """
    for key in ("clean_train", "clean", "noisy"):
        if key not in data:
            raise KeyError(f"data is missing a required key: {key!r}")
        if not isinstance(data[key], torch.Tensor):
            raise TypeError(f"data[{key!r}] must be a torch.Tensor.")

    # Converting to float32 does not normalize values from 0–255 to 0–1.
    # detach keeps exercise inputs disconnected from other computation graphs.
    clean_train = data["clean_train"][:5000].detach().to("cpu", torch.float32)
    clean_val = data["clean"].detach().to("cpu", torch.float32)
    noisy_val = data["noisy"].detach().to("cpu", torch.float32)
    for images, name in [
        (clean_train, "training targets"),
        (clean_val, "validation targets"),
        (noisy_val, "validation inputs"),
    ]:
        check_images(images, name)
    if len(clean_train) != 5000:
        raise ValueError("This exercise requires at least 5,000 training images.")
    require_same_shape(noisy_val, clean_val)

    # Set the CPU thread count, not the number of images or channels.
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model = LabDenoiser(feature_channels=feature_channels).cpu()
    # Create an optimizer for the new model parameters, with fresh Adam state.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    shuffle = torch.Generator().manual_seed(seed + 4000)
    noise_rng = torch.Generator().manual_seed(seed + 5000)
    counts = parameter_counts(model)

    initial_recon = predict_fixed(model, noisy_val)
    require_same_shape(initial_recon, clean_val)
    initial_mse = nn.MSELoss()(initial_recon, clean_val).item()
    # None means no training MSE has been recorded yet; it does not mean zero error.
    history = [[0, None, initial_mse]]
    print("Scalar parameter counts:", counts)

    for epoch in range(1, 4):  # The values are 1, 2, 3; the endpoint 4 is excluded.
        sse, count = 0.0, 0
        shuffled_ids = torch.randperm(len(clean_train), generator=shuffle)
        for ids in shuffled_ids.split(64):
            clean = clean_train[ids]
            noise = 0.25 * torch.randn(
                clean.shape,
                generator=noise_rng,
                dtype=clean.dtype,
                device=clean.device,
            )
            # After clamp, noisy-clean may differ from the original normal distribution.
            noisy = (clean + noise).clamp(0, 1)
            loss = train_step(model, optimizer, noisy, clean)
            # Weight the recorded loss by element count, including the smaller final batch.
            sse += loss * clean.numel()
            count += clean.numel()

        train_mse = sse / count
        # Unlike train_mse during training, evaluate the end-of-epoch model on fixed inputs.
        recon = predict_fixed(model, noisy_val)
        require_same_shape(recon, clean_val)
        validation_mse = nn.MSELoss()(recon, clean_val).item()
        history.append([epoch, train_mse, validation_mse])
        print(f"epoch {epoch}: train={train_mse:.4f}, validation={validation_mse:.4f}")

    baseline = nn.MSELoss()(noisy_val, clean_val).item()
    restored = nn.MSELoss()(recon, clean_val).item()
    delta = image_mse(noisy_val, clean_val) - image_mse(recon, clean_val)
    if not torch.isfinite(delta).all().item():
        raise ValueError("Per-image errors contain NaN or infinite values.")
    print(f"Input MSE={baseline:.4f}, reconstruction MSE={restored:.4f}")
    print(
        "Improved / worsened / exact ties:",
        (delta > 0).sum().item(),
        (delta < 0).sum().item(),
        (delta == 0).sum().item(),
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "recon": recon,
        "initial_recon": initial_recon,
        "clean_val": clean_val,
        "noisy_val": noisy_val,
        "baseline": baseline,
        "restored": restored,
        "delta": delta,
        "counts": counts,
    }


def plot_history(history):
    """Plot the training-time average separately from end-of-epoch validation MSE."""
    fig, ax = plt.subplots(figsize=(7, 4))
    train_records = [record for record in history if record[1] is not None]
    ax.plot(
        [record[0] for record in train_records],
        [record[1] for record in train_records],
        marker="o",
        label="train MSE (during updates)",
    )
    ax.plot(
        [record[0] for record in history],
        [record[2] for record in history],
        marker="o",
        label="validation MSE (after epoch)",
    )
    ax.set(xlabel="epoch", ylabel="MSE")
    ax.set_xticks([record[0] for record in history])
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_triplets(clean, noisy, recon, indices=None):
    """Show matched clean, noisy, and reconstructed images with a fixed [0, 1] scale."""
    for images, name in [(clean, "original"), (noisy, "input"), (recon, "reconstruction")]:
        check_images(images, name)
    require_same_shape(noisy, clean)
    require_same_shape(recon, clean)
    if indices is None:
        indices = list(range(min(6, clean.shape[0])))
    else:
        indices = list(indices)
    if not indices or any(
        not isinstance(index, int) or not 0 <= index < clean.shape[0]
        for index in indices
    ):
        raise ValueError("indices must contain at least one valid image index.")

    fig, axes = plt.subplots(3, len(indices), squeeze=False, figsize=(2.1 * len(indices), 6))
    for row, (name, images) in enumerate([
        ("clean", clean), ("noisy", noisy), ("reconstructed", recon)
    ]):
        for column, index in enumerate(indices):
            ax = axes[row, column]
            image = images[index, 0].detach().cpu().numpy()
            # Use the same intensity range instead of scaling each image automatically.
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                # This is a zero-based tensor index, not the dataset-specific image ID.
                ax.set_title(f"Index {index}")
            if column == 0:
                ax.set_ylabel(name)
    fig.tight_layout()
    plt.show()
    return fig, axes


def locate_assets():
    """Find the course helper file, or report the absolute paths that were checked."""
    candidates = [Path("assets"), Path("../assets")]
    found = next(
        (
            directory.resolve()
            for directory in candidates
            if (directory / "d2_tools.py").is_file()
        ),
        None,
    )
    if found is None:
        checked = "\n".join(
            str((directory / "d2_tools.py").resolve()) for directory in candidates
        )
        raise FileNotFoundError(
            f"Current working directory: {Path.cwd()}\n"
            f"Could not find d2_tools.py.\nChecked paths:\n{checked}"
        )
    return found


if __name__ == "__main__":
    print(
        "This file uses the data prepared for the course exercise.\n"
        "The original data and d2_tools.py are not included.\n"
        "Import the functions, then run result = run_experiment(data, seed=SEED).\n"
        "Running this file alone does not start training."
    )
