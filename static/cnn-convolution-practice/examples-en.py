"""Check convolution numerically and optionally visualize an actual MNIST image.

The default run uses only PyTorch and does not download any data.
    python examples-en.py

The MNIST example also requires torchvision and matplotlib.
    python examples-en.py --mnist
    python examples-en.py --mnist --save-figure mnist-laplacian.png

Use --save-figure to save the figure instead of opening a window.
Use --data-root to choose the folder where MNIST is stored.
"""

import argparse  # Read command-line options such as --mnist.
from pathlib import Path  # Handle the path used to save the figure.
import time  # Measure the time spent preparing data and computing the result.

import torch  # Create tensors and perform tensor calculations.
from torch import nn  # Import the convolution layer class.
import torch.nn.functional as F  # Use functional convolution, padding, and ReLU.


# 1. Arrange 20 values as one image, one channel, height 4, and width 5.
X = torch.tensor([
    [1., 2., 0., 1., 3.],
    [0., 3., 1., 2., 1.],
    [2., 1., 0., 1., 0.],
    [1., 0., 2., 3., 2.],
])
x = X.reshape(1, 1, 4, 5)
assert torch.equal(x[0, 0], X)

# Integer indices select image 0 and channel 0; :3 reads rows and columns 0, 1, 2.
patch = x[0, 0, :3, :3]
K = torch.tensor([
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.],
])

# Multiply values at matching positions, then add all nine products.
products = patch * K
row_sums = products.sum(dim=1)  # Sum the columns within each row: [1, -1, 2].
first_output = products.sum()  # Sum all elements into a 0D tensor, tensor(2.).
print("First patch:\n", patch)
print("Elementwise products:\n", products)
print("Row sums:", row_sums.tolist())
print("One output value:", first_output.item())
assert first_output.item() == 2.0

# 2. nn.Conv2d(...) creates a layer; conv(x) runs that layer on the input.
conv = nn.Conv2d(1, 1, 3, bias=False)
with torch.no_grad():  # Do not record this initialization copy for autograd.
    conv.weight.copy_(K.reshape(1, 1, 3, 3))

out = conv(x)
expected = torch.tensor([[2., 2., -3.], [0., -2., 0.]])
assert tuple(out.shape) == (1, 1, 2, 3)
assert torch.equal(out[0, 0], expected)
assert out[0, 0, 0, 0].item() == first_output.item()
assert conv.weight.requires_grad is True  # Copying in no_grad() does not permanently freeze it.
print("Convolution output:\n", out[0, 0].detach())

# Apply ReLU separately after convolution. Replace negative values with 0; keep the shape.
relu_out = F.relu(out)
assert torch.equal(relu_out[0, 0], torch.tensor([[2., 2., 0.], [0., 0., 0.]]))
assert relu_out.shape == out.shape

# 3. Change only stride and padding while keeping the same input and weights.
# These dictionary values are expected calculation results, not dataset labels.
expected_maps = {
    (1, 0): [[2., 2., -3.], [0., -2., 0.]],
    (2, 0): [[2., -3.]],
    (1, 1): [
        [-5., 0., 2., -3., 3.],
        [-6., 2., 2., -3., 4.],
        [-4., 0., -2., 0., 6.],
        [-1., 1., -3., 0., 4.],
    ],
    (2, 1): [[-5., 2., 3.], [-4., -2., 6.]],
}
H, W = x.shape[-2:]  # Read height and width, the last two entries in shape.
for (stride, padding), expected_map in expected_maps.items():
    actual = F.conv2d(x, conv.weight, stride=stride, padding=padding)
    # Coordinates refer to the padded input; range excludes its stop value.
    starts_h = list(range(0, H + 2 * padding - 3 + 1, stride))
    starts_w = list(range(0, W + 2 * padding - 3 + 1, stride))
    assert tuple(actual.shape[-2:]) == (len(starts_h), len(starts_w))
    assert torch.equal(actual[0, 0], torch.tensor(expected_map))
    print(f"stride={stride}, padding={padding}, start rows={starts_h}, start columns={starts_w}")
    print("Output shape:", tuple(actual.shape))
    print(actual[0, 0].detach())

# 4. Add one cell of padding on each side to see why the first output becomes -5.
padded_x = F.pad(x, (1, 1, 1, 1))  # Order: left, right, top, bottom.
padded_patch = padded_x[0, 0, :3, :3]
padded_products = padded_patch * K
pad_out = F.conv2d(x, conv.weight, padding=1)
assert tuple(padded_x.shape) == (1, 1, 6, 7)
assert padded_products.sum().item() == -5.0
assert pad_out[0, 0, 0, 0].item() == -5.0
assert torch.equal(padded_x[0, 0, 1:4, 1:4], patch)
assert pad_out[0, 0, 1, 1].item() == 2.0
# Do not add padding again when the input is already padded.
assert torch.equal(F.conv2d(padded_x, conv.weight), pad_out)
print("First padded patch:\n", padded_patch)
print("First padded output:", padded_products.sum().item())

# Length 6, kernel length 3, stride 2: the last valid start M is 3; selected starts are [0, 2].
L, p, k, s = 4, 1, 3, 2
last_allowed = L + 2 * p - k
starts = list(range(0, last_allowed + 1, s))
output_length = last_allowed // s + 1
assert starts == [0, 2]
assert len(starts) == output_length == 2

# 5. This fixed Laplacian filter adds the four direct neighbors and subtracts 4 times the center.
laplace = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
laplace_x = torch.tensor([
    [1., 2., 3., 0., 1.],
    [0., 1., 2., 3., 1.],
    [3., 0., 1., 2., 0.],
    [2., 3., 0., 1., 2.],
    [1., 2., 3., 0., 1.],
]).reshape(1, 1, 5, 5)
laplace_conv = nn.Conv2d(1, 1, 3, bias=False)
with torch.no_grad():
    laplace_conv.weight.copy_(laplace.reshape(1, 1, 3, 3))
laplace_out = laplace_conv(laplace_x)
assert torch.equal(
    laplace_out[0, 0],
    torch.tensor([[0., 0., -7.], [8., 0., -3.], [-8., 8., 0.]]),
)
print("Laplacian output:\n", laplace_out[0, 0].detach())

# New layers receive new weights. The next two checks compare output shapes only.
conv_pad = nn.Conv2d(1, 1, 3, padding=1, bias=False)
conv_stride = nn.Conv2d(1, 1, 3, stride=2, bias=False)
assert tuple(conv_pad(laplace_x).shape) == (1, 1, 5, 5)
assert tuple(conv_stride(laplace_x).shape) == (1, 1, 2, 2)
# Pass the existing weights explicitly when comparing values from the same filter.
same_filter_padded = F.conv2d(laplace_x, laplace_conv.weight, padding=1)
same_filter_strided = F.conv2d(laplace_x, laplace_conv.weight, stride=2)

# 6. Compute gradients in a separate layer. Do not perform an optimizer update.
grad_conv = nn.Conv2d(1, 1, 3, bias=False)
with torch.no_grad():
    grad_conv.weight.copy_(conv.weight)
weight_before = grad_conv.weight.detach().clone()  # An independent copy for comparison.
grad_out = grad_conv(x)
grad_out.retain_grad()  # Also keep the intermediate output's gradient in .grad.
grad_loss = grad_out.square().mean()  # Demonstration objective: the mean of squared outputs.
grad_loss.backward()  # Compute gradients without changing the weight values.
assert tuple(grad_out.grad.shape) == (1, 1, 2, 3)
assert tuple(grad_conv.weight.grad.shape) == (1, 1, 3, 3)
assert torch.equal(grad_conv.weight, weight_before)
# For the mean of six squared outputs, each output gradient is 2 * output / 6.
torch.testing.assert_close(grad_out.grad, 2 * grad_out.detach() / grad_out.numel())
print("Demonstration loss:", grad_loss.item())
print("Output gradient:\n", grad_out.grad[0, 0])
print("Weight gradient:\n", grad_conv.weight.grad[0, 0])

# 7. detach removes the autograd connection from its result but shares the data storage.
result = laplace_conv(laplace_x)
detached_result = result.detach()
assert torch.equal(result, detached_result)
assert result.requires_grad is True
assert detached_result.requires_grad is False
assert result.data_ptr() == detached_result.data_ptr()
# Use detach().clone() when an independent copy of the data is also needed.
copied_result = result.detach().clone()
assert copied_result.data_ptr() != result.data_ptr()
with torch.no_grad():  # Do not record the calculation in this block for autograd.
    result_without_graph = laplace_conv(laplace_x)
assert result_without_graph.requires_grad is False
assert laplace_conv.weight.requires_grad is True
print("All checks in the core example passed.")


def run_mnist(data_root, figure_path):
    """Apply the same Laplacian filter to the first MNIST image and display or save it."""
    # Import these only when MNIST is requested; the core checks do not need them.
    from torchvision import datasets, transforms
    if figure_path is not None:
        import matplotlib
        matplotlib.use("Agg")  # Save a file even in an environment without a display.
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    mnist = datasets.MNIST(
        data_root,
        train=True,  # Select the training split. This does not train a model.
        download=True,  # Reuse existing files or download them if necessary.
        transform=transforms.ToTensor(),  # Convert 8-bit brightness to 0–1 when retrieving an image.
    )
    img, label = mnist[0]  # Receive the first image and its digit label separately.
    batch = img.unsqueeze(0)  # Add a batch axis: (1,28,28) becomes (1,1,28,28).
    result = laplace_conv(batch)  # Compute using the existing Laplacian weights.
    edge = result.detach()[0, 0]  # Detach from autograd and select the 26×26 output.
    elapsed = time.perf_counter() - started

    assert tuple(img.shape) == (1, 28, 28)
    assert tuple(batch.shape) == (1, 1, 28, 28)
    assert tuple(edge.shape) == (26, 26)
    # Use absolute values only to choose the display range. Plot the signed edge values.
    limit = max(float(edge.abs().max()), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    axes[0].imshow(img[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[0].set_title(f"Original MNIST image (label={label})")
    im = axes[1].imshow(edge.cpu().numpy(), cmap="coolwarm", vmin=-limit,
                        vmax=limit, interpolation="nearest")
    axes[1].set_title("Fixed Laplacian response")
    fig.colorbar(im, ax=axes[1], shrink=0.9)
    for ax in axes:
        ax.axis("off")

    print("MNIST count:", len(mnist), "Label:", label)
    print("MNIST input/output shapes:", tuple(img.shape), tuple(edge.shape))
    print("Output range:", float(edge.min()), float(edge.max()), "Display range:", (-limit, limit))
    print("Data preparation and computation time (seconds):", round(elapsed, 3))
    if figure_path is None:
        plt.show()
    else:
        destination = Path(figure_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180, facecolor="white")
        plt.close(fig)
        print("Figure saved:", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist", action="store_true", help="Download and visualize MNIST")
    parser.add_argument("--data-root", default="./data", help="Folder for storing and reading MNIST")
    parser.add_argument("--save-figure", help="Save the figure to this path instead of displaying it")
    args = parser.parse_args()
    if args.save_figure and not args.mnist:
        parser.error("Use --save-figure together with --mnist.")
    if args.mnist:
        run_mnist(args.data_root, args.save_figure)
