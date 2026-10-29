import torch
from torch import nn
import torch.nn.functional as F

# Make inputs and initial weights easier to reproduce with the same run order.
# All models and tensors in this example use the default CPU device.
torch.manual_seed(16)


class MyProjectionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # Initialize nn.Module's management of child layers and parameters.
        super().__init__()

        # Main path 1: change channels and use the supplied spatial stride.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,  # Use one bias for each output channel.
        )

        # Main path 2: preserve the first convolution's channels and dimensions.
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # Preserve channels and spatial size for this main path's K/P settings.
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            # Process the same original input to match the main path's final shape.
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=True,
            )

    def forward(self, x):
        # F.relu replaces negative values with zero and preserves shape.
        main = F.relu(self.conv1(x))
        main = self.conv2(main)

        # Pass the original input x, not the output of conv1.
        skip = self.shortcut(x)

        # Add matching positions first, then apply ReLU to their sums.
        return F.relu(main + skip)


# Each item: input channels, output channels, stride, input shape, expected shape.
cases = [
    (8, 16, 2, (2, 8, 16, 16), (2, 16, 8, 8)),
    (8, 8, 1, (1, 8, 15, 13), (1, 8, 15, 13)),
    (8, 16, 1, (1, 8, 15, 13), (1, 16, 15, 13)),
    (8, 8, 2, (1, 8, 15, 13), (1, 8, 8, 7)),
    (16, 8, 1, (1, 16, 15, 13), (1, 8, 15, 13)),
]

for in_c, out_c, stride, input_shape, expected_shape in cases:
    block = MyProjectionResidualBlock(in_c, out_c, stride)
    x = torch.randn(input_shape)

    with torch.no_grad():
        # Inspect the paths separately, before adding their results.
        main = block.conv2(F.relu(block.conv1(x)))
        skip = block.shortcut(x)
        assert main.shape == skip.shape, "The two paths have different shapes."

        # block(x) runs forward through PyTorch's module-calling procedure.
        y = block(x)
        assert tuple(y.shape) == expected_shape

        # allclose compares values while allowing small floating-point errors.
        # atol is absolute tolerance; rtol scales with the reference value.
        assert torch.allclose(y, F.relu(main + skip), atol=1e-6, rtol=1e-5)

    expected_identity = in_c == out_c and stride == 1
    assert isinstance(block.shortcut, nn.Identity) == expected_identity

    if expected_identity:
        # Identity returns the same input object, not a copy.
        assert skip is x
        shortcut_name = "Identity"
    else:
        # Weight axes: output filters, input channels, kernel height, kernel width.
        assert tuple(block.shortcut.weight.shape) == (out_c, in_c, 1, 1)
        shortcut_name = "1x1 Conv"

    print(f"{input_shape} -> {tuple(y.shape)} | {shortcut_name}")


# Verify one 1×1 Conv output position using the numbers from the table.
# zeros creates a tensor filled with zeros in the specified shape.
point_x = torch.zeros(1, 8, 16, 16)
projection = nn.Conv2d(8, 16, kernel_size=1, stride=2, padding=0)

with torch.no_grad():
    # : selects all eight input channels.
    # torch.tensor creates a tensor from the supplied numbers.
    point_x[0, :, 4, 6] = torch.tensor([1., 2., 0., -1., 3., 1., 2., -2.])
    projection.weight[0, :, 0, 0] = torch.tensor(
        [0.5, -1., 2., 1., 0., 0.25, -0.5, 1.]
    )
    projection.bias[0] = 0.5

    point_y = projection(point_x)
    # With stride=2, output position (2,3) reads input position (4,6).
    actual = point_y[0, 0, 2, 3]
    expected = torch.tensor(-4.75)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

# item extracts a regular Python number from a one-element tensor.
print("One 1x1 Conv output value:", actual.item())


# Equal strides can produce different spatial sizes when K/P differ.
size_x = torch.zeros(1, 8, 16, 16)
settings = [
    (3, 1, 2, (8, 8)),
    (1, 0, 2, (8, 8)),
    (1, 1, 2, (9, 9)),
    (3, 0, 2, (7, 7)),
]

with torch.no_grad():
    for kernel, padding, stride, expected_hw in settings:
        conv = nn.Conv2d(8, 16, kernel, stride=stride, padding=padding)
        # shape[2:] selects axes 2 and 3: output height and width.
        actual_hw = tuple(conv(size_x).shape[2:])
        assert actual_hw == expected_hw
        print(f"K={kernel}, P={padding}, S={stride} -> {actual_hw}")


# None and False are initial values for the readiness state in this example.
projection_block = None
projection_ready = False

projection_block = MyProjectionResidualBlock(8, 16, stride=2)
with torch.no_grad():
    check_y = projection_block(torch.randn(2, 8, 16, 16))

assert tuple(check_y.shape) == (2, 16, 8, 8)
projection_ready = True  # Change the status flag after the check passes.
print("Block structure verified:", projection_ready)
