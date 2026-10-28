"""A standalone example for checking connected CNN components.

Run: python examples-en.py
Required package: torch

Use small random tensors on the CPU rather than real photographs.
Check calculations while distinguishing inputs, parameters, and outputs.
Define train_one_batch only to explain the training sequence.
Running this file does not call that function or update any weights.
"""

import torch
from torch import nn
import torch.nn.functional as F


class MyVGGBlock(nn.Module):
    """Apply two spatial-size-preserving Conv layers, then MaxPool."""

    def __init__(self, in_channels, out_channels):
        # Initialize the parent machinery for managing modules and parameters.
        super().__init__()

        # The first layer changes the input channel count to the block output count.
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

        # The second layer receives the first layer output and has separate weights.
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

        # Pool has no learnable weights. It reduces spatial size within each channel.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        result = F.relu(self.conv1(x))
        result = F.relu(self.conv2(result))
        return self.pool(result)


class MySmallVGG(nn.Module):
    """Produce five class scores per image from 32×32 RGB input."""

    def __init__(self):
        super().__init__()
        self.block1 = MyVGGBlock(3, 8)
        self.block2 = MyVGGBlock(8, 16)

        # Feature values after two pools: 16 channels × height 8 × width 8.
        # Do not multiply the per-image feature count by the batch size.
        self.fc = nn.Linear(16 * 8 * 8, 5)

    def forward(self, x):
        result = self.block1(x)              # [N, 8, 16, 16]
        result = self.block2(result)         # [N, 16, 8, 8]
        result = result.flatten(start_dim=1) # [N, 1024]: keep the batch axis.
        return self.fc(result)              # [N, 5]: scores before probabilities.


class ResidualBlock(nn.Module):
    """An educational block adding a main-path result to an equal-shape input."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        # Add values at matching batch, channel, row, and column positions.
        # This is not concatenation or matrix multiplication. x remains the input.
        return F.relu(out + x)


class ProjectionResidualBlock(nn.Module):
    """Match both paths to [N, out_channels, ceil(H/2), ceil(W/2)]."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1
        )

        # 1×1 is the spatial kernel size. Combine values across all input channels.
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1, stride=2
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        shortcut_result = self.shortcut(x)
        return F.relu(out + shortcut_result)


def train_one_batch(model, images, labels, optimizer, criterion):
    """Explain the training sequence. run_checks does not call this function.

    images: [N, 3, 32, 32] inputs on the same device as the model.
    labels: [N] targets on that device; integer class targets use torch.long, 0–4.
    criterion: this example assumes nn.CrossEntropyLoss().
    optimizer: a parameter updater constructed from model.parameters().
    """
    model.train()  # Set modules such as Dropout and BatchNorm to training behavior.

    # Clear gradients accumulated from the preceding batch.
    # Setting a gradient to None differs in detail from storing a zero tensor.
    optimizer.zero_grad(set_to_none=True)

    # Do not wrap the training forward pass in no_grad.
    # Pass logits to CrossEntropyLoss before applying Softmax.
    logits = model(images)
    loss = criterion(logits, labels)

    loss.backward()  # Calculate parameter gradients.
    optimizer.step() # Update parameters using those gradients.
    return loss.item()


def run_checks():
    """Check shapes and values only, without loss, backward, or update steps."""
    device = torch.device("cpu")
    torch.manual_seed(16)

    # A seed helps reproduce random values with the same execution order.
    # It does not guarantee identical floating-point results on every device/version.
    # The standard convolutions below use groups=1 and dilation=1.
    print("PyTorch:", torch.__version__)

    # ------------------------------------------------------------------
    # 1. Four input channels and six filters: calculate one output value by hand.
    # ------------------------------------------------------------------
    conv = nn.Conv2d(
        in_channels=4,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,  # Omit bias here to check multiplication and summation alone.
    ).to(device)
    x = torch.randn(1, 4, 7, 7, device=device)

    assert tuple(conv.weight.shape) == (6, 4, 3, 3)
    assert conv.weight.numel() == 216
    print("All weights:", tuple(conv.weight.shape), conv.weight.numel())
    print("One filter:", tuple(conv.weight[0].shape))
    print("One per-channel kernel:", tuple(conv.weight[0, 0].shape))

    with torch.no_grad():
        y = conv(x)
        actual_value = y[0, 0, 3, 3]

        # Output row 3, column 3 reads input rows/columns 2, 3, and 4.
        # The colon selects all four input channels. End index 5 is excluded.
        patch = x[0, :, 2:5, 2:5]  # [4, 3, 3]: input data.
        kernels = conv.weight[0]   # [4, 3, 3]: weights of one complete filter.
        products = patch * kernels # [4, 3, 3]: 36 elementwise products.

        # Axes 1 and 2 of this tensor are the selected height and width.
        channel_results = products.sum(dim=(1, 2)) # [4]: per-channel sums.
        manual_value = channel_results.sum()      # []: sum across channels too.
        assert torch.allclose(
            manual_value, actual_value, atol=1e-6, rtol=1e-4
        )

    assert tuple(y.shape) == (1, 6, 7, 7)
    print("Output:", tuple(y.shape))
    print("Per-channel results:", channel_results)
    print("Manual value / Conv output:", manual_value.item(), actual_value.item())

    # One filter has 36 weights, whereas one output channel has 49 values.
    # The same weights are reused at multiple spatial positions.
    assert kernels.numel() == 36
    assert y[0, 0].numel() == 49

    # ------------------------------------------------------------------
    # 2. Distinguish the ReLU function, a layer object, and a computed result.
    # ------------------------------------------------------------------
    relu_input = torch.tensor([-2.0, 3.0, 0.0, -4.0], device=device)
    relu_function = F.relu # Without parentheses, store the function itself.
    relu_layer = nn.ReLU() # Construct a layer object without processing input yet.
    relu_result1 = relu_function(relu_input)
    relu_result2 = relu_layer(relu_input)

    assert torch.equal(relu_result1, relu_result2)
    assert tuple(relu_result1.shape) == tuple(relu_input.shape)
    print("ReLU result:", relu_result1)

    # ------------------------------------------------------------------
    # 3. MaxPool selects the maximum in each 2×2 region within a channel.
    # ------------------------------------------------------------------
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    pool_input = torch.tensor(
        [[[
            [3.0, 7.0, 1.0, 4.0],
            [2.0, 5.0, 8.0, 0.0],
            [6.0, 1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0, 9.0],
        ]]],
        device=device,
    )
    pool_output = pool(pool_input)
    expected_pool = torch.tensor([[[[7.0, 8.0], [6.0, 9.0]]]], device=device)
    assert torch.equal(pool_output, expected_pool)
    assert tuple(pool_output.shape) == (1, 1, 2, 2)
    print("Pool result:", pool_output)

    # ------------------------------------------------------------------
    # 4. Count the parameters created before any input is supplied.
    # ------------------------------------------------------------------
    channels = 16
    two_small = nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
    ).to(device)
    one_large = nn.Conv2d(
        channels, channels, 5, padding=2, bias=False
    ).to(device)

    # Separate Conv objects have separate weights. ReLU has no parameters.
    small_count = sum(p.numel() for p in two_small.parameters())
    large_count = sum(p.numel() for p in one_large.parameters())
    assert small_count == 2 * 16 * 16 * 3 * 3 == 4608
    assert large_count == 16 * 16 * 5 * 5 == 6400
    print("Two 3×3 layers / one 5×5 layer weights:", small_count, large_count)

    # parameters() includes frozen parameters too.
    # Add a requires_grad condition to count only currently trainable parameters.
    trainable_count = sum(
        p.numel() for p in two_small.parameters() if p.requires_grad
    )
    assert trainable_count == small_count

    # ------------------------------------------------------------------
    # 5. Supply input, then inspect the middle 3×3 patch and original 5×5 region.
    # ------------------------------------------------------------------
    rf_x = torch.randn(1, 16, 7, 7, device=device)
    with torch.no_grad():
        rf_m = two_small[1](two_small[0](rf_x))
        rf_y = two_small[2](rf_m)
        rf_large = one_large(rf_x)

    assert rf_x.shape == rf_m.shape == rf_y.shape == rf_large.shape
    middle_patch = rf_m[0, :, 2:5, 2:5] # Intermediate values read for [0,0,3,3].
    input_region = rf_x[0, :, 1:6, 1:6] # Original input region connected to them.
    assert tuple(middle_patch.shape) == (16, 3, 3)
    assert tuple(input_region.shape) == (16, 5, 5)
    print("Middle patch / original input region:", middle_patch.shape, input_region.shape)

    # The theoretical receptive fields match; the output values need not match.
    # This describes an interior position with stride=1, dilation=1, and no pool.
    # ReLU changes one value at a time, so it does not expand the referenced region.

    # ------------------------------------------------------------------
    # 6. A VGG-style block preserves spatial size through Conv and reduces it at Pool.
    # ------------------------------------------------------------------
    vgg_block = MyVGGBlock(3, 8).to(device)
    with torch.no_grad():
        for input_shape, expected_shape in [
            ((2, 3, 32, 32), (2, 8, 16, 16)),
            ((1, 3, 30, 26), (1, 8, 15, 13)),
        ]:
            # Unpack the four tuple values into separate positional arguments.
            images = torch.randn(*input_shape, device=device)
            after_conv1 = F.relu(vgg_block.conv1(images))
            after_conv2 = F.relu(vgg_block.conv2(after_conv1))
            after_pool = vgg_block.pool(after_conv2)
            block_forward = vgg_block(images)

            assert tuple(block_forward.shape) == expected_shape
            assert torch.allclose(
                block_forward, after_pool, atol=1e-6, rtol=1e-4
            )
            print(
                "VGG block:", input_shape,
                "→", tuple(after_conv1.shape),
                "→", tuple(after_conv2.shape),
                "→", tuple(after_pool.shape),
            )

    # Run the same input twice to compare layer-by-layer and complete execution.
    # In normal use, one call to vgg_block(images) is enough.

    # ------------------------------------------------------------------
    # 7. Connect two blocks, Flatten, and Linear, then check batches of two and one.
    # ------------------------------------------------------------------
    torch.manual_seed(16)
    small_model = MySmallVGG().to(device)
    with torch.no_grad():
        for batch_size in (2, 1):
            images = torch.randn(batch_size, 3, 32, 32, device=device)
            feature1 = small_model.block1(images)
            feature2 = small_model.block2(feature1)
            flat_features = feature2.flatten(start_dim=1)
            manual_logits = small_model.fc(flat_features)
            logits = small_model(images)

            assert tuple(feature1.shape) == (batch_size, 8, 16, 16)
            assert tuple(feature2.shape) == (batch_size, 16, 8, 8)
            assert tuple(flat_features.shape) == (batch_size, 1024)
            assert tuple(logits.shape) == (batch_size, 5)
            assert torch.allclose(
                logits, manual_logits, atol=1e-6, rtol=1e-4
            )

            # Linear stores weight as [output features, input features].
            # [N,1024] @ [1024,5] + [5] = [N,5].
            # Apply the same five biases to the scores of each image.
            matrix_logits = flat_features @ small_model.fc.weight.T
            matrix_logits = matrix_logits + small_model.fc.bias
            assert torch.allclose(
                manual_logits, matrix_logits, atol=1e-6, rtol=1e-4
            )
            print(
                "Classifier:", tuple(feature1.shape),
                "→", tuple(feature2.shape),
                "→", tuple(flat_features.shape),
                "→", tuple(logits.shape),
            )

    # The 1024 feature values per image differ from the model parameter count.
    assert tuple(small_model.fc.weight.shape) == (5, 1024)
    assert small_model.fc.weight.numel() == 5120
    assert small_model.fc.bias.numel() == 5
    parameter_count = sum(p.numel() for p in small_model.parameters())
    assert parameter_count == 224 + 584 + 1168 + 2320 + 5125 == 9421
    print("Total model parameters:", parameter_count)

    # ------------------------------------------------------------------
    # 8. Convert logits to a probability form without claiming trained accuracy.
    # ------------------------------------------------------------------
    small_model.eval()
    with torch.no_grad():
        prediction_images = torch.randn(2, 3, 32, 32, device=device)
        logits = small_model(prediction_images)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = logits.argmax(dim=1)
        probability_sums = probabilities.sum(dim=1)

        assert torch.allclose(probability_sums, torch.ones(2, device=device))
        assert torch.equal(predicted_classes, probabilities.argmax(dim=1))

    print("Probability sums:", probability_sums)
    print("Largest-score class before training:", predicted_classes)

    # ------------------------------------------------------------------
    # 9. Check residual addition and ReLU with small manually chosen arrays.
    # ------------------------------------------------------------------
    residual_x = torch.tensor([[2.0, -1.0], [0.0, 3.0]], device=device)
    residual_f = torch.tensor([[-1.0, 2.0], [1.0, -4.0]], device=device)
    residual_sum = residual_f + residual_x
    residual_y = F.relu(residual_sum)
    expected_sum = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device)
    expected_y = torch.tensor([[1.0, 1.0], [1.0, 0.0]], device=device)
    assert torch.equal(residual_sum, expected_sum)
    assert torch.equal(residual_y, expected_y)
    print("Residual sum:", residual_sum)
    print("ReLU after the residual sum:", residual_y)
    print("ReLU without a shortcut:", F.relu(residual_f))

    # residual_f above was manually chosen. Use actual Conv layers below.
    block = ResidualBlock(16).to(device)
    sample = torch.randn(1, 16, 14, 14, device=device)
    with torch.no_grad():
        transformed = block.conv2(F.relu(block.conv1(sample)))
        block_output = block(sample)
        assert transformed.shape == sample.shape
        assert torch.allclose(
            block_output, F.relu(transformed + sample), atol=1e-6, rtol=1e-4
        )

    index = (0, 0, 0, 0) # First data item, first channel, first row, first column.
    x_value = sample[index].item()
    f_value = transformed[index].item()
    print("One residual position, input / main path:", x_value, f_value)
    print("Sum / actual ReLU output:", x_value + f_value, block_output[index].item())

    # A matching shape alone does not establish that a shortcut is present.
    # Also check whether forward actually calculates out + x.

    # ------------------------------------------------------------------
    # 10. Projection aligns both paths in channels and spatial size before addition.
    # ------------------------------------------------------------------
    projection_block = ProjectionResidualBlock(16, 32).to(device)
    projection_input = torch.randn(1, 16, 14, 14, device=device)
    with torch.no_grad():
        main_path = projection_block.conv2(
            F.relu(projection_block.conv1(projection_input))
        )
        shortcut_path = projection_block.shortcut(projection_input)
        projection_output = projection_block(projection_input)

        # Do not merely check whether broadcasting happens to permit addition.
        # All corresponding axis sizes of the two paths must match here.
        assert tuple(main_path.shape) == (1, 32, 7, 7)
        assert main_path.shape == shortcut_path.shape
        assert torch.allclose(
            projection_output, F.relu(main_path + shortcut_path),
            atol=1e-6, rtol=1e-4,
        )

    assert tuple(projection_block.shortcut.weight.shape) == (32, 16, 1, 1)
    assert projection_block.shortcut.weight.numel() == 512
    assert projection_block.shortcut.bias.numel() == 32
    print("Projection main path / shortcut / output:",
          main_path.shape, shortcut_path.shape, projection_output.shape)
    print("1×1 shortcut weights / biases:",
          projection_block.shortcut.weight.numel(),
          projection_block.shortcut.bias.numel())

    # A per-channel 1×1 kernel has one weight, but one complete filter has 16.
    # With 32 filters, the layer has 32×16×1×1 = 512 weights.
    # Both residual blocks are educational designs omitting BatchNorm and other details.

    # ------------------------------------------------------------------
    # 11. eval controls module behavior; no_grad controls backward-graph recording.
    # ------------------------------------------------------------------
    probe_input = torch.randn(1, 3, 32, 32, device=device)
    small_model.eval()
    tracked_output = small_model(probe_input)
    with torch.no_grad():
        untracked_output = small_model(probe_input)

    # Trainable weights can cause an output to be tracked even if its input is not.
    assert probe_input.requires_grad is False
    assert tracked_output.requires_grad is True
    assert untracked_output.requires_grad is False
    assert small_model.fc.weight.requires_grad is True
    assert small_model.training is False
    print("eval only:", tracked_output.requires_grad)
    print("eval + no_grad:", untracked_output.requires_grad)
    print("Original weight requires_grad:", small_model.fc.weight.requires_grad)

    # No backward call was made, so no parameter gradients have accumulated.
    # eval does not disable autograd, and no_grad does not clear existing .grad values.
    assert all(p.grad is None for p in small_model.parameters())
    print("All shape and calculation checks passed. No training or weight updates were run.")


if __name__ == "__main__":
    # Importing reads the definitions; direct execution also runs the checks.
    run_checks()
