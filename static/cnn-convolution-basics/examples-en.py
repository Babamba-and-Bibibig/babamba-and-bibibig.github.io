# CNN data flow: run the Python examples from the article in order.
# Requires PyTorch.
# Runs numerical examples and shape checks; real-data training is not called.
# https://babamba-and-bibibig.github.io/en/cnn-convolution-basics/

import torch  # Import tensor creation and calculation functions.
from torch import nn  # Use torch.nn layer classes through the name nn.
import torch.nn.functional as F  # Use neural-network functions through the name F.

conv_example = nn.Conv2d(
    in_channels=3,       # Match the three channels in the RGB input.
    out_channels=32,     # Create 32 filters and therefore 32 output channels.
    kernel_size=3,       # Give each per-channel kernel a spatial size of 3×3.
    stride=1,           # Move the region's starting position one step at a time.
    padding=1,          # Include one extra position on each of the four input edges.
    dilation=1,          # Apply neighboring weights to neighboring input positions.
    groups=1,           # Let every filter use all input channels.
    bias=True,          # Create one trainable bias per output channel.
    padding_mode="zeros",  # Treat the additional edge positions as zeros.
)


x7 = torch.arange(1, 50, dtype=torch.float32).reshape(1, 1, 7, 7)
# One image, one channel, and a spatial size of 7×7.

kernel3 = torch.tensor(
    [[1, 2, -1], [-2, 1, 3], [2, -1, 1]],
    dtype=torch.float32,  # Use the same floating-point type as the convolution weights.
)

patch = x7[0, 0, 0:3, 0:3]  # Select rows 0–2 and columns 0–2 in the first image/channel.
products = patch * kernel3  # Multiply matching elements: the result has shape (3, 3).


print(products.tolist())  # Convert the tensor to Python lists to inspect its values.
# [[1.0, 4.0, -3.0], [-16.0, 9.0, 30.0], [30.0, -16.0, 17.0]]

print(products.sum(dim=1).tolist())  # Sum the three values in each row: [2.0, 23.0, 31.0].
print(products.sum().item())  # Sum all nine values: 56.0.
print((products.sum() + 2).item())  # Add bias 2 to complete one output value: 58.0.


conv3 = nn.Conv2d(
    in_channels=1, out_channels=32, kernel_size=3,
    stride=1, padding=0, bias=True,
)  # Create a layer with one input channel and 32 filters.

with torch.no_grad():  # Do not record this block's operations for backpropagation.
    conv3.weight[0, 0].copy_(kernel3)  # Set the first filter's first per-channel kernel.
    conv3.bias[0].fill_(2.0)  # Set the first output channel's bias to 2.
    y3 = conv3(x7)  # Calculate the convolution: complete output shape (1, 32, 5, 5).

print(y3[0, 0])  # Inspect only the first output channel of the first image.


manual3 = torch.empty_like(y3[0, 0])  # Allocate (5, 5) storage matching the first output channel.
with torch.no_grad():
    for row in range(y3.shape[2]):  # Select vertical output indices 0–4.
        for col in range(y3.shape[3]):  # Select horizontal output indices 0–4.
            patch = x7[0, 0, row:row + 3, col:col + 3]  # The 3×3 patch at this position.
            manual3[row, col] = (patch * kernel3).sum() + conv3.bias[0]

assert torch.allclose(manual3, y3[0, 0])  # Compare the results with a small floating-point tolerance.


kernel5 = torch.tensor(
    [[1, -1, 2, 1, -2],
     [2, 1, -1, 2, 1],
     [-1, 2, 1, -2, 2],
     [1, -2, 2, 1, -1],
     [2, 1, -1, 2, 1]],
    dtype=torch.float32,  # Store these 5×5 weights as floating-point values too.
)
conv5 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
with torch.no_grad():
    conv5.weight[0, 0].copy_(kernel5)  # Assign the first filter's kernel.
    conv5.bias[0].fill_(2.0)  # Keep the same bias of 2 for this comparison.
    y5 = conv5(x7)  # Use the same original x7: output shape (1, 32, 3, 3).
    first5 = (x7[0, 0, :5, :5] * kernel5).sum() + conv5.bias[0]

print(first5.item())  # 263.0: the 25 products sum to 261, then bias 2 is added.
print(y5[0, 0].tolist())
# [[263.0, 277.0, 291.0], [361.0, 375.0, 389.0], [459.0, 473.0, 487.0]]


import torch  # Use tensors and device selection.
from torch import nn  # Use neural-network layers and the model base class.
import torch.nn.functional as F  # Use the ReLU function.


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize the mechanisms that manage child layers and parameters.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # One input channel → 32 output channels. Preserve the 28×28 spatial size.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 32 input channels → 64 output channels. Preserve the incoming spatial size.
        self.pool = nn.MaxPool2d(kernel_size=2)  # Default stride is also 2.
        self.fc = nn.Linear(64 * 7 * 7, 10)  # 3136 values per image → ten scores.

    def features(self, x, trace=False):
        if trace:
            print("Input:", tuple(x.shape))  # Display the shape as a regular Python tuple.
        x = self.conv1(x)
        if trace:
            print("conv1:", tuple(x.shape))
        x = F.relu(x)  # Replace negative values with zero while preserving shape.
        if trace:
            print("ReLU 1:", tuple(x.shape))
        x = self.pool(x)  # 28×28 → 14×14.
        if trace:
            print("Pool 1:", tuple(x.shape))
        x = self.conv2(x)
        if trace:
            print("conv2:", tuple(x.shape))
        x = F.relu(x)
        if trace:
            print("ReLU 2:", tuple(x.shape))
        x = self.pool(x)  # 14×14 → 7×7. Call the same pooling object again.
        if trace:
            print("Pool 2:", tuple(x.shape))
        return x  # Return the final feature maps.

    def forward(self, x, trace=False):
        x = self.features(x, trace=trace)  # Run convolution, activation, and pooling in order.
        x = x.flatten(start_dim=1)  # Keep the batch dimension and flatten each image.
        if trace:
            print("Flatten:", tuple(x.shape))
        if x.shape[1] != self.fc.in_features:  # Compare the actual count with Linear's requirement.
            raise ValueError(
                f"Linear requires {self.fc.in_features} values per image, "
                f"but received {x.shape[1]} values per image."
            )
        x = self.fc(x)  # Produce ten class logits.
        if trace:
            print("Linear:", tuple(x.shape))
        return x


# CUDA is the device backend for NVIDIA GPUs; MPS is the backend for Apple GPUs.
if torch.cuda.is_available():
    device = torch.device("cuda")  # Select an available NVIDIA GPU.
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Otherwise select an available Apple GPU.
else:
    device = torch.device("cpu")  # Use the CPU when neither GPU backend is available.

my_model = MyCNN().to(device)  # Create the model and move its parameters to the selected device.
my_model.eval()  # Switch to evaluation mode. This does not disable automatic differentiation.
probe = torch.zeros(2, 1, 28, 28, device=device)  # Two zero-filled inputs for checking connections.

with torch.no_grad():  # Check the forward pass without recording it for backpropagation.
    output = my_model(probe, trace=True)

assert output.shape == (2, 10)  # Check that both images receive ten output scores.


for name, parameter in my_model.named_parameters():
    print(name, tuple(parameter.shape), parameter.numel())  # Name, shape, and element count.

# Count the elements in each parameter tensor, then add the counts using Python's sum.
parameter_count = sum(p.numel() for p in my_model.parameters())
assert parameter_count == 50186  # Compare with the total counted by hand.

with torch.no_grad():
    feature_maps = my_model.features(probe)  # (2, 64, 7, 7).
    flattened = feature_maps.flatten(start_dim=1)  # (2, 3136).
    print("Linear input count:", flattened.shape[1])  # 3136: the count excluding batch size.
    scores_direct = flattened @ my_model.fc.weight.T + my_model.fc.bias
    scores_layer = my_model.fc(flattened)  # Let Linear perform the same calculation.
    assert torch.allclose(scores_direct, scores_layer)  # Compare the two sets of scores.
    predicted_classes = scores_layer.argmax(dim=1)  # The position of each image's largest score.
    print("Predicted index shape:", tuple(predicted_classes.shape))  # (2,).


def train_one_batch(model, images, labels, optimizer):
    model_device = next(model.parameters()).device  # Read the device of the first parameter.
    images = images.to(device=model_device, dtype=torch.float32)  # Floating-point image inputs.
    labels = labels.to(device=model_device, dtype=torch.long)  # Integer target class indices.
    model.train()  # Switch to training mode.
    optimizer.zero_grad(set_to_none=True)  # Clear previous .grad values by setting them to None.
    logits = model(images)  # (N, 10): scores that have not been converted to probabilities.
    loss = F.cross_entropy(logits, labels)  # The default reduction averages the image losses.
    loss.backward()  # Calculate gradients for all trainable parameters.
    optimizer.step()  # Let the optimizer update the actual parameters.
    return loss.item()  # Extract a Python number from the scalar loss tensor.


optimizer = torch.optim.Adam(
    my_model.parameters(),  # Supply the parameters to update.
    lr=0.001,  # Learning rate: a setting used to control parameter update size.
)

# After preparing actual images and their targets, call the function as follows.
# loss_value = train_one_batch(my_model, images, labels, optimizer)


def conv_output_length(input_length, kernel, padding=0, stride=1, dilation=1):
    # All arguments are integer sizes. Input, kernel, stride, and dilation must be positive.
    if input_length <= 0 or kernel <= 0 or stride <= 0 or dilation <= 0:
        raise ValueError("Input, kernel, stride, and dilation must be positive.")
    if padding < 0:
        raise ValueError("Padding must be nonnegative.")
    effective_kernel = dilation * (kernel - 1) + 1  # Kernel extent including spacing.
    padded_length = input_length + 2 * padding  # Input length including padding on both sides.
    if padded_length < effective_kernel:
        raise ValueError("The kernel is larger than the padded input.")
    return (padded_length - effective_kernel) // stride + 1  # Number of starting positions.


assert conv_output_length(7, 3) == 5  # Without padding, output length is 5.
assert conv_output_length(28, 3, padding=1) == 28  # Padding 1 preserves the length.
