+++
title = "Evaluating a Denoising Autoencoder: Per-Image MSE and Noise-Level Experiments"
description = "Build a four-channel autoencoder and connect one training batch to per-image MSE. Follow the shapes of lists, scalars, stack, and unsqueeze, distinguish evaluating one fixed model from retraining at each noise level, and interpret the reported 23% and 32% ratios."
date = 2026-11-03T18:04:22+09:00
slug = "denoising-autoencoder-training-evaluation"

[extra]
katex = true
toc = true
styles = ["denoising-autoencoder-training-evaluation/article.css"]
+++

A denoising model has produced an image. The next step is to identify what the model improved. A lower overall error, a lower error for each image, and preserved digit strokes are different things to check. When comparing stronger noise, the interpretation also depends on **whether we evaluated the same model again or trained a new model from the beginning**.

The [previous autoencoder exercise](/en/denoising-autoencoder-pytorch/) followed an image batch through the model and its training loop. Here, we build a four-channel model and follow **model implementation → one training batch → per-image error → noise-level experiments → interpretation**. The necessary inputs and function roles are introduced here so that the post can also be read on its own.

The [complete Python exercise file](/denoising-autoencoder-training-evaluation/examples-en.py) contains the model, calculation functions, and training and evaluation procedures below. The original exercise did not include the bodies of its helper functions, so the helpers in the downloadable file were implemented separately for this post. The evaluation logs quoted in the article are kept distinct from the results produced by the new code. The APIs were checked against their official documentation on November 3, 2026, and the core calculations were verified with PyTorch 2.15.0 on CPU.

## 1. Connect the input, target, and reconstruction first

A **denoising autoencoder** is a neural network trained to take a noisy image and calculate an image close to its clean original. The **encoder** calculates intermediate features from the input. The **decoder** calculates an image-shaped output from those features.

These three variable names keep the same roles throughout the post.

| Variable | Data it contains | Where it is used |
| --- | --- | --- |
| `clean` | Original images without added noise | The target against which the output is compared |
| `noisy` | Original images with added noise | The model input |
| `recon` | Reconstructed images calculated by the model | The result compared with `clean` |

The model receives `noisy`. We keep `clean` as the reference for comparing its output. A numerical measure of the difference between the two images is a **loss**. Here, we use **mean squared error, or MSE**: subtract corresponding pixel values, square each difference, and take the mean.

The trainable numbers stored in the model are its **weights** and **biases**, collectively called **parameters**. A **gradient** describes the direction and sensitivity of the loss to a small change in a parameter at its current value. **Backpropagation** calculates the gradients, and an **optimizer** uses them to update the parameters.

<figure class="dae-eval-figure">
<img src="/denoising-autoencoder-training-evaluation/model-shapes-en.svg" alt="A batch of N noisy images with one channel and a 28 by 28 spatial size passes through an encoder to four-channel, 14 by 14 features, then through a decoder to one-channel, 28 by 28 reconstructions. The clean original is the target used to calculate MSE against the reconstruction." loading="lazy">
<figcaption>The model input is noisy, and the loss target is clean. Intermediate features are values calculated from the input; they are separate from the weights stored in the model.</figcaption>
</figure>

We can inspect the same model through five distinct parts.

| Part | What to identify in this model |
| --- | --- |
| Input | A batch of noisy grayscale images |
| Neurons and layers | Convolution and ReLU, followed by transposed convolution and Sigmoid |
| Intermediate values | Four-channel feature maps calculated separately for each image |
| Weights and biases | A total of 105 trainable numbers stored in the two convolution layers |
| Final output | Reconstructed images calculated at the same size as the originals |

**Creating, training, and evaluating a model are different operations.** Creation prepares the structure and initial parameters. Training changes parameters using inputs and targets. Evaluation measures the results produced by the current parameters. Calling `model(noisy)` alone does not train its parameters.[^module]

## 2. Build the four-channel model and follow its tensor shapes

A **tensor** holds numbers together with the structure in which those numbers are arranged. Its **shape** lists the size of each axis in order. Image batches use the order `(N, C, H, W)`.

| Axis number | Symbol | Meaning | Size in `(8, 1, 28, 28)` | Valid indices |
| ---: | --- | --- | ---: | --- |
| 0 | `N` | Number of images | 8 | 0–7 |
| 1 | `C` | Channels per image | 1 | 0 |
| 2 | `H` | Height of a channel | 28 | 0–27 |
| 3 | `W` | Width of a channel | 28 | 0–27 |

Axis number 1, an axis size of 4, and indices 0–3 along that axis are different pieces of information. The input channel count of 1 comes from the grayscale data in this problem. The intermediate features have four channels, and the batch example contains eight images.

`torch.nn` contains classes for neural network layers and losses. After `from torch import nn`, we can write `nn.Conv2d` instead of `torch.nn.Conv2d`. A **class** defines how to create an object; an **object** is an instance created from that definition. In a model class, `__init__()` creates the layers, and `forward()` describes the order in which they process an input.

```python
import torch
from torch import nn  # Use the name nn for torch.nn.


class StudentDAE(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize nn.Module's layer and parameter management.
        self.encoder = nn.Conv2d(
            in_channels=1,   # Number of channels in an input image.
            out_channels=4,  # Number of feature-map channels calculated by the encoder.
            kernel_size=3,   # The spatial kernel is 3 by 3.
            stride=2,        # Move the kernel's starting position by two positions.
            padding=1,       # Add one position of padding at each input boundary.
            bias=True,       # Use a bias for each output channel.
        )
        self.decoder = nn.ConvTranspose2d(
            in_channels=4,   # Number of channels received from the encoder.
            out_channels=1,  # Number of channels in the reconstructed grayscale image.
            kernel_size=4,   # The spatial kernel is 4 by 4.
            stride=2,        # Spacing between the output regions for input positions.
            padding=1,       # Setting that adjusts the output boundaries.
            bias=True,
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, noisy):
        z = self.encoder(noisy)   # (N,1,28,28) → (N,4,14,14).
        z = self.relu(z)          # Values change; the shape stays the same.
        recon = self.decoder(z)   # (N,4,14,14) → (N,1,28,28).
        return self.sigmoid(recon)
```

`ReLU` replaces negative values with zero and keeps positive values. `Sigmoid` restricts the output to between zero and one, matching the range of the original pixels in this problem. Neither changes the tensor's shape. These functions are called **activation functions**. The model acquires its denoising ability by learning its parameters.[^activations]

The arguments used to create the two layers are settings passed to their **constructors**. Later, the argument passed to `model(noisy)` is the actual image tensor. PyTorch then runs the model's `forward()` calculation.

```python
shape_model = StudentDAE()  # A model for checking shapes; it has not been trained.
shape_input = torch.zeros(8, 1, 28, 28)  # Input used only to check the shape.
shape_output = shape_model(shape_input)
print(tuple(shape_output.shape))  # (8, 1, 28, 28)
```

This output does not measure reconstruction quality. It checks that a batch of eight images produces eight outputs with the expected image shape.

### Count output elements and parameters separately

In this configuration, one image's intermediate features contain `4 × 14 × 14 = 784` values. One input image also contains `1 × 28 × 28 = 784` values. **The spatial height and width decrease, but the channel count increases, leaving the total element count unchanged.** We can call the intermediate output a **latent representation**, but describing this structure as reducing the number of values would be incorrect.

The parameter count measures something else: weights stored in the kernels and biases for the output channels. The following calculation uses this model's `groups=1` setting.[^conv][^transpose]

| Layer | Weight shape and axis meanings | Weights | Biases | Total |
| --- | --- | ---: | ---: | ---: |
| `Conv2d` | `(4,1,3,3)` = output channels, input channels, kernel height, kernel width | `4×1×3×3=36` | 4 | 40 |
| `ConvTranspose2d` | `(4,1,4,4)` = input channels, output channels, kernel height, kernel width | `4×1×4×4=64` | 1 | 65 |
| Whole model | Trainable numbers in both layers | 100 | 5 | **105** |

Both weight shapes happen to start with `(4,1)`, but those axes have different meanings in the two layers. Kernels reuse the same weights at multiple spatial positions. The model therefore retains 105 parameters when the image count or spatial size changes. This does not mean that the number of output values or the amount of computation is fixed at 105.

Use the exact capitalization in names such as `nn.ReLU()` and `nn.Sigmoid()`. By contrast, `encoder` in `self.encoder` is an attribute name chosen by the programmer. Its definition and use must agree, and conventional spelling makes the code easier to read. Both convolution layers default to `bias=True`; the code states this explicitly to show the setting.

## 3. The 28 → 14 → 28 sizes come from kernel positions

### Convolution counts valid starting positions

Start with the height direction. An original length of 28 plus one padding position on each side gives a length of 30. The padded indices run from 0 to 29. A length-three kernel can start as far as position 27, but this configuration advances its starting position in steps of two.

| Starting position | Positions used by the kernel | Valid? |
| ---: | --- | --- |
| 0 | 0, 1, 2 | Yes |
| 2 | 2, 3, 4 | Yes |
| 26 | 26, 27, 28 | Yes |
| Next candidate: 28 | 28, 29, 30 | No; index 30 does not exist |

The actual starting positions are the 14 positions `0, 2, 4, …, 26`. There are therefore 14 output values in the height direction. Applying the same calculation to the width gives `14×14`.

```python
starts = list(range(0, 28, 2))  # Start at 0, exclude the end bound 28, step by 2.
print(starts)  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
print(len(starts))  # 14
```

We can express that position count as a formula. `L_in` is the input length, `L_out` is the output length, `k` is the kernel length, `s` is the spacing between starting positions, and `p` is the padding length on one side. We use the default `dilation=1`, which controls the spacing between kernel elements.

<div class="dae-eval-math">
$$
L_{out}=\left\lfloor\frac{L_{in}+2p-k}{s}\right\rfloor+1
$$
</div>

`L_in + 2p - k` is the largest starting position at which the kernel fits. Dividing it by `s` and taking the floor counts the moves possible after the first position. The final `+1` includes the first calculation at position zero. **Floor** selects the largest integer no greater than the value. Here, `floor((28+2−3)/2)+1 = 13+1 = 14`.[^conv]

### Transposed convolution counts the length covered by output regions

We can understand **transposed convolution** as multiplying one input value by several kernel weights, then adding those products at their corresponding output positions. An input's **contribution** here means the product of that input value and a particular weight. It is not a backpropagation gradient.

Consider input values `A` and `B` along one direction, a kernel length of four, and starting positions spaced two positions apart. Before the boundary adjustment, the products from `A` enter output positions 0–3, and the products from `B` enter positions 2–5.

| Output position | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Value added from `A` | `A·w₀` | `A·w₁` | `A·w₂` | `A·w₃` | None | None |
| Value added from `B` | None | None | `B·w₀` | `B·w₁` | `B·w₂` | `B·w₃` |

`w₀` through `w₃` are the kernel's four weights. Output position 2 receives `A·w₂ + B·w₀`. **There are eight product terms to add, but only six distinct output positions.** The two regions overlap by two positions. If the starting positions were spaced four positions apart with the same kernel length of four, the regions would cover 0–3 and 4–7 without overlapping. Overlap follows from the combination of kernel length and stride.

In two dimensions, the same placement applies to height and width. If one input channel has values `A, B, C, D` at its `2×2` positions, each value adds its products into a corresponding `4×4` output region.

<figure class="dae-eval-figure">
<img src="/denoising-autoencoder-training-evaluation/transpose-2d-en.svg" alt="The four-by-four output regions for input values A, B, C, and D start at 0,0; 0,2; 2,0; and 2,2 and overlap. Removing one position from each edge of the full six-by-six region leaves four by four." loading="lazy">
<figcaption>This shows spatial overlap from one input channel. The actual decoder also sums the calculations from all four input channels at corresponding positions in one output channel, then adds its bias.</figcaption>
</figure>

At position `(2,2)` before the boundary adjustment, this diagram adds `A·k₂₂ + B·k₂₀ + C·k₀₂ + D·k₀₀`. The two subscripts on `k` identify the height and width positions inside the kernel. Summing input-channel calculations does not multiply the output height and width by the channel count.

The actual decoder has 14 input positions along each direction. The output regions for input indices 0–13 begin at positions `0, 2, …, 26`. The final region covers 26–29. The length before boundary adjustment is therefore **four positions in the first region + 13 subsequent moves × two positions = 30 positions**. There are 13 moves covering a distance of 26 positions.

In this configuration, `padding=1` has the same size effect as excluding one boundary position on each side. Removing two positions from a length of 30 leaves 28. This explains the calculation; it does not require PyTorch to allocate a length-30 tensor internally and then crop it.

With the additional output-size adjustment `output_padding` also at its default of zero, the relationship is as follows.[^transpose]

<div class="dae-eval-math">
$$
L_{out}=(L_{in}-1)s+k-2p
$$
</div>

`L_in−1` counts the moves after the first position. Multiplying by `s` gives the final region's starting position. Adding the kernel length `k` gives the full length, and subtracting `2p` applies the two boundary adjustments. For this model, `(14−1)×2+4−2×1=28`.

The current encoder maps both lengths 27 and 28 to 14. The decoder maps that 14 to 28, so `27 → 14 → 28` is also possible. Transposed convolution is not an inverse function that always recovers the original size and information exactly. **Checking the output shape and evaluating the reconstructed pixels are separate tasks.**

## 4. Identify what actually changes during one training batch

A **batch** is a group of images processed together in one calculation. Training on one batch means calculating gradients from those inputs and targets, then updating the model's parameters once.

First, `nn.MSELoss()` creates an object for calculating the loss. We must pass predictions and targets to that object to obtain a loss value.

```python
loss_fn = nn.MSELoss()         # Create a loss object; reduction="mean" is the default.
```

Next, prepare the actual reconstruction `pred` and target `clean`, then call `loss_fn(pred, clean)` to obtain a loss tensor. `nn.MSELoss()(pred, clean)` combines object creation and calculation into one line. The constructor parentheses and the calculation call parentheses have different roles. The default MSE averages the squared errors for all corresponding image, channel, and pixel elements in the batch.[^mse]

```python
def train_one_batch(model, optimizer, noisy, clean):
    model.train()  # Set training mode; this line alone does not update parameters.
    optimizer.zero_grad(set_to_none=True)  # Clear the previous .grad values.

    pred = model(noisy)  # Calculate reconstructions with the current parameters.
    if pred.shape != clean.shape:
        raise ValueError("The reconstruction and target shapes differ.")

    loss = nn.MSELoss()(pred, clean)  # Loss measured before the update.
    loss.backward()                  # Calculate a gradient for each parameter.
    optimizer.step()                 # Update parameters using the gradients.
    return loss.item()               # Return the earlier loss as a Python number.
```

A function **parameter** is a name such as `model` or `optimizer` that receives a value. Distinguish this use of the word from a model's trainable parameters. Here, `optimizer` is an object that has already been created and passed to the function. We call one of its operations, or **methods**, using `optimizer.step()`. We do not try to call the optimizer object itself as a function with `optimizer().step()`.

### Clearing gradients does not erase learned weights

A parameter's `.grad` stores the gradient calculated by `backward()`. PyTorch accumulates gradients into existing `.grad` values. To use only the current batch's gradients, clear the earlier ones first. `set_to_none=True` resets the stored gradient values to `None`. The weights and biases retain the values learned so far.[^zero-grad]

| Call | What changes | What has not happened at that point |
| --- | --- | --- |
| `zero_grad()` | Previous stored gradients | Calculation of the new batch's gradients |
| `model(noisy)` | Newly calculated reconstruction tensor | Parameter update |
| `loss.backward()` | Each parameter's `.grad` | Update by the optimizer |
| `optimizer.step()` | Model weights, biases, and optimizer state | Recalculation of predictions and loss after the update |

The order `backward → zero_grad → step` erases the gradients before the update can use them. Clearing gradients after `step` can also work when the initial gradient state is managed correctly, but this post consistently clears them at the start of each batch.

### A loss returned after `step()` is still the value from before the update

`loss` is a result already calculated from `pred` and `clean`. Updating the model's parameters with `step()` does not automatically recalculate those existing tensors. Therefore, **`return loss.item()` returns the loss measured before the update.**

To measure the loss after the update, run another forward pass and loss calculation with the new parameters. The following example uses three artificial images only to check the timing of the measurements. `rand()` generates uniform random values in the range 0–1, and `randn_like()` generates standard normal random values with the same shape as its input. This is not an experiment measuring handwritten-digit reconstruction quality.

```python
torch.manual_seed(7)
step_model = StudentDAE().cpu()  # A separate model for checking measurement timing.
step_optimizer = torch.optim.Adam(step_model.parameters(), lr=0.01)
demo_clean = torch.rand(3, 1, 28, 28)  # Three artificial original images.
demo_noisy = (demo_clean + 0.25 * torch.randn_like(demo_clean)).clamp(0, 1)

before = train_one_batch(step_model, step_optimizer, demo_noisy, demo_clean)
with torch.no_grad():  # This comparison does not need a backpropagation graph.
    demo_recon = step_model(demo_noisy)  # Recalculate with the updated weights.
    after = nn.MSELoss()(demo_recon, demo_clean).item()
print("Measured before the update:", before, "Remeasured after the update:", after)
```

One parameter update is not guaranteed to lower the loss every time. `item()` extracts a Python number from a tensor containing one element. This loss tensor has shape `()`, and the extracted value is a `float`. If we first replace it with `loss = loss.item()`, calling `loss.backward()` fails because the Python number has no `backward()` method.[^item]

## 5. Keep one MSE value for each image

One overall mean makes it hard to distinguish which images improved and which became worse. Comparing individual images requires retaining `N` MSE values for `N` images.

First identify the corresponding positions. `n` is the image index, `c` is the channel index, and `h` and `w` identify the height and width positions. Subtracting `clean[n,c,h,w]` from `recon[n,c,h,w]` gives the error at that position. Tensors with the same shape can be subtracted directly in their multidimensional form. Calling `flatten()` first is not a requirement for subtraction.

`(recon − clean).square()` squares each error and preserves the input shape. At that point, the squared error for every pixel remains available. Next we choose **which values to average**.

| Scope of the calculation | Values retained | Result shape |
| --- | --- | --- |
| All channels, height positions, and width positions of one image | One MSE for that image | `()` |
| The same calculation separately for N images | N per-image MSE values | `(N,)` |
| All corresponding elements in the batch | One overall mean MSE | `()` |

Start with a loop that processes one image at a time. `zip()` pairs reconstructions and targets at matching positions in their sequences. If the input lengths differ, it stops at the shorter one, so first check that the shapes match.[^python-builtins]

```python
def per_image_mse(reconstruction, clean):
    if reconstruction.shape != clean.shape:
        raise ValueError("The two batches must have the same shape.")
    if reconstruction.ndim != 4 or reconstruction.shape[0] == 0:
        raise ValueError("A nonempty image batch with shape (N,C,H,W) is required.")

    losses = []  # A Python list that stores one loss tensor per image.
    for recon_img, clean_img in zip(reconstruction, clean):
        # One image has shape (C,H,W); flatten only its own values.
        error = recon_img.flatten() - clean_img.flatten()
        loss = error.square().mean()  # This image's MSE has shape ().
        losses.append(loss)          # Append the scalar tensor to the list.

    return torch.stack(losses)  # Combine N scalars into one tensor with shape (N,).
```

The function does not return one mean. It combines the MSE values into one tensor while preserving image order. Reusing the three artificial images from the previous section, we can obtain three per-image values and then calculate the overall mean separately.

```python
each = per_image_mse(demo_recon, demo_clean)  # MSE for three images, shape (3,).
overall = each.mean()                       # Mean of the three MSEs, shape ().
print(tuple(each.shape), tuple(overall.shape))  # (3,) ()
```

`losses` is a Python list and has no `losses.sum()` method. More importantly, regardless of how its values are combined, returning one mean would discard the per-image information. `torch.stack(losses)` is the return value that matches this function's purpose.

## 6. Read lists, scalar tensors, and `(N,)` separately

A **scalar** is a single value. `torch.tensor(0.1)` is a zero-dimensional tensor with no axes and shape `()`. `torch.tensor([0.1])` has one axis of size one, giving it shape `(1,)`. Both contain one element, but their numbers of axes differ.

The following numbers illustrate shapes; they are not measured training results.

```python
losses = [torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3)]
print(len(losses))               # 3: the number of tensors in the list.
print(tuple(losses[0].shape))    # (): the first tensor is a scalar.

each = torch.stack(losses)       # Combine equal-shaped tensors along a new axis.
print(each)                     # tensor([0.1000, 0.2000, 0.3000])
print(tuple(each.shape))        # (3,)

column = each.unsqueeze(1)       # Insert a new size-one axis at position 1.
print(tuple(column.shape))      # (3, 1)
print(tuple(each.shape))        # (3,): we did not reassign each.
```

<figure class="dae-eval-figure">
<img src="/denoising-autoencoder-training-evaluation/scalar-stack-en.svg" alt="A Python list without a tensor shape contains three scalar tensors with shape (). Stack produces a (3,) tensor, and unsqueeze(1) produces (3,1). Separately, per-image MSE flattens (8,1,28,28) to (8,784), then averages to produce (8,)." loading="lazy">
<figcaption>Check the list length, the shape of each tensor in the list, and the combined tensor's shape in that order. The axis added by stack has a size equal to the number of tensors combined.</figcaption>
</figure>

`losses` itself has no tensor `.shape` attribute. Each tensor inside it has shape `()`, and stacking those tensors produces `(3,)`. With eight images, the same process produces `(8,)`.[^stack]

| Shape notation | Meaning | Example of selecting a value |
| --- | --- | --- |
| `()` | A tensor with no axes | Read its number with `.item()` |
| `(8,)` | One axis of size eight | `each[0]` through `each[7]` |
| `(8,1)` | Two axes of sizes eight and one | `column[0,0]` through `column[7,0]` |

The comma in `(8,)` is Python syntax for a **tuple** containing one element. `(8)` is the number eight enclosed in parentheses; `(8,)` is a tuple containing the number eight. The comma does not stand for an omitted second axis. `torch.Size([8])` and the `(8,)` printed by `tuple(each.shape)` describe the same shape.[^tuples]

Stacking eight scalar tensors with shape `()` gives `(8,)`. Stacking eight length-one tensors with shape `(1,)` gives `(8,1)`. Check the input tensor shapes first to distinguish the two results.

## 7. What do `stack` and `unsqueeze` place on a new axis?

`stack` receives **multiple tensors** and combines them along a new axis. `unsqueeze` inserts a size-one axis into **one tensor**. Both produce one more axis than an input tensor has, but they differ in what is combined and the size of the new axis. `dim` specifies where to insert that axis.[^stack][^unsqueeze]

```python
a = torch.tensor([10, 20, 30])  # Shape (3,); original indices are 0–2.
b = torch.tensor([40, 50, 60])  # Shape (3,), matching a.

x = torch.stack([a, b], dim=0)
print(x.tolist())  # [[10, 20, 30], [40, 50, 60]]
print(tuple(x.shape))  # (2, 3)

y = torch.stack([a, b], dim=1)
print(y.tolist())  # [[10, 40], [20, 50], [30, 60]]
print(tuple(y.shape))  # (3, 2)

u = a.unsqueeze(0)
print(u.tolist())  # [[10, 20, 30]]
print(tuple(u.shape))  # (1, 3)

v = a.unsqueeze(1)
print(v.tolist())  # [[10], [20], [30]]
print(tuple(v.shape))  # (3, 1)
```

`tolist()` displays tensor values as Python lists. In the `(2,3)` result, the first axis distinguishes the two groups, `a` and `b`. In `(3,2)`, the first axis distinguishes the three original positions, and the second axis distinguishes the values from `a` and `b` at each position. For example, `x[1,2]` and `y[2,1]` both select the final value of `b`, which is 60.

`a.unsqueeze(1)` does not use `b`. It keeps the three numbers in `a` and adds a size-one axis. The returned tensor shares the original data storage. To have `a` refer to the new shape, reassign the result with `a = a.unsqueeze(1)`.

## 8. Keep the image axis to calculate the same MSE without a loop

The loop averaged the pixels of one image and repeated that operation for each image. When calculating the whole batch at once, we want to retain the same information. **Keep axis 0, which distinguishes the images, and average only the channels, height, and width within each image.**

```python
def per_image_mse_fast(reconstruction, clean):
    if reconstruction.shape != clean.shape:
        raise ValueError("The two batches must have the same shape.")
    if reconstruction.ndim != 4 or reconstruction.shape[0] == 0:
        raise ValueError("A nonempty image batch with shape (N,C,H,W) is required.")

    squared_error = (reconstruction - clean).square()  # (N,C,H,W).
    pixels = squared_error.flatten(start_dim=1)        # (N,C*H*W).
    return pixels.mean(dim=1)                          # (N,).
```

`flatten(start_dim=1)` combines axes from axis 1 through the last axis. This batch changes from `(8,1,28,28)` to `(8,784)`. The 784 squared errors for image 0 remain separate from the 784 for image 1, and so on. Next, `mean(dim=1)` averages the 784 values for each image and produces `(8,)`. By default, the reduced axis is removed from the result.[^flatten][^mean]

Comparing these three expressions makes it clear which information remains.

| Code | Result for squared errors with shape `(8,1,28,28)` | Information retained |
| --- | --- | --- |
| `error.flatten(1).mean(1)` | `(8,)` | The MSE for each image |
| `error.mean(dim=(1,2,3))` | `(8,)` | The same averaging scope as above |
| `error.flatten().mean()` | `()` | One overall mean, with the image distinction removed |

Here, `error` is the tensor of per-pixel squared errors. Calling `flatten()` without arguments also combines the image axis, producing `(6272,)`. The number 6,272 is the total element count: eight images with 784 pixels each. Once we average those values, we cannot recover the individual image MSEs from that one mean.

For a batch of equal-sized images, the mean of the per-image means is mathematically equal to the mean over all elements. Small floating-point differences can arise from the order of addition, so we compare them with `torch.allclose()`, which checks equality within numerical tolerances.

```python
# Values used to check the functions; these do not measure reconstruction quality.
clean_check = torch.zeros(8, 1, 28, 28)
recon_check = torch.full_like(clean_check, 0.2)  # Fill the same shape with 0.2.

loop_each = per_image_mse(recon_check, clean_check)
fast_each = per_image_mse_fast(recon_check, clean_check)
whole = nn.MSELoss()(recon_check, clean_check)

assert torch.allclose(loop_each, fast_each)
assert torch.allclose(fast_each.mean(), whole)
print(tuple(fast_each.shape))  # (8,)
```

The results agree because they average over the same sets of values. `fast_each` still retains individual image values, allowing us to compare improvement image by image afterward.

## 9. Compare the trained model with the input baseline

A **baseline** provides a reference for measuring the benefit of using a model. For denoising, we can use the noisy input without any processing as that baseline. We therefore calculate two errors.

| Comparison | Calculation | Question answered |
| --- | --- | --- |
| Input error | MSE between `noisy` and `clean` | How different was the input from the original before reconstruction? |
| Reconstruction error | MSE between `recon` and the same `clean` | How different was the model's output from the original? |

Here, **we evaluate one model that has already been trained**. The `model` in this section must be an object that has finished training on the training data. The earlier `shape_model` checks shapes, and `step_model` checks update timing; distinguish both from the trained model used for this evaluation. Section 17 contains the complete calls for training and evaluating a model with actual data. The evaluation function does not create a new model or call `fit()` to train again. `predict_fixed()` is not a built-in PyTorch function. It is a name for a function we define separately to process a supplied input without training.

Distinguish these two operations, which often appear together in evaluation code.[^module][^no-grad]

| Code | Role | What it does not do by itself |
| --- | --- | --- |
| `model.eval()` | Set evaluation behavior for layers such as Dropout and BatchNorm whose behavior depends on mode | Disable gradient tracking |
| `torch.no_grad()` | Stop recording operations for backpropagation inside the block | Set the model to evaluation mode |

This model contains neither Dropout nor BatchNorm. Its training and evaluation outputs therefore do not differ because of those layers, but the two operations still have different roles. `eval()` does not automatically freeze parameters, and `no_grad()` does not change the model's structure.

Use `split()` to divide an image batch that is too large to process at once. In the function below, `split(batch_size, dim=0)` divides the image axis into groups of at most `batch_size` images. `cat(..., dim=0)` joins the results again in their original order. The function uses only the supplied `noisy` tensor; it does not generate new noise.

```python
def predict_fixed(model, noisy, batch_size=256):
    if noisy.ndim != 4 or noisy.shape[0] == 0:
        raise ValueError("A nonempty image batch with shape (N,C,H,W) is required.")
    was_training = model.training  # Remember the mode used before this call.
    model.eval()
    try:
        with torch.no_grad():
            parts = []
            for batch in noisy.split(batch_size, dim=0):
                parts.append(model(batch))
            return torch.cat(parts, dim=0)
    finally:
        model.train(was_training)  # Restore the prior mode; this does not train.
```

The mode restoration in `finally` runs even when the result is returned from inside `try`. The model and its input must be on the same device. The complete exercise file creates models on CPU and checks that inputs are also CPU `float32` tensors.

Run the following code once a trained `model`, evaluation inputs `noisy`, and their originals `clean` are ready. The two image batches must correspond in the same order.

```python
recon = predict_fixed(model, noisy)  # Reconstruct with the current model, without training.
base_each = per_image_mse_fast(noisy, clean)  # (N,): input error per image.
rest_each = per_image_mse_fast(recon, clean)  # (N,): reconstruction error per image.

delta = base_each - rest_each  # Input MSE minus reconstruction MSE for the same image.
improved = int((delta > 0).sum().item())
worsened = int((delta < 0).sum().item())
tied = int((delta == 0).sum().item())

print("Per-image MSE shape:", tuple(rest_each.shape))
print(f"Input MSE={base_each.mean().item():.4f}, "
      f"Reconstruction MSE={rest_each.mean().item():.4f}")
print("Improved / worsened / tied:", improved, worsened, tied)
```

`delta` is a name for a difference. If `delta[i] > 0`, image `i` has a smaller reconstruction error than input error. `(delta > 0)` is a tensor containing a true-or-false result for each image. Applying `sum()` counts the true entries. **It counts improved images; it does not sum the amounts by which their errors decreased.**

`delta == 0` tests exact equality of the calculated values. If very small numerical differences should count as ties, define a tolerance separately. `:.4f` formats the displayed value to four decimal places; it does not round the tensor's internal value to four places. We can assign the calculated result directly to `recon`, so an earlier `recon = None` assignment is unnecessary.

## 10. What a log for 10,000 evaluation images tells us

The following output comes from the exercise record on which this article is based, with its labels translated into English. The original images and trained model were not supplied together, so these numbers are not presented as results reproduced by the new exercise file. Read them as **an example of interpreting an existing evaluation log**.

```text
Per-image MSE shape: (10000,)
Input MSE=0.0327, Reconstruction MSE=0.0078
Improved / worsened / tied: 10000 0 0
```

`(10000,)` means that each of the 10,000 images has one MSE value. The input mean is 0.0327 and the reconstruction mean is 0.0078, so the overall mean error decreased in this record. The final line also says that **the individually calculated MSE decreased for every evaluated image**, beyond the reduction in the overall mean.

An image's MSE, however, averages over many pixels. The mean can decrease even if some pixels become worse, provided that errors at other pixels decrease enough. This log therefore cannot establish that every pixel and every stroke improved.

The model's digit classification was not measured here either. Lower pixel error and higher digit-recognition accuracy are different outcomes. Assessing recognition requires a separate evaluation of classification results.

## 11. Choose the experimental question before changing the noise level

**Sigma, written σ**, controls the noise level here. In an implementation that adds Gaussian noise, it controls the **standard deviation** of the noise being added. Standard deviation describes how widely values are spread around their mean. A larger σ produces larger fluctuations in the added noise values. The next section defines the generation method explicitly.

There are two ways to compare noise levels 0.25 and 0.50.

| Aspect | Experiment A: evaluate the same model | Experiment B: train a new model at each noise level |
| --- | --- | --- |
| Question to choose first | Can the model we already have handle other noise levels? | What happens when a model is trained for each noise condition? |
| Model | Keep the same trained model | Create a new initial model for each condition |
| Training | None inside the comparison loop | Start training from the beginning for each condition |
| What changes | The noise level of the evaluation input | Training noise, the resulting trained model, and evaluation noise |
| Main comparison flow | Generate noise → predict → measure error | New model and optimizer → train → generate noise → predict → measure error |

<figure class="dae-eval-figure">
<img src="/denoising-autoencoder-training-evaluation/experiment-protocols-en.svg" alt="Experiment A holds one trained model fixed and evaluates inputs with noise levels 0.25 and 0.50. Experiment B creates and trains a new model and optimizer for each noise level before evaluating it." loading="lazy">
<figcaption>These are the comparison conditions used in this post's supplementary implementation. Both experiments use the same validation originals and the same underlying noise ε, but Experiment B starts a separate training process for each condition.</figcaption>
</figure>

**If a loop contains `StudentDAE()` and `fit()`, it is doing more than evaluation.** It creates and trains a new model for each condition, separately from the previously trained model. Locate model creation and training calls first to identify which question the code addresses.

The original exercise's noise comparison corresponds to Experiment B. Its results therefore cannot be interpreted as changing only the evaluation noise for the same model.

## 12. Define the training and validation noise explicitly

`fit`, `predict_fixed`, and `make_val_noisy` are names chosen by the programmer, not built-in PyTorch APIs. Their names alone cannot establish the number of training epochs, how randomness is fixed, or whether the data is separated. The complete exercise file for this post defines their behavior explicitly.

First prepare nonoverlapping sets of clean training images, `clean_train`, and clean evaluation images, `clean_val`. Both are CPU `float32` tensors with shape `(N,1,28,28)` and pixel values in the range 0–1. `float32` is a data type that stores each number as a 32-bit floating-point value. Adding `val` to a variable name does not separate the data. Verify the split using original image IDs or dataset split records.

### Use the same underlying noise for the same validation images

`torch.randn()` draws random values from the standard normal distribution, which has mean zero and standard deviation one. Call the resulting tensor `epsilon`, or `ε`. The noise to add to an original is `sigma * epsilon`. Then `clamp(0,1)` restricts values below zero to zero and values above one to one.[^noise]

```python
def make_val_noisy(clean, sigma, seed=9026):
    # Independent implementation for this post's CPU float32 inputs.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epsilon = torch.randn(
        clean.shape,        # One noise value for every original pixel position.
        dtype=clean.dtype,   # Use the same floating-point data type as the original.
        generator=generator,
    )
    return (clean + sigma * epsilon).clamp(0.0, 1.0)
```

A random-number **generator** manages the state used to produce a sequence of random values. A **seed** sets a starting condition so that the sequence can be produced again. This function starts a separate generator with the same seed on each call, so the same environment and shape produce the same `epsilon` again. Changing only σ applies a different multiplier to that same underlying noise.

A σ of 0.5 does not mean that only half the pixels are damaged. Also, once `clamp` clips values outside the allowed range, the distribution of the actual difference `noisy−clean` changes. We therefore cannot claim that the input MSE must equal `sigma**2` or that the standard deviation of the actual input error must be exactly σ.

### `fit()` repeats the one-batch training operation

An **epoch** is one pass through the training dataset. With 5,000 images and a maximum batch size of 128, one epoch processes 39 batches of 128 images and one final batch of eight. Updating once per batch gives 40 updates per epoch. Under these conditions, three epochs give 120 updates, not merely three calls to `step()`.

The following core implementation repeats the earlier `train_one_batch()` function. It uses separate generators for image shuffling and noise to make the random-number streams explicit. The downloadable file also includes input validation.

```python
def fit(model, optimizer, clean_train, *, sigma,
        epochs=3, batch_size=128, seed=2026, verbose=True):
    order_rng = torch.Generator(device="cpu").manual_seed(seed)
    noise_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    history = []
    n_images = len(clean_train)

    for epoch in range(1, epochs + 1):  # 1 through epochs; the end bound is excluded.
        # Arrange image indices 0 through n_images-1 in random order.
        order = torch.randperm(n_images, generator=order_rng)
        weighted_loss = 0.0

        for start in range(0, n_images, batch_size):
            ids = order[start:start + batch_size]  # The final group may be smaller.
            clean = clean_train[ids]
            epsilon = torch.randn(
                clean.shape, dtype=clean.dtype, generator=noise_rng,
            )
            noisy = (clean + sigma * epsilon).clamp(0.0, 1.0)
            loss = train_one_batch(model, optimizer, noisy, clean)
            weighted_loss += loss * len(clean)  # Account for the images in this batch.

        epoch_loss = weighted_loss / n_images
        history.append(epoch_loss)
        if verbose:  # All training above still runs when this is False.
            print(f"epoch={epoch}, train_mse={epoch_loss:.6f}")

    return history
```

Settings after `*` in the parameter list must be passed by name, such as `sigma=0.25`. `torch.randperm()` shuffles the image indices without repeating them. `order[start:start+batch_size]` includes the start and excludes the end bound; if the final slice extends beyond the sequence, it takes only the remaining elements.

Because the final batch may be smaller, we account for the image counts instead of simply averaging the batch means. All images in this exercise have the same size, so sum `batch mean × images in that batch`, then divide by the total image count. However, this record is **an average of batch losses measured at different parameter-update points**. Its measurement timing differs from validation loss measured with one fixed model at the end of an epoch.

`fit()` directly updates the parameters of the supplied model. Training therefore still occurs if the returned `history` is not assigned to a variable. In this implementation, `verbose=False` hides logs but does not skip training. The exact behavior of the identically named option in the original helper must be checked in that original function's body.

## 13. Separate fixed-model evaluation from retraining in code

### Experiment A: evaluate the already trained model unchanged

The `model` below has finished training on the training data. The loop does not create another model or optimizer. `SIGMAS` is a list containing the two noise levels to compare.

```python
SIGMAS = [0.25, 0.50]

for sigma in SIGMAS:
    val_noisy = make_val_noisy(clean_val, sigma, seed=9026)
    recon = predict_fixed(model, val_noisy)
    base = per_image_mse_fast(val_noisy, clean_val).mean().item()
    rest = per_image_mse_fast(recon, clean_val).mean().item()
    print(sigma, base, rest)
```

These results show **how one model with the same learned parameters responds to differently noisy inputs**. Its parameters are not updated during evaluation.

### Experiment B: train from the beginning at each noise level

Now change the purpose: compare models trained separately for each noise condition. Before reading the code, identify that each change in σ starts a new training process.

```python
SEED = 2026  # Initialization and training randomness chosen for this executable example.
rows = []

for sigma in SIGMAS:
    torch.manual_seed(SEED)  # Match initial weights in the same environment.
    trial = StudentDAE().cpu()  # Create a new model; do not reuse the existing model.
    trial_optimizer = torch.optim.Adam(
        trial.parameters(),  # Connect to this trial's weights and biases.
        lr=0.01,             # Learning rate: a setting that affects update size.
    )

    fit(
        trial,
        trial_optimizer,
        clean_train,
        sigma=sigma,      # Left: parameter name. Right: current loop value.
        epochs=3,         # This implementation states the epoch count explicitly.
        batch_size=128,
        seed=SEED,
        verbose=False,    # Perform training while hiding its logs.
    )

    val_noisy = make_val_noisy(clean_val, sigma, seed=9026)
    recon = predict_fixed(trial, val_noisy)  # Evaluate the trial just trained.
    base_each = per_image_mse_fast(val_noisy, clean_val)
    rest_each = per_image_mse_fast(recon, clean_val)
    rows.append((
        sigma,
        base_each.mean().item(),
        rest_each.mean().item(),
        int((base_each > rest_each).sum().item()),
    ))
```

Distinguish four connections in this code.

1. **`manual_seed()` matches the initial conditions.** In the same environment, setting the same seed immediately before creating the same model can match its initial parameters. It does not freeze the weights or prevent them from learning.
2. **`StudentDAE().cpu()` creates a new model and places it on CPU.** `.cpu()` does not load previously learned results. Even on a Mac, this code explicitly uses CPU, not GPU. The input data must also be on CPU.
3. **`Adam()` creates an optimizer for this model, and `fit()` performs the training.** Adam uses gradients and accumulated state to update parameters. Creating it does not itself train anything. Reusing the previous model's optimizer for a new model can leave the optimizer connected to the wrong parameters.[^adam]
4. **`make_val_noisy()` and `predict_fixed()` prepare and perform evaluation after training.** They contain no `backward()` or `step()` calls. The supplied `trial` object already has updated parameters, so we evaluate that same object.

Writing optimizer creation inside the function arguments, as in `fit(trial, torch.optim.Adam(...), ...)`, has the same meaning. Python first executes the inner `Adam(...)` call to create an optimizer, then passes that result as an argument to `fit()`. Splitting the operations into separate lines makes it easier to see which model's parameters are connected.

The original exercise's call `fit(..., sigma=sigma, verbose=False)` does not show an epoch count directly. Confirming the description of three epochs requires the function definition or surrounding settings. This supplementary implementation makes that condition visible with `epochs=3`. The learning rate `lr=0.01`, batch size 128, and seeds are also settings for this executable example; they are not guaranteed to be optimal.

One shared seed does not automatically control every comparison condition. Check the data split, image order, and noise generation method as well. The executable file separates randomness for initialization, training order, training noise, and validation noise. Even then, identical results are not guaranteed across different PyTorch versions or devices.[^randomness]

## 14. The 23% and 32% values describe the MSE remaining after reconstruction

The purpose of this ratio is to measure **how much error remains after reconstruction, relative to the mean squared error in the input**. Let `base` be the mean input MSE and `rest` the mean reconstruction MSE. The remaining ratio is `rest / base`. Subtracting that ratio from 100% gives the MSE reduction percentage.

| Calculation | Meaning |
| --- | --- |
| `ratio = rest / base` | Reconstruction MSE as a multiple of input MSE |
| `ratio × 100%` | Percentage of input MSE remaining |
| `(1 − ratio) × 100%` | Percentage reduction in MSE relative to the input |

If `ratio < 1`, mean MSE decreased. If `ratio > 1`, it increased. When `base` is zero, this ratio is undefined because division by zero is not possible.

### First, read the ratio from the earlier evaluation log

The earlier record has `base=0.0327` and `rest=0.0078`. These are the mean input MSE and mean reconstruction MSE, respectively, so the remaining ratio uses them in this order:

`mean reconstruction MSE / mean input MSE = 0.0078 / 0.0327 ≈ 0.2385`

Approximately **23.85%** of the input MSE remains after reconstruction, corresponding to an MSE reduction of approximately **76.15%**. This is approximate because the displayed inputs were rounded to four decimal places. When reporting actual results, calculate the ratio from unrounded `base` and `rest` values before formatting it for display.

### Read the separately trained noise-level results as a separate record

The following ratios were reported in a separate retraining comparison. Do not combine them with `0.0327` and `0.0078` as if those were absolute values from the same experiment. This record does not include the absolute input and reconstruction MSEs for each condition.

| σ used for training and evaluation | Reported reconstruction/input MSE | Corresponding MSE reduction |
| ---: | ---: | ---: |
| 0.25 | Approximately 23% | Approximately 77% |
| 0.50 | Approximately 32% | Approximately 68% |

MSE decreased relative to the input in both conditions. In this comparison, the relative reduction was smaller under stronger noise. Since `77% − 68% = 9 percentage points`, describe it as **an approximately nine-percentage-point lower MSE reduction**. A **percentage point** is a unit for the difference between two percentages. Simply saying “9% lower” can imply a different denominator for the change.

“Mean MSE decreased by approximately 77% relative to the input” identifies the measurement more precisely than “performance improved by 77%.” Ratios alone do not reveal how much the absolute reconstruction errors differed between the conditions. That comparison requires `base` and `rest` for each condition.

```python
# rows contains the results actually recorded by the earlier retraining comparison.
for sigma, base, rest, improved in rows:
    ratio = rest / base if base > 0 else float("nan")
    reduction = (1 - ratio) * 100
    print(
        f"σ={sigma:.2f}, input MSE={base:.4f}, reconstruction MSE={rest:.4f}, "
        f"remaining MSE={ratio:.1%}, reduction={reduction:.1f}%, improved images={improved}"
    )
```

`nan` marks a result that cannot be calculated as a valid number. `:.1%` multiplies a ratio by 100 and displays it as a percentage with one decimal place. `reduction` has already been multiplied by 100, so we use `:.1f` followed by a literal `%` character for that value.

The metric here is **a ratio of mean MSEs**. `rest_each.mean() / base_each.mean()` generally differs from `(rest_each / base_each).mean()`. In the first calculation, images with larger input errors account for a larger share of total input error. In the second, each image's ratio receives equal weight. The second calculation also needs a separate rule for images whose input MSE is zero. Record which calculation was used alongside the metric's name.

## 15. Compare the background and strokes of the same image

Measuring a numerical value such as MSE is **quantitative evaluation**. Inspecting the appearance and changes in actual results is **qualitative evaluation**. Image reconstruction benefits from both.

The `clean`, `noisy`, and `recon` being compared must correspond to the same original image. If the data has been shuffled, matching array positions alone may not be sufficient, so check original IDs as well. Placing an original of one digit next to a reconstruction of another would make the stroke comparison meaningless.

| What to inspect | What to compare with the original |
| --- | --- |
| Background | Did noisy dots decrease? Did new marks appear in previously empty areas? |
| Thin strokes | Are any parts broken or missing? |
| Stroke thickness and edges | Are strokes excessively thick or blurred relative to the original? |
| Overall shape | Are the features needed to distinguish the digit preserved? |

The executable file's `save_triplet()` saves three images from matching array positions. It uses the same brightness range of 0–1 for all three. Use the following code after confirming that original IDs and array order correspond.

```python
# save_triplet is defined in the accompanying complete Python file.
# It returns one pathlib.Path object for the saved PNG.
saved_path = save_triplet(clean, noisy, recon, index=0, out_dir="figures")
print(saved_path.name)      # Example: image_0.png
print(saved_path.exists())  # Check that the path actually exists.
```

The function displays `images[index,0]` with Matplotlib's `imshow()`. The first index selects an image, and the second index, zero, selects its grayscale channel. The selected tensor has shape `(28,28)`. `detach()` separates it from the backpropagation graph, `cpu().numpy()` prepares an array for plotting, and `savefig()` saves the figure. Since the actual handwritten images were not supplied, this post does not include invented examples of successful reconstruction.[^imshow]

### A list of `PosixPath(...)` objects is not automatically an error

Another visualization function might return several figure paths, producing notebook output such as this:

```text
[PosixPath('figures/trained_1.png'), PosixPath('figures/trained_2.png')]
```

`PosixPath` is an object from Python's `pathlib` that represents a path on Unix-like systems. The square brackets mean that the paths are in a list. IPython, the Python execution environment used by Jupyter, can automatically display the value of a cell's final expression. If a function displays figures and also returns their paths, that list can appear after the figures.[^pathlib][^ipython]

Assigning the result to a variable, such as `saved_paths = ...`, avoids the automatic display of the final return value. In a notebook, a semicolon at the end of the call can serve the same purpose. It does not suppress every `print()` or explicit display operation inside the function.

**Returning a path object and actually saving a file are also different facts.** Check that `exists()` is true, then open the relevant figure to inspect its contents. The original `tools.draw_triplets()` and this post's `save_triplet()` are different functions. The original implementation is unavailable, so its argument names alone do not establish how it selects and saves images.

## 16. State exactly what the results establish

Separating “what the results support” from “what remains unchecked” means **reporting what was measured without extending the conclusion to outcomes that were not measured**. It is more than adding a cautious sentence. The conclusion must match what the experiment changed and observed.

| Recorded observation | What the record establishes | What requires another check |
| --- | --- | --- |
| Per-image MSE decreased for all 10,000 images | Lower mean pixel error for each image | Improvement of every pixel and every important stroke |
| Remaining MSE of 23% and 32% in the retraining comparison | Approximately 77% and 68% reductions relative to the input in the respective conditions | How one unchanged model responds to different noise levels |
| Evaluation on one dataset and two noise levels | Measurements under those conditions | Results for real photographs, other datasets, and other noise types |
| Training under one random initialization condition | Results under that initial condition | Variation across repeated runs with different seeds |
| Measurement of pixel MSE | Squared error in pixel values | Digit-classification accuracy or performance on another practical task |

**Generalization** is a model's ability to work well on data it did not directly use for training. To treat a result as evidence of generalization, verify that validation images were not used in training. Data keys named `clean_train` and `clean_val` do not prove separation by themselves.

A **learning curve** plots losses recorded over training steps or epochs. It can reveal an overall decrease, a plateau, or large fluctuations. If training loss decreases while separately measured validation loss persistently increases, there is reason to check for **overfitting**: fitting the training data while performance on new data worsens.

The supplied records contain neither the actual learning curves nor comparison figures for matching image IDs. We therefore do not claim that a curve decreased smoothly or that a particular digit's strokes were preserved. The complete file's `fit()` returns a history of training losses, but it does not automatically create a validation-loss curve.

One paragraph describing the retraining results can connect the measurement conditions to the conclusion like this:

> In the exercise record, a new model was trained for each noise level. Mean MSE decreased relative to the input by approximately 77% at σ=0.25 and 68% at σ=0.50. Reconstruction error was lower than input error in both conditions, while the relative reduction was approximately nine percentage points smaller under stronger noise. However, the models were trained separately, so these results do not directly establish how one unchanged model responds to different noise levels. Conclusions about other data, other noise types, different initial conditions, and digit-recognition accuracy require separate evaluation.

This paragraph does not reuse the earlier single-model log, `0.0327 → 0.0078`, as absolute measurements for a different experiment. Whenever reporting a number, retain the identity of the model and data on which it was measured.

## 17. Run the complete file and distinguish the two experiments

The [Python exercise file](/denoising-autoencoder-training-evaluation/examples-en.py) brings together the model, one-batch training, full training, per-image MSE, both experiments, and figure saving. PyTorch is required; Matplotlib is needed only when saving figures. The exercise dataset is not included. Without the original images and split, the earlier evaluation numbers cannot be reproduced.

First, check the tensor shapes and function behavior without a dataset. Place the downloaded file in the same directory as the notebook and run the following. Python's `runpy.run_path()` executes a file's code and returns a dictionary of the names and objects defined there. This call does not execute the file's command-line training entry point.[^runpy]

```python
import runpy

lab = runpy.run_path("examples-en.py")  # Load the function and model definitions.
lab["self_check"]()  # Verify shapes, MSE, and training/evaluation with artificial tensors.
```

This check verifies the 105 parameters, `28→14→28` and `27→14→28` shapes, agreement between loop and vectorized MSE, `stack` and `unsqueeze`, the timing of the loss measurement, and preservation of parameters during evaluation. It uses artificial tensors and does not measure handwritten-digit reconstruction quality.

Actual training requires clean training images and separately held-out validation images. The example below assumes a prepared dictionary `data` containing these tensors under the keys `"clean_train"` and `"clean"`. The validation originals under `"clean"` must not have been used for training. `cpu().float()` matches the device and data type; it does not normalize values from 0–255 into 0–1. Verify that the input already uses the 0–1 range.

```python
import torch
import runpy

lab = runpy.run_path("examples-en.py")
clean_train = data["clean_train"].detach().cpu().float()
clean_val = data["clean"].detach().cpu().float()
lab["check_images"](clean_train, "clean_train")
lab["check_images"](clean_val, "clean_val")

# First train one model for Experiment A.
torch.manual_seed(2026)
model = lab["StudentDAE"]().cpu()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
history = lab["fit"](
    model, optimizer, clean_train,
    sigma=0.25, epochs=3, batch_size=128, seed=2026,
)

# A: change only the evaluation inputs for that trained model. No additional training.
rows_a = lab["evaluate_sigmas"](model, clean_val, sigmas=(0.25, 0.50))
lab["print_rows"]("A: evaluate the same model", rows_a)

# B: create a new model at each noise level and train each from scratch for three epochs.
rows_b = lab["retrain_sigmas"](
    clean_train, clean_val, sigmas=(0.25, 0.50),
    epochs=3, batch_size=128, verbose=False,
)
lab["print_rows"]("B: train a new model for each condition", rows_b)
```

This complete example trains one model for A and two models for B. **There are three training runs in total, each lasting three epochs.** Training does not occur inside A's evaluation loop.

The downloadable file's `rows_a` and `rows_b` are lists of dictionaries with named columns. Distinguish them from the earlier tuple-based `rows` used to explain the calculations. `print_rows()` follows the downloadable file's structure and displays the measured input MSE, reconstruction MSE, remaining ratio, reduction percentage, and counts of improved, worsened, and tied images.

Once the numbers are available, first check that input and reconstruction were compared against the same validation originals. Then identify whether one or several models were compared and how many training runs occurred. Those two conditions must be clear before the results table can answer the intended question.

## 18. Rust uses the same calculation order and retains the same axes

Rust's **tch** library connects Rust to **LibTorch**, PyTorch's C++ library. Function names and argument syntax differ from Python, but the sequence from clearing gradients to updating parameters and the method of retaining per-image MSE are the same.[^tch]

The corresponding core functions are below. `ModuleT` is a **trait**, a shared interface for a model that performs a forward pass with an argument selecting training or evaluation behavior. `&Tensor` borrows a tensor, and `&mut nn::Optimizer` borrows an optimizer while allowing it to be modified.

```rust
use tch::{nn, Kind, Reduction, Tensor};

fn train_one_batch(
    model: &impl nn::ModuleT,
    optimizer: &mut nn::Optimizer,
    noisy: &Tensor,
    clean: &Tensor,
) -> f64 {
    optimizer.zero_grad();               // Clear earlier gradients.
    let pred = model.forward_t(noisy, true); // true selects training behavior.
    assert_eq!(pred.size(), clean.size());
    let loss = pred.mse_loss(clean, Reduction::Mean); // Mean over all elements.
    loss.backward();                     // Calculate gradients.
    optimizer.step();                    // Update parameters.
    loss.double_value(&[])               // Scalar calculated before the update.
}

fn per_image_mse(reconstruction: &Tensor, clean: &Tensor) -> Tensor {
    assert_eq!(reconstruction.size(), clean.size());
    (reconstruction - clean)
        .square()
        .flatten(1, -1)   // Keep the image axis; combine axes 1 through the last.
        .mean_dim(
            &[1_i64][..], // Axes to average; i64 is a 64-bit integer type.
            false,       // Do not retain the reduced axis with size one.
            Kind::Float, // The float32 result type used in this example.
        )
}

fn show_loss_shapes() {
    // Numbers for checking shapes, not measured training performance.
    let losses = vec![
        Tensor::from(0.1_f32),
        Tensor::from(0.2_f32),
        Tensor::from(0.3_f32),
    ];
    let each = Tensor::stack(&losses, 0); // Three scalars → [3].
    let expanded = each.unsqueeze(1);    // Add a size-one axis → [3, 1].
    println!("{:?}, {:?}", each.size(), expanded.size());
}
```

`&[1_i64][..]` passes a list containing axis number 1 as a borrowed slice. The empty index in `double_value(&[])` reads the value of a scalar tensor with no axes. `forward_t(..., false)` selects evaluation behavior; disabling gradient recording requires a separate `tch::no_grad(...)` call.

| Python | Rust tch | Shared role |
| --- | --- | --- |
| `tensor.shape` | `tensor.size()` | Inspect the size of each axis |
| `loss.item()` | `loss.double_value(&[])` | Read a number from a scalar tensor |
| `torch.stack(losses, dim=0)` | `Tensor::stack(&losses, 0)` | Combine multiple tensors along a new axis |
| `tensor.unsqueeze(1)` | `tensor.unsqueeze(1)` | Insert a size-one axis |
| `flatten(1).mean(1)` | `flatten(1, -1).mean_dim(...)` | Calculate means while retaining the image axis |

The [complete Rust example](/denoising-autoencoder-training-evaluation/examples-en.rs) and [Cargo.toml](/denoising-autoencoder-training-evaluation/Cargo.toml) also include model creation and an executable entry point. The Rust code was checked against the official tch 0.26.0 API; it was not compiled or executed in this post's verification environment. A separate environment with the required LibTorch 2.13.0 must be configured for that version. Do not assume it can run by directly linking the PyTorch 2.15.0 used for the Python checks.[^tch-setup]

## References

The APIs were checked on November 3, 2026. Numerical results in the quoted evaluation logs come from the supplied exercise record, not performance measurements published in the official documentation below.

[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html). Model structure, calls, `train`, `eval`, and `cpu`.
[^activations]: PyTorch, [ReLU](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ReLU.html), [Sigmoid](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sigmoid.html). Activation calculations.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html). Output size and weight and bias shapes.
[^transpose]: PyTorch, [ConvTranspose2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ConvTranspose2d.html). Output size, overlap, and the distinction from an inverse function.
[^mse]: PyTorch, [MSELoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MSELoss.html). Elementwise squared errors and the averaging scope.
[^zero-grad]: PyTorch, [Optimizer.zero_grad](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.zero_grad.html). Clearing gradients and `set_to_none`.
[^item]: PyTorch, [Tensor.item](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.item.html). Returning a number from a one-element tensor.
[^python-builtins]: Python, [zip](https://docs.python.org/3/library/functions.html#zip). Pairing corresponding elements and stopping at the shorter input.
[^stack]: PyTorch, [stack](https://docs.pytorch.org/docs/2.15/generated/torch.stack.html). Combining equal-sized tensors along a new axis.
[^tuples]: Python, [Tuples](https://docs.python.org/3/library/stdtypes.html#tuples). Comma syntax for a one-element tuple.
[^unsqueeze]: PyTorch, [unsqueeze](https://docs.pytorch.org/docs/2.15/generated/torch.unsqueeze.html). Inserting a size-one axis and sharing storage.
[^flatten]: PyTorch, [flatten](https://docs.pytorch.org/docs/2.15/generated/torch.flatten.html). Combining a specified range of axes.
[^mean]: PyTorch, [mean](https://docs.pytorch.org/docs/2.15/generated/torch.mean.html). Axes to average and the result shape.
[^no-grad]: PyTorch, [no_grad](https://docs.pytorch.org/docs/2.15/generated/torch.no_grad.html). Disabling gradient tracking.
[^noise]: PyTorch, [randn](https://docs.pytorch.org/docs/2.15/generated/torch.randn.html), [clamp](https://docs.pytorch.org/docs/2.15/generated/torch.clamp.html). Standard normal random values and range restriction.
[^adam]: PyTorch, [Adam](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html). Optimizer creation and update settings.
[^randomness]: PyTorch, [Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html). Random seeds and the limits of reproducibility.
[^imshow]: Matplotlib, [imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html). Image arrays and a shared brightness range.
[^pathlib]: Python, [pathlib](https://docs.python.org/3/library/pathlib.html). Path objects and checking file existence.
[^ipython]: IPython, [Cell execution and output](https://ipython.readthedocs.io/en/stable/development/execution.html), [Suppressing returned output](https://ipython.readthedocs.io/en/stable/config/custommagics.html#silencing-the-magic-output). Final-expression display and semicolons.
[^runpy]: Python, [runpy.run_path](https://docs.python.org/3/library/runpy.html#runpy.run_path). Executing a file and returning its global namespace as a dictionary.
[^tch]: tch 0.26.0, [Tensor](https://docs.rs/tch/0.26.0/tch/struct.Tensor.html), [ModuleT](https://docs.rs/tch/0.26.0/tch/nn/trait.ModuleT.html), [Optimizer](https://docs.rs/tch/0.26.0/tch/nn/struct.Optimizer.html). Corresponding Rust APIs.
[^tch-setup]: tch 0.26.0, [Installation](https://docs.rs/crate/tch/0.26.0/source/README.md). LibTorch version and execution environment requirements.




