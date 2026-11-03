+++
title = "Denoising Autoencoders in PyTorch: From Tensor Shapes to Training and Evaluation"
description = "Follow noisy images through eight feature maps and back to reconstructed pixels. Connect transposed convolution, shared parameters, per-image MSE, split and cat, training, and validation through diagrams and code, then examine notebook state and the cause of StopIteration."
date = 2026-11-03
slug = "denoising-autoencoder-pytorch"

[extra]
katex = true
toc = true
styles = ["denoising-autoencoder-pytorch/article.css"]
+++

We want to take a noisy handwritten image and produce a result close to the clean original. The model is not given a target that identifies which pixels are noise. **It receives the noisy image and learns to reduce the difference between its output and the clean original.** The encoder and decoder perform this task together.

The [previous introduction to autoencoders](/en/autoencoder-denoising-basics/) examined how an input becomes a latent representation and a reconstruction. This time, we will process batches of images in PyTorch and connect that principle to the tensors and function calls in the code. Keep asking these questions as you read: **Are we working with image data or model weights? Which operation changes the arrangement of values, which computes the error, and which changes the weights?**

The [complete Python practice file](/denoising-autoencoder-pytorch/examples-en.py) and [Rust batching example](/denoising-autoencoder-pytorch/batching-en.rs) are provided alongside this article. The Python file runs on the `data` prepared in the original learning material. The attachment did not include the original data or the implementation of `d2_tools.py`, so this article does not invent training MSE values or successful reconstruction examples. The diagrams explain the computation; code for inspecting actual results appears later. The API descriptions were checked against official documentation on November 3, 2026.

## 1. First, connect the paths taken by the input and the target

An **autoencoder** is a neural network that computes an intermediate representation of its input and uses that representation to reconstruct data. The **encoder** converts the input into intermediate features. The **decoder** uses those features to compute an output with the original data format. **Reconstruction** means computing output values again; it does not guarantee exact recovery of the original.

A **denoising autoencoder** distinguishes the input from the target. Noise is added to the input, while the **target** used for comparison remains the clean original. We do not need a classification label saying whether the digit is a 3 or a 7. We need a noisy input paired with the clean image from which it was produced.[^dae]

The **loss** is a number measuring the difference between the output and the target. A **gradient** describes the direction and sensitivity of the loss change when a current weight changes slightly. **Backpropagation** computes these gradients, and an **optimizer** uses them to update the learnable numbers. With these roles distinguished, we can follow the full diagram.

<figure class="dae-figure">
<img src="/denoising-autoencoder-pytorch/overview-en.svg" alt="A clean original produces a noisy input. The noisy input passes through the encoder and decoder to become a reconstructed image, while the clean original supplies the loss target. Backpropagation computes gradients, and the optimizer updates both encoder and decoder parameters." loading="lazy">
<figcaption>The original is used to create the noisy input and also remains the reference for comparing the reconstruction. Both parts of the model learn from one final loss.</figcaption>
</figure>

We can describe this model through five distinct parts.

| What to identify | In this exercise | What changes during execution? |
| --- | --- | --- |
| Input | Noisy grayscale images | Each training batch uses different images and noise. |
| Neurons and layer structure | Convolution → ReLU → transposed convolution → Sigmoid | The same structure runs repeatedly. Output values at each position come from weighted sums and activation calculations. |
| Intermediate values | Eight feature maps produced by the encoder | They are computed from the current input and current weights on each call. |
| Weights and biases | Learnable numbers registered in the two convolution layers | The optimizer updates them. |
| Final output | Reconstructed images with the same size as the input | These are the current model's results for the input. |

The intermediate feature maps are not the model's weights. Weights are stored in the model and reused across images; feature maps are results obtained by processing particular images. There is also no separate third “denoising model” between the encoder and decoder. Denoising comes from **training the whole model to reconstruct the clean original from the noisy input**.

## 2. Retrieve a tensor from the dictionary, then select positions within it

The exercise's `data` is a **dictionary**, which lets us retrieve data by name. The strings used as names are **keys**. The image data retrieved here is a **tensor**, which stores numbers in a defined structure.

| Code | Data retrieved | Purpose |
| --- | --- | --- |
| `data["clean_train"]` | Clean training images | Create noisy training inputs and supply the training targets. |
| `data["clean"]` | Clean validation images | Use originals excluded from training as evaluation targets. |
| `data["noisy"]` | Noisy versions of the validation originals | Reuse the same inputs to compare the model as it changes. |

A key named `clean` does not guarantee that training and validation data have actually been separated. The originals must be separated during data preparation. In addition, `data["noisy"][i]` and `data["clean"][i]` must correspond to the same original image.

### Dimension number, dimension size, and index are different

A **dimension**, or **axis**, identifies a way of distinguishing the data, such as image number, channel number, or spatial position. The **shape** lists the sizes of those dimensions. An **index** selects a particular position along a dimension, starting at 0.

We will use `N` for the total number of images and `B` for the number of images in the **batch** processed by one model call. Each image is grayscale, with one channel, height 28, and width 28. The complete validation data has shape `(N, 1, 28, 28)`, and the current batch has shape `(B, 1, 28, 28)`.

| Dimension number | Meaning | Size in the complete dataset | Valid indices |
| ---: | --- | ---: | --- |
| `dim=0` | Image number | `N` | `0` through `N-1` |
| `dim=1` | Channel number | `1` | `0` |
| `dim=2` | Height position | `28` | `0` through `27` |
| `dim=3` | Width position | `28` | `0` through `27` |

A batch shape of `(64, 1, 28, 28)` means that the batch contains 64 images. Its last image index is 63. The channel dimension does not disappear just because the images are grayscale.

### Why is `data["clean"][0, 0, 5, 10]` not five-dimensional?

The first pair of brackets, `["clean"]`, **retrieves a tensor from the dictionary**. The second, `[0, 0, 5, 10]`, **selects positions along that tensor's four dimensions**. The string key is not a tensor dimension.

| Selection | What remains | Shape |
| --- | --- | --- |
| `data["clean"]` | All validation images | `(N, 1, 28, 28)` |
| `data["clean"][0]` | Image 0 | `(1, 28, 28)` |
| `data["clean"][0, 0]` | Image 0's grayscale channel | `(28, 28)` |
| `data["clean"][0, 0, 5]` | Width positions at height index 5 | `(28,)` |
| `data["clean"][0, 0, 5, 10]` | The pixel at that position | `()` |

A tensor containing one number with `shape=()` is a **scalar tensor**. Calling `.item()` retrieves its value as a Python number. Shape `(1,)` has one dimension of size 1, whereas shape `()` has no dimensions.

```python
import torch
from torch import nn  # Use torch.nn's neural-network tools under the name nn.

# The original material's data-preparation cell must already have created data.
for name in ("clean_train", "clean", "noisy"):
    images = data[name]  # Retrieve the image tensor using its string key.
    print(name, type(images), tuple(images.shape), images.dtype, images.device)

# The stop index 5000 is excluded. Select image indices 0 through 4999.
clean_train = data["clean_train"][:5000].to(device="cpu", dtype=torch.float32)
clean_val = data["clean"].to(device="cpu", dtype=torch.float32)
noisy_val = data["noisy"].to(device="cpu", dtype=torch.float32)

print(tuple(clean_val[0, 0, 5, 10].shape))  # ()
print(clean_val[0, 0, 5, 10].item())       # This pixel's value as a Python number.
```

The `dtype` is the numeric data type, and the `device` is the computation device. This exercise uses `float32` on the CPU. Calling `.to(dtype=torch.float32)` changes the data type; it does not automatically convert pixels from a 0–255 range to a 0–1 range. If the input is actually defined on 0–255, check that definition and normalize it appropriately, for example by dividing by 255. Dividing data that is already in 0–1 by 255 again would give the wrong range.

The complete practice file checks image shape, empty data, non-finite values, and the 0–1 range. **Checking equal shapes does not verify that images are paired with the correct targets.** Confirm through the creation process or image IDs that the same original occupies the same index in each tensor.

## 3. Separate code that constructs the model from code that processes images

`torch.nn` provides neural-network layers and loss tools. Its `nn.Module` is the base class that manages a model's submodules and parameters. The `__init__()` method runs when a model object is created, while `forward()` defines the sequence of computations from input to output.[^module]

Here, the constructor accepts `feature_channels`, the encoder's number of output channels. Its default is 8, matching the original exercise. Setting it to 4 will let us construct the corresponding four-channel model later.

```python
class LabDenoiser(nn.Module):
    def __init__(self, feature_channels=8):
        super().__init__()  # Initialize the management of submodules and parameters.

        self.encoder = nn.Sequential(
            # in_channels=1, out_channels=feature_channels, kernel_size=3.
            # stride=2: step by two positions; padding=1: one zero-padded position on each side.
            nn.Conv2d(1, feature_channels, 3, stride=2, padding=1),
            nn.ReLU(),  # Replace negative values with 0; preserve positive values and shape.
        )
        self.decoder = nn.Sequential(
            # Use the input feature channels to compute one output channel.
            # Transposed convolution with a 4×4 kernel, stride=2, and padding=1.
            nn.ConvTranspose2d(feature_channels, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),  # Map output values into the 0–1 range.
        )

    def forward(self, x):
        features = self.encoder(x)    # Compute intermediate features from the input.
        return self.decoder(features) # Compute reconstructed images from those features.

model = LabDenoiser(feature_channels=8).cpu()  # Prepare the layers and parameters.
with torch.no_grad():  # Disable autograd recording: we only need to inspect the shape.
    reconstructed = model(noisy_val[:2])     # Pass actual images to the prepared model.
print(tuple(reconstructed.shape))            # (2,1,28,28) if validation has at least two images.
```

The arguments to `nn.Conv2d(...)` specify channel counts and kernel settings. They do not supply the images. In contrast, `model(noisy_val[:2])` receives an actual image tensor. The slice `[:2]` excludes stop index 2, so it selects images 0 and 1.

For ordinary execution, use `model(x)` rather than calling `model.forward(x)` directly. The `nn.Module` call mechanism handles `forward()` together with additional registered functionality.[^module]

### What goes inside `nn.Sequential`?

`nn.Sequential` registers layer objects in order. When called, it passes each layer's result to the next layer. Construction receives a `Conv2d` object and a `ReLU` object; the actual image data `x` arrives later in `encoder(x)`.[^sequential]

```python
# These two lines expand the calculation performed by model.encoder(x).
conv_result = model.encoder[0](noisy_val[:2])  # First registered layer: Conv2d.
features = model.encoder[1](conv_result)      # Second registered layer: ReLU.
```

`Sequential` is useful when layers are simply connected in sequence. If the path branches, as in a residual connection that preserves an input and adds it later, write that flow explicitly in `forward()`. The name `features` refers to a computed result; it does not declare a new layer or new weights.

## 4. Read image count, feature-map count, and weight shape separately

A **feature map** stores convolution results at their spatial positions. In an ordinary `Conv2d`, one **filter** uses all input channels to produce one output channel. This encoder has eight filters, so it computes eight feature maps for each image.

<figure class="dae-figure">
<img src="/denoising-autoencoder-pytorch/shapes-en.svg" alt="A batch of B input images has shape (B,1,28,28), the encoder features have shape (B,8,14,14), and the reconstruction has shape (B,1,28,28). Encoder weights have shape (8,1,3,3), decoder weights have shape (8,1,4,4), and their biases have shapes (8,) and (1,)." loading="lazy">
<figcaption>The image count B stays the same throughout. Feature maps are computed for each image, while weights are shared across images and spatial positions.</figcaption>
</figure>

| Stage | Data shape | What the values represent |
| --- | --- | --- |
| Noisy input | `(B, 1, 28, 28)` | B images, each with one grayscale channel |
| `Conv2d` | `(B, 8, 14, 14)` | Weighted sums computed by eight filters |
| `ReLU` | `(B, 8, 14, 14)` | Intermediate features after negative sums are replaced with 0 |
| `ConvTranspose2d` | `(B, 1, 28, 28)` | Weighted sums used to produce reconstructed pixels |
| `Sigmoid` | `(B, 1, 28, 28)` | Reconstructed pixel values in the 0–1 range |

An **activation function**, such as ReLU or Sigmoid, is applied to a weighted sum. This model's ReLU and Sigmoid have no learnable parameters, but backpropagation must still account for how changes pass through these functions.

### Does reducing spatial size always compress the data?

We are counting **tensor elements used to represent one image**. Multiply the channel count by the height and width of each channel. The input has one `28×28` channel; the intermediate features have eight `14×14` channels.

| Representation | Elements per image |
| --- | ---: |
| Input | `1 × 28 × 28 = 784` |
| Intermediate features | `8 × 14 × 14 = 1,568` |

**The spatial resolution decreases, but the total element count increases.** Describing this model as necessarily compressing the input into fewer numbers would be inaccurate. With four feature channels, there are `4×14×14=784` elements, matching the input's element count. That still does not mean the computed values equal the original values.

## 5. Transposed convolution adds computed values at overlapping output positions

**Transposed convolution** uses a learnable kernel to compute values at output positions. Depending on its settings, it can increase output height and width. It is not an inverse function that automatically cancels the encoder's computation. The encoder and decoder also learn separate weights.[^transpose]

The current decoder uses `ConvTranspose2d(8, 1, 4, stride=2, padding=1)`. Its input contains eight `14×14` feature maps, and its output is one `28×28` channel. Begin with one input position in one channel.

1. Multiply the value at that position by each weight in that channel's `4×4` kernel.
2. Add the 16 products to their corresponding `4×4` output positions.
3. The neighboring input position uses the same kernel. With `stride=2`, its kernel placement in the output is shifted by two positions.
4. When both calculations reach the same output position, add their values together.
5. The other input channels calculate products with their own kernels and add them at the same output positions. Finally, add the output channel's bias.

These “values to add” are products of an input value and kernel weights. They are not a separate kind of number. Multiple multiplication results reach an overlapping position, so those results are added together.

### Why do we add the kernel length of 4 only once?

Consider just one spatial direction: height or width. The input length is 14, so its indices are `0–13`. Before adjusting the boundaries, the kernel starting positions associated with these inputs are as follows.

| Input index | Kernel start position | Output positions occupied by the kernel |
| ---: | ---: | --- |
| 0 | 0 | 0, 1, 2, 3 |
| 1 | 2 | 2, 3, 4, 5 |
| 2 | 4 | 4, 5, 6, 7 |
| 12 | 24 | 24, 25, 26, 27 |
| 13 | 26 | 26, 27, 28, 29 |

<figure class="dae-figure">
<img src="/denoising-autoencoder-pytorch/transpose-overlap-en.svg" alt="Placing length-four kernels two positions apart gives positions 0–3 for the first kernel, 2–5 for the next, and 26–29 for the last. Removing one position at each boundary from the total length of 30 leaves an output length of 28." loading="lazy">
<figcaption>The first kernel occupies four positions. Later kernels overlap earlier ones, extending the total length by two positions each. This diagram shows positions and overlap, not data values.</figcaption>
</figure>

The first kernel occupies a length of 4. There are 13 remaining input positions, and each extends the total length by 2. The length before boundary adjustment is therefore `4 + 13×2 = 30`. Counting from the final kernel gives the same result: starting position 26 plus length 4 equals 30. The first kernel's length has not been omitted.

For the current `padding=1` setting, we can understand the connections by excluding one position at each end: `30-1-1=28`. This is an equivalent explanation of the connections and size. It does not mean the implementation must allocate a larger tensor and then crop it.

### After locating the positions, summarize the size calculation

Let `L` be the input length along one direction, `k` the kernel length in that direction, `s` the stride, and `p` the padding. With `dilation=1` and transposed convolution's `output_padding=0`, as in this exercise, use the following relationships. The `floor` operation rounds down to the largest integer that does not exceed the value.[^conv][^transpose]

| Operation | Output length | Applied to this model |
| --- | --- | --- |
| Ordinary convolution | `floor((L + 2p - k) / s) + 1` | Input 28, kernel 3, stride 2, padding 1 → 14 |
| Transposed convolution | `(L - 1) × s - 2p + k` | Input 14, kernel 4, stride 2, padding 1 → 28 |

Do not infer that an output is always twice as large from `stride=2` alone. Kernel size and padding also affect output length. This model was designed for `28×28` inputs; it does not guarantee equal input and output sizes for arbitrary other image sizes.

<details>
<summary>General formula with dilation and output_padding, for later reference</summary>

**Dilation** is the spacing between kernel elements and defaults to 1. Writing dilation as `d` and transposed convolution's `output_padding` as `o`, the output length is `(L-1)×s - 2p + d×(k-1) + o + 1`.

The `output_padding` setting selects among possible output sizes. It should not be understood as simply appending zero-valued pixels to the output. Here, `d=1` and `o=0`, so the simpler formula above is sufficient. In practice, check both the target output shape and the conditions in the official documentation.

</details>

### Is transposed convolution the only way to increase spatial size?

**Upsampling** increases height and width. **Interpolation** calculates values at new positions from existing values according to a fixed rule. The `nearest` method selects a nearby existing value.

```python
# This is an alternative decoder for comparison, not a change to the original model.
alternative_decoder = nn.Sequential(
    # Double height and width without learnable weights.
    nn.Upsample(scale_factor=2, mode="nearest"),  # (B,8,14,14) → (B,8,28,28)
    nn.Conv2d(8, 1, 3, padding=1),                # (B,8,28,28) → (B,1,28,28)
    nn.Sigmoid(),
)
```

Transposed convolution uses a learned kernel to calculate values at the expanded output positions. The alternative first enlarges the spatial dimensions according to a fixed rule, then uses learned weights in a convolution. Equal output shapes do not imply equal computations or output values.[^upsample]

Images produced with transposed convolution can contain **checkerboard artifacts**, artificial grid-like patterns. Separating interpolation from convolution is one alternative for reducing them. The current kernel size of 4 is divisible by the stride of 2, reducing one source of uneven overlap, but that alone does not guarantee good reconstruction or the absence of artifacts.[^checkerboard]

## 6. Having 209 parameters does not mean performing only 209 calculations

A **parameter** is a learnable number stored in the model. Here, parameters include **weights**, which multiply input values, and **biases**, which are added to weighted sums. Reusing the same kernel across spatial positions and images is called **weight sharing**.

First, count the required connections independently of the image count. With the current `groups=1` setting, each input-channel/output-channel pair has a spatial kernel. If the decoder has eight input channels and one output channel, it needs eight `4×4` kernels. With two output channels, each output needs its own set of eight kernels, giving 16 kernels in total. Each output channel also has one bias.[^conv][^transpose]

| Layer | Weight shape | Weight count | Bias shape and count | Total |
| --- | --- | ---: | --- | ---: |
| Encoder `Conv2d` | `(8, 1, 3, 3)` | `8×1×3×3=72` | `(8,)`, 8 values | 80 |
| Decoder `ConvTranspose2d` | `(8, 1, 4, 4)` | `8×1×4×4=128` | `(1,)`, 1 value | 129 |
| `ReLU`, `Sigmoid` | None | 0 | None | 0 |
| Complete model | Four parameter tensors | 200 | 9 values | **209** |

Both weight shapes begin with `(8, 1, …, …)`, which can make their dimension order look identical. The numbers happen to match in this model, but their meanings differ.

| Layer | Weight dimension order when `groups=1` |
| --- | --- |
| `Conv2d` | `(output channels, input channels, kernel height, kernel width)` |
| `ConvTranspose2d` | `(input channels, output channels, kernel height, kernel width)` |

Processing B images does not create a separate set of 209 weights for each image. It uses the same 209 parameters more times. We must therefore distinguish **parameter elements, parameter tensors, actual multiplication and addition operations, and weight-update steps**. Parameter count alone also does not determine speed or total memory use. Intermediate feature maps and backward computations require resources too.

### When counting parameters in code, where does `part` come from?

The original compact code uses a dictionary comprehension. A **comprehension** constructs a collection through iteration. First, produce the same result with ordinary loops.

```python
counts = {}

for name, part in [("encoder", model.encoder), ("decoder", model.decoder)]:
    # First iteration: name="encoder", part=model.encoder.
    # Second iteration: name="decoder", part=model.decoder.
    total = 0
    for parameter in part.parameters():  # Receive registered parameter tensors in turn.
        total += parameter.numel()       # numel(): total number of elements in a tensor.
    counts[name] = total

print(counts)                 # {'encoder': 80, 'decoder': 129}
print(sum(counts.values()))   # Add the values 80 and 129 to get 209.

# Write the same count using a comprehension.
counts = {
    name: sum(p.numel() for p in part.parameters())
    for name, part in [("encoder", model.encoder), ("decoder", model.decoder)]
}
```

The clause `for name, part in ...` retrieves each tuple and assigns its values to the two variables. No earlier declaration of `part` is needed. Similarly, the inner `p` receives values from `for p in part.parameters()`. Python treats `part` and `Part` as different names. Unlike an ordinary `for` loop, a comprehension does not introduce its iteration variables into the surrounding scope.[^comprehension]

The word `part` does not inherently mean “part of a model.” Later, in `for part in noisy.split(256)`, it refers to a batch tensor. **A variable's role depends on the value assigned to it.**

The `parameters()` method also includes parameters frozen with `requires_grad=False`. To count only parameters enabled for training, add a condition: `sum(p.numel() for p in model.parameters() if p.requires_grad)`. All four parameter tensors in the current model are enabled for training.[^module]

## 7. Subtract corresponding image positions without flattening first

Comparing a reconstructed image with its original means comparing corresponding pixel values. The image index `i`, channel index `c`, height index `h`, and width index `w` must match.

`reconstructed[i, c, h, w] - clean[i, c, h, w]`

For example, `[0, 0, 5, 10]` selects the pixel in image 0 that we examined earlier. Subtract the original pixel at that position from the reconstructed pixel at the same position. Do not mix in pixels from different images.

An **element-wise operation** separately computes a result for each pair of values at matching positions. Two tensors with the same shape can be subtracted directly while retaining their `(B, 1, 28, 28)` structure. Squaring each difference also preserves that shape.

```python
def require_same_shape(prediction, target):
    # This reconstruction task requires the two shapes to match exactly.
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape={tuple(prediction.shape)}, target shape={tuple(target.shape)}"
        )

with torch.no_grad():
    example_clean = clean_val[:2]
    example_recon = model(noisy_val[:2])
    require_same_shape(example_recon, example_clean)

    difference = example_recon - example_clean  # (B,1,28,28): per-pixel differences.
    squared_error = difference.square()        # (B,1,28,28): square each difference.
    mse_all = squared_error.mean()             # (): mean over all elements.
```

**Mean squared error (MSE)** averages the squared differences between corresponding values. The default `nn.MSELoss()` setting is `reduction="mean"`, which averages all elements of the supplied tensor. Its divisor includes the channel, height, and width dimensions as well as image count; it does not divide by image count alone.[^mse]

| Setting | What is returned? | Shape in this exercise |
| --- | --- | --- |
| `reduction="none"` | Per-pixel squared errors | `(B, 1, 28, 28)` |
| `reduction="sum"` | Sum of all squared errors | `()` |
| `reduction="mean"` | Mean of all squared errors | `()` |

There are also two stages in `nn.MSELoss()(prediction, target)`. The first call, `nn.MSELoss()`, creates a loss object. The second passes the prediction and target to that object. It is equivalent to writing `loss_fn = nn.MSELoss()` followed by `loss_fn(prediction, target)`.

### To keep one error value for each image

The **flatten** operation combines specified dimensions into one. In `flatten(1)`, the 1 is **the dimension number at which flattening starts**, not a dimension size. It preserves image dimension 0 and combines the channel, height, and width dimensions. Each image has `1×28×28=784` pixels, so the result has shape `(B, 784)`.[^flatten]

Then `mean(1)` averages the 784 values within each image. The final shape is `(B,)`, with one error per image. The `flatten` operation does not calculate a mean; `mean` does. They perform different tasks.

<figure class="dae-figure">
<img src="/denoising-autoencoder-pytorch/mse-reduction-en.svg" alt="Subtracting corresponding reconstructed and target values, then squaring, produces pixel errors of shape (B,1,28,28). A mean over all elements returns a scalar. flatten(1) produces (B,784), and mean(1) produces per-image MSE values of shape (B,)." loading="lazy">
<figcaption>Keep the image dimension to inspect each image's error. The overall mean and per-image means aggregate the same pixel errors over different sets of dimensions.</figcaption>
</figure>

```python
def image_mse(prediction, target):
    require_same_shape(prediction, target)
    pixel_errors = (prediction - target).square()  # (B,1,28,28)
    return pixel_errors.flatten(1).mean(1)        # (B,784) → (B,)

# Averaging only channels, height, and width also gives per-image MSE without flattening.
per_image_a = squared_error.flatten(1).mean(1)
per_image_b = squared_error.mean(dim=(1, 2, 3))

# allclose allows small differences caused by floating-point rounding.
assert torch.allclose(per_image_a, per_image_b)
assert torch.allclose(per_image_a.mean(), mse_all)
```

These `assert` statements check the computation while learning. Python can remove them in optimized execution, so input checks that must remain active should use `if` and exceptions, as in `require_same_shape()` above. See the [Python assert documentation](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement).

Because all images have the same size, averaging the per-image MSE values gives the same result as averaging all pixels at once. Each image contributes the same number of pixels. Later, we will distinguish this relationship from combining the means of **batches with different sizes**.

If the starting dimension is omitted, as in `flatten()`, the image dimension is combined too, producing `(B×784,)`. The numbers are still present, but the dimension that distinguished images has disappeared. For per-image evaluation, deliberately preserve the image dimension with `flatten(1)`.

### A subtraction can run successfully and still compare the wrong images

**Broadcasting** allows differently sized tensors to be matched for computation according to defined rules. Dimensions are matched from the end, and missing leading dimensions are treated as having size 1.[^broadcast]

| Prediction shape | Target shape | Subtraction result | Interpretation |
| --- | --- | --- | --- |
| `(B,1,28,28)` | `(B,1,28,28)` | `(B,1,28,28)` | Pixel comparisons for the intended image pairs |
| `(B,1,28,28)` | `(1,1,28,28)` | `(B,1,28,28)` | One target image is reused for every prediction |
| `(B,1,28,28)` | `(B,28,28)` | `(B,B,28,28)` | Comparisons also occur between different images |

In the last case, the target is aligned as if it had a leading size-one dimension: `(1,B,28,28)`. The prediction's first B dimension and the target's second B dimension both remain, producing `(B,B,28,28)`. **A subtraction executing successfully does not establish that it compared the correct image pairs.** This is why we check shapes immediately before calculating loss.

## 8. Identify the line that actually changes weights in one training batch

A **forward pass** computes the output from the input. After producing a reconstruction, we calculate loss, use backpropagation to compute gradients, and let the optimizer update the parameters. Gradients do not specify an optimal distance to move; the update rule depends on the optimizer and its **learning rate** setting.

```python
def train_step(net, optimizer, noisy_images, clean_images):
    net.train()  # Set training mode. This call alone does not change weights.
    optimizer.zero_grad(set_to_none=True)  # Reset previous .grad values to None.

    reconstructed = net(noisy_images)      # Forward pass: noisy images → reconstructions.
    require_same_shape(reconstructed, clean_images)
    error = nn.MSELoss()(reconstructed, clean_images)  # Mean error over all pixels.

    error.backward()  # Compute gradients and accumulate them in parameter .grad fields.
    optimizer.step()  # Update weights and biases using gradients and optimizer state.

    return error.item()  # Retrieve the already-computed loss as a Python number for logging.
```

| Code | What it changes or creates | Does it update the weights themselves? |
| --- | --- | --- |
| `net.train()` | Model operating mode | No |
| `zero_grad(set_to_none=True)` | Previously stored gradients | No |
| `net(noisy_images)` | Reconstructions and computation records needed for backward | No |
| `MSELoss` | Loss comparing outputs with targets | No |
| `error.backward()` | Parameter gradients | No |
| `optimizer.step()` | Model weights and biases, and optimizer state | **Yes** |

PyTorch accumulates gradients. To use fresh gradients for each batch, as we do here, clear the previous ones. The `zero_grad()` method does not set the weights to zero. Its `set_to_none=True` argument resets stored gradients to `None`.[^zero][^autograd]

The current model has 209 parameter elements. After backpropagation, the relevant gradients are stored in the same shapes as their parameters. Computing those values, however, must account for the weights' use across many images and spatial positions. **The effects of computation paths that reuse a weight are combined into one gradient for that weight.** Producing 209 gradient values does not mean performing only 209 operations.

Defining the function does not start training either. We must call `train_step(model, optimizer, noisy, clean)` with an actual batch. Also, although the return statement follows `step()`, `error` is the **loss calculated by the forward pass before the update**. Returning it later does not change when it was measured.

The original `error.detach().item()` serves the same logging purpose. The `detach()` method returns a tensor disconnected from autograd, while `item()` returns the Python value of a one-element tensor. Since we only return a Python number here, `error.item()` is sufficient.[^item][^detach]

## 9. `split → model → cat` processes smaller batches, then joins them in order

Processing all validation images at once also computes their intermediate feature maps at once. Dividing inference into smaller batches can reduce the intermediate memory needed for one model call.

To read the original `torch.cat([model(part) for part in noisy.split(256)])`, separate its three operations.

| Operation | What it does | Return value |
| --- | --- | --- |
| `noisy.split(256, dim=0)` | Divide the image dimension into batches of at most 256 images. | A tuple of tensor chunks |
| `model(part)` | Reconstruct the current image batch. | A reconstruction tensor with the same batch size |
| `torch.cat(results, dim=0)` | Join results along the existing image dimension. | The complete reconstruction tensor |

The chunks returned by `split` are **views**, sharing storage with the original tensor. This code reads each chunk as model input. If the dimension length is not divisible by 256, the final chunk is smaller.[^split]

The diagram uses **600 images only as an explanatory example**, not as the actual validation count. Check the actual count with `noisy_val.shape[0]`.

<figure class="dae-figure">
<img src="/denoising-autoencoder-pytorch/split-cat-en.svg" alt="An illustrative set of 600 images is split into batches of 256, 256, and 88. The same model reconstructs each batch. Concatenating along dim=0 produces a reconstruction tensor of shape (600,1,28,28) in the original order." loading="lazy">
<figcaption>The first image in batch 1 has overall index 256, and the first in batch 2 has overall index 512. Processing and joining batches in order preserves their correspondence with the originals.</figcaption>
</figure>

```python
def predict_fixed(model, noisy, batch_size=256):
    if noisy.shape[0] == 0:
        raise ValueError("There are no images to reconstruct.")

    model.eval()  # Switch to evaluation mode.
    with torch.no_grad():  # Do not record computations for gradient calculation.
        results = []
        for part in noisy.split(batch_size, dim=0):
            reconstructed = model(part)
            results.append(reconstructed)
        return torch.cat(results, dim=0)  # Join along the existing image dimension.
```

The list comprehension is a compact way to write this loop and `append`. The `cat` operation does not add or average pixel values; it joins the sequences of images. In contrast, `stack` adds a new dimension and requires equal input shapes. Since the last batch can be smaller, as in 256, 256, and 88 images, `cat` is the appropriate operation here.[^cat]

Memory is still required for the original input, each batch's output, and the final concatenated result. If only MSE is needed, rather than all reconstructed images, we can instead accumulate squared-error sums and element counts batch by batch.

### Why use both `eval()` and `no_grad()`?

The `eval()` method sets layers to evaluation mode. It affects layers whose behavior depends on the mode, such as Dropout, which randomly excludes some values during training, and BatchNorm, which uses normalization statistics. The `no_grad()` context disables autograd recording. These are independent functions.[^autograd]

| Setting | Layer training/evaluation mode | Records for gradient calculation |
| --- | --- | --- |
| Only `model.eval()` | Switches to evaluation mode | Not disabled by this call |
| Only `torch.no_grad()` | Preserves the existing mode | Disabled within the context |
| Both | Evaluation mode | Disabled |

The current model has neither Dropout nor BatchNorm, so switching between `train()` and `eval()` alone does not change its output computation. Separating these responsibilities still makes the intent easier to preserve when other layers are added.

The function name `predict_fixed` does not itself fix the input. The comparison uses fixed inputs because the same `noisy_val` is passed repeatedly. The function leaves the model in evaluation mode; the next `train_step()` changes it back with `net.train()`.

## 10. Establish a clear initial state before training

A **seed** specifies the starting state of random-number generation. It helps reproduce the same initialization and sequence of random operations in the same environment. A seed alone cannot guarantee identical results across PyTorch versions, devices, or platforms.[^randomness]

**Adam** is an optimizer that uses internal state related to previous gradients as well as the current gradients. Its `lr` argument is the learning rate. The value `0.01` below comes from the original exercise; it is not a suitable setting for every dataset or model.[^adam]

```python
# SEED is the integer chosen in the original exercise.
# Running this cell starts an experiment with a new model and a new optimizer.
torch.manual_seed(SEED)
model = LabDenoiser(feature_channels=8).cpu()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Manage separate random-number streams for the two purposes.
shuffle = torch.Generator().manual_seed(SEED + 4000)
noise_rng = torch.Generator().manual_seed(SEED + 5000)

# Reconstruct the fixed validation inputs before training.
initial_recon = predict_fixed(model, noisy_val)
initial_mse = image_mse(initial_recon, clean_val).mean().item()
history = [[0, None, initial_mse]]  # [epoch, training MSE, validation MSE].
```

The `model.parameters()` call supplies the newly created model's parameters to the optimizer. The offsets in `SEED+4000` and `SEED+5000` have no special autoencoder meaning. They are choices for starting separate random streams for shuffling and noise. Each use advances a generator's state, so successive calls do not return the same random numbers.

The `history` list stores epoch numbers and loss values, not reconstructed images. The first record's `None` means that no training error has yet been measured; it does not mean zero error. The `initial_recon` tensor is the newly created model's output before training. If we keep an already-trained model and rerun only the evaluation code, its output must not be described as an untrained model's output.

The source expression `float((initial_recon-clean_val).square().mean())` also converts a scalar tensor into a Python floating-point number for logging. It serves the same purpose as `.item()` here; neither operation performs another training step.

To inspect intermediate shapes, run the registered layers one at a time as below. Ordinary inference only needs `model(x)`; this code is for examining the internal connections.

```python
with torch.no_grad():
    h = noisy_val[:2]
    print("input", tuple(h.shape))
    for name, part in [("encoder", model.encoder), ("decoder", model.decoder)]:
        for layer in part:
            h = layer(h)  # Use the current result as the next layer's input.
            # :<16 left-aligns the layer name in a field at least 16 characters wide.
            print(f"{name} {layer.__class__.__name__:<16}", tuple(h.shape))
```

With at least two validation images, the shapes are `(2,1,28,28) → (2,8,14,14) → (2,8,14,14) → (2,1,28,28) → (2,1,28,28)`. This forward-pass inspection alone does not train the weights.

## 11. Shuffle 5,000 images into batches and train for three epochs

A **training step** calculates loss and gradients for one batch and updates the parameters once. An **epoch** uses each image in the current training dataset once for training.

This exercise uses 5,000 training images, batches of at most 64 images, and three epochs. Seventy-eight full batches process 4,992 images, and the last batch processes the remaining 8. Each epoch therefore has 79 steps, giving **237 calls to `optimizer.step()`** over three epochs. The value 256 used in validation is an inference batch size, separate from the training batch size.

### First shuffle indices, then retrieve images using those indices

The call `torch.randperm(5000)` creates a tensor of shape `(5000,)` containing the indices `0–4999` in a shuffled order without repetition. Applying `.split(64)` divides these **image indices**, not the images themselves, into groups of at most 64. Then `clean_train[ids]` selects the actual images at those indices.[^randperm]

| Code | Contents | Shape |
| --- | --- | --- |
| `shuffled_ids` | 5,000 shuffled image indices | `(5000,)` |
| `ids` in a full batch | 64 indices to select | `(64,)` |
| `clean_train[ids]` | The 64 selected images | `(64,1,28,28)` |
| `ids` in the last batch | The remaining 8 indices | `(8,)` |
| Last `clean_train[ids]` | The 8 selected images | `(8,1,28,28)` |

Validation split an image tensor; here we split an index tensor. The rule used by `split` is the same, while the meaning of the data being split is different.

### Generate noise with the same shape as the originals

The call `torch.randn(clean.shape)` draws values from a normal distribution with mean 0 and standard deviation 1, arranged in the requested shape. **Standard deviation** measures how widely values are distributed around the mean. Multiplying by 0.25 gives the noise a standard deviation of 0.25 before it is added.[^randn]

The operation `clean + noise` adds each noise value to the corresponding original position. Then `.clamp(0, 1)` sets values below 0 to 0 and values above 1 to 1, leaving values within the range unchanged. Because of this clipping, the final `noisy-clean` differences do not necessarily retain the original noise distribution or standard deviation.[^clamp]

```python
for epoch in range(1, 4):  # Iterates over 1, 2, 3. The stop value 4 is excluded.
    sse, count = 0.0, 0
    shuffled_ids = torch.randperm(len(clean_train), generator=shuffle)

    for ids in shuffled_ids.split(64):
        clean = clean_train[ids]  # Usually (64,1,28,28); the last batch is (8,1,28,28).
        noise = 0.25 * torch.randn(
            clean.shape,
            generator=noise_rng,  # Use the generator dedicated to noise.
            dtype=clean.dtype,    # Match the originals' numeric data type.
            device=clean.device,  # Match the originals' computation device.
        )
        noisy = (clean + noise).clamp(0, 1)
        loss = train_step(model, optimizer, noisy, clean)

        # These are logging statistics: combine means correctly across unequal batch sizes.
        sse += loss * clean.numel()
        count += clean.numel()

    train_mse = sse / count
    recon = predict_fixed(model, noisy_val)  # Same validation inputs; no weight updates.
    validation_mse = image_mse(recon, clean_val).mean().item()
    history.append([epoch, train_mse, validation_mse])
    print(f"epoch {epoch}: train={train_mse:.4f}, validation={validation_mse:.4f}")
```

The same original image is paired with new noise in the next epoch. Validation, however, keeps using the same `noisy_val`. The `.4f` format controls only the number of decimal places displayed. It does not round the numbers stored in `history` to that precision.

This exercise directly shuffles and splits tensors already in memory. With larger datasets, a **Dataset** can define how to read an individual example, while a **DataLoader** manages batching, shuffling, and loading. The tools can change while the sequence remains the same: prepare a batch, run the forward pass, compute loss, backpropagate, and update.[^dataloader]

## 12. Combining batch means also requires their element counts

The preceding `loss` is the current batch's mean squared error. The value `clean.numel()` counts the pixel elements used in that mean. To obtain the **sum of squared errors (SSE)**, multiply the mean by the element count.

`Batch MSE = that batch's sum of squared errors / that batch's element count`

Therefore, `loss * clean.numel()` corresponds to the batch's squared-error sum. Accumulate these sums and their element counts separately, then divide to obtain the mean over all processed pixels. Actual floating-point computations can have small rounding differences.

| Batch | Image count | Pixel count used in the mean |
| --- | ---: | ---: |
| Full batch | 64 | `64×1×28×28 = 50,176` |
| Last batch | 8 | `8×1×28×28 = 6,272` |

If we average those batch MSE values with equal weight, the eight-image batch receives the same weight as the 64-image batch. Each image in the final batch then has eight times the relative weight. We want **the average squared error over all processed pixels**, rather than an equally weighted average of batch groups.

A mean that weights each group by its element count is a **weighted mean**. When all images have the same size, weighting by image count gives the same result: the common 784 pixels per image cancel from the numerator and denominator. For other loss functions, first check what quantity their averaging divides by.

This also marks an important distinction between learning and logging. The `train_step()` function has already completed `backward()` and `step()`. The `sse` and `count` variables combine the measured values for reporting. **This aggregation does not also correct the final small batch's relative contribution to parameter updates.**

### Training and validation MSE are measured at different times too

Each `history` record contains `[epoch number, training MSE, validation MSE]`. The entries `record[0]`, `record[1]`, and `record[2]` are the epoch, training error, and validation error respectively. The initial record's training value is `None`, so exclude that record from the training curve.

| Metric | Which data is used? | Which model state is used? |
| --- | --- | --- |
| Training MSE | Training images with fresh noise for each batch | Multiple model states as parameters change during the epoch |
| Validation MSE | Fixed noisy validation images | One model state at the end of the epoch |

Training MSE is different from reevaluating the training data with the final model after the epoch. A lower validation MSE than training MSE is therefore not enough to establish a bug. The data, noise, and measurement times all differ.

```python
import matplotlib.pyplot as plt

# history[1:] excludes the initial record, which has no training MSE.
plt.plot(
    [record[0] for record in history[1:]],
    [record[1] for record in history[1:]],
    label="train MSE",
)
plt.plot(
    [record[0] for record in history],
    [record[2] for record in history],
    label="validation MSE",
)
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()
```

If both errors decline, both the measured training-process error and fixed-input validation reconstruction error are decreasing. If only training error improves while validation error keeps getting worse, investigate possibilities such as **overfitting**: fitting the training data while performance on new data deteriorates. If neither improves much, first check input–target pairing, value ranges, gradients, learning rate, and model structure. Three epochs of results are not enough to establish a long-term trend.

## 13. Improving on an untrained model and improving on the noisy input are different questions

If the initial model has a very large reconstruction error, training can reduce that error while still producing a worse result than returning the noisy image unchanged. We therefore need another comparison. A **baseline** is a reference for judging improvement. Here, the baseline is the MSE obtained by returning the noisy input without using a model.

Both comparisons use the same clean originals.

| Compared data | Target | What it tells us |
| --- | --- | --- |
| Noisy input `noisy_val` | `clean_val` | Pixel error already present in the input |
| Model output `recon` | The same `clean_val` | Pixel error after processing by the model |

```python
recon = predict_fixed(model, noisy_val)
baseline = image_mse(noisy_val, clean_val).mean().item()
restored = image_mse(recon, clean_val).mean().item()
print(f"Input MSE={baseline:.4f}, reconstruction MSE={restored:.4f}")

# Subtract reconstruction error from input error for each image. Result shape: (N,).
delta = image_mse(noisy_val, clean_val) - image_mse(recon, clean_val)
improved = (delta > 0).sum().item()  # Count images for which the condition is True.
worsened = (delta < 0).sum().item()
tied = (delta == 0).sum().item()
print("Improved / worsened / exactly tied:", improved, worsened, tied)
```

The source places `recon = None` immediately before the reconstruction call. The next assignment overwrites it, so the initial assignment can be omitted in the normal flow that directly executes `recon = predict_fixed(...)`.

If `restored < baseline`, the overall mean pixel error has decreased. This does not mean every image has improved. The entry `delta[i]` measures the change for image i: a positive value means lower reconstruction MSE, while a negative value means higher reconstruction MSE.

The comparison `delta == 0` tests exact equality without tolerance. That is different from two displayed values matching to four decimal places. If you need to group practically similar results, choose a **tolerance** appropriate to the evaluation and use `delta > tolerance`, `delta < -tolerance`, and `abs(delta) <= tolerance`. Record the threshold alongside the results.

### Inspect both the background and digit strokes in the same image

One pixel MSE value cannot fully express perceived reconstruction quality. Background noise may decrease while thin digit strokes are erased. Strokes may become blurred while improvements across the larger background still lower overall MSE. Since one number does not locate the changes, examine the following together.

| What to inspect | What to check |
| --- | --- |
| Overall mean | Is reconstruction MSE lower than input MSE? |
| Per-image error | Which images improved, and which became worse? |
| Background | Have noise dots and unwanted brightness decreased? |
| Digit strokes | Are thin strokes, thickness, connections, and original shape preserved? |

Each comparison must contain **the clean original, noisy input, and reconstruction for the same source image**. Use a shared display range, such as `imshow(..., cmap="gray", vmin=0, vmax=1)`, to reduce misleading differences caused by separate contrast scaling. The `cmap` specifies how values are displayed as shades, while `vmin` and `vmax` define the shared pixel-value display range.[^imshow]

The download's `plot_triplets(clean, noisy, recon)` places three images at the same index in the same column. It defaults to at most six images. The title's `Index` is a position in the supplied tensors, not automatically the original dataset's unique image ID.

The original material's `tools.draw_triplets(...)` is a custom function, not a built-in PyTorch function. Its implementation must be inspected to determine how it uses `val_ids`, `same8`, and `OUT`. When needed, read its source with `inspect.getsource(tools.draw_triplets)`. The downloadable code can plot images without that helper file, but **the actual original image data is still required**.

When describing the results, do not start by concluding that noise was removed simply because training ran. Connect the actual input and reconstruction MSE values, the images that improved or worsened, and the changes in strokes and background that you observed.

## 14. Four and eight channels refer to the intermediate feature-map count

The following comparison **changes only the encoder's output channel count from four to eight, keeping kernel size, stride, padding, and the remaining layer structure the same**. Since the original theory slides were not provided, we cannot confirm that their model had exactly this structure.

| Property | Four intermediate channels | Eight intermediate channels |
| --- | --- | --- |
| Input | `(B,1,28,28)` | `(B,1,28,28)` |
| Encoder filter count | 4 | 8 |
| Intermediate features | `(B,4,14,14)` | `(B,8,14,14)` |
| Intermediate values per image | 784 | 1,568 |
| Final output | `(B,1,28,28)` | `(B,1,28,28)` |
| Encoder parameters | 40 | 80 |
| Decoder parameters | 65 | 129 |
| Total parameters | **105** | **209** |

The numbers four and eight do not describe image count or the final grayscale output channels. They count the intermediate feature maps computed for each image. The eight-channel model can use more intermediate features, but it is not guaranteed to produce lower MSE or preserve strokes better.

For a comparison, keep the training and validation originals, fixed validation noise, training amount, and evaluation procedure the same, and create a fresh model and optimizer for each run. In the downloadable code, calls to `run_experiment(data, seed=SEED, feature_channels=4)` and the corresponding call with `feature_channels=8` each create a new model. Shuffling and noise generators are separate from model initialization, allowing both runs to use the same image order and training noise. This does not mean that one seed makes the weights of different architectures identical one for one.

If the difference is small, examine several seeds to see whether it depends on a particular initialization. Structural differences can be established from the table, while performance differences require actual experimental results.

## 15. What starts over when a notebook cell is rerun?

A notebook retains objects created earlier. Rerunning only the training loop can restart the displayed epoch number at 1 while the model still contains already-trained weights. Clearing `history` and initializing the model are separate operations.

| Code rerun | What starts over | What may remain unchanged |
| --- | --- | --- |
| `history = []` | The log list | Model weights and Adam state |
| Random-generator and seed setup | Future random-number streams | Existing model weights and Adam state |
| `model = LabDenoiser()` | New model weights | The previous optimizer and its state |
| A new Adam instance for the new model | Parameter references and Adam state | Previously saved results |

Start a fresh experiment in this order: **set the seed → create a new model → create a new optimizer for that model → initialize generators and logs → train**. If only the model is recreated, the existing optimizer can still reference the old model's parameters.

Editing a class definition also does not automatically transform an existing `model` object into the new architecture. Execute the updated class definition, then recreate the model. The provided `run_experiment()` performs this initialization on every call.

The 4 in `torch.set_num_threads(4)` sets the number of threads used for CPU computation. It has no connection to image count, channel count, or layer count, and it is not guaranteed to be the fastest setting on every device. The complete code preserves the original exercise's CPU setup.[^threads]

## 16. Which setup operation caused `StopIteration`?

The original setup code searched for a file as follows.

```python
from pathlib import Path

# Original code for explanation: if neither candidate contains the file, next raises StopIteration.
ASSETS = next(
    p.resolve()
    for p in [Path("assets"), Path("../assets")]
    if (p / "d2_tools.py").exists()
)
```

The path `Path("assets")` refers to `assets` beneath the current working directory, while `Path("../assets")` refers to `assets` beneath its parent. The expression keeps only candidates containing `d2_tools.py`. Then `next()` retrieves the first remaining candidate.

**The immediate cause is that neither candidate passed the condition, leaving no path for `next()` to retrieve, and no default value was supplied.** This is not a model-training or convolution failure. Why the file was not found at those paths still requires investigation: the file may be missing, or relative paths may be resolved from a different working directory than expected.[^next]

The **current working directory** is the base directory for resolving relative paths. Do not assume it always matches the directory containing the notebook file; inspect it with `Path.cwd()`. The `resolve()` method resolves a path into an absolute path. It does not create a missing file.[^pathlib]

```python
def locate_assets():
    candidates = [Path("assets"), Path("../assets")]
    found = next(
        (
            directory.resolve()
            for directory in candidates
            # is_file() checks that the path refers to a file, not merely that a name exists.
            if (directory / "d2_tools.py").is_file()
        ),
        None,  # Return None if no candidate satisfies the condition.
    )

    if found is None:
        checked = "\n".join(
            str((directory / "d2_tools.py").resolve())
            for directory in candidates
        )
        raise FileNotFoundError(
            f"Current working directory: {Path.cwd()}\n"
            f"Could not find d2_tools.py.\nChecked paths:\n{checked}"
        )
    return found

# Call this in the setup cell that actually needs the original helper file.
# ASSETS = locate_assets()
```

Using `next(..., None)` does not repair a missing file. It lets us report the problem more clearly when no candidate exists. Compare the printed absolute paths with the actual file location, then restore the original `assets` directory or correct the path. Creating an empty `d2_tools.py` does not solve the problem, because it lacks the required functions.

The subsequent `importlib` code loads and executes the located Python file as a module. Distinguishing the file-location stage from the file-execution stage makes the failure point clear.

## 17. Run the complete experiment after understanding the individual steps

The short examples above explain the meaning of each stage. The [complete Python practice file](/denoising-autoencoder-pytorch/examples-en.py) brings data checks, model construction, training, evaluation, and plotting functions together. Prepare these three requirements first.

| Requirement | What must be ready |
| --- | --- |
| Python packages | The current notebook kernel must be able to import `torch` and `matplotlib`. |
| `data` | It must contain `clean_train`, `clean`, and `noisy` tensors. Training requires at least 5,000 images. All image tensors must have shape `(N,1,28,28)` with values in 0–1. |
| Original-image correspondence | Matching validation indices in `clean` and `noisy` must refer to the same original. Validation originals must be separate from training originals. |

The following assumes the downloaded file is in the notebook's current working directory. The `runpy.run_path()` function executes the specified Python file and returns a dictionary of its defined names. This file does not train a model while loading its definitions. Retrieve the functions, then call `run_experiment()`.[^runpy]

```python
import runpy

# Download examples-en.py into the current working directory before running this.
# If it is elsewhere, replace the string with the actual file path.
examples = runpy.run_path("examples-en.py")
run_experiment = examples["run_experiment"]
plot_history = examples["plot_history"]
plot_triplets = examples["plot_triplets"]

# data and SEED are the values prepared in the original learning material.
# Each call trains a new model with a new optimizer for three epochs.
result = run_experiment(data, seed=SEED, feature_channels=8)

plot_history(result["history"])
plot_triplets(result["clean_val"], result["noisy_val"], result["recon"])

print(result["counts"])    # {'encoder': 80, 'decoder': 129, 'total': 209}
print(result["baseline"])  # Actual MSE of the noisy validation inputs.
print(result["restored"])  # Actual reconstruction MSE; run the experiment to compare it.
```

The `result` dictionary contains the model, optimizer, `history`, initial and final reconstructions, validation inputs and targets, baseline and reconstruction MSE, per-image `delta`, and parameter counts. To compare four channels, make a separate call with `feature_channels=4` and save the result in another variable.

The full code was run with synthetic data using Python 3.12.14 and PyTorch 2.15.0+cpu. Validation covered shapes, parameter counts, mean calculations, batch order, 237 updates, and plot generation. Reconstruction tensors were also compared with those from the original exercise code under the same seed. This verifies **the execution flow**; it is not a reconstruction-quality evaluation on the original course images.

## 18. Rust uses the same principles for splitting, joining, and combining means

Rust's `Vec<T>` is a variable-length array that stores values of one type in order. A **slice** references an existing contiguous region of data. The method `.chunks(256)` provides slices of at most 256 elements, and `extend_from_slice()` appends their values to the result vector.[^rust]

| Concept in the Python exercise | Rust standard-library counterpart |
| --- | --- |
| Index range `range(600)` | `0..600`, excluding the end value |
| Split into batches in order | `slice.chunks(256)` |
| Join processed results in order | `Vec::extend_from_slice()` |
| Sum mean × element count, then divide by total count | The same weighted-mean calculation |

The following code checks order preservation using 600 illustrative IDs rather than actual images. It does not implement a neural network or automatic differentiation. The mean function's `Option<f64>` return type means `Some(mean)` when a value exists and `None` when there are no elements to average. The `match` expression handles these two cases separately.

```rust
// This example checks data splitting, joining, and aggregation. It is not an autoencoder.
fn weighted_epoch_mse(batches: &[(f64, usize)]) -> Option<f64> {
    // batches: a reference to tuples of (mean loss, number of elements in that mean).
    // f64 is a floating-point type; usize is an integer type used for lengths and indices.
    let mut sse = 0.0;
    let mut count: usize = 0;
    for &(mean_loss, elements) in batches {
        // The & pattern reads the referenced tuple and obtains its copyable numeric values.
        sse += mean_loss * elements as f64; // as f64 converts the count for multiplication.
        count += elements;
    }
    if count == 0 { None } else { Some(sse / count as f64) }
}

fn main() {
    // 600 is the illustrative ID count. The last value in 0..600 is 599.
    let image_ids: Vec<usize> = (0..600).collect();
    let mut combined: Vec<usize> = Vec::new();
    for part in image_ids.chunks(256) {
        println!("Images in batch: {}", part.len()); // 256, 256, 88.
        combined.extend_from_slice(part); // Join IDs in order instead of running inference.
    }
    assert_eq!(combined, image_ids); // Check that contents and order are preserved.

    // These two loss values are examples for aggregation, not actual experimental results.
    // Counts come from batches of 64 and 8 grayscale images, each 28×28.
    let batches = [(0.04, 64 * 28 * 28), (0.01, 8 * 28 * 28)];
    match weighted_epoch_mse(&batches) {
        Some(mean) => println!("Example weighted mean MSE: {mean:.6}"),
        None => println!("There are no elements to average."),
    }
}
```

Both Python and Rust receive values from an iterable into local variables and process batches while preserving their order. The difference here is that PyTorch tensors in the Python example also provide multidimensional operations and automatic differentiation. The Rust example uses only the standard library to demonstrate ordinary data flow. Building an actual neural network in Rust requires a separate implementation providing those capabilities.

## 19. What should you understand before moving to the next exercise?

The goal is not to memorize every transposed-convolution variant or its internal implementation. For this exercise, you should be able to explain what each of the following operations works on.

| Priority | What to explain or verify yourself |
| --- | --- |
| Essential | Distinguish retrieving a tensor by a dictionary key from indexing its dimensions. |
| Essential | Read the shapes of inputs, intermediate features, and reconstructions, separately from stored weight shapes. |
| Essential | Compare matching pixels in matching images, and distinguish overall from per-image means. |
| Essential | Explain that `backward()` calculates gradients and `step()` updates parameters. |
| Essential | Explain why `split → model → cat` preserves image order. |
| Essential | Log a correctly weighted mean that includes the smaller final batch. |
| Essential | Compare fixed-input baseline and reconstruction MSE while also inspecting actual strokes and background. |
| Useful to know | Recreate both the model and Adam and match conditions when comparing four and eight channels. |
| Refer to when needed | Check the general transposed-convolution formula, alternative upsampling structures, and file-location diagnostics. |

Return to the opening question. The model receives noisy images and computes reconstructions. Loss compares corresponding pixels from the same originals; gradients of that loss guide weight updates. Evaluation reconstructs the fixed inputs again and checks **whether error is actually lower than in the input itself, and whether needed digit strokes were lost along the way**. Once you can follow this connection in code, you can apply the same training flow to the next dataset and model.

## References

Architecture, shapes, parameter counts, and batch counts were calculated directly from the model and exercise settings presented here. API behavior and related concepts were checked against the following official documentation and original publications on November 3, 2026.

[^dae]: Vincent et al., [Extracting and Composing Robust Features with Denoising Autoencoders](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf), ICML 2008. Learning to reconstruct originals from corrupted inputs.
[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html). Submodule and parameter registration, and model calls.
[^sequential]: PyTorch, [Sequential](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sequential.html). Connecting registered modules in sequence.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html). Arguments, weight structure, and output sizes.
[^transpose]: PyTorch, [ConvTranspose2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ConvTranspose2d.html). Definition, output sizes, weights, and output_padding.
[^upsample]: PyTorch, [Upsample](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Upsample.html). Increasing spatial size through interpolation.
[^checkerboard]: Odena, Dumoulin, and Olah, [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/), Distill 2016. Overlap, artifacts, and resizing followed by convolution.
[^comprehension]: Python, [Displays for lists, sets and dictionaries](https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries). Comprehension iteration and variable scope.
[^mse]: PyTorch, [MSELoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MSELoss.html). The aggregation scope of `none`, `mean`, and `sum`.
[^flatten]: PyTorch, [flatten](https://docs.pytorch.org/docs/2.15/generated/torch.flatten.html) and [mean](https://docs.pytorch.org/docs/2.15/generated/torch.mean.html). Combining dimensions and calculating means.
[^broadcast]: PyTorch, [Broadcasting semantics](https://docs.pytorch.org/docs/2.15/notes/broadcasting.html). Matching dimension sizes from the end.
[^zero]: PyTorch, [Optimizer.zero_grad](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.zero_grad.html). Resetting gradients and `set_to_none`.
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html). Computation recording, backpropagation, and the distinction between evaluation mode and grad modes.
[^item]: PyTorch, [Tensor.item](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.item.html). Retrieving a Python number from a one-element tensor.
[^detach]: PyTorch, [Tensor.detach](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.detach.html). Returning a tensor disconnected from autograd.
[^split]: PyTorch, [split](https://docs.pytorch.org/docs/2.15/generated/torch.split.html). Splitting along a dimension, final chunks, and returned views.
[^cat]: PyTorch, [cat](https://docs.pytorch.org/docs/2.15/generated/torch.cat.html) and [stack](https://docs.pytorch.org/docs/2.15/generated/torch.stack.html). Joining along an existing dimension versus adding a new one.
[^randomness]: PyTorch, [Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html). Seeds and the limits of reproducibility.
[^adam]: PyTorch, [Adam](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html). Parameters, learning rate, and optimizer state.
[^randperm]: PyTorch, [randperm](https://docs.pytorch.org/docs/2.15/generated/torch.randperm.html). Random permutations of an integer range.
[^randn]: PyTorch, [randn](https://docs.pytorch.org/docs/2.15/generated/torch.randn.html). Standard-normal random values.
[^clamp]: PyTorch, [clamp](https://docs.pytorch.org/docs/2.15/generated/torch.clamp.html). Restricting values to lower and upper bounds.
[^dataloader]: PyTorch, [Dataset and DataLoader](https://docs.pytorch.org/docs/2.15/data.html). Loading data and constructing batches.
[^imshow]: Matplotlib, [imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html). Grayscale mapping and display ranges.
[^threads]: PyTorch, [set_num_threads](https://docs.pytorch.org/docs/2.15/generated/torch.set_num_threads.html). Setting CPU computation thread counts.
[^next]: Python, [next](https://docs.python.org/3/library/functions.html#next). Iterator exhaustion, default values, and StopIteration.
[^pathlib]: Python, [pathlib](https://docs.python.org/3/library/pathlib.html). Working directories, path resolution, and file checks.
[^runpy]: Python, [runpy.run_path](https://docs.python.org/3/library/runpy.html#runpy.run_path). Executing a file and returning its globals dictionary.
[^rust]: Rust, [slice::chunks](https://doc.rust-lang.org/std/primitive.slice.html#method.chunks) and [Vec::extend_from_slice](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.extend_from_slice). Splitting slices and appending to vectors.
