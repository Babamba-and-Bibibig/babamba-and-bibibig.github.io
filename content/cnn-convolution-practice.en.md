+++
title = "Convolution by Hand: From Indexing to MNIST Visualization"
description = "Calculate one convolution output from an input patch and a kernel, derive output sizes for stride and padding, distinguish PyTorch autograd from detach, and visualize a Laplacian response on MNIST."
date = 2026-10-28
slug = "cnn-convolution-practice"

[extra]
katex = true
toc = true
styles = ["cnn-convolution-practice/article.css"]
+++

A convolution can run in a single line: `out = conv(x)`. To explain the first number in its output, however, we need to know which part of the input it reads, which kernel values it multiplies, and which products it adds together. Output sizes and visualization code follow from the same calculation.

This article starts with a small array of numbers. We calculate one output value by hand, then apply the same process to a handwritten image. First, we prepare an input and a kernel and use indexing to select a small region. We multiply matching positions and add the results. Repeating the calculation at different positions produces the full output, which we eventually display as an image.

We also run a separate automatic differentiation experiment to examine how changes in the weights affect the output. Applying a fixed filter chosen by a person and training a model are separate operations. None of the examples in this article calls `optimizer.step()` to update the weights.

For channel connections and the structure of a complete convolutional neural network (CNN), see the [previous article: From Channels and Kernels to Linear](/en/cnn-convolution-basics/). Here, we introduce the concepts needed to understand the numbers and code in front of us. The Python blocks build on one another from top to bottom. A [complete runnable script with comments](/cnn-convolution-practice/examples-en.py) is also available.

## 1. Store the image values and identify the axes

An image can be represented as numbers stored at different positions. A grayscale image has one brightness value per position. An RGB image stores separate red, green, and blue components. A **channel** distinguishes these components.

A PyTorch **tensor** is a data type for storing and calculating with numbers. Each indexing position in a tensor corresponds to an **axis**. Its **shape** tells us how many elements each axis contains, and an **index** identifies a position to select. Indexing starts at 0.

Our input is a small array with four rows and five columns, representing one grayscale image. Simple numbers make the calculations easier to check by hand than a photograph would.

`torch.tensor(...)` is a function that creates a tensor from the supplied numbers. `reshape(...)` is a method that changes the sizes of the axes while keeping the total number of elements unchanged. A method is a function called on an object, as in `x.reshape(...)`.

The `tuple(...)` calls below convert the shape to a Python tuple, a data type that groups values in order. Here, this gives us a readable display such as `(4, 5)`.

```python
import torch  # Import tensors and numerical operations.
from torch import nn  # Use nn for classes such as convolution layers.
import torch.nn.functional as F  # Use F for neural network operations as functions.

X = torch.tensor([
    [1., 2., 0., 1., 3.],
    [0., 3., 1., 2., 1.],
    [2., 1., 0., 1., 0.],
    [1., 0., 2., 3., 2.],
])

# Keep all 4×5 values and add image and channel axes at the front.
x = X.reshape(1, 1, 4, 5)

print(tuple(X.shape))  # (4, 5)
print(tuple(x.shape))  # (1, 1, 4, 5)
```

`1.` is the same floating-point notation as `1.0`. `X` and `x` are different variable names: Python is case-sensitive. `F` is also a name chosen as a short alias for a module, rather than a special keyword.

A **batch** is a group of data items processed together, such as several images. In this article, we use the batched format `(N, C, H, W)` even when there is only one image. In order, these axes represent the number of images, the number of channels, height, and width.

| Axis number | What this axis selects | Size in `x.shape` | Valid indices |
| --- | --- | ---: | --- |
| 0 | Image | 1 | 0 |
| 1 | Channel of the selected image | 1 | 0 |
| 2 | Row | 4 | 0, 1, 2, 3 |
| 3 | Column | 5 | 0, 1, 2, 3, 4 |

A `1` in the shape means that the axis contains one element. Its only valid index is therefore `0`. Having two axes for images and channels does not mean that there are two images or two channels.

The element count is unchanged by `reshape`: `4×5 = 1×1×4×5 = 20`. In this example, the original row and column arrangement is also preserved. No convolution has been calculated yet.

`nn.Conv2d` also accepts an unbatched input of shape `(C, H, W)`. We use four dimensions here so that we can keep identifying images and channels in the same way throughout the examples.[^conv]

## 2. What does `x[0, 0, :3, :3]` select?

Start with `x[0, 0]`. The first `0` selects the first and only image. The second `0` selects its first and only channel. We have not specified any rows or columns, so all of them remain.

```python
image_channel = x[0, 0]  # Select image 0 and channel 0.
print(tuple(image_channel.shape))  # (4, 5)
print(torch.equal(image_channel, X))  # True: the shapes and values match.
```

The result contains the entire original 4×5 array, rather than just two numbers. Selecting the image and channel with integer indices removes those two axes from the result's shape.

In this example, `x[0, 0]` and `x[0][0]` give the same result. The latter first selects an image, then selects a channel from that result. The former places both axis indices inside one pair of brackets. Python passes the tuple `(0, 0)`, and the PyTorch tensor interprets it as selection instructions for multiple axes. An ordinary nested Python list uses `data[0][0]`. The equivalence of the two forms should not be extended to every kind of advanced indexing.[^index]

Now select only the top-left 3×3 region. A small region selected from an image is called a **patch**.

```python
# :3 is short for 0:3. It includes the start, 0, and excludes the stop, 3.
patch = x[0, 0, :3, :3]

print(patch)
# tensor([[1., 2., 0.],
#         [0., 3., 1.],
#         [2., 1., 0.]])

print(tuple(patch.shape))  # (3, 3)
```

The first `:3` selects row indices `0, 1, 2`; the second selects column indices `0, 1, 2`. The image and channel axes selected with integers disappear. The row and column axes selected with slices remain.

The patch is an input to a calculation we will perform next. Reading part of the input has not produced a convolution output, and it has not deleted that region from the original image.

## 3. Calculate one output value from a patch and a kernel

A **weight** is a number multiplied by an input value. An array of weights arranged spatially is called a **kernel**. We choose the following kernel to add the values in the left column and subtract those in the right column. The middle column contributes nothing to the sum because its weights are zero.

```python
K = torch.tensor([
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.],
])

products = patch * K  # Multiply values at matching positions.
print(products)
# tensor([[ 1., 0.,  0.],
#         [ 0., 0., -1.],
#         [ 2., 0.,  0.]])
```

Here, `*` performs **elementwise multiplication**. It multiplies values at matching positions, such as `patch[0, 0]` with `K[0, 0]` and `patch[0, 1]` with `K[0, 1]`. Nine multiplications leave nine results arranged as a 3×3 tensor. This operation does not perform matrix multiplication or automatically add the products.

The variable name `products` refers to the results of these multiplications. If an output message says `elementwise products`, it means the products of values at matching positions.

<figure class="conv-practice-figure">
<img src="/cnn-convolution-practice/patch-products-en.svg" alt="Multiplying matching positions in the first input patch and the 3×3 kernel gives row sums of 1, -1, and 2, for a total of 2." loading="lazy">
<figcaption>Elementwise multiplication produces a 3×3 result. Adding all nine values gives one convolution output value.</figcaption>
</figure>

### Which values does `sum(dim=1)` add?

`products.shape` is `(3, 3)`. Axis 0 represents rows, and axis 1 represents columns. `sum(dim=1)` adds the values across the columns in each row.[^sum]

| Row | Products in that row | Sum across the column axis |
| --- | --- | ---: |
| 0 | `[1, 0, 0]` | 1 |
| 1 | `[0, 0, -1]` | -1 |
| 2 | `[2, 0, 0]` | 2 |

One total remains per row, so the result has shape `(3,)`. The `1` in `dim=1` is the number of the second axis. It does not mean to add only the values at index 1.

```python
row_sums = products.sum(dim=1)  # Add the three column values in each row.
print(row_sums)                # tensor([ 1., -1.,  2.])
print(row_sums.tolist())       # [1.0, -1.0, 2.0]

first_output = products.sum()  # No axis is specified, so add all elements.
print(first_output)            # tensor(2.)
print(tuple(first_output.shape))  # ()
print(first_output.item())     # 2.0
```

`tolist()` converts tensor values into an ordinary Python list. It does not perform another sum. `item()` extracts an ordinary Python number from a tensor containing exactly one element. That number does not retain the automatic differentiation connection, so this conversion is used for display or logging.[^values]

Shape `()` describes a zero-dimensional tensor containing one number. It does not describe an empty tensor. `sum(dim=0)` would add values from different rows within each column; `sum()` adds all values without separating the rows or columns.

Our output value is `1 + (-1) + 2 = 2`. We wrote out the row sums to check the calculation. A convolution does not have to create a separate intermediate variable called `row_sums`.

### Run the same calculation with `nn.Conv2d`

A **layer** is a component responsible for one stage of computation in a neural network. `nn.Conv2d` is a class for creating convolution layers. Calling `nn.Conv2d(...)` creates an object that holds weights and calculation settings. Here, that object is one concrete layer we can use.

In this article, a **filter** means the complete set of weights used to produce one output channel. With one input channel and one output channel, one filter contains one 3×3 kernel. Some sources use “kernel” and “filter” interchangeably, so checking the actual weight shape and channel connections is more precise than relying on the term alone.

```python
conv = nn.Conv2d(
    in_channels=1,   # Each input image has one channel.
    out_channels=1,  # One filter produces one output channel.
    kernel_size=3,   # The spatial kernel is 3×3.
    bias=False,      # Do not add a bias after summing the products.
)

print(tuple(conv.weight.shape))  # (1, 1, 3, 3)

# Copy the manually chosen values in K into the layer's weights.
with torch.no_grad():
    conv.weight.copy_(K.reshape(1, 1, 3, 3))

out = conv(x)  # Calculate the convolution using x and the current weights.
print(out[0, 0].tolist())
# [[2.0, 2.0, -3.0], [0.0, -2.0, 0.0]]

print(tuple(out.shape))  # (1, 1, 2, 3)
```

A newly created layer starts with randomly initialized weights. `copy_` copies the numbers in `K` into the existing weight storage. In PyTorch, a trailing `_` usually marks an operation that modifies an existing tensor in place. Copying the values does not make `K` and `conv.weight` the same object.[^copy]

PyTorch provides **automatic differentiation**, which records relationships between operations so that it can later calculate rates of change. Here, `with torch.no_grad():` keeps the indented copy operation out of that record. This is a way to assign weights directly; it does not permanently freeze them. Later, we will use numbers to examine why automatic differentiation tracks outputs and what that tracking is for.[^nograd]

The unspecified settings are `stride=1`, `padding=0`, `dilation=1`, and `groups=1`. The kernel reads adjacent input cells, moves its starting position one cell at a time, and adds no cells outside the input. Each filter uses every input channel. The next sections will change the spacing between starting positions and the treatment of the boundaries.

| Expression | What it refers to |
| --- | --- |
| `conv` | A layer object containing weights and calculation settings |
| `conv.weight` | The weight tensor inside the layer |
| `conv(x)` | The output tensor calculated from the current input and weights |
| `out[0, 0]` | The entire first output channel of the first image, shape `(2, 3)` |
| `out[0, 0, 0, 0]` | One number in the first row and column of the first output channel |

Compare the first value calculated by hand with the layer's result. `assert` allows execution to continue when its condition is true and raises an error when it is false.

```python
assert first_output.item() == out[0, 0, 0, 0].item()  # Both values are 2.0.
assert torch.equal(
    out[0, 0],
    torch.tensor([[2., 2., -3.], [0., -2., 0.]]),
)
```

We reused nine weights at six different positions to produce six output values. The number of outputs and the number of weights are different quantities. An output array calculated across multiple spatial positions is also called a **feature map**.

We did not pass `products` to `conv`. The call `conv(x)` calculates with its own weights. The expressions `patch * K` and `assert` let us follow and check that calculation. Removing `print` and `assert` does not stop the convolution itself from running.

PyTorch's `Conv2d` calculates in this way without flipping the kernel. Mathematically, this operation is called cross-correlation, although deep learning commonly refers to it as convolution.[^conv]

### Weights when there are multiple channels

The first two axes already mean different things in the input shape and the weight shape. With `groups=1`, the input shape is `(number of images, input channels, height, width)`. The weight shape is `(output channels, input channels, kernel height, kernel width)`.

For an RGB input and two output filters, the weight shape is `(2, 3, 3, 3)`. Each filter contains one kernel for R, one for G, and one for B. It calculates a result for each of the three channels at the same spatial position and adds them to produce one output value. One output value uses `3×3×3=27` weights. Across two filters, there are `2×27=54` weights. If bias is enabled, each filter also has one bias value.

The first `0` in `conv.weight[0, 0]` therefore selects an output filter, rather than an image. The second `0` selects the input channel used by that filter. Output channels distinguish the results of different filters; they do not necessarily represent RGB color components.

## 4. How ReLU changes the convolution output

**ReLU** replaces a negative input with 0 and leaves a zero or positive input unchanged. It is a separate operation that can follow a convolution.[^relu]

```python
relu_out = F.relu(out)  # Apply the rule independently to each element.
print(relu_out[0, 0].tolist())
# [[2.0, 2.0, 0.0], [0.0, 0.0, 0.0]]

assert relu_out.shape == out.shape  # The element count and axis sizes stay the same.
```

ReLU does not combine multiple values, so it preserves the shape. The function itself has no weights to learn.

The reason for using ReLU is often described as introducing **nonlinearity**. A linear calculation preserves addition and scaling: performing the calculation before or after combining inputs gives the corresponding same result. ReLU does not have that property. Small numbers show why. `ReLU(1) + ReLU(-1)` is `1 + 0 = 1`. Adding the inputs first gives `ReLU(1 + (-1)) = ReLU(0) = 0`. The results differ.

Adding a stage whose rule depends on the input's sign allows a stack of layers to represent more than a single linear calculation. This does not mean that negative values are always useless. In the Laplacian visualization later in the article, negative and positive responses each carry meaning, so we will keep both and leave out ReLU.

## 5. The output size counts valid kernel starting positions

**Stride** is the spacing between the kernel's starting positions. With `stride=2`, we choose starts such as `0, 2, 4, …`. **Padding** adds values around the input boundary for the calculation. Here, we add the same number of zeros on both sides.

**Dilation** is the spacing between the input positions read by neighboring kernel weights. We keep `dilation=1` throughout this article, so the kernel reads adjacent input cells. The output height and width can each be calculated using the same reasoning.

### First, find the last allowed starting coordinate

An input of length 6 has indices `0, 1, 2, 3, 4, 5`. Place a kernel of length 3 on that input. Starting at coordinate 3 reads `3, 4, 5`, so the kernel fits. Starting at coordinate 4 would read `4, 5, 6`, but index 6 does not exist.

Now write that condition with symbols. Let `L` be the original input length, `p` the padding length on each side, and `k` the kernel length. Padding on both sides makes the total length `T=L+2p`. Let `j` be the kernel's starting coordinate in the padded input.

The kernel reads from `j` through `j+k-1`. The last cell it reads cannot go beyond the input's final index, `T-1`.

<div class="conv-practice-math">
$$
\begin{aligned}
j+k-1 &\le T-1 \\
j &\le T-k &&\text{Subtract }k-1\text{ from both sides.}
\end{aligned}
$$
</div>

Calling the last allowed starting coordinate `M` gives:

<div class="conv-practice-math">
$$M=T-k=L+2p-k$$
</div>

For input length 6 and kernel length 3, `M=6-3=3`. We have not used stride yet. We have only found the range in which the whole kernel fits.

### Why `range(0, M + 1, s)` needs `+1`

Let the stride be `s`. Python's `range(start, stop, step)` represents integers beginning at `start` and increasing by `step`. It excludes the endpoint `stop`. Wrapping it in `list(...)` collects those integers into a list.[^range]

The allowed condition is `j ≤ M`. For integer coordinates, this is equivalent to `j < M+1`, so we set the exclusive endpoint to `M+1`.

```python
L, p, k, s = 4, 1, 3, 2  # Input length 4, padding 1 per side, kernel 3, stride 2.
T = L + 2 * p             # The padded input has length 6.
M = T - k                 # The last allowed starting coordinate is 3.
starts = list(range(0, M + 1, s))

print(starts)  # [0, 2]
```

`range(0, 4, 2)` selects 0 and 2 and excludes 4. The last coordinate allowed by the input boundary is 3, but the last coordinate actually visited is 2. The stride does not land on 3. Adding `1` to the endpoint does not force coordinate 3 into the list.

<figure class="conv-practice-figure">
<img src="/cnn-convolution-practice/stride-positions-en.svg" alt="For an input of length 6, a kernel of length 3 fits through starting coordinate 3. With stride 2, the actual starts are 0 and 2; the next candidate, 4, is out of bounds." loading="lazy">
<figcaption>The last allowed start, 3, the last actual start, 2, and the output count, 2, describe different things.</figcaption>
</figure>

### Why the output-length formula divides by stride and adds 1

Let `q` be an output index. The first output is calculated at input starting coordinate 0, the second at `s`, and the third at `2s`.

| Output index `q` | Starting coordinate `j` in the input |
| ---: | ---: |
| 0 | 0 |
| 1 | `s` |
| 2 | `2s` |
| 3 | `3s` |

Thus, `j=q×s`. The starting coordinate must be no greater than `M`, so `q×s ≤ M`. Dividing both sides by the positive value `s` gives `q ≤ M/s`.

Output indices must be integers. The final output index is therefore the largest integer no greater than `M/s`. The operation that rounds down to that integer is called **floor**. If `M=3` and `s=2`, then `M/s=1.5` and the last output index is 1. Indices 0 and 1 give two outputs.

<div class="conv-practice-math">
$$
L_{\mathrm{out}}
=\left\lfloor\frac{M}{s}\right\rfloor+1
=\left\lfloor\frac{L+2p-k}{s}\right\rfloor+1
$$
</div>

This formula applies when `dilation=1` and the padding is equal on both sides. `L`, `k`, and `s` must be positive integers, `p` must be nonnegative, and the kernel must fit inside the padded input. Under these conditions, `M` is nonnegative, and Python's integer division `//` gives:[^conv]

```python
output_length = M // s + 1  # 3 // 2 + 1 = 1 + 1 = 2.
assert output_length == len(starts) == 2
```

The two occurrences of `+1` serve different purposes. In `range`, `M+1` adjusts the boundary because the endpoint is excluded. In the output-length formula, `floor(M/s)+1` converts a final index counted from zero into a number of elements.

Dividing the `range` endpoint by stride and also setting `step` to stride mixes these two calculations. The endpoint is an input coordinate; the output length is the number of starting coordinates selected within that range.

Even with a stride of 2, the output length is not always exactly half the input length. Starting position 0 also produces an output, and only positions where the whole kernel fits are included. The floor rule in the formula remains the same.

### Count rows and columns separately, then use every combination

Apply a 3×3 kernel to the original 4×5 input with `stride=2` and `padding=0`.

A starting row of 0 reads rows `0, 1, 2`. The next candidate, 2, would require rows `2, 3, 4`, but row index 4 does not exist. The starting rows are therefore `[0]`. Along the columns, the kernel fits at both 0 and 2, so the starting columns are `[0, 2]`.

The actual calculation positions are every combination of those lists: `(0, 0)` and `(0, 2)`. They produce output positions `(0, 0)` and `(0, 1)`, respectively. One starting row multiplied by two starting columns gives a 1×2 output. We do not pair the row and column lists only by matching list indices.

## 6. Compare stride and padding using the same input and weights

To observe the effect of calculation settings, we need to hold the input and weights constant. Creating a new layer with `nn.Conv2d(...)` creates new initial weights. It does not automatically copy the `K` assigned to an existing layer. Calling the already-created `conv` again, however, does not reinitialize it: the layer continues to use its existing weights.

`F.conv2d` is a function that calculates a convolution from the input and weights passed directly to it. The following code supplies the same `x` and `conv.weight` each time and changes only `stride` and `padding`.[^functional]

```python
# Read the last two entries of the shape, rather than two rows of image data.
H, W = x.shape[-2:]
settings = [(1, 0), (2, 0), (1, 1), (2, 1)]  # Pairs of (stride, padding).
shape_results = {}  # A dictionary storing the output shape for each setting.

for stride, padding in settings:
    actual = F.conv2d(
        x,                   # The same input in all four calculations.
        conv.weight,         # The same weights in all four calculations.
        stride=stride,
        padding=padding,
    )

    starts_h = list(range(0, H + 2 * padding - 3 + 1, stride))
    starts_w = list(range(0, W + 2 * padding - 3 + 1, stride))
    shape_results[(stride, padding)] = tuple(actual.shape)

    print("Settings:", (stride, padding))
    print("Starting rows:", starts_h, "Starting columns:", starts_w)
    print("Output:", actual[0, 0].tolist())
    assert tuple(actual.shape[-2:]) == (len(starts_h), len(starts_w))
```

Each item in `settings` is a tuple containing two numbers. `for stride, padding in settings` unpacks those numbers into separate variables. A dictionary associates keys with values. Here, `shape_results` stores the output shape under the key `(stride, padding)`.

When padding is present, starting coordinates are measured in the **padded input**.

| stride | padding | Starting rows | Starting columns | Output height × width |
| ---: | ---: | --- | --- | --- |
| 1 | 0 | `[0, 1]` | `[0, 1, 2]` | 2×3 |
| 2 | 0 | `[0]` | `[0, 2]` | 1×2 |
| 1 | 1 | `[0, 1, 2, 3]` | `[0, 1, 2, 3, 4]` | 4×5 |
| 2 | 1 | `[0, 2]` | `[0, 2, 4]` | 2×3 |

Compare the actual output values as well.

```text
stride=1, padding=0
[[ 2,  2, -3],
 [ 0, -2,  0]]

stride=2, padding=0
[[ 2, -3]]

stride=1, padding=1
[[-5,  0,  2, -3,  3],
 [-6,  2,  2, -3,  4],
 [-4,  0, -2,  0,  6],
 [-1,  1, -3,  0,  4]]

stride=2, padding=1
[[-5,  2,  3],
 [-4, -2,  6]]
```

The first and last settings both produce a 2×3 output, but their values differ because the input regions and spacing between positions differ. Matching shapes alone do not imply matching calculations.

The `padding=1` setting preserves the input's 4×5 size here when the kernel is 3×3, `stride=1`, and `dilation=1`. Setting padding to 1 does not preserve input size under every configuration.

Also, `F.conv2d` does not automatically read settings from the `conv` object. It uses the arguments passed to the function and the function's own defaults. Our current `conv` has no bias and uses the default stride and padding, so `conv(x)` and `F.conv2d(x, conv.weight)` perform the same calculation. A layer with different settings requires those values to be passed as well.

### Why does padding change the first output from 2 to -5?

Use `F.pad` to create the zero padding explicitly. The second argument, `(1, 1, 1, 1)`, adds one cell to the left, right, top, and bottom, in that order. It specifies the two sides of the last axis, width, first, followed by the two sides of height.[^pad]

```python
padded_x = F.pad(x, (1, 1, 1, 1))  # The default constant padding value is 0.
print(tuple(padded_x.shape))       # (1, 1, 6, 7)
print(padded_x[0, 0].tolist())
# [[0., 0., 0., 0., 0., 0., 0.],
#  [0., 1., 2., 0., 1., 3., 0.],
#  [0., 0., 3., 1., 2., 1., 0.],
#  [0., 2., 1., 0., 1., 0., 0.],
#  [0., 1., 0., 2., 3., 2., 0.],
#  [0., 0., 0., 0., 0., 0., 0.]]

padded_patch = padded_x[0, 0, :3, :3]  # The padded array's top-left 3×3 region.
padded_products = padded_patch * K

print(padded_patch.tolist())
# [[0., 0., 0.], [0., 1., 2.], [0., 0., 3.]]
print(padded_products.tolist())
# [[0., 0., 0.], [0., 0., -2.], [0., 0., -3.]]
print(padded_products.sum(dim=1).tolist())  # [0.0, -2.0, -3.0]
print(padded_products.sum().item())        # -5.0
```

The kernel is unchanged. The new first patch includes the zeros added above and to the left of the original image. Its right column contains the original values 2 and 3, giving `0-2-3=-5`. `F.pad` returns a padded tensor; it does not overwrite the shape or values of the original `x`.

The original first patch starts at `(1, 1)` in the padded array. At that position, the output is still 2.

```python
pad_out = F.conv2d(x, conv.weight, padding=1)
old_patch = padded_x[0, 0, 1:4, 1:4]  # Select indices 1, 2, and 3.

assert torch.equal(old_patch, patch)
assert pad_out[0, 0, 0, 0].item() == -5.0
assert pad_out[0, 0, 1, 1].item() == 2.0

# An explicitly padded input needs no additional padding in the convolution.
manual_pad_out = F.conv2d(padded_x, conv.weight, padding=0)
assert torch.equal(manual_pad_out, pad_out)
```

Passing the already-padded `padded_x` with `padding=1` would add padding a second time. The added zeros do not reconstruct real pixels outside the image. They define a boundary rule: values outside the image are taken to be zero.

## 7. Where does the Laplacian filter's `-4` come from?

Now use a kernel that compares the center's brightness with its neighbors. The following **Laplacian filter** contains manually chosen weights that add the differences between the center and its top, bottom, left, and right neighbors.[^laplacian]

```python
laplace = torch.tensor([
    [0.,  1., 0.],
    [1., -4., 1.],
    [0.,  1., 0.],
])
```

Call the neighboring values `top`, `bottom`, `left`, and `right`. The calculation is:

<div class="conv-practice-math">
$$
\begin{aligned}
&(\text{top}-\text{center})+(\text{bottom}-\text{center})\\
&\quad +(\text{left}-\text{center})+(\text{right}-\text{center})\\
&=\text{top}+\text{bottom}+\text{left}+\text{right}-4\times\text{center}
\end{aligned}
$$
</div>

The center is subtracted once for each of the four neighbors, which gives its weight of `-4`. Diagonal positions have zero weights because this calculation does not use them. The same expression can be written as `4×(average of the four neighbors - center)`. With this kernel, the response is negative when the center is brighter than the neighbors' average, positive when it is darker, and zero when the two are equal.

A zero response does not mean that all the neighboring values are identical. If the center is 1 and its four neighbors are `2, 0, 0, 2`, the differences are `1, -1, -1, 1`. They cancel to give zero.

Mathematically, this Laplacian approximates a second derivative with respect to spatial position. We will interpret its values through the sum of four differences above, without introducing further formulas for that term. It describes how brightness varies across an image. Automatic differentiation during training instead concerns how the loss changes with the weights.

### Check the calculation on a new 5×5 input

We use a new variable, `laplace_x`, to distinguish this input from the previous 4×5 example. Copy the kernel above into `laplace_conv`.

```python
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
first_laplace_patch = laplace_x[0, 0, :3, :3]

print((first_laplace_patch * laplace).tolist())
# [[0.0, 2.0, 0.0], [0.0, -4.0, 2.0], [0.0, 0.0, 0.0]]
print((first_laplace_patch * laplace).sum().item())  # 0.0
print(laplace_out[0, 0].tolist())
# [[0.0, 0.0, -7.0], [8.0, 0.0, -3.0], [-8.0, 8.0, 0.0]]
print(tuple(laplace_out.shape))  # (1, 1, 3, 3)
```

The first patch has a center value of 1. Its top, bottom, left, and right values are 2, 0, 0, and 2, so the result is `2+0+0+2-4×1=0`. The output height and width are each `5-3+1=3`.

Supplying weights in advance does not by itself make them pretrained. **Random initialization** chooses starting values. **Manual design** chooses values to perform a specified calculation. **Pretrained weights** come from an earlier learning process that adjusted them. This Laplacian is manually designed.

Creating new layers with `nn.Conv2d(1, 1, 3, padding=1)` or `nn.Conv2d(1, 1, 3, stride=2)` gives them separate initial weights. Their 5×5 and 2×2 output sizes can be compared, but the numerical results are not responses from the same Laplacian filter. To compare the same filter, pass the existing weights, as in `F.conv2d(laplace_x, laplace_conv.weight, padding=1)`.

## 8. Why is the output connected to autograd if it is not a trainable parameter?

So far, we have calculated output values from inputs and weights. In a typical training process, those outputs lead to predictions, and a number called the **loss** measures how the predictions differ from a target. Training then changes the weights to reduce the loss.

This requires a **gradient**: a rate of change describing how the loss responds to a very small change in a particular value, with other conditions held fixed at their current state. To learn the weights, we need the loss gradient with respect to those weights.

Changing a weight changes the convolution output, which can change the loss calculated afterward. Even though the ultimate goal is a weight gradient, we need the calculation relationships that connect the intermediate output to the loss.

### Follow the changes along one path with numbers

Before returning to convolution, consider a tiny example with one output. Fix the input at 3, call the weight `w`, and define the output as `z=3w`. Let the target be 0, with the squared difference as the loss: `L=z²`. At `w=2`, the output is `z=6` and the loss is `L=36`.

Changing the weight by `h` changes the output by `3h`. The output's rate of change with respect to the weight is therefore 3. Next, calculate the change in loss when we change the output from 6 by an amount `d`.

<div class="conv-practice-math">
$$
\begin{aligned}
(6+d)^2-6^2 &= 12d+d^2 \\
\frac{12d+d^2}{d} &= 12+d \qquad (d\ne 0)
\end{aligned}
$$
</div>

As the output change `d` becomes very small, the loss's rate of change approaches 12. A weight change is multiplied by 3 at the output, and that output change is currently multiplied by approximately 12 at the loss. The loss's rate of change with respect to the weight is therefore `12×3=36`.

We can check this directly. Changing `w` from 2 to 2.001 gives an output of 6.003 and a loss of 36.036009. Dividing the loss change, 0.036009, by the weight change, 0.001, gives 36.009, close to 36. A gradient of 36 describes the current rate of change; it is not the distance to the best weight.

The rule that multiplies intermediate rates of change to obtain the overall rate is called the **chain rule**. The notation `∂L/∂w` means the rate of change of loss `L` with respect to weight `w`, with other conditions held fixed. Similarly, `∂L/∂z` is the loss's rate of change with respect to the output, and `∂z/∂w` is the output's rate of change with respect to the weight.

<div class="conv-practice-math">
$$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial z}\times\frac{\partial z}{\partial w}$$
</div>

In convolution, the same weight is used at several output positions. Label the outputs `z₀, z₁, …`. When a weight changes, we must add the effects that reach the loss through each of those outputs. `Σ` means to add these terms, and `j` is an index distinguishing the output values.

<div class="conv-practice-math">
$$\frac{\partial L}{\partial w}=\sum_j\left(\frac{\partial L}{\partial z_j}\times\frac{\partial z_j}{\partial w}\right)$$
</div>

Using gradients with respect to the outputs as intermediate values does not turn those outputs into trainable parameters. We follow the calculation paths to obtain the gradients of the weights.[^autograd]

### When are operations recorded, gradients calculated, and weights updated?

Three PyTorch attributes help distinguish the current state of automatic differentiation.

| Attribute | What it tells us |
| --- | --- |
| `requires_grad` | Whether this tensor participates in automatic differentiation |
| `grad_fn` | Backward connection information for a tensor produced by a recorded operation |
| `grad` | A gradient calculated by backpropagation and stored on this tensor |

In ordinary gradient-recording mode, the output of `conv(x)` is connected to automatic differentiation when `conv.weight.requires_grad` is `True`, even if `x.requires_grad` is `False`. We need to calculate how the weights affect the output; we do not need to modify the input data to do that.

The process of calculating outputs is called the **forward pass**. Following the operations backward from the loss to calculate gradients is called **backpropagation**, or the backward pass. Recording operations during the forward pass does not immediately calculate gradients or update weights. `backward()` calculates gradients. An **optimizer** used in training updates its registered parameters when `step()` is called.[^autograd]

### Inspect gradients for six outputs and nine weights

Use the previous 4×5 input and kernel, but create a separate experimental layer called `grad_conv`. Our demonstration loss squares each of the six outputs and takes their average. This is mean squared error with a target of zero for every output; it is not the objective for classifying handwritten digits.

The current output values are `2, 2, -3, 0, -2, 0`, giving the following loss:

<div class="conv-practice-math">
$$L=\frac{2^2+2^2+(-3)^2+0^2+(-2)^2+0^2}{6}=\frac{21}{6}=3.5$$
</div>

`square()` squares each value, and `mean()` calculates the overall average. `retain_grad()` asks PyTorch to keep the intermediate output's gradient in its `.grad` attribute. It does not register the output values as trainable parameters.

To compare the weights before and after backpropagation, we need to keep their earlier values separately. In the code below, `detach()` obtains a tensor without the existing automatic differentiation connection, and `clone()` then makes an independent copy of its numbers. The next section revisits the difference between these two methods.

```python
grad_conv = nn.Conv2d(1, 1, 3, bias=False)
with torch.no_grad():
    grad_conv.weight.copy_(conv.weight)  # Use the same kernel values as before.

weight_before = grad_conv.weight.detach().clone()  # An independent copy for comparison.
grad_out = grad_conv(x)       # Output shape: (1, 1, 2, 3).
grad_out.retain_grad()        # Request storage of the intermediate output's gradient.
grad_loss = grad_out.square().mean()
grad_loss.backward()         # Calculate gradients from the loss back to the weights.

print(grad_loss.item())                 # 3.5
print(tuple(grad_out.grad.shape))      # (1, 1, 2, 3)
print(tuple(grad_conv.weight.grad.shape))  # (1, 1, 3, 3)
print(grad_out.grad[0, 0])
# tensor([[ 0.6667,  0.6667, -1.0000],
#         [ 0.0000, -0.6667,  0.0000]])

# backward calculated gradients, but it did not update the weights.
assert torch.equal(grad_conv.weight.detach(), weight_before)
```

The output gradients correspond to the six output values. We can also check why the first gradient is approximately 0.6667. One output `z` contributes the term `z²/6` to the loss. Changing it by `d` changes the loss by `(2zd+d²)/6`. Dividing by the change `d` gives `(2z+d)/6`. As `d` becomes very small, the rate approaches `2z/6`. At `z=2`, this is `2×2/6=2/3`; at `z=-3`, it is `2×(-3)/6=-1`.

The weight gradients correspond to the nine weights and add contributions from all output positions where each weight is used. For example, the top-left kernel weight multiplied the top-left input values of the six patches: `1, 2, 0, 0, 3, 1`. Its gradient is therefore:

<div class="conv-practice-math">
$$\frac23\times1+\frac23\times2+(-1)\times0+0\times0+\left(-\frac23\right)\times3+0\times1=0$$
</div>

Changing this weight changes individual outputs, but the effects of those paths on the current loss cancel. A zero gradient here does not mean that the weight was unused.

A tensor at the start of the recorded calculation, rather than the result of a recorded operation, is called a **leaf tensor**. `grad_conv.weight` is such a tensor. These trainable parameters receive stored `.grad` values after backpropagation by default. The intermediate output `grad_out` is a non-leaf tensor because it results from an operation. Its intermediate gradient is used during backpropagation but is not stored in `.grad` by default. We called `retain_grad()` to inspect it directly.[^autograd]

A typical training iteration clears previous gradients, performs a forward pass, calculates the loss, calls `backward()`, and then calls `optimizer.step()`. Since `backward()` accumulates gradients into existing `.grad` values, repeated training needs the clearing step. This experiment calculates gradients once on a new layer and performs no weight update. After a weight update, the next forward pass uses the updated weights to calculate its outputs.

## 9. Distinguish `detach()`, `no_grad()`, and frozen weights

`detach()` often appears in code that displays convolution results. This method returns a tensor that does not use the existing automatic differentiation connection. The numbers and shape remain the same, and the return value is still a PyTorch tensor.[^detach]

```python
result = laplace_conv(laplace_x)
detached_result = result.detach()

assert torch.equal(result, detached_result)  # The values match.
assert result.shape == detached_result.shape
assert result.requires_grad is True
assert detached_result.requires_grad is False
assert laplace_conv.weight.requires_grad is True
```

Calling `result.detach()` does not erase the original `result`'s connection or modify the weights. The returned `detached_result` is the tensor that does not use that connection. Nor does `detach()` copy the storage holding the numbers. The two tensors share storage, so modifying the detached tensor's values in place can also affect the original values.

To keep an independent copy of the numbers as well, follow it with `clone()`, which copies the data.

```python
independent_copy = result.detach().clone()  # Detach the connection and copy the data.
```

Simply displaying values with `print(result)` does not require `detach()`. To pass a tensor tracked by automatic differentiation to Matplotlib as an ordinary NumPy array, however, we can use the following sequence. A **NumPy array** is another data type used for numerical arrays in Python.

```python
array_for_plot = result.detach().cpu().numpy()
# detach(): Obtain a tensor without the existing automatic differentiation connection.
# cpu(): Move it to the CPU if it is on a GPU or another device.
# numpy(): Convert it to a NumPy array.
```

The default `numpy()` conversion has requirements, including that the tensor is on the CPU and does not require gradients. `detach()` alone does not move data to the CPU or convert it to NumPy.[^numpy]

If a calculation is only needed for display, we can avoid recording it for backpropagation in the first place.

```python
with torch.no_grad():
    result_without_graph = laplace_conv(laplace_x)

assert result_without_graph.requires_grad is False
assert laplace_conv.weight.requires_grad is True
```

`detach()` obtains a tensor without the connection from an already-calculated result. `no_grad()` prevents the operations in its block from being recorded for backpropagation. Neither permanently changes the weights' own `requires_grad` setting.[^nograd]

If a new layer will only serve as a fixed filter, `laplace_conv.weight.requires_grad_(False)` disables new gradient tracking for those weights. For a model that has already been trained, check any remaining `.grad` values and which parameters the optimizer can update. Changing `requires_grad` to `False` does not erase all previously stored gradients and update state.

This article never updates weights through an optimizer, so the filter values remain unchanged even with `requires_grad=True`. `True` does not mean that the weights have already been learned or that their values are currently changing automatically.

Applying `detach()` to an output needed for a training loss, on the other hand, breaks the gradient path back to earlier weights through that output. Keep the connection for values used to calculate the loss, and obtain a separate detached tensor for display when needed.

## 10. Apply the same Laplacian to an MNIST image

Now apply the calculation from the small numerical arrays to an actual handwritten image. **MNIST** is a dataset of 28×28 grayscale handwritten digit images and their digit labels. It provides 60,000 training images and 10,000 test images. Here, we select the first image in the training set to inspect the filter response. Its label is used only for display.[^mnist]

`torchvision.datasets` provides image dataset loading, while `transforms` provides image transformations. `datasets.MNIST(...)` creates an object that prepares the data, and `mnist[0]` retrieves the first image and label from it.

```python
import time  # Python's standard module for measuring elapsed time.
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # Import plotting functions.

DATA_ROOT = "./data"  # Read or store dataset files here.
started = time.perf_counter()  # Record the start of the measured interval.

mnist = datasets.MNIST(
    DATA_ROOT,
    train=True,                      # Select the training dataset.
    download=True,                   # Download the data if it is missing.
    transform=transforms.ToTensor(),  # Apply this transform when retrieving an image.
)

img, label = mnist[0]  # Unpack the two returned values: (image, label).
print(len(mnist))      # 60000
print(label)          # The first image's label: 5
print(tuple(img.shape))  # (1, 28, 28)
```

`train=True` does not run model training. It selects which dataset split to use. Similarly, `download=True` reuses available data when it is already present.[^mnist]

`transforms.ToTensor()` creates a transform object. Passing that object as `transform` applies it when an image is retrieved. This dataset-construction line does not transform all 60,000 images at once.

For MNIST's 8-bit grayscale images, `ToTensor` divides brightness values from 0–255 by 255 to produce floating-point values from 0–1. The resulting shape is `(channels, height, width)`, or `(1, 28, 28)`. `ToTensor` does not divide every possible kind of input by 255; this description applies to the MNIST images used here.[^totensor]

### Read a chain of operations one step at a time

Use `unsqueeze(0)` to add a batch axis before the existing axes of `img`. **unsqueeze** is a method that inserts an axis of size 1 at a specified position. The `0` here is the insertion position for the new axis, not an instruction to select image index 0.[^unsqueeze]

```python
batch = img.unsqueeze(0)       # (1, 28, 28) → (1, 1, 28, 28).
result = laplace_conv(batch)   # (1, 1, 28, 28) → (1, 1, 26, 26).
detached_result = result.detach()  # Keep the values and shape; detach the connection.
edge = detached_result[0, 0]   # Select the image and output channel to obtain (26, 26).

print(tuple(edge.shape))  # (26, 26)
elapsed = time.perf_counter() - started
print("Data preparation and calculation time in seconds:", round(elapsed, 3))
```

The same code can be written as one line: `edge = laplace_conv(img.unsqueeze(0)).detach()[0, 0]`. Breaking it down in execution order gives:

| Stage | Shape | What happened at this stage |
| --- | --- | --- |
| `img` | `(1, 28, 28)` | The original image with one channel |
| `img.unsqueeze(0)` | `(1, 1, 28, 28)` | Insert a leading batch axis for the number of images |
| `laplace_conv(batch)` | `(1, 1, 26, 26)` | Calculate the convolution using the current Laplacian weights |
| `result.detach()` | `(1, 1, 26, 26)` | Obtain a tensor detached from the automatic differentiation connection |
| `[0, 0]` | `(26, 26)` | Select output channel 0 of image 0 |

The first `1` in `img.shape` is the number of channels. The first `1` in `batch.shape` is the number of images, and the second is the number of channels. `unsqueeze` has not added pixel values or created another image.

The output height and width are each 26 because `(28-3)//1+1=26`. Neither `detach()` nor `[0, 0]` crops the spatial dimensions to make them 26. That change has already occurred in the convolution.

`edge` contains the filter's calculated response, rather than the original image. Naming the variable `edge` does not add an edge-extraction operation. The weights inside `laplace_conv` determine the calculation.

`time.perf_counter()` returns a clock value used to measure elapsed time. The difference between two calls gives the duration between them. The measured interval above includes data preparation, image selection, convolution, and intermediate printing. It therefore represents neither download time alone nor convolution time alone. `round(elapsed, 3)` rounds the duration in seconds to three decimal places.

## 11. Why a grayscale image produces a blue-and-red display

The original image contains brightness values from 0–1. The Laplacian output contains negative, zero, and positive values calculated from the differences between neighboring values and the center. The numbers in the two arrays represent different quantities.

For this kernel, a negative response means the center is brighter than the average of its four neighbors. A positive response means it is darker than that average. Different colors make those signs easier to distinguish. Color is a way to display the calculated numbers, rather than data produced by the convolution itself.

### Choose a color range with zero at the center

An absolute value gives a number's magnitude without its sign. Taking the largest absolute value in `edge.abs()` lets us use an equally sized range on either side of zero.

```python
limit = max(float(edge.abs().max()), 1e-6)
```

Reading from the inside out shows each operation's role.

1. `edge.abs()` returns a new tensor containing the absolute value of each element. It does not overwrite `edge`.
2. `.max()` selects the largest value from that entire tensor and returns it as a tensor.
3. `float(...)` extracts that value as an ordinary Python floating-point number. `.item()` would also work here.
4. The outer `max(..., 1e-6)` is Python's built-in function, which selects the larger of two numbers. `1e-6` is `0.000001`.

For example, if some output values are `[-5, -2, 0, 3]`, their absolute values are `[5, 2, 0, 3]`, with a maximum of 5. We therefore choose a display range from `-5` to `+5`. Even if the data contains no `+5`, equal bounds keep zero at the center.

If every output is zero, the maximum is also zero. Using `max(0.0, 0.000001)` avoids a display range whose two endpoints are both zero. This code sets the color range for the finite values in our example. It is not a general data-cleaning procedure for NaNs or infinities.

The values we plot are the **original signed values in `edge`**, rather than the absolute-value tensor. `abs().max()` supplies only the magnitude of the display range.

### `imshow` maps values to colors using a colormap and a range

`imshow` displays the numbers in an array. `cmap` selects the colormap, and `vmin` and `vmax` specify which numbers correspond to its two ends. We use `coolwarm` to show the two sides of zero with blue and red tones.[^imshow]

With the default linear mapping, the colormap position for a value `v` is calculated as follows. The start of the colormap is 0, and the end is 1.[^colors]

<div class="conv-practice-math">
$$\text{colormap position}=\frac{v-v_{\min}}{v_{\max}-v_{\min}}$$
</div>

For a range from `-5` to `+5`, the value -5 maps to `(-5+5)/10=0`, zero maps to `(0+5)/10=0.5`, and +5 maps to `(5+5)/10=1`. Zero lands exactly at the middle of the colormap.

| Value | Position in `coolwarm` |
| --- | --- |
| `-limit` | The blue end |
| 0 | The light-colored middle |
| `+limit` | The red end |

Specifying the name `coolwarm` alone does not automatically center zero. Zero is centered here because we also set the symmetric range `vmin=-limit`, `vmax=limit`. This display mapping does not modify the values in `edge`.

### Display the original image beside the filter response

`plt.subplots(1, 2)` creates one row with two plotting areas. The returned `fig` is a Figure object managing the whole figure. `axes` is an array containing the individual Axes objects, each representing a plotting area. In this context, Axes does not mean a tensor axis number.

```python
fig, axes = plt.subplots(
    1, 2,                # Create two plotting areas side by side.
    figsize=(8, 3),      # The full figure is 8 inches wide and 3 inches high.
    constrained_layout=True,  # Arrange titles and the colorbar without overlaps.
)

axes[0].imshow(
    img[0].cpu().numpy(),  # Select channel 0 and pass a 28×28 array.
    cmap="gray",          # Display the original brightness values in grayscale.
    vmin=0,
    vmax=1,
    interpolation="nearest",  # Do not interpolate smooth colors between cells.
)
axes[0].set_title(f"MNIST label={label}")  # Insert the actual label at {label}.

im = axes[1].imshow(
    edge.cpu().numpy(),   # The already-detached 26×26 filter response.
    cmap="coolwarm",
    vmin=-limit,          # Use equally sized negative and positive display ranges.
    vmax=limit,
    interpolation="nearest",
)
axes[1].set_title("Fixed Laplacian response")
fig.colorbar(im, ax=axes[1])  # Show im's value-to-color mapping as a legend.

for ax in axes:
    ax.axis("off")  # Hide the plotting area's ticks and border.

plt.show()  # Display the completed figure.
```

`im` is the image display object returned by `imshow`. It is separate from both the input tensor `img` and the output tensor `edge`. The colorbar shows the relationship between values and colors configured on that display object. Likewise, `axis("off")` hides the visible ticks and border; it does not remove tensor axes or data.

<figure class="conv-practice-figure conv-practice-photo">
<img src="/cnn-convolution-practice/mnist-laplacian.png" alt="The first MNIST training image, a handwritten 5 in grayscale, beside its fixed Laplacian response in blue and red with a colorbar." loading="lazy">
<figcaption>Results from the actual first MNIST training image. The original is 28×28 and the Laplacian output is 26×26. Negative and positive values are preserved; ReLU is not applied.</figcaption>
</figure>

In the run that generated this figure, output values ranged from approximately `-1.7686` to `2.0588`. The display range was approximately `-2.0588` to `2.0588`, based on the largest absolute value. The input and output have different array sizes, although each is displayed within its own plotting area. Output position `(r, c)` comes from the 3×3 patch starting at `(r, c)` in the original image. That patch's center is at original image position `(r+1, c+1)`.

Although the display uses color, `edge` remains a 26×26 numerical array. Three RGB channels have not been added. Choosing another colormap or displaying it in grayscale leaves the already-calculated convolution values unchanged.

## 12. Run the complete example and check the results

The [complete script](/cnn-convolution-practice/examples-en.py) performs the same calculations as the article. It includes checks for the input and kernel, all four stride-and-padding combinations, the Laplacian, automatic differentiation, and `detach()`. Use a Python environment with `torch`, `torchvision`, and `matplotlib` available.

```bash
# Run the small-array calculations and autograd checks without downloading data.
python examples-en.py

# Download MNIST if needed and display the filter response.
python examples-en.py --mnist

# Save the figure to a file instead of opening a display window.
python examples-en.py --mnist --save-figure mnist-laplacian.png
```

The convolution examples use small integer values, so `torch.equal` checks for exact matches. More general floating-point calculations can differ slightly with calculation order or device. Checks of floating-point results such as gradients therefore use a comparison with tolerances, such as `torch.testing.assert_close`.[^check]

The script uses `assert` for educational checks. Python can omit these statements when run with optimization enabled, so they should not replace essential input validation. `print` makes results readable, and `imshow` displays already-calculated numbers as an image. The convolution itself runs in calls such as `conv(x)` and `F.conv2d(...)`.

These checks were run with Python 3.12, PyTorch 2.14.1+cpu, and torchvision 0.29.1+cpu. The MNIST figure was also generated from actual data. Execution time depends on the environment and whether a download is needed, so no fixed timing result is given.

Multiplication and summation within one input patch produce one output value. Moving the starting position and reusing the same weights produces the entire output. Automatic differentiation tracks the calculations connecting that output to the weights, while visualization represents the stored output numbers as colors we can read. Identifying which job each line performs makes it possible to follow the numerical calculation, data shapes, learning operations, and display operations in order.

## References

The official documentation below was used to check API input formats and behavior. The small numerical examples and figures were constructed for this article.

[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): input and weight shapes, output size, and cross-correlation.
[^index]: Python, [Subscriptions](https://docs.python.org/3/reference/expressions.html#subscriptions): bracket indexing and passing a tuple.
[^sum]: PyTorch, [torch.sum](https://docs.pytorch.org/docs/stable/generated/torch.sum.html): summing all values or a specified axis.
[^values]: PyTorch, [Tensor.tolist](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html), [Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html).
[^copy]: PyTorch, [Tensor.copy_](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.copy_.html).
[^nograd]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html).
[^relu]: PyTorch, [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html).
[^range]: Python, [Ranges](https://docs.python.org/3/library/stdtypes.html#ranges).
[^functional]: PyTorch, [functional.conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html).
[^pad]: PyTorch, [functional.pad](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html).
[^laplacian]: OpenCV, [Image Filtering: Laplacian](https://docs.opencv.org/5.0/main_modules/imgproc_filter.html#laplacian).
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html): operation recording, gradient tracking, and leaf and non-leaf tensors.
[^detach]: PyTorch, [Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html): detaching from automatic differentiation and sharing storage.
[^numpy]: PyTorch, [Tensor.numpy](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html).
[^mnist]: Torchvision, [MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html).
[^totensor]: Torchvision, [ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html).
[^unsqueeze]: PyTorch, [torch.unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html).
[^imshow]: Matplotlib, [imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html), [subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html).
[^colors]: Matplotlib, [Colormap normalization](https://matplotlib.org/stable/users/explain/colors/colormapnorms.html).
[^check]: PyTorch, [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close); Python, [The assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement).
