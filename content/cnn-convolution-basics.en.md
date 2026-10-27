+++
title = "How a CNN Processes Images: From Channels and Kernels to Linear"
description = "Follow RGB channels into filters, calculate one convolution output by hand, and trace padding, pooling, Flatten, and Linear. Check values and shapes at each step with PyTorch."
date = 2026-10-27T18:12:00+09:00
slug = "cnn-convolution-basics"

[extra]
katex = true
toc = true
styles = ["cnn-convolution-basics/article.css"]
+++

An image classification model receives a picture as an array of numbers. Its first calculations use the brightness or color components stored at each position. Later calculations use new arrays formed by multiplying inputs by weights and adding the results. At the end, the model produces one score for each category it can classify.

A <strong>CNN, or convolutional neural network</strong>, repeatedly applies the same weights to local regions of an image. We will follow <strong>which input values are combined with which weights, where each result is stored, and how many results are produced</strong>.

We begin with the connections between RGB channels and filters. Then we calculate a convolution output using actual numbers. Finally, we run a PyTorch model that receives two grayscale images and produces ten scores for each. The RGB example explains channel connections; the final model takes single-channel input.

Run the code in a Python environment with PyTorch installed. The code blocks build on one another from top to bottom. You can also download the [complete runnable examples](/cnn-convolution-basics/examples-en.py).

## 1. Separate an image from a channel

A <strong>pixel</strong> is a position identified by its vertical and horizontal coordinates in an image. A grayscale image can store one brightness value at each position. An RGB image stores red (R), green (G), and blue (B) components at that same position. A <strong>channel</strong> identifies which component a value belongs to.

One RGB image therefore has three channels. Selecting one channel leaves a two-dimensional array of numbers. Intermediate layers also use channels, but those channels no longer necessarily represent red, green, and blue.

A <strong>Tensor</strong> is the PyTorch data type for storing and calculating with arrays of numbers. A <strong>dimension, or axis</strong>, is one of the coordinates used to locate a value in the array. The <strong>shape</strong> lists the size of each dimension. A group of images processed together is called a <strong>batch</strong>.

The image batches in this article use the following dimension order. An index identifies a position in an array, starting at zero.

| Dimension index | Meaning | Value in `(2, 3, 28, 28)` |
| --- | --- | --- |
| 0 | Number of images, `N` | 2 images |
| 1 | Channels per image, `C` | 3 RGB channels per image |
| 2 | Height of one channel, `H` | 28 positions vertically |
| 3 | Width of one channel, `W` | 28 positions horizontally |

Read one value with `x[image_index, channel_index, vertical_position, horizontal_position]`. For example, `x[0, 1, 2, 3]` selects the value at vertical index 2 and horizontal index 3 in the second channel of the first image.

Having three channels does not make the tensor three-dimensional. Including the batch dimension, `(2, 3, 28, 28)` is a <strong>four-dimensional tensor</strong>. Two grayscale images use the same dimension order and have shape `(2, 1, 28, 28)`.

### What changes as data passes through the model?

A <strong>layer</strong> is a component that performs one stage of a model's computation. A <strong>weight</strong> is a number multiplied by an input value. A <strong>bias</strong> is a number added after the products have been summed. Weights and biases that the model can change during training are called <strong>trainable parameters</strong>.

The diagram below shows the complete model we will build later. First, read what each operation does. We will work through the calculations in the following sections.

| Operation | What it does |
| --- | --- |
| Convolution, `Conv2d` | Multiplies a local input region by weights and sums the products. Repeats this at different positions to create new channels. |
| `ReLU` | Replaces negative values with zero and leaves the remaining values unchanged. |
| Maximum pooling, `MaxPool2d` | Selects the largest value in a small region, separately for each channel. |
| `Flatten` | Arranges each image's values into one sequence without mixing images. |
| `Linear` | Multiplies the flattened values by weights and adds them to produce the final scores. |

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/cnn-shape-flow-en.svg" alt="Two grayscale images with shape (2,1,28,28) pass through two convolution and pooling stages to become (2,64,7,7). Flatten produces (2,3136), and Linear calculates scores with shape (2,10).">
<figcaption>The leading 2 remains the number of images. The intermediate values 32 and 64 are channel counts. The final 10 is the number of output scores per image.</figcaption>
</figure>

A category that the model can predict is called a <strong>class</strong>. The final ten scores correspond to ten classes. Their meanings depend on the dataset: ten outputs alone do not tell us that the model classifies the digits 0 through 9.

## 2. One filter combines the input channels

Some resources use <strong>kernel</strong> and <strong>filter</strong> interchangeably. To make the connections clear, this article uses the following definitions.

| Name | Meaning in this article | For a 3×3 convolution on RGB input |
| --- | --- | --- |
| Per-channel kernel | A two-dimensional weight array applied to one input channel | `(3, 3)` |
| Filter | All the weights used to create one output channel | `(3, 3, 3)`, containing three per-channel kernels |
| Output channel or feature map | A two-dimensional array of results calculated by a filter at multiple positions | For example, `(7, 7)` |

<strong>The numbers inside a kernel are the weights.</strong> A kernel and its weights are not separate ingredients that are combined later. A kernel arranges those weights into a spatial shape.

This explanation uses ordinary convolution with `groups=1`. The `groups` setting divides the connections between input and output channels into groups. With one group, each filter uses every input channel.[^conv]

### Apply two filters to an RGB image

The input is one RGB image with shape `(1, 3, 7, 7)`. Filters A and B each contain a 3×3 kernel for R, another for G, and another for B. The diagram uses `padding=1`, an edge-handling setting explained later, so the output height and width are also 7.

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/rgb-filter-connections-en.svg" alt="Input channels R, G, and B each connect to their corresponding kernels in both filter A and filter B. Each filter sums the calculations from all three channels, adds its own bias once, and creates one output channel." loading="lazy">
<figcaption>Both filters use all three RGB channels. Their kernel values are separate parameters, and each filter creates one output channel.</figcaption>
</figure>

Filter A produces <strong>one value at a particular output position</strong> as follows.

1. Select a 3×3 region from R and multiply it element by element by A's R kernel. Add the nine products.
2. Select the G region at the same vertical and horizontal position. Multiply it by A's G kernel and add the products.
3. Calculate the B region with A's B kernel in the same way.
4. Add the three channel results, then add A's single bias once.
5. Store that number at the corresponding position in output channel A.

One output value uses `3 input channels × 3 × 3 = 27 multiplications`. Repeating the calculation at different spatial positions produces the whole of output channel A. Filter B processes the same input with its own weights. These steps describe how the results depend on one another; they do not require the GPU to execute every multiplication one at a time in this order.

### Weight shape and output shape describe different arrays

`C_in` is the input channel count, `C_out` is the output channel count, and `K_H` and `K_W` are the kernel height and width. With `groups=1`, the complete weight tensor has shape <strong><code>(C_out, C_in, K_H, K_W)</code></strong>.

| Selected item | Shape in this example | Contents |
| --- | --- | --- |
| `weight` | `(2, 3, 3, 3)` | All weights for both filters |
| `weight[0]` | `(3, 3, 3)` | The first filter's R, G, and B kernels |
| `weight[0, 0]` | `(3, 3)` | The first filter's R kernel |
| `weight[0, 0, 0, 0]` | `()` | One weight inside that kernel |
| `bias` | `(2,)` | One bias per output channel |
| Output `y` | `(1, 2, 7, 7)` | Two output channels for one image |
| `y[0, 0]` | `(7, 7)` | The first output channel of the first image |

Shape `()` describes a zero-dimensional tensor containing one number. Shape `(2,)` describes a one-dimensional tensor containing two numbers.

This layer has `2×3×3×3=54` weights and `2` biases. Its output contains `1×2×7×7=98` numbers. <strong>The filter count, weight count, and output element count are three different quantities.</strong>

A feature map is <strong>a result calculated from the input and the current weights</strong>. The next layer receives that result as input. The filters themselves are not passed to the next layer as input data.

### Where are the same weights used again?

Filter A reuses its weights at different positions in the input. When a batch contains several images, it uses the same weights for every image as well. This is called <strong>weight sharing</strong>.

Filter B has its own weights, separate from A's. Within A, the R and G kernels also have separate weights. Training can change their values. Sharing means <strong>using one set of model parameters at multiple positions and for multiple images</strong>.

## 3. What do you specify in `nn.Conv2d`?

`torch` is the name used to access PyTorch's functions and types. `from torch import nn` imports `torch.nn`, which provides neural-network layers, under the name `nn`. The `torch.nn.functional` module contains calculation functions such as ReLU; we use the shorter name `F` for it below.

`nn.Conv2d` is a <strong>class</strong> that creates a convolution layer. A Python class defines the structure and behavior of an object; this is a different meaning of “class” from a prediction category. Calling the class creates an actual <strong>object</strong> to use in the program.

```python
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
```

The `2d` in `Conv2d` refers to the two spatial directions: height and width. It does not mean that the complete input tensor, including batch and channels, is two-dimensional.

The same settings can be written as `nn.Conv2d(3, 32, 3, padding=1)`. The first three arguments are <strong>input channels, output channels, and kernel size</strong>, in that order. Batch size is not one of these arguments.

| Setting or parameter | What happens |
| --- | --- |
| `in_channels` | Match it to the actual channel count of the incoming data. |
| `out_channels` | Choose how many filters this layer will create. |
| `kernel_size` | Choose the spatial region read from each channel at one position. |
| `weight`, `bias` | The layer generates initial values. Training then changes the numbers inside these tensors. |

`out_channels=32` is a <strong>configuration argument</strong> to the constructor. The numbers inside `conv_example.weight` are <strong>trainable parameters</strong>. Ordinary training changes the numbers within the 32 filters; it does not increase the filter count from 32.

Calling `nn.Conv2d(...)` creates the layer and its initial weights. To calculate a convolution, pass input to the created layer, as in `conv_example(x)`.

### Match one layer's output channels to the next layer's input channels

| Stage | Input shape | Layer settings | Weight shape | Output shape |
| --- | --- | --- | --- | --- |
| First convolution | `(1, 3, 7, 7)` | `Conv2d(3, 2, 3, padding=1)` | `(2, 3, 3, 3)` | `(1, 2, 7, 7)` |
| Second convolution | `(1, 2, 7, 7)` | `Conv2d(2, 4, 3, padding=1)` | `(4, 2, 3, 3)` | `(1, 4, 7, 7)` |

The second layer receives the two channels created by the first layer. Its input is no longer RGB, so each of its filters needs two 3×3 kernels. Input and output channel counts do not have to be equal, but <strong>the incoming channel count must match the receiving layer's <code>in_channels</code></strong>.

Kernels do not have to be square. `kernel_size=(3, 5)` means height 3 and width 5. Every filter within the same `Conv2d` layer uses the specified spatial kernel size.[^conv]

## 4. Output size counts the valid starting positions

Start with a 7×7 input, no added edge values, and a 3×3 kernel that moves one position at a time. The default conditions for this section are <strong><code>stride=1</code>, <code>padding=0</code>, and <code>dilation=1</code></strong>.

The horizontal input indices are `0, 1, 2, 3, 4, 5, 6`. A region three positions wide fits in the following places.

| Starting index | Selected input indices | Corresponding horizontal output index |
| ---: | --- | ---: |
| 0 | 0, 1, 2 | 0 |
| 1 | 1, 2, 3 | 1 |
| 2 | 2, 3, 4 | 2 |
| 3 | 3, 4, 5 | 3 |
| 4 | 4, 5, 6 | 4 |
| 5 | 5, 6, 7 | Invalid: input index 7 does not exist. |

Starting positions run from 0 through 4, giving <strong>five calculation positions</strong>. Moving from the first to the last takes four steps, but the first position also produces a result. There are five valid positions vertically as well, so one filter produces a 5×5 output.

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/padding-start-positions-en.svg" alt="A length-three kernel fits at five starting positions, indices 0 through 4, in a length-seven input. One padding position on each side increases the length to nine and allows seven starting positions." loading="lazy">
<figcaption>Kernel size describes the region read at once. Output size counts the positions where that region can be placed and calculated.</figcaption>
</figure>

### Why does `L - K + 1` need the final `+1`?

Let `L` be the input length, `K` the kernel length, and `s` the starting index. The final input index is `L-1`. A kernel starting at `s` ends at index `s+K-1`.

<div class="cnn-math">
$$
\begin{aligned}
s+K-1 &\le L-1 &&\text{The kernel must end within the input.}\\
s &\le L-K &&\text{Subtract }K-1\text{ from both sides.}
\end{aligned}
$$
</div>

The possible starting indices are `0, 1, …, L-K`. Counting both endpoints gives `(L-K)-0+1`, or <strong><code>L-K+1</code></strong>. The `+1` does not add padding; it includes the starting position at index zero in the count.

Substituting numbers gives `7-3+1=5` and `28-3+1=26`. A 3×3 kernel does not always produce a 5×5 output. A different input size changes how many positions are available.

### Padding extends the region available for calculation

<strong>Padding</strong> supplies values outside the input's edges for the calculation. With `padding=1`, the calculation includes one extra position at the top, bottom, left, and right. The default mode, `padding_mode="zeros"`, treats those added positions as zero. By contrast, the default argument `padding=0` means that no positions are added.

Adding one position to each end of a length-28 input gives a calculation length of 30. A length-three kernel then fits at `30-3+1=28` positions.

<div class="cnn-math">
$$28+\underbrace{1+1}_{\text{padding on both sides}}-3+\underbrace{1}_{\text{include the first position}}=28$$
</div>

The same reasoning applies vertically and horizontally, preserving the 28×28 size. <strong>Equal sizes do not mean equal input and output values.</strong> `Conv2d` also handles this padding internally: the original input's `x.shape` does not become 30×30.

| Input shape | `padding` | Spatial size including padding | Output shape with 32 filters |
| --- | ---: | --- | --- |
| `(2, 1, 7, 7)` | 0 | 7×7 | `(2, 32, 5, 5)` |
| `(2, 1, 7, 7)` | 1 | 9×9 | `(2, 32, 7, 7)` |
| `(2, 1, 28, 28)` | 0 | 28×28 | `(2, 32, 26, 26)` |
| `(2, 1, 28, 28)` | 1 | 30×30 | `(2, 32, 28, 28)` |

All four cases use weight shape `(32, 1, 3, 3)` and bias shape `(32,)`.

### How much padding preserves the spatial size?

Let `P` be the padding length on one side. With `stride=1`, `dilation=1`, and equal padding on both sides, output length is `L+2P-K+1`. Set this equal to the input length `L`.

<div class="cnn-math">
$$
\begin{aligned}
L &= L+2P-K+1 &&\text{Set input and output lengths equal.}\\
0 &= 2P-K+1 &&\text{Subtract }L\text{ from both sides.}\\
2P &= K-1 &&\text{Add }K-1\text{ to both sides.}\\
P &= \frac{K-1}{2} &&\text{Divide both sides by 2.}
\end{aligned}
$$
</div>

| Kernel size | Padding required on each side | Setting |
| --- | --- | --- |
| 3×3 | 1 vertically and horizontally | `padding=1` |
| 5×5 | 2 vertically and horizontally | `padding=2` |
| 7×7 | 3 vertically and horizontally | `padding=3` |
| 3×5 | 1 at the top and bottom; 2 at the left and right | `padding=(1, 2)` |

You can also use `padding="same"` to let the layer preserve spatial size. PyTorch's `Conv2d` supports this setting with `stride=1`. The output <strong>channel count</strong> still comes from `out_channels`.[^conv]

<details>
<summary>Output length when stride or dilation changes</summary>

<strong>Stride</strong>, `S`, is the distance between starting positions. The positions are `0, S, 2S, …`, so with `dilation=1`, the output length is:

<div class="cnn-math">
$$\left\lfloor\frac{L+2P-K}{S}\right\rfloor+1$$
</div>

The <strong>floor</strong> operation returns the largest integer no greater than its input. For valid size settings here, this is equivalent to discarding the fractional part of the division.

<strong>Dilation</strong>, `D`, is the spacing between the input positions used by neighboring kernel weights. It can spread the application positions apart without adding weights. The kernel's effective spatial extent is `K_effective=D×(K-1)+1`. Replace `K` with `K_effective` in the formula above. Calculate height and width separately.

For an even kernel size, `(K-1)/2` is not an integer, so preserving size may require unequal padding on the two sides. Simply discarding the fractional part and applying equal padding can produce the wrong output size.

</details>

## 5. Calculate one convolution output

This example uses one image, one input channel, and 32 filters. The complete input shape is `(1, 1, 7, 7)`. Fill its single channel with the numbers 1 through 49 in order.

<div class="cnn-math">
$$
X=\begin{bmatrix}
1&2&3&4&5&6&7\\
8&9&10&11&12&13&14\\
15&16&17&18&19&20&21\\
22&23&24&25&26&27&28\\
29&30&31&32&33&34&35\\
36&37&38&39&40&41&42\\
43&44&45&46&47&48&49
\end{bmatrix}
$$
</div>

A newly created `Conv2d` has random weights drawn according to its initialization rule. To compare its output with a manual calculation, we will <strong>set only the first of the 32 filters ourselves</strong>. These values were not learned. The other 31 filters keep their initial values.

### Multiply matching elements, sum them, and add the bias

A <strong>patch</strong> is the small input region selected for the current calculation. At the first position, select the top-left 3×3 region. Set the first filter's kernel and bias as follows.

<div class="cnn-math">
$$
\text{patch}=\begin{bmatrix}1&2&3\\8&9&10\\15&16&17\end{bmatrix},
\qquad
K=\begin{bmatrix}1&2&-1\\-2&1&3\\2&-1&1\end{bmatrix},
\qquad b=2
$$
</div>

`patch * K` performs <strong>elementwise multiplication</strong>. It multiplies values at matching positions, so the result is still 3×3. This differs from the matrix multiplication operator, `@`.

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/convolution-one-position-en.svg" alt="Multiplying the first 3×3 input patch elementwise by the kernel produces rows 1, 4, -3; -16, 9, 30; and 30, -16, 17. The nine values sum to 56. Adding bias 2 gives the first output value, 58." loading="lazy">
<figcaption>Completing one output position requires summing all the products. This diagram uses one input channel. For RGB, add the results from all three channels before adding the bias once.</figcaption>
</figure>

`torch.tensor` creates a tensor from the supplied values. `torch.arange(1, 50)` generates the consecutive numbers from 1 up to, but not including, 50. `reshape` arranges the values into the requested shape without changing their order. The setting `dtype=torch.float32` stores each value as a 32-bit floating-point number.

```python
x7 = torch.arange(1, 50, dtype=torch.float32).reshape(1, 1, 7, 7)
# One image, one channel, and a spatial size of 7×7.

kernel3 = torch.tensor(
    [[1, 2, -1], [-2, 1, 3], [2, -1, 1]],
    dtype=torch.float32,  # Use the same floating-point type as the convolution weights.
)

patch = x7[0, 0, 0:3, 0:3]  # Select rows 0–2 and columns 0–2 in the first image/channel.
products = patch * kernel3  # Multiply matching elements: the result has shape (3, 3).
```

The slice `0:3` begins at zero and <strong>excludes the endpoint, 3</strong>. It therefore selects indices 0, 1, and 2.

### Why is `products` not a single number?

`products` is a variable name chosen to mean “multiplication results.” It is not a special PyTorch name. It contains nine products that have not yet been added together.

```python
print(products.tolist())  # Convert the tensor to Python lists to inspect its values.
# [[1.0, 4.0, -3.0], [-16.0, 9.0, 30.0], [30.0, -16.0, 17.0]]

print(products.sum(dim=1).tolist())  # Sum the three values in each row: [2.0, 23.0, 31.0].
print(products.sum().item())  # Sum all nine values: 56.0.
print((products.sum() + 2).item())  # Add bias 2 to complete one output value: 58.0.
```

`sum` is a method that adds values. For a `(3, 3)` tensor, `dim=1` sums along the second dimension. The first row gives `1+4-3=2`, the second gives `-16+9+30=23`, and the third gives `30-16+17=31`. With no `dim` argument, it adds every element.[^sum]

| Code | Actual result | Result shape |
| --- | --- | --- |
| `patch * kernel3` | Nine products | `(3, 3)` |
| `products.sum(dim=1)` | `[2, 23, 31]` | `(3,)` |
| `products.sum()` | `56` | `()` |
| `products.sum() + 2` | `58` | `()` |

`.tolist()` converts tensor values to Python lists or scalar values. `.item()` extracts a Python number from a tensor containing one element. Printing row sums is <strong>a way to inspect an intermediate calculation</strong>. The convolution does not return that row-sum list as its final output.

### Why copy a kernel into the weights?

Now let `Conv2d` perform the same calculation. `copy_` replaces an existing tensor's contents with supplied values. `fill_` fills the selected tensor with one value. The trailing `_` in these method names indicates that they modify the existing tensor itself.

`torch.no_grad()` prevents operations in its block from being recorded for backpropagation. We use it here while assigning known values and checking a forward pass against the manual calculation. It does not permanently freeze the weights.[^no-grad]

```python
conv3 = nn.Conv2d(
    in_channels=1, out_channels=32, kernel_size=3,
    stride=1, padding=0, bias=True,
)  # Create a layer with one input channel and 32 filters.

with torch.no_grad():  # Do not record this block's operations for backpropagation.
    conv3.weight[0, 0].copy_(kernel3)  # Set the first filter's first per-channel kernel.
    conv3.bias[0].fill_(2.0)  # Set the first output channel's bias to 2.
    y3 = conv3(x7)  # Calculate the convolution: complete output shape (1, 32, 5, 5).

print(y3[0, 0])  # Inspect only the first output channel of the first image.
```

The complete first output channel is:

<div class="cnn-math">
$$
\begin{bmatrix}
58&64&70&76&82\\
100&106&112&118&124\\
142&148&154&160&166\\
184&190&196&202&208\\
226&232&238&244&250
\end{bmatrix}
$$
</div>

Its first value is `y3[0, 0, 0, 0]=58`. One position to the right, the patch becomes `[[2,3,4],[9,10,11],[16,17,18]]`. Applying the same weights gives `64`. Calculating all 25 positions produces one 5×5 array. The other 31 filters process the same input with their own weights, so <strong>the complete output shape is <code>(1, 32, 5, 5)</code></strong>.

We can also calculate every position ourselves and compare the results. `empty_like` allocates a tensor with the same shape, data type, and device as the supplied tensor, without initializing its numerical values. The loops below fill every position.

```python
manual3 = torch.empty_like(y3[0, 0])  # Allocate (5, 5) storage matching the first output channel.
with torch.no_grad():
    for row in range(y3.shape[2]):  # Select vertical output indices 0–4.
        for col in range(y3.shape[3]):  # Select horizontal output indices 0–4.
            patch = x7[0, 0, row:row + 3, col:col + 3]  # The 3×3 patch at this position.
            manual3[row, col] = (patch * kernel3).sum() + conv3.bias[0]

assert torch.allclose(manual3, y3[0, 0])  # Compare the results with a small floating-point tolerance.
```

An `assert` statement stops execution if its condition is false. `torch.allclose` returns whether the two tensors' values match within the allowed floating-point tolerance.

<details>
<summary>What happens to the first value with the same weights and padding=1?</summary>

The top-left 3×3 region is `[[0,0,0],[0,1,2],[0,8,9]]`. The added padding zeros are included in the multiplication. Applying the same kernel gives a total of 8, and adding bias 2 produces <strong>10</strong>. More starting positions become available, so the first output channel is 7×7.

PyTorch's `Conv2d` multiplies matching positions without flipping the kernel. Mathematically, this operation is called <strong>cross-correlation</strong>, though it is conventionally called convolution in CNNs.[^conv]

</details>

## 6. How does a larger kernel change output size and computation?

With a 5×5 kernel on the same 7×7 input, the possible starting positions are `0, 1, 2` in each spatial direction. One output channel is therefore <strong>3×3</strong>. The movement is still controlled by `stride=1`; a kernel size of 5 does not automatically move the kernel five positions at a time.

The following code creates a <strong>separate layer</strong> with a 5×5 kernel. It does not place this layer after the previous 3×3 layer. Both layers receive the same `x7` input so that we can compare their calculations.

```python
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
```

Each position now requires more multiplications, but there are fewer positions. To find the total amount of computation, we need both counts.

| Quantity | Calculation with `groups=1` |
| --- | --- |
| Number of weights | `C_out × C_in × K_H × K_W` |
| Multiplications for one output value | `C_in × K_H × K_W` |
| Output values for the entire batch | `N × C_out × H_out × W_out` |
| Total multiplications using direct calculation | `N × C_out × H_out × W_out × C_in × K_H × K_W` |

The comparison below uses one image, one input channel, 32 filters, and stride 1. It counts multiplications only, excluding additions.

| Input size | Kernel | Padding | Output size | Output positions per filter | Multiplications per output value | Total multiplications |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 7×7 | 3×3 | 0 | 5×5 | 25 | 9 | 7,200 |
| 7×7 | 5×5 | 0 | 3×3 | 9 | 25 | 7,200 |
| 28×28 | 3×3 | 0 | 26×26 | 676 | 9 | 194,688 |
| 28×28 | 5×5 | 0 | 24×24 | 576 | 25 | 460,800 |
| 28×28 | 3×3 | 1 | 28×28 | 784 | 9 | 225,792 |
| 28×28 | 5×5 | 2 | 28×28 | 784 | 25 | 627,200 |

The two 7×7 cases happen to have equal totals because `25×9` equals `9×25`. This is not a general rule. <strong>When output size and all other conditions stay the same, a larger kernel requires more multiplications.</strong> If padding is fixed and output size changes, calculate the total again using that new output size.

These counts assume direct multiplication of matching elements. Actual GPU time also depends on the implementation and hardware, so the counts cannot be converted directly into execution time.

## 7. ReLU and pooling do different jobs

Between convolution layers, we can add operations that change the intermediate values or reduce their spatial size. This model applies ReLU after each convolution, then maximum pooling.

### ReLU changes values while preserving shape

An <strong>activation function</strong> transforms the values calculated by a layer. ReLU applies `max(0, value)` to each number. Negative values become zero; positive values and zeros remain unchanged.[^relu]

<div class="cnn-math">
$$
\begin{bmatrix}-2&3&0\\4&-1&5\end{bmatrix}
\quad\xrightarrow{\text{ReLU}}\quad
\begin{bmatrix}0&3&0\\4&0&5\end{bmatrix}
$$
</div>

The positions that contained negative values are not removed, so the shape stays the same. Applying ReLU to `(2, 32, 28, 28)` still produces `(2, 32, 28, 28)`.

ReLU treats negative and positive inputs differently. Adding this <strong>nonlinear transformation</strong> lets the model represent relationships that cannot be expressed by one multiplication by fixed weights followed by adding a bias.

### Maximum pooling selects a region's largest value within each channel

`nn.MaxPool2d(kernel_size=2)` selects the largest value in a 2×2 region within one channel. If `stride` is omitted, it defaults to the kernel size, 2. This differs from the default stride of 1 in `Conv2d`.[^pool]

| A 4×4 input in one channel | Result of 2×2 maximum pooling |
| --- | --- |
| `[[1,3,2,0], [4,2,5,1], [0,1,7,3], [2,6,4,8]]` | `[[4,5], [6,8]]` |

The top-left region, `[[1,3],[4,2]]`, produces 4. The top-right region produces 5. The two lower regions are processed in the same way. The next channel is processed separately using its own values. <strong>This operation neither compares maxima across channels nor combines the channels.</strong>

As a result, `(2, 32, 28, 28)` becomes `(2, 32, 14, 14)`. The image count, 2, and channel count, 32, stay unchanged; only the height and width decrease.

For length 7, a length-two region fits at starting positions 0, 2, and 4. Starting at 6 would require index 7, which is outside the input. With the default settings, only the three fully contained positions are calculated, giving <strong><code>7 → 3</code></strong>. This model uses the default `ceil_mode=False` and no padding.

Maximum pooling has no trainable weights. Its rule selects a region and returns the largest value inside it.

## 8. Flatten arranges each image's values into a sequence

Suppose the final pooling result has shape `(2, 64, 7, 7)`. Each of the two images has 64 feature maps of size 7×7. There are <strong><code>64×7×7=3136</code> numbers per image</strong>.

`flatten(start_dim=1)` leaves dimension 0, the batch dimension, intact and combines dimension 1 through the final dimension. It does not add or average values. It arranges the same values in order to produce <strong><code>(2, 3136)</code></strong>.[^flatten]

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/pooling-flatten-en.svg" alt="Maximum pooling selects the largest value in each 2×2 region within each channel. Flatten keeps the two images separate and arranges each image's 64 channels of 7×7 values into a sequence of 3136 numbers." loading="lazy">
<figcaption>The numbers at the top are a separate example of per-channel pooling. Pooling selects values and reduces spatial size; Flatten lays out all remaining values without combining them.</figcaption>
</figure>

| Input dimension | After Flatten |
| --- | --- |
| Dimension 0: batch size 2 | Remains 2 |
| Dimension 1: channels 64; dimension 2: height 7; dimension 3: width 7 | Combined into one dimension of size 3136 |

The `1` in `start_dim=1` does not mean “make the entire result one-dimensional.” It is <strong>the dimension index where flattening begins</strong>. Because we preserve the batch dimension, the complete result `(2, 3136)` is still two-dimensional.

Calling `flatten()` without arguments on this input would also combine the batch dimension, producing `(6272,)`. The shape would no longer separate the two images, which is unsuitable for this model.

## 9. Where do the two numbers in `Linear(3136, 10)` come from?

`nn.Linear(in_features, out_features)` creates a layer that multiplies input values by weights and adds the products to make output values. `in_features` is <strong>the number of input values per sample</strong>; `out_features` is <strong>the number of output values per sample</strong>. Batch size is not included in these arguments either.[^linear]

| Number | Why it has this value |
| ---: | --- |
| 3136 | The preceding layers leave `64×7×7` values per image. |
| 10 | This model is designed to produce one score for each of ten classes. |

The convolution layers reuse weights across positions to produce intermediate feature values. The final Linear layer uses <strong>all 3136 flattened values together to produce ten class scores for the whole image</strong>. Linear layers are not limited to regression; they also produce scores in classification models.

### Each output score needs 3136 weights

Let `x[0], …, x[3135]` be the flattened values for one image. The class index `j` runs from 0 through 9. `W[j,i]` is the weight connecting input `i` to the score for class `j`, and `b[j]` is that score's bias.

<div class="cnn-math">
$$\text{score}[j]=x[0]W[j,0]+x[1]W[j,1]+\cdots+x[3135]W[j,3135]+b[j]$$
</div>

One output score needs 3136 weights. Ten scores need ten such sets. PyTorch stores them as <strong><code>weight.shape=(10, 3136)</code></strong>, with <strong><code>bias.shape=(10,)</code></strong>.

<figure class="cnn-figure">
<img src="/cnn-convolution-basics/linear-weight-layout-en.svg" alt="Linear receives input shaped (2,3136) and stores weights shaped (10,3136). Multiplication by the transposed weights, shaped (3136,10), followed by adding the class biases produces scores shaped (2,10)." loading="lazy">
<figcaption>One weight row contains the 3136 weights used to calculate one output score. Both images use the same weights.</figcaption>
</figure>

<strong>Transposing</strong> a two-dimensional matrix swaps its rows and columns; in code, write `W.T`. The <strong>matrix multiplication</strong> operator, `@`, multiplies corresponding entries from one left-hand row and one right-hand column, then adds the products. The two inner sizes must match. Here the calculation is:

<div class="cnn-math">
$$\underbrace{X}_{(2,3136)}\;@\;\underbrace{W^{\mathsf T}}_{(3136,10)}+\underbrace{b}_{(10,)}=\underbrace{Y}_{(2,10)}$$
</div>

The same ten bias values are added to each image's ten scores. If you created weights yourself with shape `(input_count, output_count)`, you could use `X @ W`. PyTorch's `nn.Linear` stores them as `(output_count, input_count)`, so it performs the equivalent of `X @ W.T + b`.

### The final ten values are not probabilities yet

<strong>Logits</strong> are class scores before conversion to probabilities. They can be negative, and the ten scores do not have to sum to one. Selecting the position of the largest score gives the predicted class index.

During training, compare logits shaped `(N, 10)` with targets shaped `(N,)`, containing one target class index per image. The numerical training measure obtained from this comparison is called <strong>loss</strong>. <strong>Softmax</strong> converts one image's class scores into probabilities that sum to one. `CrossEntropyLoss` accepts logits directly, so do not apply Softmax before passing the scores to this loss.[^loss]

## 10. Check the complete model's shapes and parameter count

Now set the input to <strong>two grayscale images, each 28×28</strong>, and examine the whole model. The convolutions preserve spatial size by using 3×3 kernels with padding 1. Two maximum-pooling steps reduce each spatial dimension through `28 → 14 → 7`.

| Stage | Input shape | Weight shape | Output shape |
| --- | --- | --- | --- |
| `Conv2d(1, 32, 3, padding=1)` | `(2, 1, 28, 28)` | `(32, 1, 3, 3)` | `(2, 32, 28, 28)` |
| ReLU | `(2, 32, 28, 28)` | None | `(2, 32, 28, 28)` |
| `MaxPool2d(2)` | `(2, 32, 28, 28)` | None | `(2, 32, 14, 14)` |
| `Conv2d(32, 64, 3, padding=1)` | `(2, 32, 14, 14)` | `(64, 32, 3, 3)` | `(2, 64, 14, 14)` |
| ReLU | `(2, 64, 14, 14)` | None | `(2, 64, 14, 14)` |
| `MaxPool2d(2)` | `(2, 64, 14, 14)` | None | `(2, 64, 7, 7)` |
| `flatten(start_dim=1)` | `(2, 64, 7, 7)` | None | `(2, 3136)` |
| `Linear(3136, 10)` | `(2, 3136)` | `(10, 3136)` | `(2, 10)` |

One filter in the second convolution has shape <strong><code>(32, 3, 3)</code></strong> because it uses all 32 input channels from the preceding stage. There are 64 such filters, giving the complete weight shape `(64, 32, 3, 3)`.

Count the trainable numbers, including biases.

| Layer | Weights | Biases | Total |
| --- | ---: | ---: | ---: |
| First convolution | `32×1×3×3 = 288` | 32 | 320 |
| Second convolution | `64×32×3×3 = 18,432` | 64 | 18,496 |
| Final Linear | `10×3136 = 31,360` | 10 | 31,370 |
| Entire model | 50,080 | 106 | <strong>50,186</strong> |

ReLU, maximum pooling, and Flatten have no trainable weights in this model. Changing the batch from two images to twenty still leaves 50,186 parameters. It increases the amount of data processed with those same parameters.

### You can determine `64×7×7` before training

The final channel count, 64, comes from the developer's choice of `out_channels=64`. Height and width 7 come from the input size and the preceding operations. Training is not required to discover these numbers.

Different input image sizes change the intermediate shapes as follows.

| Original input size | After first pool | After second pool | Flattened values per image | Existing `Linear(3136,10)` |
| --- | --- | --- | ---: | --- |
| 28×28 | 14×14 | 7×7 | 3136 | Compatible |
| 7×7 | 3×3 | 1×1 | 64 | Input count does not match |
| 32×32 | 16×16 | 8×8 | 4096 | Input count does not match |

A 7×7 intermediate result does not mean that the original input image was 7×7. Default pooling also excludes incomplete regions, so input lengths from 29 through 31 become 7 after two pooling steps in this architecture. Therefore, `Linear(3136,10)` alone cannot prove that the original input was exactly 28×28.

A compatible input shape also does not tell us whether the model was trained on images of that size.

## 11. Build the PyTorch model and inspect its intermediate shapes

`nn.Module` is the base class PyTorch uses to manage a model's child layers and parameters. <strong>Inheritance</strong> is the syntax for defining a new class based on the functionality of an existing class. We use it to define the layers and computation order inside `MyCNN`.[^module]

| Syntax or method | Role |
| --- | --- |
| `__init__` | Runs when `MyCNN()` creates an object. Prepares the layers. |
| `self` | Refers to the current model object. Layers are stored in attributes such as `self.conv1`. |
| `super().__init__()` | Runs the initialization provided by the parent class, `nn.Module`. |
| `features` | A method name chosen for this example. Calculates the convolution, ReLU, and pooling stages. |
| `forward` | Defines how to calculate and return a result from the actual input. |
| `model(x)` | Runs `forward` through PyTorch's module-calling machinery. |

The name `fc` refers to a <strong>fully connected layer</strong>, which connects all input values to the calculation of each output. Here, it stores an `nn.Linear` object.

Data does not automatically flow through layers in the order they appear in `__init__`. <strong>The execution order in <code>forward</code> and the methods it calls determines the computation order.</strong>

The following block can run on its own, independently of the manual convolution examples above. Passing `trace=True` prints each intermediate shape. The `trace` argument is an inspection option added for this example.

```python
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
```

The intermediate shapes are shown below. Notice where Flatten changes the dimension arrangement.

```text
Input:   (2, 1, 28, 28)
conv1:   (2, 32, 28, 28)
ReLU 1:  (2, 32, 28, 28)
Pool 1:  (2, 32, 14, 14)
conv2:   (2, 64, 14, 14)
ReLU 2:  (2, 64, 14, 14)
Pool 2:  (2, 64, 7, 7)
Flatten: (2, 3136)
Linear:  (2, 10)
```

Written as `x = self.pool(F.relu(self.conv1(x)))`, the calculation starts with the innermost call, `conv1(x)`. ReLU receives its result, then pooling receives the ReLU result. The code above separates these same operations so that their intermediate shapes can be inspected.

`eval()` switches the behavior of layers that act differently during training and evaluation. The layers used here do not change their calculation rules when this mode changes. `no_grad()` separately disables automatic differentiation recording; these two settings do not replace one another.[^module][^no-grad]

At this point, <strong>the model has been created and its connections have been checked</strong>. Even a zero-filled input need not produce only zeros, because the model has biases. An output shape of `(2,10)` also does not show that the model can classify images well.

### Check the parameter count and the Linear calculation

`named_parameters()` provides each parameter's name and tensor. `numel()` returns the number of elements in a tensor. `parameters()` provides the parameter tensors without their names.

```python
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
```

`argmax(dim=1)` returns the <strong>index</strong> of the largest of the ten scores for each image. Replacing ten scores with one class index gives result shape `(2,)`. The model has not been trained, so these indices do not provide evidence of classification performance.

## 12. Training also needs targets, loss, backpropagation, and updates

The <strong>forward pass</strong> calculates values from input to output. <strong>Loss</strong> is a numerical training measure that compares outputs with targets. A <strong>gradient</strong> describes the direction and rate of change in loss when a parameter changes slightly. <strong>Backpropagation</strong> works backward through the recorded computation to calculate parameter gradients. An <strong>optimizer</strong> uses those gradients to update the parameters.[^training]

| Stage | Code | Result |
| --- | --- | --- |
| Create the model | `MyCNN()` | Creates layers and initial weights. |
| Select training mode | `model.train()` | Switches to training mode without changing weights by itself. |
| Clear previous gradients | `optimizer.zero_grad(...)` | Clears gradients from the previous batch. |
| Forward pass | `model(images)` | Produces class scores for each image. |
| Calculate loss | `F.cross_entropy(logits, labels)` | Calculates mean loss from the scores and targets. |
| Backpropagation | `loss.backward()` | Calculates gradients for the parameters. |
| Update parameters | `optimizer.step()` | Uses the gradients to change weights and biases. |

Feature maps are recalculated from the input and current weights. Training changes <strong>the weights and biases that produce those feature maps</strong>.

When one weight is used at several positions and for several images, those uses contribute to the gradient for that weight. The model does not create a separate trainable weight at every position. `weight.grad` has the same shape as `weight`, but its calculation includes contributions from multiple positions and images.

The function below performs one training step using an image batch and its corresponding targets. `images` is a floating-point tensor shaped `(N,1,28,28)`. `labels` is an integer tensor shaped `(N,)`, containing class indices from 0 through 9. Matching positions in the two tensors must refer to the same image.

```python
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
```

`next()` retrieves the next item from an iterator. Here it reads the first parameter so we can find the model's device. `torch.long` is a 64-bit integer data type. The `torch.optim` module provides optimizers such as Adam.[^adam]

This block <strong>defines</strong> the function and prepares the optimizer. The final call is commented out, so the block does not start training automatically. Training a real classifier requires correctly matched images and targets, repeated processing of batches, and evaluation on data that was not used for training.

## 13. Where should you look when the shapes do not match?

When an error occurs, check <strong>the actual input → each layer's output → the flattened value count → Linear's required input count</strong> in order. Changing numbers based only on layer names can leave another connection mismatched.

| Symptom | First check |
| --- | --- |
| The input channel count does not match. | Compare the actual input's `shape[1]` with the current `Conv2d.in_channels`. |
| Convolution output is spatially smaller than expected. | Check `kernel_size`, `padding`, and `stride`. The default padding is 0. |
| Pooling output is smaller than expected. | Remember that `MaxPool2d(2)` also defaults to stride 2. |
| Linear reports a matrix size error. | Compare the flattened tensor's `shape[1]` with `fc.in_features`. |
| Two images were flattened into one sequence. | Check that you used `flatten(start_dim=1)`. |
| A CPU/GPU device error occurs. | Check that the input and model parameters are on the same device. |

If calculating the size by hand is awkward, flatten the result of `my_model.features(probe)` and read its `shape[1]`, as above. This does not require training.

### LazyLinear determines the required size from its first input

`nn.LazyLinear(10)` infers `in_features` from the final dimension of its first input. Receiving `(2,3136)` first sets the input size to 3136. Passing `(2,4096)` later does not automatically create a redesigned layer. Once initialized, the layer requires the input size to match.[^lazy]

### Another design can collect a fixed number of values from different spatial sizes

`nn.AdaptiveAvgPool2d((1,1))` averages each channel's spatial values to produce one value per channel. <strong>Average pooling</strong> adds the values in a region and divides by their count.[^adaptive]

| Stage | Shape |
| --- | --- |
| Intermediate convolution output | `(N, 64, H, W)` |
| `AdaptiveAvgPool2d((1,1))` | `(N, 64, 1, 1)` |
| `flatten(start_dim=1)` | `(N, 64)` |
| `Linear(64,10)` | `(N, 10)` |

Here, Linear's input count does not depend on the spatial dimensions `H,W`. However, averaging the values instead of flattening all of them is <strong>an architectural change that changes the computation</strong>. The input must still be large enough for the preceding convolution and pooling operations to run.

## 14. Label what each number counts

The same number can appear in several places while counting different things. When reading a layer, write down the quantity each number represents to check how the layers connect.

| Expression | What it represents |
| --- | --- |
| The 3 in RGB | Input channels per image |
| A 3×3 kernel | Height and width of the region read from each channel |
| `out_channels=32` | 32 filters, producing 32 output channels |
| Weight shape `(32,1,3,3)` | 32 filters, each containing one per-channel kernel |
| Output shape `(2,32,28,28)` | Two images, each with 32 result arrays of size 28×28 |
| `64×7×7=3136` | Values remaining per image after the final pooling step |
| The 10 in Linear | Class scores produced per image |
| 50,186 parameters | Total weight and bias values changed by training |

To follow the actual values, use this order: <strong>select a region → multiply matching elements → add the results from all input channels → add the bias → store the value at one output position</strong>. Repeating this across positions produces an output channel. Each filter creates its own output channel.

The model then applies ReLU and pooling, flattens each image's values, and uses Linear to calculate scores. Keeping input data, weights, intermediate outputs, and targets separate makes it easier to see why each shape is needed.

## Additional notes: helper code and other settings

<details>
<summary>How do None, try, except, and ____ in exercise code relate to convolution?</summary>

`result = None` can indicate that a result has not been stored yet. `None` is a special Python value, distinct from zero or an empty tensor. It is not a required initialization step for running a CNN.

`try` marks a block of code that may raise an exception. `except NameError as error` catches an error caused by using an undefined name. `____` is a valid Python name, sometimes used in exercise materials to mark a blank that should be filled in. Executing it without first defining it raises `NameError`.

`if error.name != "____": raise` handles the placeholder name's error separately while re-raising errors involving other names. In this position, `raise` passes the caught exception onward. This helper code does not perform convolution or training, and it does not fix shape errors.[^python]

A name such as `count_params(model)` may also refer to a separately defined function. Check its definition instead of assuming it is built into PyTorch. This article counts parameter elements directly with `sum(p.numel() for p in model.parameters())`.

</details>

<details>
<summary>How is convolution different when groups is not 1?</summary>

When `groups` is greater than 1, input and output channels are divided into groups. Each output channel uses only the input channels belonging to its group. The general weight shape is therefore `(C_out, C_in/groups, K_H, K_W)`. Both the input and output channel counts must be divisible by `groups`.[^conv]

This article's statement that “one filter uses every input channel,” and its weight-count formulas, assume `groups=1`. We can still use “filter” to mean the complete set of weights that produces one output channel.

Increasing the filter count adds weights and output channels, but it does not guarantee that every filter learns a different feature. Also, CNNs do not have to end in Flatten followed by Linear. Other output structures are possible, including the average-pooling design discussed above.

</details>

<details>
<summary>Calculate the same output size in Python and Rust</summary>

Counting convolution output positions works the same way in both languages. The functions below <strong>calculate output length; they do not perform convolution or automatic differentiation</strong>. Both use integer dimensions and equal padding on opposite sides. They first check that the kernel fits within the input region, including padding.

Python's `//` performs floor division. In this function, the numerator is nonnegative and the denominator is positive, so this is equivalent to discarding the fractional part.

```python
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
```

Rust's `usize` is an unsigned integer type commonly used for array lengths and indices. Function arguments declare their types explicitly. Applying `/` to integers returns an integer quotient. For the nonnegative values used here, it gives the same result as Python's `//`. The return type is `-> usize`. Omitting a semicolon from the final expression makes that expression the return value.[^rust]

```rust
fn conv_output_length(
    input: usize,     // Original input length.
    kernel: usize,    // Number of kernel weights along this spatial dimension.
    padding: usize,   // Padding length on one side.
    stride: usize,    // Distance between the region's starting positions.
    dilation: usize,  // Spacing between input positions used by neighboring kernel weights.
) -> usize {
    assert!(input > 0 && kernel > 0 && stride > 0 && dilation > 0);
    let effective_kernel = dilation * (kernel - 1) + 1;
    let padded = input + 2 * padding;
    assert!(padded >= effective_kernel); // Check before a subtraction that could become negative.
    (padded - effective_kernel) / stride + 1 // Return the number of valid starting positions.
}

fn main() {
    assert_eq!(conv_output_length(7, 3, 0, 1, 1), 5);
    assert_eq!(conv_output_length(28, 3, 1, 1, 1), 28);
    let after_pool1 = 28_usize / 2; // First 2×2 pooling with stride 2: output length 14.
    let after_pool2 = after_pool1 / 2; // Second pool with the same settings: 7.
    let in_features = 64 * after_pool2 * after_pool2;
    println!("Flatten: {}", in_features); // 3136 values per image.
}
```

Python functions can have default argument values. The Rust function above takes all five arguments explicitly. In either language, a function that calculates output size does not by itself create or train a CNN.

</details>

## References

The API behavior and shape rules in these examples were checked against the official PyTorch documentation on October 27, 2026. The convolution values, intermediate shapes, and parameter counts were also checked by running the article's code on a CPU.

[^conv]: PyTorch [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): channel connections, arguments, weight/input/output shapes, padding, and initialization.
[^sum]: PyTorch [torch.sum](https://docs.pytorch.org/docs/stable/generated/torch.sum.html): summing all elements or a specified dimension. [Tensor.tolist](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html) and [Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html): conversion to Python values.
[^pool]: PyTorch [MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html): per-channel maximum pooling, default stride, and output size.
[^relu]: PyTorch [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html): elementwise transformation and input/output shapes.
[^flatten]: PyTorch [torch.flatten](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html): the range of dimensions to flatten and element order.
[^linear]: PyTorch [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html): weight storage order and the linear calculation.
[^module]: PyTorch [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html): model composition, calls, parameters, and training/evaluation modes.
[^no-grad]: PyTorch [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html): a block with automatic differentiation recording disabled.
[^loss]: PyTorch [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): logits, target class indices, and mean loss.
[^training]: PyTorch [Neural Networks](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html): the sequence from model definition to loss, backpropagation, and updates.
[^adam]: PyTorch [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html): parameters and learning-rate settings.
[^lazy]: PyTorch [LazyLinear](https://docs.pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html): initialization that infers input size from the first input.
[^adaptive]: PyTorch [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html): average pooling to a specified spatial size.
[^python]: Python [the try statement](https://docs.python.org/3/reference/compound_stmts.html#the-try-statement), [NameError](https://docs.python.org/3/library/exceptions.html#NameError), and [None](https://docs.python.org/3/library/constants.html#None).
[^rust]: The Rust Programming Language: [integer types](https://doc.rust-lang.org/book/ch03-02-data-types.html) and [functions and return values](https://doc.rust-lang.org/book/ch03-03-how-functions-work.html).
