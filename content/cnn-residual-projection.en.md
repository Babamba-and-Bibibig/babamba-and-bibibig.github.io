+++
title = "Matching Shapes in CNN Residual Connections: Stride, Padding, 1×1 Conv, and Identity"
description = "How do we add two paths that start from the same input? Read channels and spatial dimensions separately, calculate the relationship between kernel size, padding, and stride, and implement a residual block that chooses between Identity and a 1×1 convolution."
date = 2026-10-29
slug = "cnn-residual-projection"

[extra]
katex = true
toc = true
styles = ["cnn-residual-projection/article.css"]
+++

A residual connection adds the result of a convolutional path to the result of another path that starts from the input. If the convolutions change the number of channels or the height and width, we cannot add the original input directly in the intended way. The two results need matching shapes at the point of addition.

Let us build a block that changes an input of shape `(2, 8, 16, 16)` into an output of shape `(2, 16, 8, 8)`. In this example, <strong>the channel count increases from 8 to 16, while the height and width each decrease from 16 to 8.</strong> Checking these changes separately makes it easier to decide when to use `Identity` and when to use a `1×1 Conv`.

We will start with a diagram of the complete connection. Then we will identify each axis, examine the arguments that determine output size, establish why the two paths have matching shapes, and write the code. This post develops the shape-matching part of the [earlier post on VGG blocks and residual connections](/en/cnn-vgg-residual-pytorch/).

You can download the [complete Python code with comments](/cnn-residual-projection/examples-en.py) and the [Rust code for checking conditions and sizes](/cnn-residual-projection/shape-checks-en.rs). The Python examples run on the CPU in an environment with PyTorch installed and check the model's structure and computed values.

## 1. The same input enters both paths

A <strong>CNN</strong> is a neural network that uses convolutions. A convolution multiplies values in a spatial region of the input by weights and adds the products to compute new values. A <strong>weight</strong> is a number multiplied by an input value. A <strong>layer</strong> is a component that performs one stage of computation. A <strong>block</strong> groups several layers and their connections.

A <strong>residual connection</strong> adds the result of the main path to the result of another path that starts from the input. This structure is used to help train deep neural networks. Here, we will focus on implementing the connection with actual tensors.[^resnet]

These are the names used in the diagram.

| Name | Role in this block |
| --- | --- |
| Main path | Computes new feature values through two convolutional layers. |
| Shortcut | Reaches the addition point without passing through the main path's two convolutions. |
| Projection | In this post, a `1×1` convolution in the shortcut that matches channel and spatial dimensions. |
| Identity | A layer that returns its input unchanged. |
| ReLU | A function that replaces negative values with 0 and keeps values that are 0 or greater. It preserves shape. |

A tensor's <strong>shape</strong> gives the size of each axis. The numbers below are ordered as `(number of samples, channels, height, width)`. `K` means kernel size, `P` means padding on each side, and `S` means stride, the distance between calculation positions. We will examine each argument with concrete calculations later.

<figure class="residual-figure">
<img src="/cnn-residual-projection/residual-paths-en.svg" alt="Input (2,8,16,16) branches into a main path and a shortcut. The main path applies a 3×3 convolution, ReLU, and another 3×3 convolution. The shortcut applies a 1×1 convolution to the original input. Both results have shape (2,16,8,8); they are added before ReLU." loading="lazy">
<figcaption>Both paths start from the same original input x. The shortcut receives that input, not the output of the first convolution.</figcaption>
</figure>

The main path computes `conv1 → ReLU → conv2`. The shortcut separately processes the original input `x`. Once both results are available, values at matching positions are added, and ReLU is applied to the sum.

Branches in the diagram do not mean that the Python code must execute both paths at the same time. The diagram shows <strong>which input is connected to which computation</strong>.

## 2. Read channels and spatial dimensions separately

A <strong>tensor</strong> is PyTorch's data type for storing and computing arrays of numbers. An <strong>axis</strong> identifies one dimension along which positions are distinguished. This post uses the axis order `(N, C, H, W)` for image batches.

A <strong>batch</strong> is a group of samples processed together. <strong>Channels</strong> distinguish different arrays of values within one sample. An RGB image has three channels for its color components. Intermediate data produced by convolutions can have several computed channels. The eight input channels in this post represent such intermediate data.

For an input `x` with shape `(2, 8, 16, 16)`, the numbers mean the following.

| Axis number | Symbol | Meaning | Axis size | Valid indices |
| --- | --- | --- | ---: | --- |
| 0 | `N` | Number of samples processed together | 2 | 0–1 |
| 1 | `C` | Channels in one sample | 8 | 0–7 |
| 2 | `H` | Height of one channel | 16 | 0–15 |
| 3 | `W` | Width of one channel | 16 | 0–15 |

An <strong>index</strong> identifies a position to select and starts from 0. Axis number 1, a channel count of 8, and channel indices 0–7 are different pieces of information. `x[0, 3, 4, 6]` selects one number from the first sample, at channel index 3, height index 4, and width index 6.

A <strong>feature map</strong> arranges computed values by their height and width positions. In this post, spatial size means the feature map's `H×W`. It does not include the channel count `C`.

| Dimension | Input `(2,8,16,16)` | Output `(2,16,8,8)` | Change |
| --- | ---: | ---: | --- |
| Samples `N` | 2 | 2 | Unchanged |
| Channels `C` | 8 | 16 | Increased |
| Height `H` | 16 | 8 | Decreased |
| Width `W` | 16 | 8 | Decreased |

The output's valid indices are `N: 0–1`, `C: 0–15`, `H: 0–7`, and `W: 0–7`. Increasing the channel count does not turn the input into 16 separate images.

## 3. Find two kinds of information in the Conv2d arguments

`nn.Conv2d` is a class for creating a two-dimensional convolutional layer. `nn` refers to `torch.nn`, which contains PyTorch's layer classes. We bring it into the code with `from torch import nn`.

The following call creates a convolutional layer that changes eight input channels into 16 output channels.

`nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)`

| Argument | Meaning | Setting in this example |
| --- | --- | --- |
| `in_channels` | Number of channels the layer receives | 8 |
| `out_channels` | Number of channels the layer produces | 16 |
| `kernel_size`, `K` | Spatial region read at once in each input channel | `3×3` |
| `padding`, `P` | Number of cells added at the top, bottom, left, and right | 1 cell on each side |
| `stride`, `S` | Distance moved to the next calculation position | 2 cells in each direction |

The default padding mode treats the added region as zeros. With `P=1`, the calculation therefore uses two additional cells in total along the height and two along the width.[^conv]

When reading the output shape, separate two questions.

1. <strong>How many output channels are there?</strong> Check `out_channels`.
2. <strong>What are the output height and width?</strong> Check the input dimensions together with `K`, `P`, and `S`.

This post uses `dilation=1` and `groups=1`. `dilation` is the spacing between the input positions read by neighboring kernel elements; a value of 1 reads adjacent positions. `groups=1` is the default arrangement in which each output filter uses every input channel. `K`, `P`, and `S` are integers applied equally to height and width.

With these settings established, we can calculate the channels and spatial dimensions.

## 4. Output height counts the positions where the kernel is applied

Let `H` be the input height, `K` the kernel height, `P` the padding on both the top and bottom, and `S` the stride. The height including padding is `H+2P`.

For the entire kernel to fit, its starting position can move as far as `H+2P−K`. We start at position 0 and move in steps of `S`, so the output height is as follows.[^conv]

<div class="residual-math">
$$
H_{\mathrm{out}}=\left\lfloor\frac{H+2P-K}{S}\right\rfloor+1
$$
</div>

`floor`, written `⌊ ⌋`, rounds down to an integer. For example, `floor(7.5)=7`. The final `+1` includes the calculation at the initial position 0. To calculate the width, replace `H` with `W`.

### Apply a 3×3 kernel, padding 1, and stride 2 to height 16

| Step | Calculation | Result |
| --- | --- | ---: |
| Height including padding | `16+2×1` | 18 |
| Last starting position where the kernel fits | `18−3` | 15 |
| Starting positions actually used with stride 2 | `0, 2, 4, 6, 8, 10, 12, 14` | 8 positions |
| Output height | `floor(15/2)+1` | 8 |

The starting positions in this table are <strong>numbered within the array including padding</strong>. They must be distinguished from indices in the original input. The kernel would fit at starting position 15, but moving by 2 from 0 never reaches 15.

The width is also 16, so the same calculation produces 8. Combining this with `out_channels=16`, the result has shape `(2, 16, 8, 8)`.

## 5. Why use the same stride for 3×3 and 1×1 convolutions?

The main path's first convolution uses `K=3, P=1`, while the shortcut uses `K=1, P=0`. Although their kernel sizes differ, these settings share a relationship.

| Setting | Calculation of `K−2P` | Result |
| --- | --- | ---: |
| `K=3, P=1` | `3−2×1` | 1 |
| `K=1, P=0` | `1−2×0` | 1 |

In the output-size formula, `H+2P−K` becomes `H−1` for both settings. Therefore, when they receive inputs of the same height and use the same stride, their output heights match.

<div class="residual-math">
$$
K-2P=1
\quad\Longrightarrow\quad
H_{\mathrm{out}}
=\left\lfloor\frac{H-1}{S}\right\rfloor+1
=\left\lceil\frac{H}{S}\right\rceil
$$
</div>

`ceil`, written `⌈ ⌉`, rounds up to an integer. For example, `ceil(7.5)=8`. The relationship above holds for positive integer input sizes and strides. With `H=15, S=2`, `floor(14/2)+1=8`, and `ceil(15/2)` is also 8.

<strong>The same stride works here because the two kernel-and-padding combinations produce the same output-size formula.</strong> Matching stride alone is not the reason.

<figure class="residual-figure">
<img src="/cnn-residual-projection/kps-positions-en.svg" alt="For input height 16 and stride 2, a 3×3 kernel with padding 1 and a 1×1 kernel with padding 0 each have eight starting positions. A 3×3 kernel with padding 0 has seven. Positions are numbered within each array including its padding." loading="lazy">
<figcaption>Output height counts the starting positions actually used. Even with the same stride, different kernel-and-padding relationships can produce different counts.</figcaption>
</figure>

### The paths also match for odd input dimensions

| Input spatial size | Stride | Output of `3×3, P=1` | Output of `1×1, P=0` |
| --- | ---: | --- | --- |
| `16×16` | 1 | `16×16` | `16×16` |
| `16×16` | 2 | `8×8` | `8×8` |
| `15×13` | 1 | `15×13` | `15×13` |
| `15×13` | 2 | `8×7` | `8×7` |

With `stride=2`, height 15 becomes 8 rather than 7. Under these settings, each spatial dimension is divided by 2 and rounded up.

For an odd kernel size, using `P=(K−1)/2` gives the same relationship. The `K/P` combinations `1/0`, `3/1`, `5/2`, and `7/3` all satisfy it. `S=1` preserves spatial dimensions; `S=2` divides each dimension by 2 and rounds up.

More generally, under this post's assumptions, two convolutions applied to the same input produce matching spatial dimensions when their `K−2P` values match and their strides match. This is a sufficient condition for matching sizes. Observing equal output sizes for one particular input does not prove that two settings will match for every input.

## 6. Why copying padding=1 into the shortcut fails

If we copy the padding of 1 from the `3×3 Conv` into the `1×1 Conv`, the shapes in this block no longer match.

| Input height | `K` | `P` | `S` | Output height |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 1 | 2 | `floor((16+2−3)/2)+1 = 8` |
| 16 | 1 | 0 | 2 | `floor((16−1)/2)+1 = 8` |
| 16 | 1 | 1 | 2 | `floor((16+2−1)/2)+1 = 9` |
| 16 | 3 | 0 | 2 | `floor((16−3)/2)+1 = 7` |

Setting padding to 1 for a `1×1 Conv` is allowed. In this case, however, the shortcut produces `9×9`, which does not match the main path's `8×8`. The setting needed here is <strong>`K=1, P=0, S=2`</strong>.

The last row gives another example of why matching stride is insufficient. `3×3, P=0, S=2` produces `7×7`, while `1×1, P=0, S=2` produces `8×8`.

## 7. conv2 preserves the first convolution's dimensions

The shortcut must match the main path's <strong>final result</strong>. Checking only `conv1` is therefore not enough. We must also check whether subsequent layers change the dimensions again.

| Path | Layer | Channels | `K, P, S` | Output shape |
| --- | --- | --- | --- | --- |
| Main | `conv1` | `8→16` | `3, 1, 2` | `(2,16,8,8)` |
| Main | ReLU | `16→16` | Not applicable | `(2,16,8,8)` |
| Main | `conv2` | `16→16` | `3, 1, 1` | `(2,16,8,8)` |
| Shortcut | `1×1 Conv` | `8→16` | `1, 0, 2` | `(2,16,8,8)` |

`conv2` uses `K=3, P=1, S=1`, so it preserves the height and width it receives. ReLU also preserves shape. Consequently, a shortcut that matches `conv1` also matches the final result of the main path.

If `conv2` also used `stride=2`, the main path would change spatial size as `16×16 → 8×8 → 4×4`. The existing shortcut would still produce `8×8`, preventing the two results from being added.

Read `stride=stride` in the code as two separate parts. The left side is the name of the argument accepted by `Conv2d`; the right side is the variable received by the block's `__init__`. Passing `stride=2` to a block does not automatically apply that value to all its convolutions. Only layers explicitly given that value use it.

## 8. A 1×1 convolution uses all input channels at one position

`1×1` describes the <strong>spatial region</strong>. It does not mean one input channel or one weight in a filter.

A <strong>bias</strong> is a number added after summing the products of input values and weights. The weights and biases adjusted during training are called <strong>parameters</strong>. We will use <strong>filter</strong> to mean the complete set of weights that produces one output channel. Some sources use kernel and filter interchangeably; here, we distinguish the spatial region from the full calculation across channels.

With eight input channels, a filter in a standard `3×3` convolution has `8×3×3=72` weights. A filter in a `1×1` convolution on those same eight channels has `8×1×1=8` weights. The spatial region becomes smaller, but each filter still combines all eight input channels.

<figure class="residual-figure">
<img src="/cnn-residual-projection/pointwise-channels-en.svg" alt="Values from all eight input channels at the same spatial position connect to every output filter. Each filter multiplies those values by its eight weights, sums the products, and adds a bias to produce one value in its output channel. Sixteen filters produce sixteen output channels." loading="lazy">
<figcaption>Each filter processes the same eight input values with its own eight weights. The output channel count of 16 is the number of filters.</figcaption>
</figure>

### Calculate one output cell with actual numbers

Let us choose the input values at one position and the first filter's weights as follows. These numbers are assigned to check the calculation; they are not presented as the values of a trained filter.

| Input channel index | Input value | Weight in filter 0 | Product |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0.5 | 0.5 |
| 1 | 2 | −1 | −2 |
| 2 | 0 | 2 | 0 |
| 3 | −1 | 1 | −1 |
| 4 | 3 | 0 | 0 |
| 5 | 1 | 0.25 | 0.25 |
| 6 | 2 | −0.5 | −1 |
| 7 | −2 | 1 | −2 |

The eight products sum to `−5.25`. If this filter's bias is `0.5`, the value at this position in output channel 0 is `−4.75`. The other 15 filters also use the same eight input values, but they compute their outputs with their own weights and biases.

### The weight tensor has no batch axis

The complete weight tensor for `Conv2d(8, 16, 1)` has shape `(16, 8, 1, 1)`.[^conv]

| Weight axis number | Meaning | Size | Valid indices |
| --- | --- | ---: | --- |
| 0 | Number of output filters | 16 | 0–15 |
| 1 | Input channels received by one filter | 8 | 0–7 |
| 2 | Height of the kernel for one channel | 1 | 0 |
| 3 | Width of the kernel for one channel | 1 | 0 |

The first 16 in this tensor counts filters, not samples. There are `16×8×1×1=128` weights. With the default `bias=True`, there are also 16 biases. Together, these make 144 numbers that training can adjust.

The same filter weights are reused across spatial positions and across samples in a batch. Changing the batch size from 1 to 2 does not increase the layer's weight count.

### Which positions does stride=2 read?

With `1×1, P=0, S=2`, the convolution selects input indices `0, 2, 4, …` along both height and width. For a `16×16` input, it uses eight even indices from 0 through 14 in each direction. At every selected position, it still processes all input channels.

For example, calculating the output at height index 2 and width index 3 reads the original input at height index 4 and width index 6. This shortcut therefore <strong>computes new channel values while also changing spatial dimensions.</strong> It does not simply copy the input or reshape the tensor.

## 9. Where does the condition for choosing Identity come from?

The main path we have examined has fixed settings.

- `conv1`: `K=3, P=1`, using the stride received by the block.
- `conv2`: `K=3, P=1, S=1`, preserving `conv1`'s spatial dimensions.
- Final channel count: `out_channels`.

This block selects Identity when both of the following conditions hold. These settings keep the two paths' shapes equal across different input heights and widths.

| What to check | Condition for choosing Identity |
| --- | --- |
| Preserve channels | `in_channels == out_channels` |
| Preserve spatial dimensions | `stride == 1` |

Python's `and` is true when both conditions are true. The Identity condition is therefore `in_channels == out_channels and stride == 1`. Conversely, this block uses a projection when `in_channels != out_channels or stride != 1`. `or` is true when at least one of the conditions is true.

| Input and output channels | Stride | Spatial change | Choice |
| --- | ---: | --- | --- |
| `8→8` | 1 | Preserved | Identity |
| `8→16` | 1 | Preserved | `1×1 Conv` to change channels |
| `8→8` | 2 | Configured to reduce dimensions | `1×1 Conv` to change spatial dimensions |
| `8→16` | 2 | Configured to reduce dimensions | `1×1 Conv` to change channels and spatial dimensions |

<strong>Even with stride 1, a channel-count change calls for a projection in this block.</strong> This applies to both increasing channels, such as `8→16`, and decreasing channels, such as `16→8`.

This condition assumes the main path defined above. If you change the kernel or padding, or add another layer that changes dimensions, calculate the output size of the entire main path again.

### Why the choice does not require a specific input height and width

Even without specific values of `H` and `W`, we know this block's rules for changing dimensions. `conv1` produces height `ceil(H/S)`, and `conv2` preserves it. The shortcut's `1×1, P=0` convolution produces the same `ceil(H/S)` with the same stride. The width follows the same relationship.

Training can change the weight values, but the size rule remains the same as long as the kernel, padding, and stride settings stay the same. This lets us choose the layers in `__init__`, when the model is created. When we run the block, we check the actual shapes of both paths to verify the design.

## 10. Creating Identity and passing an input are different actions

A <strong>class</strong> defines an object's structure and behavior. An <strong>object</strong> is an actual instance created from that definition and used in the program. `nn.Identity` is a class; the result of `nn.Identity()` is a layer object.

`self` refers to the current block object. `self.shortcut` is an attribute that stores the layer used inside that block.

| Code | Action | Result |
| --- | --- | --- |
| `nn.Identity()` | Creates an Identity layer. | A layer object |
| `self.shortcut = nn.Identity()` | Stores the layer in a block attribute. | No input has been processed yet |
| `skip = self.shortcut(x)` | Passes input `x` to the stored layer. | The same tensor object as the input |
| `nn.Identity()(x)` | Creates a layer and immediately passes it an input. | The same tensor object as the input |
| `nn.Identity(x)` | Passes `x` to the constructor. | A layer object, not a tensor |

Identity's constructor ignores the arguments it receives, so `nn.Identity(x)` does not necessarily raise an error. It still has not processed the tensor. <strong>Creating a layer and calling that layer with an input are separate actions.</strong>[^identity]

Identity returns the same input object without copying it. It has no weights or biases. Returning a result means giving it back to the calling code, not printing it on the screen.

### The shortcut can also use x directly

When shapes match, `skip = x` can provide the shortcut result. Using Identity lets us choose and store either `Identity` or `Conv2d` during construction, then call either one as `self.shortcut(x)` during computation.

The part that passes the input through unchanged is the <strong>shortcut path</strong>. The complete block still computes the main path, adds both results, and applies ReLU. Replacing the block's `forward` with `return x` would skip all of that computation.

Using Identity in the shortcut also does not mean the block's final output always equals the original input. The main path adds its values, and the final ReLU replaces negative sums with zero.

## 11. Add matching positions, then apply ReLU

When two tensors have the same shape `(N,C,H,W)`, their values are added at matching positions.

`merged[n, c, h, w] = main[n, c, h, w] + skip[n, c, h, w]`

Here, `n`, `c`, `h`, and `w` are the sample, channel, height, and width indices. Two values at one position produce one result, so the output shape remains `(N,C,H,W)`.

<figure class="residual-figure">
<img src="/cnn-residual-projection/elementwise-add-en.svg" alt="Select a 2×3 spatial region from one channel of one sample. Adding main [[1,-2,3],[4,0,-1]] and shortcut [[2,1,-4],[-1,5,2]] at matching positions gives [[3,-1,-1],[3,5,1]]. ReLU then produces [[3,0,0],[3,5,1]]." loading="lazy">
<figcaption>Neither addition nor ReLU changes the number of spatial cells shown here. Add the two paths first, then replace negative sums with zero.</figcaption>
</figure>

This operation is different from joining channels together. Adding the example's two 16-channel tensors still gives 16 channels. It also differs from applying ReLU separately to both path results before adding them. In the last cell of the diagram's first row, `3+(−4)=−1`, so the final value is 0. Applying ReLU to each value first would give `3+0=3`, a different result.[^relu]

PyTorch's general addition rules support <strong>broadcasting</strong>, which allows some dimensions of size 1 to be used in calculations with larger dimensions. This block does not rely on that behavior. We check that both paths have exactly the same shape to verify that the residual connection has the intended structure.[^broadcast]

## 12. Implement the block and check five inputs

`nn.Module` is PyTorch's base class for models and layers. <strong>Inheritance</strong> defines a new class that receives functionality from an existing class. Inheriting from `nn.Module` and calling `super().__init__()` initializes the machinery that manages child layers and parameters. A layer stored in an attribute such as `self.conv1` is registered as a child module of the block.[^module]

A function defined inside a class is called a <strong>method</strong>. The two methods we will define have the following roles.

| Method | When it runs | Role |
| --- | --- | --- |
| `__init__` | When a block object is created | Creates and stores the layers to use. |
| `forward` | When an input is passed through `block(x)` | Computes the current input with the prepared layers. |

The block below is a teaching implementation for examining the relationship between two convolutions and a shortcut. It includes biases and does not add normalization layers. TorchVision's actual ResNet BasicBlock uses BatchNorm layers, which adjust the distribution of intermediate values, and convolutions with `bias=False`. The shared principle here is matching dimensions with `3×3/P=1` and `1×1/P=0` convolutions.[^resnet-code]

These names are used in the runnable example.

| Name | Role |
| --- | --- |
| `torch` | Package providing tensor creation and operations |
| `nn` | `torch.nn`, containing layer classes |
| `F` | Alias for `torch.nn.functional`, containing neural-network operations as functions |
| `torch.randn(...)` | Creates a tensor of random floating-point values with the specified shape |
| `torch.no_grad()` | Disables recording the enclosed computations for backpropagation |
| `assert` | Raises an error if its condition is false |
| `isinstance(...)` | Checks whether an object belongs to a specified class |
| `is` | Checks whether two names refer to the same object, rather than just equal values |

`no_grad` still computes output values. We use it because we are checking the structure and do not need a record for backpropagation. This setting does not train the model or switch it to evaluation mode.[^no-grad]

The following is the complete Python code, ready to run in one block. At the end, it also checks the `1×1` output value of `−4.75` calculated earlier and the output sizes produced by mismatched padding settings.

```python
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
```

The five inputs produce the following outputs.

```text
(2, 8, 16, 16) -> (2, 16, 8, 8) | 1x1 Conv
(1, 8, 15, 13) -> (1, 8, 15, 13) | Identity
(1, 8, 15, 13) -> (1, 16, 15, 13) | 1x1 Conv
(1, 8, 15, 13) -> (1, 8, 8, 7) | 1x1 Conv
(1, 16, 15, 13) -> (1, 8, 15, 13) | 1x1 Conv
```

In this example, `projection_block=None` indicates that no block object has been stored yet, and `projection_ready=False` marks the state as not ready. Changing the flag to `projection_ready=True` does not itself check the structure or train the model. The code above changes the flag after the actual output check passes.

## 13. What to check when the shapes do not match

First, inspect <strong>both shapes immediately before addition</strong>. For example, if the main path produces `(2,16,8,8)` and the shortcut produces `(2,8,16,16)`, the sample count `N` matches, but `C`, `H`, and `W` differ.

Then check the paths in this order.

1. Confirm that both paths start from the same original input `x`.
2. Compare the main path's final channel count with the shortcut's output channel count.
3. Calculate `K`, `P`, `S`, and the output height and width for every convolution in each path.
4. Check whether an intermediate layer changes the spatial dimensions again.
5. Once both results have the same shape, check the addition and ReLU.

| Incorrect setting or code | Cause | Correction for this block |
| --- | --- | --- |
| Using Identity when channels or spatial dimensions change | The shortcut returns the original `(N,C,H,W)` unchanged. | Choose a `1×1` projection. |
| `padding=1` in the `1×1 Conv` | The example's shortcut produces `9×9`. | Use `padding=0`. |
| `stride=2` in `conv2` as well | The main path reduces spatial dimensions twice. | Use `stride=1` in `conv2`. |
| `self.shortcut(main)` | The shortcut receives an already computed result instead of the original input. | Use `self.shortcut(x)`. |
| `nn.Identity(x)` | Creates a layer object instead of processing the input. | Pass `x` to the created object. |
| `out_channesl`, `shorcut`, `Identify` | The spelling differs from the declared argument, attribute, or class. | Check `out_channels`, `shortcut`, and `Identity`. |

Comparing actual shapes in `forward` is a normal way to check them. However, <strong>creating a new Conv every time `forward` is called</strong> also initializes new weights every time. An optimizer, the object that updates parameters, does not automatically include new parameters after it has already been created. This example creates trainable layers once in `__init__` and reuses them in `forward`.[^module]

## 14. Express the same conditions and size calculations in Rust

Python and Rust use the same logic to compare channel counts and calculate output sizes. Their condition syntax and integer type notation differ.

| Meaning | Python | Rust |
| --- | --- | --- |
| Both conditions are true | `and` | `&&` |
| At least one condition is true | `or` | `\|\|` |
| Compare two values for equality | `==` | `==` |
| Divide positive integers and keep the quotient | `//` | `/` between integers |

Below, `usize` is an unsigned integer type used for counts and indices, while `bool` represents true or false. The type after `->` is the function's return type. A final expression without a semicolon supplies the return value. `assert!` checks a condition, and `assert_eq!` checks that two values are equal.[^rust]

This Rust code checks only layer-selection conditions and spatial size calculations. It does not implement a convolution or a PyTorch model in Rust.

```rust
// Main path in this post: conv1 uses K=3/P=1; conv2 uses K=3/P=1/S=1.
fn use_identity(in_channels: usize, out_channels: usize, stride: usize) -> bool {
    in_channels == out_channels && stride == 1
}

// Output size along one axis with dilation=1 and equal integer padding on each side.
fn output_size(input: usize, kernel: usize, padding: usize, stride: usize) -> usize {
    assert!(input > 0 && kernel > 0 && stride > 0);
    let padded = input + 2 * padding;

    // Check that the kernel fits before subtracting unsigned integers.
    assert!(padded >= kernel);

    // The numerator is nonnegative, so integer division gives floor.
    (padded - kernel) / stride + 1
}

fn main() {
    assert!(use_identity(8, 8, 1));
    assert!(!use_identity(8, 16, 1));  // ! reverses a Boolean value.
    assert!(!use_identity(8, 8, 2));

    // Both kernel/padding combinations match for even and odd input sizes.
    for input in [16_usize, 15, 13] {
        let main_size = output_size(input, 3, 1, 2);
        let skip_size = output_size(input, 1, 0, 2);
        assert_eq!(main_size, skip_size);
        println!("Input {input}: main path {main_size}, shortcut {skip_size}");
    }

    assert_eq!(output_size(16, 1, 1, 2), 9);  // Change padding to 1.
    assert_eq!(output_size(16, 3, 0, 2), 7);  // Remove padding from 3×3.
    println!("Identity conditions and output sizes verified");
}
```

When modifying this block, first choose the output channel count and check the spatial changes across the entire main path. Use Identity for settings that preserve the original shape. When settings change the shape, design the shortcut to match the main path's final result, then verify both shapes with actual inputs.

## References

The APIs and implementations were checked against official documentation on October 29, 2026. The numbers substituted into the formulas and the table results were calculated from the settings used in these examples.

[^resnet]: Kaiming He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/html/1512.03385v1), sections 3.1–3.3. Residual learning and identity and projection shortcuts.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html). Input, output, and weight shapes; kernel, padding, stride, dilation, groups, and bias; output-size formulas.
[^identity]: PyTorch, [Identity](https://docs.pytorch.org/docs/stable/generated/torch.nn.Identity.html) and [Identity source](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py). Constructor arguments and the `forward` implementation that returns the input unchanged.
[^relu]: PyTorch, [functional.relu](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html). ReLU applied to each element.
[^broadcast]: PyTorch, [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html). Axis-by-axis rules for operations on tensors with different shapes.
[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) and [Optimizer](https://docs.pytorch.org/docs/stable/optim.html). Child modules and parameter registration, module calls, and parameters supplied to an optimizer.
[^resnet-code]: TorchVision, [ResNet implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html). Shape changes and layer configuration in `conv3x3`, `conv1x1`, `BasicBlock`, and `_make_layer`.
[^no-grad]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html). Disabling gradient recording while still computing outputs.
[^rust]: The Rust Programming Language, [Control Flow](https://doc.rust-lang.org/book/ch03-05-control-flow.html), [Data Types](https://doc.rust-lang.org/book/ch03-02-data-types.html), and [Functions](https://doc.rust-lang.org/book/ch03-03-how-functions-work.html). Conditions, integer types, and function return values.
