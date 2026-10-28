+++
title = "Building a CNN in PyTorch: VGG Blocks, Flatten and Linear, and Residual Connections"
description = "Distinguish inputs, weights, and outputs, then build VGG blocks, a classifier, and residual connections. Check channels, shapes, and parameter counts directly, and understand how eval, no_grad, and actual training differ."
date = 2026-10-28
slug = "cnn-vgg-residual-pytorch"

[extra]
katex = true
toc = true
styles = ["cnn-vgg-residual-pytorch/article.css"]
+++

Even after understanding how one convolutional layer computes its output, connecting several layers raises new questions. How does the output of the first convolution enter the second? Why do two stacked `3×3` convolutions connect to a `5×5` region of the original input? Which numbers should we count when passing the final feature maps to `Linear`?

This time, we will implement these connections ourselves. We will first distinguish input data, weights, and output data. Then we will combine two convolutions and pooling into a VGG-style block and attach a classifier to two such blocks. Finally, we will look at residual connections, which add the original input to a computed result, and how to match the sizes of the two paths being added.

The calculations covered in [CNN Channels and Kernels](/en/cnn-convolution-basics/) and [Calculating Convolutions Directly](/en/cnn-convolution-practice/) are our starting point. The terms and shapes needed here are explained again so that this post can also be read on its own. The examples focus on <strong>which numbers we compute, and how many remain along each axis</strong>.

The Python code blocks build on one another from top to bottom. The [complete Python code with comments](/cnn-vgg-residual-pytorch/examples-en.py) and [Rust code for checking counts](/cnn-vgg-residual-pytorch/shape-checks-en.rs) are also available. The Python examples use small random inputs to check structure and values. They do not measure real image classification performance or train a model. A training function is defined later, but it is not called.

## 1. Separate building a model from computing an input

A <strong>CNN, or convolutional neural network</strong>, computes features by reusing the same weights across small spatial regions of its input. A <strong>layer</strong> is a component responsible for one stage of computation within the network.

There are three things we will keep distinguishing throughout this post.

| Item | What it contains | Examples |
| --- | --- | --- |
| Input data | Numbers to process in the current computation | An original image, feature maps from the previous layer |
| Parameters | Numbers used to compute the input and adjusted through training | Weights and biases of convolutional and Linear layers |
| Output data | Results computed from the input and current parameters | New feature maps, scores for each class |

A <strong>weight</strong> is a number multiplied by an input value. A <strong>bias</strong> is a number added after summing those products. These numbers, which the model adjusts during training, are called <strong>parameters</strong>.

Creating a convolutional layer prepares the weights and biases it needs. If no input has been passed to it, however, no output feature maps for that input exist yet. Computing output values also does not automatically train the weights.

| Stage | What happens | What it produces |
| --- | --- | --- |
| Define a class | Describe the model's components and computation. | A definition used to build the model |
| Create a model object | Create and register actual layers according to that definition. | Layer objects and initial parameters |
| Prepare an input | Load image data or create an example tensor. | Input data |
| Call `model(x)` | Compute the current input in the defined order. | Output feature maps or scores |
| Check the structure | Inspect intermediate sizes and calculated values. | Check results |
| Train the model | Calculate a loss, compute gradients, and update parameters. | Updated parameters |

An <strong>object</strong> here means something actually created and used in a program. A tensor is an object, and so is a convolutional layer. We will revisit the difference between a class and an object before defining our own model.

## 2. A batch, a channel, and one cell are different units

A <strong>tensor</strong> is PyTorch's data type for storing and computing arrays of numbers. An <strong>axis</strong> identifies a dimension along which positions are distinguished, and a tensor's <strong>shape</strong> gives the size of each axis.

A group of images processed together is a <strong>batch</strong>. In this post, we read the axes of an image batch in the order `[N, C, H, W]`.

| Axis number | Meaning | Size in `[1, 4, 7, 7]` | Valid indices |
| --- | --- | ---: | --- |
| 0 | Number of samples, `N` | 1 | 0 |
| 1 | Channels in one sample, `C` | 4 | 0–3 |
| 2 | Height of one channel, `H` | 7 | 0–6 |
| 3 | Width of one channel, `W` | 7 | 0–6 |

An <strong>index</strong> is the number of the position we select, starting from 0. Axis number 1, its size of 4, and an index of 2 to select are different pieces of information.

In an RGB image, the three channels distinguish red, green, and blue components. After a convolution, the channels are computed <strong>feature maps</strong>. In this context, a feature map means the two-dimensional output array for one channel. Having 16 intermediate channels does not mean the original image has turned into 16 images.

When `x` has shape `[1, 4, 7, 7]`, the following expressions select objects of different sizes.

| Expression | What it selects | Resulting shape |
| --- | --- | --- |
| `x[0]` | The entire first sample | `[4, 7, 7]` |
| `x[0, 0]` | The entire first channel of the first sample | `[7, 7]` |
| `x[0, 0, 3, 3]` | One number at row 3, column 3 of that channel | `[]`, a zero-dimensional tensor |

The final indices of `3` mean the fourth row and fourth column. Computing one output cell means finding the single number to place at such a position. It does not mean reducing the entire feature map to one number.

We will use a `4→6` convolution to explain channels. Giving the input channels, output channels, and kernel size different numbers makes their roles easier to see. The classification model we build later will separately use RGB input with `3` channels.

## 3. One filter contains a kernel for each input channel

First, read the three arguments of the following layer.

```python
# This line describes creating a layer. It does not compute an input image yet.
# The setup code that imports nn appears in the next section.
# nn.Conv2d(4, 6, 3)
```

The first `4` is the number of input channels, the second `6` is the number of output channels, and the final `3` is the kernel's spatial size, `3×3`. The last argument does not mean there are three kernels.

Some sources use <strong>filter</strong> and <strong>kernel</strong> interchangeably. To distinguish the connections in this post, we will use the following definitions.

| Name | Meaning in this post | For `Conv2d(4, 6, 3)` |
| --- | --- | --- |
| Per-channel kernel | A two-dimensional weight array corresponding to one input channel | `[3, 3]` |
| Filter | The complete group of weights that produces one output channel | `[4, 3, 3]` |
| Weights of the entire layer | A tensor containing the weights of all six filters | `[6, 4, 3, 3]` |

The explanation below assumes <strong>an ordinary convolution with `groups=1`</strong>. With this setting, every filter uses all input channels. In a grouped convolution, which divides channel connections into separate groups, the set of channels received by each filter changes.[^conv]

### Channels and kernels have a one-to-one correspondence

Filter A contains per-channel kernels A0, A1, A2, and A3. A0 computes input channel 0, and A1 computes input channel 1. The remaining kernels follow the same pattern.

| Input data | Corresponding weights | Computation at one output position |
| --- | --- | --- |
| A `3×3` region in input channel 0 | Kernel A0 | Sum the nine products of corresponding positions |
| A `3×3` region in input channel 1 | Kernel A1 | Sum the nine products of corresponding positions |
| A `3×3` region in input channel 2 | Kernel A2 | Sum the nine products of corresponding positions |
| A `3×3` region in input channel 3 | Kernel A3 | Sum the nine products of corresponding positions |

Next, we add the four per-channel results. If the layer uses a bias, we also add the one bias for filter A. The result becomes one cell in output channel A.

Having four input channels and four kernels does not mean computing `4×4=16` pairs. Each channel has its corresponding kernel, so there are <strong>four pairs</strong>. Filter B uses the same four input channels, but B0 through B3 are weights trained separately from A's weights.

<figure class="cnn-vgg-figure">
<img src="/cnn-vgg-residual-pytorch/figure-channel-filter-en.svg" alt="Four input channels each correspond to one of the four per-channel 3×3 kernels in a filter. Summing the per-channel products produces one output cell, and six filters produce six output channels." loading="lazy">
<figcaption>Input channels are data; kernels are weights. Equal counts do not mean they count the same kind of object.</figcaption>
</figure>

The figure includes a bias. The manual check in the next section leaves out the bias so that we can examine the products and sums separately.

### What do 6, 24, and 216 each count?

| Item being counted | Calculation | Count |
| --- | --- | ---: |
| Filters that produce output channels | `out_channels` | 6 |
| Per-channel kernels in one filter | `in_channels` | 4 |
| Weights in one per-channel kernel | `3×3` | 9 |
| Weights in one filter | `4×3×3` | 36 |
| Per-channel kernels in the entire layer | `6×4` | 24 |
| Weights in the entire layer | `6×4×3×3` | 216 |
| Biases when bias is enabled | One per output filter | 6 |

For a `3×3` convolution with 16 input channels, one filter has `16×9=144` weights. Here, 144 is not the number of output feature maps. The number of output feature maps is still determined by `out_channels`, the number of filters.

The same filters are also reused across spatial positions and images. Doubling the batch does not create twice as many filters. The same model parameters are reused until training updates their values.

## 4. Calculate one output cell and compare it with Conv2d

First, set up the PyTorch names and the device for execution. `torch` is the package that provides tensors and operations. `nn` is a short name for `torch.nn`, which provides layers and model components. `F` is an alias for `torch.nn.functional`, which provides neural network operations as functions.

```python
import torch
from torch import nn
import torch.nn.functional as F

# Keep the model and input on the same device. Use the CPU for this small example.
device = torch.device("cpu")

# Make random inputs and initial weights easier to reproduce with the same
# execution order. This does not guarantee complete reproducibility across
# all versions, devices, and operations.
torch.manual_seed(16)

conv = nn.Conv2d(
    in_channels=4,
    out_channels=6,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,  # Check only the products and sums of weights and input values.
).to(device)

# randn creates a tensor of random real numbers. It does not load a photograph.
x = torch.randn(1, 4, 7, 7, device=device)

print(tuple(conv.weight.shape))        # (6, 4, 3, 3): all filters
print(tuple(conv.weight[0].shape))     # (4, 3, 3): filter A
print(tuple(conv.weight[0, 0].shape))  # (3, 3): A's kernel for channel 0
print(conv.weight.numel())             # 216: number of stored weight values
```

`numel()` returns the number of elements in a tensor. The axes of the weight tensor have different meanings from those of the input tensor. The first axis of `conv.weight` has size 6 because there are <strong>six output filters</strong>, not six images.

With the current settings, both the input and output have height and width 7. Computing `y[0, 0, 3, 3]` requires rows 2–4 and columns 2–4 from each input channel. The slice `2:5` selects 2, 3, and 4; it does not include the ending index 5.

`torch.no_grad()` prevents the computations below from being recorded for backpropagation. The output values are still computed normally. We use it here because we are comparing numbers without training. Its difference from training and evaluation modes is covered later.

```python
with torch.no_grad():
    y = conv(x)
    actual_value = y[0, 0, 3, 3]  # One number in the first output channel

    # Select the same 3×3 spatial region from every input channel.
    patch = x[0, :, 2:5, 2:5]    # [4, 3, 3], still input data
    kernels = conv.weight[0]    # [4, 3, 3], filter A's weights

    # * multiplies corresponding elements. It is not matrix multiplication.
    products = patch * kernels  # [4, 3, 3], 36 products

    # Axes 1 and 2 of products are height and width. Sum nine values per channel.
    channel_results = products.sum(dim=(1, 2))  # [4]
    manual_value = channel_results.sum()       # Sum the four channel results

    # Allow small floating-point differences when checking that the results agree.
    assert torch.allclose(
        manual_value, actual_value, atol=1e-6, rtol=1e-4
    )

print(tuple(y.shape))             # (1, 6, 7, 7)
print(tuple(y[0, 0].shape))       # (7, 7)
print(channel_results)           # Four per-channel sums
print(manual_value.item())       # Output value calculated manually
print(actual_value.item())       # Output value calculated by Conv2d
```

`assert` is a check that raises an error when its condition is false. `allclose` checks whether corresponding numbers are close within a tolerance. `atol` is an absolute tolerance, while `rtol` is a relative tolerance proportional to the magnitude of the comparison value. `item()` extracts an ordinary Python number from a tensor containing one element.[^check]

The shapes in this calculation change as follows.

| Stage | Shape | What it stores |
| --- | --- | --- |
| Input patch | `[4, 3, 3]` | 36 input values read from four channels |
| Elementwise multiplication | `[4, 3, 3]` | 36 products of input values and weights |
| Sum over spatial axes | `[4]` | Four per-channel results |
| Sum over the channel axis | `[]` | One value for one output cell |

Filter A has 36 weights, but output feature map A contains `7×7=49` numbers. Reapplying the same 36 weights at different positions creates those 49 cells. The complete output from all six filters contains `1×6×7×7=294` numbers.

Writing out the per-channel results separately makes the computation easier to understand. PyTorch's actual implementation does not have to store every intermediate array separately or execute operations in the same order as this code. It can process multiple positions and channels in parallel.

PyTorch's `Conv2d` uses <strong>cross-correlation</strong>, which does not flip the kernel as mathematical convolution does. In deep learning, this calculation is generally called convolution.[^conv]

## 5. Check channel counts and spatial sizes separately

When connecting convolutions, read <strong>channels</strong> separately from <strong>height and width</strong>.

| Argument | What it determines |
| --- | --- |
| `in_channels` | Number of input channels this layer expects |
| `out_channels` | Number of output channels, also the number of filters |
| `kernel_size` | Height and width of the kernel |
| `stride` | Distance between successive computation positions |
| `padding` | Width of the region treated as added around the input edges |
| `dilation` | Spacing between input positions referenced by neighboring kernel elements |

With the default padding mode, the added region is treated as zero. Every example in this post uses `dilation=1`, so the kernel reads adjacent input positions.

### The output-size formula counts valid starting positions

Let `H` be the input height, `K` the kernel height, `P` the padding width on each of the top and bottom edges, `S` the stride, and `D` the dilation. The height of the input span occupied by the kernel is `D(K−1)+1`.

Subtracting that span from the padded height `H+2P` gives the distance over which the starting position can move. Divide that distance into steps of `S` and add the first starting position to obtain the output height.[^conv]

<div class="cnn-vgg-math">
$$
H_{\mathrm{out}}
=\left\lfloor\frac{H+2P-D(K-1)-1}{S}\right\rfloor+1
$$
</div>

`floor` rounds down to the greatest integer less than or equal to its argument. Width is calculated in the same way. For the valid sizes considered here, this means discarding any remaining space that cannot hold one more kernel application.

Substituting `K=3, P=1, S=1, D=1` gives `H_out=H` and `W_out=W`. This is the condition for <strong>preserving height and width</strong>. To preserve channels as well, `in_channels` and `out_channels` must also be equal.

| Layer | Input | Output |
| --- | --- | --- |
| `Conv2d(16, 16, 3, padding=1)` | `[1,16,14,14]` | `[1,16,14,14]` |
| `Conv2d(16, 32, 3, padding=1)` | `[1,16,14,14]` | `[1,32,14,14]` |
| `Conv2d(16, 32, 3, stride=2, padding=1)` | `[1,16,14,14]` | `[1,32,7,7]` |

The final row gives `floor((14+2−2−1)/2)+1 = floor(13/2)+1 = 7`. Looking only at a `3×3` kernel and padding 1 to decide that the entire shape stays unchanged would miss changes in stride and output channels.

### Connecting two Conv layers and adding two outputs have different conditions

To connect two Conv layers in sequence, the first layer's `out_channels` must match the second layer's `in_channels`. Spatial sizes may change as long as the next kernel can still be applied.

For example, two `3×3` convolutions without padding can connect as `[1,3,28,28] → [1,8,26,26] → [1,8,24,24]`. Shrinking the intermediate height and width is not a connection error.

The residual connection introduced later, however, adds values at corresponding positions in two tensors. Their entire shapes must therefore match at the addition point. <strong>Passing a result to the next layer and adding two paths require different checks.</strong>

This also explains why height and width need not be specified when creating a Conv object. The spatial size comes from the actual input passed to `conv(x)`. `Conv2d(16,16,3,padding=1)` can process both `14×14` and `20×30` inputs, but the number of input channels must be 16.

## 6. Distinguish classes, objects, and function calls

Before writing the VGG block, let us clarify what the parentheses in the Python code do.

A <strong>class</strong> defines the attributes and behavior of an object. A <strong>function</strong> accepts arguments and performs a task. A function called through an object, as in `x.flatten(...)`, is called a <strong>method</strong>.

`nn.Module` is the base class used to build layers and models in PyTorch. Both `Conv2d`, which has weights, and `ReLU` and `MaxPool2d`, which do not, are Modules.[^module]

| Expression | Role in model code |
| --- | --- |
| `class MyVGGBlock(nn.Module):` | Define the structure of a new class based on Module. |
| `__init__` | Initialize attributes and child layers during object creation. |
| `super().__init__()` | Initialize the parent Module's management functionality. |
| `self` | Refer to the current object. |
| `self.conv1 = nn.Conv2d(...)` | Create a layer object and register it as a submodule of the current model. |
| `forward(self, x)` | Define the computation order for an actual input. |
| `model = MyVGGBlock(...)` | Use the definition to create an actual model object. |
| `model(x)` | Execute `forward` through Module's call procedure. |

Registered child layers are managed together when gathering parameters with `model.parameters()` or changing devices with `model.to(device)`. Calling `model.forward(x)` directly bypasses part of Module's call handling, so ordinary execution uses `model(x)`.[^module]

### Parentheses that create ReLU and parentheses that compute an input

<strong>ReLU</strong> returns zero for negative values and leaves values of zero or greater unchanged. For example, it transforms `[-2,3,0,-4]` into `[0,3,0,0]`. Individual values may change, but the shape stays the same, and there are no trainable weights.[^relu]

Repeated weighted sums can sometimes be combined into a single weighted sum. ReLU adds a <strong>nonlinear transformation</strong>: the computation differs for negative and positive inputs. For now, we can begin by checking how the actual values change.

| Expression | Meaning |
| --- | --- |
| `F.relu` | The function itself |
| `F.relu(x)` | The output tensor returned by calling the function |
| `F.relu()` | An invalid call with the required input missing |
| `nn.ReLU` | The class itself |
| `nn.ReLU()` | Create a ReLU layer object |
| `relu_layer(x)` | Execute an already-created layer object |
| `nn.ReLU()(x)` | Create the object and immediately compute the input |

```python
relu_input = torch.tensor([-2.0, 3.0, 0.0, -4.0], device=device)

relu_function = F.relu  # Store the function itself; do not compute an input yet.
relu_layer = nn.ReLU()  # Create a layer object; do not compute an input yet.

result1 = relu_function(relu_input)
result2 = relu_layer(relu_input)

assert torch.equal(result1, result2)  # Equal shapes and equal values at every position
print(result1)  # tensor([0., 3., 0., 0.])
```

`self.relu = F.relu` stores the function itself as an attribute and is valid. `self.relu = F.relu(x)` stores the tensor produced by a computation. `self.relu = nn.ReLU()` stores a layer object and registers it as a submodule.

`nn.Sequential` takes module objects such as `nn.ReLU()`. Inside a custom `forward` method, calling `F.relu(x)` directly is also valid. `nn.relu(x)` is not the correct API for this purpose.

Do not extend this relationship to every functional API without checking. Functional Dropout, for example, requires training or evaluation behavior to be managed through an argument. What we have checked here is the default ReLU computation.

## 7. MaxPool selects the maximum within each channel

<strong>Max pooling</strong> selects the largest number in each small spatial region of a channel. It does not combine values from different channels, so the channel count stays unchanged.

`kernel_size=2` means the region checked for its maximum is `2×2`. `stride=2` means moving two positions to the next starting point. If a problem specifies “2×2 MaxPool, stride 2,” translate each condition into code separately. A stride of 2 does not imply that the kernel must also be 2.

```python
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
print(pool_output)
# tensor([[[[7., 8.],
#           [6., 9.]]]])
print(tuple(pool_output.shape))  # (1, 1, 2, 2)
```

The upper-left region keeps 7 from `3,7,2,5`, and the upper-right region keeps 8 from `1,4,8,0`. The lower two regions keep 6 and 9, respectively. Sixteen input numbers become four maxima, one from each region.

For Conv, `kernel_size` describes the spatial size of a weight array. For MaxPool, it describes the size of the region whose values are checked. <strong>MaxPool has no trainable kernel weights.</strong>

With the current settings of `kernel_size=2, stride=2, padding=0, ceil_mode=False`, height and width become `floor(H/2)` and `floor(W/2)`. Thus `32→16` and `30→15`. For an odd size, `15→7`, the remaining edge position does not produce an additional output.[^pool]

If stride is omitted, as in `MaxPool2d(2)`, it defaults to the kernel size of 2. However, the values can also differ, as in `MaxPool2d(3, stride=2)`.

## 8. We can compare weight counts before providing an input

<strong>`nn.Sequential`</strong> groups layer objects to execute in a defined order. It registers the layers when the object is created. When an input is later passed to it, it sends each layer's output to the next layer.[^sequential]

The following code compares two structures with 16 input, intermediate, and output channels. We exclude biases and count only weights.

```python
C = 16

two_small = nn.Sequential(
    nn.Conv2d(C, C, 3, padding=1, bias=False),
    nn.ReLU(),
    nn.Conv2d(C, C, 3, padding=1, bias=False),
).to(device)

one_large = nn.Conv2d(
    C, C, 5, padding=2, bias=False
).to(device)

# Sum the number of values in each parameter tensor provided by parameters().
small_count = sum(p.numel() for p in two_small.parameters())
large_count = sum(p.numel() for p in one_large.parameters())

print(small_count)  # 4608
print(large_count)  # 6400
```

At creation time, Conv knows its input channels, output channels, and kernel size. It can therefore create the required weights without an input image. The code above has <strong>created layers and counted their parameters</strong>; it has not computed feature maps.

| Structure | Weight calculation, excluding biases | Result |
| --- | --- | ---: |
| One `3×3` Conv layer | `16×16×3×3` | 2,304 |
| Two `3×3` Conv layers | `2×16×16×3×3` | 4,608 |
| One `5×5` Conv layer | `16×16×5×5` | 6,400 |

In `16×16`, the first 16 is the number of output filters, and the second is the number of input-channel kernels per filter. Two Conv layers having weights of the same shape does not mean they share the same weight object. The total is 2,304 weights from the first layer plus 2,304 separate weights from the second.

Including biases would add `16+16=32` to the two small-kernel layers and 16 to the one large-kernel layer. Input height, width, and batch size do not multiply the parameter count.

Also, `parameters()` includes parameters whose gradient computation is currently disabled. To count only trainable parameters, add a `requires_grad` condition.

```python
trainable_count = sum(
    p.numel() for p in two_small.parameters() if p.requires_grad
)
```

## 9. Why do two 3×3 layers connect to a 5×5 input region?

The previous section counted parameters only. Now we will add an actual input and trace which parts of the original input connect to one output cell of the second layer.

```python
rf_x = torch.randn(1, 16, 7, 7, device=device)

with torch.no_grad():
    # Sequential index 0 is the first Conv, 1 is ReLU, and 2 is the second Conv.
    rf_m = two_small[1](two_small[0](rf_x))
    rf_y = two_small[2](rf_m)
    rf_large = one_large(rf_x)

print(tuple(rf_m.shape))  # (1, 16, 7, 7)
print(tuple(rf_y.shape))  # (1, 16, 7, 7)
```

Consider the single cell `rf_y[0,0,3,3]`. The second Conv reads a `3×3` region at rows 2–4 and columns 2–4 from each channel of the intermediate feature map `rf_m`. With 16 intermediate channels, it receives `16×9=144` intermediate values.

Each intermediate value was produced by the first Conv from a `3×3` region of the original input. The first Conv also uses all 16 input channels.

| Intermediate position read by the second Conv | Original input rows used to produce that value | Original input columns |
| --- | --- | --- |
| `(2,2)` | 1, 2, 3 | 1, 2, 3 |
| `(2,3)` | 1, 2, 3 | 2, 3, 4 |
| `(2,4)` | 1, 2, 3 | 3, 4, 5 |
| `(3,2)` | 2, 3, 4 | 1, 2, 3 |
| `(3,3)` | 2, 3, 4 | 2, 3, 4 |
| `(3,4)` | 2, 3, 4 | 3, 4, 5 |
| `(4,2)` | 3, 4, 5 | 1, 2, 3 |
| `(4,3)` | 3, 4, 5 | 2, 3, 4 |
| `(4,4)` | 3, 4, 5 | 3, 4, 5 |

Collecting these input positions without duplicates gives rows 1–5 and columns 1–5. A `5×5` region in each original input channel can therefore connect to the calculation of that one output cell.

<figure class="cnn-vgg-figure">
<img src="/cnn-vgg-residual-pytorch/figure-receptive-field-en.svg" alt="Tracing intermediate positions 2 through 4 read by the second 3×3 convolution back to the original input gives a 5×5 region spanning rows and columns 1 through 5." loading="lazy">
<figcaption>Both layers still have 3×3 kernels. What grows is the region of the original input that can connect to one particular output cell.</figcaption>
</figure>

The input region that can structurally affect an output value is called its <strong>theoretical receptive field</strong>. Here, both convolutions use `stride=1, dilation=1`, there is no Pool between them, and we are examining an interior position unaffected by padding.

ReLU changes values at the same positions, so it does not expand the spatial region. In practice, a zero weight or a ReLU output of zero may remove the influence of a particular input. The receptive field describes <strong>the region that can be connected</strong>.

### Reading the same region does not mean performing the same computation

One `5×5` Conv layer directly computes a weighted sum over that region of the original input. Two `3×3` layers first compute intermediate features and apply ReLU, then compute weighted sums of those intermediate values.

The two structures can therefore have the same receptive field without producing the same output values. A `3×3` kernel also does not turn into a `5×5` kernel during execution.

VGG uses stacks of small convolutions. In our example, we removed `6,400−4,608=1,792` weights, a reduction of 28% relative to the single large-kernel layer. This percentage applies to the current comparison, with all channel counts set to 16 and biases excluded. Fewer parameters alone do not guarantee better accuracy or faster execution.[^vgg]

## 10. Combine two Conv layers and one Pool into a VGG-style block

A <strong>block</strong> groups several layers into one component. This block computes features twice while preserving spatial size, then reduces height and width with the final Pool.

The computation order is `Conv1 → ReLU → Conv2 → ReLU → MaxPool`. For example, if the input has 3 channels and the block's output has 8, the first Conv must map `3→8` and the second `8→8`. The second Conv receives the result computed by the first.

We express this relationship in the initializer as follows.

```python
class MyVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()  # Initialize the parent Module's management functionality.

        # Change the original input channels to the block's output channel count.
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

        # Receive the channels produced by the first Conv.
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

        # Preserve channels while reducing height and width.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        result = self.conv1(x)
        result = F.relu(result)
        result = self.conv2(result)
        result = F.relu(result)
        return self.pool(result)
```

`result` stores each stage's output tensor in turn. `self.conv1` is a layer object, while `self.conv1(x)` is the tensor computed by that layer. We do not pass the layer object itself as the next layer's input.

The two Conv layers having the same output channel count does not mean they repeat the same computation. Each has its own weights, and the second receives data already transformed by the first.

With the current default of `bias=True`, both Conv layers also create biases. This differs from the earlier weight-count comparison, which used `bias=False`. We must include biases when counting this model's total parameters later.

### Check both square and rectangular inputs

Create one `MyVGGBlock(3,8)` object and pass in two input sizes. The `*input_shape` syntax unpacks the four numbers in a tuple into separate positional arguments.

```python
vgg_block = MyVGGBlock(3, 8).to(device)

with torch.no_grad():
    for input_shape, expected_shape in [
        ((2, 3, 32, 32), (2, 8, 16, 16)),
        ((1, 3, 30, 26), (1, 8, 15, 13)),
    ]:
        images = torch.randn(*input_shape, device=device)

        # Compute each stage separately to inspect the intermediate shapes.
        after_conv1 = F.relu(vgg_block.conv1(images))
        after_conv2 = F.relu(vgg_block.conv2(after_conv1))
        after_pool = vgg_block.pool(after_conv2)

        # Separately run the block's forward again, starting from the original input.
        block_output = vgg_block(images)

        assert tuple(block_output.shape) == expected_shape
        assert torch.allclose(
            block_output, after_pool, atol=1e-6, rtol=1e-4
        )
        print(
            input_shape,
            tuple(after_conv1.shape),
            tuple(after_conv2.shape),
            tuple(after_pool.shape),
        )
```

| Input | After Conv1 and ReLU | After Conv2 and ReLU | After Pool |
| --- | --- | --- | --- |
| `[2,3,32,32]` | `[2,8,32,32]` | `[2,8,32,32]` | `[2,8,16,16]` |
| `[1,3,30,26]` | `[1,8,30,26]` | `[1,8,30,26]` | `[1,8,15,13]` |

The final Pool reduces the spatial size. Therefore, “the Conv layers inside the block preserve spatial size” and “the entire block preserves spatial size” are different statements.

We call `vgg_block(images)` again after calculating the intermediate results to check that the separately written stages agree with `forward`. During ordinary use, calling the block once is enough. The previously computed `after_conv2` is not automatically reused by the full call.

## 11. How does this block differ from the actual VGG-16?

Our `MyVGGBlock` is for practicing a structure that stacks small convolutions and applies Pool. A small model made from two such blocks is not the complete VGG-16.

A standard VGG-16 configuration uses `2, 2, 3, 3, 3` Conv layers in its five convolutional groups. That gives 13 convolutions, followed by three fully connected layers. In a <strong>fully connected layer, or FC layer</strong>, each output is computed using all input features. Here, it is implemented with `nn.Linear`.[^vgg]

For a `224×224` RGB input, the overall structure is as follows.

| Stage | Number of Conv layers | Output channels of the stage | Height and width after Pool |
| --- | ---: | ---: | --- |
| First group | 2 | 64 | `112×112` |
| Second group | 2 | 128 | `56×56` |
| Third group | 3 | 256 | `28×28` |
| Fourth group | 3 | 512 | `14×14` |
| Fifth group | 3 | 512 | `7×7` |

The feature counts across the three FC layers change as `25,088→4,096→4,096→1,000`. The final `25,088` features come from `512×7×7`. The 16 in the name counts the 13 Conv layers and 3 FC layers with trainable weights in this configuration. It does not count every ReLU and Pool layer.

TorchVision's VGG implementation includes `AdaptiveAvgPool2d` before the classifier to make its spatial output `7×7`, along with ReLU and Dropout between FC layers. Dropout randomly sets some values to zero during training; we will discuss it later with execution modes. Also, while the paper's architecture table includes a final Softmax, TorchVision's `forward` returns scores before applying Softmax.[^vgg-code]

Our small model reduces the channel counts to 8 and 16 and uses four Conv layers and one Linear layer to examine the computation connections. We can study channel connections and Flatten without constructing the entire large classifier of the actual VGG model.

## 12. Flatten counts values in the final output data

Now let us design a small model that classifies `32×32` RGB inputs into five classes. “Five classes” means producing five output scores. Their actual meanings are determined by the correspondence between data and target class indices.

Connect two VGG-style blocks as follows.

| Stage | Shape | Why it changes |
| --- | --- | --- |
| Input | `[N,3,32,32]` | `N` RGB images |
| `MyVGGBlock(3,8)` | `[N,8,16,16]` | Conv produces 8 channels; Pool halves the spatial size |
| `MyVGGBlock(8,16)` | `[N,16,8,8]` | Conv produces 16 channels; Pool halves the spatial size |
| `flatten(start_dim=1)` | `[N,1024]` | Combine each image's `16×8×8` values along one axis |
| `Linear(1024,5)` | `[N,5]` | Compute five scores per image |

<figure class="cnn-vgg-figure">
<img src="/cnn-vgg-residual-pytorch/figure-model-shapes-en.svg" alt="RGB input of shape N,3,32,32 passes through two VGG-style blocks to become N,16,8,8. Flattening 1024 values per image and applying Linear produces scores of shape N,5." loading="lazy">
<figcaption>N remains the number of images. 16×8×8 counts the values per image in the final feature maps.</figcaption>
</figure>

Linear does not receive the Conv filters or kernels themselves. It receives <strong>output data computed by Conv and Pool</strong>. There are 16 final feature maps, each containing `8×8=64` numbers. Therefore, each image supplies `16×64=1,024` input features.

Using `16×9` would count a different object. `3×3=9` counts the weights in one per-channel kernel, while `8×8=64` counts the data values in one final output feature map.

### start_dim=1 keeps the batch axis

<strong>Flatten</strong> combines multiple axes into one. With `start_dim=1`, axis 0 is preserved, and axes 1 through the last axis are combined.[^flatten]

| Axis | Original role | After flattening |
| --- | --- | --- |
| 0 | `N` images | Preserved as `N` |
| 1 | 16 channels | Combined with the spatial axes below into 1,024 features |
| 2 | Height 8 | Included in the feature axis above |
| 3 | Width 8 | Included in the feature axis above |

This operation does not sum or average values. It changes the shape while preserving the order of values within each image. It also does not always create a new copy in memory. Depending on the input and the axes being flattened, it may return the original object, a view, or a copy. A view is a tensor that uses the same storage with a different shape.[^flatten]

The following expressions perform the same task.

```python
# These usage examples assume features is a tensor produced by the preceding block.
# flat = features.flatten(start_dim=1)         # Tensor method
# flat = torch.flatten(features, start_dim=1)  # torch function
# flatten_layer = nn.Flatten(start_dim=1)      # Create a layer object
# flat = flatten_layer(features)              # Execute the created layer
```

The API used here is not `F.flatten(...)`.

### Flattening the batch can hide a mistake when N=1

The default starting axis of `features.flatten()` is 0. It combines the batch axis as well.

| Original shape | `flatten(start_dim=1)` | `flatten()` |
| --- | --- | --- |
| `[2,16,8,8]` | `[2,1024]` | `[2048]` |
| `[1,16,8,8]` | `[1,1024]` | `[1024]` |

For `N=2`, the last axis becomes 2,048 and no longer matches `Linear(1024,5)`. For `N=1`, the last axis happens to be 1,024, so Linear can compute a result, but its shape is `[5]`. This differs from the required output `[1,5]`, which includes a batch axis.

Flatten does not interpret what an image means. The <strong>batch axis that distinguished individual images has disappeared</strong>. That is why we need a check with two images as well as one.

## 13. Connect two blocks and Linear into one model

The complete model has the two previously defined blocks and one Linear layer as attributes. Its `forward` method describes the computation order.

```python
class MySmallVGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = MyVGGBlock(3, 8)
        self.block2 = MyVGGBlock(8, 16)

        # The final feature maps contain 16×8×8 values per image.
        self.fc = nn.Linear(16 * 8 * 8, 5)

    def forward(self, x):
        result = self.block1(x)
        result = self.block2(result)
        result = result.flatten(start_dim=1)
        return self.fc(result)  # These are still scores, not probabilities.


# Create an actual object separately from defining the class.
torch.manual_seed(16)
small_model = MySmallVGG().to(device)
```

Let us check both the intermediate shapes and the full `forward` result. `logits` is the name for the classifier's raw scores; the next section distinguishes them from probabilities.

```python
with torch.no_grad():
    for batch_size in (2, 1):
        images = torch.randn(batch_size, 3, 32, 32, device=device)

        feature1 = small_model.block1(images)
        feature2 = small_model.block2(feature1)
        flat_features = feature2.flatten(start_dim=1)
        manual_logits = small_model.fc(flat_features)

        # Recompute the complete model, starting from images.
        logits = small_model(images)

        assert tuple(feature1.shape) == (batch_size, 8, 16, 16)
        assert tuple(feature2.shape) == (batch_size, 16, 8, 8)
        assert tuple(flat_features.shape) == (batch_size, 1024)
        assert tuple(logits.shape) == (batch_size, 5)
        assert torch.allclose(
            logits, manual_logits, atol=1e-6, rtol=1e-4
        )

        print(
            "block1:", tuple(feature1.shape),
            "block2:", tuple(feature2.shape),
            "flatten:", tuple(flat_features.shape),
            "logits:", tuple(logits.shape),
        )
```

The shapes we can check consistently are:

```text
block1: (2, 8, 16, 16) block2: (2, 16, 8, 8) flatten: (2, 1024) logits: (2, 5)
block1: (1, 8, 16, 16) block2: (1, 16, 8, 8) flatten: (1, 1024) logits: (1, 5)
```

Changing the number of images from 2 to 1 leaves the number of features per image at 1,024. Linear applies the same weights to each image's feature vector. A <strong>feature vector</strong> here is an array listing one image's feature values along a single axis.

### What should we check when changing the input height and width?

This model's Linear layer was designed around `32×32` inputs. That does not mean every input of another size causes an error. For example, a `33×33` input also becomes `16×16` and then `8×8` after the current two Pool operations.

The precise requirement is that <strong>each image must have 1,024 feature values when it reaches Linear</strong>. Batch size `N` may change, but arbitrary changes to height and width do not always satisfy this condition.

When the configuration changes, recalculate the feature count. For example, a separate grayscale model with the shapes `[N,1,28,28] → [N,16,14,14] → [N,32,7,7]` has `32×7×7=1,568` features. Its Linear layer for 10 classes would be `Linear(1568,10)`. Its final output size differs from that of the current RGB model, which has 1,024 features.

## 14. Separate Linear weights, logits, and probabilities

`Linear(1024,5)` uses 1,024 input values to produce five output scores. To compute one output score, it multiplies each of the 1,024 inputs by a weight, sums the products, and adds one bias for that output. Five different sets of weights each perform this computation.

PyTorch stores Linear weights in the order `[out_features, in_features]`. Thus this layer's `weight.shape` is `[5,1024]`, and its bias shape is `[5]`.[^linear]

| Data or parameter | Shape | Role |
| --- | --- | --- |
| Flattened input `X` | `[N,1024]` | 1,024 features per image |
| `weight` | `[5,1024]` | Weights used by each of the five outputs |
| `weight.T` | `[1024,5]` | Weights with the two axes exchanged for matrix multiplication |
| `bias` | `[5]` | Bias added to each output |
| Output | `[N,5]` | Five scores per image |

The expression is `X @ weight.T + bias`. `@` means matrix multiplication, and `.T` here exchanges the two axes of the two-dimensional weight matrix. The same five biases are added to each image's scores.

```python
with torch.no_grad():
    # Use flat_features from the preceding check.
    direct_scores = (
        flat_features @ small_model.fc.weight.T + small_model.fc.bias
    )
    layer_scores = small_model.fc(flat_features)
    assert torch.allclose(
        direct_scores, layer_scores, atol=1e-6, rtol=1e-4
    )

print(tuple(small_model.fc.weight.shape))  # (5, 1024)
print(small_model.fc.weight.numel())       # 5120
print(small_model.fc.bias.numel())         # 5
```

If you previously defined a matrix manually as `W=[number of inputs, number of outputs]`, its order is the reverse of PyTorch's storage order. This is why the matrix multiplication uses `weight.T`.

### Linear does not automatically output probabilities

<strong>Logits</strong> are the raw scores produced by Linear in this classification model. They can be negative, and their sum does not have to equal 1.

<strong>Softmax</strong> converts an image's class scores to values between 0 and 1 whose sum is 1. These represent the model's probabilities for the classes, but applying Softmax to an untrained model does not create useful predictions or reliable probabilities.[^softmax]

```python
small_model.eval()

with torch.no_grad():
    prediction_images = torch.randn(2, 3, 32, 32, device=device)
    logits = small_model(prediction_images)     # [2, 5]

    # Axis 1 contains the five class scores within each image.
    probabilities = torch.softmax(logits, dim=1)

    # Select the class index with the largest score.
    predicted_classes = logits.argmax(dim=1)   # [2]

print(probabilities.sum(dim=1))  # Approximately tensor([1., 1.])
print(predicted_classes)        # Class index for each image, in the range 0–4
```

`argmax` returns the index of the largest value along the specified axis. If we only need the class with the highest score, we can apply it directly to the logits without Softmax, because Softmax preserves the ordering of the scores.

What output index 0 means is established through the dataset's target indices and training. `Linear(1024,5)` does not independently choose the names of five classes.

For ordinary classification with integer target indices, pass <strong>logits without applying Softmax first</strong> to `nn.CrossEntropyLoss`. Its computation includes the necessary normalization. For the current model, its inputs have the following format.[^crossentropy]

| Value passed to the loss function | Shape | Data type and values |
| --- | --- | --- |
| `logits` | `[N,5]` | Real-valued scores for each class |
| `labels` | `[N]` | `torch.long`, with each value in 0–4 |

### The complete model has 9,421 parameters

This small RGB model includes the default biases of Conv and Linear. ReLU, Pool, and Flatten have no trainable parameters.

| Layer | Weight and bias calculation | Parameter count |
| --- | --- | ---: |
| First block's Conv1, `3→8` | `8×3×3×3 + 8` | 224 |
| First block's Conv2, `8→8` | `8×8×3×3 + 8` | 584 |
| Second block's Conv1, `8→16` | `16×8×3×3 + 16` | 1,168 |
| Second block's Conv2, `16→16` | `16×16×3×3 + 16` | 2,320 |
| Linear, `1024→5` | `5×1024 + 5` | 5,125 |
| Total | `224+584+1168+2320+5125` | **9,421** |

```python
total_parameters = sum(p.numel() for p in small_model.parameters())
assert total_parameters == 9421
print(total_parameters)
```

The 1,024 values after Flatten per image, the 5 final scores, and the model's 9,421 total parameters count different things. The earlier grayscale block configuration of `1→16→32` with `Linear(1568,10)` has 32,058 total parameters when default biases are included. Changing model channels or output sizes requires recalculating these counts.

## 15. A residual connection adds two numbers at corresponding positions

So far, we have passed each layer's output to the next layer in sequence. A <strong>residual connection</strong> includes both a path that computes the input through several layers and a path that skips those layers to reach an addition point.

The path that transforms the input through Conv and other operations is the <strong>main path</strong>. The path that skips some layers is the <strong>shortcut</strong>. When the shortcut passes the input unchanged, we add the main path's result and the original input at corresponding positions.

Writing the main path's computation mathematically as `F(x)`, our block is:

<div class="cnn-vgg-math">
$$y=\operatorname{ReLU}(F(x)+x)$$
</div>

The `F` in this equation is a symbol for the main path. In Python's `F.relu`, `F` is an alias for `torch.nn.functional`. The letter is the same, but the objects it refers to are different.

### First, check the addition with manually chosen numbers

The `residual_f` below is not yet a result computed by Conv. We choose its values manually to make the addition easy to inspect.

```python
residual_x = torch.tensor(
    [[2.0, -1.0],
     [0.0,  3.0]],
    device=device,
)
residual_f = torch.tensor(
    [[-1.0,  2.0],
     [ 1.0, -4.0]],
    device=device,
)

residual_sum = residual_f + residual_x
residual_y = F.relu(residual_sum)

print(residual_sum)
# tensor([[ 1.,  1.],
#         [ 1., -1.]])

print(residual_y)
# tensor([[1., 1.],
#         [1., 0.]])

print(F.relu(residual_f))  # Compare with the result without a shortcut.
# tensor([[0., 2.],
#         [1., 0.]])
```

The upper-left value is `-1+2=1`, and the lower-right value is `-4+3=-1`. ReLU follows the addition, so the final -1 becomes 0. This is neither channel concatenation nor matrix multiplication.

### Build the main path with two actual Conv layers

For an input of shape `[N,16,14,14]`, set both Conv layers to 16 channels, kernel size 3, stride 1, and padding 1. The main path's result then also has shape `[N,16,14,14]`, matching the original input.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        # x is the original input. Add values at the same sample, channel, row, and column.
        return F.relu(out + x)


block = ResidualBlock(16).to(device)
sample = torch.randn(1, 16, 14, 14, device=device)

with torch.no_grad():
    transformed = block.conv2(F.relu(block.conv1(sample)))
    block_output = block(sample)

    assert transformed.shape == sample.shape
    assert torch.allclose(
        block_output,
        F.relu(transformed + sample),
        atol=1e-6,
        rtol=1e-4,
    )

index = (0, 0, 0, 0)  # First sample, first channel, first row, first column
print("Input value:", sample[index].item())
print("Main-path value:", transformed[index].item())
print("ReLU after addition:", block_output[index].item())
```

One output cell is `max(main-path value at that cell + input value at that cell, 0)`. We have not overwritten `x` with another value while computing `out`, so the original input remains available for the final addition.

### Why is this called learning a residual?

Let `H(x)` be the desired transformation at the addition point. If the unchanged input `x` is added, the main path can be structured to compute the difference `H(x)−x`. This difference is called the <strong>residual</strong>.

The original ResNet research addressed the difficulty of training deeper networks when adding layers worsened even the training error. Adding a shortcut and a residual helps optimize deep models. However, it does not justify claiming that all training difficulties or vanishing gradients are completely resolved. Our small example only checks that residual addition has been implemented; it does not measure a performance improvement.[^resnet]

Also, `return F.relu(out)` and `return F.relu(out + x)` can return the same shape. Therefore, <strong>a shape check alone cannot establish whether the shortcut was used.</strong> Read the actual operations in `forward` and compare against the added result as above. For some inputs, even the values of the two expressions may happen to agree.

## 16. Match differently sized paths to the target output

Suppose we now want to increase the block's output channels while reducing height and width.

`[N,16,14,14] → [N,32,7,7]`

We can obtain this size by setting the main path's first Conv to `16→32`, `3×3`, `stride=2`, and `padding=1`. The next Conv maps `32→32` while preserving spatial size.

The problem is that the original input `x` still has shape `[N,16,14,14]`. We cannot directly add its values at corresponding positions to the main path's `[N,32,7,7]` result. Placing a `16→32`, `1×1`, `stride=2` Conv on the shortcut makes both paths produce the same output size.

Using a trainable transformation to match the shortcut's channels and size is called a <strong>projection shortcut</strong>.[^resnet]

<figure class="cnn-vgg-figure">
<img src="/cnn-vgg-residual-pytorch/figure-residual-paths-en.svg" alt="A residual block that preserves shape adds the input unchanged. A block that increases channels from 16 to 32 and reduces spatial size from 14 to 7 applies a 1×1 Conv with stride 2 on the shortcut to match both paths." loading="lazy">
<figcaption>Choose the shape at the addition point first. Configure the main path and shortcut so that each produces a tensor of that shape.</figcaption>
</figure>

```python
class ProjectionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1
        )

        # Use a 1×1 Conv to change channels and stride=2 to reduce spatial size.
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1, stride=2
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        shortcut_result = self.shortcut(x)
        return F.relu(out + shortcut_result)


projection_block = ProjectionResidualBlock(16, 32).to(device)
projection_input = torch.randn(1, 16, 14, 14, device=device)

with torch.no_grad():
    main_path = projection_block.conv2(
        F.relu(projection_block.conv1(projection_input))
    )
    shortcut_path = projection_block.shortcut(projection_input)
    projection_output = projection_block(projection_input)

    assert tuple(main_path.shape) == (1, 32, 7, 7)
    assert main_path.shape == shortcut_path.shape
    assert torch.allclose(
        projection_output,
        F.relu(main_path + shortcut_path),
        atol=1e-6,
        rtol=1e-4,
    )

print(tuple(main_path.shape))          # (1, 32, 7, 7)
print(tuple(shortcut_path.shape))      # (1, 32, 7, 7)
print(tuple(projection_output.shape))  # (1, 32, 7, 7)
```

The main path and shortcut have different Conv objects, so their weights are separate as well. They start from the same input but perform different calculations.

### Why change the original-input path?

There is no rule requiring that only the input path be changed. We could enlarge the main path's spatial result from `7×7→14×14` and change its channels from `32→16` before adding the original input.

However, that block would output `[N,16,14,14]`. It would be a different design from the original target of `[N,32,7,7]`. Enlarging the spatial dimensions also does not exactly restore input information lost earlier.

Therefore, <strong>choose the target shape to pass to the next layer first, then make both paths match it before addition.</strong> When the size stays unchanged, pass the input through directly. When channels increase and spatial size decreases, as in this example, transform the shortcut as well.

PyTorch addition permits <strong>broadcasting</strong> between some differently shaped tensors. These rules compute as though certain axes of size 1 were repeated to match the other tensor. However, our residual block intends a direct correspondence between sample, channel, and position. Check that the entire shapes match, rather than only observing that addition runs without an error.

Both residual blocks here are educational implementations that omit BatchNorm to explain the connections. The actual TorchVision ResNet includes BatchNorm and differs in details such as omitting biases in some Conv layers.[^resnet-code]

## 17. A 1×1 Conv still has weights across multiple channels

`1×1` describes <strong>the spatial size of each per-channel kernel</strong>. It does not mean the entire `Conv2d(16,32,1)` layer has only one weight.

| Item | Calculation | Parameter count |
| --- | --- | ---: |
| A kernel corresponding to one input channel | `1×1` | 1 |
| A filter producing one output channel | `16×1×1` | 16 |
| The complete layer producing 32 output channels | `32×16×1×1` | 512 |
| Default biases | One per output channel | 32 |

The weight shape is `[32,16,1,1]`. `weight[0]` contains the 16 weights that produce the first output channel. `weight[0,0]` is the `1×1` kernel within that filter corresponding to input channel 0.

At one spatial position, the output value is calculated by multiplying the values from all 16 input channels by their respective weights, summing the products, and adding a bias. <strong>A `1×1` Conv combines multiple values along the channel dimension.</strong> Its difference from reading a surrounding `3×3` region is the spatial extent.

<figure class="cnn-vgg-figure">
<img src="/cnn-vgg-residual-pytorch/figure-pointwise-en.svg" alt="One 1×1 Conv filter multiplies the values of 16 input channels at the same position by 16 weights and sums them. Thirty-two such filters have 512 weights and 32 biases and produce 32 output channels." loading="lazy">
<figcaption>The 1×1 spatial size and the 16 input channels describe different axes.</figcaption>
</figure>

### Which positions does stride=2 read?

The current shortcut uses `kernel_size=1, stride=2, padding=0`. For an input of height 14, it reads row indices `0,2,4,6,8,10,12` and follows the same pattern across the width. There are seven positions in each direction, producing a `7×7` output.

A 1×1 operation does not always preserve spatial size. With stride 1, it preserves size under the current conditions. With stride 2, it selects computation positions two steps apart.

```python
print(tuple(projection_block.shortcut.weight.shape))  # (32, 16, 1, 1)
print(projection_block.shortcut.weight.numel())       # 512
print(projection_block.shortcut.bias.numel())         # 32
```

### Are the weights fixed at 1?

The weights of a default Conv begin with initialized values and are adjusted during training. A kernel size of 1 does not mean the weight values are also 1.

If we manually set every weight to 1 and every bias to 0, each output channel sums the values of all 16 input channels at the same position. The output channels would then produce the same sum; this would not copy the original input unchanged.

Conversely, setting all weights to 0 removes the input's contribution and leaves only the bias. The bias must also be 0 for the entire output to become zero. This is why we must distinguish a shortcut that passes the original input unchanged from one that goes through a trainable 1×1 Conv.

## 18. no_grad and eval are different settings

So far, we have used `torch.no_grad()` for calculation checks and also called `small_model.eval()` when inspecting classification scores. They perform different tasks.

First, <strong>automatic differentiation, or autograd</strong>, uses the relationships between executed operations to compute rates of change called <strong>gradients</strong>. During training, these gradients describe how sensitive the loss is to changes in each parameter.

To do this, the forward pass records which operations produced which tensors. This connected record is called a <strong>computational graph</strong>. The forward pass computes outputs from inputs, while backpropagation follows those relationships backward to calculate the loss's gradients.[^autograd]

### no_grad disables backward-graph recording for the enclosed computation

Inside `with torch.no_grad():`, convolutions, ReLU, and Pool still compute their output values normally. Ordinary operations, however, are not recorded in a graph for backpropagation. During checks or inference that will not be used for training, this can reduce unnecessary recording and its associated memory use.[^nograd]

Here, `with torch.no_grad():` disables gradient recording on entry to the block and restores the previous setting on exit. The `with` statement delegates such entry and exit handling to the object being used.

| What `no_grad` does | What must be managed separately |
| --- | --- |
| Disable ordinary backward-graph recording within the block. | Permanently changing existing parameters' `requires_grad` settings |
| Continue computing outputs for the current input. | Clearing existing `.grad` values |
| Restore the previous recording setting on leaving the block. | Switching the model to evaluation mode |

`requires_grad` indicates whether a tensor needs gradients, while `.grad` is the attribute that stores computed gradients. Erasing or reinitializing existing weights is also not part of what `no_grad` does.

Some functions, including those that create a new `Parameter`, are exceptions. Do not memorize the rule as “every tensor created inside the block always has `requires_grad=False`.” Here, we are examining an ordinary forward pass through an already-created model.[^nograd]

### eval makes layers use evaluation behavior

`model.eval()` switches to evaluation mode, equivalent to `model.train(False)`. It also applies to registered submodules and remains active until `model.train()` is called again. However, it does not disable autograd recording.[^module]

Typical layers whose behavior depends on the mode include:

| Layer | Training mode, `train` | Evaluation mode, `eval` |
| --- | --- | --- |
| Dropout | Randomly set some elements to zero and rescale the remaining values. | Pass the input through unchanged. |
| BatchNorm | With default settings, normalize using the current batch's statistics and update stored statistics. | With default settings, use the stored statistics. |

<strong>Normalization</strong> in this context adjusts the distribution of values using a mean and variance. Settings that do not use stored statistics, such as BatchNorm's `track_running_stats=False`, behave differently from the defaults above.[^dropout][^batchnorm]

The Conv, ReLU, MaxPool, and Linear layers in our current small model do not change their own computation between train and eval modes. Still, explicitly calling `eval()` for evaluation makes the intended behavior clear if mode-dependent layers are added later.

### The two settings can be combined independently

| Execution combination | Backward-graph recording | Dropout and default BatchNorm behavior |
| --- | --- | --- |
| `train` + ordinary execution | Recorded when the required conditions hold | Training behavior |
| `train` + `no_grad` | Enclosed ordinary operations are not recorded | Training behavior |
| `eval` + ordinary execution | Recorded when the required conditions hold | Evaluation behavior |
| `eval` + `no_grad` | Enclosed ordinary operations are not recorded | Evaluation behavior |

Recording occurs when gradient tracking is enabled and an input or parameter requiring gradients participates in a differentiable operation. <strong>Even if the input image's `requires_grad` is False, the output can be tracked when the model weights have it set to True.</strong>

```python
probe_input = torch.randn(1, 3, 32, 32, device=device)
small_model.eval()

tracked_output = small_model(probe_input)

with torch.no_grad():
    untracked_output = small_model(probe_input)

print(tracked_output.requires_grad)        # True: eval alone does not disable tracking.
print(untracked_output.requires_grad)      # False
print(small_model.fc.weight.requires_grad) # True: existing weight settings are preserved.
```

`model.train()` also changes only the mode. Calling it alone does not calculate a loss or update weights.

### Manipulating a tensor does not always require no_grad

The deciding question is whether the operation needs to be differentiated during subsequent training. For example, if training computes `features = features * 2` and passes the result to the next layer, this multiplication affects the loss and should ordinarily be recorded. Creating the new result inside `no_grad` can cut the gradient connection to earlier layers through that operation.

By contrast, directly modifying weights between training steps is ordinarily kept out of differentiation records.

```python
# Example syntax for manually modifying weights. This example does not execute it.
# with torch.no_grad():
#     layer.weight.mul_(0.9)
```

The trailing `_` in `mul_` means it modifies the existing tensor directly. However, `no_grad` does not make modifications safe at any point in execution. Changing values needed for backpropagation after the forward pass and before backward completes can cause errors or incorrect calculations.[^autograd]

An ordinary `optimizer.step()` handles what is needed for parameter updates, so users do not need to wrap it in `no_grad` each time. Merely inspecting values with `print(layer.weight)` or `tensor.shape` does not itself require this setting either.

Inference also has `torch.inference_mode()`. It reduces additional bookkeeping overhead but places stricter constraints on using tensors created inside it in later autograd computations. It should therefore not be treated as completely equivalent to `no_grad`.[^autograd]

## 19. Actual training follows structural checks

Checking shapes with random inputs is a <strong>structural check</strong>. Determining whether a model classifies well requires actual data and targets, a training process, and a separate evaluation process.

The model has produced scores so far because initial weights alone are enough to perform a forward calculation. Creating the model or calling `eval()` does not mean training has been completed.

### What actual training adds

A <strong>loss</strong> is a training objective calculated by comparing predictions and targets. An <strong>optimizer</strong> is an object that uses computed gradients to update parameters.

| Step | Code | Role |
| --- | --- | --- |
| 1 | `model.train()` | Select the mode for training behavior |
| 2 | `optimizer.zero_grad(...)` | Clear gradients from the previous batch |
| 3 | `logits = model(images)` | Compute raw scores in a forward pass |
| 4 | `loss = criterion(logits, labels)` | Calculate loss from predictions and targets |
| 5 | `loss.backward()` | Calculate the loss's gradient for each parameter |
| 6 | `optimizer.step()` | Update parameters using the computed gradients |

`criterion` is a variable name for the loss-function object. We can use `nn.CrossEntropyLoss()` for the current five-class classification task.[^crossentropy]

The following function shows the sequence for training on one batch. <strong>Both this post and the complete runnable file define the function without calling it.</strong>

```python
def train_one_batch(model, images, labels, optimizer, criterion):
    model.train()

    # Clear previous gradients. Set them to None to prepare for the next computation.
    optimizer.zero_grad(set_to_none=True)

    # Do not wrap calculations needed for training in no_grad.
    logits = model(images)
    loss = criterion(logits, labels)

    loss.backward()   # Compute gradients; this does not yet update parameters.
    optimizer.step()  # Update parameters using those gradients.

    return loss.item()  # A Python number for logging


# Example objects to create after preparing actual data and targets.
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(small_model.parameters(), lr=0.01)
# loss_value = train_one_batch(
#     small_model, images, labels, optimizer, criterion
# )
```

`images` should be a `[N,3,32,32]` tensor on the same device as the model, and `labels` a `[N]` integer tensor on that same device. Targets use `torch.long` and values in the range 0–4. The `lr` in the comment is the <strong>learning rate</strong>, which controls the size of parameter updates.

PyTorch backpropagation can accumulate gradients, so ordinary batch-by-batch training clears the previous gradients first. `set_to_none=True` sets `.grad` to `None` instead of a tensor filled with zeros. `None` is Python's special value representing the absence of a value.[^optim]

For evaluation or prediction, `model.eval()` and `with torch.no_grad():` are commonly used together. When training resumes, switch back with `model.train()` and perform the training forward pass outside `no_grad`.

### What do ready flags and exception handling in exercise code mean?

If exercise code contains `vgg_ready = False` or `small_model_ready = False`, these are ordinary Python variables created to indicate whether checks have passed. They are not states automatically managed by PyTorch. Similarly, `small_model = None` indicates that no actual model object has been stored yet.

`raise NotImplementedError(...)` deliberately raises an exception to indicate an unfinished implementation. If it remains in `__init__`, object creation stops. If it is after `return`, it will not execute on that path, but it should still be removed from completed code.

An `except NotImplementedError` clause for a `try` block handles only that exception. It does not also catch a `RuntimeError` from a shape mismatch, an `AssertionError` from a failed check, or an `AttributeError` from invalid attribute access.[^python]

The standalone file for this post creates and checks the completed model without these exercise-progress conditions. If a check fails, it leaves the error visible so that the cause can be inspected.

### Check what the checks actually establish

Our checks verify output sizes, parameter counts, and agreement with manual calculations. They do not establish classification accuracy or generalization on actual data. Also, Python can omit educational `assert` checks when run with optimization options, so they should not replace required user-input validation.[^check]

Comparing separately executed intermediate layers with the entire `forward` result is appropriate for the current Conv, ReLU, and Pool configuration. If layers involving state or randomness are added, such as Dropout or BatchNorm during training, evaluation mode and the conditions of repeated execution must also be considered.

## 20. Check the same counts and call relationships in Rust

Channel counts, weight counts, and feature values per image stay the same when the programming language changes. We can also distinguish storing a function itself in a variable from calling it and storing its returned value.

The following Rust code uses standard features to check numerical and shape calculations only. It does not execute an entire CNN or automatic differentiation as a PyTorch model does. Running an actual neural network requires a library that provides those capabilities.

Before reading the code, let us clarify the types and syntax it uses.

| Rust expression | Meaning |
| --- | --- |
| `f32` | A 32-bit floating-point number type |
| `usize` | An unsigned integer type used for array sizes, indices, and similar values |
| `fn(f32) -> f32` | The type of a function pointer accepting one `f32` and returning an `f32` |
| `let` | Create a variable binding |
| `assert_eq!(a, b)` | A macro that checks whether two values are equal |
| `println!` | A macro that prints values |

A function pointer stores a callable function as a value. Below, writing `relu` alone refers to the function itself, while adding parentheses and an argument, as in `relu_function(-2.0)`, obtains the result of calling it.[^rust]

```rust
fn relu(x: f32) -> f32 {
    // Return zero for a negative value; otherwise, return the original value.
    x.max(0.0)
}

fn main() {
    // The same distinction as relu_function = F.relu in Python.
    let relu_function: fn(f32) -> f32 = relu;
    let result = relu_function(-2.0);
    assert_eq!(result, 0.0);

    // Conv2d(4, 6, 3): total weight count, excluding biases.
    let input_channels: usize = 4;
    let output_filters: usize = 6;
    let weights = output_filters * input_channels * 3 * 3;
    assert_eq!(weights, 216);

    // A 1×1 Conv still uses 16 input channels and 32 output filters.
    let projection_weights: usize = 32 * 16 * 1 * 1;
    assert_eq!(projection_weights, 512);

    // Calculate Linear's input size from the spatial size after two Pool operations.
    let height: usize = 32 / 2 / 2;
    let width: usize = 32 / 2 / 2;
    let in_features: usize = 16 * height * width;
    assert_eq!(in_features, 1024);

    for batch_size in [2_usize, 1_usize] {
        let flat_shape = [batch_size, in_features];
        let score_shape = [batch_size, 5];
        println!("Flatten: {flat_shape:?}, scores: {score_shape:?}");
    }

    // Check only the calculation at one position of a residual connection.
    let original: f32 = 2.0;
    let transformed: f32 = -1.0;
    let residual_result = relu(transformed + original);
    assert_eq!(residual_result, 1.0);
}
```

`x.max(0.0)` returns the larger of the two values. Because the final expression in the function body has no semicolon, its result becomes the return value. In the output string, `{flat_shape:?}` displays the array in a debug format that is convenient for inspection.

`32 / 2 / 2` is integer division here, producing 8. The final feature count is `16×8×8`, while the batch size remains a separate first axis. The `flat_shape` and `score_shape` created by this code are integer arrays describing dimensions. They are not tensors containing actual image data.

## 21. Run the complete code and review what was checked

To execute the individual calculations from the post together, use the [complete Python code](/cnn-vgg-residual-pytorch/examples-en.py). Run it in a Python environment with PyTorch available:

```bash
python examples-en.py
```

The file defines the four model classes above and `train_one_batch`. Running it directly calls `run_checks()`, which uses small inputs to check structure and computation results. It does not download a dataset or call the training function or `optimizer.step()`.

| Check | Value or relationship checked |
| --- | --- |
| One convolution output cell | Products and sums of the patch and weights agree with the Conv output within tolerance |
| ReLU | Functional and layer-object results agree |
| MaxPool | `[[7,8],[6,9]]` |
| Two small-kernel layers and one large-kernel layer | Weight counts of `4,608` and `6,400` |
| VGG-style block | Intermediate shapes for square and rectangular inputs |
| Small classification model | `[2,5]` and `[1,5]` for batches of 2 and 1 |
| Linear | `X @ weight.T + bias` agrees with the layer output |
| Total parameters | `9,421` |
| Residual block | ReLU applied after adding the main path and shortcut |
| Projection | Both paths have shape `[1,32,7,7]`; the shortcut has `512` weights and `32` biases |
| Execution modes | Tracking can remain enabled with `eval` alone; recording is disabled inside `no_grad` |

Both the Korean and English complete Python files were run with Python 3.12.14 and PyTorch 2.14.1+cpu and passed the checks. Decimal outputs from random weights may vary with execution order, version, or device. Check shapes, counts, and `allclose` results instead of relying on fixed output strings.

The Rust count calculations can be run from the [separate Rust file](/cnn-vgg-residual-pytorch/shape-checks-en.rs). In an environment with the Rust compiler, use:

```bash
rustc --crate-name cnn_shape_checks shape-checks-en.rs -o shape-checks-en
./shape-checks-en
```

Both the Korean and English Rust files were also compiled and run with rustc 1.99.1, passing all checks.

Finally, if the meaning of a number becomes unclear while reading the code, identify exactly what is being counted or computed.

| Number or expression being read | What to check first |
| --- | --- |
| `3×3` | Whether it describes a weight kernel's spatial size or the size of a patch selected from the input |
| `16×9` | Whether it counts the weights in one filter using 16 input channels |
| `16×8×8` | Whether it counts the data values per image in the final feature maps |
| `[N,5]` | Whether there are five scores for each of `N` images |
| `F(x)+x` | Whether the two paths have matching full shapes and add corresponding positions |
| `model.eval()` | Whether only the evaluation mode has been selected |
| `loss.backward()` | Whether only gradients have been computed |
| `optimizer.step()` | Whether parameters have actually been updated |

Output data produced by a convolution becomes the next layer's input. When stacking layers, check channel connections and spatial sizes. When moving to a classifier, count the values in the final output data. For residual connections, check the shapes where the two paths meet and the addition itself. Distinguishing the computation from recording it for training and from actually changing parameters makes the operation of the whole model easier to follow.

## References

API input formats and behavior were checked against the official documentation below. The VGG and ResNet structures were checked against both the original papers and the TorchVision implementations. The small models, count calculations, example checks, and figures in this post were constructed for explanation.

[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): input, output, and weight shapes; groups; output sizes; and cross-correlation.
[^check]: PyTorch, [torch.allclose](https://docs.pytorch.org/docs/stable/generated/torch.allclose.html), [torch.numel](https://docs.pytorch.org/docs/stable/generated/torch.numel.html), [Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html); Python, [the assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement).
[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html): submodule registration, calls, train and eval, and parameter management.
[^relu]: PyTorch, [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [functional.relu](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html).
[^pool]: PyTorch, [MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html): kernel, stride, output sizes, and defaults.
[^sequential]: PyTorch, [Sequential](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html).
[^vgg]: Karen Simonyan, Andrew Zisserman, [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/html/1409.1556v6), Section 2: stacked small convolutions, receptive fields, and model configurations.
[^vgg-code]: TorchVision, [VGG implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/vgg.html): configuration D's Conv sequence, avgpool, classifier, and forward return value.
[^flatten]: PyTorch, [torch.flatten](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html): start and end axes, element order, views, and copies.
[^linear]: PyTorch, [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html): weight storage order and linear computation.
[^softmax]: PyTorch, [Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html).
[^crossentropy]: PyTorch, [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): unnormalized logits and the format of targets containing class indices.
[^resnet]: Kaiming He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/html/1512.03385v1), Section 3: residual learning, identity and projection shortcuts, and changes in dimensions.
[^resnet-code]: TorchVision, [ResNet implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html): BasicBlock's Conv, BatchNorm, addition, and downsample.
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html): operation recording, grad mode, evaluation mode, inference mode, and in-place modification.
[^nograd]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html): disabling recording and exceptions when creating new parameters.
[^dropout]: PyTorch, [Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).
[^batchnorm]: PyTorch, [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html).
[^optim]: PyTorch, [Optimizer.zero_grad](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html): clearing gradients and set_to_none.
[^python]: Python, [Classes](https://docs.python.org/3/tutorial/classes.html), [Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html).
[^rust]: The Rust Programming Language, [Advanced Functions and Closures](https://doc.rust-lang.org/book/ch20-04-advanced-functions-and-closures.html): function pointers and passing functions themselves.
