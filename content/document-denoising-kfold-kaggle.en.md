+++
title = "Document Denoising with an Autoencoder: From Data Preparation to K-Fold Validation and Kaggle Submission"
description = "Pair noisy documents with clean targets, check padding and tensor shapes, and train an autoencoder. Connect cross-validation, mean image RMSE, pooled pixel RMSE, PSNR, original-size cropping, and Kaggle submission in one workflow."
date = 2026-11-07
slug = "document-denoising-kfold-kaggle"

[extra]
katex = true
toc = true
styles = ["document-denoising-kfold-kaggle/article.css"]
+++

When building a model that removes stains from documents, checking whether its output looks clean is only part of the work. We also need to check that each input matches its target, that validation images have stayed out of training, and that artificial borders have not entered the evaluation. If any of these connections is wrong, training can run without errors while leading us to the wrong conclusion.

This article uses Kaggle's **Denoising Dirty Documents** dataset to follow the process from reading image files to creating a submission. The [previous autoencoder article](/en/denoising-autoencoder-pytorch/) covered training on a batch. Here, we put together the whole experiment, including data preparation and validation.

A [complete Python example](/document-denoising-kfold-kaggle/examples-en.py) is available to download. The code in the article explains the role of each step; the downloadable program connects data loading, cross-validation, final training, and submission file creation. Dataset counts and earlier scores come from the practice records. Checking the new example's behavior is separate from reproducing the performance of those earlier experiments.

## 01. Where the input and target meet

The goal is to turn a noisy document into an image close to its clean version. An **autoencoder** combines an encoder, which computes an intermediate representation from the input, with a decoder, which turns that representation into an output. Here, we train it for restoration using pairs of noisy images and clean targets.

| Name | Contents | Where it is used |
| --- | --- | --- |
| `X`, `x_batch` | A document with stains or noise | Input to the model |
| `z` | Intermediate features computed from the input | Input to the decoder |
| `pred` | The restored image produced by the model | Prediction to compare with the target |
| `Y`, `y_batch` | The clean version of the same document | Reference for the loss and evaluation |

**The clean target does not go into the decoder.** The decoder receives `z`, which the encoder has computed. We use the clean target `Y` to judge the model's output `pred`. A classification target might be a class number; here, the target is an entire image.

<figure class="document-figure">
<img src="/document-denoising-kfold-kaggle/overview-en.svg" alt="Noisy input X passes through an encoder and decoder to become a predicted image. A separately prepared clean target Y meets the prediction in the loss calculation. During training, the loss is used to update the model parameters." loading="lazy">
<figcaption>The input, intermediate features, and output are connected through the model's computation. The target provides the reference for measuring the output error.</figcaption>
</figure>

**Parameters** are the weights and biases adjusted during training. The **loss** summarizes the difference between prediction and target according to the training objective. **Backpropagation** computes gradients: how the loss would change if a parameter changed slightly. An **optimizer** uses those gradients to update the parameters.

The same model call can serve different purposes.

| Task | Use of the target | Parameter updates | What it tells us |
| --- | --- | --- | --- |
| Training | Used to calculate the loss | Yes | How the model reduces error on its training images |
| Validation | Used to calculate scores | No | Performance on images excluded from training |
| Inference | Not required | No | The prediction for a given input |

The line `model(x)` computes a prediction. That call alone does not train the model.

## 02. Separate file counts, tensor axes, and indices

The dataset recorded in this exercise has the following structure.

| Folder | Image count | Role |
| --- | ---: | --- |
| `train` | 144 | Noisy training inputs |
| `train_cleaned` | 144 | Clean targets with matching filenames |
| `test` | 72 | Submission inputs whose targets are not public |

Of the 144 training inputs, 96 were 540 pixels wide and 420 pixels high; the other 48 were 540 pixels wide and 258 pixels high. To combine images into a batch, their tensor shapes must match. Here, we add white borders above and below the shorter images to make every image 420 pixels high.

A **tensor** holds numbers together with their arrangement along axes. Its **shape** lists the size of each axis in order. Different libraries describe the same image in different orders.

| Representation | Shape order | Example for a shorter document |
| --- | --- | --- |
| Pillow image, `img.size` | Width, height | `(540, 258)` |
| Grayscale NumPy array | Height, width | `(258, 540)` |
| One PyTorch image tensor | Channels, height, width | `(1, 258, 540)` |
| A batch of 24 padded images | Images, channels, height, width | `(24, 1, 420, 540)` |

We will write a batch shape as `(B, C, H, W)`. `B` is the number of images in the current batch, `C` is the channel count, and `H` and `W` are height and width. We use `N` for the total number of samples in a dataset.

| Axis number | Meaning | Size in `(24, 1, 420, 540)` | Valid indices |
| ---: | --- | ---: | --- |
| 0 | Image | 24 | 0–23 |
| 1 | Channel | 1 | 0 |
| 2 | Height | 420 | 0–419 |
| 3 | Width | 540 | 0–539 |

`dim=2` refers to the height axis. It does not mean that this axis has size 2. In contrast, `x[2]` selects the image at index 2 along the first axis. An axis number identifies a direction in the tensor; an index selects an element along that direction.

`x[i]` has shape `(1,H,W)`. Selecting the channel as well, with `x[i,0]`, gives `(H,W)`. To select one image while preserving the batch axis, use `x[i:i+1]`, which gives `(1,1,H,W)`.

Pillow's `img.size` is an attribute, while a tensor's `x.size()` is a method call. A tensor also exposes its shape through `x.shape`. Writing `x.size == (...)` compares the method itself with a tuple, so it does not perform the intended shape check.

## 03. Check filename pairs before relying on sorting

A file path describes where an image is stored. Creating a list of paths does not read the pixel data. The `Path` class in Python's standard `pathlib` library makes it convenient to work with directory and file paths.[^pathlib]

In the following code, `data_dir` is the directory containing `train`, `train_cleaned`, and `test`. A notebook's current working directory is not necessarily the directory in which the notebook file is saved, so print it first.

```python
from pathlib import Path

data_dir = Path.cwd()
train_paths = sorted((data_dir / "train").glob("*.png"))
clean_paths = sorted((data_dir / "train_cleaned").glob("*.png"))

print(data_dir)
print(len(train_paths), len(clean_paths))

assert train_paths, "No PNG files were found in the train directory."
assert {p.name for p in train_paths} == {p.name for p in clean_paths}
```

The `/` in `data_dir / "train"` joins path components. It does not create a directory. `glob("*.png")` finds PNG paths directly inside that directory. `sorted` orders the paths, but does not automatically interpret numbers in filenames as numeric values. Even if the order is `1.png`, `10.png`, `2.png`, the pairs remain correct when we find each target by filename.

The target for `train/12.png` is `train_cleaned/12.png`. Constructing the target path from the input's name makes that relationship explicit, instead of processing two lists separately and hoping their orders still match.

```python
from PIL import Image

noisy_imgs, clean_imgs, names = [], [], []

for noisy_path in train_paths:
    clean_path = data_dir / "train_cleaned" / noisy_path.name

    # Obtain a converted image object, then close the original file.
    with Image.open(noisy_path) as opened:
        noisy = opened.convert("L")
    with Image.open(clean_path) as opened:
        clean = opened.convert("L")

    assert noisy.size == clean.size
    noisy_imgs.append(noisy)
    clean_imgs.append(clean)
    names.append(noisy_path.name)

assert len(noisy_imgs) == len(clean_imgs) == len(names) > 0
```

`convert("L")` converts an image to 8-bit grayscale, from black at 0 to white at 255. At this point, `noisy_imgs` and `clean_imgs` are still lists of Pillow images.[^pillow]

Writing `clean_paths.append(clean)` here would mix image objects into a list intended to hold file paths. Distinguishing paths, images, and tensors in variable names helps prevent that mistake.

Check list lengths before pairing them with `zip`, too. By default, `zip` stops when the shorter input ends. If either list is empty, the loop never runs, so a “check complete” message after the loop does not prove that the data was checked.[^builtins]

To count images by size, this is enough:

```python
counts = {}
for image in noisy_imgs:
    size = image.size
    counts[size] = counts.get(size, 0) + 1

print(counts)               # Recorded counts: {(540, 420): 96, (540, 258): 48}
print(sum(counts.values())) # 144
```

The dictionary values are image counts. This code has not stored the images themselves in separate groups.

If the data arrives in ZIP files, inspect the paths inside them first. `extractall(data_dir)` does not automatically create a directory named after the ZIP file. An internal path such as `train/12.png` is preserved; an entry named only `12.png` is extracted directly into the destination. If you have already extracted the data, there is no need to repeat this step.

## 04. Tensor conversion, normalization, and padding do different jobs

To feed pixels into the network, first convert them to a floating-point tensor. Then scale their brightness to the range 0–1. Finally, make the image sizes match. These are three separate operations.

```python
import numpy as np
import torch
import torch.nn.functional as F

image = noisy_imgs[0]

x = torch.from_numpy(
    np.array(image, dtype=np.float32)
).unsqueeze(0) / 255.0
```

The expression runs in this order:

| Operation | Result | Why it is needed |
| --- | --- | --- |
| `np.array(..., dtype=np.float32)` | Floating-point array `(H,W)` | Use pixels in floating-point calculations |
| `torch.from_numpy(...)` | Tensor `(H,W)` | Use PyTorch operations |
| `unsqueeze(0)` | Tensor `(1,H,W)` | Add the grayscale channel axis |
| `/ 255.0` | Same shape, values in 0–1 | Match the brightness ranges of outputs and targets |

At this stage, `unsqueeze(0)` adds a **channel axis**. It is not a batch axis, because we have not grouped multiple images yet. The operation adds an axis without duplicating or interpolating the pixels.[^tensor-ops]

**Padding** adds an area around an image. Here, it lets us match the sizes without stretching or shrinking the original characters. Adding 81 rows above and below a 258-pixel-high document brings its height to 420.

```python
if x.shape == (1, 258, 540):
    x = F.pad(x, (0, 0, 81, 81), value=1.0)
```

For the last two spatial axes, `F.pad` takes the amounts in the order **left, right, top, bottom**. Since brightness has already been scaled to 0–1, white is `1.0`. `value=1.0` fills the newly added border; it does not overwrite the original text pixels.[^pad]

<figure class="document-figure">
<img src="/document-denoising-kfold-kaggle/padding-en.svg" alt="A document 258 pixels high and 540 pixels wide receives 81 rows of white padding above and below, making its height 420. The original area occupies height indices 81 through 338, expressed by the slice 81:339." loading="lazy">
<figcaption>Keeping the original size and border position lets us recover exactly the same document area for evaluation and submission.</figcaption>
</figure>

There is no need to create a convolutional layer just to add a white border. `F.pad` adds an area; `Conv2d` computes features using weights. Adding a trainable layer for this simple size adjustment introduces unnecessary computation and configuration.

Preparing inputs, targets, and original heights together helps keep their orders aligned.

```python
def to_tensor(image):
    # Convert Pillow's (W,H) representation to a (C,H,W) tensor.
    return torch.from_numpy(
        np.array(image, dtype=np.float32)
    ).unsqueeze(0) / 255.0


noisy_tensors, clean_tensors, heights = [], [], []

for noisy, clean in zip(noisy_imgs, clean_imgs):
    width, height = noisy.size
    if (width, height) not in {(540, 420), (540, 258)}:
        raise ValueError(f"Unexpected image size: {noisy.size}")

    x, y = to_tensor(noisy), to_tensor(clean)
    top = (420 - height) // 2
    bottom = 420 - height - top

    noisy_tensors.append(F.pad(x, (0, 0, top, bottom), value=1.0))
    clean_tensors.append(F.pad(y, (0, 0, top, bottom), value=1.0))
    heights.append(height)  # Preserve the order of filenames, inputs, and targets.

train_x = torch.stack(noisy_tensors, dim=0)
train_y = torch.stack(clean_tensors, dim=0)
heights = torch.tensor(heights, dtype=torch.int64)

assert train_x.shape == train_y.shape
print(tuple(train_x.shape))  # (144, 1, 420, 540) for this complete dataset
```

The code in this article preserves file order. Grouping images by size and joining the lists later is also possible, but filenames and original heights must be joined in exactly the same order. Matching tensor shapes do not guarantee that the metadata matches as well.

## 05. What stack, cat, and flatten change

These operations change different things. Start by checking whether you are working with Python lists or tensors.

| Operation | What it does | Example in this exercise |
| --- | --- | --- |
| List `a + b` | Joins two lists | Combines lists containing image tensors |
| `torch.stack` | Groups tensors along a new axis | Individual `(1,H,W)` images → `(N,1,H,W)` |
| `torch.cat` | Joins tensors along an existing axis | Several prediction batches → all predictions |
| `flatten(1)` | Combines axis 1 and all later axes | `(B,C,H,W)` → `(B,C×H×W)` |
| `mean` | Averages over a specified range | Computes one error value per image |

Individual image tensors do not have a batch axis yet. When `stack` groups them, the new axis is the image axis. Batch predictions already have that axis, so we combine them with `cat(..., dim=0)` along the existing axis.[^tensor-ops]

```python
# Group individual image tensors, each with shape (1,420,540).
batch = torch.stack(noisy_tensors[:3], dim=0)
print(tuple(batch.shape))  # (3,1,420,540)

# Join results that already have a batch axis.
joined = torch.cat([batch[:2], batch[2:]], dim=0)
print(tuple(joined.shape)) # (3,1,420,540)

# Flatten the shape; no average has been calculated yet.
flat = batch.flatten(1)
print(tuple(flat.shape))   # (3,226800)
```

A first-axis size of 144 in `train_x` does not mean that the training batch size is 144. If we take subsets of 24 images, the `B` passed to the model is 24.

## 06. Separate model construction from image input

Defining a model requires choosing its layers and their connections. Calling a model requires the actual input tensor.

`torch.nn` contains neural network layers and loss functions. With `from torch import nn`, we can write `nn.Conv2d` instead of `torch.nn.Conv2d`. A **class** defines how to create an object. In `CustomModel` below, `__init__` prepares the layers and `forward` specifies the sequence of calculations for an input.

```python
from torch import nn

class CustomModel(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

`nn.Sequential` runs its layers in order. `ReLU` replaces negative values with zero and leaves positive values unchanged. `Sigmoid` brings the output brightness into the same 0–1 range as the targets. Neither activation function changes the shape.[^activations]

The 16 in `CustomModel(channels=16)` is a **structural setting passed to the constructor**. The `x_batch` in `model(x_batch)` contains the **actual images to process**. A batch size of 24 is not a constructor argument to `Conv2d`.

| Stage | Output shape | Meaning of the values |
| --- | --- | --- |
| Input | `(B,1,420,540)` | Brightness of the noisy documents |
| Convolution and ReLU | `(B,16,210,270)` | 16 channels of features computed from the input |
| Transposed convolution and Sigmoid | `(B,1,420,540)` | Brightness of the restored documents |

A **feature map** contains features computed at spatial positions in the input. One convolutional filter processes all input channels to produce one output channel. This model's first layer computes 16 output channels from a single grayscale input channel. Its weights are shared across spatial positions, rather than created separately at every position.[^conv]

The intermediate tensor `z` is also called a **latent representation**. Halving its spatial dimensions does not necessarily reduce the total number of values. Height and width are each halved, but the channel count grows from 1 to 16. In this configuration, the intermediate tensor has four times as many elements as the input. Our goal here is document restoration; these shapes do not establish that the model has compressed the data for storage.

The parameter count is different from the number of intermediate values, too.

| Layer | Weight shape and axis meanings | Parameters including biases |
| --- | --- | ---: |
| `Conv2d` | `(16,1,3,3)`: output channels, input channels, kernel height, kernel width | `16×1×3×3 + 16 = 160` |
| `ConvTranspose2d` | `(16,1,4,4)`: input channels, output channels, kernel height, kernel width | `16×1×4×4 + 1 = 257` |
| Total | Trainable values stored in the two layers | **417** |

These weight shapes apply to the current `groups=1` configuration. Although both shapes begin with `(16,1)`, those axes have different meanings in the two layers.[^transpose]

## 07. Why the height shrinks and then grows again

The number of convolution output positions is determined by how many positions the kernel can occupy. Padding adds boundary space available to the computation. Increasing the **stride**, the distance between applications of the kernel, leaves fewer output positions.

A **transposed convolution** computes an output feature map from the input features and its own weights. It is not an inverse function that automatically recovers information lost by the encoder. Training against clean targets adjusts which values it should output.[^transpose]

Before checking the output shape, define the symbols. `L_in` and `L_out` are the input and output lengths along one direction; `k` is the kernel size in that direction, `s` is the stride, and `p` is the layer's `padding` setting. In convolution, this specifies padding added at the boundaries. In transposed convolution, it adjusts the output extent, so it should not be interpreted as adding the same kind of border to the input. The following formulas assume `dilation=1`, the spacing within the kernel, and `output_padding=0`, the output-size adjustment for transposed convolution.

<div class="document-math">
$$
L_{\mathrm{out,conv}}
=
\left\lfloor\frac{L_{\mathrm{in}}+2p-k}{s}\right\rfloor+1
$$
</div>

<div class="document-math">
$$
L_{\mathrm{out,transpose}}
=
(L_{\mathrm{in}}-1)s-2p+k
$$
</div>

`floor` means rounding down. Apply the formula separately to height and width. The encoder's `k=3,s=2,p=1` changes height 420 to 210 and width 540 to 270. The decoder's `k=4,s=2,p=1` brings these results back to 420 and 540.

The decoder receives height 210 from the encoder, rather than the original height of 420. Treating the original image as the direct input to the transposed convolution would lead to the wrong output-size prediction.

If you stack two downsampling layers, check whether an intermediate size becomes odd. Applying these types of settings twice can turn height 258 into `258 → 129 → 65 → 130 → 260`. This is why we check that the output still matches the target after adding layers. Also, `output_padding` should not be interpreted as a command to append zeros to the output array.

```python
shape_model = CustomModel(channels=16)
with torch.no_grad():
    sample = torch.zeros(2, 1, 420, 540)
    result = shape_model(sample)

assert result.shape == sample.shape
print(tuple(result.shape))  # (2,1,420,540)
print(sum(p.numel() for p in shape_model.parameters()))  # 417
```

This checks the output shape and parameter count. The model has not been trained yet, so the check says nothing about restoration quality.

## 08. Decide what to compare against

A **baseline** is a reference for judging a new model. It does not necessarily mean doing nothing or using the smallest possible model.

| Reference | How it is produced | Question it answers |
| --- | --- | --- |
| No-op baseline | Use the noisy input unchanged as the prediction | Does using the model improve on leaving the input as it is? |
| Starting model | Train a chosen architecture with chosen settings | Does changing the architecture or training method help? |
| Modified model | Change one condition and train again | What effect does the change have? |

The no-op baseline is simply `baseline_pred = x_val`; it needs no training. In contrast, an architecture with two encoder layers and two decoder layers, `1 → 32 → 64 → 32 → 1`, is a trained reference model. Both can be called a baseline, so specify which one you mean.

Use **the same validation images, the same original image areas, and the same metric aggregation** for comparisons. If one score includes padding and another excludes it, or one averages image scores while another pools all pixels, the difference reflects more than the model change alone.

## 09. K-Fold does not rotate the validation set of one continuing model

**K-fold cross-validation** divides the data into K groups and uses each group once for validation. In each round, a model is trained on the remaining groups. In the next round, a new model is created for a different training and validation split.[^kfold]

Splitting the 144 pairs in this exercise into six groups gives 24 pairs per group. Each round trains on 120 pairs and validates on the remaining 24.

<figure class="document-figure">
<img src="/document-denoising-kfold-kaggle/kfold-en.svg" alt="Of six data groups, A through F, one is used for validation in each round. Every round starts with a new model and optimizer, using 120 training pairs and 24 validation pairs." loading="lazy">
<figcaption>An image can be used for training in several rounds. Its validation prediction comes from the round whose model did not train on that image.</figcaption>
</figure>

While training a model within one round, keep the membership of its training and validation sets fixed. Repeating training does not move validation images into the training set.

| Unit | Meaning | Current setting |
| --- | --- | --- |
| Batch | A group passed to the model at once | 24 pairs |
| Epoch | One pass through the current training set | 120 pairs processed in five batches |
| Fold round | Training and evaluating a new model on one fixed split | Six rounds |

With one parameter update per batch, as in this code, a model receives 150 updates over 30 epochs. Its 24 validation pairs do not participate in those updates. Six folds and 30 epochs are separate settings.

The fact that 144 divides evenly by six makes the counts convenient to follow. It is not evidence that six-fold validation is statistically optimal. With five folds, validation sets can contain 28 or 29 images, and training can still handle a smaller final batch.

### KFold returns indices, not images

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=6, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_x), start=1):
    x_train = train_x[train_idx]
    y_train = train_y[train_idx]
    x_val = train_x[val_idx]
    y_val = train_y[val_idx]
    print(fold, len(train_idx), len(val_idx))
```

`train_idx` and `val_idx` are arrays of sample indices to select. Pairs stay together **because we use the same index array for inputs and targets**. `KFold` does not inspect filenames or document contents to find matching targets.

Ordinary `KFold` does not divide samples according to their target values, so we do not need to pass `train_y` to `split` here. `random_state` makes the split reproducible; it does not also fix the initial model weights or batch order.

`enumerate` adds a display number to each pair of returned arrays. We unpack that nested structure with `fold, (train_idx, val_idx)`. `fold` is a round number starting at 1, not an original image number.

We can also retrieve one split at a time with `next`.

```python
folds = kf.split(train_x)
first_train, first_val = next(folds)
second_train, second_val = next(folds)
```

Calling `next(kf.split(train_x))` afresh each time creates a new iterator. With the fixed integer seed used here, that repeatedly retrieves the first split.

## 10. Keep related versions of a document on the same side of the split

**Data leakage** occurs when information about the supposedly unseen evaluation target has entered training. Files with different names can still be modified versions of the same original document. Splitting those files between training and validation evaluates new files, but may not give a fair evaluation on new document content.

Ordinary `KFold` does not discover these relationships. If we can identify the source documents, we can assign the same group ID to samples from the same document and use **grouped cross-validation with GroupKFold**.[^groupfold]

```python
from sklearn.model_selection import GroupKFold

# groups[i] identifies the actual source document of train_x[i].
# Prepare this array by investigating the relationships before running the split.
splitter = GroupKFold(n_splits=6)
# for train_idx, val_idx in splitter.split(train_x, groups=groups):
#     ... train with each document group kept entirely on one side of the split.
```

Image sizes or arbitrary sequence numbers cannot substitute for genuine document group IDs. Six-fold validation requires at least six distinct groups. Depending on their sizes, the number of validation images can differ between rounds.

Even when training on **patches**, small areas cropped from an image, split the original documents first. Then extract patches from the training documents. Creating patches first and randomly splitting them can place parts of the same document on both sides.

The executable example in this article uses ordinary `KFold`. Its results should not be interpreted as an evaluation that has verified document relationships and applied grouped splitting.

## 11. Dataset retrieves pairs; DataLoader makes batches

`TensorDataset` retrieves data at the same index together. `DataLoader` selects samples from that dataset, optionally shuffles them, and groups them into batches. Both belong to `torch.utils.data`.[^data]

```python
from torch.utils.data import TensorDataset, DataLoader

train_idx, val_idx = next(kf.split(train_x))
x_val = train_x[val_idx]
y_val = train_y[val_idx]
val_heights = heights[val_idx]  # Select original heights from the same split.

dataset = TensorDataset(train_x[train_idx], train_y[train_idx])
loader = DataLoader(dataset, batch_size=24, shuffle=True)

one_x, one_y = dataset[0]
x_batch, y_batch = next(iter(loader))

print(tuple(one_x.shape))   # (1,420,540)
print(tuple(x_batch.shape)) # (24,1,420,540)
```

`TensorDataset` has not merged two images into one. `dataset[0]` returns a tuple containing one noisy image and its matching clean target. The tensors passed to it must have the same size along the first axis, but their remaining dimensions do not always have to match. For this restoration task, the input and target image shapes must match too.

`shuffle=True` changes the order in which samples are retrieved. It does not shuffle the input and target within a pair independently, nor does it bring validation data into training. With the default `drop_last=False`, a final batch smaller than the requested batch size is retained.

For this type of data, `list(zip(x,y))` can also serve as a dataset. `TensorDataset` is not the only option; it is a convenient way to retrieve corresponding elements by index when the tensors are already prepared.

### Choosing a device and moving data are separate steps

To use a GPU, move the model and the data being processed to the same device. NVIDIA GPUs can use CUDA, while Apple Silicon Macs can use MPS. The following code selects an available device and falls back to the CPU if neither is available.[^mps]

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = CustomModel(channels=16).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Choosing a device does not move all of `train_x` to the GPU. The complete dataset can stay on the CPU while we move only the current batch. Use the tensor returned by `to`.

```python
x_batch = x_batch.to(device)
y_batch = y_batch.to(device)

optimizer.zero_grad()         # Clear gradients left from the previous batch.
pred = model(x_batch)         # Predict with the current parameters.
assert pred.shape == y_batch.shape
loss = loss_fn(pred, y_batch) # Compare with the clean targets.
loss.backward()              # Compute gradients for the parameters.
optimizer.step()             # Update the parameters.
```

`zero_grad` does not reinitialize the parameters. `backward` computes gradients but does not apply parameter updates. The updates happen in `step`.[^training]

Create both a new model and a new optimizer for each Fold. Reusing an old optimizer with a new model can leave the optimizer pointing to the old model's parameters. On the other hand, creating a new model for every batch would discard what the preceding batches had learned.

| Code location | What belongs there |
| --- | --- |
| Inside the Fold loop, outside the epoch loop | Create a new model and optimizer |
| Inside the epoch loop | Repeat training on the same split with the same model |
| Inside the batch loop | Move inputs → predict → calculate loss → backpropagate → update |

`model.train()` puts layers whose behavior depends on mode into training mode. It does not perform training by itself. Likewise, `model.eval()` does not turn off automatic differentiation. For validation, use evaluation mode together with disabled gradient recording.[^module][^nograd]

```python
model.eval()
with torch.no_grad():
    pred = model(x_batch)
```

The current model has no Dropout or BatchNorm layers. Using this pattern still keeps training and validation properly separated if such layers are added later.

## 12. What does MSE average over?

**Mean squared error (MSE)** subtracts the target from the prediction at corresponding positions, squares each difference, and averages those squared differences over a specified range.

Adding the signed differences directly could let errors that are too bright cancel errors that are too dark. Squaring makes both directions contribute nonnegative values. It also gives larger errors more weight relative to smaller ones. If a brightness error is below 1 in magnitude, squaring can make the number itself smaller, so it is inaccurate to say that squaring always increases a value.

Calculating the squared differences alone preserves the shape.

```python
error = (pred - y_batch) ** 2
print(tuple(error.shape))  # (B,1,420,540)
```

The `reduction` argument of `nn.MSELoss` specifies how to combine the results. Set it when creating the loss object. Its default is the string `"mean"`.[^mse]

| Setting | Result | Shape |
| --- | --- | --- |
| `reduction="none"` | Squared error at every position | `(B,C,H,W)` |
| `reduction="mean"` | Mean squared error over all elements | `()` |
| `reduction="sum"` | Sum of squared errors over all elements | `()` |

`()` is a scalar tensor with no axes. It differs from `(1,)`, which has one axis of size 1.

```python
loss_fn_each = nn.MSELoss(reduction="none")
error = loss_fn_each(pred, y_batch)
mse_each = error.mean(dim=(1, 2, 3))

print(tuple(mse_each.shape)) # (B,)
```

`mean(dim=(1,2,3))` averages over channels, height, and width, leaving image axis 0 intact. It combines the errors within each image into one score, without yet combining scores from different images. A batch of `B` images produces `B` scores.

`error.flatten(1).mean(dim=1)` averages over the same range. `flatten` arranges each image's pixels along one axis; the following `mean` performs the averaging.

Do not call it as `nn.MSELoss()(pred - y_batch, reduction="none")`. Pass predictions and targets separately, and set `reduction` when creating the loss object.

Using the default `MSELoss` for training produces one loss value for the entire batch. Automatic differentiation tracks the operations that produced that value. Having a single loss does not prevent it from computing a separate gradient for each parameter.

## 13. Why changing the order of RMSE and averaging changes the result

**Root mean squared error (RMSE)** is the square root of MSE. A smaller value means less error between prediction and target.

After adding and averaging squared errors, the result is still on a squared-brightness-error scale. Taking the square root brings it back to the original brightness-error scale. It does not recover each pixel's original difference, because those differences have already been summarized. **Returning to the original scale is different from recovering the individual values.**

```python
rmse_each = mse_each.sqrt()  # One RMSE value per image
mean_image_rmse = rmse_each.mean().item()
```

The important question is when the square root was applied.

### Averaging MSE values from equally sized images

Suppose every image contains the same number of pixels. Within an image, all pixels contribute equally to its mean. Each image mean also contributes equally to the final mean. Since the images have equal pixel counts, every pixel ultimately receives the same weight.

We can also see this by considering what stays unchanged. Replacing every squared error in an image with that image's mean preserves its sum and element count. Doing this for every image preserves the overall sum and overall count as well. With equally sized images, each image mean is repeated the same number of times, so averaging those means equally gives the same result.

Under this condition, **pooled pixel MSE equals the mean of the per-image MSE values**. If image sizes differ, we must account for the number of pixels in the larger images; an unweighted mean is no longer interchangeable.

### The square root cannot freely change places with the mean

Taking the square root of each image's MSE transforms large and small image errors separately before their scores are combined. Calculating the overall MSE first allows the large squared errors to enter the combined value before one square root is applied at the end.

The square root does not reduce every input by the same factor. The ratio between large and small values changes after the transformation. Transforming each image's value first and transforming the combined value afterward are different calculations.

<figure class="document-figure">
<img src="/document-denoising-kfold-kaggle/metrics-en.svg" alt="Two aggregation paths start from squared errors in the original image areas. One averages within each image, takes a square root for each image, and then averages the image scores. The other divides the total squared error by the total pixel count and takes one square root." loading="lazy">
<figcaption>Both scores are valid definitions. Name the evaluation scope and calculation order, and compare scores calculated in the same way.</figcaption>
</figure>

For equally sized images, the difference in code is:

```python
# Finish each image's score, then average with equal weight per image.
score_a = ((pred - y_batch) ** 2).mean(dim=(1, 2, 3)).sqrt().mean()

# Combine squared errors over all pixels, then take one square root.
score_b = nn.MSELoss()(pred, y_batch).sqrt()
```

Under this condition, `score_b` is at least as large as `score_a`. They are equal when all images have the same MSE. With different original image sizes, pooled pixel scoring introduces another difference: larger images receive more weight.

Consider a small example to check the calculation. Assume that two equally sized images have MSE values of 0.01 and 0.09. These are chosen values for illustrating aggregation, not measured results from the exercise. Their RMSE values are 0.1 and 0.3, whose mean is 0.2. In contrast, taking the square root of their mean MSE, 0.05, gives about 0.223607.

```python
example_mse = torch.tensor([0.01, 0.09])
print(example_mse.sqrt().mean().item()) # About 0.200000
print(example_mse.mean().sqrt().item()) # About 0.223607
```

### Check what is being averaged across Folds, too

If every validation set contains 24 images and each image appears in validation once, the average of six Folds' **mean per-image RMSE values** equals the mean of all 144 per-image RMSE values. Every group has the same size.

However, this is not one model's score on 144 images. It combines predictions made by six different models on their own validation images. To support splits with different validation counts, it is safer to collect all per-image results and average them. The complete executable example uses this approach.

## 14. PSNR compares MSE with the brightness range

**Peak signal-to-noise ratio (PSNR)** expresses the ratio between a fixed brightness range and the reconstruction error on a logarithmic scale. For the same brightness range, a smaller MSE gives a larger PSNR. It does not detect a separate kind of error in character strokes or shapes.[^psnr]

A **decibel (dB)** expresses a ratio logarithmically. It is not an accuracy percentage. A logarithm asks which exponent of a given base produces a value. PSNR uses a base-10 logarithm.

In the formula, `R` is the width of the brightness range used for evaluation. It is 1 for the current 0–1 data and 255 for 0–255 data. It is not recomputed from the largest value observed in each image.

<div class="document-math">
$$
\mathrm{PSNR}=10\log_{10}\left(\frac{R^2}{\mathrm{MSE}}\right)
$$
</div>

Because MSE is a squared error, the brightness range is squared for the comparison too. Substituting the current `R=1` gives `10 log10(1/MSE)`. A logarithm of a quotient is the difference between two logarithms, and `log10(1)=0`, giving:

<div class="document-math">
$$
10\log_{10}(1/\mathrm{MSE})
=10\{\log_{10}(1)-\log_{10}(\mathrm{MSE})\}
=-10\log_{10}(\mathrm{MSE})
$$
</div>

```python
psnr_each = -10 * torch.log10(mse_each)
mean_image_psnr = psnr_each.mean().item()
```

`torch.log` is the natural logarithm. Use `torch.log10` for PSNR's base-10 logarithm. If `mse_each` has shape `(B,)`, so does `psnr_each`; average it before extracting a single value with `item()`.

If MSE is exactly zero, PSNR is positive infinity, `inf`. Even one perfectly matching image can make the simple mean PSNR infinite. Applying `clamp_min(1e-12)` avoids this by capping PSNR at 120dB for the current range. That adds a numerical rule and must be disclosed. The executable example does not silently discard infinite values.

For one image, a lower RMSE gives a higher PSNR. Across multiple images, **mean RMSE** and **mean PSNR** average scores after different transformations, so they need not give corresponding model rankings. Converting mean RMSE to PSNR does not produce mean per-image PSNR.

## 15. Remove artificial white borders for evaluation

The white background of a real document is part of the restoration target. White padding added to match sizes is an area that never belonged to the document. Keep the two separate.

If identical white padding is added to the noisy input and clean target, the no-op baseline has zero error in the padding. Including those zeros in the average lowers its score relative to the error on the actual document. A model may not output perfectly white padding, so its score need not change in the same direction. The problem is that artificial regions are influencing the evaluation.

For an image originally 258 pixels high, the original area is `81:339`. A Python slice includes its start and excludes its stop. After skipping the top 81 rows, the original first row is at index 81. The position immediately after its 258 original rows is 339, so the final included index is 338.

The following function receives predictions, targets, and original heights, and returns a **tensor of per-image MSE values with shape `(B,)`**. It assumes the preprocessing used here: no horizontal padding and a vertically centered original image.

```python
def per_image_mse(pred, target, original_heights):
    assert pred.shape == target.shape
    assert len(pred) == len(original_heights)
    values = []

    for i, height in enumerate(original_heights):
        height = int(height)
        assert height in (258, 420)
        top = (pred.shape[-2] - height) // 2
        stop = top + height

        restored = pred[i, :, top:stop, :]
        clean = target[i, :, top:stop, :]
        values.append(((restored - clean) ** 2).mean())

    return torch.stack(values)
```

`values` contains one scalar tensor per image. `stack` creates an image axis and combines them into shape `(B,)`. Evaluate the no-op baseline over the same area with `per_image_mse(x_val,y_val,val_heights)`.

When original sizes differ, pooled pixel RMSE requires collecting the **sum of squared errors and the actual element count**.

```python
def pooled_pixel_rmse(pred, target, original_heights):
    assert pred.shape == target.shape
    assert len(pred) == len(original_heights)
    squared_error_sum = 0.0
    pixel_count = 0

    for i, height in enumerate(original_heights):
        height = int(height)
        top = (pred.shape[-2] - height) // 2
        error = (
            pred[i, :, top:top + height, :]
            - target[i, :, top:top + height, :]
        ) ** 2

        squared_error_sum += error.sum().item()
        pixel_count += error.numel()

    return (squared_error_sum / pixel_count) ** 0.5
```

`numel()` gives the total number of elements in a tensor. Averaging MSE values from differently sized images equally gives each image the same weight. The function above gives each pixel the same weight, so the two calculations must be distinguished.

The training example uses ordinary `MSELoss`, which includes padding, while evaluation uses only the original areas. To exclude padding from training as well, use a **mask** that includes only original positions, or calculate the loss over the same cropped areas. This is different from ignoring every white background pixel in the real documents.

**Do not diagnose overfitting by directly comparing training MSE that includes padding with validation RMSE that excludes it.** To assess whether performance is worse on unseen data than on training data, predict the training images again in evaluation mode and measure both sets over the same area with the same metric.

## 16. Connect training and cross-validation in code

We can now connect `train_x`, `train_y`, `heights`, `CustomModel`, and `per_image_mse` from the earlier sections. First, put the training of one model into a function.

`train_model` receives training inputs, targets, and a device. It creates batches, **creates a new model and optimizer**, and trains for the chosen number of epochs. It returns the trained model and the mean training loss from the last epoch. Grouping these steps makes it easier to restart the same procedure for each Fold.

```python
def train_model(x, y, device, *, channels=16, epochs=30,
                batch_size=24, lr=0.001, seed=42):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    dataset = TensorDataset(x, y)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=0,
    )

    model = CustomModel(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # The training loss includes padding.

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            assert pred.shape == y_batch.shape
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            # Weight by image count so a smaller final batch is handled correctly.
            loss_sum += loss.item() * len(x_batch)

        train_mse = loss_sum / len(dataset)

    return model, train_mse
```

Defining the function does not run training. Training for a Fold begins when the loop below calls `train_model(...)`. Pass an integer of at least 1 for `epochs`.

The split seed determines which images belong to each Fold. The model and DataLoader seeds control initial parameters and batch order. Recording seeds helps comparisons, but the same seed does not guarantee identical results across devices or library versions.[^randomness]

Store validation results at their original image indices. This lets us average one result per image even if validation counts differ between Folds.

```python
kf = KFold(n_splits=6, shuffle=True, random_state=42)
batch_size = 24
epochs = 30

# Store per-image validation results on the CPU.
oof_model_mse = torch.empty(len(train_x), dtype=torch.float64)
oof_base_mse = torch.empty(len(train_x), dtype=torch.float64)
seen = torch.zeros(len(train_x), dtype=torch.int64)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_x), start=1):
    model, train_mse = train_model(
        train_x[train_idx], train_y[train_idx], device,
        channels=16, epochs=epochs, batch_size=batch_size, seed=42 + fold,
    )

    model.eval()
    with torch.no_grad():
        for start in range(0, len(val_idx), batch_size):
            idx = val_idx[start:start + batch_size]
            x_val = train_x[idx].to(device)
            y_val = train_y[idx].to(device)
            val_heights = [int(heights[int(i)]) for i in idx]

            pred = model(x_val)
            model_mse = per_image_mse(pred, y_val, val_heights)
            base_mse = per_image_mse(x_val, y_val, val_heights)

            oof_model_mse[idx] = model_mse.cpu().double()
            oof_base_mse[idx] = base_mse.cpu().double()
            seen[idx] += 1

    fold_score = oof_model_mse[val_idx].sqrt().mean().item()
    print(f"Fold {fold} | Training MSE {train_mse:.6f} | Validation RMSE {fold_score:.6f}")

assert bool((seen == 1).all())  # Check that every image was validated exactly once.

print("Mean per-image model RMSE:", oof_model_mse.sqrt().mean().item())
print("Mean per-image no-op RMSE:", oof_base_mse.sqrt().mean().item())
print("Mean per-image model PSNR:", (-10 * torch.log10(oof_model_mse)).mean().item())

# Weight by original pixel counts to obtain pooled RMSE over original pixels.
pixel_counts = torch.as_tensor(heights, dtype=torch.float64) * train_x.shape[-1]
pooled_rmse = (
    (oof_model_mse * pixel_counts).sum() / pixel_counts.sum()
).sqrt().item()
print("Pooled original-pixel model RMSE:", pooled_rmse)
```

**Out-of-fold (OOF)** results are obtained for each sample from the Fold model that did not train on it. `oof_model_mse` stores these per-image MSE values in their original order. The downloadable program stores each image's sum of squared errors and pixel count directly, then calculates all three metrics for both the model and no-op baseline.

The logged `train_mse` above is the mean of batch losses recorded during the epoch. Because the parameters change after every batch, it is not an evaluation of all batches using one fixed set of final parameters. Use the separately calculated validation metrics to compare model performance.

When the loop ends, `model` contains **the last Fold's model**. Earlier models have not been stored in the score arrays. To keep a trained model, save its weights together with its split information. The downloadable program saves each Fold's checkpoint and filename lists.

## 17. Check the device and axes before displaying images

An improved score can still hide blurred strokes or missing small characters. Showing the input, restoration, and target side by side with the same brightness range helps reveal those failures.

Three commonly used methods do different jobs before display.

| Method | What it does | What it does not do |
| --- | --- | --- |
| `detach()` | Returns a tensor detached from automatic differentiation | Does not move it to the CPU |
| `cpu()` | Returns a CPU tensor | Does not turn it into a Python number |
| `item()` | Extracts a Python number from a one-element tensor | Does not automatically average multiple scores |

For the usual NumPy conversion, use a CPU tensor that does not require gradient tracking. `tensor.detach().cpu().numpy()` is a convenient display pattern. A detached tensor can share storage with the original, so detaching is not the same as making an independent copy.[^numpy]

The next example uses **the last Fold model from the previous code and that Fold's validation indices**. It does not reevaluate all six Folds with one model.

A **Figure** is the whole figure, and an **Axes** is an individual plotting area within it. `plt.subplots` creates both. The row and column in `axes[row,col]` identify positions in the figure layout, not pixel coordinates inside an image.[^subplots]

```python
import random
import matplotlib.pyplot as plt

n_show = min(5, len(val_idx))
selected = random.Random(42).sample(list(map(int, val_idx)), n_show)

x_show = train_x[selected].to(device)
y_show = train_y[selected].to(device)
show_heights = [int(heights[i]) for i in selected]

model.eval()
with torch.no_grad():
    pred_show = model(x_show)

fig, axes = plt.subplots(
    n_show, 3, figsize=(15, 3 * n_show), squeeze=False,
)

for row, image_id in enumerate(selected):
    height = show_heights[row]
    top = (x_show.shape[-2] - height) // 2
    stop = top + height

    images = [
        x_show[row, 0, top:stop, :],
        pred_show[row, 0, top:stop, :],
        y_show[row, 0, top:stop, :],
    ]

    for col, (image, title) in enumerate(
        zip(images, ["Noisy", "Restored", "Clean"])
    ):
        axes[row, col].imshow(
            image.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1,
        )
        axes[row, col].set_title(f"{title} — {names[image_id]}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
```

`row` is the row number in the display. `image_id` is the image's index in the complete dataset. Keep them separate: use `row` to index the selected small batch and the original index to look up its filename.

Selecting the grayscale channel with `x_show[row,0]` produces the `(H,W)` array expected by `imshow`. `vmin=0,vmax=1` fixes the brightness display scale for all three images. It does not renormalize the underlying pixel values. Without a shared scale, each image may be displayed over a different range, which can make a faint output appear more contrasted than it really is.[^imshow]

`squeeze=False` preserves the `axes[row,column]` structure even when displaying one image. When increasing the number of selected images, change both `n_show` and the number of figure rows together.

To inspect differences between image scores, plot the complete OOF results.

```python
model_psnr = -10 * torch.log10(oof_model_mse)
base_psnr = -10 * torch.log10(oof_base_mse)

plt.figure(figsize=(12, 4))
plt.plot(model_psnr.cpu().numpy(), label="Model")
plt.plot(base_psnr.cpu().numpy(), label="No-op")
plt.xlabel("Image index in the stored file order")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.tight_layout()
plt.show()
```

The horizontal axis is the index in the stored file order. It does not represent time or the numeric value of a filename. If a score is `inf`, the graph may not display it at a finite position. Check whether that image's MSE is zero instead of relying on the graph alone.

## 18. Change one condition at a time when improving the model

If a restored document looks blurry, changing channels, layers, loss, and learning rate all at once makes it hard to tell which change helped. First check input–target pairs, output shapes, whether training loss decreases, scores against the no-op baseline, and whether characters are preserved. If the model cannot sufficiently reduce the loss on even one batch, inspect the data and training connections first. That check alone still cannot identify a single cause.

Distinguish the kinds of changes you can make.

| Change | What it changes | What to check |
| --- | --- | --- |
| More channels | Feature capacity at the same stage | Additional compute and memory, and whether validation improves |
| More layers | Processing features through more stages | Output size, training stability, and use of surrounding regions |
| Skip connections | Paths that pass earlier features into later computations | Whether position information and noise are both passed along |
| Different loss | Which errors matter and how strongly they are penalized | Alignment with the competition metric and character preservation |
| Patch training | The document area seen at once | Matching input–target coordinates and splitting by source document |

A **skip connection** sends earlier features directly to a later layer. When combining by addition, the two tensor shapes must match. When concatenating along channels, batch size, height, and width must match, and the next layer's input channel count changes. Adding a connection does not by itself guarantee better restoration.

More layers do not always require smaller spatial dimensions. Suitable stride and padding choices can preserve the spatial size while processing features further.

Blurry gray output can have several causes. If the model cannot adequately distinguish text from background, an intermediate brightness can reduce squared error. Experiments are needed to establish whether the actual cause is MSE, insufficient training, model capacity, information loss, or incorrectly paired targets.

**L1 loss** uses absolute differences, while a weighted loss assigns more importance to errors in selected regions. **Structural similarity (SSIM)** compares local brightness, contrast, and structure. These criteria can also be incorporated into losses, but it is more efficient to first confirm that the current code trains and evaluates correctly.[^l1][^psnr]

The background of a clean document really is white. Penalizing the model simply for restoring a white background would conflict with the restoration goal. To address disappearing characters, evaluate both text preservation and background noise removal.

For `128×128` patches, choose the crop coordinates once and apply them to both the noisy image and target. Independent random crops would train on different locations as input and target. Patches preserve the original pixel scale while providing uniform batch shapes, but they do not expose every character or the full document context at once.

Keep the split and evaluation procedure fixed, and change one condition at a time. After changing the model, check **which class is actually instantiated inside the Fold loop**. Creating a new model outside the loop does not help if the loop then creates and trains the previous class again.

## 19. Train a separate submission model after cross-validation

Cross-validation compares settings. The last Fold model trained on 120 pairs; completing all six Folds does not mean that this model has trained on all 144 pairs.

After choosing the architecture, epoch count, learning rate, and other settings, we can train a new final model using the complete training dataset. The earlier `train_model` function creates a new model on every call, so we can use it as follows.

```python
final_model, final_train_mse = train_model(
    train_x, train_y, device,
    channels=16, epochs=30, batch_size=24, lr=0.001, seed=42,
)

torch.save(
    {
        "state_dict": final_model.state_dict(),
        "channels": 16,
        "epochs": 30,
        "batch_size": 24,
        "learning_rate": 0.001,
        "seed": 42,
        "preprocessing": "L / 255; white vertical padding to (420,540)",
        "train_names": names,
    },
    "final_model.pt",
)
```

`state_dict` contains the model's parameters and registered buffers. Record the architecture, preprocessing, and training settings too, so the same computation can be reconstructed. The file above is intended for reusing the trained model in inference. To resume interrupted training with the same optimizer state, save that state and other necessary training information as well.[^saving]

`test` has no clean targets. The difference between a restoration and its noisy input only shows how much the input changed; it does not reveal how close the output is to the clean target. Local numerical comparisons of restoration quality use validation data with known targets.

Once the final model has trained on the complete training set, a score on those images is not an independent validation score. The model has already learned from them.

## 20. Preserve file order and original sizes during test inference

Apply the same grayscale conversion, normalization, and padding to test images as to training images. The difference is that test images have no targets. Original sizes and filenames are needed for submission, so keep them in the same order as the predictions.

```python
test_paths = sorted((data_dir / "test").glob("*.png"))
assert test_paths, "No PNG files were found in the test directory."
test_tensors, test_sizes = [], []

for path in test_paths:
    with Image.open(path) as opened:
        image = opened.convert("L")

    width, height = image.size
    if (width, height) not in {(540, 420), (540, 258)}:
        raise ValueError(f"Unexpected image size: {image.size}")

    x = to_tensor(image)
    top = (420 - height) // 2
    x = F.pad(x, (0, 0, top, top), value=1.0)

    test_tensors.append(x)
    test_sizes.append((width, height))

test_x = torch.stack(test_tensors)
print(tuple(test_x.shape))  # (72,1,420,540) for the complete exercise dataset
```

Batch inference reduces the amount of data that must be placed on the GPU at once.

```python
final_model.eval()
predictions = []

with torch.no_grad():
    for start in range(0, len(test_x), batch_size):
        batch = test_x[start:start + batch_size].to(device)
        pred_batch = final_model(batch)
        predictions.append(pred_batch.cpu())

test_pred = torch.cat(predictions, dim=0)
assert test_pred.shape == test_x.shape
```

Each `pred_batch` already has a batch axis, so use `cat`. Collecting results on the CPU avoids keeping every prediction on the GPU while preparing the submission file. The downloadable program goes a step further and writes each batch to the compressed file after moving it to the CPU.

## 21. Submit pixel predictions to Kaggle, not scores

Kaggle compares submitted pixel intensities with its private targets to calculate RMSE. We do not submit the RMSE or PSNR values calculated locally. The competition format is `id,value`, where `id` means **image number_row number_column number**. Intensities range from black at 0 to white at 1.[^kaggle]

```csv
id,value
1_1_1,1.0
1_1_2,0.98
```

These numbers only illustrate the format. Obtain an image number from its original filename, not its position in an array. For `test/12.png`, the image number is 12. Python tensor indices start at 0, while submission row and column numbers start at 1.

`test_pred` contains predictions padded to height 420. For images originally 258 pixels high, remove the top and bottom padding first. The following code uses `test_paths`, `test_sizes`, and `test_pred` prepared above.

```python
import csv
import io
from zipfile import ZipFile, ZIP_DEFLATED

assert len(test_paths) == len(test_sizes) == len(test_pred)
row_count = 0

with ZipFile("submission.csv.zip", "w", compression=ZIP_DEFLATED) as archive:
    with archive.open("submission.csv", "w") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "value"])

            for i, path in enumerate(test_paths):
                width, height = test_sizes[i]
                image = test_pred[i, 0].detach().cpu().numpy()

                top = (image.shape[0] - height) // 2
                image = image[top:top + height, :]

                assert image.shape == (height, width)
                assert np.isfinite(image).all()
                assert ((image >= 0) & (image <= 1)).all()

                for row, pixels in enumerate(image, start=1):
                    for col, value in enumerate(pixels, start=1):
                        writer.writerow([
                            f"{path.stem}_{row}_{col}", float(value)
                        ])
                        row_count += 1

expected_rows = sum(width * height for width, height in test_sizes)
assert row_count == expected_rows
print("Pixel rows written:", row_count)
```

`row_count` counts prediction rows, excluding the header. It must equal the total number of original image pixels. A matching row count alone does not prove that all coordinates are correct, so check the first and last filename, row, and column values as well.

Do not automatically clip values to conceal `NaN` or out-of-range output. Normal finite output from the current Sigmoid model lies in 0–1. If a check fails, inspect the model output and preprocessing first.

### Terminal and Jupyter commands use different prefixes

A **command-line interface (CLI)** lets us operate a program by entering commands in a terminal. The official Kaggle documentation checked on November 7, 2026 provides the following submission and lookup commands.[^kaggle-cli]

Run these directly in zsh on macOS, or bash or zsh on Linux. The path refers to the file created by the code above.

```bash
kaggle competitions submit denoising-dirty-documents -f submission.csv.zip -m "Document restoration"
kaggle competitions submissions denoising-dirty-documents
```

To inspect a particular submission, append its newly issued submission number to `kaggle competitions submission`. Reusing an older submission number retrieves that old record, not the new experiment.

Prefix shell commands with `!` when running them in Jupyter. This differs from `%pip`, which installs packages into the current kernel's environment.

```python
# These two lines are Jupyter cell commands, not Python script statements.
# %pip install -U kaggle
# !kaggle competitions submissions denoising-dirty-documents
```

If authentication is required, run `kaggle auth login` in a terminal and check that an actual API request, such as `kaggle quota`, succeeds. `quota` retrieves GPU and TPU usage allowances. Seeing a login message and successfully authenticating an API request are different observations.[^kaggle-auth]

During the exercise, one environment contained both an existing `kaggle.json` and OAuth credentials. It reported that the user was already logged in while API requests still failed authentication. The records show that temporarily keeping `kaggle.json` under a different name allowed the request to succeed. This establishes that excluding that file resolved the request in that environment. It does not establish a universal cause of authentication errors or explain exactly why the previous key was invalid. Do not place credential file contents in notebook output or a public repository.

An upload-success message is not a completed score. Check for submission status `COMPLETE`; if the status is `ERROR`, inspect the error details. Follow the competition's current participation requirements and submission availability at the time of submission.

## 22. Keep conclusions within what the recorded scores show

The values below come from earlier practice records. They were not reproduced by training this article's example on the original dataset.

| Record | Score | Conditions to check before interpreting it |
| --- | ---: | --- |
| Initial model, 1 epoch | Local mean RMSE about 0.45 | Architecture, split, and evaluation area at the time |
| Initial model, 30 epochs | Local mean RMSE about 0.22694 | Whether settings matched the preceding experiment |
| No-op baseline | Local mean RMSE about 0.1437 | Whether validation images and aggregation matched the model |
| Kaggle submission `58090430` | Public and Private both 0.09623, COMPLETE | Submitted model, test data, and pixel aggregation |

If a local comparison used **identical evaluation conditions** and the model's RMSE was larger than the no-op baseline's, we can say that applying the model made the result worse in that experiment.

However, Kaggle's 0.09623 being smaller than the local no-op score of 0.1437 does not by itself demonstrate improvement over doing nothing. The data differs, and so may padding treatment and image-level versus pixel-level aggregation. We also need to check whether the model changed.

It is also incorrect to turn an RMSE of 0.09623 into “90.377% accuracy.” RMSE summarizes brightness error; it is not the fraction of correct answers.

To compare experiments later, record the architecture, split filenames, epoch count, batch size, learning rate, seed, preprocessing, evaluation area, aggregation method, model file, and submission number together. Repeatedly inspecting competition scores and changing settings uses those scores for model selection too. Keep model selection centered on a separately defined local validation procedure as far as possible.[^cross-validation]

## 23. Run the example and check common points of confusion

Download the [English Python program](/document-denoising-kfold-kaggle/examples-en.py) and run it in a virtual environment with PyTorch installed. It uses `torch`, `numpy`, `Pillow`, and `scikit-learn`. The visualization code in the article also requires `matplotlib`. Choose a PyTorch installation appropriate for your CPU, CUDA, or MPS environment.

If the data directory is `./data`, it must contain `train`, `train_cleaned`, and `test`. The commands below are for a macOS or Linux terminal, assuming that `python` points to the Python interpreter in the intended virtual environment.

```bash
python examples-en.py --mode check
python examples-en.py --mode cv --data-dir ./data --output-dir ./runs
python examples-en.py --mode final --data-dir ./data --output-dir ./runs
```

| Mode | What it does | Output |
| --- | --- | --- |
| `check` | Checks core operations and submission formatting with synthetic data | Check results |
| `cv` | Trains a new model for each Fold and validates on original image areas | Files such as `fold-1.pt`, plus `cv-summary.csv` and `oof.csv` |
| `final` | Trains a new model on the complete training set and runs test inference | `final.pt` and `submission.csv.zip` |

Defaults are six folds, 30 epochs, batch size 24, 16 channels, learning rate 0.001, and seed 42. Change them with `--epochs`, `--batch-size`, `--channels`, `--folds`, `--lr`, and `--seed`. The device is selected automatically, or you can specify `--device cpu`, `--device mps`, or `--device cuda`. Reusing an output directory overwrites result files with the same names, so keep separate directories for separate experiments.

`final` creates the submission file. Uploading it to Kaggle is a separate CLI operation. With the output directory used above, the file is `./runs/submission.csv.zip`.

Common points of confusion from the exercise can be checked as follows.

| Intention | Code to check |
| --- | --- |
| NumPy floating-point type | `np.float32` |
| Add an item to a list | `.append(value)` |
| Number of cross-validation splits | `n_splits` |
| Adam optimizer | `torch.optim.Adam` |
| Repeat a fixed number of times | `for epoch in range(epochs)` |
| Check MPS availability | `torch.backends.mps.is_available()` |
| Inspect a tensor's shape | `x.shape` or `x.size()` |
| Average all MSE elements | `nn.MSELoss(reduction="mean")` |
| Print multiple scores as one number | Aggregate first, then call `.item()` |
| Mean of a Python list | `sum(values) / len(values)` |
| Base-10 logarithm | `torch.log10` |
| Create several plotting areas | `plt.subplots` |
| Select image indices without replacement | `random.sample(range(n), k)` |
| Display GPU output as an ordinary array | `tensor.detach().cpu().numpy()` |

Check **what a variable currently contains**, as well as spelling. `scores` contains scores, not models. `clean_paths` contains paths, not target pixels. Overwriting `val_idx` in another loop can make it stop matching the validation split of the model you currently have.

### Corresponding concepts in Rust

The neural network training here is implemented with Python and PyTorch. Basic tasks such as managing paths and lists have counterparts in Rust.

| Python | Rust | Shared role |
| --- | --- | --- |
| `/` on `Path` | `PathBuf::join` | Join path components |
| `list.append` | `Vec::push` | Add a value to a list |
| `dict` | `HashMap` | Store values by key |
| `zip` | `Iterator::zip` | Pair elements from two iterators in order |
| NumPy `float32` | `f32` | Represent a 32-bit floating-point number |

```rust
use std::{collections::HashMap, env, io};

fn main() -> io::Result<()> {
    // Construct a path without creating a directory or reading an image.
    let train_dir = env::current_dir()?.join("train");
    println!("{}", train_dir.display());

    // Each key is (width, height); its value is the number of images of that size.
    let sizes = [(540, 420), (540, 258), (540, 420)];
    let mut counts = HashMap::new();

    for size in sizes {
        *counts.entry(size).or_insert(0usize) += 1;
    }

    // Look up specific keys instead of relying on HashMap iteration order.
    println!("420 rows: {}", counts[&(540, 420)]); // 2
    println!("258 rows: {}", counts[&(540, 258)]); // 1
    Ok(())
}
```

This Rust code demonstrates paths and size counts only. A `Vec` is not itself an automatic-differentiation tensor, and this example does not train a neural network. Remember that `zip` stops at the shorter input by default in both languages.[^rust]

## 24. What to learn now, and what was verified

The first skill to build in this exercise is **preserving data correspondence throughout the process and evaluating performance consistently**, before moving on to complex architectures.

| Priority | What to learn here |
| --- | --- |
| Essential in practice | Filename pairing, axes and shapes, a new model per Fold, matching devices, original-area evaluation, no-op comparisons, submission coordinates and file order |
| Understand the principle | Channels versus layers, the role of transposed convolution, conditions for averaging, RMSE and PSNR transformations and aggregation |
| Extend afterward | Investigating document groups, better patch sampling, skip connections, different losses, combining predictions from several Fold models |

If related versions of the same original document are found in the actual data, grouped splitting becomes a necessary correction to the evaluation, rather than an optional later optimization.

You have reached the goal for this stage when you can answer these questions and run the code without referring back to the explanations.

- In which calculation is the clean target used?
- When is a model's training and validation membership chosen, and when does it change?
- Which axes does `mean` reduce, and which axes remain?
- Does the current score give equal weight to images or to pixels?
- Which exact padding positions are excluded from evaluation and submission?
- Is the model now used for testing the last Fold model, or a new model trained on the complete dataset?

Then implement one small change, run it on the same splits, and compare both numerical scores and restored images. Completing that experiment is more useful for the current goal than continuing to expand the mathematical detail.

The executable program's core operations were checked with **PyTorch 2.15.0+cpu**. Seven synthetic PNG pairs were used to exercise file loading, three Folds with different validation counts, model saving, retraining on the complete dataset, and submission creation for two test images. Checks covered all 366,120 pixel rows in the submission, restoration of original sizes, filename and coordinate correspondence, and matching predictions after reloading the saved model. These checks verify that the code connections and aggregation behave as intended. They do not reproduce restoration quality on the original Kaggle data or the earlier competition scores.

## References

Official API and Kaggle CLI documentation was checked on November 7, 2026. Dataset counts and earlier scores are taken from the supplied practice records.

[^pathlib]: Python, [pathlib](https://docs.python.org/3/library/pathlib.html) — joining paths and finding files.
[^pillow]: Pillow, [Image.open and Image.convert](https://pillow.readthedocs.io/en/stable/reference/Image.html) — reading images and grayscale mode.
[^builtins]: Python, [built-in functions](https://docs.python.org/3/library/functions.html#zip) — zip, enumerate, and next.
[^tensor-ops]: PyTorch, [unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html), [stack](https://docs.pytorch.org/docs/stable/generated/torch.stack.html), [cat](https://docs.pytorch.org/docs/stable/generated/torch.cat.html), [flatten](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html) — adding, joining, and flattening axes.
[^pad]: PyTorch, [functional.pad](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html) — padding order and fill values.
[^activations]: PyTorch, [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [Sigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html) — activation functions.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — arguments, output shapes, and weight shapes.
[^transpose]: PyTorch, [ConvTranspose2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) — the operation and its output shape.
[^kfold]: scikit-learn, [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) — split indices and Fold sizes.
[^groupfold]: scikit-learn, [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) — splits with non-overlapping groups.
[^data]: PyTorch, [Dataset and DataLoader](https://docs.pytorch.org/docs/stable/data.html) — samples, batches, and shuffling.
[^mps]: PyTorch, [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html), [CUDA](https://docs.pytorch.org/docs/stable/notes/cuda.html) — using devices.
[^training]: PyTorch, [training loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — loss, backpropagation, and parameter updates.
[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) — model structure and training and evaluation modes.
[^nograd]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) — disabling gradient recording.
[^mse]: PyTorch, [MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) — elementwise squared errors and reduction.
[^psnr]: scikit-image, [PSNR and SSIM](https://scikit-image.org/docs/stable/api/skimage.metrics.html) — brightness ranges and image quality metrics.
[^randomness]: PyTorch, [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) — seeds and differences between environments.
[^numpy]: PyTorch, [Tensor.numpy](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html), [Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html) — array conversion and detaching from differentiation.
[^subplots]: Matplotlib, [subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) — creating Figures and Axes.
[^imshow]: Matplotlib, [imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html) — image display and brightness ranges.
[^l1]: PyTorch, [L1Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) — absolute error loss.
[^saving]: PyTorch, [saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) — state_dict and checkpoints.
[^kaggle]: Kaggle, [Denoising Dirty Documents](https://www.kaggle.com/c/denoising-dirty-documents) — evaluation and pixel-level submission format.
[^kaggle-cli]: Kaggle, [competition CLI commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md) — submission, status, and score lookup.
[^kaggle-auth]: Kaggle, [CLI authentication guide](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md), [CLI source](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/cli.py) — OAuth login and quota.
[^cross-validation]: scikit-learn, [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) — the roles of model selection and validation.
[^rust]: Rust, [Path](https://doc.rust-lang.org/std/path/struct.Path.html), [HashMap](https://doc.rust-lang.org/std/collections/struct.HashMap.html), [Iterator::zip](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.zip) — paths, counting, and iteration.
