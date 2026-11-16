+++
title = "Image Classification Evaluation: From Dataset Indices to Misclassified Images"
description = "Inspect the links between ImageFolder, Subset, and DataLoader, collect labels and predictions, and calculate accuracy, precision, recall, and F1. Follow evaluation results back to the original files while checking what each validation step can establish."
date = 2026-11-16T20:39:00+09:00
slug = "image-classification-evaluation"

[extra]
toc = true
styles = ["image-classification-evaluation/article.css"]
+++

Printing classification accuracy takes only a few lines of code. Finding the images the model got wrong requires knowing which samples were processed and in what order. The 0 in `y_true[0]` and the 0 in `classes[0]` are the same integer, but they refer to different things. Confusing them can produce correct metrics while displaying the wrong image.

This article follows file paths and labels through model inputs, predictions, and evaluation. The earlier [chest X-ray classification tutorial](/en/chest-xray-classification-pytorch/) covers model construction and the overall training process. Here, we examine how the information and numbers in each object connect. We begin with the necessary data structures, so you can follow without reading the earlier article.

`NORMAL` and `PNEUMONIA` are the two class names used below. Example file counts and confusion-matrix values are provided for explanation and calculation. Execution and performance with an actual model must be verified separately. The final section introduces a [complete Python example](/image-classification-evaluation/examples-en.py) for checking the execution flow and explains the scope of its verification.

## 01. Decide which information must stay connected

The goal is to predict an image's class and, when the prediction is wrong, identify the file and the class it was mistaken for. A **class** is a category to distinguish, and a **label** is the known answer attached to an image. One image together with its label is a **sample**.

First, separate the inputs and outputs needed at each stage.

<div class="eval-table">

| Stage | Input | Information that must remain after processing |
| --- | --- | --- |
| Inspect the source | Class-based image folders | File paths, class names and IDs, sample counts |
| Split the data | All sample indices and labels | Lists of original indices for training and validation |
| Preprocess and form batches | Selected images and labels | Image tensors and label tensors in matching order |
| Run the model | Image tensors | One score per class for each image |
| Train | Scores and labels | Loss and updated model weights |
| Validate | Scores from the current model and labels | Lists of true and predicted labels |
| Analyze results | Both lists and their links to the source | Metrics, a confusion matrix, misclassified files, and both class names |

</div>

A **sample index** is an integer that selects a sample. A **class ID** is an integer that identifies the category a sample belongs to. Indices start at 0. Having 100 samples means there are sample indices from 0 through 99; it does not mean there are 100 classes.

Before writing code, ask three questions. Do you need pixels, a file path, or a label? Which object holds that information? When connecting it to other information, what establishes that both refer to the same sample?

## 02. Choose ImageFolder's root from the class folders

`ImageFolder` is a torchvision dataset class that provides samples from class-based folders. Here, a **Dataset** is an object that reports the sample count and returns an image and its label when given a sample index.

By default, `ImageFolder` uses the names of the subdirectories directly below `root` as class names. It also searches deeper directories within each class folder. It does not understand the word `train` and automatically find the training folder.[^folder]

<div class="eval-table">

| Example file paths | `root`, the parent of the class folders |
| --- | --- |
| `animals/cat/a.png`, `animals/dog/b.png` | `animals` |
| `animals/images/cat/a.png` | `animals/images` |
| `chest_xray/train/NORMAL/a.jpeg` | `chest_xray/train` |
| `images/a.png` and a separate `labels.csv` | The default ImageFolder cannot attach labels from the CSV |

</div>

If `train` and `test` are directly below `chest_xray`, using `chest_xray` as `root` makes those two names the classes. That creates a label system different from the intended distinction between normal and pneumonia.

`Path` is a Python class for handling paths. The `/` operator joins path components; it does not create directories. `Path.cwd()` returns the current working directory, `Path.home()` returns the user's home directory, and `iterdir()` iterates over the entries directly below a directory.[^path]

~~~python
from pathlib import Path
from torchvision.datasets import ImageFolder

print("Current working directory:", Path.cwd())

# Change this to the directory where you extracted the files.
data_dir = Path.home() / "Downloads" / "chest_xray"
root = data_dir / "train"

if not root.is_dir():
    raise FileNotFoundError(f"Check the parent directory of the class folders: {root}")

print("Immediate subdirectories:", [p.name for p in root.iterdir() if p.is_dir()])
dataset = ImageFolder(root=root)
print("Class IDs:", dataset.class_to_idx)
~~~

The relative path `Path("chest_xray/train")` is interpreted from the current working directory. Creating a path object and confirming that a file actually exists are separate steps.

## 03. Distinguish the dataset from the lists it contains

Constructing an `ImageFolder` inspects file paths and labels. It does not load and store every image's pixels in advance. The pixels are read when a sample is requested.[^folder]

<div class="eval-table">

| Expression | Type | Contents or role |
| --- | --- | --- |
| `dataset` | `ImageFolder` object | The rules for reading samples and all the information below |
| `dataset.classes` | List of strings, `list[str]` | Class names in class-ID order |
| `dataset.class_to_idx` | Dictionary, `dict[str, int]` | Class name → class ID |
| `dataset.samples` | List of tuples, `list[tuple[str, int]]` | `(file path, label ID)` for each sample |
| `dataset.targets` | List of integers, `list[int]` | Label IDs in the same order as samples |
| `len(dataset)` | Integer, `int` | Total number of samples |

</div>

A list is an ordered collection. A tuple groups the path and label into a pair here. A dictionary retrieves a value using a key, such as a class name. `targets` is the list of labels within the dataset, not the complete dataset.

For illustration, suppose there are two normal images and two pneumonia images. If the default class ordering gives `classes` as `["NORMAL", "PNEUMONIA"]`, their IDs are 0 and 1. If the labels in `samples` are 0, 0, 1, 1 in order, then `targets` is also `[0, 0, 1, 1]`. When the default labels have not been changed, the following connections hold.

~~~python
# i is an original sample index. Inspect the first sample here.
i = 0
img_path, label = dataset.samples[i]

assert label == dataset.targets[i]
print("File path:", img_path)
print("Label ID:", label)
print("Label name:", dataset.classes[label])
~~~

`assert` checks a condition and stops with an error if it is false. The check above verifies the correspondence between the lists. It cannot establish whether an image was put in the wrong class folder. `ImageFolder` connects the labels expressed by folders to numbers; it does not inspect image content to label it automatically.

### Read only the labels when counting samples

Counting images per class does not require pixels. Start every class count at 0, read each label, and increase the corresponding class count by one.

~~~python
# Each key is a class name; its value is the running image count.
class_count = {name: 0 for name in dataset.classes}

for label in dataset.targets:
    class_name = dataset.classes[label]
    class_count[class_name] += 1

print(class_count)
assert sum(class_count.values()) == len(dataset)
~~~

`class_count.values()` provides the count values in the dictionary, and `sum()` adds them. Iterating with `for image, label in dataset` would also read files and apply preprocessing that this task does not need. To count how many items belong to each category, use the information that identifies their categories.

## 04. Retrieving a file path and reading an image are different operations

The same sample index can produce different return values depending on what you access.

<div class="eval-table">

| Expression | Operation | Return value |
| --- | --- | --- |
| `dataset.samples[i]` | Look up the stored list | A path string and an integer label |
| `dataset[i]` | Read the file and apply configured transforms | An image object or tensor, together with its label |

</div>

With the default PIL loader and no transform, the image is a **Pillow image**. Pillow is a library for reading and working with image files. `type(img)` reports the object's type, and `img.size` is a tuple containing width and height.

~~~python
img, label = dataset[0]
img_path, stored_label = dataset.samples[0]

print("File name:", Path(img_path).name)
print("Object type:", type(img))
print("Color mode:", img.mode)
print("Width, height:", img.size)
assert label == stored_label
img.close()
~~~

Get the file name from the path in `samples` instead of assuming it is available as `img.name`. The default PIL loader converts the image to RGB. An RGB `img.mode` therefore does not establish that the original file on disk was also RGB. To inspect the original mode, open the file itself with `Image.open(img_path)`.[^folder]

A **tensor** is PyTorch's data type for calculating with numbers arranged along multiple axes. Once preprocessing converts the image to a tensor, the way to inspect it changes too.

<div class="eval-table">

| What to inspect | Pillow image | PyTorch tensor |
| --- | --- | --- |
| Object type | `type(img)` | `type(img)` |
| Dimensions | `img.size`: `(width, height)` | `img.shape`: usually `(channels, height, width)` for one image |
| Numeric data type | `np.asarray(img).dtype` when needed | `img.dtype` |
| Color mode | `img.mode` | No equivalent `mode` attribute |

</div>

Here, `np` refers to NumPy imported with `import numpy as np`. `asarray` converts the image to a numeric array for inspection. `dtype` describes the type of the numbers inside the object, rather than the object's own type. Also distinguish the tensor method `img.size()`, which is called, from the Pillow attribute `img.size`.

## 05. Match preprocessing to the model's input requirements

**Preprocessing** converts source data into a form the model can receive. Before choosing transforms, determine the required channel count, height, and width. This example uses a floating-point RGB tensor with 3 channels, a height of 224, and a width of 224.

In the single-image shape `(C, H, W)`, `C` is the channel count, `H` is height, and `W` is width. Axis numbers 0, 1, and 2 correspond to those three meanings. `Resize((224, 224))` receives **one tuple containing height and width**. It is different from `Resize(224, 224)`.

`Compose` runs transforms in their listed order. For ordinary 8-bit RGB or grayscale images, `ToTensor` creates a floating-point tensor with channels first and divides pixel values by 255. It does not apply the same range conversion to every input data type.[^transforms]

~~~python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

tensor_dataset = ImageFolder(root=root, transform=val_transform)
image_tensor, label = tensor_dataset[0]

print(image_tensor.shape)  # (3, 224, 224) with this preprocessing
print(image_tensor.dtype)
print(image_tensor.min().item(), image_tensor.max().item())
~~~

`min()` and `max()` find the smallest and largest values. `.item()` extracts a Python number from a tensor containing one value. With the preprocessing above, ordinary 8-bit inputs already produce values between 0 and 1, so do not divide by 255 again. Forcing both dimensions to a fixed size can change the original aspect ratio. This is a basic example of making shapes consistent; in an actual task, check whether cropping or padding would be more suitable.

Using `Grayscale(num_output_channels=1)` gives the image tensor one channel. The model must also be configured for one input channel. For Normalize, explained below, do not apply the three RGB means and standard deviations directly to a single-channel tensor.

**Data augmentation** modifies training images in ways that preserve their labels. `RandomHorizontalFlip(p=0.5)` flips an image horizontally with a 50% probability when it is read. It does not change the file or increase the number of samples. Do not add flipping automatically when image orientation carries meaning. The validation preprocessing here uses no random augmentation.

### Normalize uses the reference values you supply

**Standardization** changes the numerical reference by subtracting a mean and dividing by a standard deviation. The mean describes the center of the values, and the standard deviation describes their spread around that center. Subtracting the mean expresses whether a value is above or below the center. Dividing by the standard deviation expresses that difference relative to a chosen scale of variation.

`Normalize(mean=..., std=...)` uses the fixed mean and standard deviation supplied for each channel. It does not recalculate statistics for each image. Its output can fall outside the 0–1 range, and using reference values from another dataset does not make your image's mean zero. If you calculate your own reference statistics, calculate them from the training data and reuse them for validation and testing.[^normalize]

**Pretraining** means learning weights from another dataset in advance. **Transfer learning** means using that model for a new task. When reusing weights, first check the preprocessing those weights expect.

~~~python
from torchvision.models import ResNet18_Weights

# This configuration object identifies the weights version to use.
weights = ResNet18_Weights.IMAGENET1K_V1
preprocess = weights.transforms()
print(preprocess)
~~~

`weights` itself is not a tensor of learned numbers. It is a configuration object associated with a weights file and preprocessing information. `weights.transforms()` creates the registered preprocessing for evaluation and inference. This ResNet18 configuration includes resizing, center cropping, range conversion, and normalization with ImageNet channel statistics. Do not automatically append another `ToTensor` and `Normalize`.[^weights]

Creating a preprocessing object is separate from constructing a model and loading its weights with `resnet18(weights=weights)`. The latter does not automatically determine the number of classes in the new task either. Changing the output layer is covered in the [earlier article's model construction](/en/chest-xray-classification-pytorch/).

## 06. Design sample selection separately from preprocessing

Training uses labels to change the model's weights. Validation evaluates the current model and helps select settings. Testing evaluates the selected model on data held aside for the final assessment. Validation and testing do not update the weights.

First, divide the original sample indices into training and validation lists. There is no need to read image pixels to make this split. `targets` identifies each sample's class, providing the information needed to consider class proportions.

A **stratified split** preserves the original class proportions in both parts as closely as possible. If the original ratio is 20:80, the split tries to retain a similar ratio; it does not balance the classes to 50:50.[^split]

~~~python
from sklearn.model_selection import train_test_split

# range excludes its endpoint. Build all original sample indices.
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=dataset.targets,
    random_state=42,
)

assert len(train_idx) + len(val_idx) == len(dataset)
assert set(train_idx).isdisjoint(val_idx)
~~~

`test_size=0.2` assigns about 20% to the second returned list. Although the parameter contains the word `test`, we use that part for validation here. `stratify` receives the per-sample labels, and `random_state` is a random seed for reproducing the split with the same inputs and conditions. `set(...).isdisjoint(...)` checks that the two sets share no elements. Stratification can fail when a class has too few samples or a split is too small.

Next, use **Subset** to provide only the selected indices. A Subset connects an original dataset to an index list; it does not copy image files. `ImageFolder` has no `indices` argument for this purpose, and the default `dataset[train_idx]` is not a way to select a whole list of integer indices.[^data]

To configure training and validation preprocessing independently, create separate `ImageFolder` objects for reading the source. Verify that both use the same sample list and class IDs, then apply split indices that refer to that common list.

~~~python
from torch.utils.data import Subset

train_base = ImageFolder(root=root, transform=train_transform)
val_base = ImageFolder(root=root, transform=val_transform)

# Equal numbers are not enough: their paths, labels, and order must match.
assert dataset.samples == train_base.samples == val_base.samples
assert dataset.class_to_idx == train_base.class_to_idx == val_base.class_to_idx

train_dataset = Subset(train_base, train_idx)
val_dataset = Subset(val_base, val_idx)
~~~

Both `train_base` and `val_base` can access the same complete set of samples. Each Subset determines which of those samples to use. If two Subsets reference the same ImageFolder object, they share its `transform`. Adding a `transform` attribute to a Subset does not make the default Subset execute it.

Checking that the index lists do not overlap only prevents repeated use of the same index across splits. Copied files or several images from the same patient require separate inspection. Separating groups such as patients requires reliable group information and an appropriate splitting method. PyTorch's `random_split` returns randomly divided Subsets but has no `stratify` argument.[^data][^split]

## 07. Subset changes what an index refers to

Suppose the original dataset has four samples and an illustrative selection list is `[3, 1]`. The first sample in the Subset is the fourth sample in the original dataset. The indices used to request them differ: 0 and 3. They are connected because the first entry in the selection list contains original index 3.

<div class="eval-table">

| Sample index within the Subset | Original sample index retrieved from the selection list | Sample actually requested |
| ---: | ---: | --- |
| 0 | 3 | `val_base[3]` |
| 1 | 1 | `val_base[1]` |

</div>

In this example, requesting `val_dataset[0]` reads `val_dataset.indices[0]`, then uses the resulting 3 to request `val_base[3]`. The index's reference changes, while the linked original file and its default class ID remain the same. With a random transform, reading the same file again can still produce different transformed pixels.

<figure class="eval-figure">
<img src="/image-classification-evaluation/data-identity-en.svg" alt="In the illustrative selection list [3, 1], validation sample index 0 maps to original sample index 3. The samples list provides the file and label, while class ID 1 maps to PNEUMONIA in classes." loading="lazy">
<figcaption>Follow the sample index used to find a file separately from the class ID used to find a name. The selection list is an example of this mapping.</figcaption>
</figure>

Once we connect a `DataLoader` to provide batches, the objects have the following roles. A **DataLoader** retrieves samples from a dataset and provides them in **batches**, which are groups of samples.

<div class="eval-table">

| Variable | Actual type | Scope and role |
| --- | --- | --- |
| `val_base` | ImageFolder | Paths, labels, and class information for the complete source, plus validation preprocessing |
| `val_dataset` | Subset | Samples selected from the source for validation |
| `val_loader` | DataLoader | Provides the selected validation samples in batches |

</div>

`val_dataset.dataset is val_base` checks whether the two expressions refer to the same object. `is` compares object identity, not whether their values are similar. The `val_loader.dataset` created below is exactly `val_dataset`. If an ImageFolder were passed directly to the DataLoader, that attribute would instead be an ImageFolder. Do not infer its type from the name `.dataset` alone.

## 08. Distinguish batch counts, sample counts, and iteration state

Training uses the same set of samples repeatedly, so their reading order can be shuffled. For validation, keep the default order to make results easy to connect to the source, and include the smaller final batch.

~~~python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    drop_last=False, num_workers=0,
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False,
    drop_last=False, num_workers=0,
)

assert val_loader.dataset is val_dataset
assert val_dataset.dataset is val_base

images, labels = next(iter(val_loader))
print(images.shape, labels.shape)
print(images.dtype, labels.dtype)
assert images.shape[0] == labels.shape[0]
~~~

`batch_size` is the maximum number of samples in a batch. `shuffle` changes the order of samples within that set; it does not shuffle images and labels separately. `drop_last` determines whether to discard a smaller final batch. `num_workers=0` avoids creating separate loading worker processes, making the flow simpler to inspect here.[^data]

Splitting divides samples into sets, while shuffling changes the order within a set. Creating a DataLoader does not train the model. Training or validation depends on the calculations performed afterward.

Let `B` be the number of samples in the current batch. The image batch shape is `(B, C, H, W)`, and the label shape is `(B,)`. Axis 0 represents samples, axis 1 channels, axis 2 height, and axis 3 width. Default batching requires image tensors of the same size. ImageFolder itself does not make all image dimensions equal.

Consider a **counting example** with 100 validation samples and a batch size of 32.

<div class="eval-table">

| What is inspected | Value | What is counted |
| --- | ---: | --- |
| `len(val_dataset)` | 100 | Validation samples |
| `len(val_loader.dataset)` | 100 | Samples in the dataset passed to the DataLoader |
| `len(val_loader)` | 4 | Batches |
| `len(labels)` during iteration | 32, 32, 32, 4 | Actual samples in the current batch |

</div>

Use `total += len(labels)` to accumulate the number of processed samples. Adding `len(val_loader)` on each iteration repeatedly adds the batch count. With `drop_last=True`, this example would process only 96 samples, excluding the last 4.

### next retrieves the next batch from an iterator

An **iterable** is an object from which a traversal can begin. An **iterator** tracks how far that traversal has progressed. Calling `iter()` on a DataLoader gives an iterator that retrieves batches. `next()` retrieves one next batch from that iterator.[^iteration]

~~~python
batch_iter = iter(val_loader)
images1, labels1 = next(batch_iter)

# Run this only if a second batch exists.
if len(val_loader) >= 2:
    images2, labels2 = next(batch_iter)
~~~

Repeating `next(iter(val_loader))` does not advance the same iterator. Each call starts a new traversal and retrieves its first batch. With the default sequential settings, it retrieves the same first batch; shuffling or random transforms can change the contents of a new traversal. Calling `next()` on an exhausted iterator raises `StopIteration`. To process everything, use `for images, labels in val_loader` and let the loop manage the iterator and its termination.

## 09. Model scores, activation functions, and loss functions have different roles

Consider a classification task with one correct class per image. The model uses the image's numbers and learnable **weights** to calculate intermediate values, then produces a score for each class. An **activation function** transforms intermediate values before passing them to the next calculation. A **loss function** compares the final scores with the true label and produces a number that defines what training should improve.

For example, ReLU changes negative values to 0 and leaves positive values unchanged. A sequence consisting only of weighted sums can be combined into one calculation of the same form. Adding a transform such as ReLU, whose response depends on its input, allows more complex relationships to be represented. An activation function does not receive the true label, while a classification loss needs to compare predictions with that label.

The model's scores before conversion to probabilities are called **logits**. They can be negative and do not need to sum to 1. Let `K` be the number of classes. The structures are then:

<div class="eval-table">

| Variable | Type and shape | Meaning of the axes |
| --- | --- | --- |
| `images` | Floating-point tensor `(B, C, H, W)` | Samples, channels, height, width |
| `logits` | Floating-point tensor `(B, K)` | Samples, per-class scores |
| `labels` | `torch.int64` tensor `(B,)` | One true class ID per sample |
| `preds` | Integer tensor `(B,)` | One predicted class ID per sample |
| `loss` | Tensor of shape `()` with the default mean reduction | One value representing the batch loss |

</div>

The model must be prepared to accept `C` input channels and produce `K` scores. Leaving an ImageNet model's default 1,000-class output unchanged does not complete a model for this two-class task.

In this case, `CrossEntropyLoss` receives logits and integer labels. It evaluates whether the correct class receives a sufficiently high score relative to the other classes, handling the necessary log-probability calculation internally. Do not apply Softmax beforehand.[^loss]

**Softmax** transforms one sample's scores into probabilities that sum to 1. **argmax** returns the index of the largest value, rather than that value itself. Axis 1 of `logits` is the class axis, so `argmax(dim=1)` returns a class ID for each sample. Softmax preserves the ordering of the scores; if only a predicted class ID is needed, there is no reason to calculate Softmax first.

~~~python
from torch import nn

# model must have the required input channels and output class count.
# Assume model, images, and labels are on the same computing device.
loss_fn = nn.CrossEntropyLoss()
logits = model(images)
loss = loss_fn(logits, labels)
preds = logits.argmax(dim=1)
~~~

`dim=1` represents channels in `images` and classes in `logits`. The meaning of 1 depends on the tensor's complete structure. Also, an output shaped like probabilities does not establish that it accurately represents the chance of a prediction being correct.

Other losses require their own matching output and label structures. `BCEWithLogitsLoss` is used for binary classification with one score or for several independent binary decisions. `MSELoss` and `L1Loss` are used for errors in continuous numerical values. Do not swap them into this code by changing only the function name.

## 10. Separate training mode from weight updates

A **gradient** describes how the loss responds to changes in a weight. **Backpropagation** starts from the loss and calculates gradients for the weights. An **optimizer** uses those gradients to update the weights.

Training proceeds through clearing previous gradients, calculating scores, calculating loss, calculating gradients, and updating weights. Constructing the model's layers and neurons is separate from using data to train their weights.

<div class="eval-table">

| Expression | What it does | A common confusion |
| --- | --- | --- |
| `model.train()` | Switches to training mode | Calling it does not update weights |
| `optimizer.zero_grad()` | Clears previous gradients | It does not set weights to zero |
| `loss.backward()` | Calculates gradients | Weight updates have not happened yet |
| `optimizer.step()` | Updates weights using gradients | This is the parameter-update step in training |
| `model.eval()` | Switches to evaluation mode | It does not automatically disable gradient recording |
| `with torch.no_grad():` | Disables recording for backpropagation within the block | It does not automatically switch to evaluation mode |

</div>

Training and evaluation modes affect layers whose behavior changes between modes, such as Dropout and BatchNorm. **Dropout** randomly excludes some intermediate outputs during training. **BatchNorm** uses different statistical handling during training and evaluation. For now, understanding that `eval()` and `no_grad()` control different settings matters more than the internal implementation of these layers.[^autograd]

Place the model, inputs, and labels on the same device for the calculation. For an ordinary single-device model, the first parameter tensor's `.device` identifies the model's device. Here, `next(model.parameters())` retrieves the first parameter; it retrieves a different kind of object from the batch iterator in the earlier section.

~~~python
import torch

# Prepare the optimizer after moving model to the chosen device.
# lr is the learning rate, which controls the size of each weight update.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = next(model.parameters()).device

model.train()
for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
~~~

Ordinary batch training clears previous gradients so that the update uses the new batch's gradients. This is necessary because PyTorch accumulates gradients during `backward()`. A traversal that uses every training sample once, as in this setup, is one **epoch**.

The device names are `mps` for an Apple GPU on a Mac, `cuda` for an NVIDIA GPU, and `cpu` for a CPU. Select a device after checking actual support. The code fragments in the article assume that an appropriate model is ready. The final section provides a runnable example that also constructs the model.

## 11. Collect validation results in matching sample order

The total correct count and total sample count are enough to calculate accuracy. They cannot reconstruct which class was mistaken for which. We therefore collect true class IDs in `y_true` and predicted IDs in `y_pred`. Both are `list[int]`, and matching indices must refer to results for the same image.

Use `extend()` to join batch results into one long list. `append([0, 1])` adds a whole list as one element, giving `[[0, 1]]`, while `extend([0, 1])` adds its integers individually, giving `[0, 1]`.[^lists]

The following code processes every validation sample once and collects counts, loss, labels, and predictions together. It uses the default mean loss from `CrossEntropyLoss()` with no weighting and no ignored label values.

~~~python
model.eval()
correct = 0
total = 0
loss_sum = 0.0
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        assert logits.shape == (len(labels), len(val_base.classes))
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)

        batch_count = len(labels)
        correct += (preds == labels).sum().item()
        total += batch_count
        loss_sum += loss.item() * batch_count

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

assert total > 0
assert len(y_true) == len(y_pred) == total == len(val_dataset)

accuracy = correct / total
mean_loss = loss_sum / total
print(f"Accuracy: {accuracy:.2%}, mean loss: {mean_loss:.4f}")
~~~

`preds == labels` creates a tensor with True for each correct prediction and False for each wrong one. `.sum()` counts each True as 1, and `.item()` converts the result to a Python number. `.cpu()` returns a CPU tensor, and `.tolist()` converts its values to a Python list. `tolist()` itself handles a CPU transfer when needed, so the preceding `.cpu()` is not mandatory.[^tolist]

Datasets and DataLoaders are not tensors, so do not move them with expressions such as `val_loader.dataset.cpu()`. Calling `.cpu()` does not overwrite the original variable either. When inspecting a tensor that tracks gradients as a NumPy array, the common expression `tensor.detach().cpu().numpy()` performs three different operations: `detach()` separates it from gradient tracking, `cpu()` moves it to the required device, and `numpy()` converts it to an array.

### Why not simply average the batch averages?

Within each batch average, the images in that batch have equal weight. If batch sizes differ, averaging those averages equally gives each image in a smaller batch more influence. To count every image once with equal weight, multiply each batch average by its actual sample count to recover a sum, then divide the combined sums by the total sample count.

<figure class="eval-figure">
<img src="/image-classification-evaluation/batch-mean-en.svg" alt="An example with 100 samples divided into batches of 32, 32, 32, and 4. Averaging batch means equally gives the final 4 samples more influence. Multiplying each batch mean by its sample count, adding, and dividing by 100 weights all samples equally." loading="lazy">
<figcaption>First decide what should receive equal weight. Here, the unit is one image sample.</figcaption>
</figure>

Accuracy likewise uses the total correct count divided by the total processed count. When every batch is the same size, averaging batch accuracies gives the same result; with a smaller last batch, it generally does not. Class-weighted losses or ignored labels can change the denominator of the mean, so do not apply the `loss × batch_count` rule to those cases without checking it.[^loss]

## 12. Accuracy, precision, and recall use different sample groups

Before memorizing metrics, consider which samples form the denominator. **Accuracy** is the proportion of all evaluated samples whose predicted class is correct. **Precision** and **recall** focus on one class but evaluate it against different groups of samples.

If PNEUMONIA is the class we want to find, we call it **positive** and the other class **negative**. Positive does not mean a good outcome; it identifies the target we are looking for.

<div class="eval-table">

| Metric | Samples selected first | What to check within that group |
| --- | --- | --- |
| Accuracy | All evaluated samples | Whether the true and predicted classes match |
| PNEUMONIA precision | Samples the model predicts as PNEUMONIA | Whether they are actually PNEUMONIA |
| PNEUMONIA recall | Samples that are actually PNEUMONIA | Whether the model finds them as PNEUMONIA |
| PNEUMONIA support | Samples that are actually PNEUMONIA | Their count, rather than a proportion |

</div>

Precision and recall have the same numerator: the number of correct predictions for that class. Their reference groups differ. Precision asks, “How many of the reported findings are correct?” Recall asks, “How many of the actual targets were found?”[^metrics]

If every actual pneumonia image is found, recall is 100%. Suppose the model now incorrectly calls more normal images pneumonia, while the number of actual pneumonia images and the number found remain unchanged. Recall does not change. Precision falls because more incorrect results are included among the samples called pneumonia. This is why the two metrics reveal different kinds of mistakes.

**Class imbalance** means that classes have different sample counts. As a calculation example, predicting normal for all 990 normal samples and 10 defective samples gives 99% accuracy but 0% recall for defects. Accuracy counts each sample once, so a class with more samples has more influence on the overall value.

`support` is the number of samples with that true label in the current evaluation. It does not automatically mean the class count in the original complete dataset. Nor should 777 images be interpreted as 777 patients without separately establishing that each patient has exactly one image.

Choose which metric to prioritize by considering what happens after a prediction. Recall matters when missing a target is costly; precision can matter when an incorrect finding leads to costly blocking or deletion. Do not choose a fixed priority from the field's name alone. In document retrieval, precision is the proportion of retrieved documents that are relevant, while recall is the proportion of all relevant documents that were retrieved. First define what counts as a relevant document.

## 13. A confusion matrix preserves the direction of mistakes

A **confusion matrix** counts samples for each combination of true and predicted classes. In scikit-learn, rows represent true classes and columns represent predicted classes. Each horizontal row corresponds to one true class, and each vertical column to one predicted class.[^cm]

The following values are a **calculation example** for understanding the metrics. They are not performance results obtained by training a model for this article.

<div class="eval-table">

| True class ↓ / Predicted class → | NORMAL | PNEUMONIA | Row total |
| --- | ---: | ---: | ---: |
| NORMAL | 222 | 48 | 270 |
| PNEUMONIA | 12 | 765 | 777 |
| Column total | 234 | 813 | 1,047 |

</div>

The 222 samples are normal images correctly predicted as normal, while 48 are normal images incorrectly predicted as pneumonia. The 12 samples are pneumonia images missed as normal, while 765 are pneumonia images correctly predicted as pneumonia.

With pneumonia defined as positive, the four cells have the following names.

<div class="eval-table">

| Name | Meaning | Example count |
| --- | --- | ---: |
| True Negative, TN | An actual negative correctly predicted as negative | 222 |
| False Positive, FP | An actual negative incorrectly predicted as positive | 48 |
| False Negative, FN | An actual positive missed as negative | 12 |
| True Positive, TP | An actual positive correctly predicted as positive | 765 |

</div>

<figure class="eval-figure">
<img src="/image-classification-evaluation/confusion-en.svg" alt="The example confusion matrix uses rows for true classes and columns for predictions. Pneumonia precision uses the predicted-pneumonia column total of 813 as its denominator, while recall uses the actual-pneumonia row total of 777." loading="lazy">
<figcaption>The same 765 correct findings give different metrics depending on whether the reference is all predicted targets or all actual targets.</figcaption>
</figure>

Because the row and column class orders match, the diagonal counts correct predictions. A class's total actual count is its row sum, not its diagonal value. Its column sum counts predictions of that class.

After identifying what each count represents, calculate the metrics as follows.

<div class="eval-table">

| Quantity | Calculation in this example | Result |
| --- | --- | ---: |
| Overall accuracy | Add the 222 and 765 correct predictions, then divide by all 1,047 samples | About 94.27% |
| NORMAL precision | 222 correctly predicted normal / 234 predicted normal | About 94.87% |
| NORMAL recall | 222 correctly predicted normal / 270 actually normal | About 82.22% |
| PNEUMONIA precision | 765 correctly predicted pneumonia / 813 predicted pneumonia | About 94.10% |
| PNEUMONIA recall | 765 correctly predicted pneumonia / 777 actually pneumonia | About 98.46% |

</div>

High overall accuracy does not mean the two classes have similar recall. Here, normal recall differs from pneumonia recall. To see which kinds of mistakes occur more often, read the per-class metrics alongside overall accuracy.

## 14. labels selects class values and order; target_names provides display names

The minimum inputs for calculating a confusion matrix are `y_true` and `y_pred` in matching sample order. File paths and the model object are not needed for the calculation. Class names help people read the result.

By default, `confusion_matrix(y_true, y_pred)` collects and sorts the class values appearing in either list. Since not every class necessarily appears, we may need to specify which classes the output table includes.[^cm]

~~~python
from sklearn.metrics import confusion_matrix

# Use separate demo variables to avoid mixing this with validation results.
demo_true = [1, 1, 1]
demo_pred = [1, 1, 1]

print(confusion_matrix(demo_true, demo_pred))
# [[3]]: includes only class 1, which appears in the data.

print(confusion_matrix(demo_true, demo_pred, labels=[0, 1]))
# [[0 0]
#  [0 3]]: also includes the empty class 0 in the specified order.
~~~

`labels` does not assign new label IDs. It specifies which existing class values to include and in what order. ImageFolder's default IDs run from 0 through `K-1` for `K` classes, so we can create the complete ID list as follows.

~~~python
class_names = val_base.classes
class_ids = list(range(len(class_names)))

cm = confusion_matrix(y_true, y_pred, labels=class_ids)
~~~

`len(class_names)` is the class count. `range()` provides numbers from 0 up to, but excluding, that count, and `list()` makes them into a list. Two classes give `[0, 1]`; three give `[0, 1, 2]`. Creating `class_ids` without passing it to the function has no effect on the table.

`target_names` supplies the names displayed in a report. With `labels=[1, 0]`, the names must follow the same order: `["PNEUMONIA", "NORMAL"]`. Supplying names does not make the function automatically match them to the meanings of the numbers. In another task where the original label values are strings, `labels` can contain those strings too.

## 15. Calculating metrics from the matrix makes the axes explicit

`cm` is a two-dimensional NumPy array. Here, it is a **count matrix** containing all classes, with no normalization or sample weighting. Under those conditions, its total must equal the number of evaluated samples.

~~~python
assert len(y_true) > 0
assert cm.sum() == len(y_true) == len(y_pred) == len(val_dataset)

accuracy_from_cm = cm.diagonal().sum() / cm.sum()
print(f"Accuracy calculated from the confusion matrix: {accuracy_from_cm:.2%}")

class_recall = {}
for row_idx, row in enumerate(cm):
    class_name = class_names[row_idx]
    support = row.sum()

    # Distinguish no actual samples from getting every sample wrong.
    if support == 0:
        class_recall[class_name] = None
    else:
        class_recall[class_name] = row[row_idx] / support

print(class_recall)
~~~

`cm.diagonal()` returns an array of diagonal entries, whose sum is the total correct count. `enumerate(cm)` retrieves one row at a time, so `row_idx` is a row index and `row` is the one-dimensional array for that row.

In `row[row_idx]`, however, the same integer is used as a column index within that row. Because the row and column class orders match, selecting the same number retrieves the diagonal count of correct predictions. Calling the variable `col` would not change where the number came from: iteration over rows. The name `row_idx` makes that origin easier to see.

`None` is the value chosen here to mark an unavailable measurement. Recall of 0 when actual samples exist but all were missed is different from recall that cannot be calculated because no actual samples exist. Also, if `labels` contains only some classes or `normalize` produces a matrix of proportions, the total-count check above no longer applies unchanged.

## 16. Read F1 and the classification report by class

The **F1 score** evaluates precision and recall together. It uses their **harmonic mean**, making a high score difficult to achieve when only one of the two is high. Before considering the mathematical derivation, focus on its purpose: checking both whether reported findings are correct and whether actual targets were missed.

Writing precision as `P` and recall as `R`, F1 is `2 × P × R / (P + R)`, with both values expressed as proportions from 0 to 1. For example, 100% precision and 20% recall give an F1 of about 33.3%. If both metrics are 80%, F1 is also 80%. Equal values alone are not evidence of good performance: if both are 10%, F1 is 10% too.

When zeros occur, do not rely only on the appearance of the formula. If a class exists in the true labels or predictions but has no correct predictions, its F1 is 0. A class absent from both true labels and predictions has no basis for an F1 measurement and is handled according to `zero_division`. Using the counts from the earlier section, `2 × TP / (2 × TP + FP + FN)` makes this distinction visible.[^f1]

`classification_report` collects precision, recall, F1, and support for each class. Its default return value is a string intended for display.[^report]

~~~python
from sklearn.metrics import classification_report

report = classification_report(
    y_true,
    y_pred,
    labels=class_ids,
    target_names=class_names,
    digits=4,
    zero_division=0,
)
print(report)
~~~

`digits=4` sets the number of decimal places displayed. `zero_division=0` chooses to display an undefined metric as 0. It does not resolve missing samples or predictions, or turn every displayed zero into an observed performance value. Check support and prediction counts as well.

<div class="eval-table">

| Report entry | What is averaged | What receives equal weight |
| --- | --- | --- |
| `accuracy` | The proportion of correctly classified samples overall | Each evaluated sample |
| `macro avg` | Per-class metrics with equal class weights | Each class |
| `weighted avg` | Per-class metrics weighted in proportion to support | A class's weight for the metric follows its actual sample count |

</div>

Macro F1 is the **average of the per-class F1 scores**. It generally differs from first calculating macro precision and macro recall, then using those two values to calculate F1. Check both what is being averaged and the order of operations.

To check the calculation with five samples without overwriting actual results, use this example.

~~~python
demo_true = [0, 1, 1, 0, 1]
demo_pred = [0, 1, 0, 0, 1]
demo_names = ["NORMAL", "PNEUMONIA"]

print(confusion_matrix(demo_true, demo_pred, labels=[0, 1]))
print(classification_report(
    demo_true, demo_pred, labels=[0, 1],
    target_names=demo_names, zero_division=0,
))
~~~

The expected confusion matrix is `[[2, 0], [1, 2]]`. Four of the five samples are correct, giving 80% accuracy.

<div class="eval-table">

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| NORMAL | 0.67 | 1.00 | 0.80 | 2 |
| PNEUMONIA | 1.00 | 0.67 | 0.80 | 3 |

</div>

With the same result lists and class order, the report's recall must match the recall calculated directly from the confusion matrix, allowing only display-rounding differences. A high F1 alone does not establish that a model is the best choice for a task. Also inspect individual metrics against the costs of misses and incorrect detections.

## 17. Correct evaluation and correct file mapping are separate requirements

Accuracy and a confusion matrix require true labels and predictions paired for the same samples. Applying the same reordering to both complete lists leaves the metrics unchanged. To use a result index to retrieve an original file, however, the result order must also correspond to the Subset order.

The current code preserves that correspondence under these conditions:

- `val_dataset` is a single Subset directly wrapping `val_base`.
- `val_loader` uses that Subset with default sequential sample selection and batch ordering.
- `shuffle=False` and `drop_last=False` process every sample once.
- True labels and predictions are appended together in processing order.
- Neither result list nor the index list is independently sorted or filtered afterward.

Under these conditions, result-list index `i` corresponds to `val_dataset[i]`. That sample's original index is stored in `val_dataset.indices[i]`. Looking up `val_base.samples` with the original index gives the file path and original label.

<div class="eval-table">

| Variable | Type | Where it comes from | What it indexes |
| --- | --- | --- | --- |
| `i` | Integer | The index attached while iterating over results | `val_dataset.indices[i]` |
| `origin_idx` | Integer | The original sample index stored in the selection list | `val_base.samples[origin_idx]` |
| `true_label` | Integer | An ImageFolder label passed through a batch into `y_true` | `val_base.classes[true_label]` |
| `pred_label` | Integer | The argmax of the per-class scores | `val_base.classes[pred_label]` |

</div>

To convert a predicted ID to a class name, model output 0 must represent class 0 in the training data. This is why the training and validation `class_to_idx` mappings must match. The lookup works because the stored value's origin and numbering system match, not because the variable happens to be named `true_label`.

Metrics can still be calculated with shuffled validation order if true labels and predictions are collected together. But result index `i` can no longer be used directly as a Subset index. If the design needs a different order, return and store the file path or original index alongside each prediction.[^data]

Equal list lengths verify counts only. Comparing original and collected labels also cannot detect different images of the same class being swapped. File identity must be established by a structure that preserves sample order or by file identifiers retained alongside the results.

## 18. Display both the true and predicted names for misclassified images

**Error analysis** examines incorrect samples to investigate the kinds of errors that occur. Break the process into comparing labels and predictions, selecting an error, looking up its original index, retrieving its file path, reading the image, and displaying the true and predicted names.

`zip(y_true, y_pred)` retrieves pairs of values at matching list indices. `enumerate()` adds the result index `i` to each pair. Because `zip` stops at the shorter input, check the lengths first.

Pillow's `Image.open()` opens an image file. The `with ... as img` form closes the opened file when the block ends. In Matplotlib, `imshow()` places the image on a figure, `title()` sets the title, `show()` displays it, and `close()` closes the figure. An image file and a Matplotlib figure are separate resources.[^image]

~~~python
from PIL import Image
import matplotlib.pyplot as plt

assert len(y_true) == len(y_pred) == len(val_dataset)
found_error = False

for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
    if true_label != pred_label:
        origin_idx = val_dataset.indices[i]
        img_path, original_label = val_base.samples[origin_idx]

        # This auxiliary check does not prove file order within a class.
        assert true_label == original_label

        true_name = val_base.classes[true_label]
        pred_name = val_base.classes[pred_label]
        print("File:", img_path)
        print("True:", true_label, true_name, "/ Predicted:", pred_label, pred_name)

        with Image.open(img_path) as img:
            plt.figure()
            # Prevent a grayscale source from being displayed with a misleading colormap.
            plt.imshow(img.convert("RGB"))
            plt.title(f"True: {true_name} | Pred: {pred_name}")
            plt.axis("off")
            plt.show()
            plt.close()

        found_error = True
        break

if not found_error:
    print("There are no misclassified samples in this evaluation list.")
~~~

`found_error` is a Boolean value recording whether an error was found. `break` ends the loop after the first error is processed. If there are no errors, it is normal for no image to appear; the message explains that outcome.

Calling `plt.title(pred_name)` after `plt.title(true_name)` replaces the same title with the second value. To display both, combine them into one string and set the title once. The English title above can be used without configuring a Korean font. With a Korean font configured, the true and predicted labels can instead be written in Korean.

The displayed image is **read from the original file**. The model received a tensor processed by transforms such as Resize or Normalize, so do not assume the two have the same appearance. The original image can suggest possible causes of an error, but one image does not establish the cause. The next step is to inspect the actual model input as well.

## 19. Check object types and the scope of a check before changing syntax

When an error occurs, identify the current variable's type and the required operation before changing function names.

<div class="eval-table">

| Purpose | Correct expression | What to distinguish |
| --- | --- | --- |
| Retrieve a selected index | `val_dataset.indices[i]` | List lookup uses `[]`; function calls use `()` |
| Retrieve a confusion-matrix cell | `cm[row_idx, col_idx]` | Do not call an array as if it were a function with `cm(...)` |
| Sum a Python list | `sum(values)` | Do not assume a list has a `.sum()` method |
| Sum a NumPy array or tensor | `values.sum()` | These objects provide a sum method |
| Retrieve a class name | `val_base.classes[label]` | Use a class ID, not a Subset index |
| Clear previous gradients | `optimizer.zero_grad()` | This is not an `optimizer.no_grad()` operation |
| Disable gradient recording | `with torch.no_grad():` | Separate this from clearing gradients or switching evaluation mode |
| Stop when a condition fails | `assert cm.sum() == len(y_true)` | Writing a comparison alone differs from stopping on a failed check |

</div>

Here, `values` is an example name for the list, array, or tensor being summed, and `col_idx` is a column index. Names such as `transform`, `drop_last`, `plt.imshow`, and `classification_report` must also be spelled correctly. `val_loader` and `val_laoder` are different variable names.

A passed check does not establish meanings or relationships it did not inspect.

<div class="eval-table">

| Passed check | What it establishes | What it does not establish |
| --- | --- | --- |
| Per-class counts sum to the dataset length | The counted total | Whether the folder labels are actually correct |
| Training and validation indices do not overlap | Separation of identical original indices | No leakage from copied files or the same patient |
| Two `.samples` lists are equal | Matching paths, labels, and order | Whether image content and labels are valid |
| Image and label batch sizes match | Matching counts | Correct pairing of every image and label |
| True-label and prediction list lengths match | Matching result counts | Result order and original file identity |
| The full-class count matrix sums to the result count | All results were counted | Performance on new data |

</div>

`assert` is useful for experiments and development checks. Python can omit it when run with optimization options, so do not rely only on assertions for required input validation in deployed programs.[^assert]

<details>
<summary>Check the same index mapping in Rust</summary>

The relationship between an original list and a selection list remains the same in another language. This example compares only the lookup from selected index to original sample and from label to name; it does not implement a neural network. Rust's `usize` is an integer type used for array indices, and `let` declares a variable. `&str` is the type used here for a string reference.

~~~rust
fn main() {
    let classes = ["NORMAL", "PNEUMONIA"];
    let samples: [(&str, usize); 4] = [
        ("normal01.jpeg", 0),
        ("normal02.jpeg", 0),
        ("pneumonia01.jpeg", 1),
        ("pneumonia02.jpeg", 1),
    ];
    let selected_indices: [usize; 2] = [3, 1];

    let subset_index: usize = 0; // The first sample in the selection list
    let original_index = selected_indices[subset_index];
    let (path, label) = samples[original_index];

    // Use the original index for the file and the class ID for its name.
    println!("{} | {}", path, classes[label]);
}
~~~

The output is `pneumonia02.jpeg | PNEUMONIA`. The lookup follows the same order as Python's `samples[indices[i]]`. Python lists can change length, while the arrays in this Rust example have lengths fixed in their declarations. The underlying mapping is the same in both languages.

</details>

## 20. Define what you should be able to explain independently

The practical goal of this article is to inspect data and connect evaluation results back to the source files. Memorizing more function names is secondary. First answer the following questions without code, then check your reasoning with a small example if needed.

1. Why can samples be counted per class without reading their pixels?
2. What kind of number does each of `val_dataset.indices[i]` and `val_base.classes[true_label]` receive?
3. Why can metrics still be calculated after shuffling validation order, while original-file lookup becomes a problem?
4. When batch sizes differ, which samples receive more weight if batch means are averaged equally?
5. How can recall be 100% while precision is low?

<details>
<summary>Answers to check after thinking it through</summary>

1. The task needs each sample's class ID, not its image content. `targets` alone is enough to count samples per class.
2. The first receives a sample index within the validation Subset and returns an original sample index. The second receives a true class ID and returns its class name.
3. Metrics require true labels and predictions paired for the same samples. File lookup additionally requires a result index to match a Subset index, and shuffled reading order breaks that condition.
4. Each sample in a smaller batch receives more weight. Every batch mean contributes equally, but fewer samples share that contribution within the smaller batch.
5. All actual targets may have been found while many non-targets were included too. Recall uses the actual targets as its denominator, while precision uses the predicted targets.

</details>

<div class="eval-table">

| Learning priority | What to do now |
| --- | --- |
| Essential for practical work | Inspect the source structure, verify class mappings, prepare splits, transforms and batches, collect validation results, and connect metrics to misclassified files |
| Understand the principle | Roles of loss and weight updates, weighting batch means, the reference groups for precision and recall, and the meaning of F1 and class averages |
| Refer to when needed | Mathematical derivation of F1, complex nested Subsets, parallel loading settings, and group-splitting implementation details |

</div>

The stopping point is being able to start from an empty notebook, inspect the data structure, connect `ImageFolder → Subset → DataLoader`, collect labels and predictions, and produce metrics and misclassified file paths. You should be able to explain whether each number is a sample count, batch count, original index, or class ID.

In an actual project, continue with model-input visualization, training and validation records across several epochs, saving and reloading the best model, independent testing, and inference on new images. Preserve the class-ID mapping and preprocessing settings along with the model weights. Separately verify the actual labels and the independence of the data splits.

## 21. Run the complete example and distinguish what was verified

The article separates code into sections to explain the decisions behind each task. To run it as one file, use the [complete Python example with comments](/image-classification-evaluation/examples-en.py). The English and [Korean example](/image-classification-evaluation/examples-ko.py) perform the same processing.

The script creates 24 blue images and 24 red images so that the connections can be checked without a separate dataset. The default split contains 36 training images and 12 validation images. With a batch size of 5, the validation batches contain 5, 5, and 2 samples. This is a synthetic-image example, not an X-ray performance experiment.

For a quick run, the downloadable example uses 64×64 inputs and a 25% validation split. The 224×224 inputs and 20% split in the article are separate settings used to explain individual concepts. For an actual model, choose values that match its input structure and the requirements of the split.

The example's small CNN accepts RGB inputs of shape `(B, 3, 64, 64)`, calculates intermediate features with convolutions and ReLU, and outputs `K` class scores. **Convolution** applies the same kernel weights across different parts of an image. The intermediate channel counts are 8 and 16. Two pooling operations reduce height and width to 16×16, after which the features are flattened per sample and passed to the output layer. **Pooling** here reduces spatial size by taking a representative value from each small region. The constructor receives the class count; `forward` receives the actual image batch. The model is intended to check execution of the evaluation flow, not to establish the best architecture for a real task.

The required packages are `torch`, `torchvision`, `scikit-learn`, `numpy`, `Pillow`, and `matplotlib`. Install them in the Python environment you use, with a compatible PyTorch and torchvision combination for your operating system and GPU. Select GPU installation commands for your environment from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

The following commands work in **macOS zsh and Linux bash or zsh** from the directory where you downloaded the example file. Run them with `python` from your active Python environment.

~~~bash
python examples-en.py --demo --device cpu --output-dir runs/demo-01
~~~

`--demo` runs with synthetic data, `--device cpu` selects the CPU, and `--output-dir` specifies a new directory for the results. Existing output directories are not overwritten, so use a new directory name when running again. To use an actual image directory, use the following form and replace the path with the parent of your class folders.

~~~bash
python examples-en.py --data-root ./chest_xray/train --epochs 3 --device auto --output-dir runs/real-01
~~~

`--epochs` is the number of training epochs. `--device auto` checks availability and selects CUDA, then MPS, then CPU in that order. The default is 1 training epoch. With `--epochs 0`, the script evaluates untrained weights to check evaluation and file mapping only.

The following files are saved when `--output-dir` is specified. The output directory must be a new directory outside `data-root`. A results directory created inside the parent of the class folders could be mistaken for a new class on the next inspection.

<div class="eval-table">

| Output file | Contents |
| --- | --- |
| `summary.json` | Class mapping, preprocessing and split settings, training history, and validation metrics |
| `classification-report.txt` | Precision, recall, F1, and support for each class |
| `predictions.csv` | Result index, original index, file path, true label, and prediction |
| `errors.csv` | Only the misclassified samples from those records |
| `first-error.png` | The original image with both true and predicted names, when an error exists |

</div>

When synthetic data is preserved in the output directory, a `demo-data/` directory is created too. A synthetic run without an output directory cleans up its temporary data afterward. The values 222, 48, 12, and 765 are printed as a separate arithmetic example, distinct from the executed model's performance.

Before publication, the Korean example was run on the CPU for synthetic-data training and evaluation, and the English example was run to evaluate untrained weights. Checks covered aggregation with unequal batch sizes, file mapping through a non-sorted Subset, rejection of a shuffled validation loader for file mapping, and retaining report entries for classes absent from true labels and predictions. The execution environment used PyTorch 2.15.0+cpu and torchvision 0.29.2+cpu.

These checks establish the program's processing and correspondence between records. They do not measure actual X-ray training performance or MPS/CUDA execution performance. The example does not include saving the best weights or evaluating an independent test set. For an actual task, continue with the remaining implementation items in the previous section.

## References

[^folder]: [torchvision ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) · [Folder inspection, labels, and image-loading implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html).
[^path]: [Python pathlib](https://docs.python.org/3/library/pathlib.html).
[^transforms]: [torchvision image transforms](https://docs.pytorch.org/vision/stable/transforms.html) · [ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html).
[^normalize]: [torchvision Normalize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html).
[^weights]: [ResNet18 and preprocessing for its weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) · [Weights configuration implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/_api.html).
[^split]: [scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) · [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
[^data]: [PyTorch Dataset, Subset, and DataLoader](https://docs.pytorch.org/docs/stable/data.html).
[^iteration]: [Python iterators](https://docs.python.org/3/tutorial/classes.html#iterators) · [iter and next](https://docs.python.org/3/library/functions.html#iter).
[^loss]: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) · [Training and validation loops](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
[^autograd]: [PyTorch gradient recording and evaluation mode](https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation).
[^lists]: [Python list append and extend](https://docs.python.org/3/tutorial/datastructures.html).
[^tolist]: [Tensor.tolist](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html) · [Tensor.numpy](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html).
[^metrics]: [scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
[^cm]: [scikit-learn confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).
[^f1]: [scikit-learn f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
[^report]: [scikit-learn classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
[^image]: [Pillow image-file handling](https://pillow.readthedocs.io/en/stable/reference/open_files.html) · [Matplotlib title](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html) · [close](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.close.html).
[^assert]: [Python assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement).
