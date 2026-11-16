+++
title = "Chest X-ray Classification: From Data Inspection to CNNs and Transfer Learning"
description = "Inspect ImageFolder paths and labels, preserve sample correspondence through splitting and preprocessing, and connect training to validation. Compare a custom CNN with ResNet18 while examining loss aggregation, per-class metrics, BatchNorm, and frozen weights."
date = 2026-11-16
slug = "chest-xray-classification-pytorch"

[extra]
toc = true
styles = ["chest-xray-classification-pytorch/article.css"]
+++

Starting an image classification project with the model makes it easy to overlook what the model actually receives. A file path can be mistaken for an image, and an index used to select a sample can be confused with its class label. After training, overall accuracy alone can hide which class the model gets wrong.

Here, we build a classification experiment using chest X-rays labeled normal or pneumonia. We begin by inspecting the file list, then follow one image through model scores and evaluation. The earlier [introduction to CNNs](/en/cnn-convolution-basics/) explains convolution; this article connects those ideas to data preparation and experiment comparison.

The code in each section explains the decisions at that stage. To run the example from the beginning, use the [complete Python example](/chest-xray-classification-pytorch/examples-en.py). The file counts and performance tables come from an earlier experiment record. The final section separately describes what was checked in the new runnable example.

## 01. Define the roles of the image, label, and model output

The program takes one X-ray image and predicts either normal or pneumonia. A <strong>class</strong> is one of the categories the model distinguishes. A <strong>label</strong> is the known answer attached to an image.

The class numbers checked in this experiment are:

| Class name | Label number | Meaning |
| --- | ---: | --- |
| NORMAL | 0 | An image labeled normal |
| PNEUMONIA | 1 | An image labeled pneumonia |

These numbers are not a universal convention for image datasets. Read the data, check the actual mapping, and use that mapping consistently.

We will call an input image and its label together one <strong>sample</strong>. The model receives the image and calculates a score for each class. The label provides the reference used to evaluate those scores and train the model. The model does not read the label number from the image.

The work uses three distinct sets of data.

| Data | Purpose | Are weights updated? |
| --- | --- | --- |
| Training data | Compare predictions with labels and train the model | Yes |
| Validation data | Compare and select model structures and training settings | No |
| Test data | Measure the final performance of the selected model | No |

<strong>Weights</strong> and biases are values the model uses in its calculations. Values that training can adjust are called <strong>parameters</strong>. Settings chosen by a person, such as the learning rate or number of training epochs, are <strong>hyperparameters</strong>.

If we keep changing settings after looking at test results, the test data also becomes part of model selection. To retain a final evaluation, we need to keep model selection separate from testing.[^cv]

<figure class="xray-figure">
<img src="/chest-xray-classification-pytorch/workflow-en.svg" alt="The original train and val folders contain 5,232 images in total, split into 4,185 training and 1,047 validation images. Training updates weights, validation selects the model, and the 624 test images are evaluated after selection." loading="lazy">
<figcaption>Distinguish the original folder names from the new training and validation roles. Use the test set after choosing the model.</figcaption>
</figure>

## 02. Inspect the original folders and image counts

The dataset is [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). We start with a dataset directory containing `train`, `val`, and `test`, each with class subdirectories. First check where the files were actually extracted.

| Example path | Contents |
| --- | --- |
| `chest_xray/train/NORMAL/` | Normal images in the original training folder |
| `chest_xray/train/PNEUMONIA/` | Pneumonia images in the original training folder |
| `chest_xray/val/NORMAL/` | Normal images in the original validation folder |
| `chest_xray/test/PNEUMONIA/` | Pneumonia images in the original test folder |

The experiment record contains the following file counts. Count the files again when using another copy of the dataset instead of assuming these numbers will match.

| Original split | NORMAL | PNEUMONIA | Total |
| --- | ---: | ---: | ---: |
| train | 1,341 | 3,875 | 5,216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |
| train + val | 1,349 | 3,883 | 5,232 |

With only 16 validation images, changing the outcome for a single image changes accuracy by 6.25 percentage points. That makes small performance differences an unstable basis for selecting a model. In this experiment, the original `train` and `val` folders were combined and split into new training and validation sets. The original `test` set was kept separate.

This changes the split for the purposes of the experiment. If an official evaluation requires a fixed split, or the original split separates patients or institutions, check those conditions first. A small validation set is not a general reason to replace the original split.

## 03. Separate ImageFolder metadata from the images it returns

Here, a <strong>Dataset</strong> is an object that defines how many samples exist and what to return when given a sample index. A <strong>sample index</strong> selects a sample. It has a different role from the label number that represents normal or pneumonia.

`ImageFolder` is a torchvision Dataset class for class-based folders. It uses subdirectory names as class names, reducing the code needed to collect file paths and labels ourselves.[^folder]

Below, `data_dir` is the directory that actually contains `train`, `val`, and `test`. `Path` is a Python class for handling file paths, and the `/` operator joins path components. Joining a path does not create a folder.

~~~python
from pathlib import Path
from torchvision import datasets

data_dir = Path("chest_xray")
train_dir = data_dir / "train"
val_dir = data_dir / "val"

original_train = datasets.ImageFolder(train_dir)
original_val = datasets.ImageFolder(val_dir)

print(original_train.class_to_idx)
print(len(original_train), len(original_val))

# Check that the same integer means the same class in both datasets.
assert original_train.class_to_idx == original_val.class_to_idx
~~~

`assert` checks a condition and stops execution with an error if it is false. The check above does not compare the number of images in the two folders.

The complete `original_train` object and the information it stores have different roles.

| Expression | Type and contents | Purpose |
| --- | --- | --- |
| `original_train` | ImageFolder object | The complete rules for reading samples |
| `.classes` | List of class-name strings | Inspect the names |
| `.class_to_idx` | Dictionary mapping names to integer labels | Check what labels mean |
| `.samples` | List of `(path string, integer label)` tuples | Inspect the file list |
| `.targets` | List of integer labels in the same order as samples | Count samples per class |
| `original_train[i]` | One actual image and its label | Read a sample |

Here, `i` is a sample index. `original_train.samples[0]` retrieves a file path and label. `original_train[0]` opens the file at that path, applies any configured image transformation, and returns the image and label.

Without preprocessing, the default PIL loader returns a Pillow image. A <strong>Pillow image</strong> is a Python object for working with image files. Once we convert the image into numerical data, we receive a <strong>tensor</strong>: a PyTorch data type for storing and calculating with numbers arranged along multiple axes.

A custom Dataset follows the same division of responsibilities. `__init__` stores paths, labels, and transformation settings; `__len__` returns the number of samples; and `__getitem__` constructs and returns the requested sample. A Dataset that creates a colored square on demand and ImageFolder differ in how they obtain the image.[^data]

## 04. Combining datasets does not change the original object types

We do not need to move image files to use the original `train` and `val` sets together. We can connect the two Datasets in order and access them through a combined sample index.

`ConcatDataset` concatenates multiple Datasets. `Subset` connects to an original Dataset but exposes only selected indices. Neither tool copies the files.[^data]

~~~python
from torch.utils.data import ConcatDataset, Subset

merged_dataset = ConcatDataset([original_train, original_val])

print(type(merged_dataset))             # ConcatDataset
print(type(merged_dataset.datasets[0])) # ImageFolder
print(len(merged_dataset))              # Experiment record: 5232
~~~

`merged_dataset.datasets` is a list of the original objects being combined. The combined object is a ConcatDataset, while the two objects inside it remain ImageFolder instances.

The original training set uses sample indices 0 through 5,215. The 16 validation images follow them. Requesting combined index 5,220 therefore passes the first 5,216 samples and requests index 4 from the original validation set.

`Subset` introduces another index mapping. If we use `[5220, 8, 10]` as an example selection, index 0 in the Subset refers to index 5,220 in its source. The selection below is only an example of this mapping.

~~~python
example_subset = Subset(merged_dataset, [5220, 8, 10])

# Requesting example_subset[0] follows this mapping:
# merged_dataset[5220] → original_val[4]
print(example_subset.indices[0]) # 5220
print(example_subset.dataset is merged_dataset) # True
~~~

<figure class="xray-figure">
<img src="/chest-xray-classification-pytorch/index-map-en.svg" alt="Index 0 in an example Subset selects combined dataset index 5220, which maps to index 4 in the original val dataset. Looking up samples metadata and reading an image are separate operations." loading="lazy">
<figcaption>Track which Dataset each index refers to.</figcaption>
</figure>

Standard ConcatDataset and Subset objects do not automatically combine ImageFolder attributes such as `.targets`, `.samples`, or `.class_to_idx`. Retrieve the information from the original objects or construct it explicitly. A variable name such as `train_dataset` does not identify its type; check what object was assigned to it.

## 05. Count labels when you need class counts

To count images per class, we need the label attached to each image. There is no reason to read pixels or convert them into tensors. The stored label list already contains the information.

Below, `all_targets` is the label list for the entire combined dataset. Its name distinguishes it from `labels`, which will later hold the labels of one batch returned by a DataLoader.

~~~python
from collections import Counter

# Preserve the order used to construct ConcatDataset.
all_targets = original_train.targets + original_val.targets
counts = Counter(all_targets)
total = len(all_targets)

for class_name, class_id in original_train.class_to_idx.items():
    count = counts[class_id]
    print(f"{class_name}: {count} images, {count / total:.2%}")
~~~

`Counter` counts how often each value appears. `class_name` is a class-name string, `class_id` is its integer label, and `count` is the number of occurrences of that label. For the labels in the experiment record, there are 1,349 normal images, or about 25.78%, and 3,883 pneumonia images, or about 74.22%.

`.class_to_idx` is a dictionary attribute. `.items()` is a method for iterating over names and numbers together. `.samples` is also a stored attribute—a list—so it is not called as `.samples()`.

`Counter(merged_dataset)` would perform a different task. Iterating over a Dataset produces images read from files together with their labels, not labels alone. It would add image reading to a task that only needs counts.

Use the same principle when inspecting a few filenames: retrieve only the information the task needs.

~~~python
normal_id = original_train.class_to_idx["NORMAL"]
filenames = []

for path, target in original_train.samples:
    if target == normal_id:
        filenames.append(Path(path).name)

print(filenames[:5])
~~~

`path` and `target` are the path string and integer label unpacked from each tuple in samples. `Path(path).name` retrieves the final filename component of the path. It does not read the image.

<details>
<summary>Writing the same selection as a list comprehension</summary>

A <strong>list comprehension</strong> combines iteration, a condition, and result collection into an expression that creates a list.

~~~python
filenames = [
    Path(path).name
    for path, target in original_train.samples
    if target == normal_id
]
~~~

The reasoning stays the same: iterate over samples, select entries with the normal label, and add their filenames to the result. There is no need to begin with the shortest syntax.

</details>

## 06. A stratified split preserves class proportions

Splitting training and validation data does not change image contents. It decides which samples serve each purpose. Because a Dataset reads samples by index, we only need to store lists of selected indices.

A <strong>stratified split</strong> preserves each class's proportion as closely as possible in the resulting sets. We select a certain fraction of normal images for training and the same fraction of pneumonia images. Neither class is preferentially assigned to training, so the class composition of both new sets remains close to the original.

It does not turn a dataset with about 26% normal and 74% pneumonia images into a 50:50 dataset. Since image counts are integers, the proportions may not match exactly.[^split]

~~~python
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    range(len(merged_dataset)),
    test_size=0.2,
    stratify=all_targets,
    random_state=42,
)

train_set = Subset(merged_dataset, train_idx)
val_set = Subset(merged_dataset, val_idx)
~~~

| Argument | Meaning in this example |
| --- | --- |
| `range(len(merged_dataset))` | Sample indices from zero up to, but excluding, the total length |
| `test_size=0.2` | Assign 20% of the samples to the second returned set |
| `stratify=all_targets` | Use the label corresponding to each index to preserve class proportions |
| `random_state=42` | A random-state setting for reproducing the split with the same inputs and settings |

Although the argument is named `test_size`, we use the second result for **validation**. This code does not modify the original `test` folder. The function also does not always return indices: it returns split index lists here because indices are what we supplied.

The essential condition is that `all_targets[i]` must be the label of the sample returned by `merged_dataset[i]`. This is why we concatenate the label lists in the same order as the two source Datasets.

The counts and settings used in the experiment produce the following result.

| New split | NORMAL | PNEUMONIA | Total |
| --- | ---: | ---: | ---: |
| Training | 1,079 | 3,106 | 4,185 |
| Validation | 270 | 777 | 1,047 |

After splitting, check lengths, overlap, and class counts.

~~~python
assert len(all_targets) == len(merged_dataset)
assert len(train_idx) + len(val_idx) == len(merged_dataset)
assert set(train_idx).isdisjoint(val_idx)
assert set(train_idx) | set(val_idx) == set(range(len(merged_dataset)))

print(Counter(all_targets[i] for i in train_idx))
print(Counter(all_targets[i] for i in val_idx))
~~~

A `set` holds distinct values without duplicates. `isdisjoint` checks that two sets have no common elements, and `|` combines their elements. These checks confirm that training and validation do not overlap and together cover every source index.

`val_idx` contains indices defined relative to merged_dataset. Applying them to another Dataset, as in `Subset(original_val, val_idx)`, either gives the same numbers a different meaning or accesses an index outside that Dataset.

PyTorch's `random_split` randomly divides a Dataset and returns Subsets, but it does not deliberately preserve class proportions. Random splitting and splitting with class composition in mind are different decisions.

## 07. Distinct images do not guarantee distinct patients

To evaluate performance on images from new patients, we need to check whether different images from the same patient appear in both training and validation. Evaluating another image of a patient already seen during training can change the condition we intended to test.[^cv]

Stratification asks whether the class proportions were preserved. A patient-group split asks whether all images of a patient were assigned to the same side. These conditions need separate checks.

The strongest evidence is a mapping between patient IDs and images supplied by the dataset provider. Using filenames requires a verified rule identifying which part of each filename represents the patient.

For example, the shared text in `person1000_bacteria_2931.jpeg` and `person1000_virus_1681.jpeg` is a clue worth investigating. It does not, by itself, establish that the images belong to the same patient. A filename convention for another kind of image must not be transferred to X-rays without verification either.

The experiment record did not establish reliable patient IDs, so it used an image-level stratified split. It checked that the selected indices did not overlap; it did not establish that patients were independent across the sets.

The performance below therefore represents preliminary validation on this image split. A claim about new patients requires investigating patient information and verifying the split again. For that goal, separating patients is a condition of the evaluation, not merely an optional way to improve performance.

## 08. Preprocessing runs when a sample is read

Images with different sizes and representations are difficult to combine into batches for a model. First, we make their size consistent, convert them into tensors, and choose the floating-point type and value range used for calculations.

The basic CNN uses the following preprocessing.[^transforms]

~~~python
import torch
from torchvision.transforms import v2

def make_basic_transform():
    return v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
~~~

`Compose` is an object that runs transformations in the order listed. This function creates and returns that transformation configuration. Calling it does not immediately transform every image.

| Transformation | What changes | Condition to keep in mind |
| --- | --- | --- |
| `Resize((224, 224))` | Sets height and width to 224 | Does not preserve the original aspect ratio in general |
| `ToImage()` | Represents the image as an image tensor | Does not, by itself, scale values to 0–1 |
| `ToDtype(torch.float32, scale=True)` | Changes the numerical type and value range | Maps 0–255 to 0–1 for the current 8-bit inputs |

ImageFolder with its default PIL loader converts images to RGB. Even an X-ray that looks grayscale can therefore return a three-channel tensor. Representing the same grayscale information as RGB does not create three different sources of examination data.[^folder]

ImageFolder is the object that calls the transformation. Assigning `train_set.transform = ...` to a standard Subset does not make its sample-reading logic use that attribute. Successfully assigning an attribute and actually applying a transformation are separate facts.

The train_set and val_set created earlier share the same merged_dataset. Changing a source ImageFolder's transform affects both. To add a random transformation only to training later, we need separate objects that read the same files with different preprocessing.

The function below creates and combines two new ImageFolder objects for the original folders. It also checks that their file order matches the original metadata.

~~~python
def make_merged_dataset(transform):
    first = datasets.ImageFolder(train_dir, transform=transform)
    second = datasets.ImageFolder(val_dir, transform=transform)

    # Check that each index selects the same file and label.
    assert first.samples == original_train.samples
    assert second.samples == original_val.samples
    assert first.class_to_idx == second.class_to_idx

    return ConcatDataset([first, second])

train_source = make_merged_dataset(make_basic_transform())
val_source = make_merged_dataset(make_basic_transform())

train_set = Subset(train_source, train_idx)
val_set = Subset(val_source, val_idx)
~~~

Training and validation currently use the same kind of transformation, but their ImageFolder objects and transformation configurations are separate. We have prepared separate ways to read the same files, not duplicated the files.

Giving the original train folder a training transform and the original val folder a validation transform would not correctly represent the new split. Both new sets contain images selected from both original folders.

Read an actual sample to check whether the configuration produces what we intended.

~~~python
image, target = train_set[0]

print(type(image), image.shape, image.dtype)
print(image.min().item(), image.max().item(), target)

assert isinstance(image, torch.Tensor)
assert image.shape == (3, 224, 224)
assert image.dtype == torch.float32
assert 0.0 <= image.min().item() <= image.max().item() <= 1.0
~~~

`isinstance` checks whether an object has the specified type or a subtype of it. A v2 image tensor can be a subtype of Tensor. `.item()` extracts a Python number from a tensor containing one element.

This value-range check applies to the basic preprocessing used here. After ImageNet normalization later in the article, negative values and values greater than one can be expected.

## 09. DataLoader combines samples into batches

While a Dataset defines how to retrieve one sample, a <strong>DataLoader</strong> determines the sample order and how many samples to group together. A <strong>batch</strong> is a group of samples passed to the model in one computation.[^data]

~~~python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_set, batch_size=32, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_set, batch_size=32, shuffle=False, num_workers=0
)
~~~

`batch_size=32` puts up to 32 images in a batch. `num_workers=0` reads the data in the current Python process. It is a simple starting point for checking behavior before introducing environment-specific parallel-loading concerns.

Constructing a DataLoader does not run training. The loader requests samples from the Dataset when we iterate over it. With the ordinary configuration used here, one complete pass through the training data is called an <strong>epoch</strong>.

| Set | Images | Batches with a batch size of 32 | Batch count |
| --- | ---: | --- | ---: |
| Training | 4,185 | 130 batches of 32 + one batch of 25 | 131 |
| Validation | 1,047 | 32 batches of 32 + one batch of 23 | 33 |

`len(train_set)` is the number of images. `len(train_loader)` is the number of batches under the current configuration. The remaining images are included because `drop_last`, which controls whether to discard the last incomplete batch, defaults to False.

With the default sampling behavior, `shuffle=True` shuffles all sample indices and forms batches in that order. It does not independently draw 32 images with replacement from the whole dataset for every batch. An image and its label move together as a pair.

During training, one batch updates the weights before the next batch is processed, so order can affect the outcome. Validation evaluates every sample with the same model state and preprocessing. If each image is processed independently as it is here, changing the order should preserve the total number of correct predictions. Using `shuffle=False` for validation makes results easier to trace. If preprocessing is random or calculations depend on other samples in a batch, check those conditions again.

<details>
<summary>What iter and next each do</summary>

An <strong>iterator</strong> tracks how far a pass through the data has progressed. `iter(train_loader)` starts a new pass, and `next` retrieves its next batch.

~~~python
iterator = iter(train_loader)
imgs, labels = next(iterator) # First batch
imgs, labels = next(iterator) # Next batch in the same pass
~~~

`next` does not create a new iterator every time. In contrast, running `next(iter(train_loader))` repeatedly starts a new pass each time. It does not automatically continue after the batch retrieved previously.

</details>

## 10. Track tensor axes and image–label correspondence together

After retrieving a batch, `imgs` is an image tensor and `labels` is a tensor of known class numbers. A tensor's <strong>shape</strong> lists the size of each axis in order.

We write the image-batch shape as `(B, C, H, W)`: B is the number of images in the current batch, C is the channel count, H is the height, and W is the width.

| Axis number | Meaning | Size for a 32-image batch | Valid indices |
| ---: | --- | ---: | --- |
| 0 | Image | 32 | 0–31 |
| 1 | Channel | 3 | 0–2 |
| 2 | Height | 224 | 0–223 |
| 3 | Width | 224 | 0–223 |

The shape of `imgs` is `(32, 3, 224, 224)`, while `labels` has shape `(32,)`. A label only needs one class number per image, so its shape differs from the image tensor's.

`imgs[i]` is one image in the current batch, and `labels[i]` is that image's label. We establish this correspondence from the processing steps, rather than assuming that equal indices must refer to the same image.

1. The Dataset returns the image and label as a pair.
2. The DataLoader adds those sample pairs to the batch in the same order.
3. The current CNN preserves the image order within the batch and computes scores for each image.
4. We select a predicted class from each image's scores.

Preserving this sequence connects the image, label, and prediction at the same batch index. Shuffling only the labels or sorting only the predictions would break that correspondence.

Check axis order when displaying an RGB image too. A single PyTorch image uses `(channels, height, width)`, while Matplotlib expects `(height, width, channels)`. `permute(1, 2, 0)` reorders the existing axes to axis 1, then axis 2, then axis 0. It does not average pixels.[^tensor]

~~~python
import matplotlib.pyplot as plt

imgs, labels = next(iter(val_loader))
image = imgs[0]
class_name = original_train.classes[labels[0].item()]

plt.imshow(image.permute(1, 2, 0).cpu())
plt.title(class_name)
plt.axis("off")
plt.show()
~~~

This display code applies to the 0–1 images produced by the basic preprocessing. For a tensor with ImageNet normalization, undo the normalization before displaying it to recover a view closer to the original brightness.

If images were stored in a separate list with `images.append(image)`, that list contains image tensors only. We cannot later retrieve an image–label pair with `image, label = images[0]`. The structure used to retrieve data must match the structure we stored.

## 11. Design the CNN by working backward from the required output

Before choosing the model's layers, consider what it ultimately needs to produce. Each image needs one normal score and one pneumonia score. For B input images, the output must therefore have shape `(B, 2)`.

To calculate those scores, we first need to extract features from the image and arrange them for a classification layer. A <strong>convolutional neural network (CNN)</strong> uses convolutional layers that repeatedly apply the same weights to small image regions. A tensor that preserves the spatial arrangement of the resulting responses is called a <strong>feature map</strong>.

Here, we apply two convolutions and reduce the spatial size after each one. Finally, we flatten the features and calculate two class scores.

| Operation | Role | Output shape |
| --- | --- | --- |
| Input | Preprocessed image batch | `(B, 3, 224, 224)` |
| Conv1 + ReLU | Calculate 16 feature channels | `(B, 16, 224, 224)` |
| MaxPool | Reduce height and width | `(B, 16, 112, 112)` |
| Conv2 + ReLU | Calculate 32 feature channels | `(B, 32, 112, 112)` |
| MaxPool | Reduce height and width | `(B, 32, 56, 56)` |
| Flatten | Arrange each image's features along one axis | `(B, 100352)` |
| Linear | Calculate class scores | `(B, 2)` |

`Conv2d` combines small regions of the input channels using weights. To calculate one output channel, it sums the contributions from the input channels. Sixteen output channels do not mean sixteen images. They are sixteen kinds of features calculated with different weights for each image.[^conv]

`ReLU` is an activation function that replaces negative values with zero and preserves positive values. Adding it between layers allows the model to represent relationships beyond repeated linear calculations.

`MaxPool2d(2)` selects the largest value in each 2×2 region of each channel. The stride is also two here, so height and width are halved. That statement depends on the current even dimensions and settings. Pooling has no weights to learn.[^pool]

Sharing convolution weights across image coordinates does not mean the weights remain unchanged throughout training. Weights shared within a computation can change through backpropagation and parameter updates.

<figure class="xray-figure">
<img src="/chest-xray-classification-pytorch/model-shapes-en.svg" alt="CustomCNN preserves batch size B while a three-channel 224×224 input passes through 16-channel and 32-channel feature maps, then 100352 features and two class scores." loading="lazy">
<figcaption>The batch axis B is preserved. Channel count, spatial size, and the number of classifier inputs are separate design choices.</figcaption>
</figure>

### Flatten the features or average each channel?

The final feature map currently has shape `(B, 32, 56, 56)`. <strong>Flatten</strong> combines the channel, height, and width axes into one axis. It neither averages nor discards feature values. Preserving the batch axis keeps features from different images separate.

For the comparison below, assume `features` contains the feature map before flattening, with shape `(B, 32, 56, 56)`.

~~~python
flat_features = torch.flatten(features, start_dim=1)
# (B, 32, 56, 56) → (B, 100352)
~~~

Each image has 56×56 values in each of 32 channels, giving 100,352 features. `start_dim=1` means to begin flattening at axis 1. A four-dimensional tensor has axes numbered 0 through 3, so `flatten(x, 2, 4)` would refer to an axis that does not exist. Flatten's `end_dim` is the last included axis number.[^tensor]

Another option is <strong>global average pooling (GAP)</strong>. For each image and channel, it averages all spatial values and leaves one value per channel.

~~~python
gap = torch.nn.AdaptiveAvgPool2d((1, 1))
summary = gap(features) # (B, 32, 1, 1)
summary = torch.flatten(summary, 1) # (B, 32)
~~~

| Method | Information passed to the classifier | Tradeoff |
| --- | --- | --- |
| Flatten | Feature values at each coordinate in each channel | More classifier inputs and weights |
| GAP | The mean spatial response of each channel | Does not directly retain the coordinates of responses |

Flatten can follow GAP; they are not necessarily alternatives. Also, `AdaptiveAvgPool2d((7, 7))` leaves 49 values per channel, unlike GAP's single value per channel.[^gap]

This model uses Flatten followed by a Linear layer with two outputs. The last layer has 100,352 weights per output and one additional bias per output, for a total of 200,706 parameters. The full model has 205,794. We should inspect that scale before deciding whether the model is too large, rather than drawing that conclusion from the use of Flatten alone.

## 12. Separate layer construction from the order of computation

A PyTorch model is built on `nn.Module`. In `__init__`, we create and store the layers. In `forward`, we specify the order in which those layers process the input tensor. The order of layer declarations alone does not determine the computation.

The constructor argument `num_classes` below sets the number of output scores. In contrast, `x` passed to `forward` is the actual batch of image tensors.

~~~python
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
~~~

`self` refers to the current model object, and `self.fc` is the classification layer stored on it. In `self.pool = nn.MaxPool2d(2)`, two configures the pooling region size. In `self.pool(x)`, x is the tensor to process. Calling `self.pool(2)` does not reconfigure the region size.

To check the layer connections, first pass a dummy input through the model and inspect the output shape. `torch.randn` creates a tensor of random floating-point values with the specified shape. The `torch.no_grad()` block below avoids recording operations for backpropagation during this check.

~~~python
shape_model = CustomCNN(num_classes=2)

with torch.no_grad():
    output = shape_model(torch.randn(4, 3, 224, 224))

assert output.shape == (4, 2)
print(sum(p.numel() for p in shape_model.parameters())) # 205794
~~~

`parameters()` provides the model's parameters, and `numel()` returns the number of elements in a tensor. This check verifies layer connections and shapes. It does not verify the label mapping of real images or classification performance.

<details>
<summary>Common naming and calling errors</summary>

| Incorrect expression | Correct expression | Reason |
| --- | --- | --- |
| `super.__init__()` | `super().__init__()` | Call super before using the parent initializer |
| `F.ReLU(x)` | `F.relu(x)` | The functional API's function name is lowercase |
| `nn.Fc(...)` | `nn.Linear(...)` | The class name for a fully connected layer |
| `self.pool(2)` | `self.pool(x)` | Pass the tensor, not the configuration value |
| `fc(x)` | `self.fc(x)` | Use the layer stored on the model |

</details>

Running `model = CustomCNN(2)` again creates a model with newly initialized weights. To preserve a trained model, use a separate name for a model created for checks, and pay attention to which cells you rerun.

## 13. Scores, true labels, and predicted labels are different values

The raw scores calculated by the model are called <strong>logits</strong>. This model returns two class scores per image.

| Variable | Shape | Type and contents |
| --- | --- | --- |
| `imgs` | `(B, 3, 224, 224)` | Floating-point input image tensor |
| `logits` | `(B, 2)` | Floating-point normal and pneumonia scores |
| `labels` | `(B,)` | Integer tensor of true class labels |
| `preds` | `(B,)` | Predicted class numbers selected by comparing scores |

`logits[i, 0]` is the normal score for the image at batch index i. The first index selects an image; the second selects a class.

We can check this structure with the `shape_model` created earlier and the matching `imgs` and `labels` batch retrieved from the DataLoader. This model has not been trained yet.

~~~python
logits = shape_model(imgs)
preds = logits.argmax(dim=1)
~~~

`argmax` returns the index of the largest value. `dim=1` identifies the class-score axis. If the scores are `[1.5, 3.2]`, the result is class number 1, not the largest value, 3.2.

Logits can be negative or greater than one. When probabilities are needed, <strong>Softmax</strong> converts the scores into positive values that sum to one. This preserves score order, so we can apply argmax directly to logits when selecting a class. Adding the same number to every score also preserves their relative differences. The absolute value of the correct class's score alone is therefore not a measure of confidence.

### Choose the loss to match the meaning of outputs and labels

A <strong>loss</strong> quantifies how predictions disagree with labels according to the training objective. Our outputs are scores with shape `(B, 2)`, and our targets are class numbers with shape `(B,)`. This combination can use `CrossEntropyLoss`.[^ce]

This loss does not reshape scores and labels to match and then subtract their elements. For each image, it receives all class scores and evaluates how much probability was assigned to the class identified by the label. One label number identifies which of the two scores corresponds to the correct class, so the input shapes can differ.

~~~python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
~~~

For the integer-label approach used here, the conditions are:

- logits contains floating-point scores before Softmax.
- labels is a `torch.long` integer tensor containing zeros or ones.
- Both tensors must have the same image order and batch size and be on the same device.

Do not pass `preds` from argmax to the training loss. Keeping only the selected class number discards information about the difference between class scores. Do not apply a separate Softmax before CrossEntropyLoss either: the loss performs the necessary calculation in a numerically stable form.

The default `reduction="mean"` returns one batch average of the per-image losses. Its shape is `()`, a zero-dimensional tensor. `reduction="none"` returns the per-image losses with shape `(B,)`, while `"sum"` returns their sum.

Two-class problems can also use a different output design. For example, one score can represent whether pneumonia is present, using `BCEWithLogitsLoss`. The targets must then be floating-point zeros and ones with the same shape as the output. Merely replacing the loss function name in the current two-score design does not make that change correctly.[^bce]

## 14. train, eval, and no_grad control different things

To understand whether a model is learning, distinguish its operating mode, gradient calculation, and actual parameter updates.

A <strong>gradient</strong> describes the direction and amount by which the loss changes when a parameter changes slightly. <strong>Backpropagation</strong> follows the connections in the model's computation to calculate those gradients. An <strong>optimizer</strong> uses the gradients and its own update rules to change parameters.

| Code or setting | What it does | What it does not do by itself |
| --- | --- | --- |
| `model.train()` | Sets mode-dependent layers to training behavior | Run a training loop |
| `model.eval()` | Sets those layers to evaluation behavior | Disable gradient recording |
| `torch.no_grad()` | Disables operation recording for backpropagation inside the block | Switch the model to eval mode |
| `param.requires_grad` | Specifies whether a parameter needs gradients | Update the parameter |
| `loss.backward()` | Calculates gradients | Directly change the weights |
| `optimizer.step()` | Updates the parameters it manages | Update parameters belonging to another model |

**A model in eval mode can still record gradients and run backpropagation.** Conversely, calling train without running the training computations does not train its weights.[^autograd]

The current CustomCNN has no BatchNorm or Dropout layers whose calculations differ between training and evaluation modes. Even so, it is useful to explicitly call train for training and eval for validation. Switching to ResNet18 later makes this distinction affect the actual computation.

Writing `@torch.no_grad()` immediately above a function disables gradient recording throughout that function's execution. This syntax is called a <strong>decorator</strong>. It is suitable for an evaluation function, but applying it to an ordinary training function prevents the recording needed for backpropagation.

## 15. Put one batch's training steps in order

Training on a batch requires calculating scores with the current weights, comparing those scores with the labels, calculating gradients, and then updating the weights. Gradients left from the previous batch would accumulate with the new ones, so clear them first.

The model and tensors must be on the same computational <strong>device</strong>. We can use CUDA with an NVIDIA GPU, available MPS support on Apple Silicon, or the CPU. The following code checks what is actually available in the current Python environment.[^device]

~~~python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = CustomCNN(2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
~~~

`.to(device)` moves a model or tensor to the specified device. Move the model first, then construct the optimizer from that model's parameters. Adam is one kind of optimizer. `lr` is the <strong>learning rate</strong>, a setting that affects update size. Here, 0.001 is a starting setting, not the best value for every problem.

Create the model and optimizer when starting an experiment. Creating a new model every epoch would discard the weights learned in preceding epochs.

~~~python
model.train()
total_loss = 0.0
total_samples = 0

for imgs, labels in train_loader:
    imgs = imgs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad() # Clear only the previous batch's gradients.
    logits = model(imgs)
    loss = loss_fn(logits, labels)

    loss.backward()      # Calculate gradients of the loss with respect to parameters.
    optimizer.step()     # Use those gradients to update the parameters.

    total_loss += loss.item() * len(labels)
    total_samples += len(labels)

print("Mean training loss:", total_loss / total_samples)
~~~

`zero_grad` does not reinitialize weights. `backward` calculates gradients, and `step` performs the update. `loss.item()` extracts a Python number for logging. Replacing the tensor with `loss = loss.item()` and then trying to backpropagate from that number will not work.

This training loop contains no line that recreates the model. Each batch uses the weights updated by the preceding batch.

## 16. Why multiply a batch mean by the image count again?

The quantity we want to record is a mean loss in which every image has equal weight. The loss function directly returns the mean for each batch. When batch sizes differ, these batch means cannot simply be averaged to obtain the image-level mean.

Adding the means of a large and a small batch once each and dividing by the number of batches gives both batches equal influence. As a result, an individual image in the smaller batch has more influence.

If each image should count equally, first recover each batch's total loss from its mean. Sum those totals, then divide by the number of images actually processed. Every image then contributes with the same weight.

~~~python
total_loss += loss.item() * len(labels)
total_samples += len(labels)
epoch_loss = total_loss / total_samples
~~~

Here, `len(labels)` is the number of images in the current batch, not the dataset size or the number of batches. The final training batch contains 25 images, so its mean is multiplied by 25.

Check the conditions under which this aggregation is valid. With the default CrossEntropyLoss mean used here, without class weights or ignored targets, multiplying the batch mean by the image count recovers the loss sum. Class weights or `ignore_index` can change the mean's denominator, so the aggregation must then be reconsidered.[^ce]

The training loss is also recorded while weights change from batch to batch. It differs from evaluating the whole training set again with the final model held fixed. A loss of 0.3166 does not mean an error rate of 31.66%.

## 17. Keep weights fixed while collecting validation results

Validation evaluates the current model on images not used to train its weights. Switch the model to eval mode, disable gradient recording, and collect each batch's labels and predictions in matching order.

~~~python
all_labels = []
all_preds = []

model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        logits = model(imgs.to(device))
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())

assert len(all_labels) == len(all_preds) == len(val_set)
~~~

This code leaves labels on the CPU. It moves model inputs to device, then brings predictions from that device back to the CPU before converting them to a list. To calculate validation loss as well, move the label tensor to the same device as logits before passing it to the loss function.

`all_labels` and `all_preds` are lists for the entire validation set, with one integer per image. Entries at the same list index must describe the same image.

`append` adds the supplied object as a single element. `extend` adds each of its elements to the existing list.

| Starting list | Operation | Result |
| --- | --- | --- |
| `[0, 1]` | `append([1, 0])` | `[0, 1, [1, 0]]` |
| `[0, 1]` | `extend([1, 0])` | `[0, 1, 1, 0]` |

We use extend because the goal is to combine the batch lists into one flat list. Both methods modify the existing list; their return values should not be assigned back to the list variable. Initialize the result lists outside the batch loop so that previous batches are preserved.

### Counting results per class explains what the metrics measure

To count normal images that were correctly predicted as normal, check two conditions. First identify images whose true label is normal, then identify correct predictions among them.

A tensor indicating whether each element satisfies a condition is called a <strong>Boolean mask</strong>. Below, we move the labels from the same validation batch onto the device holding preds before comparing them.

~~~python
labels = labels.to(preds.device) # Put both tensors on the same device for comparison.
normal_mask = labels == 0
correct_mask = preds == labels

normal_total = normal_mask.sum().item()
normal_correct = (normal_mask & correct_mask).sum().item()
~~~

`labels == 0` does not modify the labels. It marks which true labels are normal. `&` checks whether both conditions are true at each matching index. `sum` counts True as one and False as zero, producing an image count.

For totals across the entire validation set, initialize the per-class counts outside the loop and add each batch's results. Grouping these calculations into tensor operations can reduce unnecessary device synchronization compared with repeatedly using Python `if` and `.item()` for individual images on a GPU.

## 18. Use the confusion matrix to identify each metric's denominator

A <strong>confusion matrix</strong> counts combinations of true and predicted labels. In scikit-learn, rows represent true labels and columns represent model predictions.[^metrics]

~~~python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

print(cm)
print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(
    all_labels,
    all_preds,
    labels=[0, 1],
    target_names=["NORMAL", "PNEUMONIA"],
    digits=4,
    zero_division=0,
))
~~~

`labels=[0, 1]` specifies the class order for the rows and columns. `target_names` supplies display names in that same order. Neither setting reassigns the label numbers themselves.

Below is the recorded validation result for the Partial model discussed later. We first lay it out as a confusion matrix to identify the groups behind the numbers.

| True label ＼ Model prediction | NORMAL | PNEUMONIA | Actual count |
| --- | ---: | ---: | ---: |
| NORMAL | 252 | 18 | 270 |
| PNEUMONIA | 16 | 761 | 777 |
| Predicted count | 268 | 779 | 1,047 |

<figure class="xray-figure">
<img src="/chest-xray-classification-pytorch/metrics-en.svg" alt="The Partial model's validation confusion matrix contains 252 correct normal predictions and 761 correct pneumonia predictions. Accuracy, per-class recall, and precision use different denominators." loading="lazy">
<figcaption>Recall starts with the true class; precision starts with the class predicted by the model.</figcaption>
</figure>

| Metric | Question to ask first | Calculation from this record |
| --- | --- | --- |
| Accuracy | How many images were classified correctly overall? | 1,013 / 1,047 |
| Normal recall | How many actual normal images were identified? | 252 / 270 |
| Pneumonia recall | How many actual pneumonia images were identified? | 761 / 777 |
| Pneumonia precision | Of the images predicted as pneumonia, how many actually had that label? | 761 / 779 |

Recall and precision can share a numerator while using different denominators. First decide which group provides the basis for the assessment.

The <strong>F1 score</strong> combines precision and recall. A low value for either makes a high F1 score difficult to obtain. In the report, `macro avg` gives each class equal weight, while `weighted avg` weights classes by their actual image counts. `support` is the number of images with each true label, not the number predicted as that class.

`zero_division=0` displays zero when a zero denominator makes a metric undefined. It does not mean the situation did not occur. Also check whether the model never predicted a particular class.[^report]

About 74% of this validation set is labeled pneumonia. Always predicting pneumonia would yield about 74% accuracy, but normal recall would be zero. This is why we inspect per-class recall and precision alongside accuracy. Stratification does not automatically eliminate this bias.

If a set contains both classes and normal recall is 1.0 while pneumonia recall is 0.0, every image was predicted as normal. That result alone does not establish an error in the loss function or metric code. The experiment record also included a case where a newly initialized model was evaluated before training. First check whether the training loop ran, whether the model was recreated afterward, and whether the optimizer manages the intended model.

## 19. Load ResNet18 weights together with their preprocessing

The custom CNN started with randomly initialized weights. We can also start with weights learned on another dataset. Learning those weights first is called <strong>pretraining</strong>, and using them for a new task is <strong>transfer learning</strong>. Further training of the imported weights on new data is <strong>fine-tuning</strong>.

ResNet18 is a CNN with residual connections that add a layer's input to a later computation. The [earlier article](/en/cnn-residual-projection/) explains residual connections. Here, we use torchvision's model and its ImageNet-pretrained weights without reimplementing the internal architecture.

~~~python
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
resnet_model = resnet18(weights=weights)

print(resnet_model.fc)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
~~~

`fc` is the final classification layer. It originally receives 512 features and calculates scores for 1,000 classes. We replace it because our task needs two scores. `in_features` is the number of features the layer receives.

The new Linear layer has newly initialized weights. Loading a pretrained model does not mean this new normal-versus-pneumonia classifier has already been trained.[^resnet]

### Match the input representation too

Pretrained weights were learned from inputs processed in a particular way. Even for the same pixels, changing the input range and reference values can change the features calculated afterward. Start by inspecting the preprocessing provided with the weights.

~~~python
resnet_transform = weights.transforms()
print(resnet_transform)
~~~

The evaluation preprocessing checked for the default ResNet18 weights follows this sequence.

| Step | Operation | Result |
| ---: | --- | --- |
| 1 | Resize the shorter edge to 256 | Resize while preserving aspect ratio |
| 2 | Take a central 224×224 crop | Fix the spatial input size |
| 3 | Convert to floating point and rescale | Values in 0–1 before channel normalization |
| 4 | Subtract channel means and divide by standard deviations | Values matched to the pretrained input convention |

The final step is <strong>normalization</strong>. This configuration uses channel means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. These values belong to the preprocessing supplied with the weights; they are not calculated from our X-ray validation set. Negative values and values greater than one can occur after normalization.[^resnet]

`weights.transforms()` is the official evaluation preprocessing. We can also use it for a quick training baseline, but it is not the only possible training procedure and does not include random augmentation. Cropping the center can discard information near image edges, so its suitability for X-rays needs separate evaluation.

For model comparisons, reuse the existing indices instead of randomly splitting the samples again. The earlier `make_merged_dataset` function checks file order against the original data, allowing preprocessing to change while sample correspondence is preserved.

~~~python
res_train_source = make_merged_dataset(weights.transforms())
res_val_source = make_merged_dataset(weights.transforms())

res_train_set = Subset(res_train_source, train_idx)
res_val_set = Subset(res_val_source, val_idx)

res_train_loader = DataLoader(
    res_train_set, batch_size=32, shuffle=True, num_workers=0
)
res_val_loader = DataLoader(
    res_val_set, batch_size=32, shuffle=False, num_workers=0
)
~~~

The same indices select different samples if the source file order changes. That is why we check both the file list and its label mapping.

## 20. Frozen, Partial, and Full specify which weights can learn

How much of a pretrained model to change is an experimental choice. This article compares three configurations.

| Name | Weights to train | Weights to keep fixed |
| --- | --- | --- |
| Frozen | The new classifier, fc | The entire existing feature extractor |
| Partial | The last feature block, layer4, and fc | The earlier feature extractor |
| Full | The entire feature extractor and fc | None |

Full also starts with pretrained weights. It does not mean training from randomly initialized weights. Both Partial and Full are forms of fine-tuning.

`requires_grad` controls whether a weight needs gradients. For Partial, we select only parameters whose names begin with `layer4.` or `fc.`.

~~~python
partial_model = resnet18(weights=weights)
partial_model.fc = nn.Linear(partial_model.fc.in_features, 2)

for name, param in partial_model.named_parameters():
    param.requires_grad = (
        name.startswith("layer4.") or name.startswith("fc.")
    )

partial_model = partial_model.to(device)
trainable_params = [
    param
    for param in partial_model.parameters()
    if param.requires_grad
]
partial_optimizer = torch.optim.Adam(trainable_params, lr=0.0001)
~~~

`named_parameters()` provides each name string together with the actual parameter object. Names include `conv1.weight`, `layer4.0.conv1.weight`, and `fc.bias`.

`startswith` checks whether a string begins with the specified text. The True or False result of the condition is assigned to `requires_grad`. We then collect the parameters marked True in a list and pass them to this model's optimizer.

The essential connection is between the model that calculated the scores and the model whose parameters the optimizer manages. Calculating a loss from `partial_model(imgs)` and then calling `optimizer.step()` on the CustomCNN optimizer does not train the intended ResNet18 parameters.

The recorded starting learning rates were 0.001 for Frozen, 0.0001 for Partial, and 0.00001 for Full. These are not prescribed answers for the three training scopes. A very small learning rate applied to the whole model can also make the new classifier learn slowly. Even when every experiment runs for one epoch, conditions other than the trainable scope can affect performance.

For independent comparisons, start each model again from the same pretrained weights. Training Frozen and then switching that same model to Partial is also possible, but it is a separate experiment that progressively expands the training scope.

## 21. Separate frozen weights from fixed BatchNorm statistics

ResNet18 contains <strong>Batch Normalization, or BatchNorm</strong>, layers that adjust the distribution of feature values. Before considering the formulas, distinguish two kinds of state.[^batchnorm]

| State | What it contains | How it changes |
| --- | --- | --- |
| Trainable parameters | Weights and biases that adjust values | Updated through gradients and the optimizer |
| Running statistics | Values reflecting means and variances observed during training | Updated during forward passes in training mode |

Even when parameters have `requires_grad=False`, the running means and variances can change in training mode. These statistics are not updated by the optimizer in the same way as ordinary weights.

To preserve the existing feature-extraction behavior in Frozen, keep both the weights and the statistics fixed. Partial also requires a decision about which statistics may change.

### Option A: Keep all running statistics fixed

~~~python
partial_model.eval()

for imgs, labels in res_train_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    partial_optimizer.zero_grad()
    loss = loss_fn(partial_model(imgs), labels)
    loss.backward()
    partial_optimizer.step()
~~~

This code trains in eval mode. Gradient recording remains enabled, layer4 and fc are trainable, and the optimizer updates those parameters. The configuration keeps BatchNorm running statistics fixed while training the selected weights.[^autograd]

### Option B: Also update statistics in the last trainable block

~~~python
partial_model.eval()
partial_model.layer4.train()
partial_model.fc.train()
~~~

Running the same training loop after these mode settings keeps the earlier blocks in evaluation behavior and puts layer4 in training behavior. The standard ResNet18 fc layer is Linear, so its own calculation does not change between train and eval modes. Whether its weights are trainable is configured separately.

Full can likewise use train mode throughout to update running statistics, or eval mode with gradient recording enabled to keep those statistics fixed. Neither policy is universally better. Record which one the experiment used.

When a validation function calls `model.eval()`, that state persists after validation ends. Reapply the experiment's mode policy at the start of the next epoch. Unconditionally calling `model.train()` for every model could change BatchNorm statistics that were intended to remain fixed.

## 22. Decide what the recorded results can support

The earlier preliminary experiment used 1,047 validation images: 270 normal and 777 pneumonia. The values below organize and calculate metrics from that record. No Full result or final test result was provided.

Confusion-matrix rows are true labels and columns are predictions, in the order NORMAL, PNEUMONIA.

| Model | Confusion matrix | Accuracy | Normal recall | Pneumonia recall | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| CustomCNN | `[[250,20],[24,753]]` | 95.80% | 92.59% | 96.91% | 0.9454 |
| Frozen | `[[233,37],[24,753]]` | 94.17% | 86.30% | 96.91% | 0.9227 |
| Partial | `[[252,18],[16,761]]` | 96.75% | 93.33% | 97.94% | 0.9575 |
| Full | No recorded result | — | — | — | — |

The CustomCNN confusion matrix was reconstructed from 250 correct normal predictions and 753 correct pneumonia predictions recorded for the same validation samples. The Frozen and Partial confusion matrices were supplied directly in the record.

Partial performed best among the observed results. Compared with Frozen, it correctly classified 19 more normal images and eight more pneumonia images. Total errors decreased from 61 to 34. These are image counts. Without verified patient IDs, they cannot be described as patient counts.

There is a possible explanation for the difference. Frozen preserved the existing feature weights, while Partial could adapt the last feature block to X-rays. The record does not establish this as the only cause of the improvement.

Even with the same samples, preprocessing, learning rate, initialization of the new classifier, batch order, BatchNorm policy, and training amount can affect the results. This record does not establish that Partial is always best or that transfer learning always outperforms a custom CNN.

For subsequent comparisons, record at least the following conditions.

| Record | What it establishes |
| --- | --- |
| File list and split indices | Whether evaluation used the same images |
| Class-number mapping | Whether score axes and labels have the same meaning |
| Preprocessing | How input size, cropping, and normalization differ |
| Initial weights and trainable scope | What changed from the starting point |
| Learning rate, epochs, and batch size | How much training was performed |
| BatchNorm policy | Whether running statistics also changed |
| Selection metric | The criterion used to choose a model |

At this stage, the priority is to understand the one-epoch preliminary results and complete the comparison procedure. They should not be interpreted as extensive repeated experiments or validation on independent patients.

## 23. Multiple epochs and saving require managing different states

Multiple epochs repeat passes through the training data with the same model and optimizer. Validate after each pass and preserve the model state when the chosen metric improves.

Separate what needs initialization from what needs to persist.

| Item | When to create or reset it | How long to keep it |
| --- | --- | --- |
| Model and optimizer | At the start of an independent experiment | Across multiple epochs |
| Epoch loss and sample count | At the start of an epoch | Across all batches in that epoch |
| All validation labels and predictions | At the start of validation | Across all batches in that evaluation |
| Gradients | Each batch in the ordinary setup used here | For the current backpropagation and update |

`state_dict` is a dictionary containing model parameters and stored state such as BatchNorm statistics. It can be saved to a file.[^saving]

~~~python
torch.save(model.state_dict(), "custom_cnn_weights.pt")
~~~

This file alone does not automatically restore the model architecture, preprocessing, class numbers, train or eval mode, or requires_grad settings. Preserve the other settings needed when using the model again.

Be careful with simple assignment when keeping the best model in memory.

~~~python
import copy

best_weights = copy.deepcopy(model.state_dict())
~~~

`deepcopy` makes an independent copy. Using only `best_weights = model.state_dict()` can leave references to values that change during subsequent training. Another option is to save directly to a file whenever the validation metric improves.

To resume training, also preserve optimizer state, the completed epoch, and random-number state. Save the original file list and class mapping alongside split indices. If file order changes, the same indices no longer reproduce the same experiment.

Fixing the split with `random_state` and setting PyTorch randomness for operations such as model initialization with `torch.manual_seed` are separate steps. A fixed seed still does not guarantee identical calculations across different library versions and devices.[^randomness]

### Diagnose execution problems by checking each kind of state

Selecting a GPU does not automatically move the model and tensors onto it. Check their actual devices with:

~~~python
print(device)
print(next(model.parameters()).device)
print(imgs.device)
~~~

Data processing includes file reading, image decoding, preprocessing, batch assembly, device transfer, and model computation. When training is slow, identify which stage takes the time. Reading many small files from a network drive is one possible cause, but slow execution alone does not establish it.

Rerunning only the training loop in a notebook can continue from the current weights. Rerunning model construction starts with initial weights again. Recreating only the optimizer resets Adam's internal state.

In environments such as Colab, a session reset can remove the model from memory even when the code file is still saved. If `FileNotFoundError` occurs, first check the drive connection and current paths. Scores left in a notebook cannot reconstruct unsaved weights.

## 24. Check the full workflow with the runnable example

The [Python example with Korean comments](/chest-xray-classification-pytorch/examples-ko.py) and [Python example with English comments](/chest-xray-classification-pytorch/examples-en.py) use the same processing structure. Run either in a Python environment with PyTorch, torchvision, scikit-learn, NumPy, and Pillow installed.

Set `--data-dir` to the actual directory containing `train`, `val`, and `test`. The commands below use zsh or bash on macOS or Linux. `python` must refer to the environment where the required packages are installed.

First, check the code connections without real X-ray images.

~~~bash
python examples-en.py --self-check
~~~

This option checks dataset connections and model behavior using temporary image folders. It is not an experiment measuring classification performance.

To train the custom CNN for one epoch:

~~~bash
python examples-en.py --data-dir ./chest_xray --model custom --epochs 1 --output runs/custom-best.pt
~~~

For Partial transfer learning with fixed BatchNorm running statistics:

~~~bash
python examples-en.py --data-dir ./chest_xray --model partial --epochs 3 --bn-policy fixed --output runs/partial-best.pt
~~~

| Option | Meaning |
| --- | --- |
| `--model custom` | Train the custom CNN from newly initialized weights |
| `--model frozen` | Train only the new ResNet18 classifier |
| `--model partial` | Train ResNet18's layer4 and fc |
| `--model full` | Fine-tune the entire ResNet18 |
| `--bn-policy fixed` | Keep running means and variances fixed while training the selected weights |
| `--bn-policy adapt` | Update running statistics in layer4 for Partial, or throughout the model for Full |
| `--device auto` | Choose an available device in the order CUDA, MPS, CPU |
| `--output` | A new file path for saving the selected model |

Frozen keeps the existing feature extractor's running statistics fixed. The default training command does not read test data. It keeps an independent copy of the weights from the epoch with the highest validation Macro F1, then saves them to a file after training finishes. If the session ends during training, that run will not leave a checkpoint file. Existing output files are not overwritten, so use a new path for each experiment.

After finishing training configuration and model selection, evaluate the saved model on the final test set.

~~~bash
python examples-en.py --data-dir ./chest_xray --evaluate-test runs/partial-best.pt
~~~

This command loads the selected checkpoint and evaluates test data without training again. Continually changing the model after seeing test results would make the test set part of model selection, so record the purpose of each evaluation.

A <strong>checkpoint</strong> is a saved file used to reload a model or preserve training state. The example stores model and optimizer state, class numbers, relative file paths, split indices, preprocessing, training settings, and evaluation records. Its command-line functions provide training and test evaluation of a saved model; they do not provide exact training resumption from an interruption.

### What was checked in the runnable example

The example was checked with synthetic RGB image folders using PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, and scikit-learn 1.8.0. Checks covered data splitting and label correspondence, independent preprocessing objects, the final smaller batch, CNN outputs and actual weight changes, evaluation-loss aggregation, ResNet18's trainable scope, and changes in BatchNorm statistics.

The workflow was also run for two epochs without a test folder, saving the best validation model and loading it again. A separately created test folder was then evaluated with the saved model.

ResNet structure checks used `weights=None` to avoid downloading pretrained weights. The actual transfer-learning command loads the specified pretrained weights. These checks verify program connections and state changes. They do not reproduce the earlier performance figures on real X-ray data.

## 25. Keep the reasoning useful for an unfamiliar dataset

The recurring question in this experiment was what information we have and what we intend to change. Distinguishing paths from images avoids unnecessary reads. Knowing which dataset an index refers to allows us to keep tracking the same sample. Giving each image equal weight also requires checking the denominator when combining averages.

| Learning scope | What to learn now |
| --- | --- |
| Essential in practice | Inspect folders and labels, choose split criteria, verify preprocessing and input/output shapes, separate training from validation, evaluate each class, and save results |
| Understand the principle | Information retained by Flatten versus GAP, denominators in averages, and frozen weights versus fixed BatchNorm statistics |
| Extend later | Repeated experiments, learning-rate searches, augmentation comparisons, loading optimization, and mixed precision |

If the goal is evaluation on independent patients, verifying patient information is not an optimization to postpone. Establish the evaluation conditions required by the goal first.

Being able to explain the following questions and check them with small pieces of code is a useful stopping point for the basic structure covered here.

1. Why use `.targets` when only class counts are needed?
2. Does Subset index 0 always refer to the same sample as index 0 in the original Dataset?
3. If training and validation share an ImageFolder, which samples are affected by changing its transform?
4. When the final batch is smaller, which images gain more influence if batch means are averaged directly?
5. Under what conditions can layer4 weights learn while the model is in eval mode?
6. How can high overall accuracy and low normal recall occur together?

<details>
<summary>Check the answers against your own explanation</summary>

1. The required information is each sample's label, not its pixels. Counting targets does not add image-file reading.
2. A Subset index refers to its selection list. The first value in that list gives the index to select in the original Dataset.
3. The change affects both sets that read through the shared source. Different transformations require separate objects that preserve sample correspondence.
4. Equal weight for large and small batches gives an individual image in the smaller batch more influence.
5. The relevant weights must have requires_grad set to True, gradient recording must be enabled, and backward and the corresponding model optimizer's step must run.
6. Correctly predicting the larger class can produce high overall accuracy. Accuracy uses all images as its denominator, while normal recall uses only images whose true label is normal.

</details>

For the next experiment, change one condition and compare results on the same validation samples. Before making the model more complex, complete the workflow while being able to explain the data correspondence, the states being updated, and the evaluation criterion.

## References

Official API documentation was checked on November 16, 2026. Image counts and preliminary performance figures come from the earlier experiment record. The original X-ray files and earlier model weights were not reevaluated while preparing this article.

[^folder]: torchvision, [ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) · [Implementation source](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html) — Class ordering, metadata, RGB loading, and sample retrieval.
[^data]: PyTorch, [Dataset, ConcatDataset, Subset, and DataLoader](https://docs.pytorch.org/docs/stable/data.html) — Dataset connections, sample selection, batching, and shuffling.
[^split]: scikit-learn, [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — Stratification and returned results.
[^cv]: scikit-learn, [Cross-validation and grouped data](https://scikit-learn.org/stable/modules/cross_validation.html) — Model selection, final evaluation, and patient-group separation.
[^transforms]: torchvision, [v2 image transforms](https://docs.pytorch.org/vision/stable/transforms.html) · [ToDtype](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html) — Tensor representation, numerical types, and value ranges.
[^tensor]: PyTorch, [flatten](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html) · [permute](https://docs.pytorch.org/docs/stable/generated/torch.permute.html) — Combining and reordering axes.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — Channels, weights, and output shapes.
[^pool]: PyTorch, [MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) — Pooling regions and stride.
[^gap]: PyTorch, [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html) — Specifying the output spatial size.
[^ce]: PyTorch, [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — Logits, target representations, and mean denominators.
[^bce]: PyTorch, [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) — Matching output and floating-point target shapes.
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) — Differences between requires_grad, no_grad, and evaluation mode.
[^device]: PyTorch, [CUDA](https://docs.pytorch.org/docs/stable/notes/cuda.html) · [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) — Tensor computation on each device.
[^metrics]: scikit-learn, [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) — True-label rows and predicted-label columns.
[^report]: scikit-learn, [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) — Per-class metrics, averages, and zero denominators.
[^resnet]: torchvision, [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) — Pretrained weights and evaluation preprocessing.
[^batchnorm]: PyTorch, [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) — Trainable parameters and running statistics.
[^saving]: PyTorch, [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) — state_dict, independent copies, and checkpoints.
[^randomness]: PyTorch, [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) — Random-state settings and environment-dependent limits.

