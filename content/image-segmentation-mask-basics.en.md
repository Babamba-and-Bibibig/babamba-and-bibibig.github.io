+++
title = "Image Segmentation Basics: Array Axes, Mask Conversion, and Augmentation"
description = "Start with the difference between RGB images and class masks, then connect axes, Boolean indexing, connected components, one-hot encoding, and argmax. Learn how to preserve masks during augmentation and prepare them for PyTorch."
date = 2026-11-25
slug = "image-segmentation-mask-basics"

[extra]
toc = true
styles = ["image-segmentation-mask-basics/article.css"]
+++

In image segmentation code, arrays can represent the same image while having different shapes. One variable has shape `(H, W, 3)`, another `(H, W)`, and another `(C, H, W)`. If we copy the code without understanding these differences, it is easy to miss why `axis=-1` is needed or what `mask[condition] = 2` actually changes.

The first question is **what each pixel stores, and how many values it stores**. Three color components, one class ID, and one membership value per class require different operations. Once we know the output structure we need, we can decide which axes to keep and which to reduce.

This article builds a small image containing sky, ground, and two balls. We use the same example throughout to connect array selection, mask conversion, visualization, and augmentation. We then check the formats needed to pass the mask into PyTorch. The runnable script is available as [examples.py](/image-segmentation-mask-basics/examples.py).

## 01. Images and masks store different information about the same pixels

**Segmentation** assigns each image pixel to a region or category. A **class** is a category we want to distinguish. A **mask** is an array that stores information such as a class ID or a selection flag for each pixel. The word “mask” alone does not specify one particular data type or shape.

For this example, we define the class IDs as `0=sky`, `1=ground`, and `2=ball`. The size of an ID does not indicate a category's size or importance. Nor does `0` always mean background. It means sky in our semantic mask; another dataset may use a different convention.

`H` is the image height and `W` is its width. **shape** lists the size of each axis in order, and an **axis** is a dimension along which we index an array. In the table below, `C` is the number of classes, not the number of RGB components.

<div class="seg-table">

| Variable | Shape | Data type | Information stored for one pixel |
| --- | --- | --- | --- |
| `toy` | `(H, W, 3)` | `uint8` | Three R, G, and B color values |
| `semantic` | `(H, W)` | `uint8` | One class ID |
| `ball` | `(H, W)` | `bool` | `True` for a ball pixel, otherwise `False` |
| `onehot` | `(C, H, W)` | `uint8` | A membership value of 0 or 1 for each class |

</div>

`uint8` stores integers from 0 to 255. `bool` stores the truth values `True` and `False`. With only three classes in this example, `uint8` is sufficient for the class mask.

In the RGB image, `toy[y, x]` contains three color values. At the same pixel, `semantic[y, x]` contains one class ID. Here, `y` is a row index and `x` is a column index. Matching height and width is necessary, but it is not the whole condition connecting these arrays. **The same row and column indices must refer to the same image pixel.** Flipping the image without its mask breaks this correspondence even though both shapes remain unchanged.

**Semantic segmentation** gives every ball the same class ID. **Instance segmentation** also distinguishes the individual balls. The `2` in a semantic mask identifies the category “ball”; an instance ID identifies an individual region.

<figure class="seg-figure">
  <img src="/image-segmentation-mask-basics/representations-en.svg" alt="The same two balls represented as an RGB image, a class mask, and connected-region labels" loading="lazy" />
  <figcaption>Colors, class IDs, and individual region IDs are different kinds of information. Colors added for display are also distinct from the numbers stored in a mask.</figcaption>
</figure>

## 02. Create the source array, then inspect its axes and indices

We will create an image with solid colors so that the example runs without external files. NumPy is an array-processing library, imported under the short name `np`. OpenCV is an image-processing library, imported in Python as `cv2`.

`np.zeros` creates an array of the requested shape and fills it with zeros. We first fill the image with the sky color, change the lower rows to green, and draw two balls.

~~~python
import numpy as np
import cv2

H, W = 120, 200
CLASS_NAMES = ["sky", "ground", "ball"]
C = len(CLASS_NAMES)  # List indices and class IDs use the same convention.
BALL_RGB = (230, 60, 50)

toy = np.zeros((H, W, 3), dtype=np.uint8)
toy[:] = (150, 200, 255)
toy[90:] = (90, 160, 70)

# Circle centers use (x, y); -1 fills the interior.
cv2.circle(toy, (55, 60), 25, BALL_RGB, -1)
cv2.circle(toy, (145, 60), 20, BALL_RGB, -1)

print(toy.shape, toy.dtype, toy.size)
# (120, 200, 3) uint8 72000
~~~

`toy.size` is the total number of array elements. Each RGB component counts as an element, so this differs from the pixel count, `120 × 200 = 24,000`. To compare only the spatial dimensions of an image and a mask, compare `toy.shape[:2]` with `semantic.shape`.

Every expression below starts from the same `toy` array. **An integer index selects one entry and removes that axis; a slice selects a range while retaining the axis.** Omitted trailing axes are selected in full.[^indexing]

<div class="seg-table">

| Expression | What it selects | Result shape |
| --- | --- | --- |
| `toy[0]` | All pixels in the first row | `(200, 3)` |
| `toy[0:1]` | The first row, retaining a row axis of length 1 | `(1, 200, 3)` |
| `toy[:, 0]` | All pixels in the first column | `(120, 3)` |
| `toy[:, :, 0]` | The R component of every pixel | `(120, 200)` |
| `toy[60, 55]` | RGB values at row 60, column 55 | `(3,)` |
| `toy[90:]` | Rows from index 90 through the last row | `(30, 200, 3)` |
| `toy[:, 90:]` | Columns from index 90 through the last column | `(120, 110, 3)` |

</div>

`start:stop` includes the start index and excludes the stop index. `toy[90:]` specifies a range only for the first axis, so it selects rows. `toy[:, 90:]` keeps all rows and selects a range of columns on the second axis.

NumPy selects pixels as `toy[y, x]`, while OpenCV takes a circle center as `(x, y)`. Keep those orders separate. Here, we interpret the array we created as RGB. `cv2.circle` writes the component values in the order supplied, so this convention works. In contrast, a typical color image read with `cv2.imread` uses BGR order and needs conversion when used as RGB.[^opencv-image]

There is another practical distinction: basic slicing produces a **view**, which shares data with the original array. To edit a crop independently, make a copy with `crop = toy[20:80, 30:100].copy()`. Without `.copy()`, editing `crop` can also change the corresponding region in the original image.[^indexing]

## 03. Comparing three RGB values produces three Boolean results

We used exactly the same RGB color for every ball pixel. To find the balls, we can check whether all three components of each pixel match that color. In real photographs, lighting and shadows change an object's colors, so we use this approach only for the solid-color image we just created.

First, compare the individual components.

~~~python
result = toy == BALL_RGB

print(result.shape, result.dtype)
# (120, 200, 3) bool
~~~

`==` compares corresponding elements. It does not treat the R, G, and B values as one indivisible item. A pixel with values `(230, 60, 50)` produces `[True, True, True]`. If only the last component differs, the result is `[True, True, False]`.

The image contains many elements, but `BALL_RGB` contains only three values. **Broadcasting** makes this comparison possible. When arrays have different shapes, a value on an axis of size 1 can be applied repeatedly along the corresponding axis of the other array. NumPy does not need to build a huge copied array first.[^broadcasting]

Compare the shapes from right to left. A pair of axes is compatible if their sizes are equal or one of them is 1. Missing leading axes are treated as having size 1. For this example:

<div class="seg-table">

| Operand | Row axis | Column axis | Component axis |
| --- | --- | --- | --- |
| `toy` | 120 | 200 | 3 |
| How the three RGB values align | 1 | 1 | 3 |

</div>

The component-axis sizes match. Along the row and column axes, the same three RGB values can be reused. Every pixel's R component is therefore compared with 230, its G component with 60, and its B component with 50. This also explains why `toy[:] = (150, 200, 255)` can fill the entire image.

This rule does not depend on the meaning of an image. NumPy works from axis sizes. When adding or multiplying another array with an image, compare the two shapes from the rightmost axis to determine whether the operation is supported.

## 04. axis determines which elements are grouped for a calculation

`result` currently stores three Boolean results per pixel. We want `ball` to store one result per pixel. We must not mix results from different pixels: **only the three components at the same row and column indices belong together.**

`np.all` checks whether all the supplied values are true. With `axis=-1`, it groups values along the last axis. That axis corresponds to the RGB components, so it matches the operation we need.[^all]

~~~python
ball = np.all(result, axis=-1)

print(ball.shape, ball.dtype)
# (120, 200) bool
~~~

The row and column indices remain. For each row–column combination, NumPy checks the three values along the component axis and produces one result. It does not turn the entire `ball` array into a single truth value. The result is a two-dimensional array containing a decision for every pixel.

<figure class="seg-figure">
  <img src="/image-segmentation-mask-basics/axis-all-en.svg" alt="Keeping each row and column fixed while all combines three RGB comparison results into one Boolean decision per pixel" loading="lazy" />
  <figcaption>The component axis is reduced. Keeping the row and column axes preserves the link to each original image pixel.</figcaption>
</figure>

A calculation that combines several elements into one result, such as a sum or maximum, is called a **reduction**. `sum`, `mean`, `max`, and `all` are examples. By default, the reduced axis disappears and the other axes remain. `argmax` also compares values along an axis and returns one index from that axis, but its result has a different meaning from the value returned by `max`.

For `result` with shape `(120, 200, 3)`, we can predict the following output structures.

<div class="seg-table">

| Operation | Elements grouped in one calculation | Remaining axes and shape |
| --- | --- | --- |
| `result.sum(axis=0)` | All rows at the same column and component indices | Column, component: `(200, 3)` |
| `result.sum(axis=1)` | All columns at the same row and component indices | Row, component: `(120, 3)` |
| `result.all(axis=2)` | All components at the same row and column indices | Row, column: `(120, 200)` |
| `result.all()` | The entire array | One result |

</div>

`axis` does not select a limited region of the array. It specifies which elements are grouped for each combination of indices on the other axes. To limit the region being processed, select it with indexing first.

<details>
<summary>Applying the same reasoning to four axes</summary>

Suppose `a` has shape `(3, 4, 30, 3)`. `a.sum(axis=2)` sums the 30 elements along axis 2. For one sum, the indices on axes 0, 1, and 3 stay fixed while only the index on axis 2 changes.

One such sum uses the 30 elements in `a[0, 0, :, 0]`. The other axes have other index combinations, so the same calculation runs separately for each combination. There are `3 × 4 × 3 = 36` results, and the output shape is `(3, 4, 3)`.

`a[0, 0, :, 0].sum()` selects just one of these combinations and produces one result. `a.sum(axis=2)` produces results for every combination. The grouping does not depend on whether stored values happen to be equal; it depends on axes and indices.

</details>

To retain the reduced axis with size 1, use `keepdims=True`. For example, `result.all(axis=-1, keepdims=True)` has shape `(120, 200, 1)`. That shape can be useful for broadcasting a later operation back across all three components of each pixel.

## 05. Separate creating a condition from changing values

Now that we have found the ball pixels, we can store their class IDs. Filling `semantic` with zeros initially marks every pixel as sky. We change the ground rows to 1 and the ball pixels to 2.

~~~python
semantic = np.zeros((H, W), dtype=np.uint8)
semantic[90:] = 1
semantic[ball] = 2

print(np.unique(semantic))  # Inspect the distinct stored values.
# [0 1 2]
~~~

`semantic[ball] = 2` does not compare the values of the two arrays. It selects the row and column indices where `ball` is true and writes 2 into `semantic` at those indices. Unselected values stay unchanged. Selecting elements with a true/false array is called **Boolean indexing**.[^indexing]

<div class="seg-table">

| Expression | Operation | Result or change |
| --- | --- | --- |
| `semantic == 2` | Check which pixels are balls | An `(H, W)` Boolean array |
| `semantic[semantic == 2]` | Extract the stored values at ball pixels | A one-dimensional array containing only 2s |
| `semantic[semantic == 2] = 1` | Change values at ball pixels | Modifies the original `semantic` array |

</div>

The last row illustrates the difference between these expressions. In the code that follows, we continue to use 2 as the ball class ID.

`semantic == 2` needs no `np.all`. `semantic` stores one value per pixel, so the comparison already produces one Boolean result per pixel. By contrast, `toy == BALL_RGB` produces three results per pixel and needs a step that combines them.

An integer array containing 0s and 1s is not the same as a Boolean array. Inside `[]`, an integer array supplies integer indices. Use `binary.astype(bool)` **only when the intended rule is to exclude zeros and select every nonzero value**. To find only balls in a class mask, state the condition explicitly with `semantic == 2`.

## 06. Separate connected regions to distinguish the two balls

`ball` currently marks every ball pixel as true. To handle each ball separately, we can group pixels that are connected to one another and give each group its own number. Such a group is a **connected component**.

`cv2.connectedComponents` takes an 8-bit, single-channel array. Converting a Boolean array to `uint8` turns `False` into 0 and `True` into 1. We explicitly set `connectivity=8`, so pixels touching diagonally also count as connected. With `4`, only pixels sharing a side—above, below, left, or right—count as connected.[^components]

~~~python
num_labels, labeled = cv2.connectedComponents(
    ball.astype(np.uint8), connectivity=8
)

print(num_labels)       # 3: includes the background label.
print(labeled.shape)    # (120, 200)
print(labeled.dtype)    # int32

instances = [labeled == i for i in range(1, num_labels)]
print(len(instances))   # 2

# Do not pass an empty list to stack when an image contains no balls.
if instances:
    instance_masks = np.stack(instances, axis=0)
else:
    instance_masks = np.zeros((0, H, W), dtype=bool)

print(instance_masks.shape)  # (2, 120, 200)
~~~

`num_labels` counts the labels, including background label 0. `labeled` stores a region ID at each pixel. Here, background means **every pixel that is zero in the current input `ball`**, which includes both sky and ground. It is not the same region as class 0, sky, in `semantic`.

`range(1, num_labels)` excludes 0 and includes the last non-background label. `instances` is a Python list of `(H, W)` Boolean arrays. `np.stack` combines arrays of the same shape along a new axis. The size of that new first axis is the number of individual regions, not the number of classes. When no region is present, explicitly using shape `(0, H, W)` keeps the structure consistent for later processing.[^stack]

To reconstruct the semantic mask, give every individual ball the same class ID.

~~~python
rebuilt = np.zeros((H, W), dtype=np.uint8)
rebuilt[90:] = 1
for instance_mask in instances:
    rebuilt[instance_mask] = 2

assert np.array_equal(rebuilt, semantic)
~~~

`assert` checks a condition and stops execution if it is false. `np.array_equal` checks that two arrays have the same shape and values.

This method does not solve instance segmentation in general. Touching balls can form one connected component, and a gap within one ball's mask can split it into several components. Noise can also form a separate component. A component ID does not encode an object category or guarantee a fixed left-to-right order. We interpret the component count as the ball count only under **this example's condition that each separate ball corresponds to exactly one connected region**.

## 07. One-hot encoding replaces class IDs with class membership

`semantic[y, x]` currently contains one class ID. For some operations, it is more convenient to keep a separate mask for each class. **One-hot encoding** represents each pixel with 1 for its class and 0 for every other class.

First, build a lookup table for our three classes. `np.eye(C)` creates a `(C, C)` array whose diagonal entries are 1 and whose other entries are 0. We can use its row indices as class IDs.[^eye]

~~~python
class_vectors = np.eye(C, dtype=np.uint8)
print(class_vectors)
# [[1 0 0]
#  [0 1 0]
#  [0 0 1]]

# With real data, validate the IDs before indexing a lookup table.
assert np.issubdtype(semantic.dtype, np.integer)
assert semantic.min() >= 0 and semantic.max() < C

onehot_hwc = class_vectors[semantic]       # (H, W, C)
onehot = onehot_hwc.transpose(2, 0, 1)    # (C, H, W)
~~~

`class_vectors[semantic]` uses integer array indexing. Each integer stored in `semantic` becomes a row index into the lookup table. The original pixel stored one ID; the lookup returns an array of length `C` for that ID. The shape therefore changes from `(H, W)` to `(H, W, C)`.

`transpose(2, 0, 1)` arranges the original axes in the order 2, 0, 1. It does not mix the stored values or invent new class IDs. After this change, `onehot[2]` is the ball class's `(H, W)` mask.

<figure class="seg-figure">
  <img src="/image-segmentation-mask-basics/onehot-roundtrip-en.svg" alt="A separate 3-by-4 class mask converted into three one-hot class masks and restored to the original class IDs with argmax" loading="lazy" />
  <figcaption>This separate 3×4 mask makes every stored value readable. The same row and column indices still correspond after splitting it into class masks, and argmax along the class axis restores the original IDs.</figcaption>
</figure>

This one-hot representation assumes that each pixel belongs to exactly one class and that class IDs are integers from 0 through `C-1`. If real data contains values such as 255 or -1 for unlabeled regions, define how to handle them first. In particular, -1 selects the last row in NumPy. Without validation, it can silently map an unlabeled pixel to the wrong class.

## 08. argmax returns the index of the largest value

To recover a class ID from one-hot data, find the class containing 1 at each pixel. We constructed the table so that its class indices match the original class IDs. Selecting the index of the largest value therefore recovers the ID.

`max` returns the largest **value**; `argmax` returns the **index** where that value is stored. The class axis of `onehot` is axis 0, so that is the axis to compare.[^argmax]

~~~python
back = onehot.argmax(axis=0)

print(back.shape)  # (120, 200)
print(np.array_equal(back, semantic))  # True

# np.eye guarantees that this onehot array contains only 0s and 1s.
assert np.all(onehot.sum(axis=0) == 1)
~~~

If `onehot[:, y, x]` at one pixel is `[0, 0, 1]`, the maximum value is 1 and its index is 2. Across the entire array, the same operation keeps each row and column fixed while comparing the class axis. That is why the output shape is `(H, W)`.

`argmax` does not validate whether the input is correctly encoded as one-hot data. If every value is 0 or several values tie for the maximum, it returns the first maximum's index. Check the representation rules before using it to restore a class mask. On model output scores, `argmax` selects the highest-scoring class; whether that prediction is correct must be checked separately against the target.

## 09. Use class IDs as a common reference when counting pixels

Pixel counts by class are immediately useful for inspecting a dataset. If a class is almost absent or much more common than expected, that can prompt a check of mask conversion or data selection. These are **pixel counts, not object counts**.

We can find the ball class ID in `CLASS_NAMES` because we explicitly made the list indices match the actual class IDs.

~~~python
ball_id = CLASS_NAMES.index("ball")  # 2

count_from_mask = int((semantic == ball_id).sum())
count_from_onehot = int(onehot[ball_id].sum())

# ravel flattens the array to one dimension while retaining its values.
counts = np.bincount(semantic.ravel(), minlength=C)
count_from_all = int(counts[ball_id])

assert count_from_mask == count_from_onehot == count_from_all
assert int(counts.sum()) == H * W

ratio = count_from_mask / semantic.size * 100
print(counts)
print(f"ball: {count_from_mask} pixels, {ratio:.2f}%")
# [14782  6000  3218]
# ball: 3218 pixels, 13.41%
~~~

When a Boolean array is summed, `True` contributes 1 and `False` contributes 0. The one-hot ball mask also contains 1 only at ball pixels, so both methods give the same count. `np.bincount` counts occurrences of nonnegative integers; each output index is the integer being counted.[^bincount]

`minlength=C` makes the output **at least** `C` elements long. It does not restrict the length to exactly `C` or remove invalid labels. A mask containing 255 can produce an output extending through index 255, so the class-range checks in Section 07 must come first.

## 10. Distinguish stored IDs from display colors

To display a mask in color, create a **palette** that maps class IDs to RGB values. These colors help distinguish categories on screen; they do not change the target class IDs used for training.

~~~python
PALETTE = np.array([
    [222, 231, 240],  # 0: sky
    [73, 128, 93],    # 1: ground
    [185, 43, 62],    # 2: ball
], dtype=np.uint8)

color = PALETTE[semantic]
print(color.shape, color.dtype)
# (120, 200, 3) uint8
~~~

This is the same lookup principle used to obtain one-hot arrays from class IDs in Section 07. Here, each table row contains three RGB values, so each pixel receives three values. The original `semantic` remains an `(H, W)` integer array.

Matplotlib can also display the class mask directly. A **colormap** maps numbers to display colors. Fixing `vmin` and `vmax` ensures that the same ID keeps the same color even when a particular class is absent from an image.[^imshow]

~~~python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(PALETTE / 255.0)
plt.imshow(
    semantic, cmap=cmap, vmin=0, vmax=C - 1,
    interpolation="nearest"
)
plt.axis("off")
plt.show()
~~~

The display option `interpolation="nearest"` prevents boundaries from looking smoothly blended. This differs from a resize operation that changes the source array's shape or stored class IDs. A Boolean mask can be displayed with `plt.imshow(ball, cmap="gray", vmin=0, vmax=1)`.

## 11. An overlay blends colors at corresponding pixels

An **overlay** displays mask colors over an original image. It is useful for checking whether an image is paired with the right target and whether their boundaries still align after a transformation.

To make the mask look translucent, blend the original and display colors component by component at each pixel. `alpha` is the mask color's contribution: 0 keeps only the original image, while 1 keeps only the mask color.

~~~python
# Use a display color different from the ball color to inspect boundaries.
overlay_palette = PALETTE.copy()
overlay_palette[2] = (255, 220, 0)

alpha = 0.45
color_float = overlay_palette[semantic].astype(np.float32)
image_float = toy.astype(np.float32)
mixed = image_float * (1 - alpha) + color_float * alpha

# Apply the overlay only to ball pixels in this example.
overlay = toy.copy()
overlay[ball] = np.clip(mixed[ball], 0, 255).astype(np.uint8)
~~~

`np.float32` stores values with fractional parts. We use floating-point calculations while blending, restrict the result to 0–255 with `np.clip`, and then convert it to integers for display. Pixels outside the selected ball regions remain unchanged.

Here, `toy` has three dimensions and `ball` has two. Because their first two axes match, `toy[ball]` selects all three RGB components of each selected pixel. If `P` pixels are selected, the result has shape `(P, 3)`. This operation selects pixels; it does not reduce their RGB component axis.

## 12. Apply the same spatial transformation to the image and target

**Data augmentation** modifies existing training data so that a model can learn from a wider range of inputs. In segmentation, **each transformed image pixel must still correspond to the same array index in the target mask**.

A horizontal flip reverses the column order. For the image, retain the RGB component axis. For the mask, move the class IDs without changing their values.

~~~python
image_flip = toy[:, ::-1, :].copy()
mask_flip = semantic[:, ::-1].copy()

assert np.array_equal(
    mask_flip == 2,
    np.all(image_flip == BALL_RGB, axis=-1)
)
~~~

`::-1` selects indices in reverse order along that axis. `.copy()` creates an independent array and removes the reversed traversal of the original data. This also makes the result easier to convert to a PyTorch tensor later.

The check above works because our synthetic image uses solid colors, allowing us to verify the target again through a color comparison. With real photographs, check that the image and mask share the same transformation rules and inspect their boundaries with an overlay.

<figure class="seg-figure">
  <img src="/image-segmentation-mask-basics/paired-flip-en.svg" alt="Comparison of the original image and mask, both flipped together, and a misaligned pair where only the image is flipped" loading="lazy" />
  <figcaption>An image and its mask can have identical spatial shapes while referring to different pixels. The two balls have different sizes to make this misalignment visible.</figcaption>
</figure>

If the flip is random, make that decision only once. Drawing a separate random value for the image and the mask can flip one without the other.

~~~python
rng = np.random.default_rng(42)  # A reproducible random-number generator.
do_flip = rng.random() < 0.5     # Share this one decision between both arrays.

if do_flip:
    image_aug = toy[:, ::-1, :].copy()
    mask_aug = semantic[:, ::-1].copy()
else:
    image_aug = toy.copy()
    mask_aug = semantic.copy()
~~~

The same reasoning applies to cropping and rotation. Choose the crop's row and column ranges or the rotation angle once, then apply them to both arrays. Brightness and color changes, in contrast, apply only to the image and must preserve both pixel correspondence and the meaning of the target. Do not brighten or normalize the mask's class IDs.

In a typical training setup, split the data into training and validation sets first, then apply random augmentation to the training data. Validation images may still require a fixed resize to match the model's input dimensions. Keep training augmentation distinct from validation preprocessing.

## 13. Do not average class IDs when resizing a mask

**Interpolation** determines new pixel values when an image is resized. Image colors can be blended to produce intermediate colors. Class IDs, however, identify categories, so averaging them has no useful meaning.

For example, blending a boundary between classes 0 and 2 can produce 1, introducing the ground class where there was no ground. An integer output data type does not prevent this error. Converting a fractional result back to an integer does not restore the original meaning.

**Nearest-neighbor interpolation** chooses a source pixel and takes its value. Use it when resizing a class-ID mask to avoid blending the IDs.[^resize]

~~~python
new_H, new_W = 60, 100

image_small = cv2.resize(
    toy, (new_W, new_H), interpolation=cv2.INTER_AREA
)
mask_small = cv2.resize(
    semantic, (new_W, new_H), interpolation=cv2.INTER_NEAREST
)

assert image_small.shape[:2] == mask_small.shape
assert np.isin(np.unique(mask_small), np.unique(semantic)).all()
~~~

`cv2.resize` takes its size argument in `(width, height)` order. We resize the image and mask to the same target dimensions, using `INTER_AREA` to shrink the image and `INTER_NEAREST` for the class-ID mask.

This check confirms that resizing did not introduce a new class ID. It does not require every original class to survive: a very small region can disappear when downsampling. Nor does the presence of only valid IDs prove that boundaries align correctly, so inspect the result visually as well.

## 14. Inspect file formats and label conventions before reading real masks

We created every array ourselves, so we knew what its values meant. With a real dataset, first distinguish the image files, mask files, and metadata that maps class names to IDs. Do not assume that separately sorting two file lists makes their matching indices valid image–mask pairs. Check the dataset's filename or sample-ID convention.

<div class="seg-table">

| What to inspect | Why it matters |
| --- | --- |
| The image–mask pairing rule | Prevents assigning another image's target |
| Mask shape and data type | Distinguishes an ID array from an RGB color array |
| `np.unique(mask)` and the class table | Establishes how stored IDs map to training IDs |
| Special values such as 255 or -1 | Identifies ignored regions or actual classes |
| Image and mask height and width | Checks the spatial grid and pixel correspondence |

</div>

PNG masks do not all use the same storage scheme. Some store a class ID at each pixel, while others use RGB colors to identify classes. A palette PNG can store palette indices as pixel values. A colorful display does not prove that the underlying array is RGB.[^pillow]

Pillow's `Image.open` reads an image file. Inspect the mask's original mode and stored values before converting it to RGB or grayscale.

~~~python
from PIL import Image

# mask_path is the path to a mask file in the dataset.
with Image.open(mask_path) as mask_image:
    print(mask_image.mode, mask_image.size)
    raw_mask = np.array(mask_image)

print(raw_mask.shape, raw_mask.dtype)
if raw_mask.ndim == 2:
    print(np.unique(raw_mask))
~~~

This file-inspection example requires a real `mask_path`. Pillow's `size` uses `(width, height)`, while a two-dimensional NumPy shape uses `(height, width)`. If `raw_mask` is RGB, compare each pixel's three components with the dataset's color table to obtain class IDs. Calling only `np.unique(raw_mask)` collects individual component values, not RGB combinations, so its result cannot be interpreted as a class list.

Store target IDs in a lossless format such as PNG. JPEG compression can change pixel values, making it unsuitable for masks that must preserve exact class IDs.

## 15. Match the input, target, and output formats separately in PyTorch

We can now connect these array operations to deep learning code. A **tensor** is PyTorch's data structure for multidimensional data. A **batch** is a group of images processed together. `N` below is the number of images in that batch.

For a task that assigns one class to each pixel, the following formats apply when passing class-ID targets to `CrossEntropyLoss`. A **loss** measures the disagreement between model scores and targets.[^crossentropy]

<div class="seg-table">

| Data | Shape | Data type and meaning |
| --- | --- | --- |
| Input `images` | `(N, 3, H, W)` | `float32`, RGB images |
| Target `targets` | `(N, H, W)` | `long`, one class ID per pixel |
| Model output `logits` | `(N, C, H, W)` | Floating-point class scores at each pixel |
| Predicted IDs `predictions` | `(N, H, W)` | The index of the highest score on the class axis |

</div>

A **logit** is a model output score before conversion to a probability. Pass these scores directly into `CrossEntropyLoss`. When targets are supplied as class IDs, as they are here, there is no need to convert them to one-hot arrays. The target type `long` is PyTorch's 64-bit integer type.

First, create tensors from the NumPy arrays. `torch.from_numpy` creates a CPU tensor that shares the array's data. `.float()` and `.long()` change the data type, while `.unsqueeze(0)` adds a leading batch axis of size 1. If later tensor edits could affect shared source data that you need to preserve, make a copy.[^fromnumpy]

~~~python
import torch
from torch import nn

# Image: move RGB components to the front and convert to floats in 0–1.
image_chw = np.ascontiguousarray(toy.transpose(2, 0, 1))
images = torch.from_numpy(image_chw).float().unsqueeze(0) / 255.0

# Target: preserve class IDs. Do not divide them by 255.
targets = torch.from_numpy(semantic.copy()).long().unsqueeze(0)

print(tuple(images.shape), images.dtype)
# (1, 3, 120, 200) torch.float32
print(tuple(targets.shape), targets.dtype)
# (1, 120, 200) torch.int64
~~~

`np.ascontiguousarray` prepares the transposed data as a contiguous memory array. We use it here to make the converted result straightforward to handle. Apply `/ 255.0` only under the condition that the original RGB values are in the 0–255 range. With a pretrained model, also check the input normalization that model requires.

Now connect the input through to the loss. The model below uses one convolutional layer to take RGB components and produce `C` class scores. The layer's weights are learnable values controlling how much the surrounding pixel components contribute. `kernel_size=3` uses a 3×3 region. With the default stride of 1, `padding=1` adds one cell at each edge so that the output height and width stay unchanged.[^conv2d]

~~~python
model = nn.Conv2d(3, C, kernel_size=3, padding=1)
criterion = nn.CrossEntropyLoss()

logits = model(images)
assert logits.shape[0] == targets.shape[0]
assert logits.shape[1] == C
assert logits.shape[2:] == targets.shape[1:]

loss = criterion(logits, targets)
predictions = logits.argmax(dim=1)

print(tuple(logits.shape))       # (1, 3, 120, 200)
print(tuple(predictions.shape))  # (1, 120, 200)
print(loss.ndim)                 # 0: one scalar loss value.
~~~

This small model checks that the input and output formats connect correctly. It does not demonstrate a trained model that segments real photographs well. No weights have been updated: we have only obtained predicted IDs from scores produced by randomly initialized weights. Training would additionally require calculating gradients from the loss and updating the weights with an optimizer.

NumPy calls the axis argument `axis`; the PyTorch operation above calls it `dim`. The class axis is axis 1, so we use `argmax(dim=1)`, keeping the batch, height, and width axes. If only predicted class IDs are needed, select the index of the largest score directly.

Pass `logits` into the training loss. Passing the class IDs produced by `argmax` discards the class scores and the differentiable connection needed for training. Also, applying image-oriented `ToTensor` directly to an integer class mask can divide values by 255 for some input formats. Convert the target directly to integers as shown above.[^totensor]

Validation and inference do not update the weights. With a real model, call `model.eval()` to select evaluation behavior and produce outputs inside `torch.no_grad()`. When targets are available, compare them with predictions to evaluate the model. Inference without targets can still produce and display predictions.

## 16. Verify the results and apply the same reasoning to the next dataset

The array examples are collected in [examples.py](/image-segmentation-mask-basics/examples.py). The script runs without downloading external data or a trained model. Its default checks cover image and mask creation, connected components, one-hot round trips, pixel counts, overlays, paired horizontal flips, and nearest-neighbor resizing.

In macOS zsh or Linux bash/zsh, run the following commands from the folder containing the script. A virtual environment is a directory that keeps this example's installed packages separate from other work.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy opencv-python-headless
python examples.py
~~~

`opencv-python-headless` provides OpenCV without its GUI window functions. It is sufficient because this example does not use `cv2.imshow`. If your existing environment already contains `opencv-python`, there is no need to add both packages. The Matplotlib display snippets additionally require `matplotlib`, and the file-inspection snippet requires `pillow`.

If PyTorch is installed, you can also run the optional check. GPU installation depends on the device and environment; this small format check runs on the CPU.

~~~bash
python examples.py --torch-check
~~~

The execution environment and actual output are recorded in [verification.txt](/image-segmentation-mask-basics/verification.txt). A convincing-looking image is not the only success criterion.

<div class="seg-table">

| Check | Relationship being verified |
| --- | --- |
| Compare image spatial dimensions with mask shape | Do they use the same pixel grid? |
| Validate class-ID ranges | Does each lookup refer to a valid table row? |
| Check one-hot sums and the `argmax` round trip | Does changing the representation preserve each pixel's class? |
| Sum the per-class pixel counts | Has every pixel been counted? |
| Rebuild the semantic mask from component masks | Can the original class mask be recovered after separating regions? |
| Flip the image and mask together | Is pixel–target correspondence preserved? |
| Inspect IDs after nearest-neighbor resizing | Have class IDs avoided being blended? |

</div>

The first practical skills to learn are inspecting image and mask structures, using indexing and `axis`, validating class IDs, applying spatial transformations jointly, and matching model input and output shapes. Use one-hot arrays, palettes, and connected components when a particular representation or processing task needs them. With real data, do not assume that a connected component always equals an object or that every loss function accepts the same target format.

You have reached the goal of these basics when, on a new dataset, you can explain and verify **what each variable stores, what its axes mean, how the arrays correspond, and which relationships must survive a transformation**.

## References

[^indexing]: NumPy, [Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html). Basic slicing, integer array indexing, Boolean indexing, and views.
[^opencv-image]: OpenCV, [Image file reading and writing](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html). Image reading and the default color-channel order.
[^broadcasting]: NumPy, [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). Shape compatibility checked from the rightmost axis.
[^all]: NumPy, [numpy.all](https://numpy.org/doc/stable/reference/generated/numpy.all.html). Axis selection and `keepdims`.
[^components]: OpenCV, [Structural Analysis and Shape Descriptors](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html). Connected-component inputs and outputs, background labels, and 4- and 8-way connectivity.
[^stack]: NumPy, [numpy.stack](https://numpy.org/doc/stable/reference/generated/numpy.stack.html). Combining arrays of equal shape along a new axis.
[^eye]: NumPy, [numpy.eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html). Creating a lookup table with ones on its diagonal.
[^argmax]: NumPy, [numpy.argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html). Maximum-value indices and tie handling.
[^bincount]: NumPy, [numpy.bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html). Integer occurrence counts and minimum output length.
[^imshow]: Matplotlib, [matplotlib.pyplot.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html). Colormaps, value ranges, and display interpolation.
[^resize]: OpenCV, [Geometric Image Transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html). Resize dimensions and interpolation methods.
[^pillow]: Pillow, [Concepts — Modes](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes). Image modes including `P`, `L`, and `RGB`.
[^crossentropy]: PyTorch, [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). Shapes and data types for raw output scores and class-ID targets.
[^fromnumpy]: PyTorch, [torch.from_numpy](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html). Memory sharing between NumPy arrays and tensors.
[^conv2d]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html). Input and output channels, kernels, and padding.
[^totensor]: TorchVision, [ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html). Value scaling for certain image inputs and the warning about target masks.
