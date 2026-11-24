+++
title = "Evaluating SSD Object Detection: Predictions, AP, mAP, and Visualization"
description = "Distinguish images, batches, and boxes, then connect confidence scores and IoU to TP, FP, FN, COCO AP, and dataset-wide mAP. Read Oxford-IIIT Pet validation results alongside test inference and the settings needed to preserve them."
date = 2026-11-24T08:45:00+09:00
slug = "ssd-detection-evaluation"

[extra]
toc = true
styles = ["ssd-detection-evaluation/article.css"]
+++

When I printed the predictions from an SSD model trained to detect dogs and cats, a single image produced 200 boxes. There were nowhere near that many animals in the photograph. At first, I was also unsure how to read the confidence scores beside the boxes and the mAP reported during validation.

The first distinction to make was not how large a number was, but **what it counted or compared**. The number of images differs from the number of predicted boxes. Counting incorrect predictions also differs from counting missed ground-truth objects. Even when two quantities range from 0 to 1, a model's confidence score and the overlap between two boxes describe different information.

This article follows the process of reading a trained object detector's output, evaluating a validation dataset, and inspecting individual images. The earlier [image classification evaluation article](/en/image-classification-evaluation/) compared one class per image. Here, we compare classes and boxes whose counts can vary between images. We begin with the necessary data structures, so you can follow without reading the earlier article.

The model scores in this article come from the existing exercise record. The small AP calculation examples are separate examples for understanding the evaluation rules. They should not be confused with a new run of the trained model or the original notebook. The final section introduces the runnable [evaluation and visualization functions and calculation examples](/ssd-detection-evaluation/examples.py).

## 01. Start with what the ground truth asks the model to find

**Object detection** predicts both an object's category and the coordinates of a box around it. A **bounding box** represents the target region as a rectangle. This exercise uses an object detector called SSD to distinguish dogs and cats. SSD stands for Single Shot MultiBox Detector.

There is an important condition: the bounding-box annotations in the Oxford-IIIT Pet dataset describe the animal's **head region**. Judging a prediction around the head against the animal's whole body would use a different target from the one the model was trained on.[^pets]

First, separate the complete source dataset from the information used in this task.

<div class="ssd-table">

| Source information | Contents | Role in this detection task |
| --- | --- | --- |
| JPG images | Image pixels | Model input and display |
| TXT lists | Image names, species and breed information, split lists | Identify the files to use |
| XML annotations | Object names and head-box coordinates | Construct detection targets for each image |
| Segmentation masks | Pixel-level regions | Not used in this box evaluation |

</div>

Images and XML files are matched by their filenames without extensions. Confirm that correspondence before forming image–target pairs. Sorting the JPG and XML lists separately is not enough to establish that the same index in both lists refers to the same image.

In this exercise, names read from XML were converted into the following numbers. A **class** is a category to distinguish; a **label** here is the class ID assigned to an object.

~~~python
# Convert XML object names into the integer labels used by this model.
class_map = {"dog": 1, "cat": 2}

# Convert model labels back into names for display.
class_names = {number: name for name, number in class_map.items()}
~~~

Both variables are dictionaries. `class_map` maps strings to integers, and `class_names` maps integers to strings. Here, `class_names` contains `{1: "dog", 2: "cat"}`.

This SSD uses 0 for the background class, so the model has three output classes including background. In contrast, the original TXT `SPECIES` IDs are `1=cat` and `2=dog`. The difference exists because the exercise remapped XML names instead of using those original IDs directly. **Read an ID together with the rule that created it.**

Training, validation, and test inference also have distinct roles.

<div class="ssd-table">

| Stage | Input | Work performed | Weight updates |
| --- | --- | --- | --- |
| Training | Images and box targets | Calculate losses and modify the model | Yes |
| Validation | Images and box targets | Generate predictions and compare them with targets | No |
| Test inference in this exercise | Separately held-out images | Generate predictions and inspect examples | No |

</div>

The test files inspected in this exercise did not have the XML box targets required for detection evaluation. That does not mean species and breed information was also absent. Predictions can be displayed in this situation, but detection mAP cannot be calculated without the box targets it must compare against.

## 02. Separate one image, one batch, and one box

A **tensor** is PyTorch's data type for working with numbers arranged along multiple axes. Its **shape** lists the size of each axis. One input image here is a `float32` tensor with shape `(3, H, W)`. `H` is the image height and `W` is its width. Axis 0 contains the three RGB channels, axis 1 is the vertical axis, and axis 2 is the horizontal axis.

Fetching one image returns an image tensor and a target dictionary. Here, `val_dataset` is the dataset object that provides validation samples.

~~~python
image, target = val_dataset[0]

print(image.shape)             # (3, H, W)
print(target["boxes"].shape)   # (N, 4)
print(target["labels"].shape)  # (N,)
~~~

`N` is the **number of ground-truth boxes** in this image. One row of `boxes` contains the four coordinates `[x1, y1, x2, y2]` of one box. The same index in `labels` holds that box's class ID. The coordinates describe its top-left and bottom-right boundaries; its width is `x2-x1` and its height is `y2-y1`. Do not confuse integer indices that select pixels with coordinates that describe boundaries.[^ssd]

A **batch** is a group of images processed together. A **DataLoader** retrieves samples from a dataset and combines them into batches. In this detection task, image tensors and target dictionaries are grouped into separate tuples. A tuple is a Python structure that holds values in an ordered sequence.

~~~python
# Pass this batch-combining function as the DataLoader's collate_fn argument.
def detection_collate(batch):
    return tuple(zip(*batch))
~~~

`batch` is a list of `(image, target)` pairs. `*batch` expands those pairs into separate inputs to `zip`, which groups the first elements together and the second elements together. The result is `(image tuple, target tuple)`. This sequence structure can accommodate different image sizes and different numbers of boxes.

<div class="ssd-table">

| Expression | What it retrieves or counts |
| --- | --- |
| `val_dataset[0]` | The first validation image and its target |
| `next(iter(val_loader))` | The first batch supplied by the loader |
| `len(val_dataset)` | Number of images |
| `len(val_loader)` | Number of batches |
| `len(target["boxes"])` | Number of ground-truth boxes in one image |

</div>

A regular DataLoader is not indexed as `val_loader[0]`. `iter` creates an iterator that can provide batches in sequence, and `next` retrieves its next item. Calling `next(iter(val_loader))` afresh each time also creates a new iterator each time. This differs from repeatedly retrieving the next batch from one iterator. If the loader is configured to shuffle, the contents of each new iteration's first batch can also differ.

The exercise record contains 735 validation images and a batch size of 4. With the final incomplete batch retained, that produces 183 batches of four images and one batch of three, or 184 batches in total. This is the size of a particular validation split, not the total size of the source dataset.

### The `[0]` after the model call selects the first image's result

TorchVision SSD in evaluation mode accepts a list of image tensors. It returns a list containing one prediction dictionary for each input image.[^ssd]

Below, `model.eval()` selects the behavior that returns predictions, while `torch.no_grad()` avoids recording operations for training. Section 04 explains the distinction in more detail.

~~~python
import torch

# model and device are the trained model and the device used for computation.
model_images = [image.to(device)]
model.eval()
with torch.no_grad():
    predictions = model(model_images)
pred = predictions[0]
~~~

`image.to(device)` prepares a tensor on the same device as the model. The square brackets create a Python list containing that one tensor. They do not create a new tensor with shape `(1, 3, H, W)`. The code uses the value returned by `.to(device)`; it does not automatically reassign the original `image` variable.

If there are `B` input images, the length of `predictions` is also `B`. If the first image has `M` predictions, the structure is as follows.

<div class="ssd-table">

| Expression | Type or shape | Meaning |
| --- | --- | --- |
| `predictions` | List of dictionaries, length `B` | Predictions for all input images |
| `predictions[0]` | Dictionary | All predictions for the first image |
| `pred["boxes"]` | Floating-point tensor `(M, 4)` | Predicted box coordinates for that image |
| `pred["labels"]` | Integer tensor `(M,)` | Class ID of each prediction |
| `pred["scores"]` | Floating-point tensor `(M,)` | Class score of each prediction |
| `pred["boxes"][0]` | Floating-point tensor `(4,)` | Four coordinates of the first predicted box |

</div>

Thus, `model([image.to(device)])[0]` does not retain only the first box. It selects the first image's result, which may contain many boxes. The ground-truth box count `N` and the prediction count `M` do not need to match either.

<figure class="ssd-figure">
<img src="/ssd-detection-evaluation/prediction-structure-en.svg" alt="One prediction dictionary per input image, with the box, label, and score at the same j index forming one prediction" loading="lazy">
<figcaption>An index selecting an image and an index selecting a prediction within that image refer to different objects.</figcaption>
</figure>

## 03. Preserve prediction correspondence through conversions and loops

Reading a file and converting an image into a tensor are separate operations. Pillow is an image-handling library, and `Image.open` takes a file path and creates an image object. `.convert("RGB")` converts it to three channels. For an ordinary RGB JPG such as those used here, `to_tensor` produces the `(3, H, W)` layout with pixel values in the 0–1 range.[^tensor]

~~~python
from PIL import Image
from torchvision.transforms.functional import to_tensor

# image_path is the path of an existing image file.
with Image.open(image_path) as source:
    pil_image = source.convert("RGB")

image_tensor = to_tensor(pil_image)
~~~

You cannot call `.to(device)` directly on a Pillow image. Conversely, if the dataset has already returned a tensor, you do not need to reopen that tensor with `Image.open`. First identify whether you have a path, an image object, or a tensor.

The relationship to preserve when iterating through predictions is straightforward: **the box, label, and score at the same index form one prediction**. If `j` is a prediction index within an image, handle `boxes[j]`, `labels[j]`, and `scores[j]` together.

~~~python
# assert raises an error if its condition is false.
assert len(pred["boxes"]) == len(pred["labels"]) == len(pred["scores"])

for box, label, score in zip(
    pred["boxes"], pred["labels"], pred["scores"]
):
    x1, y1, x2, y2 = box.tolist()
    name = class_names[label.item()]
    score_value = score.item()
~~~

`zip` groups values at the same index in its three inputs. The loop assigns them separately to `box`, `label`, and `score`. It does not place all three in `box`. Because `zip` stops at the shortest input, checking the lengths first helps detect correspondence errors.

During one iteration, `box` is a tensor with shape `(4,)`. Its `.tolist()` method returns a Python list of four coordinates, which can be unpacked into four variables. This does not assume that the image has only one box: the loop processes multiple boxes one at a time.

`.item()` retrieves a Python number from a tensor containing one element. `label.item()` returns an integer for looking up a class name; `score.item()` returns a floating-point number for comparison or display. Keep code that handles one value distinct from code that handles an entire tensor with multiple elements.

## 04. Evaluation mode and disabling gradient recording have different roles

During training, the difference between a model's result and its target is measured with a **loss**, and the weights are adjusted to reduce that loss. A **gradient** describes how a loss changes when a weight changes. PyTorch records the training computation so that it can calculate these gradients later.

`model.train()` and `model.eval()` change the model's operating mode. `torch.no_grad()` prevents recording operations for gradient calculation within its block. Calling `eval()` alone does not automatically disable gradient recording.[^autograd]

<div class="ssd-table">

| Purpose | Mode and call | TorchVision SSD return value |
| --- | --- | --- |
| Training | `model.train()`, then `model(images, targets)` | Dictionary of classification and box-regression losses |
| Evaluation or inference | `model.eval()`, then `model(images)` | List of image-level dictionaries containing `boxes`, `labels`, and `scores` |

</div>

**Box regression** here means predicting adjustments to align box coordinates with the targets. Receiving predictions in evaluation mode does not mean a validation loss is returned alongside them.[^ssd]

~~~python
import torch

model.eval()
with torch.no_grad():
    pred = model([image.to(device)])[0]
~~~

Call `model.train()` again before the next training stage. Also distinguish `loss.item()`, used for logging, from `loss`, used for backpropagation. Backpropagation calculates weight gradients from the loss, so it needs the original loss tensor. Accumulating Python numbers for logging avoids unnecessarily attaching the logging total to the computation graph.

`loss_sum / len(train_loader)` is the **mean of the loss values returned for the batches**. It cannot immediately be described as the mean of individual image losses. The last batch may have a different size, and SSD itself normalizes losses using the number of matched object candidates.[^ssdsource]

If each batch loss really is a simple mean of individual image losses, and all batches contain the same number of images, averaging the batch means gives every image equal weight. Verify those conditions before changing what you say the average represents.

## 05. Distinguish confidence, IoU, and NMS

A **confidence score** is the score a model assigns to a predicted class. SSD calculates image features, then produces class outputs and box adjustments for multiple object candidates. **Softmax** converts the class outputs so that the class scores for one candidate sum to 1. The score for the class associated with each final prediction is stored in `scores`.[^ssdsource]

This score is not calculated by comparing the prediction with a ground-truth box. That is why it can also be returned for a new image without annotations. A score of `0.9` does not by itself establish that the prediction has a verified 90% probability of being correct.

**Intersection over Union (IoU)** measures the overlap between two boxes. It is the proportion of their combined covered region that belongs to both boxes. To avoid counting the overlap twice, the combined area is the sum of the two box areas minus the overlap area once.

The denominator is not the area of a large rectangle enclosing both boxes. Such a rectangle can contain regions that belong to neither box. IoU is 0 when there is no overlap and 1 when the two boxes match exactly.

<div class="ssd-table">

| Criterion | What it uses | What it decides |
| --- | --- | --- |
| Confidence | The model's class output | Which predictions to prioritize, use, or display |
| Evaluation IoU | A predicted box and a ground-truth box | Whether the prediction sufficiently overlaps the target |
| Duplicate-removal IoU | A predicted box and another predicted box | Whether to retain multiple similar predictions |

</div>

A box with a confidence score of 0.39 can overlap a target with an IoU of 0.9. These numbers measure different things, so a confidence score should not be compared directly with an IoU threshold.

### A count of 200 boxes describes predictions left after postprocessing

SSD does not know the true object count in advance and then create exactly that many boxes. It creates multiple candidates, removes low-scoring ones, reduces similar predictions, and limits the final number it returns.

**Non-Maximum Suppression (NMS)** prioritizes higher-scoring boxes among strongly overlapping predictions. TorchVision SSD applies this process separately by class. Ground-truth boxes are not required for this step.[^ssdsource]

The official SSD implementation checked on November 24, 2026 uses defaults of `0.01` for the score threshold, `0.45` for the NMS IoU threshold, and `200` for the final per-image detection limit. These settings can be changed when constructing the model, so inspecting the actual object is more reliable.

~~~python
print("Score threshold:", model.score_thresh)
print("NMS IoU threshold:", model.nms_thresh)
print("Per-image detection limit:", model.detections_per_img)
~~~

The exercise recorded 200 predictions, the first five labels `[2, 2, 2, 2, 2]`, and the scores `[0.9987, 0.1008, 0.0680, 0.0614, 0.0612]`. Those values establish that the first five predictions use `cat` in the current mapping. They do not establish that all 200 predictions have that class or that any prediction matches a target. Nor does the limit of 200 mean the model must always return 200 boxes.

## 06. FP counts predictions; FN counts ground-truth objects

Before deciding whether a detection is correct, check that the prediction and target belong to the same image and class and meet the IoU threshold. For the ordinary objects considered here, **multiple predictions cannot all count as separate correct detections of the same ground-truth object**. Special rules for crowd objects or ignored annotations are outside this explanation.[^coco]

Within each image and class, predictions are processed from highest confidence downward. Each prediction is matched to the best-overlapping eligible target that is still unmatched and meets the IoU threshold. The score order therefore also affects which duplicate prediction is matched first.

<div class="ssd-table">

| Name | What is counted | Condition |
| --- | --- | --- |
| True positive (TP) | A retained prediction | Meets the IoU threshold with an unmatched target of the same image and class |
| False positive (FP) | A retained prediction | Has no eligible target to match |
| False negative (FN) | A ground-truth object | Has no corresponding correct prediction |

</div>

A duplicate box around the same target can be an FP. An incorrect class prediction is an FP for the predicted class. If no other correct prediction finds the actual object, that target also remains an FN for its true class. One image can contain both FP and FN errors.

### Does removing a prediction scored 0.39 turn it into an FN?

The removed prediction itself does not turn into an FN. First check whether a ground-truth object exists and whether another prediction still finds it. The following cases assume a score threshold is raised to 0.4, removing that prediction **from the evaluation input**.

<div class="ssd-table">

| Situation | Before removal | After removal |
| --- | --- | --- |
| There is no dog, but the model predicts one | That prediction is an FP | The FP disappears; there is no missed dog |
| A dog exists, but the 0.39 box is elsewhere and no other prediction finds it | Both an FP and an FN | The FP disappears; the FN remains |
| The 0.39 box is the only correct prediction finding the dog | A TP | The unmatched target becomes an FN |
| Another correct prediction continues to find the target | The target is matched | Removing the low-scoring prediction does not by itself make the target an FN |

</div>

If, instead, `if score < 0.4: continue` appears **only inside the drawing loop**, it simply omits a box from the display. It does not change the original prediction dictionary, predictions already passed to the metric, or an mAP value already calculated. Read the condition together with the stage at which it is applied.

## 07. Why precision and recall still need confidence scores

**Precision** is the proportion of used predictions that are correct detections. **Recall** is the proportion of ground-truth objects that have been found. Precision asks about predictions; recall asks about targets.

- Precision divides TP by `TP + FP`.
- Recall divides TP by `TP + FN`, the total number of ground-truth objects.

With no predictions, the precision denominator is zero; with no targets, the recall denominator is zero. Such cases require the evaluator's defined handling rules. The arithmetic examples below contain both targets and predictions.

Once the set of predictions has been fixed, these ratios can be calculated from TP, FP, and FN alone. But the set still has to be chosen. Keeping only high-confidence predictions excludes some detections; lowering the threshold includes lower-scoring ones too. This changes TP and FP and therefore changes the two ratios.

**Average Precision (AP)** summarizes the precision and recall obtained as the included predictions change. Confidence is not added directly to either ratio, but it supplies the **order in which predictions are included**.

“Adding predictions one at a time” does not mean running the model again. It means sorting predictions that have already been generated and evaluating the first prediction alone, then the first two, and so on. With distinct scores, this corresponds to progressively lowering a confidence threshold. The item being added is a predicted box, not an image or a batch.

## 08. Calculate COCO-style AP from start to finish

Fix the conditions first: one class, dog; an IoU threshold of 0.5; and two ground-truth dogs. After matching within the same image and class and checking duplicate matches, suppose the score-ordered outcomes are `TP → FP → TP`. The corresponding scores are 0.9, 0.8, and 0.7.

<div class="ssd-table">

| Included predictions | Cumulative TP | Cumulative FP | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| First prediction | 1 | 0 | 1 | 1/2 |
| First two predictions | 1 | 1 | 1/2 | 1/2 |
| All three predictions | 2 | 1 | 2/3 | 1 |

</div>

At the second step, adding an FP does not increase the number of targets found, so recall stays the same. More predictions are used, so precision decreases. At the third step, the remaining target is found and recall reaches 1. There are three intermediate precision–recall pairs, not three AP values.

The default COCO evaluation used by TorchMetrics produces one AP using the following rules.[^coco]

1. Set recall thresholds from 0.00 through 1.00 in increments of 0.01. Including both endpoints gives 101 thresholds.
2. For each threshold, find the intermediate results that reach at least that recall.
3. Select the highest precision among those results. Use 0 if none reaches the threshold.
4. Average the resulting 101 precision values.

This rule for determining precision between observed points is called **interpolation** here. It is not the same as drawing straight lines between adjacent points and using the values along those lines.

In this example, precision 1 is available for the 51 recall thresholds from 0.00 through 0.50. For the 50 thresholds from 0.51 through 1.00, use the precision 2/3 from the result that reaches recall 1. Thus AP is `(51 × 1 + 50 × 2/3) / 101`, approximately **0.8349835**.

This differs from simply averaging the three observed precisions as `(1 + 1/2 + 2/3) / 3`. It is also neither the average of precision and recall nor the average of the three confidence scores.

<figure class="ssd-figure">
<img src="/ssd-detection-evaluation/ap-ranking-en.svg" alt="Include TP, FP, and TP in score order, calculate intermediate precision and recall, then average precision over 101 recall thresholds" loading="lazy">
<figcaption>This is an illustrative calculation. TP/FP matching, intermediate precision and recall, and final AP are separate calculation stages.</figcaption>
</figure>

### Why collect precision at recall thresholds?

With the ground-truth set fixed, recall provides a common measure of how much of the required target set has been found. We can compare precision from finding a small portion through finding nearly all of it. Unreachable recall thresholds receive 0, so the score also reflects a model that finds a few objects accurately but misses the rest.

Even when the final TP and FP counts match, changing their order changes the intermediate results. For the same two targets, the sequence `FP → TP → TP` gives AP of 2/3, approximately 0.667. The final precision and recall match the preceding example, but the outcomes encountered when starting with high-confidence predictions differ.

There is a limit to what this tells us. With one target and a prediction sequence of `TP → FP`, the first prediction alone reaches both precision 1 and recall 1. COCO interpolated AP can therefore be 1. **AP of 1 does not mean every returned box is correct.** A high AP and incorrect boxes displayed at a low confidence threshold can coexist.

## 09. Why the mean of batch AP values differs from dataset-wide AP

Split the preceding predictions and targets into two batches. In this example, there is only one class and one IoU threshold, so AP and mAP are the same.

<div class="ssd-table">

| Evaluation scope | Target count | Predictions in score order | AP |
| --- | ---: | --- | ---: |
| Batch 1 | 1 | TP(0.9) | 1 |
| Batch 2 | 1 | FP(0.8), TP(0.7) | 0.5 |
| Entire dataset | 2 | TP(0.9), FP(0.8), TP(0.7) | Approximately 0.8349835 |

</div>

Averaging the batch AP values gives `(1 + 0.5) / 2 = 0.75`, which differs from the approximately 0.835 obtained by evaluating everything together. Neither predictions nor targets changed, and randomness did not cause the difference.

For an ordinary mean, averaging equally sized groups and then averaging those group means preserves each element's weight. AP, however, is not an average of fixed per-image scores. It uses the total number of targets as recall's denominator, sorts predictions of the same class across the entire evaluation set, and then interpolates. Combining batches changes both **the ranking and the reference set for recall**.

During validation, therefore, collect predictions and targets from each batch with `metric.update(predictions, targets)`. Call `metric.compute()` once after all batches have been processed. Batches divide the work supplied to the model; the final evaluation scope is the entire validation dataset.

This does not mean matching predicted and target boxes across different images. Matching remains within each image. The ranking used for AP then combines predictions of the same class across images. Combining the evaluation scope is different from removing image boundaries.

## 10. Check what mAP averages and which fields are returned

So far, we have calculated AP for one class at one IoU threshold. **mAP (mean Average Precision)** averages AP across classes. The default setting used here gives classes equal weight; a class with more images does not automatically receive more weight.

`mAP@0.50` averages class AP values at IoU 0.50. `mAP@0.75` works the same way. When multiple IoUs are specified, TorchMetrics' `results["map"]` averages across evaluable classes and **the specified IoU thresholds**.[^torchmetrics]

The default list contains ten thresholds: `0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95`. The final threshold is not `0.75`. If only six thresholds are supplied, `map` is calculated using those six, so preserve the list together with the score.

### A variable name you choose differs from a library argument name

`iou_values` is a Python variable name chosen for this code. The official argument accepted by `MeanAveragePrecision` is `iou_thresholds`. Pass the variable as `iou_thresholds=iou_values` when constructing the metric object. The constructor receives the calculation rules; the subsequent `update` method receives actual predictions and targets. A method is a function belonging to an object.

<div class="ssd-table">

| Constructor setting | What it specifies |
| --- | --- |
| `box_format="xyxy"` | Interpret coordinates as `[x1, y1, x2, y2]` |
| `iou_type="bbox"` | Evaluate bounding boxes rather than pixel masks |
| `iou_thresholds=iou_values` | IoU thresholds used for matching targets |
| `max_detection_thresholds=[1, 10, 100]` | Detection limits used during evaluation |
| `class_metrics=True` | Also return per-class results |
| `extended_summary=True` | Also return detailed arrays for extracting results by IoU |

</div>

Adding IoU 0.95 to the list does not automatically create a `results["map_95"]` key. The standard summary contains `map`, `map_50`, and `map_75`. **Supported calculation conditions and dictionary field names are separate matters.** A missing `map_95` key does not mean evaluation at 0.95 is unsupported.[^torchmetrics]

<div class="ssd-table">

| Result field | Meaning |
| --- | --- |
| `map` | Score averaged across the specified IoUs and classes, without an object-size restriction |
| `map_50`, `map_75` | mAP at the corresponding IoU |
| `map_small`, `map_medium`, `map_large` | mAP for each object-area range |
| `mar_1`, `mar_10`, `mar_100` | Mean Average Recall under the respective detection limits |
| `mar_small`, `mar_medium`, `mar_large` | Mean Average Recall for each object-area range |
| `map_per_class` | AP for each class, averaged across the specified IoUs |
| `mar_100_per_class` | Mean Average Recall for each class under the limit of 100 detections |
| `classes` | Class IDs corresponding to the per-class results |

</div>

**mAR (mean Average Recall)** averages how many targets are found across IoUs and classes. It is not a mean of precision. However, high-scoring incorrect predictions can occupy a limited detection budget, displacing correct predictions and affecting recall too.

The size ranges apply to object area, not the dimensions of the entire image. COCO uses `32²` and `96²` square pixels as area boundaries. This does not require both width and height to equal 32 or 96. Box area is used if separate `area` information is absent. The all-size `map` is not calculated by averaging the small, medium, and large scores.[^coco]

### Treat 0 and -1 differently

A score of 0 is a valid result for an evaluable entry. A value of -1 indicates an uncalculated or undefined entry. For example, check whether per-class output is disabled, whether the area range contains any targets, or whether the IoU needed by that summary is absent from the configured list.

Exclude -1 when averaging, but retain 0. Also, do not assume `classes` lists every class the model knows. Use the correspondence between IDs observed in the evaluation input and the result arrays. TorchMetrics does not automatically treat 0 as background in every input; background handling here is part of this SSD and its label scheme.

### SSD's limit of 200 and the evaluator's limit of 100 have different scopes

SSD's default limit of 200 applies to predictions across all classes in one image. The default COCO class-wise evaluation limit of 100 applies per image and class. Returning 200 predictions therefore does not establish that evaluation discarded exactly 100 of them.

A warning about many detections is not the same as a failed calculation. Changing `max_detection_thresholds` merely to remove a warning also changes the evaluation conditions. Detection limits must match when comparing results.[^torchmetrics][^coco]

## 11. Accumulate the full validation set and retrieve scores at each IoU

Now define the processing order. The inputs are the trained `model`, its computation `device`, and the validation `val_loader`. Generate predictions for each batch, preserve image order when transferring predictions and targets to CPU, and accumulate them in the metric object. Calculate the overall result at the end.

The following code does not train a new model or construct a dataset by reading XML. `val_loader` must provide images and targets in the structure described in Section 02. Images must be RGB floating-point tensors in the 0–1 range; target `boxes` must be floating-point tensors of shape `(N, 4)`, and `labels` must be `torch.int64` tensors of shape `(N,)`. If images have been transformed, their target coordinates must use the same coordinate system as those images.

If `MeanAveragePrecision` cannot be imported in Jupyter, use the installation cell below. `%pip` is notebook syntax, not a terminal command. PyTorch and torchvision must already be installed as a compatible pair for that environment.

~~~python
%pip install torchmetrics pycocotools
~~~

`torchmetrics` provides the interface for accumulating and retrieving metrics. `pycocotools` supplies the COCO evaluation implementation selected here. In a regular terminal, use `python -m pip install torchmetrics pycocotools` with the same Python environment.

### Calculate the full validation result first

Specify the six IoUs inspected separately during the exercise. If the intended result is the default ten-IoU average, use the default ten thresholds instead.

~~~python
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

iou_values = [0.50, 0.75, 0.80, 0.85, 0.90, 0.95]
class_names = {number: name for name, number in class_map.items()}

# Create a fresh object to avoid mixing in data from earlier validation runs.
metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    iou_thresholds=iou_values,
    max_detection_thresholds=[1, 10, 100],
    class_metrics=True,
    extended_summary=True,
    backend="pycocotools",
)

image_count = 0
model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        assert len(images) == len(targets)
        predictions = model([image.to(device) for image in images])
        assert len(predictions) == len(images)

        # Use model-postprocessed predictions without an extra display filter.
        predictions_cpu = [
            {key: pred[key].cpu() for key in ("boxes", "labels", "scores")}
            for pred in predictions
        ]

        # Preserve the evaluation meaning of area and iscrowd when provided.
        targets_cpu = [
            {
                key: target[key].cpu()
                for key in ("boxes", "labels", "area", "iscrowd")
                if key in target
            }
            for target in targets
        ]
        metric.update(predictions_cpu, targets_cpu)
        image_count += len(images)

if image_count == 0:
    raise ValueError("The validation loader provided no images.")

results = metric.compute()
print("Validation images:", image_count)
print("IoU thresholds:", iou_values)
print(f"mAP across the {len(iou_values)} specified IoUs: {results['map'].item():.4f}")
~~~

`predictions_cpu` and `targets_cpu` are lists of image-level dictionaries in matching order. `area` records object area, and `iscrowd` is a 0/1 indicator for crowd annotations. The present material uses `boxes` and `labels`, but if another dataset supplies these optional fields, do not discard them arbitrarily.[^torchmetrics]

Moving data to CPU does not eliminate accumulation memory. Predictions and targets still occupy CPU memory, and detailed summaries add further cost. If the same metric object is reused for another validation pass, clear it with `metric.reset()` before starting. `compute()` does not automatically save files or reset the object.

### Read the five axes of the detailed array first

With `extended_summary=True`, `results["precision"]` contains interpolated precision values. Its shape is `(T, R, K, A, M)`. Unlike the earlier use of `M` for a prediction count, `M` here means **the number of detection-limit choices in this library's output**. The same symbol has been defined within a different scope.[^torchmetrics]

<div class="ssd-table">

| Axis | Symbol | Meaning | Size in this configuration |
| --- | --- | --- | --- |
| 0 | T | IoU thresholds | 6 |
| 1 | R | Recall thresholds | 101 |
| 2 | K | Evaluation classes | Determined by the input |
| 3 | A | Object-area ranges | 4: all, small, medium, large |
| 4 | M | Detection-limit choices | 3: 1, 10, 100 |

</div>

`precision[i, :, :, 0, -1]` selects IoU index `i`, every recall threshold and class, the all-size range, and the last detection limit. `:` selects an entire axis; an index of `-1` selects its last entry. This index -1 differs from an **array value** of -1 indicating an undefined result.

The selection leaves an `(R, K)` array. Excluding only undefined values and taking the mean gives mAP at that IoU.

~~~python
map_by_iou = {}

for i, iou in enumerate(iou_values):
    values = results["precision"][i, :, :, 0, -1]
    valid = values[values >= 0]  # Keep valid zero scores; exclude undefined -1.
    score = valid.mean().item() if valid.numel() > 0 else -1.0
    map_by_iou[iou] = score
    print(f"mAP@{iou:.2f}: {score:.4f}")

# Link class IDs to names, retaining a 1D sequence even for a single class.
class_ids = results["classes"].reshape(-1).tolist()
class_ap = results["map_per_class"].reshape(-1).tolist()
class_ar = results["mar_100_per_class"].reshape(-1).tolist()

for class_id, ap, ar in zip(class_ids, class_ap, class_ar):
    print(f"{class_names[class_id]}: AP={ap:.4f}, AR@100={ar:.4f}")
~~~

`enumerate` provides a list's indices together with its values. `numel()` reports the number of tensor elements, and `mean()` calculates their mean. `reshape(-1)` flattens all elements onto one axis, with the axis length inferred automatically. Here it ensures an iterable list even when there is only one class. Apply the undefined-value rules from Section 10 if a per-class result is -1.

Start verification by checking whether two existing routes produce the same result, rather than adding another metric. Because this IoU list includes 0.50 and 0.75, the extracted values should agree with the official summaries within a small numerical tolerance.

~~~python
import math

assert math.isclose(map_by_iou[0.50], results["map_50"].item(), abs_tol=1e-6)
assert math.isclose(map_by_iou[0.75], results["map_75"].item(), abs_tol=1e-6)
~~~

`math.isclose` checks whether two numbers are sufficiently close under a tolerance. `1e-6` is 0.000001. These checks verify that the selected array axes correspond to the summary fields. They do not establish that the dataset split or the annotations themselves are correct.

## 12. Do not compare the recorded 0.5555 and 0.3357 as the same metric

The initial exercise record reported `mAP ≈ 0.5555` and `mAP@0.50 ≈ 0.9610`. If the first value was calculated with the default settings, it averages ten IoU thresholds. The separately inspected scores at each IoU were as follows.

<div class="ssd-table">

| Evaluation IoU | Recorded mAP |
| --- | ---: |
| 0.50 | 0.9610 |
| 0.75 | 0.6274 |
| 0.80 | 0.3219 |
| 0.85 | 0.0961 |
| 0.90 | 0.0075 |
| 0.95 | 0.0002 |

</div>

Adding these six values and dividing by 6 gives approximately **0.3357**. A `map` result around 0.33 after specifying these six thresholds is therefore consistent with the record. The default ten-threshold average and the selected six-threshold average use different evaluation conditions. Their difference alone is not evidence that the model became worse.

<figure class="ssd-figure">
<img src="/ssd-detection-evaluation/iou-results-en.svg" alt="Recorded mAP for six IoU thresholds: 0.9610 at 0.50, 0.6274 at 0.75, 0.3219 at 0.80, 0.0961 at 0.85, 0.0075 at 0.90, and 0.0002 at 0.95" loading="lazy">
<figcaption>This figure plots the existing exercise record. Individual results for 0.55, 0.60, 0.65, and 0.70 were not supplied and have not been filled in. Horizontal spacing follows the actual differences between IoU thresholds.</figcaption>
</figure>

It would also be incorrect to read `mAP@0.50 = 0.9610` as “96.1% of images were classified correctly.” mAP is not per-image accuracy. It averages AP calculated from class-specific prediction rankings and target matching.

Assuming the same data and predictions were used, the observation supported by this record is that the scores were lower under stricter IoU conditions. These numbers alone cannot establish the direction of box misalignment, whether coordinate processing was incorrect, or whether failures were concentrated in a particular class.

I use this result to guide the next checks. First, confirm that the data, class mapping, IoU list, and detection limits match. Then inspect results by class and object size and plot the targets and predictions for examples showing large discrepancies. A proposed cause needs actual examples that support it.

Suitability for practical use also cannot be decided by one number. The relative cost of missed objects and false detections, and the required box accuracy, affect the confidence threshold and evaluation conditions to use. Choose those settings on validation data, then inspect an independent test set after the choices are complete.

## 13. Draw targets and predictions on the same image

An overall score summarizes many examples. It cannot reveal whether class names were displayed in reverse, predictions from another image were attached, or a particular box boundary was misaligned with the target. **Visualization** examines these specific cases. A few attractive examples do not establish overall performance.

Understand two conversions before drawing. Matplotlib expects a color image in `(H, W, C)` axis order, so `permute(1, 2, 0)` rearranges the current `(C, H, W)` tensor. The numbers 1, 2, and 0 identify the original axes. This operation neither runs inference again nor changes box coordinates.

`Rectangle`, used to draw a box, accepts a starting coordinate, width, and height. Of the four stored boundary coordinates, the start is `(x1, y1)`; width and height are `x2-x1` and `y2-y1` respectively.[^rectangle] Passing `y2-y2` executes successfully but produces a rectangle with zero height.

### State the display condition in words first

If the intention is “display predictions with confidence at least 0.4,” skip predictions **below** 0.4. `continue` skips the rest of the current iteration and proceeds to the next prediction.

The next two lines are an excerpt from inside the prediction loop. `display_threshold` is the floating-point minimum display score, set to 0.4 here. Do not run this excerpt as a standalone cell; it belongs inside the complete function below.

~~~python
if score.item() < display_threshold:
    continue
~~~

Conversely, `if display_threshold < score.item(): continue` skips the higher-scoring predictions. `if 0.3 < 0: continue` is an always-false condition unrelated to the prediction's score. Describing which predictions to retain and which to skip before choosing the comparison operator makes these mistakes easier to spot.

Here, 0.4 is an example display setting, not an established optimum. Seeing only a few boxes even at a low threshold establishes only that few predictions passed the filter. It does not prove that the targets were detected correctly.

### Put repeated prediction drawing into a function

The following function accepts the plotting area `ax`, the prediction dictionary for one image `pred`, the ID-to-name mapping `class_names`, and the display threshold `display_threshold`. It does not modify the input predictions. It returns the number of boxes drawn as an integer.

~~~python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_predictions(ax, pred, class_names, display_threshold):
    count = 0
    assert len(pred["boxes"]) == len(pred["labels"]) == len(pred["scores"])

    for box, label, score in zip(
        pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu()
    ):
        score_value = score.item()
        if score_value < display_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="red", linewidth=2,
        ))
        ax.text(
            x1, y1, f"{class_names[label.item()]} {score_value:.2f}",
            color="red",
        )
        count += 1

    return count
~~~

`ax.add_patch` adds a rectangle to the plotting area. `fill=False` leaves its interior unfilled, and `ax.text` writes the class name and score at the given coordinates. The returned `count` is **the number of displayed predictions**. Do not assume it equals either the number returned by the model or the number of actual objects.

Now retrieve an image and its target together, and display predictions generated from **that same image**. `fig` is the entire figure, and `ax` is the plotting area for one image.

The label `GT` beside a target box abbreviates Ground Truth.

~~~python
image, target = val_dataset[0]
model.eval()
with torch.no_grad():
    pred = model([image.to(device)])[0]

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image.permute(1, 2, 0).cpu())

# Draw ground truth in green and predictions in red.
for box, label in zip(target["boxes"], target["labels"]):
    x1, y1, x2, y2 = box.tolist()
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor="lime", linewidth=2,
    ))
    ax.text(x1, y1, f"GT: {class_names[label.item()]}", color="lime")

display_threshold = 0.4
shown = draw_predictions(ax, pred, class_names, display_threshold)
ax.set_title(f"Returned: {len(pred['boxes'])}; shown: {shown}")
ax.axis("off")
plt.show()
~~~

`axis("off")` hides axis ticks and borders without changing image or box coordinates. The current image is in the 0–1 range produced by `to_tensor`. If other code applied additional normalization, that transformation may need to be reversed before display.

First use this figure to check the connection between data and display. Does the target surround the head? Does `class_names` agree with the actual mapping? Were this image and `pred` paired by the same inference call? Inspect overlap after those checks to avoid confusing coordinate errors with model prediction errors.

## 14. Keep filenames and predictions together during test inference

Construct the test list from filenames. `test_ids` is the set of names read from the test TXT file; `jpg_ids` is the set of existing JPG filenames without extensions. The set operator `&` selects elements present in both sets, and `sorted` converts that result into a sorted list.

~~~python
test_names = sorted(test_ids & jpg_ids)

assert len(test_names) > 0
assert len(set(test_names) & set(train_names)) == 0
assert len(set(test_names) & set(val_names)) == 0
~~~

`train_names` and `val_names` are the filename lists used for training and validation. These checks detect overlap by name. They do not detect identical image content stored under different filenames. Because this test set lacks XML box targets, do not also intersect it with `xml_ids` when constructing the list.

When displaying multiple images, finish reading, tensor conversion, inference, and drawing within the same loop iteration. This reduces the risk of drawing a previous image's predictions over a new image.

~~~python
import random
from PIL import Image
from torchvision.transforms.functional import to_tensor

# The start of a sorted list may concentrate on one breed; sample up to six
# without replacement. The seed 42 repeats the same selection from the same list.
sample_names = random.Random(42).sample(test_names, min(6, len(test_names)))
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax in axes.flat:
    ax.axis("off")

model.eval()
with torch.no_grad():
    for ax, name in zip(axes.flat, sample_names):
        # image_dir is a pathlib.Path pointing to the JPG directory.
        with Image.open(image_dir / f"{name}.jpg") as source:
            pil_image = source.convert("RGB")
        image_tensor = to_tensor(pil_image)
        pred = model([image_tensor.to(device)])[0]

        ax.imshow(pil_image)
        shown = draw_predictions(ax, pred, class_names, display_threshold=0.4)
        ax.set_title(f"{name}\nshown: {shown}")

plt.tight_layout()
plt.show()
~~~

`plt.subplots(2, 3)` creates a two-row, three-column arrangement of plotting areas. `axes.flat` iterates through those areas individually, and `zip` pairs each area with a filename in the same order. Hiding all axes first leaves no ticks in unused panels when there are fewer than six images. `tight_layout` adjusts spacing between the plots.

After the loop, `pred` holds the last image's result. Drawing it again over the first image would create an incorrect pairing. The six displayed examples do not represent overall test performance either. Without ground-truth boxes, this checks the process of reading files, generating predictions, and displaying them on the corresponding images.

## 15. Save weights together with the conditions needed to interpret the scores

A **checkpoint** stores state needed to resume work or run inference later. `model.state_dict()` is a dictionary containing model weights and related state. An **optimizer** updates weights using gradients, and `optimizer.state_dict()` contains its internal state needed to continue training.[^checkpoint]

Weights alone do not tell you whether class 1 meant dog or cat, or which IoU thresholds were averaged to produce an mAP value. Save the score together with the conditions required to interpret it.

The following example assumes that `results`, `map_by_iou`, and `iou_values` from the earlier evaluation and the training `optimizer` are available. `Path.cwd()` returns the current working directory. The output path is chosen explicitly instead of depending on a Colab-specific directory, and the code checks for an existing file to avoid overwriting it accidentally.

The postprocessing attribute `topk_candidates` limits the candidates kept per class before NMS. Save it together with the final `detections_per_img` limit to preserve the inference settings.

~~~python
from pathlib import Path

output_path = Path.cwd() / "pet_ssd_evaluated.pt"
if output_path.exists():
    raise FileExistsError("A file with this name already exists. Choose a new name.")

summary_keys = [
    "map", "map_50", "map_75",
    "map_small", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100",
    "map_per_class", "mar_100_per_class", "classes",
]

torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "class_map": class_map,
    "iou_thresholds": iou_values,
    "max_detection_thresholds": [1, 10, 100],
    "postprocess": {
        "score_thresh": float(model.score_thresh),
        "nms_thresh": float(model.nms_thresh),
        "detections_per_img": int(model.detections_per_img),
        "topk_candidates": int(model.topk_candidates),
    },
    "validation_names": list(val_names),
    "metrics": {key: results[key] for key in summary_keys},
    "map_by_iou": map_by_iou,
}, output_path)
~~~

This example adds the exercise's evaluation settings to the saved state. It does not claim to contain everything required to continue training identically. Exact reproduction also requires managing the model-construction code, preprocessing and data versions, package versions, training progress, random state, and other relevant conditions. Neither `metric.update` nor `loss_sum += loss.item()` automatically creates such a record.

When restoring, **construct a model with the same architecture first**. The `model` below must have the same structure as the saved model and output three classes including background. The original model-construction code is not included here, so do not arbitrarily substitute a different SSD configuration.

~~~python
checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
model.to(device)

# These settings are not weight tensors, so restore them separately.
for name, value in checkpoint["postprocess"].items():
    setattr(model, name, value)

class_map = checkpoint["class_map"]
class_names = {number: name for name, number in class_map.items()}
model.eval()
~~~

`map_location="cpu"` initially loads the saved tensors onto CPU. The model is then moved to the intended `device`. `setattr` restores a value to an object attribute identified by its name. Inference alone does not require restoring optimizer state. To continue training, prepare an optimizer with the same settings and load its state too.

A successful save call does not complete restoration verification. Use the same image and preprocessing before and after saving to check that boxes, labels, and scores are reproduced. Restoring the actual exercise weights was not among the checks run while preparing this article.

## 16. Separate what the runnable example verifies from what remains untested

The article divides its code according to the decisions being explained. [examples.py](/ssd-detection-evaluation/examples.py) combines full-dataset evaluation and plotting into functions and includes an AP calculation example that runs without a model. The file contains Korean comments and English function descriptions.

Run the following command in the folder containing the downloaded file. It works with the active Python environment's `python` in **macOS zsh and Linux bash or zsh**.

~~~bash
python examples.py --demo
~~~

`--demo` uses only the Python standard library and checks the following values.

~~~text
TP, FP, TP:       AP = 0.8349835
FP, TP, TP:       AP = 0.6666667
Mean batch AP:         0.7500000
Global AP:             0.8349835
TP, FP (one GT):  AP = 1.0000000
~~~

This function takes outcomes that have already been sorted by confidence and classified as TP or FP. It is not a complete implementation of COCO evaluation, including box matching, tie ordering, and crowd handling. Actual box evaluation uses TorchMetrics.

`--verify-metric` creates illustrative box coordinates and checks that actual TorchMetrics and `pycocotools` results agree with the preceding calculations. It requires `torch`, `torchvision`, `torchmetrics`, and `pycocotools`. The plotting functions also require `Pillow` and `matplotlib`.

~~~bash
python examples.py --verify-metric
~~~

To apply the functions to your trained model, place `examples.py` in the same folder as the notebook and import them as follows. The inputs match the article's validation code. The return values are the full result dictionary and an IoU-to-mAP dictionary.

~~~python
from examples import evaluate_detector, print_evaluation

results, map_by_iou = evaluate_detector(
    model, device, val_loader, class_map,
    iou_thresholds=[0.50, 0.75, 0.80, 0.85, 0.90, 0.95],
)
print_evaluation(results, map_by_iou, class_map)
~~~

Omitting `iou_thresholds` uses the default ten thresholds. The function leaves the model in evaluation mode; call `model.train()` before training again. The validation-image function returns the complete figure, that image's predictions, and the number displayed.

~~~python
import matplotlib.pyplot as plt
from examples import show_validation_image, show_test_images

fig, pred, shown = show_validation_image(
    model, device, val_dataset, class_map,
    index=0, display_threshold=0.4,
)
plt.show()

# First check file existence and separation from train/validation in Section 14.
fig, selected_names = show_test_images(
    model, device, image_dir, test_names, class_map,
    display_threshold=0.4, seed=42,
)
plt.show()
~~~

Before publication, CPU checks verified the AP calculations, accumulation across two batches, changes in AP caused by prediction order, and agreement between per-IoU extraction and `map_50` or `map_75`. The single-class case and the six-IoU list were also checked. Synthetic images were used to run validation plotting and test JPG loading and display, and to confirm that a display threshold leaves the original predictions unchanged.

The verification environment was Python 3.12.14, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, TorchMetrics 1.9.0, and pycocotools 2.0.11. The [verification record](/ssd-detection-evaluation/verification.txt) describes the execution scope in more detail. These versions document the environment used; they are not a requirement that every reader install identical versions. Consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for a compatible combination for your operating system and device.

These checks establish the calculations and information correspondence in the evaluation code. They did not reproduce the trained Pet SSD's performance, run CUDA or MPS execution, or restore the actual trained weights. Distinguish recorded model scores such as 0.9610 from the execution results of the synthetic examples.

## 17. Start the next dataset with the same questions

The goal here is not to memorize an entire AP implementation. It is to identify what a new prediction result refers to, how its information corresponds, and which evaluation conditions apply.

<div class="ssd-table">

| Learning priority | Scope to understand |
| --- | --- |
| Essential in practice | Separate images, batches, and boxes; verify class mappings and input/output structure; accumulate full-dataset evaluation; separate display from evaluation; save conditions with weights |
| Understand the principle | Why confidence ordering affects AP, why precision is averaged over recall thresholds, and why batch-average AP differs from dataset-wide AP |
| Refer to when needed | NMS internals, memory optimization for large evaluations, crowd and ignored-target handling, automatic confidence-threshold selection |

</div>

Try answering these questions before looking back at the code.

1. What does `[0]` select in `model([image.to(device)])[0]`?
2. When a prediction scored 0.39 is removed, under what condition does an actual target remain an FN?
3. Why can mean batch AP differ from dataset-wide AP even when the batches contain equal numbers of images?
4. If `map_95` is absent, how do you distinguish an unsupported calculation from a different result-access method?
5. Why can the number of visible boxes decrease while an already calculated mAP stays unchanged?

<details>
<summary>Check the answers after thinking through them</summary>

1. It selects the prediction dictionary for the first input image. That dictionary can contain multiple boxes for that image.
2. A ground-truth object exists and no other correct prediction matches it. The removed prediction itself does not become an FN.
3. AP is not an arithmetic mean of fixed per-image scores. Combining the data changes the class-specific prediction ranking and the target count used as recall's denominator.
4. Check the constructor's IoU list separately from the returned dictionary's keys. Include 0.95 in the calculation list and select that IoU from the detailed `precision` array.
5. The drawing-loop filter changed only what was drawn, without changing the predictions supplied to the metric object.

</details>

The completion criterion for this stage is being able to decide what to pair and what to count when given new images and targets, then inspect both overall metrics and individual examples. Record the data and evaluation conditions alongside each score. Those conditions must stay consistent for the next model comparison to answer the same question.

## References

[^pets]: [Official Oxford-IIIT Pet dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/) — head-region boxes and source annotation definitions.
[^ssd]: [TorchVision SSD300 API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html) — input, target, and prediction structures and mode-dependent return values.
[^ssdsource]: [TorchVision SSD implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/detection/ssd.html) — loss normalization, class scores, NMS, and detection limits.
[^tensor]: [TorchVision to_tensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.to_tensor.html) · [ToTensor input and value ranges](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html).
[^autograd]: [PyTorch gradient recording and evaluation mode](https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation).
[^coco]: [Official COCO evaluation implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) — matching by image and class, 101 recall thresholds, area ranges, and detection limits.
[^torchmetrics]: [TorchMetrics MeanAveragePrecision](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html) — constructor settings, output fields, and detailed-array axes.
[^rectangle]: [Matplotlib Rectangle](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html) — starting-coordinate, width, and height arguments.
[^checkpoint]: [Saving and loading models in PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
