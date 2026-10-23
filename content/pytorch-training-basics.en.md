+++
title = "Your First PyTorch Training Loop: Shapes, Backpropagation, and Optimizers"
description = "Learn tensor shapes, object identity, and gradient recording separately before connecting them to regression training, minibatch classification, and optimizer comparisons."
date = 2026-10-23T16:00:00+09:00
slug = "pytorch-training-basics"

[extra]
katex = true
toc = true
styles = ["pytorch-training-basics/article.css"]
+++

Suppose we want to predict one number from another. A model multiplies the input by a weight and adds a bias. When predictions differ from targets, we modify the weight and bias. Three questions matter: **which numbers are being compared, where calculation results are stored, and what information determines the updates**.

Even with 100 predictions and 100 targets, incorrect tensor shapes can compare every combination instead of matching samples. Also, two subtraction expressions can produce equal numbers while differing in whether they modify an existing tensor or create a new one.

We use synthetic regression data and a two-moon classification dataset, not measurements of production performance. The examples assume you can read Python variable assignments, function calls, and basic loops, and have `torch`, `scikit-learn`, and `matplotlib` installed. PyTorch features and additional syntax are explained where needed. Code blocks can run from top to bottom; the complete regression example also includes its own data creation.

## 1. Separate the stages of training

A **sample** is one data case, and a **feature** is one input item describing it. A **weight** multiplies an input; a **bias** is added to the result. Such numbers modified during training are called **parameters**.

| Name | Role in this example |
| --- | --- |
| Input | Data used to calculate a prediction. |
| Target | The comparison reference corresponding to each input. |
| Prediction | The result calculated with the current weights and biases. |
| Loss | A numerical measure of how poorly predictions match targets. |
| Gradient | The rate at which loss changes when a particular parameter changes slightly. |
| Optimizer | A tool that uses gradients to update parameters according to a specified rule. |

The **forward pass** calculates predictions from inputs. We then calculate loss using predictions and targets, obtain parameter gradients through **backpropagation**, and finally update parameters using those gradients. Backpropagation and parameter updates are separate operations.

```text
Prepare inputs and targets → prepare weights and biases
    ↓
predictions → loss → gradients → update weights and biases
    ↑                                       |
    └──────── calculate again with new values ┘
```

A **batch** is a group of samples processed together. One pass through all training samples is an **epoch**. We first process the whole regression dataset as one batch, then introduce smaller batches.

## 2. Inspect tensor values, shape, and dimensionality separately

A **Tensor** is the PyTorch data type for storing and calculating with numbers in a structured arrangement. `torch.tensor(...)` creates a tensor from supplied numbers. An actual item created in a program, such as a tensor, is an **object**. A variable is a name referring to that object.

`shape` gives the size of each dimension, `ndim` gives the number of dimensions, and `numel()` returns the total element count. `dtype` specifies how individual elements are represented. Information read without parentheses, such as `shape`, is an **attribute**. A function called through an object, such as `numel()`, is a **method**.

Each of `a`, `b`, `c`, and `d` below represents one number.

| Arrangement | Shape | Dimensions | Elements |
| --- | --- | ---: | ---: |
| `a` | `()` | 0 | 1 |
| `[a]` | `(1,)` | 1 | 1 |
| `[[a]]` | `(1, 1)` | 2 | 1 |
| `[[a, b], [c, d]]` | `(2, 2)` | 2 | 4 |

The comma in `(1,)` is Python syntax for a one-item tuple. Containing one number does not make a tensor zero-dimensional. Start by inspecting shape without any training settings.

```python
import torch  # Import PyTorch tensor operations.

values = torch.tensor([2.0])  # Store one number in a one-dimensional arrangement.
print(values.shape)  # torch.Size([1]): shape (1,).
print(values.ndim)  # 1: one dimension.
print(values.numel())  # 1: one element too.
```

For input `X` shaped `(500, 2)`, this article uses **rows for samples** and **columns for features**. There are 500 samples, each with two features, in a two-dimensional tensor.

A model description saying “two-dimensional input features” may mean two features. One sample can have shape `(2,)`, and a batch can have shape `(N, 2)`, where `N` is the sample count. **Feature count and tensor dimensionality are different measurements.**

## 3. Match each input row with its target row

**Regression** predicts numerical values. We create targets by adding small amounts of noise to `2*x + 1`. **Noise** is an added value that makes targets deviate slightly from the rule. The coefficients `2` and `1` generate the targets; they are not the model's initial weight and bias.

### 3.1. Prepare the input shape first

`torch.linspace(start, end, count)` generates evenly spaced numbers including both endpoints. `unsqueeze(dim)` returns a tensor with a size-one dimension inserted at the specified position. Dimension positions start at zero. Element values and element count stay unchanged.[^shape]

<figure class="pt-figure">
<img src="/pytorch-training-basics/unsqueeze.svg" alt="unsqueeze(0) changes (N,) to (1,N); unsqueeze(1) changes it to (N,1).">
<figcaption>We need N samples with one feature each, so we use (N, 1). Shape (1, N) would put N features into one sample row.</figcaption>
</figure>

```python
x = torch.linspace(0, 1, 100)  # 100 numbers including zero and one: (100,).
x2 = x.unsqueeze(1)  # 100 samples with one feature each: (100, 1).
```

The `1` in `unsqueeze(1)` is **the new dimension's position**, not a count of new elements. `x` remains `(100,)`; the returned `x2` is `(100, 1)`. Calling `x.unsqueeze(1)` and ignoring the result does not change `x`'s shape.

### 3.2. Add randomly generated noise

**Random numbers** are values drawn using a defined random-generation procedure. `torch.manual_seed(42)` sets its starting state. It helps reproduce an experiment executed in the same order and environment. The number `42` has no special significance.

`torch.randn(100, 1)` draws from the **standard normal distribution**, with mean zero and standard deviation one, and returns a `(100, 1)` tensor. Standard deviation measures spread around the mean. Multiplying by `0.05` gives the noise distribution a standard deviation of `0.05`; it does not confine all values between `-0.05` and `0.05`.[^random]

```python
torch.manual_seed(42)  # Set the starting state for subsequent random operations.
noise = 0.05 * torch.randn(100, 1)  # One noise value per sample: (100, 1).
y2 = 2 * x2 + 1 + noise  # The targets corresponding to the input rows: (100, 1).
```

```text
first input row  → first target row, including the first noise value
second input row → second target row, including the second noise value
...
x2 (100, 1)                       y2 (100, 1)
```

`linspace` does not use randomness, so the seed does not affect it. The observed mean of 100 random values need not be exactly zero. A shared seed alone also cannot guarantee identical results across different devices or PyTorch versions.

<details>
<summary>Additional detail: does unsqueeze copy the numbers?</summary>

`x2` shares the underlying numerical storage with `x`. A tensor that presents shared storage with a different arrangement is called a **view**. Although the variables have different shapes, modifying an element through one changes the value visible through the other. Here we only adjust the shape; we do not modify the input values.

</details>

## 4. An operation can run while comparing the wrong samples

Specify the intended output first. We need one prediction for each of 100 samples, so the prediction should have shape **`(100, 1)`**, matching the targets.

`*` is elementwise multiplication. `@` is matrix multiplication: it multiplies matching entries of a left-hand row and a right-hand column, then adds the products. For two-dimensional inputs, `(rows, shared_size) @ (shared_size, columns)` requires matching shared sizes and produces `(rows, columns)`. But when the right operand is a one-dimensional vector, this example produces a one-dimensional result.[^shape]

We are only checking shape here, so the weight and bias have no training settings. `torch.zeros(1)` creates a `(1,)` tensor containing one zero.

```python
w = torch.randn(1)  # One weight value: (1,).
b = torch.zeros(1)  # One bias value: (1,).
wrong_pred = x2 @ w + b  # (100, 1) @ (1,) → (100,).
print(wrong_pred.shape)  # torch.Size([100]), not the intended (100, 1).
```

### 4.1. How broadcasting changes the subtraction

**Broadcasting** is the shape-matching rule used in elementwise operations on differently sized tensors. Compare dimensions from the right: their sizes must match or one must be one. Missing leading dimensions are treated as size one. The calculation acts as though a value is used at multiple positions; it does not require physically copying the data.

```text
prediction: (100,)     → treated like (1, 100) because a leading dimension is missing
target:     (100, 1)   → one target per row
result:     (100, 100) → each target is compared with all 100 predictions
```

<figure class="pt-figure">
<img src="/pytorch-training-basics/broadcasting.svg" alt="A row of predictions and a column of targets produce all combinations. Two matching columns compare corresponding rows.">
<figcaption>Only three samples are shown. We need the lower one-to-one comparison, not the upper all-combinations comparison.</figcaption>
</figure>

```python
wrong_error = wrong_pred - y2  # Compare every combination of 100 predictions and 100 targets.
print(wrong_error.shape)  # torch.Size([100, 100]): 10,000 errors.
```

### 4.2. Produce and check the intended shape

Using a `(1, 1)` weight in the calculation preserves a `(100, 1)` output. Below, `assert` stops execution when its condition is false. `torch.allclose` checks whether tensor values match while allowing small floating-point differences.

```python
y_hat = x2 @ w.unsqueeze(1) + b  # (100, 1) @ (1, 1) → (100, 1).
assert y_hat.shape == y2.shape  # Check prediction and target shapes.
same_pred = x2 * w + b  # Equivalent in this particular one-feature problem.
assert torch.allclose(y_hat, same_pred)  # Also compare the calculated values.
```

With multiple features, `*` and `@` generally differ. Matching shapes do not fix targets arranged in the wrong sample order. Check **intended output shape → actual output shape → correspondence between samples**.

## 5. Distinguish value transformations by their calculation units

An **activation function** transforms values, including intermediate neural-network outputs. A **class** is a category to predict. **Logits** are class scores before conversion to probabilities; they may be negative.

| Function | Which values are calculated together? | Result |
| --- | --- | --- |
| Sigmoid | Transforms each element independently. | Each value is between zero and one; the total need not be one. |
| ReLU | Transforms each element independently. | Changes negatives to zero and preserves the rest. |
| Softmax | Calculates across one sample's class scores. | That sample's probabilities sum to one. |

### 5.1. Inspect Sigmoid and ReLU first

`torch.sigmoid` and `torch.relu` return tensors with the input shape. The following input is `(2, 3)`, so both outputs are `(2, 3)`.[^activation]

```python
z = torch.tensor([[-2.0, 0.0, 3.0], [1.0, -4.0, 2.0]])  # Values to compare: (2, 3).
sigmoid_out = torch.sigmoid(z)  # Apply Sigmoid to each element.
relu_out = torch.relu(z)  # Result: [[0, 0, 3], [1, 0, 2]].
```

To write the calculations directly, first introduce two more functions. `torch.exp(z)` raises `e` to the power of each element; `e` is the mathematical constant approximately `2.71828`. `torch.where(condition, value_if_true, value_if_false)` chooses a value at each position according to the condition.

```python
sigmoid_manual = 1 / (1 + torch.exp(-z))  # The Sigmoid formula.
relu_manual = torch.where(z > 0, z, 0.0)  # Keep positive values and use zero otherwise.
assert torch.allclose(sigmoid_out, sigmoid_manual)  # Check values for this input.
assert torch.allclose(relu_out, relu_manual)  # Check the transformation's output values.
```

ReLU leaves values above one unchanged. Mathematically, Sigmoid maps finite inputs strictly between zero and one, but limited precision can represent extreme results as zero or one. A manually calculated exponential can also have problems with large inputs, so use built-in functions in actual models. The comparison above checks output values; it does not claim identical derivative behavior at every point.

### 5.2. Softmax's dim chooses which values are grouped

Rows of `logits2` represent samples and columns represent classes. There are two samples with three class scores each. In `torch.softmax(logits2, dim=1)`, `dim=1` means calculating across those class-score columns. The shape stays `(2, 3)`.

```python
logits2 = torch.tensor([[2.0, 0.5, 0.3], [0.2, 2.5, 0.1]])  # Scores: (2, 3).
probs = torch.softmax(logits2, dim=1)  # Produce three class probabilities per row.
```

The calculation is **exponentiate each score → add values within its row → divide each value by its row's sum**. `.sum(dim=1, keepdim=True)` reduces the columns but retains that dimension with size one. The sums therefore have shape `(2, 1)`, not `(2,)`.

<figure class="pt-figure">
<img src="/pytorch-training-basics/softmax.svg" alt="Divide exponentials in each row by their own row sum. Store two row sums with shape (2,1).">
<figcaption>a through f are exponentiated values. The first row uses s₁ and the second s₂, keeping samples separate.</figcaption>
</figure>

```python
exp_logits = torch.exp(logits2)  # Use small scores to inspect the calculation: (2, 3).
row_sums = exp_logits.sum(dim=1, keepdim=True)  # One sum per row: (2, 1).
probs_manual = exp_logits / row_sums  # Divide by the corresponding sum: (2, 3).
assert torch.allclose(probs, probs_manual)  # Compare against the built-in function.
```

`dim=-1` means the last dimension, so it equals `dim=1` for this input. A one-dimensional vector containing only class scores uses `dim=0` or `dim=-1`.

<details>
<summary>Additional detail: calculating directly with large scores</summary>

The manual example above uses small scores to expose the steps. Exponentiating large positive numbers can exceed the computer's numerical range. Subtracting each row's maximum first makes the exponential inputs nonpositive.

`.amax(dim=1, keepdim=True)` returns one maximum per row with shape `(2, 1)`. Subtracting the same value from every score in a row scales all its exponentials by the same factor. That factor cancels between numerator and denominator, leaving the probabilities unchanged.

```python
row_max = logits2.amax(dim=1, keepdim=True)  # Row maxima: (2, 1).
shifted = logits2 - row_max  # Subtract each row's maximum from that row.
exp_shifted = torch.exp(shifted)  # Avoid exponentiating large positive values.
shifted_sums = exp_shifted.sum(dim=1, keepdim=True)  # Row sums.
probs_stable = exp_shifted / shifted_sums  # Calculate probabilities.
assert torch.allclose(probs, probs_stable)  # Compare with the original Softmax output.
```

</details>

## 6. Separate the loss calculation from using its implementation

### 6.1. MSE squares errors and averages them

**MSE, mean squared error**, subtracts matching predictions and targets, squares the differences, and averages them. `.square()` squares each element; `.mean()` averages all elements.

```text
predictions: [2.5, 0.0, 2.0]       targets: [3.0, -0.5, 2.0]
errors:      [-0.5, 0.5, 0.0]
squares:     [0.25, 0.25, 0.0]
mean:        (0.25 + 0.25 + 0.0) / 3 = 0.166666...
```

```python
pred = torch.tensor([2.5, 0.0, 2.0])  # Three predictions.
target = torch.tensor([3.0, -0.5, 2.0])  # Three targets in the same order.
manual_loss = (pred - target).square().mean()  # Create a loss tensor with shape ().
```

Default MSE averages **all output elements**. With multiple outputs per sample, its denominator is not just the sample count. The method is `.square()`, not `.squared()`.[^loss]

### 6.2. Creating a loss object is separate from calculating loss

In Python, a **class** is a definition used to create objects. This is a different meaning from “class” as a classification category. `nn.MSELoss` is a class that creates loss-calculation objects. First create one with `nn.MSELoss()`, then supply predictions and targets to calculate a loss.

```text
nn.MSELoss()           → create the loss-calculation object
loss_fn                → the variable referring to that object
loss_fn(pred, target)  → calculate loss using actual predictions and targets
```

```python
from torch import nn  # Import the part providing layer and loss classes.

loss_fn = nn.MSELoss()  # Prepare the tool; no data have been compared yet.
loss = loss_fn(pred, target)  # This call performs the actual loss calculation.
assert torch.allclose(loss, manual_loss)  # It matches the manually calculated MSE.
```

Do not put data into the construction call as `nn.MSELoss(pred, target)`. `nn.MSELoss()(pred, target)` combines construction and calling, but two lines make their roles clearer.

### 6.3. Classification loss takes original scores and target indices

`nn.CrossEntropyLoss` creates a classification-loss object. Here we provide **one correct class index per sample**. Assigning a lower probability to the correct class produces a larger loss. The tool includes the necessary calculations, so pass **logits without applying Softmax first**.[^loss]

`N` is sample count and `C` is class count. `torch.long` is a 64-bit integer type suitable for class indices; it is the same as `torch.int64`.

| Item | Shape | Meaning |
| --- | --- | --- |
| `logits` | `(N, C)` | Floating-point scores for each sample and class. |
| `labels` | `(N,)` | One target index per sample, from 0 through C−1. |
| Default result | `()` | One number averaging the sample losses. |

```python
labels = torch.tensor([0, 1], dtype=torch.long)  # First sample: class 0; second: class 1.
ce_fn = nn.CrossEntropyLoss()  # Create the loss-calculation object.
ce = ce_fn(logits2, labels)  # Pass the earlier scores, not probs.
print(ce)  # tensor(0.2577)
```

`labels.long()` converts an existing tensor to this integer type. No extra conversion is needed when it was specified during creation. Probability-distribution targets are also supported, but this article uses integer indices.

<details>
<summary>Additional detail: names for the internal cross-entropy operations</summary>

Let `p` be the correct class's probability. The sample loss in this setup is `-log(p)`. Here `log` is the natural logarithm, the inverse of the exponential introduced earlier: if `exp(a)=p`, then `log(p)=a`. Loss is zero when the probability is one and grows as the probability approaches zero.

`LogSoftmax` calculates the natural logarithms of class probabilities. `NLLLoss` selects the correct-class entry and negates it to obtain the loss. CrossEntropyLoss with integer labels is equivalent to connecting those operations. In actual code, use `nn.CrossEntropyLoss()` rather than assembling the functions separately.

</details>

## 7. Automatic differentiation calculates gradients from recorded operations

So far we have calculated prediction and loss values. We now need **gradients for deciding how to modify the weights**. **Automatic differentiation** is PyTorch's feature for obtaining those gradients using the calculation relationships.

There are two stages. While calculating predictions and loss, PyTorch records which operations were applied to which tensors. Later it follows that record backward from the loss to calculate gradients. Not having calculated a gradient yet is different from having no computation record.

| Feature | When it is used | What it does |
| --- | --- | --- |
| `requires_grad=True` | When preparing a trainable tensor. | Enables later gradient calculation for that tensor. |
| `loss.backward()` | After calculating loss. | Uses recorded relationships to calculate gradients. |
| `w.grad` | After gradient calculation. | Stores gradients corresponding to the elements of `w`. |

`True` is the Boolean value for true. `None` represents the absence of a value. A `.grad` value of `None` does not mean the weight or its gradient is zero.

```python
w = torch.randn(1, requires_grad=True)  # Now enable gradient calculation for the weight.
b = torch.zeros(1, requires_grad=True)  # The bias is also trainable.
y_hat = x2 @ w.unsqueeze(1) + b  # Predictions: (100, 1), with operations recorded.
loss = (y_hat - y2).square().mean()  # Mean loss: ().
print(w.grad)  # None: backward has not run yet.
loss.backward()  # Calculate weight and bias gradients for the current loss.
print(w.grad.shape, b.grad.shape)  # Both are torch.Size([1]).
```

```text
forward:  w, b → predictions → loss
backward: loss → computations producing predictions → weight and bias gradients
```

One loss value can depend differently on each parameter, so each parameter has its own gradient. The differentiation rule linking the effects of intermediate calculations is the **chain rule**. `backward()` carries out that calculation without modifying the values of `w` and `b`.[^autograd]

Another forward and backward pass adds new gradients to existing `.grad` values. To update independently for each batch, clear the old gradients. `w.grad = None` **clears only the gradient storage and preserves the weight**.

<details>
<summary>Additional detail: which tensor stores the gradient?</summary>

The `w` and `b` above were created directly as starting tensors, rather than as results of other differentiable operations. Such tensors are called **leaf tensors**. Leaf tensors with `requires_grad=True` store `.grad` by default.

`w.unsqueeze(1)` is an intermediate tensor derived from `w`. Using it in prediction still allows gradients to follow the connection to the original `w`. Intermediate tensors do not all retain `.grad` by default. Keep the original variable `w` and use the result of `w.unsqueeze(1)` where the shape is needed.

</details>

## 8. Equal calculated values do not mean the same tensor was modified

This section examines **tensor values and variable references** without training or differentiation. Compare `w -= 0.5` with `w = w - 0.5`. Equal starting values produce equal subtraction results. The difference is whether the original tensor changes.

First, `same_object = w` does not copy the tensor: it makes two names refer to one tensor. Python's `is` operator tests whether two names refer to **the same object**.

### 8.1. -= changes the existing tensor's values

```python
w = torch.tensor([2.0])  # Create a tensor containing 2.0.
same_object = w  # Give that same tensor another name; do not create a new tensor.
w -= 0.5  # Change the existing tensor's value from 2.0 to 1.5.
print(w)  # tensor([1.5000])
print(same_object)  # tensor([1.5000]): the same tensor exposes the changed value.
print(w is same_object)  # True: these names still refer to one object.
```

```text
before: w ───────────┐       after: w ───────────┐
                    ├→ tensor [2.0]             ├→ same tensor [1.5]
        same_object ┘              same_object ┘
```

### 8.2. w = w - ... connects w to a new tensor

Reset the starting value to `2.0` so that both cases start from the same value.

```python
w = torch.tensor([2.0])  # Start the second comparison from 2.0 as well.
same_object = w  # Both names refer to the same tensor.
w = w - 0.5  # Create the subtraction-result tensor and connect only w to it.
print(w)  # tensor([1.5000]): the new tensor's value.
print(same_object)  # tensor([2.]): the original tensor was not modified.
print(w is same_object)  # False: these are now different objects.
```

```text
after: w ───────────→ new tensor [1.5]
       same_object ─→ original tensor [2.0]
```

Reading `w` produces `1.5` in both cases. Reading `same_object`, which also referred to the original tensor, exposes the difference. **Equality of numbers and identity of objects are separate questions.** This describes PyTorch tensor subtraction behavior. Applying `-=` to an ordinary Python number does not modify the existing number object.[^assignment]

Reversing the symbols as `w =- 0.5` means `w = -0.5`. It assigns negative 0.5 instead of subtracting 0.5 from the current value. If an update was intended, that is a typing mistake.

## 9. Learn how to disable recording, then connect it to training

### 9.1. Executing a calculation and recording it are different operations

Section 7 explained that automatic differentiation records forward-pass relationships. Use `torch.no_grad()` when **numerical calculations should run without recording them for backward**. It does not distinguish modifying a tensor from assigning a new tensor.

Python's `with` syntax applies setup when entering a defined block and handles cleanup when leaving it. Here, `with torch.no_grad():` disables recording for **the indented lines below it**. When the indentation ends, the preceding recording setting is restored.[^no_grad]

First, perform the same multiplication three times without updating a weight, so only recording behavior differs.

```python
source = torch.tensor([2.0], requires_grad=True)  # A starting tensor for gradient calculation.
recorded = source * 2  # Multiply with recording enabled.

with torch.no_grad():
    unrecorded = source * 2  # Perform the multiplication without a differentiation record.

recorded_again = source * 2  # The earlier recording setting has been restored outside the block.
print(recorded.requires_grad)  # True
print(unrecorded.requires_grad)  # False
print(recorded_again.requires_grad)  # True
print(source.requires_grad)  # True: the original tensor's setting did not change.
```

All three multiplications produce the number `4`. What differs is **whether the operation producing that result was recorded for differentiation**. `no_grad` does not erase an existing `.grad` value or permanently disable the original tensor's `requires_grad` setting.

### 9.2. Why use it around a weight update?

This training procedure calculates the gradient of the current loss with respect to the current weights, then uses it to update those weights. **We are not trying to include the update operation itself in the next backward calculation.** Recording is therefore disabled during the update. PyTorch raises an error if we try to modify such a trainable starting tensor directly while recording is enabled.[^autograd]

```text
recording enabled: calculate predictions → calculate loss → backward calculates gradients
recording disabled: use existing gradients to change the existing weight and bias values
after the block: clear gradients and record new operations during the next forward pass
```

The two choices do different jobs. **`-=` keeps the original tensor and changes its values.** **`no_grad` excludes the update operation from differentiation recording.** Neither one is the meaning of the other.

Creating a new subtraction result with `w = w - ...` inside `no_grad` produces a result without gradient tracking. Rebinding `w` to it replaces the original training tensor. Below, we keep the original `w` and `b` and update their values. This is an update after backward, not permission to overwrite arbitrary intermediate values needed during a forward pass.

### 9.3. Connect the complete regression loop

The **learning rate** `lr` scales the gradient-based update. The basic rule here is `new value = current value - lr × gradient`. The gradient is not the distance to an optimum, so an excessive learning rate can increase loss.

Also separate recording for display from training. `loss.item()` extracts a Python number from a one-element tensor. `losses = []` creates an empty list, and `.append(...)` adds a value at its end. The extracted number is for logging; backward uses the original `loss` tensor. `range(500)` makes the loop run 500 times.

```python
import torch  # This block can run the regression experiment on its own.

torch.manual_seed(42)  # Set the random-generation starting state.
x2 = torch.linspace(0, 1, 100).unsqueeze(1)  # Inputs: (100, 1).
y2 = 2 * x2 + 1 + 0.05 * torch.randn(100, 1)  # Targets: (100, 1).
w = torch.randn(1, requires_grad=True)  # One weight shared by all samples.
b = torch.zeros(1, requires_grad=True)  # One bias shared by all samples.
lr = 0.3  # Learning rate for this experiment.
losses = []  # Store loss numbers.

for step in range(500):
    y_hat = x2 @ w.unsqueeze(1) + b  # Predict using current parameters: (100, 1).
    assert y_hat.shape == y2.shape  # Reject an unintended comparison shape.
    loss = (y_hat - y2).square().mean()  # Mean loss: ().
    losses.append(loss.item())  # Record the loss computed before the update.
    loss.backward()  # Calculate gradients for the current loss.

    with torch.no_grad():
        w -= lr * w.grad  # Update the original weight without recording the update.
        b -= lr * b.grad  # Update the original bias the same way.

    w.grad = None  # Avoid accumulating old gradients in the next iteration.
    b.grad = None  # Clear the bias gradient too.

print("w:", w.item(), "b:", b.item())  # Inspect the learned numbers.
```

Each iteration uses all 100 samples, making it one full-batch training update. There is one weight and one bias, but all 100 samples contribute to loss and gradient computation. **Parameter count and amount of computation are not the same thing.** Because targets contain noise, learned values need not be exactly `2` and `1`.

## 10. Select the device used for tensor calculations

A **device** is where tensors are stored and calculations run. `cpu` uses the CPU, `cuda` is the NVIDIA GPU path here, and `mps` is the Apple GPU path on supported Macs. `is_available()` returns a Boolean indicating whether that path is currently available.[^device]

**`.to(device)` returns a tensor on the requested device.** It does not automatically rebind the original variable, so retain the returned value. If device and data type already match, it may return the original tensor.

```python
if torch.cuda.is_available():
    device = "cuda"  # Choose CUDA when available.
elif torch.backends.mps.is_available():
    device = "mps"  # Otherwise choose MPS when available.
else:
    device = "cpu"  # Use the CPU when neither GPU path is available.

t = torch.randn(2, 2)  # With default settings, create a tensor on the CPU.
t_on_device = t.to(device)  # Keep the returned tensor in a new variable.
print(t_on_device.device)  # Inspect its actual device.
```

This device-only transfer preserves shape and the meaning of values. When using a model later, prepare **its parameters, the current inputs, and the current targets on the same computation device**. We will connect model transfer after explaining how to create a model object.

## 11. Use an existing layer before wrapping it in a model class

### 11.1. Predict with one nn.Linear layer

A **neuron** multiplies inputs by corresponding weights, adds the products, and adds a bias. A **layer** groups such calculation units. `nn.Linear(input_features, output_neurons)` creates a layer performing that computation. Its two arguments are integers, not shape tuples such as `(1,)`.[^linear]

`nn.Linear(1, 1)` takes one feature and outputs one value. It also creates the required weight and bias. These numbers are managed as **Parameters**, special tensors registered as trainable model parameters.

```python
linear = nn.Linear(1, 1)  # Create a layer object with a weight and bias.
linear_prediction = linear(x2)  # Input (100, 1) → output (100, 1).
print(linear.weight.shape)  # torch.Size([1, 1]): one weight value.
print(linear.bias.shape)  # torch.Size([1]): one bias value.
```

**Creating** a layer, **predicting** with it, and **training** it using loss and gradients are different stages. This code only creates a layer and makes predictions.

### 11.2. Describe the computation order in a model class

To manage layers as one model, define a model class based on `nn.Module`. **Inheritance** means using an existing class's functionality in a new class. Here it provides management of parameters and child layers.

| Name or syntax | Role |
| --- | --- |
| `class LinearRegression(nn.Module)` | Define a model class using `nn.Module` functionality. |
| `__init__` | A method describing preparation when a model object is created. |
| `self` | The model object currently being created or used. |
| `super().__init__()` | Initialize the base `nn.Module` management first. |
| `self.linear` | The linear layer stored as part of the current model. |
| `forward` | A method describing how inputs are calculated and returned. |

`def` starts a function or method definition, and `return` sends its result back to the caller. Calling `model(x2)` makes PyTorch execute this `forward` calculation.

```python
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()  # Prepare management of child layers and parameters.
        self.linear = nn.Linear(1, 1)  # Store a one-input, one-output layer.

    def forward(self, x):
        return self.linear(x)  # Predict with the stored layer and return its result.

model = LinearRegression()  # Execute the preparation in __init__.
prediction = model(x2)  # Execute the calculation in forward.
```

Attaching the layer as `self.linear` lets the model discover that layer's parameters. Merely storing an ordinary tensor in an attribute does not automatically register it as a trainable parameter.

### 11.3. Inspect parameters and transfer the model

`model.parameters()` returns an **iterator** providing trainable parameter tensors. An iterator is an object from which items can be obtained one at a time. `named_parameters()` also provides each tensor's name. The following loop receives those two items as `name` and `p`.

```python
for name, p in model.named_parameters():
    print(name, p.shape)  # Inspect linear.weight (1, 1) and linear.bias (1,).

model = model.to(device)  # Move registered parameters and related state to the chosen device.
```

Unlike tensor `.to()`, model `.to()` transforms registered parameters and related state within the model and returns the model. Prepare it in this order: **create the model → move it to the device → create an optimizer for that model's parameters**. Section 14 connects the optimizer calls.

## 12. Identify what each number in 2 → 16 → 2 counts

Now switch from regression to classification. The new input has two coordinates, and the target is class 0 or 1. An **MLP** is a neural network connecting linear layers and activation functions. A **hidden layer** computes intermediate values between inputs and final outputs.

This model has one hidden layer and one output layer. Its 16 hidden neurons receive the same two input features and calculate with their own weights. Its two output neurons produce one score per class. `fc1` and `fc2` name the first and second linear layers.

```text
2 input features → 16 hidden neurons → 2 output neurons producing class scores
```

`B` is the current batch's sample count. `z1` is the first linear layer's result; `h1` is that result after ReLU. Start with the shapes:

```text
x (B, 2) → fc1 → z1 (B, 16) → ReLU → h1 (B, 16) → fc2 → logits (B, 2)
```

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # Prepare model parameter management.
        self.fc1 = nn.Linear(2, 16)  # Produce 16 intermediate values from two features.
        self.fc2 = nn.Linear(16, 2)  # Produce two class scores from 16 intermediate values.

    def forward(self, x):
        z1 = self.fc1(x)  # (B, 2) → (B, 16).
        h1 = torch.relu(z1)  # Replace negatives with zero; shape remains (B, 16).
        logits = self.fc2(h1)  # (B, 16) → (B, 2).
        return logits  # Return the original scores for classification loss.
```

In “16-dimensional hidden representation,” 16 counts intermediate features per sample. Tensor `h1` itself is two-dimensional, shaped `(B, 16)`. As learned weights change, the way those intermediate features are calculated changes too. Learning features this way is **representation learning**, not copying input values into extra columns.

### 12.1. Data rows and weight rows mean different things

`nn.Linear` stores weights as **`(output_neurons, input_features)`**. Its first-layer `weight` is `(16, 2)`. One weight row holds the two weights used by one neuron.

A **transpose** exchanges the rows and columns of a two-dimensional matrix and is written `.T`. The actual calculation is equivalent to `X @ weight.T + bias`.[^linear]

<figure class="pt-figure">
<img src="/pytorch-training-basics/linear-weight.svg" alt="Stored weight is (16,2), transposed weight is (2,16), and multiplying input (B,2) produces (B,16).">
<figcaption>Input rows represent samples; stored weight rows represent the weight sets of neurons. Being a row does not give every tensor the same meaning.</figcaption>
</figure>

`i` is a sample index and `j` an output-neuron index, both starting at zero. `X[i, 0]` is sample `i`'s first feature; `weight[j, 0]` is neuron `j`'s weight for that feature.

```text
z1[i, j] = X[i, 0] * weight[j, 0]
         + X[i, 1] * weight[j, 1]
         + bias[j]

shapes: (B, 2) @ (2, 16) + (16,) → (B, 16)
```

For a manual `X @ W`, arrange `W` as `(input_features, output_neurons)`. `nn.Linear.weight` uses the opposite storage orientation and transposes it for calculation. The underlying calculation has not changed.

### 12.2. Separate layers, parameter tensors, and scalar values

| Parameter | Shape | Scalar values |
| --- | --- | ---: |
| `fc1.weight` | `(16, 2)` | 32 |
| `fc1.bias` | `(16,)` | 16 |
| `fc2.weight` | `(2, 16)` | 32 |
| `fc2.bias` | `(2,)` | 2 |
| Total | Four tensors | 82 |

There are two trainable linear layers, four parameter tensors, and 82 numbers inside them. Gradients stored for those parameters after backward also total 82 elements. More samples do not increase parameter count, but they increase computation because each sample contributes.

Also separate the reason for ReLU. A sequence of linear layers can be rearranged into one linear layer's multiply-and-add calculation. ReLU changes some values to zero depending on their signs, enabling **nonlinear relationships** that cannot be combined into that single calculation. `torch.relu` is a function called inside `forward`, so it does not appear in `print(model)` as a registered child layer. Its absence from that list does not mean its calculation is missing.

## 13. Pair inputs and targets, then construct batches

`make_moons` generates coordinates arranged in two moon shapes and target indices. It returns two NumPy arrays. A NumPy array stores numbers in a structured shape; here we immediately convert those outputs to PyTorch tensors.

`n_samples` is sample count, `noise` controls the noise added to coordinates, and `random_state` sets this function's randomness. Inputs use the 32-bit floating-point type `torch.float32`; targets use the previously introduced integer type `torch.long`.

```python
from sklearn.datasets import make_moons  # Generate coordinates and class indices.
from torch.utils.data import TensorDataset, DataLoader  # Pair samples and construct batches.

X_np, y_np = make_moons(n_samples=500, noise=0.2, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)  # Inputs: (500, 2).
y = torch.tensor(y_np, dtype=torch.long)  # Target class indices: (500,).
```

`TensorDataset(X, y)` creates an object pairing `X[i]` and `y[i]` at the same index. The tensors must have the same first-dimension size: the same sample count. `DataLoader` constructs and supplies batches from that object.[^data]

| Setting | Meaning |
| --- | --- |
| `batch_size=64` | Put up to 64 samples in one batch. |
| `shuffle=True` | Shuffle sample order each epoch while preserving input–target pairs. |
| `drop_last=False` | Include the smaller final batch. False is also the default. |

```python
dataset = TensorDataset(X, y)  # Manage input–target pairs.
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
```

<figure class="pt-figure">
<img src="/pytorch-training-basics/batches.svg" alt="500 samples divided into batches of 64 produce seven full batches and one final batch of 52.">
<figcaption>64 × 7 + 52 = 500. This article's loop performs one parameter update per batch.</figcaption>
</figure>

The first seven batches have inputs `(64, 2)` and targets `(64,)`; the last has inputs `(52, 2)` and targets `(52,)`. One call to `model(xb)` processes the samples in that batch together. A batch of 64 does not require 64 model calls.

A **step** here is one parameter update. Eight batches mean eight updates per epoch, and 100 epochs mean 800 updates. Section 9 used the whole dataset as one batch, producing one update per epoch instead.

## 14. Let an optimizer handle parameter updates

### 14.1. Identify construction and method roles first

We previously modified `w` and `b` directly. With more model parameters, give an optimizer **the tensors to update and the update rule**.

`torch.optim.Adam` is an optimizer class. Adam uses both current gradients and a history of earlier gradients to determine per-parameter updates. This section covers construction and call order; Section 17 separately explains how the update amounts are calculated.[^optim]

| Code | Role |
| --- | --- |
| `model.parameters()` | Supply this model's trainable parameter tensors. |
| `Adam(..., lr=0.01)` | Create an Adam object for them, without updating them yet. |
| `zero_grad(set_to_none=True)` | Clear previous parameter gradients by setting them to `None`. |
| `loss.backward()` | Calculate gradients for the current loss. |
| `optimizer.step()` | Update parameters using stored gradients and the update rule. |

The optimizer's default behavior excludes its updates from differentiation recording, so this loop does not need a manually wrapped update block. `zero_grad` clears **parameter `.grad` values**, not learned weights or Adam's separate history.

### 14.2. Decide how to record the epoch average before writing the loop

Default CrossEntropyLoss returns a batch mean. **Batch means should not receive equal weight when batch sizes differ.** Let `B` be the batch sample count and `l₁` through `l_B` the individual sample losses.

```text
batch mean = (l₁ + ... + l_B) / B
batch sum  = batch mean × B
epoch mean = sum of all batch sums / samples actually processed
```

If batch means are `L₁` through `L₈`, this epoch's mean is:

```text
(64*L₁ + 64*L₂ + ... + 64*L₇ + 52*L₈) / 500
```

`(L₁ + ... + L₈) / 8` gives 52-sample and 64-sample batches the same weight, so it generally differs. Calculate **batch mean → batch sum → total sum → overall mean**.

`len(xb)` is the current batch's sample count. Accumulate loss sums in `epoch_loss_sum` and actual processed counts in `sample_count`. For numbers, `+=` adds the right-hand value to the current value and assigns the result back to that variable. These are logging numbers, separate from the `loss` tensor used for backward.

### 14.3. Repeat one-batch training inside each epoch

<figure class="pt-figure">
<img src="/pytorch-training-basics/workflow.svg" alt="Clear gradients, calculate predictions and loss, run backward, and call optimizer.step for the current batch.">
<figcaption>The features introduced so far are now connected into one batch's execution order. Targets yb enter the loss calculation.</figcaption>
</figure>

The outer loop handles epochs and the inner loop handles batches. `for xb, yb in loader` receives the batch inputs and targets into separate variables. Pass **current-batch targets `yb`**, not the entire `y`, to the loss.

```python
torch.manual_seed(42)  # Set randomness for model initialization.
model = MLP().to(device)  # Create the model and move it before creating the optimizer.
criterion = nn.CrossEntropyLoss()  # Mean loss with integer class targets.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Prepare the update tool.
epoch_losses = []  # Store epoch loss numbers.

for epoch in range(100):
    epoch_loss_sum = 0.0  # Start a fresh total for this epoch.
    sample_count = 0  # Count samples actually processed in this epoch.

    for xb, yb in loader:
        xb = xb.to(device)  # Current batch inputs: (B, 2).
        yb = yb.to(device)  # Current batch targets: (B,).
        optimizer.zero_grad(set_to_none=True)  # Clear old gradients.
        logits = model(xb)  # Class scores: (B, 2).
        loss = criterion(logits, yb)  # Compare with current-batch targets.
        loss.backward()  # Calculate gradients.
        optimizer.step()  # Update parameters.

        batch_size = len(xb)  # Use the actual size, including 52 for the last batch.
        epoch_loss_sum += loss.item() * batch_size  # Recover and accumulate the loss sum.
        sample_count += batch_size  # Accumulate the processed sample count.

    epoch_losses.append(epoch_loss_sum / sample_count)  # Record once after all batches.
```

Accumulate batch sums inside the inner loop and store the epoch mean after that loop finishes. This is why the final `append` is indented one level less than the batch-loop body.

The record combines losses measured while processing different batches. Because weights change between batches, it may differ from reevaluating the whole dataset with the model at the epoch's end. Reading `loss.item()` after `step()` does not change an already computed loss.

<details>
<summary>Additional detail: the scope of this averaging rule</summary>

The rule uses default CrossEntropyLoss with equal importance assigned to all samples. Counting the samples actually processed includes the smaller final batch correctly.

**Class weights**, which give some classes greater importance, or `ignore_index`, which excludes a particular target index, can change the loss denominator. Check the documented averaging rule before using those settings. Neither is used in this example.[^loss]

</details>

## 15. Understand function passing before shortening it with lambda

### 15.1. Passing a function name is not executing the function

A function is also an object that can be stored in a variable or passed to another function. `double` without parentheses is the function itself; `double(3)` executes it and receives its return value.

The following `apply` takes **one function and one value, then calls the function**. `operation` names the supplied function, and `value` names the argument to pass to it.

```python
def double(p):
    return p * 2  # Calculate using p when the function is called.

def apply(operation, value):
    return operation(value)  # The actual function call happens here.

answer = apply(double, 3)  # Pass the double function and the number 3 separately.
print(answer)  # 6
```

```text
apply(double, 3)
    → operation receives double; value receives 3
    → operation(value) executes double(3)
    → double's parameter p receives 3
    → 3 * 2 returns 6 to the caller
```

**Passing a function does not immediately execute its body.** Execution happens at an actual call such as `operation(value)`.

### 15.2. lambda also defines a function

`lambda p: p * 2` creates a function performing the same calculation as `double`. The name before the colon receives the argument, and the expression after it is evaluated and returned when called. `lambda` defines a function; it is not syntax for calling itself.[^python]

```python
answer = apply(lambda p: p * 2, 3)  # Pass the new function; apply calls it internally.
print(answer)  # 6
```

### 15.3. Apply the same idea to optimizer creation

By default, `torch.optim.SGD` creates an optimizer that subtracts the learning rate times the current gradient. The following function accepts parameters and returns an SGD object.

```python
def make_sgd(p):
    return torch.optim.SGD(p, lr=0.01)  # Create an optimizer for the supplied parameters.

optimizer_fn = make_sgd  # Store the function itself; no SGD object has been created yet.
```

The equivalent lambda is `lambda p: torch.optim.SGD(p, lr=0.01)`. Here `p` is a **function parameter**, a name for a supplied value. The value supplied will be **an iterator providing model parameter tensors**. It is neither SGD itself nor a single scalar weight.

<figure class="pt-figure">
<img src="/pytorch-training-basics/lambda.svg" alt="Pass the function first, create a new model, then call the function with model.parameters() to receive an SGD object.">
<figcaption>optimizer_fn(model.parameters()) supplies p. Separating definition from calling shows that no recursion occurs.</figcaption>
</figure>

Because the shared training function will create a new model internally, its caller cannot yet supply that new model's parameters. Instead it passes **a function that will create an optimizer when it receives those parameters later**. The next section shows the call site.

## 16. Compare optimizers through a shared training function

`train_with` receives an optimizer-creation function and an epoch count, **trains a fresh MLP, and returns a list of loss numbers**. `epochs=200` is the default when a caller omits the epoch count. `_` is a conventional name indicating that the loop index is unused.

Here we process all 500 samples at once. Unlike Section 14's minibatches, each epoch produces one update. Resetting the seed inside the function starts each experiment from the same initial model within the same environment.

```python
X_dev = X.to(device)  # All inputs: (500, 2).
y_dev = y.to(device)  # All targets: (500,).

def train_with(optimizer_fn, epochs=200):
    torch.manual_seed(0)  # Match initialization conditions across experiments.
    model = MLP().to(device)  # Create a new model on every call.
    criterion = nn.CrossEntropyLoss()  # Create the loss object.
    optimizer = optimizer_fn(model.parameters())  # This supplies the function's p argument.
    losses = []  # Loss history for this experiment.

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)  # Clear previous gradients.
        logits = model(X_dev)  # Calculate scores for the entire input batch.
        loss = criterion(logits, y_dev)  # Calculate full-batch mean loss.
        loss.backward()  # Calculate gradients.
        optimizer.step()  # Apply the chosen update rule.
        losses.append(loss.item())  # Record the already-computed loss number.

    return losses  # Return the loss list to the caller.
```

`results = {}` creates a **dictionary**, which associates keys with values. Here experiment names are associated with loss lists. The first call executes in this order: **create lambda function → call train_with → create new model → call lambda → return SGD object → run training loop**.

```python
results = {}  # Associate experiment names with loss histories.
results["SGD lr=0.01"] = train_with(lambda p: torch.optim.SGD(p, lr=0.01))
results["SGD lr=0.5"] = train_with(lambda p: torch.optim.SGD(p, lr=0.5))
results["SGD lr=10.0"] = train_with(lambda p: torch.optim.SGD(p, lr=10.0))
results["Adam lr=0.01"] = train_with(lambda p: torch.optim.Adam(p, lr=0.01))
```

Each value contains 200 losses. `results.items()` supplies name–value pairs. `plt.plot(losses)` plots a loss list as a line. `label` names that line, and `legend()` displays a key identifying the lines.

```python
import matplotlib.pyplot as plt  # Use plotting functions through the name plt.

plt.figure(figsize=(8, 5))  # Create one figure eight inches wide and five inches tall.
for name, losses in results.items():
    plt.plot(losses, label=name)  # Plot one experiment's loss history.
plt.xlabel("epoch index")  # Record indices start at zero.
plt.ylabel("training loss")  # The losses are measured on training data.
plt.legend()  # Display the experiment names.
plt.show()  # Inspect the complete plotting range.
```

A small learning rate may reduce loss slowly within the same iteration budget. An excessively large rate may cause spikes or oscillations. But `lr=10` does not always diverge, and Adam is not always best. Data, initialization, and iteration count matter.

A display limit such as `plt.ylim(0, 2)` hides losses above two. Inspect the full range first when investigating instability. These are **training losses**. Performance on new samples must be evaluated using separate **validation data** held out for that purpose.

## 17. Which records determine an optimizer's update?

Separate the training loop's call order from the optimizer's internal calculation. Here we compare the rules used by `step()`. **Basic SGD** means SGD without the momentum option. PyTorch's SGD can also be configured with momentum.[^optim]

| Method | Information used |
| --- | --- |
| Basic SGD | Current gradient. |
| Momentum | Current gradient and an accumulation of previous gradients. |
| RMSProp | Current gradient and an averaged record of squared gradients. |
| Adam | Two records: averaged gradients and averaged squared gradients. |

In the formulas, `t` is the update index, starting at `t=1`. `w_(t-1)` is a weight before the update, `w_t` after it, and `g_t` its gradient for the current loss. `lr` is the learning rate. Each formula describes a basic update for one scalar weight and is applied separately to other weights and biases.

### 17.1. Basic SGD uses the current gradient

<div class="pt-math">
$$w_t=w_{t-1}-lr\,g_t$$
</div>

Multiply the current gradient by the learning rate and subtract it from the current weight. This is the basic rule written manually in Section 9.

### 17.2. Momentum retains the influence of earlier gradients

`u_t` is an accumulated gradient record. `μ` controls how much of the previous record is retained, usually with a value at least zero and below one. Start with `u_0=0`.

<div class="pt-math">
$$u_t=\mu u_{t-1}+g_t$$
$$w_t=w_{t-1}-lr\,u_t$$
</div>

First add the current gradient to a retained portion of the earlier record, then update using that accumulation. Repeated signs can accumulate; a changed sign can partially cancel the previous value. Both signs and magnitudes are retained.

### 17.3. RMSProp scales updates using gradient-magnitude history

A **moving average** updates a record by combining the previous record and a new value with specified proportions. Here older values have less influence as updates continue.

`v_t` is the moving average of squared gradients. `β` is the retention ratio, at least zero and below one. `ε` is a small positive number preventing a zero denominator. `√` denotes the square root. Squaring collects magnitude information; taking the square root returns a scale corresponding to the gradient's original magnitude. Start with `v_0=0`.

<div class="pt-math">
$$v_t=\beta v_{t-1}+(1-\beta)g_t^2$$
$$w_t=w_{t-1}-lr\frac{g_t}{\sqrt{v_t}+\varepsilon}$$
</div>

The first equation updates the squared-gradient record. The second uses it to adjust the update based on the current gradient. Persistently large gradients increase the denominator. **Each weight maintains its own record** rather than sharing one across all weights.

### 17.4. Adam uses two averaged records

`m_t` is the moving average of gradients, and `v_t` is the moving average of squared gradients. `β₁` and `β₂` are their retention ratios, both at least zero and below one. Initialize `m_0=0` and `v_0=0`.

<div class="pt-math">
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$
</div>

Starting from zero makes the early records small, so they are corrected. For the first update, `m_0=0` gives `m_1=(1-β₁)g_1`. Dividing by `1-β₁` removes that initial scaling. At a general update index `t`, use the corrections below. `β₁^t` and `β₂^t` mean the respective values raised to the power `t`; the hats in `m̂` and `v̂` indicate corrected records.

<div class="pt-math">
$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}$$
$$w_t=w_{t-1}-lr\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}$$
</div>

The numerator reflects recent gradient signs and magnitudes, and the denominator adjusts the update using the squared-gradient record. Adam uses core ideas associated with Momentum and RMSProp; it does not call those optimizers one after another.

With every rule, **`backward()` calculates gradients, and `step()` uses them to determine and apply updates.** These formulas show basic forms without additional options.

## Appendix: Python syntax for choosing values conditionally

A **conditional expression** chooses one value according to a condition: `value_if_true if condition else value_if_false`. A **list comprehension** combines iteration with a value expression to create a new list. Distinguishing the two explains why `if` can appear in different positions.[^python]

`%` calculates a division remainder. `x % 2 == 0` checks whether dividing by two leaves zero remainder.

```python
numbers = [1, 2, 3, 4, 5, 6]  # Numbers to process.

# A trailing if retains only elements meeting the condition.
even_numbers = [x for x in numbers if x % 2 == 0]  # [2, 4, 6]

# A leading if else chooses the output value for each element.
parity = ["even" if x % 2 == 0 else "odd" for x in numbers]
# ['odd', 'even', 'odd', 'even', 'odd', 'even']

# Another condition can be checked after else.
sizes = ["small" if x < 3 else "medium" if x < 5 else "large" for x in numbers]
# ['small', 'small', 'medium', 'medium', 'large', 'large']
```

The sequence is: get one element → evaluate the condition → add the chosen value to the result. For longer conditions, an ordinary `for` loop with `if/elif/else` is often easier to read.

[^shape]: PyTorch documentation: [unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html), [matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html), [broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html).
[^random]: PyTorch documentation: [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [linspace](https://docs.pytorch.org/docs/stable/generated/torch.linspace.html), [randn](https://docs.pytorch.org/docs/stable/generated/torch.randn.html).
[^activation]: PyTorch documentation: [Sigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html), [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html).
[^loss]: PyTorch documentation: [MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
[^autograd]: PyTorch documentation: [autograd and in-place operations](https://docs.pytorch.org/docs/stable/notes/autograd.html).
[^assignment]: Python documentation: [assignment and augmented assignment](https://docs.python.org/3/reference/simple_stmts.html).
[^no_grad]: [PyTorch no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) and [Python with statement](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement).
[^device]: PyTorch documentation: [Tensor.to](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html), [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).
[^linear]: PyTorch documentation: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).
[^data]: [PyTorch datasets and DataLoader](https://docs.pytorch.org/docs/stable/data.html), [scikit-learn make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).
[^python]: Python documentation: [expressions](https://docs.python.org/3/reference/expressions.html), [lambda](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions).
[^optim]: PyTorch documentation: [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html), [RMSprop](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html), [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).
