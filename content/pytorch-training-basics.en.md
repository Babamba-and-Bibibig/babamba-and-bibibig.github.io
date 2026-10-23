+++
title = "Your First PyTorch Training Loop: Shapes, Backpropagation, and Optimizers"
description = "Connect tensor shapes to manual regression updates and minibatch classification. Illustrated explanations cover broadcasting, Linear weight layout, epoch loss, and passing an optimizer factory with lambda."
date = 2026-10-23T16:00:00+09:00
slug = "pytorch-training-basics"

[extra]
katex = true
toc = true
styles = ["pytorch-training-basics/article.css"]
+++

Suppose a model takes one input number and predicts another number. It must produce one prediction per sample, compare it with that sample's target, and update the weights and biases used for prediction. Code running without an error is not enough. Arrange 100 predictions and 100 targets incorrectly, and the code can compare values that do not belong to the same sample.

We will first use **synthetic regression data** to examine tensor shapes and one training update. Then we will train a small neural network on **two-moon classification data**. Both datasets are generated to study the computations. They are not measurements of a production system's predictive performance.

Start with the complete flow, then follow what the code creates at each stage. The examples require a Python environment with `torch`, `scikit-learn`, and `matplotlib` installed.

## 1. What does training calculate, and what does it change?

A **sample** is one data case. A **feature** is one input item describing a sample. **Parameters** are the numbers training modifies. Here they are **weights**, which multiply inputs, and **biases**, which are added to the results.

| Name | Role | Does this training procedure modify it? |
| --- | --- | --- |
| Input | Data the model uses for prediction. | No. |
| Target | The reference data used to judge predictions. | No. |
| Prediction | The result calculated with the current parameters. | Recomputing with different parameters can produce a different result. |
| Loss | A value measuring how poorly predictions match targets. | We do not reduce it by directly assigning it a smaller number. |
| Gradient | The rate at which loss changes when a particular parameter changes slightly. | Backpropagation calculates it. |
| Optimizer | An object that uses gradients and an update rule to modify parameters. | It modifies weights and biases. |

The **forward pass** calculates predictions and loss from inputs. **Backpropagation** follows those computation relationships backward from the loss to obtain gradients. Backpropagation does not itself update the weights.

<figure class="pt-figure">
<img src="/pytorch-training-basics/workflow.svg" alt="Each batch clears previous gradients, calculates predictions and loss, runs backward, and updates parameters. Targets enter the loss calculation.">
<figcaption>One batch of training. backward calculates gradients; step modifies parameters. The next batch uses the updated parameters.</figcaption>
</figure>

A **batch** is a group of samples processed together. An **epoch** is one pass through the training dataset. The basic loops in this article perform one backward pass and one parameter update per batch. Sample count, model-call count, and parameter count are different quantities.

## 2. Element count is not tensor dimensionality

A **Tensor** is the PyTorch data type for storing and calculating with numbers in a structured arrangement. `shape` gives the size of each dimension, `ndim` gives the number of dimensions, and `numel()` returns the total number of elements. `dtype` specifies the element data type.

In this table, `a`, `b`, `c`, and `d` each represent one number.

| Arrangement | Shape | Dimensions | Elements |
| --- | --- | ---: | ---: |
| `a` | `()` | 0 | 1 |
| `[a]` | `(1,)` | 1 | 1 |
| `[[a]]` | `(1, 1)` | 2 | 1 |
| `[[a, b], [c, d]]` | `(2, 2)` | 2 | 4 |

The comma in `(1,)` is Python syntax for a one-element tuple. A tensor containing one number is not necessarily a zero-dimensional tensor.

```python
import torch  # Use tensors and automatic differentiation.
from torch import nn  # Use model layers and loss classes.

w = torch.randn(1, requires_grad=True)  # A one-dimensional training tensor with one element.
print(w.shape, w.ndim, w.numel())  # torch.Size([1]) 1 1
```

`requires_grad=True` enables gradient tracking for later computations involving this tensor. It does not change the shape or calculate a gradient immediately.

Now suppose input `X` has shape `(500, 2)`. In this article, **each row is one sample** and **each column is one feature**. There are 500 samples with two features each. `X` itself is a two-dimensional tensor.

However, a model description saying “two-dimensional input” may mean two input features. A single such sample without a batch dimension has shape `(2,)`; a collection has shape `(N, 2)`, where `N` is the sample count. **Do not treat feature count and tensor dimensionality as the same thing.**

## 3. Generate regression data and understand unsqueeze

**Regression** predicts numerical values, such as quantities. Here we create targets by adding a little noise to `2*x + 1`. We do not give those coefficients to the model as its learned parameters. The model learns a weight and bias from input–target pairs.

First, distinguish the functions involved.

| Code | Purpose | Shape in this example |
| --- | --- | --- |
| `torch.manual_seed(42)` | Sets the initial state of PyTorch's random number generator. | A setting used to reproduce the random experiment below. |
| `torch.linspace(0, 1, 100)` | Creates 100 evenly spaced numbers, including both endpoints. | `(100,)` |
| `x.unsqueeze(1)` | Adds a dimension of size one at position 1. | `(100,) → (100, 1)` |
| `torch.randn(100, 1)` | Draws values from the standard normal distribution. | `(100, 1)` |

Dimension positions start at zero. The `1` in `unsqueeze(1)` specifies **where the new dimension goes**, not how many numbers to add. Element values and the number of elements stay unchanged.[^shape]

<figure class="pt-figure">
<img src="/pytorch-training-basics/unsqueeze.svg" alt="Applying unsqueeze 0 to a length-N tensor produces one row and N columns. Applying unsqueeze 1 produces N rows and one column.">
<figcaption>This regression problem needs one feature per sample. We use (N, 1), with N sample rows, rather than (1, N), with all values in one row.</figcaption>
</figure>

```python
torch.manual_seed(42)  # Reproduce the sequence of random operations in the same environment.

x = torch.linspace(0, 1, 100)  # 100 input numbers, with spacing 1/99.
x2 = x.unsqueeze(1)  # Represent 100 samples with one feature each: (100, 1).
noise = 0.05 * torch.randn(100, 1)  # Noise added to each sample's target.
y2 = 2 * x2 + 1 + noise  # Training targets, not model predictions.
```

The standard normal distribution has mean zero and standard deviation one. **Standard deviation** measures how widely values are spread around their mean. Multiplying by `0.05` makes the noise distribution's standard deviation `0.05`. It does not restrict every noise value to the interval from `-0.05` to `0.05`. The mean of the 100 values actually drawn need not be exactly zero either.

`manual_seed` affects random number generation, not the evenly spaced values from `linspace`. The same seed does not guarantee identical results across different PyTorch versions, devices, and computation methods.[^random]

The tensor returned by `unsqueeze` shares the original tensor's underlying storage. Such a result is called a **view**. Calling `x.unsqueeze(1)` and ignoring its return value leaves the original variable `x` with shape `(100,)`. The code above gives the returned view the name `x2`.

## 4. A valid operation is not necessarily the intended operation

Our regression prediction multiplies an input by a weight and adds a bias. Because each sample needs one prediction, **the intended prediction shape is `(100, 1)`, matching the targets**.

`*` multiplies corresponding elements, while `@` follows matrix-multiplication rules appropriate to the tensor dimensions. They are not interchangeable just because an operation runs.

```python
w = torch.randn(1, requires_grad=True)  # One weight element, shape (1,).
b = torch.zeros(1, requires_grad=True)  # One bias element, shape (1,).

wrong_pred = x2 @ w + b  # (100, 1) @ (1,) → (100,). This operation is valid.
wrong_error = wrong_pred - y2  # (100,) - (100, 1) → (100, 100).
print(wrong_error.shape)  # torch.Size([100, 100])
```

Multiplying a two-dimensional matrix by a one-dimensional vector produces a one-dimensional result in this case. Memorizing only the **two-dimensional-by-two-dimensional rule**, `(k, m) @ (m, n) → (k, n)`, misses this behavior.[^shape]

The subtraction then uses **broadcasting**. Dimension sizes are compared from the right; each pair must match or contain a size of one. Missing leading dimensions are treated as size one. This does not mean the data must physically be copied.

During subtraction, predictions of shape `(100,)` are treated like `(1, 100)`. Every row of targets `(100, 1)` is compared with all 100 predictions, producing `(100, 100)`. We needed 100 matched errors, but calculated all 10,000 combinations instead.

<figure class="pt-figure">
<img src="/pytorch-training-basics/broadcasting.svg" alt="One row of predictions and one column of targets create a three-by-three grid of all differences. Two matching columns create only the three corresponding differences.">
<figcaption>Only three samples are shown. The upper calculation compares every combination; the lower calculation compares each prediction with its own target.</figcaption>
</figure>

Treating the weight as `(1, 1)` preserves the intended output shape.

```python
y_hat = x2 @ w.unsqueeze(1) + b  # (100, 1) @ (1, 1) → (100, 1).
assert y_hat.shape == y2.shape  # Stop rather than calculate loss on mismatched shapes.

same_pred = x2 * w + b  # Equivalent here because there is only one input feature and weight.
assert torch.allclose(y_hat, same_pred)  # Compare values allowing small floating-point differences.
```

`assert` raises an error when its condition is false. `allclose` compares values while allowing small floating-point differences. With multiple input features, `*` and `@` generally do different calculations. Matching shapes also cannot fix samples that are in different orders. **Check both the shapes and which elements correspond.**

## 5. What do Sigmoid, ReLU, and Softmax change?

An **activation function** transforms values, including intermediate neural-network outputs. These three functions preserve input shape but differ in which values they calculate together.[^activation]

| Function | Calculation unit | Result |
| --- | --- | --- |
| Sigmoid | Each element independently. | Maps each value between zero and one. The total need not sum to one. |
| ReLU | Each element independently. | Replaces negative values with zero and leaves the rest unchanged. Results can exceed one. |
| Softmax | Values together along a specified dimension. | Here, converts each sample's class scores into probabilities summing to one. |

```python
z = torch.tensor([[-2.0, 0.0, 3.0], [1.0, -4.0, 2.0]])  # Shape (2, 3).

sigmoid_manual = 1 / (1 + torch.exp(-z))  # Elementwise Sigmoid formula.
sigmoid_builtin = torch.sigmoid(z)  # The built-in function used in practice.

relu_manual = torch.where(z > 0, z, 0.0)  # Choose z where the condition is true, otherwise zero.
relu_builtin = torch.relu(z)  # Returns [[0, 0, 3], [1, 0, 2]].
```

`exp(z)` means the mathematical constant `e` raised to the power `z`. The constant is approximately `2.71828`. `torch.exp` applies this calculation to each tensor element. `torch.where(condition, value_if_true, value_if_false)` chooses the value at each position.

Mathematically, Sigmoid maps finite inputs to values strictly between zero and one. Limited computer precision can represent extreme results as zero or one. The exponential in the manual formula can also overflow, so use built-in functions in actual models. `torch.clamp(z, min=0)` also clips negatives to zero, but do not assume every manually assembled expression has exactly the same derivative conventions as built-in ReLU.

### Softmax operates across the classes of one sample

A **class** is a category to predict. **Logits** are class scores before conversion to probabilities; they may be negative. Each row of `logits2` below is one sample, and each column is one class score.

```python
logits2 = torch.tensor([[2.0, 0.5, 0.3], [0.2, 2.5, 0.1]])  # (2, 3).

row_max = logits2.amax(dim=1, keepdim=True)  # One maximum per row: (2, 1).
shifted = logits2 - row_max  # Subtract the same maximum within each row: (2, 3).
exp_logits = torch.exp(shifted)  # Elementwise exponential: (2, 3).
row_sums = exp_logits.sum(dim=1, keepdim=True)  # One sum per row: (2, 1).
probs_manual = exp_logits / row_sums  # Divide each row by its own sum: (2, 3).

probs = torch.softmax(logits2, dim=1)  # Perform the same operation with the built-in function.
assert torch.allclose(probs_manual, probs)  # Verify matching results.
```

`amax` is a method that finds maximum values. In this table, `dim=1` means calculating across the class-score columns. `keepdim=True` retains the reduced dimension with size one. Thus the column size falls from three to one, changing `(2, 3)` into `(2, 1)`.

<figure class="pt-figure">
<img src="/pytorch-training-basics/softmax.svg" alt="A two-by-three array of exponentials has one sum per row, stored as two-by-one. Each original row is divided by its own sum.">
<figcaption>a through f are values after exponentiation. The first row is divided by s1 and the second by s2. Scores from different samples are not summed together.</figcaption>
</figure>

Subtracting the row maximum avoids sending large positive values into the exponential. Subtracting the same value from every score in a row leaves the Softmax probabilities unchanged. With shape `(samples, classes)`, use `dim=1`; `dim=-1`, meaning the final dimension, is equivalent here. For a one-dimensional vector of class scores, use `dim=0` or `dim=-1`.

## 6. Creating a loss object and calculating loss are different operations

`nn.MSELoss` and `nn.CrossEntropyLoss` are **classes**: definitions used to create objects. Parentheses after the class name create a loss object. Calling that object with predictions and targets performs the actual calculation.

```python
loss_fn = nn.MSELoss()  # Create the loss object. No predictions have been compared yet.
pred = torch.tensor([2.5, 0.0, 2.0])  # Three predictions.
target = torch.tensor([3.0, -0.5, 2.0])  # Their corresponding targets.

loss = loss_fn(pred, target)  # Call the object to obtain a loss tensor.
manual_loss = (pred - target).square().mean()  # Write the same calculation directly.
assert torch.allclose(loss, manual_loss)  # The result is approximately 0.1667.
```

**MSE, mean squared error**, subtracts matching values, squares the errors, and averages all elements. `.square()` is the method for elementwise squaring; its name is not `.squared()`.

```text
prediction - target  [2.5-3.0, 0.0-(-0.5), 2.0-2.0]
errors               [-0.5, 0.5, 0.0]
squared errors       [0.25, 0.25, 0.0]
mean                 (0.25 + 0.25 + 0.0) / 3 = 0.166666...
```

`nn.MSELoss(pred, target)` puts data in the object-construction call and is not the intended usage. `nn.MSELoss()(pred, target)` performs construction followed by calculation, but two separate lines make the distinction clearer.

Default MSE averages **all output elements**. With multiple output values per sample, dividing only by the number of samples is a different calculation.[^loss]

### CrossEntropyLoss with integer class targets

For classification, we compare each sample's class scores with its target class index. Let `N` denote sample count and `C` class count.

| Object | Shape | Type and contents |
| --- | --- | --- |
| `logits` | `(N, C)` | Floating-point class scores. Do not apply Softmax first. |
| `labels` | `(N,)` | `torch.long` integers from `0` through `C-1`. |
| Default mean loss | `()` | A zero-dimensional tensor containing one number. |

```python
labels = torch.tensor([0, 1], dtype=torch.long)  # Sample one is class 0; sample two is class 1.
ce_fn = nn.CrossEntropyLoss()  # Create the classification loss object.
ce = ce_fn(logits2, labels)  # Pass the original scores from the previous section, not probs.
print(ce)  # tensor(0.2577)
```

Under this setup, a lower probability assigned to the correct class produces a larger loss. Integer-label cross-entropy is equivalent to combining `LogSoftmax` with `NLLLoss`; it uses the negative logarithm of the correct-class probability. The default setup returns the mean across samples.[^loss]

`labels.long()` converts the data type to `torch.int64`. No additional conversion is needed when the labels are already `torch.long`. CrossEntropyLoss is not limited to integer targets: it also supports class-probability targets. This article uses the `(N,)` integer-target mode.

## 7. backward calculates gradients; updates inside no_grad change values

Automatic differentiation uses the computation relationships recorded during the forward pass. Weights and biases produce predictions, and predictions produce loss. By following those relationships backward, it can calculate how loss depends on each parameter. The differentiation rule connecting intermediate effects is the **chain rule**.[^autograd]

```text
Forward:    w, b → prediction → loss L
Backward:   start from loss L → follow prediction computations → w.grad, b.grad
Update:     use those gradients to modify the existing w and b values
```

| Code | What it does | What it does not do |
| --- | --- | --- |
| `loss.backward()` | Calculates parameter gradients and accumulates them into `.grad`. | Does not update parameter values. |
| `w.grad` | Contains the derivative of the current loss with respect to each element of `w`. | Does not give the distance to an optimal weight. |
| `with torch.no_grad():` | Excludes operations in the block from reverse-mode gradient recording. | Does not erase already stored `.grad` values. |
| `w.grad = None` | Clears the stored gradient. | Does not set the weight itself to zero. |

A loss can be one number while depending differently on many weights. Each weight therefore has its own gradient. `w.grad` has the same shape as `w`.

A directly created starting tensor such as `w` or `b`, rather than the result of another differentiable operation, is a **leaf tensor**. Leaf tensors with `requires_grad=True` store `.grad` by default. `w.unsqueeze(1)` is an intermediate result connected to the original `w`. Using that view in prediction still lets the gradient reach `w`. Using a view at the point of calculation, rather than replacing the training variable with it, keeps this distinction clear.

Running a new forward pass and then `backward()` again adds gradients to existing `.grad` values. To update separately using each batch's gradients, as in this article, clear them between batches.

## 8. Updating weights: modifying an existing tensor or assigning a new one

`w -= lr * w.grad` and `w = w - lr * w.grad` calculate the same subtraction result when given the same current values and gradients. In PyTorch, however, they differ in **whether they modify the existing tensor or create a new tensor and bind the variable to it**.

A Python variable is a name referring to an object. Two tensors containing equal numbers are different from two names referring to the same tensor.

| Code | Meaning for a PyTorch tensor |
| --- | --- |
| `w = w - lr * w.grad` | Binds the name `w` to the new tensor produced by subtraction. |
| `w -= lr * w.grad` | Modifies the values of the existing `w` tensor. |

Inside `no_grad`, `w = w - ...` creates a new subtraction-result tensor without gradient tracking. Replacing the training variable with that result changes the connection needed for later training. In the same position, `w -= ...` preserves the existing object's `requires_grad=True` setting while changing its values.

```python
original = torch.tensor([2.0], requires_grad=True)  # The training tensor object.
same_object = original  # Refer to the same object; do not copy it.

with torch.no_grad():
    original -= 0.1  # Modify the existing tensor's values.

assert original is same_object  # is tests whether both names refer to the same object.
assert original.requires_grad  # Gradient tracking remains enabled.

with torch.no_grad():
    replacement = original - 0.1  # Create a new tensor as the subtraction result.

assert replacement is not original  # It is a different object.
assert not replacement.requires_grad  # This operation result was created inside no_grad.
```

This explanation concerns **updating parameters after backward has finished**. Overwriting needed intermediate values during a forward pass can break backpropagation. The behavior of `-=` also depends on the data type. Do not apply PyTorch tensor identity behavior to ordinary Python numbers.[^assignment]

### Typing mistake: do not reverse the symbols

`w =- lr * w.grad` is not another update method. It is an example of **accidentally reversing the symbols in `-=`**. Separating the assignment and minus signs makes it clearer: `w = -lr * w.grad`.

| Case | Code | Meaning |
| --- | --- | --- |
| Intended update | `w -= lr * w.grad` | Subtract `lr * w.grad` from the current weight. |
| Reversed-symbol mistake | `w = -lr * w.grad` | Assign `-(lr * w.grad)` as the new value, omitting subtraction from the current weight. |

`-=` is one operator with **minus followed by equals**. `= -` is an assignment with **equals followed by unary minus**. The second line can still run as valid Python syntax, so this typing mistake need not produce a syntax error.

## 9. Train regression by updating the parameters directly

The following code runs the regression procedure in one block. The **learning rate** `lr` scales the overall size of the gradient-based update. The basic rule subtracts the learning rate times the gradient from the current weight.

```python
import torch  # Use tensors and automatic differentiation.

torch.manual_seed(42)  # Set the random generation sequence.
x2 = torch.linspace(0, 1, 100).unsqueeze(1)  # Inputs: (100, 1).
y2 = 2 * x2 + 1 + 0.05 * torch.randn(100, 1)  # Targets: (100, 1).

w = torch.randn(1, requires_grad=True)  # One weight shared by all samples.
b = torch.zeros(1, requires_grad=True)  # One bias shared by all samples.
lr = 0.3  # Learning rate for this example.
losses = []  # A Python list used to record losses.

for step in range(500):
    y_hat = x2 @ w.unsqueeze(1) + b  # Forward pass: (100, 1).
    assert y_hat.shape == y2.shape  # Check the intended matching shape.
    loss = (y_hat - y2).square().mean()  # Loss tensor with shape ().
    losses.append(loss.item())  # Keep a Python number for plotting or inspection.

    loss.backward()  # Calculate w and b gradients for the current loss.
    with torch.no_grad():
        w -= lr * w.grad  # Update the existing weight's value.
        b -= lr * b.grad  # Update the existing bias's value.

    w.grad = None  # Prevent old gradients from accumulating in the next iteration.
    b.grad = None  # Clear the bias gradient as well.

print(f"w={w.item():.3f}, b={b.item():.3f}")  # Inspect the learned parameters.
```

`.item()` extracts a Python number from a one-element tensor. It is suitable for recording values, but that Python number cannot be used for backpropagation. Call backward on the original `loss` tensor.

Every iteration here uses all 100 samples, so it is one full-batch training update. There is one weight and one bias, but all 100 samples contribute to loss and gradient calculations. **Parameter count is not a count of the computation performed.**

Because targets include noise, the learned parameters need not be exactly `w=2` and `b=1`. Gradients do not reveal the distance to the optimum, so an excessively large learning rate can make loss increase.

## 10. What must be placed on the same CPU, CUDA, or MPS device?

A **device** is where tensors are stored and calculations run. `cpu` uses the CPU, `cuda` is the NVIDIA GPU path in this example, and `mps` is the Apple GPU path on supported Macs. `is_available()` returns a Boolean indicating current availability.[^device]

```python
if torch.cuda.is_available():
    device = "cuda"  # Choose CUDA when available.
elif torch.backends.mps.is_available():
    device = "mps"  # Otherwise choose MPS when available.
else:
    device = "cpu"  # Use the CPU when neither GPU path is available.

t = torch.randn(2, 2)  # Create a tensor on the default device.
t_on_device = t.to(device)  # Keep the returned tensor on the requested device.
print(t_on_device.device)  # Inspect its actual device.
```

`Tensor.to(device)` returns a tensor; it does not automatically rebind the original variable name. If device and data type already match, it may return the original tensor. The example changes only the device, so shape and the meaning of the values are preserved.

A model's `.to(device)` transforms registered parameters and other relevant state within the model. Do not assume it behaves identically to tensor `.to()`. Prepare training in this order: **create the model → move it to the device → create the optimizer from that model's parameters**. Move the targets used for loss computation as well as the inputs to the appropriate device.

## 11. Do nn.Module and nn.Linear perform training for us?

`nn.Module` is the base class used to define models and layers. `__init__` prepares layers and parameters when an object is created. `forward` describes how inputs become outputs. `self` refers to the current object, and `super().__init__()` performs the base `nn.Module` initialization.

```python
from torch import nn  # Import the model building classes.

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize the machinery managing modules and parameters.
        self.linear = nn.Linear(1, 1)  # One input feature → one output value.

    def forward(self, x):
        return self.linear(x)  # Calculate predictions with the prepared layer.

model = LinearRegression()  # Create a model; it has not been trained.
prediction = model(x2)  # Run a forward pass; this alone does not update parameters.

for name, p in model.named_parameters():
    print(name, tuple(p.shape), type(p).__name__)  # Inspect name, shape, and type.
```

The parameters are `linear.weight`, shaped `(1, 1)`, and `linear.bias`, shaped `(1,)`. Both arguments of `nn.Linear(input_features, output_neurons)` are integers, not shape tuples such as `(1,)`.[^linear]

`nn.Linear` creates and manages weights and biases as **Parameters**. A Parameter is a special tensor registered as a model parameter. Attaching the layer to `self.linear` lets the model discover its parameters. Assigning any ordinary tensor to an attribute does not automatically register it as a training parameter.

`named_parameters()` provides parameter names and tensors, whereas `parameters()` provides only the tensors, one after another.

| Manual code | Model-and-optimizer equivalent |
| --- | --- |
| Directly created `w` and `b` | A layer's `weight` and `bias` |
| `(pred - target).square().mean()` | Calling an object created by `nn.MSELoss()` |
| `w -= lr * w.grad` | Basic SGD optimizer `step()` |
| `w.grad = None` | `zero_grad(set_to_none=True)` |

## 12. Why does a 2 → 16 → 2 model have a (16, 2) weight tensor?

We now switch to classifying two-feature inputs as class 0 or class 1. An **MLP** connects multiple linear layers and activation functions. This model has one hidden layer and one output layer. “Two layers” counts the parameterized linear layers; “16” counts neurons in the hidden layer, not hidden layers.

A **neuron** multiplies each input feature by its corresponding weight, adds the products, and adds a bias. The first layer's 16 neurons receive the same two features, but each calculates one value using its own weights and bias.

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize the model's base machinery.
        self.fc1 = nn.Linear(2, 16)  # Two features → sixteen hidden values.
        self.fc2 = nn.Linear(16, 2)  # Sixteen hidden values → two class scores.

    def forward(self, x):
        z1 = self.fc1(x)  # (B, 2) → (B, 16); B is the current sample count.
        h1 = torch.relu(z1)  # (B, 16) → (B, 16); replace negative values with zero.
        logits = self.fc2(h1)  # (B, 16) → (B, 2); return unnormalized scores.
        return logits
```

`z1` is the first linear layer's output, and `h1` is its output after ReLU. Calling `h1` a “16-dimensional hidden representation” does not change the fact that this tensor is two-dimensional, with shape `(B, 16)`. Its 16 columns are new features calculated from the original features in different ways. Developing such features through learned weights is **representation learning**. This is not merely copying values into extra columns.

`nn.Linear` stores weights as **`(output_neurons, input_features)`**. The first layer's weight shape is therefore `(16, 2)`. Each weight row contains the weights used by one output neuron.

<figure class="pt-figure">
<img src="/pytorch-training-basics/linear-weight.svg" alt="X is B by 2, stored weight is 16 by 2, and transposed weight is 2 by 16. X times transposed weight plus bias produces B by 16.">
<figcaption>Rows of the input represent samples. Rows of the stored weight represent the weight sets of output neurons. A row does not have the same meaning in every tensor.</figcaption>
</figure>

The calculation is equivalent to `X @ weight.T + bias`. For a two-dimensional matrix, `.T` exchanges rows and columns; this is called a **transpose**.[^linear]

```text
X:            (B, 2)
weight:       (16, 2)
weight.T:     (2, 16)
bias:         (16,)
result:       (B, 2) @ (2, 16) + (16,) → (B, 16)

For sample i and neuron j:
z1[i, j] = X[i, 0] * weight[j, 0]
         + X[i, 1] * weight[j, 1]
         + bias[j]
```

When manually writing `X @ W`, arrange `W` as `(input_features, output_neurons)`. `nn.Linear.weight` stores the opposite orientation and transposes it for the calculation. The mathematics is unchanged; the storage orientation differs.

| Parameter tensor | Shape | Number of scalar values |
| --- | --- | ---: |
| `fc1.weight` | `(16, 2)` | 32 |
| `fc1.bias` | `(16,)` | 16 |
| `fc2.weight` | `(2, 16)` | 32 |
| `fc2.bias` | `(2,)` | 2 |
| Total | Four tensors | 82 |

After backward, the gradients stored for these parameters are also four tensors with 82 elements in total. Two layers do not mean two gradients. Intermediate gradients participate in backpropagation, but they are not all retained in `.grad` by default. Larger batches leave the parameter count unchanged while increasing the work contributed by samples.

Without ReLU, a sequence of linear layers can be combined into one linear layer. ReLU introduces an input-dependent nonlinear transformation. Because `torch.relu` is a function called inside `forward`, it does not appear as a registered child module in `print(model)`. Its absence from that list does not mean its calculation is missing.

## 13. What TensorDataset and DataLoader organize

We use `make_moons` to generate classification data. It returns NumPy arrays of coordinates arranged in two moon shapes and their target class indices. We convert those arrays into PyTorch tensors. `noise` and `random_state` are settings for this synthetic dataset; changing them can change training results.

```python
from sklearn.datasets import make_moons  # Generate synthetic classification data.
from torch.utils.data import TensorDataset, DataLoader  # Pair samples and build batches.

X_np, y_np = make_moons(n_samples=500, noise=0.2, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)  # (500, 2); two coordinates per row.
y = torch.tensor(y_np, dtype=torch.long)  # (500,); a class index, 0 or 1, per sample.

dataset = TensorDataset(X, y)  # Pair X[i] with y[i] at the same index.
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)
```

The tensors passed to `TensorDataset` must have matching first-dimension sizes: the same number of samples. `dataset[i]` provides `(X[i], y[i])`. `DataLoader` must receive the dataset it should use.[^data]

`shuffle=True` shuffles sample order when traversing an epoch while preserving input–target pairs. `drop_last=False` includes the smaller final batch; this is the default.

<figure class="pt-figure">
<img src="/pytorch-training-basics/batches.svg" alt="500 samples at batch size 64 form seven batches of 64 and one batch of 52. Processing all eight completes one epoch.">
<figcaption>64 × 7 + 52 = 500. This training loop performs one update per batch and eight updates per epoch.</figcaption>
</figure>

For the first seven batches, inputs `xb` have shape `(64, 2)` and targets `yb` have shape `(64,)`. For the last batch, those shapes are `(52, 2)` and `(52,)`. A batch of 64 samples does not require 64 calls to `model(xb)`: one call produces results for all 64 samples together.

A **step** here means one call to `optimizer.step()`. One hundred epochs produce 800 updates. With all 500 samples in one full batch, each epoch instead produces one update. Inspect the actual data being processed, rather than relying on whether a loop variable is named `step` or `epoch`.

## 14. Minibatch training and epoch loss

Connect the previously defined `MLP`, `device`, `dataset`, and `loader`. **Move the model to its device first**, then create the optimizer using that model's parameters.

```python
torch.manual_seed(42)  # Set the random state used to initialize the model.
model = MLP().to(device)  # Create the model and move it to the computation device.
criterion = nn.CrossEntropyLoss()  # Default mean loss with integer class targets.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Specify parameters and update rule.
epoch_losses = []  # Store one recorded average loss per epoch.

for epoch in range(100):
    epoch_loss_sum = 0.0  # Recorded loss sum for this epoch.
    sample_count = 0  # Number of samples actually processed.

    for xb, yb in loader:
        xb = xb.to(device)  # Move this batch's inputs.
        yb = yb.to(device)  # Move this batch's targets too.
        optimizer.zero_grad(set_to_none=True)  # Clear previous batch gradients.

        logits = model(xb)  # Predict only this batch: (B, 2).
        loss = criterion(logits, yb)  # Compare with this batch's yb, not the entire y.
        loss.backward()  # Calculate current parameter gradients.
        optimizer.step()  # Update parameters using those gradients and Adam's rule.

        batch_size = len(xb)  # The final batch contains 52, not 64.
        epoch_loss_sum += loss.item() * batch_size  # Convert the batch mean to a sum.
        sample_count += batch_size  # Count the samples actually processed.

    epoch_losses.append(epoch_loss_sum / sample_count)  # Record once after all batches.
```

`zero_grad` clears parameter gradients. It does not reset learned weights or Adam's separately maintained history. `step()` performs the update task we previously wrote manually, but **Adam's update formula differs from basic SGD**.

### Why multiply an average, then divide again?

This setup uses default CrossEntropyLoss without class weights or ignored labels. Let `B` be the batch's sample count, and let individual losses be `l₁` through `l_B`.

```text
batch mean loss = (l₁ + ... + l_B) / B
batch loss sum  = loss × B
epoch loss sum  = sum of the batch loss sums
epoch mean loss = epoch loss sum / samples actually processed
```

If the eight batch means are `L₁` through `L₈`, the correctly weighted epoch mean is:

```text
(64*L₁ + 64*L₂ + ... + 64*L₇ + 52*L₈) / 500
```

`(L₁ + ... + L₈) / 8` gives the 52-sample batch the same weight as a 64-sample batch and generally produces a different result. We are not dividing an average again arbitrarily. The calculation is **batch mean → batch sum → total sum → overall mean**.

`loss` is the tensor used for backpropagation. `epoch_loss_sum` and `epoch_losses` contain Python numbers for observation and recording; they do not update the parameters. Indentation matters: accumulate batch contributions inside the inner loop and record the epoch mean once after that loop finishes.

The epoch record combines losses calculated using different parameter states across batches. It need not equal the loss obtained by reevaluating the whole dataset with the model at the end of the epoch. Calling `loss.item()` after `step()` does not retroactively recalculate an already-computed loss with the new parameters.

The code counts samples actually processed. When every sample is used exactly once, this equals `len(dataset)`. Dropping samples or sampling with replacement may change that relationship. With class weights or `ignore_index`, check the loss function's averaging denominator as well.[^loss]

## 15. Who supplies the p in lambda p?

Comparing optimizers does not require copying the complete training loop for each choice. Keep the common loop and pass **a function that creates an optimizer**.

```python
def make_sgd(p):
    return torch.optim.SGD(p, lr=0.01)  # Create SGD using the supplied model parameters.

optimizer_fn = make_sgd  # Pass the function without calling it. No optimizer exists yet.
```

The following `lambda` has the same role. **A lambda expression defines an unnamed function.** The name `p` before the colon is a function parameter; the expression after the colon is evaluated and returned when the function is called.[^python]

```python
optimizer_fn = lambda p: torch.optim.SGD(p, lr=0.01)  # Define a function; do not execute its body yet.
```

A function parameter named `p` and a model's learned parameters are different concepts. `p` is a name for a supplied value. The model parameters are the actual weight and bias tensors.

<figure class="pt-figure">
<img src="/pytorch-training-basics/lambda.svg" alt="The lambda function is passed to train_with as optimizer_fn. Calling it with model.parameters supplies p, after which it creates and returns an SGD optimizer.">
<figcaption>p is not SGD itself. The call optimizer_fn(model.parameters()) supplies its value. No function calls itself recursively here.</figcaption>
</figure>

`model.parameters()` returns an **iterator**, an object that provides items one at a time. For this MLP, it provides four parameter tensors. `p` does not mean one individual weight.

The execution order is:

```text
1. Evaluate the lambda expression to create a function object.
2. Pass that function to train_with.
3. Create a new model inside train_with.
4. Call model.parameters() to obtain an object providing that model's parameters.
5. Call optimizer_fn with that object. This is when p receives its value.
6. Execute the function body, torch.optim.SGD(p, lr=0.01).
7. Return the new SGD object and store it in optimizer.
```

Because the model is created inside `train_with`, the caller passes a function that can create an optimizer for the new model, rather than creating an optimizer for a different model beforehand.

## 16. Compare optimizers from the same initial model

This experiment uses all 500 samples together in full-batch training. It uses the earlier `X`, `y`, `MLP`, and `device`. Each `train_with` call creates a new model and returns a list of loss numbers. Resetting the seed inside the function matches model initialization conditions within the same environment.

```python
X_dev = X.to(device)  # Move all inputs for full-batch training.
y_dev = y.to(device)  # Move all targets too.

def train_with(optimizer_fn, epochs=200):
    torch.manual_seed(0)  # Match initialization conditions for each experiment.
    model = MLP().to(device)  # Create a fresh model on each call.
    criterion = nn.CrossEntropyLoss()  # Compare class scores with target indices.
    optimizer = optimizer_fn(model.parameters())  # This call supplies the lambda's p.
    losses = []  # Loss history for this experiment.

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)  # Clear old gradients.
        logits = model(X_dev)  # Calculate scores for all 500 samples.
        loss = criterion(logits, y_dev)  # Mean loss for this full batch.
        loss.backward()  # Calculate gradients.
        optimizer.step()  # Apply the chosen optimizer's update.
        losses.append(loss.item())  # Record the loss computed before the update.

    return losses  # Return one Python number per epoch.

results = {}  # Associate experiment names with their loss lists.
results["SGD lr=0.01"] = train_with(lambda p: torch.optim.SGD(p, lr=0.01))
results["SGD lr=0.5"] = train_with(lambda p: torch.optim.SGD(p, lr=0.5))
results["SGD lr=10.0"] = train_with(lambda p: torch.optim.SGD(p, lr=10.0))
results["Adam lr=0.01"] = train_with(lambda p: torch.optim.Adam(p, lr=0.01))
```

`_` is a conventional variable name indicating that the loop index is not used. Each value in `results` contains 200 loss numbers. Because this is full-batch training, each epoch performs one update.

```python
import matplotlib.pyplot as plt  # Plot the loss histories.

plt.figure(figsize=(8, 5))  # Create one plot.
for name, losses in results.items():
    plt.plot(losses, label=name)  # Draw the history for each experiment.
plt.xlabel("epoch")  # Horizontal axis: epoch record order.
plt.ylabel("loss")  # Vertical axis: training loss.
plt.legend()  # Identify each experiment.
plt.show()  # First inspect the full result without manually clipping the y-axis.
```

A small learning rate may reduce loss slowly within a fixed iteration budget. An overly large rate may cause spikes, oscillation, or divergence. But `lr=10` does not guarantee divergence, and Adam is not always the best choice. Initial parameters, data, and iteration count affect results.

A limit such as `plt.ylim(0, 2)` hides losses above two. When investigating instability, inspect the full range first. Also, these are **training-data losses**. Evaluating performance on new data requires a separate validation dataset.

## 17. How SGD, Momentum, RMSProp, and Adam differ

All these methods use gradients to update parameters. The distinction is **which information they use to determine the next update**. The formulas below show basic forms without options such as weight decay. “Basic SGD” here means SGD without momentum; PyTorch's SGD can also be configured with momentum.[^optim]

| Method | Information used to determine the update |
| --- | --- |
| Basic SGD | Current gradient. |
| Momentum | Current gradient and an accumulation of earlier gradients. |
| RMSProp | Current gradient and a history of squared gradients. |
| Adam | A moving average of gradients, a moving average of squared gradients, and initial-bias corrections. |

### Symbols used in the formulas

`t` is the update index, starting with `t=1`. `w_(t-1)` is one particular weight before the update. `g_t` is the derivative of the current batch loss with respect to that weight. `lr` is the learning rate. Each formula describes one scalar weight; it is also applied separately to other weights and biases.

**Basic SGD** subtracts the learning rate times the current gradient.

<div class="pt-math">
$$w_t = w_{t-1} - lr\,g_t$$
</div>

For **Momentum**, `u_t` is the accumulated value including earlier gradients, and `μ` controls how much previous accumulation is retained, usually with a value at least zero and below one. Starting from `u_0=0`, the basic form is:

<div class="pt-math">
$$u_t = \mu u_{t-1} + g_t,\qquad w_t = w_{t-1} - lr\,u_t$$
</div>

Repeated gradients of the same sign can accumulate. When signs change, the new gradient can partially cancel the previous value. The stored information includes gradient magnitude as well as sign.

A **moving average** combines an existing record with a newly received value. Here, the influence of older values decreases as updates continue. **RMSProp** maintains a moving average of squared gradients, `v_t`, separately for each weight. `β` is the retention ratio, at least zero and below one. `ε` is a small positive number added to reduce problems such as division by zero. Initialize `v_0=0`.

<div class="pt-math">
$$v_t=\beta v_{t-1}+(1-\beta)g_t^2$$
$$w_t=w_{t-1}-lr\frac{g_t}{\sqrt{v_t}+\varepsilon}$$
</div>

The square-root operation appears in the denominator. A weight with persistently large gradients develops a larger squared-gradient record, adjusting the size of its update. The weights do not all share a single record.

**Adam** uses both a moving average of gradients, `m_t`, and a moving average of squared gradients, `v_t`. `β₁` and `β₂` are the respective retention ratios. Both records start at zero.

<div class="pt-math">
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$
</div>

Starting at zero makes these records small early on. The following corrects that effect. `β₁^t` and `β₂^t` mean the respective values raised to the power `t`.

<div class="pt-math">
$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}$$
$$w_t=w_{t-1}-lr\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}$$
</div>

The numerator reflects recent gradient signs and magnitudes. The denominator adjusts update size using recent magnitude information. Adam uses ideas associated with Momentum and RMSProp, but it does not run those two optimizers one after the other. With every method, distinguish the gradient calculated by `backward()` from the update applied by `step()`.

## Appendix: Python expressions that choose values

The device-selection expression `"cuda" if condition else "cpu"` is a **conditional expression**: it chooses one value according to a condition. The same syntax works inside a list comprehension. A **list comprehension** creates a new list by combining iteration with a value expression.[^python]

```python
numbers = [1, 2, 3, 4, 5, 6]  # Numbers to process.

# The trailing if filters elements. % calculates the remainder.
even_numbers = [x for x in numbers if x % 2 == 0]  # [2, 4, 6]

# The leading if else chooses a value for each element.
parity = ["even" if x % 2 == 0 else "odd" for x in numbers]
# ['odd', 'even', 'odd', 'even', 'odd', 'even']

# Another conditional expression can follow else.
sizes = ["small" if x < 3 else "medium" if x < 5 else "large" for x in numbers]
# ['small', 'small', 'medium', 'medium', 'large', 'large']
```

Take one element, evaluate the condition, and append the chosen value to the result. For complicated conditions, an ordinary `for` loop with `if/elif/else` is often easier to read.

[^shape]: PyTorch documentation: [unsqueeze](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html), [matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html), [broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html).
[^random]: PyTorch documentation: [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [linspace](https://docs.pytorch.org/docs/stable/generated/torch.linspace.html), [randn](https://docs.pytorch.org/docs/stable/generated/torch.randn.html).
[^activation]: PyTorch documentation: [Sigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html), [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), [Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html).
[^loss]: PyTorch documentation: [MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
[^autograd]: PyTorch documentation: [autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html), [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html).
[^assignment]: Python documentation: [assignment and augmented assignment](https://docs.python.org/3/reference/simple_stmts.html). For tensor restrictions, see PyTorch's [in-place operations with autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd).
[^device]: PyTorch documentation: [Tensor.to](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html), [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [MPS](https://docs.pytorch.org/docs/stable/notes/mps.html).
[^linear]: PyTorch documentation: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).
[^data]: PyTorch documentation: [TensorDataset and DataLoader](https://docs.pytorch.org/docs/stable/data.html).
[^python]: Python documentation: [conditional expressions and comprehensions](https://docs.python.org/3/reference/expressions.html), [lambda expressions](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions).
[^optim]: PyTorch documentation: [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html), [RMSprop](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html), [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).
