+++
title = "Predicting Tomorrow's Visitor Count with an RNN: From Input Data to Training and Evaluation"
description = "Follow one task—predicting the next day from 14 days of records—to connect input windows, batches, hidden states, shared weights, RNNCell and RNN, backpropagation, and training. Check the essentials with diagrams and executable code, with manual differentiation in a separate reference section."
date = 2026-11-02
slug = "rnn-window-to-forecast"

[extra]
katex = true
toc = true
styles = ["rnn-window-to-forecast/article.css"]
+++

Suppose we want to predict tomorrow's visitor count from the previous 14 days. The first decision concerns the data: **which dates belong in one input, and which date supplies the target for that input?** We need to settle this before choosing the neural network.

We then need a computation that processes those dates in order. A **recurrent neural network (RNN)** combines the current input with the result of an earlier computation to produce a new result. The intermediate result passed to the next time step is called the **hidden state**. The hidden state is computed as the model processes the input; the weights are the learned parameters used to compute it.

This post follows one prediction task through data tables, diagrams, array structures, and code. We first connect inputs to targets in a short record, then extend the same operation to 84 days. Next, we run the RNN, predict the following day, train the model, and evaluate it on future dates. The main example uses a basic RNN with **one layer, one direction, and input windows of equal length**. Unidirectional means that it processes the inputs from beginning to end.

You can download the [complete Python example](/rnn-window-to-forecast/examples-en.py), [manual backpropagation checks](/rnn-window-to-forecast/bptt-en.py), and [Rust comparison example](/rnn-window-to-forecast/recurrence-en.rs). On a first reading, follow the main text; expand the reference sections later for calculations you want to examine more closely. The API descriptions were checked against the official PyTorch documentation on November 2, 2026.

## 1. Connect the complete workflow first

A **sample** is one input supplied to the model. Here, one input window containing 14 days of records is one sample. A **batch** groups several samples for processing together, and `B` is the number of samples processed in the current call.

In the diagram, the `RNN` computes four hidden values at each time step. `Linear` uses the final four hidden values to produce one prediction for the next day. The **loss** is a number that measures the difference between predictions and targets. A **gradient** describes how the loss responds to a small change in a particular value. An **optimizer** uses the computed gradients to update the numbers being learned.

<figure class="rnn-guide-figure">
<img src="/rnn-window-to-forecast/overview-en.svg" alt="The RNN processes a 14-day input. Linear uses the final hidden state to predict the next day. Predictions and targets produce a loss, backpropagation computes gradients, and the optimizer updates the weights." loading="lazy">
<figcaption>Producing a prediction computes values using the current weights. The loss then provides the starting point for computing gradients, and step updates the weights.</figcaption>
</figure>

When reading a neural network, distinguish these five parts.

| What to identify | Its role in this model |
| --- | --- |
| Input | Each sample contains the previous 14 days of visitor counts in order. |
| Neurons and structure | Four hidden neurons in the RNN compute at every time step, and Linear makes the prediction at the end. |
| Intermediate values | The four newly computed hidden values from each time step enter the next computation. |
| Weights | The RNN and Linear each store their own numbers that training updates. |
| Final output | Each input window corresponds to one prediction of the next day's visitor count. |

Reducing a visitor record to a single mean removes information about order. For example, `10 → 20 → 30` and `30 → 20 → 10` have the same mean, but their final values and changes from beginning to end differ. In an RNN, a previously computed state enters the next computation, so changing the order can change the result. This does not guarantee different results for every possible setting of the weights.

## 2. Pair one input window with exactly one target

Start with just seven days of records. We use visitor counts of `10, 20, 30, 40, 50, 60, 70` to make the positions easy to follow. The task is to **predict the next day from the preceding three days**.

An **index** is a position number used to select a value. Python indices start at 0, so the first day has index 0 and the seventh day has index 6. The **length** of the complete record is 7. The length and the final index are different quantities.

| Sample number | Input indices | Input values | Target index | Target |
| ---: | --- | --- | ---: | ---: |
| 0 | 0, 1, 2 | `[10, 20, 30]` | 3 | 40 |
| 1 | 1, 2, 3 | `[20, 30, 40]` | 4 | 50 |
| 2 | 2, 3, 4 | `[30, 40, 50]` | 5 | 60 |
| 3 | 3, 4, 5 | `[40, 50, 60]` | 6 | 70 |

<figure class="rnn-guide-figure">
<img src="/rnn-window-to-forecast/windows-en.svg" alt="A three-day input window moves one position at a time through seven days of records. Each of the four input windows has a target at the immediately following index." loading="lazy">
<figcaption>The final input value and the target belong to different dates. The first three dates still serve as inputs for predicting later dates.</figcaption>
</figure>

We could also form the input window `[50, 60, 70]`. However, the record contains no following day, so we cannot attach a training target to it. There are therefore **four samples with available targets**. After checking this relationship, we can express the count as follows.

`Number of windows with targets = length of the complete record − input window length`

This relationship applies when the window moves one position at a time and the target is the single value immediately after it. Predicting several days at once or changing the distance between consecutive windows changes the count.

Python's `range(4)` provides `0, 1, 2, 3`. The slice `data[0:3]` also excludes its ending boundary, 3, selecting indices 0, 1, and 2. If `i` is the start of a window, its input is therefore `data[i:i+3]` and its target is `data[i+3]`.[^python-sequence]

```python
data = [10, 20, 30, 40, 50, 60, 70]
seq_len = 3  # Number of days to include in an input.

for i in range(len(data) - seq_len):  # range(4): starting indices 0–3.
    window = data[i:i + seq_len]     # Include the start; exclude the end.
    target = data[i + seq_len]       # Select the position just after the input.
    print(window, "->", target)

# [10, 20, 30] -> 40
# [20, 30, 40] -> 50
# [30, 40, 50] -> 60
# [40, 50, 60] -> 70
```

### Extend the same operation to 84 days

For the exercise, we repeat a synthetic weekly pattern 12 times and add small random variations. These are not records collected from an actual business. Dividing the visitor counts by 100 is a choice that reduces the numerical scale used in the computation. It does not turn them into probabilities, so a value can still exceed 1.

```python
import torch
from torch import nn  # Use the neural network classes in torch.nn as nn.

torch.manual_seed(1)  # Starting configuration for this data's random numbers.
weekly = torch.tensor(
    [62, 55, 58, 60, 75, 118, 105],
    dtype=torch.float32,  # Store 32-bit floating-point numbers.
)
value_scale = 100.0  # Multiply predictions by this to restore visitor units.
repeated = weekly.repeat(12)  # Repeat the whole week: (7,) -> (84,).
noise = torch.randn(len(repeated)) * 3.0  # Synthetic daily variation.
visitors = (repeated + noise) / value_scale  # Shape (84,).
```

A **tensor** is a PyTorch data type that stores numbers together with their array structure. Its **shape** lists the size of each axis in order. The shape `(84,)` of `visitors` means it has one axis containing 84 values.

Now set the input window length to 14. The first 14 days do not have 14 earlier days in the record, so they cannot serve as target dates. Each date from day 15 through day 84 does have the required history, giving us 70 input windows with targets.

| Sample number | Input dates | Python input selection | Target |
| ---: | --- | --- | --- |
| 0 | Days 1–14 | `visitors[0:14]` | `visitors[14]`, day 15 |
| 1 | Days 2–15 | `visitors[1:15]` | `visitors[15]`, day 16 |
| 69 | Days 70–83 | `visitors[69:83]` | `visitors[83]`, day 84 |

The following function repeats these selections. Its **arguments** are the tensor containing the actual records and the window length. Its **return values** are the inputs `X` and the targets `y`. The `append()` method adds an item to the end of a list.

```python
def make_windows(values, seq_len):
    windows = []
    for i in range(len(values) - seq_len):
        windows.append(values[i:i + seq_len])  # Each tensor has shape (14,).

    # stack groups tensors of the same size along a new axis.
    X_2d = torch.stack(windows, dim=0)  # 70 windows -> (70,14).
    # unsqueeze adds an axis of size 1 without changing the number of values.
    X = X_2d.unsqueeze(-1)  # Add the final feature axis -> (70,14,1).
    y = values[seq_len:].unsqueeze(-1)  # Values for the next days -> (70,1).
    return X, y

seq_len = 14
X, y = make_windows(visitors, seq_len)
print(tuple(X.shape), tuple(y.shape))  # (70,14,1) (70,1)
```

`visitors[14:]` selects from index 14 through the end. Because the final index is 83, it is equivalent to `visitors[14:84]`. Omitting the ending boundary already includes the rest of the sequence, so no additional `+1` is needed.

`unsqueeze(-1)` changes the structure so that each day's visitor count sits inside a feature list, such as `[visitor_count]`. It does not add observations. We also prepare the targets as `(70,1)` so that each sample's target can be compared with its one predicted number.[^stack][^unsqueeze]

## 3. Read samples, time steps, and features along separate axes

A **sequence** is an ordered group of data. In this task, one input window is both one sequence and one sample. A **time step** is a position within that sequence, and a **feature** is one kind of information observed at a time step.

| Symbol | Meaning | In this example |
| --- | --- | --- |
| `N` | Total number of prepared samples | 70 input windows |
| `B` | Number of samples in the current model call | 2 when we select two windows for explanation |
| `T` | Number of time steps in one sample | 14 days per window |
| `F` | Number of input features at one time step | One feature: visitor count |
| `H` | Number of hidden values computed at each time step | 4, chosen by the developer |

`F=1` follows from using only visitor counts as inputs. An RNN's input is not always a single number. Including visitor count, temperature, rainfall, and advertising spend would give `F=4`. If we still observed 14 days, `T=14` would remain unchanged. The hidden size `H` is not automatically determined by the number of features or days.

The complete `X` has shape `(N,T,F)=(70,14,1)`. Selecting its first two windows as `xb = X[:2]` gives shape `(B,T,F)=(2,14,1)`. **The size of the complete dataset and the size of the current batch describe different things.**

| Axis number in `xb` | Meaning | Axis size | Valid indices |
| ---: | --- | ---: | --- |
| `dim=0` | Which sample? | 2 | 0, 1 |
| `dim=1` | Which time step in that sample? | 14 | 0–13 |
| `dim=2` | Which feature at that time step? | 1 | 0 |

Keep axis number 1, axis size 14, and final index 13 separate. The tensor has three axes, so it is a **three-dimensional tensor**. Having 70 samples does not make it 70-dimensional.

```python
xb = X[:2]  # Select the first two input windows as a batch.
print(tuple(xb.shape), xb.ndim, xb.numel())
# (2,14,1) 3 28
# shape: sizes of the axes / ndim: number of axes / numel(): total values.
```

**Indexing** selects the required parts of existing values; it does not compute a new prediction.

| Selection | Selected content | Result shape |
| --- | --- | --- |
| `xb[0]` | The first sample's 14 days | `(14,1)` |
| `xb[:1]` | The first sample, retaining the batch axis | `(1,14,1)` |
| `xb[:, 0, :]` | The first time step from each of the two samples | `(2,1)` |
| `xb[:, -1, :]` | The final input time step from each of the two samples | `(2,1)` |
| `xb[:, -1:, :]` | The same final time step, retaining the time axis | `(2,1,1)` |

`:` selects the entire axis. Selecting an axis with a single integer removes it from the result; selecting with a slice retains it. To feed `xb[0]` back into this post's model, add the batch axis with `unsqueeze(0)`, or select it as `xb[:1]` from the start.

## 4. The RNN computes from the current input and the previous state

A **weight** is a learned number multiplied by an input. A **bias** is a learned number added after the multiplication results have been summed. Together, weights and biases are **parameters**. A **neuron** computes a weighted sum of the values connected to it and applies an activation function to produce one result.

A **weighted sum** multiplies each value by its corresponding weight and adds the results. An **activation function** transforms that sum into the next result. Here, we use the hyperbolic tangent, `tanh`. It acts independently on each number, producing a value between −1 and 1. It is different from the trigonometric function `tan`.[^tanh]

An RNN **cell** computes the complete hidden state for one time step. Our cell has four hidden neurons. Each neuron reads **all of the current input and all four previous hidden values**, then uses its own weights to produce one result. The four results remain as four values in the new hidden state; we do not add them together into a single number.

We also need a consistent time notation. In the main text, `t` is a time index starting at 0, just like in the code. `h_init` is the initial state before processing any input, and `h_0` is the state produced by processing the input at index 0. The final input index is 13, and the final state is `h_13`.

| Processing step | Current input | Previously available state | State to compute |
| --- | --- | --- | --- |
| First time step, `t=0` | The first date in each sample | `h_init`: four zeros per sample | `h_0`: four hidden values per sample |
| Second time step, `t=1` | The second date in each sample | The newly computed `h_0` | `h_1` |
| Final time step, `t=13` | The fourteenth date in each sample | The previously computed `h_12` | `h_13` |

<figure class="rnn-guide-figure">
<img src="/rnn-window-to-forecast/recurrence-en.svg" alt="Two samples use the same RNN weights, but each computes from its own initial and previous hidden states. States connect the first through the final time step, with no state passed between samples." loading="lazy">
<figcaption>Time order continues within each sample. The cells drawn at different time steps represent repeated computations of the same RNN layer, not different layers.</figcaption>
</figure>

The first window begins at original record index 0, and the second begins at index 1. Consequently, `t=0` in the two samples refers to different original dates. Selecting the “same time step” across a batch means selecting **the same position within each input window**.

The two samples' states are stored in `(B,H)=(2,4)`. Sample 0's four values enter sample 0's next computation, while sample 1's four values enter sample 1's next computation. The samples do not combine their hidden states or pass them to each other. Even two input windows that overlap in their original dates are handled as separate samples in this model.

### Compute new hidden states and reuse the weights

| Aspect | Hidden state | Weights and biases |
| --- | --- | --- |
| What it is | An intermediate result computed from the input and previous state | Parameters stored by the model object |
| After processing one time step | A new state has been computed. | The current parameter values have been used. |
| At the next time step | The state from the preceding step becomes an input. | The same parameters are used again. |
| Where it changes in a standard training loop | The cell's forward computation | `optimizer.step()` |

**Weight sharing** means repeatedly using the same stored parameters in computations at different time steps. It does not mean creating separate weights at every time step and setting their values equal.

Write the weights for the current input as `W_ih` and those for the previous hidden state as `W_hh`. Here, `i` refers to input and `h` to hidden. PyTorch stores them in the following structures.[^rnncell]

| Parameter | Shape | Count and reason in this example |
| --- | --- | --- |
| `weight_ih` | `(H,F)=(4,1)` | Four values: each of four neurons uses one input feature. |
| `weight_hh` | `(H,H)=(4,4)` | Sixteen values: each of four current neurons uses four previous hidden values. |
| `bias_ih`, `bias_hh` | Each `(H,)=(4,)` | Eight values: PyTorch stores two sets of biases. |

The RNN therefore has `4+16+8=28` learned numbers. Extending the input from 14 to 28 days leaves this RNN's parameter count at 28. The number of processed time steps, the computation, and the intermediate values to store increase. **Parameter count and computational work are different quantities.**

<details>
<summary>Check how the current input and previous state combine in the calculation</summary>

`x_t` is the current input `(B,F)`, `h_prev` is the previous state `(B,H)`, and `a_t` is the weighted sum before activation `(B,H)`. `h_t` is the new state `(B,H)`. The following code directly expresses the computation inside the cell.

```python
# @ is matrix multiplication; .T swaps the two axes of these 2D weights.
a_t = (
    x_t @ cell.weight_ih.T       # (B,F) @ (F,H) -> (B,H).
    + h_prev @ cell.weight_hh.T  # (B,H) @ (H,H) -> (B,H).
    + cell.bias_ih              # Input bias for each neuron.
    + cell.bias_hh              # State bias for each neuron.
)
h_t = torch.tanh(a_t)  # Transform each weighted sum into a hidden value.
```

`W_ih[j,k]` connects input feature `k` to current neuron `j`. `W_hh[j,q]` connects previous hidden value `q` to current neuron `j`. The first index identifies the current neuron; the second identifies the position of the value that neuron reads.

With four input features and three hidden neurons, each neuron has four input weights and three state weights. The total parameter count is `3×4 + 3×3 + 2×3 = 27`. The structural principle is the same as in the visitor model; only `F` and `H` differ. The [connection diagrams for a multivariate RNN](@/rnn-sequence-hidden-state.en.md) examine this case in more detail.

The two biases can be added and represented as one effective bias in the forward computation. However, PyTorch manages them as separate parameters, so we should not assume that training behaves identically to another implementation with just one bias.

</details>

## 5. RNNCell processes one time step; RNN receives the sequence

A **class** defines an object's structure and behavior. `nn.RNNCell(...)` calls the class to **create a model object**. Calling the resulting object as `cell(x_t, state)` computes with actual data. The arguments to these two calls have different roles.

| Stage | Code | What is supplied |
| --- | --- | --- |
| Create the object | `nn.RNNCell(input_size=1, hidden_size=4)` | The number of input features and hidden values |
| Compute with data | `cell(x_t, state)` | The current input tensor and previous state tensor |

We do not pass weights as arguments to `cell()` because the cell object stores its weights internally. `state` is the result of an earlier computation, not a weight.

```python
cell = nn.RNNCell(input_size=1, hidden_size=4)
xb = X[:2]  # (B=2,T=14,F=1).
state = xb.new_zeros(xb.shape[0], cell.hidden_size)
# new_zeros: create zeros with xb's dtype and device. Initial state: (2,4).

states = []
for t in range(xb.shape[1]):  # 0–13: iterate over time steps, not samples.
    x_t = xb[:, t, :]        # Current inputs for both samples, (2,1).
    next_state = cell(x_t, state)  # (2,1) and (2,4) -> (2,4).
    states.append(next_state)
    state = next_state      # Set the state to use in the next iteration.

cell_output = torch.stack(states, dim=1)  # All time-step states, (2,14,4).
```

The shorter form `state = cell(x_t, state)` has the same meaning. The right-hand side first uses the existing `state` to compute a new result; the name `state` on the left then refers to that result. This is not a recursive call to itself, and it does not update the weights.

Completing `h_1` requires `h_0`. This dependency makes the state computation follow time order. Operations that do not depend on the previous state, such as multiplying an input by its weights, can still be performed ahead of time. We do not need to interpret the dependency as a claim that the hardware executes every operation one at a time.

### unbind specifies which axis to iterate over

`xb.unbind(dim=1)` returns a tuple of tensors selected one at a time along the time axis. That selected axis is removed from each returned tensor. We can therefore write the preceding loop as follows.[^unbind]

```python
state = xb.new_zeros(xb.shape[0], cell.hidden_size)
for x_t in xb.unbind(dim=1):  # 14 tensors, each with shape (2,1).
    state = cell(x_t, state)  # Update each sample's own state.
```

If we write the original record's `visitors[k]` as `v_k`, the first iteration receives `[[v_0],[v_1]]` and the second receives `[[v_1],[v_2]]`. The position within each window has advanced by one. In contrast, `for row in xb` iterates over the first axis—the sample axis—and selects two windows, each with shape `(14,1)`.

### RNN performs the time loop internally

```python
rnn = nn.RNN(
    input_size=1,       # Input features at one time step.
    hidden_size=4,      # Hidden values at one time step.
    num_layers=1,       # Number of RNN layers.
    nonlinearity="tanh",  # Activation applied to the weighted sum.
    batch_first=True,  # Arrange input and output axes as (B,T,features).
)
output, h_n = rnn(xb)  # Supply all 14 days in one call.
```

Although we call `rnn(xb)` once, it computes states for all 14 time steps internally. Omitting the initial state makes it start from zeros. It does not automatically carry the previous call's final state into the next call.[^rnn]

The separately created `cell` and `rnn` above also have separately initialized weights. Matching structures and shapes therefore do not guarantee matching output numbers. The downloadable example copies the same weights and biases into both objects before checking that their results agree within a numerical tolerance.

## 6. Select the final state and compute a prediction

First, distinguish the two tensors returned by the RNN. For the single-layer, unidirectional setup in the main example, they are as follows.

| Return value | Stored content | Shape for two samples |
| --- | --- | --- |
| `output` | Hidden states at every time step from the final RNN layer | `(2,14,4)` |
| `h_n` | The final hidden state for each layer and direction | `(1,2,4)` |

The leading 1 in `h_n` is not the sample count. It indicates one layer–direction combination. The following 2 is the sample count. `batch_first=True` applies to the input and `output`; it does not change the axis order of `h_n`.

<figure class="rnn-guide-figure">
<img src="/rnn-window-to-forecast/outputs-en.svg" alt="Selecting all four hidden values at the final time step for both samples changes output from (2,14,4) to (2,4). Selecting h_n[0] from h_n with shape (1,2,4) gives the same state. Linear computes one prediction per sample from it." loading="lazy">
<figcaption>Selecting the final time step picks values that have already been computed. Linear uses those selected values to compute new prediction numbers.</figcaption>
</figure>

Read `output[:, -1, :]` from left to right: **select all samples, their final time step, and all hidden values**. It does not select only the final neuron. Only the time axis disappears; each sample retains all four values.

```python
last = output[:, -1, :]  # (2,14,4) -> (2,4).
last_from_hn = h_n[0]    # (1,2,4) -> (2,4).
assert torch.allclose(last, last_from_hn)
# allclose returns a bool indicating whether matching values agree within tolerance.
```

The 0 in `h_n[0]` selects a layer–direction combination, not just sample 0. The complete `output` and complete `h_n` cover different sets of states, but selecting the final state as above gives the same values.

The four numbers in that state are not yet a visitor-count prediction. Our **prediction head**, `nn.Linear(4,1)`, takes the four hidden values and computes one prediction. `Linear` transforms the final axis while preserving the axes before it.[^linear]

| Tensor supplied | Result from Linear | Meaning |
| --- | --- | --- |
| Final state `(B,4)` | `(B,1)` | One next-day prediction per sample |
| All states `(B,14,4)` | `(B,14,1)` | One prediction per time step |

We use the first option because our task requires one next-day prediction from each input window. The hidden size of 4 is a chosen structural setting; it was not calculated from the 14 input days.

### Separate model construction from computation with data

`nn.Module` is the base class used to define PyTorch models. `__init__` runs when the model is created and prepares its internal layers. `forward` describes the computation performed when the created model receives an input.

```python
class VisitorRNN(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()  # Run nn.Module's initialization.
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(
            in_features=hidden_size,  # Number of final hidden values received.
            out_features=1,          # One next-day value per sample.
        )

    def forward(self, x):
        output, _ = self.rnn(x)  # (B,14,1) -> (B,14,4).
        # _ conventionally names the h_n return value we do not use here.
        last = output[:, -1, :]  # (B,4): select existing values.
        prediction = self.head(last)  # (B,1): compute new values.
        return prediction

model = VisitorRNN(hidden_size=4)  # Prepare the structure and parameters.
prediction = model(X[:2])          # Execute forward with actual data.
print(tuple(prediction.shape))     # (2,1).
```

`model.rnn(X)` runs only the internal RNN and returns two tensors. `model(X)` runs everything written in `forward`—the RNN, final-time-step selection, and Linear—and returns one prediction tensor. Calling both consecutively computes the RNN twice. To reuse a previously computed `output`, write `model.head(output[:, -1, :])` explicitly.

This model has **33 learned numbers**: 28 RNN parameters plus Linear's four weights and one bias. The 70 input windows and 14 time steps do not each have their own separate model weights.

## 7. Distinguish predictions, loss, gradients, and weight updates

When comparing predictions and targets, we must match values belonging to the same sample. Each sample has one prediction and one target in this task, so we give both shape `(B,1)`.

The **error** is the prediction minus the target. Squaring an error gives a **squared error**. Adding all squared errors gives the sum of squared errors; averaging them gives the **mean squared error (MSE)**. A loss does not have to be an average. We use an average here because we chose MSE.

```python
xb, yb = X[:2], y[:2]  # Inputs and targets for the same two samples.
prediction = model(xb)  # (2,1).
assert prediction.shape == yb.shape  # Check the correspondence.

errors = prediction - yb         # Error for each sample, (2,1).
squared_errors = errors.square() # Square each element, (2,1).
loss = squared_errors.mean()    # Average all elements; shape ().

loss_fn = nn.MSELoss()          # Default reduction='mean'.
same_loss = loss_fn(prediction, yb)  # The same MSE calculation as above.
```

Shape `()` describes a zero-dimensional tensor containing one number. It has not become a Python `float`. A tensor loss can remain connected to the operations that produced it, allowing it to serve as the starting point for backpropagation. By default, `MSELoss` averages across all output elements. Here, there is one output per sample, so the number of elements being averaged equals `B`.[^mse]

Subtracting a target of shape `(B,)` directly from a `prediction` of shape `(B,1)` can produce an unintended `(B,B)` comparison. This follows from **broadcasting**, the rules for automatically matching array sizes during operations. Keep both tensors as `(B,1)`, or make both `(B,)`.[^broadcast]

### Separate one training step into five stages

The **forward pass** computes values from the input through to the prediction. **Backpropagation** follows the connections in that computation to obtain loss gradients. A typical training step has the following stages.[^optimization]

```python
optimizer = torch.optim.Adam(
    model.parameters(),  # Register all learned parameters in the RNN and Linear.
    lr=0.01,             # Learning rate chosen for this example.
)

optimizer.zero_grad(set_to_none=True)  # 1. Clear previously stored gradients.
prediction = model(xb)                # 2. Predict using the current weights.
loss = loss_fn(prediction, yb)        # 3. Compute loss from predictions and targets.
loss.backward()                      # 4. Compute the loss gradient for each parameter.
optimizer.step()                     # 5. Update parameters using the gradients.
```

`zero_grad()` does not set hidden states or weights to zero. It clears gradients left on the parameters from an earlier computation. `set_to_none=True` clears those stored values to `None` instead of a zero-filled array. The next `backward()` creates the gradients it needs again.[^zero-grad]

`loss.backward()` stores gradients in each parameter's `.grad`. If the recurrent weight matrix has shape `(4,4)`, its gradient also has shape `(4,4)`: each individual weight needs one corresponding gradient value. Our model has 33 learned numbers, so its 33 scalar parameters have 33 corresponding gradients. That does not mean that obtaining them requires only 33 differentiation operations.

`optimizer.step()` modifies the parameters using the prepared gradients. The next forward pass then uses different weights. If we print the `loss` computed before `step()`, it is still **the loss of the predictions made before the update**. Updating the weights does not automatically change an already computed loss value.

The basic update for the simplest form of stochastic gradient descent, SGD, is `new weight = current weight − learning rate × that weight's gradient`. It does not subtract the learning rate multiplied by the weight itself. The gradient describes the local rate of change of the loss; it does not directly give the optimal weight or the optimal distance to move. Adam, selected above, also uses information from earlier gradients, so we should not describe it as the same algorithm as this simple SGD update.[^sgd][^adam]

## 8. Why can the final loss train weights used at earlier time steps?

The final state was computed from the preceding state. That preceding state was itself computed from an earlier state. Consequently, computations from earlier time steps are connected to the loss of the final prediction. Following these connections backward through time to calculate gradients is called **backpropagation through time (BPTT)**.

First, identify the two values related by each gradient. For one hidden neuron in one sample, let `a` be its weighted sum before activation, `h` the result of `tanh(a)`, and `L` the final loss.

| Gradient to calculate | The precise question |
| --- | --- |
| Gradient of `h` with respect to `a` | If weighted sum `a` changes slightly, how much does hidden value `h` change? |
| Gradient of `L` with respect to `h` | If hidden value `h` changes slightly, how much does final loss `L` change? |
| Gradient of `L` with respect to `a` | If weighted sum `a` changes slightly, how much does final loss `L` change? |

The first gradient alone does not tell us the third. **We must connect how a change in `a` changes `h` with how that change in `h` changes the loss.** The rule that multiplies these two rates of change is the **chain rule**. The reference section later examines this connection through small changes.

<figure class="rnn-guide-figure">
<img src="/rnn-window-to-forecast/gradients-en.svg" alt="The loss gradient for the current hidden state passes through the tanh derivative to give the weighted-sum gradient. This branches into gradients for the current parameter uses and the previous hidden state. Contributions from time steps using the same parameter are added to obtain its final grad." loading="lazy">
<figcaption>The gradient for a parameter and the gradient for the previous state are different results. The previous state's gradient continues through the earlier computation that produced that state.</figcaption>
</figure>

The current weighted sum includes the result of applying the recurrent weights to the previous hidden state. At this connection, we can ask two questions.

1. **How would changing a weight used here change the loss?** This calculation produces a term to add to that weight's final gradient.
2. **How would changing the previous hidden state used here change the loss?** This calculation produces a gradient to pass to the preceding time step.

Once we have the second result, we continue differentiating the `tanh` and weighted sum that produced that previous state. We are not moving backward because we cannot compute the previous state's gradient at the current step. We are **using the gradient already obtained for that state to continue through the operations that produced it**. There are also gradient paths for the input weights, biases, and current input; the two questions do not imply that only two kinds of gradient exist.

Backpropagation does not reconstruct past input values. It does not remove `tanh` or predict a previous hidden value again. What we keep calculating is **the rate at which the loss responds to each computed value and each weight**.[^autograd]

### Reusing a weight requires adding the effects of its different uses

Choose one recurrent weight. For example, `W_hh[0,1]` determines how previous hidden value 1 enters the computation of current neuron 0. This **same number** is reused at all 14 time steps in an input window.

Changing that weight can affect computations at several time steps where it was used. Looking only at its direct use at the final time step therefore cannot describe the complete rate of change of the loss. Some contributions may be zero, but we must account for and add the terms from all of its uses to obtain the final gradient of that one weight.

A **time-step contribution** means one term in the sum that forms the shared weight's final gradient, arising from the weight's use at that time step. We have not created a new weight or a new loss for each time step. Each term already accounts for the effect on the final loss through the computations that follow that step.

When differentiating the current use, we treat the previous state supplied to that multiplication as a given value. The fact that this previous state was also produced using the same weight at earlier steps is handled by **continuing to differentiate the earlier operations using the gradient passed to the previous state**. We calculate these paths separately and collect the results for the parameter.

If the weight matrix has shape `(4,4)`, each time step's contribution also has shape `(4,4)`. Adding corresponding entries gives the final `.grad`, again with shape `(4,4)`. More time steps mean more computations to combine, while the number of entries in the final gradient remains equal to the number of weight entries.

We must also distinguish **combining the uses of shared weights within one `backward()` call** from **accumulating `.grad` across several `backward()` calls**. `zero_grad()` clears stored values left by an earlier call. It does not remove the necessary summation inside the current `backward()`. If differentiation starts from a mean loss, its averaging factor is already included, so we should not arbitrarily divide the completed gradient by the batch size again.

## 9. Train on earlier dates and evaluate on later dates

Creating input windows has not yet trained the model. We now divide the 70 windows into sets for training and evaluation.

The **training set** is used to update the weights. The **validation set** helps us decide which weight state to select during training. The **test set** is used to assess performance after that selection is complete.

Because the task is to predict future dates, we split by **the target's date index in the original record**. A **time series** is a record of observations in time order. To assess future prediction performance, we need observations later than the past data used for training.[^time-split]

| Set | Target indices in the original record | Target dates | Input windows |
| --- | --- | --- | ---: |
| Training | 14–55 | Days 15–56 | 42 |
| Validation | 56–69 | Days 57–70 | 14 |
| Test | 70–83 | Days 71–84 | 14 |

<figure class="rnn-guide-figure">
<img src="/rnn-window-to-forecast/split-en.svg" alt="Within the 84-day record, target indices 14 through 55 are used for training, 56 through 69 for validation, and 70 through 83 for testing. Each target's input contains only the 14 actual observations preceding that date." loading="lazy">
<figcaption>The first validation target, at index 56, uses inputs 42–55. Using previously observed dates as inputs is different from using future targets to train the model.</figcaption>
</figure>

```python
target_positions = torch.arange(seq_len, len(visitors))  # 14–83.
train_mask = target_positions < 56
valid_mask = (target_positions >= 56) & (target_positions < 70)
test_mask = target_positions >= 70
# A mask is a True/False array indicating whether to select each sample.

X_train, y_train = X[train_mask], y[train_mask]  # 42 samples.
X_valid, y_valid = X[valid_mask], y[valid_mask]  # 14 samples.
X_test, y_test = X[test_mask], y[test_mask]     # 14 samples.
```

Neighboring windows share past records. The 70 windows are not 70 independently collected time series. However, shared historical inputs between training and validation windows do not automatically constitute **data leakage**. The relevant question here is **whether information unavailable at prediction time has entered the input or training process**.

This evaluation assumes that we observe the actual visitor count each day and then predict the following day. Once the first test day's actual count is observed, it can be used as historical input to predict the next day. This is different from predicting all 14 future days at once today. For that task, we could not insert future actual observations as intermediate inputs.

Dividing by the fixed constant 100 does not estimate preprocessing statistics such as a mean or standard deviation from the whole dataset. If we later switch to preprocessing that uses a mean or standard deviation, those statistics must be calculated from the training period only.

### DataLoader groups samples

`TensorDataset(X_train, y_train)` creates a dataset that pairs each input with the target at the same position. `DataLoader` groups these pairs into **minibatches** and returns them.[^data]

```python
from torch.utils.data import DataLoader, TensorDataset

generator = torch.Generator().manual_seed(7)  # Random configuration for shuffling.
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=16,     # At most 16 input windows per call.
    shuffle=True,     # Shuffle only the order of complete training windows.
    drop_last=False,  # Keep the final group even when fewer than 16 remain.
    generator=generator,
)
```

The 42 training samples form three groups of `16,16,10`. `shuffle=True` does not shuffle the dates inside a window. Each window starts with its own initial state in this model, so we can shuffle the order in which complete windows are processed. We should not apply this setting unchanged to a model that carries state between consecutive data segments.

An **epoch** processes all training samples once. A **training step**, in this example, updates the weights once using one minibatch's loss.

| Loop | What repeats in this example | Count |
| --- | --- | --- |
| Time processing within one sample | Window positions 0–13 | 14 time steps |
| Minibatch processing within one epoch | Groups of 16, 16, and 10 samples | 3 training steps |
| Complete training | Process the same training set again | 300 epochs |

The code therefore performs 900 weight updates in total. A window's 14 time steps do not mean calling `step()` at every time step. If we grouped all 70 windows into batches of 16 without splitting the data, there would be five groups of `16,16,16,16,6`. This training run uses only the 42 windows assigned to the training set.

## 10. Run training and validation, then compare with simple predictions

Before writing the training loop, distinguish two types of mode. `model.train()` and `model.eval()` set the behavior of modules such as Dropout that act differently during training and evaluation. `torch.no_grad()` disables the recording needed for backpropagation inside its block. **Calling `eval()` alone does not disable automatic differentiation.** This small model has no Dropout, but we still write training and evaluation as separate roles.[^grad-modes]

Training uses predictions and a loss from a new forward pass. We do not reuse observational predictions previously computed under `no_grad()` as the training predictions.

The following code uses the previously defined `VisitorRNN`, data, and `train_loader`. A **dtype** describes how numbers are stored, while a **device** specifies where they are computed, such as a CPU or GPU. This run keeps the model, inputs, and targets as CPU `float32` values.

```python
import copy  # Create an independent copy of the best weights.

torch.manual_seed(7)  # Reproduce the training model's weight initialization.
device = torch.device("cpu")  # Device used to measure the results in this post.
model = VisitorRNN(hidden_size=4).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
best_valid_loss = float("inf")  # Any finite loss is initially smaller.
best_state = None
best_epoch = 0

for epoch in range(1, 301):  # 1–300; the ending boundary 301 is excluded.
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xb)  # The current B is 16, 16, or 10.
        assert prediction.shape == yb.shape
        loss = loss_fn(prediction, yb)
        loss.backward()
        optimizer.step()

        # Weight each batch mean by its sample count to prepare the overall mean.
        loss_sum += loss.item() * len(xb)
        sample_count += len(xb)

    train_loss = loss_sum / sample_count
    model.eval()
    with torch.no_grad():  # Compute values without recording for backpropagation.
        valid_prediction = model(X_valid.to(device))
        valid_loss = loss_fn(valid_prediction, y_valid.to(device)).item()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        # state_dict contains named parameters and buffers.
        # An independent copy prevents later training from changing the saved values.

model.load_state_dict(best_state)  # Restore the weights selected by validation.
model.eval()
with torch.no_grad():
    test_prediction = model(X_test.to(device)).cpu()
```

`loss.item()` extracts a Python number for display or logging from a tensor containing one number. Do not replace the training `loss` itself with `loss.item()` and then try to call `backward()`.

Consider why we did not compute the epoch loss as a simple average of the three batch losses. The first two batches have 16 samples each, while the last has 10. Multiplying each batch mean by its sample count, adding the results, and dividing by 42 gives each sample equal weight. This record averages losses observed under different weights at successive training steps. It is distinct from evaluating the entire training set again using the weights at the end of the epoch.

Validation does not update the weights. We select the weights that achieved the lowest validation loss, restore that state, and then run the test evaluation. This run does not use the test results to choose new settings afterward.

### Interpret the results in visitor-count units

The **mean absolute error (MAE)** averages the absolute differences between predictions and targets. Reversing the `/100` scaling lets us read it in visitor-count units. We train with MSE and use MAE below to make the performance comparison easier to interpret.

We also define two **baselines** for comparison: “tomorrow will equal today” and “tomorrow will equal the same weekday last week.” Both use only values observed before the prediction date.

```python
test_positions = target_positions[test_mask]  # Target indices 70–83.
yesterday = visitors[test_positions - 1].unsqueeze(-1)
previous_week = visitors[test_positions - 7].unsqueeze(-1)

def mae_in_visitors(prediction):
    return (prediction - y_test).abs().mean().item() * value_scale

print("RNN", mae_in_visitors(test_prediction))
print("Yesterday", mae_in_visitors(yesterday))
print("Previous week", mae_in_visitors(previous_week))
```

These results were obtained with Python 3.12.14, PyTorch 2.15.0+cpu, and CPU `float32`. The data-generation seed was 1; the model-initialization and training-window-shuffle seeds were 7. Within the maximum of 300 epochs, epoch 289 had the lowest validation loss. The downloadable example reproduces these measurements.

| Method | Test MAE, in visitor-count units |
| --- | ---: |
| Trained RNN | 5.649 |
| Reuse yesterday's visitor count | 19.324 |
| Reuse the count from the same weekday last week | 2.726 |

**In this run, the RNN outperformed yesterday's-value baseline but was less accurate than last week's-value baseline.** We constructed the data by repeating a weekly pattern and adding small amounts of noise, so last week's value is a strong comparison. This interpretation applies to the data-generation conditions in this exercise; it is not a conclusion about performance in an actual service. Training a neural network and obtaining predictions useful for the task are separate things to establish.

We can also supply the final 14 days of the 84-day record to predict day 85, whose target is not yet available. The input shape is `(1,14,1)`, and this run predicts approximately 59.58 visitors. Because we have no actual count for day 85, we cannot calculate that prediction's error.

## 11. Checking an implementation differs from measuring predictive performance

Comparing predictions from individually supplied windows with predictions from the same windows in one batch helps check that batch processing has not mixed the samples. This comparison assumes the same weights and initial states, independent computations between samples, and no differences in random behavior during evaluation.

```python
model.eval()
with torch.no_grad():
    check = X_test[:4]  # Four input windows to compare, (4,14,1).
    one_pred = model(check[:1])  # Prediction for the first window only, (1,1).
    individual = torch.cat(
        [model(row.unsqueeze(0)) for row in check],
        dim=0,  # Join the existing sample axes of (1,1) predictions -> (4,1).
    )
    together = model(check)  # Process all four windows in one call -> (4,1).

assert torch.allclose(individual, together, atol=1e-6, rtol=1e-6)
# atol: absolute tolerance / rtol: tolerance proportional to the reference value.
```

`one_pred` is the result for the first sample alone. `individual` collects the results obtained by processing all the samples individually. The name `individual` does not mean it contains only one sample.

Floating-point operations can produce small differences when their order changes. For ordinary finite values, `allclose` checks whether `abs(a-b) <= atol + rtol*abs(b)`. The tolerances above are settings for this example's check, not fixed values suitable for every model, dtype, and device.[^allclose]

The structural checks in the complete example used all 70 windows. The maximum difference between individual and batched calls was approximately `4.47×10⁻⁸`. The maximum difference between `RNNCell` and `RNN` after copying identical weights was approximately `5.96×10⁻⁸`. Both were within the tolerances used for those checks. These comparisons test whether the implementations perform the same computation. **They do not measure prediction accuracy against actual targets.**

### Distinguish detaching for display from cutting the training connection

A **computation graph** records which operations produced which values and how those operations connect. Automatic differentiation uses these connections. `detach()` returns a tensor containing the same numbers, separated from the previous differentiation connection.[^detach]

```python
print(state.detach())  # Create a detached tensor for display.
# This does not change the original state's differentiation connection
# when that original state is used in later computations.

# state = state.detach()
# Reassigning this way makes later computations use the detached state.
```

Calling `state = state.detach()` at every iteration of the time loop cuts the path through which a later loss would propagate back to earlier steps. Information from earlier inputs does not immediately disappear from the forward computation, but backpropagation through that connection is stopped. We therefore leave this out of a loop intended to learn through the complete 14-day connection.

`detach()` does not copy the data; it shares storage with the original tensor. To preserve an independent copy of the numbers as well, use `detach().clone()`. Using `clone()` alone copies the numbers but can retain their differentiation connection to the original computation.

### Problems to check with long sequences

A **vanishing gradient** becomes very small as it is propagated toward earlier time steps. An **exploding gradient** becomes excessively large. In an RNN, the local derivatives of `tanh` and the repeated recurrent-weight operations both contribute. The derivative of `tanh` alone does not establish that the complete gradient must always shrink.[^rnn-training]

**Gradient clipping** limits gradient magnitude. When needed, apply it after `backward()` and before `step()`. The **norm** below measures the combined magnitude of gradients across the parameters. This code does not simply cut every individual entry to the range −1 to 1.[^clip]

```python
# Use this position in the training loop when large gradients need limiting.
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# max_norm=1.0 is an illustrative setting; choose it for the actual problem.
optimizer.step()
```

Clipping does not restore gradients that have already become small. If the task requires dependencies over long periods, first examine the input length and training behavior. GRUs and LSTMs, whose structures regulate information retention and updates, are also options to investigate. Any architecture still needs evaluation on future periods and comparison with baselines.

After running this far, try answering these five questions.

1. At which position in the original record is the target for `X[i]`?
2. Which axis does `output[:, -1, :]` select, and which values does it retain?
3. What does `state = cell(x_t, state)` change, and what does `optimizer.step()` change?
4. What is counted by 14 time steps, and what is counted by 3 training steps per epoch?
5. Does agreement between batched and individual computations establish that future predictions are accurate?

If you can explain these answers and run the training and evaluation code, you have connected the parts needed to build a next-day prediction model with a basic RNN. Memorizing the complete manual differentiation is not a prerequisite for moving on to the next exercise.

## Reference: expand the calculations and syntax you need

<details>
<summary>A. Selecting values and changing array structure</summary>

A **function** is called as in `torch.stack(...)`; a **method** is called on an object, as in `X.unsqueeze(...)`. Calling the class `nn.RNN(...)` creates a model object. The table below groups expressions from this post by the work they perform.

| Expression | Operation | Example result |
| --- | --- | --- |
| `len(X)` | Read the length of the first axis | 70 for the complete dataset |
| `X.shape` | Read the size of each axis | `torch.Size([70,14,1])` |
| `tuple(X.shape)` | Display the shape as an ordinary tuple | `(70,14,1)` |
| `X.ndim` | Read the number of axes | 3 |
| `X.numel()` | Count all elements | 980 |
| `torch.stack(windows, dim=0)` | Group same-size tensors along a new axis | Seventy `(14,)` tensors → `(70,14)` |
| `torch.cat(predictions, dim=0)` | Join along an existing axis | Seventy `(1,1)` tensors → `(70,1)` |
| `stack(dim=0)` on the same predictions | Group along a new axis | Seventy `(1,1)` tensors → `(70,1,1)` |
| `row.unsqueeze(0)` | Add an axis of size 1 at the beginning | `(14,1)` → `(1,14,1)` |
| `X_2d.unsqueeze(-1)` | Add an axis of size 1 at the end | `(70,14)` → `(70,14,1)` |
| `y.squeeze(-1)` | Remove the final axis if its size is 1 | `(70,1)` → `(70,)` |
| `state.flatten()` | By default, flatten all axes into one | `(2,4)` → `(8,)` |

Calling `squeeze()` without specifying an axis can remove every axis of size 1. If the batch size is 1, this can also remove the batch axis, so specify the intended axis, as in `squeeze(-1)`.

`reshape()` changes the shape while preserving the total number of elements. To swap the sample and time axes, use an axis exchange such as `transpose(0,1)`. Simply changing `(B,T,F)` to `reshape(T,B,F)` does not perform the same meaningful axis exchange. Similarly, flattening `[sample 0's four values, sample 1's four values]` into one sequence removes the original sample boundary from the shape.[^tensor-views]

The comma in shape `(3,)` is Python syntax for a tuple with one item; it does not create another axis. Converting `[10,20,30]` to a tensor gives shape `(3,)`, while converting `[[10],[20],[30]]` gives `(3,1)`. Both contain three numbers, but their array structures differ.

A **list comprehension** is compact syntax for building a list through iteration. The main text's loop can be shortened as follows. If this syntax is unfamiliar, keep using the explicit `for` loop and `append` from the main example.

```python
X = torch.stack([
    visitors[i:i + seq_len]
    for i in range(len(visitors) - seq_len)
]).unsqueeze(-1)
y = visitors[seq_len:].unsqueeze(-1)
```

`X = y = None` assigns a placeholder meaning that no value has been supplied yet to both names. It is unnecessary if we immediately create new values for both `X` and `y`. Annotations such as `values: torch.Tensor`, `seq_len: int`, and `-> tuple[torch.Tensor, torch.Tensor]` are **type hints**, indicating the expected argument and return types. These annotations alone do not automatically check inputs at runtime. The downloadable example uses actual conditional checks to verify that the input is a one-dimensional tensor, the window length is positive, and at least one next-day target remains.[^typing]

Conversions for displaying values also have different roles.

| Expression | What it actually does |
| --- | --- |
| `tensor.item()` | Converts a tensor containing exactly one element into one Python number. |
| `tensor.tolist()` | Converts a tensor to a Python number or nested list, moving it to the CPU if necessary. |
| `state.detach().flatten().tolist()` | Separates from the differentiation connection → flattens into one axis → converts to a Python list. |
| `tensor.detach().cpu().numpy()` | Separates from the differentiation connection → moves to the CPU → converts to a NumPy array. |

`tolist()` is not differentiable, so `detach()` is not required merely to print a list. It can still make the intention to separate data for display explicit. Keep Python values converted for inspection distinct from tensors used in training computations.[^tensor-convert]

</details>

<details>
<summary>B. Synthetic data, numerical scale, and plot coordinates</summary>

`repeat(12)` repeats the whole sequence of seven weekly values 12 times in order. It does not write the first number 12 times, then the second number 12 times. The existing `(7,)` axis grows to `(84,)`; there is still only one axis.

`torch.randn()` draws random numbers from a **standard normal distribution**, which has mean 0 and standard deviation 1. The mean is the sum of values divided by their count; standard deviation measures how spread out values are around the mean. A finite sample of 84 numbers does not necessarily have an observed mean and standard deviation of exactly 0 and 1. Multiplying the random numbers by 3 multiplies the distribution's standard deviation by 3. This 3 sets the magnitude of our synthetic variation; it is not a required constant in an RNN.[^randn]

`torch.manual_seed()` helps reproduce random results in the same environment and execution order. It does not guarantee identical computation results across different versions and devices. That is why the main text specifies the software versions, device, and seeds together.[^reproducibility]

We can also plot the visitor records with Matplotlib. `plt` is an alias for `matplotlib.pyplot`. The following code requires Matplotlib and NumPy to be installed.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 3))  # Figure width and height, measured in inches.
plt.plot(visitors.numpy())  # This tensor is on the CPU without gradient tracking.
plt.xlabel("day index")     # Horizontal-coordinate label.
plt.ylabel("visitors / 100")  # Vertical-coordinate label.
plt.show()
```

When horizontal coordinates are omitted, `plot(y_values)` uses `0,1,...,len(y_values)-1`. Matplotlib has not inferred that those positions represent dates. Even though the plot uses horizontal and vertical coordinates, the original `visitors` remains a one-dimensional tensor of shape `(84,)`. A plot's two coordinates and a tensor's number of axes are different concepts. The default behavior of plotting each column as a separate line when given a two-dimensional array is also separate from the RNN's axis conventions.[^plot]

Under the conditions above, `numpy()` can share memory with the original tensor. Dtype and device also need to be considered separately from shape. `float32` and `float64` differ in floating-point precision; calling `.double()` on a model converts its floating-point parameters and related values to `float64`. Prepare the input with the corresponding dtype and device so the RNN computation can proceed correctly.

</details>

<details>
<summary>C. Connect the tanh derivative to the loss gradient</summary>

This explanation examines **one element for the same sample, time step, and hidden neuron**. `a` is the weighted sum before activation, and `h=tanh(a)` is the hidden value. `L` is the final loss obtained through subsequent computations. `Δa`, `Δh`, and `ΔL` denote small changes in those respective values.

Differentiating `tanh` does not mean differentiating one stored number in isolation. It means finding a rate of change in **the functional relationship through which output `h` depends on input `a`**. At the current point, that rate is `1-h²`. We call it a **local derivative** because it describes this function at this point in the computation.

If we change the weighted sum slightly, the change in the hidden value is approximated by:

<div class="rnn-guide-math">
$$
\Delta h \approx (1-h^2)\Delta a
$$
</div>

This is not an equation for the values themselves, such as `h=(1-h²)×a`. It relates small changes. Because `tanh` is not a straight line, its rate of change varies with position.

Now define `grad_h` as **the gradient of the final loss with respect to the hidden value**. We can connect the preceding change to its effect on the later loss.

<div class="rnn-guide-math">
$$
\Delta L \approx \mathrm{grad}_h\,\Delta h
\approx \mathrm{grad}_h(1-h^2)\Delta a
$$
</div>

The loss gradient with respect to the weighted sum, `grad_a`, is therefore computed as follows.

```python
grad_a = grad_h * (1 - h**2)
# grad_h: gradient of L with respect to h.
# 1-h**2: gradient of h with respect to a.
# grad_a: gradient of L with respect to a.
```

We multiply `1-h²` by `grad_h`, not by `h` itself. Knowing the local derivative of `tanh` does not tell us how sensitive the loss is to that hidden value, so we need `grad_h` as well. This computation does not recover `tanh`'s original input `a`. Evaluating an **inverse function** to recover an input and performing **differentiation** are different operations.

In the manual check, `adj_h` and `adj_a` correspond to `grad_h` and `grad_a` above. `adj` is a variable name chosen by the author, not a Python or PyTorch keyword. The same principle applies to backpropagation in an MLP. RNNs do not use a separate set of differentiation rules.

</details>

<details>
<summary>D. Exactly what gets summed in matrix backpropagation?</summary>

Start with a single multiplication, `z=w×u`. Holding `u` fixed and changing `w` slightly changes `z` by `u×Δw`. Conversely, holding `w` fixed and changing `u` changes `z` by `w×Δu`. **Even within the same multiplication, the value to multiply by depends on which input's gradient we are calculating.**

Now return to the RNN weighted sum at one time step. `grad_a`, with shape `(B,H)`, contains the final loss gradient with respect to each element of that weighted sum. `x_t` is the current input `(B,F)`, and `h_prev` is the previous state `(B,H)`. `G_ih_t` and `G_hh_t` are the gradient contributions from the weights' uses at the current time step. `grad_h_prev` is the loss gradient with respect to the previous state, which will be passed to the preceding step.

```python
G_ih_t = grad_a.T @ x_t          # Contribution from this use of the input weights.
G_hh_t = grad_a.T @ h_prev       # Contribution from this use of the recurrent weights.
grad_h_prev = grad_a @ W_hh      # Loss gradient with respect to the previous hidden state.
```

| Calculation | Shape with the main example's `B=2,F=1,H=4` | What is summed |
| --- | --- | --- |
| `grad_a.T @ x_t` | `(4,2) @ (2,1) → (4,1)` | Samples that used the same input weight |
| `grad_a.T @ h_prev` | `(4,2) @ (2,4) → (4,4)` | Samples that used the same recurrent weight |
| `grad_a @ W_hh` | `(2,4) @ (4,4) → (2,4)` | Current neurons connected to a previous hidden value |

Indices make the correspondence more precise. `b` identifies a sample, `j` a current hidden neuron, `k` an input feature, and `q` a previous hidden value. Each is a zero-based index on its respective axis.

| One result element | Terms added to compute that element |
| --- | --- |
| `G_ih_t[j,k]` | `grad_a[b,j] × x_t[b,k]` over all samples `b` |
| `G_hh_t[j,q]` | `grad_a[b,j] × h_prev[b,q]` over all samples `b` |
| `grad_h_prev[b,q]` | `grad_a[b,j] × W_hh[j,q]` over all current neurons `j` |

The first two results combine effects from samples that used the same weight. The last result combines connections because one previous hidden value feeds multiple current neurons. We then add `G_ih_t` and `G_hh_t` **across time steps** to obtain the shared weights' final gradients. Summation across samples, across neurons, and across time are separate operations.

For a multivariate example with `B=2,F=4,H=3`, the three result shapes become `(3,4)`, `(3,3)`, and `(2,3)`, respectively. Changing the number of input features does not change which quantity each result corresponds to. These arrays are gradients because they combine the differentiation rules for the multiplications, not merely because their shapes happen to match.

This explanation assumes that the loss is calculated from the final time step's prediction. A model with separate losses at every time step also needs the gradient from the direct path between a state and its loss. If `grad_a` starts from a mean loss, the averaging factor is already included; do not divide by the batch size again after these sums.

</details>

<details>
<summary>E. Check three time steps of manual backpropagation against PyTorch</summary>

Only this section uses **a separate verification example** with a single-number hidden state. It does not change the visitor-prediction model in the main text. First, fix its conditions.

| Condition | Setting and role |
| --- | --- |
| Input | One sample containing `[1,2,3]`, selected as values for a three-time-step check |
| Structure | One input feature, one hidden value, no bias |
| Input weight | 0.5, the parameter multiplied by the current input |
| Recurrent weight | 0.5, a separate parameter multiplied by the previous state |
| Initial state | 0 |
| Target | 0.5, the value compared with the final hidden value |
| Loss | `0.5 × (final hidden value − target)²` |
| Dtype and device | CPU `float64` to inspect small numerical differences |

Although the table uses 0.5 repeatedly, the two weights, target, and loss coefficient have different roles. The 0.5 before the loss cancels the factor of 2 introduced when differentiating the square; it is not included in the default `nn.MSELoss()`. This special setup uses the final hidden value directly as the prediction, so there is no separate Linear layer. An output that uses a value unchanged in this way is sometimes called an **identity head**.

The check's `hs` array also stores the initial state. The main text numbers time steps from 0, like the code, but this array reserves its first position for the initial state. Consequently, **the index of a processed state in `hs` is one greater than the corresponding input index**. Some official documentation uses `h_0` for the initial state, so matching names alone do not establish that they refer to the same value.

| Actual value | Main-text notation | Array in this check |
| --- | --- | --- |
| State before the first input | `h_init` | `hs[0]` |
| Result after processing input index 0 | `h_0` | `hs[1]` |
| Result after processing input index 1 | `h_1` | `hs[2]` |
| Result after processing input index 2 | `h_2` | `hs[3]` |

We process three inputs, but the array has four entries because it also includes the initial state. The initial state does not add another input time step. The following small check can be run from the beginning on its own.

```python
import torch
from torch import nn

scalar_rnn = nn.RNN(1, 1, bias=False, batch_first=True).double()
with torch.no_grad():  # Exclude the initialization from backpropagation records.
    scalar_rnn.weight_ih_l0.fill_(0.5)
    scalar_rnn.weight_hh_l0.fill_(0.5)
    # fill_ changes existing tensor values in place; l0 denotes the first RNN layer.

x_scalar = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float64)
target = 0.5
output_scalar, _ = scalar_rnn(x_scalar)  # (1,3,1): hidden values, not a loss.
loss_scalar = 0.5 * (output_scalar[0, -1, 0] - target).square()
scalar_rnn.zero_grad(set_to_none=True)
loss_scalar.backward()
autograd_whh = scalar_rnn.weight_hh_l0.grad.item()

hs = [0.0] + output_scalar.detach().flatten().tolist()
# Only in this special example does flatten give one number per time step.
recurrent_weight = scalar_rnn.weight_hh_l0.detach().item()
adj_h = hs[-1] - target  # Loss gradient with respect to the final hidden value.
pieces = [0.0] * 3      # Storage for the recurrent weight's time-step contributions.

for t in range(3, 0, -1):  # Stored indices 3,2,1; exclude the ending boundary 0.
    adj_a = adj_h * (1 - hs[t] ** 2)  # Loss gradient for the current weighted sum.
    pieces[t - 1] = adj_a * hs[t - 1] # Weight branch: multiply by the previous state.
    adj_h = adj_a * recurrent_weight # Previous-state branch: multiply by the weight.

print(pieces)
print(sum(pieces), autograd_whh)
assert abs(sum(pieces) - autograd_whh) < 1e-12
```

Although `t` proceeds through 3, 2, and 1, storing into `pieces[t-1]` leaves the completed array in first-, second-, and third-time-step order. The weight branch and previous-state branch are both computed from the same `adj_a`. The previous-state result becomes `adj_h` in the next iteration, continuing through the earlier `tanh` that produced that state.

| Use of the recurrent weight | Value added to its gradient |
| --- | ---: |
| First time step | 0 |
| Second time step | 0.00251786232041447 |
| Third time step | 0.03172237391003767 |
| Manual sum of the three terms | 0.03424023623045214 |
| PyTorch automatic differentiation | 0.03424023623045215 |

The first contribution is zero because the initial state multiplied by the recurrent weight is zero. **This does not remove the entire first time step from learning.** The gradient contribution for the input weight multiplied by the first input can still be nonzero. To check a use of the input weight, multiply that time step's `adj_a` by the current input instead of the previous state.

The gradient for the initial state calculated at the end is not used further in this example, because the initial state is fixed at zero rather than being a learned parameter. Calculating `pieces` as Python numbers also does not automatically change the model's `.grad`. We are only comparing it with the result already obtained by `backward()`. There is no `step()` here to train the weights.

A **finite difference** approximates a rate of change by changing a weight slightly and recalculating the actual loss. A central finite difference evaluates the loss on both sides of the current recurrent weight `w`, at `w+ε` and `w−ε`. Here, `ε` is a small interval of change.

<div class="rnn-guide-math">
$$
\frac{L(w+\varepsilon)-L(w-\varepsilon)}{2\varepsilon}
$$
</div>

The downloadable check evaluates a separate loss function with each candidate recurrent weight, `w+ε` and `w−ε`, without modifying the original model. The input weight remains fixed. With `ε=10⁻⁶`, the result was `0.03424023624354344`. It is close to the automatic and manual derivatives, but a finite difference remains an approximation. An excessively small interval can also be affected by floating-point precision. A tolerance such as `1e-12` is a choice for this small CPU `float64` check, not a value to apply unchanged to general GPU `float32` computations.[^gradcheck]

</details>

<details>
<summary>F. Other RNN settings and debugging checks</summary>

In the main example, `B` and `T` describe the amounts of data supplied in a call. `F` and `H` determine the weight structure. We can pass an input with `T=28` instead of `T=14` to the same RNN object, but we cannot directly supply four input features to a model created with `F=1`.

| Setting changed | What it means for the same basic model |
| --- | --- |
| `B`, batch size | The number of samples processed in the call and represented in the initial state changes. |
| `T`, number of time steps | The number of processed steps and the size of `output`'s time axis change. |
| `F`, number of input features | The input-weight structure must change. |
| `H`, number of hidden values | The RNN weights and the input size of the connected Linear layer must change. |

`id(model)` identifies a Python object. Even if the object remains the same, `step()` can change the weight values inside it. To check whether the values are unchanged as well, compare copies of the parameters from before and after the call. Conversely, running a forward pass under `no_grad()` with a different time length does not train the weights. Even if all inputs are zero, biases can make the output nonzero.

With multiple layers or directions, we need to interpret the return values again. `K` is the number of RNN layers, and `D` is the number of processing directions: 1 for unidirectional and 2 for bidirectional. Each layer and direction uses separate parameters.

| Setting | Return shapes and final states for batched input |
| --- | --- |
| General `batch_first=True` | `output=(B,T,D×H)`, `h_n=(K×D,B,H)` |
| Multiple layers, one direction | `output[:, -1, :]` corresponds to `h_n[-1]` from the final layer. |
| Bidirectional | Inspect the beginning-to-end and end-to-beginning states separately. Do not directly generalize the main text's `h_n[0]` selection. |
| Padded inputs with variable lengths | Each sample may have a different actual final position, so `-1` may not identify its last real observation. |
| RNN input without a batch axis | Input `(T,F)`, `output=(T,D×H)`, and `h_n=(K×D,H)`; `batch_first` does not apply. |
| RNNCell input without a batch axis | Use input `(F,)`, previous state `(H,)`, and returned state `(H,)`. |

Support for unbatched input in `nn.RNN` itself does not mean our `VisitorRNN.forward` supports it. The main example's `forward` uses three-axis indexing, `output[:, -1, :]`, so we retain the batch axis in its input. Read a typical MLP's `(B,F)` and Conv2d's `(B,C,height,width)` the same way, distinguishing the batch axis from the structure within each sample. Here, `C` is the number of input image channels.

Modes and gradient recording have the following distinct roles.

| Code | What it changes |
| --- | --- |
| `model.train()` | Sets modules with different training and evaluation behavior to training mode. |
| `model.eval()` | Sets those modules to evaluation mode. |
| `with torch.no_grad():` | Does not create backpropagation records for ordinary tensor operations inside the block. Numerical computation still runs. |
| `with torch.inference_mode():` | Further reduces automatic-differentiation bookkeeping for pure inference. Tensors created here have restrictions on later use in training. |

The effect of a `with` block ends with its indentation. A `no_grad()` block used earlier to set initial weights does not also cover a separate training loop below it. For evaluation, use `eval()` together with `no_grad()`, or use `inference_mode()` for pure inference that will not connect to later automatic-differentiation computations.

Needing weight gradients does not mean the input data must also have `requires_grad=True`. Ordinary module parameters are registered for learning. In usual training, where input gradients are not needed, we can obtain parameter gradients without enabling differentiation tracking for the inputs.

`model.zero_grad()` clears gradients for the model's parameters, while `optimizer.zero_grad()` clears gradients for the parameters managed by that optimizer. Their targets coincide when all parameters have been registered, but can differ when the optimizer manages only some of them. `step()` generally does not clear `.grad` automatically. A new computation graph is created by a new forward pass with gradient tracking enabled, not by `zero_grad()`.

A **leaf tensor** generally refers to a starting tensor rather than an intermediate value produced by another operation tracked by automatic differentiation. A module's learned parameters retain their `.grad` by default, while computed intermediates such as hidden states may not retain their own `.grad`. **A `None` value in an intermediate tensor's `.grad` alone does not establish that backpropagation did not traverse that connection.** To inspect the gradient of an intermediate tensor, call `retain_grad()` beforehand on the tensor that is actually connected to the loss computation.[^retain-grad]

```python
debug_model = VisitorRNN(hidden_size=4)
debug_output, _ = debug_model.rnn(X[:2])
debug_output.retain_grad()  # Retain the gradient of this intermediate tensor too.
debug_pred = debug_model.head(debug_output[:, -1, :])
debug_loss = nn.MSELoss()(debug_pred, y[:2])
debug_loss.backward()
print(tuple(debug_output.grad.shape))  # (2,14,4).
```

In this check, `debug_output.grad` describes the gradient through the path where the returned `output` array is subsequently used by the prediction head. We should not interpret it directly as an array of the total gradients for all hidden states passed between time steps inside the RNN. This example uses only the returned array's final time step in the prediction head. To investigate internal gradients at individual time steps, inspect intermediate states in a computation you explicitly construct with `RNNCell`.

A loss does not have to have shape `()` to support `backward()` without an argument. A real-valued tensor that requires gradients can also have shape `(1,)` or `(1,1)` if it contains exactly one element. Calling `backward()` directly on a prediction tensor containing multiple elements, however, differs from the main example's approach of first computing a single-number loss.[^backward]

</details>

<details>
<summary>G. Express the same input windows and manual backpropagation in Rust</summary>

The relationships between input windows, time steps, and weights are the same in Python and Rust. The difference here is that the Rust code below uses only standard arrays and floating-point numbers. These arrays do not have PyTorch's automatic-differentiation records, `.grad`, or `detach()`, so we calculate the gradients explicitly. Rust can also use separate libraries that support automatic differentiation; here, we are comparing how the code is expressed in the two languages.

`Vec` is an array whose length can grow, and `push` adds an item to its end. `f64` is a 64-bit floating-point type. `&` denotes a reference for borrowing a value to read it. Python's `range(start,end)` and Rust's `start..end` exclude the end. Rust's `1..=3` includes both endpoints, and `.rev()` reverses the order.[^rust]

The following code performs the same scalar check as reference E. At the end, it adds one simple SGD update to distinguish calculating a gradient from actually changing a weight.

```rust
fn main() {
    // First, pair input windows and next-day targets from the same seven-day record.
    let data = [10, 20, 30, 40, 50, 60, 70];
    let seq_len = 3;
    let mut pairs = Vec::new();
    for i in 0..data.len() - seq_len {
        let window = data[i..i + seq_len].to_vec(); // Exclude the end and copy.
        let target = data[i + seq_len];
        pairs.push((window, target));
    }
    println!("{pairs:?}"); // Four input windows and four targets.

    // Separate scalar check: repeat the same rule for each input and previous state.
    let inputs = [1.0_f64, 2.0, 3.0];
    let input_weight = 0.5_f64;
    let recurrent_weight = 0.5_f64;
    let target = 0.5_f64;
    let mut hs = [0.0_f64; 4]; // Initial state plus the states from three time steps.

    for (index, &value) in inputs.iter().enumerate() {
        hs[index + 1] = (
            input_weight * value + recurrent_weight * hs[index]
        ).tanh();
    }

    let mut adj_h = hs[3] - target; // Gradient of 0.5*(hs[3]-target)^2.
    let mut pieces = [0.0_f64; 3];
    for t in (1..=3).rev() { // Order: 3,2,1.
        let adj_a = adj_h * (1.0 - hs[t].powi(2));
        pieces[t - 1] = adj_a * hs[t - 1]; // Contribution from this recurrent-weight use.
        adj_h = adj_a * recurrent_weight; // Pass the gradient to the previous state.
    }
    let gradient: f64 = pieces.iter().sum();
    println!("gradient={gradient}");

    // Actually update a number here, separately from the gradient calculation above.
    let learning_rate = 0.01_f64;
    let mut updated_weight = recurrent_weight;
    updated_weight -= learning_rate * gradient; // One simple SGD step.
    println!("updated_weight={updated_weight}");
}
```

This code computes a recurrent-weight gradient of `0.034240236230452...`. The downloadable Rust file also includes a multivariate forward pass with four input features and three hidden values. In that computation, every current neuron reads the same previous state, and the state is replaced only after all new hidden values have been calculated. Overwriting the previous state after computing each neuron would let the next neuron read already updated values, producing a different computation from the original RNN.

</details>

## Executable examples and official references

The [Python example](/rnn-window-to-forecast/examples-en.py) contains the complete training exercise, and the [BPTT check](/rnn-window-to-forecast/bptt-en.py) contains manual differentiation under its special conditions. Run these files in a Python environment with PyTorch installed. Matplotlib and NumPy are additionally required only for the plotting reference code. The [Rust example](/rnn-window-to-forecast/recurrence-en.rs) uses the standard library without external crates.

The experimental numbers in the main text were calculated from synthetic data using the published example code. The documents below were consulted to verify API behavior and training and evaluation principles.

[^python-sequence]: [Official Python documentation — sequence operations and range](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations).
[^stack]: [PyTorch — torch.stack](https://docs.pytorch.org/docs/2.15/generated/torch.stack.html).
[^unsqueeze]: [PyTorch — torch.unsqueeze](https://docs.pytorch.org/docs/2.15/generated/torch.unsqueeze.html).
[^unbind]: [PyTorch — torch.unbind](https://docs.pytorch.org/docs/2.15/generated/torch.unbind.html).
[^rnncell]: [PyTorch — RNNCell](https://docs.pytorch.org/docs/2.15/generated/torch.nn.RNNCell.html).
[^rnn]: [PyTorch — RNN](https://docs.pytorch.org/docs/2.15/generated/torch.nn.RNN.html).
[^linear]: [PyTorch — Linear](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html).
[^tanh]: [PyTorch — Tanh](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Tanh.html).
[^mse]: [PyTorch — MSELoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MSELoss.html).
[^broadcast]: [PyTorch — Broadcasting semantics](https://docs.pytorch.org/docs/2.15/notes/broadcasting.html).
[^optimization]: [Official PyTorch training tutorial — Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).
[^zero-grad]: [PyTorch — Optimizer.zero_grad](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.zero_grad.html).
[^sgd]: [PyTorch — SGD](https://docs.pytorch.org/docs/2.15/generated/torch.optim.SGD.html).
[^adam]: [PyTorch — Adam](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html).
[^autograd]: [PyTorch — Automatic Differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).
[^data]: [PyTorch — torch.utils.data, TensorDataset, and DataLoader](https://docs.pytorch.org/docs/2.15/data.html).
[^time-split]: [scikit-learn — time series cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split).
[^grad-modes]: [PyTorch — Autograd mechanics: grad mode, no-grad mode, inference mode, and evaluation mode](https://docs.pytorch.org/docs/2.15/notes/autograd.html).
[^allclose]: [PyTorch — torch.allclose](https://docs.pytorch.org/docs/2.15/generated/torch.allclose.html).
[^detach]: [PyTorch — Tensor.detach](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.detach.html).
[^rnn-training]: [Pascanu, Mikolov, Bengio — On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063).
[^clip]: [PyTorch — clip_grad_norm_](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html).
[^tensor-views]: [PyTorch — Tensor Views](https://docs.pytorch.org/docs/2.15/tensor_view.html).
[^typing]: [Official Python documentation — typing](https://docs.python.org/3/library/typing.html).
[^tensor-convert]: [PyTorch — Tensor.tolist](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.tolist.html), [Tensor.numpy](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.numpy.html).
[^randn]: [PyTorch — torch.randn](https://docs.pytorch.org/docs/2.15/generated/torch.randn.html).
[^reproducibility]: [PyTorch — Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html).
[^plot]: [Matplotlib — pyplot.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html).
[^gradcheck]: [PyTorch — gradcheck: compare automatic differentiation with finite differences](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.gradcheck.gradcheck.html).
[^retain-grad]: [PyTorch — Tensor.retain_grad](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.retain_grad.html).
[^backward]: [PyTorch — Tensor.backward](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.backward.html).
[^rust]: [Rust standard library — Iterator](https://doc.rust-lang.org/std/iter/trait.Iterator.html), [Vec](https://doc.rust-lang.org/std/vec/struct.Vec.html).
