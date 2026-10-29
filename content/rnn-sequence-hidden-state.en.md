+++
title = "RNN Sequences and Hidden States: Cells, Neurons, Weights, and PyTorch"
description = "Calculate an RNN with two sequences, five time steps, four input features, and three hidden neurons. Use diagrams and executable code to examine sequence structure, connections inside a cell, the two weight matrices, biases, and the output shapes of RNNCell and RNN."
date = 2026-10-29
slug = "rnn-sequence-hidden-state"

[extra]
katex = true
toc = true
styles = ["rnn-sequence-hidden-state/article.css"]
+++

RNN (recurrent neural network) diagrams often contain several similar boxes, each containing several neurons. To follow the calculation, we need to distinguish the number of boxes, the number of neurons, and the number of input values.

Start with the basic operation. <strong>A basic RNN computes a new hidden state from the current input and the previous hidden state.</strong> A hidden state is a group of numbers produced by processing earlier inputs. With three hidden neurons, the RNN produces three numbers for one sequence at one time step, then passes all three to the next step.

This post uses four input features, three hidden neurons, and five time steps. We will process two independent sequences together. First, we will examine the structure through diagrams and numbers. Then we will run the same calculation in PyTorch. The example is <strong>the forward computation of a basic RNN with one layer, one direction, and tanh activation.</strong> Unidirectional means processing from beginning to end in one direction. A forward computation calculates results using the current weights. The additional structures of LSTM and GRU, and the training process, are outside this example.

You can download the [complete Python code with comments](/rnn-sequence-hidden-state/examples-en.py) and the [Rust code for checking the numerical calculation](/rnn-sequence-hidden-state/recurrence-en.rs). The Python code runs on the CPU in an environment with PyTorch installed.

## 1. The current input and the previous result enter together

Before reading the diagram, define four terms.

| Name | Meaning | In this example |
| --- | --- | --- |
| Feature | An individual piece of information in an input | Four numbers at one time step |
| Time step | The position currently being processed in ordered data | First through fifth |
| Sequence | Inputs from several time steps grouped in order | All five inputs together |
| Hidden state | The group of results from one RNN time step | Three numbers produced by three neurons |

If temperature, humidity, wind speed, and air pressure are measured every hour, each time step has four features. Grouping five measurements in time order produces one sequence. The numbers below are for practicing the calculation; they are not actual weather observations. We label the features `f0, f1, f2, f3`.

Write the first input as `x₁` and the hidden state produced from it as `h₁`. The first calculation has no previous result, so we provide an initial state `h₀`. Here, we start with `[0, 0, 0]`.

<figure class="rnn-figure">
<img src="/rnn-sequence-hidden-state/time-flow-en.svg" alt="One RNN layer processes five time steps. Each step receives its four input features and the three hidden values from the previous step, then produces three new hidden values. All five calculations use the same weights and biases." loading="lazy">
<figcaption>All three numbers in h₁ pass together to the next calculation. x₂ is the new input at the second time step, while h₁ is the result from the first step.</figcaption>
</figure>

The complete flow is <strong>prepare the inputs → select the first time step → compute from the current input and initial state → store the three results as the new state → repeat with the next input and the new state.</strong>[^rnn]

Completing the second hidden state `h₂` requires `h₁`. However, operations that do not use the previous state, such as multiplying the input by its weights, can be performed ahead of time. The part that depends on time order is <strong>completing the new hidden state.</strong>

## 2. Find one sequence inside the input x

A <strong>batch</strong> groups several samples for processing together. Here, one batch contains two sequences. The sequences share the same weights, but each has its own hidden state.

| Symbol | What it counts | Value |
| --- | --- | ---: |
| `B` | Sequences in the batch | 2 |
| `T` | Time steps in one sequence | 5 |
| `F` | Input features at one time step | 4 |
| `H` | Hidden neurons, or hidden values produced per time step | 3 |
| `L` | Stacked RNN layers | 1 |

We call the hidden neurons A, B, and C. The symbol `B=2` denotes the batch size; “neuron B” names one of the three neurons.

<figure class="rnn-figure">
<img src="/rnn-sequence-hidden-state/sequence-data-en.svg" alt="The complete input x contains two sequences. Each sequence has five time steps, and each time step has four features. The second time step of the first sequence is [3,8,12,6], and the same step of the second sequence is [9,5,10,7]." loading="lazy">
<figcaption>One sequence contains 5×4=20 numbers. Keep the five time steps separate from the four features at each step.</figcaption>
</figure>

In this post, `x` is <strong>a PyTorch tensor created with `torch.tensor(...)`.</strong> A tensor stores and computes numbers and arrays of numbers. Treating the nested arrays in the diagram as an ordinary Python list and applying `x[0, 1, :]` would not work in the same way.

A sequence does not have to represent clock time. The word order in a sentence can also form a sequence. An RNN processes inputs in the order it receives them. It does not infer actual dates or time intervals from indices, or automatically sort shuffled observations by time. If a task needs actual time intervals, provide that information through input features or another explicit representation.

## 3. Read axis numbers, sizes, and indices separately

A <strong>shape</strong> lists the size of each axis in order. An <strong>axis</strong> identifies a dimension along which positions are distinguished, and an <strong>index</strong> selects a position on that axis. Python and PyTorch indices start at 0.

The complete input has shape `(B, T, F) = (2, 5, 4)`.

| Axis number | Meaning | Size | Valid indices |
| ---: | --- | ---: | --- |
| 0 | Select a sequence | 2 | 0, 1 |
| 1 | Select a time step | 5 | 0, 1, 2, 3, 4 |
| 2 | Select a feature | 4 | 0, 1, 2, 3 |

“Axis 1” is the number of the second axis. Its size is 5, and its last index is 4. These three numbers describe different things.

| Selection | Selected data | Result shape |
| --- | --- | --- |
| `x[0]` | All five time steps of the first sequence | `(5, 4)` |
| `x[0, 1, :]` | The second time step of the first sequence: `[3, 8, 12, 6]` | `(4,)` |
| `x[0, 1, 2]` | The third feature of that input: `12` | `()` |
| `x[:, 1, :]` | The second time step of both sequences: `[[3,8,12,6], [9,5,10,7]]` | `(2, 4)` |
| `x[:, 1:2, :]` | The same step, retaining a time axis of size 1 | `(2, 1, 4)` |

`:` selects the entire axis. Selecting with the integer index `1` removes that axis from the result. Selecting with the slice `1:2` retains an axis of size 1. The slice `1:2` includes the starting index 1 and excludes the ending index 2.

One number is a <strong>scalar</strong>. Numbers arranged along one axis form a <strong>vector</strong>, and numbers arranged along two axes form a <strong>matrix</strong>. The vector `[3, 8, 12, 6]` contains four numbers but has one axis, so its shape is `(4,)`. A scalar tensor holding one number has shape `()`.[^tensor]

## 4. What layers, cells, and neurons compute

A <strong>weight</strong> is a trainable number multiplied by an input value. A <strong>bias</strong> is a trainable number added to the sum of the products. Together, weights and biases are the model's <strong>parameters</strong>.

A <strong>neuron</strong> is a computational unit that multiplies its connected values by weights, adds all the products, adds the biases, and applies an activation function to produce a result. An activation function transforms that sum into the next value. We use `tanh` here.

| Unit | Scope of the calculation | In this example |
| --- | --- | --- |
| One neuron | Computes its own result from the current input and previous hidden values | One number for one sequence at one time step |
| One cell calculation | Computes the complete hidden state at that time step | Three numbers containing the results of A, B, and C |
| One RNN layer | Processes a sequence using the same calculation rule and parameters | Applies the same cell calculation at five time steps |

<figure class="rnn-figure">
<img src="/rnn-sequence-hidden-state/cell-connections-en.svg" alt="The four current input features and three previous hidden values enter the calculations of current neurons A, B, and C. Each neuron uses its own weights and biases. Their results form [A_t,B_t,C_t]. There is no connection that feeds current A's result into current B." loading="lazy">
<figcaption>Neurons A, B, and C all read the same complete current input and the same complete previous hidden state. Each neuron has separately stored weights.</figcaption>
</figure>

Call the values produced by A, B, and C for one sequence at one time step `Aₜ`, `Bₜ`, and `Cₜ`. The cell's result is `hₜ = [Aₜ, Bₜ, Cₜ]`. <strong>The cell's output contains the results of all three neurons.</strong> It does not return only the value of the last neuron, C.

Inside one neuron, several multiplication results are added together. When forming the cell's output, the three neurons' results remain separate and are collected into a vector. These are different operations.

When both sequences are processed together, the state has shape `(B, H) = (2, 3)`. Axis 0 selects sequence indices `0–1`, and axis 1 selects neuron indices `0–2`. Each sequence has three values, giving six numbers in total. A tensor containing hidden states does not always have only one axis.

PyTorch performs these calculations using weight matrices and tensor operations. The neurons drawn in a diagram do not need to correspond to separate Python objects.

## 5. The two groups of weights multiply different values

A basic RNN has weights for the current input and weights for the previous hidden state. PyTorch names them `weight_ih` and `weight_hh`, respectively. `i` stands for input, and `h` stands for hidden.[^cell]

| Name | Connection | Shape | Size here |
| --- | --- | --- | --- |
| `weight_ih` | Current input → current hidden value | `(H, F)` | `(3, 4)` |
| `weight_hh` | Previous hidden value → current hidden value | `(H, H)` | `(3, 3)` |

In both matrices, axis 0 selects <strong>the neuron whose current result is being computed.</strong> Axis 1 of `weight_ih` selects an input feature, while axis 1 of `weight_hh` selects a previous hidden value.

<figure class="rnn-figure">
<img src="/rnn-sequence-hidden-state/weight-layout-en.svg" alt="weight_ih is a 3×4 matrix storing four input weights for each current neuron A, B, and C. weight_hh is a 3×3 matrix storing each current neuron's weights for previous A, B, and C. Multiplying the batch input (2,4) by the transposed input weight matrix (4,3) produces (2,3)." loading="lazy">
<figcaption>The first axis of PyTorch's stored weights selects the current neuron. The weight matrix is transposed when multiplying a batch-first input by it.</figcaption>
</figure>

Neuron A's input weights are `[0.11, -0.07, 0.04, 0.02]`. A uses all four input features, and B and C use the same four input features. Each neuron performs the calculation with its own weights.

Neuron A's weights for the previous state are `[0.21, -0.14, 0.07]`. Computing current A therefore uses <strong>all three previous values: A, B, and C.</strong> B and C also read the complete previous state. A structure in which each output calculation connects to every value in its input group is called fully connected.

A previous hidden value is the result of an earlier calculation. A recurrent weight is a parameter multiplied by that result. The calculation uses `previous hidden value × recurrent weight`, and these two kinds of numbers are stored separately.

### Why append .T to the weights?

`@` performs matrix multiplication. It multiplies corresponding values from a row of the left matrix and a column of the right matrix, then adds the products. Here, `.T` transposes a two-dimensional matrix by exchanging its rows and columns.

| Calculation | Left shape | Right shape | Result shape |
| --- | --- | --- | --- |
| `xt @ weight_ih.T` | `(2, 4)` | `(4, 3)` | `(2, 3)` |
| `previous_state @ weight_hh.T` | `(2, 3)` | `(3, 3)` | `(2, 3)` |

The first calculation matches each sequence's four input values with the four weights belonging to each current neuron. The second matches each sequence's three previous values with each current neuron's three recurrent weights. Because `weight_hh` is square, transposing it preserves its shape but changes the positions of its values. Omitting `.T` may not trigger a shape error, so the direction of the connections still needs checking.

## 6. Each neuron has 7 weights; the whole cell has 27 scalar parameters

One hidden neuron uses `F` input features and `H` previous hidden values. It therefore has `F+H` weights. Here, that is `4+3=7`.

There are three neurons, so the total weight count is `H×(F+H) = 3×7 = 21`. Biases are counted separately.

PyTorch's `nn.RNNCell` stores two bias vectors when `bias=True`. `bias_ih` belongs to the input calculation, and `bias_hh` belongs to the previous-state calculation. Each has shape `(H,)`, giving two bias values per neuron.[^cell]

| Neuron | `bias_ih` | `bias_hh` | Sum added in the forward computation |
| --- | ---: | ---: | ---: |
| A | 0.015 | -0.008 | 0.007 |
| B | -0.025 | 0.018 | -0.007 |
| C | 0.035 | -0.028 | 0.007 |

Having two input paths does not mathematically require two biases. In the calculation, `bias_ih + bias_hh` can be represented as one combined bias. When counting the trainable numbers actually stored by PyTorch, however, both vectors must be included.

| Parameter tensor | Shape | Number of values |
| --- | --- | ---: |
| `weight_ih` | `(3, 4)` | 12 |
| `weight_hh` | `(3, 3)` | 9 |
| `bias_ih` | `(3,)` | 3 |
| `bias_hh` | `(3,)` | 3 |
| Total | 4 tensor objects | 27 |

The cell has 21 weights and 6 biases, giving 27 trainable numbers. “Four parameter tensor objects” and “27 trainable numbers” count different units. With `bias=False`, both bias vectors are absent, leaving two tensor objects and 21 numbers.

Do not multiply the parameter count by 2 because there are two sequences, or by 5 because there are five time steps. Multiple inputs reuse the same parameters. This formula applies to the single basic RNN layer examined here.

## 7. Calculate the three hidden values at the first time step

`tanh` is an activation function that transforms the sum nonlinearly. For example, `tanh(0.377)` is approximately `0.360099`. Applied to a vector, it transforms each element separately. Mathematically, its result for a finite input lies strictly between −1 and 1; computer rounding can produce a displayed value of ±1.

The first input of the first sequence is `x₁ = [2, 5, 7, 11]`, and the initial state is `h₀ = [0, 0, 0]`. A zero initial state does not mean zero weights. We use the weights specified earlier. PyTorch's basic RNNCell and RNN start from zero when the initial state is omitted.[^cell][^rnn]

Separate neuron A's calculation into three parts.

| Part | Calculation | Result |
| --- | --- | ---: |
| Current input | `2×0.11 + 5×(-0.07) + 7×0.04 + 11×0.02` | 0.370 |
| Previous state | `0×0.21 + 0×(-0.14) + 0×0.07` | 0 |
| Bias | `0.015 + (-0.008)` | 0.007 |

Adding these three results gives `0.377`. Applying `tanh` produces A's new hidden value, approximately `0.360099`. B and C use the same input and initial state, but their own weights.

| Neuron | Current-input part | Previous-state part | Bias sum | tanh result |
| --- | ---: | ---: | ---: | ---: |
| A | 0.370 | 0 | 0.007 | 0.360099 |
| B | 0.560 | 0 | -0.007 | 0.502765 |
| C | 1.310 | 0 | 0.007 | 0.866036 |

Collecting the three results gives `h₁ ≈ [0.360099, 0.502765, 0.866036]`. The previous-state term exists even at the first time step. Its products are zero here because the initial state is zero.

## 8. The second time step uses all three previous values

The second input is `x₂ = [3, 8, 12, 6]`. This calculation receives the newly computed `h₁` as its previous hidden state. Neuron A reads four input values and three previous hidden values, for seven values in total.

| Part | Calculation | Result |
| --- | --- | ---: |
| Current input | `3×0.11 + 8×(-0.07) + 12×0.04 + 6×0.02` | 0.370 |
| Previous state | `A₁×0.21 + B₁×(-0.14) + C₁×0.07` | Approximately 0.065856 |
| Bias | `0.015 + (-0.008)` | 0.007 |

A's new result is therefore `tanh(0.370 + 0.065856… + 0.007) ≈ 0.416009`. The current-input sum happens to equal the one at the first time step, but the previous-state part changes, so the final result changes too.

| Neuron | Current-input part | Previous-state part | Bias sum | New hidden value |
| --- | ---: | ---: | ---: | ---: |
| A | 0.370 | 0.065856 | 0.007 | 0.416009 |
| B | 1.310 | 0.172000 | -0.007 | 0.900527 |
| C | 0.250 | 0.306724 | 0.007 | 0.510735 |

The new state is `h₂ ≈ [0.416009, 0.900527, 0.510735]`. The table rounds to six decimal places; subsequent calculations use the unrounded values.

For the complete batch, the calculation can be written as follows. `xₜ` is the `(B,F)` input selected at the current time step, and `hₜ₋₁` is the `(B,H)` previous state. `Wᵀ` denotes a transposed weight matrix. Each bias is added at the same neuron position in every sequence.

<div class="rnn-math">
$$
h_t=\tanh\left(x_tW_{ih}^{\mathsf T}+b_{ih}+h_{t-1}W_{hh}^{\mathsf T}+b_{hh}\right)
$$
</div>

The original first input `x₁` does not return as the current input of the second time step. Its computed result `h₁` enters the next calculation. This allows earlier inputs to influence later results. It does not mean that the hidden state stores every original input or preserves all past information without loss.

## 9. Separate time-step order from neuron calculation order

Within the same sequence, the states depend on one another in the order `h₁ → h₂ → h₃`. Within one time step, however, neurons A, B, and C all read <strong>the state from the previous time step.</strong>

| Result to compute | Previous hidden values used | Current neuron results used |
| --- | --- | --- |
| `A₂` | `A₁, B₁, C₁` | None |
| `B₂` | `A₁, B₁, C₁` | None |
| `C₂` | `A₁, B₁, C₁` | None |

Replacing `A₁` with the newly computed `A₂` while calculating `B₂` would change the operation. Even in a manual loop, <strong>keep the previous state unchanged while computing new A, B, and C, then replace the entire state with the three completed results.</strong>

The neuron calculations within a time step and the calculations for different sequences can be grouped into matrix operations. Actual parallel execution depends on the device and implementation. Writing code that computes A, B, and C in that order does not mean B requires current A's result.

### Do five cell boxes in a diagram mean five objects?

When one `nn.RNNCell` object receives the complete batch at each time step, distinguish the following counts.

| What is counted | Count | Reason |
| --- | ---: | --- |
| RNNCell objects | 1 | Created once and reused |
| Calculation positions in the time diagram | 5 | The same operation is applied at five time steps |
| Python calls to `cell(...)` | 5 | Each call handles one time step |
| Individual sequence–time-step combinations | 10 | `B×T = 2×5` |
| Output numbers across all time steps | 30 | `B×T×H = 2×5×3` |
| Shared trainable scalar parameters | 27 | 21 weights and 6 biases |

The 30 output numbers count results produced by the calculation. They do not mean that the model creates 30 new neuron objects.

More time steps increase the number of calculations and the size of an output that stores every step. The parameter count stays the same. Memory use also depends on whether all results and the automatic differentiation history are retained. A forward pass that keeps only the final state and a training pass that retains the complete history have different memory requirements.

## 10. RNNCell computes one time step

`torch` is the PyTorch package, and `torch.nn` is the module that provides neural network components. `from torch import nn` imports `torch.nn` under the name `nn`.

`nn.RNNCell` is a class used to create cell objects. `cell = nn.RNNCell(input_size=4, hidden_size=3)` creates an object with parameters. Calling `cell(xt, state)` then supplies input to that object and performs a calculation.[^cell]

| Argument | What it determines | Value here |
| --- | --- | --- |
| `input_size` | Input features `F` at one time step | 4 |
| `hidden_size` | Hidden neurons `H` | 3 |
| `bias` | Whether biases are used | `True` |
| `nonlinearity` | Activation function applied to the sum | `"tanh"` |

Do not put the batch size 2 or the time-step count 5 into `input_size`. Those counts are determined by the structure of the data supplied to the model.

| One call | `xt` | Previous `state` | Return value |
| --- | --- | --- | --- |
| Two sequences in a batch | `(2, 4)` | `(2, 3)` | A `(2, 3)` tensor |
| One sequence without a batch axis | `(4,)` | `(3,)` | A `(3,)` tensor |

A single RNNCell call has no time axis. The developer selects the current time step with `x[:, t, :]` and calls the cell repeatedly. At each step, a `(2,4)` input enters and a `(2,3)` state is returned.

### In what order does state = cell(xt, state) run?

An assignment finishes evaluating the right-hand side before binding the result to the name on the left.[^python]

1. Read the previous tensor referenced by `state` on the right.
2. Compute a new state tensor from `xt` and that previous state.
3. Bind the name `state` on the left to the returned tensor.

Having the same variable name on both sides does not create infinite recursion. For clarity, we can give the previous state another name with `previous_state = state`, then write `state = cell(xt, previous_state)`. The first assignment does not itself copy the tensor. This example does not overwrite the previous tensor's values.

A name such as `cell_scalar` has no special meaning in PyTorch. It is a variable name chosen by the author. Check the object's construction arguments and the tensor shapes to determine the actual input and output sizes.

## 11. reshape changes the axes without adding numbers

Selecting `x[0, 1, :]`, the second time step of the first sequence, gives `[3, 8, 12, 6]` with shape `(4,)`. To supply it as an input with batch size 1, `.reshape(1, 4)` produces `[[3, 8, 12, 6]]`.[^reshape]

| Form | Actual values | Shape | Axis meanings |
| --- | --- | --- | --- |
| Batch axis omitted | `[3, 8, 12, 6]` | `(4,)` | Four features |
| Batch axis retained | `[[3, 8, 12, 6]]` | `(1, 4)` | One sequence, four features |

Both forms contain four elements. What changes is the arrangement of the axes. If the input has shape `(1,4)`, prepare its matching previous state with shape `(1,3)`. When starting from the first time step, `torch.zeros(1, 3, dtype=torch.float64)` can provide that state. `torch.zeros` creates a tensor of the specified shape filled with zeros. `dtype` determines the data type of the stored numbers, and `torch.float64` is the 64-bit floating-point type. Use the same data type for the input and model as well.

RNNCell also accepts a `(4,)` input with a `(3,)` state, so reshape is not always necessary. Choose consistently whether both the input and state include a batch axis or both omit it.

In the full-batch code here, `xt = x[:, t, :]` already returns `(2,4)`, so no reshape is added. Reshaping `(4,)` into `(2,2)` preserves the number of values but changes their meaning as sequences and features. Such an input cannot be passed to this cell with `input_size=4`.

## 12. Read the code that stores and combines states

Before examining the complete example, define the data type and the way results are recorded. <strong>float64</strong> represents floating-point numbers using 64 bits. For numerical verification, this example uses float64 consistently for the model, inputs, and initial state.

| Expression | Kind and operation | Result or effect |
| --- | --- | --- |
| `cell.double()` | Module method: converts floating-point parameters and related values to float64 | Returns the same module |
| `torch.tensor(..., dtype=DTYPE)` | Function: creates a tensor from the supplied values | A tensor with the specified data type |
| `torch.zeros(B, H, dtype=DTYPE)` | Function: creates a state filled with zeros | A `(2,3)` tensor |
| `with torch.no_grad():` | Context that disables automatic differentiation recording in its block | Runs the calculation without recording it for gradients |
| `weight.copy_(values)` | Tensor method: copies prepared values into the existing tensor | Changes the existing tensor's values |
| `weight.fill_(value)` | Tensor method: fills every element with the same value | Changes the existing tensor's values |
| `state[0].tolist()` | Tensor method: converts the first sequence's state to a Python list | A list of three numbers |
| `state[0, 0].item()` | Tensor method: converts a single element to a Python number | One number |

Automatic differentiation records calculations so that gradients can be computed later. We use `no_grad` because we are assigning the weights directly and inspecting forward results. The model's ability to train is not permanently disabled after the block ends.[^no-grad]

A trailing `_` in a PyTorch tensor method name conventionally indicates that the existing tensor is modified. No rule requires all weights to have the same value. Here, we use `copy_` instead of `fill_` to assign different values to the different connections.[^module][^tensor]

Use `item()` when a tensor contains exactly one element. For `state[0]`, which contains three numbers, use `tolist()`. `tolist()` returns nested lists for tensors with multiple axes and a Python number for a scalar tensor. Keep tensors for subsequent calculations; use list conversion to inspect printed values.[^item]

### trace.append and stack do not add the values together

`trace` is a Python list that holds the result from each time step. `trace.append(state)` adds the current state tensor to the end of the list. When RNNCell returns a new tensor on the next call, the name `state` refers to the new result, but the tensor previously added to the list remains there. Modifying that stored tensor itself can still change its contents.

`torch.stack(trace, dim=1)` <strong>combines tensors of the same shape by creating a new axis.</strong> Here, it arranges five `(2,3)` tensors along axis 1 of the result, producing `(2,5,3)`. The 1 in `dim=1` is the number of the new time axis. The operation does not sum or average the values across time steps.[^stack]

In the function that omits the batch axis, five `(3,)` states are combined with `dim=0` to produce `(5,3)`. Both approaches keep the five time-step results separate.

## 13. nn.RNN's output and h_n contain different selections of states

`nn.RNN` is a class that processes complete sequences. With `nn.RNNCell`, the developer writes the time-step loop. With `nn.RNN`, the complete sequence is supplied in one call. These are different interfaces; this does not mean that `nn.RNN` must call RNNCell objects through a Python loop.[^rnn]

This post uses `input_size=4`, `hidden_size=3`, `num_layers=1`, `batch_first=True`, and `bidirectional=False`. `num_layers` is the number of RNN layers, and `bidirectional=False` means that only the forward direction is used.

`output, h_n = rnn(x)` returns two tensors. The following explanation applies to our current example: one layer, one direction, and two sequences of length 5.

<figure class="rnn-figure">
<img src="/rnn-sequence-hidden-state/output-shapes-en.svg" alt="Processing input (2,5,4) with the RNN returns output (2,5,3), containing hidden states from all time steps, and h_n (1,2,3), containing final states. output[:, -1, :] and h_n[0] give the same (2,3) final state." loading="lazy">
<figcaption>output contains all three hidden values at every time step. Selecting the final time step still leaves the three values A, B, and C for each sequence.</figcaption>
</figure>

| Tensor | Contents | Axis meanings | Shape |
| --- | --- | --- | --- |
| `x` | Complete input | Sequence, time step, input feature | `(2,5,4)` |
| `output` | Hidden states from all time steps | Sequence, time step, hidden value | `(2,5,3)` |
| `h_n` | Final hidden state for each layer and direction | Layer and direction, sequence, hidden value | `(1,2,3)` |
| `output[:, -1, :]` | State at the final time step | Sequence, hidden value | `(2,3)` |
| `h_n[0]` | Final state for the only layer–direction combination | Sequence, hidden value | `(2,3)` |

`-1` selects the last index on its axis. Axis 0 of `h_n` has size 1 because this model has one layer and one direction, not because the batch size is 1. Axis 1 has size 2 for the two sequences, with valid indices `0–1`. Axis 2 has size 3 for the hidden values, with valid indices `0–2`.

### batch_first sets the axis order of the input and output

| Setting | Input axis order | output axis order | h_n axis order |
| --- | --- | --- | --- |
| `batch_first=True` | `(B,T,F)` | `(B,T,H)` | `(L,B,H)` |
| `batch_first=False` | `(T,B,F)` | `(T,B,H)` | `(L,B,H)` |

The default is `batch_first=False`. <strong>Setting `batch_first=True` does not change the axis order of `h_n`.</strong> In the table, `L` is the layer count for a unidirectional model. In a bidirectional model, the first axis includes both the layer count and the direction count.[^rnn]

Supplying the whole input in one call preserves the time dependency between hidden states. One function call and five time steps processed inside that call are different counts.

## 14. What to pass when splitting calls or stacking layers

### Continuing the same sequence across multiple calls

RNNCell and RNN do not automatically pass the final state of one call into the next. When processing parts of the same sequence, pass the returned state explicitly.

| Order | Expression | Meaning |
| --- | --- | --- |
| First call | `out1, h1 = rnn(x[:, 0:1, :])` | Process the first time step from a zero state |
| Next call | `out2, h2 = rnn(x[:, 1:2, :], h1)` | Process the second time step using the first call's final state |

The slices `0:1` and `1:2` retain a time axis of length 1. Each input has shape `(2,1,4)`, and the supplied `h1` has shape `(1,2,3)`. When starting independent new sequences, avoid accidentally carrying over the previous sequences' states.

### A later time step in the same layer differs from a higher layer

The subscript 2 in `h₂` denotes the second time step, not the second hidden layer. The same layer shares its weights across time and keeps the same hidden size `H`.

When several RNN layers are stacked, each has its own weights and hidden state. A higher layer uses <strong>the lower layer's output at the same time step</strong> and <strong>its own state from the previous time step.</strong>

| Connection | What is passed as input | Parameter relationship |
| --- | --- | --- |
| Previous time step → current time step in the same layer | That layer's previous hidden state | The same parameters are used across time |
| Lower layer → higher layer at the same time step | The lower layer's current hidden state | Each layer has separate parameters |

Increasing `num_layers` in one `nn.RNN` gives every layer the same `hidden_size`. To use different hidden sizes across layers, connect separate modules and match each higher layer's input size to the lower layer's output size. In a multilayer `nn.RNN`, `output` contains <strong>every time step from the top layer</strong>, while `h_n` contains <strong>the final state of each layer and direction.</strong>[^rnn]

### Distinguish the hidden state from the task's prediction

The three hidden values do not automatically become a temperature prediction or a classification result. If the task requires an output size of `K`, applying `nn.Linear(H,K)` to the final state `(B,H)` produces `(B,K)`. `Linear` multiplies values on the last input axis by weights and adds a bias.[^linear]

For predictions at every time step, apply the same output layer to each step of `(B,T,H)` to produce `(B,T,K)`. `H` is the internal hidden-state size; `K` is the output size required by the task. For classification, do not automatically call these outputs probabilities. Interpret them according to the loss function and prediction procedure being used.

## 15. Run the complete Python example with the same numbers

The code below does not train the model. It checks that RNNCell, manual matrix calculations, and RNN produce matching results with the specified weights. Run the complete code in order from the top.

`assert` is Python syntax that raises an error when its condition is false. Here, it checks the example's results. `torch.allclose` checks whether corresponding elements agree within a tolerance. `rtol` is a relative tolerance scaled by the reference value's magnitude, and `atol` is an absolute tolerance. They allow for small rounding differences caused by the order of floating-point operations.[^allclose]

The code follows this flow: <strong>create a cell → set its parameters → prepare the input batch → compare manual and cell calculations at each time step → collect the outputs → experiment with order and length → compare with RNN.</strong> Every parameter is assigned explicitly, so the random values used when the model is first created do not enter the later calculations.

```python
# RNN sequences and hidden states: CPU / float64 / forward computation only
import torch
from torch import nn

# B: batch size, T: time steps, F: input features, H: hidden neurons
B, T, F, H = 2, 5, 4, 3

# Use 64-bit floating-point numbers to inspect the numerical calculation.
DTYPE = torch.float64

# Create one cell object.
# bias=True: use both bias vectors.
# nonlinearity="tanh": apply tanh to the summed inputs and biases.
cell = nn.RNNCell(
    input_size=F,
    hidden_size=H,
    bias=True,
    nonlinearity="tanh",
).double()

# Replace the initial parameters with the weights and biases from the article.
# Do not record these assignments for automatic differentiation.
with torch.no_grad():
    cell.weight_ih.copy_(torch.tensor([
        [ 0.11, -0.07,  0.04,  0.02],  # Current neuron A
        [-0.09,  0.13,  0.06, -0.03],  # Current neuron B
        [ 0.05,  0.08, -0.12,  0.15],  # Current neuron C
    ], dtype=DTYPE))

    cell.weight_hh.copy_(torch.tensor([
        [ 0.21, -0.14,  0.07],  # Previous A, B, C -> current A
        [-0.16,  0.25,  0.12],  # Previous A, B, C -> current B
        [ 0.19, -0.06,  0.31],  # Previous A, B, C -> current C
    ], dtype=DTYPE))

    cell.bias_ih.copy_(torch.tensor(
        [0.015, -0.025, 0.035], dtype=DTYPE
    ))

    cell.bias_hh.copy_(torch.tensor(
        [-0.008, 0.018, -0.028], dtype=DTYPE
    ))

# shape = (B, T, F) = (2, 5, 4)
x = torch.tensor([
    [
        [2., 5., 7., 11.],
        [3., 8., 12., 6.],
        [4., 6., 9., 13.],
        [8., 3., 5., 10.],
        [7., 9., 2., 14.],
    ],
    [
        [6., 4., 8., 3.],
        [9., 5., 10., 7.],
        [12., 11., 6., 4.],
        [5., 14., 3., 8.],
        [11., 7., 13., 2.],
    ],
], dtype=DTYPE)

# Record the identity of each parameter object.
ids_before = tuple(id(p) for p in cell.parameters())

# Make separate copies of the parameter values as well.
# detach(): disconnect from the automatic differentiation history.
# clone(): create a separate copy of the data.
values_before = [p.detach().clone() for p in cell.parameters()]

# Each of the two sequences has three hidden values.
# shape = (B, H) = (2, 3)
state = torch.zeros(B, H, dtype=DTYPE)

# Store the (B, H) result tensor from each time step.
trace = []

with torch.no_grad():
    for t in range(T):
        # t takes the values 0, 1, 2, 3, and 4.
        # Select the same time-step index from both sequences.
        # shape: (B, F) = (2, 4)
        xt = x[:, t, :]

        # Every current neuron reads this same previous state.
        # Do not modify this tensor while computing individual neurons.
        previous_state = state

        # @ performs matrix multiplication.
        # .T transposes the two-dimensional weight matrix here.
        #
        # Input:        (B, F) = (2, 4)
        # weight_ih:    (H, F) = (3, 4)
        # Transposed:   (F, H) = (4, 3)
        # Product:      (B, H) = (2, 3)
        input_part = xt @ cell.weight_ih.T

        # Previous state: (B, H) = (2, 3)
        # weight_hh.T:    (H, H) = (3, 3)
        # Product:        (B, H) = (2, 3)
        #
        # Transposing a square matrix still changes the positions of its values.
        previous_part = previous_state @ cell.weight_hh.T

        # Each bias vector has shape (H,).
        # The same bias is added to the matching neuron in every sequence.
        manual_state = torch.tanh(
            input_part
            + cell.bias_ih
            + previous_part
            + cell.bias_hh
        )

        # Perform the same calculation using PyTorch's RNNCell.
        # Returned shape: (B, H) = (2, 3)
        state = cell(xt, previous_state)

        # allclose: compare corresponding elements within a numerical tolerance.
        # rtol: relative tolerance, scaled by the reference value's magnitude.
        # atol: absolute tolerance.
        assert torch.allclose(
            state, manual_state, rtol=1e-10, atol=1e-12
        )

        trace.append(state)

        # Print only the new hidden state of the first sequence.
        # state[0] has shape (3,).
        print(f"Time step {t + 1}: {state[0].tolist()}")

    # Combine five (2, 3) tensors by adding a time axis.
    # Result shape: (B, T, H) = (2, 5, 3)
    output = torch.stack(trace, dim=1)

print("Full input:", tuple(x.shape))
print("Full output:", tuple(output.shape))
print("Final state:", tuple(state.shape))

# Outputs for the first sequence, rounded to six decimal places:
#
# Time step 1: [0.360099,  0.502765, 0.866036]
# Time step 2: [0.416009,  0.900527, 0.510735]
# Time step 3: [0.567644,  0.654344, 0.940266]
# Time step 4: [0.824405, -0.150259, 0.956839]
# Time step 5: [0.645847,  0.176484, 0.997772]


# -------------------------------------------------------------------
# A function that processes one sequence
# Input: (T, F), return value: (T, H)
# -------------------------------------------------------------------

def run_given_cell(values: torch.Tensor) -> torch.Tensor:
    # ndim is the number of tensor axes.
    # This function receives a sequence (T, F), without a batch axis.
    if values.ndim != 2 or values.shape[1] != F or values.shape[0] == 0:
        raise ValueError("values must be a nonempty (T, F) tensor.")

    # Check that the model and input use the same data type and device.
    if (
        values.dtype != cell.weight_ih.dtype
        or values.device != cell.weight_ih.device
    ):
        raise ValueError("values and cell must have the same data type and device.")

    # Create a fresh initial state for every function call.
    # The final state of an earlier call is therefore not carried over.
    # The batch axis is omitted, so the shape is (H,).
    state = torch.zeros(
        H,
        dtype=values.dtype,
        device=values.device,
    )

    trace = []

    with torch.no_grad():
        # Iterating over 2D values yields a (F,) tensor for each time step.
        for xt in values:
            # Use the previous state as input, then receive the new state.
            state = cell(xt, state)
            trace.append(state)

    # Combine T tensors of shape (H,) into a tensor of shape (T, H).
    return torch.stack(trace, dim=0)


# -------------------------------------------------------------------
# Experiment A: What changes when the inputs are reordered?
# -------------------------------------------------------------------

original = run_given_cell(x[0])

# Change only the time-step order in the first sequence.
# [2, 0, 1, 3, 4] specifies the order of time-step indices to use.
reordered = run_given_cell(x[0, [2, 0, 1, 3, 4], :])

print("Final state in the original order:", original[-1].tolist())
print("Final state after reordering:", reordered[-1].tolist())

# Original order:
# [0.645847, 0.176484, 0.997772]
#
# Reordered inputs:
# [0.638208, 0.188234, 0.997711]

# Reordering the inputs changes the final state in this example.
assert not torch.allclose(original[-1], reordered[-1])

# Reason:
# Reordering earlier inputs changes the previous hidden state,
# and that state is used in the calculation at the next time step.
#
# A changed order is not guaranteed to produce a different result for
# every possible choice of weights and inputs. This example does differ.


# -------------------------------------------------------------------
# Experiment B: What changes when the sequence is shortened?
# -------------------------------------------------------------------

# Process only the first three time steps.
short = run_given_cell(x[0, :3, :])

# Compare with the first three results from the complete five-step sequence.
assert torch.allclose(
    short, original[:3], rtol=1e-10, atol=1e-12
)

# Reason:
# The initial state, input prefix, and weights are the same.
# In a unidirectional RNN, later inputs do not retroactively change
# the hidden states that were computed at earlier time steps.


# -------------------------------------------------------------------
# Experiment C: Are the parameter objects and their values unchanged?
# -------------------------------------------------------------------

# Compare object identities.
assert ids_before == tuple(id(p) for p in cell.parameters())

# Compare the numerical values separately.
# zip: pair each current parameter with its saved copy in order.
# torch.equal: check that shapes and element values match.
# all: check that every comparison is True.
assert all(
    torch.equal(p, before)
    for p, before in zip(cell.parameters(), values_before)
)

print("Parameter tensor count:", len(tuple(cell.parameters())))
print("Trainable scalar count:", sum(p.numel() for p in cell.parameters()))

# Output:
# Parameter tensor count: 4
# Trainable scalar count: 27


# -------------------------------------------------------------------
# Experiment D: Does nn.RNN produce the same sequence of states?
# -------------------------------------------------------------------

rnn = nn.RNN(
    input_size=F,
    hidden_size=H,
    num_layers=1,          # One RNN layer.
    nonlinearity="tanh",
    bias=True,
    batch_first=True,     # Input and output: (B, T, features).
    bidirectional=False,  # Process only in the forward direction.
).double()

with torch.no_grad():
    # Use identical parameter values for the comparison.
    # _l0 refers to the first RNN layer, whose layer index is 0.
    rnn.weight_ih_l0.copy_(cell.weight_ih)
    rnn.weight_hh_l0.copy_(cell.weight_hh)
    rnn.bias_ih_l0.copy_(cell.bias_ih)
    rnn.bias_hh_l0.copy_(cell.bias_hh)

    # Pass the entire sequence batch in one call.
    # The omitted initial hidden state defaults to zeros.
    rnn_output, h_n = rnn(x)

# Compare with the repeated RNNCell calls.
assert torch.allclose(
    rnn_output, output, rtol=1e-10, atol=1e-12
)

# Compare the final hidden states.
assert torch.allclose(
    h_n[0], output[:, -1, :], rtol=1e-10, atol=1e-12
)

# Compare the first sequence processed in the batch with the same
# sequence processed on its own.
assert torch.allclose(
    output[0], original, rtol=1e-10, atol=1e-12
)

print("nn.RNN output:", tuple(rnn_output.shape))  # (2, 5, 3)
print("nn.RNN h_n:", tuple(h_n.shape))            # (1, 2, 3)
print("All checks passed.")
```

## 16. Check four things in the results

### Every time step of the first sequence retains three results

| Time step | Hidden value A | Hidden value B | Hidden value C |
| ---: | ---: | ---: | ---: |
| 1 | 0.360099 | 0.502765 | 0.866036 |
| 2 | 0.416009 | 0.900527 | 0.510735 |
| 3 | 0.567644 | 0.654344 | 0.940266 |
| 4 | 0.824405 | -0.150259 | 0.956839 |
| 5 | 0.645847 | 0.176484 | 0.997772 |

All five time steps produce the results of A, B, and C together. Each row is one time step of the first sequence. The complete batch output also contains the second sequence's results separately.

### Reordering the inputs changes this example's final state

`x[0, [2, 0, 1, 3, 4], :]` selects the first sequence's time steps in index order 2, 0, 1, 3, 4. The feature order within each time step stays the same.

| Time-step index order | Final state |
| --- | --- |
| Original `[0,1,2,3,4]` | `[0.645847, 0.176484, 0.997772]` |
| Reordered `[2,0,1,3,4]` | `[0.638208, 0.188234, 0.997711]` |

The final input is the same in both runs. However, processing the earlier inputs produces different hidden states, which affect the final calculation. Reordering is not guaranteed to produce a different result for every possible combination of weights and inputs. This example verifies a case where the results do differ.

### Processing only the first three steps matches the full run's first three results

`run_given_cell(x[0, :3, :])` processes only the first three time steps. Its output matches the first three steps of the full five-step output because the initial state, input prefix, and weights are the same. In this unidirectional RNN, later inputs do not retroactively change states already computed at earlier steps.

`run_given_cell` creates a zero state inside the function on every call, so it does not inherit the final state from an earlier call. Its flow is <strong>create the initial state → select one input → compute with the previous state → save the result → repeat → return all results.</strong> Calling a function and designing it to carry state across calls are separate matters.

### Check parameter identities and values separately

| Code | What it checks or provides |
| --- | --- |
| `cell.parameters()` | An iterator that provides the module's registered parameters in order |
| `id(p)` | The identity of a Python object |
| `p.numel()` | The number of values in the tensor |
| `p.detach().clone()` | A separate data copy disconnected from the automatic differentiation history |
| `torch.equal(p, before)` | Whether the shapes and corresponding values are equal |

`detach()` disconnects from the automatic differentiation history but shares data with the original tensor. The following `clone()` creates a separate data copy, allowing the values from before the forward pass to be retained.[^tensor]

An object can keep the same identity while its internal numbers change. The example therefore compares object identities and numerical values separately. `zip` pairs each current parameter with its saved copy in order, and `all` checks whether every comparison is true.[^python]

In this example, `state` changes at each time step while the 27 parameter values remain unchanged. This separates changes in the computed state from model training.

## 17. Which counts to compare across MLPs, CNNs, and RNNs

An <strong>MLP</strong> is a neural network built from fully connected layers and related components. A <strong>CNN</strong> uses convolutions to compute values over spatial regions. When comparing these structures, compare the same kinds of quantities.

| Structure | Inputs read by one output calculation | Weight structure | Where the weights are reused |
| --- | --- | --- | --- |
| Fully connected layer in an MLP | All input features of that layer | One weight per input feature for each output neuron | The same layer is applied to multiple samples |
| Standard `Conv2d` with `groups=1` | The kernel region across all input channels | `input channels × kernel height × kernel width` weights per filter | Multiple spatial positions and samples |
| Basic RNN | The complete current input and previous hidden state | `F+H` weights per hidden neuron | Multiple time steps and sequences |

One CNN filter reuses its weights at multiple spatial positions to produce one output channel. A filter does not always produce only one number. `groups=1` is the standard setting in which every filter uses all input channels.[^conv]

For a later MLP layer, its number of input features is the number of outputs from the preceding layer. Input feature count and output neuron count do not have to match. An MLP can receive order-related information as input, but a basic fully connected layer has no recurrent connection carrying the previous time step's state as an RNN does.[^linear]

In an RNN, the input feature count determines <strong>the number of input weights in each neuron.</strong> The number of cell boxes drawn along the time axis corresponds to the time-step count. Read these as separate relationships.

For the basic structure of matrix multiplication, see the [post on neural network matrix calculations](/en/neural-network-matrix-basics/). For channel and filter connections, see the [post on convolution basics](/en/cnn-convolution-basics/).

## 18. Changing a state during a forward pass is different from training

In this example, `state` is newly computed whenever an input is processed. We do not change the weights or biases, and we do not call `backward()` or `optimizer.step()`.

A typical process that includes training follows these steps.

1. Prepare the initial state for a new sequence.
2. Compute hidden states through the time-step forward calculations.
3. Produce predictions using an output layer or another component suited to the task.
4. Compare the predictions with the targets to compute a loss. A loss measures how the predictions differ from the targets.
5. Use backpropagation to compute each parameter's gradient. A gradient describes the direction and rate at which the loss changes when a parameter changes slightly.
6. Let the optimizer update the weights and biases using the gradients. An optimizer is an object that applies a rule for modifying parameters.

Processing the next input along the time axis and updating parameters are different stages. A typical forward pass through a sequence uses the same parameters at several time steps. Training needs an automatic differentiation record of those calculations, so do not carry this example's inspection-only `no_grad` block into the training forward pass unchanged.[^autograd]

The initial state in this example is an ordinary tensor filled with zeros. It is not registered as a separate trainable parameter. An initial state can be designed to be learned, but here it is a fixed starting value.

## 19. Check these points when results differ from expectations

| Observed problem or interpretation | Cause | What to inspect |
| --- | --- | --- |
| The cell's result is interpreted as one number. | One returned tensor has been confused with the number of elements it contains. | One sequence's state contains three hidden values. |
| Newly computed A is used while calculating current B. | The previous state was overwritten before the new state was complete. | Every neuron must read the same previous state. |
| New weights appear at each time step. | The cell object was recreated inside the loop. | Create the cell outside the loop and reuse it. |
| Every calculation starts from zero, as if it were the first step. | The previous result was not passed into each call. | Inspect how the state is passed in `state = cell(xt, state)`. |
| An input shape error occurs. | `F`, `T`, and `B` have been confused, or the input and state use batch axes inconsistently. | Check for current input `(2,4)` and previous state `(2,3)`. |
| Shapes match, but the manual results differ. | The recurrent weight transpose or one of the two biases may be missing. | Check `weight_hh.T`, `bias_ih`, and `bias_hh`. |
| `state[0].item()` raises an error. | A single number was requested from three elements. | Use `tolist()` for all values, or select one element before calling `item()`. |
| `h_n` is read as `(B,L,H)`. | `batch_first` was assumed to change the state's axis order too. | Here, the shape is `(L,B,H)=(1,2,3)`. |
| A new input was created, but the result did not change. | The new variable may not have been used in the actual cell call. | Check the variable passed to `cell(...)` or `rnn(...)`. |
| Results for independent sequences depend on an earlier run. | The earlier run's final state may be carried over. | Decide whether this is a fresh start or a continuation, then prepare the state accordingly. |

When investigating an error, first inspect <strong>the actual values and shapes of the input and previous state passed to the model.</strong> Then check the cell arguments, weight connections, and the location of state updates to narrow down the cause.

## 20. Verify the same two time steps in Rust

Rust can perform the same calculation: sum the weighted current inputs, add the sum of the weighted previous hidden values and the biases, then apply `tanh`. The code below selects only the first two of the first sequence's five time steps to check the numbers.

The Python example uses PyTorch's tensor operations and parameter management. The Rust example below uses standard arrays and loops to calculate forward values only. It does not implement automatic differentiation or parameter training. This difference concerns the scope of these examples; it does not mean that training is unavailable in the Rust language.

| Meaning | Python/PyTorch | Rust code below |
| --- | --- | --- |
| Floating-point data type | `torch.float64` | `f64` |
| Four features | A tensor with shape `(4,)` | `[f64; 4]` |
| Input weights for each neuron | A tensor with shape `(3,4)` | `[[f64; 4]; 3]` |
| Compute from the previous state | Matrix multiplication | Array indexing and loops |
| Replace the state after computing all new results | Assign the returned tensor to `state` | Build `next_state`, then assign it |

`[f64; 4]` is a fixed-size array containing four floating-point numbers. `[[f64; 4]; 3]` contains three such arrays. `mut` allows a variable to be assigned a new value. `0..3` starts at 0 and excludes 3, yielding `0,1,2`. `iter()` reads items by reference, and `enumerate()` also provides an index starting at 0.[^rust]

```rust
// Reproduce the first two steps of the first sequence using standard Rust.
fn main() {
    // Input weights: 3 current neurons x 4 input features.
    let w_ih: [[f64; 4]; 3] = [
        [ 0.11, -0.07,  0.04,  0.02],
        [-0.09,  0.13,  0.06, -0.03],
        [ 0.05,  0.08, -0.12,  0.15],
    ];

    // Recurrent weights: 3 current neurons x 3 previous hidden values.
    let w_hh: [[f64; 3]; 3] = [
        [ 0.21, -0.14,  0.07],
        [-0.16,  0.25,  0.12],
        [ 0.19, -0.06,  0.31],
    ];

    let b_ih = [0.015, -0.025, 0.035];
    let b_hh = [-0.008, 0.018, -0.028];

    // The first two time steps of the first sequence.
    let inputs: [[f64; 4]; 2] = [
        [2.0, 5.0, 7.0, 11.0],
        [3.0, 8.0, 12.0, 6.0],
    ];

    // mut allows a new state to be assigned for the next time step.
    let mut state = [0.0_f64; 3];

    // iter(): read each time-step array by reference, in order.
    // enumerate(): also provide its index, starting from 0.
    for (t, xt) in inputs.iter().enumerate() {
        let mut next_state = [0.0_f64; 3];

        // This code uses a loop to compute the neurons, but each neuron
        // reads the previous state, never another neuron's current result.
        for neuron in 0..3 {
            let mut input_sum = 0.0;
            let mut previous_sum = 0.0;

            for feature in 0..4 {
                input_sum += xt[feature] * w_ih[neuron][feature];
            }

            for previous_neuron in 0..3 {
                previous_sum +=
                    state[previous_neuron] * w_hh[neuron][previous_neuron];
            }

            next_state[neuron] = (
                input_sum
                + previous_sum
                + b_ih[neuron]
                + b_hh[neuron]
            ).tanh();
        }

        // After every neuron has used the same previous state,
        // replace it with the complete new state.
        state = next_state;

        // {:?}: debug formatting, which can display an array.
        println!("Time step {}: {:?}", t + 1, state);
    }
}
```

The Rust loop computes the neurons one after another, starting with A. All three still read the unchanged previous `state` and write their results into a separate `next_state`. Only after every result is complete is the entire state replaced, preserving the same connections as RNNCell.

The first output is `[0.360099…, 0.502765…, 0.866036…]`, and the second is `[0.416009…, 0.900527…, 0.510735…]`. Both languages show the same process: four current input values and three previous hidden values produce three new hidden values, then the same weights are used at the next time step.

## References

The APIs in this post were checked against the official documentation on October 29, 2026. The examples were run with PyTorch 2.15.0 on the CPU and Rust 1.99.1. The input, weight, and output values are example numbers chosen to make the calculation traceable.

[^cell]: PyTorch, [RNNCell](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNNCell.html). The basic cell's formula, input and hidden-state shapes, two weight matrices, and two bias vectors.
[^rnn]: PyTorch, [RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html). Time-step computation, connections across layers, `batch_first`, `output`, `h_n`, and parameter shapes.
[^tensor]: PyTorch, [Tensors](https://docs.pytorch.org/docs/stable/tensors.html), [Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html), [torch.clone](https://docs.pytorch.org/docs/stable/generated/torch.clone.html). Tensor indexing and methods, data sharing, and copying.
[^module]: PyTorch, [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html). Calling modules, `parameters()`, `double()`, and parameter management.
[^no-grad]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html). A context for computation with gradient recording disabled.
[^item]: PyTorch, [Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html), [Tensor.tolist](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html). Converting one element or all tensor values to Python data types.
[^stack]: PyTorch, [torch.stack](https://docs.pytorch.org/docs/stable/generated/torch.stack.html). Combining tensors of the same size along a new axis.
[^reshape]: PyTorch, [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html). Changing tensor shape while preserving the element count.
[^allclose]: PyTorch, [torch.allclose](https://docs.pytorch.org/docs/stable/generated/torch.allclose.html). Elementwise comparison with relative and absolute tolerances.
[^linear]: PyTorch, [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html). A linear layer applied to the last axis and its input and output shapes.
[^conv]: PyTorch, [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html). Channels, filters, groups, and weight shapes.
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html), [Optimizers](https://docs.pytorch.org/docs/stable/optim.html). Recording calculations, backpropagation, and parameter updates.
[^python]: Python, [Assignment statements](https://docs.python.org/3/reference/simple_stmts.html#assignment-statements), [Built-in Functions](https://docs.python.org/3/library/functions.html). Assignment semantics and `id`, `zip`, `all`, and `sum`.
[^rust]: Rust, [array](https://doc.rust-lang.org/std/primitive.array.html), [f64::tanh](https://doc.rust-lang.org/std/primitive.f64.html#method.tanh), [Control Flow](https://doc.rust-lang.org/book/ch03-05-control-flow.html). Fixed-size arrays, floating-point operations, and loops.
