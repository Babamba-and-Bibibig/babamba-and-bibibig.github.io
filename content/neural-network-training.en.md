+++
title = "How Neural Networks Learn: Forward Passes, Backpropagation, and Gradient Descent"
description = "Follow one complete training step in a small neural network that predicts the duration of two packing jobs. Connect batch matrix operations, mean loss, parameter gradients, backpropagation, learning rates, and actual updates through diagrams, tables, and PyTorch code."
date = 2026-10-21T19:00:00+09:00
slug = "neural-network-training"

[extra]
katex = true
toc = true
styles = ["neural-network-training/article.css"]
+++

Suppose we want to predict how long it takes to pack boxes. We tell the model how many small boxes and large boxes a job contains, and the model calculates an estimated duration in minutes. When its estimate differs from the recorded duration, we need to adjust numbers inside the model. This article explains how to determine which numbers to adjust and in which direction.

We will use <strong>two fictional job records and a small neural network</strong> so that we can check every calculation. Job A has the input `[2, 1]` and a target duration of 6 minutes. Job B has the input `[1, 1]` and a target duration of 4 minutes. These are not measurements of real packing performance. We also choose the network's initial values ourselves to make the calculations easy to follow.

<strong>Training a neural network repeats four tasks: make predictions with the current settings, evaluate those predictions, calculate how each setting affects the evaluation, and then adjust the settings.</strong> We will follow the same example to understand what forward and backward passes exchange, why one loss produces several gradients, and why a gradient's magnitude is not a distance to move.

## 1. What actually changes during one training step?

First, distinguish the four tasks involved in training. A <strong>parameter</strong> is a number inside the model that training can adjust. Here, the parameters are <strong>weights</strong>, which multiply inputs, and <strong>biases</strong>, which are added to the sum of those products.

| Task | What it does | Do the parameters change at this stage? |
| --- | --- | --- |
| Forward pass | Calculates predictions using the current parameters. | No. |
| Loss calculation | Compares predictions with targets to produce an evaluation value. | No. |
| Backward pass | Calculates how the loss changes when parameters change. | No. |
| Optimizer update | Changes parameter values using gradients and an update rule. | Yes, at this stage. |

A <strong>loss</strong> is a value chosen to evaluate predictions, and training aims to reduce it. A <strong>gradient</strong> describes the local rate at which the loss changes when a parameter changes by a very small amount. An <strong>optimizer</strong> uses this information to choose and apply the actual parameter updates.[^optimization]

A group of samples processed together is a <strong>batch</strong>. In this article, A and B form one batch. We perform one backward pass and one update for each batch.

<figure class="nt-figure">
<a href="/neural-network-training/workflow-en.svg" target="_blank" rel="noopener"><img src="/neural-network-training/workflow-en.svg" alt="Inputs pass through the forward computation to produce predictions. Targets are compared with those predictions in the loss calculation. Backpropagation calculates parameter gradients, and the optimizer updates the parameters. The next batch uses the updated values."></a>
<figcaption>The gradients calculated by backpropagation and the parameters changed by the optimizer are different information. Select the diagram to open a larger version.</figcaption>
</figure>

Before training begins, the model structure and the initial parameters of every layer must be ready. We do not train an early layer during the forward pass and then create the next layer. The next batch uses the same model, with the parameter values changed by the previous update.

## 2. Distinguish two samples, two features, and nine parameters

A <strong>sample</strong> is one case consisting of inputs and their corresponding target. A <strong>feature</strong> is one item that makes up the input. Here, one packing job is a sample, and the counts of small and large boxes are its two features.

| Sample | Small-box count `x1` | Large-box count `x2` | Actual duration `y` |
| --- | ---: | ---: | ---: |
| A | 2 | 1 | 6 minutes |
| B | 1 | 1 | 4 minutes |

`[2, 1]` is <strong>one sample containing two features</strong>, rather than two samples. The inputs for the two jobs contain four numbers, but there are only two samples.

We use `y` for the actual target and `ŷ`, pronounced “y hat,” for the prediction calculated by the model. Predicting a numerical quantity such as an amount or a duration is a <strong>regression problem</strong>. The target is already provided in the data. The model changes its weights and biases to bring its predictions closer to those targets.

A <strong>neuron</strong> is a calculation unit that multiplies several inputs by their respective weights and adds the products and a bias. It can also apply an activation function, which transforms the result before passing it to the next layer. A <strong>layer</strong> groups such units into a calculation stage. A <strong>hidden layer</strong> is an intermediate stage between the input and the final output.

Our network has <strong>2 input features → 2 hidden neurons → 1 output neuron</strong>. Every value from the previous stage connects to each neuron in the next layer, so both layers are <strong>fully connected layers</strong>.

| What are we counting? | Count in this model | What the count refers to |
| --- | ---: | --- |
| Samples in the batch | 2 | Jobs A and B |
| Input features per sample | 2 | Small-box count and large-box count |
| Hidden neurons | 2 | Two intermediate calculations |
| Output neurons | 1 | The estimated duration of one job |
| Trainable fully connected layers | 2 | The hidden layer and the output layer |
| Individual trainable parameter values | 9 | 6 weights and 3 biases |

We do not count the input positions as trainable layers. Adding more samples does not increase the number of weights or biases in this model. <strong>The model applies the same parameters to every sample.</strong>

## 3. Neurons in the same layer calculate separately from the same input

Let the first hidden neuron's weights be `w1` and `w2`, with bias `b1`. For the second hidden neuron, use `w3`, `w4`, and `b2`. Name the two weighted sums, including their biases, `z1` and `z2`.

An <strong>activation function</strong> transforms a neuron's weighted sum and bias result before passing it to the next layer. We use <strong>ReLU</strong>, which leaves positive values unchanged and returns 0 for all other values. Call the two values after ReLU `h1` and `h2`.

<div class="nt-math">
$$
\begin{aligned}
z_1&=x_1w_1+x_2w_2+b_1,& h_1&=\max(0,z_1)\\
z_2&=x_1w_3+x_2w_4+b_2,& h_2&=\max(0,z_2)
\end{aligned}
$$
</div>

The output neuron receives `h1` and `h2`. It multiplies them by weights `v1` and `v2` and adds bias `b3`. The final prediction is:

<div class="nt-math">
$$\hat y=h_1v_1+h_2v_2+b_3$$
</div>

We do not add a separate ReLU to the output layer. A system that predicts real durations would also need design decisions about issues such as whether to allow negative predictions. Here, we use a simple regression model to examine the training calculations.

Fix the initial parameters to the following values.

| Calculation unit | Weights multiplied by its inputs | Bias added at the end |
| --- | --- | ---: |
| Hidden neuron 1 | `w1=1`, `w2=2` | `b1=0` |
| Hidden neuron 2 | `w3=−1`, `w4=1` | `b2=2` |
| Output neuron | `v1=2`, `v2=−1` | `b3=1` |

<figure class="nt-figure">
<a href="/neural-network-training/network-en.svg" target="_blank" rel="noopener"><img src="/neural-network-training/network-en.svg" alt="Inputs x1 and x2 each connect to both hidden neurons. Each hidden neuron calculates with its own weights and bias and then applies ReLU. The output neuron uses h1 and h2 together." loading="lazy"></a>
<figcaption>Hidden neuron 2 does not receive hidden neuron 1's output. The output layer waits because its calculation needs both hidden outputs.</figcaption>
</figure>

### Calculate the prediction for job A

A's inputs are `x1=2` and `x2=1`. Calculate the first and second neurons separately.

<div class="nt-math">
$$
\begin{aligned}
z_1&=2\times1+1\times2+0=4\\
h_1&=\operatorname{ReLU}(4)=4\\[3pt]
z_2&=2\times(-1)+1\times1+2=1\\
h_2&=\operatorname{ReLU}(1)=1\\[3pt]
\hat y_A&=4\times2+1\times(-1)+1=8
\end{aligned}
$$
</div>

In the second neuron, the sum of the products alone is −1. <strong>Adding the bias of 2 gives `z2=1`.</strong> Including the bias is necessary to calculate the model we defined. The prediction for A is 8 minutes, while its recorded duration is 6 minutes.

### Calculate job B with the same parameters

For B, only the inputs change to `x1=1` and `x2=1`. Calculating A has not already updated the model's weights.

<div class="nt-math">
$$
\begin{aligned}
z_1&=1\times1+1\times2+0=3,& h_1&=3\\
z_2&=1\times(-1)+1\times1+2=2,& h_2&=2\\
\hat y_B&=3\times2+2\times(-1)+1=5
\end{aligned}
$$
</div>

| Sample | Hidden outputs `[h1, h2]` | Prediction | Target |
| --- | --- | ---: | ---: |
| A | `[4, 1]` | 8 minutes | 6 minutes |
| B | `[3, 2]` | 5 minutes | 4 minutes |

The hidden outputs are intermediate values used to calculate the final duration. We are not given a separate “correct duration” for `h1` or `h2` individually.

## 4. A batch groups sample calculations into matrices

A <strong>matrix</strong> arranges numbers into rows and columns. An array's <strong>shape</strong> gives the size of each dimension. Here, the input shape is `(number of samples, number of features)`. One row represents one job.

Call the input matrix `X`, the hidden weight matrix `W_hidden`, and the bias array `b_hidden`. <strong>One column of the weight matrix contains all the weights for one hidden neuron.</strong>

<div class="nt-math">
$$
X=\begin{bmatrix}2&1\\1&1\end{bmatrix},\quad
W_{\mathrm{hidden}}=\begin{bmatrix}w_1&w_3\\w_2&w_4\end{bmatrix}
=\begin{bmatrix}1&-1\\2&1\end{bmatrix},\quad
b_{\mathrm{hidden}}=[0,2]
$$
</div>

Python's `@` is the <strong>matrix multiplication operator</strong>. To produce one result entry, take the corresponding row from the left matrix and column from the right matrix, multiply their matching entries, and add the products. For example, A's second hidden-neuron entry uses `[2, 1]` and `[-1, 1]` to calculate `2×(−1)+1×1=−1`. We then add the second neuron's bias of 2.

<div class="nt-math">
$$
Z=XW_{\mathrm{hidden}}+b_{\mathrm{hidden}}
=\begin{bmatrix}4&-1\\3&0\end{bmatrix}+[0,2]
=\begin{bmatrix}4&1\\3&2\end{bmatrix}
$$
</div>

We apply the bias `[0, 2]` to both rows in the same way. <strong>Broadcasting</strong> is the array-operation rule that applies values from a smaller array to compatible positions. We add a bias <strong>once to each neuron's weighted sum</strong>, rather than once per input feature. That calculation is applied to every sample.[^broadcasting]

Let `H` be the result of applying ReLU to every entry of `Z`. All the current values are positive, so `H=Z`. Calculate the output layer in the same way.

<div class="nt-math">
$$
W_{\mathrm{out}}=\begin{bmatrix}2\\-1\end{bmatrix},\quad b_{\mathrm{out}}=[1]
$$
$$
\mathrm{pred}=HW_{\mathrm{out}}+b_{\mathrm{out}}
=\begin{bmatrix}4&1\\3&2\end{bmatrix}
\begin{bmatrix}2\\-1\end{bmatrix}+[1]
=\begin{bmatrix}8\\5\end{bmatrix}
$$
</div>

<figure class="nt-figure">
<a href="/neural-network-training/batch-matrix-en.svg" target="_blank" rel="noopener"><img src="/neural-network-training/batch-matrix-en.svg" alt="Input X stores A and B in separate rows. Shared weights and biases produce hidden outputs A [4,1] and B [3,2], followed by predictions A 8 and B 5. Each sample keeps its own row." loading="lazy"></a>
<figcaption>Grouping the calculations into matrices does not combine A and B into one sample. The same weights and biases are applied to both rows.</figcaption>
</figure>

Let `N` be the actual number of samples in the batch. The shapes relate as follows.

| Value | Shape | What happens when `N` changes? |
| --- | --- | --- |
| Input `X` | `(N, 2)` | The number of sample rows changes. |
| Hidden weights `W_hidden` | `(2, 2)` | The shape stays the same. |
| Hidden outputs `H` | `(N, 2)` | The number of sample rows changes. |
| Output weights `W_out` | `(2, 1)` | The shape stays the same. |
| Predictions `pred` | `(N, 1)` | There is one prediction per sample. |

When one call to `model(X)` processes 100 samples, there is <strong>one forward-pass call</strong> and <strong>100 prediction values</strong>. The number of calls does not tell us the number of numerical operations. How much a GPU can execute in parallel also depends on the operations and hardware.

## 5. Error, individual loss, and mean loss are different values

Let `i` identify a sample. `ŷ_i` is its prediction, and `y_i` is its target. Define <strong>the error `e_i` as prediction minus target</strong>. A positive error means the prediction is too high; a negative error means it is too low.

Define an individual sample's loss as half its squared error. Use `ℓ_i` for the individual loss and `L` for the batch's mean loss. The symbol `ℓ` is a lowercase ell.

<div class="nt-math">
$$
e_i=\hat y_i-y_i,\qquad
\ell_i=\frac12e_i^2,\qquad
L=\frac{\ell_1+\cdots+\ell_N}{N}
$$
</div>

Squared errors are nonnegative, so positive and negative errors cannot simply cancel when we add them. As the magnitude of an error grows, its square grows more quickly. Our goal is to reduce this evaluation value. Depending on the data and model, reaching a loss of 0 may be impossible.

| Sample | Prediction | Target | Error `prediction−target` | Individual loss `½×error²` |
| --- | ---: | ---: | ---: | ---: |
| A | 8 | 6 | 2 | 2 |
| B | 5 | 4 | 1 | 0.5 |

There are two samples in the batch, so the mean loss is `(2+0.5)/2=1.25`. This does not mean 1.25 minutes or an error rate of 1.25%. Because we squared errors measured in minutes, this evaluation value has units of minutes².

We divide by 2 twice, but for different reasons.

| Factor | Why it is present | What happens when batch size changes? |
| --- | --- | --- |
| `1/2` in the individual loss | Cancels the factor of 2 from differentiating the square, simplifying the calculation. | It remains `1/2`. |
| `1/N` in the mean | Divides the sum of sample losses by the number of samples. | It changes to match the actual sample count. |

<strong>MSE, or mean squared error</strong>, is the average squared error. Its usual definition does not include the initial `1/2`. Our `L` is therefore exactly <strong>half the MSE</strong>. Applying PyTorch's `nn.MSELoss()` to these predictions gives 2.5. Multiplying that result by 0.5 gives 1.25.[^mse]

PyTorch's default MSE reduction takes the <strong>mean over all output elements</strong>. This example has one output per sample, so the element count equals the sample count. If each sample has several outputs, those two counts need to be distinguished.

Defining the error as `target−prediction` would give the same squared loss. However, we would need to differentiate that entire expression consistently. We will continue to use `prediction−target` throughout.

## 6. At which “current position” is a gradient calculated?

First, distinguish a prediction graph from a training graph. In the simple prediction formula `ŷ=wx`, if the horizontal axis is input `x`, then `w` is the slope of the prediction. In training, however, we <strong>hold inputs and targets fixed and ask how changing parameters changes the loss</strong>. When the horizontal axis of a loss graph is `w1`, that axis is not the box count `x1`.

For this batch, the “current position” is the following combination of nine numbers.

<div class="nt-math">
$$
(w_1,w_2,w_3,w_4,v_1,v_2,b_1,b_2,b_3)
=(1,2,-1,1,2,-1,0,2,1)
$$
</div>

The current position is not the number of the layer being executed. It is <strong>the state of all model parameters used for this forward pass</strong>.

| Stage | Parameters | What is newly calculated or stored? |
| --- | --- | --- |
| Forward pass | Keep the initial values above | Intermediate values and predictions |
| Loss calculation | Keep the initial values above | Individual losses and mean loss |
| Backward pass | Keep the initial values above | Each parameter's gradient |
| Update | Change at this stage | New parameter values |

For a function with several inputs, calculating its rate of change with respect to one input is called <strong>partial differentiation</strong>. `∂L/∂w1` describes how the mean loss changes when only `w1` changes slightly, with the other parameters and this batch's inputs and targets held fixed. The symbol `∂` indicates a partial derivative.

The gradient for `w2` is calculated at <strong>the same parameter state</strong>. We do not update `w1` first and then differentiate with respect to `w2` at the new state. A different batch or a different loss definition can produce different gradients even when the parameter values are unchanged.

We calculate the gradients needed for this batch at the current parameter state, rather than calculating gradients in advance for every possible weight value.

## 7. A gradient of 5 does not mean “move the weight by 5”

Hold every other condition fixed and recalculate the results while changing only `w1`. This comparison helps explain rates of change; it is not yet a training update.

| `w1` | Prediction for A | Prediction for B | Mean loss `L` |
| ---: | ---: | ---: | ---: |
| 0.9 | 7.6 | 4.8 | 0.80 |
| 1.0 | 8.0 | 5.0 | 1.25 |
| 1.1 | 8.4 | 5.2 | 1.80 |

The symbol `Δ`, pronounced “delta,” indicates the difference between two values. <strong>A change is the new value minus the old value.</strong> The weight change is `Δw1`, and the loss change is `ΔL`.

Changing `w1` from 1 to 1.1 increases the weight by 0.1 and the loss by 0.55. The <strong>average rate of change</strong> over that interval is `0.55/0.1=5.5`. Changing it from 1 to 0.9 instead gives `(−0.45)/(−0.1)=4.5`. These are averages over different intervals, so they can differ.

As we reduce the size of the weight change, the rate approaches 5.

| Change added to `w1=1` | Loss change divided by weight change |
| ---: | ---: |
| −0.01 | 4.95 |
| −0.001 | 4.995 |
| +0.001 | 5.005 |
| +0.01 | 5.05 |

The derivative at the current point is <strong>the rate approached as the change becomes arbitrarily close to 0</strong>. We do not actually divide by 0. In this example, `∂L/∂w1=5`.

<figure class="nt-figure">
<a href="/neural-network-training/loss-slope-en.svg" target="_blank" rel="noopener"><img src="/neural-network-training/loss-slope-en.svg" alt="With all other conditions fixed, the current point on the w1 loss curve is (1,1.25), and the tangent slope is 5. The loss is 0.8 at w1=0.9 and 1.8 at w1=1.1." loading="lazy"></a>
<figcaption>Distinguish the current value on the curve from the slope at that point. This graph changes only <code>w1</code>; the other eight parameters stay fixed.</figcaption>
</figure>

For a small change, a gradient of 5 lets us predict <strong>`ΔL ≈ 5×Δw1`</strong>. The symbol `≈` means approximately equal. Increasing `w1` very slightly points toward increasing loss, while decreasing it very slightly points toward decreasing loss.

| Gradient | Meaning near the current position |
| --- | --- |
| Positive | A small increase in the parameter points toward increasing loss. |
| Negative | A small increase in the parameter points toward decreasing loss. |
| 0 | The current gradient alone cannot determine a decreasing direction for this parameter. It does not guarantee zero loss or a minimum. |
| Large absolute value | The loss is more sensitive to the same small parameter change. |

The gradient is neither the loss itself nor the remaining distance to a minimum. If we move far, the curve's slope can change, so an estimate based on the current rate may no longer be accurate.

Over the range shown, this graph follows `L=5(w1−0.5)²`. Holding everything else fixed, its minimum is therefore at `w1=0.5`. In this small example we already know the entire expression. For a general neural network, however, a current gradient alone does not reveal where the minimum is.

## 8. Why one mean loss produces a gradient for each prediction

The number 1.25 by itself cannot tell us the gradients of the weights. We need <strong>the calculations that produced that 1.25</strong>. Here, the predictions for A and B each produce an individual loss, and the mean of those two losses becomes `L`.

During calculations with automatic differentiation enabled, PyTorch records how operations connect and saves values needed for the backward pass. This network of operations is called a <strong>computation graph</strong>. Calculating the mean does not erase the relationships that produced it.[^autograd]

<div class="nt-math">
$$L=\frac{\ell_A+\ell_B}{2}$$
</div>

Changing A's loss alone changes `L`, and changing B's loss alone also changes `L`. We can therefore calculate each prediction's effect on `L` separately.

### Derive the prediction gradient from the formula

Keep the definitions of error and individual loss introduced earlier. Hold sample `i`'s target fixed and add a small change `δ` to its prediction. The symbol `δ` is also pronounced “delta.” Here, it means <strong>the change added to the prediction</strong>, rather than a hidden output or a weight.

Increasing the prediction by `δ` changes the error from `e_i` to `e_i+δ`. Let `Δℓ_i` be the resulting loss change.

<div class="nt-math">
$$
\begin{aligned}
\Delta\ell_i
&=\frac12(e_i+\delta)^2-\frac12e_i^2\\
&=\frac12\left(e_i^2+2e_i\delta+\delta^2-e_i^2\right)\\
&=e_i\delta+\frac12\delta^2
\end{aligned}
$$
</div>

The second line expands the product `(e_i+δ)(e_i+δ)`. The matching `e_i²` terms cancel, leaving the other two terms. Dividing the loss change by the prediction change `δ` gives `e_i+δ/2`. As `δ` approaches 0, the `δ/2` term vanishes, leaving `e_i` as the individual loss's derivative.

Each individual loss contributes a factor of `1/N` to the batch mean. The other samples' losses stay fixed when this prediction changes, so the mean loss's rate of change is the individual loss's rate divided by `N`.

<div class="nt-math">
$$\frac{\partial L}{\partial\hat y_i}=\frac{e_i}{N}=\frac{\hat y_i-y_i}{N}$$
</div>

The initial `1/2` has already canceled with the factor of 2 from expanding the square. <strong>The remaining `/N` comes from taking the batch mean.</strong>

Now substitute the values for A and B.

| Which prediction changes? | Gradient of the mean loss |
| --- | --- |
| A's prediction `ŷ_A` | `(8−6)/2=1` |
| B's prediction `ŷ_B` | `(5−4)/2=0.5` |

Changing only A's prediction from 8 to 8.01 changes the mean loss to 1.260025. The average rate is `(1.260025−1.25)/0.01=1.0025`, close to the derivative of 1. Changing only B's prediction from 5 to 5.01 gives a rate of 0.5025, close to the derivative of 0.5.

Here, we treated each prediction as an independent intermediate value when calculating its rate of change. In the actual model, changing a shared weight can change several predictions together. <strong>Prediction gradients are intermediate information used to calculate parameter gradients.</strong> The optimizer does not update the prediction array itself as though it were a trainable parameter.

## 9. Backpropagation multiplies rates along a path and adds contributions

The path of `w1`'s influence is `w1 → z1 → h1 → prediction → individual loss → mean loss`. The weight `w1` is not directly added to the final loss; its effect passes through intermediate calculations.

The <strong>chain rule</strong> connects the rates of change in these linked calculations. We multiply the rates along one path. If one calculation doubles a change and the next triples it, the final change is six times the original change. When the same value affects the result through several paths, we add each path's contribution.

For each current sample, four rates connect `w1` to the loss.

| Link | Why this is its rate of change | A | B |
| --- | --- | ---: | ---: |
| Prediction → mean loss | The previously derived `(prediction−target)/N` | 1 | 0.5 |
| `h1` → prediction | The output formula multiplies `h1` by `v1`. | 2 | 2 |
| `z1` → `h1` | ReLU passes positive inputs through unchanged. | 1 | 1 |
| `w1` → `z1` | The formula for `z1` multiplies `w1` by input `x1`. | 2 | 1 |

The forward pass evaluates the linked functions in order. The backward pass uses those relationships to <strong>connect rates of change with respect to the final loss, starting from the end</strong>.

<div class="nt-math">
$$
\begin{aligned}
\text{A's contribution}&=1\times2\times1\times2=4\\
\text{B's contribution}&=0.5\times2\times1\times1=1\\
\frac{\partial L}{\partial w_1}&=4+1=5
\end{aligned}
$$
</div>

<figure class="nt-figure">
<a href="/neural-network-training/backward-en.svg" target="_blank" rel="noopener"><img src="/neural-network-training/backward-en.svg" alt="The mean-loss gradients for the predictions are 1 for A and 0.5 for B. Multiplying each by output weight 2, ReLU rate 1, and input x1 gives contributions of 4 and 1 to w1. Adding them gives a gradient of 5 for the shared weight." loading="lazy"></a>
<figcaption>Add both paths' contributions to the weight shared by A and B. The prediction gradients 1 and 0.5 already include the mean's <code>÷2</code> factor.</figcaption>
</figure>

We therefore do not divide the final 5 by 2 again. Alternatively, if we first calculate <strong>gradients of the individual losses</strong>, A gives `2×2×1×2=8`, and B gives `1×2×1×1=2`. We then average them as `(8+2)/2=5`. Applying the mean factor at the beginning or at the end gives the same result, as long as we apply it once.

### What can we reuse by calculating backward?

The loss gradient of 1 with respect to A's prediction can be reused when differentiating the output weights `v1` and `v2` and the output bias `b3`. We can also propagate it toward the earlier hidden outputs. Starting at the final loss allows these intermediate gradients to be reused across several parameter calculations.[^autograd]

What travels toward earlier layers is <strong>the gradient of the loss with respect to intermediate outputs</strong>. It is not simply the `.grad` stored for a later layer's weights. Backpropagation also does not copy the loss value 1.25 backward or reconstruct the original inputs.

Differentiation can also be organized in the forward direction. For a calculation such as this one, however, where many parameters lead to a single loss, sharing intermediate gradients in the reverse direction is efficient. Backpropagation does not test weights by changing them one at a time. It uses each operation's differentiation rule and values obtained during the forward pass.

### Why not compare each hidden layer directly with the target?

The target of 6 minutes belongs to A's <strong>final output</strong>. It is not a target for each value in the hidden output `[4, 1]`. With the current output formula `ŷ=2h1−h2+1`, we could lower the prediction by lowering `h1` or by raising `h2`. The size of a hidden value alone does not tell us which direction to change it.

This is why ordinary training first calculates the final loss and then works backward to find how each hidden value affected it. Some models assign separate losses to intermediate layers, but this training procedure does not include such losses.

## 10. Matrix backpropagation also multiplies and adds sample contributions

Let names beginning with `g_`, such as `g_pred`, mean <strong>the gradient of the mean loss with respect to that value</strong>. For example, `g_H` describes how the mean loss changes with the hidden outputs `H`. The `.T` operation is a <strong>transpose</strong>, which exchanges rows and columns.

First, collect the prediction gradients into an array.

<div class="nt-math">
$$g_{\mathrm{pred}}=\begin{bmatrix}1\\0.5\end{bmatrix}$$
</div>

The output weight `v1` multiplies each sample's `h1`. A therefore contributes `4×1`, and B contributes `3×0.5`, for a total of 5.5. For `v2`, the same reasoning gives `1×1+2×0.5=2`. The expression `H.T @ g_pred` calculates both sums together.

<div class="nt-math">
$$
g_{W_{\mathrm{out}}}=H^Tg_{\mathrm{pred}}
=\begin{bmatrix}4&3\\1&2\end{bmatrix}
\begin{bmatrix}1\\0.5\end{bmatrix}
=\begin{bmatrix}5.5\\2\end{bmatrix}
$$
</div>

The gradients propagated toward the hidden outputs are `g_pred @ W_out.T`. Each prediction gradient is multiplied by the output weight attached to the corresponding hidden value.

<div class="nt-math">
$$
g_H=\begin{bmatrix}1\\0.5\end{bmatrix}
\begin{bmatrix}2&-1\end{bmatrix}
=\begin{bmatrix}2&-1\\1&-0.5\end{bmatrix}
$$
</div>

All current `Z` values are positive, so ReLU has a rate of change of 1 at each entry. The gradients `g_Z` propagated to the weighted sums are therefore unchanged. At a negative input, ReLU keeps producing 0, so we use a rate of 0 for that entry. At exactly 0, there is no single ordinary derivative; this code and PyTorch use 0 there.[^relu]

For hidden weights, we again multiply inputs by gradients and add the sample contributions.

<div class="nt-math">
$$
g_{W_{\mathrm{hidden}}}=X^Tg_Z
=\begin{bmatrix}2&1\\1&1\end{bmatrix}
\begin{bmatrix}2&-1\\1&-0.5\end{bmatrix}
=\begin{bmatrix}5&-2.5\\3&-1.5\end{bmatrix}
$$
</div>

The upper-left entry is `2×2+1×1=5`, matching the gradient of `w1` calculated earlier. The upper-right entry is the gradient of `w3`, and the lower entries are the gradients of `w2` and `w4`. <strong>Each gradient occupies the same position as its corresponding weight.</strong>

A bias is added directly to each sample's weighted sum. Increasing the bias by 1 therefore increases that weighted sum by 1. We add the sample contributions without multiplying by another input.

| Gradient | Calculation | Result |
| --- | --- | --- |
| Output bias | `g_pred.sum(dim=0)` | `[1.5]` |
| Hidden biases | `g_Z.sum(dim=0)` | `[3, −1.5]` |

`sum` is a tensor <strong>method</strong> that calculates a sum. Here, `dim=0` specifies the first dimension, where samples are arranged. Starting with `(N, 2)`, summing the sample contributions for each neuron removes the sample dimension and leaves `(2,)`. Starting with `(N, 1)` leaves `(1,)`. In both cases, we calculate <strong>the gradient of a shared bias</strong>, rather than creating a separate bias for each sample.

The manual matrix calculations can be collected in the following order. The `*` operator multiplies matching elements; it does not also sum products as `@` does. `(Z > 0)` is an array of true or false values indicating which entries are positive. In multiplication, those values act as 1 and 0.

```python
# pred, y: (N, 1). Differentiate this example's mean loss with one output per sample.
g_pred = (pred - y) / N
g_W_out = H.T @ g_pred              # (2, N) @ (N, 1) → (2, 1)
g_b_out = g_pred.sum(dim=0)         # (N, 1) → (1,): sum sample contributions
g_H = g_pred @ W_out.T             # (N, 1) @ (1, 2) → (N, 2)
g_Z = g_H * (Z > 0)                # (N, 2): apply ReLU's rate to each entry
g_W_hidden = X.T @ g_Z             # (2, N) @ (N, 2) → (2, 2)
g_b_hidden = g_Z.sum(dim=0)        # (N, 2) → (2,): sum sample contributions
```

This snippet shows how the calculations connect. In the complete code, the output weights stored in the model are accessed as `model.W_out`. This is the same matrix called `W_out` above. The complete runnable code, including the input data and parameter definitions, is collected near the end of the article.

## 11. The optimizer turns gradients into actual parameter changes

<strong>Gradient descent</strong> updates parameters in the direction opposite to their current gradients. We use the basic SGD rule without momentum or weight decay. The name SGD is used when training with gradients from samples or minibatches rather than the entire dataset. In this calculation, we use the average gradient from A and B.[^sgd]

PyTorch's SGD optimizer applies the same basic update formula regardless of batch size. In this small example, all the prepared samples—A and B—form one batch.

Let `w` be one parameter and `g` its current gradient. The <strong>learning rate</strong> is a positive value that controls the size of an update. We write it as `η`, pronounced “eta,” in formulas and `lr` in code. The symbol `Δw` is the actual change added to `w`.

<div class="nt-math">
$$\Delta w=-\eta g,\qquad w_{\mathrm{new}}=w+\Delta w=w-\eta g$$
</div>

Choose `η=0.01`. The current value of `w1` is 1 and its gradient is 5, so the change is `−0.01×5=−0.05`, and the new value is `1−0.05=0.95`. The gradient of `w3` is −2.5, so its change is `+0.025`, increasing its value from −1 to −0.975.

| Parameter | Current value | Gradient of the mean loss | Value after an update with `0.01` |
| --- | ---: | ---: | ---: |
| `w1` | 1 | 5 | 0.95 |
| `w2` | 2 | 3 | 1.97 |
| `w3` | −1 | −2.5 | −0.975 |
| `w4` | 1 | −1.5 | 1.015 |
| `v1` | 2 | 5.5 | 1.945 |
| `v2` | −1 | 2 | −1.02 |
| `b1` | 0 | 3 | −0.03 |
| `b2` | 2 | −1.5 | 2.015 |
| `b3` | 1 | 1.5 | 0.985 |

All nine gradients were calculated at <strong>the same state before the update</strong>. The update applies each result to its corresponding parameter.

Predicting A and B again with the new parameters gives <strong>7.3522 minutes and 4.50995 minutes</strong>, respectively. The new mean loss is:

<div class="nt-math">
$$L_{\mathrm{new}}=\frac{\tfrac12(7.3522-6)^2+\tfrac12(4.50995-4)^2}{2}
=0.522123460625$$
</div>

This is smaller than the original 1.25. Unlike the earlier experiment that changed only `w1`, this result comes from <strong>updating all nine parameters</strong>. It also only shows that the loss decreased on these two samples; it does not establish that predictions will be accurate for previously unseen packing jobs.

After `optimizer.step()`, the existing `loss` tensor still contains the 1.25 calculated earlier. The stored `.grad` values also describe the previous position. <strong>To obtain new loss and gradient values, we must perform a new forward pass, loss calculation, and backward pass.</strong>

## 12. Why use gradient magnitude if it is not a distance to a minimum?

The gradient's sign tells us which direction locally decreases the loss. Its magnitude tells us <strong>how much the loss changes for the same small movement</strong>. That magnitude alone, however, does not determine the best distance to move.

To see this, compare two separate losses with one parameter, `u`. This does not change our neural network. It is a small calculation specifically for distinguishing a gradient from a distance.

| Loss formula | Current `u` | Minimum location | Change needed to reach the minimum | Current gradient |
| --- | ---: | ---: | ---: | ---: |
| `(u−3)²` | 1 | 3 | +2 | −4 |
| `9(u−3)²` | 1 | 3 | +2 | −36 |

The second loss is nine times the first, so its gradient is also nine times as large. Its minimum remains in the same place. <strong>A gradient nine times as large does not imply a distance nine times as far from the minimum.</strong> Scaling the loss also changes the update produced by the same learning rate, so the learning rate must be considered together with the loss definition.

### Choosing a decreasing direction and choosing a movement size are different tasks

If the current gradient is `g`, the loss change for a sufficiently small movement is approximately `ΔL ≈ gΔw`. Substituting gradient descent's `Δw=−ηg` gives `ΔL ≈ −ηg²`. This estimate is negative when `η` is positive and `g` is nonzero.

This calculation explains <strong>why the chosen direction decreases the loss locally</strong>. It does not prove why we must choose this particular movement size or why it would be best in every situation. A large movement can make the original approximation inaccurate.

### One criterion that produces the basic gradient descent update

We can choose a criterion that seeks a small predicted loss change, `gΔw`, while also penalizing movements that are too large. Add a squared movement cost, `(Δw)²/(2η)`, and call the resulting expression <strong>the movement-selection criterion `Q`</strong>.[^quadratic]

<div class="nt-math">
$$Q(\Delta w)=g\Delta w+\frac{(\Delta w)^2}{2\eta}$$
</div>

`Q` is not the actual loss `L`. Its first term estimates the loss change from the current gradient. Its second term is a chosen cost that limits large movements. Using a squared cost is itself a design choice, not the only possible choice.

Keeping the current `g` and `η` fixed and differentiating with respect to the movement `Δw` gives `g+Δw/η`. Differentiating the square produces a factor of 2, so the derivative of the second term is `Δw/η`. Now, to locate the minimum of `Q`, consider <strong>the movement that makes this derivative zero</strong>.

<div class="nt-math">
$$
\frac{dQ}{d(\Delta w)}=g+\frac{\Delta w}{\eta}=0
\quad\Longrightarrow\quad\Delta w=-\eta g
$$
</div>

We can also verify that this is the minimum of `Q` by rewriting the expression.

<div class="nt-math">
$$Q(\Delta w)=\frac{(\Delta w+\eta g)^2}{2\eta}-\frac{\eta g^2}{2}$$
</div>

Because `η>0`, the first term is nonnegative. It becomes 0 at `Δw=−ηg`, while the second term does not depend on the movement. That movement therefore minimizes `Q`.

<strong>With the same movement cost, a larger gradient magnitude predicts more loss reduction from a small movement, so this criterion chooses a larger movement.</strong> This is one mathematical reason for using the gradient's magnitude. It does not mean we have discovered the distance to the actual loss's minimum.

<details class="nt-details">
<summary>What conditions are needed to guarantee that the actual loss decreases?</summary>

<div class="nt-detail-body">

Consider the case where <strong>all other parameters stay fixed, only `w` changes, and the loss is differentiable throughout the movement interval</strong>. The actual loss must not bend too rapidly away from the current approximation. One sufficient condition is that, for any two positions `r` and `s` in the movement interval, the difference between their gradients is no greater than `|r−s|/η`. Here, `L′(r)` is the gradient at position `r`, and vertical bars indicate absolute value.

<div class="nt-math">
$$|L'(r)-L'(s)|\le\frac{|r-s|}{\eta}$$
</div>

When the gradient's change over the interval is limited this way, we can use the following upper bound. “Upper bound” means that the actual new loss is no greater than the expression on the right. Guaranteeing a decrease when several parameters change together requires a separate check of conditions for that whole movement.

<div class="nt-math">
$$L(w+\Delta w)\le L(w)+g\Delta w+\frac{(\Delta w)^2}{2\eta}$$
</div>

Substituting `Δw=−ηg` shows that the new loss is at most `L(w)−ηg²/2`. If `g≠0`, the actual loss decreases. This condition does not automatically hold for every loss and learning rate. A large current gradient therefore does not, by itself, make a large update safe.[^quadratic]

</div>
</details>

### Using only the gradient's sign is also possible

The sign function `sign(g)` returns 1 for a positive value, −1 for a negative value, and 0 for zero. If we separately choose a positive movement size `step_size`, we can update with `new w = w−step_size×sign(g)`. Methods such as signSGD have indeed been studied using gradient signs.[^sign]

A fixed movement size can repeatedly cross back and forth around a minimum, so it may need adjustment. A method using the gradient directly also needs its learning rate adjusted. <strong>The optimizer determines how gradient information becomes an update.</strong> The basic SGD rule is one possible method, not the only one.

| Optimizer | Information used to choose updates |
| --- | --- |
| Basic SGD | Current gradient and learning rate |
| SGD with momentum | An accumulated direction that retains a proportion of past gradients |
| RMSprop | Recent averages of squared gradients to adjust update sizes |
| Adam | Recent gradient averages and squared-gradient averages |
| AdamW | A separate parameter-shrinking operation alongside Adam's update |

Here, a “recent average” is updated by mixing the previous record and the new value in fixed proportions. AdamW's separate shrinking operation is called <strong>weight decay</strong>. These algorithms also use information from backpropagation, but their update formulas differ from basic SGD.[^optimizers]

## 13. What changes when only the learning rate changes?

To isolate the effect of the learning rate, this section uses the separate small loss <strong>`J(u)=(u−3)²`</strong>. The parameter being adjusted is `u`. This is a different function from the neural network's loss `L`, and it does not have a leading `1/2`. Its gradient is therefore `J′(u)=2(u−3)`.

In all three experiments, keep the initial value `u=0`, the loss formula, the gradient formula, and the number of updates—three—the same. Change only the learning rate: 0.05, 0.25, or 1.1. Recalculate the gradient at the current position each time and update with `u←u−η×2(u−3)`. The arrow `←` means replacing the value on the left with the result on the right.

| Learning rate | Initial `u` | After update 1 | After update 2 | After update 3 |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0 | 0.3 | 0.57 | 0.813 |
| 0.25 | 0 | 1.5 | 2.25 | 2.625 |
| 1.1 | 0 | 6.6 | −1.32 | 8.184 |

For example, with a learning rate of 0.25, the first gradient is `2(0−3)=−6`. The new value is `0−0.25×(−6)=1.5`. For the second update, we do not reuse −6. We calculate the gradient at the new position as `2(1.5−3)=−3`.

| Learning rate | Initial loss | Loss after update 1 | Loss after update 2 | Loss after update 3 |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 9 | 7.29 | 5.9049 | 4.782969 |
| 0.25 | 9 | 2.25 | 0.5625 | 0.140625 |
| 1.1 | 9 | 12.96 | 18.6624 | 26.873856 |

<figure class="nt-figure">
<a href="/neural-network-training/learning-rate-en.svg" target="_blank" rel="noopener"><img src="/neural-network-training/learning-rate-en.svg" alt="Starting at u=0, three updates minimize J(u)=(u−3) squared. The loss falls slowly with learning rate 0.05, falls faster with 0.25, and rises with 1.1. Four points include the initial state." loading="lazy"></a>
<figcaption>There are three updates and four records because the initial state is also recorded. Plotting displays values already recorded; it does not perform additional training.</figcaption>
</figure>

For this function, we can calculate exactly why the results differ. Rearrange the difference between the new position and the minimum at 3.

<div class="nt-math">
$$
\begin{aligned}
u_{\mathrm{new}}-3
&=(u-3)-2\eta(u-3)\\
&=(1-2\eta)(u-3)
\end{aligned}
$$
</div>

The signed difference from the minimum, `u−3`, is multiplied by `1−2η` at each update. With a learning rate of 0.05, the factor is 0.9, so the position approaches the minimum slowly. With 0.25, the factor is 0.5. With 1.1, the factor is −1.2: the position crosses to the other side while its distance grows by a factor of 1.2 each time.

Approaching a target value through repeated steps is called <strong>convergence</strong>. When the distance from the minimum and the loss keep growing, as in this example, we call it <strong>divergence</strong>. Crossing past the minimum in one movement is <strong>overshooting</strong>. A step can cross the minimum while still getting closer to it, so overshooting alone does not mean divergence.

For this function, when the starting point is not the minimum, `0<η<1` gives convergence, and `η=0.5` reaches the minimum in one update. At `η=1`, the position alternates between opposite sides at the same distance, and for `η>1` it diverges. <strong>The value 0.25 reduced the loss fastest among the three values in the table; that does not make it the best of all possible learning rates.</strong> These boundaries also change with a different loss formula.

## 14. Count gradient results separately from the amount of computation

Let a fully connected layer have `D` inputs and `K` output neurons. Each output neuron has one weight per input and one bias. With biases enabled, the number of values is:

<div class="nt-math">
$$\text{Weights}=D\times K,\quad
\text{Biases}=K,\quad
\text{Parameters}=D\times K+K$$
</div>

The model calculated earlier has `2×2+2=6` parameters in its hidden layer and `2×1+1=3` in its output layer, for a total of 9. There is also one gradient for each of those nine values.

Now consider <strong>a separate model used only for counting</strong>, with the structure `2 → 2 → 2 → 2 → 1`. It has four trainable fully connected layers, each with biases. We do not assign the predictions or numerical gradients from our earlier two-layer model to this different model.

| Fully connected layer | Inputs → outputs | Weights | Biases | Total |
| --- | --- | ---: | ---: | ---: |
| 1 | 2 → 2 | 4 | 2 | 6 |
| 2 | 2 → 2 | 4 | 2 | 6 |
| 3 | 2 → 2 | 4 | 2 | 6 |
| 4 | 2 → 1 | 2 | 1 | 3 |
| Entire model | | 14 | 7 | 21 |

<strong>Having four layers does not mean there are four gradients.</strong> This structure has 21 trainable values, so it has 21 parameter-gradient values. ReLU has no trainable weights or biases.

### The number of tensors differs from the number of values inside them

For this article, a <strong>tensor</strong> can be understood as an array holding numbers. PyTorch's `nn.Linear` is a class for creating fully connected layers. It stores weights with shape `(number of outputs, number of inputs)` and calculates `X @ weight.T + bias`.[^linear]

Our earlier hand calculation stored weights as `(number of inputs, number of outputs)`. The storage orientations differ, but transposing gives the same calculation. The main model in the complete code directly uses the hand calculation's storage orientation, while the separate four-layer counting model uses `nn.Linear`.

| Storage in the four-layer model | `weight` and `weight.grad` | `bias` and `bias.grad` |
| --- | --- | --- |
| First layer | `(2, 2)` | `(2,)` |
| Second layer | `(2, 2)` | `(2,)` |
| Third layer | `(2, 2)` | `(2,)` |
| Output layer | `(1, 2)` | `(1,)` |

There are four weight tensors and four bias tensors: <strong>8 tensors containing 21 numbers</strong>. The call `loss.backward()` does not return them as one array. It returns `None`, and each parameter's gradients are stored in that parameter's `.grad`.[^backward]

For trainable parameters connected to the loss as in this example, the parameter and its `.grad` have the same shape. Some gradient values may be 0. A parameter not connected to the loss may have `.grad` equal to `None`, so distinguish a numerical zero from the absence of a stored gradient.

### Why 100,000 samples still produce 21 gradient results

Each sample contributes to <strong>the same 21 parameters</strong>. We combine those contributions into one result per parameter. When the loss is the mean of individual sample losses and sample computations are independent, as in this example, the batch gradient is the mean of the individual sample-loss gradients.

| Samples in one batch | Final predictions | Final parameter-gradient values | Weight multiplications in the forward formulas |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 21 | 14 |
| 2 | 2 | 21 | 28 |
| 100,000 | 100,000 | 21 | 1,400,000 |

The last column counts only the multiplications that apply the four-layer model's 14 weights to each sample. It excludes additions, biases, activation functions, backpropagation, and updates. It does not count hardware instructions or measure execution time.

<strong>Having 21 final results does not mean performing only 21 operations in total.</strong> Implementations combine contributions through matrix operations, so they do not need to store a separate array of parameter gradients for every sample. Contributions from different samples can also cancel if they point in opposite directions. Reducing the mean loss does not guarantee that every individual sample's loss decreases.

## 15. What do batches, steps, and epochs count?

The <strong>batch size</strong> is the number of samples in one group. In this article, a <strong>step</strong> means performing one backward pass and one update with one batch. An <strong>epoch</strong> is one pass through the entire training dataset.[^optimization]

With 100,000 samples and 100 samples per batch, one epoch processes 1,000 batches. Each batch goes through forward pass → loss → backward pass → update.

| What is counted in one epoch? | Count |
| --- | ---: |
| Samples used | 100,000 |
| Batches | 1,000 |
| Batch forward-pass calls | 1,000 |
| Backward-pass calls | 1,000 |
| Optimizer updates | 1,000 |

For the 21-parameter model, the same 21 numbers are updated 1,000 times. Counting all gradient values produced across the steps gives `21×1,000=21,000`. This does not mean that the model grows to 21,000 parameters or that all these gradients remain stored.

If the dataset size is not divisible by the batch size, the final batch can be smaller. For example, 1,003 samples with a batch size of 100 produce ten full batches followed by a final batch of three samples. In PyTorch's `DataLoader`, `drop_last=False` uses that final batch, while `True` excludes the incomplete batch.[^data]

When calculating a mean manually, use <strong>the number of samples actually present</strong>, instead of always inserting 100 for the final batch. With <strong>gradient accumulation</strong>, several batches' gradients are collected before a later update, so the number of backward passes can differ from the number of updates. The table above assumes one update per batch.

Finishing an epoch means using the dataset once. It does not mean that every prediction is correct or that training is complete.

## 16. PyTorch executes the training stages separately

<strong>PyTorch</strong> is a Python library that provides tensor operations, automatic differentiation, and neural network training tools. Automatic differentiation applies the operation-by-operation derivative rules we connected manually earlier. The following five lines are the core sequence for one batch.[^optimization]

```python
# Clear previous gradients. This does not set weights and biases to zero.
optimizer.zero_grad(set_to_none=True)
pred = model(batch_X)                         # (N, 2) → (N, 1): forward pass
loss = 0.5 * (pred - batch_y).square().mean()  # (N, 1) → (): mean loss
loss.backward()                              # Store current gradients in parameter .grad
optimizer.step()                             # Update parameters using those gradients
```

The shape `()` describes a <strong>zero-dimensional tensor containing one number</strong>. It does not mean that `loss` is a Python float. Being a tensor allows it to retain its calculation connections and support `backward()`.

### Distinguish creating a model from running training

The complete code defines `StudyMLP`, a <strong>custom model class</strong> based on `nn.Module`. Its `__init__` prepares parameters, while `forward` defines how inputs become predictions. Writing `model = StudyMLP()` creates a model object; the forward pass runs when we call `model(X)`.

| Feature used in the code | Kind | Role and result |
| --- | --- | --- |
| `torch.tensor(...)` | Function | Creates a tensor from input numbers. Our `X` has shape `(2, 2)`. |
| `nn.Parameter(...)` | Class | Assigning this object to an attribute of an `nn.Module` registers it as a trainable parameter. It tracks gradients by default. |
| `model.parameters()` | Method | Provides the model's registered parameter tensors in sequence. |
| `torch.optim.SGD(...)` | Class | Creates an optimizer for the supplied parameters and learning rate. |
| `square()` | Tensor method | Squares each element and preserves the shape. |
| `mean()` | Tensor method | Here, averages all elements and returns a zero-dimensional tensor. |
| `numel()` | Tensor method | Returns the number of values in a tensor as a Python integer. |

The calculation checks in this article use a CPU and `torch.float64`. The type `float64` represents 64-bit floating-point numbers; we choose it to reduce numerical errors when comparing these small calculations. This code does not propose the fastest execution settings for a large model.

### Why clear gradients?

PyTorch's `backward()` adds its calculated gradients to existing `.grad` values. For independent updates on each batch, we must clear the previous batch's gradients. Calling `zero_grad(set_to_none=True)` does so by setting `.grad` to `None`. The call `optimizer.step()` does not perform this clearing for us.[^backward][^zero]

Adding A and B's contributions within one batch is necessary for calculating the gradient of the mean loss. This is different from <strong>leaving a previous batch's `.grad` in place so that the next backward pass adds to it</strong>. When deliberately accumulating gradients across batches, the loss scaling and update timing must be chosen accordingly.

### Check the shapes of predictions and targets first

Our predictions and targets must both have shape `(N, 1)`. If predictions have shape `(2, 1)` while targets have shape `(2,)`, subtraction can broadcast them to `(2, 2)`. That no longer compares each job's prediction only with its own target.[^broadcasting]

Before calculating the loss, use `assert pred.shape == y.shape` to check the shapes. An `assert` statement raises an error when its condition is false. It is useful for checking learning examples, but Python can omit assertions under optimization options, so it cannot replace every required production check.[^assert]

## 17. Check that manual calculations, automatic differentiation, and numerical differences agree

The complete script uses exactly the initial parameters calculated earlier. It compares the hand-calculated values with automatic differentiation results, rather than treating arbitrary numbers as correct answers.

| Check | What it examines |
| --- | --- |
| Manual matrix backpropagation | Results from explicitly connecting rates of change, such as `X.T @ g_Z` |
| PyTorch automatic differentiation | Results stored in each parameter's `.grad` by `loss.backward()` |
| Numerical differentiation | Differences between losses after moving a parameter slightly in both directions |

<strong>Numerical differentiation</strong> approximates a derivative using function values at nearby positions. The method here, which uses positions on both sides, is called a <strong>central difference</strong>. Let `loss_at(w)` be the loss with all other conditions fixed, and let `ε`, pronounced “epsilon,” be a small positive comparison distance.

<div class="nt-math">
$$\frac{\mathrm{loss\_at}(w+\varepsilon)-\mathrm{loss\_at}(w-\varepsilon)}{2\varepsilon}$$
</div>

The denominator is the difference between the two positions: `(w+ε)−(w−ε)=2ε`. The code restores each parameter to its original value after checking it. This is a separate calculation for verifying derivatives. Ordinary training does not change each parameter individually as a replacement for automatic differentiation. If `ε` is too large, the local derivative approximation can be poor. If it is too small, floating-point rounding errors can become more significant.

Computers store real-number approximations in a finite number of bits, so a result may contain a tiny difference such as `0.5221234606250005`. The function `torch.testing.assert_close` checks that two tensors agree within a tolerance. The function `math.isclose` returns true or false for whether two numbers are close. Instead of repeatedly rounding intermediate calculations, <strong>use tolerances when comparing results or adjust the displayed precision only when printing them</strong>.[^close]

Also distinguish the methods used to inspect intermediate values.

| Feature | What it makes possible |
| --- | --- |
| `retain_grad()` | Retains the `.grad` of an intermediate tensor after backpropagation, when it would not normally be stored. |
| `detach()` | Produces a tensor detached from differentiation connections. It does not itself create an independent copy of the values. |
| `clone()` | Makes a separate copy with the same values. Use `detach().clone()` to preserve previous parameters. |
| `item()` | Extracts a Python number from a tensor with one element, for printing or recording. |

The result of `loss.item()` is a Python number, so it cannot receive a `backward()` call. Differentiate using the original `loss` tensor. It is unnecessary to retain gradients for every intermediate value; here, `retain_grad()` is used to inspect the calculation flow.[^retain]

### Complete code and expected output

The code below was run and verified on a CPU with Python 3.12 and PyTorch 2.14.1. The file includes the main model's forward pass and gradients, updates to all nine parameters, central-difference checks, the separate four-layer model's 21 parameters, a batch training loop, and the learning-rate experiment. It runs in Python with PyTorch installed and does not require a GPU.

<strong>[Download the complete Python script](/neural-network-training/examples-en.py)</strong>

| Item calculated and checked by the code | Expected value |
| --- | --- |
| Initial predictions | `[[8], [5]]` |
| Initial mean loss | `1.25` |
| Gradient of `w1` | `5` |
| Gradient of `W_hidden` | `[[5, −2.5], [3, −1.5]]` |
| `w1` after one basic SGD update | `0.95` |
| Predictions after updating all nine parameters | `[[7.3522], [4.50995]]` |
| Mean loss after the update | Approximately `0.522123460625` |
| Separate four-layer model | 8 parameter tensors containing 21 numbers |

<details class="nt-details">
<summary>Expand the complete runnable Python code</summary>

```python
"""Inspect neural-network training: forward pass, loss, gradients, and updates.

PyTorch is the only external package required. The script runs on the CPU.
Fictional data: [small-box count, large-box count] -> packing time in minutes.
The target for A=[2, 1] is 6 minutes; the target for B=[1, 1] is 4 minutes.
These two examples illustrate calculations; they do not establish predictive quality.
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# float64 stores 64-bit floating-point numbers for these small numerical checks.
DTYPE = torch.float64


class StudyMLP(nn.Module):
    """Two input features -> two hidden neurons -> one output neuron."""

    def __init__(self):
        super().__init__()  # Initialize nn.Module's parameter management.
        # Parameter registers a trainable tensor (an array of numbers) in the model.
        # Our matrices use rows for input features and columns for output neurons.
        # W_hidden = [[w1, w3], [w2, w4]]: (2, 2).
        self.W_hidden = nn.Parameter(
            torch.tensor([[1.0, -1.0], [2.0, 1.0]], dtype=DTYPE)
        )
        self.b_hidden = nn.Parameter(torch.tensor([0.0, 2.0], dtype=DTYPE))
        self.W_out = nn.Parameter(torch.tensor([[2.0], [-1.0]], dtype=DTYPE))
        self.b_out = nn.Parameter(torch.tensor([1.0], dtype=DTYPE))

    def forward(self, X):
        # @ is matrix multiplication. Each sample receives the same neuron biases.
        Z = X @ self.W_hidden + self.b_hidden  # (N, 2).
        H = torch.relu(Z)                     # Elementwise max(0, z), (N, 2).
        return H @ self.W_out + self.b_out    # (N, 1).


def check_central_differences(model, X, y, expected_grads):
    """Change one parameter at a time and compare numerical and analytic gradients."""
    eps = 1e-5  # The two comparison positions are current value +/-0.00001.
    checked = 0
    # no_grad disables new differentiation records, not the numerical calculations.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            numeric = torch.empty_like(parameter)  # Result array with the same shape.
            # view(-1) views the same storage as one dimension; it does not copy here.
            flat_parameter = parameter.view(-1)
            flat_numeric = numeric.view(-1)
            for index in range(parameter.numel()):
                original = flat_parameter[index].item()
                try:
                    flat_parameter[index] = original + eps
                    plus = (0.5 * (model(X) - y).square()).mean().item()
                    flat_parameter[index] = original - eps
                    minus = (0.5 * (model(X) - y).square()).mean().item()
                finally:
                    flat_parameter[index] = original  # Restore after each comparison.
                # The denominator is the distance between positions: (w+eps)-(w-eps).
                flat_numeric[index] = (plus - minus) / (2 * eps)
                checked += 1
            # assert_close checks agreement within a tolerance and raises if it fails.
            # rtol sets relative tolerance; atol sets absolute tolerance.
            torch.testing.assert_close(
                numeric, expected_grads[name], rtol=1e-6, atol=1e-8
            )
    assert checked == 9
    print("Parameters checked by central differences and backpropagation:", checked)


def inspect_one_update(X, y):
    """Compare manual gradients, autograd, and one SGD update from the same state."""
    model = StudyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # zero_grad clears previous .grad values; it does not change the parameters.
    optimizer.zero_grad(set_to_none=True)
    # detach removes the differentiation connection; clone copies to separate storage.
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    # 1. Expand forward's operations so we can inspect intermediate values.
    Z = X @ model.W_hidden + model.b_hidden  # [[4, 1], [3, 2]].
    H = torch.relu(Z)                       # All entries are positive, so H equals Z.
    pred = H @ model.W_out + model.b_out    # [[8], [5]].
    # Intermediate .grad values are normally not retained. Keep them for inspection.
    for intermediate in (Z, H, pred):
        intermediate.retain_grad()

    # 2. square returns elementwise squares; mean averages all tensor elements.
    assert pred.shape == y.shape == (2, 1)
    per_loss = 0.5 * (pred - y).square()  # [[2], [0.5]], shape=(2, 1).
    loss = per_loss.mean()               # 1.25, shape=(): a single number.
    print("\n[Forward pass and loss]")
    for name, value in [("Z", Z), ("H", H), ("Prediction", pred), ("Per-sample loss", per_loss)]:
        print(name, value.detach().tolist())  # tolist converts a tensor to Python lists.
    print("Mean loss:", loss.item())  # item extracts a one-element tensor as a Python number.
    torch.testing.assert_close(pred, torch.tensor([[8.0], [5.0]], dtype=DTYPE))
    assert loss.item() == 1.25

    # 3. Manual backpropagation. Each g_ is the mean loss's gradient for that value.
    N = X.shape[0]  # Actual number of samples in this batch: 2.
    with torch.no_grad():
        g_pred = (pred - y) / N             # (2, 1): [[1], [0.5]].
        g_W_out = H.T @ g_pred              # (2, 1): [[5.5], [2]].
        g_b_out = g_pred.sum(dim=0)         # (2, 1) -> (1,): [1.5].
        g_H = g_pred @ model.W_out.T        # (2, 2): gradient passed to the hidden layer.
        g_Z = g_H * (Z > 0)                 # ReLU uses derivative 1 above zero, else 0.
        g_W_hidden = X.T @ g_Z              # (2, 2): sum the sample contributions.
        g_b_hidden = g_Z.sum(dim=0)          # (2, 2) -> (2,): [3, -1.5].
    # .T swaps rows and columns. sum(dim=0) sums samples and removes that dimension.
    # g_pred already includes /N; do not divide by N again later.
    # ReLU has no ordinary derivative at zero; PyTorch uses zero at that point.
    manual = {"W_hidden": g_W_hidden, "b_hidden": g_b_hidden,
              "W_out": g_W_out, "b_out": g_b_out}
    expected = {"W_hidden": [[5.0, -2.5], [3.0, -1.5]],
                "b_hidden": [3.0, -1.5], "W_out": [[5.5], [2.0]],
                "b_out": [1.5]}

    # 4. Automatic backpropagation returns None and accumulates results into .grad.
    result = loss.backward()
    assert result is None
    torch.testing.assert_close(pred.grad, g_pred)
    torch.testing.assert_close(H.grad, g_H)
    torch.testing.assert_close(Z.grad, g_Z)
    print("\n[Gradients for each parameter]")
    for name, parameter in model.named_parameters():
        # Also verify that backward did not change the parameter values.
        torch.testing.assert_close(parameter.detach(), before[name])
        torch.testing.assert_close(parameter.grad, manual[name])
        torch.testing.assert_close(parameter.grad, torch.tensor(expected[name], dtype=DTYPE))
        print(name, "gradient:", parameter.grad.tolist(), "shape:", tuple(parameter.grad.shape))
    # numel counts numerical elements: these four tensors contain nine parameters.
    assert sum(p.numel() for p in model.parameters()) == 9
    assert sum(p.grad.numel() for p in model.parameters()) == 9
    check_central_differences(model, X, y, manual)

    # 5. Only now do we actually update parameters according to the learning rule.
    optimizer.step()  # New value = old value - 0.01 * gradient.
    changed = 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name] - 0.01 * manual[name])
        torch.testing.assert_close(parameter.grad, manual[name])  # Still the old-state gradient.
        changed += (parameter.detach() != before[name]).sum().item()
    assert changed == 9  # All nine gradients are nonzero in this particular example.
    with torch.no_grad():
        pred_after = model(X)  # Run a new forward pass using the new parameters.
        loss_after = (0.5 * (pred_after - y).square()).mean()
    print("\nPredictions after update:", pred_after.tolist())
    print("Mean loss after update:", loss_after.item())
    # isclose compares numbers while allowing small floating-point representation errors.
    assert math.isclose(loss_after.item(), 0.522123460625, rel_tol=1e-12)
    assert loss.item() == 1.25  # The previously computed loss does not change itself.


def inspect_four_layers(X, y):
    """Distinguish four Linear layers, eight tensors, and 21 parameter numbers."""
    # Linear(D, K) computes K outputs from D inputs.
    # Sequential runs its registered modules in order.
    model = nn.Sequential(
        nn.Linear(2, 2, dtype=DTYPE), nn.ReLU(),
        nn.Linear(2, 2, dtype=DTYPE), nn.ReLU(),
        nn.Linear(2, 2, dtype=DTYPE), nn.ReLU(),
        nn.Linear(2, 1, dtype=DTYPE),
    )
    # Linear stores weight as (outputs, inputs) and computes X @ weight.T + bias.
    # This model has its own initial values, so its gradient values may differ.
    loss = 0.5 * (model(X) - y).square().mean()
    loss.backward()
    parameters = list(model.parameters())
    assert len(parameters) == 8
    assert sum(p.numel() for p in parameters) == 21
    assert sum(p.grad.numel() for p in parameters) == 21
    print("\n[Four-layer model: value shape / gradient shape]")
    for name, p in model.named_parameters():
        assert p.shape == p.grad.shape
        print(name, tuple(p.shape), tuple(p.grad.shape))
    # 0,2,4,6 are module positions; the ReLUs at 1,3,5 have no learned parameters.
    # The 21 gradients count results, not the number of multiplications or additions.


def train_three_epochs(X, y):
    """Inspect a standard batch training loop and inference using the same two samples."""
    model = StudyMLP()  # Start from the initial values in a separate model.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # TensorDataset pairs input and target rows to form samples.
    # DataLoader is the class that retrieves those samples in batches.
    loader = DataLoader(TensorDataset(X, y), batch_size=2, shuffle=True, drop_last=False)
    # shuffle moves inputs and targets together; drop_last=False keeps a smaller final batch.
    total_steps = 0
    print("\n[Three training epochs]")
    for epoch in range(3):
        model.train()  # Switch modes; this call alone performs no training computation.
        loss_total, sample_total = 0.0, 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)  # Clear previous gradients.
            batch_pred = model(batch_X)            # Forward pass.
            assert batch_pred.shape == batch_y.shape
            batch_loss = 0.5 * (batch_pred - batch_y).square().mean()
            batch_loss.backward()                  # Gradients at the current state.
            optimizer.step()                       # Update the parameters.
            total_steps += 1
            # Weight each batch mean by its actual sample count for the epoch record.
            count = batch_X.shape[0]
            loss_total += batch_loss.item() * count
            sample_total += count
        print(f"epoch={epoch + 1}, mean loss observed during training={loss_total / sample_total:.6f}")
    assert total_steps == 3  # Two samples / batch size two * three epochs.
    # Those records use losses at batch-processing time, not a final-model reevaluation.
    model.eval()  # Set Dropout/BatchNorm to evaluation mode; this model has neither.
    with torch.no_grad():  # eval does not disable differentiation records on its own.
        prediction = model(X)
    print("Inference after training:", prediction.tolist())


def compare_learning_rates():
    """Keep other conditions fixed for the separate function J(u)=(u-3)^2."""
    histories = {}  # A dictionary mapping each learning rate to its record list.
    expected_u = {0.05: [0.0, 0.3, 0.57, 0.813],
                  0.25: [0.0, 1.5, 2.25, 2.625],
                  1.1: [0.0, 6.6, -1.32, 8.184]}
    print("\n[Learning-rate comparison for J(u)=(u-3)^2]")
    for learning_rate in [0.05, 0.25, 1.1]:
        u = 0.0  # Begin every experiment at the same position.
        records = [{"step": 0, "u": u, "loss": (u - 3) ** 2}]
        for step in range(1, 4):  # Three values: 1,2,3. The stop value 4 is excluded.
            gradient = 2 * (u - 3)  # Recompute at the new position on every iteration.
            u = u - learning_rate * gradient
            records.append({"step": step, "u": u, "loss": (u - 3) ** 2})
        # append adds to a list and returns None; do not assign that result to records.
        histories[learning_rate] = records
        # zip retrieves items at the same position from the two lists as pairs.
        for record, expected in zip(records, expected_u[learning_rate]):
            assert math.isclose(record["u"], expected, abs_tol=1e-12)
        print("Learning rate:", learning_rate)
        print("u:", [round(row["u"], 6) for row in records])
        print("J:", [round(row["loss"], 6) for row in records])
    # Each experiment has three updates and four records, including the initial state.
    # This function satisfies (u_new-3)=(1-2*learning_rate)*(u-3).
    # 0.25 is faster among these three choices, not optimal for every loss function.
    return histories


def main():
    # manual_seed makes random choices reproducible; the nine manual parameters are fixed.
    torch.manual_seed(0)
    # tensor creates an array of numbers. Input shape is (samples, features)=(2, 2).
    X = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=DTYPE)
    # Match target and prediction shapes at (N, 1) to avoid accidental broadcasting with (N,).
    y = torch.tensor([[6.0], [4.0]], dtype=DTYPE)
    inspect_one_update(X, y)
    inspect_four_layers(X, y)
    train_three_epochs(X, y)
    compare_learning_rates()
    print("\nAll numerical checks passed.")


if __name__ == "__main__":
    main()  # Run the operations above in order when this file is executed directly.
```

</details>

## 18. What carries over to the next batch and to inference?

The complete training loop uses `TensorDataset(X, y)` to pair the input and target at each row into one sample. <strong>`TensorDataset` and `DataLoader` are classes.</strong> The former creates the pairs, and the latter retrieves batches from that dataset. With `shuffle=True`, sample order is shuffled, but inputs and targets move together, preserving their correspondence.[^data]

After one batch's update, the next batch uses <strong>the same model with its updated parameters</strong>. Clearing gradients does not reset learned weights to their initial values. The same model also carries over into the next epoch.

The runnable example has two samples and a batch size of 2, so there is one update per epoch. Three epochs produce three updates. With a larger dataset, the relationship is the same: the outer loop runs epochs, and the inner loop processes batches in sequence.

### Check what an epoch's recorded mean loss represents

The final batch can have a different size, so taking a simple average of batch means can give samples unequal weight. Instead, multiply each batch's mean loss by its actual sample count, add those values, and divide by the total number of samples used.

During training, however, parameters change after each batch. This record therefore <strong>combines losses calculated when each batch was processed</strong>. It differs from evaluating the entire dataset again with the model at the end of the epoch. Even in an example with only one batch, the recorded loss was calculated before that update.

In the learning-rate experiment, `records.append(...)` is a method that adds a new record to the end of a list and returns `None`. Storing a record does not itself perform training. If the initial state is also recorded, there is one more record than there are updates.

### Run inference when only predictions are needed

Using a trained model to calculate predictions is called <strong>inference</strong>. Inference usually does not need differentiation records or updates. It still performs a forward pass: the model receives inputs and calculates outputs.

| Setting | What it does | What it does not do |
| --- | --- | --- |
| `model.train()` | Switches layers whose behavior depends on mode into training mode. | The call itself does not run training. |
| `model.eval()` | Switches those layers into evaluation mode. | It does not disable differentiation recording. |
| `torch.no_grad()` | Disables new automatic differentiation recording for operations performed inside the context. | It does not stop the forward calculation. |

Two common layers whose behavior depends on mode are Dropout and BatchNorm. During training, Dropout performs operations such as randomly setting some values to 0. BatchNorm adjusts values using means and variances, with different ways of using statistics during training and evaluation. This small model includes neither layer, but <strong>evaluation mode and differentiation recording remain separate controls</strong>.[^autograd]

The inference code calls `model.eval()` and then runs `model(X)` inside `with torch.no_grad():`. The model uses the weights and biases changed during training to calculate new predictions. When evaluating new data, the targets are not given to the model as inputs. If needed, they are compared with predictions after the predictions are produced.

---

To examine how one input row and one weight column produce one result entry, see [Understanding Neural Networks Through Matrix Operations](/en/neural-network-matrix-basics/) for the same operation rules in another small example.

### Official documentation and original sources

Documentation checked on <strong>October 21, 2026</strong>. The small numerical examples in this article were calculated directly from the displayed data and parameters.

[^optimization]: PyTorch, [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html). Loss calculation, clearing gradients, backpropagation, optimizer updates, and training loops.
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html). Computation graphs, values saved for backpropagation, evaluation mode, and automatic differentiation modes.
[^mse]: PyTorch, [MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html). Squared errors and averaging over all elements.
[^linear]: PyTorch, [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html). Fully connected layer formulas and weight and bias storage shapes.
[^backward]: PyTorch, [Tensor.backward](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html). Gradient calculation through the chain rule and accumulation in `.grad`.
[^data]: PyTorch, [torch.utils.data](https://docs.pytorch.org/docs/stable/data.html). TensorDataset, DataLoader, and batch configuration options.
[^broadcasting]: PyTorch, [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html). Rules for matching array sizes and the resulting shapes.
[^relu]: PyTorch, [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html) and [Gradients for non-differentiable functions](https://docs.pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions).
[^sgd]: PyTorch, [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html). Update rules involving learning rates, momentum, and weight decay.
[^quadratic]: Neal Parikh and Stephen Boyd, [Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf), §4.2.1. Interpreting gradient descent through a first-order function approximation and a squared movement cost.
[^sign]: Jeremy Bernstein et al., [signSGD: Compressed Optimisation for Non-Convex Problems](https://proceedings.mlr.press/v80/bernstein18a.html), ICML 2018. Optimization using gradient signs.
[^optimizers]: PyTorch, [torch.optim](https://docs.pytorch.org/docs/stable/optim.html) and [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html). Update information used by different optimizers and decoupled weight decay.
[^zero]: PyTorch, [Optimizer.zero_grad](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html). `set_to_none` and clearing gradients.
[^retain]: PyTorch, [Understanding Leaf and Non-leaf Tensors](https://docs.pytorch.org/tutorials/beginner/understanding_leaf_vs_nonleaf_tutorial.html), [Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html), and [Tensor.item](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html). Retaining intermediate gradients and inspecting values.
[^close]: Python, [math.isclose](https://docs.python.org/3/library/math.html#math.isclose); PyTorch, [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close). Comparing floating-point results with tolerances.
[^assert]: Python, [The assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement). Assertion behavior and optimized execution options.
