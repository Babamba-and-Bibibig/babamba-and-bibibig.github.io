+++
title = "Autoencoders and Denoising: From Latent Representations to Reconstruction and Learning"
description = "Follow a small calculation to see how an input becomes a latent representation and a reconstruction. Distinguish samples, features, MSE, backpropagation, and information loss, then train and evaluate autoencoders and denoising models in PyTorch."
date = 2026-11-02
slug = "autoencoder-denoising-basics"

[extra]
katex = true
toc = true
styles = ["autoencoder-denoising-basics/article.css"]
+++

A model that identifies handwritten digits takes an image and produces a classification such as “3.” Here, we will look at a model that takes an image and produces its pixel values again. An <strong>autoencoder (AE)</strong> is a neural network trained to compute an intermediate representation of its input and use that representation to reconstruct the input.

Why recreate an original that we already have? If we limit the number of intermediate values or add noise to the input, the model must learn how to reconstruct the original under those conditions. The intermediate representation can be used for feature extraction, and the reconstruction can be used for tasks such as denoising. However, successful reconstruction alone does not establish that the learned features are useful for every task.[^ae]

We will first work through a small model that turns two input features into one latent value and then reconstructs two values. After examining how both sets of weights learn, we will connect that process to a model for reducing image noise. The small numerical model, the comparison of two weight configurations, the MNIST array discussion, and the digits training example are separate examples.

You can download the [complete Python examples](/autoencoder-denoising-basics/examples-en.py) and the [Rust forward-pass example](/autoencoder-denoising-basics/forward-en.rs). The article introduces code in the order needed for each explanation; the download files organize the examples into a sequence you can run.

## 1. From an input through a latent representation to a reconstruction

First, distinguish the names used in the diagram.

| Name | Meaning | What does it receive or produce? |
| --- | --- | --- |
| Encoder | The part of the network that transforms the input into an intermediate representation | Receives input `x` and computes `z`. |
| Latent representation | The intermediate data computed by the encoder | A collection of numbers computed for each sample. |
| Decoder | The part of the network that turns the intermediate representation into values with the original form | Receives `z` and computes the reconstruction `x_hat`. |
| Reconstruction | The result computed in the form of the input | Written as `x_hat`, or `x̂` in mathematical notation. |
| Target | The desired data against which the output is compared | The original `x` in a basic AE. |
| Loss | A number measuring the difference between the output and the target | This article uses mean squared error. |

In the diagram, `N` is the number of samples processed together, `F` is the number of input features per sample, and `K` is the number of latent features per sample. Parentheses show the <strong>shape</strong>, meaning the size of each dimension, rather than the actual data values. `θe` and `θd` denote the collections of encoder and decoder parameters.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/overview-en.svg" alt="A basic AE passes the original input through an encoder, a latent representation, and a decoder, then compares the reconstruction with the same original. Input and output have shape (N,F); the latent representation has shape (N,K). A DAE adds noise to the model input while keeping the clean original as the target." loading="lazy">
<figcaption>Follow both the path through the model and the path supplying the comparison target for the loss. In a basic AE, the same original serves as both input and target.</figcaption>
</figure>

A <strong>weight</strong> is a learnable number multiplied by an input value. A <strong>bias</strong> is a learnable number added to a weighted sum. Values adjusted during learning, including weights and biases, are called <strong>parameters</strong>. Different samples use the same model parameters.

In contrast, `z` is a result computed for a particular sample. In an ordinary AE, each sample’s `z` is not registered as an independent model parameter and updated directly. When the encoder weights change, the next computation can produce a different `z`.

| What to identify in a neural network | The small model in this article |
| --- | --- |
| Input | One sample contains two feature values. |
| Neurons and structure | One encoder output connects to the decoder’s two output calculations. |
| Intermediate value | One latent value, `z`, is computed. |
| Weights | Two encoder weights and two decoder weights are used. |
| Final output | Two reconstructed values correspond to the original two features. |

This connection also explains why the encoder and decoder learn together. Updating only the decoder can adjust only how a given `z` becomes a reconstruction. Updating the encoder can also adjust how that `z` is produced. One final reconstruction loss trains both parts together.

### What can the model compare without class labels?

Handwritten digit classification pairs an image with a digit-class label. A basic AE uses the image itself as its target. It can therefore compute a loss without separate classification labels.

Basic AEs are traditionally treated as a representation-learning method within <strong>unsupervised learning</strong>. They can also be described through <strong>self-supervised learning</strong>, because the data itself supplies the training target. The presence of a loss function does not imply that training uses labels assigned by a person.

## 2. Separate the number of samples from the number of features

A <strong>sample</strong> is one data example, and a <strong>feature</strong> is an input item that describes that sample. If a sample is written as `B=[2,1]`, there is one sample named B, with first feature value 2 and second feature value 1. Here, B does not denote batch size. This article uses `N` for batch size.

A <strong>batch</strong> is a group of samples processed in one model call. A <strong>tensor</strong> is the data type PyTorch uses to represent numbers and arrays of numbers. A <strong>dimension</strong>, or <strong>axis</strong>, provides a way to identify positions in the data; the shape gives the size of each dimension.

| Representation | Actual values | Shape | Interpretation |
| --- | --- | --- | --- |
| Sample B without a batch dimension | `[2,1]` | `(2,)` | Two feature values |
| Sample B with a batch dimension | `[[2,1]]` | `(1,2)` | One sample, two features per sample |
| A batch containing A and B | `[[1,2],[2,1]]` | `(2,2)` | Two samples, two features per sample |

The actual data `[2,1]` and the shape `(2,1)` mean different things. The former says that the stored values are 2 and 1. The latter says that the first dimension has size 2 and the second has size 1.

The dimensions and indices of the batch `X=[[1,2],[2,1]]` are shown below. An <strong>index</strong> is a position number, starting at 0.

| Dimension number | Meaning | Size | Valid indices | Example selection |
| --- | --- | ---: | --- | --- |
| 0 | Sample | 2 | 0, 1 | `X[0]` is A; `X[1]` is B |
| 1 | Feature within a sample | 2 | 0, 1 | `X[1,0]` is B’s first feature value, 2 |

The structure `2 → 1 → 2` means <strong>two features per sample become one latent value, which then produces two reconstructed values</strong>. It does not mean combining two samples into one.

| Stage | Input | Latent representation | Reconstruction |
| --- | --- | --- | --- |
| Batch shape | `(N,2)` | `(N,1)` | `(N,2)` |
| Number of samples | `N` | `N` | `N` |
| Values per sample | 2 | 1 | 2 |

This model preserves the number of samples because its operations are defined to transform each sample’s features. If a developer explicitly sums over the sample dimension, that dimension can disappear. This is why the dimension selected by an operation matters.

## 3. How two feature values become two reconstructed values

The following calculation uses <strong>no biases or activation functions</strong>. An activation function is a function applied to a weighted sum; the later image model uses ReLU and Sigmoid. For now, we will examine the connections formed by multiplication and addition.

The sample is `A=[1,2]`, the encoder weights are `e=[0.5,0.5]`, and the decoder weights are `d=[0.8,1.2]`. There are four learnable numbers: `e₁, e₂, d₁, d₂`. The diagram also abbreviates the reconstructed values `x_hat₁` and `x_hat₂` as `r₁` and `r₂`.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/tiny-forward-en.svg" alt="Multiplying sample A’s two features, 1 and 2, by encoder weights of 0.5 and adding the results gives z=1.5. The decoder multiplies this same z by 0.8 and 1.2 separately, producing 1.2 and 1.8. MSE against the original is 0.04." loading="lazy">
<figcaption>The encoder adds its two multiplication results. The decoder keeps its two multiplication results as separate components of the reconstructed vector.</figcaption>
</figure>

### Encoder: what gets multiplied, and what gets added?

We want to compute the latent value `z`. The symbols `x₁, x₂` are the two input feature values, and `e₁, e₂` are the encoder weights multiplying those features. A <strong>weighted sum</strong> multiplies each input by its corresponding weight and adds the results.

<div class="ae-math">
$$z=e_1x_1+e_2x_2$$
</div>

| Connection | Input value | Encoder weight | Product |
| --- | ---: | ---: | ---: |
| First feature → latent value | `x₁=1` | `e₁=0.5` | 0.5 |
| Second feature → latent value | `x₂=2` | `e₂=0.5` | 1.0 |

Adding these results gives `z=0.5+1.0=1.5`. The two input features have been used to compute one latent value.

### Decoder: how does one latent value produce two values?

We want `x_hat₁, x_hat₂`, corresponding to the original two features. The decoder receives the newly computed `z=1.5`, rather than the original `[1,2]`.

<div class="ae-math">
$$\hat{x}_1=d_1z,\qquad \hat{x}_2=d_2z$$
</div>

| Output | Latent value used | Decoder weight | Reconstructed value |
| --- | ---: | ---: | ---: |
| Reconstruction of the first feature | 1.5 | `d₁=0.8` | `0.8×1.5=1.2` |
| Reconstruction of the second feature | 1.5 | `d₂=1.2` | `1.2×1.5=1.8` |

The reconstructed vector is therefore `[1.2,1.8]`. A <strong>component</strong> is an individual value within a vector. There is one variable named `reconstruction`, but it contains two reconstructed components.

### Express the same calculation with tensors

`torch.tensor()` is a function that creates a tensor from data. The setting `dtype=torch.float64` represents each value using 64-bit floating-point numbers. <strong>Floating-point representation</strong> approximates real numbers with a finite number of bits, so printed results can contain small numerical differences.

```python
import torch

def tiny_ae(x, e, d):
    # This function accepts one sample without a batch dimension.
    # x, e, and d each contain two values and have shape=(2,).
    if x.shape != (2,) or e.shape != (2,) or d.shape != (2,):
        raise ValueError("x, e, and d must each have shape (2,).")

    # Sum the elementwise products [e1*x1, e2*x2] into one latent value.
    z = (e * x).sum()                  # shape=(): a scalar tensor.

    # Multiply the same z by d1 and d2 separately. Do not add the outputs.
    reconstruction = d * z            # shape=(2,).

    # Average the squared errors of the two corresponding components.
    loss = ((reconstruction - x) ** 2).mean()
    return z, reconstruction, loss

A = torch.tensor([1.0, 2.0], dtype=torch.float64)
e = torch.tensor([0.5, 0.5], dtype=torch.float64)
d = torch.tensor([0.8, 1.2], dtype=torch.float64)
z, reconstruction, loss = tiny_ae(A, e, d)

print(z.item())                    # One-element tensor → Python number: about 1.5.
print(reconstruction.tolist())    # Tensor → Python list: about [1.2, 1.8].
print(loss.item())                 # About 0.04; the next section explains the calculation.
```

A <strong>scalar</strong> is a single number. Above, `z.shape` is `()`, with no dimensions. A tensor with shape `(1,)` also has one element, but it is a vector with one dimension of length 1. Shape `(1,1)` has two dimensions. Read the element count and the dimension structure separately.

`.item()` and `.tolist()` convert values for display or logging. PyTorch’s backpropagation connections do not continue through the resulting Python numbers or lists.

## 4. MSE averages squared errors between corresponding components

<strong>Mean squared error (MSE)</strong> subtracts the reconstruction and target at corresponding positions, squares each difference, and averages the results. The sequence is <strong>match positions → subtract → square each difference → average</strong>.

In the previous section, the target was A’s original `[1,2]`, and the reconstruction was `[1.2,1.8]`.

| Calculation | First component | Second component |
| --- | ---: | ---: |
| Target | 1 | 2 |
| Reconstruction | 1.2 | 1.8 |
| Reconstruction − target | +0.2 | −0.2 |
| Squared error | 0.04 | 0.04 |

There are two values to average, so the result is `(0.04+0.04)/2=0.04`. <strong>The denominator 2 is the number of compared components, not the number of samples.</strong> Adding raw differences can let positive and negative values cancel; squaring prevents that cancellation.

Writing the original components as `x₁, x₂`, the reconstructed components as `x_hat₁, x_hat₂`, and the loss as `L`, the calculation is:

<div class="ae-math">
$$L=\frac{(\hat{x}_1-x_1)^2+(\hat{x}_2-x_2)^2}{2}$$
</div>

Because the differences are squared, `original−reconstruction` and `reconstruction−original` give the same MSE. We will consistently use `reconstruction−target` when explaining gradients.

### What happens if you call `.mean()` after computing a sum?

Adding the two squared errors first leaves one value. A Python float has no `.mean()` method. For a PyTorch scalar tensor, `.mean()` averages one element, so the value remains unchanged. It does not automatically divide by 2.

```python
# reconstruction and target are tensors with the same shape.
target = A
squared_errors = (reconstruction - target).square()  # [0.04, 0.04].
correct = squared_errors.mean()                      # About 0.04.
sum_then_mean = squared_errors.sum().mean()           # About 0.08.

# If two scalar tensors already exist, stack can combine them along a new dimension.
error1 = squared_errors[0]
error2 = squared_errors[1]
also_correct = torch.stack([error1, error2]).mean()
```

The methods `square()`, `sum()`, and `mean()` compute elementwise squares, a sum, and a mean, respectively. The function `torch.stack()` combines tensors along a new dimension. Here, it combines two scalars into a tensor of shape `(2,)`.[^stack]

Do not build a training loss by recreating the tensors `error1` and `error2` as `torch.tensor([error1,error2])`. `torch.tensor()` copies their values into a new tensor and breaks the automatic-differentiation connection to the original computation. Adding `requires_grad=True` to the new tensor does not restore its connection to the original weights.[^tensor]

### A mean over the batch versus a mean for each sample

In practical code, `torch.nn.MSELoss()` creates a loss object, which you then call as `loss_fn(reconstruction,target)`. Its default setting, `reduction="mean"`, averages <strong>the squared errors of all corresponding elements</strong> in the two tensors.[^mse]

| Input and target shape | Number of squared errors averaged | Default result shape |
| --- | --- | --- |
| `(N,F)` | `N×F` | `()` |
| `(N,C,H,W)` | `N×C×H×W` | `()` |

To inspect the error of each image separately, keep the sample dimension. Starting with `(N,C,H,W)`, `flatten(start_dim=1)` preserves dimension 0 and combines the remaining dimensions into `(N,C×H×W)`. Applying `mean(dim=1)` then produces `(N,)`, with one error per sample.

Before computing the loss, check that `reconstruction.shape == target.shape`. If the shapes differ, <strong>broadcasting</strong>—the rules that match values across compatible dimension sizes—can produce an unintended comparison. Also avoid comparing MSE numbers directly when the pixel ranges or averaging conventions differ.

## 5. Compute gradients for both parts from one loss

The loss tells us how far the current reconstruction is from the target. A <strong>gradient</strong> tells us the local direction in which to adjust the weights. The expression `∂L/∂w` does not divide loss `L` by weight `w`. It is a <strong>partial derivative</strong>: the sensitivity of the loss to a small change in `w`, with other independent weights and the input held fixed. Collecting these derivatives for multiple variables gives a gradient.

At the current point, a positive gradient means that increasing `w` slightly moves toward higher loss. A negative gradient means that increasing `w` slightly moves toward lower loss. Gradient magnitude describes the current rate of change; it is neither the distance to the correct weight nor the optimal update size.

The <strong>forward pass</strong> computes from the input through the reconstruction to the loss. <strong>Backpropagation</strong> follows those connections backward from the loss to compute gradients for earlier variables. To determine an earlier weight’s effect on the loss, we need to know how the intermediate value changed by that weight affects the loss through subsequent calculations.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/backward-en.svg" alt="Output gradients 0.2 and -0.2 produce decoder gradients 0.3 and -0.3. The two paths back to the latent value contribute 0.16 and -0.24, summing to -0.08. Multiplying by inputs 1 and 2 gives encoder gradients -0.08 and -0.16." loading="lazy">
<figcaption>Distinguish the gradients stored for weights from the intermediate-value gradient passed to earlier computations. The gradient of z continues the calculation back to the encoder.</figcaption>
</figure>

### Start with the loss’s rate of change with respect to the reconstruction

To keep the expressions short, write the reconstructed values `x_hat₁, x_hat₂` as `r₁, r₂`. In the previous section, `r₁=1.2`, `r₂=1.8`, and the targets were `x₁=1`, `x₂=2`.

For MSE over `m` elements, the gradient with respect to one output is `2/m × (reconstruction−target)`. Here, there are two elements to average, so `2/m=1`. Thus, `∂L/∂r₁=+0.2` and `∂L/∂r₂=−0.2`. The difference itself is not the gradient for every MSE calculation.[^mse]

We can check this rate of change directly. Holding everything else fixed and changing only the first reconstructed value from `1.2` to `1.21` changes the loss from `0.04` to `0.04205`. Dividing the loss change, `0.00205`, by the output change, `0.01`, gives `0.205`. Reducing the output change to `0.001` gives a ratio of `0.2005`. As the change becomes smaller, the ratio approaches the gradient, `0.2`.

### Decoder weights and the latent value are different gradient targets

The <strong>chain rule</strong> multiplies rates of change along connected computations to obtain the rate of change with respect to an earlier variable. When one variable affects the loss along multiple paths, add the values calculated along those paths.

The decoder relationships are `r₁=d₁z` and `r₂=d₂z`. A small change in `d₁` changes `r₁` by `z` times that amount. Therefore, the gradient of `d₁` is the first output gradient multiplied by `z`.

| Gradient to compute | Calculation along the connection | Values used | Result |
| --- | --- | --- | ---: |
| Gradient of `d₁` | `(∂L/∂r₁)×z` | `+0.2×1.5` | +0.30 |
| Gradient of `d₂` | `(∂L/∂r₂)×z` | `−0.2×1.5` | −0.30 |

Next comes the gradient of `z`. Changing the single value `z` changes both reconstructed values. We need to account for the effect through the first reconstruction and the effect through the second.

| Path | Multiplier from z to the output | Loss gradient of that output | Value added to the loss’s rate of change with respect to z |
| --- | ---: | ---: | ---: |
| `z → r₁ → L` | `d₁=0.8` | +0.2 | `+0.2×0.8=+0.16` |
| `z → r₂ → L` | `d₂=1.2` | −0.2 | `−0.2×1.2=−0.24` |

Therefore, `∂L/∂z=+0.16−0.24=−0.08`. “Adding the effects of two paths” means that <strong>the same z is used in two output calculations, so we add the two effects that a small change in z has on the total loss</strong>. We are not simply adding the gradients of two weights.

### Pass the latent-value gradient back to the encoder weights

The encoder relationship is `z=e₁x₁+e₂x₂`. A change in `e₁` is multiplied by the first input, `x₁`, on its way to `z`; a change in `e₂` is multiplied by the second input, `x₂`.

| Gradient to compute | Calculation along the connection | Values used | Result |
| --- | --- | --- | ---: |
| Gradient of `e₁` | `(∂L/∂z)×x₁` | `−0.08×1` | −0.08 |
| Gradient of `e₂` | `(∂L/∂z)×x₂` | `−0.08×2` | −0.16 |

The final gradients for the four weights are `e.grad=[−0.08,−0.16]` and `d.grad=[+0.30,−0.30]`. This count does not mean that only four operations were performed. These results were obtained through the output and intermediate-value gradients, multiplications, and additions shown above.

PyTorch handles these connections through <strong>automatic differentiation</strong>, also called <strong>Autograd</strong>. Tensors at the start of the calculation, such as directly created learnable weights, are called <strong>leaf tensors</strong>. By default, gradients accumulate in the `.grad` attributes of learnable leaf tensors. The intermediate gradient of `z` can be used in the calculation without being retained in its `.grad` attribute. To inspect it, call `z.retain_grad()` before backpropagation.[^autograd]

## 6. Computing gradients and changing weights are separate operations

<strong>Gradient descent</strong> moves weights a small distance opposite the current gradient. The <strong>learning rate</strong> is a positive factor controlling the update size.

Why the opposite direction? If `Δw` is a small change in a weight, the nearby loss change is approximately `(∂L/∂w)×Δw`. Choosing `Δw` with the opposite sign to the gradient makes this product negative, giving a direction that reduces the loss. If the step is too large, the local rate of change is insufficient to predict the result, so an actual loss decrease is not guaranteed.

The basic update is `new weight = old weight − learning rate×gradient`. Apply a learning rate of `0.1` to the gradients computed in the previous section.

| Weight | Old value | Gradient of A’s loss | Update calculation | New value |
| --- | ---: | ---: | --- | ---: |
| `e₁` | 0.5 | −0.08 | `0.5−0.1×(−0.08)` | 0.508 |
| `e₂` | 0.5 | −0.16 | `0.5−0.1×(−0.16)` | 0.516 |
| `d₁` | 0.8 | +0.30 | `0.8−0.1×0.30` | 0.770 |
| `d₂` | 1.2 | −0.30 | `1.2−0.1×(−0.30)` | 1.230 |

The following code reproduces this update. It uses the earlier `tiny_ae()` function but creates the weights again from their initial values. `requires_grad=True` enables gradients with respect to these weights.

```python
# Run tiny_ae() and import torch first.
A = torch.tensor([1.0, 2.0], dtype=torch.float64)
B_input = torch.tensor([2.0, 1.0], dtype=torch.float64)
e = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
d = torch.tensor([0.8, 1.2], dtype=torch.float64, requires_grad=True)

# Compute this loss with differentiation connections back to the weights.
z_before, recon_before, loss_before = tiny_ae(A, e, d)

# B is not used for training here; observe only its pre-update result.
with torch.no_grad():
    z_B0, recon_B0, loss_B0 = tiny_ae(B_input, e, d)

loss_before.backward()       # Compute gradients. The weights have not changed yet.
grad_e = e.grad.clone()       # Copy gradients calculated at the pre-update state.
grad_d = d.grad.clone()

# Do not record the update operations themselves for automatic differentiation.
with torch.no_grad():
    e -= 0.1 * grad_e         # Actually change the encoder weights.
    d -= 0.1 * grad_d         # Use decoder gradients from the same pre-update state.

# Run a new forward pass to observe the new weights' results.
with torch.no_grad():
    for name, x in [("A", A), ("B", B_input)]:
        z, recon, loss = tiny_ae(x, e, d)
        print(name, z.item(), recon.tolist(), loss.item())
```

The two update lines run in sequence, but both use <strong>gradients from the same pre-update model</strong>. The code does not update the encoder and then recompute the decoder gradients using that new encoder. In practical training loops, an <strong>optimizer</strong> performs the updates. Optimizers such as Adam also use gradient history, so their behavior differs from the simple update formula above.

### What happens to B after training on A?

| Quantity observed | Before the update | After the update |
| --- | --- | --- |
| A’s latent value | 1.5 | 1.54 |
| A’s reconstruction | `[1.2,1.8]` | `[1.1858,1.8942]` |
| A’s MSE | 0.04 | 0.02285764 |
| B’s latent value | 1.5 | 1.532 |
| B’s reconstruction | `[1.2,1.8]` | `[1.17964,1.88436]` |
| B’s MSE | 0.64 | 0.7275415696 |

<strong>A’s loss decreased, but B’s loss increased.</strong> The update used the gradient of A’s loss. There is no guarantee that this direction also reduces B’s loss.

This single observation does not establish <strong>overfitting</strong>, a condition in which a model fits its training data well but performs poorly on new data. It shows that improvement on one sample does not guarantee improvement on another. Actual training performance must be checked on multiple examples not used for training.

## 7. Distinguish `no_grad`, `backward`, and `step`

In PyTorch, running a model, computing gradients, and changing weights are different operations.

| Code | What it does | Does it directly change weights? |
| --- | --- | --- |
| `model(x)` | Computes an output using the current weights. | Not in an ordinary forward pass. |
| `loss.backward()` | Accumulates loss gradients in the relevant `.grad` attributes. | No. |
| `optimizer.step()` | Updates the parameters managed by the optimizer. | Yes. |
| `optimizer.zero_grad(set_to_none=True)` | Clears stored gradients by setting them to `None`. | No. |
| `with torch.no_grad():` | Omits backward-pass recording for ordinary operations inside the block. | Weights can change if you explicitly write an update operation. |

`no_grad()` is a <strong>context manager</strong> that skips automatic-differentiation recording. The setting applies while execution is inside the `with` block; the previous state is restored when the block ends. It reduces recording and memory use that are unnecessary for evaluation.[^no-grad]

Removing `no_grad()` from the earlier B observation code would not automatically include B in training. Even if both A’s and B’s operations are recorded, calling only `loss_A.backward()` computes gradients connected to A’s loss. To use both losses, backpropagate from a loss that combines them, such as `(loss_A+loss_B)/2`.

`no_grad()` does not clear existing `.grad` values, permanently change a parameter’s `requires_grad`, or prohibit weight modification. That is why the earlier manual update could run inside `no_grad()`.

### How is `model.eval()` different?

`model.train()` and `model.eval()` configure the training or evaluation behavior of certain layers. <strong>Dropout</strong> randomly sets some values to 0 during training. <strong>BatchNorm</strong> normalizes values using statistics that can differ between training and evaluation. Such layers change their behavior with the mode.

`model.eval()` does not disable automatic-differentiation recording. Evaluation usually calls `model.eval()` and then runs the model inside `with torch.no_grad():`. The Linear–ReLU–Sigmoid model in this article has no layers whose behavior changes between training and evaluation, but the evaluation code follows the same pattern.[^autograd]

### What do `detach()` and `clone()` separate?

| Expression | Existing differentiation connection | Storage relative to the original | Example use |
| --- | --- | --- | --- |
| `tensor.detach()` | Disconnects it. | Shares storage. | Read values without the connection. |
| `tensor.clone()` | Preserves it when the source is tracked. | Creates independent storage. | Make a copy of the values. |
| `tensor.detach().clone()` | Disconnects it. | Creates independent storage. | Preserve the initial weight state. |

For example, after `e0=e.detach().clone()`, subsequent changes to `e` do not affect `e0` through shared storage. Neither a variable name such as `fixed` nor `detach()` makes the values immutable.[^detach][^clone]

A training loop repeats <strong>clear gradients → new forward pass → new loss → backpropagation → update</strong>. Because `backward()` can add to existing gradients, clear them unless gradient accumulation is intentional. Repeating an old `loss.backward()` line does not constitute the next training step.

## 8. Specify the summation dimension when processing a batch

The earlier `tiny_ae()` function accepts one sample of shape `(2,)`. Summing its two features with `(e*x).sum()` therefore does not mix different samples.

Now use `X.shape=(2,2)`, containing A and B together. Dimension 0 indexes samples, and dimension 1 indexes features. <strong>To sum features within each sample, specify `dim=1`.</strong>

```python
import torch

X = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
e = torch.tensor([0.5, 0.5], dtype=torch.float64)
d = torch.tensor([0.8, 1.2], dtype=torch.float64)

products = X * e                          # [[0.5,1.0],[1.0,0.5]].
Z = products.sum(dim=1, keepdim=True)     # [[1.5],[1.5]], shape=(2,1).
X_hat = Z * d                            # [[1.2,1.8],[1.2,1.8]], (2,2).

print(products.tolist())
print(Z.tolist(), tuple(Z.shape))
print(X_hat.tolist(), tuple(X_hat.shape))
```

`dim=1` requests a sum along dimension number 1, and `keepdim=True` keeps the reduced dimension with size 1. Broadcasting multiplies each latent value in `(2,1)` by the two decoder weights separately. It does not combine latent values from different samples.

Omitting the dimension, as in `products.sum()`, adds all four elements into the scalar `3.0`. That is not a separate latent representation for each of the two samples.

### What shape do the weights have in `nn.Linear`?

`from torch import nn` imports `torch.nn` under the name `nn`. `nn.Linear` is a layer class that computes weighted sums of input features. Its <strong>constructor</strong> creates a layer; calling that layer performs the calculation on actual data.[^linear]

| Item | Encoder | Decoder |
| --- | --- | --- |
| Construction | `nn.Linear(2,1,bias=False)` | `nn.Linear(1,2,bias=False)` |
| Input → output | `(N,2) → (N,1)` | `(N,1) → (N,2)` |
| Weight shape | `(1,2)` | `(2,1)` |

The constructor’s first argument, `in_features`, is the number of input features per sample. The second, `out_features`, is the number of output features per sample. `bias=False` disables the bias. Batch size `N` is not a constructor argument. The actual batch is supplied when calling the constructed encoder as `encoder(X)`.

The weight shape in PyTorch Linear is <strong>`(output features, input features)`</strong>. The earlier `e.shape=(2,)` and `d.shape=(2,)` were direct representations of the same calculation. Creating a Linear layer assigns weights according to its initialization rule; reproducing the earlier numbers requires copying those particular values into its weights.

## 9. Reducing the number of values and preserving information

<strong>Dimensionality reduction</strong> reduces the number of features representing a sample. A restricted intermediate part with fewer values than the input is called a <strong>bottleneck</strong>. A small bottleneck is one way to construct an AE, but it is not required for every AE.[^ae]

Whether fewer values still allow reconstruction depends on the patterns in the data and the information retained. We can examine this using a fixed rule that does not learn.

From an input `[a,b,c,d]`, keep only its first two values, `[a,b]`. To reconstruct it, repeat those two values to produce `[a,b,a,b]`.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/information-loss-en.svg" alt="Keeping the first two values of different originals [0.2,0.8,0.2,0.8] and [0.2,0.8,0.7,0.1] produces the same code, [0.2,0.8]. Repeating that code produces the same reconstruction, [0.2,0.8,0.2,0.8], perfectly reconstructing the first original but giving MSE 0.185 for the second." loading="lazy">
<figcaption>This is a fixed rule, not a learned model. The example asks whether this decoder can distinguish two inputs that have been mapped to the same code.</figcaption>
</figure>

| Original | Retained code | Reconstruction | MSE |
| --- | --- | --- | ---: |
| `[0.2,0.8,0.2,0.8]` | `[0.2,0.8]` | `[0.2,0.8,0.2,0.8]` | 0 |
| `[0.2,0.8,0.7,0.1]` | `[0.2,0.8]` | `[0.2,0.8,0.2,0.8]` | 0.185 |

The first original is reconstructed exactly because its final two values repeat its first two. The second original ends with `0.7,0.1`, but that information is absent from the code. Its squared errors are `[0,0,0.25,0.49]`, with a mean of `0.185`.

<strong>Deterministic</strong> means producing the same result from the same input and model state. A fixed deterministic decoder receiving only the same code cannot distinguish the two originals above and reconstruct both exactly.

```python
import torch

def repeat_first_half(x):
    # x has shape=(4,). The slice excludes endpoint 2, selecting only indices 0 and 1.
    code = x[:2].clone()

    # cat joins along an existing dimension: two (2,) tensors → (4,).
    reconstruction = torch.cat([code, code])
    mse = (reconstruction - x).square().mean()
    return code, reconstruction, mse

x = torch.tensor([0.2, 0.8, 0.7, 0.1], dtype=torch.float64)
code, reconstruction, mse = repeat_first_half(x)
print(code.tolist(), reconstruction.tolist(), mse.item())  # MSE is about 0.185.
```

`torch.cat()` extends an existing dimension, whereas `torch.stack()` creates a new one. Both combine tensors, but they determine the resulting shape differently.

### Separate information loss from reconstruction error

<strong>Information loss</strong> occurs when a representation no longer retains information that distinguishes different originals. <strong>Reconstruction error</strong> is the actual difference between a reconstruction and its original. A <strong>loss function</strong> is the calculation used to measure that difference numerically.

Even if the code retains the necessary information, a decoder can produce reconstruction errors if it has not learned sufficiently or lacks enough expressive capacity. Conversely, data with a repeating pattern, such as the first original above, can be reconstructed exactly from fewer values. Do not attribute every error to information lost by the encoder, or assume that reducing values makes exact reconstruction impossible for every kind of data.

### If there are fewer latent values, does the file shrink by the same ratio?

An RGB `32×32` image has `3×32×32=3,072` values. Encoding it into 128 latent values reduces the value count. Actual storage also depends on the bits per element, storage format, additional information, and whether the reconstruction model must be transmitted.

Practical learned compression adds steps such as <strong>quantization</strong>, which maps latent values to a defined set of values, and <strong>entropy coding</strong>, which uses information such as value probabilities to produce a bitstream. The stored or transmitted bitstream is decoded before the decoder reconstructs the data.[^compression]

The relationship between reconstruction quality and the required number of bits is called the <strong>rate–distortion trade-off</strong>. The number of AE features alone does not determine the actual file compression ratio.

## 10. What are points and dimensions in latent space?

<strong>Latent space</strong> is the space in which latent representations are treated as coordinates. One latent value per sample gives one coordinate; two values give two coordinates; `K` values give `K`-dimensional coordinates.

If `Z.shape=(100,16)`, each of 100 samples is represented by 16 latent features. <strong>There are 100 points, and each point has 16 coordinate components.</strong> The sample count of 100 is not the dimension of the latent space.

Keeping the encoder `e=[0.508,0.516]` from the single update on A and passing A and B through it gives the following results. The reconstruction MSE also uses the same updated decoder, `d=[0.770,1.230]`.

| Input | Latent value from the same encoder | Reconstruction MSE |
| --- | ---: | ---: |
| A=`[1,2]` | 1.540 | 0.02285764 |
| B=`[2,1]` | 1.532 | 0.7275415696 |

The coordinate gap is `1.540−1.532=0.008`. This shows <strong>where the same encoder places the two inputs</strong>. It does not mean that smaller latent values produce better reconstructions, or that nearby coordinates necessarily imply similar meaning.

### How latent representations relate to embeddings

An <strong>embedding</strong> represents data as a numerical vector. An AE’s latent representation can be considered an embedding in this broad sense. However, a representation trained for reconstruction has a different objective from one trained for semantic search, such as finding related sentences.

<strong>Cosine similarity</strong> measures how similar the directions of two vectors are. It is 1 for the same direction, 0 for perpendicular directions, and −1 for opposite directions. Computing this measurement does not by itself train a useful embedding. A model first produces vectors, and those vectors are then compared.[^embedding]

The values `1.540` and `1.532` above are both positive and one-dimensional. They point in the same positive direction, so their mathematical cosine similarity is 1. Here, we are examining the gap between coordinates on a number line. A zero vector has no direction, so its mathematical cosine similarity is undefined.

<details>
<summary>Code and arguments for plotting two latent values</summary>

A `Figure` manages the entire figure, while an `Axes` is an individual plotting area inside it. `plt.subplots()` creates both. Below, `y=0` is a display coordinate that places the points on one horizontal line; it is not a second latent feature.

```python
import matplotlib.pyplot as plt

values = [("A", 1.540), ("B", 1.532)]
fig, ax = plt.subplots(figsize=(7, 2.4))  # Figure width and height in inches.
ax.axhline(y=0, color="gray")           # Draw a horizontal line at data coordinate y=0.
ax.scatter([value for _, value in values], [0, 0])

for name, value in values:             # Unpack the name and value from each tuple.
    ax.annotate(
        f"{name}: z={value:.3f}",      # Display three digits after the decimal point.
        xy=(value, 0),                # Actual data coordinates of the annotated point.
        xytext=(0, 18 if name == "A" else -28),
        textcoords="offset points", # Units for the text offset from the point.
        ha="center",                 # Center the text horizontally.
    )

ax.set_xlim(1.525, 1.547)               # Set the display range without changing z.
ax.set_ylim(-0.45, 0.45)
ax.set_yticks([])                      # Hide vertical ticks that carry no meaning here.
ax.set_xlabel("Latent value z")
fig.tight_layout()                    # Adjust spacing around titles and tick labels.
plt.show()
```

The list comprehension `[value for _,value in values]` produces `[1.540,1.532]`. The conventional variable `_` receives the name, which is unused in this calculation. `annotate()` places explanatory text; a point in `offset points` is `1/72` of an inch. The value `18` does not move the latent coordinate by 18.[^plot]

The `y` argument to `axhline()` uses actual data coordinates, but `xmin` and `xmax` are fractions from 0 to 1 of the plotting area’s width. `set_title`, `set_xticks`, `tick_params`, and `spines` adjust titles, ticks, text, and borders. These settings do not train any values. If a separate color dictionary is used, it must also be defined in the code.

</details>

## 11. A denoising AE uses different inputs and targets

A <strong>denoising autoencoder (DAE)</strong> learns to reconstruct the original, uncorrupted data from a corrupted input. Call the original `clean` and the version with added noise `noisy`. Here, clean means the reference data before artificial noise is added.[^dae]

| Item | Basic AE | DAE |
| --- | --- | --- |
| Model input | `clean` | `noisy` |
| Comparison target | `clean` | `clean` |
| Direct encoder output | Latent representation `z` | Latent representation `z` |
| Direct decoder output | Reconstruction | Reconstruction |
| Final training objective | Reconstruct close to the original input | Reconstruct close to the original before noise was added |

The key difference is what enters the model and what its output is compared against. Even with the same architecture, connecting the training data as follows changes the task.[^keras]

```python
# model is an already constructed model; loss_fn is nn.MSELoss().

# Basic AE: receive the original and compare against the original.
reconstruction = model(clean_batch)
loss = loss_fn(reconstruction, clean_batch)

# DAE: receive the noisy input and compare against the original before noise.
reconstruction = model(noisy_batch)
loss = loss_fn(reconstruction, clean_batch)
```

A DAE can also consist of one encoder and one decoder. The encoder does not need to complete a clean image first for the decoder to reconstruct a second time. The encoder directly outputs intermediate features, and the decoder computes output values in the original form.

For example, if 784 image pixels become 64 latent features, those 64 values are not a completed 784-pixel image. A basic DAE usually supplies no separate correct `z` for these intermediate values. It trains both parts together using the loss between the final output and clean.

<strong>Robustness</strong> is the ability to use relevant information reliably despite noise or changes in the input. DAE training can encourage representations that withstand noise, but it does not guarantee that all noise disappears from the latent representation. Denoising ability is evaluated at the final output. A small bottleneck is not required for a DAE either.

## 12. Distinguish samples, channels, height, and width in images

MNIST consists of `28×28` grayscale handwritten digit images. Consider one grayscale image of that size. A <strong>pixel</strong> is a spatial position in an image; a grayscale image stores one brightness value at each position. A <strong>channel</strong> is a dimension separating image components. This grayscale image has one channel.

In the PyTorch image-batch convention `(N,C,H,W)`, `N` is the number of samples, `C` the number of channels, `H` the height, and `W` the width.

| Dimension number | Meaning | Example size | Valid indices |
| --- | --- | --- | --- |
| 0 | Samples `N` | `N` | `0` through `N−1` |
| 1 | Channels `C` | 1 | 0 |
| 2 | Positions along height `H` | 28 | 0–27 |
| 3 | Positions along width `W` | 28 | 0–27 |

Putting one image in a batch gives `(1,1,28,28)`. A channel-and-spatial representation of one image without a batch dimension is `(1,28,28)`. However, <strong>shape `(1,28,28)` alone does not establish that its first dimension is channels</strong>. The file creator might have omitted channels and stored `(N,H,W)`. Check the code and convention used to create the array.

Once you confirm that an array stores one grayscale image as `(C,H,W)`, use `torch.from_numpy(clean).unsqueeze(0)` to add a batch dimension. `from_numpy()` creates a tensor that shares memory with the NumPy array; `unsqueeze(0)` inserts a dimension of size 1 at position 0. This does not generate a new image.[^tensor]

### How is noise added?

The <strong>standard normal distribution</strong> is a normal distribution with mean 0 and standard deviation 1. <strong>Standard deviation</strong> measures how widely values are spread around the mean. The following code creates noise with the same shape as clean, scales that noise, and limits pixel values to the range 0–1.

```python
# clean_batch is an (N,1,28,28) tensor with values in the range 0–1.
noise = torch.randn_like(clean_batch)          # Standard normal noise with the same shape.
noisy_batch = (clean_batch + 0.25 * noise).clamp(0, 1)
```

`randn_like()` creates a random tensor with the same shape as the reference tensor. Multiplying by `0.25` gives the noise a standard deviation of `0.25` before range limiting. `clamp(0,1)` replaces values below 0 with 0 and values above 1 with 1. This range-limiting operation is called <strong>clipping</strong>.

After clipping, the actual change `noisy_batch−clean_batch` no longer necessarily follows a normal distribution with standard deviation 0.25. Values that crossed the boundaries have been clipped.

### What does comparing the same pixel mean?

For an array of shape `(1,28,28)`, `image[0,14,14]` selects first-dimension index 0, height index 14, and width index 14. Since indices start at 0, this is the 15th row and 15th column in the displayed image.

For example, a pair might have a clean value of approximately `0.7921569` and a noisy value of approximately `0.89783067` at the same position. A DAE receives the entire noisy image, computes an entire reconstructed image, and compares this reconstructed position with the same clean position. It is not a digit-class prediction task. The full loss compares all pixels, not only this one.

| Name | When is it available? | Role |
| --- | --- | --- |
| `noisy` | Before running the model | Input |
| `clean` | Before running the model | Comparison target for training and evaluation |
| `reconstruction` | After running the model | Actual predicted output |

Displaying clean does not mean that the model has reconstructed it. During actual use, called <strong>inference</strong>, the model usually receives noisy and produces a reconstruction without knowing clean. Clean supplies the comparison reference during training and evaluation.

## 13. What does a four-panel clean/noisy figure check?

Loading arrays and inspecting them visually is useful before model training. It checks whether the input and target come from the same original, their positions align, and their value ranges are appropriate.

<strong>NPZ</strong> is a file format that can contain multiple named NumPy arrays. The extension alone does not establish that the arrays were stored with compression. The following code reads a file containing numerical arrays named `clean` and `noisy`. The file must be supplied separately; the digits example later in the article does not need it.

```python
from pathlib import Path
import numpy as np

# Set this to the location of the actual file.
pair_path = Path("mnist_day1_pair.npz")
if not pair_path.is_file():
    raise FileNotFoundError(f"Check the file location: {pair_path}")

# allow_pickle=False: load numerical arrays without restoring Python objects.
# Leaving the with block closes the open NPZ file.
with np.load(pair_path, allow_pickle=False) as pair:
    clean = pair["clean"].copy()      # Keep an independent array copy.
    noisy = pair["noisy"].copy()

assert clean.shape == noisy.shape == (1, 28, 28)
for name, image in [("clean", clean), ("noisy", noisy)]:
    print(name, image.shape, image.dtype, image.min(), image.max())
```

When `np.load()` reads an NPZ file, it returns an object from which arrays can be retrieved by name. `dtype` is the element data type; `min()` and `max()` return the actual minimum and maximum values in the array. `float32` means 32-bit floating-point values.[^numpy]

A clean maximum slightly below `1.0`, such as `0.996...`, is not by itself a scaling error. Being within 0–1 and containing a pixel exactly equal to 1 are different conditions. If file loading fails, correct the path and load again. Do not interpret arrays or figures left over from earlier notebook cells as results from the new file.

### Two full-image panels and two zoomed panels

`fig, axes=plt.subplots(2,2)` creates one figure and four plotting areas. `axes.shape=(2,2)` describes the array of plotting-area objects. It is unrelated to the model’s input shape.

| Plotting area | Data displayed | Array size |
| --- | --- | --- |
| `axes[0,0]` | Full noisy image | `28×28` |
| `axes[0,1]` | Full clean image | `28×28` |
| `axes[1,0]` | Zoomed region of noisy | `7×7` |
| `axes[1,1]` | The same zoomed region of clean | `7×7` |

These are not four separate samples. They show <strong>one noisy image and one clean image, each displayed both in full and as a zoomed region</strong>. With no model call or backpropagation, this is not a denoising-results figure either.

`axes[0,1]` selects the upper-right plotting area. `image[0,14,14]` selects a pixel in the data. Keep the objects selected by these two index expressions separate.

### Keep the original pixel indices when zooming

`image[0,11:18,11:18]` selects height and width indices `11,12,13,14,15,16,17`. <strong>The endpoint 18 is excluded.</strong> Each direction has `18−11=7` values, so the cropped array has shape `(7,7)`. Original index 14 becomes local index `14−11=3` in the crop.

Displaying this small array with default coordinates gives indices 0–6. To retain the original coordinates 11–17, specify `extent=(10.5,17.5,17.5,10.5)`. The order is `(left,right,bottom,top)`. Pixel centers run from 11 through 17, with each pixel boundary 0.5 away from its center, so the full boundaries are 10.5–17.5.[^imshow]

```python
# Run after loading the clean/noisy file. This display code does not train a model.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for col, (name, image) in enumerate([("noisy", noisy), ("clean", clean)]):
    full = axes[0, col]                  # Area for the full image.
    zoom = axes[1, col]                  # Area for a crop of the same image.
    full.imshow(image[0], cmap="gray", vmin=0, vmax=1,
                interpolation="nearest", origin="upper")
    zoom.imshow(image[0, 11:18, 11:18], cmap="gray", vmin=0, vmax=1,
                interpolation="nearest", origin="upper",
                extent=(10.5, 17.5, 17.5, 10.5))
    full.set_title(f"{name}: full")
    zoom.set_title(f"{name}: rows/cols 11–17")

    # A pixel centered at (column 14, row 14) has boundaries 13.5–14.5 in each direction.
    # Rectangle adds a display border without changing the array's pixel values.
    zoom.add_patch(Rectangle((13.5, 13.5), 1, 1,
                             fill=False, edgecolor="red", linewidth=2))

fig.tight_layout()
plt.show()
```

`cmap="gray"` is a color map that displays numbers as grayscale brightness. Applying the same `vmin=0,vmax=1` to both images makes the same number appear with the same brightness. `interpolation="nearest"` helps distinguish pixel boundaries when displayed, and `origin="upper"` places the array’s first row at the top. These settings do not predict pixels or remove noise.

For `Rectangle`, `fill=False` leaves the interior unfilled. Two contrasting borders, such as black and white, could also be overlaid, but that would still be a display setting. Understand coordinates and brightness ranges precisely; look up font sizes and border widths when needed.

## 14. Build a model and connect the image shapes

We can now assemble PyTorch layers instead of writing weights manually. A <strong>class</strong> defines an object’s structure and behavior, and `__init__` is the initialization method run when creating an object. `nn.Module` is the base class for neural-network models, and `nn.Sequential` is a container that executes registered layers in order.

This model’s constructor takes `in_features`, `hidden_features`, and `latent_features`: the input features, intermediate hidden features, and latent features per sample. The actual input tensor goes into `model(x)`, rather than the constructor. That call executes the computation defined in the model’s `forward()` method.

<strong>ReLU</strong> replaces negative values with 0 and preserves nonnegative values. <strong>Sigmoid</strong> maps outputs to the range 0–1. This example uses a final Sigmoid because its targets are scaled to 0–1. A Sigmoid output is not required for every AE.

```python
import torch
from torch import nn

class DenseAutoencoder(nn.Module):
    def __init__(self, in_features, hidden_features, latent_features):
        super().__init__()  # Initialize nn.Module.

        # Each Linear uses the default bias=True, learning both weights and biases.
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, latent_features),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N,F), z: (N,K), reconstruction: (N,F).
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

model = DenseAutoencoder(64, 32, 16)  # Create the layers and parameters.
```

This model computes `(N,64) → (N,32) → (N,16) → (N,32) → (N,64)`. `model.parameters()` supplies all weights and biases registered in the encoder and decoder, allowing one optimizer to manage both parts.

### Connect an image of MNIST size

A single-channel `28×28` image contains `1×28×28=784` values. For this case, create `DenseAutoencoder(784,128,64)` and combine the batch’s image dimensions into a feature dimension.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/image-shapes-en.svg" alt="Flattening a noisy image batch of shape (N,1,28,28) per sample gives (N,784). The encoder produces (N,64), and the decoder computes (N,784). These values are arranged back into (N,1,28,28) and compared with the clean target." loading="lazy">
<figcaption>Flatten changes the pixel arrangement, while the neural network computes new pixel values. The final reshape arranges those computed values into image form. The sample dimension N is preserved.</figcaption>
</figure>

```python
# noisy_batch and clean_batch are (N,1,28,28) tensors from corresponding originals.
mnist_model = DenseAutoencoder(784, 128, 64)
loss_fn = nn.MSELoss()

flat_input = noisy_batch.flatten(start_dim=1)          # (N,784).
flat_reconstruction = mnist_model(flat_input)          # (N,784).
reconstruction = flat_reconstruction.reshape_as(noisy_batch)

assert reconstruction.shape == clean_batch.shape       # (N,1,28,28).
loss = loss_fn(reconstruction, clean_batch)
```

`flatten(start_dim=1)` keeps sample dimension 0 and combines the remaining dimensions. `reshape_as(noisy_batch)` changes the shape to match the reference tensor while preserving element order. The decoder has already computed the reconstructed pixel values. Reshape does not predict a new reconstruction.

## 15. Train and evaluate without a separate image file

The runnable example uses the <strong>digits</strong> dataset included with scikit-learn. <strong>This is a different dataset from MNIST.</strong> Its images are `8×8`, with original pixel values ranging from 0 to 16. Dividing the 1,797 images by `16.0` scales them to 0–1, with 64 features per image.[^digits]

| Item | Earlier MNIST-size explanation | Runnable digits example |
| --- | --- | --- |
| Image size | `28×28` | `8×8` |
| Flattened feature count | 784 | 64 |
| Model configuration | `DenseAutoencoder(784,128,64)` | `DenseAutoencoder(64,32,16)` |
| External NPZ file | Used only in the file-loading discussion | Not needed |

The execution sequence is <strong>split original data → create training batches → add any required noise only to the input → compare reconstruction with clean → backpropagate and update → evaluate on held-out data</strong>. Split originals first so that clean and noisy versions of the same original do not end up in both training and evaluation data.

A <strong>dataset</strong> is an object organizing samples so that they can be retrieved. `TensorDataset` groups tensor values with the same first-dimension index into one sample. A <strong>DataLoader</strong> creates batches from a dataset. `batch_size=64` means at most 64 samples per batch, and `shuffle=True` shuffles sample order on each pass.

An <strong>epoch</strong> is one pass through the training data. It differs from a <strong>training step</strong>, which processes one batch and updates the weights. This example groups 1,437 training samples into batches of up to 64, giving 23 batches per epoch. The final batch contains 29 samples.

<details>
<summary>The full training function, from model creation to evaluation</summary>

Run this after the `DenseAutoencoder` definition in the previous section. The function uses the same architecture for the basic AE and DAE. It adds noise to the input only when `denoise=True`; the target is always clean.

```python
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader

def run_digits(denoise=False, epochs=20):
    # A seed sets the random starting state, allowing comparisons in the same environment.
    torch.manual_seed(42)

    # (1797,8,8) → (1797,64). Scale original pixels from 0–16 to 0–1.
    X = torch.tensor(load_digits().images, dtype=torch.float32)
    X = X.flatten(start_dim=1) / 16.0

    # randperm places indices 0 through len(X)-1 in a random order.
    indices = torch.randperm(len(X))
    split = int(len(X) * 0.8)               # 1,437 training samples; 360 evaluation samples.
    train_x = X[indices[:split]]             # The endpoint split is excluded.
    test_x = X[indices[split:]]

    loader = DataLoader(TensorDataset(train_x), batch_size=64, shuffle=True)
    model = DenseAutoencoder(64, 32, 16)     # Create a new model on each function call.
    loss_fn = nn.MSELoss()                   # Mean squared error over all corresponding elements.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):              # Iterate from 0 through epochs-1.
        model.train()
        total_loss = 0.0

        # TensorDataset received one tensor, so it returns a one-element collection.
        # (clean_batch,) unpacks the actual batch tensor from that collection.
        for (clean_batch,) in loader:
            if denoise:
                input_batch = (
                    clean_batch + 0.25 * torch.randn_like(clean_batch)
                ).clamp(0, 1)
            else:
                input_batch = clean_batch

            optimizer.zero_grad(set_to_none=True)     # Clear previous gradients.
            reconstruction = model(input_batch)       # New forward pass.
            loss = loss_fn(reconstruction, clean_batch)
            loss.backward()                            # Compute gradients.
            optimizer.step()                           # Update weights and biases.

            # Feature counts match across samples, so weight the recorded loss by batch size.
            # Do not give the final small batch the same weight as a 64-sample batch.
            total_loss += loss.item() * len(clean_batch)

        if (epoch + 1) % 5 == 0:
            print(f"epoch={epoch + 1}, train_MSE={total_loss / len(train_x):.6f}")

    # Fix evaluation noise with a separate Generator, independent of training randomness.
    generator = torch.Generator().manual_seed(123)
    if denoise:
        test_input = (
            test_x + 0.25 * torch.randn(test_x.shape, generator=generator)
        ).clamp(0, 1)
    else:
        test_input = test_x

    model.eval()
    with torch.no_grad():
        z = model.encoder(test_input)                  # (360,16).
        reconstruction = model.decoder(z)              # (360,64).
        model_mse = loss_fn(reconstruction, test_x).item()
        baseline_mse = loss_fn(test_input, test_x).item()

    print("Input / latent / reconstruction:", test_input.shape, z.shape, reconstruction.shape)
    print("Model MSE:", model_mse)
    if denoise:
        print("Unprocessed noisy MSE:", baseline_mse)
        print("MSE improvement:", baseline_mse - model_mse)

    return model, test_input, reconstruction, test_x

# Call the task you want. Defining the function above does not start training.
ae_model, ae_input, ae_output, ae_target = run_digits(denoise=False, epochs=20)
# dae_model, dae_input, dae_output, dae_target = run_digits(denoise=True, epochs=20)
```

</details>

Extracting a Python number with `loss.item()` is for logging. Backpropagation uses the original tensor, `loss`, before conversion. The epoch MSE printed during training combines losses from multiple model states as batches update the model. It is not always the same as applying the final model from that epoch to the entire training dataset again.

### Completing training does not establish successful denoising

A <strong>baseline</strong> is a basic result used to judge model performance. For a DAE, the error obtained by returning the noisy input unchanged can serve as the baseline.

| Compared result | Calculation | Meaning |
| --- | --- | --- |
| Unprocessed baseline | `MSE(noisy,clean)` | Error when the noisy input is left unchanged |
| Model result | `MSE(reconstruction,clean)` | Error after passing the input through the model |
| Improvement | `baseline MSE−model MSE` | A positive value means lower mean error in this evaluation |

The following results were obtained by <strong>running the download code’s default settings on CPU for 20 epochs</strong>. The data were split into 1,437 training images and 360 evaluation images. Model creation and training used seed 42; evaluation noise used seed 123, with a noise multiplier of 0.25.

| Task | Model evaluation MSE | Unprocessed noisy baseline MSE | Interpretation |
| --- | ---: | ---: | --- |
| Basic AE | 0.03209316 | Not applicable | Reconstruction error for clean inputs |
| DAE | 0.03996541 | 0.03788888 | Under these settings, the model output has higher mean error. |

The DAE improvement is approximately `−0.00207653`. <strong>This experiment does not support a claim of successful denoising.</strong> Running training and computing a loss are separate from improving on the baseline. The AE and DAE also receive different input conditions, so their numbers should not be read as a ranking on an identical task.

The verification environment used Python 3.12.14, PyTorch 2.15.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0, and Matplotlib 3.10.8, with one CPU computation thread. Final decimal places or training results can vary across environments. The table reports observations from this run, not guaranteed values for every run.

<figure class="ae-figure">
<img src="/autoencoder-denoising-basics/digits-dae-en.svg" alt="The first four evaluation samples from the trained DAE. Each row shows the same 8×8 digits original as a noisy input, an actual model reconstruction, and a clean target. Across all 360 images, model MSE is higher than unprocessed-input MSE." loading="lazy">
<figcaption>The first four images in the evaluation array are shown, without selecting favorable examples. All three columns use the same brightness range, 0–1. This is an actual digits model result, separate from the MNIST file-inspection display.</figcaption>
</figure>

### What to check when running the code

Run the downloaded `examples-en.py` in a Python environment with PyTorch installed. The digits example also needs NumPy and scikit-learn; saving figures requires Matplotlib. This CPU example does not require a GPU.

| Command | What it does |
| --- | --- |
| `python examples-en.py --mode tiny` | Checks the small forward pass, one update, batch calculation, and Q1/Q2. |
| `python examples-en.py --mode ae --epochs 20` | Creates, trains, and evaluates a new basic AE. |
| `python examples-en.py --mode dae --epochs 20 --output-dir figures` | Trains and evaluates a DAE and saves result figures to the specified directory. |
| `python examples-en.py --pair mnist_day1_pair.npz --output-dir figures` | Also displays the input and target from a separately supplied NPZ file. |

`--mode` selects the task, `--epochs` the number of training passes, `--output-dir` the results directory, and `--pair` an optional NPZ path. If the interpreter command in your environment is `python3`, use `python3` pointing to that same environment. The commands in this table have the same form in Bash and zsh.

If you repeatedly choose model size or training duration by examining evaluation results, you are using that data as <strong>validation data</strong>. For final performance reporting, keep separate <strong>test data</strong> that did not inform those choices. If an image pair was set aside for validation inspection, using it directly for training changes its original validation role.

## 16. Compare the results of two weight configurations

Now calculate two configurations with the same input but different weights. We will call them Q1 and Q2. These are separate from the earlier experiment that made one update using A.

The shared input and target are `B=[2,1]`: one sample with two features and a `2→1→2` structure. There are no biases or activation functions, and the loss is MSE over the two reconstructed components.

| Configuration | Encoder weights `e` | Decoder weights `d` |
| --- | --- | --- |
| Q1 | `[0.2,0.4]` | `[1.0,1.5]` |
| Q2 | `[0.38,0.49]` | `[1.096,1.484]` |

<strong>Both the encoder and decoder differ in Q2.</strong> Computing Q2’s latent value and then applying Q1’s decoder, `[1.0,1.5]`, does not calculate the given Q2 configuration.

The calculation follows the same order as before. First, multiply the two input features by their encoder weights and add the results. Then multiply that latent value by each decoder weight to produce two reconstructed components. Finally, compare them with the corresponding target components.

| Calculation stage | Q1 | Q2 |
| --- | --- | --- |
| Latent value `z=e₁x₁+e₂x₂` | `0.2×2+0.4×1=0.8` | `0.38×2+0.49×1=1.25` |
| First reconstruction `d₁z` | `1.0×0.8=0.8` | `1.096×1.25=1.37` |
| Second reconstruction `d₂z` | `1.5×0.8=1.2` | `1.484×1.25=1.855` |
| First component squared error, target 2 | `(0.8−2)²=1.44` | `(1.37−2)²=0.3969` |
| Second component squared error, target 1 | `(1.2−1)²=0.04` | `(1.855−1)²=0.731025` |
| MSE over the two components | 0.74 | 0.5639625 |

The first reconstruction changes from `0.8` to `1.37`, moving closer to target 2. The second changes from `1.2` to `1.855`, moving farther from target 1. The decrease in the first squared error exceeds the increase in the second, so the mean decreases.

The conclusion is that <strong>Q2 has lower mean reconstruction error on the same input B</strong>. This does not establish that every output component improved, that Q2 is better on other inputs, or that a larger latent value indicates a better model.

Different weights can also produce the same output on a particular input if their changes cancel. Compute the actual reconstruction and error instead of stopping at the fact that the weights changed. Here, we compared forward passes under two supplied configurations. These calculations alone do not reveal the training process that produced those weights.

## 17. Compute the same forward pass in Rust

Python and Rust can use the same encoder–decoder relationships and MSE definition. The distinction here is how the examples are implemented. The Python PyTorch tensors used automatic differentiation, while the Rust code below uses only fixed-length arrays and floating-point operations from the standard library. Therefore, <strong>this Rust example computes a forward pass without automatic differentiation or training</strong>. Rust can also implement training through a library that provides automatic differentiation.

`[f64;2]` is an array of two 64-bit floating-point values. The return type `(f64,[f64;2],f64)` returns the latent value, reconstructed array, and MSE in one <strong>tuple</strong>. `.powi(2)` squares a value using the integer exponent 2.

```rust
// Run this Rust forward-pass example as a separate file. It uses no external libraries.
fn tiny_ae(
    x: [f64; 2],
    e: [f64; 2],
    d: [f64; 2],
) -> (f64, [f64; 2], f64) {
    // Multiply the two features by their encoder weights and add the results.
    let z = e[0] * x[0] + e[1] * x[1];

    // Compute two reconstructed components separately from the same z.
    let reconstruction = [d[0] * z, d[1] * z];

    // Average the squared errors of the two corresponding components.
    let mse = (
        (reconstruction[0] - x[0]).powi(2)
        + (reconstruction[1] - x[1]).powi(2)
    ) / 2.0;

    (z, reconstruction, mse)
}

fn main() {
    let x = [2.0, 1.0];
    let conditions = [
        ("Q1", [0.2, 0.4], [1.0, 1.5]),
        ("Q2", [0.38, 0.49], [1.096, 1.484]),
    ];

    for (name, e, d) in conditions {
        let (z, reconstruction, mse) = tiny_ae(x, e, d);
        // :.3 and :.7 control displayed decimal places without changing the computed values.
        println!("{name}: z={z:.3}, reconstruction={reconstruction:?}, MSE={mse:.7}");
    }
}
```

Q1’s MSE is `0.7400000` and Q2’s is `0.5639625`, matching Python. Printing the reconstructed array can reveal small differences in the final digits due to floating-point representation. The downloadable Rust file also includes checks for these results.

## 18. What to check in an actual application

Before changing the model, check how the data and computations connect.

| Question to check | Where to inspect the code |
| --- | --- |
| Does the first input dimension represent samples or channels? | Data-creation code, `shape`, and actual indexing |
| Does the encoder output connect to the decoder input? | `z=encoder(x)`, `decoder(z)` |
| Is the DAE loss target clean? | `loss_fn(reconstruction,clean_batch)` |
| Has the computation graph been disconnected midway? | Uses of `tensor(...)`, `.item()`, and `detach()` while computing the loss |
| Have both gradient computation and updates run? | The order of `zero_grad`, `backward`, and `step` |
| Has an old result been mistaken for a result from the updated model? | A fresh forward pass after the update; whether file loading succeeded |
| Has performance improved on data not used for training? | Data-splitting criteria, model MSE, and the noisy baseline |
| Does the figure show an actual model output? | The distinction between displaying clean arrays and results from `model(input)` |

Trace the input, latent value, reconstruction, and loss directly in the small model. Preserve those same connections in batch processing and evaluation code for a practical model. When trying the next model, start by checking input–target correspondence, the paths connected to the loss, and the result relative to a baseline.

## References

The numerical examples were checked using the inputs and weights specified in the article. The API and concept explanations were checked against the original sources below. Documentation was consulted on 2026-11-02.

[^ae]: Goodfellow, Bengio, Courville, [Deep Learning — Chapter 14: Autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html). Encoders and decoders, representation learning, small bottlenecks, and other constraints.
[^dae]: Vincent et al., [Stacked Denoising Autoencoders](https://www.jmlr.org/papers/v11/vincent10a.html), JMLR, 2010. Original research using reconstruction from corrupted input as a training objective.
[^keras]: Keras, [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder/). A training example connecting noisy inputs with clean targets.
[^linear]: PyTorch, [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html). Constructor, input/output shapes, and weight dimension order.
[^mse]: PyTorch, [MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html). Elementwise squared errors and the default averaging scope.
[^tensor]: PyTorch, [torch.tensor](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html). Copy construction, automatic-differentiation history, and guidance on tensors sharing NumPy storage.
[^stack]: PyTorch, [torch.stack](https://docs.pytorch.org/docs/stable/generated/torch.stack.html). Combining tensors by adding a new dimension.
[^autograd]: PyTorch, [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html). Operation recording, gradient accumulation, evaluation mode, and differentiation modes.
[^no-grad]: PyTorch, [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html). A context manager that omits recording for gradient computation.
[^detach]: PyTorch, [Tensor.detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html). Disconnecting automatic-differentiation history while sharing storage.
[^clone]: PyTorch, [torch.clone](https://docs.pytorch.org/docs/stable/generated/torch.clone.html). A differentiable copy operation.
[^numpy]: NumPy, [numpy.load](https://numpy.org/doc/stable/reference/generated/numpy.load.html). Reading NPZ files, closing files, and configuring `allow_pickle`.
[^imshow]: Matplotlib, [Axes.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html). Brightness ranges, origin, display coordinates, and interpolation.
[^plot]: Matplotlib, [Axes.axhline](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.axhline.html), [Axes.annotate](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html). Distinguishing data coordinates from display coordinates.
[^digits]: scikit-learn, [load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). Sample count, image size, and pixel range.
[^embedding]: Sentence Transformers, [Semantic Textual Similarity](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html). Connecting embedding generation with similarity calculation.
[^compression]: TensorFlow Compression, [Learned data compression](https://tensorflow.github.io/compression/). Bitstreams and entropy coding for learned compression.
