+++
title = "Weekly Paper 05: Vanishing Gradients and Clearing Gradients"
date = 2026-10-25
slug = "weekly-paper-05"
+++

<style>
#main-content article h4.research-topic {
  margin: 2.75rem 0 .65rem;
  font-size: 1.4rem;
  font-weight: 750;
  line-height: 1.5;
  text-wrap: pretty;
  word-break: keep-all;
  overflow-wrap: anywhere;
}
#main-content article h3 + h4.research-topic {
  margin-top: 1.25rem;
}
#main-content article h5.research-detail {
  margin: 1.5rem 0 .6rem;
  font-size: 1.125rem;
  font-weight: 650;
  line-height: 1.55;
}
#main-content article h4.research-topic + h5.research-detail {
  margin-top: 0;
}
</style>

## Question 1

You built a deep neural network with 20 layers and used sigmoid as the activation function. During training, the layers near the output learn well, but the weights of the layers near the input barely change even after several epochs. What cause would you suspect? How would you check and resolve it?

### 1) Google Research Notes

#### 1. Googling Why Neural Networks Have Multiple Layers {#research-network-depth .research-topic}

##### 1. Hierarchical Feature Extraction (Step-by-step Abstraction) {.research-detail}

As layers are stacked, the nature of the data changes at each stage. An image model, for example, can learn features such as the following.

- **Lower layers (near the input):** Find simple, local features such as lines, points, and edges.
- **Middle layers:** Identify partial forms such as eyes, noses, ears, or textures that combine simpler features.
- **Higher layers (near the output):** Learn complex, higher-level meanings (abstractions) representing a whole, such as a human face or a car.

##### 2. Solving Complex Nonlinear Problems {.research-detail}

- Real-world data, such as images, audio, and natural language, has a highly complex structure that cannot be separated by a single straight line.
- Stacking multiple hidden layers and placing activation functions, such as ReLU or sigmoid, between them makes it possible to create decision boundaries that flexibly separate even complex, intertwined data.
- If only linear operations are stacked without activation functions, no matter how deep the network becomes, they can ultimately be combined into a single linear transformation and a bias.

##### 3. Parameter Efficiency and Computational Savings {.research-detail}

- A shallow, wide network may require an enormous number of nodes, making computation inefficient.
- A deeper network reuses simple features combined in earlier layers, so it can achieve much more complex and extensive expressive power with fewer neurons.

#### 2. Googling the Weights of Layers Near the Input {#research-early-layer-weights .research-topic}

##### 1. Characteristics and Roles {.research-detail}

- **Detecting basic patterns:** In an image model, the weights of the first hidden layer after the input can learn the most basic, local features, such as edges, lines, and color changes.
- **Reflecting data dimensions:** If there are N input features and M neurons in the first hidden layer, the weight matrix has shape `(N, M)` under the `X @ W` convention, where the input is multiplied by the weights on the right. PyTorch's `nn.Linear` stores its weights with shape `(M, N)`.

##### 2. A Problem in Deep Neural Networks: Vanishing Gradients {.research-detail}

- **Vanishing gradients:** During backpropagation in a deep neural network (DNN), gradients are propagated from the output toward the input.
- As these operations accumulate, gradients in layers closer to the input can become extremely small, preventing their weights from being updated effectively.
- **Solutions:** To mitigate this problem, use ReLU-family activation functions or suitable weight initialization methods, such as He initialization or Xavier initialization.

#### 3. Googling the Vanishing Gradient Problem {#research-vanishing-gradients .research-topic}

##### 1. Main Causes {.research-detail}

- **Saturating activation functions:** The derivatives of sigmoid and hyperbolic tangent (tanh) approach 0 when their inputs are very large or very small.
- **Repeated application of the chain rule:** Backpropagation involves multiplying derivatives, weights, and other factors across layers. If the effect of shrinking the gradient is repeated, the gradient can become exponentially smaller.
- **Deep network structure:** As the network gets deeper, gradients in the early layers near the input can become extremely small, causing learning to stall.
- **Recurrent neural networks (RNNs):** Over longer sequences of time steps, gradients propagated to earlier time steps can become smaller, making long-term dependencies difficult to learn.

##### 2. Main Solutions and Mitigation Methods {.research-detail}

- **Change the activation function:**
  - Use ReLU, whose derivative is always 1 for positive inputs, or Leaky ReLU, which also has a small positive derivative for negative inputs.
- **Weight initialization:**
  - Use Xavier (Glorot) initialization for sigmoid/tanh networks and He initialization for ReLU networks to adjust the initial variance appropriately.
- **Batch normalization:**
  - Normalize values for each feature within a batch to reduce the tendency for activations to concentrate in a particular range.
- **Residual connections / skip connections:**
  - Use a structure that adds the input directly to the output, as in ResNet, to create a path through which gradients can be propagated directly to earlier layers during backpropagation.
- **Improve the architecture (replace a basic RNN):**
  - Use LSTMs or GRUs, which introduce gating structures, to mitigate the vanishing gradient problem in RNNs.

#### 4. Googling Sigmoid and Backpropagation Problems {#research-sigmoid-backpropagation .research-topic}

##### 1. The Sigmoid Function and Its Derivative {.research-detail}

- **Sigmoid function (σ(x)):** It takes the form `σ(x) = 1 / (1 + e^(−x))` and compresses input values into the interval between 0 and 1.
- **Derivative (σ′(x)):** Differentiating it gives `σ′(x) = σ(x)(1 − σ(x))`.

##### 2. A Problem During Backpropagation: Vanishing Gradients {.research-detail}

- **Upper limit on the derivative:** The sigmoid derivative reaches its maximum of 0.25 when the input is 0. As the input moves farther from 0, becoming very large or very small, the derivative approaches 0.
- **Repeated multiplication through the chain rule:**
  - During backpropagation, local gradients are repeatedly multiplied across layers.
  - Repeatedly multiplying sigmoid derivatives of at most 0.25 across many hidden layers can contribute to gradients becoming extremely small.
- **Stalled learning:** Weights in earlier hidden layers may barely update, preventing the neural network from learning effectively.

##### 3. Solution {.research-detail}

- **Use ReLU:** Use an activation function such as ReLU, whose derivative is always 1 for positive inputs, to mitigate vanishing gradients. Related material can be found in Deep Learning Basics — Forward Propagation and Backpropagation.

### 2) My Answer

I'm not sure, but based on what I found through searching, backpropagation starts at the output layer and moves toward the input while calculating gradients by differentiating the loss with respect to the weights. As it gets closer to the input, the gradients become so small that the update amounts in `w -= lr * grad` also become too small, so I think the weights are not being updated properly.

The search results simply say to use ReLU as a solution, but I don't really understand why it works. One thought I have right now is to use different learning rates near the output and near the input, making the update amounts larger as we get closer to the input.

## Question 2

You left out the line `optimizer.zero_grad()` in a PyTorch training loop, and now the loss does not decrease properly and jumps around unexpectedly. What does this line do, and why does leaving it out cause problems?

### My Answer

Gradient values are filled in after backpropagation, and these values need to be cleared again after updating the weights. Otherwise, the gradients calculated for the previous batch remain, and when gradients are calculated for the next batch, they are added to the existing values and stored as accumulated gradients.

The weight update then uses accumulated gradients rather than the current batch's gradients, which can have a large negative effect on the update. As a result, training does not work properly.
