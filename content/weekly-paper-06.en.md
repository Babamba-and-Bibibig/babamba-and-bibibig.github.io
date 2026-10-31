+++
title = "Weekly Paper 06: Weight Sharing in CNNs and Vanishing Gradients in RNNs"
date = 2026-10-31
slug = "weekly-paper-06"
+++

## Question 1

You built an image classification model using only fully connected layers, and the number of parameters exceeded several million. The predictions also change significantly when the position of an object in a photo shifts even slightly. Why do these problems occur, and how do CNNs address them?

### My Answer

Connecting the pixel values in each image channel to every neuron in the next layer with separate weights requires too many weight parameters. This makes computation difficult, and I expect that having so many weights to learn could also make overfitting more likely when there is not enough data.

In contrast, a CNN uses kernels that share weights across different positions. It applies a kernel to each input channel to scan the image, then combines the results from the channels to produce one feature map per filter. This allows it to learn with fewer weights than a fully connected layer, which can reduce the risk of overfitting and, depending on the architecture, can also reduce computation.

Also, because it scans the image using the same kernel weights, I expect it to be able to find the same features even if the object's position changes slightly.

## Question 2

You used a basic RNN to classify the sentiment of long movie reviews as positive or negative. However, information from important words at the very beginning of a sentence seems to have little influence when the model makes its prediction at the end. What cause would you suspect? How would you resolve it?

### My Answer

I think this could also be a vanishing gradient problem. During backpropagation, gradients can become smaller as they are propagated to earlier time steps. Since reaching the time steps at the beginning of the sentence requires passing through more computational steps, the model may not learn the features of the words at the beginning properly.

I think I should try replacing the basic RNN with an LSTM or GRU, which addresses this weakness, and then test it again and evaluate the results.
