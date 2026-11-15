+++
title = "Weekly Paper 08"
date = 2026-11-15
slug = "weekly-paper-08"
+++

## Question 1

I trained a model to classify whether products are defective using 5,000 images. Its accuracy is very high on the training data, but its performance drops sharply on new images in practice. Taking additional images is difficult. What methods would you try?

### My answer

This looks like overfitting. I would try augmenting the data and training again.

I would augment the data by creating several variations from one image, such as converting it to grayscale, applying different filters, or rotating it slightly. Learning from slightly different views of the same product may help the model classify new images a little better.

## Question 2

I want to train a classification model from scratch using 300 medical images to identify a rare disease. What problems could this approach have, and what alternatives would you suggest?

### My answer

Three hundred images seem like too few. I do not know which classification model is being used, but the size of the model and the amount of data need to be appropriate for each other, and 300 images seem like a small amount. The model may not learn well, and I think there is a high chance of overfitting.

As an alternative, I would try transfer learning by taking an already trained model and training it further on the medical images.
