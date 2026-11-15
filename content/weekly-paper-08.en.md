+++
title = "Weekly Paper 08"
date = 2026-11-15
slug = "weekly-paper-08"
+++

## Question 1

I trained a model to classify whether products are defective using 5,000 images. Its accuracy is very high on the training data, but its performance drops sharply on new images in practice. Taking additional images is difficult. What methods would you try?

### My answer

This looks like overfitting. The model seems to have adapted too closely to the images it was trained on and struggles to classify new images. I would also check whether the lighting or camera angles differ substantially between the training images and the images used in practice.

If taking more images is difficult, I would try data augmentation and train again. This means creating variations of an existing image, such as rotating it slightly or adjusting its brightness, and using those variations for training. This may help the model become familiar with changes that can occur when taking real photographs.

However, I would only convert images to grayscale or apply filters when the features needed to identify defects are preserved. If a difference in color is a clue to a defect, converting the image to grayscale could remove that information. A filter could also erase a small scratch. Rotations and brightness adjustments should likewise preserve these clues and the original label.

I would split the original images into training and validation sets first, then apply augmentation only to the training images. I would keep variations of the same original image from appearing in both sets and check whether performance also improves on the held-out validation images.

## Question 2

I want to train a classification model from scratch using 300 medical images to identify a rare disease. What problems could this approach have, and what alternatives would you suggest?

### My answer

Three hundred images seem like a small amount of data for training an image classification model from scratch. It would depend on the model and the diversity of the images, but a model with many weights to learn may struggle to learn the features of the disease from this amount of data. There may also be a high risk of overfitting, where the model performs well on the images it was trained on but fails to classify images from other patients correctly.

As an alternative, I would try transfer learning with a model already trained on other data. I would initially leave the feature-extraction part unchanged, replace the final classification part to predict whether the disease is present, and train that new part. This reuses features the model has already learned while reducing the number of weights that need to be learned, which may help when data is limited compared with learning all the weights from scratch.

Transfer learning does not guarantee better performance, so I would check the results on images that were not used for training. If there are several images from the same patient, I would keep all of that patient's images together in either the training set or the validation set.
