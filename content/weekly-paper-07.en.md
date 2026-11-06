+++
title = "Weekly Paper 07"
date = 2026-11-06
slug = "weekly-paper-07"
+++

## Question

I trained an autoencoder for image compression, but when I checked the results, it seemed to simply copy the input to the output. The reconstruction error is almost zero, yet it does not seem to have learned meaningful features. What might have gone wrong, and how would you fix it?

### My answer

The image produced by the decoder looks almost identical to the input, and the reconstruction error is almost zero, so it seems to reconstruct the input well. Looking only at the reconstruction, I would think the model had learned well. However, the question says it does not seem to have learned meaningful features. From the information given, it is hard to tell what that judgment is based on.

If the model reconstructs the images it was trained on very well but struggles to reconstruct other, similar images, I would suspect overfitting. However, an almost-zero reconstruction error alone is not enough to conclude that this is extreme overfitting. I would need to check whether the model also reconstructs images that were not used for training well.

I would also check whether I accidentally evaluated the model on training data, thinking it was validation data. Evaluating on training data again may give a good score, but that result alone does not show that the model can reconstruct unseen images well. I would check that the training and validation sets do not overlap, then evaluate the model again on a separate validation set.
