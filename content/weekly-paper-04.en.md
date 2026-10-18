+++
title = "Weekly Paper 04: Strengths and Weaknesses of Boosting and Bagging Models"
description = "Examining the gap between training and test accuracy through bias, variance, and L1/L2 regularization, then comparing how random forests and gradient boosting combine trees and respond to adding more trees."
date = 2026-10-18
slug = "weekly-paper-04"
+++

<style>
/* Keep the heading sizes and spacing used in earlier Weekly Paper posts. */
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
#main-content article table th:first-child,
#main-content article table td:first-child {
  white-space: nowrap;
}
</style>

## Question 1

A model achieved 99% accuracy on the training data but 70% on the test data. Is this a bias or a variance problem? How do L1 and L2 regularization help reduce it?

### 1) Google Research Notes

#### How Bias and Variance Work {#research-bias-variance .research-topic}

- **Bias:** How far a model's average prediction deviates from the true relationship when the model is trained on different training datasets for the same problem. A model that is too simple may be unable to represent the relevant relationship, which can lead to high bias. Trying to predict a complex, curved relationship using only a straight line is one example. This failure to learn the relevant relationship well enough is called **underfitting**.
- **Variance:** How much a model's prediction for the same input changes when the training data changes. If a complex model follows chance fluctuations that are specific to its training data, its predictions can become sensitive to which data it was trained on. This is related to **overfitting**, where a model fits the training data well but performs poorly on new data.

In regression, if we imagine repeatedly drawing training data and new data from the same data-generating process, we can break down the average squared error as follows.

**Average squared error = Bias² + Variance + Noise**

Noise refers to random variation that is difficult to predict from the input information currently available. This is a decomposition of squared error; it is not a formula that directly splits classification accuracy into three terms.

References: scikit-learn's [bias–variance decomposition example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html) and [documentation on using training and validation scores to examine underfitting and overfitting](https://scikit-learn.org/stable/modules/learning_curve.html).

#### The Trade-off Between Bias and Variance {#research-tradeoff .research-topic}

A trade-off is a relationship in which improving one aspect can make another worse. Increasing model complexity allows a model to represent more complex relationships, but it also creates a risk of following chance fluctuations in the training data. Conversely, making a model too simple may make its predictions less variable while causing it to miss relevant relationships.

Model complexity should therefore not be chosen solely by how well the model fits its training data. It is also necessary to check whether it predicts well on data that was not used for training. This does not mean that bias and variance always change in fixed directions as complexity increases.

Reference: scikit-learn's [documentation on model complexity and validation curves](https://scikit-learn.org/stable/modules/learning_curve.html).

#### How L1 and L2 Regularization Reduce Overfitting {#research-regularization .research-topic}

A loss function calculates how far predictions are from their targets. Weights are values that multiply the inputs, and the model adjusts them during training. L1 and L2 regularization add a **penalty**, or extra cost, to the loss based on the size of the weights. The model chooses weights that make the sum of the prediction error and this cost smaller.

- **L1 regularization:** Adds the sum of the absolute values of the weights, multiplied by the regularization strength, to the loss. Some weights can become exactly 0, reducing the number of features the model uses.
- **L2 regularization:** Adds the sum of the squared weights, multiplied by the regularization strength, to the loss. It limits how large the weights become overall.

The regularization strength determines how much the penalty contributes. Whether a weight is positive or negative, a larger magnitude produces a larger penalty. Making weights larger to reduce the training error is therefore not always beneficial.

References: scikit-learn's [documentation on Ridge and L2 regularization](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) and [Lasso and L1 regularization](https://scikit-learn.org/stable/modules/linear_model.html#lasso).

Appropriate regularization can help reduce variance by limiting how closely the model fits its training data. Performance on new data may improve even if training performance decreases slightly. If regularization is too strong, however, the model may fail to learn relevant relationships and underfit.

Reference: scikit-learn's [example of training and validation performance at different regularization strengths](https://scikit-learn.org/stable/auto_examples/model_selection/plot_train_error_vs_test_error.html).

### 2) My Answer

My first suspicion would be overfitting: the model reached 99% accuracy on the training data but only 70% on the test data. It fits the data it learned from but does not predict new data well, so I would put this closer to a variance problem than a bias problem.

I learned that L1 and L2 regularization can help reduce this kind of overfitting. A model might be able to represent complex relationships, but it can still run into trouble if it tries too hard to match every small change in the training data. Regularization limits how closely it fits those details.

L1 and L2 regularization add a penalty to the loss function based on the size of the weights. L1 uses the sum of their absolute values, while L2 uses the sum of their squares. The model has to find weights that account for this penalty as well as the training error. My understanding is that L1 can make some weights 0 in the process, while L2 limits how large the weights become.

If regularization is too strong, though, the model could miss relationships it needs to learn. It would be helpful if the model predicted new data better even with slightly lower training accuracy, but I do not think stronger regularization is always better.

## Question 2

Random forests and gradient boosting both combine multiple decision trees. How do they differ in the way they build and combine trees? What do you expect would happen to each model if you kept increasing the number of trees?

### 1) Google Research Notes

#### The Main Differences in How Trees Are Combined {#research-ensemble-comparison .research-topic}

An **ensemble** combines the predictions of several models. A single tree here means one whole decision tree, containing internal nodes where splits are made and terminal nodes that produce predictions. Increasing the number of trees is different from increasing the number of nodes within one tree.

| Aspect | Random Forest | Gradient Boosting |
| --- | --- | --- |
| How trees are combined | Combines the results of multiple trees using a bagging-based approach | Adds corrections from new trees sequentially |
| Training order | Each tree is trained without waiting for the results of other trees | The next tree is trained based on the predictions of the complete model built so far |
| What each tree learns | Each tree learns to predict the original targets | Each tree learns corrections that reduce the errors of the current complete model |
| How data is used | By default, rows are sampled with replacement, and features are randomly selected at each split | Uses the input features but recalculates the correction targets at each stage |
| How predictions are combined | Averages predictions for regression; uses voting or probability averaging for classification | Accumulates corrections multiplied by the learning rate; for classification, converts the accumulated scores into probabilities and classes |
| Main effect | Reduces prediction variance by combining multiple trees | Helps reduce bias by correcting the remaining errors |

How classification results are combined varies by implementation. **The random forest classifier in scikit-learn averages each tree's predicted probabilities for each class, then selects the class with the highest average probability.** A class is a category that the model distinguishes.

References: scikit-learn's [documentation on ensemble models](https://scikit-learn.org/stable/modules/ensemble.html) and [how random forest classification predictions are calculated](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict).

#### Strengths and Weaknesses of Random Forests and Gradient Boosting {#research-ensemble-details .research-topic}

##### 1. Random Forests: Combining the Results of Separately Trained Trees {.research-detail}

Bagging is short for **Bootstrap Aggregating**. Training datasets are created by **sampling with replacement**, which allows the same row from the original data to be selected more than once. Models are then trained separately, and their results are combined.

Random forests also add randomness to feature selection. Features are the input attributes used for prediction. Whenever a split condition is chosen within a tree, some features are selected at random as candidates. Differences in the rows and features considered can cause the trees to produce different predictions. Settings control how many rows and features are used.

Each tree learns to predict the original targets from the data it is given. For a new input, each tree produces a prediction, and the complete model combines the results through averaging or voting.

- **Strengths:** Combining the results of different trees can reduce how strongly the final result depends on a single tree's prediction. Trees do not need to wait for earlier trees to finish training, so they can be trained in parallel.
- **Points to watch:** Individual trees can still overfit the training data. If many trees make similar mistakes, averaging may not reduce those mistakes enough. Memory use and computation also increase with the number and size of the trees.

Reference: scikit-learn's [documentation on how random forests work and support parallel processing](https://scikit-learn.org/stable/modules/ensemble.html#random-forests).

##### 2. Gradient Boosting: Correcting the Errors Remaining in the Overall Prediction {.research-detail}

Gradient boosting trains each new tree based on **the predictions of the complete model**, which combines the trees built so far. It does not consider only the errors of the immediately preceding tree.

For regression using squared error, the process starts from an initial prediction and repeats the following steps. A **residual** is the actual value minus the current prediction.

1. Use the current complete model to predict values for the training data, then calculate the residual for each row.
2. Train **one new decision tree** to predict these residuals using the original input features.
3. Multiply the new tree's predictions by the learning rate and add them to the existing overall predictions. The learning rate determines how much of the new tree's correction is applied.
4. Recalculate the residuals using the updated overall predictions, then train the next tree.

**New overall prediction = Existing overall prediction + Learning rate × New tree's prediction**

In general, the correction targets are calculated according to the loss function. They are obtained from **the negative gradient of the loss**, which indicates the direction and magnitude of a change in the current predictions that would reduce the loss. With squared-error regression, this leads to learning the residuals.

Reference: scikit-learn's [documentation on gradient boosting regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).

- **Strengths:** New trees can improve predictive performance by correcting the parts that the current complete model does not predict well enough.
- **Points to watch:** Each stage needs the preceding overall predictions, so the trees cannot all be trained independently as they can in a random forest. The learning rate, tree size, and number of trees need to be considered together. Repeated corrections may also fit chance fluctuations in the training data.

Sequential training alone does not mean that boosting is always slower. Actual speed depends on the size of the data and the implementation. Neither method can be assumed to always have better predictive performance.

Reference: scikit-learn's [comparison of random forests and histogram-based gradient boosting](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_hist_grad_boosting_comparison.html).

#### What Happens If We Keep Adding Trees? {#research-number-of-trees .research-topic}

##### 1. Random Forests {.research-detail}

If only the number of trees is increased while other settings, such as tree depth, stay the same, averaging more predictions tends to reduce the influence of a small number of trees. However, this does not eliminate mistakes that the trees have in common.

Once there are enough trees, adding more often produces smaller performance gains. This does not mean that the actual validation score improves every time or becomes completely fixed after a certain point. Scores can rise and fall as trees are added.

Random forests are not immune to overfitting. **Increasing the number of trees is different from making individual trees deeper.** Even after performance gains become small, adding trees increases the number of trees that must be stored and the computation needed to make predictions.

References: Probst and Boulesteix's [paper on whether to tune the number of trees in random forests](https://jmlr.org/papers/v18/17-269.html) and scikit-learn's [documentation on random forest parameters](https://scikit-learn.org/stable/modules/ensemble.html#parameters).

##### 2. Gradient Boosting {.research-detail}

Adding trees in gradient boosting means continuing to correct the current predictions. This can further reduce the training error, but predictive performance on new data may worsen if the model starts fitting chance fluctuations that exist only in the training data.

Training performance and **validation performance** should therefore be examined together when adding trees. Validation data is not used directly for training; it is used to choose the model's settings. Its role is different from that of the test data used for the final evaluation.

**Early stopping** stops training when validation performance has not improved enough for a specified number of iterations. It can help reduce unnecessary tree additions and overfitting. The number of trees and the learning rate can also be chosen based on validation performance, so early stopping is not the only option.

Adding many trees does not necessarily cause overfitting. Even with the same number of trees, results depend on the learning rate and the size of each tree, so these settings need to be considered together.

References: scikit-learn's [early stopping example for gradient boosting](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_early_stopping.html) and [documentation on the relationship between the learning rate and the number of trees](https://scikit-learn.org/stable/modules/ensemble.html#shrinkage-via-learning-rate).

### 2) My Answer

In a random forest, each decision tree learns to predict the target, and the results are then combined. For regression, we average the predicted values. In the scikit-learn classification model I learned about, we average the probabilities for each class. The trees can be trained in parallel because they do not have to wait for one another's training results.

As the number of trees grows, there are more results to include in the average. Since the final result does not rely on just one tree, I would expect it to be less affected even if some trees fit the training data too closely.

Gradient boosting trains trees in sequence so that each new tree helps correct the errors of the complete model built so far. For regression with squared error, the new tree learns the remaining residuals, and its predictions are multiplied by the learning rate before being added to the existing predictions. Continuing to correct the remaining errors could make the model fit the training data better, but if this goes on too long, I think it could start fitting random changes that only exist in that data. That makes me think I need to pay closer attention to whether I should keep adding trees when using boosting.
