+++
title = "Weekly Paper 04: Strengths and Weaknesses of Boosting and Bagging Models"
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

#### 1. How Bias and Variance Work {#research-bias-variance .research-topic}

A model's total prediction error consists of bias² + variance + noise (irreducible error).

- **Bias:** Indicates how accurately the model reflects the training data.
  - **How it works:** If a model is too simple (for example, predicting complex curved data with a straight line), it cannot learn the underlying relationships in the data, so the difference between actual and predicted values (bias) increases. This is called **underfitting**.
- **Variance:** Indicates how much a model's predictions fluctuate (variability) when it is given different training data.
  - **How it works:** If a model is too complex (for example, a high-degree polynomial or a deep neural network), it memorizes all the random noise in the data. It fits the training data perfectly, but its predictions swing wildly when new data comes in. This is **overfitting**.

#### 2. The Bias–Variance Trade-off {#research-tradeoff .research-topic}

Increasing model complexity reduces bias but increases variance. Conversely, reducing complexity decreases variance but increases bias. The goal of model training is to find the optimal balance that minimizes the sum of the two.

### 2) My Answer

The model has 99% accuracy on the training data but 70% on the test data, so I think this is overfitting.
Overfitting is more of a variance problem than a bias problem.

We therefore need to reduce variance, and I learned that L1 and L2 regularization are often used to do this. As a model becomes more complex, its understanding of the problem improves. However, if it learns too closely from the data, overfitting occurs. L1 and L2 regularization reduce overfitting by lowering the model's complexity a little. With lower complexity, the model may miss detailed changes in the data, so there is a trade-off where accuracy can decrease.

My understanding of regularization is that it deliberately adds a penalty term to the loss function to interfere with learning a little. The model cannot simply keep increasing its weights; it has to find the best weights again while also taking this penalty into account. This is how I understand it reduces variance.

## Question 2

Random forests and gradient boosting both combine multiple decision trees. How do they differ in the way they build and combine trees? What do you expect would happen to each model if you kept increasing the number of trees?

### 1) Google Research Notes

#### 1. The Main Differences in How Trees Are Combined {#research-ensemble-comparison .research-topic}

| Aspect | Random Forest | Boosting |
| --- | --- | --- |
| Ensemble type | Bagging: Bootstrap + Aggregation | Sequential boosting |
| Tree creation order | Independent: multiple trees are created simultaneously in parallel | Sequential: each new tree corrects the errors of the previous tree |
| Training data | Different samples drawn with replacement (bootstrap sampling) + random feature selection | Trains with higher weights on data the previous model got wrong (large errors) |
| How results are combined | Majority voting for classification or averaging for regression | Sums the weighted predictions of each tree |
| Main purpose | Reduces variance: prevents overfitting and improves stability | Reduces bias: aims for high accuracy by reducing prediction errors |

#### 2. Detailed Comparison {#research-ensemble-details .research-topic}

##### ① Random Forest (Bagging-based) {.research-detail}

- **How it works:** Multiple independent decision trees are created simultaneously using subsamples drawn with replacement from the original data. At each split node, only some features are randomly selected, rather than using all features, to make the trees more diverse.
- **How results are combined:** The results from all trees are collected, and the final decision is made by majority vote for classification or by averaging for regression.
- **Advantages:** Individual trees have a low risk of noise or overfitting, parallel processing is easy, and hyperparameter tuning is relatively easy.

##### ② Boosting (Boosting-based) {.research-detail}

- **How it works:** After the first tree is created, greater weight is assigned to the data it did not predict well (errors). The second tree focuses on learning these highly weighted errors, and this process is repeated sequentially (for example, using gradient descent).
- **How results are combined:** Weights are assigned to the trees' predictions in the order the trees were created, and the final prediction is obtained by accumulating them.
- **Advantages:** Generally achieves higher predictive accuracy than random forests and learns complex patterns in the data well.
- **Disadvantages:** Trees must be created sequentially, making parallel processing difficult and training time long. The risk of overfitting noisy data is relatively high.

#### What Happens When More Trees Are Added {#research-number-of-trees .research-topic}

##### 1. Bagging and Parallel Combination {.research-detail}

Bagging, represented by random forests, is a parallel combination method that trains multiple trees independently and then averages their predictions or takes a majority vote.

- **There is no risk of overfitting.** Even if the number of trees is increased without limit, overfitting caused by increasing model complexity does not occur.
- **Variance decreases, and error converges to a particular lower bound.** As trees are added, the randomness (noise) of individual trees cancels out, stabilizing predictions. Beyond a certain number of trees, performance reaches a plateau where it neither improves nor worsens.
- **Only computing costs increase.** Performance stagnates while memory usage and computation time increase unnecessarily.

##### 2. Boosting and Sequential Combination {.research-detail}

Boosting, represented by Gradient Boosting, XGBoost, and LightGBM, is a sequential combination method in which the next tree corrects the errors of the previous tree.

- **The risk of overfitting is very high.** If trees keep being added, the model learns even the tiniest amounts of noise and all the outliers in the training data.
- **Bias becomes extremely low, but variance increases.** The model reaches a point where it memorizes the patterns in the data perfectly. Its training error approaches 0, but its predictive performance on new data (test data) deteriorates drastically.
- **Early stopping is essential to prevent this.** Further trees must be prevented from being added once the loss on the validation data no longer decreases.

### 2) My Answer

A random forest runs several decision trees in parallel and uses the average of their results, while boosting combines decision trees in sequence to learn by correcting the previous model's errors.
Because a random forest uses an average of several results, I think it would have some resistance to overfitting even with more trees. Boosting, on the other hand, immediately continues learning from the residuals to correct them, so I think it learns more directly. However, if we add more trees, it may learn the training data too closely, making overfitting relatively more likely than with a random forest.
