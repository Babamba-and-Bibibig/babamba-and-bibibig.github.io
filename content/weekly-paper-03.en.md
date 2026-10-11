+++
title = "Weekly Paper 03"
description = "Notes on what happens when the learning rate is too high or too low, and whether to use classification or regression to predict content age ratings."
date = 2026-10-11
slug = "weekly-paper-03"
+++

<style>
/* Use the same sizes for research topics and subtopics as in earlier Weekly Paper posts. */
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
/* Keep the short labels in the first column on one line. */
#main-content article table th:first-child,
#main-content article table td:first-child {
  white-space: nowrap;
}
</style>

## Question 1

What happens if you set the learning rate too high when training a model with gradient descent? What if you set it too low?

### 1) Research Notes

#### Google search: "gradient descent and learning rate" {#research-learning-rate .research-topic}

##### 1. What the Learning Rate Controls {.research-detail}

Loss is a value that measures how different a model's predictions are from the correct answers. Gradient descent repeatedly adjusts the weights and bias to reduce this loss. A weight is a value multiplied by an input, and a bias is a value added to the result.

The gradient indicates the direction and how much the loss changes when a weight or bias changes slightly. In gradient descent, we multiply the gradient by the learning rate and subtract the result from the current value.

**New weight = current weight − learning rate × gradient**

So, for the same gradient, a higher learning rate means a larger change in each update.

Reference: [Google Machine Learning Crash Course — Gradient Descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent). I checked the steps and calculations for updating the weights and bias.

##### 2. What Happens If It Is Too High or Too Low? {.research-detail}

| Learning rate | Possible problems |
| --- | --- |
| Too high | Each update may change the values too much and repeatedly pass a point with low loss. The loss may fail to decrease steadily or may keep increasing. |
| Too low | Each update changes the values so little that reducing the loss takes many iterations. The model may not learn enough within the set number of training iterations. |

The right learning rate depends on the model and the data. A higher value does not always mean faster learning, and a lower value does not always produce better results.

Reference: [Google Machine Learning Crash Course — Learning Rate in Hyperparameters](https://developers.google.com/machine-learning/crash-course/linear-regression/hyperparameters#learning-rate). I checked the explanations and graphs showing how the learning rate affects loss.

### 2) My Answer

I understand the learning rate as a value that controls how much the weights and bias change in one update. If it is too high, the updates can go past a point with low loss, and the model may struggle to learn. If it is too low, the changes are small, so learning can take a long time.

I would not choose a high learning rate just to make training faster. Setting it as low as possible is not the answer either, so I would first look at how the loss changes during training. I would try adjusting the learning rate while checking whether the loss keeps swinging up and down without decreasing, or is decreasing too slowly.

## Question 2

You are planning to build an AI model that predicts appropriate age ratings (such as ALL, 12, 15, and 19) from scripts for new content, including films and webtoons. You could approach this as a classification problem with four categories or convert the ratings into continuous numerical values and use regression. Which approach would you choose, and why? What are the advantages and disadvantages of each?

### 1) Research Notes

#### Google search: "the difference between classification and regression" {#research-classification-regression .research-topic}

- **Classification:** Predicts which category something belongs to. In this problem, it means choosing one of ALL, 12, 15, and 19 based on a script.
- **Regression:** Predicts a numerical value itself. To use it for this problem, the ratings need to be converted into numbers, and the predicted number needs to be converted back into one of the four ratings.

Classification models can also calculate numerical scores or probabilities internally. So, the presence of continuous numbers in the calculations does not make a task regression. What matters is what we use as the correct answer and what we are trying to predict.

References: Google's [Introduction to Machine Learning — Regression and Classification](https://developers.google.com/machine-learning/intro-to-ml/what-is-ml#supervised-learning) and [Classification](https://developers.google.com/machine-learning/crash-course/classification). I checked the difference between predicting numbers and predicting categories, and how probabilities can be used to decide on a category.

#### Google search: "predicting ordered ratings with regression" {#research-ordered-ratings .research-topic}

##### 1. The Order of Ratings and the Gaps Between Numbers Are Different {.research-detail}

The ratings in this problem have an order: ALL, 12, 15, and 19. But having an order does not mean the difference in how mature the content is can be expressed as a fixed numerical gap between ratings.

For example, I considered assigning 0 to ALL and keeping the others as 12, 15, and 19. The difference between 0 and 12 is 12, while the difference between 12 and 15 is 3. But the problem gives no reason to say that the first difference in content maturity is exactly four times the second. Using 0, 1, 2, and 3 would also involve a choice: treating the gaps between all adjacent ratings as equal.

If regression calculates error from the difference between the predicted number and the correct number, it can treat mistakes involving numerically close ratings differently from mistakes involving distant ones. But we also need to consider that the numbers assigned to the ratings affect the error calculation.

##### 2. Advantages and Disadvantages of Each Approach {.research-detail}

| Approach | Advantages | What to watch out for |
| --- | --- | --- |
| Classification | The model can be trained to distinguish directly between the four ratings we need. | Ordinary classification does not automatically account for the order of the ratings or the seriousness of predicting a rating that is far from the correct one. |
| Regression | The numbers can represent the order of the ratings, and the numerical distance of a wrong prediction can be reflected in the error. | We need to decide which numbers to assign to the ratings and what the gaps mean. We also need rules for converting a predicted number into an actual rating. |

For example, if regression predicts 13.8, we need a rule for deciding whether that means a rating of 12 or 15. This does not mean regression cannot be used, but predicting a number still leaves the step of deciding on a rating.

Also, simply giving an ordinary classification model the rating labels does not make it automatically understand their order. There are methods that predict ordered categories and account for their order during training, so ordinary regression is not the only way to use the order of the ratings.

References: [Pedregosa, Bach, and Gramfort, On the Consistency of Ordinal Regression Methods](https://jmlr.org/papers/volume18/15-495/15-495.pdf). I referred to the introduction and pages 10–11 for explanations of ordered categories and converting regression predictions into categories. I checked how numerical differences are reflected in error in [scikit-learn's explanation of mean squared error](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error). The age-rating example above is one I considered for this question.

### 2) My Answer

I would start with classification. The answer we need is one of ALL, 12, 15, and 19, and I am not sure the numbers in those ratings can be used directly as scores for how mature the content is. I would first try putting scripts together with their actual ratings and training a model to distinguish between the four ratings.

I think regression does have the advantage of using the order of the ratings and measuring how far a prediction is off through the difference between numbers. But I would also have to decide which numbers to assign and how to turn the predicted number into a rating, so I would prefer to start with classification.

If I started with ordinary classification, I would also need to remember that it does not automatically account for the order of the ratings. In particular, a prediction lower than the actual rating could expose younger audiences to content that is not appropriate for their age. I would put the actual and predicted ratings side by side and check these mistakes separately.

I think I would need to compare the two approaches on actual data to know which is more accurate.
