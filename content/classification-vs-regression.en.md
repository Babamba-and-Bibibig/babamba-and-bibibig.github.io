+++
title = "The Fundamental Difference Between Classification and Regression"
description = "Why classification and regression differ even when they use the same numerical calculations, explained through delivery times, animal categories, and content ratings. From the meaning of the target to loss calculations and practical choices."
date = 2026-10-11T17:42:50+09:00
slug = "classification-vs-regression"

[extra]
katex = true
toc = true
styles = ["classification-vs-regression/article.css"]
+++

One thing kept confusing me while I was studying classification and regression. Both take numbers as input, perform calculations, and adjust the values used in those calculations when a prediction is wrong. Classification models also calculate decimals such as 0.2 and 0.7 internally. Could the difference simply be whether the model returns a number or attaches a name to it at the end?

The distinction is **what we treat as the correct answer and what makes one prediction better than another**. Changing those criteria also changes the mistakes we try to reduce during training.

Let’s start by making two different problems from the same delivery records. We will decide what answer we need, look at how to represent that answer with numbers, and then work through the calculations used to evaluate a prediction.

## 1. The same delivery records can give us two different problems

A delivery company wants to tell customers when a new delivery will arrive. At the time of prediction, it knows the delivery distance, the time of departure, and the traffic conditions when the vehicle leaves. Records of completed deliveries also contain the actual time each delivery took, in minutes.

Information we give the model, such as distance and departure time, is called the **input**. The actual delivery time we want it to predict is the **correct answer**, or **target value**. Teaching a model prediction rules by showing it both inputs and their correct answers is called **supervised learning**.[^intro]

The following numbers and definition of a delay are made up for this explanation. We will count a delivery as **delayed if it takes more than 60 minutes** after departure. A delivery that arrives at exactly 60 minutes is not delayed.

| Delivery | Actual delivery time | Delayed? |
| --- | ---: | --- |
| A | 45 minutes | No |
| B | 60 minutes | No |
| C | 90 minutes | Yes |

The first problem we can make from these records is **“How many minutes will this delivery take?”** We want to predict the time itself: 45, 60, or 90 minutes. Predicting a quantity or magnitude like this is called **regression**.

The second problem is **“Will this delivery be delayed?”** The answer we need is either “delayed” or “not delayed.” Predicting which type something belongs to is called **classification**. A **category** is one of the types we use to group things.[^intro]

Even with the same distance and traffic information, one task predicts the time taken, while the other predicts whether the delivery will be delayed. **Classification and regression are distinguished by the answer we want from the input, more than by the input data itself.**

Whether a number has a decimal point is not the deciding factor either. Tomorrow’s sales quantity may be recorded as an integer, such as 42 items. But the difference between 40 and 50 represents a real difference in the number of items sold, so we can treat this as regression. Conversely, the integers 1, 2, and 3 attached to product types may simply be identification numbers.

## 2. A shared training process, with different training goals

A **model** is a set of calculation rules that takes input and produces a prediction. **Training** is the process of adjusting those rules using past inputs and their correct answers.

Suppose we use a **neural network**, which connects several stages of calculation. A neural network has **weights**, which multiply input values, and **biases**, which are added to calculation results. Training changes these adjustable numbers.

Before changing them, though, we need to judge how wrong the current prediction is. The rule that expresses this as a number is a **loss function**, and the number it produces is the **loss**. During training, we try to reduce that loss.

A **gradient** tells us the direction and rate at which the loss changes when we make a small change to a weight. **Differentiation** is the mathematical method used to calculate this change. **Gradient descent** uses gradients to repeatedly adjust weights and biases in a direction that reduces the loss. Whether the size of each adjustment is appropriate also affects the training result.

![The neural network training loop: generate a prediction from inputs, compare it with the actual answer to calculate loss, then use gradients to adjust weights and biases](/classification-vs-regression/learning-loop-en.svg)

Both regression and classification can use this process. The earlier calculations that find useful features in delivery distance and traffic information can also be similar. What we ask the model to predict at the end, and which loss we use to evaluate that prediction, need to match our goal.

**Gradient descent reduces the loss we have chosen. It does not decide for us which mistakes should matter.** Changing the formula for the loss can change the gradients calculated from it, which can in turn affect the direction and size of the adjustments to the weights.

The process described here is for training a neural network with gradient descent. Not every machine learning model learns this way. A decision tree, for example, learns by finding conditions that split the data, and it can be used for both regression and classification.[^trees] For a more detailed walkthrough of differentiation and weight updates, see [Supervised Learning: Basic Concepts and Introductory Coding](/en/supervised-learning-basics/).

## 3. In regression, a numerical difference measures how far a prediction is off

Suppose a delivery actually took 60 minutes, but the model predicted 50 minutes. Subtracting the actual time from the predicted time gives us:

<div class="comparison-math">
$$
50-60=-10
$$
</div>

The unit is minutes. The result, −10, means the prediction was 10 minutes shorter than the actual time. If the prediction had been 55 minutes, the difference would have been −5 minutes. In terms of the time difference alone, 55 minutes is closer to the actual time.

The difference between a predicted value and the actual value is called the **error**. Here, we will calculate it as “predicted value − actual value.”

### MSE squares the errors and takes their average

One loss function often used in regression is **mean squared error**, or **MSE**. As the name suggests, we square each error and then average those values. Squaring a number means multiplying it by itself.[^mse]

| Actual time | Predicted time | Error | Squared error |
| ---: | ---: | ---: | ---: |
| 60 minutes | 50 minutes | −10 minutes | 100 |
| 90 minutes | 85 minutes | −5 minutes | 25 |

The squared error for the first delivery is `(-10) × (-10) = 100`, and for the second it is `(-5) × (-5) = 25`. Adding the two values and dividing by 2, the number of deliveries, gives the average.

<div class="comparison-math">
$$
\text{MSE}=\frac{100+25}{2}=62.5
$$
</div>

The unit of 62.5 here is **minutes²**, because we squared a time difference. It does not mean “the predictions were off by 62.5 minutes on average.” MSE is the average of the squared errors.

Squaring prevents positive and negative errors from canceling each other out. Also, an error of 10 minutes is twice an error of 5 minutes, but their squares are 100 and 25, so its contribution to the loss is four times as large. This is how MSE gives larger errors more weight.

This calculation fits delivery time prediction because **the difference between 60 and 50 actually means 10 minutes**. The numerical difference corresponds to the prediction error we want to reduce.

## 4. Category numbers identify types, not quantities

Now suppose we want to identify the animal in a photo as a cat, dog, or bird. We assign the three types these numbers:

| Animal | Identification number |
| --- | ---: |
| Cat | 0 |
| Dog | 1 |
| Bird | 2 |

If the actual answer is cat but the prediction is dog, the numerical difference is 1. If the prediction is bird, the difference is 2. Does that alone let us say that mistaking a cat for a bird is twice as bad as mistaking it for a dog?

No. **The numbers 0, 1, and 2 do not measure distances between animals.** Giving the cat the number 2 and the bird the number 0 would not change the relationships between the animals just because we reassigned the numbers.

For delivery times, a difference of 10 meant 10 minutes. For animal numbers, a difference of 1 does not tell us how much the prediction is off. That is why ordinary classification does not calculate loss by directly subtracting these numbers.

### Category probabilities and the correct category number have different jobs

Here, we will assume that each photo has one correct animal category and that a neural network calculates a probability for each animal. Predicting one of three or more categories is called **multiclass classification**. Distinguishing between two categories, as with delayed or not delayed, is called **binary classification**.

A **probability** expresses how likely an outcome is considered to be, using a number between 0 and 1. A probability of 0.2 is the same as 20%, and 0.7 is 70%. In this example, only one of the three mutually exclusive animal categories is correct, so we make the three probabilities add up to 1.

| Output position | Animal represented by that position | Model’s predicted probability |
| ---: | --- | ---: |
| 0 | Cat | 0.2 = 20% |
| 1 | Dog | 0.7 = 70% |
| 2 | Bird | 0.1 = 10% |

Written in order, this is `[0.2, 0.7, 0.1]`. The square brackets hold the values in sequence, and here we number their positions starting at 0: position 0, position 1, and position 2.

If we follow the rule of choosing the animal with the highest probability, the result is dog. But if the actual answer is cat, that prediction is wrong. A probability of 70% calculated by the model does not mean its prediction is correct.

The correct category number, 0, is used to **find the output position corresponding to cat**. Looking at position 0 tells us that the model assigned a probability of 0.2 to the correct answer, cat.[^cross-entropy]

There are three kinds of numbers to distinguish here. **The category number 0 identifies cat, the probability 0.2 tells us how likely the model considers cat to be, and the weights are used in the calculations that produce that probability.** They are all written as numbers, but they serve different purposes.

### Cross-entropy evaluates the probability assigned to the correct answer

**Cross-entropy**, a loss function often used in classification, evaluates a prediction using the probability assigned to the correct answer. For one example with a single correct category and no special weighting, its basic form is:[^cross-entropy]

<div class="comparison-math">
$$
\text{Loss}=-\ln(\text{probability assigned to the correct answer})
$$
</div>

`ln` is the name of a mathematical function called the **natural logarithm**. In this formula, it takes a probability greater than 0 and no greater than 1, producing a smaller loss as the probability rises. Comparing actual calculations shows what it does.

| Probability assigned to the correct answer | Calculation | Loss |
| ---: | --- | ---: |
| 20% | −ln(0.2) | About 1.609 |
| 70% | −ln(0.7) | About 0.357 |
| 90% | −ln(0.9) | About 0.105 |

The losses in this table come from applying `-ln` to each probability and rounding to three decimal places. **When the probability assigned to the correct answer rises from 20% to 70%, the loss falls from about 1.609 to 0.357.** When training on many photos, one approach is to calculate this loss for each photo and average the results.

For the earlier photo, cat is the correct answer, so we use 0.2 in the calculation. We do not use 0.7, the probability the model assigned to dog, even though it was the highest. We do not subtract the predicted category number 1 from the correct category number 0 either.

With this calculation, **if the probability of the correct category stays the same, the basic cross-entropy loss for that photo stays the same, regardless of how the remaining probability is divided among the other categories.** No special order or distance between category numbers is automatically included.

If we change the numbers and also update the connection between the correct category number and its output position, we still find the same cat probability and calculate the same loss. What does not matter is the magnitude of an arbitrarily assigned category number. This does not mean that the magnitudes of weights, scores, and probabilities are unimportant.

<details>
<summary>If the natural logarithm is unfamiliar</summary>

The natural logarithm tells us what power we must raise the number `e`, approximately 2.71828, to in order to get a given value. For example, `e` raised to the power of 0 is 1, so `ln(1) = 0`. The value of `e` raised to the power of −1 is `1/e`, approximately 0.3679, so `ln(0.3679)` is approximately −1.

The natural logarithm of a number between 0 and 1 is negative, and placing a minus sign in front makes it positive. As the probability assigned to the correct answer approaches 1, this loss approaches 0. As that probability approaches 0, the loss becomes very large. The calculation gives a large loss when the model predicts that the correct answer is almost impossible.

</details>

## 5. MSE and probabilities alone do not tell us which task we are doing

So far, we have used MSE for regression and cross-entropy for classification. These are common pairings, but they are not the only possibilities.

For regression, we can also use a loss such as **mean absolute error**, or **MAE**, which averages the sizes of the errors without squaring them. For example, errors of −10 minutes and −5 minutes have magnitudes of 10 minutes and 5 minutes, whose average is 7.5 minutes. We choose the loss according to which errors we want to give more importance to.[^metrics]

### We can calculate squared errors in classification too

Let’s return to the photo whose correct answer is cat. Instead of writing the answer as the single category number 0, we can mark the positions as follows:

| Animal | Is this the correct animal? | Model’s predicted probability |
| --- | ---: | ---: |
| Cat | 1 | 0.2 |
| Dog | 0 | 0.7 |
| Bird | 0 | 0.1 |

A representation such as `[1, 0, 0]`, with 1 only in the position of the correct category and 0 elsewhere, is called a **one-hot representation**. The 1 here is not the category number for cat. It means “the animal in this position is the correct answer.”

We can compare each answer indicator with the predicted probability in the same position and calculate the average squared error.

<div class="comparison-math">
$$
\begin{gathered}
\frac{(1-0.2)^2+(0-0.7)^2+(0-0.1)^2}{3}
\\
=\frac{0.64+0.49+0.01}{3}=0.38
\end{gathered}
$$
</div>

We divide by 3 because, in this example, we are averaging the squared errors across three positions. In this way, classification can also use squared errors to compare each category’s predicted probability with its actual answer indicator.[^brier]

The target is still a category: cat, dog, or bird. We have not calculated the difference between category numbers. We have **compared each category’s predicted probability with its actual answer indicator**. Defining the loss this way does not turn animal classification into a quantity prediction task. The formula does differ from cross-entropy, though, so we cannot assume the changes during training will be identical.

### Regression can also predict probabilities for possible values

We do not always have to predict delivery time as a single number. Along with a value such as “estimated time: 55 minutes,” we may want to know how likely the delivery time is to fall within different ranges.

A **probability distribution** describes how probabilities are assigned to possible values. If we build a regression model that predicts a distribution over continuous delivery times, we can group the results into intervals and display them as follows. These probabilities are also made up for illustration.[^probabilistic]

| Delivery time interval | Model’s predicted probability |
| --- | ---: |
| Less than 30 minutes | 10% |
| At least 30 minutes and no more than 60 minutes | 60% |
| More than 60 minutes and no more than 90 minutes | 25% |
| More than 90 minutes | 5% |

The total is 100%. In this example, the probability of taking more than 60 minutes is the sum of the last two intervals: `25% + 5% = 30%`.

Looking only at the table, it might seem as though we predicted four categories. But the model assumed here **predicts a distribution over continuous time, and the table groups that result into intervals for display**. With such a distribution, we can also calculate probabilities for other time intervals chosen later.

In contrast, if we converted the actual times into the names of four intervals from the start and trained the model using only those four categories as the answers, we would have framed the task as classification. Two models can produce tables that look the same while having different training goals and working with different information.

We need to look at **what the probabilities are about**, rather than just whether probabilities are used. Losses involving natural logarithms are not exclusive to classification either. Regression that predicts probability distributions can also use a loss involving logarithms.[^probabilistic]

## 6. Even for the same delivery, a better prediction depends on what we need

It is reasonable to still wonder: “If both perform calculations to reduce loss, does it really matter which way we solve the problem?” Looking at delivery time and delay status together makes the difference clearer.

Consider a new delivery that actually takes **61 minutes**. Our rule is unchanged: a delivery is delayed if it takes more than 60 minutes. We will first compare two time predictions and then apply this rule to each one.

| What we check | Prediction A | Prediction B |
| --- | ---: | ---: |
| Predicted delivery time | 59 minutes | 64 minutes |
| Difference from the actual 61 minutes | −2 minutes | +3 minutes |
| Squared time error | 4 | 9 |
| Delay status inferred from the predicted time | Not delayed | Delayed |
| Is the actual delay status predicted correctly? | No | Yes |

**A is better when we evaluate the time error. B is better when we evaluate whether the delay status is correct.** Producing a number closer to the actual time does not necessarily mean getting the classification we need right more often.

That is why we first need to decide how the service will use the result. If we want to tell customers an estimated arrival time, “delayed” is not enough. We also need to know whether the delivery will take 65 minutes or 120 minutes. On the other hand, if the feature is meant to identify likely delays in advance and send notifications, delay status or the probability of a delay can be a more direct target.[^framing]

We can also predict the time with regression and then convert it into a delay status. If the delay threshold changes from 60 to 75 minutes, we can change the rule applied to the predicted time. However, **a small squared error in time predictions does not by itself guarantee the best predictions of delay status.**

Conversely, a classification model trained only on delay status has not been trained to answer how many minutes the delivery will take. That timing information is lost. The advantages and disadvantages of classification and regression come from how we intend to use the results. Neither is more accurate or faster for every problem.

Also, if only 5% of all deliveries are delayed, predicting “not delayed” for every delivery will be correct 95% of the time. Yet it will identify no delayed deliveries in advance. We need to check **how many actual delays were missed and how many deliveries that were not delayed were incorrectly flagged**, as well as how many predictions were correct overall.

## 7. Ratings have an order, but that does not establish numerical distances

Consider predicting one of four content age ratings: ALL, 12, 15, and 19. The ratings in this example have an order from lower to higher. That is different from the identification numbers for cat, dog, and bird.

However, we cannot directly treat the ages in those ratings as measured amounts of mature content. The difference between 12 and 15 is 3, and the difference between 15 and 19 is 4. This calculation alone does not tell us that the difference in content maturity between the latter pair is 4/3 times the difference between the former pair.

**Having an order and having differences that can be measured as quantities are separate properties.** We can design this task in several ways.

### Distinguishing the four ratings with ordinary classification

We can predict the probability of each rating and train the model to assign a high probability to the actual rating. One advantage is that this directly handles the four ratings we ultimately need.

But the basic cross-entropy calculation we saw earlier does not directly include distances or order between the ratings. Suppose two predictions have 19 as the correct answer and assign the same probability to 19. The loss for that example will not change simply because one gives more of the remaining probability to 15 while the other gives more to ALL.

This does not mean the model cannot learn features related to ratings from the data. It means **simply assigning numbers to ratings does not automatically incorporate their order into the loss calculation**.

### Converting the ratings into numbers and predicting them with regression

We could also convert ALL, 12, 15, and 19 to 0, 1, 2, and 3, respectively, and predict a number. If we apply MSE to these numbers, every pair of adjacent ratings has a numerical difference of 1.

Under the same rule, squaring an error of two rating steps gives 4, while squaring an error of one step gives 1. We can reflect differences between ratings in the loss, but we are also **choosing to calculate the spacing between ratings and the size of mistakes in this particular way**.

We also need a rule for deciding which rating to use when the prediction is, say, 1.7. Rounding to the nearest integer would give 2, meaning a rating of 15, but we must separately judge whether that rule suits the service’s purpose.

### Including the order of the ratings with ordinal classification

Methods designed to predict ordered categories while incorporating their order into training are called **ordinal classification** methods. One approach creates a question at each rating boundary:[^ordinal]

- Is the rating 12 or higher?
- Is the rating 15 or higher?
- Is the rating 19?

If the actual rating is 15, the correct answers are “yes, yes, no.” For ALL, they are “no, no, no,” and for 19, they are “yes, yes, yes.” In addition to distinguishing the individual ratings, training uses which boundaries the rating crosses. The predictions for these questions need to be designed together so that they do not contradict each other.

These methods are also called **ordinal regression** in the literature. The target here is an ordered category, not a continuous quantity. Assuming it is the same as ordinary quantity prediction just because “regression” appears in the name can be confusing.

Including order does not resolve every important type of error, either. If we want to reduce the number of predictions that give content a lower rating than it should have, we must also account for the importance of the direction of the error in training or in the final decision rule. Adding order alone does not automatically meet that requirement.

## 8. In practice, start with the answer you need and the mistakes you want to reduce

Before asking which model looks better, it helps to ask the following questions.

**First, what answer will the service actually use?** If it needs a quantity itself, such as delivery time, sales volume, or estimated cost, regression is a reasonable starting point. If it needs one of a fixed set of types, such as whether a customer inquiry concerns delivery, a refund, or a product, classification is a reasonable starting point.

**Second, what do the numbers in the answers mean?** The difference between 10 minutes and 20 minutes is 10 minutes. The difference between inquiry type 1 and inquiry type 2 does not mean “the inquiries differ by one unit.” Where there is only an order, as with ratings, we also need to consider whether using the numerical gaps directly is appropriate.

**Third, which mistakes do we need to reduce?** Predicting delivery times that are close on average and avoiding missed delays are not the same goal. The loss used for training and the criteria used to evaluate the service should match its purpose.

**Fourth, have we compared actual results using the same criteria?** If the same original problem can be framed as either regression or classification, compare the results you need on the same data that was not used for training. If the purpose is to send delay notifications, convert the regression model’s time predictions into delay statuses before comparing. We cannot place a regression MSE value next to a classification cross-entropy value and declare the smaller one the better model. The two values measure different things using different rules.

What I have taken from this is that, when distinguishing classification from regression, I should first check **what the numbers in the answers mean**. Once I know whether a number is a quantity to predict or an identifier for a category, I can start thinking about how to choose the output and the loss.

Regression and classification are not simply names for different calculation principles. **They distinguish the meaning of the target and the training goal.** Being able to use the same calculation methods does not mean we can choose the answer we ask the model to learn arbitrarily.

## References

[^intro]: Google, [What is Machine Learning](https://developers.google.com/machine-learning/intro-to-ml/what-is-ml). Supervised learning and the basic distinction between regression and classification.
[^trees]: scikit-learn, [Decision Trees](https://scikit-learn.org/stable/modules/tree.html). Classification and regression with decision trees, including their splitting criteria.
[^mse]: scikit-learn, [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html). Mean squared error between actual and predicted values.
[^cross-entropy]: PyTorch, [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html), Keras, [CategoricalCrossentropy and SparseCategoricalCrossentropy](https://keras.io/api/losses/probabilistic_losses/). Cross-entropy calculations using predicted values and either a correct category number or a one-hot representation.
[^metrics]: scikit-learn, [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics). Regression evaluation criteria, including mean absolute error.
[^brier]: scikit-learn, [Brier score loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html). Squared errors between category answer indicators and predicted probabilities.
[^probabilistic]: TensorFlow Probability, [Probabilistic Layers Regression](https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression). A regression example that predicts a probability distribution over continuous values.
[^framing]: Google, [Framing an ML problem](https://developers.google.com/machine-learning/problem-framing/ml-framing). The output a service needs and the relationship between numerical predictions and final decision rules.
[^ordinal]: Cao et al., [Rank consistent ordinal regression for neural networks with application to age estimation](https://arxiv.org/html/1901.07884v7). Framing ordered category prediction as prediction tasks at multiple rating boundaries.
