+++
title = "Supervised Learning: Core Concepts and Basic Coding"
description = "Start with inputs and targets, then explore MSE, derivatives, gradient descent, NumPy arrays, and training/test splits through small numerical examples and Python and Rust code."
date = 2026-10-07
updated = 2026-10-07
slug = "supervised-learning-basics"

[extra]
katex = true
toc = true
styles = ["supervised-learning-basics/article.css"]
+++

When you first study machine learning, unfamiliar words such as `fit`, `MSE`, `gradient`, and `shape` arrive all at once. Even when the code runs, it is easy to lose track of what went in, what changed, and what the numbers in the output mean.

This article follows **the process of learning a prediction rule from data with inputs and targets** from the beginning. To understand the calculations, we use small datasets that we can work through by hand. To assess prediction performance, we use separate data that did not take part in training. No prior knowledge of derivatives or matrices is assumed. Mathematical symbols and the values returned by the code are explained where they are needed.

The complete Python code is collected in section 17. You can first read what each part of the code means, then run it from beginning to end. Section 18 also contains a short Rust version of the same gradient descent calculation.

## 1. What does supervised learning learn?

**Supervised learning** uses inputs together with their corresponding targets to learn a rule for predicting the target of a new input. Here, “supervised” means that the training data includes the answers. It does not mean that a person gives an instruction every time the computer performs a calculation.

For example, suppose we have records of the weight, size, and shipping cost of various items. We can use weight and size as the inputs and shipping cost as the target. After training, we can enter the weight and size of an item whose shipping cost is not yet known and predict its cost.

Two common types of problem are distinguished by the nature of the target.

| Problem | What we want to predict | Examples |
| --- | --- | --- |
| Regression | A numerical quantity whose size and differences are meaningful | Shipping cost, temperature, a measured indicator |
| Classification | A predefined type or category | Spam/legitimate email, cat/dog |

Classification targets can also be stored as numbers such as `0` and `1`. Storing them as numbers does not make the problem regression. If `0=legitimate` and `1=spam`, the numbers stand in for category names. We need to consider both what is being predicted and how the predictions are evaluated.

The coding examples in this article use **regression to predict a single numerical value**. The overall learning process is shown below.[^sklearn-start]

![Inputs and targets are split into training and test data; the training data is used to build a model, and predictions for the test inputs are compared with the test targets.](/supervised-learning-basics/workflow-en.svg)

Training uses both inputs and targets. Prediction uses inputs and the model that has already been trained. Evaluation compares those predictions with the targets that were set aside. First, remember that **these three stages use different information**.

## 2. Reading inputs `X` and targets `y` from a table

When data is arranged in a table, each row represents one case. This case is called a **sample**, or observation. Each input item used for prediction is called a **feature**. In the shipping example, one item is a sample, while weight and size are its features.

To work through rows and columns, we will use the deliberately simple dataset below. The two features are practice numbers; they do not represent particular physical quantities.

| Sample | First feature $x_1$ | Second feature $x_2$ | Target $y$ |
| --- | ---: | ---: | ---: |
| First | 0 | 0 | 1 |
| Second | 1 | 0 | 3 |
| Third | 0 | 1 | 4 |
| Fourth | 1 | 1 | 6 |

By convention, an array containing only the input columns is called **`X`**, and an array containing the targets is called **`y`**. Python treats uppercase and lowercase letters as different names. `X` and `x` are different names, too.

| Name | Contents | Layout |
| --- | --- | --- |
| `X` | `[[0, 0], [1, 0], [0, 1], [1, 1]]` | 4 samples × 2 features |
| `y` | `[1, 3, 4, 6]` | 1 target per sample, 4 in total |
| `pred` | Predictions calculated by the model | 1 prediction per sample, 4 in total |

**The input and target at the same position must form a pair.** If the second input row is accidentally paired with the third target value, the model learns from the wrong case. We must preserve this pairing when we split the data later.

In mathematical notation, the target is written as $y$ and the prediction as $\hat y$. We read $\hat y$ as “y hat”; the small hat marks it as a predicted value. In code, `pred` is short for prediction. The answer we want to predict is called the **target**, and especially in classification, it is also called a **label**.

## 3. Models and coefficients: distinguishing given values from values we change

A **function** maps inputs to outputs according to a defined rule. If a function doubles an input $x$ and adds 1, then for $x=3$ its output is $1+2\times3=7$.

A **model** represents a prediction rule of this kind. In this article, we use a **linear model**, which multiplies inputs by numbers and adds the results. With one input, it looks like this:

<div class="supervised-math">
$$
\hat y=b+wx
$$
</div>

$wx$ is a shorter way to write $w\times x$. Code cannot omit the multiplication symbol, so we write `b + w * x`.

| Symbol | Name | Role |
| --- | --- | --- |
| $x$ | Input feature value | A value provided by the sample |
| $w$ | Weight | A coefficient that determines how much to multiply the input by |
| $b$ | Intercept or bias | A base term that is added even when the input is 0 |
| $\hat y$ | Prediction | The result calculated from the current coefficients and input |
| $y$ | Target | The value against which the prediction is compared |

A **coefficient** is a number that multiplies a term. Values such as $w$ and $b$ that are determined through training are called the model's **parameters**. The word “bias” has several meanings in other fields; here it refers to the intercept added to the expression.

The training data provides $x$ and $y$. During training, we change $w$ and $b$, then recalculate predictions using the new coefficients. **We change the coefficients that produce the predictions, rather than editing each predicted value independently.**

With two inputs, the expression expands as follows:

<div class="supervised-math">
$$
\hat y=b+w_1x_1+w_2x_2
$$
</div>

$w_1$ is the weight multiplying the first feature, and $w_2$ is the weight multiplying the second. The small numbers written below the letters identify the features; they are not powers. The earlier four-row dataset was constructed so that $b=1$, $w_1=2$, and $w_2=3$ match every target. Later, we will start all the coefficients at 0 and check whether they approach these values.

Here, “linear” refers to this form of multiplying coefficients and inputs and adding the results. Not every real relationship can be represented exactly in this way. Also, holding the other features fixed, increasing $x_1$ by 1 changes **this model's prediction** by $w_1$. This does not establish a causal relationship guaranteeing that changing the feature in the real world will change the outcome by that amount.

## 4. How wrong are the predictions? SSE, MSE, MAE, and RMSE

To improve the coefficients, we need a number that measures how wrong the current predictions are. First, let us define the **error** for one sample. In our training calculations, we subtract in the following order:

<div class="supervised-math">
$$
e=\hat y-y
$$
</div>

In other words, **subtract the target from the prediction.** The error is negative when the prediction is below the target and positive when it is above the target. The following three predictions form a separate example just for calculating errors.

| Target $y$ | Prediction $\hat y$ | Error $e$ | Squared error $e^2$ | Absolute error $\lvert e\rvert$ |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 9 | −1 | 1 | 1 |
| 20 | 18 | −2 | 4 | 2 |
| 30 | 27 | −3 | 9 | 3 |

To **square** a number is to multiply it by itself. For example, $(-2)^2=(-2)\times(-2)=4$. The **absolute value** is its magnitude without the sign, so $\lvert-2\rvert=2$.

If we simply add the errors, positive and negative values can cancel out. For example, $-3$ and $+3$ add up to 0, even though neither prediction was correct. Squaring the errors or taking their absolute values prevents this cancellation.

### Square the errors, add them, and divide by the count

**SSE (sum of squared errors)** is the sum of the squared errors. In the table above, it is $1+4+9=14$.

**MSE (mean squared error)** divides this sum by the number of samples. To calculate a mean, add all the values and divide by how many values there are. The table has 3 samples, so its MSE is $14/3\approx4.666667$.[^mse]

The symbol $\approx$ means “approximately equal”: the two sides are close, but not exactly equal. We have rounded the decimal value to keep it short.

For $n$ samples, the same calculation can be written mathematically as follows. Here, $n$ is the number of samples and $i$ is a sample number. $e_i$ is the error for the $i$th sample. **$\sum$ is the summation symbol: it tells us to add all the specified terms.**

<div class="supervised-math">
$$
\mathrm{SSE}=\sum_{i=1}^{n}e_i^2,
\qquad
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}e_i^2
$$
</div>

In Python, `e ** 2` squares each error, `np.sum(e ** 2)` calculates SSE, and `np.mean(e ** 2)` calculates MSE. `np` is the short name we will use for NumPy, a numerical computing tool introduced below.

As we collect more samples with similar error sizes, SSE generally grows. Dividing by the number of samples makes MSE **the average squared error per sample**. This does not mean that MSE values from different problems or targets with different units can be compared without any conditions.

### The training loss in this article is half the MSE

A **loss function** calculates how wrong predictions are. Training usually changes the coefficients to make this value smaller. In this article, we call our training loss $J$ and define it as follows:

<div class="supervised-math">
$$
J=\frac{\mathrm{MSE}}{2}
=\frac{1}{2n}\sum_{i=1}^{n}e_i^2
$$
</div>

In the earlier example, $J=7/3\approx2.333333$. **Dividing by 2 is not part of the definition of MSE.** We choose this constant for the training objective in this article because it conveniently cancels the 2 that appears when differentiating.

For the same data, the coefficients that minimize MSE also minimize half the MSE. Every loss value has simply been multiplied by the same positive number, $1/2$. However, the slope calculated later is also halved, so using the same learning rate does not produce the same step size.

### How do MAE and RMSE differ?

| Measure | Calculation | Value for the table above | How to interpret it |
| --- | --- | ---: | --- |
| SSE | Sum of squared errors | 14 | Total magnitude of the squared errors |
| MSE | Mean of squared errors | About 4.666667 | Average squared error per sample |
| $J$ | In this article, MSE divided by 2 | About 2.333333 | The objective value used to update the coefficients |
| MAE | Mean of absolute errors | 2 | Average error magnitude per sample |
| RMSE | Square root of MSE | About 2.160247 | A measure based on squared errors, converted back to the original unit |

A **square root** of a number is a value that gives that number when squared. For example, because 3 squared is 9, the nonnegative square root of 9 is 3. `np.sqrt(mse)` calculates this square root.

If the target is measured in Korean won, the error, MAE, and RMSE are also measured in won, while MSE is measured in won squared. RMSE is not the same calculation as MAE. Squaring turns an error of 1 into 1 but an error of 3 into 9, giving larger errors more weight.

## 5. Python and NumPy: looking at both array values and shapes

A Python **variable** is a name attached to a value or object. In `w = 1.0`, `=` performs **assignment**: it associates the value on the right with the name on the left. It is not a mathematical proof of equality. Code uses `==` to compare whether two values are equal.

A **list** is a Python data type that holds multiple values in order, such as `[1, 2, 3]`. An **array** also holds multiple values, but the NumPy arrays used here perform numerical calculations with a defined shape and data type. `np.array([1.0, 2.0, 3.0])` turns a list into a NumPy array. The decimal points indicate values to be used in floating-point calculations.

An **element** is one value inside an array. When two NumPy arrays have the same shape, `+`, `-`, `*`, and `**` normally operate element by element at corresponding positions. For example, multiplying arrays made from `[1, 2, 3]` and `[10, 20, 30]` gives `[10, 40, 90]`. The `*` operator on Python lists does not perform the same operation, so we need to distinguish lists from NumPy arrays.

### `shape` tells us how many values there are along each direction

A direction used to locate values in an array is called an **axis**. A two-dimensional table has two axes: one for selecting rows and another for selecting columns. `shape` gives the size along each axis, and `ndim` gives the number of axes.

| Example array | `shape` | `ndim` | Meaning |
| --- | --- | ---: | --- |
| `[1, 3, 5]` | `(3,)` | 1 | A one-dimensional array of length 3 |
| `[[1], [3], [5]]` | `(3, 1)` | 2 | 3 rows and 1 column |
| `[[1, 3, 5]]` | `(1, 3)` | 2 | 1 row and 3 columns |
| Our earlier `X` | `(4, 2)` | 2 | 4 rows and 2 columns |

The result of `shape` is displayed as a Python data type called a **tuple**. A tuple groups values in order, and its entries cannot be replaced after it is created. A tuple with one element includes a comma, as in `(3,)`. This comma does not mean that an empty second axis exists. `(3,)` means there is one axis; `(3, 1)` means there are two.

Strictly calling a one-dimensional array of shape `(3,)` “3 rows and 1 column” leads to confusion in later calculations. Even with the same three numbers, `(3,)`, `(3, 1)`, and `(1, 3)` can behave differently. When the code behaves unexpectedly, **print and inspect the `shape` as well as the numbers**.

## 6. Indexing: choosing values by position

Selecting part of an array is called **indexing**. An **index** is a number identifying a position. In Python, the first position is `0`, the second is `1`, and the third is `2`.

Consider the following array with 3 rows and 3 columns. The row and column numbers shown below are indices.

| Row index | Column 0 | Column 1 | Column 2 |
| --- | ---: | ---: | ---: |
| 0 | 10 | 11 | 12 |
| 1 | 20 | 21 | 22 |
| 2 | 30 | 31 | 32 |

Call this array `A`. We can select along both axes within one pair of square brackets, using `A[row selection, column selection]`. Here, `:` means to select every position along that axis.[^indexing]

| Code | Actual result | Resulting `shape` |
| --- | --- | --- |
| `A[1]` | `[20, 21, 22]` | `(3,)` |
| `A[[1]]` | `[[20, 21, 22]]` | `(1, 3)` |
| `A[:, 2]` | `[12, 22, 32]` | `(3,)` |
| `A[:, [2]]` | `[[12], [22], [32]]` | `(3, 1)` |
| `A[1, 2]` | `22` | `()` — a value with no axes |

Selecting one row with the integer `1` removes the axis used to select that row. In contrast, `[1]` is a list of length 1 containing a row index to select, so the result remains two-dimensional with one row. The difference between `2` and `[2]` when selecting a column works the same way. Later, this distinction lets us preserve `(number of samples, 1)` when passing just one feature to scikit-learn.

### Two pairs of brackets perform two operations

`A[1][2]` first uses `A[1]` to obtain `[20, 21, 22]`, then selects index 2 of **that result**, giving 22.

Do not memorize “the first brackets select rows, and the second brackets select columns” as a universal rule. With `row_ids = [0, 2]`, `A[row_ids]` produces the new selection `[[10, 11, 12], [30, 31, 32]]`. Applying `[1]` to it, as in `A[row_ids][1]`, selects the second row of that result: `[30, 31, 32]`.

To select column 2 from those two rows while keeping a two-dimensional result, write `A[row_ids][:, [2]]`. The result is `[[12], [32]]`, with shape `(2, 1)`.

Also distinguish the case where two lists appear inside one pair of brackets. In this example, `A[[0, 2], [1, 2]]` pairs the indices to select `A[0, 1]` and `A[2, 2]`, giving `[11, 32]`. It does not select every combination of the two rows and two columns to form a table. Performing the selections separately with `A[[0, 2]][:, [1, 2]]` gives `[[11, 12], [31, 32]]`.

## 7. `None`, `reshape`, and broadcasting: the same numbers can have different shapes

Suppose `v` is a one-dimensional NumPy array containing `[1, 3, 5]`. In indexing, `None` **adds a new axis of length 1** at that position. It does not insert the number 0 or a missing value.

| Code | Result | `shape` |
| --- | --- | --- |
| `v` | `[1, 3, 5]` | `(3,)` |
| `v[:, None]` | `[[1], [3], [5]]` | `(3, 1)` |
| `v[None, :]` | `[[1, 3, 5]]` | `(1, 3)` |
| `v.reshape(-1, 1)` | `[[1], [3], [5]]` | `(3, 1)` |
| `v.T` | `[1, 3, 5]` | `(3,)` |

`reshape` changes an array's shape while preserving the number of elements. In `reshape(-1, 1)`, `1` sets the number of columns to 1, and `-1` asks NumPy to calculate the remaining size from the total number of elements. With 3 elements, the number of rows becomes 3. It does not create a negative number of rows.

For a two-dimensional array, `.T` performs a **transpose**, swapping rows and columns. But `v` has only one axis, so there is no second axis to swap with. `v.T` therefore still has shape `(3,)`. To give a one-dimensional array a column layout, add an axis or use `reshape`.

### An incorrect error array can be created without an error message

**Broadcasting** is a NumPy feature that allows arrays of different shapes to be used together when certain rules are satisfied. It compares each `shape` starting from the rightmost axis. An axis is compatible if the sizes are equal or one of the sizes is 1. Missing leading axes are treated as having size 1.[^broadcasting]

If predictions `p = [0, 2, 4]` and targets `v = [1, 3, 5]` both have shape `(3,)`, then `p - v` gives `[-1, -1, -1]`. These are three errors, each comparing values from the same sample.

But if we change only the targets to `v[:, None]`, the operation becomes `(3,) - (3, 1)`. NumPy treats the predictions as having shape `(1, 3)` and produces the following **3-by-3 result comparing every prediction with every target**.

| Target being subtracted | Prediction 0 | Prediction 2 | Prediction 4 |
| --- | ---: | ---: | ---: |
| Target 1 | −1 | 1 | 3 |
| Target 3 | −3 | −1 | 1 |
| Target 5 | −5 | −3 | −1 |

We wanted one error per sample, or 3 in total. Instead, we obtained 9 values without an error message. Taking the mean of this result still produces a normal-looking single number, so the mistake is easy to miss.

For code like ours, which predicts one target and uses one-dimensional predictions, **first check that both predictions and targets have shape `(number of samples,)`**. The expression `assert pred.shape == y.shape` checks whether the two shapes match. Section 13 explains `assert` in more detail.

## 8. Derivatives and gradients: which way does the loss change?

After calculating the loss, the next question is: “If we change a coefficient slightly, will the loss decrease?” **Differentiation** helps us answer this question.

### Divide the change in the output by the change in the input

First, consider a simple function $f(t)=t^2/2$ with one numerical input, $t$. Here, $f$ is the function's name, and $f(t)$ is its output when the input is $t$.

Changing $t$ from 3 to 3.1 increases the input by 0.1. The function value changes from $4.5$ to $4.805$, an increase of 0.305. The output change per unit of input change is $0.305/0.1=3.05$.

Writing the input change as $h$, we get the following calculation. **At this stage, $h$ is a nonzero value.**

<div class="supervised-math">
$$
\frac{f(t+h)-f(t)}{h}
=\frac{(t+h)^2-t^2}{2h}
=\frac{2th+h^2}{2h}
=t+\frac{h}{2}
$$
</div>

As we bring $h$ closer and closer to 0, this value approaches $t$. **Differentiation finds the rate of change at the current position when the input moves by a very small amount.** The expression above does not divide by 0. We simplify it with a nonzero $h$, then examine which value it approaches as $h$ approaches 0.[^derivative]

On a function's graph, this rate of change at the current position is called the **slope**.

| Concept | Expression in this example | Meaning |
| --- | --- | --- |
| Original function | $f(t)=t^2/2$ | A rule that calculates the function value from the input |
| Derivative function | $f'(t)=t$ | A rule that calculates the slope at the input position |
| Function value at $t=3$ | $f(3)=4.5$ | The height of the function at that position |
| Derivative value at $t=3$ | $f'(3)=3$ | The rate of change at that position |

The **derivative function** is the whole function obtained by differentiating. A **derivative value** is the number obtained by evaluating it at a particular position. The small mark in $f'$ is read as “prime.” **The loss value and the derivative value are different numbers.**

### With several coefficients, vary one at a time

The regression loss $J$ depends on $w$ and $b$. We can hold $b$ fixed and change only $w$ slightly to calculate how much the loss changes. We can also hold $w$ fixed and change only $b$. Differentiating with respect to just one of several inputs in this way gives a **partial derivative**.

| Expression | Meaning |
| --- | --- |
| $\partial J/\partial w$ | The rate of change of loss $J$ when $w$ changes while the other coefficients are held fixed |
| $\partial J/\partial b$ | The rate of change of loss $J$ when $b$ changes while the other coefficients are held fixed |

$\partial$ is the symbol used for partial derivatives. Although the expression looks like a fraction, for now read it as a single notation indicating “which function is being differentiated with respect to which variable.”

The **gradient** is the collection of partial derivative values for the coefficients, arranged in a fixed order. It is a **vector**, which here can be understood as several numbers grouped in order. If the coefficient order is `[b, w1, w2]`, the gradient has three values in that same order.[^gradient]

## 9. Why are `db = mean(e)` and `dw = mean(e * x)`?

Before memorizing a formula, start with one sample. Its prediction is $b+wx$ and its target is $y$, so its error is $e=b+wx-y$. Let the loss for this one sample be $\ell=e^2/2$. The symbol $\ell$ is a lowercase “ell,” used to distinguish this single-sample loss from the average loss $J$ over several samples.

### Changing the intercept changes the error by the same amount

If we change $b$ to $b+h$, the new error is $e+h$. Subtracting the old loss from the new loss gives:

<div class="supervised-math">
$$
\frac{(e+h)^2}{2}-\frac{e^2}{2}
=\frac{e^2+2eh+h^2-e^2}{2}
=eh+\frac{h^2}{2}
$$
</div>

Dividing this by $h$, the change in $b$, gives $e+h/2$. As $h$ approaches 0, the remaining value is $e$, so $\partial\ell/\partial b=e$.

### Changing the weight also multiplies the change by the input

If we change $w$ to $w+h$, the $wx$ part of the prediction becomes $(w+h)x=wx+hx$. The new error is therefore $e+hx$.

<div class="supervised-math">
$$
\frac{(e+hx)^2}{2}-\frac{e^2}{2}
=\frac{e^2+2ehx+h^2x^2-e^2}{2}
=ehx+\frac{h^2x^2}{2}
$$
</div>

Dividing by $h$, the change in $w$, gives $ex+hx^2/2$, which approaches $ex$ as $h$ approaches 0. Thus, $\partial\ell/\partial w=ex$. **The input $x$ appears in the derivative with respect to the weight because changing the weight by $h$ changes the prediction by $hx$.**

The **chain rule** gives another way to express the same result. A change in a coefficient changes the error, and a change in the error changes the loss. We multiply the rates of change for these two stages.[^chain-rule]

<div class="supervised-math">
$$
\frac{\partial\ell}{\partial w}
=\frac{\partial\ell}{\partial e}\frac{\partial e}{\partial w}
=e\cdot x,
\qquad
\frac{\partial\ell}{\partial b}
=\frac{\partial\ell}{\partial e}\frac{\partial e}{\partial b}
=e\cdot1
$$
</div>

Although this looks like canceling parts of a fraction, it is not a rule obtained by simply deleting symbols. As we have just seen, it expresses how actual changes are connected. We must also distinguish the fact that the derivative of the error with respect to $b$ is 1 from the fact that **the derivative of the loss with respect to $b$ is $e$**.

### For the average loss over several samples, average the derivatives too

Our loss $J$ is the mean of the individual sample losses $\ell$. Differentiating a sum adds the derivatives of its terms, and differentiating a value divided by a constant retains that same division. We therefore obtain:

<div class="supervised-math">
$$
\frac{\partial J}{\partial b}=\frac{1}{n}\sum_{i=1}^{n}e_i,
\qquad
\frac{\partial J}{\partial w}=\frac{1}{n}\sum_{i=1}^{n}e_ix_i
$$
</div>

These expressions are `db = np.mean(e)` and `dw = np.mean(e * x)` in code. Here, $i$ is the sample number. Because this model has one input, $x_i$ is the input value for the $i$th sample. Distinguish this context from the earlier two-feature model, where $x_1$ and $x_2$ identified different features.

`db` and `dw` are ordinary variable names chosen by the programmer. Adding `d` to the beginning of a name does not make Python differentiate automatically. We calculate the expressions we derived and store the results ourselves. Also, **these derivatives are for $J=\mathrm{MSE}/2$**. Differentiating MSE itself gives twice each value.

## 10. Updating the coefficients once with gradient descent

If the slope at the current position is positive, increasing the coefficient slightly moves in a direction that increases the loss. We therefore decrease the coefficient a little. If the slope is negative, we move by increasing the coefficient a little. This method of **updating coefficients in the direction opposite to the gradient** is called **gradient descent (GD)**.[^gradient-descent]

<div class="supervised-math">
$$
w_{\mathrm{new}}=w-\alpha\frac{\partial J}{\partial w},
\qquad
b_{\mathrm{new}}=b-\alpha\frac{\partial J}{\partial b}
$$
</div>

The subscript `new` marks the new value after an update. The symbol $\alpha$, read as “alpha,” is the **learning rate**, a positive number that controls how far to move in one step. We call it `lr` in code. Unlike the coefficients the model uses for prediction, the learning rate is a value we set for the training method. Such a setting is called a **hyperparameter**.

Let us perform one calculation using the following three-row dataset with one input. These targets are different from the targets used just to illustrate errors in section 4.

| $x$ | Target $y$ | Initial prediction $0+1\times x$ | Error $e$ | $ex$ |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | −1 | 0 |
| 1 | 3 | 1 | −2 | −2 |
| 2 | 5 | 2 | −3 | −6 |

The initial coefficients are $w=1$ and $b=0$. The errors are `[-1, -2, -3]`, the initial MSE is $14/3$, and the initial $J$ is $7/3$. The two derivatives are:

<div class="supervised-math">
$$
dw=\frac{0-2-6}{3}=-\frac{8}{3},
\qquad
db=\frac{-1-2-3}{3}=-2
$$
</div>

With a learning rate of 0.1, the updated values are:

<div class="supervised-math">
$$
w_{\mathrm{new}}=1-0.1\left(-\frac{8}{3}\right)
=\frac{19}{15}\approx1.266667,
\qquad
b_{\mathrm{new}}=0-0.1(-2)=0.2
$$
</div>

Both coefficients increased because we subtracted negative values. The new predictions are approximately `[0.2, 1.466667, 2.733333]`, and the MSE is approximately **2.709630**, down from the initial value of about 4.666667. The new $J$ is half of that, approximately 1.354815.

We calculate **both `dw` and `db` at the same coefficients before updating them**. If we change one coefficient first and calculate the other derivative using the new coefficient, that differs from the single simultaneous update described here.

The predictions do not match the targets perfectly after one step, but the loss has decreased. We can repeat this process. However, the gradient describes the current position, so it does not guarantee a decrease in loss if we move too far. Section 12 examines the effect of the learning rate separately.

## 11. Combining two inputs into a matrix calculation

We now return to the dataset from section 2, with **two inputs and four samples**. Both its sample count and input count differ from the three-row dataset in the previous section.

A **matrix** is an arrangement of numbers in rows and columns. We can express matrix calculations using two-dimensional NumPy arrays. To include the intercept in the same calculation, add a column of ones to the left of the original `X` and call the result `Xb`.

| 1 for the intercept | $x_1$ | $x_2$ | Prediction calculated for this row |
| ---: | ---: | ---: | --- |
| 1 | 0 | 0 | $b$ |
| 1 | 1 | 0 | $b+w_1$ |
| 1 | 0 | 1 | $b+w_2$ |
| 1 | 1 | 1 | $b+w_1+w_2$ |

In code, `np.ones(n)` creates $n$ ones. `np.c_[np.ones(n), X]` joins them as a column to the left of `X`. This column of ones is not a newly observed feature. **Because $b\times1=b$, it lets us include the intercept in the same multiply-and-add calculation.**

We collect the coefficients in the order `theta = [b, w1, w2]`. The mathematical symbol $\theta$ is read as “theta” and is the name of our coefficient collection here. The name theta does not always mean exactly two values, a weight and an intercept. Its length depends on the number of coefficients the model needs.

### `Xb @ theta` produces one prediction per row

For NumPy arrays, `@` performs **matrix multiplication**. In the case used here—a two-dimensional array multiplied by a one-dimensional coefficient vector—it multiplies matching positions in **each row** of the first array and the second vector, then adds those products. This produces one result per row.[^matmul]

For example, multiplying the last row `[1, 1, 1]` by the coefficients `[1, 2, 3]` gives $1\times1+1\times2+1\times3=6$. Applying this to every row gives the predictions `[1, 3, 4, 6]`.

| Part of the calculation | `shape` | Reason |
| --- | --- | --- |
| `Xb` | `(4, 3)` | 4 samples and 3 columns corresponding to the coefficients |
| `theta` | `(3,)` | An intercept and two weights, 3 values in total |
| `Xb @ theta` | `(4,)` | One prediction per sample |
| `Xb.T` | `(3, 4)` | The rows and columns of `Xb` are swapped |
| Error `e` | `(4,)` | Prediction minus target for each sample |
| `Xb.T @ e / n` | `(3,)` | One derivative value per coefficient |

**The 3 coefficients and the 4 predictions are different kinds of value.** The model expression determines the number of coefficients, while the number of samples supplied determines the number of predictions. Calculating `@` does not change the original `theta` into the shape of the predictions. It produces a separate result. We must also distinguish this from elementwise multiplication with `*`.

### Working through `Xb.T @ e / n` with numbers

Starting the coefficients at `[0, 0, 0]` makes all four predictions 0. The errors are `[-1, -3, -4, -6]`, and the loss is:

<div class="supervised-math">
$$
J=\frac{1+9+16+36}{2\times4}=7.75
$$
</div>

The derivative for the intercept is the mean of the errors. The derivative for the first weight is the mean of each error multiplied by the first feature. The second weight works the same way. In `Xb.T`, the inputs from each column are arranged as rows so that we can perform these calculations.

| Row of `Xb.T` | Sum of the products of that row and the error vector | Derivative after dividing by 4 |
| --- | --- | ---: |
| `[1, 1, 1, 1]` | $-1-3-4-6=-14$ | −3.5 |
| `[0, 1, 0, 1]` | $0-3+0-6=-9$ | −2.25 |
| `[0, 0, 1, 1]` | $0+0-4-6=-10$ | −2.5 |

Thus, `gradient = Xb.T @ e / n` gives `[-3.5, -2.25, -2.5]`. **The transpose `.T` does not perform differentiation.** It arranges the array in the right direction to calculate all the “mean of input × error” expressions that we have already derived.

Separately calculating one step with a learning rate of 0.1 gives `[0.35, 0.225, 0.25]`. In code, `trial_theta = theta - 0.1 * g0` stores this result in `trial_theta`. Since we have not assigned the result back to `theta`, the original `theta` is still `[0, 0, 0]`.

## 12. Repeated training, loss records, and the learning rate

The actual loop starts the coefficients at 0 and performs **2000 updates with a learning rate of 0.2**. This differs from the trial calculation with 0.1 shown in the previous section. Each iteration performs the following steps:

1. Calculate predictions for the four samples using the current coefficients.
2. Subtract the targets from the predictions to obtain the errors.
3. Record the current loss $J$.
4. Use the errors and inputs to calculate the derivative for each coefficient.
5. Subtract `learning rate × derivative` from each coefficient.

Because each update uses all the training samples, this method is called **batch gradient descent**. Other variations use only some of the samples, but this article uses all four rows.

After 2000 updates, the coefficients are approximately `[1, 2, 3]`, and the predictions are approximately `[1, 3, 4, 6]`. They have approached the answers used to construct the practice data. Starting at 0 is suitable for this simple linear regression problem. Do not extend that conclusion to the initialization of every kind of model, especially every neural network.

### Loss recorded before an update is one step behind loss recorded after it

`history` is a Python list that collects losses in order. `.append(value)` adds one value to the end of a list. In this example, we append **before updating**.

| Stored value | Corresponding coefficient state |
| --- | --- |
| `history[0]` | 0 updates completed: the initial coefficients |
| `history[1]` | 1 update completed |
| `history[1999]` or `history[-1]` | 1999 updates completed |
| `final_j`, calculated outside the loop | 2000 updates completed |

The negative list index `-1` means the last element. Performing 2000 updates does not automatically make `history[-1]` the loss after the 2000th update. We need to look at **when the value was stored**.

`history + [final_j]` adds the final state, giving 2001 states from 0 through 2000 completed updates. To make the early changes easier to read, the following figure shows only **81 states, from 0 through 80 completed updates**.

![Training loss J decreases from 7.75 to approximately 0.002055 over 0 through 80 completed updates.](/supervised-learning-basics/loss-en.svg)

The horizontal axis is the number of completed coefficient updates, and the vertical axis is $J=\mathrm{MSE}/2$. This is **the loss curve for the four training rows**. It does not yet show performance on new data. After 2000 updates, the loss in the execution environment is approximately $4.8\times10^{-30}$, numerically very close to 0. The number $10^{-30}$ is 1 divided by $10^{30}$, a very small value.

### Assignment, in-place changes, and copies serve different purposes

`theta = theta - lr * gradient` first produces the result on the right, then reassigns the name `theta` to that result. For the NumPy floating-point array used here, `theta -= lr * gradient` performs an **in-place update**, directly changing the existing array's contents.

If we write only `saved = theta`, both names refer to the same array. Changing `theta` in place later also changes what we read through `saved`. To preserve an earlier result independently, make a copy with `saved = theta.copy()`. The line `theta_gd = theta.copy()` stores a separate copy of the coefficients after training.

### A larger learning rate can change the sign; a still larger one can increase the loss

To isolate the effect, step away from the regression problem briefly and consider only the function $J(t)=t^2/2$. Its derivative value is $t$, and the initial $t$ is 3. The initial loss is 4.5.

<div class="supervised-math">
$$
t_{\mathrm{new}}=t-\alpha t=(1-\alpha)t
$$
</div>

| Learning rate $\alpha$ | $t$ after the first update | $J$ after the first update | What to notice |
| ---: | ---: | ---: | --- |
| 0.2 | 2.4 | 2.88 | The value moves closer to 0 with the same sign. |
| 0.8 | 0.6 | 0.18 | The first step takes the value closer to 0 by a larger amount. |
| 1.5 | −1.5 | 1.125 | The loss decreases even though the sign changes. |
| 2.2 | −3.6 | 6.48 | The value moves farther from 0, increasing the loss. |

For this function, each step multiplies the value by $1-\alpha$. For a nonzero starting value to approach 0, we therefore need $\lvert1-\alpha\rvert<1$, or **$0<\alpha<2$**. A value getting progressively closer to a target is called **convergence**. At $\alpha=2$, the value alternates between 3 and −3 without reducing the loss.

**This range is a result for the particular function $J(t)=t^2/2$.** A different loss shape or different input magnitudes can call for a different learning rate. Also, a change in the sign of a coefficient does not by itself mean that training has failed. Examine the loss and the size of the movement together.

## 13. Least squares, `fit` and `predict`, and implementation checks

Our training objective has been to find coefficients that make the sum of squared errors small. This is called a **least-squares** problem. For the same fixed data, the coefficients that minimize SSE, MSE, and half the MSE are the same.

The gradient descent code we wrote is not the only way to solve this problem. For linear regression, we can also use a least-squares solver based on linear algebra calculations.

### What does NumPy's `lstsq` return?

`np.linalg.lstsq(Xb, y, rcond=None)` finds coefficients that make `Xb @ theta` close to `y`. The name `linalg` stands for linear algebra and identifies the collection of functions for those calculations. `lstsq` is short for least squares. The setting `rcond=None` uses the library's default criterion for distinguishing very small numerical components.[^lstsq]

A single call returns the following four values in order:

| Returned value | Name used in this article | Meaning |
| --- | --- | --- |
| The solution coefficients | `theta_ls` | Here, 3 numbers in the order `[b, w1, w2]` |
| Array of residual sums of squares | `residual_sums` | The summed squared errors for each target column |
| Matrix rank | `rank` | The number of linearly independent column directions |
| Singular values | `singular_values` | Values used to assess magnitudes along the matrix's directions and distinguish them numerically |

Here, an **independent column** is one that cannot be fully replaced by multiplying the other columns by fixed numbers and adding them. Calculating singular values requires more linear algebra, so we will not work through that calculation here. We should still distinguish the fact that `lstsq` returns more than the coefficients and understand what its other outputs represent.

In particular, although its name contains `residual`, the second output is **not an array listing the error for each sample**. In this example it is a length-1 array containing a sum of squares; depending on the matrix dimensions and rank, it can also be empty. To obtain the errors for individual samples, calculate predictions separately and subtract the targets.

### A scikit-learn model is an object that stores calculation results

A **library** is software that collects reusable functionality. NumPy provides array operations, while **scikit-learn** provides functionality for training, prediction, and evaluation.

Calling `LinearRegression()` creates a linear regression model **object**. Here, an object is an entity that stores settings and training results and performs related operations. A function belonging to an object is called a **method**.

| Code | What it does | What it returns or stores |
| --- | --- | --- |
| `model = LinearRegression()` | Creates a model object to train. | A model that has not yet been trained on this data |
| `model.fit(X, y)` | Finds coefficients from the inputs and targets. | Stores the training results in the model and returns the same model object |
| `model.coef_` | Reads the learned weights. | Here, `[w1, w2]` |
| `model.intercept_` | Reads the learned intercept. | Here, $b$ |
| `model.predict(X)` | Makes predictions with the current coefficients. | An array with one prediction per sample |

The return value of `fit` is not an array of predictions. Calling `predict` does not train the coefficients again. In scikit-learn, the trailing underscore in names such as `coef_` and `intercept_` is a convention for attributes obtained through training.[^sklearn-start]

With the **dense NumPy arrays and default settings** in this article, `LinearRegression` uses SciPy's least-squares solver. Here, a dense array stores all its elements in the ordinary array format, and SciPy is a scientific computing library. The model does not internally repeat the same 2000 gradient descent steps we wrote. `fit` is a common interface for “train on the supplied data”; the actual solution method depends on the model and settings.[^linear-regression]

The default `fit_intercept=True` means that the model also estimates an intercept. We therefore supply **the original `X`, before adding a column of ones**. Distinguish this input format from our direct `Xb @ theta` calculation.

For the four-row example, all three methods predict approximately `[1, 3, 4, 6]`. Placing them side by side with `np.c_[pred_ls, pred_gd, pred_sk]` produces **4 rows and 3 columns**: a comparison of three methods' predictions for four samples.

### Preserve the feature order for new inputs

For a new input $x_1=3$, $x_2=4$, our manually implemented model puts a 1 for the intercept first and calculates with `[1, 3, 4]`.

<div class="supervised-math">
$$
[1,3,4]\cdot[1,2,3]=1\times1+3\times2+4\times3=19
$$
</div>

The centered dot means to multiply corresponding positions and add the results. Changing the training order `[1 for the intercept, first feature, second feature]` changes the calculation. For scikit-learn, supply a two-dimensional input such as `[[3, 4]]`, representing **1 sample with 2 features**.

Being able to produce the prediction 19 does not prove that the model predicts real outcomes well. That requires a separate evaluation against the targets of new cases.

### `assert` checks a condition; it does not train a model

`assert condition` continues if the condition is true and raises an error if it is false. `True` and `False` are Python values representing true and false. The examples use assertions to check array shapes and results that we have already calculated by hand.

Computers store real numbers with a finite number of digits, using a representation called **floating point**. Even if a mathematical result is 1, the stored result might differ slightly, such as `0.9999999999999998`. Instead of always comparing floating-point arrays with `==`, we can specify how much difference to allow.

| Tool | Result and purpose |
| --- | --- |
| `np.isclose(a, b)` | Checks whether values at corresponding positions are sufficiently close. Array inputs produce an array of true/false values. |
| `np.allclose(a, b)` | Returns one true/false value indicating whether all the compared values are sufficiently close. |
| `np.isfinite(a)` | Checks whether each value is neither NaN nor infinite. |
| `.all()` | Checks whether every value in a true/false array is true. |

**NaN**, short for “not a number,” is a special value used for numerical results that are undefined or invalid, among other cases. **Infinity** represents a value that is not finite. If these values appear during repeated training, examine the calculations.

The basic comparison rule used by `allclose` is:[^allclose]

<div class="supervised-math">
$$
|a-b|\leq\mathrm{atol}+\mathrm{rtol}|b|
$$
</div>

`atol` is the absolute difference allowed, and `rtol` determines an additional allowance proportional to the magnitude of the reference value $b$. To allow only an absolute difference of $10^{-8}$, write `atol=1e-8, rtol=0`. The code notation `1e-8` means $1\times10^{-8}=0.00000001$. The `e` in this notation is unrelated to the error variable we created earlier.

Because `allclose` also allows broadcasting, **it does not replace a shape check**. Check that the shapes match before comparing the numbers. Also, `final_j < history[0]` checks that the final loss is smaller than the initial loss; it does not check that the loss never increased at any intermediate step. Tiny fluctuations can occur in numerical calculations that have become very close to 0.

Checking whether the coefficients are close to `[1, 2, 3]` verifies the implementation on this particular practice dataset, which was constructed with known answers. We should not expect these coefficients or the same loss threshold for arbitrary real data. Python's `assert` can be removed when running with the `-O` optimization option, so do not rely on it for every input validation that must run in a real service.[^assert]

## 14. Splitting real data into training and test sets

The small practice dataset let us check the calculations. Now let us examine **whether the model also predicts cases that were not used for training**. This ability is called **generalization**.

We will use the `diabetes` dataset included in scikit-learn. It has 442 samples and 10 input features. The target is **a numerical measure of disease progression one year after the initial measurements**. The task is not to predict diabetes status as 0 or 1, nor to predict blood glucose itself.[^diabetes]

| Column position | Name | Meaning needed for this exercise |
| --- | --- | --- |
| 0 | `age` | Age |
| 1 | `sex` | Sex category recorded numerically |
| 2 | `bmi` | Body mass index, an indicator calculated from height and weight |
| 3 | `bp` | Average blood pressure |
| 4–9 | `s1`–`s6` | Six indicators related to blood tests, including cholesterol |

The purpose of this article is to practice regression using inputs and a numerical target from the same row, rather than to interpret each medical indicator. The official documentation also notes uncertainty about the exact meanings of some original features. We will not assign meanings to the numbers by guessing.[^diabetes]

`load_diabetes(return_X_y=True, scaled=False)` returns the input array followed by the target array. `return_X_y=True` means that we want the two arrays `X` and `y` directly. `scaled=False` means that we **receive the stored feature values without applying the additional mean centering and scaling offered by the loader**. Mean centering subtracts the feature's mean from each value; scaling transforms values, for example by dividing them by a chosen reference quantity. This does not mean that no column has ever undergone any earlier processing.

This exercise directly solves ordinary least-squares regression with an intercept. Additional scaling is not required for this example, but do not generalize that conclusion to say input scaling is always unnecessary for the earlier gradient descent code or for other algorithms.

### Split row indices, then apply the same selection to inputs and targets

**Training data** is used to find the coefficients. **Test data** contains cases not used in that training, allowing us to evaluate the predictions.

`np.arange(len(y_real))` creates one row index per target, from `0` through `441`. The function `arange` creates evenly spaced numbers; here, it excludes the endpoint 442. We split that index array with `train_test_split`.[^split]

| Setting or result | Meaning |
| --- | --- |
| `test_size=0.2` | Select approximately 20% of the data for testing. |
| `random_state=42` | Fix the starting point for random number generation so that the same inputs, settings, and environment reproduce the same random split. |
| `train_idx` | Array of row indices selected for training, with length 353 |
| `test_idx` | Array of row indices selected for testing, with length 89 |

Random numbers are numbers used to make random selections. The value 42 does not guarantee a better model than another number. It is a setting that makes the split reproducible. It also does not mean that exactly the same cases will be selected if the data itself or its order changes.

`train_idx` is **an array of multiple row indices**, not a single number. Applying the same selection in `X_real[train_idx]` and `y_real[train_idx]` preserves the input–target pairs. The resulting shapes are:

| Array | `shape` |
| --- | --- |
| `X_train` | `(353, 10)` |
| `y_train` | `(353,)` |
| `X_test` | `(89, 10)` |
| `y_test` | `(89,)` |

### First, build a baseline that predicts only the mean

A **baseline** is a simple method against which we can measure how much a more complex method improves. Here, we calculate the mean of the training targets and predict that mean for every test sample. `DummyRegressor(strategy="mean")` performs this role.[^baseline]

Why the mean? If we predict the same number $c$ for every sample, the derivative of half the mean squared loss with respect to $c$ is `mean(c - y)`. This equals $c-\operatorname{mean}(y)$. When $c$ equals the mean of the training targets, the derivative is 0 and the upward-opening squared loss reaches its minimum. In other words, **if we predict just one constant and evaluate it with MSE, the mean of the training targets is the right choice**.

Here, the training target mean is approximately **153.736544**. If we used the test target mean to choose the baseline, we would be using the evaluation answers for training. We must calculate it from **the training targets alone**. This baseline does not use individual feature values, such as weight or BMI, to make its predictions.

### Compare BMI alone with all features on the same test rows

The first linear model uses only the BMI column. Its column index is 2, so we pass `X_train[:, [2]]`. Using `[2]` keeps the input two-dimensional, with shape `(353, 1)`.

The second linear model uses all 10 features. We create a separate model object for each and call `fit` using only the 353 training rows. We then call `predict` on the inputs for the same 89 test rows and compare both sets of predictions with the same `y_test`.

| Method | What it learns | Test MSE |
| --- | --- | ---: |
| Mean baseline | One number: the mean of the training targets | 5361.533457 |
| Linear regression using only BMI | A BMI weight and an intercept | 4061.825928 |
| Linear regression using all 10 features | 10 weights and an intercept | 2900.193628 |

This table rounds results obtained with Python 3.12.14, NumPy 2.3.5, and scikit-learn 1.8.0, using `scaled=False`, `test_size=0.2`, and `random_state=42`. A smaller MSE means smaller squared errors on this test set.

**For this split, the model using all features has the smallest MSE.** This does not establish that adding features always improves performance or that the ranking will remain the same in every situation. Additional features may contain unhelpful information, and a model may fit incidental patterns in its training data.

In the code, we store the three prediction arrays by name in a **dictionary** called `predictions`. A dictionary pairs a name, called a key, with its corresponding value. We distinguish the results as `baseline`, `bmi`, and `all` so that after the loop we do not look only at the final `prediction` variable and lose track of which model it belongs to.

## 15. Residual plots: how far off, and in which direction?

MSE summarizes errors across many samples as a single number. To see the direction of each sample's error, examine its **residual**. For the residual plot in this article, we subtract in the following order:

<div class="supervised-math">
$$
\text{Residual}=y-\hat y
$$
</div>

This has **the opposite sign** from the training error $e=\hat y-y$. Different texts or programs may use the word “error” differently, so check the subtraction order. Either order gives the same squared value and therefore the same MSE, but the interpretations of positive and negative values differ.

| Residual | Meaning |
| --- | --- |
| Positive | The actual value is greater than the prediction. The model predicted too low. |
| 0 | The actual value and prediction are equal. |
| Negative | The actual value is less than the prediction. The model predicted too high. |

The first test sample for the model using all features is row index 287 in the original data. Its actual value is 219 and its prediction is approximately 139.547558, giving this residual:

<div class="supervised-math">
$$
219-139.547558\approx+79.452442
$$
</div>

A **scatter plot** shows each sample as a point positioned by its horizontal and vertical values. The figure below places predictions on the horizontal axis and residuals on the vertical axis. Each point is one test sample, for a total of 89 points.

![A scatter plot of predictions and residuals for 89 test samples. The first sample is highlighted with target 219, prediction about 139.55, and positive residual about 79.45.](/supervised-learning-basics/residuals-en.svg)

The horizontal reference line marks residual 0. The farther a point is from this line, the larger the error for that sample. The highlighted point is the first test sample calculated above. It lies above the line, so its prediction was too low.

This type of plot lets us look for patterns, such as residuals leaning in one direction as predictions increase or errors becoming more spread out in a particular range. Such patterns can be reasons to investigate the model's relationship or the conditions of the data further. A single plot cannot establish their cause or guarantee performance in actual use.

## 16. What this evaluation tells us, and what else needs checking

### Implementation checks and model evaluation answer different questions

Checking whether the small dataset produces coefficients `[1, 2, 3]` asks **whether the calculation is implemented as intended**. Calculating MSE on 89 real-data rows that were not used for training asks **how well the trained model predicts separate cases**.

Passing `assert` checks does not mean prediction performance is good. A small MSE on a particular dataset does not mean the entire implementation is free of errors, either. Performing 2000 training iterations and splitting the data into training and test sets also serve different purposes. Iteration updates the coefficients; splitting separates the cases used to assess performance.

### Repeatedly choosing models from the same test results changes that data's role

**Overfitting** occurs when a model fits the training data well but does not predict new data well. Looking only at the training loss is not enough to detect it reliably.

Use **validation data** when repeatedly comparing model types, learning rates, or feature choices. The training data determines the coefficients, the validation data helps compare settings, and a separately reserved test set provides the final evaluation.

If we keep changing the model based on the test table above and choose whichever gives the lowest MSE, those 89 rows effectively become validation data. After repeating that process, we should no longer describe the result as performance on a completely untouched final test set.[^leakage]

When data is limited, we can also use **cross-validation**. Split the data into several groups, leave one group out for validation in turn, train on the remaining groups, and collect the results. This helps us examine how sensitive the result is to one particular split. Plan both how to reserve data for the final evaluation and how to use cross-validation to compare settings.[^cross-validation]

### Apply the training-data boundary to preprocessing too

**Preprocessing** means cleaning or transforming data before passing it to a model. It includes filling in **missing values**, which are entries with no value, and adjusting input scales.

If a transformation derives quantities such as a mean from the data, **split into training and test sets first, then calculate those quantities using only the training data**. Apply the transformation determined during training unchanged to the test data. Filling missing entries using the mean of the entire dataset before splitting can allow test information into training. When information that should not be available during evaluation enters training or model selection, the problem is called **data leakage**.[^leakage]

Scikit-learn's `Pipeline` connects preprocessing and a model in sequence, helping apply the transformation learned during training when making predictions too. This article focuses on the basic least-squares regression calculation, so we have not added a preprocessing model. If new data requires transformations, the same boundary must be respected.

Also check whether the rows represent independent cases. If one person's records occupy several rows, consider the consequences of distributing that person's records across both training and test sets. Predicting the future may require a split that follows time order. A random split is not appropriate for every dataset.[^cross-validation]

When working with real data, also check the input–target pairing, missing values, duplicate records, units, and plausible ranges. The fixed split and three MSE values in this article form **a reproducible example of the full training, prediction, and evaluation process**. There is no universal passing MSE for every regression problem; comparisons and separate evaluation must fit the actual purpose.

## 17. Complete Python code to run from the beginning

The following code runs in Python 3. A **package** is a unit of software distributed so that it can be installed and used. The required packages are `numpy`, `matplotlib`, and `scikit-learn`. `matplotlib` draws the graphs, and scikit-learn is imported in code under the name `sklearn`.

In Jupyter Notebook, prepare the packages in a separate cell with `%pip install numpy matplotlib scikit-learn`, then place the following code in one cell or run it in order from top to bottom. `%pip` is an installation command for the notebook environment, so do not put it directly into an ordinary `.py` file. If the packages are already installed, there is no need to install them again.

**`import`** makes library functionality available for use. In `import numpy as np`, `as np` means that we will refer to `numpy` by the shorter name `np`. The form `from ... import ...` brings in a specific feature by name.

A `for` loop takes values one at a time and repeats the same work. `range(2000)` supplies 2000 numbers, from 0 through 1999. In `for _ in ...`, `_` is a conventional name indicating that the iteration number itself is not used in the calculation. The indented lines run as part of that iteration, so **preserve the indentation when copying the code**.

In the code, `len` gives the number of elements, `float` converts a value to a Python floating-point number, and `print` displays output. An array's `.shape[0]` is the first entry in its shape tuple, which is the number of rows for this two-dimensional input. Passing `name=value` to a function sets the named option to that value.

Writing several names side by side, as in `w, b = 1.0, 0.0`, assigns the values on the right to those names in order. The same applies to `X_real, y_real = ...` when a function returns multiple values. The operator `is` checks whether two names refer to the same object. Thus, `fit_return is model_small` checks whether `fit` returned the original model object itself.

`history_complete[:shown]` is **slicing**: it selects from the beginning up to, but not including, index `shown`. Because the endpoint is excluded, `shown=81` selects indices 0 through 80. The function `min` finds the smaller of the supplied values, while `np.max` finds the largest value in an array. `scores = {}` creates an empty dictionary, and `scores[name] = value` stores an entry under that name.

To create a graph, `plt.subplots` returns the overall figure and a plotting area. `plot` draws a line graph, and `scatter` draws a scatter plot. `set` sets the axis labels and title, `legend` displays the names of points or lines in a legend, and `plt.show()` displays the prepared figures. The relevant code lines also explain the individual settings.

[Download the Python example](/supervised-learning-basics/examples-en.py)

```python
"""Supervised Learning: Core Concepts and Basic Coding — runnable Python examples.

Required packages: numpy, matplotlib, scikit-learn
In Jupyter, install in a separate cell: %pip install numpy matplotlib scikit-learn
Run the code below from top to bottom in one pass.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Change only the number of displayed decimal places, not the calculation precision.
np.set_printoptions(precision=6, suppress=True)

# 1. Errors, sum of squares, mean squared error, and training loss J
y_demo = np.array([10.0, 20.0, 30.0])
pred_demo = np.array([9.0, 18.0, 27.0])
e_demo = pred_demo - y_demo
sse = np.sum(e_demo ** 2)       # ** 2: square each element
mse = np.mean(e_demo ** 2)      # mean: divide the sum of the elements by their count
j_demo = mse / 2                # The factor 1/2 is not part of the definition of MSE itself
mae = np.mean(np.abs(e_demo))    # abs: magnitude without the sign
rmse = np.sqrt(mse)             # sqrt: square root
print("Errors:", e_demo)
print("SSE, MSE, J, MAE, RMSE:", sse, mse, j_demo, mae, rmse)

# 2. Indexing: select rows and columns, and check the resulting shape
A = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
row_ids = [0, 2]
index_examples = {
    "A[1]": A[1],
    "A[[1]]": A[[1]],
    "A[:, 2]": A[:, 2],
    "A[:, [2]]": A[:, [2]],
    "A[row_ids][1]": A[row_ids][1],
    "A[row_ids][:, [2]]": A[row_ids][:, [2]],
    "A[[0, 2], [1, 2]]": A[[0, 2], [1, 2]],
}
# items(): retrieve each name and result as a pair from the dictionary
for label, values in index_examples.items():
    print(label, "=", values, "shape =", values.shape)
print("A[1][2] =", A[1][2])

# 3. Adding axes, reshaping, and possible broadcasting mistakes
v = np.array([1.0, 3.0, 5.0])
p = np.array([0.0, 2.0, 4.0])
print("Shapes of v and v.T:", v.shape, v.T.shape)
print("v[:, None]:", v[:, None], "shape =", v[:, None].shape)
print("v[None, :]:", v[None, :], "shape =", v[None, :].shape)
print("reshape(-1, 1):", v.reshape(-1, 1))
print("Correct errors at matching positions:", p - v)
print("Unintended 3×3 result:\n", p - v[:, None])
assert p.shape == v.shape

# 4. Use derivatives to update a model with one input feature once
x_one = np.array([0.0, 1.0, 2.0])
y_one = np.array([1.0, 3.0, 5.0])
w, b = 1.0, 0.0
lr_one = 0.1
pred_one = b + w * x_one
e_one = pred_one - y_one
j_before = np.mean(e_one ** 2) / 2
dw = np.mean(e_one * x_one)      # Current value of the derivative of J with respect to w
db = np.mean(e_one)              # Current value of the derivative of J with respect to b
# Compute both derivatives at the existing w and b before updating either coefficient.
w = w - lr_one * dw
b = b - lr_one * db
pred_after = b + w * x_one
mse_after = np.mean((pred_after - y_one) ** 2)
print("J, dw, db before the single update:", j_before, dw, db)
print("w, b, MSE after the update:", w, b, mse_after)
assert np.isclose(j_before, 7 / 3, atol=1e-12, rtol=0)
assert np.isclose(mse_after, 1829 / 675, atol=1e-12, rtol=0)

# 5. A separate dataset with four rows and two input features
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = np.array([1.0, 3.0, 4.0, 6.0])
n = len(y)
# ones(n): create n ones. c_: join arrays side by side as columns.
Xb = np.c_[np.ones(n), X]
theta = np.zeros(3)              # Coefficient order: [b, w1, w2]
e0 = Xb @ theta - y              # @: matrix multiplication of each row by the coefficient vector
g0 = Xb.T @ e0 / n              # .T: swap the rows and columns of this two-dimensional array
j0 = np.mean(e0 ** 2) / 2
trial_theta = theta - 0.1 * g0   # Store the result under another name; theta is still zero
print("Xb:\n", Xb)
print("Initial J, gradient:", j0, g0)
print("Trial update with learning rate 0.1:", trial_theta)
print("Original theta after the trial calculation:", theta)

# 6. Run the actual iterations with learning rate 0.2, starting from zero coefficients
lr = 0.2
n_steps = 2000
history = []                     # List that stores each loss immediately before the corresponding update
for _ in range(n_steps):         # range(2000): 0 through 1999, for a total of 2000 iterations
    pred = Xb @ theta
    e = pred - y
    history.append(float(np.mean(e ** 2) / 2))
    gradient = Xb.T @ e / n
    theta -= lr * gradient      # In-place update that changes the contents of the existing array

theta_gd = theta.copy()          # Independent copy that preserves the result if theta changes later
pred_gd = Xb @ theta_gd
final_j = float(np.mean((pred_gd - y) ** 2) / 2)
history_complete = np.array(history + [final_j])
print("GD coefficients:", theta_gd)
print("GD predictions:", pred_gd)
print("First loss / last stored loss / final loss:", history[0], history[-1], final_j)
print("Length including the loss after 2000 updates:", len(history_complete))

# subplots returns the entire figure, fig, and the plotting area, ax.
fig_loss, ax_loss = plt.subplots(figsize=(7, 4))  # figsize is measured in inches
shown = min(81, len(history_complete))           # 0 through 80 completed updates
ax_loss.plot(np.arange(shown), history_complete[:shown])
ax_loss.set(xlabel="Completed updates", ylabel="J = MSE / 2",
            title="Gradient descent on four samples")
ax_loss.grid(alpha=0.25)           # alpha: opacity of the grid lines
fig_loss.tight_layout()           # Adjust margins to prevent overlap of the title and axis labels

# 7. Compare learning rates using the separate function J(t)=t²/2
# Here, t is a single number, separate from theta in the original regression model.
for rate in [0.2, 0.8, 1.5, 2.2]:
    t = 3.0
    t_history = [t]              # Start recording at the state with zero completed updates
    for _ in range(8):
        t -= rate * t           # The derivative of this function at t is t
        t_history.append(t)
    losses = np.array(t_history) ** 2 / 2
    print("Learning rate", rate, "t/J after the first update:", t_history[1], losses[1],
          "J after 8 updates:", losses[-1])

# 8. Apply least squares and scikit-learn to the same small dataset
# Four returned values: coefficients, array of residual sums of squares, rank, singular values.
theta_ls, residual_sums, rank, singular_values = np.linalg.lstsq(
    Xb, y, rcond=None
)
pred_ls = Xb @ theta_ls
model_small = LinearRegression()  # Default: fit_intercept=True
fit_return = model_small.fit(X, y)  # Use X before adding the column of ones
pred_sk = model_small.predict(X)
print("Least-squares coefficients:", theta_ls)
print("Other values returned by lstsq:", residual_sums, rank, singular_values)
print("Does fit return the same model object:", fit_return is model_small)
print("sklearn intercept / weights:", model_small.intercept_, model_small.coef_)
print("Compare least-squares / GD / sklearn predictions by column:\n", np.c_[pred_ls, pred_gd, pred_sk])
new_input = np.array([1.0, 3.0, 4.0])  # [1 for the intercept, x1, x2]
print("GD prediction for x1=3, x2=4:", new_input @ theta_gd)

# 9. Check the implementation: verify shapes first, then compare numbers within a tolerance
assert theta_gd.shape == (3,)
assert pred_gd.shape == y.shape
assert np.isfinite(theta_gd).all()  # Check that every coefficient is neither NaN nor infinite
assert np.isfinite(history_complete).all()
assert np.isclose(j0, 7.75, atol=1e-12, rtol=0)
assert np.allclose(g0, [-3.5, -2.25, -2.5], atol=1e-12, rtol=0)
assert np.allclose(theta_gd, [1.0, 2.0, 3.0], atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_ls, atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_sk, atol=1e-8, rtol=0)
assert final_j < history_complete[0]
print("Implementation checks passed. Maximum GD/least-squares prediction difference:", np.max(np.abs(pred_gd - pred_ls)))

# 10. Real data: stored feature values without the additional scaling offered by the loader
X_real, y_real = load_diabetes(return_X_y=True, scaled=False)
# Split the row indices and apply the same indices to both arrays.
all_ids = np.arange(len(y_real))
train_idx, test_idx = train_test_split(
    all_ids, test_size=0.2, random_state=42
)
X_train, X_test = X_real[train_idx], X_real[test_idx]
y_train, y_test = y_real[train_idx], y_real[test_idx]
assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]
print("Full / training / test input shapes:", X_real.shape, X_train.shape, X_test.shape)

# 11. Baseline: predict the mean of the training targets for every test row
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_test)

# 12. Regression using only the BMI feature. Select with [2] to retain the column dimension.
model_bmi = LinearRegression()
model_bmi.fit(X_train[:, [2]], y_train)
pred_bmi = model_bmi.predict(X_test[:, [2]])

# 13. Regression using all ten features. This is a separate object from the BMI model above.
model_all = LinearRegression()
model_all.fit(X_train, y_train)
pred_all = model_all.predict(X_test)
predictions = {"baseline": pred_baseline, "bmi": pred_bmi, "all": pred_all}
scores = {}
for name, prediction in predictions.items():
    assert prediction.shape == y_test.shape
    assert np.isfinite(prediction).all()
    scores[name] = mean_squared_error(y_test, prediction)
    print(name, "Test MSE:", scores[name])
print("Mean training target used by the baseline:", y_train.mean())

# 14. Residual = actual value - prediction, the opposite sign from e used during training.
residual = y_test - pred_all
print("Original row index of the first test sample:", test_idx[0])
print("First test target / prediction / residual:", y_test[0], pred_all[0], residual[0])
fig_residual, ax_residual = plt.subplots(figsize=(7, 4))
ax_residual.scatter(pred_all, residual, alpha=0.65, label="Test samples")
ax_residual.scatter([pred_all[0]], [residual[0]], color="crimson",
                    s=90, label="First test sample")  # s: marker area
ax_residual.axhline(0, color="black", linewidth=1)  # Reference line at residual 0
ax_residual.set(xlabel="Prediction", ylabel="Residual = target - prediction",
                title="Residuals on 89 held-out samples")
ax_residual.legend()
fig_residual.tight_layout()
plt.show()                       # Display the two prepared figures
```

Running the code prints the errors, array shapes, update results, and test MSE values for the three methods calculated above, then displays the loss and residual plots. When their conditions hold, `assert` checks pass without printing anything. Check that execution reaches the end and that the final output appears. The last decimal places may vary slightly between environments.

## 18. Implementing the same gradient descent in Rust

The Python example uses `@` and `np.mean` to express many multiplications and additions at once. In Rust, we can write the same calculations directly with loops and no external numerical package. The code below translates **only gradient descent on the four-row practice dataset**. The earlier Python example includes the real-data split and evaluation.

**Rust** is another programming language. Normally, a **compiler** converts the source code into an executable, which is then run. A compiler checks the syntax and data types of the code and converts it into an executable form. You can put the following code into `main.rs` in a Rust binary project.

| Rust expression | Meaning and correspondence with the Python example |
| --- | --- |
| `fn main()` | The function where the program starts |
| `let` | Declares a variable |
| `mut` | Allows the variable's value to be changed later |
| `f64` | A 64-bit floating-point number type |
| `[f64; 3]` | A fixed-length array containing 3 floating-point numbers |
| `[[f64; 3]; 4]` | 4 rows of length 3: `Xb` in this example |
| `[0.0_f64; 3]` | An array containing 3 zeros of type `f64` |
| `0..2000` | A range excluding its endpoint, giving 2000 iterations |
| `as f64` | Converts a length to a floating-point value for division |
| `+=`, `-=` | Adds to or subtracts from an existing value to update it |
| `assert!(condition)` | Checks whether the condition holds |
| `println!` | Prints one line of output |

Array positions start at 0, as in Python. Here, braces delimit functions and loops, and statements end with semicolons. In `println!`, `{:?}` displays an array's contents, while `{:.6}` displays a number with six digits after the decimal point.[^rust]

The code first calculates a prediction for each sample, then accumulates each error's contribution to the derivative for each coefficient. **It updates the coefficients only after calculating the derivatives for all four rows.** It therefore uses the same mean gradient as Python's `Xb.T @ e / n`.

[Download the Rust example](/supervised-learning-basics/gradient-descent-en.rs)

```rust
// Compute gradient descent on the same four-row dataset without external packages.
// Save this file as main.rs in a Rust binary project to run it.
fn main() {
    // Each row is [1 for the intercept, x1, x2]. f64 is a 64-bit floating-point type.
    let xb: [[f64; 3]; 4] = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let y: [f64; 4] = [1.0, 3.0, 4.0, 6.0];
    let mut theta = [0.0_f64; 3]; // mut: a variable that can be updated, [b, w1, w2]
    let lr = 0.2;
    let n = y.len() as f64;       // Convert the array length for floating-point calculations

    for _ in 0..2000 {           // The upper bound 2000 is excluded, giving 2000 iterations
        let mut gradient = [0.0_f64; 3];
        for i in 0..y.len() {
            let mut prediction = 0.0;
            for j in 0..theta.len() {
                prediction += xb[i][j] * theta[j];
            }
            let error = prediction - y[i];
            for j in 0..theta.len() {
                gradient[j] += xb[i][j] * error / n;
            }
        }
        // Compute gradients for all rows at the existing coefficients, then update them together.
        for j in 0..theta.len() {
            theta[j] -= lr * gradient[j];
        }
    }

    let expected = [1.0, 2.0, 3.0];
    for j in 0..theta.len() {
        assert!((theta[j] - expected[j]).abs() < 1e-8);
    }
    let new_input = [1.0, 3.0, 4.0];
    let mut new_prediction = 0.0;
    for j in 0..theta.len() {
        new_prediction += new_input[j] * theta[j];
    }
    println!("Coefficients [b, w1, w2]: {:?}", theta);
    println!("Prediction for the new input: {:.6}", new_prediction);
}
```

The expected result is a set of coefficients close to `[1, 2, 3]` and a prediction of `19` for the new input. The Python example in this article was run and checked from beginning to end. The Rust example received a static review of the corresponding expressions, arrays, and update order. Rust compilation and execution were not performed in the environment used to write this article.

We can now connect the roles of the values in the code as follows:

| Concept | What it is | Code in this article |
| --- | --- | --- |
| Input | Information used for prediction | `X`, `X_train`, `X_test` |
| Target | The value used for comparison during training or evaluation | `y`, `y_train`, `y_test` |
| Coefficients | Numbers that define the prediction rule and are determined by training | `theta`, `coef_`, `intercept_` |
| Prediction | The result calculated from inputs and coefficients | `pred`, `model.predict(...)` |
| Loss | A number summarizing how wrong predictions are | `MSE`, and this article's `J = MSE / 2` |
| Gradient | The current rate of change of the loss with respect to each coefficient | `Xb.T @ e / n` |
| Training | The process of finding coefficients from data | Our manually written update loop or `fit` |
| Evaluation | Comparing predictions on separate data with the targets | Test MSE and the residual plot |

When learning a new model, first identify **the input shape, the meaning of the target, the coefficients being changed, the loss being reduced, and the data used for evaluation**. These connections help you read the code as a sequence of related calculations.

## References

The calculations in the main text were checked directly with the examples included above. The following official documentation was used to confirm library behavior and dataset descriptions.

[^sklearn-start]: scikit-learn, [Getting Started](https://scikit-learn.org/stable/getting_started.html). Input and target shapes, `fit`, `predict`, and the convention for attributes obtained through training.
[^mse]: scikit-learn, [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html). The definition of mean squared error and the function's inputs and outputs.
[^indexing]: NumPy, [Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html). Integer and list indexing, preserving axes, and multiple selections.
[^broadcasting]: NumPy, [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). Shape rules that compare axes from the right.
[^derivative]: MIT OpenCourseWare, [Introduction to Derivatives](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/session-1-introduction-to-derivatives/). The basics of rates of change and differentiation.
[^gradient]: MIT OpenCourseWare, [Partial Derivatives and the Gradient](https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter06/section06.html). Partial derivatives and the gradient vector.
[^chain-rule]: MIT OpenCourseWare, [The Chain Rule (PDF)](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/66ba9836b3c9e99138bc8d766d913bc5_MIT18_01SCF10_Ses11a.pdf). Differentiation when functions are composed.
[^gradient-descent]: MIT OpenCourseWare, [Gradient Descent: Downhill to a Minimum](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-22-gradient-descent-downhill-to-a-minimum/). The basic principles of gradient descent.
[^matmul]: NumPy, [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html). Matrix–vector multiplication and dimension rules.
[^lstsq]: NumPy, [numpy.linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html). Least-squares solutions and the meanings of the four returned values.
[^linear-regression]: scikit-learn, [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). The default intercept setting and solution methods for different input formats.
[^allclose]: NumPy, [numpy.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html). Absolute and relative tolerances, and broadcasting.
[^assert]: Python, [The assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement). Assertions and their behavior during optimized execution.
[^diabetes]: scikit-learn, [load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) and [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset). Dataset size, features, target, and the `scaled` option.
[^split]: scikit-learn, [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Split sizes and reproducibility settings.
[^baseline]: scikit-learn, [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html). A baseline using the mean of the training targets.
[^leakage]: scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html). Separating test data, limiting preprocessing to the appropriate data, and data leakage.
[^cross-validation]: scikit-learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html). The distinction between validation and test data, and splits that respect groups or time order.
[^rust]: The Rust Programming Language, [Variables and Mutability](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html), [Data Types](https://doc.rust-lang.org/book/ch03-02-data-types.html), [Control Flow](https://doc.rust-lang.org/book/ch03-05-control-flow.html).
