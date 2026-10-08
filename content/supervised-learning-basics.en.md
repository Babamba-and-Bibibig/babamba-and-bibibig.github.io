+++
title = "Supervised Learning: Core Concepts and Basic Coding"
description = "Follow a small delivery-fee prediction problem to understand supervised learning, errors and loss, derivatives, and gradient descent, then check the calculations with NumPy and scikit-learn."
date = 2026-10-07
updated = 2026-10-08
slug = "supervised-learning-basics"

[extra]
katex = true
toc = true
styles = ["supervised-learning-basics/article.css"]
+++

Imagine building a program that gives customers an estimated delivery fee when they place an order. The program needs to estimate the fee before the delivery is complete. The information we have comes from completed orders: their delivery distances and the fees actually charged.

To find a prediction rule in these records, we need to answer three questions. **What information will we give the program? How will we judge how far its predictions are from the recorded fees? Which parts of the rule should we change, and how, to improve the predictions?**

This article works through those questions using one delivery-fee example. Derivatives and gradient descent will arise as we need them. We will first adjust the base fee or the per-kilometer fee in our prediction rule by a small amount and observe how the predicted fees and the loss change. We will then express those changes mathematically and put the same calculations into code.

## 1. Using past delivery records to predict the fee for a new order

The following **fictional training records** were created to help us work through the calculations. They are not a real delivery company's price list.

Here, the distance is **the additional distance beyond the distance included in the base fee**, rather than the total delivery distance. An order completed within the included distance has an additional distance of 0 km. To keep the numbers simple, we express delivery fees in thousands of Korean won.

| Order | Additional distance | Actual delivery fee |
|---|---:|---:|
| A | 0 km | 1 thousand won |
| B | 1 km | 3 thousand won |
| C | 2 km | 5 thousand won |

Now suppose a new order has an additional distance of 3 km. We want to use the past records to predict this order's delivery fee.

For this problem, we give the program the order's **additional distance**. Information used to make a prediction is called an **input**. An individual item of input information is also called a **feature**. For now, we have one feature: additional distance.

The **actual delivery fee** for a past order is the value against which we compare the program's prediction. A value supplied as the answer to predict in training data is called a **target**, or **target value**.

**Supervised learning** uses data containing both inputs and targets to find a rule for predicting the target from the input. One piece of training data containing an input and a target, such as a single order in the table, is called a **sample**.[^sklearn-start]

A supervised learning problem that predicts a numerical value, such as a delivery fee, is called **regression**. A problem that predicts a category, such as whether an email is spam or legitimate, is called **classification**. This article focuses on regression.

![Prepare inputs and targets, repeatedly adjust the prediction rule, then check it using data that was not used for training.](/supervised-learning-basics/learning-cycle-en.svg)

Once training is complete, we give the program the additional distance for a new order. The actual fee for that order is not yet known when we make the prediction. If we can obtain the actual fee later, we can use it to evaluate how accurate the prediction was.

## 2. Building a prediction rule from a base fee and a fee per kilometer

Let us predict the delivery fee as the sum of two parts:

- A **base fee** applied to every order.
- A **fee per additional kilometer**, which adds more to the total as the additional distance increases.

If the base fee is 1 thousand won and the fee per additional kilometer is 2 thousand won, an order with an additional distance of 3 km has the following predicted fee:

<div class="supervised-math">
$$
1+2\times 3=7
$$
</div>

The result, 7, is also in thousands of won, so the predicted delivery fee is 7 thousand won.

We will give each value a name so that we can use the same calculation with other fees and distances.

| Symbol | What it represents in this example | Unit |
|---|---|---|
| $x$ | The order's additional distance | km |
| $y$ | The actual delivery fee in the record | Thousands of won |
| $\hat y$ | The delivery fee predicted by the rule | Thousands of won |
| $b$ | The base fee used by the rule | Thousands of won |
| $w$ | The fee per additional kilometer used by the rule | Thousands of won/km |

We read $\hat y$ as “y hat.” The hat above $y$ marks it as **a predicted value**, rather than the value in the actual record.

With these symbols, we can write our prediction rule as follows:

<div class="supervised-math">
$$
\hat y=b+wx
$$
</div>

$wx$ is a shorter way to write $w\times x$. We multiply the fee per kilometer by the distance, then add the base fee to obtain the delivery fee.

A rule that takes an input and produces a prediction is called a **model**. In this model, $b$ and $w$ are the numbers we adjust. Such numbers are called **coefficients** or **parameters**.

Predicting a number by multiplying each input by a fixed coefficient and adding a base value is the basic form of **linear regression**. In machine learning, $b$ is also called the **bias**, and $w$ the **weight**. On a graph, the predicted value at $x=0$ is $b$, so $b$ is also the **intercept**.

The key is that **we apply the same $b$ and $w$ to every order**. The distance $x$ varies between orders. Each order also has its own actual fee $y$, but we do not change those records. The values we change to improve the predictions are $b$ and $w$.

To examine the learning process, we will deliberately start with coefficients that do not fit well: $b=0$, $w=1$.

| Order | Additional distance $x$ | Actual fee $y$ | Calculation with the current rule | Predicted fee $\hat y$ |
|---|---:|---:|---|---:|
| A | 0 | 1 | $0+1\times0$ | 0 |
| B | 1 | 3 | $0+1\times1$ | 1 |
| C | 2 | 5 | $0+1\times2$ | 2 |

This rule predicts a fee below the actual fee for all three orders. How should we change it to make the predictions more accurate?

With this small table, we can see that $b=1$, $w=2$ would match all three records exactly. But when there are many records and several input features, choosing coefficients by inspection becomes difficult. The calculations that follow will let the computer **evaluate its current predictions and adjust the coefficients**.

## 3. Comparing prediction rules with a numerical score

### How far is each prediction off, and in which direction?

First, we will subtract the actual delivery fee from the predicted fee. In this article, we call the result the **error** and write it as $e$.

<div class="supervised-math">
$$
e=\hat y-y
$$
</div>

| Order | Predicted fee | Actual fee | Error $e$ | Interpretation |
|---|---:|---:|---:|---|
| A | 0 | 1 | −1 | The prediction is 1 thousand won below the actual fee. |
| B | 1 | 3 | −2 | The prediction is 2 thousand won below the actual fee. |
| C | 2 | 5 | −3 | The prediction is 3 thousand won below the actual fee. |

The sign of the error tells us the direction. With our definition, a negative error means an underprediction, and a positive error means an overprediction. Other sources may subtract in the opposite order, so check the subtraction order before interpreting an expression.

### Why not just add the errors?

Suppose a rule predicts one order's fee 2 thousand won too high and another order's fee 2 thousand won too low. Adding the errors gives $2+(-2)=0$. But both predictions were wrong. We cannot call them perfect predictions just because their errors add up to 0.

To prevent errors in opposite directions from canceling, we will use **the square of each error**. Squaring a number means multiplying it by itself. Since $(-2)^2=(-2)\times(-2)=4$, a squared error is nonnegative whether the original error was positive or negative.

The current errors for our three orders are −1, −2, and −3. Their squares are 1, 4, and 9, which add up to 14. This total is called the **sum of squared errors**, or **SSE**.

To make comparisons on a per-order basis even when datasets contain different numbers of orders, we divide that total by the number of orders, which is 3. Adding values and dividing by their count gives their **mean**.

<div class="supervised-math">
$$
\text{MSE}=\frac{(-1)^2+(-2)^2+(-3)^2}{3}
=\frac{14}{3}\approx4.666667
$$
</div>

This value is the **mean squared error**, or **MSE**. The symbol $\approx$ means that the values are close but not exactly equal because we shortened the decimal representation. When we compare two models predicting the same orders, the model with the smaller MSE produces smaller squared errors on average.[^mse]

Squaring gives a larger error more weight in the score. When the error magnitude triples from 1 to 3, its square increases ninefold, from 1 to 9. As a result, training that reduces MSE is strongly influenced by orders with large prediction errors.

### The loss we will use when adjusting the coefficients

The score we try to reduce when training a model is called the **loss**. In this article, we will use half the MSE as our loss and call its value $J$.

The symbols $e_1$, $e_2$, and $e_3$ are the errors for the first, second, and third orders, respectively. The small numbers below the letters indicate the order number.

<div class="supervised-math">
$$
J=\frac{\text{MSE}}{2}
=\frac{e_1^2+e_2^2+e_3^2}{2\times3}
$$
</div>

Why is it acceptable to halve the score? If two models have MSE values of 2 and 6, their $J$ values are 1 and 3. Dividing every model's score by the same positive number, 2, does not change which score is smaller. The coefficients that make MSE smallest therefore also make $J$ smallest. Later, when we expand the change in the loss, a factor of 2 from the square cancels the 2 in the denominator, simplifying the calculation.

The current loss is $J=14/6=7/3\approx2.333333$. To check the calculation one order at a time, think of assigning each order a loss of $e^2/2$, then taking the mean of those three losses.

| Order | Error | Loss for that order, $e^2/2$ |
|---|---:|---:|
| A | −1 | 0.5 |
| B | −2 | 2 |
| C | −3 | 4.5 |

The mean of these three values is $(0.5+2+4.5)/3=7/3$.

Neither $J$ nor MSE is an actual delivery fee. Each is a score used to evaluate predictions. If the fee is measured in thousands of won, squared error is measured in “thousands of won, squared.” An MSE of 4 does not mean that the average prediction is off by 4 thousand won.

<details>
<summary>Two ways to express errors in thousands of won: MAE and RMSE</summary>

**Mean absolute error (MAE)** averages the error magnitudes with their signs removed. The vertical bars denoting an absolute value mean that we take only the magnitude of the number. For example, $|-2|=2$. In our current example, MAE is $(1+2+3)/3=2$, so the average error magnitude per order is 2 thousand won.

**Root mean squared error (RMSE)** is the square root of MSE. Here, taking a square root means finding the nonnegative number that gives the specified value when squared. The current RMSE is $\sqrt{14/3}\approx2.160247$ thousand won.

Both MAE and RMSE can be expressed in the same unit as the delivery fee, but they use different calculations. RMSE squares the errors first, making it more sensitive to large errors. In a practical task, you can examine both measures according to how much ordinary errors matter and how serious unusually large errors are.

</details>

## 4. What happens to the predictions if we raise the base fee by 100 won?

The current rule uses $b=0$, $w=1$. Let us raise only the base fee $b$, from 0 to 0.1. Because our monetary unit is a thousand won, adding 0.1 means **raising the base fee by 100 won**. We leave the per-kilometer fee $w$ and the order records unchanged.

Every order has the same base fee, so each predicted delivery fee increases by 0.1. The actual fees do not change. Each error, calculated as “predicted fee minus actual fee,” therefore also increases by 0.1. In this case, the negative errors move closer to 0, as when −1 becomes −0.9. The magnitude of each prediction error therefore decreases.

![A calculation showing how predicted fees and errors change when the base fee changes from 0 to 0.1.](/supervised-learning-basics/intercept-change-en.svg)

We can check whether this was a useful change by calculating the loss.

| Order | Error before the change | Error after the change | Loss before the change | Loss after the change | Change in loss |
|---|---:|---:|---:|---:|---:|
| A | −1 | −0.9 | 0.5 | 0.405 | −0.095 |
| B | −2 | −1.9 | 2 | 1.805 | −0.195 |
| C | −3 | −2.9 | 4.5 | 4.205 | −0.295 |

The last column is “loss after the change minus loss before the change.” For example, order B now has a loss of $(-1.9)^2/2=1.805$, so its loss changed by $1.805-2=-0.195$. The negative value means that the loss decreased.

Taking the mean of the three orders' losses, the overall loss decreased from approximately 2.333333 to 2.138333. The change in the overall loss is −0.195. Raising the base fee by 100 won improved the predictions on these records.

But how much would the loss change if we raised the fee by just 10 won or 1 won instead of 100 won? A 100-won adjustment and a 10-won adjustment start with different changes to the fee, so the decrease in loss alone is difficult to compare. We will **divide the change in loss by the change in the base fee** to examine how strongly the loss responds to each adjustment.

<div class="supervised-math">
$$
\text{Rate of change of the loss}
=\frac{\text{Loss after the change}-\text{Loss before the change}}{\text{Change in the base fee}}
$$
</div>

For the 100-won adjustment, this ratio is $-0.195/0.1=-1.95$. This is the rate of change over the interval in which the base fee moves from 0 to 0.1. It does not mean that raising the base fee by a thousand won would decrease the loss by exactly 1.95. To examine **the change very close to the current base fee**, we need to make the adjustment smaller.

Now return to the original $b=0$, $w=1$ before each calculation, and change only the base fee by each of the following amounts. These adjustments are not applied cumulatively.

| Change in the base fee | Change in the overall loss | Change in loss ÷ change in the base fee |
|---:|---:|---:|
| 0.1 | −0.195 | −1.95 |
| 0.01 | −0.01995 | −1.995 |
| 0.001 | −0.0019995 | −1.9995 |
| −0.001 | 0.0020005 | −2.0005 |
| −0.01 | 0.02005 | −2.005 |

As we make the increases and decreases in the base fee very small, the values in the last column approach −2 from both sides. This value tells us **how sensitively the loss responds near the current base fee**.

If the rate of change approaches a particular value as the size of the adjustment approaches 0, we call that value the **derivative at the current point**. **Differentiation** is the calculation used to find this rate of change.[^derivative]

We do not set the adjustment to 0 at the start, because we cannot divide by 0. Instead, we divide by small, nonzero adjustments and examine which value the results approach.

In this example, the negative derivative tells us that increasing the current base fee slightly decreases the loss. That agrees with our situation: we are raising predicted fees that were too low. In the next section, we will find out why the derivative is −2 and derive a calculation we can also use at other coefficient values.

## 5. Calculating how the loss changes with the base fee

### Giving the change in the base fee a name

Earlier, we changed the base fee by 0.1, 0.01, and 0.001. To express all these adjustments together, **we will write the amount added to the base fee as $h$**. An adjustment of $h=0.1$ raises the base fee by 100 won, while $h=-0.1$ lowers it by 100 won.

If the current base fee is $b$, the adjusted base fee is $b+h$. Again, we hold the per-kilometer fee $w$, the order's distance $x$, and its actual fee $y$ fixed.

First, let us calculate how the error changes for one order. If its error before the adjustment is $e$, then $e=b+wx-y$. To find the error after the adjustment, **replace $b$ in the existing expression with $b+h$**.

<div class="supervised-math">
$$
\begin{aligned}
\text{Error after the change}
&=(b+h)+wx-y\\
&=(b+wx-y)+h\\
&=e+h
\end{aligned}
$$
</div>

On the second line, we rearranged the terms being added and subtracted. On the third line, we replaced $b+wx-y$ with the original error, $e$. This is the same result we observed earlier: raising the base fee by 100 won also increases the error by 100 won. Throughout the following expansion, $e$ continues to mean **the error before the adjustment**.

### How much does one order's loss change?

The loss for one order is its error squared and then divided by 2. Its original loss is $e^2/2$, and its loss after the adjustment is $(e+h)^2/2$. To find the difference, we first need to expand $(e+h)^2$.

<div class="supervised-math">
$$
\begin{aligned}
(e+h)^2
&=(e+h)(e+h)\\
&=e\times e+e\times h+h\times e+h\times h\\
&=e^2+2eh+h^2
\end{aligned}
$$
</div>

The two middle terms, $eh$ and $he$, have the same value, so we combined them as $2eh$. Now subtract the original loss from the new loss.

<div class="supervised-math">
$$
\begin{aligned}
\text{Change in one order's loss}
&=\frac{(e+h)^2}{2}-\frac{e^2}{2}\\
&=\frac{e^2+2eh+h^2-e^2}{2}\\
&=eh+\frac{h^2}{2}
\end{aligned}
$$
</div>

The original $e^2$ terms cancel, and dividing $2eh$ by 2 leaves $eh$.

We can check the result with order B's original error, $e=-2$, and a base-fee adjustment of $h=0.1$.

<div class="supervised-math">
$$
(-2)\times0.1+\frac{0.1^2}{2}
=-0.2+0.005
=-0.195
$$
</div>

This matches the result from the previous section, where the loss changed from 2 to 1.805, a change of −0.195.

### What do we add when evaluating all three orders?

The value $J$ that we want to reduce is the mean of the three orders' losses. Let us write the overall loss before adjusting the base fee as $J$ and the overall loss after the adjustment as $J_{\text{new}}$. The subscript “new” marks the new value after the adjustment.

First, check the difference in the overall loss using the numbers.

<div class="supervised-math">
$$
\begin{aligned}
J_{\text{new}}-J
&=\frac{0.405+1.805+4.205}{3}
-\frac{0.5+2+4.5}{3}\\
&=\frac{(0.405-0.5)+(1.805-2)+(4.205-4.5)}{3}\\
&=\frac{-0.095-0.195-0.295}{3}\\
&=-0.195
\end{aligned}
$$
</div>

**To find the change in the overall loss, we add the changes in the individual orders' losses and divide by 3.** The expression above shows why this works by combining two fractions with the same denominator.

Now write the same calculation using $h$. The first order's loss changes by $e_1h+h^2/2$. For the second and third orders, we put each order's own error into the same expression.

<div class="supervised-math">
$$
\begin{aligned}
J_{\text{new}}-J
&=\frac{
(e_1h+h^2/2)+(e_2h+h^2/2)+(e_3h+h^2/2)
}{3}\\
&=\frac{h(e_1+e_2+e_3)+3h^2/2}{3}\\
&=h\frac{e_1+e_2+e_3}{3}+\frac{h^2}{2}
\end{aligned}
$$
</div>

On the second line, we took the common factor $h$ outside the brackets. The term $h^2/2$ also appears three times, giving $3h^2/2$. Dividing by the number of orders, 3, gives the last line.

This expression is not yet a derivative. It calculates the **actual change in the loss** exactly when we change the base fee by $h$.

### Divide by the change in the base fee, then make that change smaller

To obtain the rate of change from the previous section, divide the change in loss by the change in the base fee, $h$. When $h$ is not 0, the expression simplifies as follows:

<div class="supervised-math">
$$
\frac{J_{\text{new}}-J}{h}
=\frac{e_1+e_2+e_3}{3}+\frac{h}{2}
$$
</div>

The first term on the right is **the mean of the three original errors**. The second term is $h/2$. As the adjustment $h$ approaches 0, $h/2$ also approaches 0. The rate of change of the overall loss therefore approaches the mean of the three original errors.

Therefore, **the derivative of the overall loss with respect to the current base fee equals the mean of the errors produced by the current model**. We arrived at this result by calculating how each order's loss changed, adding those changes, and dividing by the number of orders.

Substituting the current errors, −1, −2, and −3, gives:

<div class="supervised-math">
$$
\frac{-1-2-3}{3}=-2
$$
</div>

The algebra now explains why the rates of change in our experiment approached −2 as we made smaller adjustments.

In the program, we will store the result of this calculation under the name `db`. **`db` is the rate at which the loss changes when we change the base fee $b$. It is neither a new base fee nor an amount to add directly to the base fee.** We will decide how much to adjust the coefficient in a later step.

<details>
<summary>Notation for more than three orders</summary>

If we write the total number of orders as $n$, the same calculation replaces the order count 3 with $n$. We use $i$ for an order number and $e_i$ for the error of the $i$th order.

Adding all the errors gives $e_1+e_2+\cdots+e_n$. The dots indicate that we continue adding the intervening terms in the same way up to the final term. We can shorten this long sum to $\sum_{i=1}^{n}e_i$. The symbol $\sum$, called “sigma,” means a sum, and the $i=1$ below it together with the $n$ above it specifies the range: add from the first order through the $n$th order.

For $n$ orders, the derivative of the loss with respect to the base fee is therefore:

<div class="supervised-math">
$$
\frac{1}{n}\sum_{i=1}^{n}e_i
=\frac{e_1+e_2+\cdots+e_n}{n}
$$
</div>

Even with more data, the calculation is the same: add the errors for all the orders, then divide by the number of orders.

</details>

## 6. Why do we multiply the error by distance when adjusting the per-kilometer fee?

### Unlike a change in the base fee, the effect is larger for more distant orders

Return once more to the original $b=0$, $w=1$. This time, hold the base fee fixed and raise $w$, the fee per additional kilometer, from 1 to 1.1. This means **adding 100 won to the fee for each additional kilometer**.

| Order | Additional distance | Prediction before the adjustment | Prediction after the adjustment | Increase in the prediction |
|---|---:|---:|---:|---:|
| A | 0 km | 0 | 0 | 0 |
| B | 1 km | 1 | 1.1 | 0.1 |
| C | 2 km | 2 | 2.2 | 0.2 |

Order A has no additional distance, so raising the per-kilometer fee adds nothing to its predicted total. Order B's predicted fee increases by 100 won, while order C's increases by 200 won. **The effect of a change in the per-kilometer fee is proportional to the order's additional distance.**

The errors after the adjustment are −1, −1.9, and −2.8. The loss is:

<div class="supervised-math">
$$
J_{\text{new}}
=\frac{(-1)^2+(-1.9)^2+(-2.8)^2}{2\times3}
=\frac{12.45}{6}
=2.075
$$
</div>

The loss decreased by approximately 0.258333 from its original value of $7/3$. We changed the per-kilometer fee by 0.1, so the rate of change is approximately −2.583333. Returning to the original coefficients each time and reducing the adjustment to 0.01 and 0.001 gives rates of approximately −2.658333 and −2.665833, respectively. These values approach $-8/3\approx-2.666667$.

### Putting the change in the per-kilometer fee into the expression

This time, we will use $h$ for **the amount added to the per-kilometer fee**. This is a separate calculation from the previous section's experiment with the base fee. If the current per-kilometer fee is $w$, the adjusted value is $w+h$. We hold the base fee $b$, distance $x$, and actual fee $y$ fixed.

Write the original error as $e=b+wx-y$, and replace $w$ in that expression with $w+h$.

<div class="supervised-math">
$$
\begin{aligned}
\text{Error after the change}
&=b+(w+h)x-y\\
&=b+wx+hx-y\\
&=(b+wx-y)+hx\\
&=e+hx
\end{aligned}
$$
</div>

When we changed the base fee, we added $h$ to the error. When we change the per-kilometer fee, we add **$h$ multiplied by the order's distance $x$**. For order C, $h=0.1$ and $x=2$, so the error increases by 0.2.

Let us calculate the change in the loss for one order.

<div class="supervised-math">
$$
\begin{aligned}
\text{Change in one order's loss}
&=\frac{(e+hx)^2-e^2}{2}\\
&=\frac{e^2+2ehx+h^2x^2-e^2}{2}\\
&=ehx+\frac{h^2x^2}{2}
\end{aligned}
$$
</div>

We expand $(e+hx)^2$ in the same way as in the previous section. The two cross terms are $ehx$ and $hxe$, which add up to $2ehx$. The last term is $(hx)(hx)=h^2x^2$.

### Substitute the numbers for the same three orders

We will now put the errors and distances for A, B, and C into the change in one order's loss, $ehx+h^2x^2/2$.

| Order | Original error $e$ | Additional distance $x$ | Change in that order's loss |
|---|---:|---:|---|
| A | −1 | 0 | $0$ |
| B | −2 | 1 | $-2h+h^2/2$ |
| C | −3 | 2 | $-6h+2h^2$ |

For example, the calculation for order C is:

<div class="supervised-math">
$$
(-3)\times h\times2+\frac{h^2\times2^2}{2}
=-6h+2h^2
$$
</div>

Order A has an additional distance of 0, so changing the per-kilometer fee does not change its loss. Adding the changes in the losses for B and C, then dividing by the total number of orders, 3, gives the change in the overall loss.

<div class="supervised-math">
$$
\begin{aligned}
J_{\text{new}}-J
&=\frac{0+(-2h+h^2/2)+(-6h+2h^2)}{3}\\
&=\frac{-8h+(1/2+2)h^2}{3}\\
&=-\frac{8h}{3}+\frac{5h^2}{6}
\end{aligned}
$$
</div>

The terms containing $h$ combine as $-2h-6h=-8h$. The terms containing $h^2$ give $(1/2+2)h^2=(5/2)h^2$. Dividing that result by 3 gives $5h^2/6$.

To find the rate of change in the loss, we divide by $h$, the change in the per-kilometer fee. When $h$ is not 0, the expression simplifies to:

<div class="supervised-math">
$$
\frac{J_{\text{new}}-J}{h}
=-\frac83+\frac{5h}{6}
$$
</div>

As $h$ approaches 0, $5h/6$ also approaches 0, so the rate of change approaches $-8/3$. This is the same value we found in our earlier numerical experiment.

### Where did −8 come from?

In the calculation above, −8 came from adding **each order's error multiplied by that order's distance**.

<div class="supervised-math">
$$
(-1)\times0+(-2)\times1+(-3)\times2
=0-2-6=-8
$$
</div>

Dividing this sum by the number of orders, 3, gives the derivative we found. The same expansion works with other error and distance values. Let $x_1$, $x_2$, and $x_3$ be the distances of the first, second, and third orders. Just as in the error symbols, these subscripts are **order numbers**. The result is:

<div class="supervised-math">
$$
\text{Derivative of the loss with respect to the per-kilometer fee}
=\frac{e_1x_1+e_2x_2+e_3x_3}{3}
$$
</div>

In the program, we will store the result of this calculation under the name `dw`. **To calculate `dw`, multiply each order's error by that order's distance, then take the mean of those products.**

To understand why we multiply by distance, return to the delivery example. To change the prediction for order A, we must change the base fee. Its additional distance is 0, so no adjustment to the per-kilometer fee can change that order's prediction. In contrast, an adjustment to the per-kilometer fee changes order C's prediction twice as much as order B's.

When calculating which way to adjust the per-kilometer fee, we must therefore **account for both the size of the error and how strongly that coefficient affects the prediction for the order**. In this model, that effect is represented by the distance $x$.

<details>
<summary>Check that the same result follows when errors and distances are written as symbols</summary>

The change in one order's loss was $ehx+h^2x^2/2$. This expression does not require the current error or distance to be a particular number. We can substitute each order's own error and distance.

| Order | Change in that order's loss |
|---|---|
| First | $h e_1x_1+(h^2/2)x_1^2$ |
| Second | $h e_2x_2+(h^2/2)x_2^2$ |
| Third | $h e_3x_3+(h^2/2)x_3^2$ |

When adding the three changes, we group the terms multiplied by $h$ and the terms multiplied by $h^2/2$, then divide by the number of orders, 3.

<div class="supervised-math">
$$
\begin{aligned}
J_{\text{new}}-J
&=h\frac{e_1x_1+e_2x_2+e_3x_3}{3}\\
&\quad+h^2\frac{x_1^2+x_2^2+x_3^2}{6}
\end{aligned}
$$
</div>

The denominator in the second term is 6 because we multiply the 2 in each individual loss by the number of orders, 3. Dividing the change in the overall loss by $h$ gives:

<div class="supervised-math">
$$
\begin{aligned}
\frac{J_{\text{new}}-J}{h}
&=\frac{e_1x_1+e_2x_2+e_3x_3}{3}\\
&\quad+h\frac{x_1^2+x_2^2+x_3^2}{6}
\end{aligned}
$$
</div>

The distances are fixed values in the data, while $h$ approaches 0. The second term therefore approaches 0, and the overall rate of change approaches the first term: the mean of each error multiplied by its corresponding distance.

</details>

## 7. Using the rates of change of the loss to adjust both fees

### Naming the two values we have calculated

We performed one calculation in which only the base fee $b$ changed, and a separate calculation in which only the per-kilometer fee $w$ changed. Differentiating with respect to one coefficient while holding the other coefficients fixed is called **partial differentiation**.[^gradient]

The symbol $\partial$ is used to write a partial derivative. We can now attach mathematical notation to the values we have calculated.

| What we calculated | Mathematical notation | Name used to store it in the code | Current value |
|---|---|---|---:|
| Rate of change of the overall loss when only $b$ changes | $\partial J/\partial b$ | `db` | −2 |
| Rate of change of the overall loss when only $w$ changes | $\partial J/\partial w$ | `dw` | $-8/3$ |

We read $\partial J/\partial b$ as “the partial derivative of J with respect to b.” It is not an instruction to divide the symbols above and below the line as though they were numbers. The notation tells us whose change we calculated and which coefficient we changed to calculate it.

Collecting these partial derivative values in the same order as the coefficients gives the **gradient**. If we arrange the coefficients in the order $b$, $w$, the current gradient is $[-2,-8/3]$.

If you encounter the word “slope” here, keep track of the quantities involved. The coefficient $w$ describes **how much the predicted delivery fee increases as distance increases**. In contrast, `dw` describes **how much the loss changes when we change the per-kilometer fee $w$**. These are rates of change for different relationships.

### Choosing the direction and size of an adjustment

The current `db` and `dw` are both negative. Near the current coefficients, increasing either the base fee or the per-kilometer fee slightly therefore decreases the loss.

To turn this direction into a calculation, subtract **the derivative value multiplied by a fixed positive number** from the existing coefficient. Subtracting a negative derivative increases the coefficient, while subtracting a positive derivative decreases it.

We call the positive multiplier the **learning rate**. In code, we will write it as `lr`; in formulas, we will use the Greek letter $\alpha$, pronounced “alpha.” The learning rate controls how strongly we use the rate of change of the loss to adjust the coefficients. It serves a different purpose from $h$, the adjustment we used while finding the derivative.

We calculate the new coefficients as follows:

<div class="supervised-math">
$$
\begin{aligned}
b_{\text{new}}&=b-\alpha\,\texttt{db}\\
w_{\text{new}}&=w-\alpha\,\texttt{dw}
\end{aligned}
$$
</div>

Set the learning rate to 0.1 and substitute the current values.

<div class="supervised-math">
$$
\begin{aligned}
b_{\text{new}}
&=0-0.1\times(-2)=0.2\\
w_{\text{new}}
&=1-0.1\times(-8/3)\\
&=1+4/15\approx1.266667
\end{aligned}
$$
</div>

The base fee has increased by 200 won, and the fee per additional kilometer has increased by approximately 266.67 won. A learning rate of 0.1 does not mean that we changed each coefficient by 0.1. **The change added to each coefficient is “−learning rate × derivative of the loss with respect to that coefficient.”** For the base fee, this gives $-0.1\times(-2)=+0.2$.

Calculate both `db` and `dw` using **the same model before the adjustment**. Changing one coefficient before calculating the derivative for the other would produce a different calculation from the simultaneous adjustment defined above.

### Checking whether the adjusted rule actually improved

Use the new rule to predict the fees for the three orders again.

| Order | Distance | New predicted fee | New error |
|---|---:|---:|---:|
| A | 0 | 0.2 | −0.8 |
| B | 1 | About 1.466667 | About −1.533333 |
| C | 2 | About 2.733333 | About −2.266667 |

The new loss is shown below. The displayed decimals are rounded; the actual calculation uses the values before rounding.

<div class="supervised-math">
$$
J_{\text{new}}
\approx\frac{(-0.8)^2+(-1.533333)^2+(-2.266667)^2}{6}
\approx1.354815
$$
</div>

This is smaller than the original loss of 2.333333. The predictions still differ from the targets, so we can calculate the derivative values again using the new errors and adjust the coefficients again. Repeating this process is the method called **gradient descent**.[^gradient-descent]

We do not reuse the initial values of −2 and $-8/3$ for the next adjustment. Substituting the new errors gives `db` of approximately −1.533333 and `dw` of approximately −2.022222. **Changing the model also changes its predictions and errors. At each step, we therefore calculate the rates of change of the loss again using the new errors, then use those results to determine the next adjustments.**

We used small adjustments $h$ to explain derivatives, but the actual training code does not need to approximate derivatives by repeatedly shrinking $h$. It calculates the current derivative values directly from the expressions we derived: the mean of the errors, and the mean of each error multiplied by its order's distance.

<details>
<summary>What connection does the chain rule describe in this calculation?</summary>

Changing the base fee or the per-kilometer fee changes the prediction. Since the actual fee is fixed, that changes the error, which in turn changes the loss. The **chain rule** is a rule of differentiation that calculates rates of change along this sequence.[^chain-rule]

We can check it using the result we have already derived for one order. When the error is $e$ and we add a small amount to it, the change in loss consists of “original error × amount added” plus “half the square of the amount added.” Dividing by the amount added and then letting that amount approach 0 leaves $e$. In other words, **the derivative of one order's loss with respect to its error is $e$**.

Raising the base fee by $h$ increases the error by $h$. The rate of change of the error with respect to the base fee is therefore 1. Multiplying the rates for the two stages gives $e\times1=e$, the derivative of one order's loss with respect to the base fee.

Raising the per-kilometer fee by $h$ increases the error by $hx$. The rate of change of the error with respect to the per-kilometer fee is therefore $x$. Multiplying the rates for the two stages gives $e\times x=ex$.

The chain rule is not a trick that produces a different answer from the earlier expansion. It lets us calculate more briefly the same connection that we checked by directly expanding the changes in the error and the loss.

</details>



## 8. Calculate all three orders at once with NumPy

We will now translate the calculations we performed one order at a time into Python. Python is a programming language in which we can write and run a sequence of calculations. **NumPy** is a tool that lets us group numbers together and calculate with them.

The complete runnable code appears later in the article. First, we will look at how each line corresponds to the delivery-fee calculations we have already worked through. We use short names such as `x` and `y` in this explanation. The complete code adds `_one`, as in `x_one` and `y_one`, to distinguish these records from the four-order dataset introduced later.

The first line, `import numpy as np`, loads NumPy and lets us refer to it by the name `np`. The name `np.array` refers to a function that creates an **array**. A function is a piece of code that takes inputs and performs a defined task. We put the values it should work with inside the parentheses. Writing `np.array([0, 1, 2])` creates an array that stores those three numbers in that order.

| Code | What it stores or calculates |
|---|---|
| `x = np.array([0, 1, 2], dtype=float)` | Stores the additional distances for orders A, B, and C, in that order. |
| `y = np.array([1, 3, 5], dtype=float)` | Stores the actual delivery fees in the same order. |
| `b, w = 0.0, 1.0` | Sets the current base fee and per-kilometer fee. |
| `pred = b + w * x` | Calculates the predicted delivery fees `[0, 1, 2]` for the three orders. |
| `e = pred - y` | Calculates each order's error: `[-1, -2, -3]`. |
| `loss = np.mean(e ** 2) / 2` | Squares the errors, takes their mean, and divides by 2 to calculate $J$. |
| `db = np.mean(e)` | Calculates the mean of the errors, which is −2. |
| `dw = np.mean(e * x)` | Calculates the mean of `[0, -2, -6]`, which is $-8/3$. |

Here, `=` is an assignment sign: it stores the result on the right under the name on the left. A name under which we keep a value for use in calculations is called a **variable**. The option `dtype=float` creates an array using a numerical type that can represent decimal values. The operator `*` means multiplication, and `** 2` squares a value. The function `np.mean(...)` calculates the mean of the values supplied inside the parentheses.

When we multiply or subtract arrays of the same shape, NumPy calculates with **the values in matching positions**. The expression `e * x` does not mix all the errors with all the distances. It multiplies A's error by A's distance, B's error by B's distance, and C's error by C's distance. These are the same calculations we performed row by row in the earlier table.

There is no need to memorize the loss calculation as a single unexplained line. The array `e` contains `[-1, -2, -3]`, so `e ** 2` contains `[1, 4, 9]`. Their mean is $14/3$. Dividing that result by 2 gives $J=7/3$, exactly as in our calculation by hand.

### Arrays need matching shapes so that we compare the same orders

A NumPy array's `shape` tells us how its numbers are arranged. Our current `pred` and `y` are both **one-dimensional arrays**, each containing a sequence of three numbers. They are not stored as tables that use both row and column indices. Both have the shape `(3,)`. The comma in this notation indicates that the array has just one dimension, with a size of 3.

If we instead store three numbers as a table with 3 rows and 1 column, we have a **two-dimensional array** with the shape `(3, 1)`. It still contains three values, but its shape is different.

That difference can cause a real calculation error. If the prediction array has shape `(3,)` and the target array has shape `(3, 1)`, subtraction produces the following 3-by-3 result, rather than comparing just three pairs.

| Target being subtracted | Prediction A: 0 | Prediction B: 1 | Prediction C: 2 |
|---|---:|---:|---:|
| Target A: 1 | **−1** | 0 | 1 |
| Target B: 3 | −3 | **−2** | −1 |
| Target C: 5 | −5 | −4 | **−3** |

The three bold values compare predictions and targets for the same order. Those are the values we want. Every other value compares a prediction for one order with a target for a different order. Even if NumPy reports no error, calculating the loss from this entire array would give us the wrong evaluation.

NumPy's ability to combine arrays of different shapes according to a set of rules is called **broadcasting**. It is useful when, for example, we add one number to every value in an array. However, it also permits unintended calculations like this one.[^broadcasting]

Before calculating errors in this example, we therefore check `pred.shape == y.shape`. The operator `==` asks whether two values are equal. **We need to check both that the shapes match and that matching positions refer to the same order.** If we shuffle the orders differently in the two arrays, their shapes can still match while the comparisons are wrong.

<details>
<summary>Why does subtracting arrays of shape (3,) and (3, 1) produce (3, 3)?</summary>

Broadcasting compares shapes starting from the right. Along each dimension, the two sizes must either be equal or one of them must be 1. Values along a dimension of size 1 can be reused to match the other array's size.

When comparing `(3,)` and `(3, 1)`, NumPy treats the missing leftmost dimension of the first array as size 1. It therefore compares `(1, 3)` with `(3, 1)`. On the right, the sizes 3 and 1 can match at 3. On the left, the sizes 1 and 3 can also match at 3. That is why the result has shape `(3, 3)`.

The runnable code uses `y[:, None]` to reproduce this situation by turning the target array into 3 rows and 1 column. Inside the brackets, `:` selects all the existing values, while `None` adds a new dimension of size 1. The three values stay the same, but the shape changes from `(3,)` to `(3, 1)`.[^indexing]

In this case, `y.reshape(3, 1)` can produce the same shape. The `reshape` method changes an array's shape while keeping the same number of values.

</details>

## 9. Matrices organize the same calculations

As we add more inputs, we can organize the numbers into a table instead of writing an ever longer prediction expression in our code. A rectangular table of numbers like this is called a **matrix**. Rows run horizontally, and columns run vertically.

Before introducing any new data, we will calculate with the same three orders using a matrix.

### Include the base fee in the multiplication

Our prediction for one order is $b+wx$. Multiplying $b$ by 1 leaves its value unchanged, so we can also write the prediction as $1\times b+x\times w$. Each order then supplies the numbers `[1, additional distance]`, while the coefficients shared by all orders are `[base fee, per-kilometer fee]`.

We will use the name $X_b$ for the input matrix that includes the extra 1s for the base fee. We will use $\theta$, pronounced “theta,” for the array that collects the two coefficients in a fixed order. A collection of numbers arranged along one direction is also called a **vector**. For our three orders, these arrays are:

<div class="supervised-math">
$$
X_b=
\begin{bmatrix}
1&0\\
1&1\\
1&2
\end{bmatrix},
\qquad
\theta=
\begin{bmatrix}
b\\
w
\end{bmatrix}
$$
</div>

In the formula, we wrote the two coefficients vertically. In code, we store them in a one-dimensional array, as in `theta = np.array([b, w])`, with shape `(2,)`. We need to distinguish the vertical mathematical notation from the shape of the array actually stored in NumPy.

We multiply the first column of $X_b$ by $b$ and the second column by $w$. Adding those two products within one row gives the prediction for that order.

<div class="supervised-math">
$$
X_b\theta=
\begin{bmatrix}
1b+0w\\
1b+1w\\
1b+2w
\end{bmatrix}
$$
</div>

With $b=0$ and $w=1$, the result is again $[0,1,2]$. This calculation is a **matrix–vector product**, which we write as `Xb @ theta` in NumPy. The `@` operator performs matrix multiplication.[^matmul]

The matrix needs two columns and the coefficient array needs two values so that we can pair each column with its coefficient. The shapes in this calculation are therefore `(3, 2) @ (2,) → (3,)`. Each of the three input rows produces one prediction.

### Organize the same sums to calculate the gradient

The derivative of the loss with respect to the base fee was the mean of the three errors. If we write out the multiplications as well, this is $(1e_1+1e_2+1e_3)/3$. The derivative with respect to the per-kilometer fee was $(x_1e_1+x_2e_2+x_3e_3)/3$.

We can calculate both sums together by putting the 1s in one row and the distances in another row. Moving the columns of $X_b$ into rows gives us exactly that arrangement. Swapping rows and columns is called **transposing** a matrix, and we write it as `Xb.T` in NumPy.

<div class="supervised-math">
$$
X_b^{\mathsf T}=
\begin{bmatrix}
1&1&1\\
0&1&2
\end{bmatrix}
$$
</div>

Here, $e$ will represent the vector containing the three orders' errors, $[-1,-2,-3]$. In the earlier calculations for one order, $e$ was a single number. Now that we are calculating for several orders together, we collect their errors in an array. Multiplying by this error vector and dividing by the number of orders gives:

<div class="supervised-math">
$$
\begin{aligned}
\frac{X_b^{\mathsf T}e}{3}
&=\frac{1}{3}
\begin{bmatrix}
1(-1)+1(-2)+1(-3)\\
0(-1)+1(-2)+2(-3)
\end{bmatrix}\\
&=\frac{1}{3}
\begin{bmatrix}
-6\\
-8
\end{bmatrix}
=\begin{bmatrix}
-2\\
-8/3
\end{bmatrix}
\end{aligned}
$$
</div>

The first value in the result is `db`, and the second is `dw`. In the code, we also store the three orders' error array under the name `e`.

If we store the number of orders as `n`, the code is `gradient = Xb.T @ e / n`. The name `gradient` stores the gradient. Transposing and multiplying matrices have not replaced the work of differentiation. **We have arranged the arrays so that they calculate the two sums we already derived, together and in the same order.**

| Value | Shape in this example | Meaning of its positions |
|---|---|---|
| `Xb` | `(3, 2)` | Three orders, each with a 1 for the base fee and a distance |
| `theta` | `(2,)` | Base fee, then per-kilometer fee |
| `pred`, `e` | `(3,)` | One prediction and one error per order |
| `Xb.T` | `(2, 3)` | The three orders' inputs to multiply for each coefficient |
| `gradient` | `(2,)` | Rates of change in loss with respect to the base fee and the per-kilometer fee |

## 10. Add special packaging as an input

Suppose delivery fees depend on special packaging as well as distance. We will use **four new fictional records with two inputs**.

In the table, $u$ is the same additional distance we used earlier, and $v$ indicates whether an order needs special packaging. We record $v=0$ when there is no special packaging and $v=1$ when there is. The numbers 0 and 1 distinguish the two possibilities.

| Order | Additional distance $u$ (km) | Special packaging $v$ | Actual delivery fee $y$ (thousands of won) |
|---|---:|---:|---:|
| D | 0 | 0 | 1 |
| E | 1 | 0 | 3 |
| F | 0 | 1 | 4 |
| G | 1 | 1 | 6 |

The base fee is still $b$. We will call the fee per additional kilometer $w_u$ and the special-packaging fee $w_v$. Our prediction rule is now:

<div class="supervised-math">
$$
\hat y=b+w_u u+w_v v
$$
</div>

We multiply $w_u$ by the distance $u$ and $w_v$ by the packaging indicator $v$. When an order does not need special packaging, $v=0$, so no packaging fee is added. When it does, $v=1$, so we add the packaging fee $w_v$ once.

The coefficients that match these fictional records exactly are $b=1$, $w_u=2$, and $w_v=3$. For example, order G has a fee of $1+2\times1+3\times1=6$ thousand won.

As in the previous section, we add a column of 1s for the base fee and collect the coefficients in the same order as the columns:

<div class="supervised-math">
$$
X_b=
\begin{bmatrix}
1&0&0\\
1&1&0\\
1&0&1\\
1&1&1
\end{bmatrix},
\qquad
\theta=
\begin{bmatrix}
b\\
w_u\\
w_v
\end{bmatrix}
$$
</div>

In the code, `X` stores only the two columns for distance and packaging. This time we have four orders, so `n` is 4 and `np.ones(n)` creates `[1, 1, 1, 1]`. The expression `np.column_stack([np.ones(n), X])` attaches these values as a new column to the left of `X`, creating `Xb`. The shape of `X` is therefore `(4, 2)`, while the shape of `Xb` is `(4, 3)`.

If we start all three coefficients at 0, the predictions for all four orders are also 0. The errors are $[-1,-3,-4,-6]$, giving the following initial loss:

<div class="supervised-math">
$$
J=\frac{1+9+16+36}{2\times4}=7.75
$$
</div>

The same reasoning applies whichever coefficient we adjust. The base fee changes every order's prediction by the same amount. The effect of the per-kilometer fee depends on the additional distance. The packaging fee affects only orders that need special packaging. We therefore calculate the derivatives as follows.

| Coefficient to adjust | Value to multiply by each order's error | Sum across the four orders | Derivative after dividing by 4 |
|---|---|---|---:|
| Base fee $b$ | 1 | $-1-3-4-6=-14$ | −3.5 |
| Per-kilometer fee $w_u$ | Additional distance $u$ | $0-3+0-6=-9$ | −2.25 |
| Packaging fee $w_v$ | Packaging indicator $v$ | $0+0-4-6=-10$ | −2.5 |

The matrix calculation `Xb.T @ e / n` also gives $[-3.5,-2.25,-2.5]$. Adding another input has not required a new calculation principle. **We still multiply each error by the corresponding input value because that input determines how strongly a change in the coefficient affects the order's prediction.**

Even with two inputs, this is still a regression problem. Regression and classification are distinguished by **what we predict**, not by whether the inputs contain 0s and 1s. We are still predicting a numerical delivery fee.


## 11. Repeat the calculations and examine the learning rate

### Recalculate predictions and errors each time

For the four-order example, we will use a learning rate of 0.2. All coefficients start at 0, and the gradient we just calculated is $[-3.5,-2.25,-2.5]$.

In the first update, the base fee becomes $0-0.2(-3.5)=0.7$, the per-kilometer fee becomes $0-0.2(-2.25)=0.45$, and the packaging fee becomes $0-0.2(-2.5)=0.5$. The new coefficients are $[0.7,0.45,0.5]$.

We now use those new coefficients to predict the four fees again, then calculate a new gradient from the new errors. To repeat this process in code, we follow these steps:

1. Calculate predictions using the current coefficients: `pred = Xb @ theta`.
2. Compare predictions and targets for the same orders: `e = pred - y`.
3. Calculate the current loss: `loss = np.mean(e ** 2) / 2`.
4. Calculate the rates of change in loss at the current coefficients: `gradient = Xb.T @ e / n`.
5. Update all coefficients together: `theta = theta - lr * gradient`.

Within one iteration, the first four calculations all use the same current coefficients. We use their results to update the coefficients, then start the calculations again in the next iteration.

The complete code performs 2,000 updates. We chose that number to give this small example enough iterations. It is not a fixed number of updates that every training problem requires.

| Stage | Coefficients `[base fee, per-kilometer fee, packaging fee]` | Loss $J$ |
|---|---|---:|
| Before any updates | `[0, 0, 0]` | 7.75 |
| After the first update | `[0.7, 0.45, 0.5]` | 3.784375 |
| After 2,000 updates | Approximately `[1, 2, 3]` | Numerically very close to 0 |

![Loss decreases as the model learns the delivery fees for the four orders](/supervised-learning-basics/loss-en.svg)

The code stores **the loss before each update** in `history`. With 2,000 updates, this array contains 2,000 losses, from the initial state through the state after 1,999 updates. After the loop ends, we calculate the loss at the final coefficients as `final_j`, then append it to create the complete record, `history_complete`. This complete record contains 2,001 values, including the initial state. On the graph, horizontal position 0 means before any updates, and 1 means after one update.

The figure shows 81 loss values, from the initial state through the 80th update. The loss falls quickly at first, then the curve appears to lie close to 0. However, the loss after 80 updates is about 0.002055, which is still positive. Appearing to be 0 on the graph is different from being exactly 0 in the calculation.

The code uses `theta.copy()` to create and save a separate array with the same coefficient values at that point. We use these saved final coefficients to make predictions and compare the results with other solution methods.

### Why can a larger learning rate make the result worse?

When we say that increasing a fee will help, we mean **near its current value**. A small increase in the base fee can bring predictions that are too low closer to the actual fees. A very large increase, however, can make the predictions much too high.

We will examine this using the original three orders. In this experiment, we **hold the per-kilometer fee fixed at $w=1$** and adjust only the base fee. This is separate from the earlier experiment in which we trained $b$ and $w$ together.

Order A has an error of $b+1\times0-1=b-1$, order B has an error of $b+1\times1-3=b-2$, and order C has an error of $b+1\times2-5=b-3$. The distances and actual fees still come from the original table; only the base fee $b$ remains unspecified. The derivative of the loss with respect to the base fee is therefore:

<div class="supervised-math">
$$
\texttt{db}
=\frac{(b-1)+(b-2)+(b-3)}{3}
=\frac{3b-6}{3}
=b-2
$$
</div>

When $b$ is less than 2, this value is negative, so increasing the base fee helps. When $b$ is greater than 2, it is positive, so decreasing the base fee helps. **In this experiment, where the per-kilometer fee is fixed at 1**, a base fee of 2 therefore gives the smallest loss. The predictions at that point are $[2,3,4]$, and the errors are $[1,0,-1]$, giving a loss of $(1+0+1)/6=1/3$. They do not match all the targets because we have not also fitted the correct per-kilometer fee. A rate of change of 0 in the loss is different from a loss of 0.

Starting each experiment at $b=0$ gives `db = -2`. We will change only the learning rate and perform one update.

| Learning rate | Base fee after one update | Loss after the update | What happened |
|---:|---:|---:|---|
| 0.2 | 0.4 | Approximately 1.613333 | We move a little closer to the better base fee of 2. |
| 0.8 | 1.6 | Approximately 0.413333 | We move much closer to 2. |
| 1.5 | 3.0 | Approximately 0.833333 | We pass 2, but end up closer to it than where we started. |
| 2.2 | 4.4 | Approximately 3.213333 | We go too far past 2, and the loss exceeds the initial 2.333333. |

Even if we pass the best base fee of 2, the loss decreases if we end up closer to 2 than before. With a learning rate of 1.5, the base fee moves from 0 to 3, so its distance from 2 shrinks from 2 to 1. What matters is **how far we overshoot and how the loss changes in subsequent iterations**.

A learning rate that works well here may not work on different data. For example, writing the same distances in meters instead of kilometers changes the input values from 1 and 2 to 1,000 and 2,000. Because the calculation of `dw` multiplies by the input, its magnitude also changes. Using the same learning rate can then change the coefficient by far too much. A suitable learning rate depends on the units and scale of the inputs, the loss we use, and the structure of the model.

<details>
<summary>When do repeated updates settle down in the base-fee-only experiment?</summary>

Substituting `db = b - 2` into the update rule gives $b_{\text{new}}=b-\alpha(b-2)$. We will examine how far the new base fee is from 2, the value that minimizes the loss.

<div class="supervised-math">
$$
\begin{aligned}
b_{\text{new}}-2
&=b-\alpha(b-2)-2\\
&=(b-2)-\alpha(b-2)\\
&=(1-\alpha)(b-2)
\end{aligned}
$$
</div>

Every update multiplies the previous difference $b-2$ by $1-\alpha$. If the absolute value of this multiplier is less than 1, the magnitude of the difference shrinks. We therefore need $-1<1-\alpha<1$.

Rearranging the left-hand condition, $-1<1-\alpha$, gives $\alpha<2$. Rearranging the right-hand condition, $1-\alpha<1$, gives $\alpha>0$. In this experiment, the base fee therefore approaches 2 as we keep updating it when $0<\alpha<2$.

We obtained this range **for our current data and loss, while holding the per-kilometer fee at 1 and learning only the base fee**. It is not a universal range of learning rates for other models.

</details>

## 12. Check the result with a least-squares solver and scikit-learn

### Solve the same problem using a different calculation method

Our loop reduced squared errors by adjusting the coefficients a little at a time. Finding coefficients that minimize the sum of squared errors is called **solving a least-squares problem**. For the same data, SSE, MSE, and half the MSE differ only by fixed positive multipliers, so the same coefficients minimize all three scores.

There are other ways to solve the least-squares problem for linear regression besides the gradient descent loop we wrote. NumPy provides one through `np.linalg.lstsq`.[^lstsq]

Passing the four orders' input matrix and actual fees to `result = np.linalg.lstsq(Xb, y)` returns a collection of results. The first item is the coefficient array, which we retrieve with `theta_ls = result[0]`. Python starts numbering positions at 0, so `[0]` selects the first item.

In this example, the calculated coefficients are approximately $[1,2,3]$. We can use them to make predictions and check whether those predictions are close to the ones from our own gradient descent code.

### `fit` finds and stores coefficients; `predict` uses them to make predictions

**scikit-learn** is a Python tool that provides many machine-learning methods. Its `LinearRegression` model is a linear regression model, like our “base fee plus a charge for each input” rule. With the default settings we use here, it finds coefficients that minimize squared errors.[^linear-regression]

The expression `model_small = LinearRegression()` creates a **model object** to store the training results and make predictions. Here, you can think of an object as a part of the program that keeps the training functionality, prediction functionality, and learned coefficients together.

| Code | What it does in the delivery-fee problem |
|---|---|
| `model_small.fit(X, y)` | Finds coefficients from the four orders' distances, packaging indicators, and actual fees, then stores them in the model. |
| `model_small.intercept_` | Retrieves the learned base fee, approximately 1. |
| `model_small.coef_` | Retrieves the learned per-kilometer and packaging fees, approximately `[2, 3]`. |
| `model_small.predict(X)` | Uses the stored coefficients to predict fees for the four orders. |

The trailing underscore is scikit-learn's naming convention for values determined during training. The return value of `fit` is the fitted model itself, not an array of predictions. To obtain predictions, we call `predict`.

By default, `LinearRegression` handles the intercept separately. We therefore pass **the two-column `X` containing only distance and packaging** to `fit`. We do not pass `Xb`, to which we added a column of 1s for our own calculations. Also, the name `fit` does not mean “run the gradient descent loop from earlier.” It is a common name for training; the calculation method depends on the model.

### Use the same units and column order for new orders

Let us predict the fee for an order with an additional distance of 3 km and special packaging. For our own matrix calculation, the input is `[1, 3, 1]`: a 1 for the base fee, a distance of 3, and a packaging indicator of 1.

With learned coefficients of $[1,2,3]$, the prediction is:

<div class="supervised-math">
$$
1\times1+3\times2+1\times3=10
$$
</div>

The predicted delivery fee is 10 thousand won. For scikit-learn, we remove the 1 for the base fee and provide `[[3, 1]]`. The two layers of brackets keep the input in a 1-row, 2-column shape: one order with two inputs.

If we swap the distance and packaging columns, each coefficient will multiply the wrong input. Using meters for prediction when we used kilometers for training also produces an incorrect result. **The input columns must have the same order and units during training and prediction.**

This 3 km order lies outside the 0–1 km range in our fictional training records. Here, we are checking the calculation for a rule we deliberately built into the data. With real data, we need to check separately whether the same relationship continues beyond the range we observed.

### When calculated decimals differ slightly

Even when two methods give the same mathematical answer, a computer may produce slightly different final digits because it stores decimals with finite precision and performs operations in different orders. When comparing prediction arrays, we can therefore allow a specified difference instead of requiring exact equality.

For example, if the expected value is 3 and the calculated value is 3.000000002, the difference is 0.000000002. That result is within an allowed difference of 0.00000001.

The code `np.allclose(actual, expected, atol=1e-8, rtol=0)` checks whether corresponding values in the arrays agree within this tolerance. Here, `actual` is the array we calculated, and `expected` is the array we compare it with. The notation `1e-8` is how we write $10^{-8}=0.00000001$ in code. The argument `atol` sets the allowed absolute difference. The argument `rtol` sets an additional allowance proportional to the magnitude of the comparison value. We set `rtol=0` here to use only the absolute difference.[^allclose]

An `assert` statement checks whether the condition that follows it is true and stops execution if it is false. Our code uses these statements to check agreement with calculations by hand or with another solution method. They help us find calculation errors; they do not train a more accurate model.[^assert]

We still need to compare arrays with matching shapes. Since `allclose` can also use broadcasting, we check `shape` first.


## 13. Check predictions on data that was not used for training

We created delivery records that followed an exact rule so that we could check our calculations. In real work, fitting past records well is not enough. The program's purpose is **to predict the fees for orders that will arrive in the future**.

A program that memorizes every past order number and its fee may still have no answer for an order it has never seen. The ability to predict well for examples that were not used during training is called **generalization**.

To examine this ability, we will set aside some of the data before training. We use the **training data** to find the coefficients and the **test data** to evaluate predictions on examples that were not used for training.

### Move to a real dataset that anyone can run

We will now move from our handmade delivery records to the `diabetes` dataset included with scikit-learn. This lets anyone run the same exercise without obtaining records from a delivery company.

The dataset contains records for 442 people and 10 input features. The target is a numerical measure of **disease progression one year after the initial measurements**. We are predicting a different kind of number from a delivery fee, but the process is the same: train using paired inputs and targets, then evaluate predictions on new examples.[^diabetes]

| Column index | Name | What the input records |
|---|---|---|
| 0 | `age` | Age |
| 1 | `sex` | Sex recorded as a numerical category |
| 2 | `bmi` | Body mass index: a measure calculated from height and weight |
| 3 | `bp` | Average blood pressure |
| 4–9 | `s1`–`s6` | Six measures related to blood tests |

This exercise is a regression problem that predicts the recorded target number, not a classification problem that predicts whether someone has diabetes. We are not assigning our own medical interpretations to the measurements or using the model to make clinical decisions. The official documentation also notes that the precise meanings of some original features are unclear.[^diabetes]

The call `load_diabetes(return_X_y=True, scaled=False)` returns the input array followed by the target array. The option `return_X_y=True` requests these two arrays directly. The option `scaled=False` returns the stored feature values without the additional mean centering and scaling offered by the loading function. It does not mean that no column has ever been processed before.

Mean centering subtracts a column's mean from each of its values. Scaling changes their scale, for example by dividing them by a chosen reference quantity. Here, we will use the least-squares solver in `LinearRegression`. We are not applying our earlier delivery-example gradient descent loop and learning rate directly to this real dataset.

### Split row indices first to keep inputs paired with their targets

We will store the 442-by-10 input array as `X_real` and the 442 targets in the same order as `y_real`. Shuffling the inputs and targets independently could pair values from different people. We therefore **split the row indices first, then apply the same indices to both arrays**.

The expression `np.arange(len(y_real))` creates indices from 0 to 441. The value of `len(y_real)` is the number of targets, 442. The call `np.arange(442)` creates integers starting at 0 and stopping just before 442.

We pass these indices to `train_test_split`. The setting `test_size=0.2` leaves roughly 20% of them for testing. The option `random_state=42` fixes the starting state of the random selection process so that we can reproduce the split with the same data and settings. The number 42 does not guarantee a particularly good split.[^split]

We store the training indices returned by the splitting function as `train_ids` and the test indices as `test_ids`. The expressions `X_real[train_ids]` and `y_real[train_ids]` select rows with the same indices in the same order. We apply the same procedure to the test indices.

| Purpose | Input array shape | Target array shape |
|---|---|---|
| Training | `X_train: (353, 10)` | `y_train: (353,)` |
| Testing | `X_test: (89, 10)` | `y_test: (89,)` |

In the execution results, the first record in the test array has **index 287** in the original dataset. Since Python starts its indices at 0, that was the 288th record in the original order.

| Value to check | Position in the code | Value |
|---|---|---:|
| Original index of the first test record | `test_ids[0]` | 287 |
| BMI for that record | `X_test[0, 2]` | 25.8 |
| Target for that record | `y_test[0]` | 219 |

The indices `[0, 2]` select the first row and third column of the current array. Distinguishing the original row index, 287, from its position in the test array, 0, lets us keep track of which record's target each prediction is compared with.

## 14. Compare the model with predicting the mean every time

### We need a point of comparison to interpret MSE

If a model has a test MSE of 2,900, is that a good result? The number alone does not tell us. We need to know the units and scale of the targets, as well as how well a simple method can predict them.

A simple prediction method chosen as a point of comparison for more complex models is called a **baseline**. Our baseline will ignore differences between inputs and **predict the mean of the training targets for every person**.

In the delivery-fee example, this would mean quoting the same fee for every order, regardless of distance or packaging. If a new model performs worse than this baseline on the same test data, the more complex model has not produced better predictions in this evaluation.

scikit-learn's `DummyRegressor(strategy="mean")` calculates and stores the mean of the training targets, then uses that value for every prediction.[^baseline] The mean in our training data is approximately 153.736544. The baseline therefore predicts that same value for the first test record, the second test record, and all the others.

We use `fit(X_train, y_train)` and `predict(X_test)` just as with the other models. However, this baseline does not change its predictions according to age, BMI, or any other input value. It returns the same prediction once for each row in the test input.

<details>
<summary>Why use the mean when predicting the same value for every example?</summary>

Suppose we predict the same number $c$ for every record. The value $c$ is the only coefficient we can adjust in this method, and it is the **shared prediction** for all records. The error for one record is $c-y$. We will again use the loss we defined earlier, $J=\text{MSE}/2$.

We can use the same calculation we used for the derivative of the loss with respect to the base fee. Increasing the shared prediction $c$ raises every prediction by the same amount. The derivative of the loss with respect to $c$ is therefore also the mean of the errors.

Let the training targets be $y_1$, $y_2$, and $y_3$. Using the same prediction $c$ for every record gives three errors: $c-y_1$, $c-y_2$, and $c-y_3$. Adding these errors and dividing by 3 gives:

<div class="supervised-math">
$$
\begin{aligned}
\frac{(c-y_1)+(c-y_2)+(c-y_3)}{3}
&=\frac{3c-(y_1+y_2+y_3)}{3}\\
&=c-\frac{y_1+y_2+y_3}{3}
\end{aligned}
$$
</div>

The derivative is therefore “the current shared prediction minus the mean of the training targets.”

When the shared prediction is below the mean, the derivative is negative, so raising the prediction reduces the loss. When it is above the mean, the derivative is positive, so lowering the prediction reduces the loss. **When we predict the same value for every example and evaluate it using MSE, the mean of the training targets gives the smallest loss.** The same reasoning applies to any number of records.

We calculate this mean using the training targets. If we looked at the test targets first and used them to choose the mean, we would already have used the answers from our future evaluation data to create the prediction rule.

</details>

### A model using only BMI and a model using all ten features

Our first linear regression model uses only BMI as its input. Since the BMI column has index 2, we select it with `X_train[:, [2]]`. Inside the brackets, `:` selects all rows, while `[2]` selects the third column and keeps it in a one-column table. The result has shape `(353, 1)`.

Writing `X_train[:, 2]` instead produces an array of shape `(353,)`, a sequence of 353 numbers. This scikit-learn model expects inputs as a table of “number of samples × number of features,” so we keep a two-dimensional shape even when there is only one feature.[^sklearn-start]

Our second model uses all 10 input columns. Each model learns its own coefficients, and **both are fitted using only the same 353 training rows**. We then predict the same 89 test rows and compare both sets of predictions with the same `y_test`.

| Method | Values learned during training | Test MSE |
|---|---|---:|
| Mean baseline | One mean of the training targets | 5361.533457 |
| Linear regression using only BMI | A weight for BMI and an intercept | 4061.825928 |
| Linear regression using all 10 features | 10 weights and an intercept | 2900.193628 |

These are rounded results from Python 3.12.14, NumPy 2.3.5, and scikit-learn 1.8.0, using `scaled=False`, `test_size=0.2`, and `random_state=42`. The function call `mean_squared_error(y_test, prediction)` calculates the same MSE we learned earlier.

On this split, the BMI model had a lower MSE than the mean baseline, and the model using all features had the lowest MSE. We can say that **the model using all features produced the smallest mean squared error on these 89 held-out records**.

This does not mean that adding features always improves a model. Some inputs may not help, and a model can learn relationships that appeared in the past data only by chance. This comparison tells us how the three methods performed on this particular dataset and split.

## 15. Examine individual mistakes and understand what the evaluation tells us

### MSE alone does not show which way predictions were wrong

The model using all features predicts approximately 139.547558 for the first test record. Its target is 219, so the prediction is about 79.452442 too low.

Now we will examine how far each record's actual value is above its prediction. To represent this difference, we use the **residual**, calculated by subtracting the prediction from the actual value. Plotting residuals for multiple records lets us examine which way the model's predictions were wrong. We calculate them as follows:

<div class="supervised-math">
$$
\text{residual}=\text{actual value}-\text{predicted value}=y-\hat y
$$
</div>

This subtraction runs in the opposite order from the error $e=\hat y-y$ that we used for training. A positive residual therefore means the prediction was too low, and a negative residual means it was too high. Reversing the subtraction does not change the squared value, so it does not change MSE.

The residual for the first test record is:

<div class="supervised-math">
$$
219-139.547558\approx79.452442
$$
</div>

The figure below is a **scatter plot**: it shows one point per test record, with the prediction on the horizontal axis and the residual on the vertical axis. There are 89 points in total. The first record, whose residual we just calculated, is highlighted.

![Predictions and residuals for 89 test records. The highlighted first record has an actual value of 219, a prediction of about 139.55, and a residual of about 79.45.](/supervised-learning-basics/residuals-en.svg)

The horizontal line marks a residual of 0. The farther a point is from this line, the larger the error for that record. The highlighted point is above the line, so its prediction was below the actual value.

In delivery work, we might look for something that orders with underestimated fees have in common. For example, if predictions are consistently too low for orders with special packaging, that gives us a reason to examine the packaging inputs or the prediction rule more closely. Examining errors record by record can reveal problems that an overall MSE hides. The shape of the plot alone, however, does not establish the cause.

### Fitting past records well is different from predicting new examples well

**Overfitting** occurs when a model fits even the chance details of its training records so closely that it predicts new examples poorly.

If a delivery model memorizes a separate fee for every order number, it can have very little loss on orders it has already seen. For a new order number, however, it has no memorized answer. To check whether it has learned useful relationships with distance and packaging, we need to evaluate it on separate examples.

Earlier, we checked that the small delivery table produced coefficients of $[1,2,3]$ to **verify our calculation implementation**. Here, we calculated MSE on 89 records that were not used for training to **evaluate predictions on separate examples**. Passing assertions in the code does not automatically answer the second question.

### Separate the data for model selection from the data for final evaluation

If we want to try different model types or learning rates and choose the settings that perform best, we set aside **validation data** for those comparisons. We use training data to learn coefficients, validation data to choose settings, and the remaining test data for a final evaluation.

Repeatedly studying the same exam questions and answers makes us familiar with that particular exam. Similarly, if we keep checking the same test results and changing our model, that dataset influences our choice of model. If we repeatedly make choices using the 89 records above, those records are effectively serving as validation data. We can no longer interpret the subsequent result as performance on a final exam we have never seen.[^cross-validation]

When data is limited, **cross-validation** divides it into several groups and rotates which group is held out for validation. Here, imagine leaving the final test data aside and splitting the remaining data into five groups. We train on four groups and validate on the remaining group, repeating the process five times. These results help us examine how much performance depends on one particular split.[^cross-validation]

### Use only information that will be available when making a prediction

Suppose we are predicting the delivery fee when an order arrives, but include the final billed amount—which is only known after delivery—as an input. The model might fit past records well, but we cannot provide that input when a new order arrives.

**Data leakage** occurs when **information unavailable at prediction time is used to build the prediction rule, or when test information reserved for final evaluation enters training or model selection**.[^leakage]

The same principle applies when preparing inputs. Filling missing values or adjusting the scale of inputs is called **preprocessing**. Suppose we fill missing distance values with the mean distance. We first split the data into training and test sets, then **calculate the mean using only distances in the training data**. We use that same mean to fill missing values in the test data. Filling values with a mean calculated from both sets before splitting would allow information from the held-out data into training.[^leakage]

The way we split the data should also reflect the prediction task. To predict next month's orders, it makes sense to train on earlier orders and evaluate on later ones. To check performance for customers we have never served before, we can keep each customer's records entirely on one side of the training–evaluation split. Randomly splitting rows is not suitable for every problem.[^cross-validation]

Our original goal was to predict the fee for a new order. To serve that goal, we need to **prepare inputs, compare predictions with targets for the same orders, learn coefficients, and check results in situations that were not used for training**. Calculating a loss and running gradient descent each contribute to part of that process.


## 16. The complete Python code, ready to run from the beginning

The code below runs through the calculations in order. The names `x_one` and `y_one` refer to the original three-order dataset, while `X` and `y` later refer to the four-order dataset with special packaging. In variable names, `initial` identifies values before an update, `trial` identifies an experiment that changes one coefficient, and `after` identifies values after an update.

The tools we need are `numpy`, `matplotlib`, and `scikit-learn`. We use `matplotlib` to draw graphs, and import scikit-learn under the name `sklearn`. A collection of software that we install and use in this way is called a **package**.

In Jupyter Notebook, you can prepare the packages in a separate cell with `%pip install numpy matplotlib scikit-learn`, then run the code. To run a regular Python file, prepare them in a terminal with `python -m pip install numpy matplotlib scikit-learn`. You do not need to reinstall packages that are already installed.

[Download the Python code](/supervised-learning-basics/examples-en.py)

As you read the code, follow where it uses the current coefficients and which line stores the new ones. The indented lines below a `for` statement are the operations to repeat. The expression `range(n_steps)` makes the loop repeat the specified number of times. Here, `_` is a conventional name indicating that we do not use the iteration number itself in the calculation.

The call `history.append(...)` adds a new value to the end of the loss history. Later in the code, `predictions` is a **dictionary** that pairs each model name with its prediction array. Using `predictions.items()` retrieves one name and prediction array at a time so that we can evaluate all three models in the same way.

The function `np.abs` calculates magnitudes with the signs removed, while `np.sqrt` calculates square roots. The function `np.isfinite` checks that results are neither infinite nor values indicating an undefined numerical result. Calling `.all()` on its result checks whether every value in the array satisfies that condition.

The `print` calls display results, and the `assert` statements check calculations; these lines help us examine the training results. On your first run, execute the complete code from top to bottom. When changing values to experiment, also check which data and initial coefficients you are using.

```python
"""Supervised Learning: Core Concepts and Basic Coding, from delivery fees to real data.

Required packages: numpy, matplotlib, scikit-learn
In Jupyter, install them in a separate cell with %pip install numpy matplotlib scikit-learn
Run the code below from top to bottom.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=6, suppress=True)

# These fictional, simplified data describe delivery fees.
# x_one: additional distance beyond the distance included in the base fee (km); y_one: fee (thousand won).
# b is the base fee, and w is the fee per additional km. Start with b=0 and w=1.
x_one = np.array([0.0, 1.0, 2.0])
y_one = np.array([1.0, 3.0, 5.0])
b_initial, w_initial = 0.0, 1.0
pred_one = b_initial + w_initial * x_one
e_one = pred_one - y_one
sse = np.sum(e_one ** 2)
mse = np.mean(e_one ** 2)
j_before = mse / 2
mae = np.mean(np.abs(e_one))
rmse = np.sqrt(mse)
print("Initial delivery fee predictions:", pred_one)
print("Error = prediction - target:", e_one)
print("SSE, MSE, J, MAE, RMSE:", sse, mse, j_before, mae, rmse)
assert np.allclose(e_one, [-1.0, -2.0, -3.0], atol=1e-12, rtol=0)
assert np.isclose(j_before, 7 / 3, atol=1e-12, rtol=0)

# Increase only the base fee by 0.1 thousand won; keep the original fee per km.
b_trial = b_initial + 0.1
pred_b_trial = b_trial + w_initial * x_one
e_b_trial = pred_b_trial - y_one
j_b_trial = np.mean(e_b_trial ** 2) / 2
print("Predictions / errors / J after increasing only b by 0.1:", pred_b_trial, e_b_trial, j_b_trial)
assert np.allclose(pred_b_trial, [0.1, 1.1, 2.1], atol=1e-12, rtol=0)
assert np.allclose(e_b_trial, [-0.9, -1.9, -2.9], atol=1e-12, rtol=0)
assert np.isclose(j_b_trial, 1283 / 600, atol=1e-12, rtol=0)

# Start again from the original b=0, w=1, and increase only the fee per km by 0.1.
# Do not carry over b_trial from the previous experiment.
w_trial = w_initial + 0.1
pred_w_trial = b_initial + w_trial * x_one
e_w_trial = pred_w_trial - y_one
j_w_trial = np.mean(e_w_trial ** 2) / 2
print("Predictions / errors / J after increasing only w by 0.1:", pred_w_trial, e_w_trial, j_w_trial)
assert np.allclose(pred_w_trial, [0.0, 1.1, 2.2], atol=1e-12, rtol=0)
assert np.allclose(e_w_trial, [-1.0, -1.9, -2.8], atol=1e-12, rtol=0)
assert np.isclose(j_w_trial, 83 / 40, atol=1e-12, rtol=0)

# Use the derivatives to update the base fee and the fee per km together once.
# First calculate both derivatives from the errors at the original b=0, w=1.
db = np.mean(e_one)
dw = np.mean(e_one * x_one)
lr_one = 0.1
b = b_initial - lr_one * db
w = w_initial - lr_one * dw
pred_after = b + w * x_one
mse_after = np.mean((pred_after - y_one) ** 2)
j_after = mse_after / 2
print("db, dw before the update:", db, dw)
print("b, w, J after one update:", b, w, j_after)
assert np.allclose([db, dw], [-2.0, -8 / 3], atol=1e-12, rtol=0)
assert np.allclose([b, w], [0.2, 19 / 15], atol=1e-12, rtol=0)
assert np.isclose(j_after, 1829 / 1350, atol=1e-12, rtol=0)

# Writing the same three deliveries as a matrix gives the same predictions and derivatives.
# The leading column of ones adds the base fee b once to each delivery.
Xb_one = np.column_stack([np.ones(len(y_one)), x_one])
theta_one = np.array([b_initial, w_initial])
pred_matrix_one = Xb_one @ theta_one
gradient_one = Xb_one.T @ (pred_matrix_one - y_one) / len(y_one)
theta_after_matrix = theta_one - lr_one * gradient_one
print("Three-row delivery matrix Xb:\n", Xb_one)
print("Matrix predictions / [db, dw]:", pred_matrix_one, gradient_one)
print("[b, w] after one matrix update:", theta_after_matrix)
assert np.array_equal(Xb_one, [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
assert np.allclose(pred_matrix_one, pred_one, atol=1e-12, rtol=0)
assert np.allclose(gradient_one, [db, dw], atol=1e-12, rtol=0)
assert np.allclose(theta_after_matrix, [b, w], atol=1e-12, rtol=0)

# Select deliveries or the distance column and check the resulting array shapes.
print("Second delivery row:", Xb_one[1], "shape =", Xb_one[1].shape)
print("Keep the second delivery as one row:", Xb_one[[1]], "shape =", Xb_one[[1]].shape)
print("Distance column:", Xb_one[:, 1], "shape =", Xb_one[:, 1].shape)
print("Keep distance as a single column:", Xb_one[:, [1]], "shape =", Xb_one[:, [1]].shape)

# Subtract the target from the prediction for the same delivery.
# Turning only one array into a column produces an unintended 3-by-3 set of differences.
print("Prediction and target shapes:", pred_one.shape, y_one.shape)
print("Shape after turning targets into a column:", y_one[:, None].shape)
print("Correct errors for matching deliveries:", pred_one - y_one)
print("Unintended 3-by-3 result:\n", pred_one - y_one[:, None])
assert pred_one.shape == y_one.shape

# Compare learning rates on the same deliveries by fixing w=1 and updating only the base fee.
# Here db is b-2. Restart every experiment from the same b=0.
learning_rate_results = []
for rate in [0.2, 0.8, 1.5, 2.2]:
    b_rate = 0.0
    db_rate = b_rate - 2.0
    b_rate = b_rate - rate * db_rate
    pred_rate = b_rate + 1.0 * x_one
    j_rate = np.mean((pred_rate - y_one) ** 2) / 2
    learning_rate_results.append([rate, b_rate, j_rate])
    print("Learning rate / b after the first update / J:", rate, b_rate, j_rate)
assert np.allclose(np.array(learning_rate_results)[:, 1], [0.4, 1.6, 3.0, 4.4],
                   atol=1e-12, rtol=0)
assert np.allclose(np.array(learning_rate_results)[:, 2],
                   [121 / 75, 31 / 75, 5 / 6, 241 / 75], atol=1e-12, rtol=0)

# Add a second input: u is additional distance (km); v indicates special packaging (no=0, yes=1).
# Fees for these four deliveries are still measured in thousand won.
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = np.array([1.0, 3.0, 4.0, 6.0])
n = len(y)
Xb = np.column_stack([np.ones(n), X])
theta = np.zeros(3)              # [base fee b, fee per km w_u, special packaging fee w_v]
e0 = Xb @ theta - y
g0 = Xb.T @ e0 / n
j0 = np.mean(e0 ** 2) / 2
print("Four-row delivery matrix Xb:\n", Xb)
print("Initial J, gradient:", j0, g0)

# Train all three fees from zero with learning rate 0.2 for 2000 updates.
lr = 0.2
n_steps = 2000
history = []
for _ in range(n_steps):
    pred = Xb @ theta
    e = pred - y
    history.append(float(np.mean(e ** 2) / 2))  # Loss immediately before this update
    gradient = Xb.T @ e / n
    theta = theta - lr * gradient

theta_gd = theta.copy()
pred_gd = Xb @ theta_gd
final_j = float(np.mean((pred_gd - y) ** 2) / 2)
history_complete = np.array(history + [final_j])
print("GD coefficients [b, w_u, w_v]:", theta_gd)
print("GD delivery fee predictions:", pred_gd)
print("Initial / last stored / final loss:", history[0], history[-1], final_j)
print("History length including the loss after 2000 updates:", len(history_complete))

fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
shown = min(81, len(history_complete))  # Show the starting loss and the first 80 completed updates.
ax_loss.plot(np.arange(shown), history_complete[:shown])
ax_loss.set(xlabel="Completed updates", ylabel="J = MSE / 2",
            title="Gradient descent on four delivery samples")
ax_loss.grid(alpha=0.25)
fig_loss.tight_layout()

# Fit the same four deliveries with least squares and scikit-learn as well.
# The first item in the least-squares result is the coefficient array we need.
result = np.linalg.lstsq(Xb, y)
theta_ls = result[0]
pred_ls = Xb @ theta_ls
model_small = LinearRegression()
fit_return = model_small.fit(X, y)  # The model handles the base fee, so use X without a leading column of ones.
pred_sk = model_small.predict(X)
print("Least-squares coefficients:", theta_ls)
print("Does fit return the same model:", fit_return is model_small)
print("sklearn base fee / distance and packaging fees:", model_small.intercept_, model_small.coef_)
print("Least-squares / GD / sklearn predictions as columns:\n", np.column_stack([pred_ls, pred_gd, pred_sk]))
new_input = np.array([1.0, 3.0, 1.0])  # [1 for the base fee, 3 additional km, special packaging]
new_prediction = new_input @ theta_gd
print("Fee for 3 additional km with special packaging (thousand won):", new_prediction)

# Compare arrays of matching shapes to check that all three methods predict the same fees.
assert theta_gd.shape == (3,)
assert pred_gd.shape == y.shape
assert np.isfinite(theta_gd).all()
assert np.isfinite(history_complete).all()
assert np.isclose(j0, 7.75, atol=1e-12, rtol=0)
assert np.allclose(g0, [-3.5, -2.25, -2.5], atol=1e-12, rtol=0)
assert np.allclose(theta_gd, [1.0, 2.0, 3.0], atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_ls, atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_sk, atol=1e-8, rtol=0)
assert np.isclose(new_prediction, 10.0, atol=1e-8, rtol=0)
assert final_j < history_complete[0]
print("Implementation checks passed. Largest GD/least-squares prediction difference:", np.max(np.abs(pred_gd - pred_ls)))

# Move to real data, using stored feature values without the loader's additional scaling.
X_real, y_real = load_diabetes(return_X_y=True, scaled=False)
# Apply the same row indices to inputs and targets to preserve their pairing.
all_ids = np.arange(len(y_real))
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
X_train, X_test = X_real[train_ids], X_real[test_ids]
y_train, y_test = y_real[train_ids], y_real[test_ids]
assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]
print("Full / training / test input shapes:", X_real.shape, X_train.shape, X_test.shape)

# Build a baseline that predicts the mean training target for every test row.
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_test)

# Train separate models using BMI alone and all ten features.
model_bmi = LinearRegression()
model_bmi.fit(X_train[:, [2]], y_train)
pred_bmi = model_bmi.predict(X_test[:, [2]])
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

# Residuals are target minus prediction, the opposite sign of the earlier errors e.
residual = y_test - pred_all
print("Original row index of the first test sample:", test_ids[0])
print("First test sample BMI:", X_test[0, 2])
print("First test target / all-feature prediction / residual:", y_test[0], pred_all[0], residual[0])
assert test_ids[0] == 287
assert np.isclose(X_test[0, 2], 25.8, atol=1e-12, rtol=0)
assert y_test[0] == 219.0
fig_residual, ax_residual = plt.subplots(figsize=(7, 4))
ax_residual.scatter(pred_all, residual, alpha=0.65, label="Test samples")
ax_residual.scatter([pred_all[0]], [residual[0]], color="crimson",
                    s=90, label="First test sample")
ax_residual.axhline(0, color="black", linewidth=1)
ax_residual.set(xlabel="Prediction", ylabel="Residual = target - prediction",
                title="Residuals on 89 held-out samples")
ax_residual.legend()
fig_residual.tight_layout()
plt.show()
```

Running the code shows the initial errors `[-1, -2, -3]`, the coefficients `[0.2, 1.266667]` after the first simultaneous update, the coefficients `[1, 2, 3]` learned from the four orders, and a predicted fee of 10 thousand won for the new order. It then prints the three MSE values for the real dataset and the residual for the first test record.

The numbers checked in this code are **the results of these particular examples under the stated execution settings**. If you replace the examples with your own data, you should not expect the same coefficients or the same MSE.

## 17. Calculate the same sums directly with Rust loops

To see which values NumPy's matrix multiplication adds together, we can also calculate with the same four orders using loops in Rust. Rust is a different programming language from Python. The mathematics below is unchanged; we simply write the multiplications and additions one item at a time instead of performing them together.

You do not need to learn Rust syntax before understanding the Python example. The example below lets us follow the process of **calculating one order's prediction, then calculating that order's contribution to the gradient for each coefficient**.

In the Rust code, `i` is an order's position and `j` is a coefficient's position. Both start at 0. The expression `xb[i][j]` is the input value to multiply by that coefficient for that order.

For example, when all initial coefficients are 0, the first order has input `[1, 0, 0]` and an actual fee of 1. Its prediction is 0 and its error is −1. This order therefore contributes $[-1/4,0,0]$ to the overall gradient. The second order contributes $[-3/4,-3/4,0]$, the third $[-1,0,-1]$, and the fourth $[-1.5,-1.5,-1.5]$. Adding the four arrays position by position gives the same $[-3.5,-2.25,-2.5]$ we calculated earlier.

**We must finish adding the contributions from every order before updating the coefficients.** That is why the loop that updates coefficients sits outside the per-order calculations, after they are complete.

<details>
<summary>View the Rust code and how to run it</summary>

This code uses no external packages. If you save it as `gradient-descent-en.rs`, you can compile it in an environment with Rust installed by running `rustc gradient-descent-en.rs -o gradient-descent`, then run the resulting executable.

The keyword `let` declares a name, and `mut` allows its value to change later. The type `f64` stores floating-point numbers. The type `[f64; 3]` is an array of three such numbers, while `[[f64; 3]; 4]` contains four rows of three values. The range `0..2000` runs from 0 through 1999.[^rust]

The contents of `fn main()` run when the program starts. The operator `+=` adds a calculated value to the existing value, while `-=` subtracts it. The expression `y.len()` gives the number of orders, and `as f64` converts that number to a floating-point type for division. The macro `assert!` checks a condition, and `println!` prints a result. Inside the assertions, `.abs()` removes the sign of a difference to obtain its magnitude.

[Download the Rust code for this example](/supervised-learning-basics/gradient-descent-en.rs)

```rust
// Learn fees for four deliveries with gradient descent and no external packages.
// Each row is [1 for the base fee, additional distance (km), special packaging (no=0, yes=1)].
// Fees are in thousand won. Save this as main.rs and run it in a Rust project.
fn main() {
    let xb: [[f64; 3]; 4] = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let y: [f64; 4] = [1.0, 3.0, 4.0, 6.0];
    let mut theta = [0.0_f64; 3]; // [base fee b, fee per km w_u, special packaging fee w_v]
    let lr = 0.2;
    let n = y.len() as f64;

    for _ in 0..2000 {
        let mut gradient = [0.0_f64; 3];
        for i in 0..y.len() {
            // Add the base fee, distance fee, and packaging fee for one delivery.
            let mut prediction = 0.0;
            for j in 0..theta.len() {
                prediction += xb[i][j] * theta[j];
            }
            let error = prediction - y[i];
            for j in 0..theta.len() {
                gradient[j] += xb[i][j] * error / n;
            }
        }
        // Calculate derivatives for all deliveries at the old coefficients, then update all three fees.
        for j in 0..theta.len() {
            theta[j] -= lr * gradient[j];
        }
    }

    let expected = [1.0, 2.0, 3.0];
    for j in 0..theta.len() {
        assert!((theta[j] - expected[j]).abs() < 1e-8);
    }
    // Predict a delivery with 3 additional km and special packaging.
    let new_input = [1.0, 3.0, 1.0];
    let mut new_prediction = 0.0;
    for j in 0..theta.len() {
        new_prediction += new_input[j] * theta[j];
    }
    assert!((new_prediction - 10.0).abs() < 1e-8);
    println!("Coefficients [b, w_u, w_v]: {:?}", theta);
    println!("Predicted delivery fee (thousand won): {:.6}", new_prediction);
}
```

The calculation implemented by this code gives coefficients of approximately `[1, 2, 3]` and a predicted fee of approximately 10 thousand won for the new order. The environment used to revise this article did not have a Rust compiler, so the Rust file itself has not been compile-checked. The same calculations, in the same loop order, were checked in Python, and the complete Python example above was actually executed.

</details>

## References

The official lessons and documentation below explain the mathematical principles and the behavior of the functions we use. The delivery records are fictional data created to explain the calculations. The calculations in the article and the evaluation results on real data were checked with the accompanying examples. The documentation was checked on October 8, 2026.

[^sklearn-start]: scikit-learn, [Getting Started](https://scikit-learn.org/stable/getting_started.html). Input and target shapes, training and prediction, and the naming convention for fitted attributes.
[^mse]: scikit-learn, [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html). The definition of mean squared error and the function that calculates it.
[^derivative]: OpenStax, [Calculus Volume 1, 3.1 Defining the Derivative](https://openstax.org/books/calculus-volume-1/pages/3-1-defining-the-derivative), MIT OpenCourseWare, [Introduction to Derivatives](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/session-1-introduction-to-derivatives/). Ratios of changes and the meaning of a derivative at a point.
[^gradient]: MIT OpenCourseWare, [Partial Derivatives and the Gradient](https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter06/section06.html). Partial derivatives when changing one variable, and the gradient.
[^gradient-descent]: MIT OpenCourseWare, [Gradient Descent: Downhill to a Minimum](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-22-gradient-descent-downhill-to-a-minimum/). How gradient descent works.
[^chain-rule]: MIT OpenCourseWare, [The Chain Rule (PDF)](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/66ba9836b3c9e99138bc8d766d913bc5_MIT18_01SCF10_Ses11a.pdf). The chain rule for rates of change through a sequence of calculations.
[^broadcasting]: NumPy, [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). Rules for combining different array shapes.
[^indexing]: NumPy, [Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html). Selecting values by position and preserving row and column shapes.
[^matmul]: NumPy, [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html). Matrix–vector multiplication and its shape rules.
[^lstsq]: NumPy, [numpy.linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html). Calculating coefficients that minimize squared errors, and the returned results.
[^linear-regression]: scikit-learn, [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). Fitting linear regression, handling the intercept, and retrieving coefficients.
[^allclose]: NumPy, [numpy.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html). Comparing arrays with a specified tolerance.
[^assert]: Python, [The assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement). How assertions work. Running Python with the `-O` option can remove assertion statements.
[^diabetes]: scikit-learn, [load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html), [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset). Dataset size, features, targets, and loading options.
[^split]: scikit-learn, [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Splitting training and test data and reproducing the split.
[^baseline]: scikit-learn, [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html). A baseline that uses the mean of the training targets.
[^cross-validation]: scikit-learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html). The roles of validation and test data, and choosing a split appropriate to the task.
[^leakage]: scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html). Separating training and evaluation, and preventing leakage during preprocessing.
[^rust]: The Rust Programming Language, [Variables and Mutability](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html), [Data Types](https://doc.rust-lang.org/book/ch03-02-data-types.html), [Control Flow](https://doc.rust-lang.org/book/ch03-05-control-flow.html).
