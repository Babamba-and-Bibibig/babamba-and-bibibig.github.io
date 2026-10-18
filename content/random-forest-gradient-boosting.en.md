+++
title = "Predicting with Multiple Decision Trees: Random Forests and Gradient Boosting"
description = "Use the prices of four houses to understand decision trees, bagging, random forests, and gradient boosting. Follow the calculations and diagrams to distinguish nodes from trees, see why leaves predict means, learn how residuals become training targets, and predict a new house."
date = 2026-10-18T22:35:00+09:00
slug = "random-forest-gradient-boosting"

[extra]
katex = true
toc = true
styles = ["random-forest-gradient-boosting/article.css"]
scripts = ["random-forest-gradient-boosting/embed.js"]
+++

Suppose we want to predict a house's price from its location and facilities. We could train a single decision tree, or combine predictions from several different trees. <strong>A method that combines predictions from multiple models is called an ensemble.</strong>

Random forests and gradient boosting both use multiple decision trees, but the trees have different jobs. <strong>In a random forest for regression, each tree predicts a house price, and the complete model averages those prices. In gradient boosting for regression with squared error, each tree predicts a correction to the current house price prediction, and the complete model adds those corrections in sequence.</strong>[^ensemble]

To understand this difference, we first need to know what one tree learns and what it returns. We will use the same four houses to follow the process from training an individual tree to combining several trees. All houses have equal weight in these calculations, and we evaluate predictions using squared differences from the actual prices. In the diagram below, <strong>a residual is the actual target minus the current prediction of the complete model</strong>.

<figure class="boost-figure"><img src="/random-forest-gradient-boosting/overview-en.svg" alt="A regression random forest averages house price predictions from separately trained trees. Squared-error regression boosting accumulates corrections from new trees trained on the current residuals."></figure>

## 1. Pair each house's inputs with its actual price

The following is a <strong>fictional house price dataset</strong> for working through the calculations. Each row represents one house. The near-station, renovated, and parking columns use `0=no` and `1=yes`. Prices are measured in <strong>units of KRW 100 million</strong>.

| House | Near a station | Renovated | Parking | Actual price |
| --- | ---: | ---: | ---: | ---: |
| A | 0 | 0 | 1 | 5 |
| B | 0 | 1 | 0 | 7 |
| C | 1 | 0 | 0 | 11 |
| D | 1 | 1 | 1 | 17 |

Input information used for prediction is called a <strong>feature</strong>. Here, station access, renovation, and parking are the features. The actual house price is the <strong>target</strong>: the value the model is trying to predict.

By convention, we name the input table `X` and the array of actual targets `y`. In this article, `X` contains three features for each of four houses, and `y` is `[5, 7, 11, 17]`. House A has one target, the first value, 5. The other three values belong to B, C, and D, respectively.

<strong>An array is a data structure that holds multiple values in order.</strong> An array's `shape` gives the size of each dimension. Here is how to read the shapes in this example.

| Data | Contents | `shape` |
| --- | --- | --- |
| `X` | 4 houses × 3 input features | `(4, 3)` |
| `y` | One actual price per house, 4 values in total | `(4,)` |
| The model's predictions for `X` | One prediction per house, 4 values in total | `(4,)` |

`(4,)` means a one-dimensional array with four values. Inputs and targets must remain in matching order. If we reorder only the houses in `X` and leave `y` unchanged, we will pair inputs with another house's price.

<strong>Predicting a number whose magnitude has meaning, such as a house price, is a regression problem.</strong> <strong>Predicting which category something belongs to, such as whether a purchase happens or which species a penguin belongs to, is a classification problem.</strong> Decision trees can be used for both. A regression tree also divides data rows using conditions, but its final output is a number such as a price.[^tree]

## 2. Distinguish a node, a tree, and the complete model

A <strong>node</strong> is a component inside a tree. The starting node is called the <strong>root</strong>. A node that checks a condition chooses where to go next. A terminal node that does not split further is a <strong>leaf</strong>, and it outputs a prediction.

For example, a tree that splits once at the root and then reaches one of two leaves has three nodes: one root and two leaves. <strong>The whole structure containing those three nodes is one decision tree.</strong>

| Unit | What it refers to |
| --- | --- |
| One node | One component within a tree that checks a condition or outputs a result |
| One decision tree | One individual model containing its own root and all its branches and leaves |
| One complete ensemble model | Multiple individual models and a rule for combining their predictions |
| One regression boosting stage | One step that trains a new regression tree and applies its correction to the overall prediction |

If boosting uses three trees, each has its own root and leaves. Building the second tree does not attach new branches to the first tree's leaves. <strong>The complete model combines outputs from several complete trees.</strong>

### Depth is counted within one tree

<strong>Depth</strong> is the number of connections traversed from the root to a node. The root has depth 0, and its immediate children have depth 1.

`max_depth` sets the <strong>maximum permitted depth</strong> of each tree. A scikit-learn binary decision tree with `max_depth=1` can have at most two leaves and three nodes. It may be smaller if the conditions for splitting are not met. `max_depth=3` permits depths 0, 1, 2, and 3; it does not require the tree to reach depth 3.[^regressor]

`n_estimators`, in contrast, determines how many models or stages are used in the ensemble. In this regression boosting example, `n_estimators=3` means <strong>three correction trees</strong>. Each tree's depth is set separately through `max_depth`. The initial prediction used at the start is not counted as one of these three trees.

### One call to `fit()` trains the whole tree

`DecisionTreeRegressor` is a <strong>class</strong> that creates regression-tree objects. An individual object created with the class's settings is the model we use. A function belonging to an object is a <strong>method</strong>, and a value stored on an object is an <strong>attribute</strong>.

| Code | Operation | Return value |
| --- | --- | --- |
| `DecisionTreeRegressor(...)` | Create a new tree object with the specified settings | An untrained model object |
| `tree.fit(X, target)` | Determine split conditions and leaf values from inputs and targets | The fitted `tree` object itself |
| `tree.predict(X)` | Apply the learned conditions to each input row | An array containing one prediction per row |

Here, `target` refers to the training targets supplied to the tree. We use `y` when learning house prices. Later, when learning corrections, we will supply calculated residuals.

Inside `fit()`, the algorithm compares candidate conditions, creates child nodes, repeats the necessary splits, and stores leaf outputs. <strong>A single call to `fit()` therefore performs many calculations across multiple nodes.</strong> Calling `fit()` again on a standard `DecisionTreeRegressor` rebuilds the tree using the newly supplied data.[^regressor]

`predict()` uses the rules learned during training. If the input has four rows, one method call returns predictions for all four houses. <strong>The number of models, the number of method calls, and the number of values in the output array must each be counted separately.</strong>

## 3. How does a regression tree choose a split and an output?

When the near-station column contains 0 or 1, the condition `near_station <= 0.5` separates the houses into two groups. A and B satisfy the condition; C and D do not. The threshold `0.5` distinguishes the two input values, 0 and 1.

Once a condition has been chosen, we can divide rows using only the inputs `X`. <strong>During training, however, the algorithm must choose a useful condition and decide which price to output for each resulting group.</strong> This requires the actual targets `y`.

### Square each prediction error, then take the mean

<strong>Mean squared error (MSE)</strong> is calculated by squaring the difference between the target and prediction for each row, adding those squared differences, and dividing by the number of rows. A quantity used to assess a model's errors in this way is called a <strong>loss</strong>. In this article, training aims to produce predictions with a smaller MSE.

Suppose a leaf predicts the same price, 6, for every house that reaches it. If those houses are A and B, with actual prices 5 and 7, the calculation is:

<div class="boost-math">
$$
\mathrm{MSE}=\frac{(5-6)^2+(7-6)^2}{2}
=\frac{1+1}{2}=1
$$
</div>

The two differences are −1 and +1, but squaring makes both of them 1. If we averaged the differences without squaring them, they could cancel and give 0 even though the predictions were wrong.

Squaring also changes the units. If prices are measured in units of KRW 100 million, <strong>MSE is measured in `(KRW 100 million)²`</strong>. An MSE of 1 does not mean that the average error is exactly KRW 100 million.

### Why predict the mean for houses in the same leaf?

A regression tree trained using squared error outputs the mean of the targets in each leaf.[^regressor] The reason is that <strong>when we must output one common number for every row in a group, the mean gives that group the smallest MSE</strong>. We can establish this with a general calculation.

| Symbol | Meaning |
| --- | --- |
| $k$ | The number of training rows in one leaf; at least 1 |
| $t_i$ | The training target for the $i$th row in that leaf |
| $c$ | The common number that the leaf will output for every row |
| $a$ | The arithmetic mean of the targets in that leaf |
| $\sum_{i=1}^{k}$ | Add all terms as $i$ takes each value from 1 through $k$ |

We write the mean $a$ and the MSE for a common prediction $c$ as follows:

<div class="boost-math">
$$
a=\frac{1}{k}\sum_{i=1}^{k}t_i,
\qquad
\mathrm{MSE}(c)=\frac{1}{k}\sum_{i=1}^{k}(t_i-c)^2
$$
</div>

Now split each difference $t_i-c$ into two parts around the mean. Adding $(t_i-a)+(a-c)$ cancels $-a$ with $+a$, leaving the original difference.

<div class="boost-math">
$$
\begin{aligned}
\mathrm{MSE}(c)
&amp;=\frac{1}{k}\sum_{i=1}^{k}\bigl((t_i-a)+(a-c)\bigr)^2\\
&amp;=\frac{1}{k}\sum_{i=1}^{k}(t_i-a)^2
+\frac{2(a-c)}{k}\sum_{i=1}^{k}(t_i-a)
+(a-c)^2
\end{aligned}
$$
</div>

The second line expands the square of the sum. Because $(a-c)^2$ is the same for every row, adding it $k$ times and dividing by $k$ leaves $(a-c)^2$.

The sum in the middle term is 0. This follows from the definition of the mean, which gives $ka=\sum_{i=1}^{k}t_i$.

<div class="boost-math">
$$
\sum_{i=1}^{k}(t_i-a)
=\sum_{i=1}^{k}t_i-ka
=0
$$
</div>

The expression therefore becomes:

<div class="boost-math">
$$
\boxed{\mathrm{MSE}(c)=\mathrm{MSE}(a)+(a-c)^2}
$$
</div>

$(a-c)^2$ is always at least 0 and equals 0 only when $c=a$. Choosing any number other than the mean adds $(a-c)^2$ to the MSE obtained with the mean. That is why the leaf uses the mean as its common prediction.

This conclusion assumes <strong>squared error and equal row weights</strong>. If rows have different weights, the tree uses a weighted mean. A regression tree trained using absolute error uses the median instead of the mean. The median is the middle value when the numbers are placed in order; with an even number of values, it is conventionally the mean of the two middle values.[^regressor]

### Calculate the means, then compare candidate splits

A tree of depth 1 that learns house prices directly can compare the MSE obtained by splitting on each column. Every prediction array in the following table is ordered A, B, C, D.

| Candidate split | Mean price in each group | Predictions for A, B, C, D | Overall MSE |
| --- | --- | --- | ---: |
| No split | Overall mean: 10 | `[10, 10, 10, 10]` | 21 |
| Near a station | A, B: 6 / C, D: 14 | `[6, 6, 14, 14]` | 5 |
| Renovated | A, C: 8 / B, D: 12 | `[8, 12, 8, 12]` | 17 |
| Parking | B, C: 9 / A, D: 11 | `[11, 9, 9, 11]` | 20 |

For example, splitting on station access gives leaf outputs of $(5+7)/2=6$ and $(11+17)/2=14$. The MSE across all four houses is:

<div class="boost-math">
$$
\frac{(5-6)^2+(7-6)^2+(11-14)^2+(17-14)^2}{4}
=\frac{1+1+9+9}{4}=5
$$
</div>

This is the smallest value among the candidates, so station access is selected for the first split. The mean is calculated <strong>to set each group's prediction after splitting and compare the candidates' prediction errors</strong>. It is not needed simply to divide rows according to a condition. If the two groups have different numbers of rows, those row counts must be reflected in the overall MSE.

If we allow greater depth, the algorithm can compare further conditions within each group. Here, depth is limited to 1, so A and B receive the same leaf value, 6. Choosing a good split at each node in this way does not guarantee the best structure among all possible complete trees.[^tree]

## 4. Bagging trains multiple models on differently sampled data

A small change in training data can change a tree's split conditions. If the first condition changes, the groups formed later can change too. <strong>Bagging</strong> trains multiple models on different samples of the training data and combines their predictions to reduce this variation.[^ensemble]

First, consider <strong>sampling with replacement</strong>. A row that has already been selected remains eligible for the next draw. If the original rows are A, B, C, and D, four draws might produce the following samples. These groups illustrate the sampling method.

| Individual model | Houses used for training, in order | Matching actual prices |
| --- | --- | --- |
| Model 1 | A, A, C, D | 5, 5, 11, 17 |
| Model 2 | B, C, C, A | 7, 11, 11, 5 |
| Model 3 | D, D, B, C | 17, 17, 7, 11 |

Each sample contains four rows, but it does not necessarily contain every house once. Model 1 receives A twice and does not receive B. <strong>Whenever a row is sampled, its inputs and target are taken together.</strong>

Resampling data with replacement in this way is called <strong>bootstrapping</strong>. Bagging trains individual models on the resulting samples. For regression, the final prediction for a new input is the mean of the individual models' predictions.[^bagging]

All the models can use the same learning algorithm and settings. They are still separate model objects, and because they receive different data, their learned conditions and outputs need not be identical. <strong>Training each model does not require waiting for another model's predictions.</strong> This makes it straightforward to train models at the same time, a form of parallel processing.

If the models make almost identical errors, averaging will not reduce those errors by much. For combining predictions to help, the models need some opportunity to predict differently. Bagging creates that difference by sampling different data. The individual models used for bagging do not have to be decision trees.[^bagging]

## 5. Random forests also control which columns are considered for splits

A <strong>random forest</strong> combines predictions from multiple decision trees. With scikit-learn's default settings, each tree receives training rows sampled with replacement. We can also limit <strong>the number of columns considered as candidates for splitting at each node</strong>.[^forest]

If every tree examines all three columns in the house dataset, several trees may select the same station-access condition first. Randomly limiting the candidate columns gives other conditions a chance to be selected. This is one way to reduce how similar the trees become.

For example, `max_features=2` selects two candidate columns when looking for a split. The following table illustrates how the candidates can differ between nodes. It is not a record of the depth-1 tree trained earlier.

| Where candidates are selected | Example of two columns that can be considered |
| --- | --- |
| A tree's root | Near a station, renovated |
| That tree's left child node | Renovated, parking |
| That tree's right child node | Near a station, parking |

<strong>Candidates are selected for the individual node being split.</strong> The algorithm does not assign one column to each tree or require every node at the same depth to use the same column. A column can be used again at multiple nodes and in multiple trees. Candidate selection and condition comparison happen inside `fit()`.

### The same name, `max_features`, applies at different places

| Model | What `max_features` controls |
| --- | --- |
| `BaggingRegressor` | The number of input columns passed to each individual model |
| `RandomForestRegressor` | The number of candidate columns considered when searching for a split at each node |

The defaults also differ between classification and regression. `RandomForestClassifier` uses `max_features="sqrt"` by default: take the square root of the total number of columns, keep the integer part, and consider at least one column. The default for `RandomForestRegressor` is <strong>`max_features=1.0`</strong>, which considers all columns.[^forest][^forest-classifier]

A regression random forest therefore does not always consider only a subset of columns. There is also a difference between the integer `1`, meaning one column, and the float `1.0`, meaning 100% of the columns. In addition, the implementation may inspect more than the specified number of columns to find a valid split.[^forest]

An ordinary `DecisionTreeRegressor` can also limit candidates at each node. A random forest combines this tree-training process with the predictions of multiple trees. Individual models in bagging do not need to contain nodes at all, whereas the individual models in a random forest are decision trees.

### After training, average predictions for the same house

Each tree in a regression random forest outputs a house price. With three trees, <strong>add the three predictions for the same house and divide by 3</strong>. Predictions for different houses are not averaged together.

For classification, the combination rule changes to match the output. A scikit-learn classification random forest <strong>averages each class's probability across trees</strong>, then selects the class with the highest mean probability. Classification bagging also averages probabilities when its individual models provide probability predictions. Otherwise, it uses a vote among their predicted classes.[^forest-classifier][^bagging-classifier]

## 6. Boosting learns corrections to the current prediction

In the random forest, each tree predicted the original house price. Now consider a structure that <strong>keeps the current prediction and improves it by adding the output of a new tree</strong>.

<div class="boost-math">
$$
\text{new overall prediction}=\text{current overall prediction}+\text{new tree's correction}
$$
</div>

<strong>Boosting</strong> is a strategy that trains and combines models in sequence, using the results of earlier stages when training the next model. <strong>Gradient boosting</strong> calculates a direction that reduces the loss, then adds a model that predicts that direction. We will first follow the squared-error regression calculation. Section 9 connects this calculation to the word *gradient* in the name.[^boosting]

This section and the next numerical example add each correction in full. That corresponds to a <strong>learning rate</strong> of 1. The learning rate, explained later, determines the fraction of a correction to apply.

### Subtract the current overall prediction from the actual target

What must we add to the current prediction to obtain the actual target? The answer is the difference between them.

<div class="boost-math">
$$
\text{residual}=\text{actual target}-\text{current overall prediction}
$$
</div>

A positive <strong>residual</strong> means the prediction is below the target and needs to increase. A negative residual means the prediction is above the target and needs to decrease. The residual is the required adjustment to the current prediction and has the same units as the house price.

The outer boosting code calculates the residuals, then passes them to a new regression tree through `fit(X, residual)`. Here, `residual` is the name of the residual array. The new tree learns to predict these numbers from the input features. It does not need to know what the word *residual* means or how the previous tree is structured.

We also <strong>keep the original targets `y`</strong> throughout this process. Whenever the current predictions change, we need to compare them with those targets again.

### Why does predicting residuals reduce the final error?

Let $y_{\text{true}}$ be the actual target, $\hat y$ the current overall prediction, and $v$ the correction predicted by the new tree. Define the current residual as $r=y_{\text{true}}-\hat y$. The squared error after applying the correction is:

<div class="boost-math">
$$
\begin{aligned}
\bigl(y_{\text{true}}-(\hat y+v)\bigr)^2
&amp;=\bigl(y_{\text{true}}-\hat y-v\bigr)^2\\
&amp;=\bigl((y_{\text{true}}-\hat y)-v\bigr)^2\\
&amp;=(r-v)^2
\end{aligned}
$$
</div>

The first line compares the new overall prediction with the actual target. The last line compares the new tree's correction with the residual. Because the expressions have the same value, <strong>with the current overall prediction held fixed, training a new tree to predict the residuals has the same objective as reducing the squared error of the corrected overall prediction</strong>.

That is why the new tree learns the current residuals rather than predicting the house prices again. However, it must output a common number for houses in the same leaf, so it may not match every individual residual exactly. The differences that remain after this correction become targets for the next tree.

## 7. Correct the four house predictions with three trees

At the start, before any correction trees exist, we predict the same price for every house. As established earlier, the best common number under squared error is the mean.

<div class="boost-math">
$$
\text{initial prediction}=\frac{5+7+11+17}{4}=10
$$
</div>

For squared-error regression, scikit-learn's `GradientBoostingRegressor` also uses the mean of the training targets as its default initial prediction.[^boosting] Here, 10 is the <strong>starting prediction</strong> for each house. It is stored separately from the correction trees.

### First tree: divide the current residuals into two groups

| House | Actual price | Current overall prediction | Residual used as the new tree's target |
| --- | ---: | ---: | ---: |
| A | 5 | 10 | $5-10=-5$ |
| B | 7 | 10 | $7-10=-3$ |
| C | 11 | 10 | $11-10=1$ |
| D | 17 | 10 | $17-10=7$ |

Create a new regression tree of depth 1 and train it on these residuals. Splitting on station access puts A and B in one leaf and C and D in the other. Each leaf outputs <strong>the mean of the residuals that reach it</strong>.

<div class="boost-math">
$$
\text{correction for A and B}=\frac{-5+(-3)}{2}=-4
$$
</div>

<div class="boost-math">
$$
\text{correction for C and D}=\frac{1+7}{2}=4
$$
</div>

<figure class="boost-figure"><img src="/random-forest-gradient-boosting/first-tree-en.svg" alt="The complete first correction tree. It has one root at depth 0 and two leaves at depth 1, and outputs -4 or +4 according to station access."></figure>

The entire diagram is the first tree. A's actual residual is −5 and B's is −3, but the tree predicts −4 for both. <strong>The residual used as a training target can differ from the correction predicted by the tree.</strong>

Splitting on station access gives an MSE of 5 after correction. Splitting on renovation gives 17, and splitting on parking gives 20, so station access is selected again. Subtracting the same initial value, 10, from all original targets also lowers every group's mean by 10. The differences between targets and their group means stay the same. That is why the candidates have the same MSE values as when the tree learned house prices directly.

Adding the chosen corrections to the existing predictions gives:

| House | Previous prediction | First tree's correction | Updated overall prediction | Remaining residual |
| --- | ---: | ---: | ---: | ---: |
| A | 10 | −4 | 6 | −1 |
| B | 10 | −4 | 6 | +1 |
| C | 10 | +4 | 14 | −3 |
| D | 10 | +4 | 14 | +3 |

House C's actual price is 11, but its prediction increased from 10 to 14. Its absolute error therefore grew from 1 to 3. Because C and D share a leaf, both receive the same correction of +4. <strong>An individual house's prediction can get worse even while the overall MSE decreases</strong>, in this case from 21 to 5.

The result, `[6, 6, 14, 14]`, matches the depth-1 tree that learned prices directly. Under these conditions, outputting each group's mean house price gives the same result as adding the group's mean residual to the initial value, 10. Boosting continues by training new trees on the residuals that remain.

### The correction −4 and the MSE 1 serve different purposes

The correction used in the leaf for A and B is the mean, −4. To evaluate how well it predicts the two target residuals, calculate:

<div class="boost-math">
$$
\frac{(-5-(-4))^2+(-3-(-4))^2}{2}=1
$$
</div>

−4 determines how much to lower the predictions. The value 1 measures the squared error of those correction predictions.

Squaring and averaging the residuals themselves gives $((-5)^2+(-3)^2)/2=17$. This is <strong>the MSE while the correction is still 0</strong>. We do not add 17 to the prediction. A correction needs a sign to indicate whether to raise or lower the prediction.

If the residuals within one leaf are −3 and +3, the common correction is their mean, 0. The MSE is still $(9+9)/2=9$. A correction of 0 is the best common adjustment available for those two houses; it does not mean that both predictions are correct.

### Second tree: recalculate residuals from the updated predictions

After the first stage, the current overall predictions are `[6, 6, 14, 14]`. Subtracting this array from the actual prices gives the second set of training targets.

<div class="boost-math">
$$
[5,7,11,17]-[6,6,14,14]=[-1,1,-3,3]
$$
</div>

This time, the new tree splits on renovation. The mean residual for unrenovated houses A and C is $(-1-3)/2=-2$. For renovated houses B and D, it is $(1+3)/2=2$.

| House | Previous overall prediction | Second tree's correction | Updated overall prediction | Remaining residual |
| --- | ---: | ---: | ---: | ---: |
| A | 6 | −2 | 4 | +1 |
| B | 6 | +2 | 8 | −1 |
| C | 14 | −2 | 12 | −1 |
| D | 14 | +2 | 16 | +1 |

The values −4 and +4 stored in the first tree are unchanged. The second tree has separately learned corrections of −2 and +2, which are added to the overall predictions.

### Third tree: learn the differences that remain

The new residuals are `[1, -1, -1, 1]`. Houses A and D, which have parking, both have a residual of +1. B and C, which do not have parking, both have −1. The third tree can separate these values using the parking condition.

<div class="boost-math">
$$
[4,8,12,16]+[1,-1,-1,1]=[5,7,11,17]
$$
</div>

The following table collects the accumulated results at each stage. Every array is ordered A, B, C, D.

| State | Correction predicted by the tree added at this stage | Overall price predictions | Overall MSE |
| --- | --- | --- | ---: |
| Initial | No correction trees yet | `[10, 10, 10, 10]` | 21 |
| Stage 1 | `[-4, -4, 4, 4]` | `[6, 6, 14, 14]` | 5 |
| Stage 2 | `[-2, 2, -2, 2]` | `[4, 8, 12, 16]` | 1 |
| Stage 3 | `[1, -1, -1, 1]` | `[5, 7, 11, 17]` | 0 |

Station access, renovation, and parking were not assigned to the trees in advance. This order emerged from comparing conditions against each stage's residuals. With different training data or settings, the same column may be selected repeatedly across stages.

Select a stage below to see <strong>the new tree created at that stage</strong> alongside each house's residual, correction, and overall prediction. The diagram displays the calculation above with a learning rate of 1.

<iframe class="boost-demo" data-boosting-demo src="/random-forest-gradient-boosting/stages.html?lang=en" title="Trees and house price calculations at each regression boosting stage" loading="lazy" height="1180"></iframe>

[Open the stage diagrams in a separate view](/random-forest-gradient-boosting/stages.html?lang=en)

The training MSE for these four houses has reached 0, but that value alone does not show that other houses will be predicted accurately. What we have checked so far is <strong>the calculation for the four rows used in training</strong>.

## 8. Predict an unseen house using the stored trees

Now consider a new house, E. It is <strong>near a station, has not been renovated, and has parking</strong>. Its input is `[[1, 0, 1]]`. The outer brackets contain the houses, and the inner brackets contain the three features of one house, so its `shape` is `(1, 3)`.

All three trees receive this input. The first tree checks station access and outputs +4. The second checks renovation and outputs −2. The third checks parking and outputs +1.

| Component used for prediction | Calculation for E | Output |
| --- | --- | ---: |
| Initial prediction | Use the starting value stored during training. | 10 |
| First tree | Follow the branch for houses near a station to its leaf. | +4 |
| Second tree | Follow the branch for unrenovated houses to its leaf. | −2 |
| Third tree | Follow the branch for houses with parking to its leaf. | +1 |

The overall prediction is:

<div class="boost-math">
$$
10+4-2+1=13
$$
</div>

The model predicts a price of <strong>13 × KRW 100 million</strong> for E. Its actual price has not been provided, so we cannot yet evaluate the accuracy of this prediction.

Predicting a new house does not require its actual target or actual residual. <strong>The model applies rules learned from residuals during training to the new house's inputs, producing corrections.</strong> There is no new call to `fit()`.

The second tree also receives E's original three features, not the first tree's output of +4. <strong>During training, the next tree depends on the overall predictions from the previous stage. After training, each tree can calculate its own output from the same input.</strong> The order in which trees are trained is different from the way a new input is passed to them.

## 9. Why is it called gradient boosting if it learns residuals?

<strong>A derivative is an instantaneous rate of change: it describes how a function's value changes when an input number changes by a very small amount.</strong> If several numbers can change, the collection of derivatives with respect to those numbers is called the <strong>gradient</strong>.

<strong>Gradient descent</strong> calculates the loss's rate of change, then adjusts values in small steps in a direction that lowers the loss. Here, we examine how the <strong>prediction output by the complete model</strong> should change.

### Change the prediction slightly and calculate the change in loss

As before, write the actual target as $y_{\text{true}}$ and the current prediction as $\hat y$. Define the loss function $L$ for one row as:

<div class="boost-math">
$$
L(\hat y)=\frac12(y_{\text{true}}-\hat y)^2
$$
</div>

The factor $1/2$ is a positive constant included to simplify the derivative. It scales every loss value by the same amount, so it does not change which prediction minimizes the loss.

Let $\Delta$ be a small change added to the prediction. It can be positive or negative; when dividing by it, assume it is nonzero. Changing the prediction to $\hat y+\Delta$ changes the loss by:

<div class="boost-math">
$$
\begin{aligned}
L(\hat y+\Delta)-L(\hat y)
&amp;=\frac12\left[(y_{\text{true}}-\hat y-\Delta)^2
-(y_{\text{true}}-\hat y)^2\right]\\
&amp;=-(y_{\text{true}}-\hat y)\Delta+\frac12\Delta^2\\
&amp;=(\hat y-y_{\text{true}})\Delta+\frac12\Delta^2
\end{aligned}
$$
</div>

The second line follows by expanding the square and canceling the common terms. Dividing this change in loss by the change in prediction, $\Delta$, gives the rate of change over that interval.

<div class="boost-math">
$$
\frac{L(\hat y+\Delta)-L(\hat y)}{\Delta}
=\hat y-y_{\text{true}}+\frac12\Delta
$$
</div>

As $\Delta$ approaches 0, the final term approaches 0. We write the remaining instantaneous rate of change as $\frac{dL}{d\hat y}$. This means the derivative of the loss with respect to the prediction $\hat y$.

<div class="boost-math">
$$
\frac{dL}{d\hat y}=\hat y-y_{\text{true}}
$$
</div>

If the prediction exceeds the target, this value is positive. Increasing the prediction further increases the loss, so the prediction should decrease. If the prediction is below the target, the derivative is negative, and increasing the prediction lowers the loss. For a small correction, we therefore move in <strong>the direction with the opposite sign to the derivative</strong>.

<div class="boost-math">
$$
\boxed{-\frac{dL}{d\hat y}=y_{\text{true}}-\hat y=\text{residual}}
$$
</div>

The residual learned by each new tree is exactly this direction for reducing the loss. <strong>Gradient boosting trains a new tree to predict this direction from input features, then uses the tree's output to change the overall prediction.</strong> The individual regression tree still learns by comparing candidate splits and calculating leaf values.[^ensemble]

### Differentiating the full MSE introduces a constant factor

The exact equality above holds <strong>when the loss for one row is defined as half the squared error</strong>. Ordinary MSE averages the squared errors across all rows.

Let $n$ be the total number of rows, $y_i$ and $\hat y_i$ the target and prediction for row $i$, and $J$ the overall MSE. Then:

<div class="boost-math">
$$
J=\frac1n\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$
</div>

The derivative when changing just one prediction, $\hat y_i$, is written $\frac{\partial J}{\partial\hat y_i}$. Changing this prediction leaves the other rows' losses unchanged. Only this row's squared error changes, giving:

<div class="boost-math">
$$
\frac{\partial J}{\partial\hat y_i}
=\frac1n\cdot2(\hat y_i-y_i),
\qquad
-\frac{\partial J}{\partial\hat y_i}
=\frac2n(y_i-\hat y_i)
$$
</div>

This is the residual multiplied by the positive constant $2/n$. The direction that lowers the loss is therefore the same, but the negative derivative is not always numerically equal to the residual regardless of how the loss is defined.

## 10. What changes with the learning rate or loss function?

The <strong>learning rate</strong> determines how much of a new tree's predicted correction is applied. Writing the learning rate as $\eta$, the update becomes:

<div class="boost-math">
$$
\text{new overall prediction}
=\text{current overall prediction}+\eta\times\text{new tree's correction}
$$
</div>

The numerical example above used $\eta=1$, applying each correction in full. With a learning rate of 0.1, only 10% of the first tree's correction of −4 for A is applied.

<div class="boost-math">
$$
10+0.1\times(-4)=9.6
$$
</div>

A's next residual is then $5-9.6=-4.6$. This differs from the residual of −1 obtained with a learning rate of 1. <strong>Changing the learning rate also changes what the next tree will learn.</strong> Multiplying the outputs of trees already trained with a learning rate of 1 by 0.1 at the end is a different process from training new trees with a learning rate of 0.1 from the start.

A smaller learning rate makes each stage's correction smaller. It may therefore take more trees to reach a similar training error. The learning rate and number of trees need to be considered together, and larger corrections do not necessarily improve performance.[^boosting]

### Classification boosting corrects numerical scores

The residual means calculated so far apply to <strong>squared-error regression</strong>. Gradient boosting can use other loss functions, which change the new model's training targets and the way leaf values are calculated.[^ensemble]

For example, in binary classification that distinguishes purchase from no purchase, the complete model can first produce a numerical score and then convert it to a probability between 0 and 1. The <strong>sigmoid</strong> is the function that converts this score into a probability.

In typical binary gradient boosting with log loss, each regression tree corrects this <strong>score before it is converted to a probability</strong>. Log loss assigns a larger loss when the actual class receives a lower probability. The model adds the trees' score corrections before calculating the probability. The corrections in the regression example should therefore not be treated as values added directly to probabilities.

The complete model performs classification, but its individual trees are regression trees that output numerical corrections. Also, scikit-learn's `GradientBoostingClassifier` trains one regression tree per stage for binary classification and one regression tree per class per stage for multiclass classification. <strong>The earlier statement that one stage adds one tree refers to the regression example.</strong>[^boosting-classifier]

Boosting as a whole does not always learn simple residuals. Methods such as AdaBoost adjust the weights assigned to training samples and use those weights when training the next model. The algorithm and its loss function must be considered together.[^ensemble]

## 11. Check performance on new data as the number of trees grows

Adding a tree to a random forest adds another house price prediction to the average. The new tree is not specifically trained on the errors of houses that the existing trees predicted incorrectly. Instead, the model averages more predictions obtained from different samples and conditions.

As more predictions enter the average, any one tree has less influence. Variation caused by the random training process therefore tends to decrease, but the improvements can become smaller as similar predictions accumulate. The score on a finite validation dataset is not guaranteed to improve with every added tree.[^ensemble]

In boosting, each new tree learns the differences that remain in the current overall predictions. More stages can express the relationship between house features and prices in greater detail, but they can also start following differences that happen to occur only in the training data.

<strong>Overfitting</strong> occurs when a model fits the training data so closely that its performance does not carry over to new data. It must be checked for deep individual trees and random forests as well as for boosting. In particular, the training MSE of 0 in the example is not evidence that the model has correctly predicted a new house's actual price.

| What to compare | Regression random forest | Gradient boosting for squared-error regression |
| --- | --- | --- |
| Each tree's training target | Original prices of the sampled houses | Residuals of the current overall predictions |
| When other trees' results are needed | Each tree can be trained separately. | The previous stage's predictions are needed to calculate the next tree's targets. |
| Final prediction for a new house | Mean of the trees' house price predictions | Initial prediction plus the sum of corrections scaled by the learning rate |
| What adding trees does | Adds predictions to the average | Adds stages that correct the current prediction |
| What else to check | Whether gains become small and whether errors from deep trees remain | Whether validation performance improves and whether continued correction causes overfitting |

For both methods, more trees can require more computation for training and prediction, as well as more space to store the model. Random-forest trees can readily be trained separately. Boosting uses the previous stage's results to create the next targets, so all its trees cannot be trained at once. This does not mean that computations within each stage cannot be parallelized.

### Give training, validation, and test data separate roles

In an actual analysis, first decide what to predict and check that each column's meaning and values are appropriate. Check missing values, duplicate records, and values that are impossible in the real world. Then divide the data according to their purpose.

| Data | Purpose |
| --- | --- |
| Training | Learn split conditions, leaf values, the initial mean, and stage-specific residuals. |
| Validation | Compare manually chosen settings such as tree depth, tree count, and learning rate. |
| Test | Evaluate final performance after selecting the settings. |

These settings are called <strong>hyperparameters</strong>. The following settings control different parts of the process.

| Setting | What it controls |
| --- | --- |
| `max_depth` | The maximum depth of each individual tree |
| `min_samples_leaf` | The minimum number of training samples that must remain in each leaf |
| `n_estimators` | The number of trees in a forest, or the number of training stages in gradient boosting |
| `max_features` | Input columns per model in bagging, or candidate split columns per node in a forest |
| `learning_rate` | How much of each new correction boosting applies |
| `random_state` | Reproducing random choices under the same conditions |

The number 42 in `random_state=42` has no special property that improves accuracy. It is a value used to make comparisons under matching conditions.

Boosting can also use <strong>early stopping</strong>. This ends training when validation performance no longer improves according to a specified criterion. With early stopping enabled, training can finish before reaching the specified `n_estimators`. Early stopping was not enabled in the three-tree example.[^boosting]

<strong>Data leakage</strong> occurs when information that should not be used for training enters model training, or when test data reserved for final evaluation are used to select settings. For example, boosting's initial mean must be calculated from training targets alone. Including validation or test house prices in that mean, or using those rows to train new trees on residuals, allows evaluation targets into training. The same principle applies when calculating means or other reference values during preprocessing.[^pitfalls]

Compare settings on validation data, then evaluate on test data after the choice is complete. The four-row calculation example shows how the trees are combined. Determining which method works well for a real problem requires separate data and an evaluation suited to that problem.

## 12. Run the complete calculation in Python

The code below creates a single regression tree, a bagging model, and a random forest using the same four houses. It then implements the boosting training loop directly. Finally, it trains scikit-learn's complete boosting model and checks that the results match. The required packages are NumPy and scikit-learn.

<strong>NumPy is a Python library for storing numbers in arrays and performing calculations on them.</strong> The following functions and methods are introduced in the code.[^numpy]

| Name and kind | Input and return value |
| --- | --- |
| `np.array(...)` function | Create a NumPy array from values in a list. Nested lists of rows form a two-dimensional array. |
| `np.full(shape, fill_value)` function | Create an array of the specified size and fill every element with the same value. |
| `y.mean()` method | Here, return the number 10, the mean of the four house prices. |
| `np.mean(...)` function | Here, take the four squared house price errors and return their mean as one number. |
| `trees.append(tree)` method | Add a trained tree object to the end of the list. Return `None`. |
| `tree.apply(X)` method | Return the node index of the leaf reached by each row. The result has shape `(4,)` here. |
| `boosting.staged_predict(X)` method | Provide the overall prediction array for each stage in sequence. Each array has shape `(4,)`. |

In NumPy, `y - current_prediction` subtracts elements at matching positions when the arrays have the same shape. `** 2` squares each element. Therefore, `np.mean((y - current_prediction) ** 2)` is the overall MSE for the four houses.

The array `forest_outputs`, which collects the forest's individual predictions, has shape `(3, 1)`: <strong>three trees × one new house</strong>. `mean(axis=0)` reduces the first dimension, which contains the three trees, by averaging. This leaves an array of shape `(1,)` containing <strong>the mean prediction for that one house</strong>. With several houses, it averages across trees while preserving each house's position.

`tree.tree_` is an attribute containing the learned internal tree structure. `tree_.node_count` gives the number of nodes, and `get_depth()` returns the actual depth. The values 1 and 2 returned by `apply(X)` are <strong>leaf node indices within an individual tree</strong>.

The attribute `boosting.estimators_` contains the trained individual trees. Here, its shape is `(3, 1)`, with one regression tree for each of three stages. `boosting.estimators_[0, 0]` is the first complete tree object. Python positions are indexed starting at 0.[^regressor][^boosting]

The manual loop passes residuals to `tree.fit(X, residual)`. In contrast, <strong>the complete model's `boosting.fit(X, y)` receives the actual house prices `y`</strong>. Inside that call, the library calculates the initial prediction and stage-specific training targets, then trains the individual trees in sequence. One call to the complete model's `fit()` therefore contains the training of multiple trees.

[Download the executable Python file](/random-forest-gradient-boosting/examples-en.py)

```python
"""Check decision trees, bagging, random forests, and regression boosting with four houses.

Required packages: numpy, scikit-learn.
The data are fictional and illustrate the calculations. Prices are in units of KRW 100 million.
All rows have equal weight, and the regression trees are trained using squared error.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

# Each row is one house. Column order: [near a station, renovated, parking].
# For each input, 0 means no and 1 means yes.
X = np.array([
    [0, 0, 1],  # House A
    [0, 1, 0],  # House B
    [1, 0, 0],  # House C
    [1, 1, 1],  # House D
])
y = np.array([5.0, 7.0, 11.0, 17.0])  # Actual prices in the same order.
new_X = np.array([[1, 0, 1]])  # New house E: 1 row, 3 columns. Its actual price is unknown.

print("Input and target shapes:", X.shape, y.shape)  # (4, 3), (4,)

# 1. Create one regression tree that learns the house prices themselves.
single_tree = DecisionTreeRegressor(
    criterion="squared_error",  # Evaluate candidate splits using squared error.
    max_depth=1,                # Allow only depths 0 (the root) and 1.
    random_state=42,            # Make random choices reproducible.
)
fit_result = single_tree.fit(X, y)  # Learn all the tree's conditions and leaf values.
assert fit_result is single_tree   # fit returns the model object itself.
print("Tree trained directly on prices:", single_tree.predict(X))  # [6, 6, 14, 14]

# 2. Bagging trains three models on different samples of the training data.
bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=1),
    n_estimators=3,
    bootstrap=True,    # Sample training rows with replacement.
    max_features=1.0,   # Pass all columns to each model.
    random_state=42,
)
bagging.fit(X, y)       # Each model learns the original prices of its sampled houses.
print("Number of bagging models:", len(bagging.estimators_))  # 3

# 3. In a random forest, max_features sets the number of candidate columns at a node.
forest = RandomForestRegressor(
    n_estimators=3,
    max_depth=1,
    bootstrap=True,
    max_features=2,    # Select two candidate columns for each split in this example.
    random_state=42,
)
forest.fit(X, y)

# Each tree predicts a price for the same new house. Their mean is the final prediction.
forest_outputs = np.array([tree.predict(new_X) for tree in forest.estimators_])
print("Individual forest price predictions:", forest_outputs)
print("Mean forest price prediction:", forest.predict(new_X))
np.testing.assert_allclose(forest.predict(new_X), forest_outputs.mean(axis=0))

# 4. Write the outer boosting loop directly. Train it separately from the models above.
initial_value = y.mean()  # Calculate the mean training target, 10.0.
learning_rate = 1.0      # Apply each new correction in full for this calculation.
current_prediction = np.full(y.shape, initial_value)  # [10, 10, 10, 10]
trees = []              # A list to store the trained trees.

print("Initial overall predictions:", current_prediction)
print("Initial MSE:", np.mean((y - current_prediction) ** 2))  # 21.0

# range(1, 4) provides 1, 2, and 3 in order. The final value, 4, is excluded.
for step in range(1, 4):
    residual = y - current_prediction  # Calculate each house's current residual.

    # Create a separate new tree on every iteration.
    tree = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=1,
        random_state=42,
    )
    tree.fit(X, residual)       # Train this entire new tree on the residuals.
    correction = tree.predict(X)  # Calculate four corrections, one per house.

    # Add the new tree's predictions to the existing overall predictions.
    current_prediction = current_prediction + learning_rate * correction
    trees.append(tree)         # Store the completed tree. Earlier trees are unchanged.
    mse = np.mean((y - current_prediction) ** 2)

    print(f"\nBoosting stage {step}")
    print("Residual training targets:", residual)
    print("New tree's predicted corrections:", correction)
    print("Updated overall price predictions:", current_prediction)
    print("Overall MSE:", mse)    # 5.0, 1.0, and 0.0 across the three stages.

np.testing.assert_allclose(current_prediction, y)

# 5. Inspect a trained tree's structure. Index 0 refers to the first element.
first_tree = trees[0]
print("First tree's actual depth:", first_tree.get_depth())       # 1
print("First tree's node count:", first_tree.tree_.node_count)     # 3
print("Leaf node index for each house:", first_tree.apply(X))      # [1, 1, 2, 2]

# 6. Predict a new house without fitting again or calculating its actual residual.
new_prediction = np.full(new_X.shape[0], initial_value)  # Starting value [10] for one house.
for tree in trees:
    new_prediction = new_prediction + learning_rate * tree.predict(new_X)

print("\nBoosting prediction for new house E:", new_prediction)  # [13.]
np.testing.assert_allclose(new_prediction, [13.0])

# 7. Train the library's complete boosting model with the same settings.
boosting = GradientBoostingRegressor(
    loss="squared_error",  # Set the loss that the complete model will reduce.
    n_estimators=3,        # Train three correction trees in sequence.
    max_depth=1,           # Apply this depth limit to each individual tree.
    learning_rate=1.0,
    random_state=42,
)
boosting.fit(X, y)  # Supply actual prices. The library calculates residuals internally.

print("Library prediction for the new house:", boosting.predict(new_X))  # [13.]
print("Internal tree array shape:", boosting.estimators_.shape)           # (3, 1)

# staged_predict provides the accumulated overall predictions at each stage in sequence.
for stage_prediction in boosting.staged_predict(X):
    print("Library's overall predictions at this stage:", stage_prediction)

# Check that both implementations produce the same results with these data and settings.
# assert_allclose passes if values match within tolerance and raises an error otherwise.
np.testing.assert_allclose(boosting.predict(X), current_prediction)
np.testing.assert_allclose(boosting.predict(new_X), new_prediction)
```

Both the manual boosting loop and `GradientBoostingRegressor` produce the stage predictions `[6, 6, 14, 14]`, `[4, 8, 12, 16]`, and `[5, 7, 11, 17]`. They also both predict `[13.]` for the new house E. The notation `13.` is a display of the floating-point value 13.0.

With this code's settings, the forest uses three trees that predict 17, 17, and 9 for E. Its final output is $(17+17+9)/3\approx14.3333$. This differs from boosting's prediction of 13, but E's actual price is unavailable, so these two numbers alone cannot tell us which model is more accurate.

The example was run with <strong>scikit-learn 1.8.0 and NumPy 2.3.5</strong>, confirming that the manual boosting loop and the library produce matching results. The manual loop exposes the sequence of operations for squared-error regression; it does not reproduce all of scikit-learn's internal implementation. The official documentation was checked on October 18, 2026.

[^tree]: scikit-learn, [Decision Trees](https://scikit-learn.org/stable/modules/tree.html).
[^regressor]: scikit-learn, [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).
[^ensemble]: scikit-learn, [Ensembles: gradient boosting, random forests, bagging](https://scikit-learn.org/stable/modules/ensemble.html).
[^bagging]: scikit-learn, [BaggingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html).
[^bagging-classifier]: scikit-learn, [BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html).
[^forest]: scikit-learn, [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
[^forest-classifier]: scikit-learn, [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
[^boosting]: scikit-learn, [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).
[^boosting-classifier]: scikit-learn, [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).
[^pitfalls]: scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).
[^numpy]: NumPy, [Array creation routines](https://numpy.org/doc/stable/reference/routines.array-creation.html), [numpy.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html).
