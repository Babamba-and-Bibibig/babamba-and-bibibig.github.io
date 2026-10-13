+++
title = "How a Decision Tree Learns and Predicts: From Gini Impurity to Pruning"
description = "Connect model creation, training, prediction, and evaluation through penguin classification. Read a small tree's prediction path, calculate Gini and entropy using six samples, and interpret depth limits, pruning, and feature importance."
date = 2026-10-13T18:00:00+09:00
slug = "decision-tree-basics"

[extra]
katex = true
toc = true
styles = ["decision-tree-basics/article.css"]
+++

Suppose we want to identify a penguin's species from measurements such as bill length and body mass. Creating a model and printing its accuracy takes only a few lines. Reading those lines requires separating several questions: What did the model learn? What does accuracy compare? Does a number printed in the tree identify a class or a node?

A decision tree **learns which feature to compare with which threshold. During prediction, it follows those conditions and determines a class at the final node.** We will connect that process first, then examine the calculations used to select conditions and control tree size.

This article organizes the material studied on October 13, 2026. Its three examples use different data.

| Example | Available information | Purpose |
| --- | --- | --- |
| Penguin exercise | 204 training samples, 69 validation samples, four features, and recorded experiment tables | Training, evaluation, and configuration comparison |
| Small tree, `tiny_model` | A structure that compares `x1` with 0.5 and 1.5, plus its recorded output | Nodes, leaves, and prediction paths |
| Fictional dataset A01 | Six samples with two features and class labels | Gini, entropy, and split calculations |

The penguin tables and the small tree use results recorded during the exercise. The complete code at the end **runs independently using the A01 data provided in this article**. Its results must not be interpreted as the penguin results or as a reconstruction of `tiny_model`.

## 1. Separate inputs from targets, then training from validation

A **sample** is one observed subject. In the penguin table, one row represents one penguin. A **feature** is information supplied to the model. A **class** is one of the categories to predict, and a **label** is the recorded class assigned to an individual sample.

Labels such as 0, 1, and 2 do not mean incorrect, correct, and something else. Their meanings come from the dataset's label definitions.

We use `X` for inputs and `y` for actual targets. The suffixes `train` and `val` distinguish the roles of the data.

| Name | Whose information does it contain? | Shape in the penguin exercise |
| --- | --- | --- |
| `X_train` | Features of the training penguins | `(204, 4)` |
| `y_train` | Actual labels of those same penguins | `(204,)` |
| `X_val` | Features of the validation penguins | `(69, 4)` |
| `y_val` | Actual labels of those same validation penguins | `(69,)` |

**`shape` gives the size of each dimension.** `(204, 4)` describes a two-dimensional input with 204 rows and four columns. `(204,)` describes a one-dimensional array containing 204 labels. Its trailing comma does not indicate an omitted second dimension.

`X_train` and `y_train` are not a training set and a validation set. They are **the inputs and targets of the same training subjects**. The model partitions samples using their inputs, then checks their targets to evaluate how well the split separates classes. That is why `fit(X_train, y_train)` needs both.

The first input row must correspond to the first target. Matching lengths are not enough. Sorting the target array independently can attach another penguin's label to an input. Converting targets with `np.asarray(y_train)` does not repair an existing mismatch.

### The complete workflow

**Training** determines the values or structure needed for prediction from data. **Validation** evaluates candidates to help choose a configuration. A **test set** is held separately for final evaluation after that choice.

```text
Define the prediction target and class meanings.
    ↓
Prepare X and y, checking that each input matches its target.
    ↓
Assign separate roles to training, validation, and final test data.
    ↓
Create the model and specify its settings.
    ↓
Use fit to learn split conditions and tree structure from training data.
    ↓
Use predict to obtain a class for each validation input.
    ↓
Use accuracy_score to compare predictions with validation targets.
    ↓
Compare validation performance and the actual size of candidate trees.
    ↓
Inspect one input's path and the model's feature importance.
    ↓
After choosing the configuration, evaluate on the separate test set.
```

Validation inputs must contain the same features in the same order as training inputs. When preprocessing is involved, apply the criteria learned from training data to validation data. Learning preprocessing statistics from validation or test data lets evaluation information enter the training process.[^pitfalls]

## 2. Creating a model is not the same as training it

`DecisionTreeClassifier` is a **Python class** used to create a classification-tree object. An object is an individual instance that we actually use. A function belonging to an object is a **method**, and a value it exposes is an **attribute**. A Python class is a different concept from a prediction class such as a penguin species.

| Code | Operation | Return value |
| --- | --- | --- |
| `DecisionTreeClassifier(...)` | Create a configured model object | An untrained model |
| `model.fit(X_train, y_train)` | Build the tree from training data | The fitted model itself |
| `model.predict(X_val)` | Apply learned conditions to validation inputs | An array of predicted labels |
| `accuracy_score(y_val, val_pred)` | Compare actual and predicted labels | An accuracy number with default settings |
| `model.get_depth()` | Inspect actual maximum depth | An integer |
| `model.get_n_leaves()` | Count leaves | An integer |
| `model.feature_importances_` | Read feature importance | An array with one value per feature |

Executing `model = DecisionTreeClassifier(...)` does not complete training. Training data are used when `fit` runs. Writing `row = model.fit(...)` also stores the model in `row`, not a row of evaluation results.[^classifier]

`feature_importances_` is an attribute, so do not append `()`. Prediction with this estimator is `model.predict(X_val)`, not `model(X_val)`.

### Learning does not necessarily mean gradient descent

**Gradient descent** updates learnable numbers using the gradient of a loss function. The decision tree in this exercise does not learn its split conditions through gradient descent.

Its `fit` compares feature-and-threshold combinations and selects conditions for partitioning samples. The resulting model stores feature indices, thresholds, child-node connections, and class composition at nodes.

Saying that a tree “learns questions” does not mean that it generates sentences. It means **choosing which feature to compare with which value and which node to use next for each outcome**.[^tree]

`random_state=41` controls reproducibility of random choices during training. It helps reproduce results when the data and other conditions remain the same. It does not mean training 41 times or setting accuracy to 41. The exercises below keep it at 41 and use the default split criterion, `criterion="gini"`.

## 3. Distinguish a model's experiment result from a sample's prediction

`measure`, `show`, `candidate`, and `row` are not special scikit-learn syntax. `measure` and `show` are functions defined for the exercise; the others are variable names chosen by the programmer.

`measure(model, name)` trains and evaluates the supplied model, returning a **dictionary**. A dictionary groups values under keys such as `"metric_name": value`. `show(rows)` displays multiple results as a table.

```text
candidate or review_model
    Refers to a model object.
          ↓ measure trains and evaluates it.
row or review_row
    A dictionary describing one model experiment.
          ↓ rows.append(row) adds that result.
rows
    A list containing multiple experiment results.
          ↓ show(rows) displays the table.
One row in the result table
    Represents one model configuration, not one penguin.
```

The string `"depth=2"` is an experiment name. The actual setting comes from `max_depth=2` supplied to the model. Changing only the name to `"depth=4"` does not change the model.

Check where `measure` obtains its data. When it is defined to use `X_train` and other variables outside the function, it reads those variables. The library is not finding a dataset automatically. In this article's complete code, four data objects are passed to `compare_settings`, and the `measure` defined inside it uses those objects.

Because `measure` already calls `fit`, there is no need to fit the same model immediately before calling it. Calling `fit` again on this decision tree trains it anew rather than adding further learning to the old tree.

### The accuracy function does not need to know the model

`accuracy_score(y_val, val_pred)` takes **actual labels and predictions that have already been computed**. It does not take a model object. With default settings:[^accuracy]

$$
\text{accuracy}=\frac{\text{number of correct predictions}}{\text{number of evaluated samples}}
$$

Division converts a count into a fraction of the total. For 67 correct predictions among 69 validation samples:

$$
\frac{67}{69}\approx0.971014\approx97.10\%
$$

Comparing target and prediction arrays with `==` produces `True` or `False` at each position. When summed, `True` contributes 1 and `False` contributes 0, giving the number of correct predictions.

Training accuracy evaluates inputs already used in training. Validation accuracy evaluates the validation inputs not used by that particular `fit`. Their difference can be recorded as `gap = train_accuracy - validation_accuracy`. A gap of 0.05 means **five percentage points**, not a relative five-percent decrease.

A gap alone does not establish the cause of overfitting. Conversely, a small gap does not make a model good when both scores are low. `accuracy_score` also does not identify the model type and automatically switch to Gini or MSE. Choose the evaluation metric for the target and the evaluation objective.

## 4. A binary tree has at most two children per node, not necessarily two classes

A **node** is a component of the tree that stores a condition or class information. The starting node is the **root**. A **split node** checks a condition and selects a child. A **leaf** has no children and ends the prediction path.

A **binary tree** gives each node at most two children. In this decision tree, split nodes have two children and leaves have none. This differs from **binary classification**, which has two target classes. A binary tree can classify three or more classes.

The recorded `tiny_model` has this structure:

```text
node 0: x1 <= 0.5 ?                            depth 0
│
├─ true  → node 1: predict class 0 [leaf]        depth 1
│
└─ false → node 2: x1 <= 1.5 ?                  depth 1
              │
              ├─ true  → node 3: class 0 [leaf] depth 2
              │
              └─ false → node 4: class 1 [leaf] depth 2
```

Node 1 does not split into nodes 3 and 4. Node 1 is already a leaf; node 2 splits again. The three leaves are nodes 1, 3, and 4. Leaves do not have to occur at the same depth.

| Quantity | Value in this tree | Meaning |
| --- | --- | --- |
| Node IDs | 0, 1, 2, 3, 4 | Identifiers for nodes |
| Predicted classes | 0, 1 | Target categories |
| Leaf count | 3 | Number of terminal nodes without children |
| Actual maximum depth | 2 | Number of connections from the root to the deepest leaf |
| Training sample count | Not determined by this diagram | Number of training observations |

Different leaves can both predict class 0. **Leaf count is neither class count nor sample count.** `leaf 3` means the leaf with node ID 3, not depth 3.

### Depth starts at zero at the root

Depth counts the **connections** traversed from the root. At the root, no connection has been traversed, so depth is zero.

`max_depth=2` permits at most three levels including the root; `max_depth=3` permits four. Each additional depth can double the number of nodes, giving maximum leaf counts of `1 → 2 → 4 → 8`. For a maximum depth $d$, the leaf-count upper bound is $2^d$.

`max_depth` is an **allowed upper bound**, whereas `get_depth()` is the **actual result**. Setting a limit of four does not force a tree to reach depth four. `max_depth=None` does not mean splitting forever: depth is unrestricted, but stopping conditions such as pure nodes, the absence of a valid split, and minimum sample requirements remain.[^classifier]

The counting convention and the parameter's valid values are also distinct. `max_depth` accepts a positive integer or `None`. A root-only fitted tree can have actual depth zero, but that does not mean specifying `max_depth=0`.

## 5. Read the predicted class and the prediction path separately

Consider the input `[[1, 1]]` used with `tiny_model`. The feature checked along this path is `x1`, whose value is 1.

| Step | Condition | Result and next operation |
| --- | --- | --- |
| 1 | At node 0, `1 <= 0.5` | False: use the right child, node 2. |
| 2 | At node 2, `1 <= 1.5` | True: use the left child, node 3. |
| 3 | Node 3 is a leaf. | Finish prediction using its learned class distribution. |

“Going right” does not move the original row into another file. It means **selecting the next node to use in the calculation**. Left does not always mean class 0, nor right class 1. Direction comes from the condition; the predicted class comes from information at the reached leaf.

The final recorded line was:

```text
leaf 3, class distribution {0: 0.5, 1: 0.5}, prediction 0
```

The training samples at node 3 had equal class proportions. **These values are not 50% validation accuracy.** Proportions alone also do not reveal how many training samples reached that leaf. For a tie, this classifier chooses the tied class appearing first in `classes_`; here that produces class 0. If the actual label is 1, the prediction is wrong.[^classifier]

Conditions and input values can identify the reached leaf. Without that leaf's prediction information, however, the final class cannot be determined.

### Inspection tools return different information

| Tool | What it returns or displays |
| --- | --- |
| `export_text(model, ...)` | A string describing the whole tree's conditions and leaf predictions |
| `model.predict(X)` | A predicted class for each input |
| `model.apply(X)` | The ID of the leaf reached by each input |
| `model.decision_path(X)` | A matrix indicating which nodes each input uses |
| A custom `path_for(...)` | A readable record of conditions and leaf information for one input |

Default `export_text` output is not a trace for one input or a table of training sample counts. An `[3]` returned by `apply` does not mean predicted class 3.[^structure]

`path_for` is a custom explanation function, not a standard function for finding a filesystem path. It receives `feature_names` to attach readable names to the stored feature indices. These names must follow training-column order. Supplying display names does not change the fitted split conditions.

Why does a false test of `1 <= 0.5` appear as `1 > 0.5` in a trace? The expression `relation = "<=" if value <= threshold else ">"` prints the **branch actually taken**. The model did not change its threshold.

This explanation assumes no missing values. `NaN` is a special value that can represent missing data; a simple `<=` comparison should not be assumed to describe the model's actual missing-value routing. Rounded display values also require care near thresholds. The complete code's `path_for` accepts only one finite-valued input row and verifies that its calculated leaf agrees with `apply`.

### Check position and dimensionality when selecting a row

When the full prediction array `val_pred` already exists, select the same position to inspect one subject. Re-prediction is unnecessary as long as the model has not been refitted and validation rows have not been reordered.

| Selection | Return shape and type | Meaning |
| --- | --- | --- |
| `X_val.iloc[1]` | Series, `(4,)` | Values from the second row |
| `X_val.iloc[1:2]` | DataFrame, `(1, 4)` | The second row retained as two-dimensional input |
| `val_pred[1:2]` | NumPy array, `(1,)` | One corresponding prediction retained as an array |

`.iloc` selects by integer **position**. Positions start at zero, and slices include the start but exclude the end. Thus `[1:2]` selects the second row. In this exercise, `predict` takes a two-dimensional `(number of samples, number of features)` input, so one penguin remains shaped `(1, 4)`.[^indexing]

For a pandas Series with integer index labels, `y_val[1]` selects index label 1. Use `y_val.iloc[1]` for the second position. For a NumPy array, `[1]` selects the second position. Original row identifiers and current array positions need not be equal.

### decision_path uses samples as rows and node IDs as columns

Used nodes are marked 1 and unused nodes 0. The path `0 → 2 → 3` produces:

```text
node ID       0  1  2  3  4
used         [1, 0, 1, 1, 0]
matrix shape (1, 5)
```

A matrix with many zeros is **sparse**. `decision_path` returns CSR storage, which primarily records nonzero values and their positions. For a one-sample query, `.indices` identifies the nodes recorded for that row. Do not interpret the entire `.indices` array from multiple samples as one sample's path. A sparse matrix's column-index list is not, in general, a chronological execution log.[^csr]

`.toarray()` expands the matrix into an ordinary array. This is convenient for a small example but can consume substantial memory for a large matrix. Child connections are available through `model.tree_.children_left[node_id]` and `model.tree_.children_right[node_id]`. A child ID of `-1` means no child in that direction. Writing only `children_right[0]` will not work unless that standalone variable has been defined.

## 6. Gini impurity measures how classes are mixed

To choose a split, the model needs a way to compare the resulting groups. **Gini impurity** describes whether actual labels at a node concentrate in one class or mix across classes. It is not an error rate computed by comparing predictions with targets.

First define the symbols. $K$ is the number of classes; $k$ identifies a class from 0 through $K-1$. $n_k$ is the number of samples in class $k$ at the current node, and $n$ is that node's total sample count. Assume $n>0$ and no separate sample or class weights.

The class proportion $p_k$ and Gini impurity $G$ are:

$$
p_k=\frac{n_k}{n},\qquad
G=1-\sum_{k=0}^{K-1}p_k^2
$$

The summation symbol means to add the specified terms. **There is one term per class, not one term per sample.** Two classes give two terms; three classes give three. The formula is not restricted to binary classification.[^tree]

### Why square the proportions instead of just adding them?

Class proportions always sum to one. Their sum alone cannot distinguish a distribution dominated by one class from an even mixture.

Squaring multiplies a number by itself. When $0<p<1$, $p^2<p$, so the probability itself becomes smaller. For example, $9/10=90/100$, while its square is $81/100$. Both numerator and denominator change.

However, squaring increases the **relative ratio** of a larger positive number to a smaller one. If $a>b>0$:

$$
\frac{a^2}{b^2}=\left(\frac{a}{b}\right)^2>\frac{a}{b}
$$

Because $a/b>1$, squaring that ratio makes it larger. This does not claim that the absolute difference between two values always grows.

Compare distributions whose proportions sum to one:

| Class proportions | Sum of squared proportions | Gini after subtracting from one |
| --- | --- | --- |
| `[0.9, 0.1]` | $0.81+0.01=0.82$ | $0.18$ |
| `[0.5, 0.5]` | $0.25+0.25=0.50$ | $0.50$ |

The more concentrated distribution has the larger sum of squares. Subtracting it from one makes the impurity smaller when there is less mixing. A pure node has Gini zero. For a fixed $K$, the maximum occurs at equal class proportions and equals $1-K(1/K)^2=1-1/K$. **0.5 is the maximum for two classes, not for every classification problem.**

**MSE, mean squared error**, averages squared prediction errors. Both MSE and Gini use properties of squaring, but square different quantities: prediction errors versus class proportions. Squaring is also not the only possible basis for an impurity measure. Entropy is another criterion for the same purpose.

### Calculate Gini using six fictional samples

In A01, `x1` and `x2` are numeric feature names; labels 0 and 1 identify two classes. No physical units or penguin-species meanings are assigned to this fictional table.

| Sample | x1 | x2 | Actual class |
| --- | --- | --- | --- |
| 1 | 1 | 1 | 0 |
| 2 | 2 | 2 | 0 |
| 3 | 3 | 4 | 0 |
| 4 | 4 | 3 | 1 |
| 5 | 5 | 5 | 1 |
| 6 | 6 | 6 | 1 |

The condition `x2 <= 2.5` puts samples 1 and 2 on the left, and samples 3, 4, 5, and 6 on the right. Listing counts in class-0, class-1 order gives `[2, 0]` on the left and `[1, 3]` on the right.

The total right-hand count, 4, is different information from class counts `[1, 3]`. Calculating Gini requires the class composition.

$$
G_R
=1-\left[\left(\frac14\right)^2+\left(\frac34\right)^2\right]
=1-\left(\frac1{16}+\frac9{16}\right)
=\frac6{16}
=0.375
$$

The fractions $1/4$ and $3/4$ describe classes 0 and 1, not probabilities defined as correct and incorrect. The three class-1 samples are already represented by the numerator of $3/4$. Adding $(3/4)^2$ three times would count the same class proportion repeatedly.

Both left-hand samples belong to class 0:

$$
G_L=1-\left[\left(\frac22\right)^2+\left(\frac02\right)^2\right]=0
$$

## 7. Weight each child's impurity by its sample count

If the children have different sizes, adding their two Gini values and dividing by two gives a two-sample child the same influence as a four-sample child. A **weighted average** accounts for how much each value should contribute.

Let $n_L$ and $n_R$ be the child sample counts and $n=n_L+n_R$ the parent's count. Let $G_L$ and $G_R$ be the child impurities. Counting each child's impurity once for each sample in that child and dividing by the total gives:

$$
\text{weighted Gini}
=\frac{n_LG_L+n_RG_R}{n}
=\frac{n_L}{n}G_L+\frac{n_R}{n}G_R
$$

Division calculates each child's share of the parent. Multiplication applies that share to the child's impurity. Addition combines the two contributions.

For the two left-hand and four right-hand A01 samples:

$$
\text{weighted Gini}
=\frac26\times0+\frac46\times0.375
=0.25
$$

The calculation order is **class counts → class proportions → child Gini values → child sample shares → weighted average**. Substituting $(1/16+9/16)$ directly into the right-hand term uses the sum of squares, not Gini. First calculate $1-(1/16+9/16)$.

For comparison, `x2 <= 3.5` produces class counts `[2, 1]` and `[1, 2]`. Both children have Gini $1-(4/9+1/9)=4/9$. With three samples per child, their weighted average is also $4/9\approx0.4444$.

For the same parent, **`x2 <= 2.5`, scoring 0.25, reduces mixing more than the candidate scoring 0.4444**. This compares training-time split criteria, not validation accuracy.

## 8. Compare distinct partitions instead of infinitely many real thresholds

For a continuous number, we could keep inventing thresholds such as 2.1, 2.11, and 2.111. A decision tree does not test every real number. It compares **ways to partition the current training samples differently**.

Assume a numeric feature, no missing values, and the default `splitter="best"`. Sorting A01's `x2` gives `[1, 2, 3, 4, 5, 6]`. Both `x2 <= 2.1` and `x2 <= 2.9` put values 1 and 2 on the left, producing identical training partitions and impurity.

For distinct adjacent values $a<b$, thresholds satisfying $a\le t<b$ in the condition $x\le t$ yield the same training partition. Their midpoint $(a+b)/2$ is commonly used as a representative. With $m$ distinct values, there are at most $m-1$ numeric boundaries producing nonempty children. The width of the numeric range does not determine that count.

But **identical training partitions do not guarantee identical predictions for every future input**. A new value between 2.1 and 2.9 can go to different children depending on the chosen threshold.

### Targets must follow their samples during sorting

Sorting A01 by `x2` changes sample order to `1, 2, 4, 3, 5, 6`. Labels must therefore become `[0, 0, 1, 0, 1, 1]`. Sorting only the feature while keeping the original target order would calculate a different dataset.

Start with left counts `[0, 0]` and right counts `[3, 3]`. As the boundary moves right, update only the class count of the newly transferred sample:

| Threshold | Left class counts | Right class counts | Weighted Gini |
| --- | --- | --- | --- |
| 1.5 | `[1, 0]` | `[2, 3]` | 0.4000 |
| 2.5 | `[2, 0]` | `[1, 3]` | 0.2500 |
| 3.5 | `[2, 1]` | `[1, 2]` | 0.4444 |
| 4.5 | `[3, 1]` | `[0, 2]` | 0.2500 |
| 5.5 | `[3, 2]` | `[0, 1]` | 0.4000 |

This avoids recounting all sample labels from the beginning for every candidate. The cost of evaluating class statistics does not disappear. The implementation also applies additional candidate restrictions and optimizations; sorting is not necessarily performed only once for the entire training procedure.[^implementation]

This table compares candidates for **`x2` alone**. The full model compares the feature-and-threshold combinations it considers. A01 also offers `x1 <= 3.5`, which produces `[3, 0]` and `[0, 3]`, giving weighted Gini zero. The best score of 0.25 for `x2` therefore is not the best score across all features.

After selecting a condition at the current node, the method repeats in the children. It does not compare all possible future tree structures and guarantee a global optimum.[^tree]

## 9. Entropy is not a second evaluation of Gini

**Entropy** is another measure of uncertainty in class composition. `criterion="gini"` uses Gini; `criterion="entropy"` compares splits using entropy reduction. The model is not first calculating Gini and then grading it with entropy.

A **logarithm** is the inverse of exponentiation. $\log_2(p)=r$ means $2^r=p$. Define the **information content** of an outcome with probability $p$ as:

$$
I(p)=-\log_2(p),\qquad 0<p\le1
$$

A certain outcome has $p=1$ and information content zero. Rarer outcomes have larger information content. For $0<p<1$, the logarithm is negative, so the minus sign makes the quantity nonnegative. When independent outcomes have a joint probability expressed as a product, logarithms make their information content additive:

$$
-\log_2(pq)=-\log_2(p)-\log_2(q)
$$

Entropy $H$ is the **weighted average** of class information content using class proportions. Retaining the same $p_k$ and class indices:[^entropy]

$$
H=\sum_{k=0}^{K-1}p_kI(p_k)
=-\sum_{k=0}^{K-1}p_k\log_2(p_k)
$$

The logarithm converts probability into information content. Multiplying by proportions and adding computes its average. A term with $p_k=0$ contributes zero; do not evaluate `log2(0)` directly. Entropy is not merely “Gini with the square replaced by a logarithm.”

### Information gain is a before-and-after difference

**Information gain** subtracts the children's weighted entropy from the parent's entropy. Greater uncertainty reduction gives greater gain.

A01's parent counts are `[3, 3]`. Both proportions are $1/2$, and $\log_2(1/2)=-1$, so parent entropy is 1. For `x2 <= 2.5`, the left child `[2, 0]` has entropy zero, while `[1, 3]` on the right gives:

$$
H_R=-\left[\frac14\log_2\left(\frac14\right)+\frac34\log_2\left(\frac34\right)\right]
\approx0.811278
$$

Weighted entropy is $(2/6)\times0+(4/6)\times0.811278\approx0.540852$. Information gain is $1-0.540852\approx0.459148$.

| Condition | Weighted Gini | Weighted entropy | Information gain |
| --- | --- | --- | --- |
| `x2 <= 3.5` | 0.4444 | 0.9183 | 0.0817 |
| `x2 <= 2.5` | 0.2500 | 0.5409 | 0.4591 |

Lower post-split Gini and entropy are better; higher information gain is better. Both criteria prefer the same candidate here, but need not always choose the same split on other data.

## 10. Depth limits and pruning reduce trees in different ways

A **hyperparameter** specifies a training rule or constraint. `max_depth` and `ccp_alpha` are hyperparameters. The actual split features, thresholds, connections, and class distributions are determined from data by `fit`.

`max_depth` limits how deep a tree may grow. **Pruning** removes the structure below a split node and makes that node a leaf. It does not delete the original training records.

`ccp_alpha` controls **cost-complexity pruning**. This method assigns a cost to leaf count as well as training impurity. Let $T$ denote a candidate tree, $R(T)$ its leaf impurities weighted by their shares of the entire training set, and $L(T)$ its leaf count. Let $\alpha$ be the complexity cost per leaf.

$$
\text{cost}(T)=R(T)+\alpha L(T)
$$

Charging each leaf the same cost $\alpha$ produces the multiplication $\alpha L(T)$. Adding it to training impurity gives the quantity used to compare trees. Starting from the same tree, increasing $\alpha$ penalizes larger leaf counts more strongly and favors simpler trees.[^pruning]

The default `ccp_alpha=0` does not perform this pruning. `0.01` does not mean removing 1% of branches, lowering accuracy by 1%, or setting a learning rate. Structure changes at particular boundaries, so a small alpha increase may leave the result unchanged.

**`fit` does not inspect validation accuracy to decide which branches to remove.** Pruning uses the cost criterion. Comparing alphas on validation data is a separate experiment outside that fitting operation.

## 11. Read what an experiment gains and loses

In a loop, `[1, 2, 4, None]` is the full candidate list, while `depth` is the single candidate used in the current iteration. Pass `max_depth=depth`, not the whole list.

Create a fresh model each iteration, obtain its result from `measure`, and store it with `rows.append(row)`. `append` changes the existing list and returns `None`, so do not write `rows = rows.append(row)`. The number of elements in the result list counts experiments, not validation samples.

### The recorded penguin depth experiment

The following results used the same training and validation data and `random_state=41`:

| Depth limit | Actual depth | Leaves | Validation accuracy |
| --- | --- | --- | --- |
| 1 | 1 | 2 | 0.782609 |
| 2 | 2 | 4 | 0.913043 |
| 4 | 4 | 9 | 0.971014 |
| `None` | 4 | 9 | 0.971014 |

Among the tested candidates, depth 4 and `None` achieved the highest validation accuracy. Removing the depth limit still produced actual depth 4. This means **allowing more depth did not make this fitted tree deeper**, not that depths greater than four are useless for every dataset.

In unweighted, single-target classification, exactly 100% training accuracy means that every leaf correctly assigns one class to all training samples reaching it. Those leaves therefore have no class mixture. But the model did not stop because it decided that roughly 97% validation accuracy was sufficient. Stopping is determined from the node's training samples and training constraints.

### The recorded penguin pruning experiment

| Alpha | Actual depth | Leaves | Training accuracy | Validation accuracy |
| --- | --- | --- | --- | --- |
| 0 | 4 | 9 | 1.000000 | 0.971014 |
| 0.005 | 4 | 9 | 1.000000 | 0.971014 |
| 0.01 | 4 | 6 | 0.980392 | 0.927536 |
| 0.02 | 2 | 4 | 0.965686 | 0.913043 |
| 0.05 | 2 | 3 | 0.955882 | 0.913043 |

Changing alpha from 0 to 0.01 reduced the number of leaves from nine to six. With 69 validation samples, we can translate the accuracy values back into counts:

$$
0.971014\approx\frac{67}{69},\qquad
0.927536\approx\frac{64}{69}
$$

**The model lost three leaves, but also made three fewer correct validation predictions.** The accuracy difference is $(3/69)\times100\approx4.35$ percentage points. Rather than simply calling pruning a failure, describe the observed tradeoff: a simpler tree with lower accuracy on this validation set.

Alphas 0 and 0.005 tied for the highest validation accuracy in this table. Zero was not the unique winner. Alphas 0.02 and 0.05 also tied on validation accuracy, while 0.05 used fewer leaves, providing a reason to prefer simplicity at the same observed performance.

Equal depth, leaf count, and accuracy do not prove that every split condition is identical. Summary statistics and the complete tree structure are different information.

### State the scope of configuration choices

**Overfitting** occurs when detailed rules fitted to training data do not work well on new data. This is why 100% training accuracy does not guarantee 100% accuracy on future inputs.

With 69 validation samples, one additional correct prediction changes accuracy by approximately $(1/69)\times100=1.45$ percentage points. Do not treat a single observed score as a fixed measurement of ability.

**Cross-validation** repeatedly evaluates a model while changing which data are used for training and evaluation according to a chosen splitting scheme. `GridSearchCV` compares combinations of candidate settings in this way. Time-ordered observations or repeated records of the same subject require splits appropriate to actual use. Repeatedly adjusting settings after looking at test scores uses the test set for selection too.[^selection]

`cost_complexity_pruning_path(X_train, y_train)` calculates pruning boundaries for the training data and settings. It returns a **Bunch object**, which groups named results; `ccp_alphas` is its array of candidate alpha values. It does not directly return the alpha with the highest validation accuracy. Candidates still need separate training and evaluation, and changing settings such as depth limits can change the candidate boundaries.[^pruning]

The scope of this experiment's conclusion is:

> Prioritizing current validation accuracy, `ccp_alpha=0` or `0.005` is a leading choice among the tested candidates. Alpha `0.01` reduced the tree to six leaves but lowered validation accuracy to about 92.75%. This does not establish an optimum over all possible alphas; the result should also be checked with other appropriate splits.

## 12. Feature importance describes this tree's contribution, not causal strength

In this exercise, `feature_importances_` reports **importance based on impurity reduction**. It weights each split's reduction by the share of training samples reaching that node, adds the contributions of splits using the same feature, and normalizes by total feature contribution.

When there is no impurity reduction, importance values can all be zero. Do not assume that their sum must always be one. These values are neither validation accuracies nor causal effects.[^importance]

For example, if `flipper_length_mm` has the largest importance, read it as: “Splits using flipper length contributed the most to weighted impurity reduction in this fitted tree.” The suffix `mm` indicates millimeters.

| Interpretation | Why it is not supported |
| --- | --- |
| Importance 0.55 means this feature alone predicts 55% correctly. | This is not validation accuracy from a feature-only model. |
| Removing the feature will reduce accuracy by 55%. | No removal, retraining, and evaluation have been performed. |
| Increasing the feature value leads to class 1. | A sum of reductions does not encode the direction of prediction changes. |
| Flipper length causes the penguin species. | No intervention changing flipper length has established a causal effect. |

Inspecting conditions and input-specific predictions can show how the model responds, but that response is not automatically a real-world causal effect. This does not mean that causal analysis is impossible; it means **this importance measure alone does not establish causality**.

An importance of zero means that the feature received no impurity-reduction contribution in this model. It often means the feature was unused, but it could occur in a zero-reduction split. Inspect feature indices in `model.tree_.feature` to check actual usage.

This importance measure can favor features with many distinct values. Similar features may share contributions or concentrate them in one feature. To ask whether a feature helps validation performance, retrain without it or evaluate how the score changes when its validation values are shuffled. Neither procedure automatically proves causality.[^importance]

### Pair names with zip, then sort with sorted

`zip(feature_names, importance)` returns an **iterator** producing **tuples** from corresponding positions. A tuple groups values such as `(feature_name, importance)`; an iterator supplies items one at a time. Use `list(...)` to collect a completed list.

`zip` does not understand the meanings of names. Names must follow training-column order. In Python 3.10 or later, `strict=True` raises an error for unequal lengths. Default `zip` stops when the shorter input ends.[^python]

`sorted(pairs, key=lambda pair: pair[1], reverse=True)` returns a **new list** in descending importance order. `pair` is one tuple, `pair[0]` is its name, and `pair[1]` its importance. `lambda pair: pair[1]` is a small function extracting the sorting key; `reverse=True` requests descending order.

Sorted output is not the original feature order. A `zip` iterator is also consumed: after collecting it fully once, a second `list(...)` call on that same iterator has no items left. Printed `np.float64(...)` indicates a NumPy numeric type, not another evaluation metric.

## 13. Connect the calculations and inspection with complete Python code

The code below includes every A01 data row and all function definitions. It calculates Gini, entropy, and threshold scores, then trains an A01-specific tree and inspects its path and importance. **Accuracy calculated on the same six A01 training samples is training accuracy, not validation performance.**

The condition `x1 <= 3.5` separates the classes completely, producing actual depth 1 and two leaves. That does not contradict the upper bound `max_depth=2`. The second sample uses path `0 → 1`, not the recorded `tiny_model` path `0 → 2 → 3`.

| Custom function | Input and return value |
| --- | --- |
| `proportions_from_counts` | Class counts in; a same-length proportion array out. |
| `gini_from_counts`, `entropy_from_counts` | Class counts in; one impurity number out. |
| `path_for` | A fitted model, one numeric input row, and feature names in; a list of trace strings out. |
| `compare_settings` | Prepared training/validation inputs and targets in; a result table and collection of fitted models out. |

`path_for` is a separate implementation defined here. It reads leaf proportions through the public `predict_proba` method, which returns a `(number of samples, number of classes)` probability array. Selecting `[0]` gives the first sample's class probabilities. It does not assume that `tree_.value` contains counts without checking the version.

Defining `compare_settings` does not run training. Once actual penguin data are prepared, the commented call at the end can run the experiments on those data. Since the original penguin rows and split were not supplied, the A01 results are not claimed to reproduce the penguin tables.

<details class="dt-code">
<summary>Expand the complete Python code</summary>

```python
# Python 3.10+. Requires numpy, pandas, and scikit-learn.
# Runs calculations, training, and path inspection on six fictional A01 samples.
# Does not reproduce the recorded penguin scores or the original tiny_model.
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text


def proportions_from_counts(counts):
    """Class counts to a one-dimensional proportion array of the same length."""
    counts = np.asarray(counts, dtype=float)  # Preserve order; use floating point.
    # ndim counts dimensions; isfinite excludes NaN and infinity.
    if counts.ndim != 1 or not np.all(np.isfinite(counts)):
        raise ValueError("Class counts must be a finite one-dimensional sequence.")
    total = counts.sum()
    if np.any(counts < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("Counts must be nonnegative with a finite positive sum.")
    return counts / total


def gini_from_counts(counts) -> float:
    """Square each class proportion once and subtract their sum from one."""
    proportions = proportions_from_counts(counts)
    return float(1 - np.sum(proportions**2))


def entropy_from_counts(counts) -> float:
    """Average information content using class proportions and base-2 logs."""
    proportions = proportions_from_counts(counts)
    positive = proportions[proportions > 0]  # Do not evaluate log2(0).
    return float(-np.sum(positive * np.log2(positive)))


def path_for(model, one_row, *, feature_names):
    """Return a list of readable path strings for one numeric input row."""
    # Arguments after * must be named, as in feature_names=...
    # This implementation supports single-target classification and finite inputs.
    prediction = model.predict(one_row)[0]  # Also validates model/input compatibility.
    if model.n_outputs_ != 1:
        raise ValueError("Only single-target classification is supported.")
    # Match the tree's internal input type: float32 is a 32-bit floating-point type.
    values = np.asarray(one_row, dtype=np.float32)
    if values.shape != (1, model.n_features_in_):
        raise ValueError("Supply one row with the same features used in training.")
    if not np.all(np.isfinite(values)):
        raise ValueError("This explanation function does not accept NaN or infinity.")
    if len(feature_names) != model.n_features_in_:
        raise ValueError("Feature-name count does not match the trained model.")

    node_id = 0  # Root node ID, not the current depth.
    trace = []
    tree = model.tree_
    while tree.children_left[node_id] != -1:  # -1 means no child.
        feature_id = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        value = float(values[0, feature_id])
        go_left = value <= threshold
        relation = "<=" if go_left else ">"
        trace.append(
            f"node {node_id}: {feature_names[feature_id]} "
            f"{value:.9g} {relation} {threshold:.9g}"
        )
        node_id = int(
            tree.children_left[node_id] if go_left
            else tree.children_right[node_id]
        )

    # predict_proba returns a two-dimensional sample-by-class probability array.
    probabilities = model.predict_proba(one_row)[0]
    # Use a public method instead of assuming tree_.value contains sample counts.
    distribution = dict(zip(
        model.classes_.tolist(), probabilities.tolist(), strict=True
    ))
    trace.append(
        f"leaf {node_id}, distribution {distribution}, prediction {prediction}"
    )
    # assert stops on a false condition. Compare manual routing with apply.
    assert node_id == int(model.apply(one_row)[0])
    return trace


def compare_settings(X_train, y_train, X_val, y_val):
    """Compare depth and alpha candidates using prepared training/validation data."""
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_val, pd.DataFrame):
        raise TypeError("Supply input features as pandas DataFrames.")
    y_train_array, y_val_array = np.asarray(y_train), np.asarray(y_val)
    if y_train_array.ndim != 1 or y_val_array.ndim != 1:
        raise ValueError("Targets must be one-dimensional, with one label per sample.")
    if len(X_train) != len(y_train_array) or len(X_train) == 0:
        raise ValueError("Check the training input and target counts.")
    if len(X_val) != len(y_val_array) or len(X_val) == 0:
        raise ValueError("Check the validation input and target counts.")
    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Training and validation columns must match in order.")
    # Shape checks do not prove subject alignment or train/validation separation.

    def measure(model, name):
        # Read data passed to the outer function; the library is not finding them.
        model.fit(X_train, y_train)
        train_accuracy = float(accuracy_score(y_train, model.predict(X_train)))
        validation_accuracy = float(accuracy_score(y_val, model.predict(X_val)))
        return {
            "setting": name,
            "depth": int(model.get_depth()),
            "leaves": int(model.get_n_leaves()),
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "gap": train_accuracy - validation_accuracy,
        }

    rows, models = [], {}
    for depth in [1, 2, 4, None]:
        name = f"depth={depth}"  # The string only names the result row.
        candidate = DecisionTreeClassifier(max_depth=depth, random_state=41)
        rows.append(measure(candidate, name))  # Do not assign append's return value.
        models[name] = candidate  # Keep fitted models separately from result rows.
    for alpha in [0, 0.005, 0.01, 0.02, 0.05]:
        name = f"alpha={alpha}"
        candidate = DecisionTreeClassifier(ccp_alpha=alpha, random_state=41)
        rows.append(measure(candidate, name))
        models[name] = candidate
    return pd.DataFrame(rows), models


# 1. Recreate the article's A01 table: one fictional sample per row.
a01 = pd.DataFrame({
    "sample_id": [1, 2, 3, 4, 5, 6],
    "x1": [1, 2, 3, 4, 5, 6],
    "x2": [1, 2, 4, 3, 5, 6],
    "label": [0, 0, 0, 1, 1, 1],
})
left_gini = gini_from_counts([2, 0])
right_gini = gini_from_counts([1, 3])
weighted_gini = (2 / 6) * left_gini + (4 / 6) * right_gini
print("left / right / weighted Gini:", left_gini, right_gini, weighted_gini)
# isclose compares floating-point results within a tolerance.
assert np.isclose(left_gini, 0)
assert np.isclose(right_gini, 0.375)
assert np.isclose(weighted_gini, 0.25)

# 2. Compare boundaries between all distinct adjacent x2 values.
# sort_values sorts entire rows, preserving the input/target correspondence.
sorted_a01 = a01.sort_values("x2")
x2_sorted = sorted_a01["x2"].to_numpy()  # Series to a one-dimensional array.
y_sorted = sorted_a01["label"].to_numpy()
left_counts, right_counts = [0, 0], [3, 3]
total = len(a01)
parent_entropy = entropy_from_counts(right_counts)
threshold_rows = []
for i in range(total - 1):
    label = int(y_sorted[i])
    left_counts[label] += 1
    right_counts[label] -= 1
    left_count, right_count = i + 1, total - i - 1
    threshold = (x2_sorted[i] + x2_sorted[i + 1]) / 2
    weighted_gini = (
        left_count / total * gini_from_counts(left_counts)
        + right_count / total * gini_from_counts(right_counts)
    )
    weighted_entropy = (
        left_count / total * entropy_from_counts(left_counts)
        + right_count / total * entropy_from_counts(right_counts)
    )
    threshold_rows.append({
        "threshold": threshold,
        # copy preserves this iteration's counts when the lists change later.
        "left_counts": left_counts.copy(),
        "right_counts": right_counts.copy(),
        "weighted_gini": weighted_gini,
        "weighted_entropy": weighted_entropy,
        "information_gain": parent_entropy - weighted_entropy,
    })
print(pd.DataFrame(threshold_rows).to_string(index=False))
assert np.allclose(
    [row["weighted_gini"] for row in threshold_rows],
    [0.4, 0.25, 4 / 9, 0.25, 0.4],
)

# 3. Fit a model for these six A01 samples. It is not tiny_model.
X = a01[["x1", "x2"]]  # Exclude sample_id and the target from input features.
y = a01["label"]
model = DecisionTreeClassifier(max_depth=2, random_state=41)
model.fit(X, y)
pred = model.predict(X)
# Evaluating the same training inputs does not produce validation accuracy.
print("training accuracy:", accuracy_score(y, pred))
print("actual depth / leaves:", model.get_depth(), model.get_n_leaves())
print(export_text(model, feature_names=list(X.columns)))

# 4. Select the second sample as a two-dimensional, one-row input.
one_row = X.iloc[1:2]
print("input / prediction shape:", one_row.shape, pred[1:2].shape)
for line in path_for(model, one_row, feature_names=list(X.columns)):
    print(line)
indicator = model.decision_path(one_row)
print("node-use matrix:", indicator.toarray())  # Expand only this small row.
print("leaf ID:", model.apply(one_row))

# 5. Attach names in training-column order before sorting importance.
pairs = list(zip(X.columns, model.feature_importances_, strict=True))
ordered_importance = sorted(pairs, key=lambda pair: pair[1], reverse=True)
print("feature importance:", ordered_importance)
for position, name in enumerate(X.columns):  # Positions begin at zero.
    print(name, "used:", bool(np.any(model.tree_.feature == position)))
probe = DecisionTreeClassifier(random_state=41)
pruning_path = probe.cost_complexity_pruning_path(X, y)
print("candidate alphas:", pruning_path.ccp_alphas)

# 6. Use this call only after preparing actual penguin training/validation data.
# table, fitted_models = compare_settings(X_train, y_train, X_val, y_val)
# print(table.to_string(index=False))
# Defining compare_settings alone does not train a penguin model.
```

</details>

The key A01 output is below. Local verification used scikit-learn 1.8.0, NumPy 2.3.5, and pandas 2.2.3. API descriptions were checked against official documentation available on the article's date. These local version numbers do not identify the reader's installation or the original penguin exercise environment.

```text
left / right / weighted Gini: 0.0 0.375 0.25
training accuracy: 1.0
actual depth / leaves: 1 2
node 0: x1 2 <= 3.5
leaf 1, distribution {0: 1.0, 1: 0.0}, prediction 0
node-use matrix: [[1 1 0]]
leaf ID: [1]
candidate alphas: [0.  0.5]
```

`candidate alphas` contains pruning boundaries, not optimal validation scores. The training accuracy of 1.0 must not be presented as performance on new data.

## 14. Calculate the same Gini values in Rust

The mathematics is the same in Python and Rust: convert class counts to proportions, subtract their squared sum from one, then weight child impurities by their sample shares. Types and operator notation differ.

Python's `/` performs floating-point division even for integer operands. Rust's division between integers is integer division, so this example uses the **64-bit floating-point type `f64`** and values such as `2.0`. The `f64` method `.powi(2)` performs the squaring written as `**2` in Python.[^rust]

`&[f64]` is a borrowed slice of floating-point values; it does not copy the whole array. `iter()` reads items one at a time, `map` transforms each item, and `sum` adds the results. `|&count|` extracts the numeric value from the referenced item.

This is an **independent Rust program verifying the same A01 Gini calculation**, not a demonstration of training a decision-tree library in Rust.

<details class="dt-code">
<summary>Expand the complete Rust code</summary>

```rust
fn gini(counts: &[f64]) -> f64 {
    // Check that the total is positive and finite.
    let total: f64 = counts.iter().sum();
    assert!(total.is_finite() && total > 0.0);
    // all checks whether every item satisfies the condition.
    assert!(counts.iter().all(|&count| count.is_finite() && count >= 0.0));

    // Square each class proportion once. powi(2) uses integer exponent 2.
    let squared_sum: f64 = counts
        .iter()
        .map(|&count| (count / total).powi(2))
        .sum();
    // With no trailing semicolon, the final expression is returned.
    1.0 - squared_sum
}

fn main() {
    // Use the same class counts as in Python.
    let left_gini = gini(&[2.0, 0.0]);
    let right_gini = gini(&[1.0, 3.0]);
    // Floating-point division preserves the shares 2/6 and 4/6.
    let weighted_gini =
        (2.0 / 6.0) * left_gini + (4.0 / 6.0) * right_gini;

    // abs computes absolute value; use a small floating-point tolerance.
    assert!((left_gini - 0.0).abs() < 1e-12);
    assert!((right_gini - 0.375).abs() < 1e-12);
    assert!((weighted_gini - 0.25).abs() < 1e-12);
    println!("Weighted Gini: {weighted_gini}"); // 0.25
}
```

</details>

## 15. Review code by asking what enters and what returns

| Common confusion | What to check |
| --- | --- |
| `fit` receives training and validation data. | It receives training inputs and those subjects' targets. |
| `fit` returns an evaluation table. | It returns the model itself; a separate function builds the table. |
| `max_depth=3` means three levels in total. | Root depth is zero, so it allows up to four levels. |
| Seven leaves mean seven classes or samples. | There are seven terminal nodes without children. |
| Left means class 0. | Direction follows the condition; prediction uses leaf information. |
| Three class-1 samples require adding $(3/4)^2$ three times. | The count is already in the proportion; add one term per class. |
| Pruning made the tree simpler, so it improved the model. | Check complexity reduction and validation performance separately. |
| `None` always marks an exercise blank. | `row=None` can mark an absent result; `max_depth=None` is a real setting. |
| `model is not None` proves training is complete. | It only checks that the value is not `None`. |
| `hasattr(model, "tree_")` proves this cell trained the model. | The attribute may remain from earlier training. |

Numbers printed together in a decision-tree exercise answer different questions. **Gini compares split conditions during learning. Validation accuracy evaluates agreement between predictions and actual labels. A path shows the conditions applied to one input. Importance summarizes contributions to impurity reduction in the current tree.**

When reading code, inspect the input type and shape, identify the operation, determine its return value, and follow the variable receiving it. Only then decide what conclusions the result supports.

## References

The penguin configuration tables and the small tree's structure and class distribution come from the October 13, 2026 study record. A01 calculations and the Python example were checked separately. API behavior and calculation criteria were checked against these official sources.

[^classifier]: scikit-learn, [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Creation, fitting, prediction, settings, ties, and inspection methods.
[^tree]: scikit-learn, [Decision Trees](https://scikit-learn.org/stable/modules/tree.html). Tree learning, split criteria, Gini, and entropy.
[^accuracy]: scikit-learn, [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html). Inputs and default return value.
[^structure]: scikit-learn, [Understanding the decision tree structure](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html). Nodes, leaves, `apply`, and `decision_path`.
[^indexing]: pandas, [Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html). Position-based and label-based selection.
[^csr]: SciPy, [csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html). Row-oriented sparse storage.
[^implementation]: scikit-learn, [split-search implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_splitter.pyx) and [class statistics and criteria](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx). The article's loop explains the principle rather than copying the internal implementation.
[^entropy]: SciPy, [entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html). Information content and entropy definitions.
[^pruning]: scikit-learn, [Post pruning decision trees with cost complexity pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html) and [Minimal Cost-Complexity Pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning).
[^selection]: scikit-learn, [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html). Candidate configurations and cross-validation.
[^pitfalls]: scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html). Consistent preprocessing and data leakage.
[^importance]: scikit-learn, [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html). Impurity-based importance bias and interpretation limits.
[^python]: Python, [Built-in functions](https://docs.python.org/3/library/functions.html). `zip`, `sorted`, and `enumerate`.
[^rust]: Rust, [f64](https://doc.rust-lang.org/std/primitive.f64.html) and [Iterator](https://doc.rust-lang.org/std/iter/trait.Iterator.html). Floating-point powers and iterator operations.
