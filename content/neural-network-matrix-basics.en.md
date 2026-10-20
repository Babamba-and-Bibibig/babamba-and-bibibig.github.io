+++
title = "Understanding Neural Networks Through Matrix Operations: From Neurons to Softmax and Cross-Entropy"
description = "Calculate a small neural network that tells whether two inputs are the same or different. Follow how an input row and a weight column produce one output value, then explore ReLU, hidden representations, broadcasting, softmax, and cross-entropy with diagrams and NumPy code."
date = 2026-10-20T19:37:00+09:00
slug = "neural-network-matrix-basics"

[extra]
katex = true
toc = true
styles = ["neural-network-matrix-basics/article.css"]
scripts = ["neural-network-matrix-basics/embed.js"]
+++

Suppose we want to tell whether two devices are in the same state. We represent each device as `0` when it is off and `1` when it is on. The states match in `[0, 0]` and `[1, 1]`. They differ in `[0, 1]` and `[1, 0]`.

A conditional statement could handle this rule. Here, having only four possible inputs lets us <strong>follow every calculation a neural network makes to turn an input into a prediction.</strong> There are few enough numbers to calculate every matrix entry by hand.

A neural network connects several stages of multiplying inputs by weights and adding the results. Between these stages, activation functions change how the values are related. In this article, a person chooses the weights and biases in advance. The experiment therefore performs a <strong>forward pass: calculating the results with the current settings.</strong> It does not train the model by updating its weights. We will explain what training adds near the end.

## 1. What does processing four samples produce?

One input case is a <strong>sample</strong>, and an input item that describes the case is a <strong>feature</strong>. Let `x1` be the first device's state and `x2` the second device's state. In this problem, one row containing both states is one sample.

A category we want to predict is a <strong>class</strong>. We assign class numbers as `0=same` and `1=different`. The true label `y` in the following table is an answer we already know for each input, rather than a value calculated by the model.

| Sample | Array row index | First device `x1` | Second device `x2` | True label `y` |
| --- | ---: | ---: | ---: | --- |
| First | 0 | 0 | 0 | 0: same |
| Second | 1 | 0 | 1 | 1: different |
| Third | 2 | 1 | 0 | 1: different |
| Fourth | 3 | 1 | 1 | 0: same |

This rule, which returns 1 only when the two values differ, is called <strong>XOR</strong>. A number identifying a position in an array is an <strong>index</strong>. Python and NumPy indices start at 0, so the second sample has row index 1.[^indexing]

<strong>An array is a data structure that stores values in an order. A matrix arranges numbers into rows and columns.</strong> Each value stored in an array is an <strong>element</strong>. <strong>NumPy is a collection of tools for array calculations in Python.</strong> We will represent matrices with two-dimensional NumPy arrays. The array containing the inputs is named `X`, and the array containing the true labels is named `y`.

<div class="nn-math">
$$
X=\begin{bmatrix}0&0\\0&1\\1&0\\1&1\end{bmatrix}
\qquad
y=\begin{bmatrix}0&1&1&0\end{bmatrix}
$$
</div>

Writing `y` horizontally above shows the order of the four labels. In the code, `y` is a <strong>one-dimensional array</strong> of length 4.

`shape` is an <strong>attribute</strong> that tells us the size of each array dimension. It does not take parentheses like a function call. `X.shape` is `(4, 2)`, meaning four samples and two features. `y.shape` is `(4,)`. The shape `(4,)` is one-dimensional, whereas `(4, 1)` is two-dimensional with four rows and one column. These are different shapes. `X.size` is 8, the total number of elements. Eight input numbers do not mean eight samples.

Our network has <strong>two input features, two hidden neurons, and two output neurons</strong>. A neuron is a calculation unit that multiplies inputs by their weights, adds the results, and applies a bias and other operations. A layer groups these calculation units into a stage. A hidden layer is an intermediate stage between the input and the final output.

| Stage | Variable name | Values calculated for one sample | Shape for all four samples |
| --- | --- | --- | --- |
| Input | `X` | Two original states | `(4, 2)` |
| Hidden weighted sums and biases | `Z1` | One value from each of two hidden neurons | `(4, 2)` |
| Hidden activation function | `H` | Two newly calculated features | `(4, 2)` |
| Output layer | `S` | Scores for two classes | `(4, 2)` |
| Prediction selection | `pred` | One predicted class index | `(4,)` |
| Probability calculation | `probabilities` | Probabilities for two classes | `(4, 2)` |
| Loss calculation | `losses` | One loss compared with the true label | `(4,)` |
| Mean loss | `mean_ce` | Average of the four samples' losses | One number |

<strong>A hidden layer with two neurons produces eight values when it processes four samples.</strong> The same two neurons are applied to each of the four samples. The number of input features, neurons, and classes are different quantities. We have simply chosen all three counts to be 2 in this example.

In the following flow, ReLU replaces negative values with 0, and softmax converts class scores into probabilities. Cross-entropy is a loss that evaluates the probability assigned to the true class. We will work through each calculation in order.

<figure class="nn-figure">
<a href="/neural-network-matrix-basics/workflow-en.svg" target="_blank" rel="noopener"><img src="/neural-network-matrix-basics/workflow-en.svg" alt="Input X produces hidden weighted sums Z1, ReLU outputs H, and output scores S. The path then branches into choosing the highest-scoring class and calculating softmax probabilities. The true-class probability is used to calculate cross-entropy."></a>
<figcaption>Follow both the role of each value and the size of the full array. Select the diagram to open a larger version.</figcaption>
</figure>

## 2. One neuron has one weight for each input

A <strong>weight</strong> is a number multiplied by an input. A <strong>weighted sum</strong> adds the results of multiplying each input by its weight. A <strong>bias</strong> is an extra number added to that sum. For a neuron with two inputs, the calculation is as follows. `w1` and `w2` are the input weights, `b` is the neuron's bias, and `z` is the result.[^dense]

<div class="nn-math">
$$z=x_1w_1+x_2w_2+b$$
</div>

With a positive weight, a positive input contributes to increasing the sum. With a negative weight, a positive input contributes to decreasing the sum. We add the bias at the end without multiplying it by either input.

Let us choose the following calculations for our two hidden neurons. Here, `z1` and `z2` are the values calculated by the first and second hidden neurons.

<div class="nn-math">
$$
\begin{aligned}
z_1&=x_1\times1+x_2\times(-1)+0\\
z_2&=x_1\times(-1)+x_2\times1+0
\end{aligned}
$$
</div>

Both neurons <strong>receive the same original inputs, `x1` and `x2`, but use different weights.</strong> They do not run in a sequence where the second neuron receives the first neuron's result. A layer in which every input from the previous stage connects to every neuron in the current stage is called a <strong>fully connected layer</strong>.[^dense]

`W1` collects the weights for these two calculations into one matrix. Because this article calculates `input @ weights`, <strong>one column of the weight matrix holds all the weights for one neuron.</strong>

| Input row in `W1` | First hidden neuron: column 0 | Second hidden neuron: column 1 |
| --- | ---: | ---: |
| Weights multiplied by `x1`: row 0 | 1 | −1 |
| Weights multiplied by `x2`: row 1 | −1 | 1 |

<div class="nn-math">
$$W_1=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}\qquad b_1=[0,0]$$
</div>

The values `[1, -1]` in column 0 of `W1` are the first neuron's two weights. The values `[1, -1]` in row 0 are the two weights used when the first input, `x1`, goes to the two neurons. Although the numbers happen to match in this example, the row and the column have different roles.

In `W1[k, j]`, `k` identifies the input feature and `j` identifies the destination neuron. Using `:` to select all rows, `W1[:, j]` retrieves all the weights for one neuron. Conversely, `W1[k, :]` retrieves the weights used to send one input to each neuron.[^indexing]

## 3. Matrix multiplication uses one input row and one weight column to produce one entry

In NumPy, <strong>`@` is the matrix multiplication operator</strong>. `X @ W1` collects the calculations that multiply each sample's inputs by each neuron's weights and add the products.[^matmul]

Start with the second sample, `[0, 1]`. The first hidden neuron uses weights `[1, -1]`, and the second uses `[-1, 1]`.

<div class="nn-math">
$$
\begin{aligned}
\text{First neuron}&:\quad0\times1+1\times(-1)=-1\\
\text{Second neuron}&:\quad0\times(-1)+1\times1=1
\end{aligned}
$$
</div>

The resulting row for the second sample is therefore `[-1, 1]`. <strong>The row index selected from the input becomes the result's row index. The column index selected from the weights becomes the result's column index.</strong> Each neuron's bias is added to its sum. Both biases are currently 0, so the values stay the same.

<figure class="nn-figure">
<a href="/neural-network-matrix-basics/matmul-en.svg" target="_blank" rel="noopener"><img src="/neural-network-matrix-basics/matmul-en.svg" alt="Select the second row [0,1] of X and the second column [-1,1] of W1. Pair the values by input position: 0 times -1 plus 1 times 1 equals 1. Store the result at row 1, column 1 of Z1." loading="lazy"></a>
<figcaption>Producing one entry adds the contributions from two inputs. Inputs from other samples do not enter this calculation.</figcaption>
</figure>

Applying the same weights to the remaining samples completes this table.

| Original input | First hidden neuron's calculation | Second hidden neuron's calculation | Result row |
| --- | --- | --- | --- |
| `[0, 0]` | `0×1 + 0×(−1) = 0` | `0×(−1) + 0×1 = 0` | `[0, 0]` |
| `[0, 1]` | `0×1 + 1×(−1) = −1` | `0×(−1) + 1×1 = 1` | `[−1, 1]` |
| `[1, 0]` | `1×1 + 0×(−1) = 1` | `1×(−1) + 0×1 = −1` | `[1, −1]` |
| `[1, 1]` | `1×1 + 1×(−1) = 0` | `1×(−1) + 1×1 = 0` | `[0, 0]` |

### Why the inner dimensions must match

To produce each result entry, we need <strong>one matching weight for every input value</strong>. If an input row contains two numbers, a weight column must also contain two numbers. That is why the left matrix's column count must equal the right matrix's row count.

Let `N` be the sample count, `D` the number of features entering this layer, and `M` the number of neurons in this layer. The shapes then follow this rule.

<div class="nn-math">
$$\underbrace{(N,D)}_{\text{input}}\ @\ \underbrace{(D,M)}_{\text{weights}}\ \longrightarrow\ \underbrace{(N,M)}_{\text{result}}$$
</div>

The two occurrences of `D` identify the number of input-and-weight pairs multiplied and added to produce one entry. The result keeps the sample count `N` and neuron count `M`. Our current calculation is `(4, 2) @ (2, 2) → (4, 2)`. The multiplication `(1, 2) @ (1, 2)` cannot be performed: there are two input values, but each weight column contains only one value to match them.[^matmul]

Check the shape when selecting a single row or column in code. Selecting with one integer index removes that axis. Selecting with a <strong>slice</strong> written as `start:stop` keeps the axis. The stopping index of a slice is excluded.[^indexing]

| Selection code | Selected values | Actual shape |
| --- | --- | --- |
| `X[1]` | The second sample's `[0, 1]` | `(2,)` |
| `X[1:2]` | The same sample kept as one row: `[[0, 1]]` | `(1, 2)` |
| `W1[:, 1]` | The second neuron's weights: `[-1, 1]` | `(2,)` |
| `W1[:, 1:2]` | The same weights kept as one column: `[[-1], [1]]` | `(2, 1)` |

Selecting one weight column in a mathematical diagram does not make the result of `W1[:, 1]` in code automatically two-dimensional.

`*` has a different role. NumPy's `*` <strong>multiplies matching elements without adding the products.</strong> Use `@` when you need matrix multiplication. The following separate, small matrices show the difference between the two operations.

<div class="nn-math">
$$A=\begin{bmatrix}1&2\\3&4\end{bmatrix}\qquad B=\begin{bmatrix}5&6\\7&8\end{bmatrix}$$
</div>

| NumPy operation | First result entry | Full result |
| --- | --- | --- |
| `A * B` | `1×5 = 5` | `[[5, 12], [21, 32]]` |
| `A @ B` | `1×5 + 2×7 = 19` | `[[19, 22], [43, 50]]` |

### Select a result entry to inspect its calculation

Select an entry in the result matrix below to highlight the <strong>input row and weight column</strong> used to calculate it. Start with the hidden layer. After reading about the output layer, switch layers to check that the same rule applies.

<iframe class="nn-demo" data-neural-demo src="/neural-network-matrix-basics/matrix-lab.html?lang=en" title="Select a matrix multiplication result entry to trace its input row and weight column" loading="lazy" height="1300"></iframe>

<p class="nn-small"><a href="/neural-network-matrix-basics/matrix-lab.html?lang=en" target="_blank" rel="noopener">Open the matrix calculator in a larger window</a></p>

## 4. Add the biases, then apply ReLU

The full hidden-layer calculation is `Z1 = X @ W1 + b1`. The values in `b1=[0, 0]` are the first and second neurons' biases. Every sample uses these same values.

We can add them even though `X @ W1` has shape `(4, 2)` and `b1` has shape `(2,)`. NumPy uses a rule called <strong>broadcasting</strong>: when array sizes are compatible for an elementwise operation, it uses the smaller array's values at the required matching positions. Here, the same two biases are applied to all four rows.[^broadcasting]

| Values to add | Applied to the neuron in column 0 | Applied to the neuron in column 1 |
| --- | --- | --- |
| `b1`, shared by every sample | Add `b1[0]` to the first value of each row. | Add `b1[1]` to the second value of each row. |

Next, we apply an <strong>activation function</strong>. This function transforms the result of a neuron's weighted sum and bias before it goes to the next stage. Our activation function, <strong>ReLU</strong>, replaces negative inputs with 0 and leaves inputs of 0 or above unchanged.[^mlp]

<div class="nn-math">
$$\operatorname{ReLU}(z)=\max(0,z)$$
</div>

For example, `−1` becomes `0`, while `1` remains `1`. The value `0` also remains `0`. The code `H = np.maximum(0, Z1)` performs this calculation. <strong>`np.maximum` is a function that compares corresponding elements of two inputs and returns the larger value.</strong> Here, it compares each of the eight numbers in `Z1` with 0.[^maximum]

<figure class="nn-figure">
<a href="/neural-network-matrix-basics/relu-en.svg" target="_blank" rel="noopener"><img src="/neural-network-matrix-basics/relu-en.svg" alt="Apply ReLU to the four rows of Z1: [0,0],[-1,1],[1,-1],[0,0]. Only the two negative entries become 0, producing H with rows [0,0],[0,1],[1,0],[0,0]. The row and column counts stay the same." loading="lazy"></a>
<figcaption>ReLU changes individual entries. It does not combine rows or remove samples, so the shape remains `(4, 2)`.</figcaption>
</figure>

Column 0 of `H` holds the first hidden neuron's output for each sample. Column 1 holds the second hidden neuron's output. For one sample, we call these two hidden outputs `h1` and `h2`. `H` is a new array calculated from the original `X`; this code does not overwrite `X`.

## 5. The output layer turns hidden values into class scores

The hidden values `h1` and `h2` now become the output layer's inputs. We assign one output neuron to each class. The first output neuron scores whether the original two devices are in the <strong>same</strong> state. The second scores whether they are <strong>different</strong>.

Choose the following output weights and biases. `W2` means the second weight matrix. In this network, it holds the <strong>output layer's weights</strong>; its name does not mean there are two hidden layers.

<div class="nn-math">
$$W_2=\begin{bmatrix}-1&1\\-1&1\end{bmatrix}\qquad b_2=[1,0]$$
</div>

| Row of `W2` | Column 0: same score | Column 1: different score |
| --- | ---: | ---: |
| Weights multiplied by `h1` | −1 | 1 |
| Weights multiplied by `h2` | −1 | 1 |

Both output neurons receive both `h1` and `h2`. Let `s0` be the score for class 0 and `s1` the score for class 1.

<div class="nn-math">
$$
\begin{aligned}
s_0&=h_1\times(-1)+h_2\times(-1)+1\\
s_1&=h_1\times1+h_2\times1+0
\end{aligned}
$$
</div>

The second sample's hidden output was `[0, 1]`. Its same score is therefore `0×(−1) + 1×(−1) + 1 = 0`. Its different score is `0×1 + 1×1 + 0 = 1`. The result `[0, 1]` means that this sample has a higher score for different.

The code `S = H @ W2 + b2` processes all four samples together. The diagram below shows the addition of bias `[1, 0]` to each row as a separate step.

<figure class="nn-figure">
<a href="/neural-network-matrix-basics/output-bias-en.svg" target="_blank" rel="noopener"><img src="/neural-network-matrix-basics/output-bias-en.svg" alt="Add bias [1,0] to every row of the matrix product H @ W2. Adding 1 to the first class score and 0 to the second produces S=[[1,0],[0,1],[0,1],[1,0]]." loading="lazy"></a>
<figcaption>`(4, 2) + (2,) → (4, 2)`. Apply each neuron's bias to every sample. The repeated values show which calculations match; NumPy does not need to build copies of the smaller array.</figcaption>
</figure>

Output scores such as `S` are also called <strong>logits</strong>. Scores may be negative or greater than 1, and their sum does not have to be 1. The scores here happen to be 0 and 1, but that does not make them probabilities.[^ce]

Throughout the calculation, the output layer still answers <strong>whether the original two device states match</strong>. It does not switch to predicting whether the intermediate values `h1` and `h2` match.

## 6. The position of the highest score identifies the predicted class

`S.argmax(axis=1)` gives the class index with the highest score for each sample. <strong>`argmax` is an array method that returns the position of a maximum, rather than the maximum value itself.</strong> A method is a function that belongs to an object, such as an array.[^argmax]

First, consider what `axis` specifies. An <strong>axis is a dimension along which positions are distinguished in an array</strong>. With `S.shape=(4, 2)`, axis 0 distinguishes the four samples, and axis 1 distinguishes the two classes within a sample. Finding the maximum's position with `axis=1` compares the two class scores in each row and reduces them to one position.

| Sample row | Scores `[same, different]` | Highest score | Column of that score | Prediction |
| --- | --- | ---: | ---: | --- |
| 0 | `[1, 0]` | 1 | 0 | same |
| 1 | `[0, 1]` | 1 | 1 | different |
| 2 | `[0, 1]` | 1 | 1 | different |
| 3 | `[1, 0]` | 1 | 0 | same |

The result is `pred=[0, 1, 1, 0]`, with `pred.shape=(4,)`. <strong>Two output neurons do not give two final answers for one sample.</strong> They produce scores for two classes, and we select the one with the higher score.

To make the difference between a maximum and its position clearer, briefly consider scores for three classes. We choose the following values separately to compare the operations; they are not outputs from the XOR network.

| Class scores `[cat, dog, rabbit]` | `.max(axis=1)` | `.argmax(axis=1)` | Selected class |
| --- | ---: | ---: | --- |
| `[2.1, 0.4, 5.0]` | 5.0 | 2 | rabbit |
| `[1.3, 4.2, 0.7]` | 4.2 | 1 | dog |

The returned class index can be 2 even though there are only two rows. The result identifies <strong>which of the three class columns has the maximum</strong>, rather than a sample row. If values tie, NumPy's `argmax` selects the first occurrence.[^argmax]

Returning to XOR, we stored the output-column class names in the order `class_names=["same", "different"]`. The expression `class_names[pred]` retrieves the name for each index in `pred=[0, 1, 1, 0]`, producing `["same", "different", "different", "same"]`.

Assigning one output neuron to each class is the design used in this article. Binary classification can also use one output neuron with a <strong>sigmoid function, which maps one score to a value between 0 and 1</strong>. We will keep the two-score design here so that the correspondence between matrix columns and classes stays visible.

## 7. How do original coordinates differ from hidden coordinates?

When we plot the original inputs, the horizontal axis is `x1` and the vertical axis is `x2`. When we plot the hidden outputs, the axes are `h1` and `h2`. These are <strong>two graphs that represent the same samples using different kinds of values</strong>.

| Original coordinates `(x1, x2)` | Calculated hidden coordinates `(h1, h2)` | True class |
| --- | --- | --- |
| `(0, 0)` | `(0, 0)` | same |
| `(0, 1)` | `(0, 1)` | different |
| `(1, 0)` | `(1, 0)` | different |
| `(1, 1)` | `(0, 0)` | same |

The original points `(0, 0)` and `(1, 1)` both produce hidden coordinates `(0, 0)`. On the hidden-coordinate graph, <strong>two samples from the same class therefore overlap at one position.</strong> No sample has disappeared. The array `H` still contains all four rows.

<figure class="nn-figure">
<a href="/neural-network-matrix-basics/xor-spaces-en.svg" target="_blank" rel="noopener"><img src="/neural-network-matrix-basics/xor-spaces-en.svg" alt="In the original x1,x2 coordinates, the two same-class XOR points lie on a diagonal. In hidden h1,h2 coordinates, those two samples overlap at the origin, while the different-class points are at (0,1) and (1,0). The hidden boundary h1+h2=0.5 corresponds to two original-coordinate boundaries, x2=x1±0.5." loading="lazy"></a>
<figcaption>The values used for the axes have changed. All four samples were calculated with the same weights and ReLU, and their true labels remain unchanged.</figcaption>
</figure>

With the current weights, `h1` is the positive part of `x1−x2`, and `h2` is the positive part of `x2−x1`. Their sum is `|x1−x2|`. The two vertical bars mean <strong>absolute value</strong>: replace a negative number's sign with a positive sign, and leave zero and positive numbers unchanged.

For these inputs, `h1+h2=0` when the device states match and `h1+h2=1` when they differ. The output layer calculates `s0=1−(h1+h2)` and `s1=h1+h2`. The condition for the different score to be higher is therefore:

<div class="nn-math">
$$
\begin{aligned}
s_1>s_0
&\quad\Longleftrightarrow\quad h_1+h_2>1-(h_1+h_2)\\
&\quad\Longleftrightarrow\quad 2(h_1+h_2)>1\\
&\quad\Longleftrightarrow\quad h_1+h_2>0.5
\end{aligned}
$$
</div>

The symbol `⇔` means that the conditions on either side are equivalent. To obtain the second line, add `h1+h2` to both sides. To obtain the last line, divide both sides by 2. The <strong>decision boundary</strong>, where the two class scores are equal, is `h1+h2=0.5`. This threshold of 0.5 follows from the output weights and biases chosen by a person; the experiment did not learn it.

In hidden coordinates, this boundary is one straight line. In original coordinates, it becomes `|x1−x2|=0.5`, giving two boundaries: `x2=x1+0.5` and `x2=x1−0.5`. The boundary looks different depending on which coordinates express the classification rule.

### What ReLU changes in this example

Before ReLU, the two hidden values are `z1=x1−x2` and `z2=−x1+x2`. Their terms cancel when added, so their sum is always 0. After ReLU, only the negative values become 0, changing the sum.

| Input | Sum of the two values before ReLU | Sum of the two values after ReLU |
| --- | --- | --- |
| `[0, 0]` | `0+0=0` | `0+0=0` |
| `[0, 1]` | `−1+1=0` | `0+1=1` |
| `[1, 0]` | `1+(−1)=0` | `1+0=1` |
| `[1, 1]` | `0+0=0` | `0+0=0` |

The sum before ReLU is 0 because <strong>we chose two weight columns that are exact opposites in this example</strong>. This does not mean that every neural network without ReLU outputs 0.

ReLU uses different rules for negative and positive inputs. It is therefore a <strong>nonlinear function</strong>: a single straight line cannot describe the whole function. Nonlinear does not necessarily mean a smooth curve or a squared term. Connecting only weighted sums and biases can ultimately be reduced to one weighted sum and bias. Adding a nonlinear calculation such as ReLU between them makes it possible to express a wider range of relationships.[^mlp]

Intermediate features calculated by a neural network are called a <strong>representation</strong>. Adjusting the weights through training so that the network produces representations useful for prediction is <strong>representation learning</strong>. In this example, a person constructed the useful representation directly.

## 8. Softmax turns one sample's scores into probabilities

Now return to the output scores `S`. If we only need the highest-scoring class, `argmax` finishes the prediction. To see the probability assigned to each class, we calculate <strong>softmax</strong>.[^softmax]

Softmax takes the exponential of each score, then divides it by the sum of all exponentials for the same sample. <strong>`exp(s)` means `e` raised to the power `s`.</strong> The number `e` is a mathematical constant of approximately 2.718. We have `exp(0)=1` and `exp(−1)≈0.367879`. NumPy's `np.exp` function applies this calculation to every array element.[^exp]

Let `s0` and `s1` be one sample's two scores, and `p0` and `p1` its class probabilities. The calculations are:

<div class="nn-math">
$$
p_0=\frac{\exp(s_0)}{\exp(s_0)+\exp(s_1)}
\qquad
p_1=\frac{\exp(s_1)}{\exp(s_0)+\exp(s_1)}
$$
</div>

Adding the two numerators gives the same value as the denominator, so the probabilities sum to 1. This is not an average that divides by the number of classes. It calculates <strong>the share each exponential contributes to their total</strong>.

In practice, we first subtract the sample's maximum score from all its scores to avoid large exponentials. This leaves the softmax probabilities unchanged. We will explain why in the next section; first, follow the arrays as they change.

<figure class="nn-figure">
<a href="/neural-network-matrix-basics/softmax-en.svg" target="_blank" rel="noopener"><img src="/neural-network-matrix-basics/softmax-en.svg" alt="Subtract each row's maximum from its scores, apply exp element by element, then divide by that row's sum of exponentials. The second sample's scores [0,1] become [-1,0], then [0.3679,1], and finally probabilities [0.2689,0.7311]." loading="lazy"></a>
<figcaption>There is one maximum and one sum per sample. Each sample uses only the values in its own row to calculate probabilities.</figcaption>
</figure>

### ① Find one maximum for each row

Run `row_max = S.max(axis=1, keepdims=True)`. The array method `max` returns a maximum along the specified axis. Setting `axis=1` reduces each sample's two class scores to one maximum. Setting `keepdims=True` <strong>retains the reduced class axis with size 1</strong>.[^max]

| Input | Operation | Result |
| --- | --- | --- |
| `S`, shape `(4, 2)` | `.max(axis=1)` | `[1, 1, 1, 1]`, shape `(4,)` |
| The same `S` | `.max(axis=1, keepdims=True)` | `[[1], [1], [1], [1]]`, shape `(4, 1)` |

Both results store the same maxima, but their array shapes differ. All four maxima happen to be 1 here; row maxima do not always have to match. The earlier three-class score example would give `[[5.0], [4.2]]`.

### ② Subtract each row's maximum from every score in that row

`shifted = S - row_max` performs `(4, 2) - (4, 1) → (4, 2)`. For the first row, subtract its maximum of 1 from both entries in `[1, 0]` to get `[0, -1]`. For the second row, subtract its maximum of 1 from `[0, 1]` to get `[-1, 0]`.

This is also broadcasting. When adding the bias array `(2,)`, we applied <strong>two neuron-specific values to every row</strong>. Here, the array `(4, 1)` holds <strong>one sample-specific value to apply across that sample's columns</strong>.

NumPy compares shapes starting from the right. At each position, the sizes must be equal, or one of them must be 1. Missing leading dimensions are treated as size 1. With `(4, 2)` and `(4, 1)`, the rightmost sizes 2 and 1 are compatible, and the leftmost sizes both equal 4. The earlier bias array `(2,)` matches as though its shape were `(1, 2)`, allowing it to apply to all four rows. Removing `keepdims=True` makes the row maxima `(4,)`, whose rightmost size 4 does not match 2. The subtraction then fails.[^broadcasting]

If the sample count and class count happen to match, a subtraction in the wrong direction can execute without an error. Check not only whether the code runs, but also <strong>whether each row received its own reference value</strong>.

To see what happens when row maxima differ, reuse the three-class scores from the earlier `max` and `argmax` comparison. This separate example has shape `(2, 3)`. We must subtract a maximum of 5.0 from the first row and 4.2 from the second. The shape `(2, 1)` produced by `keepdims=True` preserves this correspondence.

<figure class="nn-figure">
<a href="/neural-network-matrix-basics/row-broadcast-en.svg" target="_blank" rel="noopener"><img src="/neural-network-matrix-basics/row-broadcast-en.svg" alt="Subtract the maximum 5.0 from the first three-class score row [2.1,0.4,5.0] to get [-2.9,-4.6,0]. Subtract the maximum 4.2 from the second row [1.3,4.2,0.7] to get [-2.9,0,-3.5]. Subtracting shape (2,1) from (2,3) leaves the result shape (2,3)." loading="lazy"></a>
<figcaption>Even when the row maxima differ, each reference value applies only to the three scores in its own row.</figcaption>
</figure>

### ③ Apply the exponential function to every element

Run `exp_scores = np.exp(shifted)`. The first row `[0, -1]` becomes `[1, 0.367879…]`. Only the individual values change, so the shape remains `(4, 2)`.

`exp_scores` is an intermediate result calculated from output scores. Its role differs from the neuron weights `W1` and `W2` used earlier.

### ④ Add the exponentials within each row

Run `row_sums = exp_scores.sum(axis=1, keepdims=True)`. The array method `sum` adds values. `axis=1` combines the two values on the class axis, and `keepdims=True` retains that axis with size 1. The shape changes from `(4, 2)` to `(4, 1)`.[^sum]

The first row's sum is `1 + 0.367879… = 1.367879…`. The other three rows happen to have the same sum in this example.

### ⑤ Divide each element by the sum for its own row

Run `probabilities = exp_scores / row_sums`. Expanding the first row's two calculations gives:

<div class="nn-math">
$$
\begin{aligned}
p_0&=\frac{1}{1+0.367879\ldots}\approx0.731059\\
p_1&=\frac{0.367879\ldots}{1+0.367879\ldots}\approx0.268941
\end{aligned}
$$
</div>

This is elementwise division, rather than matrix multiplication. The shapes are `(4, 2) / (4, 1) → (4, 2)`, so the probability array still contains four samples and two classes.

| Sample row | Original scores | Scores after subtracting the maximum | Exponentials | Final probabilities `[same, different]` |
| --- | --- | --- | --- | --- |
| 0 | `[1, 0]` | `[0, −1]` | `[1, 0.3679]` | `[0.7311, 0.2689]` |
| 1 | `[0, 1]` | `[−1, 0]` | `[0.3679, 1]` | `[0.2689, 0.7311]` |
| 2 | `[0, 1]` | `[−1, 0]` | `[0.3679, 1]` | `[0.2689, 0.7311]` |
| 3 | `[1, 0]` | `[0, −1]` | `[1, 0.3679]` | `[0.7311, 0.2689]` |

The table rounds values to four decimal places for readability. The calculation uses the values before rounding.

ReLU applies independently to each element. Softmax first calculates each element's exponential, then uses <strong>a total formed by all classes in the same row</strong>. Changing one class's score can therefore change the other class probabilities in that row.

Softmax preserves the ordering of scores within a row. If we only want the highest-scoring class, we can skip softmax and use `S.argmax(axis=1)`. The probabilities are model outputs: a result of `0.73` does not guarantee a real-world accuracy of 73%.

## 9. Subtracting the maximum preserves the probabilities while keeping calculations manageable

A computer's floating-point type can represent only a finite range of values. When a calculation exceeds that range, the result <strong>overflows</strong>. Applying an exponential directly to a large score can cause this problem.[^softmax]

Subtracting the maximum in each row makes the largest shifted score 0 and all others 0 or below. The largest exponential is then `exp(0)=1`. The aim is not to make the original scores as small as possible. It is to <strong>avoid large exponentials while calculating the same softmax probabilities</strong>.

We can briefly check why subtracting the same number preserves probabilities. Let `s0` and `s1` be one sample's scores, and `m` the maximum subtracted from both. A property of exponentials gives `exp(s0−m)=exp(s0)×exp(−m)`, so:

<div class="nn-math">
$$
\begin{aligned}
\frac{\exp(s_0-m)}{\exp(s_0-m)+\exp(s_1-m)}
&=\frac{\exp(s_0)\exp(-m)}{\exp(s_0)\exp(-m)+\exp(s_1)\exp(-m)}\\
&=\frac{\exp(s_0)\exp(-m)}{\exp(-m)\{\exp(s_0)+\exp(s_1)\}}\\
&=\frac{\exp(s_0)}{\exp(s_0)+\exp(s_1)}
\end{aligned}
$$
</div>

We factor out the shared `exp(−m)` in the denominator, then cancel that factor from the numerator and denominator. The same calculation applies to the second class. What stays unchanged is the <strong>softmax probability</strong>, rather than the ratio between the original scores.

In contrast, dividing scores by the same positive number changes their differences, so the probabilities generally change. We can compare the following transformations of the separate score pair `[1000, 1002]`.

| Transformation before applying the exponential | Transformed scores | Softmax probabilities |
| --- | --- | --- |
| Subtract the maximum, 1002. | `[-2, 0]` | `[0.1192, 0.8808]` |
| Divide both scores by 1000. | `[1, 1.002]` | `[0.4995, 0.5005]` |
| Map the minimum to 0 and the maximum to 1. | `[0, 1]` | `[0.2689, 0.7311]` |

The highest score is in the second position in every row, but that does not make the probabilities equal. Use the first approach when the calculation needs to preserve the original probabilities. The example file linked near the end includes code that calculates these comparisons.

## 10. Select the true class's probability when calculating loss

The probability array contains two class probabilities per sample. To calculate loss for this problem, we need <strong>the one probability assigned to each sample's true class</strong>.

For sample row `i`, the true class index is `y[i]`. The value we want is therefore `probabilities[i, y[i]]`. We call the array containing one such value per sample `correct_p`. The name means <strong>select the true class's probability</strong>, rather than “select only the samples predicted correctly.”

`np.arange` is a function that creates evenly spaced values within a specified range. Here, `np.arange(len(y))` creates integer row indices starting at 0 and stopping before `len(y)`, returning `[0, 1, 2, 3]`. `len(y)` is 4, the number of true labels.[^arange]

| Row index in `row_indices` | True class column in `y` | Position to select |
| ---: | ---: | --- |
| 0 | 0 | `probabilities[0, 0]` |
| 1 | 1 | `probabilities[1, 1]` |
| 2 | 1 | `probabilities[2, 1]` |
| 3 | 0 | `probabilities[3, 0]` |

`correct_p = probabilities[row_indices, y]` pairs entries at matching positions in the two index arrays. It selects the true-class column from the first row, then the true-class column from the second row, and so on. The result is `[0.7311, 0.7311, 0.7311, 0.7311]` with shape `(4,)`. This operation does not select every combination of the row and column indices.[^indexing]

In the current example, the model's predictions `pred` all match the true labels `y`. Even so, we must not substitute one for the other when calculating loss. <strong>If the true class is 0 but the model returns probabilities `[0.1, 0.9]`, we must select 0.1 from the true-class column.</strong> Selecting 0.9 from the model's chosen class 1 would fail to properly evaluate its confident but incorrect prediction.

## 11. Cross-entropy increases as the true-class probability decreases

A <strong>loss</strong> compares the model's output with the true answer and produces a value that training aims to reduce. In this article, each sample has one true class, and we apply no separate weights to samples or classes. Under these conditions, <strong>cross-entropy (CE)</strong> takes the natural logarithm of the true-class probability and reverses its sign.[^ce]

The natural logarithm, `ln`, reverses the exponential calculation used earlier. If `exp(a)=q`, then `ln(q)=a`. Here, `a` is the exponent, and `q` is the exponential's result. For example, `exp(0)=1`, so `ln(1)=0`. In NumPy, the function `np.log` calculates the natural logarithm element by element.[^log]

Let `q` be one sample's true-class probability and `ℓ` its loss.

<div class="nn-math">
$$\ell=-\ln(q)$$
</div>

| True-class probability `q` | Natural logarithm `ln(q)` | Loss `−ln(q)` |
| ---: | ---: | ---: |
| 1 | 0 | 0 |
| 0.9 | Approximately −0.1054 | Approximately 0.1054 |
| 0.7310586 | Approximately −0.3133 | Approximately 0.3133 |
| 0.1 | Approximately −2.3026 | Approximately 2.3026 |

A smaller true-class probability produces a larger loss. The code `losses = -np.log(correct_p)` converts four true-class probabilities into four losses. The shape stays `(4,) → (4,)`.

Next, `mean_ce = losses.mean()` averages the four losses. The array method `mean` divides the sum of the elements by their count. We do not specify an axis here, so it returns one number: the average of the whole array.[^mean]

<div class="nn-math">
$$\text{Mean CE}\approx\frac{0.3133+0.3133+0.3133+0.3133}{4}=0.3133$$
</div>

<figure class="nn-figure">
<a href="/neural-network-matrix-basics/indexing-loss-en.svg" target="_blank" rel="noopener"><img src="/neural-network-matrix-basics/indexing-loss-en.svg" alt="Use the true labels y=[0,1,1,0] to select four entries from the probability matrix. All four true-class probabilities are approximately 0.7311. Their negative natural logarithms are approximately 0.3133 each, and the mean of the four losses is approximately 0.3133." loading="lazy"></a>
<figcaption>From eight probabilities, select four true-class probabilities, calculate four losses, then calculate one mean.</figcaption>
</figure>

### Why is the loss nonzero when all four predictions are correct?

To check whether a prediction is correct, we compare <strong>the class chosen by `argmax` with the true label</strong>. The `argmax` operation itself compares only scores; it does not inspect the true label. Cross-entropy also evaluates <strong>how much probability the model assigns to that true class</strong>.

If the true class has probability 0.51 and the other class has 0.49, the class with the highest probability is correct. But the true-class probability is not 1, so `−ln(0.51)` is not 0. Each of our four samples has a true-class probability of approximately 0.7311. Their predictions are all correct, while their mean loss is approximately 0.3133.

Getting these four inputs right <strong>confirms that our small, predefined calculation matches the XOR rule</strong>. It is not a performance measurement of a model trained on a separate real-world dataset.

### Follow the second sample from its input to its loss

Collecting the values for just the second sample gives the following sequence. Its array row index is 1, and its true label `y[1]=1` was known from the start.

| Stage | Second sample's value | Meaning of this value |
| --- | --- | --- |
| `X[1]` | `[0, 1]` | The first device is off, and the second is on. |
| `Z1[1]` | `[-1, 1]` | The weighted sums and biases calculated by the two hidden neurons. |
| `H[1]` | `[0, 1]` | The hidden outputs after ReLU replaces the negative value with 0. |
| `S[1]` | `[0, 1]` | The same score is 0, and the different score is 1. |
| `probabilities[1]` | Approximately `[0.2689, 0.7311]` | The two class probabilities calculated by softmax. |
| `pred[1]` | `1` | The highest-scoring class, different, is the prediction. |
| `correct_p[1]` | Approximately `0.7311` | The selected probability of the true class, `y[1]=1`. |
| `losses[1]` | Approximately `0.3133` | The natural logarithm of the true-class probability with its sign reversed. |

The input, hidden outputs, and class scores happen to appear as `[0, 1]` at several stages, but these values have different meanings. Read each value together with the array it comes from to identify its role in the calculation.

## 12. The same matrix rules apply when we add layers

So far, the input feature count, hidden neuron count, and output neuron count have all been two. To see the same rule with different numbers, consider a separate structure with <strong>two samples, four input features, two hidden layers, and three output classes</strong>. Give the first hidden layer five neurons and the second hidden layer six.

| Stage | Incoming array | Weight matrix | Bias | Result |
| --- | --- | --- | --- | --- |
| First hidden layer | `X_demo (2, 4)` | `W1_demo (4, 5)` | `(5,)` | `H1_demo (2, 5)` |
| Second hidden layer | `H1_demo (2, 5)` | `W2_demo (5, 6)` | `(6,)` | `H2_demo (2, 6)` |
| Output layer | `H2_demo (2, 6)` | `W3_demo (6, 3)` | `(3,)` | `S_demo (2, 3)` |

The hidden layers apply ReLU after their weighted sums and biases to produce `H1_demo` and `H2_demo`. The output layer produces scores `S_demo`. The `_demo` suffix distinguishes these names from the earlier XOR variables.

<figure class="nn-figure">
<a href="/neural-network-matrix-basics/layer-shapes-en.svg" target="_blank" rel="noopener"><img src="/neural-network-matrix-basics/layer-shapes-en.svg" alt="Two samples pass from four input features to five hidden neurons, then six hidden neurons, and finally three output neurons. Weight shapes are 4x5, 5x6, and 6x3; the sample count remains 2. Layer parameter counts are 25, 36, and 21, totaling 82." loading="lazy"></a>
<figcaption>At each stage, the weight row count equals the number of incoming features, and the weight column count equals the number of neurons in that stage.</figcaption>
</figure>

### A previous output column connects to a row of the next weight matrix

The first hidden layer's third neuron stores its results in <strong>column 2</strong> of `H1_demo`. In the next layer, this value becomes an input feature, so it corresponds to <strong>row 2</strong> of `W2_demo`.

The second hidden layer's first neuron multiplies this value by `W2_demo[2, 0]` and adds the contribution to its sum. The second neuron multiplies it by `W2_demo[2, 1]`, and the sixth by `W2_demo[2, 5]`. <strong>Column 2 of the previous layer does not go only to column 2 of the next layer.</strong> In a fully connected layer, every next-layer neuron uses this input with its own weight.

We can also inspect one result entry. In the example file, the first input row is `[2, 3, 5, 7]`. The first hidden layer's third neuron has the weight column `[-0.3, 0.6, 0.5, -0.4]` and bias 0.

<div class="nn-math">
$$
\begin{aligned}
z&=2\times(-0.3)+3\times0.6+5\times0.5+7\times(-0.4)+0\\
&=-0.6+1.8+2.5-2.8=0.9
\end{aligned}
$$
</div>

Here, `z` is this neuron's value before activation. ReLU leaves the positive value 0.9 unchanged, so `H1_demo[0, 2]=0.9`. The downloadable example includes all the weights and the code for all three layers.

### Parameter count is separate from sample count

A <strong>parameter</strong> is a number that determines the model's calculation and can be adjusted through training. Here, the parameters are weights and biases. A fully connected layer with `D` incoming values and `M` neurons has `D×M` weights. Using one bias per neuron adds `M` more parameters.[^dense]

| Structure | Weights | Biases | Total |
| --- | ---: | ---: | ---: |
| XOR hidden layer: 2 → 2 | `2×2=4` | 2 | 6 |
| XOR output layer: 2 → 2 | `2×2=4` | 2 | 6 |
| **Whole XOR model** | **8** | **4** | **12** |
| Additional example, first hidden layer: 4 → 5 | `4×5=20` | 5 | 25 |
| Additional example, second hidden layer: 5 → 6 | `5×6=30` | 6 | 36 |
| Additional example, output layer: 6 → 3 | `6×3=18` | 3 | 21 |
| **Whole additional example** | **68** | **14** | **82** |

Adding samples increases the number of result rows to calculate. The parameter count stays the same because every sample uses the same weights and biases. The input `X`, intermediate result `H`, and scores `S` are not included in the parameter count.

Later layers do not necessarily have fewer weights. The second hidden layer above has 30 weights, more than the first hidden layer's 20. Multiply the number of values entering a layer by its neuron count to find the correct weight count.

## 13. Training updates the weights and biases

The calculations above moved from inputs to loss while keeping all weights and biases fixed. Actual training adds a process that <strong>updates weights and biases in a direction that reduces the loss</strong>.

| Process | What it does |
| --- | --- |
| Forward pass | Uses the current weights and biases to calculate hidden outputs, scores, and loss from the inputs. |
| Backpropagation | Works backward through the connected calculations to find how changing each parameter would change the loss. |
| Parameter update | Uses those directions and rates of change to update weights and biases. |
| Next forward pass | Calculates results and loss again using the updated parameters. |

A <strong>gradient</strong> describes how much and in which direction a result changes when a value changes slightly. Backpropagation uses the final loss and intermediate calculations to find a gradient for each parameter. Obtaining one mean loss does not mean that every weight receives an equal update.[^autograd]

Basic <strong>gradient descent</strong> subtracts a value proportional to the gradient from each parameter. Let `L` be the mean loss, `w` one weight, `dw` its gradient, and `lr` the learning rate that controls the size of an update.

<div class="nn-math">
$$dw=\frac{\partial L}{\partial w}\qquad w_{\text{new}}=w_{\text{old}}-lr\times dw$$
</div>

The expression `∂L/∂w` means the rate at which loss `L` changes as `w` changes slightly while the other parameters are held fixed. Each bias is updated in the same way using its own gradient.

ReLU itself has no trainable weights. However, changing which values pass to the next stage also affects the gradients passed backward. ReLU's slope is 1 for positive inputs and 0 for negative inputs. At exactly 0, there is no single mathematical derivative, so common implementations use 0. Consequently, not every parameter necessarily changes on every update.[^mlp]

Hidden-layer weights are also connected to the final loss. During training, changes to those weights adjust the intermediate features used for prediction. Representations like the `H` that we constructed directly can thus be adjusted using data and loss.

Settings that determine the structure and training process—such as layer count, neuron count, activation functions, and learning rate—are called <strong>hyperparameters</strong>. We choose a starting structure, train it, and compare performance on <strong>validation data</strong> that was not used directly for fitting, together with time and memory costs. We then adjust the settings. Test data reserved for final evaluation is not used for these repeated choices.[^tuning]

The input feature count alone cannot tell us the best number of neurons. More neurons may allow a wider range of relationships, but performance may not improve under the available data and training conditions. Reducing the input to too few intermediate values may also discard information needed for prediction. Choosing a structure requires both actual validation results and consideration of computational cost.

## 14. Two details to check when moving to a library

### Implementations may store weights in different directions

This article calculates `input @ W + b` and stores `W` with shape `(input count, output neuron count)`. PyTorch's `nn.Linear` class stores weights as `(output neuron count, input count)` and transposes them for the calculation. A <strong>transpose</strong> swaps the rows and columns of a two-dimensional matrix. In code, it is written as `.T`.[^linear]

| Implementation | Stored weight shape | Calculation |
| --- | --- | --- |
| This article's NumPy example | `(input count, output count)` | `input @ W + b` |
| PyTorch `nn.Linear` | `(output count, input count)` | `input @ weight.T + bias` |

The underlying calculation is the same: each neuron multiplies all incoming values by its own weights and adds the products. When moving code between implementations, first check the direction in which the library stores its weights.

### Check the inputs expected by a cross-entropy function

For explanation, we calculated softmax probabilities before taking `−log(true-class probability)`. However, very large score differences can cause tiny exponential values to be stored as 0, even after subtracting the maximum. Taking their logarithm then produces an infinite loss.

In this situation, calculate <strong>log probabilities</strong> directly from scores. A log probability is the natural logarithm of a class probability. Using our existing `shifted` and `row_sums`, we can calculate `log_probabilities = shifted - np.log(row_sums)`. This follows by turning the logarithm of division in `log(exp(shifted) / row_sums)` into subtraction and using `log(exp(shifted))=shifted`. It avoids first storing a probability as 0 and then taking its logarithm.[^logsoftmax]

In the common case where true labels are class indices, a loss function created with PyTorch's `CrossEntropyLoss` class expects <strong>scores before softmax and the true class indices</strong>. It combines log-softmax with the loss calculation internally, so we do not first convert the input scores into softmax probabilities.[^ce]

After using the direct calculations to understand what each array means, pass scores and labels according to the input requirements of the library you use.

## 15. Check the full calculation with NumPy

The following code runs from top to bottom in Python or a Jupyter Notebook with NumPy installed. The function `np.array` creates a NumPy array from supplied values, and `dtype=float` stores the numbers as floating-point values. `np.zeros(2)` creates an array containing two zeros with shape `(2,)`.

The Python statement `assert` stops execution with an `AssertionError` when its condition is false. The function `np.allclose` checks whether all array values are close while allowing small floating-point differences. `np.array_equal` checks that array shapes and elements match exactly. `np.isclose` compares two values for closeness; here it checks the single mean-loss value.[^assert]

<!-- CORE_CODE_START -->
```python
# Reading a neural network through matrix operations: XOR and additional examples.
# NumPy is the only external library required.
# A person chose these weights to make the calculations easy to inspect.
# There is no backpropagation or parameter update, so this script does not train a model.

import numpy as np  # Use the array library under the name np.

np.set_printoptions(precision=4, suppress=True)  # Only the displayed precision changes.

# np.array() creates an array from the supplied values; dtype=float stores real numbers.
# Each row is one sample. The two columns are input features x1 and x2.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)  # (4, 2): four samples and two features.
y = np.array([0, 1, 1, 0])  # (4,): the true class index for each corresponding row.
class_names = np.array(["same", "different"])  # Define the meaning of columns 0 and 1.

# W1 rows identify input features; W1 columns identify destination hidden neurons.
# The first neuron computes x1 - x2; the second computes -x1 + x2.
W1 = np.array([
    [1, -1],
    [-1, 1],
], dtype=float)  # (2, 2): two input features and two hidden neurons.
b1 = np.zeros(2)  # np.zeros() creates a zero-filled array. Two biases, shape (2,).

# @ is matrix multiplication: pair an input row with a W1 column, multiply, then sum.
# Add the same two biases to every sample row.
Z1 = X @ W1 + b1  # (4, 2) @ (2, 2) + (2,) -> (4, 2).
# np.maximum() chooses the larger value element by element. This ReLU replaces negatives.
H = np.maximum(0, Z1)  # (4, 2) -> (4, 2). The original X remains unchanged.

# Both output neurons receive both hidden values, h1 and h2.
# The first column computes the 'same' score; the second computes the 'different' score.
W2 = np.array([
    [-1, 1],
    [-1, 1],
], dtype=float)  # (2, 2): two hidden outputs and two output neurons.
b2 = np.array([1, 0], dtype=float)  # (2,): add 1 to the first score and 0 to the second.
S = H @ W2 + b2  # (4, 2). These are class scores, not probabilities.

# argmax() returns the index of the largest value, rather than the value itself.
# axis=1 compares the class scores within each sample. (4, 2) -> (4,).
pred = S.argmax(axis=1)
pred_names = class_names[pred]  # Retrieve the four names at the predicted indices.

# max() reduces two class scores to one maximum for each sample.
# keepdims=True retains the reduced class axis with size one.
row_max = S.max(axis=1, keepdims=True)  # (4, 2) -> (4, 1).
shifted = S - row_max  # (4, 2) - (4, 1) -> (4, 2). Subtract each row's maximum.
# np.exp() applies e**value to every element. exp(0)=1 and exp(-1)≈0.3679.
exp_scores = np.exp(shifted)  # (4, 2). Intermediate values, not learned weights.
# sum(axis=1) adds the class values in each row; keepdims=True retains the sum's column.
row_sums = exp_scores.sum(axis=1, keepdims=True)  # (4, 2) -> (4, 1).
probabilities = exp_scores / row_sums  # Divide each element by its row's sum: (4, 2).

# len(y) is the number of labels, 4. np.arange(4) creates the integers [0, 1, 2, 3].
row_indices = np.arange(len(y))  # (4,): the sample row indices.
# Pair the indices at matching positions to select [0,0], [1,1], [2,1], and [3,0].
# Use the true labels y, rather than the model's predictions pred, to select probabilities.
correct_p = probabilities[row_indices, y]  # Select one value per row: (4, 2) -> (4,).
losses = -np.log(correct_p)  # np.log() applies the natural logarithm. Four losses, (4,).
mean_ce = losses.mean()  # mean() returns the arithmetic mean of four losses: one number.

# This calculation reduces problems caused by tiny probabilities rounding to zero.
# Compute log probabilities from scores, rather than taking log of stored probabilities.
log_probabilities = shifted - np.log(row_sums)  # (4, 2) - (4, 1) -> (4, 2).
stable_losses = -log_probabilities[row_indices, y]  # Select true-class log probabilities: (4,).

# .size is an array's element count. sum() adds the counts for all parameter arrays.
parameter_count = sum(a.size for a in [W1, b1, W2, b2])  # 4 + 2 + 4 + 2 = 12.

# assert stops if a condition is false. allclose() allows small floating-point differences.
# array_equal() checks array shapes and values; isclose() checks whether two values are close.
assert X.shape == (4, 2) and y.shape == (4,)
assert H.shape == (4, 2) and S.shape == (4, 2)
assert row_max.shape == (4, 1) and row_sums.shape == (4, 1)
assert np.array_equal(Z1, [[0, 0], [-1, 1], [1, -1], [0, 0]])
assert np.array_equal(H, [[0, 0], [0, 1], [1, 0], [0, 0]])
assert np.array_equal(S, [[1, 0], [0, 1], [0, 1], [1, 0]])
assert np.allclose(probabilities.sum(axis=1), 1.0)  # Each sample's probabilities sum to 1.
assert np.array_equal(pred, y)  # Every sample's predicted class must match its true class.
assert np.allclose(stable_losses, losses)
assert np.isclose(mean_ce, 0.3132616875182228) and parameter_count == 12

# Unpack each name-and-array pair, then print .shape (axis lengths) and the actual values.
for name, values in [
    ("X", X), ("Z1", Z1), ("H", H), ("S", S),
    ("row_max", row_max), ("shifted", shifted), ("exp_scores", exp_scores),
    ("row_sums", row_sums), ("probabilities", probabilities),
    ("correct_p", correct_p), ("losses", losses),
    ("log_probabilities", log_probabilities), ("stable_losses", stable_losses),
]:
    print(name, "shape=", values.shape, "\n", values)
print("Predictions:", pred, pred_names)
print("Mean cross-entropy:", mean_ce, "Parameter count:", parameter_count)
print("All XOR checks passed.")
```
<!-- CORE_CODE_END -->

The code prints `Predictions` as `[0 1 1 0]`, `Mean cross-entropy` as approximately `0.3132616875`, and `Parameter count` as `12`. The four-decimal values in the diagrams and tables are rounded versions of these calculation results.

The <a href="/neural-network-matrix-basics/examples-en.py" download>complete example code, examples-en.py</a>, also runs the full matrices for two hidden layers, compares `max` with `argmax`, and compares subtracting the maximum with dividing the scores. The additional network has 82 parameters, and entry `[0, 2]` in its first hidden output is approximately 0.9.

## Official documentation

The API behavior was checked against the following official documentation and author-written textbook on October 20, 2026. The XOR numbers, matrices, diagrams, and additional layer example come from directly running this article's example calculations. The PyTorch links point to version 2.9 documentation used to check weight orientation and the loss function's input requirements.

[^dense]: Keras, [Dense layer](https://keras.io/api/layers/core_layers/dense/). Describes a fully connected layer that multiplies inputs by a kernel and applies a bias and activation function.
[^matmul]: NumPy, [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html). Describes matrix multiplication shape rules and the meaning of the `@` operator.
[^broadcasting]: NumPy, [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html). Describes how elementwise operations match dimension sizes starting from the right.
[^maximum]: NumPy, [numpy.maximum](https://numpy.org/doc/stable/reference/generated/numpy.maximum.html). Documents the function that compares corresponding elements of two inputs.
[^indexing]: NumPy, [Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html). Explains zero-based indices, slices, and integer array indexing.
[^argmax]: NumPy, [numpy.argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html). Describes the positions of maximum values and how ties are handled.
[^max]: NumPy, [numpy.max](https://numpy.org/doc/stable/reference/generated/numpy.max.html). Explains `axis` and `keepdims`.
[^sum]: NumPy, [numpy.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html). Describes summing along a specified axis and retaining dimensions.
[^exp]: NumPy, [numpy.exp](https://numpy.org/doc/stable/reference/generated/numpy.exp.html). Documents the elementwise exponential function.
[^softmax]: SciPy, [scipy.special.softmax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html). Describes the softmax definition, probabilities summing to 1 along an axis, and stabilization through shifting by the maximum.
[^arange]: NumPy, [numpy.arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html). Describes creating regularly spaced values while excluding the stopping value.
[^log]: NumPy, [numpy.log](https://numpy.org/doc/stable/reference/generated/numpy.log.html). Documents the elementwise natural logarithm.
[^mean]: NumPy, [numpy.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html). Describes the arithmetic mean over all elements or a specified axis.
[^ce]: PyTorch 2.9, [CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html). Describes loss calculations that take raw scores and true class indices.
[^logsoftmax]: SciPy, [scipy.special.log_softmax](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_softmax.html). Describes a calculation more numerically stable than taking the logarithm of probabilities directly.
[^mlp]: Dive into Deep Learning, [Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html). Explains hidden layers, nonlinear activation functions, and ReLU.
[^assert]: Python, [The assert statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement); NumPy, [allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html), [array_equal](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html), [isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html).
[^linear]: PyTorch 2.9, [Linear](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Linear.html). Describes the stored weight shape and the calculation using a transpose.
[^autograd]: PyTorch, [Automatic Differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html). Explains how connected calculations produce gradients for parameters.
[^tuning]: KerasTuner, [Getting started](https://keras.io/keras_tuner/getting_started/); scikit-learn, [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html). Explain searching over settings such as neuron count and the roles of evaluation data.
