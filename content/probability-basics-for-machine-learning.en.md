+++
title = "Probability Basics for Machine Learning: Random Variables, the Binomial Distribution, Expected Value, and Variance"
description = "An introduction to probability that connects trials and events to random variables, the binomial distribution, expected value, variance, and standard deviation through small tables and calculations."
date = 2026-10-06
updated = 2026-10-06
slug = "probability-basics-for-machine-learning"

[extra]
katex = true
styles = ["probability-basics/article.css"]
+++

In machine learning, we learn patterns from data and use those patterns to make predictions about new data. Some problems call for predicting a number. Others call for predicting a probability, such as the chance that an email is spam.

Learning probability gives us a way to interpret these numbers. It helps us distinguish **what varies randomly, what value we are measuring, and how often we expect that value to occur**. Expected value and variance begin with these distinctions, too.

This article is for readers who are encountering probability and statistics for the first time. We will explain small examples in words and tables before introducing formulas as a shorter way to write the calculations. Each unfamiliar symbol is explained where it first appears.

## 1. Start with the overall picture

When we toss a coin or roll a die, it is difficult to determine the exact outcome before we do it. Here, a **random phenomenon** means a situation in which we consider several possibilities using probabilities, rather than treating the outcome of a single attempt as known in advance. The word random does not, by itself, mean that all outcomes have the same probability.

Before we can calculate anything in such a situation, we need to decide two things.

First, **which outcomes are possible and what probability to assign to each one**. Second, **what number to record from each outcome**. Once both are specified, we can list the possible numbers and their probabilities.

![Possible outcomes and their probabilities combine with a rule for assigning numbers to form a probability distribution. The distribution is then used to calculate expected value and variance.](/probability-basics/overview-en.svg)

It is fine if the terms in the diagram are unfamiliar. We will work through the following questions one at a time.

| Concept | The question it answers |
| --- | --- |
| Random variable | What number should we record from an outcome? |
| Probability distribution | Which numbers can occur, and what is the probability of each? |
| Binomial distribution | If we repeat success/failure trials under the same conditions, how can the total number of successes vary? |
| Expected value | What is the average when we account for the probabilities? |
| Variance and standard deviation | How widely are the values spread around that average? |

We will mainly consider cases in which **the possible values can be counted one by one**, such as a number of successes or the number showing on a die. These are called **discrete random variables**. Discrete means that the possible values are individually distinguishable. A discrete random variable does not have to take only integer values. For example, a score whose only possible values are 0 and 2.5 is also a discrete random variable.

## 2. The terms we need first: trial, outcome, event, and probability model

### A trial is one execution of a procedure

A **trial** is an execution of a specified procedure to observe an outcome. Tossing a coin once and rolling a die once are examples.

An **outcome** is what actually results from that trial. If we roll a die once and get a 4, the outcome of that trial is 4.

The collection of all possible outcomes is called the **sample space**. For an ordinary six-sided die, it consists of `1, 2, 3, 4, 5, 6`. In mathematics, such a collection is called a **set** and is written inside braces, as in `{1, 2, 3, 4, 5, 6}`.

An **event** is a collection of outcomes that satisfy a condition we are interested in.

| Term | Example: rolling a die once |
| --- | --- |
| Trial | Roll the die once. |
| Actual outcome | The result is 4. |
| Sample space | `{1, 2, 3, 4, 5, 6}` |
| The event "an even number occurs" | `{2, 4, 6}` |
| The event "a 6 occurs" | `{6}` |

An event can contain several outcomes or just one. If the actual outcome is 4, the event "an even number occurs" has occurred, while the event "a 6 occurs" has not.

### Probability is a number describing how likely an event is

A **probability** is a number assigned to an event, from 0 to 1 inclusive. We use 1 to represent the full probability, or 100%.

<div class="probability-math">
$$
0.5=\frac{1}{2}=50\%
$$
</div>

These three expressions represent the same amount. `0.5` is a decimal, `1/2` is a fraction, and `50%` is a percentage. A percentage expresses a proportion by treating the whole as 100.

If we assign the same probability to all six outcomes of a die, each has probability $1/6$. We will call a die a **fair die** when we assume that each face has the same probability of occurring. A fair coin is one for which we assign probability $1/2$ to each of heads and tails.

The coin models in this article include only heads and tails as outcomes. They do not include possibilities such as a coin landing on its edge. We are deciding what to include and which probabilities to use so that we can work with reality in a form we can calculate.

A **probability model** provides this basis for calculation by specifying the possible outcomes and their probabilities. The assumption of fairness is part of the model. It does not automatically prove that a physical object is fair.[^model]

For a fair die, three of the six outcomes are even, so the probability is $3/6=1/2$. We can calculate "number of qualifying outcomes ÷ total number of outcomes" here **because every outcome has the same probability**. If the probabilities differ, we must add the probabilities of the qualifying outcomes rather than just count them.

## 3. A random variable is a rule for representing outcomes with numbers

Suppose we toss a fair coin once and record a score using the following rule.

| Coin outcome | Score to record |
| --- | ---: |
| Heads | 0 points |
| Tails | 10 points |

We will call this score $R$. **The random variable $R$ is the rule that assigns the number 0 or 10 according to the coin's outcome**. The scoring rule is already fixed before we toss the coin. What we do not yet know is which value that rule will produce on this particular trial.

In mathematics, a rule that determines an output from an input is called a **function**. In this example, the coin's outcome is the input and the score is the output. This is why we describe a random variable as a function.[^random-variable]

We need to distinguish these three things.

- **Random outcome:** whether this toss lands heads or tails.
- **Random variable:** the rule $R$ that records 0 points for heads and 10 points for tails.
- **Observed value:** the score obtained after actually tossing the coin. If the outcome is tails, this observed value is 10.

### We can define different numbers to measure from the same outcome

Now consider one roll of a fair die. We can define the following two random variables using the same die outcomes.

| Die outcome | $D$, which records the number showing | $Q$, which records 999 points only for a 6 |
| --- | ---: | ---: |
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 3 | 0 |
| 4 | 4 | 0 |
| 5 | 5 | 0 |
| 6 | 6 | 999 |

$D$ records the number showing. $Q$ records a score based on whether a 6 occurred. They use the same trial, but they are different random variables because the numbers we are interested in are different.

A random variable does not have to take only the values 0 and 1. It can take 0 and 10, the numbers 1 through 6, or 0 and 999. However, **the meaning of the average also depends on how we assign the numbers.**

For example, labeling red, blue, and green with the numbers 1, 2, and 3 does not make the average of those numbers an "average color." Being able to represent something with numbers and having an average that serves the purpose of our analysis are separate matters.

### Defining a random variable does not determine its probabilities by itself

Knowing only the rule "0 points for heads, 10 points for tails" does not tell us the probability of getting 10 points. We also need to know the probabilities of heads and tails.

The calculation therefore proceeds as follows: **specify the probability model and recording rule → find the possible values and their probabilities → calculate expected value and variance**. Once the model and rule have been specified, we cannot choose an expected value independently of them.

## 4. How to read P(X = x), and what a probability distribution shows

We use $P$ to write probabilities mathematically. It stands for *Probability*. **$P(\text{event})$ means the probability of the event inside the parentheses**.

For the coin score $R$ defined earlier, we write the following.

<div class="probability-math">
$$
P(R=10)=0.5
$$
</div>

Read this as **"the probability that the score $R$ is 10 equals 0.5."**

| Part of the expression | Meaning |
| --- | --- |
| $R$ | The score determined by the coin's outcome |
| $R=10$ | The event that the score is 10 |
| $P(R=10)$ | The probability of that event |
| $0.5$ | The probability: 50% |

Here, 10 is a **score** and 0.5 is a **probability**. The expression does not say that the probability is 10.

### Uppercase and lowercase letters have different roles

Textbooks often write this structure in the general form $P(X=x)$. Here, the uppercase $X$ represents a random variable, and the lowercase $x$ represents one value that variable can take. The $X$ here is a general name used to explain the notation, not a new name for a coin in our example.

For example, if we see $P(X=10)=0.3$, we read it as "the probability that the random variable $X$ defined in this text equals 10 is 30%." We must first check how that text has defined $X$.

### Listing the possible values and their probabilities gives a probability distribution

Let us return to the coin score $R$.

| Possible score $r$ | Probability of that score, $P(R=r)$ |
| ---: | ---: |
| 0 | 0.5 |
| 10 | 0.5 |

This table is the **probability distribution of the score $R$**. It shows which values are possible and what probability is assigned to each. For a discrete random variable, the function that gives the probability of each value is called the **probability mass function**, or **PMF**. For now, think of it as a rule that returns the probability in this table for each value.[^random-variable]

When writing a discrete probability distribution, check two things.

1. Is each probability between 0 and 1 inclusive?
2. Do the probabilities of all possible values add up to 1, with no values left out?

The random variable $Q$ also has two possible values. Their probabilities are different, however, because of the die-scoring rule we defined earlier.

<div class="probability-math">
$$
P(Q=0)=\frac{5}{6},\qquad P(Q=999)=\frac{1}{6}
$$
</div>

The five die outcomes `1, 2, 3, 4, 5` all give a score of 0, so we add their probabilities. **Having two possible numerical values does not mean that each has probability one-half.**

## 5. Bernoulli trials and recording 0 or 1

In some trials, it is enough to divide all outcomes into two categories: **the event of interest occurs, or it does not occur**.

| What are we checking? | Record 1 if it occurs | Record 0 if it does not |
| --- | --- | --- |
| A purchase | Purchased | Did not purchase |
| A defect | Defective | Not defective |
| A coin landing heads | Heads | Tails |

A single trial whose outcomes are divided into two categories in this way is called a **Bernoulli trial**. By convention, occurrence of the event of interest is called a **success**, and non-occurrence is called a **failure**. Success does not mean that something good happened. If we want to count defects, we can define the occurrence of a defect as a success.

A random variable that represents success with 1 and failure with 0 is a **Bernoulli random variable**. We can apply this to a die, too. If we record only whether a 6 occurred, the outcomes fall into two categories: a 6 is a success, and the other five numbers are failures.

### Use different symbols for one trial and a sum of trials

From now on, when we repeat success/failure trials, **$X_i$ will denote the random variable that records 1 for success and 0 for failure on the $i$th trial**. The small $i$ below the letter identifies the trial number. The first is $X_1$ and the second is $X_2$. This small number or label is called a **subscript**.

Let $p$ be the probability of success on one trial. Since each result must be classified as either success or failure, the probability of failure is $1-p$.

<div class="probability-math">
$$
P(X_i=1)=p,\qquad P(X_i=0)=1-p
$$
</div>

If $p=0.3$, the success probability is 30% and the failure probability is 70%. The success probability does not have to be 0.5.

We can write this distribution briefly as follows.

<div class="probability-math">
$$
X_i\sim\operatorname{Bernoulli}(p)
$$
</div>

Here, $\sim$ is read as **"follows the probability distribution on the right."** `Bernoulli` is the name of the distribution, and $p$ in parentheses is its success probability. This use of the tilde is different from contexts in which it means "approximately equal."

Let $n$ be the number of trials, and define $S_n$ as **the total number of successes from the first through the $n$th trial**.

<div class="probability-math">
$$
S_n=X_1+X_2+\cdots+X_n
$$
</div>

The symbol $\cdots$ means that the intervening additions continue in the same way. Adding the 0s and 1s counts the number of 1s, which is the number of successes.

For example, if five trials produce `1, 0, 1, 1, 0`, we get the following.

<div class="probability-math">
$$
S_5=1+0+1+1+0=3
$$
</div>

| Symbol | What does it mean? | Possible values |
| --- | --- | --- |
| $X_{10}$ | The result of the tenth trial alone | 0 or 1 |
| $X_{10}=1$ | The event that the tenth trial succeeds | We determine whether this statement is true or false |
| $S_{10}$ | The total number of successes in ten trials | 0 through 10 |
| $S_{10}=3$ | The event that exactly three of the ten trials succeed | We determine whether this statement is true or false |

**The 10 in $X_{10}$ indicates a position in the sequence; the 10 in $R=10$ is a score.** Throughout this article, $X_i$ will refer to an individual trial and $S_n$ to the total number of successes.

## 6. Conditional probability and independence: what information are we using?

### Conditional probability accounts for information we are given

Suppose a fair die has been rolled once. Before seeing the number, we are told only that the result is even.

Without that information, the probability of a 6 is $1/6$. Once we know that the result is even, however, we can narrow the possibilities to `2, 4, 6`. These three outcomes remain equally likely, so the probability of a 6 is now $1/3$.

This **probability given some fact** is called conditional probability. Let us give the events short names.

- $A$: the event that the die shows 6.
- $B$: the event that the die shows an even number.

<div class="probability-math">
$$
P(A\mid B)=\frac{1}{3}
$$
</div>

Read $\mid$ as **"given the condition on the right."** Thus, $P(A\mid B)$ means "the probability of a 6, given that the result is even." Reversing $A$ and $B$ changes the question. $P(B\mid A)$ asks for "the probability of an even number, given that the result is 6," which is 1 in this example.

The general formula is as follows.[^conditional]

<div class="probability-math">
$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad P(B)>0
$$
</div>

$A\cap B$ is **the event that both $A$ and $B$ occur**. The symbol $\cap$ denotes the **intersection**. In this example, the condition is "a 6 and an even number," so the only qualifying outcome is 6.

The **numerator**, the top of the fraction, is the probability of satisfying both conditions. The **denominator**, the bottom, is the probability of the known condition $B$. Here, the calculation is $(1/6)\div(1/2)=1/3$. **We are recalculating a probability relative to condition $B$, after originally expressing it relative to all outcomes.** The condition $P(B)>0$ is needed to avoid dividing by zero in this formula.

### Independence means that learning another outcome does not change the probability

**Independence** is a relationship in which knowing that one event occurred does not change the probability of the other.

Assume that we toss a fair coin independently each time under the same conditions. If heads counts as success, knowing the first result does not change the probability of heads on the second toss.

<div class="probability-math">
$$
P(X_2=1\mid X_1=1)=P(X_2=1)=0.5
$$
</div>

Heads on the first toss does not mean that tails must become more likely on the second. Even after several consecutive heads, the probability of heads on the next toss stays the same in a model with independent trials and an unchanged success probability.

Repeated trials are not always independent. Suppose a bag contains one red ball and one blue ball. If we draw a ball and **do not put it back**, drawing the red ball first makes the probability of drawing a red ball second equal to 0. Knowing the earlier result has changed the later probability.

Independence of all the trials is a stronger condition than simply considering a few pairs of results separately. When this article assumes independent repeated trials, it means **a situation in which even knowing the combined results of the previous trials does not change the success probability on the next trial**.

The term Bernoulli trial tells us that one trial's outcome is classified as success or failure. **Whether several Bernoulli trials are independent and whether they share a success probability must be checked separately.**

## 7. Multiply within one sequence; add the probabilities of different sequences

To understand the binomial distribution, we need to distinguish **cases in which all conditions must hold together** from **cases in which any one of several possibilities is enough**.

### AND: all the specified conditions must hold

Let us find the probability that three independent tosses of a fair coin produce exactly **heads → tails → heads**, in that order. Here, we will call a list of outcomes with their order specified a **sequence**.

For this sequence to occur, **all** three conditions must hold: heads first, tails second, and heads third. This is the meaning of AND.

Since the three trials are independent, we calculate as follows.

<div class="probability-math">
$$
0.5\times0.5\times0.5=0.125
$$
</div>

To see why we multiply, start with two steps. The probability of heads on the first toss is 0.5. Within those cases, the probability of tails on the second toss is again 0.5. The probability of satisfying both conditions is therefore $0.5\times0.5=0.25$. Multiplying again by the probability of satisfying the third condition, heads, gives 0.125.

**The multiplication rule also works without independence, but from the second step onward we must use probabilities that account for the preceding conditions.** The general multiplication rule for two events is as follows.[^conditional]

<div class="probability-math">
$$
P(A\cap B)=P(B)P(A\mid B),\qquad P(B)>0
$$
</div>

In the die example from the previous section, $A$ meant a 6 and $B$ meant an even number. The probability of "a 6 and an even number" is as follows.

<div class="probability-math">
$$
\frac{1}{2}\times\frac{1}{3}=\frac{1}{6}
$$
</div>

Ignoring the condition that the number is even and multiplying $P(B)P(A)=(1/2)(1/6)$ gives the wrong answer. Only when the two events are independent does $P(A\mid B)=P(A)$, allowing the formula to simplify as follows.

<div class="probability-math">
$$
P(A\cap B)=P(A)P(B)
$$
</div>

### OR: any one of the different sequences is enough

Now let us find the probability of **exactly one head** in the same three coin tosses. We place no restriction on which toss produces that head.

| Position of the head | Sequence | Probability of that sequence |
| --- | --- | ---: |
| First | Heads·Tails·Tails | 0.125 |
| Second | Tails·Heads·Tails | 0.125 |
| Third | Tails·Tails·Heads | 0.125 |

If **any one** of these three sequences occurs, the condition "exactly one head" is satisfied. This is the meaning of OR.

One set of three tosses cannot match two of these sequences at the same time. Such events are called **non-overlapping events**, or **mutually exclusive events**. In this case, we add their probabilities directly.

<div class="probability-math">
$$
P(S_3=1)=0.125+0.125+0.125=0.375
$$
</div>

Since we have added the same number three times, we can also write it as follows.

<div class="probability-math">
$$
0.125+0.125+0.125=3\times0.125
$$
</div>

Here, **$3\times$ is ordinary arithmetic that abbreviates the addition of three probabilities**. It is not a multiplication of probabilities for three independent events that must all occur. The factor at the beginning of the binomial formula will have this same meaning.

### OR does not always mean that we can simply add

$A\cup B$ is **the event that at least one of $A$ or $B$ occurs**. The symbol $\cup$ denotes the **union**. It also includes cases in which both occur.

If the two events share any outcomes, $P(A)+P(B)$ counts the probabilities of those outcomes twice. We must subtract the shared probability once to correct that double counting.[^model]

<div class="probability-math">
$$
P(A\cup B)=P(A)+P(B)-P(A\cap B)
$$
</div>

For the die used earlier, the event "a 6 or an even number" contains the outcomes `{2, 4, 6}`. The 6 is already included among the even numbers. We therefore add $(1/6)+(3/6)$ and then subtract $1/6$, the probability of the 6 we counted twice, to obtain $3/6$.

| What we want to calculate | General calculation | When it simplifies |
| --- | --- | --- |
| Both events occur | Probability of one event × probability of the other given that event | If independent, multiply their original probabilities. |
| At least one of two events occurs | Sum of the two probabilities − probability of their overlap | If they do not overlap, simply add their probabilities. |

**Independent and mutually exclusive mean different things.** Independence means that learning the information does not change the probability. Mutual exclusivity means that the events cannot occur together. Heads and tails on the same coin toss are mutually exclusive, but they are not independent: once we know the toss was heads, the probability of tails becomes 0.

## 8. Combinations: how many ways can we choose the positions of the successes?

To get exactly two heads in four coin tosses, we can choose which two of the four positions will contain heads. The remaining positions must then contain tails.

| Positions of the heads | Resulting sequence |
| --- | --- |
| `{1, 2}` | Heads·Heads·Tails·Tails |
| `{1, 3}` | Heads·Tails·Heads·Tails |
| `{1, 4}` | Heads·Tails·Tails·Heads |
| `{2, 3}` | Tails·Heads·Heads·Tails |
| `{2, 4}` | Tails·Heads·Tails·Heads |
| `{3, 4}` | Tails·Tails·Heads·Heads |

There are six possibilities. A **selection of some items from a collection, where the order in which we select them does not matter**, is called a **combination**.

We need to read "order does not matter" carefully here. It means that **choosing position 1 and then position 2 selects the same two positions as choosing position 2 and then position 1**. In contrast, `{1, 2}` and `{1, 3}` are different selections because they place the heads in different positions.

### Why is the answer 6, rather than 4 × 3?

There are four choices for the first position we select, followed by three choices for a different position. Thus, **if we distinguish the order of selection**, there are $4\times3=12$ possibilities.

But this calculation counts `position 1 → position 2` separately from `position 2 → position 1`. Each selection of the same two positions has been counted twice, so we divide by 2.

<div class="probability-math">
$$
\frac{4\times3}{2}=6
$$
</div>

### Factorial is a short notation for multiplying consecutive integers

The **factorial** of a positive integer is the product of that integer and every integer down to 1. We write it by placing an exclamation mark after the number.

<div class="probability-math">
$$
3!=3\times2\times1=6
$$
</div>

<div class="probability-math">
$$
4!=4\times3\times2\times1=24
$$
</div>

If we select three distinct positions and distinguish the **order of selection**, the same group of positions is counted $3!=6$ times. The combination formula generalizes this principle.

If there are $n$ positions in total and we choose $k$ of them, we write the number of combinations as $\binom{n}{k}$. The two numbers appear one above the other inside parentheses, but this is not a fraction. The notation $C(n,k)$ has the same meaning.

<div class="probability-math">
$$
\binom{n}{k}=\frac{n!}{k!(n-k)!}
$$
</div>

First, $n!/(n-k)!$ counts selections of $k$ distinct items **with the order of selection distinguished**. For $n=4$ and $k=2$, we can cancel the common factor $2\times1$ in the numerator and denominator, as follows.

<div class="probability-math">
$$
\frac{4!}{2!}=\frac{4\times3\times2\times1}{2\times1}=4\times3=12
$$
</div>

These are the twelve possibilities we counted earlier. Each selection of the same $k$ items is counted in $k!$ different orders, so dividing again by $k!$ gives the number of combinations.

<div class="probability-math">
$$
\binom{4}{2}=\frac{4!}{2!2!}=\frac{24}{2\times2}=6
$$
</div>

Also remember the definition $0!=1$. There is one way to select no positions: select nothing. There is also one way to select every position. Thus, $\binom{n}{0}=\binom{n}{n}=1$.

## 9. The binomial distribution: counting successes across several trials

Earlier, we defined $X_i$ to record whether an individual trial succeeds, and $S_n$ as the total number of successes in $n$ trials. When the following four conditions hold, $S_n$ follows a **binomial distribution**.[^binomial]

| Condition | What to check |
| --- | --- |
| The number of trials is fixed. | Specify the number of trials, $n$, in advance. |
| Each trial is classified as success or failure. | Record each $X_i$ as 1 or 0. |
| The success probability is the same. | Use the same $p$ for every trial. |
| The trials are independent. | Knowing the results of other trials does not change the probability for a given trial. |

We can express these conditions in one line as follows.

<div class="probability-math">
$$
S_n\sim\operatorname{Binomial}(n,p)
$$
</div>

`Binomial` is the name of the distribution. Read this as "the distribution of the total number of successes in $n$ independent trials, each with success probability $p$." Numbers such as $n$ and $p$ that determine the specific shape of a distribution are called **parameters**.

The possible values of $S_n$ range from 0 through $n$. We will use $k$ for the particular number of successes whose probability we want to calculate. For example, to find the probability of two successes in four trials, we use $n=4$ and $k=2$. **$n$ is the total number of trials; $k$ is the success count whose probability we are asking for.**

### Start with the probability of one specific sequence

For four tosses of a fair coin, the probability of the one sequence `Heads·Heads·Tails·Tails` is as follows.

<div class="probability-math">
$$
0.5\times0.5\times0.5\times0.5=0.0625
$$
</div>

Now extend this to a success probability of $p$. If there are $k$ successes, there are $n-k$ failures. **All the results of the independent trials must match the specified order**, so we multiply the success probability $p$ a total of $k$ times and the failure probability $1-p$ a total of $n-k$ times.

Multiplying the same number repeatedly is called taking a **power**. For example, $p^3$ means $p\times p\times p$, and $p^k$ means the product of $k$ copies of $p$.

The probability of one specific sequence is therefore as follows.

<div class="probability-math">
$$
p^k(1-p)^{n-k}
$$
</div>

Writing the two expressions next to each other also means multiplication: it is the same as $p^k\times(1-p)^{n-k}$. When a probability is strictly between 0 and 1, requiring that outcome zero times gives a factor such as $p^0=1$. There is no additional condition to multiply in.

### Add the probabilities of sequences with the same success count

There are $\binom{n}{k}$ ways to choose the positions of the successes. Because the trials are independent and the success probability is the same each time, every sequence with $k$ successes has the same probability.

If any one of those sequences occurs, the total number of successes is $k$. The sequences do not overlap, so we add their probabilities. Adding the same probability $\binom{n}{k}$ times gives the binomial probability formula.

<div class="probability-math">
$$
P(S_n=k)=\binom{n}{k}p^k(1-p)^{n-k}
$$
</div>

| Part of the formula | What it calculates |
| --- | --- |
| $p^k(1-p)^{n-k}$ | The probability that all conditions within one sequence hold. We multiply probabilities for independent trials. |
| $\binom{n}{k}$ | The number of different sequences containing $k$ successes. |
| Number of sequences × probability of one sequence | Repeated addition of the same probability for sequences that do not overlap. |

**The fact that the formula contains multiplication does not mean that every part uses the multiplication rule for independent events.** One formula contains both the step of combining conditions within a sequence and the step of adding probabilities across sequences.

Calculating the probability of exactly two heads gives the following.

<div class="probability-math">
$$
P(S_4=2)=6\times0.0625=0.375=37.5\%
$$
</div>

The explanation of powers above assumes $0<p<1$. A binomial distribution with $p=0$ is also possible; it always gives zero successes. If $p=1$, it always gives $n$ successes. These two cases can be understood directly because all the probability is concentrated on one value.

### The distribution becomes visible when we list every success count

For four tosses of a fair coin, with $n=4$ and $p=0.5$, we get the following.

| Success count $k$ | Number of sequences | Probability of one sequence | $P(S_4=k)$ |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0.0625 | 0.0625 |
| 1 | 4 | 0.0625 | 0.2500 |
| 2 | 6 | 0.0625 | 0.3750 |
| 3 | 4 | 0.0625 | 0.2500 |
| 4 | 1 | 0.0625 | 0.0625 |
| Total | 16 | — | 1.0000 |

Having five possible success counts does not mean that each has probability $1/5$. Six sequences produce two successes, while only one produces four successes.

In the bar charts below, **the horizontal axis $k$ is the number of successes**, and **the vertical axis is the probability of that success count**. A taller bar means that the corresponding success count is more likely. The upper chart uses $p=0.5$, matching the table. The lower chart shows a different binomial model in which only the success probability changes, to $p=0.3$.

![Probability bar charts for a binomial distribution with four trials, comparing success probabilities of 0.5 and 0.3.](/probability-basics/binomial-n4-en.svg)

The bar heights sum to 1 in each chart. With $p=0.3$, the lower success probability places more probability on smaller success counts.

### Not all 0/1 data follow a binomial distribution

If purchase probabilities differ across customers, or one customer's purchase is related to another's, we need to recheck the four conditions. An experiment that continues until a success occurs also differs from the fixed number of trials considered here.

A Bernoulli distribution describes **one 0/1 result**. A binomial distribution describes **the total number of successes over several trials under specified conditions**. We should not apply the binomial formula simply because the data are recorded as 0s and 1s.

## 10. Expected value: average the possible values using their probabilities

We usually calculate an average by adding the numbers we observed and dividing by how many numbers there are. But what should we average if we have not yet run the trial, or if we are considering all possible outcomes through a probability model?

In that case, **multiply each possible value by its probability and add the results**. This theoretical average of a random variable is its **expected value**.[^expectation]

Read $E[R]$ as "the expected value of the random variable $R$." The $E$ stands for *Expectation*. Here, expectation does not mean a desired result or a number we hope to see. It means an average calculated using probabilities.

For the fair-coin score defined earlier, heads gives 0 points and tails gives 10 points.

| Possible score | Probability | Score × probability |
| ---: | ---: | ---: |
| 0 | 0.5 | 0 |
| 10 | 0.5 | 5 |
| Total | 1 | 5 |

The expected value is therefore as follows.

<div class="probability-math">
$$
E[R]=0\times0.5+10\times0.5=5
$$
</div>

### We multiply by probabilities because values occur in different proportions

A number specifying how much a value contributes to an average is called a **weight**. An average calculated by applying weights to values is a **weighted average**.

For expected value, probabilities serve as weights. A value expected to occur more often receives more weight, while a rare value receives less. The probabilities already sum to 1, so **after multiplying the values by their probabilities and adding, we do not divide again by the number of distinct values**.

Consider $Q$ again, which gives 999 points only when a fair die shows 6.

<div class="probability-math">
$$
E[Q]=0\times\frac{5}{6}+999\times\frac{1}{6}=166.5
$$
</div>

We should not calculate $(0+999)/2$ simply because the possible scores are 0 and 999. The two values do not have equal probabilities. Five of the six die outcomes give 0 points, and one gives 999 points.

### Sigma tells us to repeat a calculation and add the results

$\sum$ is the summation symbol, called **sigma**. It tells us to repeat the calculation written after it for the specified values and add all the results.

Using the lowercase $r$ to represent one possible score, we can write the formula for the expected value of $R$ as follows.

<div class="probability-math">
$$
E[R]=\sum_r rP(R=r)
$$
</div>

In this article, $\sum_r$ means **sum over every score $r$ that $R$ can take**. For the score $R$, we substitute $r=0$ and $r=10$. The formula is therefore a short way to write the calculation we just made: $0\times0.5+10\times0.5$.

### An expected value does not have to be a possible outcome

The score $R$ can only be 0 or 10 points. An expected value of 5 points does not mean that a single coin toss can produce 5 points.

We calculate the expected value of $D$, the number showing on a fair die, in the same way.

<div class="probability-math">
$$
E[D]=\frac{1+2+3+4+5+6}{6}=3.5
$$
</div>

There is no face showing 3.5 on the die, but its expected value is 3.5. **An average combines several possibilities, so it does not have to appear in the list of individual outcomes.**

Expected value also does not mean the value with the highest probability. Finding the most likely value and calculating an average that accounts for all values and probabilities are different questions.

The variables $D$ and $Q$ use the same die, but their expected values are 3.5 and 166.5, respectively. We have not changed an average arbitrarily. **We have calculated different expected values because we defined different numbers to measure.**

## 11. How does expected value differ from the average of actual data?

Suppose we actually observe the score $R$ five times and obtain the following.

`0, 10, 0, 0, 10`

The average of the actual data is as follows.

<div class="probability-math">
$$
\frac{0+10+0+0+10}{5}=4
$$
</div>

The expected value calculated from the probability model was 5, but the average of this set of observations is 4. This difference is possible because 0 and 10 do not have to occur equally often in a set of five observations.

Let $m$ be the number of observations, and write the values actually obtained as $r_1,r_2,\ldots,r_m$. The subscripts on the lowercase $r$ identify the first, second, and last observed values. We write their average as $\bar r$. The bar above the letter denotes an average in this context.

<div class="probability-math">
$$
\bar r=\frac{r_1+r_2+\cdots+r_m}{m}
$$
</div>

We use $m$ for the number of observations here to distinguish it from $n$, the number of trials in one group in a binomial experiment.

| Quantity | What the calculation uses | Fair-coin score example |
| --- | --- | --- |
| Expected value $E[R]$ | Possible values and their probabilities in the probability model | $0\times0.5+10\times0.5=5$ |
| Observed mean $\bar r$ | Actual observed values and the number of observations | 4 for these five values |

The two numbers can happen to be equal. **The distinction is not that their numerical values always differ, but that their calculations are based on different information.**

### What "long-run average" means precisely

For a random variable with a finite list of possible values, such as the coin score in this article, **taking many independent observations from the same distribution** makes the observed mean increasingly likely to be close to the expected value. This property is called the **law of large numbers**.[^large-numbers]

This is why expected value is described as "the average over a very large number of repetitions under the same conditions." It does not, however, mean any of the following.

- The next observed value must equal the expected value.
- Every additional observation must bring the average closer to the expected value.
- Reaching a specified number of observations must make the average exactly equal to the expected value.

The observed mean can temporarily move farther away. Also, if we record only selected outcomes or the observation conditions keep changing, we must first recheck the assumed probability model and the actual data-collection process.

## 12. Why the expected values of Bernoulli and binomial distributions are simple

### A Bernoulli variable's expected value is its success probability

$X_i$ is 1 for success and 0 for failure. Substitute these values directly into the definition of expected value.

<div class="probability-math">
$$
E[X_i]=1\times p+0\times(1-p)=p
$$
</div>

Because this recording rule adds 1 for a success and 0 for a failure, the average calculated using probabilities equals the success probability.

For actual 0/1 data, too, the average is the **observed proportion** of 1s. For example, `1, 0, 1, 0, 0` contains two 1s out of five values.

<div class="probability-math">
$$
\frac{1+0+1+0+0}{5}=\frac{2}{5}=0.4
$$
</div>

The success proportion in these data is exactly 40%. However, observing 40% successes in five trials does not establish that the model's actual success probability $p$ must be 0.4. **Using observed proportions to learn about unknown probabilities** connects to statistical estimation, which we will encounter later.

### The expected value of a binomial distribution is n × p

The total number of successes is $S_n=X_1+\cdots+X_n$. Expected value has the property that **the expected value of a sum equals the sum of the individual expected values**. This works because we can rearrange the additions when averaging the sum for each possible outcome using its probability.[^expectation]

<div class="probability-math">
$$
E[S_n]=E[X_1]+E[X_2]+\cdots+E[X_n]
$$
</div>

Each trial has expected value $p$, so we add $p$ a total of $n$ times.

<div class="probability-math">
$$
E[S_n]=np
$$
</div>

$np$ is $n\times p$ written without the multiplication sign. **This addition property of expected values does not itself require independence.** The independence condition specified earlier is, however, required for us to call $S_n$ binomial in this setting.

For four tosses of a fair coin, $E[S_4]=4\times0.5=2$. Calculating directly from the probability table in section 9 gives the same value.

<div class="probability-math">
$$
\begin{aligned}
E[S_4]&=0\times0.0625+1\times0.25+2\times0.375\\
&\qquad+3\times0.25+4\times0.0625\\
&=2
\end{aligned}
$$
</div>

### When interpreting an expected success count, identify what counts as one group

If $n=10$ and $p=0.3$, then $E[S_{10}]=3$. This does not mean that every ten trials must contain exactly three successes.

It means that **if we define an experiment of ten trials as one group and repeat that group many times independently under the same conditions, the average success count per group approaches 3**.

| Example group of trials | Number of trials in the group | Example observed success count |
| --- | ---: | ---: |
| First group | 10 | 2 |
| Second group | 10 | 4 |
| Third group | 10 | 1 |

The success counts above are possible observations used for illustration. There is no reason that just three groups must have an average of 3. The expected value of 3 is calculated using the probabilities in the binomial model.

If $n=4$ and $p=0.3$, the expected value is 1.2. The success count itself is an integer, but an average of success counts can have a fractional part.

## 13. Variance: how do we average departures from the expected value?

Expected value alone does not tell us how widely the values are spread. Compare the following two scoring rules. Under each rule, assume that the two scores each have probability 50%.

| Rule | Possible scores | Expected value | Distance from the expected value |
| --- | --- | ---: | --- |
| $R$, as defined earlier | 0 points, 10 points | 5 points | 5 points each |
| A different rule for comparison | 4 points, 6 points | 5 points | 1 point each |

The two rules have the same mean, but the values of $R$ are farther from it. **Variance** is one way to describe this spread.

### First, subtract the mean from the value: deviation

A **deviation** is the difference obtained by subtracting the reference mean from a value. Since the expected value of $R$ is 5, we get the following.

| Score | Score − expected value | Meaning |
| ---: | ---: | --- |
| 0 | $0-5=-5$ | 5 points below the mean. |
| 10 | $10-5=5$ | 5 points above the mean. |

But averaging these deviations directly using their probabilities gives 0, as follows.

<div class="probability-math">
$$
(-5)\times0.5+5\times0.5=0
$$
</div>

The positive and negative values cancel each other out. The result is 0 even though the values are away from the mean, so this calculation alone cannot describe their spread.

### Variance squares the deviations

**Squaring** means multiplying a number by itself. We have $(-5)^2=(-5)\times(-5)=25$ and $5^2=25$. A deviation multiplied by itself is called a **squared deviation**.

Squaring prevents positive and negative deviations from canceling. A distance of 2 from the mean becomes 4, while a distance of 4 becomes 16, so it also gives greater emphasis to larger differences.

Squaring is not the only way to prevent cancellation. We could use the **absolute value**, which removes the sign and keeps only the size of the distance. **Variance is specifically defined as a measure that averages squared deviations**.[^variance]

### Weight each squared deviation by the probability of its original value

Let us calculate the variance of $R$ directly.

| Score $r$ | Probability | Deviation | Squared deviation | Squared deviation × probability |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5 | −5 | 25 | 12.5 |
| 10 | 0.5 | 5 | 25 | 12.5 |
| Total | 1 | — | — | 25 |

The variance is therefore 25. We write it as $\operatorname{Var}(R)$. `Var` is short for *Variance*.

<div class="probability-math">
$$
\operatorname{Var}(R)=(0-5)^2\times0.5+(10-5)^2\times0.5=25
$$
</div>

The probability 0.5 in this calculation represents **how often each original score, $R=0$ and $R=10$, occurs**. We account not only for the size of a squared deviation, but also for how often the original score that produces it occurs.

The general calculation follows these steps.

1. List the random variable's possible values and their probabilities.
2. Calculate the expected value from that probability distribution.
3. Subtract the expected value from each possible value.
4. Square each difference.
5. Multiply each squared deviation by the probability of its original value.
6. Add all the results.

For $R$, we can write these steps as the following formula.

<div class="probability-math">
$$
\operatorname{Var}(R)=\sum_r\bigl(r-E[R]\bigr)^2P(R=r)
$$
</div>

A shorter version is as follows.

<div class="probability-math">
$$
\operatorname{Var}(R)=E\bigl[(R-E[R])^2\bigr]
$$
</div>

Inside the expression, $R-E[R]$ is a deviation, and its square is a squared deviation. The outer $E[\ ]$ means **average these squared deviations using their probabilities**. This is where variance connects to expected value.

### What if two original values produce the same squared deviation?

For $R$, both 0 points and 10 points produce a squared deviation of 25. Let us define this squared deviation itself as a new random variable $T$.

<div class="probability-math">
$$
T=(R-5)^2
$$
</div>

| Original result | Probability of the original result | Transformed value $T$ |
| --- | ---: | ---: |
| $R=0$ | 0.5 | 25 |
| $R=10$ | 0.5 | 25 |

**The probabilities of the two cases that produce $T=25$ add up to 1.**

<div class="probability-math">
$$
P(T=25)=0.5+0.5=1
$$
</div>

Thus, if we build the distribution of $T$ alone and calculate its expected value, we get $E[T]=25\times1=25$. This is the same answer as $25\times0.5+25\times0.5$, calculated using the original results. A rule can still define a random variable even if transforming the outcomes makes its numerical value always the same. $T$ is an example.

This is the same principle we used when several die outcomes all gave a score of 0. **If different original outcomes produce the same number, the probability of that number is the sum of the probabilities of those outcomes.** Neither individual 0.5 in the table is "the total probability of $T=25$."

## 14. The variance of actual data and the variance of a probability distribution

### In actual data, each observation contributes once to the average

If the two actual observations are `0, 10`, their mean is 5. To summarize **the spread of these two observed values themselves**, we can average their squared deviations as follows.

<div class="probability-math">
$$
\frac{(0-5)^2+(10-5)^2}{2}=25
$$
</div>

We divide by 2 **because there are two observations**, not because there are two distinct kinds of value.

Now suppose the actual data are `0, 0, 0, 0, 0, 0, 0, 0, 0, 10`. There are still two distinct values, but now there are ten observations. Their mean is 1.

| Observed value | Number of occurrences | Squared deviation from the mean of 1 | Contribution including all occurrences |
| ---: | ---: | ---: | ---: |
| 0 | 9 | 1 | 9 |
| 10 | 1 | 81 | 81 |
| Total | 10 | — | 90 |

The variance of these ten observed values themselves is $90/10=9$. We have merely grouped equal values together; the calculation still accounts for all ten observations.

We can rearrange the expression slightly as follows.

<div class="probability-math">
$$
\frac{9(0-1)^2+1(10-1)^2}{10}=0.9(0-1)^2+0.1(10-1)^2=9
$$
</div>

**Number of occurrences ÷ total number of observations** is called **relative frequency**. Frequency is the number of occurrences; relative frequency is the proportion of the total represented by that count. In these data, the relative frequency of 0 is 0.9, and that of 10 is 0.1.

### In a probability distribution, the model supplies the weights

Separately from the fair-coin score $R$, define **a score $V$ that equals 0 with probability 0.9 and 10 with probability 0.1**. This is a different model with different probabilities.

<div class="probability-math">
$$
E[V]=0\times0.9+10\times0.1=1
$$
</div>

<div class="probability-math">
$$
\operatorname{Var}(V)=(0-1)^2\times0.9+(10-1)^2\times0.1=9
$$
</div>

The two calculations give the same numbers because the observed proportions above and the probabilities in this model are both 0.9 and 0.1. The observations have not conclusively proved that this is the correct model.

| Whose variance is being calculated? | Weights used in the average |
| --- | --- |
| The variance of the observed set of values itself | Relative frequencies calculated from actual occurrence counts |
| The variance of a random variable | Probabilities assigned to each value in the probability model |

Both calculations are weighted averages of squared deviations. **What differs is where the weights come from.**

### Why do some books divide by "number of observations − 1"?

The difference comes from the purpose of the analysis. First, distinguish the following terms.

- **Population:** the entire collection whose characteristics we want to understand.
- **Sample:** the part of the population that we actually survey or observe.
- **Estimation:** using a sample to calculate an unknown characteristic of the population.

We have just calculated **the spread of the observed values we have in hand**. If we instead want to use a sample to estimate the variance of a larger population, we must account for the effect of using a mean that changes from sample to sample.

Assume that $m$ observations are drawn independently from the same distribution and that its variance is finite. **Finite** here means that the variance has a definite numerical value, rather than being infinite. The coin and die examples in this article meet this condition.

The sample mean is the reference point that makes the sum of squared deviations within that sample as small as possible. If we use a mean calculated from the sample in place of the unknown population mean, the calculation can therefore tend to understate the spread.

If we divide the sum of squared deviations from the sample mean by $m$, then, when the population variance is nonzero, the result is smaller than the population variance on average over repeated samples. To correct this property, we use the **unbiased sample variance**, which divides by $m-1$. Here, **unbiased** means that the estimates have no systematic deviation on average when the estimation is repeated.[^sample-variance]

This correction is used when $m\ge2$. It does not guarantee an estimate closer to the actual variance for every individual sample. For now, distinguish **summarizing the observed values themselves from estimating a population characteristic using a sample**. We can return to the proof of this correction when studying samples and estimation in their own right.

## 15. Standard deviation: reading spread in the original units

The variance of the score $R$ was 25. Since we squared the deviations, however, its unit is not points but **points squared**, written as points². Squaring a difference of 5 points gives 25 points².

To express the spread in the same units as the score, we take the **square root** of the variance. A square root of a number is a number that produces it when squared. Both 5 and −5 give 25 when squared, so 25 has the square roots 5 and −5.

We write $\sqrt{25}=5$ to denote the **nonnegative square root**: the one that is greater than or equal to 0. The symbol is $\sqrt{\phantom{a}}$. Standard deviation uses this nonnegative value.

The **standard deviation** is the square root of the variance. It is often written using the Greek letter $\sigma$, pronounced "sigma." This is a different symbol from the summation sign $\sum$ introduced earlier. Writing $\sigma_R$ to indicate the standard deviation of $R$ gives the following.[^scale]

<div class="probability-math">
$$
\sigma_R=\sqrt{\operatorname{Var}(R)}=\sqrt{25}=5
$$
</div>

| Quantity | What it calculates | Value and units for the score $R$ |
| --- | --- | --- |
| Expected value | Probability-weighted average of the score | 5 points |
| Variance | Probability-weighted average of squared deviations from the mean | 25 points² |
| Standard deviation | Square root of the variance | 5 points |

Standard deviation uses the same units as the original values, which makes comparisons easier. However, **it is not the simple average of the distances from the mean.** We first square those distances and average them, then take the square root. Nor does it mean that "all values lie within one standard deviation of the mean."

When comparing the same kind of value in the same units, we can interpret a larger variance or standard deviation as greater spread around the mean. If the quantities or units differ, as with scores and amounts of money, we should not directly compare only the sizes of the numbers.

## 16. Why is the variance of a binomial distribution np(1 − p)?

The variance of a binomial distribution is not a separate concept. Applying the **probability-weighted average of squared deviations** that we have already learned to the binomial distribution gives a compact formula.

### Start with one 0/1 trial

$X_i$ equals 1 for success and 0 for failure, and its expected value is $p$. Insert these two values directly into the definition of variance.

<div class="probability-math">
$$
\operatorname{Var}(X_i)=(1-p)^2p+(0-p)^2(1-p)
$$
</div>

The deviation is $1-p$ for success and $-p$ for failure. We squared each deviation, then multiplied by the probability of success, $p$, or the probability of failure, $1-p$, respectively.

Both terms contain $p(1-p)$. Factoring it out gives the following.

<div class="probability-math">
$$
\operatorname{Var}(X_i)=p(1-p)\bigl[(1-p)+p\bigr]=p(1-p)
$$
</div>

The expression inside the square brackets is $1-p+p=1$, leaving only $p(1-p)$. This is the variance of a Bernoulli random variable representing one trial.

### Add the variances of independent trials

In the binomial model, $S_n=X_1+\cdots+X_n$, and the trials are independent. **The variance of a sum of independent random variables can be calculated by adding their individual variances.**[^variance]

Let us examine the role of independence here. The expected total number of successes is $np$, so the deviation of the total from its expected value is as follows.

<div class="probability-math">
$$
S_n-np=(X_1-p)+(X_2-p)+\cdots+(X_n-p)
$$
</div>

In other words, **the deviation of the sum is the sum of the individual trials' deviations**. To find the variance, we square this sum and take its probability-weighted average.

For two numbers $a$ and $b$, $(a+b)^2$ is not merely $a^2+b^2$: it is $a^2+2ab+b^2$. Likewise, when we add deviations from the mean and then square the sum, we get both the square of each deviation and **products of different deviations**.

Deviations obtained from independent trials are also independent. The probability-weighted average of the product of two independent values equals the product of their expected values. This is because, under independence, the probability of two outcomes occurring together is the product of their individual probabilities, allowing the average calculation to separate.

Each deviation has expected value $E[X_i-p]=E[X_i]-p=p-p=0$. Letting $i$ and $j$ denote different trial numbers, we have the following.

<div class="probability-math">
$$
E[(X_i-p)(X_j-p)]=E[X_i-p]E[X_j-p]=0\times0=0
$$
</div>

Here, **$i\ne j$**: we are referring to different trials. Multiplying the deviation of the same trial by itself produces a squared deviation, so we cannot discard it as zero.

Thus, the products of different deviations disappear when averaged, leaving only the averages of each trial's squared deviations: their variances. This is why we can add the variances in this situation.

<div class="probability-math">
$$
\operatorname{Var}(S_n)=\operatorname{Var}(X_1)+\cdots+\operatorname{Var}(X_n)
$$
</div>

Each variance is the same, $p(1-p)$, so adding it $n$ times gives the following.

<div class="probability-math">
$$
\operatorname{Var}(S_n)=np(1-p)
$$
</div>

Unlike the rule for adding expected values, this variance calculation used the independence of the trials. If different trials tend to succeed together or fail together, the products of different deviations may contribute to the result.

### Use numbers to read the expected value and spread together

For a binomial distribution with $n=10$ and $p=0.3$, we have the following.[^binomial-reference]

| Measure | Calculation | Meaning |
| --- | --- | --- |
| Expected value | $10\times0.3=3$ | The theoretical average of the total number of successes in one group of trials |
| Variance | $10\times0.3\times0.7=2.1$ | The spread of the total number of successes around that average; the units are the square of a count |
| Standard deviation | $\sqrt{2.1}\approx1.45$ | The spread expressed in the same units as the number of successes |

The symbol $\approx$ means "approximately equal to." We use it because we have shortened the decimal representation of the square root. An expected value of 3 and a standard deviation of about 1.45 do not mean that the number of successes is always 3 or must lie within $3\pm1.45$. The symbol $\pm$ means "plus or minus."

### With more trials, the total count can have greater variance while the proportion becomes more stable

For a fixed success probability with $0<p<1$, increasing the number of trials $n$ increases the variance $np(1-p)$ of the **total number** of successes. The **proportion** of successes, however, is $S_n/n$. The total count and the proportion are different quantities.

If we divide every value by the same number $n$, each difference from the mean is also divided by $n$. Each squared difference is therefore divided by $n^2$, so the variance of the proportion is as follows.

<div class="probability-math">
$$
\operatorname{Var}\left(\frac{S_n}{n}\right)=\frac{np(1-p)}{n^2}=\frac{p(1-p)}{n}
$$
</div>

This formula applies when $n\ge1$. As we collect more independent trials with the same success probability satisfying $0<p<1$, the variance of the success proportion decreases. **Distinguishing "how many successes?" from "what percentage of the total?"** explains how the variance of the total count can increase while the observed proportion becomes more stable.

If $p=0$ or $p=1$, the outcome is already determined, so the variance of both the total count and the proportion is 0.

## 17. How does this connect to machine learning?

### Distinguish a 0/1 ground-truth label from a model's predicted probability

In machine learning, a task that assigns an answer to one of two categories is called **binary classification**. Classifying email as spam or legitimate email is one example. The actual correct answer recorded in the training data is called a **label**.

If we assign 1 to spam and 0 to legitimate email, the ground-truth label is either 0 or 1. A model, on the other hand, can look at the email's content and output a predicted probability such as "I estimate the probability of spam to be 0.8."

| Value | What does it represent? |
| --- | --- |
| Ground-truth label 1 | A record that the email is spam |
| Predicted probability 0.8 | The model's estimate of how likely the email is to be spam, based on its information |

A predicted probability of 0.8 does not mean that the email's correct label is 0.8. We are distinguishing whether an event occurs from how likely it is to occur. The model estimates **a probability conditional on the given input information**, which also connects to conditional probability.

A model output of 0.8 also does not guarantee that the actual probability is exactly 0.8. For example, we can collect emails assigned a predicted probability near 0.8 and check whether about 80% are actually spam. **Calibration** concerns how closely predicted probabilities agree with actual occurrence rates. Adjusting predicted probabilities to improve this agreement is called **probability calibration**.[^calibration]

### The mean of 0/1 labels is the event's proportion in the data

If we record a canceled reservation as 1 and a reservation that was not canceled as 0, the sum of the labels is the number of cancellations, and their mean is the observed cancellation proportion.

For `1, 0, 1, 0, 0`, two of the five reservations were canceled, giving 40%. This is the cancellation proportion in the current data. To conclude that the probability of cancellation for future reservations is the same, we would also need to examine the conditions that produced the data, such as the customer mix and collection period.

### Average loss also connects to expected value

A **loss** is a number calculated by a specified rule to measure how far a prediction differs from the actual correct answer. During training, we adjust the model to make this loss smaller.

For example, in a task that predicts a quantity, if the actual value is 5 and the model predicts 4, the error is $4-5=-1$. Using a **loss rule that squares the error** gives a loss of 1 for that example. If the actual value is 5 and the model predicts 7, the loss is $(7-5)^2=4$.

Add one more example whose actual value and prediction are both 5. The three losses are then `1, 4, 0`, and the average loss for these data is $(1+4+0)/3=5/3$. We can average the losses of observed examples in this way.

The inputs and correct answers we will encounter in the future are uncertain. The probability-weighted average of loss across those possibilities is the **expected loss**. We should distinguish the average loss calculated on the current data from the expected loss under the distribution of future data.

Classification also uses other rules, such as **cross-entropy loss**, which uses the correct label and the predicted probability. This rule assigns a large loss to a prediction that gives a low probability to the correct answer. The connection we need here is **evaluate predictions numerically → average the losses across examples → train the model to predict well on average on new examples, too**.[^loss]

### Before interpreting variance, identify what is spread out

The variance of input data, the variance of predictions, and the variance of errors concern different quantities. First, identify whose variance is being calculated.

For example, customers' purchase amounts may have a large variance because the amounts vary widely. This alone does not tell us that the data are wrong or that the model's predictions are poor. The spread of predicted probabilities assigned to different customers is also a different concept from how much we can trust the prediction for a single customer.

**The habit of checking how a random variable is defined is just as necessary in machine learning.**

## 18. Before comparing numbers, check what was calculated

Organizing the key ideas by what we calculate gives the following overview.

| Concept | Question to ask | Notation or calculation in this article |
| --- | --- | --- |
| Random variable | Which outcomes are recorded as which numbers? | Coin score $R$; success or failure of each trial $X_i$ |
| Probability of a value | How likely is the specified value? | $P(R=10)$ |
| Probability distribution | Have all possible values and their probabilities been listed? | A table whose probabilities sum to 1 |
| Bernoulli distribution | Are we considering whether one trial succeeds? | $X_i$ is 0 or 1 |
| Binomial distribution | Is there a fixed number of trials, a constant success probability, and independence? | Total number of successes $S_n$ |
| Expected value | Which probabilities weight the possible values in the average? | $E[R]=\sum_r rP(R=r)$ |
| Observed mean | What data were actually obtained? | Sum of observations ÷ number of observations |
| Variance | Which mean is used to calculate the squared deviations being averaged? | $\operatorname{Var}(R)=E[(R-E[R])^2]$ |
| Standard deviation | Is the spread being read in the original units? | Square root of the variance |

For the binomial distribution in particular, we should be able to distinguish the following three calculations.

<div class="probability-math">
$$
P(S_n=k)=\binom{n}{k}p^k(1-p)^{n-k}
$$
</div>

This expression calculates **the probability of exactly $k$ successes**.

<div class="probability-math">
$$
E[S_n]=np
$$
</div>

This expression calculates **the average total number of successes**.

<div class="probability-math">
$$
\operatorname{Var}(S_n)=np(1-p)
$$
</div>

This expression calculates **how spread out the total number of successes is around its average**. Although the same $n$ and $p$ appear in these formulas, the quantities being calculated differ.

From here, we can continue to **continuous probability distributions**, which describe values that vary continuously; the **normal distribution**, a familiar bell-shaped distribution; **Bayes' theorem**, which updates probabilities using new information; and **likelihood**, which compares model settings while holding the observed data fixed.

Before moving on, check whether you can explain the following three statements in your own words.

1. **A random variable's distribution is determined only when both the probability model and the rule for assigning numbers are specified.**
2. **We construct a binomial distribution by multiplying probabilities to find the probability of one sequence, then adding the probabilities of all sequences with the same number of successes.**
3. **Expected value is a probability-weighted average of values; variance is a probability-weighted average of their squared differences from the expected value.**

## References

The following public educational resources were used to check definitions, formulas, and their conditions. The coin and die calculations and tables in the article were calculated directly from the models used in the explanations.

[^model]: MIT OpenCourseWare, 18.05 — [Probability: Terminology and Examples](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class02-prep.pdf). Sample spaces, events, basic probability rules, and addition for overlapping events.
[^random-variable]: MIT OpenCourseWare, 18.05 — [Discrete Random Variables](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class04-prep-a.pdf). Random variables, probability mass functions, and Bernoulli and binomial distributions.
[^conditional]: MIT OpenCourseWare, 18.05 — [Conditional Probability, Independence and Bayes' Theorem](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class03-prep.pdf). Conditional probability, the multiplication rule, and independence.
[^binomial]: MIT OpenCourseWare, 18.05 — [Discrete Random Variables: the binomial distribution](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class04-prep-a.pdf#page=7). Trial conditions and the process of counting sequences and adding their probabilities.
[^expectation]: MIT OpenCourseWare, 18.05 — [Discrete Random Variables: Expected Value](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class04-prep-b.pdf). Expected value as a weighted average and the expected value of a sum.
[^large-numbers]: MIT OpenCourseWare, 18.05 — [Central Limit Theorem and the Law of Large Numbers](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class06-prep-b.pdf). Conditions connecting the observed mean to the distribution's mean and the limits of repeated observation.
[^variance]: MIT OpenCourseWare, 18.05 — [Variance of Discrete Random Variables](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class05-prep-a.pdf). The definition of variance, standard deviation, and the variance of a sum of independent variables.
[^sample-variance]: UC Berkeley, Philip B. Stark — [Estimating Parameters from Simple Random Samples](https://www.stat.berkeley.edu/~stark/SticiGui/Text/estimation.htm). Estimation using samples and the correction for sample variance.
[^scale]: NIST/SEMATECH e-Handbook — [Measures of Scale](https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm). The squared units of variance and the distinction between standard deviation and absolute deviations.
[^binomial-reference]: NIST/SEMATECH e-Handbook — [Binomial Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda366i.htm). The binomial probability formula, mean, and standard deviation.
[^calibration]: scikit-learn official documentation — [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html). Comparing a model's predicted probabilities with actual occurrence rates.
[^loss]: scikit-learn official documentation — [log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html). Classification loss using correct labels and predicted probabilities, and average loss across examples.
