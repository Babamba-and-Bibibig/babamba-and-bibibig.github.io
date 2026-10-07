+++
# The title shown in the post list and on the individual post page.
title = "Weekly Paper 01"

# A short introduction shown in the post list.
description = "Weekly research on AI engineering skills, date-handling modules, PyTorch model definitions, and customer data processing."

# Keep the original publication date and record when the writing structure was updated.
date = 2026-09-24
updated = 2026-09-25
slug = "weekly-paper-01"
aliases = ["/en/first-post/"]
+++

> **Work in progress** · This page sets out this week's questions and writing structure. I am filling in the research and answers myself.

This is a weekly study log where I research the assigned questions on Google and write up what I have understood in my own words.

## 01. The Most Important Skill for an AI Engineer

> What do you think is the most important skill for an AI engineer? Choose either technical or nontechnical skills and explain your reasons in your own words.

### Research Notes

<!-- Record what you have found through your own searches, separately from the draft answer. -->
Definitions of technical and nontechnical skills for developers

Technical skills

1. Programming skills
2. Architecture design
3. Managing code and maintaining its quality

Nontechnical skills

1. Communication skills
2. Problem-solving and analytical skills
3. Collaboration and teamwork
4. Domain knowledge beyond coding

### My Answer

<!-- Write about your chosen skill and your reasons in your own words. -->
Both are important, but if I had to choose just one these days, I think nontechnical skills might matter more.
AI can help with writing code and looking up specialized information, but communicating directly with people is not something that is easy to improve through theory alone.
I also think that knowing the field you work in helps you be more specific about what to ask AI and how to ask it.
These days, AI can answer questions about things we want to know. But when I do not even know what I am missing or what I should ask, coming up with the right question is difficult in the first place.

## 02. What If an Existing Module Is Missing Just One Feature?

> You need a way to process date data, and you find an existing module that already supports most of what you need, such as calculating days of the week and intervals between dates. But it is missing just one thing: support for an unusual date format used only at your company, such as '25년 3월 첫째 주' ('the first week of March ’25'). What would you do?

### Research Notes

<!-- Record what you have found through your own searches. -->
Can a Python date module be customized?

The standard Python `datetime` module itself cannot be modified directly.
However, it is possible to implement custom behavior by creating a subclass.
You can inherit from `datetime.datetime` to add new features or change its behavior.

```python,linenos
from datetime import datetime
class CustomDateTime(datetime):
    def to_kr_string(self):
        # Add a custom output method for the desired format.
        # The Korean labels mean year, month, day, hour, and minute.
        return self.strftime("%Y년 %m월 %d일 %H시 %M분")

now = CustomDateTime.now()
print(now.to_kr_string())
```

### My Answer

<!-- Explain which approach you would choose and why, in your own words. -->
Following the advice not to reinvent the wheel, I think it is best to make full use of an existing module when it supports what I need, and make only the smallest necessary changes or additions for anything it does not support.

## 03. Why Define a PyTorch Model as a Class?

> When building a deep learning model in PyTorch, why would you define the model as a class? Explain by comparing this approach with an implementation that uses only functions.

### Research Notes

<!-- Record what you have found through your own searches. -->
Reasons to represent data using classes

1. Structuring data and state: Related variables such as a name, age, and contact information can be grouped into a single user-defined data type.
This makes the code simpler and easier to read than managing each variable separately.

2. Reusability and extensibility: Multiple instances with the same data structure can be created and used efficiently.
You can define both the data and the functions related to it, called methods, together and handle them consistently.

3. Easier maintenance: When the data format needs to change, you can update the definitions inside the class, reducing the need to make changes throughout the code.
In practice, classes are widely used as DTOs for transferring data or as database entities.

### My Answer

<!-- Explain why a model is defined as a class and compare it with an implementation using only functions. -->
I think that to understand why a deep learning model is defined as a class in PyTorch, it helps to first consider why we group data with the functions that work on it.
There are several reasons to use classes, but I think being able to keep related data and functions in one place is important.
In particular, when data needs to change, I see an advantage in keeping the rules for those changes together with the functions that carry them out.

A model needs not only the calculations that process its input, but also values such as the weights that are adjusted during training.
The weights registered with the model and the calculations can be managed within one object, and the values needed for training can be retrieved together.
This means using the model management features provided by PyTorch, as well as using a class.

So I think the reason to define a model as a class is that it makes it easier to manage the model's structure and the data needed for training together, rather than because it would be impossible to build one with functions alone.

## 04. Improving Code That Processes Data for One Million Customers

> A colleague's code uses Python lists and loops to process data for one million customers, and each run takes several minutes. Diagnose how the code could be improved and suggest specific changes.

### Research Notes

Using a loop with the NumPy library

```python,linenos
import numpy as np

arr = np.array([1, 2, 3, 4])
for x in arr:
    print(x)
```

### My Answer

<!-- Write your diagnosis and specific suggestions for improvement in your own words. -->

In a real work situation, I might ask AI for help, but before that I would try to understand why the loop is needed and what calculation it is repeating.

First, I would check whether the same calculation is being repeated unnecessarily. For example, if a value that applies equally to every customer is being calculated again each time, it could be moved outside the loop and calculated just once.
Also, if only customers who meet certain conditions need additional processing, I would check those conditions first to avoid unnecessary calculations for the other customers.

I would also look into using NumPy, a library designed to work with large amounts of data.
NumPy's array operations provide convenient access to optimized internal computations, so I would consider whether NumPy could be used here.
