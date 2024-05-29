
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Ep4kCi5E)
# hw1-knn
# HW1: k-Nearest Neighbors
Name: Josh Dawson

CSCI 373 (Spring 2024)

Website: [https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW1/](https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW1/)

# Data Sets 

This assignment contains four data sets which are based on three publicly available benchmarks:

1. monks1.csv: A data set describing two classes of robots using all nominal attributes and a binary label.  This data set has a simple rule set for determining the label: if head_shape = body_shape  jacket_color = red, then yes, else no. Each of the attributes in the monks1 data set are nominal.  Monks1 was one of the first machine learning challenge problems (http://www.mli.gmu.edu/papers/91-95/91-28.pdf).  This data set comes from the UCI Machine Learning Repository: http://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems

2. penguins.csv: A data set describing observed measurements of different animals belonging to three species of penguins.  The four attributes are each continuous measurements, and the label is the species of penguin.  Special thanks and credit to Professor Allison Horst at the University of California Santa Barbara for making this data set public: see this Twitter post and thread with more information (https://twitter.com/allison_horst/status/1270046399418138625) and GitHub repository (https://github.com/allisonhorst/palmerpenguins).

3. mnist100.csv: A data set of optical character recognition of numeric digits from images.  Each instance represents a different grayscale 28x28 pixel image of a handwritten numeric digit (from 0 through 9).  The attributes are the intensity values of the 784 pixels. Each attribute is ordinal (treat them as continuous for the purpose of this assignment) and a nominal label.  This version of MNIST contains 100 instances of each handwritten numeric digit, randomly sampled from the original training data for MNIST.  The overall MNIST data set is one of the main benchmarks in machine learning: http://yann.lecun.com/exdb/mnist/.  It was converted to CSV file using the python code provided at: https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html

4. mnist1000.csv: The same as mnist100, except containing 1000 instances of each handwritten numeric digit.

# Research Questions

### 1. 

***Pick a single random seed and a single training set percentage (document both in your README.md) and run k-Nearest Neighbors with a k = 1 on each of the four data sets first with the Hamming distance function.  What is the accuracy you observed on each data set?***
```
seed = 1, percentage = 0.75, Hamming distance
Accuracies and time taken on lab computers:
monks1.csv - 83.33% - 0.02s
penguins.csv - 70.93% - 0.02s
mnist100.csv - 57.6% - 2.69s
mnist1000.csv - 76.36% - 285.77s
```
***Then, rerun k-Nearest Neighbors with the same seed, training set percentage, and k = 1 on only the penguins, mnist100, and mnist1000 datasets using the Euclidian distance function.  What is the accuracy you observed on each data set?  How do your accuracies compare between using the Hamming vs. Euclidian distance functions?***
```
seed = 1, percentage - 0.75. Euclidian distance
Accuracies and time taken on lab computers: 
penguins.csv - 81.40% - 0.04s
mnist100.csv - 87.60% - 9.02s
mnist1000.csv - 94.88% - 919.17s
```
Here, we can see that all accuracies increase by 10-30% when using Euclidian distance over Hamming, a gigantic increase!

### 2. 

***Using the accuracies from Question 1, calculate a 95% confidence interval around each accuracy.  Show your arithmetic that you used to calculate the intervals.***

To find the confidence interval for each accuracy, we must first find the standard errors ($SE$) using the following formula:

$$ 
SE = \sqrt{\frac{p * (1-p)}{n}},
$$

where $p$ is accuracy and $n$ is the total number of predictions made.

First we calculate the Hamming $SE$ s,

$$ 
SE_{monks} = \sqrt{\frac{0.8333 * (1-0.8333)}{108}}
= 0.036
$$

$$ 
SE_{Hpenguins} = \sqrt{\frac{0.7093 * (1-0.7093)}{86}}
= 0.049
$$

$$ 
SE_{Hmnist100} = \sqrt{\frac{0.576 * (1-0.576)}{250}}
= 0.031
$$

$$ 
SE_{Hmnist1000} = \sqrt{\frac{0.7636 * (1-0.7636)}{2500}}
= 0.008.
$$

We can calculate the Euclidian $SE$ s similarly. 

In doing so, we find $SE_{Epenguins}=0.042, SE_{Emnist100}=0.021,$ and $SE_{Emnist1000}=0.004$.

Now, we may calculate our 95% confidence interval, $I$, using the following formula:

$$
I = p \pm Z_{0.95} * SE. 
$$

Here, $Z_{0.95}$ is the $Z$ score of 95% confidence, which is $1.96$.

Plugging our $7$ calculated standard errors and accuracies into this equation, we find our $7$ confidence intervals, $I$.

Calculating the first one, 

$$
I_{monks}=[0.8333 - (1.96 * SE_{monks}),0.8333+(1.96 * SE_{monks})] = [0.76,0.90].
$$

Similarly we may calculate the other 3 Hamming distance intervals, 

$$
I_{Hpenguins}=[0.61,0.81]
$$

$$
I_{Hmnist100}=[0.51,0.64]
$$

$$
I_{Hmnist1000}=[0.75,0.78],
$$

and our 3 Euclidian distance intervals, 

$$
I_{Epenguins}=[0.73,0.90]
$$

$$
I_{Emnist100}=[0.84,0.92]
$$

$$
I_{Emnist1000}=[0.94,0.96].
$$

### 3. 

***How did your accuracy compare between the mnist100 and mnist1000 data sets when using the Euclidian distance function?  Which had the higher average?  Why do you think you observed this result?  Did their confidence intervals overlap?  What conclusion can we draw based on their confidence intervals?***

When using the Euclidian distance function, I found an accuracy of $87.60\%$ for the mnist100 data set, and an accuracy of $94.88\%$ for the mnist1000 data set. Clearly, mnist1000 had the higher average, and this makes sense, as it had $10x$ the amount of data to observe the same pattern.

In the case of this particular algorithm (k nearest neighbors), there were $\approx10x$ more neighbors for each number, so there was a lower chance that the nearest neighbor would have the wrong label.

In fact, this $10x$ increase in data was so significant that their confidence intervals ($I_{Emnist100}$ & $I_{Emnist1000}$) don't overlap. This means that $I_{Emnist1000}$, statistically significantly outperformed $I_{Emist100}$.

### 4. 

***Pick one data set and three different values of k (document both in your README.md).  Run the program with each value of k on that data set and compare the accuracy values observed.  Did changing the value of k have much of an effect on your results?  Speculate as to why or why not that observation occurred.***

I picked three values of k (1,3,5) and decided to first use mnnist100.csv to test how changing k would change the accuracy. After I saw that it barely had an effect, I tested the same values of k on two other data sets, penguins.csv and monks1.csv.

This is what I found: 
``` 
file: mnist100.csv, Euclidian distance, percentage = 0.75, seed = 1
Accuracies and k values: 
87.6%, 1
87.6%, 3
88%, 5

file: penguins.csv, Euclidian distance, percentage = 0.75, seed = 1
Accuracies and k values: 
81.4%, 1
75.58, 3
73.26%, 5

file: monks1.csv, Euclidian distance, percentage = 0.75, seed = 1
Accuracies and k values: 
83.33%, 1
87.04%, 3
92.59%, 5
```
The way this algorithm works, we find the k nearest neighbors in the training data for each point in the test set. By raising this hyperparameter, we take the average of more nearby neighbors. 

In mnist100, the first data set I picked, raising this hyperparameter had almost no impact on the accuracy. I speculate this is because the disagreement in identification of numbers was a systematic error. I believe that most of the numbers are distinctly separated from their neighbors, and therefore by increasing the number of near neighbors I consider, there is no difference in the prediction. But for what looks to be 2/9th (22%) of the data, the numbers are nearby and are conflated 50% of the time. This would mean, in considering more nearby neighbors, we are still wrong 50% the itme, leading to almost no difference in accuracy. In this case, since the data represent the numbers 1-9, this might occur if two of the numbers are grouped together often.

In penguins, I believe that since the data set is so small and (as we saw in lab2 through scikit-learn) so close together, by considering more neighbors nearby, we are grabbing the wrong species of penguin, and the more neighbors we consider, the more often we make the wrong prediction (at least for k = 1,3,5).

For monks1, since there are only two groups of data (yes and no) we can see that by raising the amount of nearby neighbors we consider, the average label is more often correct. 

## Bonus Question (Optional)

### 5. 

***Pick 10 different random seeds (document them in your README.md file) and rerun k-Nearest Neighbors with a k = 1 on the penguins.csv data.  Record the average of the accuracy across the 10 runs.***

I will run penguins with Euclidian distance and 75% training data. 

My 10 seeds: 1, 12, 123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 12345678910

Accuracy and seed number: 
```
81.40%, 1
91.86%, 12
88.37%, 123
86.05%, 1234
84.88%, 12345
79.07%, 123456
84.88%, 1234567
76.74%, 12345678
81.40%, 123456789
84.88%, 12345678910
```

***Next, rerun the program on the same 10 seeds but only consider two attributes at a time (ignoring the other two attributes not in the chosen pair).  Record the average accuracy for each pair of  attributes across the 10 seeds.  Since there are four attributes, there are six possible pairs of attributes (e.g., bill_length_mm-bill_depth_mm is one pair, so flipper_length_mm and body_mass_g would be ignored for this pair).***

The 4 possible attributes: bill_length_mm (bill_len), bill_depth_mm (bill_d), flipper_length_mm (flip), body_mass_g (mass).

Average accuracy for bill_len and bill_d: 
```
93.02%, 1
89.53%, 12
95.35%, 123
96.51%, 1234
93.02%, 12345
94.19%, 123456
94.19%, 1234567
89.53%, 12345678
95.35%, 123456789
91.86%, 12345678910
```
Average accuracy for bill_len and flip: 
```
95.35%, 1
98.84%, 12
95.35%, 123
91.86%, 1234
95.35%, 12345
96.51%, 123456
95.35%, 1234567
90.70%, 12345678
91.86%, 123456789
91.86%, 12345678910
```
Average accuracy for bill_len and mass: 
```
79.07%, 1
86.05%, 12
84.88%, 123
84.88%, 1234
88.37%, 12345
79.07%, 123456
81.40%, 1234567
76.74%, 12345678
84.88%, 123456789
88.37%, 12345678910
```
Average accuracy for bill_d and flip: 
```
79.07%, 1
72.09%, 12
75.58%, 123
79.07%, 1234
75.58%, 12345
73.26%, 123456
74.42%, 1234567
73.26%, 12345678
76.74%, 123456789
74.42%, 12345678910
```
Average accuracy for bill_d and mass: 
```
68.60%, 1
76.74%, 12
68.60%, 123
72.09%, 1234
65.12%, 12345
60.47%, 123456
63.95%, 1234567
58.14%, 12345678
69.77%, 123456789
72.09%, 12345678910
```
Average accuracy for mass and flip: 
```
80.23%, 12
76.74%, 123
79.07%, 1234
72.09%, 12345
68.60%, 123456
72.09%, 1234567
69.77%, 12345678
70.93%, 123456789
70.93%, 12345678910
```
***Finally, compare the average accuracy results between (1-6) all six pairs of attributes and (7) the results using all four attributes.  Did any pairs of attributes do as well (or better) than learning using all four attributes?  Speculate why you observed your results.***
 
By taking all permutations of the four attributes, we can see which ones are strongly and weakly correlated with the species of penguin. Some of the permutations were weaker and some were stronger, the accuracy found using all four lies in the middle. 

Based off the results above, I think that mass was weakly correlated among the penguins. Whenever a permutation had mass it seemed to do worse than the average of the four. But to answer the question above, bill_len and bill_d, and bill_len and flip did noticeably better than the average and I speculate that these are attributes of the penguins that are strongly correlated with species.

# Additional Questions

Please answer these questions after you complete the assignment:

1. ***What was your experience during the assignment (what did you enjoy, what was difficult, etc.)?***

    I found the assignment to be quite fun. For me, it was in between the intro course labs, where we are walked through the assignment, and a research/winter term, where we have a task but no route to success. I struggled with optimizing once I had created code that worked, especially since I was running this on a MacBook Air (not exactly known for its processing power). However, once I ran this on the lab machines, it was a lot smoother.
    
    I will say, it would have been nice to have a test set of data with a test output for a given seed, that way we know what to aim for/have something to compare against. I believe, since we are using a seed to shuffle, we should be able to recreate the same outcome given the same input. This would allow for verification to make sure there are no bugs/small errors in our algorithm. Otherwise, we are left to guess on our own when we think we have reasonable results. I do understand that we don't know this information in a real world scenario, but for the first hw it would have been a nice guide.

2. ***Approximately how much time did you spend on this assignment?***
    
    8-9+ hours (I spent a lot of time fiddling with things to fix runtime when it was a hardware issue...)

3. ***Did you adhere to the Honor Code?***

    Yes
