[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/42rKmsKG)
# HW2: Decision Trees and Random Forests
Name: Josh Dawson and Mario Stinson-Maas

CSCI 373 (Spring 2024)

Website: [https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW2/](https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW2/)

# Data Sets 

This assignment contains five data sets that are based on publicly available benchmarks:

1. **banknotes.csv**: A data set describing observed measurements about banknotes (i.e., cash) under an industrial print inspection camera.  The task in this data set is to predict whether a given bank note is authentic or a forgery.  The four attributes are each continuous measurements.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/banknote+authentication](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

2. **mnist1000.csv**: A data set of optical character recognition of numeric digits from images.  Each instance represents a different grayscale 28x28 pixel image of a handwritten numeric digit (from 0 through 9).  The attributes are the intensity values of the 784 pixels. Each attribute is ordinal (treat them as continuous for the purpose of this assignment) and a nominal label.  This version of MNIST contains 1,000 instances of each handwritten numeric digit, randomly sampled from the original training data for MNIST.  The overall MNIST data set is one of the main benchmarks in machine learning: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).  It was converted to CSV file using the python code provided at: [https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html](https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html)

3. **occupancy.csv**: A data set of measurements describing a room in a building for a Smart Home application.  The task in this data set is to predict whether or not the room is occupied by people.  Each of the five attributes are continuous measurements.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+)

4. **penguins.csv**: A data set describing observed measurements of different animals belonging to three species of penguins.  The four attributes are each continuous measurements, and the label is the species of penguin.  Special thanks and credit to Professor Allison Horst at the University of California Santa Barbara for making this data set public: see this Twitter post and thread with more information [https://twitter.com/allison_horst/status/1270046399418138625](https://twitter.com/allison_horst/status/1270046399418138625) and GitHub repository [https://github.com/allisonhorst/palmerpenguins](https://github.com/allisonhorst/palmerpenguins).

5. **seismic.csv**: A data set of measurements describing seismic activity in the earth, measured from a wall in a Polish coal mine.  The task in this data set is to predict whether there will be a high energy seismic event within the next 8 hours.  The 18 attributes have a mix of types of values: 4 are ordinal attributes, and the other 14 are continuous.  The label is “no event” if there was no high energy seismic event in the next 8 hours, and “event” if there was such an event.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/seismic-bumps](https://archive.ics.uci.edu/ml/datasets/seismic-bumps)

# Research Questions

Please answer the following research questions using the `trees.py` program that you created.  Show your work for all calculations.

#### Question 1

Choose a random seed and training set percentage (document them here as part of your answer).  Using those choices, train a single tree for each of the five data sets.  

***The following was observed with a training set percentage of 0.75 and a seed of 1234.***

a. What is the accuracy of the test set predictions of each tree that you learned for each of the five data sets?

*To calculate accuracy we used the following formula a = c/t, where the accuracy, a, is equal to the correct predictions / total predicted on. The numbers below were calculated using with line 209*

- banknotes: 99.71%

- mnist1000: 79.80%

- occupancy: 98.95%

- penguins: 95.35%

- seismic: 87.46%

b. Calculate the 95% confidence interval around each accuracy in your answer to Question 1a.

*This was done here the same way as it was in hw 1. We first calculated the standard error and then used this to calculate the confidence interval. The exact operations used to calculate the intervals below can be seen on lines 211 and 212 of trees.py.*

- banknotes: Calculated Standard Error: 0.0029. Calculated 95% Confidence Interval: [0.9914, 1].

- mnist1000: Calculated Standard Error: 0.008. Calculated 95% Confidence Interval: [0.7823, 0.8137].

- occupancy: Calculated Standard Error: 0.0014. Calculated 95% Confidence Interval: [0.9868, 0.9922].

- penguins: Calculated Standard Error: 0.0227. Calculated 95% Confidence Interval: [0.909, 0.998].

- seismic: Calculated Standard Error: 0.013. Calculated 95% Confidence Interval: [0.8491, 0.9001].

#### Question 2

Using your program, visualize the tree for the penguins.csv data set learned in Question 1a.  Upload the tree visualization to GitHub as an image file. 

**Decision tree vizualization file is 'tree_penguins_1t_75p_1234'**

a. What rules were learned by this tree?

Here are the first 10 rules from left to right, per your instructions for listing them.

- bill length <= 39.5, 
bill depth <= 16.65, 
flipper length <= 207.5,
On this branch, the label predicted is Adelie

- 39.5 <= bill length <= 42.4,
bill depth <= 16.65, 
flipper length <= 207.5,
This predicted label is Chinstrap

- bill length <= 42.4, 
bill depth >= 16.65, 
flipper length <= 207.5,
This predicted label is Adelie

- 42.4 <= bill length <= 43.35,
flipper length <= 189.5,
This predicted label is Chinstrap

- 42.4 <= bill length <= 43.35,
189.5 <= flipper length <= 207.5,
This predicted label is Adelie

- bill length >= 43.35,
body mass <= 4125, 
flipper length <= 207.5,
This predicted label is Chinstrap

- 43.35 <= bill length <= 48.3,
body mass >= 4125, 
flipper length <= 207.5,
This predicted label is Adelie

- bill length >= 48.3,
body mass >= 4125, 
flipper length <= 207.5,
This predicted label is Chinstrap

- bill depth <= 18.1, 
flipper length >= 207.5,
This predicted label is Gentoo

- bill length <= 44.9
bill depth >= 18.1, 
flipper length >= 207.5,
This predicted label is Adelie

b. How do these rules relate to your analysis in Lab 2 when you compared the average values of each attribute for each label (e.g., how the average bill length differed between the three species)?

Many of these rules relate to (or in some cases are the same as) numbers that are at or near the averanges observed in Lab 2. This is especially true for rules that had much higher 'values', meaning more instances fit the rules. This makes sense as the rules that can be thought of as the 'biggest buckets' of penguin indentification would be more in line with the averages observed. We also observed that the body mass of the penguin was used in very few rules and thus didn't have a very high correlation with accuracy (this was also observed in the bonus question for hw1). This issue is likely due to the large differences between body masses, which are greatly exaggerated by squaring the difference in the Euclidian metric (which was touched on in Lab 4). 

#### Question 3

Use a seed of 1234 and a training set percentage of 75%. Train a single tree to classify only the mnist1000.csv data set. 

a. Calculate the recall for your tree for each label.

- zero: 85.53%

- one: 89.84%

- two: 78.49%

- three: 76.65%

- four: 79.1%

- five: 79.56%

- six: 79.27%

- seven: 84.67%

- eight: 75.98%

- nine: 69.78%

b. Which label had the *lowest* recall?  Why do you think this label had the lowest?

*nine* - a low recall value means that the model struggled to identify nines, which makes sense since they look a lot like other numbers. In particular, ones, eights, and sevens.


c. Which label had the *highest* recall? Why do you think this label had the highest?

*one* - a high recall value means the model succeeded at identifing ones, this means that the ones stood out more than other numbers and were not confused for them less often. This might be because 

#### Question 4

Use a seed of 1234 and a training set percentage of 75%. Train a single tree to classify only the seismic.csv data set.

a. Calculate the recall for your tree for each label.

- event: 8.89%

- no event: 93.34%

b. What do you think these recalls imply about the usefulness of your model?

- These recalls imply that the model is not very useful at all; less than 10% of the time when we need to be warned about seismic activity does this model actually predict this.

c. Based on the data in the seismic.csv data set and the counts in your confusion matrix, why do you think this trend between the two recalls occurred?

- Since there are many more instances of *no event* than *event* (as would be expected), the model seems to predict no event almost all of the time which leads to the recalls noted above. 

#### Question 5

Using a seed of 1234, train a tree for each of the training percentages in [0.2, 0.4, 0.6, 0.8] with the mnist1000.csv, occupancy.csv, and penguins.csv data sets.  Plot the accuracies of the trees as a line chart, with training percentage as the x-axis, accuracy as the y-axis, and separate lines (in the same chart) for each data set.  Upload your line chart to GitHub as an image file.

**Line charts are 'Q5_line_plot' and 'Q5_line_plot_scaled'**

a. What trends do you see for each data set?

- mnist1000: Mnist1000 increases the most overall, especially in the beginning, before it tapers off.  

- occupancy: Occupancy has no (visible) increase in accuracy as the training percent is increased.

- penguins: Penguins had a large increase in accuracy but not as much as mnist1000.

b. Why do you think those trends occurred for each data set?

- As the training percentage is increased the increase in accuracy tapers off and settles around 80 ish%. I wonder if this is because of the limitations of the model on this data set and the conflation of two of the numbers. 

- Occupancy has no increase, which might mean the model identifies the same patterns no matter the percent of training used. Also there are 20000 data points, so 20% of the data is still 5000 data points, which is sufficient to identify the trends present.

- It makes sense there is an increase in accuracy when more data is used for the penguins set since the set is so small, this data set also tapers off but around 95%. I'm not sure why 95% specifically, but I believe similar to mnist1000, this is a limitation in the model where 5-10% of the penguin data is conflated with another penguin. It would be interesting to identify the incorrect predictions in the tree and see if there are commonalities between them, since the data set is so small.

#### Question 6

Using the same random seed and training percentage as Question 1, train a forest with 100 trees for each of the five data sets.

a. What is the accuracy of the test set predictions of each forest that you learned for each of the five data sets?

*Accuracy here was calculated the same as it was in 1a*

- banknotes: 99.71%

- mnist1000: 95.16%

- occupancy: 99.18%

- penguins: 96.51%

- seismic: 92.88%

b. Calculate the 95% confidence interval around each accuracy in your answer to Question 6a.

*Confidence Intervals here were calculated the same as  in 1b*

- banknotes: Calculated Standard Error: 0.0029. Calculated 95% Confidence Interval: [0.9914, 1].

- mnist1000: Calculated Standard Error: 0.0043. Calculated 95% Confidence Interval: [0.9432, 0.96].

- occupancy: Calculated Standard Error: 0.0013. Calculated 95% Confidence Interval: [0.9893, 0.9944].

- penguins: Calculated Standard Error: 0.0198. Calculated 95% Confidence Interval: [0.9263, 1].

- seismic: Calculated Standard Error: 0.0101. Calculated 95% Confidence Interval: [0.909, 0.9486].

#### Question 7

Compare the confidence intervals for each data set between Questions 1b and 6b.

a. For each data set, did any model (tree or forest) statistically significantly outperform the other?

- banknotes: no 

- mnist1000: yes 

- occupancy: no 

- penguins: no

- seismic: yes

b. Based on your answer to Question 7a, what conclusions can you draw about the usefulness of training more than one tree in a random forest?

More trees allows us to water down any inconsistencies that might be present in a single tree. In each case this should give us more accurate predictions because averaging the results of multiple trees will minimize the random error in a given tree.

#### Question 8

Using a seed of 1234, a training percentage of 0.75, and each of the numbers of trees in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], train a random forest for only the mnist1000.csv data set.  Plot the accuracies of the forest as a line chart, with number of trees as the x-axis and accuracy as the y-axis.  Upload your line chart to GitHub as an image file.

**Line charts are 'Q8_line_plot' and 'Q8_line_plot_scaled'**

a. What trend do you observe? 

There is a sharp increase in accuracy that tapers off around 30 trees. 

b. Why do you think this trend occurred for this data set? 

We think the increase is due to the increased number of trees. Any potential inaccuries/inconsistenties are watered down by the larger amount of trees. We think that the taper off is due to a limitation in the number of patterns that are identifiable in the numbers (variance between people's handwritings). 
 
# Additional Questions

Please answer these questions after you complete the assignment:

1. What was your experience during the assignment (what did you enjoy, what was difficult, etc.)?

We had a lot of fun implementing the methods, it was cool to see how much less code it was, and the documentation was super useful for understanding how the methods work, although it was a little tedious to read at times. Some of the plot stuff was a little tricky, but the labs were really helpful in showing us how the graphs work. I (josh) personally had a lot of fun making the graphs, and scaling them and analyzing them. The data analysis/visualization was super enjoyable. 

2. Approximately how much time did you spend on this assignment?

7-8 hours each.

3. Did you adhere to the Honor Code?
    
Yes
