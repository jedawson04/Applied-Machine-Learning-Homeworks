[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/u5RbYg_2)
# HW4: Classification and Regression with Neural Networks
Name: Josh Dawson and Mario Stinson-Maas

CSCI 373 (Spring 2024)

Website: [https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW4/](https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW4/)

# Data Sets 

This assignment contains four data sets that are based on publicly available benchmarks:

1. **mnist1000.csv**: A data set of optical character recognition of numeric digits from images.  Each instance represents a different grayscale 28x28 pixel image of a handwritten numeric digit (from 0 through 9).  The attributes are the intensity values of the 784 pixels. Each attribute is ordinal (treat them as continuous for the purpose of this assignment) and a nominal label.  This version of MNIST contains 1,000 instances of each handwritten numeric digit, randomly sampled from the original training data for MNIST.  The overall MNIST data set is one of the main benchmarks in machine learning: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).  It was converted to CSV file using the python code provided at: [https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html](https://quickgrid.blogspot.com/2017/05/Converting-MNIST-Handwritten-Digits-Dataset-into-CSV-with-Sorting-and-Extracting-Labels-and-Features-into-Different-CSV-using-Python.html)

2. **penguins.csv**: A data set describing observed measurements of different animals belonging to three species of penguins.  The four attributes are each continuous measurements, and the label is the species of penguin.  Special thanks and credit to Professor Allison Horst at the University of California Santa Barbara for making this data set public: see this Twitter post and thread with more information [https://twitter.com/allison_horst/status/1270046399418138625](https://twitter.com/allison_horst/status/1270046399418138625) and GitHub repository [https://github.com/allisonhorst/palmerpenguins](https://github.com/allisonhorst/palmerpenguins).

3.	**energy.csv**: A data set describing the energy consumption in 10-minute increments by appliances in a low-energy residence in Belgium.  The task is to predict how much energy was consumed by appliances.  Each of the 27 attributes are numeric and describe measurements from sensors in the residence or nearby weather stations, as well as energy usage by lights.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

4.	**seoulbike.csv**: Another data set describing bike rentals in a metropolitan area (Seoul, South Korea).  Again, the task is to predict how many bikes will be rented hourly throughout the day over a two-year period.  The 11 attributes are a mix of 2 categorical and 9 numeric attributes, including information such as the season, whether it was a holiday, and current weather conditions.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

# Research Questions

Please answer the following research questions using the `neuralnet.py` program that you created.

#### Question 1

Choose a random seed and training set percentage (document them here as part of your answer).  Using those choices, train neural networks for the `penguins.csv` and `mnist1000.csv` data sets with each of the following numbers of hidden neurons `[8, 16, 32, 64, 128, 256]` and learning rates `[0.00001, 0.0001, 0.001, 0.01, 0.1]`.

Random Seed: 12345
Training Split: 75%

a. Using the resulting accuracies from the 30 neural networks trained for each data set, create one **line chart** for each data set where the x-axis is the number of neurons, the y-axis is the accuracy of the neural network on the test set, and there are separate lines for each learning rate.  Save the two line charts to image files with names that clearly describe to which data set they correspond (e.g., `penguins_line.png`).  Upload those two image files to your GitHub repository.

Seen in figures as penguins_line.png and mnist1000_line.png

b. For each classification data set, what general trend do see as the *number of neurons increases*?  Did performance improve or worsen?  Why do you think this occurred?

penguins: In general, it seems that the number of neurons had little effect on the accuracy of the model past around 50. There were some discrepancies in the model with the smallest learning rate, but this could be seed-dependent here; we aren't quite sure why the accuracy spiked at 64 neurons. We think that the accuracy is mostly unaffected by the number of neurons when there are greater than 50 because penguins is such a small dataset, only consisting of 4 attributes and 3 labels.

mnist: As neurons increase, we notice that performance increases if the learning rate is sufficiently small, i.e. at 0.1 or higher (we theorize) the accuracy can be harmed when neurons are added. Every model had an increase in accuracy when moving from 4 to 8 neurons, and again from 8 to 16. The rest of the learning rates all increased with added neurons, but the performance improvement tapers off past 100 neurons. It makes sense that this cutoff is larger than as noted in penguins, because mnist has many more attributes, and more labels as well. 

c. For each classification data set, what general trend do you see as the *learning rate increases*?  Did performance improve or worsen?  Why do you think this occurred?

penguins: We noticed that the learning rate generally improved as the learning rate increased; the models with the two largest learning rates performed identically, we think, both at 100% accuracy. We think this occurs because the larger learning rates allows the models to search past local minima and find the true global minimum. There are multiple ways of comparing these rates, and multiple (area under curve, pointwise comparison) hold for each of our curves.

mnist: We note here that increasing learning rate doesn't necessarily improve performance. In fact, we can see that the most extreme learning rates (largest and smallest) have the two worst performances. Since a learning rate is proportional to the amt we change each weight by in backpropogation this makes sense. By choosing a learning rate in the middle, we are less likely to jump over or get stuck in a local/global minima. We believe this differs from penguins due to the larger size of the dataset, since there are likely more local minima. This means, unlike in penguins, where raising the learning rate always improved the performance, a learning rate in the middle is more effective.

#### Question 2

Choose a random seed and training set percentage (document them here as part of your answer).  Using those choices, train neural networks for the `energy.csv` and `seoulbike.csv` data sets with each of the following numbers of hidden neurons `[8, 16, 32, 64, 128, 256]` and learning rates `[0.00001, 0.0001, 0.001, 0.01, 0.1]`.

a. Using the resulting MAEs from the 30 neural networks trained for each data set, create one **line chart** for each data set where the x-axis is the number of neurons, the y-axis is the MAE of the neural network on the test set, and there are separate lines for each learning rate.  Save the two line charts to image files with names that clearly describe to which data set they correspond (e.g., `energy_line.png`).  Upload those two image files to your GitHub repository.

Seen in figures as seoulbike_line.png and energy_line.png

b. For each regression data set, what general trend do see as the *number of neurons increases*?  Did performance improve or worsen?  Why do you think this occurred?

energy: When looking at the energy plot, nothing jumps out more than the huge drop in MAE from 0 - 64 neurons with the smallest learning rate. There are similar, albeit much smaller, drops in MAE by other learning rates within the first 64 neurons. Generally across all learning rates the the MAE slowly increases and starts to plateu from 64-256 neurons. We aren't sure exactly why this spike occurred at 64, but I think it's clear that with less than 64 neurons, the model is not able to perform as well to represent the data. This isn't that surprising as the energy data set has so many attributes (27 if we counted correctly) and is 20000 lines long, so it will need more neurons to properly encompass all patterns in the attributes.

seoulbike: The dramatic spike present in the smallest learning rate in energy_line.png, is the norm in seoulbike. Across all learning rates there are dramatic decreases in MAE from 0 to 64 neurons. For many of the learning rates, this MAE continues to drop/plateu (but not increase!) as more neurons are added. Seoulbike also has many attributes (although less than energy), which could explain why there are large drops from 0 to 64, but we're not sure there continues to be drops in MAE as the number of neurons goes up, unlike in energy.

c. For each regression data set, what general trend do you see as the *learning rate increases*?  Did performance improve or worsen?  Why do you think this occurred?

energy: As the learning rate increases (from 0.00001 -> 0.1) the MAE drops pretty consistently. The only exception to this is the largest learning rate of 0.1 which starts off lower than the second lowest, but ends up with a higher MAE. Although, this could be do to the randomness associated with the training. This trend makes sense as, with such a large data set, (and so many attributes) finding the global minima will require having a larger learning rate.

seoulbike: In seoulbike the same trends occured as in energy except on a much larger scale and with no exceptions. Here, the lower the learning rate, the higher the MAE, and there is no overlap across there lines, even with an increase in neurons. This can similar be understood as the larger learning rates having a larger chance to find the global minima (or a smaller chance to get stuck in a local minima).

#### Question 3

a. Based on your answers to Questions 1 and 2 above, which do you think is more important to optimize first -- the learning rate or the number of neurons in the network?  Why is it more important to optimize that hyperparameter first?

We think one should first optimize the learning rate; however we note that performance also depends on the static value held for the other one generally 50-100 neurons performed the best. Accuracy generally plateus as neurons increase past a certain point, whereas learning rate has a larger impact on performance. The middle seems to perform the best, i.e. the learning rates from .0001 to .01 had the highest accuracies.

b. Based on your answers to Questions 1 and 2 above, when might we want to start with a small learning rate (closer to 0)?  When might we want want to start with a larger learning rate (closer to 1)?

We can see a very noticeable performance improvement with higher learning rates in seoulbike -- and regression tasks in general. Lower learning rate never seems to be great; but in theory, if there are a lot of really close local minima, or if the only local minima is the global one, a low learning rate will ensure this is found. 

c. Based on your answers to Questions 1 and 2 above, when might we want to start with a small number of neurons?  When might we want want to start with a larger number of neurons?

If we are using a high learning rate, starting with less neurons is okay; it doesn't have as much of an impact. However, in general, we need to consider the complexity of the datasets we are working with. For example, the number of attributes and possible labels for a classification task would give us information regarding how many neurons we might need; if the data is not complex, there is no need using more neurons than are necessary. With a larger dataset, however, it is worth it to start with at least 50 neurons.

#### Question 4

a. Using the `penguins.csv` data set, a number of hidden neurons of `128`, a training percentage of `0.75`, and a random seed of `54321`, create two line charts that demonstrate the performance on the **training** and **validation** sets during training: one line chart for a learning rate of `0.01` and another for a learning rate of `0.00001`.  As in Lab 5, the x-axis should be the epoch, the y-axis should be the loss and accuracy, and there should be four lines -- one each for the `[loss, val_loss, accuracy, val_accuracy]` tracked by the `CSVLogger` during training.  Save the two line charts to image files with names that clearly describe to which data set they correspond (e.g., `penguins_0.01.png`).

Seen in figures as penguins_0.01.png and penguins_1e-05.png

b. Similarly, using the `seoulbike.csv` data set, a number of hidden neurons of `128`, a training percentage of `0.75`, and a random seed of `54321`, create two line charts that demonstrate the performance on the **training** and **validation** sets during training: one line chart for a learning rate of `0.01` and another for a learning rate of `0.00001`.  The x-axis should be the epoch, the y-axis should be the loss and MAE, and there should be four lines -- one each for the `[loss, val_loss, mean_absolute_error, val_mean_absolute_error]` tracked by the `CSVLogger` during training.  Save the two line charts to image files with names that clearly describe to which data set they correspond (e.g., `seoulbike_0.01.png`).

Seen in figures as seoulbike_0.01.png and seoulbike_1e-05.png -- (within this set the loss drowned out the MAE)

c. How did the accuracy curves of the two learning rates differ for the `penugins.csv` classification task?  How does this compare to the results you observed in Question 1?

The chart that used learning rate 1e-05 shows a 40% accuracy for the first ~200 epochs. This indicates that the function was only slightly better than the null model, which predicts the most common species in the dataset. Only after epoch 200 does the accuracy jump up to around 70%. In stark contrast, with the higher learning rate, the accuracy started around the same 40% but then quickly jumped up to around 95-98%. This is clearly indicative of the fact that 1e-05 is simply too low of a learning rate for this smaller dataset. This agrees with our observation in Question 1 that for the penguins dataset, learning rate has a much larger impact than number of neurons.

d. How did the loss curves of the two learning rates differ?  Were there any common trends across the data sets? Did any level off to a near-constant value?  Were any continually decreasing? 

In the penguins dataset, the loss functions continually declined; there were some sharp variations within the validation loss function, which stops decreasing around epoch 100.

It is tricky to determine a change in the seoulbike dataset loss functions, as the common scale destroys the information on the smaller scale.

e. What do the shapes of the loss curves for the two learning rates imply?  Is there a relationship between the number of epochs needed for training a neural network model and the learning rate used?

Yes; it appears that higher learning rates need less time to be "fully trained", where the change in performance from epoch to epoch is no longer significant. The lower learning rates actually were incapable of improving accuracy until around the 200th epoch.


# Additional Questions

Please answer these questions after you complete the assignment:

1. What was your experience during the assignment (what did you enjoy, what was difficult, etc.)?
    We had a lot of fun putting things together, sorting the methods and making the graphs look pretty. 
    However, Waiting for errors = :( makes us want to look into compiled python options.
    Also, trying to determine where the errors are coming from when sorting through 10 libraries of error calls was annoying.
2. Approximately how much time did you spend on this assignment?
    Counting waiting -- 20 ish hours
3. Did you adhere to the Honor Code?
    Yes

