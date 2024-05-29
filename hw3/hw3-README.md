[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/l1FvqcXt)
# HW3: Regression with Weighted Models
Names: Josh Dawson and Mario Stinson-Maas

CSCI 373 (Spring 2024)

Website: [https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW3/](https://cs.oberlin.edu/~aeck/Spring2024/CSCI373/Assignments/HW3/)

# Data Sets 

This assignment contains five data sets that are based on publicly available benchmarks:

1.	**capitalbike.csv**: A data set describing bike rentals within the Capital bikeshare system.  The task is to predict how many bikes will be rented hourly throughout the day over a two-year period.  The 12 attributes are a mix of 6 categorical and 6 numeric attributes, including information such as the season, day of the week, whether it was a holiday, and current weather conditions.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/data sets/275/bike+sharing+data sets](https://archive.ics.uci.edu/data sets/275/bike+sharing+data sets)

2.	**seoulbike.csv**: Another data set describing bike rentals in a metropolitan area (Seoul, South Korea).  Again, the task is to predict how many bikes will be rented hourly throughout the day over a two-year period.  The 11 attributes are a mix of 2 categorical and 9 numeric attributes, including information such as the season, whether it was a holiday, and current weather conditions.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/data sets/560/seoul+bike+sharing+demand](https://archive.ics.uci.edu/data sets/560/seoul+bike+sharing+demand)

3.	**energy.csv**: A data set describing the energy consumption in 10-minute increments by appliances in a low-energy residence in Belgium.  The task is to predict how much energy was consumed by appliances.  Each of the 27 attributes are numeric and describe measurements from sensors in the residence or nearby weather stations, as well as energy usage by lights.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/data sets/374/appliances+energy+prediction](https://archive.ics.uci.edu/data sets/374/appliances+energy+prediction)

4.	**forestfires.csv**: A data set describing forest fires in northeastern Portugal.  The task is to predict how much area was burned by forest fires.  The 12 attributes are a mix of 2 categorical and 10 numeric values, including date and weather data, as well as the geographic location of the area within the Montesinho park.  This data set comes the UCI Machine Learning Repository: [https://archive.ics.uci.edu/data sets/162/forest+fires](https://archive.ics.uci.edu/data sets/162/forest+fires)

5.	**wine.csv**: A data set of measurements of wine samples.  The task is to predict the quality of the wine (on a numeric scale).  The attributes are a mix of 11 numeric measurements from the wine, along with 1 categorical attribute describing the color of the type of wine.  This data set is the most popular regression task from the UCI Machine Learning Repository: [https://archive.ics.uci.edu/data sets/186/wine+quality](https://archive.ics.uci.edu/data sets/186/wine+quality)


# Research Questions

Please answer the following research questions using the `regression.py` program that you created.

#### Question 1

Choose a random seed and training set percentage (document them here as part of your answer).  Using those choices and a value of `true` for whether you want to rescale the numeric attributes using max-min normalization, train the requested eight models on each of the five data sets and calculate the MAEs on the corresponding test sets.  (You do *not* have to list those MAEs here)

Seed: 12345,
Training %: 75

a. Using the resulting MAEs, create one **bar chart** for each data set and save the five bar charts to image files that clearly describe to which data set they correspond (e.g., `capitalbike_rescaled_bar.png`).  Upload those five image files to your GitHub repository.

*See '../Figures/hw3-figures/bar_rescaled/'*

b. Comparing the performance of Linear Regression and Ridge Regression across the five data sets, what trends do you observe?  Does adding regularization improve the performance of Ridge Regression over Linear Regression?  Why do you think this result occurred?

They are almost exactly the same, with and without the regularization term. With the specific seed we chose, it looks like ridge performs better in forestfires, but it doesn't appear to be a significant difference. Since adding regularization terms helps with overfitting the data, and we didn't struggle with that, it wouldn't affect our results; we didn't really see a difference between the two.

c. Comparing the performance of Linear Regression and LASSO across the five data sets, what trends do you observe?  Does adding regularization improve the performance of LASSO over Linear Regression?  Why do you think this result occurred?

Linear regression and LASSO yield very similar performances across the 5 data sets; if any difference, we generally note that Linear regression performs better than LASSO, except for in the forestfires data sets. We guessed that the similar performance follows from the same reason as Ridge, that regularization didn't benefit the data. We're not sure why exactly LASSO did slightly worse, but since it's able to set weights to 0, perhaps it would sometimes zero weights that would have been helpful to have in small amounts instead of zero. Overall, LASSO performs similarly to both Linear and Ridge Regression.

d. Comparing the performance of Linear Regression and the four SVM models across the five data sets, what trends do you observe?  Do the improvements of the SVM approach lead to improved predictions?

The SVM models outperform linear regression in nearly every case we came across. In the capitalbike data sets, we saw worse performance from the SVM RBF model than linear, although the other SVM models still did better than linear model. The forestfires data sets saw the largest performance improvement in SVM over linear, SVMs having around 60% of linear's MAE on average. In general, the ability of the SVM approach to transform attributes lead to improved predictions.

e. Comparing the performance of the different kernels within the four SVM models across the five data sets, what trends do you observe?  Is one kernel a better choice for these data sets than the other?  How does the choice of degree for the Polynomial kernel affect learning performance?

Generally, The RBF model performs the worst out of the four. The polynomial kernel of degree 4 performs the best generally, although in some data sets there was a negligible difference. The best improvement from this higher degree polynomial kernel came from the energy and seoulbike data sets, and the worst relative performance from the RBF kernel came from the these same two data sets, meaning that these sets yielded the most variation amongst SVM models. Thus the polynomial kernel seems to be the best choice for these data sets, with higher degree polynomials generally having lowest errors.

f. Comparing the performance of the Decision Tree with the other models across the five data sets, what trends do you observe?  When did the decision tree do better or worse than the other models?  Why do you think this might have occurred?

The decision tree outperformed nearly all other models in most of the data sets. The seoulbike and capitalbike data sets have the starkest contrast between trees and the other models; e.g. in capitalbike, the decision tree yielded a MAE of around 1/3 the others, signifying a major improvement. However, in forestfires, the decision tree model performed similarly to LASSO, linear, and ridge, while underperforming relative to all SVM models. Since the Decision Tree is able to isolate the most important attributes (via entropy/Gain), we think it may have outperformed other models which were not able to successfully identify these best attributes. We are not sure if the trees subpar performance compared to SVM models is more due to it's underperformance or the SVM models excelling, either case is plausible.

#### Question 2

Using the same random seed and training set percentage as in Question 1 and a value of `false` for whether you want to rescale the numeric attributes using max-min normalization, rerun your program to train the requested eight models on each of the five data sets and calculate the resulting MAEs on the test sets.  (You do *not* have to list those MAEs here)

a.  Using the resulting MAEs, create one **bar chart** for each data set and save the five bar charts to image files that clearly describe to which data set they correspond (e.g., `capitalbike_unscaled_bar.png`).  Upload those five image files to your GitHub repository.

*See '../Figures/hw3-figures/bar/'*

b. Comparing the performances of Linear Regression within the five data sets, do you observe any changes in the performance of the model whether or not you rescale the numeric attributes?  Why do you think this result occurred?

The linear regression model performs the same regardless of rescaling. This trend occurs in every data sets, and makes sense. When training, linear models determine the weight of each attribute in influencing the dependent variable. This means that rescaling just changes the weights of each predictor to match this rescaling, and the final model behaves the exact same, predicting the same label regardless of numeric attribute scales.

c. Comparing the performances of LASSO and Ridge Regression within the five data sets, do you observe any changes in the performances of the models whether or not you rescale the numeric attributes?  Why do you think this result occurred?

The LASSO and ridge regression models perform similarly whether or not the inputs are being rescaled. However, we do see slightly better performance when rescaling with LASSO and ridge models in the forestfires and energy data sets. We think this is the same because the actual relationship between the predictor variables remains similar when rescaling; this is similar to why the linear models are unchanged by attribute rescaling.

d. Comparing the performances of SVM within the five data sets, do you observe any changes in the performances of the models whether or not you rescale the numeric attributes?  Why do you think this result occurred?

We observed massive improvements in performance of all SVM models when rescaling the data; this held for all data sets except for forestfires and wine. The biggest performance improvement seemed to come in the seoulbike data sets, where rescaling numeric attributes cut the MAE by 20-25%. This trend happens because SVMs are heavily influenced by large values, and can be thrown off when attributes fall on different scales. The difference here is that the function optimized by SVMs changes distance based on the scale, yielding very different models.

e. Comparing the performances of the Decision Tree within the five data sets, do you observe any changes in the performances of the models whether or not you rescale the numeric attributes?  Why do you think this result occurred?

The decision tree appeared to be unaffected by rescaling the numeric attributes; this could be due to the fact that decision trees don't consider the relationship between variables directly as in other models; the training process instead focuses on splitting individual attributes one at a time. Thus this trend makes sense because the regressor shouldn't be altered at all.

f. If given a new regression task, for which types of models might you consider rescaling numeric attributes?

We would consider rescaling numeric attributes for SVM models because we noticed that those saw the largest performance improvement from rescaling in these 5 data sets. LASSO and ridge also benefit from this rescaling, but the gains are less noticeable and would definitely be dependent on the data and random seeds used.

#### Question 3

Using the **energy.csv** data set, a random seed of `12345`, and a value of `true` for whether you want to use max-min normalization for the numeric attributes, calculate the MAE on the test set for each of the eight models using each of the following training percentages: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]`. (You do *not* have to list those MAEs here)

a. Create a **single line chart** plotting the test set MAE for each of the eight models for each of the training percentages.  That is, in your line chart, the training percentage should be the x-axis, the MAE should be the y-axis, and there should be one line for each of the eight models.  Save the line chart to an image file with the name `energy_rescaled_line.png` and upload it to your GitHub repository.

*See '../Figures/hw3-figures/energy_rescaled_line.png/'*

b. What trends do you observe as the training percentages increase?  Which models improved?  Which models had consistent performance?  Did any models do *worse* with more training data, and if so, which ones?

As the training percentages increase, the first identifiable trend is with the Decision Tree Regressor which seems to plummet its mean average error in comparison to the rest. We think this might be due to how it picks the best attribute, as it gets access to more data, it can make a more informed decision. It's a sort of risk/reward. LASSO performed the worst overall and is also the only model that had a higher mean average error with the highest training percentage than lowest (albeit not by much). The remaining models improved slightly from beginning to end, but not drastically. Linear and ridge performed similarly for all percentages, and both saw a drop in MAE at first followed by a slow incline. The four SVM models are all grouped near each other, but they don't overlap. From worst to best performing the SVM models were rmf, poly2, pol3 and pol4.

c. What do these trends imply about how you should choose a type of regresssion model, based on the amount of data you have available?  If you have a small training set, which model would you choose and why?  If you have a large training set, which model would you choose and why?

As touched on before, tree is a great model to choose if you train on a majority of the data. It won't be helpful if you have a small training set, but it is surprisingly effective as the training set gets closer to 75-80% of the data. On the other hand, SVM's stay solid the whole time, especially the polynomials, and are good choices for small training sets. Although, we are unsure if these trends are characteristic to this data set and seed or are more generalizable.

#### Question 4

Using the **seoulbike.csv** data set, a random seed of `12345`, and a value of `true` for whether you want to use max-min normalization for the numeric attributes, calculate the MAE on the test set for each of the eight models using each of the following training percentages: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]`. (You do *not* have to list those MAEs here)

a. Create a **single line chart** plotting the test set MAE for each of the eight models for each of the training percentages.  That is, as in Question 3, the training percentage should be the x-axis of your line chart, the MAE should be the y-axis, and there should be one line for each of the eight models.  Save the line chart to an image file with the name `seoulbike_rescaled_line.png` and upload it to your GitHub repository.

*See '../Figures/hw3-figures/seoulbike_rescaled_line.png/'*

b. What trends do you observe as the training percentage increases that are **similar** to the trends in Question 3 with the **energy.csv** data set?

We notice again that LASSO and ridge seem to have higher mean average errors as the training percent increases, although it is nearly constant. There is still a large drop in the tree model's MAE as the training percentage increases, although its MAE starts much lower relative to the energy results. We still notice a significant general downward trend as training percentage is increased.

c. What trends do you observe as the training percentage increases that are **different** from the trends in Question 3 with the **energy.csv** data set?

First, we notice that, on average, the slope is more negative for the lines in this chart. For example, the low training percentage MAE is a lot higher than the high, although we don't see tree start with such egregiously high MAE as in energy. Moreover, the SVM RBF model performs the worst in this data sets, whereas LASSO performed the worse in the other energy. Here, the decision tree regressor is the best model by a lot, whereas SVM polynomial of degree 4 was the best performer in energy.

d. Based on your answers to Question 4b and 4c above, what does this reveal about the process of designing a machine learning solution?  Are there some general rules of thumb that we can trust to help us make choices (e.g., on what model to use)?  How can we know whether those choices were the right ones to make for a particular data set?

There are some rules we can use; for example, normalizing the data almost never hurts the model's effectiveness, so we can do this most of the time. SVM models generally performed better than linear models, and decision trees were very hit and miss. We can try to rerun these experiments on the data sets in question to determine what will be the most effective, since we won't fully know until we try them. Trying these with different seeds and training percentages is also very useful to verify that a model can perform well on a particular data sets; i.e., through cross validation we can ensure that our model is better than alternatives.

# Additional Questions

Please answer these questions after you complete the assignment:

1. What was your experience during the assignment (what did you enjoy, what was difficult, etc.)?
    The research questions were the hardest, and the coding was the most enjoyable for us.

2. Approximately how much time did you spend on this assignment?
    10 hours
3. Did you adhere to the Honor Code?
    Yes
