import time, csv, math
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sys import argv as args
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import * 

# This script is written by Josh Dawson and Mario Stinson-Maas
# This script is for the implementation of trees and forests algorithm CART from scikit-learn libraries

# split_data method splitts data into training and testing sets
# takes in csv filename, percentage of training set to total data, and a random seed
def split_data(data_set, train_percentage, seed):
    # load the data set
    data_frame = pd.read_csv(data_set)
    # randomize the order of the data set in a new variable
    shuffled = data_frame.sample(frac=1, random_state=seed)
    total_rows = shuffled.shape[0] # grabs number of rows
    training_rows = int(train_percentage * total_rows)

    # create the training set
    training = shuffled.iloc[:training_rows, :] # grabs rows upto training_rows
    testing = shuffled.iloc[training_rows:, :] # grabs rows after training_rows
  
    # split the training attributes and labels
    training_X = training.drop("label", axis=1)
    training_y = training["label"]
       
    # split the testing attributes and labels
    testing_X = testing.drop("label", axis=1)
    testing_y = testing["label"]

    return training_X, training_y, testing_X, testing_y


# creates the model based on num, predicts with it and returns a confusion matrix of the results
def generate_predictions(num, training_X, training_y, testing_X, testing_y, seed):

    # set model to Decision Tree if num == 1 else Forest with num trees
    model = DecisionTreeClassifier(random_state=seed) if num == 1 else RandomForestClassifier(n_estimators=num, random_state=seed)
    
    # provide the training data to inform the classifier
    model.fit(training_X, training_y)
    
    # make predictions on the testing data
    predictions = model.predict(testing_X)
    
    # grab labels from classes methods from model
    labels = model.classes_.tolist()        

    # create a confusion matrix with testing_y and predictions with labels 
    cm = confusion_matrix(testing_y, predictions, labels=labels)

    # creates a confusion matrix normalized to predictions 
    cmpred = confusion_matrix(testing_y, predictions, labels=labels, normalize='pred')

    # recall is the diagonal values in cmpred
    print()
    for ind,rows in enumerate(cmpred):
        print(f'{labels[ind]} recall = {round(rows[ind]*100,2)}%') # print recalls
                                                
    return cm, labels

# Code from Adam to plot the decision tree
def log_tree(tree, dataset_infile, train_percentage, seed):
    dataset = pd.read_csv(dataset_infile)
    # create the filename
    filename = ("tree"
                + "_" + dataset_infile[:-4]
                + "_1t"
                + "_" + str(int(train_percentage * 100)) + "p"
                + "_" + str(seed) + ".png")

    attributes = list(dataset.drop("label", axis=1))
    labels = sorted(list(dataset["label"].unique()))
    # create and display plot
    fig = plt.figure(figsize=(100, 100))
    plotted = plot_tree(tree,
        feature_names=attributes,
        class_names=labels,
        filled=True,
        rounded=True)
    fig.savefig(filename)

# creates line plot of accuracy of predictions for 10 different tree values for 1 dataset
def tree_plot():
    trees = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    accuracies = []
    for tree in trees:
        # create a forest with tree trees
        model = RandomForestClassifier(n_estimators=tree, random_state=1234)
        # split data based on parameters outlined in question 8
        training_X, training_y, testing_X, testing_y = split_data("mnist1000.csv", 0.75, 1234)
        # fit the model to the data 
        model.fit(training_X, training_y)
        # make predictions on the testing data
        predictions = model.predict(testing_X)
        # calculate accuracy of predictions
        accuracy = sum(predictions == testing_y) / len(testing_y)
        # add accuracy to list
        accuracies.append(accuracy)
    # create dictionary to make data frame
    data = {}
    # add trees column
    data["Amt of Trees"] = trees
    # add accuracy column
    data["Accuracy"] = accuracies
    # create the DataFrame object
    results = pd.DataFrame(data)
    # create line chart
    line_chart = (
        ggplot(results) 
            + aes(x="Amt of Trees", y="Accuracy")
            + geom_line()
            + ylim(0.75, 1)
            + ggtitle(f'Accuracy of mnist1000 Random Forest Predictions for Different Number of Trees')
    )
    # display the chart to the user
    line_chart.show()

# creates a line plot of accuracy of predictions for 4 different training percentages for 3 data sets 
def percentage_plot():
    percentages = [0.2, 0.4, 0.6, 0.8] # different training percentages
    model = DecisionTreeClassifier(random_state=1234)
    files = ["mnist1000", "occupancy", "penguins"]
    dataframes = []
    for ind, file in enumerate(files):
        accuracies = []
        for p in percentages:
            training_X, training_y, testing_X, testing_y = split_data(f'{file}.csv', p, 1234)
            # provide the training data to inform the classifier
            model.fit(training_X, training_y)
            # make predictions on the testing data
            predictions = model.predict(testing_X)
            # calculate and return accuracy of predictions
            accuracy = sum(predictions == testing_y) / len(testing_y)
            # add accuracy to list
            accuracies.append(accuracy)
        # create a dictionary to make the data frame
        data = {}
        # save the percent column
        data["Percent of Training Data"] = percentages
        # save the accuracy column
        data["Accuracy"] = accuracies
        # create the DataFrame object
        file_results = pd.DataFrame(data)
        # add a 'dataset' column that saves the name of the data set    
        file_results["Dataset"] = files[ind]
        # add the data set to dataframes
        dataframes.append(file_results)
    # combines the dataframes together to one
    results = pd.concat(dataframes, ignore_index=True)
    # create a line chart of the accuracy values
    line_chart = (
        ggplot(results) 
            + aes(x="Percent of Training Data", y="Accuracy", color="Dataset")
            + geom_line()
            + ylim(0, 1)
            + ggtitle(f'Accuracy of Decision Tree Predictions for Different Training Percentages')
    )
    # display the chart to the user
    line_chart.show()
   
# Main method handles user input and calls above methods to generate the output csv
def main():
    start_time = time.time()
    # color codes for clearer error message
    [YELLOW, PINK, WHITE, DARKCYAN, BOLD, UNBOLD] = ['\033[33m', '\033[35m', '\033[37m', '\033[36m', '\033[1m', '\033[0m']

    # check num of args
    if len(args) != 5:
        print(f'{BOLD}{YELLOW}Need 4 arguments that correspond to file, num of trees, percentage to train on, and an integer to use as a random seed')
        print(f'Correct Usage: {PINK}python trees.py file.csv num_trees percent seed{WHITE}{UNBOLD}')
        exit()    

    # convert args to variables for readability
    [file, trees, per, seed] = args[1:]
    
    # creates the training and testing data by calling split data 
    training_X, training_y, testing_X, testing_y = split_data(file, float(per), int(seed))

    # creates the confusion matrix and list of labels by calling generate predictions
    matrix, labels = generate_predictions(int(trees), training_X, training_y, testing_X, testing_y, int(seed))

    # vizualize the plot
    # log_tree(model, file, float(per), seed)

    # generates percentage plot for question 5
    # percentage_plot() 
    
    # generates tree plot for question 8
    # tree_plot()

    # format of output csv: results_<DataSet>_<NumTrees>t_<TrainingPercentage>p_<Seed>.csv
    name = str('results_'+file[:-4]+'_'+trees+'t_'+per+'p_'+seed+'.csv')
    labels.append("") # add an empty label for an extra comma in csv output
    
    with open(f'./results/{name}', 'w', newline='') as file: # creates a .csv named name in results
        writer = csv.writer(file)
        writer.writerow(labels) # label row
        # for every label create a row and add it
        for ind,row in enumerate(matrix): 
            line = [elem for elem in row]
            line.append(labels[ind])
            writer.writerow(line) # write row 
    
    accuracy = matrix.diagonal().sum() / len(testing_y) # calculate accuracy
    # SE and confidence interval used to verify math was correct
    SE = round((math.sqrt(accuracy*(1-accuracy)/len(testing_y))),4)
    Con_Int = [max(round(accuracy-1.96*SE,4),0),min(round(accuracy+1.96*SE,4),1)]
    print()
    print(f'{BOLD}{DARKCYAN}Accuracy: {round(accuracy*100,2)}%') # print accuracy to the user
    print(f'Time taken: {round(time.time()-start_time,2)}s') # print time taken
    # print SE and Con Int
    print(f'Calculated Standard Error: {SE}. Calculated 95% Confidence Interval: {Con_Int}.{UNBOLD}') 
    
if __name__ == "__main__":
    main()