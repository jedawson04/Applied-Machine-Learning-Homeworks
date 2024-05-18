import time, csv
from sklearn import tree, linear_model, svm, metrics
from sys import argv as args
import pandas as pd
from plotnine import * 
import numpy as np
import warnings
warnings.filterwarnings('ignore') # to ignore warnings from plotnine

# This script is written by Josh Dawson and Mario Stinson-Maas
# This script is for the creating machine learning models to solve regression problems, evaluating performance and, visualizing and interpreting the resulting data

# one_hot_encoding method partially from Lab 4
def one_hot_encoding(dataframe):
    for col in dataframe.columns[1:]: # iterate through non label cols
        # only transform strings
        if isinstance(dataframe[col][0], str): 
            # perform one hot encoding
            onehots = pd.get_dummies(dataframe[col], col, drop_first=True, dtype=int)
            dataframe = pd.concat([dataframe.drop(col, axis=1), onehots], axis=1)
    return dataframe

# min-max normalization method partially from Lab 4
def scale_dataset(dataframe):
    # make a deep copy of the data set
    df_scaled = dataframe.copy() 
    print(f'{df_scaled.min()[0] = }')
    print(f'{df_scaled.max()[0] = }')
    for col in df_scaled.columns[1:]: # iterate through non label cols
        # scale if col is numeric
        if isinstance(dataframe[col][0], (int,float,np.integer)): 
            # get the min and max value to normalize to
            col_max = df_scaled[col].max()
            col_min = df_scaled[col].min()
            # normalize column to max and min
            df_scaled[col] = (df_scaled[col] - col_min) / (col_max - col_min)

    return df_scaled

# split_data method splits data into training and testing sets
# The data is pre-processessed before splitting
def split_data(dataset, train_percentage, seed, norm_bool):

    # load the data set
    dataframe = pd.read_csv(dataset)

    # perform min-max normalization if designated by the user
    if norm_bool:
        dataframe = scale_dataset(dataframe)

    # perform one hot encoding
    dataframe = one_hot_encoding(dataframe)

    # randomize the order of the data set in a new variable
    shuffled = dataframe.sample(frac=1, random_state=seed)
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

# Creates and trains 8 models 
def train_models(training_X, training_y, seed):
    # create and train models
    lasso = linear_model.Lasso(random_state=seed).fit(training_X, training_y) 
    linear = linear_model.LinearRegression().fit(training_X, training_y)
    ridge = linear_model.Ridge(random_state=seed).fit(training_X, training_y)
    svm_poly2 = svm.SVR(kernel='poly', degree=2).fit(training_X, training_y) 
    svm_poly3 = svm.SVR(kernel='poly').fit(training_X, training_y) 
    svm_poly4 = svm.SVR(kernel='poly', degree=4).fit(training_X, training_y) 
    svm_rbf = svm.SVR().fit(training_X, training_y) 
    decision_tree = tree.DecisionTreeRegressor(random_state=seed).fit(training_X, training_y) 

    # return the trained models
    return [lasso, linear, ridge, svm_poly2, svm_poly3, svm_poly4, svm_rbf, decision_tree] 

# Creates line charts to answer questions 3 and 4
def line_chart():
    # seed, percentages, and names are fixed
    seed = 12345
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    names = ["LASSO", "linear", "ridge", "svm_poly2", "svm_poly3", "svm_poly4", "svm_rbf", "tree"]
    # energy file for question 3 and seoulbike file for question 4
    files = ["energy", "seoulbike"] 
    for file in files:
        results = []
        for training_percent in percentages:
            print(training_percent)
            # split data for every training percentage
            training_X, training_y, testing_X, testing_y = split_data(f'{file}.csv', training_percent, seed, True)
            # train all 8 models on the sets generated
            models = train_models(training_X, training_y, seed)
            # predict with the models and add results of predictions for all 8 models to a results list
            results.append(calculate_MAE(models, testing_X, testing_y))
        # results now has [prediction for percentages [all 8 models[name, model MAE]]]
        dfs = [] # create dataframes list
        for ind, model in enumerate(names): 
            # for every model create a dict and add training percentages col to it
            data = {}
            data["training percentage"] = percentages
            # create MAE col 
            MAE = [pred_for_per[ind][1] for pred_for_per in results]
            data["MAE"] = MAE # add MAE col to dict
            df = pd.DataFrame(data) # create df with cols for percentage and MAE 
            df["model"] = model # add a model col to all dicts
            dfs.append(df) # append to dataframes list
        # combines the dataframes together to one
        results = pd.concat(dfs, ignore_index=True)
        # create a line chart of the accuracy values
        line_chart = (
            ggplot(results) 
                + aes(x="training percentage", y="MAE", color="model")
                + scale_color_brewer(type="qual")
                + geom_line()
                + ggtitle(f'{file} MAE vs training percentage for 8 models')
        )
        # display the chart to the user
        ggsave(line_chart, f'{file}_rescaled_line', path="./figures/") 

# Creates bar charts to answer questions 1 and 2
def bar_chart(): 
    files = ["capitalbike","energy","forestfires","seoulbike","wine"]
    scaled = [True, False]
    seed = 12345 
    training_percent = 0.75
    for scale in scaled: 
        for file in files:
            training_X, training_y, testing_X, testing_y = split_data(f'{file}.csv', training_percent, seed, scale)
            # train all 8 models on the sets generated
            models = train_models(training_X, training_y, seed)
            # predict with the models and add results of predictions for all 8 models to a results list
            pred = calculate_MAE(models, testing_X, testing_y)
            data = {} # create dict to transform into df
            # add model and MAE cols to data based off of predictions
            data["model"] = [p[0] for p in pred]
            data["MAE"] = [p[1] for p in pred]
            # create df and bar chart
            results = pd.DataFrame(data)
            bar_chart = (
            ggplot(results) 
                + aes(x="model", y="MAE", fill="model")
                + geom_col(position="dodge")
                + ggtitle(f'MAE of 8 different ML models for {file}' + (" rescaled" if scale else ""))
                + theme(axis_text_x = element_text(angle = 45,hjust = 10))
            )
            # save chart
            name = f'{file}' + ("_rescaled" if scale else "") + "_bar"
            path = f'./figures/bar' + ("_rescaled" if scale else "")
            ggsave(bar_chart, name, path = path) 

# create and save predictions with names
def calculate_MAE(models, testing_X, testing_y):
    mean_absolute_errors = []
    # list of names to pair with models
    names = ["LASSO", "linear", "ridge", "svm_poly2", "svm_poly3", "svm_poly4", "svm_rbf", "tree"]
    for i, model in enumerate(models):
        # create predictions and calculate MAE
        prediction = model.predict(testing_X)
        mean_absolute_error = metrics.mean_absolute_error(testing_y, prediction)

        # add name and MAE to predictions
        mean_absolute_errors.append([names[i], mean_absolute_error])

    return mean_absolute_errors

# Main method handles user input and calls above methods to generate csv output
def main():
    start_time = time.time()

    # color codes for clearer error message
    [YELLOW, PINK, WHITE, DARKCYAN, BOLD, UNBOLD] = ['\033[33m', '\033[35m', '\033[37m', '\033[36m', '\033[1m', '\033[0m']

    # check num of args
    if len(args) != 5:
        print(f'{YELLOW}Requires Four arguments corresponding to: file, percentage to train on, int to use as a seed, and if desired to use min-max normalization on numeric attributes')
        print(f'Correct Usage: {PINK}python regression.py file.csv percent seed true/false{WHITE}')
        exit()

    # convert args to variables for readability
    [file, per, seed, norm_bool] = args[1:]
    
    # if user entered lowercase bools correct to uppercase
    if norm_bool == 'true':
        norm_bool = True
    elif norm_bool == 'false':
        norm_bool = False
    
    # line_chart() # line chart to plot 8 training percentages for all models on 2 files

    # creates the training and testing data by calling split data 
    training_X, training_y, testing_X, testing_y = split_data(file, float(per), int(seed), norm_bool)
    
    # creates and trains all 8 models
    models = train_models(training_X, training_y, int(seed))
    
    # calculated MAE for all 8 models and returns a list with [name, MAE] pairs
    predictions = calculate_MAE(models, testing_X, testing_y)

    # bar_chart() # creates bar chart based on prediction calculated from CL args 

    # format of output csv: results_<DataSet>_<TrainingPercentage>p_<Seed>[_rescaled].csv 
    name = f'results_{file[:-4]}_{float(per)*100}p_{seed}' 
    if norm_bool: 
        name += "_rescaled"
    
    # creates a .csv named name in results
    with open(f'./results/{name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model","MAE"]) # label row
        # for every model create a row and add its MAE
        [writer.writerow(model) for model in predictions]
    
    print(f'{BOLD}{DARKCYAN}Your output file named "{name}" is at /results/{name}') # let user know where their output file is
    print(f'Time taken: {round(time.time()-start_time,2)}s{WHITE}{UNBOLD}') # print time taken
if __name__ == "__main__":
    main()