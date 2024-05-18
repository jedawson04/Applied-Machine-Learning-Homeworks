import time, csv
from sys import argv as args
import pandas as pd
import numpy as np
import tensorflow as tf 
from sklearn import metrics
from plotnine import *

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
def scale_dataset(df):
    for col in df.columns[1:]: # iterate through non label cols
        # scale if col is numeric
        if isinstance(df[col][0], (int,float,np.integer)): 
            # get the min and max value to normalize to
            col_max = df[col].max()
            col_min = df[col].min()
            # normalize column to max and min
            if col_max == col_min:
                df.drop(col, axis=1, inplace=True) # drop the col if this is the case (for mnist)
            else: 
                df[col] = (df[col] - col_min) / (col_max - col_min)
    return df

# converts the the labels of mnist1000/penguins into numbers
def convert_labels(filename, dataset):
    # create a dictionary mapping 
    if filename == "mnist1000.csv":
        numbers = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
    elif filename == "penguins.csv":
        numbers = {"Adelie" : 0, "Chinstrap" : 1, "Gentoo" : 2}
    # convert each of the string labels into the corresponding number
    for name in numbers:
        number = numbers[name]
        dataset.loc[dataset["label"] == name, "label"] = number

# split_data method splits data into training and testing sets
# The data is pre-processessed before splitting
def split_data(dataset, train_percentage, seed, classification):

    # load the data set
    dataframe = pd.read_csv(dataset)

    # convert labels to integers for classification tasks
    if classification:
        convert_labels(dataset, dataframe)

    # perform min-max normalization on numeric attributes
    dataframe = scale_dataset(dataframe)

    # perform one hot encoding on categorical attributes
    dataframe = one_hot_encoding(dataframe)

    # randomize the order of the data set based on seed
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

# creates a neural network with one hidden layer
def create_network(hidden_neurons, output_neurons):

    hidden_layer = tf.keras.layers.Dense(hidden_neurons, activation='sigmoid')  # creates a layer with specified amt of neurons
    output_layer = tf.keras.layers.Dense(output_neurons) # number of output neurons depends on task

    # add the layers to a list and create the network
    all_layers = [hidden_layer, output_layer] 
    network = tf.keras.models.Sequential(all_layers)

    return network

# trains a neural network given training data
def train_network(network, training_X, training_y, learning_rate, classification, filename):
    # create the algorithm that learns the weight of the network with specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # random seed with network -- not sure. -- you could seed the training by using the `tf.random.set_seed` method (https://www.tensorflow.org/api_docs/python/tf/random/set_seed).  All weights are initially random, so setting the seed should cause the same weights to be chosen with each run of the program using the same seed.  I'm not expecting students to do this, however -- it felt like one extra step when there was already a lot of work to do. -- we decided not to do this for the purposes of this hw

    # save metrics and loss function
    if classification: 
        metrics = ["accuracy"]
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else: 
        metrics = ["mean_absolute_error"]
        loss_function = tf.keras.losses.MeanSquaredError()

    # configure the network with losses and metrics
    network.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    # create a logger to save the training details to file
    csv_logger = tf.keras.callbacks.CSVLogger(f'./Logs/{filename[:-4]}_training.csv')

    # train the network for 250 epochs (setting aside 20% of the training data as validation data)
    network.fit(training_X, training_y, validation_split=0.2, epochs=250, callbacks=[csv_logger])
    
def calculate_performance(network, classification, testing_X, testing_y): 
    
    # predict
    predictions = network.predict(testing_X)

    if classification:
        # use softmax to pick the most probable label for each test instance
        probs = tf.nn.softmax(predictions)
        predictions = tf.argmax(probs, 1).numpy()
        # calculate the accuracy of the predictions
        accuracy = sum(predictions == testing_y.numpy()) / len(predictions)
        return accuracy 

    # calcuate mean average error for regression
    mean_absolute_error = metrics.mean_absolute_error(testing_y, predictions)

    return mean_absolute_error

# Same method for: questions one and two
def line_chart_30_networks(classification): 
    # pass in classification set / regression set for desired line charts
    files = ["penguins.csv","mnist1000.csv"] if classification else ["seoulbike.csv","energy.csv"]
    # default params
    [seed, training_percent] = [12345, 0.75]
    # specified rates and neurons
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    num_neurons = [8, 16, 32, 64, 128, 256]
    for file in files: # each of these is a separate plot
        # determine num output neurons and metric
        if not classification: 
            output_neurons = 1
            metric = "MAE"
        else: 
            output_neurons = 3 if file == "penguins.csv" else 10
            metric = "accuracy"
        dfs = [] # create a df for every rate
        for rate in learning_rates: # each of these is a separate line in the plot 
            # create a dict and add x - axis as a col
            data = {}
            data["neurons"] = num_neurons
            performances = [] # store performance for all neurons
            for neurons in num_neurons: 
                training_X, training_y, testing_X, testing_y = split_data(file, float(training_percent), int(seed), classification)
                # convert sets to tensors to train on network
                training_X = tf.convert_to_tensor(training_X, dtype=tf.float32)
                training_y = tf.convert_to_tensor(training_y, dtype=tf.int32)
                testing_X = tf.convert_to_tensor(testing_X, dtype=tf.float32)
                testing_y = tf.convert_to_tensor(testing_y, dtype=tf.int32)
                
                # create the network
                network = create_network(neurons, output_neurons) # with neurons hidden neurons

                # train the network
                train_network(network, training_X, training_y, rate, classification, file)

                performances.append(calculate_performance(network, classification, testing_X, testing_y))

                print(f'Trained with {neurons = } and {rate = } for {file = }.\n')

            # add performances as y - axis col
            data[metric] = performances
            # create df with cols for number of neurons and performance
            df = pd.DataFrame(data) 
            # add a rate col for each line
            df["learning_rate"] = rate 
            dfs.append(df) # append to dataframes list
        # combines the dataframes together to one
        results = pd.concat(dfs, ignore_index=True)
        # represents learning rate categorically
        results["learning_rate"] = results["learning_rate"].astype("category") 
        print(f'Finished {rate = } for {file = }.\n')

        # create a line chart of the metrics
        sd = 642 if file == "seoulbike.csv" else 102 # standard deviation of MAEs determined from diagnostics.py
        line_chart = (
            ggplot(results, aes(x="neurons", y=metric, color="learning_rate"))
                + scale_color_brewer(type="qual")
                + geom_line()
                + labs(title=f'{file}, Number of Neurons vs. {metric}', x="Number of Neurons")
                + expand_limits(y=0)
                + scale_color_discrete(name="Learning Rate")
        )
        if not classification: # add std dev line for regression figures
            line_chart = line_chart + geom_hline(yintercept=sd, linetype="dotted") + annotate("label", x=15, y=sd-1, label="1 STD DEV")
        # save the chart for the user
        ggsave(line_chart, filename = f'{file[:-4]}_line.png', path="./figures/")

# method for question 4
def line_chart_validation(): 
    # save the defauls
    files = ["penguins.csv", "seoulbike.csv"]
    [num_neurons, training_percent, random_seed, learning_rates] = [128, 0.75, 54321, [0.01, 0.00001]]
    # demonstrate performance of training and validation sets
    for file in files: # 2 files
        classification = True if file == "penguins.csv" else False 
        # set defaults depending on classification/regression task
        output_neurons = 3 if classification else 1
        if classification: 
            metrics = ["accuracy"]
            loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else: 
            metrics = ["mean_absolute_error"]
            loss_function = tf.keras.losses.MeanSquaredError()
        for rate in learning_rates: # one plot for each learning rate per file (4 plots total)
            
            training_X, training_y, testing_X, testing_y = split_data(file, float(training_percent), int(random_seed), classification)
            # convert sets to tensors to train on network
            training_X = tf.convert_to_tensor(training_X, dtype=tf.float32)
            training_y = tf.convert_to_tensor(training_y, dtype=tf.int32)
            testing_X = tf.convert_to_tensor(testing_X, dtype=tf.float32)
            testing_y = tf.convert_to_tensor(testing_y, dtype=tf.int32)
            
            # create the network
            network = create_network(num_neurons, output_neurons) # with neurons hidden neurons

            # train the network 
            optimizer = tf.keras.optimizers.Adam(learning_rate=rate)

            # configure the network with losses and metrics
            network.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

            # create a logger to save the training details to file
            csv_logger = tf.keras.callbacks.CSVLogger(f'./Logs/QuestionFour_{file[:-4]}_training.csv') # special title

            # train the network for 250 epochs (setting aside 20% of the training data as validation data)
            network.fit(training_X, training_y, validation_split=0.2, epochs=250, callbacks=[csv_logger])

            log = pd.read_csv(f'./Logs/QuestionFour_{file[:-4]}_training.csv') # read the ouput log into a df
            
            # convert the data from wide to long format (so that each measure for each epoch is a different row)
            log = log.reset_index()
            long = pd.melt(log, id_vars='epoch', value_vars=[metrics[0], 'loss', f'val_{metrics[0]}', 'val_loss']) 

            # x - axis is epoch, y - axis is [loss, val_loss, accuracy, val_accuracy] (line for each)
            line_chart = ( 
                ggplot(long, aes(x="epoch",y="value", color="variable"))
                + scale_color_brewer(type="qual")
                + geom_line()
                + labs(title="Performance Values by Epoch", x="Epoch", y="Value")
                + scale_color_discrete(name="Metric", labels=[metrics[0], "Loss", f"Validation {metrics[0]}", "Validation Loss"])
            )
            # save the plot
            ggsave(line_chart, f'./figures/{file[:-4]}_{rate}.png') 

            
# Main method handles user input and calls above methods to generate csv output
def main():
    start_time = time.time()

    # color codes for clearer error message
    [YELLOW, PINK, WHITE, DARKCYAN, BOLD, UNBOLD] = ['\033[33m', '\033[35m', '\033[37m', '\033[36m', '\033[1m', '\033[0m']

    # check num of args
    if len(args) != 6:
        print(f'{YELLOW}{BOLD}Requires Five parameters corresponding to: file, learning rate for training, neurons per hidden layer, percentage to train on, and a random seed.')
        print(f'Correct Usage: {PINK}python neuralnet.py file.csv rate neurons percent seed {WHITE}{UNBOLD}')
        exit()

    # convert args to variables for readability
    [filename, learning_rate, neurons_per_layer, training_percentage, random_seed] = args[1:]
    
    # set seed -- doesn't seem to work unfortunately :-( -- by work I mean same parameters will still have variable performance
    tf.random.set_seed(int(random_seed))

    # determine if task is classification or regression
    classification = True if filename == "mnist1000.csv" or filename == "penguins.csv" else False

    # line_chart_30_networks(classification) # method for q1 and q2
    
    line_chart_validation() # method for q4

    # preprocesses and splits up data set
    training_X, training_y, testing_X, testing_y = split_data(filename, float(training_percentage), int(random_seed), classification)
    
    # convert sets to tensors to train on network
    training_X = tf.convert_to_tensor(training_X, dtype=tf.float32)
    training_y = tf.convert_to_tensor(training_y, dtype=tf.int32)
    testing_X = tf.convert_to_tensor(testing_X, dtype=tf.float32)
    testing_y = tf.convert_to_tensor(testing_y, dtype=tf.int32)
    
    # determine number of output neurons
    if not classification: 
        output_neurons = 1 # 1 output for regression tasks
    else: 
        # for classification, 3 outputs for penguins, and 10 for mnist1000
        output_neurons = 3 if filename == "penguins.csv" else 10
    
    # create the network
    network = create_network(neurons_per_layer, output_neurons)

    # train the network
    train_network(network, training_X, training_y, float(learning_rate), classification, filename)

    # # predict with the network
    performance = calculate_performance(network, classification, testing_X, testing_y)

    results = [filename[:-4], learning_rate, neurons_per_layer, performance]

    with open('results.csv', 'a', newline='') as file: # using 'a' for append as opposed to write
        writer = csv.writer(file)
        writer.writerow(results)

    print(f'{BOLD}{DARKCYAN}Time taken: {round(time.time()-start_time,2)}s{WHITE}{UNBOLD}') # print time taken

if __name__ == "__main__":
    main()