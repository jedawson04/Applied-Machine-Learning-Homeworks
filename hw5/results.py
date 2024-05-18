import time
import pandas as pd
import numpy as np
import tensorflow as tf 
from sklearn import metrics, tree, linear_model, svm, neighbors, ensemble
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
    if filename == "mnist1000":
        numbers = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
    elif filename == "penguins":
        numbers = {"Adelie" : 0, "Chinstrap" : 1, "Gentoo" : 2}
    # convert each of the string labels into the corresponding number
    for name in numbers:
        number = numbers[name]
        dataset.loc[dataset == name] = number

# split_data method splits data into training and testing sets
# The data is pre-processessed before splitting
def split_data(dataset, train_percentage, seed):

    # load the data set
    dataframe = pd.read_csv(f'{dataset}.csv')

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

# creates a neural network with one hidden layer that contains hidden_neurons neurons
def create_network(hidden_neurons, output_neurons):
    hidden_layer = tf.keras.layers.Dense(hidden_neurons, activation='sigmoid')  # creates a layer with specified amt of neurons
    output_layer = tf.keras.layers.Dense(output_neurons) # number of output neurons depends on task

    # add the layers to a list and create the network
    all_layers = [hidden_layer, output_layer] 
    network = tf.keras.models.Sequential(all_layers)

    return network

# creates and trains a neural network based on training data and other parameters
def train_nn(training_x, training_Y, learning_rate, neurons, output_neurons, classification, filename):

    # create the network
    network = create_network(neurons, output_neurons) # with neurons hidden neurons

    # convert sets to tensors to train on network
    training_X = tf.convert_to_tensor(training_x, dtype=tf.float32)
    training_y = tf.convert_to_tensor(training_Y, dtype=tf.int32)

    # create the algorithm that learns the weight of the network with specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
    csv_logger = tf.keras.callbacks.CSVLogger(f'./Logs/{filename}_training.csv')

    # train the network for 250 epochs (setting aside 20% of the training data as validation data)
    network.fit(training_X, training_y, validation_split=0.2, epochs=250, callbacks=[csv_logger])
    return network

def calculate_vmae(model, testing_X, testing_y):
    # calcualte mae
    predictions = model.predict(testing_X).flatten()
    mean_absolute_error = metrics.mean_absolute_error(testing_y, predictions)
    # calculate vmae based off predictions and mae
    absErr = np.abs(np.subtract(testing_y, predictions))
    vmae = np.mean((absErr - mean_absolute_error)**2)
    return vmae

def calculate_performance(model, classification, testing_X, testing_y):
    predictions = model.predict(testing_X)
    if classification:
        were_correct = predictions == testing_y
        num_correct = sum(were_correct)
        return (num_correct / len(testing_y))
    
    # create predictions and calculate MAE
    mean_absolute_error = metrics.mean_absolute_error(testing_y, predictions)
    return mean_absolute_error

def calculate_nn_performance(network, classification, testing_X, testing_y): 
    # predict
    predictions = network.predict(testing_X)

    if classification:
        # use softmax to pick the most probable label for each test instance
        probs = tf.nn.softmax(predictions)
        predictions = tf.argmax(probs, 1).numpy()
        # calculate the accuracy of the predictions
        accuracy = sum(predictions == testing_y) / len(predictions)
        return accuracy 

    # calcuate mean average error for regression
    mean_absolute_error = metrics.mean_absolute_error(testing_y, predictions)

    return mean_absolute_error

# trains all models 
def train_models():
    files = ["mnist1000"]#["penguins","mnist1000","seoulbike","energy"]
    # default params
    [seed, training_percent] = [12345, 0.75]
    # create dataframe to store results
    df = []

    for file in files: # each of these is a separate plot
        classification = True if file == "penguins" or file == "mnist1000" else False
        training_X, training_y, testing_X, testing_y = split_data(file, training_percent, seed)

        if file == "energy":
            # create a list to store results, set hyperparameters
            results = []
            hyperParams = {"neurons": 64, "learning_rate": .01}

            # train models
            linear = linear_model.LinearRegression()
            linear.fit(training_X, training_y)
            svmModel = svm.SVR(kernel='poly', degree=4)
            svmModel.fit(training_X, training_y)
            nn = train_nn(training_X, training_y, hyperParams["learning_rate"], hyperParams["neurons"], 1, classification, file)

            # calculate maes here
            results.append({"model": "Linear", "mae": calculate_performance(linear, False, testing_X, testing_y), "vmae": calculate_vmae(linear, testing_X, testing_y)})
            results.append({"model": "SVM", "mae": calculate_performance(svmModel, False, testing_X, testing_y), "vmae": calculate_vmae(svmModel, testing_X, testing_y)})
            results.append({"model": "Neural Net", "mae": calculate_nn_performance(nn, False, testing_X, testing_y), "vmae": calculate_vmae(nn, testing_X, testing_y)})

            for result in results:
                df.append({"model": result["model"], "MAE": result["mae"], "VMAE": result["vmae"], "file": "energy"})

        elif file == "seoulbike":
            # create list to store results, set hyperparameters
            results = []
            hyperParams = {"neurons": 64, "learning_rate": .1}

            # train models
            linear = linear_model.LinearRegression()
            linear.fit(training_X, training_y)
            svmModel = svm.SVR(kernel='poly', degree=4)
            svmModel.fit(training_X, training_y) 
            nn = train_nn(training_X, training_y, hyperParams["learning_rate"], hyperParams["neurons"], 1, classification, file)

            # calculate maes
            results.append({"model": "Linear", "mae": calculate_performance(linear, False, testing_X, testing_y), "vmae": calculate_vmae(linear, testing_X, testing_y)})
            results.append({"model": "SVM", "mae": calculate_performance(svmModel, False, testing_X, testing_y), "vmae": calculate_vmae(svmModel, testing_X, testing_y)})
            results.append({"model": "Neural Net", "mae": calculate_nn_performance(nn, False, testing_X, testing_y), "vmae": calculate_vmae(nn, testing_X, testing_y)})

            for result in results:
                df.append({"model": result["model"], "MAE": result["mae"], "VMAE": result["vmae"], "file": "seoulbike"})
        
        elif file == "mnist1000":
            # create list to store results, set hyperparameters
            results = []
            hyperParams = {"neurons": 64, "learning_rate": .01, "neighbors": 1, "numtrees": 100}

            # train models
            knn = neighbors.KNeighborsClassifier(n_neighbors=hyperParams["neighbors"])
            knn.fit(training_X, training_y)
            dt = tree.DecisionTreeClassifier()
            dt.fit(training_X, training_y)
            rf = ensemble.RandomForestClassifier(n_estimators=hyperParams["numtrees"])
            rf.fit(training_X, training_y)
            convert_labels(file, training_y)
            nn = train_nn(training_X, training_y, hyperParams["learning_rate"], hyperParams["neurons"], 10, classification, file)

            results.append({"model": "K-NN", "accuracy": calculate_performance(knn, True, testing_X, testing_y)})
            results.append({"model": "Tree", "accuracy": calculate_performance(dt, True, testing_X, testing_y)})
            results.append({"model": "Forest", "accuracy": calculate_performance(rf, True, testing_X, testing_y)})
            convert_labels(file, testing_y)
            results.append({"model": "Neural Net", "accuracy": calculate_nn_performance(nn, True, testing_X, testing_y)})
            
            for result in results:
                df.append({"model": result["model"], "accuracy": result["accuracy"], "file": "mnist1000"})

        elif file == "penguins":
            # create list to store results, set hyperparameters
            results = []
            hyperParams = {"neurons": 16, "learning_rate": .1, "neighbors": 1, "numtrees": 100}

            # train models
            knn = neighbors.KNeighborsClassifier(n_neighbors=hyperParams["neighbors"])
            knn.fit(training_X, training_y)
            dt = tree.DecisionTreeClassifier()
            dt.fit(training_X, training_y)
            rf = ensemble.RandomForestClassifier(n_estimators=hyperParams["numtrees"])
            rf.fit(training_X, training_y)
            convert_labels(file, training_y)
            nn = train_nn(training_X, training_y, hyperParams["learning_rate"], hyperParams["neurons"], 3, classification, file)

            results.append({"model": "K-NN", "accuracy": calculate_performance(knn, True, testing_X, testing_y)})
            results.append({"model": "Tree", "accuracy": calculate_performance(dt, True, testing_X, testing_y)})
            results.append({"model": "Forest", "accuracy": calculate_performance(rf, True, testing_X, testing_y)})
            convert_labels(file, testing_y)
            results.append({"model": "Neural Net", "accuracy": calculate_nn_performance(nn, True, testing_X, testing_y)})

            for result in results:
                df.append({"model": result["model"], "accuracy": round(result["accuracy"],2), "file": "penguins"})

    return pd.DataFrame(df)

# makes plots 
def make_plots(df):
    # create a plot for each of the datasets
    for file in ["mnist1000"]: #["penguins","mnist1000","seoulbike","energy"]:
        fileDf = df.loc[df['file'] == file]
        testLengths = {
            "penguins": 342,
            "mnist1000": 10000,
            "seoulbike": 8465,
            "energy": 19735
        }

        classification = True if file == "penguins" or file == "mnist1000" else False
        if classification:
            metric = "accuracy"
            fileDf['CIlower'] = fileDf.apply(lambda row: row.accuracy - 2.39 * np.sqrt(row.accuracy*(1-row.accuracy)/(.75*testLengths[file])), axis=1)
            fileDf['CIupper'] = fileDf.apply(lambda row: row.accuracy + 2.39 * np.sqrt(row.accuracy*(1-row.accuracy)/(.75*testLengths[file])), axis=1)
        else:
            metric = "MAE"
            fileDf['CIlower'] = fileDf.apply(lambda row: row.MAE - 2.24 * np.sqrt(row.VMAE/(.75*testLengths[file])), axis=1)
            fileDf['CIupper'] = fileDf.apply(lambda row: row.MAE + 2.24 * np.sqrt(row.VMAE/(.75*testLengths[file])), axis=1)
        # create the plot
        print(fileDf)
        plot = (
            ggplot(fileDf, aes(x="model", y=metric, fill="model"))
            + geom_bar(stat="identity")
            + geom_errorbar(aes(x="model", ymin="CIlower", ymax="CIupper"))
            + labs(title=f'{file.capitalize()} {metric[0].upper()}{metric[1:]}, by Model', x="Model", y=f"{metric[0].upper()}{metric[1:]}")
            + theme(legend_position = "none")  
        )
        # save the plot to a file
        plot.save(f'./figures/{file}.png')

            
# main methods calls train_models and make_plots to generate desired plots
def main():
    start_time = time.time()

    # color codes for clearer error message
    [WHITE, DARKCYAN, BOLD, UNBOLD] = ['\033[37m', '\033[36m', '\033[1m', '\033[0m']

    fullDF = train_models()
    # print(fullDF)
    make_plots(fullDF)

    print(f'{BOLD}{DARKCYAN}Time taken: {round(time.time()-start_time,2)}s{WHITE}{UNBOLD}') # print time taken

if __name__ == "__main__":
    main()