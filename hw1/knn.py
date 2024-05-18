from sys import argv as args
import random as ran
import csv 
import math
import time 

# This script is for the implementation of the n nearest neighbors algorithm
# This script can take in a csv file, run k nearest neighbors and output the accuracy and time taken

"""
Takes in csv filename, percentage of training set to total data, and a random seed

The generate_sets method opens and reads in the csv file, adding all the data to a 
list and splitting it into training and test data based on a random seed that is entered
"""
def generate_sets(filename, per, seed):
    # color codes to make error handling clearer
    RED =  '\033[31m' # Red text
    YELLOW = '\033[33m' # Yellow text
    # check validity of .csv and filename here
    try: 
        csv_file = open(filename, 'r') # open the file with 'r' as a parameter to specify read
    except: 
        print(YELLOW+"Usage: python knn.py "+RED+"file.csv"+YELLOW+" E/H k percent seed ")
        print(RED+"Make sure you have entered a .csv file that is available"+YELLOW)
        exit()
    reader = csv.reader(csv_file) # use the reader method of csv to create an iterable object
    instances = []
    for row in reader: 
        instances.append(row) # append all rows to the instances list
    instances = instances[1:] # remove labels
    # check validity of seed being an int
    try:
        ran.seed(int(seed)) # generate a number based on the seed
    except:
        print(YELLOW+"Usage: python knn.py file.csv E/H k percent"+RED+" seed")
        print("Enter an integer for the seed"+YELLOW)
        exit()
    shuffled = list(instances) 
    ran.shuffle(shuffled) # shuffles instances based on seed
    cutoff = int(len(shuffled)*per) # calculates the cutoff point
    # creates test and training sets 
    training = shuffled[:cutoff] 
    test = shuffled[cutoff:]
    return [training,test]

# The equivalent of java's helper function. 
# Calculates distance between two instances, given the instances and the metric to use
def knn_dist(test,neighbor,dis):
    dist = 0
    if dis == 'E': # Euclidian
        dist = math.sqrt(sum([(test[i] - float(neighbor[i]))**2 for i in range(1, len(test))]))
    if dis == 'H': # Hammond 
        for i in range(1, len(test)):
            dist += 0 if neighbor[i] == test[i] else 1 
    return [dist,neighbor[0]]

"""
Takes in test and training sets, distance metric, and hyperparameter k

The predict method runs the k nearest neighbor algorithm on the test set based off the training instances 
and returns the prediction of a given test instance in a [predicted label, actual label] pair. 
This method differentiates between Hammond and Euclidian distance and is capable of calculating both

The first instance of the predictions list is a set of all possible labels
"""
def predict(sets, dis, k): 
    RED =  '\033[31m' # Red text
    WHITE = '\033[37m' # White text
    pos_labels = set() # label set 
    training = sets[0]
    tests = sets[1]
    # makes predictions for each instance in the test based off the training data, using k nearest neighbors
    predictions = []
    for test in tests:
        if dis == 'E':
            try:
                # test if euclidian entered on an invalid data set
                for i in range(1,len(test)):
                    test[i] = float(test[i])
            except: 
                print(RED+"Trying to use Euclidian metric to determine distance between ordinal attributes"+WHITE)
                exit()
        pos_labels.add(test[0]) # add labels to labels set
        distances = []
        for neighbor in training: # calculate distance from all neighbors for each test
            output = knn_dist(test,neighbor,dis)
            distances.append(output) # store all distances with their labels 
        # sort all distances and grab k nearest
        distances.sort(key=lambda x: x[0]) # sort by distance
        k_nearest = distances[:k]
        labels = {}
        for near in k_nearest: # populate dictionary
            labels[near[1]] = labels[near[1]]+1 if near[1] in labels else 1 
        prediction = max(labels,key=labels.get) # find most common label and predict with it
        predictions.append([prediction, test[0]]) # add (prediction, actual) pair to predictions
    predictions.insert(0,pos_labels) # add label set to beg of predictions
    return predictions

"""
Main method to handle (most of the) command args, call the generate sets and predict methods 
and output the confusion matrix to the user

Since we are implementing this ourselves without libraries I tried to make the user experience 
more enjoyable with clearer error handling messages. 
"""
def main():
    # color codes for clearer error handlings
    # color codes found from this resource: https://www.instructables.com/Printing-Colored-Text-in-Python-Without-Any-Module/
    start_time = time.time()
    RED =  '\033[31m' # Red text
    YELLOW = '\033[33m' # Yellow text
    PINK = '\033[35m' # Pink text
    WHITE = '\033[37m' # White text
    # check num of args
    if len(args) != 6:
        print(YELLOW+"Need 5 arguments that correspond to file, metric used, # of neighbors, percentage to train on, and an integer to use as a random seed")
        print ("Correct Usage: "+PINK+"python knn.py file.csv E/H k percent seed"+WHITE)
        exit()    
    # check validity of percentage
    try:
        percent = float(args[4])
        if percent >= 1 or percent < 0:
            exit()
    except:
        print (YELLOW+"Usage: python knn.py file.csv E/H k"+RED+" percent"+YELLOW+" seed")
        print(RED+"Enter a number between 0 and 1"+YELLOW)
        exit()
    sets = generate_sets(args[1],percent,args[5]) # generate sets
    # check hammond/euclidian
    if (args[2] != 'H' and args[2] != 'E'):
        print (YELLOW+"Usage: python knn.py file.csv"+RED+" E/H "+YELLOW+"k percent seed ")
        print(RED+"Choose E or H as a metric"+YELLOW)
        exit()
    # check k is a pos int
    try:
        k = int(args[3])
        if k < 1:
            exit()
    except: 
        print(YELLOW+"Usage: python knn.py file.csv E/H "+RED+"k "+YELLOW+"percent seed")
        print(RED+"Please enter a non zero positive integer for k"+YELLOW)
        exit()
    output = predict(sets,args[2],k) # output[i] is in form of [prediction, actual]
    pos_labels = list(output[0]) # make labels subscriptable
    output = output[1:] # get rid of labels
    dic = {} # dict stores the tuples and counts 
    for i in range(len(output)):
        pair = tuple(output[i]) # stored as [prediction,test] tuple
        dic[pair] = dic[pair]+1 if pair in dic else 1
    # generating csv file output 
    file = args[1].split('.')
    name = str('results_'+file[0]+'_'+args[2]+'_'+str(k)+'_'+args[5]+'.csv')
    n = len(pos_labels)
    pos_labels.append("") # add an empty label for an extra comma
    accuracy = 0
    with open(name, 'w', newline='') as file: # creates a .csv named name
        writer = csv.writer(file)
        writer.writerow(pos_labels) # label row
        # for every pos label create a row and add it
        for i in range(n): 
            row = []
            for j in range(n):
                key = tuple([pos_labels[i],pos_labels[j]])
                val = dic.get(key) if dic.get(key) != None else 0 # get vals from dic, if not in dic then 0
                accuracy += val if pos_labels[i] == pos_labels[j] else 0 # calculate accuracy
                row.append(val)
            row.append(pos_labels[i]) # add actual labels to end of rows
            writer.writerow(row) # write row 
    
    # SE and confidence interval used to verify math was correct
    accuracy = accuracy/len(output)
    SE = (accuracy*(1-accuracy)/len(output))**0.5
    Con_Int = [accuracy-1.96*SE,accuracy+1.96*SE]
    
    print(f'Accuracy: {round(accuracy*100,2)}%') # print accuracy to the user
    print(f'Time taken: {round(time.time()-start_time,2)}s') # print time taken
if __name__ == "__main__":
    main()