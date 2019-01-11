from sklearn import datasets
import numpy as np
import math
import operator
import pandas as pd
import os
from sklearn.cross_validation import train_test_split


def load_data():
    iris = pd.read_csv("./chapter3/data/iris.csv")
    X_train,X_test,y_train,y_test = train_test_split(iris,)
    return iris


# Defining a function which calculates euclidean distance between two data points
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

def sort(distances):
    sort_data = sorted(distances.items(), key=operator.itemgetter(1))
    return sort_data

# Calculating the most freq class in the neighbors
def calculate_frequecies(neighbors,trainingSet):
    classVotes = {}
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
 
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #### End of STEP 3.4
    return classVotes



def knn(training_data,test_data,k_param):
    dist = {}
    
    ## defining two variables  
    test_length = test_data.shape[1]

    length_ = test_data.shape[1]
    # Now as we have stated in our blog we will be finding euclidean distance between each row of trainin
    # and test dataset
    neighbors = []
    for i in range(len(training_data)):
        # Lets start calculating
        distance = euclideanDistance(test_data,training_data.loc[i],length)

        # storing distances
        dist[i] = distance

        # now we need to get the nearest neighbours
        for neighbours in range(k_param):
            neighbors.append(sort[i][0])

        classVotes = calculate_frequecies(neighbors)
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0],neighbors











