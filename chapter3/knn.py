from sklearn import datasets
import numpy as np
import math
import operator
import pandas as pd
import os
from sklearn.cross_validation import train_test_split

# load iris in data folder
def load_data():
    iris = pd.read_csv("./data/iris.csv")

    return iris


# Defining a function which calculates euclidean distance between two data points
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

def sort_data(distances):
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
    return classVotes



def knn(training_data,test_data,k_param):
    dist = {}
    sort = {}

    length = test_data.shape[1]
    # Now as we have stated in our blog we will be finding euclidean distance between each row of trainin
    # and test dataset
    neighbors = []
    for i in range(len(training_data)):
        # Lets start calculating
        distance = euclideanDistance(test_data,training_data.iloc[i],length)

        # storing distances
        dist[i] = distance[0]

        # now we need to get the nearest neighbours
    sorted_ = sorted(dist.items(),key=operator.itemgetter(1)) 
    # extracting top k neighbors
    
    for x in range(k_param):
            neighbors.append(sorted_[x][0])

    classVotes = calculate_frequecies(neighbors,training_data)
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0],neighbors






