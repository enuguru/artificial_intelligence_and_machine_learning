
# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(2):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
                #print(response)
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
# prepare data
trainingSet=[]
testSet=[]
split = 0.67
loadDataset('bankloan.data', split, trainingSet, testSet)
print()
print("This application shows a Classificaion model using KNN Algorithm")
print()
print("In this model we are predicting if a bank customer will repay the loan or will not repay the loan")
print()
print("This model is very useful in the banking domain for doing credit risk assesment for the bank so that the bank does not end up having NPAs (non performing assets)")
print()
print("We are splitting the given data in to training set and test set")
print()
print("The number of Training Sets are")
print('Train set: ' + repr(len(trainingSet)))
print()
print("The number of Test Sets are")
print('Test set: ' + repr(len(testSet)))
#print(trainingSet)
# generate predictions
predictions=[]
testsetactuals=[]
for i in range(len(testSet)):
	testsetactuals.append(testSet[i][2])
print("The actual data in the test set are")
print(testsetactuals)
k = 3
for x in range(len(testSet)):
	neighbors = getNeighbors(trainingSet, testSet[x], k)
	result = getResponse(neighbors)
	predictions.append(result)
print()
print("The Predictions for the customers in the Test Set are")
print(predictions)
print()
print("The Accuracy of the Predictions is")
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')
