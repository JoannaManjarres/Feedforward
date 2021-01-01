#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 06:18:48 2020

@author: joanna
"""

import numpy as np
import csv
import tensorflow as tf
import matplotlib.pyplot as plt

def obtener_coordenadas_validas():

    # filename = "../../data/processed/coord_input/CoordVehiclesRxPerScene_s008.csv"
    filename = "data/CoordVehiclesRxPerScene_s008.csv"
    limit_ep_train = 1564
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        coordinates_train = []
        coordinates_test = []      
        x_coordinates_train=[]
        y_coordinates_train=[]
        
        for row in reader:
            isValid = row['Val'] #canal valido -> V o Canal invalido-> I 
            if isValid == 'V': #check if the channel is valid
                if int(row['EpisodeID']) <= limit_ep_train:
                    # coordinates_train.append([int(row['EpisodeID']),float(row['x']),float(row['y']),float(row['z'])])
                    x_coordinates_train.append(float(row['x']))
                    y_coordinates_train.append(float(row['y']))
                if int(row['EpisodeID']) > limit_ep_train:
                    coordinates_test.append([int(row['EpisodeID']),float(row['x']),float(row['y']),float(row['z'])])
    
    
    print(x_coordinates_train)
    print(max(x_coordinates_train))
    print("x: ",max(x_coordinates_train), "y: ", max(y_coordinates_train))
    print("x: ",min(x_coordinates_train), "y: ", min(y_coordinates_train))
     
    #normalizando as entradas
    x_train = tf.keras.utils.normalize([x_coordinates_train])
    y_train = tf.keras.utils.normalize(y_coordinates_train)
    
   
    
    coordinates_train = x_train + y_train
    
    #coordenadas x e y (entradas)
    # x_train = coordinates_train[:,1:3]
    # x_test = coordinates_test[:,1:3]
    
    return coordinates_train, coordinates_test, limit_ep_train

coordinates_train, coordinates_test, limit_ep_train = obtener_coordenadas_validas()

print(max(coordinates_train[0]), max(coordinates_train[1]) )

# #coordenadas x e y (entradas)
# x_coordenate_train = coordinates_train[:,1:3]
# x_coordenate_test = coordinates_test[:,1:3]

# #normalizando as entradas
# x_train = tf.keras.utils.normalize(x_coordenate_train)
# x_test = tf.keras.utils.normalize(x_coordenate_test)


# print("vr maximo: ",max(x_coordenate_train[0]), max(x_coordenate_train[1]))
# print("vr minimo: ",min(x_coordenate_train[0]), min(x_coordenate_train[1]))     
      
def getBeamOutput(output_file):
    
    thresholdBelowMax = 1
    
    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']
    
    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]
    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y = beamsLogScale(y,thresholdBelowMax)
    
    return y,num_classes

def beamsLogScale(y,thresholdBelowMax):
        y_shape = y.shape
        
        for i in range(0,y_shape[0]):            
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs + 1e-30)
            minValue = np.amax(logOut) - thresholdBelowMax
            zeroedValueIndices = logOut < minValue
            thisOutputs[zeroedValueIndices]=0
            thisOutputs = thisOutputs / sum(thisOutputs)
            y[i,:] = thisOutputs
        
        return y
 
baseline_path = "data/processed/output_beam/baseline/"

output_train_file = baseline_path+'beams_output_train.npz'
y_train,num_classes = getBeamOutput(output_train_file)

output_test_file = baseline_path+'beams_output_validation.npz'
y_test, _ = getBeamOutput(output_test_file)

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    error_vector=[]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        error_vector.append(sum_error)
        print('>epoch=%d, lrate=%.3f, error=%.5f' % (epoch, l_rate, error))
    plt.plot(error_vector)
    # plt.plot(history.history['val_acc'])
    plt.ylabel('Error')
    plt.xlabel('epoch')
    plt.show()
    return weights

# Make a prediction with weights
def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
    
	return correct / float(len(actual)) * 100.0

def evaluate_algorithm(train_set, test_set, algorithm, n_folds, *args):
    predicted = algorithm(train_set, test_set, *args)

# evaluate algorithm
n_folds = 2
l_rate = 0.5
n_epoch = 10
scores = evaluate_algorithm(x_coordenate_train, x_coordenate_test, perceptron, n_folds, l_rate, n_epoch)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
