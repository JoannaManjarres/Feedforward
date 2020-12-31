#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 20:39:28 2020

@author: joanna
"""

import numpy as np


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

def getBeamOutput(output_file):
    
    thresholdBelowMax = 1
    
    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']
    print("Forma yMatrix",yMatrix.shape)
    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]
    
    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y = beamsLogScale(y,thresholdBelowMax)

    return y,num_classes

def generate_groups_beams(y,k):
    labels=[]
    
    for i in range(y.shape[0]):
        valor = round(np.argmax(y[i,:], axis=0)/k)
        labels.append(valor)
    
    
    return np.asarray(labels)

########################## MAIN #####################################################

def process_and_save_output_beams(k):
    if (k==0):
        print("k should be greater than zero")
    else:
        baseline_path = "../../data/processed/output_beam/baseline/"
        
        #train
        output_train_file = baseline_path+'beams_output_train.npz'
        y_train,num_classes = getBeamOutput(output_train_file)
        
        output_validation_file = baseline_path+'beams_output_validation.npz'
        y_validation, _ = getBeamOutput(output_validation_file)
        
        savePath = "../../data/processed/output_beam/"
        debugPath = "../../data/processed/output_beam/debug/"
        
        #train
        label_train = generate_groups_beams(y_train,k)
        debug_train = label_train[0:20]
        np.savez(savePath+'beams_output_train'+'.npz',output_training=label_train)
        np.savez(debugPath+'beams_output_train'+'.npz',output_training=debug_train)

        #test
        label_validation = generate_groups_beams(y_validation,k)
        debug_validation = label_validation[0:2]
        np.savez(savePath+'beams_output_validation'+'.npz',output_test=label_validation)
        np.savez(debugPath+'beams_output_validation'+'.npz',output_test=debug_validation)

