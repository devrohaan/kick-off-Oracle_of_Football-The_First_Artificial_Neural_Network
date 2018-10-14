#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:47:10 2018

@author: Rohan
"""
from tabulate import tabulate
from math import ceil
print(tabulate([
                
                [24, 37, 3290, 5, 1, 34],
                [25, 52, 4653, 5, 0, 53],
                [26, 56, 5046, 6, 0, 60],
                [27, 57, 5009, 13, 0, 59],
                [28, 54, 4794, 7, 1, 65],
                [29, 48, 3290, 5, 1, 51],
                [30, 57, 5243, 6, 0, 56],
                [31, 44, 4893, 3, 1, 48],
                [32, 40, 3000, 0, 0, 'Predict']
    
               ], 
                headers=['Age', 'All appearance(s)','Playing Time (min)','Yellow Card','Red Card','Goals']
            ))


import numpy as np

X = np.array((  
                [24, 37, 3290/60, 5, 1],
                [25, 52, 4653/60, 5, 0],
                [26, 56, 5046/60, 6, 0],
                [27, 57, 5009/60, 13, 0],
                [28, 54, 4794/60, 7, 1],
                [29, 48, 3290/60, 5, 1],
                [30, 57, 5243/60, 6, 0],
                [31, 44, 4893/60, 3, 1]),
             dtype=float)


Y = np.array(([34], [53], [60], [59], [65], [51], [56], [48]), dtype=float)



X = X/np.amax(X, axis=0) # maximum of X array # maximum of xPredicted (our input data for the prediction)
Y = Y/100 



class NN(object):
    
    def __init__(self):
        
        self.iplayer_neurons = X.shape[1] # Total Number of features : 5
        self.hiddenlayer_neurons = 3
        self.oplayer_neurons = 1
        self.epoch = 1000
        self.learning_rate = 0.05
        
        
        self.w_hidden = np.random.randn(self.iplayer_neurons, self.hiddenlayer_neurons) # (5x3) weight matrix from input to hidden layer
        self.b_hl = np.random.randn(1, self.hiddenlayer_neurons)
        self.w_out = np.random.randn(self.hiddenlayer_neurons, self.oplayer_neurons) # (3x1) weight matrix from hidden to output layer
        self.b_ol = np.random.randn(1, self.oplayer_neurons)
        
      
        
        
    def forward_Propogation(self, X):
        
        self.hl_input = np.dot(X, self.w_hidden) # 8*5 dot 5*3 = 8 * 3
        self.hl_input = self.hl_input + self.b_hl
        
        self.hl_activations = self.sigmoid(self.hl_input)
        
        self.ol_input = np.dot(self.hl_activations, self.w_out) # 8*3 dot 3*1 = 8 * 1
        self.ol_input = self.ol_input + self.b_ol
        output = self.sigmoid(self.ol_input)
        return output # or ol_activation
    
    def backward_Propogation(self, X, Y, output):
        
        """
        gradient of hidden and output layer neurons 
        """
        #print(output)
        #print(Y)
        E = Y-output
        print("Error Calculated: ",E)
        slope_ol = self.sigmoidPrime(output)
        slope_hl = self.sigmoidPrime(self.hl_activations)
    
        delta_ol = E * slope_ol
        Error_at_hidden_layer = delta_ol.dot(self.w_out.T)
        delta_hl = Error_at_hidden_layer * slope_hl
    
        # Update weight at both output and hidden layer
    
        self.w_out += self.hl_activations.T.dot(delta_ol) * self.learning_rate
        self.b_ol += np.sum(delta_ol, axis=0,keepdims=True) * self.learning_rate
        self.w_hidden += X.T.dot(delta_hl) * self.learning_rate
        self.b_hl += np.sum(delta_hl, axis=0,keepdims=True) * self.learning_rate
    
    # Activation Funtion: Sigmoid 
    def sigmoid (self, x):
        return 1/(1 + np.exp(-x))

    #Derivative of Sigmoid Function
    def sigmoidPrime(self, x):
        return x * (1 - x)
    
    def train(self, X, Y):
        output = self.forward_Propogation(X)
        self.backward_Propogation(X, Y, output)
    
    def saveWeights(self):
        
        np.savetxt("/Users/Rohan/Desktop/3rdAug/FirstNeuralNetwork/w_hidden.txt", self.w_hidden, fmt="%s")
        np.savetxt("/Users/Rohan/Desktop/3rdAug/FirstNeuralNetwork/w_out.txt", self.w_out, fmt="%s")
    
    def loadWeights(self):
        
        w_hidden = np.loadtxt("/Users/Rohan/Desktop/3rdAug/FirstNeuralNetwork/w_hidden.txt", delimiter=" ")
        w_out = np.loadtxt("/Users/Rohan/Desktop/3rdAug/FirstNeuralNetwork/w_out.txt", delimiter=" ")
        
    
    def goalScorePrediction(self, age, appearance, playingtime, yellowcards,redcards):
        
        predict = np.array(([age, appearance, playingtime, yellowcards, redcards]), dtype=float)
        predict = predict/np.amax(predict, axis=0)
        print("Oracle_of_Football predicts %s goals will be scored by Ronaldo when he is %d years old." %("".join(str(ceil(self.forward_Propogation(predict).item(0)*100))), age))
        #print(str(self.forward_Propogation(predict)*100))
        self.saveWeights()
        
        
    
nn = NN()

if __name__=='__main__':
    
    for i in range(nn.epoch): # trains the NN 1,000 times
        
        nn.train(X, Y)
        #print(str(nn.forward_Propogation(X)))
    
    
    nn.goalScorePrediction(32, 40, 3000, 0, 0)
    nn.saveWeights()
    nn.loadWeights()

"""
OUTPUT:
    
    Oracle_of_Football predicts 56 goals will be scored by Ronaldo when he is 32 years old.
    
"""   