# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:27:37 2018

@author: Mateusz
"""
import numpy as np
from matplotlib import pyplot as  plt
from inspect import getsourcefile
from random import shuffle
import os
from math import sqrt
import math
import pickle as pkl 
import matplotlib.pyplot as plt

class MLP():
    def __init__(self, mu, gamma, low, high, activation, n_iter, hidden, batch_size, eps=None, decr = 0):
        '''class representing multi layer neural network
        attributes:
            -path - path where file is stored
            -alpha - learning rate
            -weights - weights of a single neuron
            -activation - activation function one of the following linear, sigmoid, tanh
            -eta - adaptive learning rate coefficient
            -n_iter - number of iterations to be done 
            -beta - parameter for sigmoid function only 
            -batches - number of minibatches for which we split training set
            -n_hidden - number of neurons in hidden layer
            -eps - cost which indicate stopping train function
            -erl_stop - if true then early stopping technique is applied '''
        self.path= os.path.split(os.path.abspath(__file__))[0]
        os.chdir(self.path)
        self.mu = mu
        self.gamma = gamma
        self.low, self.high = low, high 
        self.n_iter = n_iter
        self.beta = 1
        self.batch_size = batch_size
        self.hidden = hidden
        self.decr = decr
        self.eps = eps
        self.err_incr = 1.04
        self.read_weights()
        self.read_sets()
        self.set_activation_function(activation)
        self.train_acc = []
        self.val_acc = []
        self.loss = []
        self.loss_epoch = []
        self.erl_stop = True
        
    def read_sets(self):
        '''method for reading test, validation and training sets'''
        with open('E:\\Sieci neuronowe\\mnist.pkl', 'rb') as file:
            train_set, valid_set, test_set = pkl.load(file, encoding='iso-8859-1')
        
        self.train_set = train_set[0]
        self.valid_set = valid_set[0]
        self.test_set = test_set[0]  
        
        #one-hot encoding output results
        self.train_set_hot = np.eye(10)[train_set[1]].T 
        self.valid_set_hot = np.eye(10)[valid_set[1]].T  
        self.test_set_hot = np.eye(10)[test_set[1]].T  
        
        self.n_features = self.train_set.shape[1]
        
    def add_bias_row(self, X):
        '''adding single row of bias to matrix X'''
        bias = np.ones((1, X.shape[1]))
        return np.vstack([bias, X])
        
    def add_bias_col(self, X):
        '''adding single column of bias to matrix X'''
        bias = np.ones((X.shape[0], 1))
        return np.hstack([bias, X])
        
    def set_weights(self):
        '''set random weights for input and hidden layers '''
        self.weights = np.random.uniform(low = self.low, high = self.high, size = (self.hidden,self.n_features+1))
        self.weights_hidden = np.random.uniform(low = self.low, high = self.high, size = (10,self.hidden+1))
        
        
    def train(self):
        '''main train function for training multilayer neural network'''
        self.set_weights() # randomly set weights
        p = self.train_set.shape[0] #total number of training samples
        weights_t = np.zeros(self.weights.shape) #weights at time t
        weights_hidden_t = np.zeros(self.weights_hidden.shape) #hidden weights at time t
        self.batches = int(p/self.batch_size)
        
        for epoch in range(self.n_iter):
            idx_split = np.array_split(range(p),self.batches) #split training array on smaller batches
            self.mu /= (1 + self.decr*epoch) 
            
            idx = np.random.permutation(self.train_set.shape[0]) #shuffling numbers in each epoch to prevent from adapting too much to pattern
            X_data = self.train_set[idx] #choose X_data 
            y_data = self.train_set_hot[:,idx]
            
            for idx in idx_split:
                #forward propagation algorithm start
                X = X_data[idx]
                X = self.add_bias_col(X)
                X = X.T
                z_2 = self.weights.dot(X)
                a_2 = self.activation(z_2)
                a_2 = self.add_bias_row(a_2)
                z_3 = self.weights_hidden.dot(a_2)
                y_p = self.activation(z_3)
                #forward propagation algorithm finished
                y_true = y_data[:,idx]
                Err = sum(sum((y_p-y_true)**2))/p
                self.loss.append(Err)
                #backpropagation algorithm start
                sigma3 = ( y_p - y_true)
                z_2 = self.add_bias_row(z_2)
                sigma2 = self.weights_hidden.T.dot(sigma3)*self.der_activation(z_2)
                sigma2 = sigma2[1:, :]
                X = X.T
                grad1 = sigma2.dot(X)
                grad2= sigma3.dot(a_2.T)
                delta_w1, delta_w2 = self.mu*grad1, self.mu*grad2
                #momentum w(t+1) = w(t) - (grad(t+1) + alpha*grad(t-1))
                self.weights -= (delta_w1 + (self.gamma*weights_t) )
                self.weights_hidden -= (delta_w2 + (self.gamma*weights_hidden_t) )
                weights_t, weights_hidden_t = np.copy(delta_w1), np.copy(delta_w2)
                #backpropagation algorithm finished
                print (Err)
            cost = np.mean(self.loss[epoch*self.batches:(epoch+1)*self.batches])
            epoch +=1 
            self.train_acc.append(self.get_acc('train'))
            self.val_acc.append(self.get_acc('valid'))
            print ('Training accuracy:', self.get_acc('train'))
            print ('Validation accuracy:', self.get_acc('valid'))
            print('Error at epoch ' ,str(epoch), cost)
            if (self.get_acc('valid')) > 0.99:
                print ('DONE')
                break
            self.loss_epoch.append(cost)
            
        np.save('weights.npy', self.weights)
        np.save('weights_hidden.npy', self.weights_hidden)
        return epoch
    
    def early_stopping(self):
        if len(self.val_acc)>3:
            return self.val_acc[-1] < self.val_acc[-2] < self.val_acc[-3]
        return False
    
    def test(self,type = 'test'):
        '''method for testing data along with forward propagation algorithm'''
        if type == 'test':
            test_set = self.add_bias_row(self.test_set.T)
        elif type =='valid':
            test_set = self.add_bias_row(self.valid_set.T)
        elif type =='train':
            test_set = self.add_bias_row(self.train_set.T)
        else:
            print ('Choose one of the following "test", "valid" or "train"')
            return 
        z_1 = self.weights.dot(test_set)
        a_1 = self.activation(z_1)
        a_1 = self.add_bias_row(a_1)
        z_2 = self.weights_hidden.dot(a_1)
        y_p = self.activation(z_2)
        pred = np.argmax(y_p,0)
        return pred

    def opt_mu(self):
        if len(self.loss)>1:
            ro_i = 1.05
            ro_d = 0.7
            k_w = 1.04
            if math.sqrt(self.loss[-1]) > k_w *math.sqrt(self.loss[-2]):
                self.mu *= ro_i
            elif (math.sqrt(self.loss[-1]) <= k_w *math.sqrt(self.loss[-2]) ):
                self.mu *= ro_d
            
    def get_acc(self, type ='test'):
        '''method for checking accuracy of data, either testing, validation or training'''
        y_p = self.test(type)
        if type == 'test':
            y_true = np.argmax(self.test_set_hot,axis=0)
        elif type == 'valid':
            y_true = np.argmax(self.valid_set_hot,axis=0)
        elif type == 'train':
            y_true = np.argmax(self.train_set_hot,axis=0)
        else:
            return 0 
        return np.sum(y_p== y_true)/y_p.shape[0]
        
    def set_activation_function(self, activation):
        self.activation = {'linear':self.linear, 'sigmoid':self.sigmoid, 'tanh':self.tanh}[activation]
        self.der_activation = {'linear':self.linear_der, 'sigmoid':self.sigmoid_der, 'tanh':self.tanh_der}[activation]

    def linear(self, X):
        return X
    
    def linear_der(self, X):
        return np.ones(X.shape)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-self.beta*X))

    def sigmoid_der(self, X):
        return self.beta*self.sigmoid(X)*(1-self.sigmoid(X))
    
    def tanh(self, X):
        return 2 / (1 + np.exp(self.beta*X)) - 1 
    
    def tanh_der(self, X):
        return self.beta*(1 - self.tanh(X) )**2
    
    def read_weights(self):
        '''function for reading weights from file'''
        self.weights = np.load('weights.npy')
        self.weights_hidden = np.load('weights_hidden.npy')

    def visualize(self, option, figName='default'):
        vis = Visualize(self.path)
        if option == 'show_figs':
            vis.show_figs()
            plt.show()
        elif option == 'show_nr':
            vis.show_nr(4)
        elif option =='show_mistakes':
            y_pred = self.test()
            y_true = np.argmax(self.test_set_hot,axis=0)
            vis.show_mistakes(y_true, y_pred)
        elif option == 'plot_accuracy':
            vis.plot_accuracy(self.val_acc, self.train_acc, figName)
        elif option == 'plot_loss':
            vis.plot_loss(self.loss_epoch, figName)
    
if __name__ == '__main__':
    #im mniejsza jest paczka tym wiekszy wspolczynnik uczenia
    mu = 0.1
    gamma = 1
    low, high = -1,1
    activation = 'sigmoid'
    n_iter = 10
    hidden = 500
    batches = 500
    mlp = MLP(mu, gamma, low, high, activation, n_iter, hidden, batches, eps=None, decr = 0)
    pred = mlp.train()
    acc = mlp.get_acc()
    print (acc)
    mlp.visualize('show_mistakes')