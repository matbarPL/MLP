# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 21:57:07 2018

@author: Mateusz
"""
import pylab
import os
from matplotlib import pyplot as plt
plt.ioff()
import imageio
import subprocess
from random import shuffle
import numpy as np
from math import sqrt 
import seaborn as sn
import pandas as pd
import pickle as pkl

class Visualize():
    def __init__(self, path):
        '''class to visualize training results'''
        self.path = path
        with open('mnist.pkl', 'rb') as file:
            self.train_set, val_set, self.test_set = pkl.load(file, encoding='iso-8859-1')

    def printProgressBar(self,iteration, total, prefix = '', suffix = '', decimals = 1, length = 70, fill = '█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    
        if iteration == total: 
            print()
            
    def plot_accuracy(self,val_acc,train_acc,name):
        fig, ax = plt.subplots()
        ax.plot(range(len(val_acc)), val_acc, label='walidacyjny')
        ax.plot(range(len(train_acc)), train_acc, label='treningowy')
        ax.set_xlabel('Numer epoki')
        ax.set_ylabel('Dokladnosc')
        ax.set_title('Porównanie dokładnosci sieci na zbiorach')
        plt.legend()
        plt.savefig('accuracy_'+name+'.png')
        
    def plot_loss(self, loss, name):
        fig, ax = plt.subplots()
        ax.plot(range(len(loss)), loss, label='strata')
        ax.set_xlabel('Numer epoki')
        ax.set_ylabel('Calkowity koszt')
        ax.set_title('Funcja kosztu w zaleznosci od epoki')
        plt.legend()
        plt.savefig('loss_'+name+'.png')
        
    def show_figs(self):    
        fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
        ax = ax.flatten()
        for i in range(10):
            img = self.train_set[0][self.train_set[1]== i][0].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
    
    def show_nr(self, nr):
        fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
        ax = ax.flatten()
        for i in range(25):
            img = self.train_set[0][self.train_set[1]== nr][0].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
        
    def show_mistakes(self, y_true, y_pred):
        miscl_img = self.test_set[0][y_true != y_pred][:25]
        correct_lab = self.test_set[1][y_true != y_pred][:25]
        miscl_lab= y_pred[y_true != y_pred][:25]
        
        fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
        ax = ax.flatten()
        for i in range(25):
            img = miscl_img[i].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
            ax[i].set_title('%d) pr: %d est: %d' % (i+1, correct_lab[i], miscl_lab[i]))
        
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        #plt.savefig('./rysunki/12_09.png', dpi=300)
        plt.show()
            
