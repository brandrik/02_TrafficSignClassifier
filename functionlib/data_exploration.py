# DATASET SUMMARY AND EXPLORATION


# IMPORT
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from itertools import groupby

from functionlib.imageprocessing import *



# FUNCTIONS
def getCSVcellData(filepath) -> np.ndarray:
    return pd.read_csv(filepath).values

sign_names_and_labels = getCSVcellData('./signnames.csv')

def plotTrafficSigns(X, y, n_classes, fig) -> np.ndarray:
    for i in (range (n_classes)):       
        #print (range (n_classes))    
        #subplot(nrows, ncols, index, **kwargs) 
        plt.subplot(15, 4, i+1)
        selected = X[y == i] # select train image with index i equal to index in 'signnames.csv'
        
        if (X.shape[3]==1):
            plt.imshow(selected[0, :, :, -1], cmap='gray') # takes x*y dimensions draw only first image of every class grayscale
        else:  
            plt.imshow(selected[0, :, :, :]) # draw only first image of every class
        
        plt.title(sign_names_and_labels[i])
        plt.axis('off')

    plt.show()


    

# Distribution of images
def plotSignsHistogram(y, title):  
    
    fig, ax = plt.subplots(figsize=(10, 15))
    
    labels = np.arange(len(sign_names_and_labels))   # 0 .. 42
    
    tmp = np.sort(y, axis=0, kind='heapsort')
    
    numOfEach_trainset = [len(list(group)) for key, group in groupby(tmp)]
    
    ax.barh(labels, numOfEach_trainset, align='center', color='blue')
    ax.set_yticks(labels)
    ax.set_yticklabels(sign_names_and_labels)
    ax.invert_yaxis()  # display labels from top-to-bottom
    ax.set_xlabel('Number of each')
    ax.set_title(title)

    plt.show()
    
    
    
def augment_data(X,y,n_classes, limit=800):
    print('X, y shapes:', X.shape, y.shape)
    for class_n in range(n_classes):
        print(class_n, ': ', end='')
        class_indices = np.where(y == class_n)
        n_samples = len(class_indices[0])
        if n_samples < limit:
            for i in range(limit - n_samples):
                new_img = X[class_indices[0][i % n_samples]]
                new_img = random_translate(random_scale(random_warp(random_brightness(new_img))))
                X = np.concatenate((X, [new_img]), axis=0)
                y = np.concatenate((y, [class_n]), axis=0)
                # show progress:
                if i % 100 == 0:
                    print('|', end='')
                elif i % 50 == 0:
                    print('+',end='')    
                elif i % 10 == 0:
                    print('-',end='')
        print('')
                
    print('X augmented, y augmented shapes:', X.shape, y.shape)
    return (X,y)    
