# Load pickled data
import pickle
import os

# TODO: Fill this in based on where you saved the training and testing data

path = "./traffic-signs-data/"
training_file = path + "train.p"
validation_file = path + "valid.p"
testing_file = path + "test.p"

# cwd= os.getcwd()
# print(cwd)

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print ("Image Shape: {}".format(X_train[0].shape))
print()
print ("Training Set: {} samples".format(len(X_train))) 
print ("Validation Set: {}".format(len(X_valid)))
print ("Test Set: {}".format(len(X_test)))


#### Step 1   Dataset Summary & Exploration

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import csv

all_labels = []

with open ('signnames.csv', 'r') as csvfile:
    readcsv = csv.reader(csvfile, delimiter=',')
    for line in readcsv: 
        all_labels += [line[1]]

all_labels.pop(0) # remove header string from array

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(all_labels)


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



# Plotting traffic sign images and their label

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
# Visualizations will be shown in the notebook.


def plotSigns(X,y) -> np.ndarray:
    #n_samples = []
    for i in (range (n_classes)):       
        #print (range (n_classes))    
        #subplot(nrows, ncols, index, **kwargs) 
        plt.subplot(15, 4, i+1)
        selected = X[y == i] # select train image with index i equal to index in 'signnames.csv'
        
        #print(selected)
        
        plt.imshow(selected[0, :, :, :]) # draw only first image of every class
        
        #print("selected")
        #print(selected[0, :, :, :])
        
        plt.title(all_labels[i])
        plt.axis('off')
        #n_samples.append(len(selected))
    #plt.show()

    
# Distribution of images
def plotHistogram(y, title):  
    
    fig, ax = plt.subplots(figsize=(10, 15))
    
    labels = np.arange(len(all_labels))   # 0 .. 42
    numOfEach_trainset = [len(list(group)) for key, group in groupby(y)]
    
    ax.barh(labels, numOfEach_trainset, align='center', color='blue')
    ax.set_yticks(labels)
    ax.set_yticklabels(all_labels)
    ax.invert_yaxis()  # display labels from top-to-bottom
    ax.set_xlabel('Number of each')
    ax.set_title(title)

    plt.show()



plt.close('all')

plt.figure(1, figsize=(20,20))
plotSigns(X_train, y_train)

## Histogram training set
plt.figure(2, figsize=(20,20))
plotHistogram(y_train, "Histogram of Trainining Set")


### Histogram validation set
#plt.figure(3, figsize=(20,20))
#plotHistogram(y_valid, "Histogram of Validation Set")

## Histogram test set
plt.figure(4, figsize=(20,20))
plotHistogram(y_test, "Histogram of Test Set")