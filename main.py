# Load pickled data
import pickle
import os
from IPython.core.debugger import set_trace

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


def plotSigns(X,y, fig) -> np.ndarray:
    #n_samples = []
    for i in (range (n_classes)):       
        #print (range (n_classes))    
        #subplot(nrows, ncols, index, **kwargs) 
        plt.subplot(15, 4, i+1)
        selected = X[y == i] # select train image with index i equal to index in 'signnames.csv'
        
        #print(selected)
        
        if (X.shape[3]==1):
            plt.imshow(selected[0, :, :, -1], cmap='gray') # takes x*y dimensions draw only first image of every class grayscale
        else:  
            plt.imshow(selected[0, :, :, :]) # draw only first image of every class
        
        #print("selected")
        #print(selected[0, :, :, :])
        
        plt.title(all_labels[i])
        plt.axis('off')
        #n_samples.append(len(selected))
    fig.show()

    
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

f1 = plt.figure(1, figsize=(20,20))
plotSigns(X_train, y_train, f1)

## Histogram training set
f2 = plt.figure(2, figsize=(20,20))
plotHistogram(y_train, "Histogram of Trainining Set")


### Histogram validation set
#plt.figure(3, figsize=(20,20))
#plotHistogram(y_valid, "Histogram of Validation Set")

### Histogram test set
#plt.figure(4, figsize=(20,20))
#plotHistogram(y_test, "Histogram of Test Set")



#### Step 2: Design and Test a Model Architecture

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 

from sklearn.utils import shuffle
import cv2

# Shuffle data
X_train, y_train = shuffle(X_train, y_train)

 

### converting to grayscale, etc.
def grayScale(images) -> np.ndarray:
    return (np.sum(images/3, axis=3, keepdims=True))

X_train_g = grayScale(X_train) 
X_test_g = grayScale(X_test) 
X_valid_g = grayScale(X_valid) 


# Normalize
def normalize(data):
    return (data - 128) / 128

X_train_ng = normalize(X_train_g) 
X_test_ng = normalize(X_test_g) 
X_valid_ng = normalize(X_valid_g)


print("last")
f3 = plt.figure(3, figsize=(20,20))
#plotSigns(X_train_ng, y_train, f3)



#### MODEL ARCHITECTURE

from tensorflow.contrib.layers import flatten
import tensorflow as tf

def LeNet(x, n_labels):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # 5 by 5 filter, input depth of 1 and output depth of 6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    
    # Initialize the bias:
    
    conv1_b = tf.Variable(tf.zeros(6))
    
    # use conv2 function to convolve the filter with the input (over the images) 
    # and add the bias
    #conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], use_cudnn_on_gpu=True, padding='VALID') + conv1_b

    # SOLUTION: Activation. with relu activation function
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6. 
    # Resampling: 2 by 2 kernel with a 2 by 2 stride, which gives a pooling output of 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    #conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], use_cudnn_on_gpu=True, padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    # Flatten output into a vector
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_labels), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_labels))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

# tf.one_hot() on windows in GPU mode failed with CUDA_ERROR_ILLEGAL_ADDRESS
# https://github.com/tensorflow/tensorflow/issues/6509
def one_hot_workaround(y, num_labels):
    sparse_labels = tf.reshape(y, [-1, 1])
    derived_size = tf.shape(sparse_labels)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.concat(0, [tf.reshape(derived_size, [1]), tf.reshape(num_labels, [1])])
    return tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))

# y_one_hot = tf.one_hot(y, n_classes)
y_one_hot = one_hot_workaround(y, n_classes)
logits = LeNet(x, n_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_one_hot)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation)

print('Done!')


## TRAIN, VALIDATE AND TEST THE MODEL