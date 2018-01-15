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

#f1 = plt.figure(1, figsize=(20,20))
#plotSigns(X_train, y_train, f1)

## Histogram training set
#f2 = plt.figure(2, figsize=(20,20))
#plotHistogram(y_train, "Histogram of Trainining Set")


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


## Preprocess data
# 1 Grayscale
# 2 Normalize
def preprocess(data):
    #return (data - 128) / 128
    ret_arr = np.ndarray(shape = (data.shape[0:3] + tuple([1])))
    a = -0.5
    b = 0.5
    minimum = 0
    i = 0
    maximum = 255
    for img in data:
        YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)  # change color space: Y Cr and Cb images, where Y essentially is a grayscale picture
        img_g = np.resize(YCrCb[:,:,0], (32,32,1))
        img_gn = a + ((img_g - minimum) * (b - a)) / (maximum - minimum)
        ret_arr[i, :, :, :] = img_gn
        i+=1
    return ret_arr


X_train_gn = preprocess(X_train) 
X_test_gn = preprocess(X_test) 
X_valid_gn = preprocess(X_valid)


#print("last")
f3 = plt.figure(3, figsize=(20,20))
#plotSigns(X_train_gn, y_train, f3)


# check if all grascaled by plotting 1st preprocessed image

image = X_train_gn[1, :, : ,-1]
cmap ='gray'
ex_per_sign = 1
fig = plt.figure(figsize = (ex_per_sign, 1))
fig.subplots_adjust(hspace = 0, wspace = 0)
i = 0
axis = fig.add_subplot(1,ex_per_sign, i+1, xticks=[], yticks=[])
axis.imshow(image, cmap=cmap)


#### MODEL ARCHITECTURE

from tensorflow.contrib.layers import flatten
import tensorflow as tf

#def LeNet(x, n_labels):    
    ## Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    #mu = 0
    #sigma = 0.1
    
    ## SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    ## 5 by 5 filter, input depth of 1 and output depth of 6
    #conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    
    ## Initialize the bias:
    
    #conv1_b = tf.Variable(tf.zeros(6))
    
    ## use conv2 function to convolve the filter with the input (over the images) 
    ## and add the bias
    ##conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    #conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], use_cudnn_on_gpu=True, padding='VALID') + conv1_b

    ## SOLUTION: Activation. with relu activation function
    #conv1 = tf.nn.relu(conv1)

    ## SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6. 
    ## Resampling: 2 by 2 kernel with a 2 by 2 stride, which gives a pooling output of 14x14x6
    #conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    ## SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    #conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    #conv2_b = tf.Variable(tf.zeros(16))
    ##conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    #conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], use_cudnn_on_gpu=True, padding='VALID') + conv2_b
    
    ## SOLUTION: Activation.
    #conv2 = tf.nn.relu(conv2)

    ## SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    #conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    ## SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    ## Flatten output into a vector
    #fc0   = flatten(conv2)
    
    ## SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    #fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    #fc1_b = tf.Variable(tf.zeros(120))
    #fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    ## SOLUTION: Activation.
    #fc1    = tf.nn.relu(fc1)

    ## SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    #fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    #fc2_b  = tf.Variable(tf.zeros(84))
    #fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    ## SOLUTION: Activation.
    #fc2    = tf.nn.relu(fc2)

    ## SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    #fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_labels), mean = mu, stddev = sigma))
    #fc3_b  = tf.Variable(tf.zeros(n_labels))
    #logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    #return logits


#x = tf.placeholder(tf.float32, (None, 32, 32, 1))
#y = tf.placeholder(tf.int32, (None))

#y_one_hot = tf.one_hot(y, n_classes)

#logits = LeNet(x, n_classes)
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_one_hot)
#loss_operation = tf.reduce_mean(cross_entropy)
#optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
#training_operation = optimizer.minimize(loss_operation)

#print('Done!')


def LeNet(x, num_labels):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # Convolutional Layer. Input = 32x32x1. Output = 28x28x48.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 48), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros([48]))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)

    # Max Pooling. Input = 28x28x48. Output = 14x14x48.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Convolutional Layer. Output = 10x10x96.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 48, 96), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros([96]))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)

    # Max Pooling. Input = 10x10x96. Output = 5x5x96.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Convolutional Layer. Input = 5x5x96. Output = 3x3x172.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 96, 172), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros([172]))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    conv3 = tf.nn.relu(conv3)
    
    # Max Pooling. Input = 3x3x172. Output = 2x2x172.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
    
    # Flatten. Input = 2x2x172. Output = 688.
    fc1 = flatten(conv3)
    
    # Fully Connected. Input = 688. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(688 , 84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros([84]))
    fc2 = tf.nn.xw_plus_b(fc1, fc2_W, fc2_b)
    fc2 = tf.nn.relu(fc2)

    # Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, num_labels), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros([num_labels]))
    logits = tf.nn.xw_plus_b(fc2, fc3_W, fc3_b)
    
    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))

y_one_hot = tf.one_hot(y, n_classes)
#y_one_hot = one_hot_workaround(y, n_classes)
logits = LeNet(x, n_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_one_hot)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation)


## TRAIN, VALIDATE AND TEST THE MODEL

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.



correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

EPOCHS = 35
BATCH_SIZE = 128

def evaluate(X_data, y_data):
    #set_trace()
    n_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, n_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / n_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_examples = len(X_train_gn)

    print("Training...")
    for i in range(EPOCHS):
        print("EPOCH {} ... ".format(i+1), end='')
        X_train_gn, y_train = shuffle(X_train_gn, y_train)
        #set_trace()
        for offset in range(0, n_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_gn[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        validation_accuracy = evaluate(X_valid_gn, y_valid)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        
    saver.save(sess, './lenet')
    print("Model saved")
    


## Test accuracy 
print("Compute Accuracy")
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_gn, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))