# NEURAL NETWORK MODELS

# IMPORT
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

# MODELS

def LeNet(x, num_labels, keep_prob):
    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # LAYER 1: Convolutional, Input = 32x32x1. Output = 28x28x48.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 48), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros([48]))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)

    # L1: Max Pooling, Input: 28x28x48, Output: 14x14x48
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    # LAYER 2: Convolutional, Input: 14x14x48, Output: 10x10x96.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 48, 96), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros([96]))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)

    # L2: Max Pooling, Input: 10x10x96, Output: 5x5x96.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    
    # LAYER 3: Convolutional, Input: 5x5x96. Output: 3x3x172.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 96, 172), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros([172]))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    conv3 = tf.nn.relu(conv3)
    
    # L3: Max Pooling, Input: 3x3x172, Output: 2x2x172.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
    
    # L3: Flatten. Input = 2x2x172. Output = 688.
    fc1 = flatten(conv3)
    
    
    # LAYER 4: Fully Connected, Input: 688,  Output: 84
    fc2_W = tf.Variable(tf.truncated_normal(shape=(688 , 84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros([84]))
    fc2 = tf.nn.xw_plus_b(fc1, fc2_W, fc2_b)
    fc2 = tf.nn.relu(fc2)

    # LAYER 5: Fully Connected, Input: 84, Output: 43
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, num_labels), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros([num_labels]))
    logits = tf.nn.xw_plus_b(fc2, fc3_W, fc3_b)
    
    return logits



def LeNetSermanet(x, n_labels, keep_prob):    
    
    # Hyperparameters
    mu = 0          # mean value of weights
    sigma = 0.1     # standard deviation of weights
    

    # LAYER 1: Convolutional - Input: 32x32x1, Output: 28x28x48
    
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 48), mean = mu, stddev = sigma), name="W1")
    conv1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(48), name="b1")
    conv1 = tf.nn.bias_add(conv1, b1)
    print("layer 1 shape:",conv1.get_shape())

    # L1: Activation.
    conv1 = tf.nn.elu(conv1)
    
    # L1: Pooling. Input: 28x28x6, Output: 14x14x48.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer1 = conv1   # 1st stage features later fed to classifier
    
    # END OF 1st STAGE
    
    
    # SECOND STAGE
    
    # LAYER 2: Convolutional - Input: 14x14x48,  Output: 10x10x16
    
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 48, 96), mean = mu, stddev = sigma), name="W2")
    conv2 = tf.nn.conv2d(conv1, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(96), name="b2")
    conv2 = tf.nn.bias_add(conv2, b2)
                     
    # L2: Activation
    conv2 = tf.nn.elu(conv2)

    # L2: Pooling - Max Pooling, Input: 10x10x96, Output: 5x5x96.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer2 = conv2
    
    
    
    # LAYER 3: Convolutional - Input = 5x5x96, Output = 3x3x192.
    
    # Input: 5x5x96. Output: 3x3x172.
    W3 = tf.Variable(tf.truncated_normal(shape=(3, 3, 96, 192), mean = mu, stddev = sigma), name="W3")
    conv3 = tf.nn.conv2d(conv2, W3, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(192), name="b3")
    conv3 = tf.nn.bias_add(conv3, b3)
    
                     
    # L3: Activation
    conv3 = tf.nn.elu(conv3)
    layer3 = conv3
    
    

    # L1: Additional pooling / subsampling to enable conacatenation with other branch
    layer1 = tf.nn.max_pool(layer1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID')

    # L1: Flatten - Input = 14x14x48, Output = 9408
    layer1flat = flatten(layer1)
    print("layer1flat shape:",layer1flat.get_shape())
    
    
    # L3: Flatten - Input = 3x3x192, Output
    conv3flat = flatten(conv3)
    print("conv3flat shape:",conv3flat.get_shape())
    
    
    # COMBINE BOTH BRANCHES
    # Concat layer1flat and conv3flat: Input:,  Output:

    common = tf.concat_v2([conv3flat, layer1flat], 1)
    print("common shape:", common.get_shape())
    
    # Regularization - Dropout:
    common = tf.nn.dropout(common, keep_prob)
    
    # LAYER 4: Fully Connected, Input:,  Output: 43 (=n_labels)
    W4 = tf.Variable(tf.truncated_normal(shape=(3456, n_labels), mean = mu, stddev = sigma), name="W4")
    b4 = tf.Variable(tf.zeros(n_labels), name="b4")    
    logits = tf.add(tf.matmul(common, W4), b4)

    
    return logits

print('Modified Model - Done')