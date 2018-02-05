# Load pickled data
from IPython.core.debugger import set_trace
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf


from functionlib.data_exploration import *
from functionlib.models import *

#from lib.imageprocessing import *
#from lib.data_exploration import *

# LOAD DATA

path = "./traffic-signs-data/"
training_file = path + "train.p"
validation_file = path + "valid.p"
testing_file = path + "test.p"

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


# LOAD STORED DATA INSTEAD


# DATA EXPLORATION

sign_names_and_labels = []   

def getCSVcellData(filepath):
    return pd.read_csv(filepath).values

sign_names_and_labels = getCSVcellData('./signnames.csv')
# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(sign_names_and_labels)




## TODO: Number of training examples
#n_train = len(X_train)

## TODO: Number of validation examples
#n_validation = len(X_valid)

## TODO: Number of testing examples.
#n_test = len(X_test)

## TODO: What's the shape of an traffic sign image?
#image_shape = X_train[0].shape



# Plots
plt.close('all')

#f1 = plt.figure(1, figsize=(20,20))
#plotTrafficSigns(X_train, y_train, n_classes, f1)

## Histogram training set
#f2 = plt.figure(2, figsize=(20,20))
#plotSignsHistogram(y_train, "Histogram of Trainining Set")

# save image to image folder

### Histogram validation set
#plt.figure(3, figsize=(20,20))
#plotHistogram(y_valid, "Histogram of Validation Set")

## Histogram test set
#plt.figure(4, figsize=(20,20))
#plotHistogram(y_test, "Histogram of Test Set")


# PREPROCSS DATA

X_train_gn = preprocess(X_train) 
X_test_gn = preprocess(X_test) 
X_valid_gn = preprocess(X_valid)



# SET UP MODEL
tf.reset_default_graph() # Clears the default graph stack and resets the global default graph



x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32) # probability to keep units

y_one_hot = tf.one_hot(y, n_classes)

# Select model
learning_rate = 0.007



logits = LeNetSermanet(x, n_classes, keep_prob)

graph = tf.get_default_graph()

#The default graph is a property of the current thread. 
#If you create a new thread, and wish to use the default 
#graph in that thread, you must explicitly add a 
#with g.as_default(): in that thread's function. 
#(See - https://www.tensorflow.org/api_docs/python/tf/get_default_graph.)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_one_hot)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.007)
training_operation = optimizer.minimize(loss_operation)
    
    
    ## Histogram training set
    #f3 = plt.figure(3, figsize=(20,20))
    #plotHistogram(y_train, "Histogram of Augmented Trainining Set")    


# TRAIN, VALIDATE AND TEST THE MODEL

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

####
path = './augm_data_2000_flat.npz'
loaded = np.load(path)
X_training = loaded[loaded.files[0]]
y_training = loaded[loaded.files[1]]

#[X_training, y_training] = augment_data(X_train_gn, y_train,n_classes, limit = 1)


#f3 = plt.figure(2, figsize=(10,10))
#plotSignsHistogram(y_training, "Histogram of Augmented Trainining Set")


EPOCHS = 40
BATCH_SIZE = 128


def evaluate(X_data, y_data):
    #set_trace()
    n_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, n_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        
        # turn off drop out for evaluation with validation data (by setting keep_prop = 1)
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1}) 
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / n_examples


with tf.Session() as sess:
    ####
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    n_examples = len(X_training)

    print("Training...")
    for i in range(EPOCHS):
        print("EPOCH {} ... ".format(i+1), end='')
        X_training, y_training = shuffle(X_training, y_training)
        #set_trace()
        for offset in range(0, n_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_training[offset:end], y_training[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.3})
        validation_accuracy = evaluate(X_valid_gn, y_valid)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        
    saver.save(sess, './lenet')
    print("Model saved")
    
# Validate
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_gn, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))