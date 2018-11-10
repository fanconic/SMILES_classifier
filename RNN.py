'''
Neural Network Project for CSCI3230, Fundamentals of Artificial Intelligence
Classifier that predicts if a drug SMILES value is toxious or not

Author: fanconic
'''
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import csv
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

'''--------------------------------------Data Preprocessing-------------------------------'''
path_train_names = './data/NR-ER-train/names_onehots.npy'
path_train_labels = './data/NR-ER-train/names_labels.csv'
path_test_names = './data/NR-ER-test/names_onehots.npy'
path_test_labels = './data/NR-ER-test/names_labels.csv'

# Write Lables from csv to onehot list
def construct_labels(path_to_file):
        labels = []
        with open(path_to_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter= ',')
                for row in csv_reader:
                        if int(row[1]) == 0:
                                labels.append([1,0])
                
                        elif int(row[1]) == 1:
                                labels.append([0,1])

        return np.asarray(labels)


# Write OneHots to list
def construct_names (path_to_file):
        names = []
        df = np.load(path_to_file).tolist()
        names = df.get('onehots')
        return np.asarray(names)

def create_validation(val_size ,X, y):
        index = np.arange(0, len(X))
        np.random.shuffle(index)
        index1 = index[:val_size]
        index2 = index[val_size:]
        X_val = [X[i] for i in index1]
        y_val = [y[i] for i in index1]
        X_train = [X[i] for i in index2]
        y_train = [y[i] for i in index2] 
        return np.asarray(X_val), np.asarray(y_val), np.asarray(X_train), np.asarray(y_train)

y_train = construct_labels(path_train_labels)
y_test = construct_labels(path_test_labels)

X_train = construct_names(path_train_names)
X_test = construct_names(path_test_names)

X_train = np.append(X_train, X_test, axis= 0)
y_train = np.append(y_train, y_test, axis= 0)

X_test, y_test, X_train, y_train = create_validation(265, X_train, y_train)
X_val, y_val, X_train, y_train = create_validation(265, X_train, y_train)




'''--------------------------------------Auxiliary Functions------------------------------'''

# Flatten the training data
def flatten_X(X):
        X_temp = np.empty([len(X), 72*398])
        for i in range(X.shape[0]):
                X_temp[i] = np.array(X[i]).flatten()
        return X_temp

# Make balanced 50/50 Batches from Training and Testing data
def next_batch_balanced(n, X, y):
        # Negative (non-toxic) Data
        neg_index, _ = np.where(y == ([1,0]))
        neg_index = neg_index[::2]
        X_neg = X[neg_index]
        y_neg = y[neg_index]
        neg_index = np.arange(0, len(X_neg))
        np.random.shuffle(neg_index)
        neg_index = neg_index[:int(n/2)]
        X_neg_shuffle = [X_neg[i] for i in neg_index]
        y_neg_shuffle = [y_neg[i] for i in neg_index]

        # Positive (Toxic) Data
        pos_index, _ = np.where(y == ([0,1]))
        pos_index = pos_index[::2]
        X_pos = X[pos_index]
        y_pos = y[pos_index]
        pos_index = np.arange(0, len(X_pos))
        np.random.shuffle(pos_index)
        pos_index = pos_index[:int(n/2)]
        X_pos_shuffle = [X_pos[i] for i in pos_index]
        y_pos_shuffle = [y_pos[i] for i in pos_index]

        # Merge Data to batch
        X_ = np.append(X_neg_shuffle, X_pos_shuffle, axis= 0)
        y_ = np.append(y_neg_shuffle, y_pos_shuffle, axis= 0)
        index = np.arange(0, n)
        np.random.shuffle(index)
        X_shuffle = [X_[i] for i in index]
        y_shuffle = [y_[i] for i in index]

        return np.asarray(X_shuffle), np.asarray(y_shuffle)

# Normal Next Batch function
def next_batch(n, X, y):
        index = np.arange(0, len(X))
        np.random.shuffle(index)
        index = index[:n]
        X_shuffle = [X[i] for i in index]
        y_shuffle = [y[i] for i in index]
        return np.asarray(X_shuffle), np.asarray(y_shuffle)

'''--------------------------------------Neural Network------------------------------------'''
# Neural Network Parameters
n_classes = 2
batch_size = 16
n_epochs = 5
num_hidden = 128 # hidden layer
num_input = 72
timesteps = 398
lr = 1e-3

# Placeholders
X = tf.placeholder(tf.float32, shape=[None, timesteps, num_input])
y = tf.placeholder(tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Deep Neural Feed Forward Network Model Method
def RNN_model(X):
        '''----------------------------------Network Architecture---------------------------------------------'''
        # Output Layer, fully connected
        out = {
                'weights': tf.Variable(tf.truncated_normal([num_hidden,n_classes])),
                'biases': tf.Variable(tf.truncated_normal([n_classes]))
        }

        '''-----------------------------------Network Calculation-----------------------------------------------'''
        
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        X = tf.unstack(X, timesteps, 1)

        # 1-layer LSTM with n_hidden units
        rnn_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Generate Prediction
        outputs, _ = rnn.static_rnn(rnn_cell, X, dtype= tf.float32)

        # There are n_input outputs, but we only want the last one
        return tf.matmul(outputs[-1], out['weights']) + out['biases']



# Method to train Neural Network Model
def train_RNN(X, epochs, lr):
        # Write to Log file
        file = open('./models/trash/log.txt','w') 

        pred = RNN_model(X)
        pred_classes = tf.argmax(pred, axis=1)
        labels = tf.argmax(y, axis=1)

        # Error function and Optimizer
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= pred, labels= y))
        optimizer = tf.train.AdamOptimizer(lr).minimize(error)


        # Tensorflow Session
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for epoch in range(epochs):
                        epoch_loss = 0
                        start = 0
                        # Training step
                        while start < len(y_train):
                                end = batch_size + start

                                X_batch, y_batch = next_batch_balanced(batch_size, X_train, y_train)
                                X_batch = X_batch.reshape((-1, timesteps, num_input))
                                

                                start += batch_size
                                # Running training
                                _, cost = sess.run([optimizer, error], feed_dict= {X: X_batch, y: y_batch})
                                epoch_loss += cost

                        print('Training:\nEpoch', epoch+1, 'completed out of', n_epochs, 'loss:', epoch_loss)
                        
                        y_true= sess.run(labels,feed_dict={X: X_val.reshape((-1, timesteps, num_input)), y: y_val, keep_prob: 1}) 
                        y_pred= sess.run(pred_classes, feed_dict={X: X_val.reshape((-1, timesteps, num_input)), y: y_val, keep_prob: 1})
                        print('Validation')
                        print("Balanced Accuracy: {:.4f}".format(balanced_accuracy_score(y_true, y_pred)))
                        print("Precision (positive): {:.4f}".format(precision_score(y_true, y_pred, pos_label=1)))
                        print("Precision (negative): {:.4f}".format(precision_score(y_true, y_pred, pos_label=0)))
                        print("Recall (positive): {:.4f}".format(recall_score(y_true, y_pred, pos_label=1)))
                        print("Recall (negative): {:.4f}".format(recall_score(y_true, y_pred, pos_label=0)))
                        
                # Testing
                y_true = sess.run(labels,feed_dict={X: X_test.reshape((-1, timesteps, num_input)), y: y_test, keep_prob: 1}) 
                y_pred = sess.run(pred_classes, feed_dict={X: X_test.reshape((-1, timesteps, num_input)), y: y_test, keep_prob: 1})
                
                print('Testing')
                print("Balanced Accuracy: {:.4f}".format(balanced_accuracy_score(y_true, y_pred)))
                print("Precision (positive): {:.4f}".format(precision_score(y_true, y_pred, pos_label=1)))
                print("Precision (negative): {:.4f}".format(precision_score(y_true, y_pred, pos_label=0)))
                print("Recall (positive): {:.4f}".format(recall_score(y_true, y_pred, pos_label=1)))
                print("Recall (negative): {:.4f}".format(recall_score(y_true, y_pred, pos_label=0)))
                #file.write(testing_string)

                saver = tf.train.Saver()
                saver.save(sess, './models/trash/dnn', global_step = 1)

# Train Neural Network
train_RNN(X, n_epochs, lr)

