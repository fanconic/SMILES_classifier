
'''
Neural Network Project for CSCI3230, Fundamentals of Artificial Intelligence
Classifier that predicts if a drug SMILES value is toxious or not

Author: fanconic
'''
import tensorflow as tf
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

# A Molecule is composed of a onehot matrix of 72 x 396 Dimension
n_rows = len(X_train[0])
n_cols = len(X_train[0][0])
n_input_vec = n_cols * n_rows



'''--------------------------------------Neural Network------------------------------------'''
# Neural Network Parameters
batch_size = 128
n_classes = 2
n_epochs = 10
lr = 1e-4

# Placeholders
X = tf.placeholder(tf.float32, shape=[None, n_rows, n_cols])
y = tf.placeholder(tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(tf.float32)

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

def next_batch(n, X, y):
        index = np.arange(0, len(X))
        np.random.shuffle(index)
        index = index[:n]
        X_shuffle = [X[i] for i in index]
        y_shuffle = [y[i] for i in index]
        return np.asarray(X_shuffle), np.asarray(y_shuffle)

# Convolutional Neural Network Model Method
def CNN_model(X):
        ''' Convolutional Neural Network Model Method

        Parameters
        -----------
        X: placeholder, will be fed with featues

        Return
        ----------
        out: tensor, output layer of the convolutional neural network
        '''

        ''' ------------Network Calculation-------------'''
        # Reshaping vector
        X_train = tf.reshape(X, shape=[-1, n_rows, n_cols, 1], name= 'Input')
        tf.summary.image('input', X_train, 3)

        # First Convolutional Layer
        conv1 = tf.layers.conv2d(X_train, filters =16,
                                        kernel_size= 3,
                                        activation= tf.nn.relu,
                                        padding='SAME',
                                        name='conv1',
                                        kernel_initializer= tf.truncated_normal_initializer(),
                                        bias_initializer= tf.constant_initializer(0.1))

        # First Max Pooling
        pool1 = tf.layers.max_pooling2d(conv1, 3, 3, name='MaxPool_3x3')

        # First Convolutional Layer
        conv2 = tf.layers.conv2d(pool1, filters =32,
                                        kernel_size= 3,
                                        activation= tf.nn.relu,
                                        padding='SAME',
                                        name='conv2',
                                        kernel_initializer= tf.truncated_normal_initializer(),
                                        bias_initializer= tf.constant_initializer(0.1))

        # Second Max Pooling
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='MaxPool_2x2')

        # Second Convolutional Layer
        conv3 = tf.layers.conv2d(pool2, filters =64,
                                        kernel_size= 3,
                                        activation= tf.nn.relu,
                                        padding='SAME',
                                        name='conv3',
                                        kernel_initializer= tf.truncated_normal_initializer(),
                                        bias_initializer= tf.constant_initializer(0.1))

        # Flatten
        flat = tf.contrib.layers.flatten(conv3)
        
        # First Layer Fully Connected
        local1 = tf.layers.dense(flat, 128,
                                activation= tf.nn.leaky_relu,
                                name='fc_1',
                                kernel_initializer= tf.truncated_normal_initializer(),
                                bias_initializer= tf.constant_initializer(0.1))

        # First Dropout
        local1 = tf.layers.dropout(local1, rate= (1-keep_prob), name='dropout_1')

        # Second Layer Fully Connected
        local2 = tf.layers.dense(local1, 256,
                                activation= tf.nn.leaky_relu,
                                name= 'fc_2', kernel_initializer= tf.truncated_normal_initializer(),
                                bias_initializer= tf.constant_initializer(0.1))

        # Second Dropout
        local2 = tf.layers.dropout(local2, rate= (1-keep_prob), name= 'dropout_2')

        # Third Fully Connected Layer
        local3 = tf.layers.dense(local2, 512,
                                activation= tf.nn.leaky_relu,
                                name= 'fc_3', kernel_initializer= tf.truncated_normal_initializer(),
                                bias_initializer= tf.constant_initializer(0.1))

        # Third Dropout
        local3 = tf.layers.dropout(local3, rate= (1-keep_prob), name= 'dropout_3')

        # Readout Layer, fully connected
        out = tf.layers.dense(local3, n_classes, name= 'Output')

        return out



# Method to train Neural Network Model
def train_CNN(X, epochs, lr):

        pred = CNN_model(X)
        pred_classes = tf.argmax(pred, axis=1)
        labels = tf.argmax(y, axis=1)

        # give weight
        '''
        ratio = 1/10
        class_weight = tf.constant([ratio, 1-ratio])
        weighted_pred = tf.multiply(pred, class_weight)
        '''

        # Error function and Optimizer
        with tf.name_scope('Error'):
                error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= pred, labels= y))
        tf.summary.scalar('cross_entropy', error)

        with tf.name_scope('Optimizer'):        
                optimizer = tf.train.AdamOptimizer(lr).minimize(error)

        # Tensorflow Session
        with tf.Session() as sess:
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter('./Visualization', sess.graph)
                sess.run(tf.global_variables_initializer())

                for epoch in range(epochs):
                        epoch_loss = 0
                        start = 0
                        # Training step
                        while start < len(y_train):
                                end = start + batch_size
                                
                                #X_batch = X_train[start:end]
                                #y_batch = y_train[start:end]
                                X_batch, y_batch = next_batch_balanced(batch_size, X_train, y_train)
                                start += batch_size
                                
                                # Running training
                                _, cost = sess.run([optimizer, error], feed_dict= {X: X_batch, y: y_batch, keep_prob: 0.6})
                                epoch_loss += cost

                        s = sess.run(merged, feed_dict={X: X_val, y: y_val, keep_prob: 1})
                        writer.add_summary(s, epoch+1)
                        print('Training:\nEpoch', epoch+1, 'completed out of', n_epochs, 'loss:', epoch_loss)
                        
                        y_true= sess.run(labels,feed_dict={X: X_val, y: y_val, keep_prob: 1}) 
                        y_pred= sess.run(pred_classes, feed_dict={X: X_val, y: y_val, keep_prob: 1})
                        print('Validation')
                        print("Balanced Accuracy: {:.4f}".format(balanced_accuracy_score(y_true, y_pred)))
                        print("Precision (positive): {:.4f}".format(precision_score(y_true, y_pred, pos_label=1)))
                        print("Precision (negative): {:.4f}".format(precision_score(y_true, y_pred, pos_label=0)))
                        print("Recall (positive): {:.4f}".format(recall_score(y_true, y_pred, pos_label=1)))
                        print("Recall (negative): {:.4f}".format(recall_score(y_true, y_pred, pos_label=0)))
                        
                # Testing
                y_true = sess.run(labels,feed_dict={X: X_test, y: y_test, keep_prob: 1}) 
                y_pred = sess.run(pred_classes, feed_dict={X: X_test, y: y_test, keep_prob: 1})
                
                print('Testing')
                print("Balanced Accuracy: {:.4f}".format(balanced_accuracy_score(y_true, y_pred)))
                print("Precision (positive): {:.4f}".format(precision_score(y_true, y_pred, pos_label=1)))
                print("Precision (negative): {:.4f}".format(precision_score(y_true, y_pred, pos_label=0)))
                print("Recall (positive): {:.4f}".format(recall_score(y_true, y_pred, pos_label=1)))
                print("Recall (negative): {:.4f}".format(recall_score(y_true, y_pred, pos_label=0)))
                #file.write(testing_string)

                saver = tf.train.Saver()
                saver.save(sess, './models/dnn', global_step = 1)
                writer.close()

# Train Neural Network
train_CNN(X, n_epochs, lr)

