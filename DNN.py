
'''
Neural Network Project for CSCI3230, Fundamentals of Artificial Intelligence
Classifier that predicts if a drug SMILES value is toxious or not

Author: fanconic
'''
import tensorflow as tf
import numpy as np
import csv
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score, precision_score

'''--------------------------------------Data Preprocessing-------------------------------'''
path_train_names = './data/NR-ER-train/names_onehots.npy'
path_train_labels = './data/NR-ER-train/names_labels.csv'
path_test_names = './data/NR-ER-test/names_onehots.npy'
path_test_labels = './data/NR-ER-test/names_labels.csv'

# Write Lables and weights from csv to onehot list
def construct_labels_and_weights(path_to_file):
        labels = []
        weights = []
        with open(path_to_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter= ',')
                for row in csv_reader:
                        if int(row[1]) == 0:
                                # Not Toxic
                                labels.append([1,0])
                                weights.append(1)
                        elif int(row[1]) == 1:
                                # Toxic
                                labels.append([0,1])
                                weights.append(10)

        return np.asarray(labels), np.asarray(weights)

# Write OneHots to list
def construct_features(path_to_file):
        features = []
        new_features = []
        df = np.load(path_to_file).tolist()
        features = df.get('onehots')
        #new_features = np.sum(features, axis=2)
        return np.asarray(features)

# Create Validation Set
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

# Flatten the training data
def flatten_X(X):
        X_temp = np.empty([len(X), n_input_vec])
        for i in range(X.shape[0]):
                X_temp[i] = np.array(X[i]).flatten()
        return X_temp


y_train, class_weights = construct_labels_and_weights(path_train_labels)
y_test, _ = construct_labels_and_weights(path_test_labels)

X_train = construct_features(path_train_names)
X_test = construct_features(path_test_names)

X_train = np.append(X_train, X_test, axis= 0)
y_train = np.append(y_train, y_test, axis= 0)

X_test, y_test, X_train, y_train = create_validation(265, X_train, y_train)
X_val, y_val, X_train, y_train = create_validation(265, X_train, y_train)


# A Molecule is composed of a onehot matrix of 72 x 396 Dimension
n_rows = len(X_train[0])
n_cols = len(X_train[0][0])
n_input_vec = n_rows*n_cols
X_train = flatten_X(X_train)
X_test = flatten_X(X_test)
X_val = flatten_X(X_val)


'''--------------------------------------Neural Network------------------------------------'''
# Neural Network Parameters
n_classes = 2
batch_size = 16
n_hl1 = 256
n_hl2 = 256
n_hl3 = 256
n_hl4 = 256
n_hl5 = 256
n_epochs = 100
lr = 1e-3

# Placeholders
X = tf.placeholder(tf.float32, shape=[None, n_input_vec])
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

# Deep Neural Feed Forward Network Model Method
def DNN_model(input_data):
        ''' ------------Network Architecture-------------'''
                # First hidden Layer, weights and biases are randomly normally distributed
        hl_1 = {'weights': tf.Variable(tf.truncated_normal([n_input_vec,n_hl1])),
               'biases': tf.Variable(tf.truncated_normal([n_hl1]))}

        # Second hidden Layer, weights and biases are randomly normally distributed
        hl_2 = {'weights': tf.Variable(tf.truncated_normal([n_hl1,n_hl2])),
               'biases': tf.Variable(tf.truncated_normal([n_hl2]))}

        # third hidden Layer, weights and biases are randomly normally distributed
        hl_3 = {'weights': tf.Variable(tf.truncated_normal([n_hl2,n_hl3])),
               'biases': tf.Variable(tf.truncated_normal([n_hl3]))}

        # fourth hidden Layer, weights and biases are randomly normally distributed
        hl_4 = {'weights': tf.Variable(tf.truncated_normal([n_hl3,n_hl4])),
               'biases': tf.Variable(tf.truncated_normal([n_hl4]))}

        # fifth hidden Layer, weights and biases are randomly normally distributed
        hl_5 = {'weights': tf.Variable(tf.truncated_normal([n_hl4,n_hl5])),
               'biases': tf.Variable(tf.truncated_normal([n_hl5]))}
        
        # output Layer, weights and biases are randomly normally distributed
        out_l =  {'weights': tf.Variable(tf.truncated_normal([n_hl5,n_classes])),
               'biases': tf.Variable(tf.truncated_normal([n_classes]))}

        ''' ------------Network Calculation-------------'''
         # h_k+1 = g(h_k * w_j + b_i)
        l1 = tf.add(tf.matmul(input_data, hl_1['weights']), hl_1['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hl_2['weights']), hl_2['biases'])
        l2 = tf.nn.relu(l2)
        l2 = tf.nn.dropout(l2, keep_prob)

        l3 = tf.add(tf.matmul(l2, hl_3['weights']), hl_3['biases'])
        l3 = tf.nn.relu(l3)
        l3 = tf.nn.dropout(l3, keep_prob)

        l4 = tf.add(tf.matmul(l3, hl_4['weights']), hl_4['biases'])
        l4 = tf.nn.relu(l4)
        l4 = tf.nn.dropout(l4, keep_prob)

        l5 = tf.add(tf.matmul(l4, hl_5['weights']), hl_5['biases'])
        l5 = tf.nn.relu(l5)
        l5 = tf.nn.dropout(l5, keep_prob)

        # O_m = h_m-1 *(w_j + b_i)
        ol = tf.matmul(l5, out_l['weights']) + out_l['biases']

        return ol

# Method to train Neural Network Model
def train_DNN(X, epochs, lr):
        file = open('./models/trash/log.txt','w') 

        pred = DNN_model(X)
        pred_classes = tf.argmax(pred, axis=1)
        labels = tf.argmax(y, axis=1)
        
        # Create Weights for classes
        ratio = 1/20
        class_weight = tf.constant([ratio, 1-ratio])
        weighted_pred = tf.multiply(pred, class_weight)

        # Error and optimizer
        error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= weighted_pred, labels= y))
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

                                start += batch_size
                                # Running training
                                _, cost = sess.run([optimizer, error], feed_dict= {X: X_batch, y: y_batch, keep_prob: 1})
                                epoch_loss += cost

                        print('Training:\nEpoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)
                        
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
                saver.save(sess, './models/trash/dnn', global_step = 1)

# Train Neural Network
train_DNN(X, n_epochs, lr)

