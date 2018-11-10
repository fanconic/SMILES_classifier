
'''
Neural Network Project for CSCI3230, Fundamentals of Artificial Intelligence
Classifier that predicts if a SMILES represenantation of a molecule is toxious or not

Author: fanconic
'''
# Depedencies
import tensorflow as tf
import numpy as np
import csv
from tensorflow.contrib import rnn

# sklearn dependencies
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

'''--------------------------------------Data Preprocessing-------------------------------'''
path_train_names = './data/NR-ER-train/names_onehots.npy'
path_train_labels = './data/NR-ER-train/names_labels.csv'
path_test_names = './data/NR-ER-test/names_onehots.npy'
path_test_labels = './data/NR-ER-test/names_labels.csv'
path_train_smiles = './data/NR-ER-train/names_smiles.csv'
path_test_smiles = './data/NR-ER-test/names_smiles.csv'


# Write Lables from csv to onehot list
def construct_labels(path_to_file):
        labels = []
        weights = []
        with open(path_to_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter= ',')
                for row in csv_reader:
                        if int(row[1]) == 0:
                                # Not Toxic
                                labels.append(0)
                                weights.append(1)
                        elif int(row[1]) == 1:
                                # Toxic
                                labels.append(1)
                                weights.append(10)
        return labels, weights


# Write OneHots to list
def construct_onehots(path_to_file):
        onehots = []
        df = np.load(path_to_file).tolist()
        onehots = df.get('onehots')
        return np.asarray(onehots)

# Construct Nmaes
def construct_smiles(path_to_file):
        smiles = []
        with open(path_to_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter= ',')
                for row in csv_reader:
                        smiles.append(row[1])

        return smiles

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
        X_temp = np.empty([len(X), 72*398])
        for i in range(X.shape[0]):
                X_temp[i] = np.array(X[i]).flatten(order= 'F')
        return X_temp

y_train, w_train = construct_labels(path_train_labels)
y_test, _ = construct_labels(path_test_labels)

X_train = construct_onehots(path_train_names)
X_test = construct_onehots(path_test_names)

X_train = np.append(X_train, X_test, axis= 0)
y_train = np.append(y_train, y_test, axis= 0)

X_test, y_test, X_train, y_train = create_validation(265, X_train, y_train)

X_train = flatten_X(X_train)
X_test = flatten_X(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Balanced Accuracy: {:.4f}".format(balanced_accuracy_score(y_test, y_pred)))
print("Precision (positive): {:.4f}".format(precision_score(y_test, y_pred, pos_label=1)))
print("Precision (negative): {:.4f}".format(precision_score(y_test, y_pred, pos_label=0)))
print("Recall (positive): {:.4f}".format(recall_score(y_test, y_pred, pos_label=1)))
print("Recall (negative): {:.4f}".format(recall_score(y_test, y_pred, pos_label=0)))
