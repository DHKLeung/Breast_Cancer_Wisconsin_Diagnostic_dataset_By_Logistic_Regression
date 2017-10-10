#   Breast_Cancer_Wisconsin_Diagnostic_dataset_By_Logistic_Regression
#   Programed by Daniel H. Leung 10/10/2017 (DD/MM/YYYY)

import numpy as np

def loadData():
    X_train = np.genfromtxt('X_train.csv', delimiter = ',')
    Y_train = np.genfromtxt('Y_train.csv', delimiter = ',')
    X_test = np.genfromtxt('X_test.csv', delimiter = ',')
    Y_test = np.genfromtxt('Y_test.csv', delimiter = ',')
    return X_train.T, Y_train.reshape(Y_train.shape[0], 1).T, X_test.T, Y_test.reshape(Y_test.shape[0], 1).T

def featureScaling(X_train):
    minval = np.amin(X_train, axis = 1).reshape(X_train.shape[0], 1)
    maxval = np.amax(X_train, axis = 1).reshape(X_train.shape[0], 1)
    X_train = (X_train - minval) / (maxval - minval)
    return X_train, minval, maxval

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def FFBP(X_train, Y_train, X_test, Y_test, learning_rate, epoch):
    m = X_train.shape[1]
    n = X_train.shape[0]
    W = np.zeros((n, 1))
    b = 0.0
    for i in range(epoch):
        A = sigmoid(np.dot(W.T, X_train) + b)
        J = -np.mean(np.multiply(Y_train, np.log(A)) + np.multiply((1 - Y_train), np.log(1 - A))) 
        dJdW = np.dot(X_train, (A - Y_train).T) / m
        dJdb = np.mean(A - Y_train)
        W = W - learning_rate * dJdW
        b = b - learning_rate * dJdb
        A = sigmoid(np.dot(W.T, X_train) + b)
        J = -np.mean(np.multiply(Y_train, np.log(A)) + np.multiply((1 - Y_train), np.log(1 - A))) 
        acc = np.mean(((A > 0.5) == Y_train).astype(float)) * 100.0
        A_t = sigmoid(np.dot(W.T, X_test) + b)
        J_t = -np.mean(np.multiply(Y_test, np.log(A_t)) + np.multiply((1 - Y_test), np.log(1 - A_t))) 
        acc_t = np.mean(((A_t > 0.5) == Y_test).astype(float)) * 100.0 
        print('Iter: ', i + 1, 'Train Cost: ', J, 'Test Cost: ', J_t, 'Train Accuracy: ', acc, '%', 'Test Accuracy: ', acc_t, '%')

    return W, b

learning_rate = 0.3
epoch = 3600
X_train, Y_train, X_test, Y_test = loadData()
X_train, minval, maxval = featureScaling(X_train)
X_test = (X_test - minval) / (maxval - minval)
W, b = FFBP(X_train, Y_train, X_test, Y_test, learning_rate, epoch)