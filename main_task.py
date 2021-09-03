from __future__ import print_function

import sys
import pyspark
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

from pyspark import SparkContext

######### FUNCTIONS #########

# --------- ACTIVATION FUNCTIONS ---------- #
# General function to apply any activation function
def activation(x, f):
    return f(x)

# Sigmoid Activation function
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

# Sigmoid prime function (used for backward prop)
def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Hyperbolic Tangent Activation function
def tanh(x):
    return np.tanh(x);

# Hyperbolic Tangent prime function (used for backward prop)
def tanh_prime(x):
    return 1-np.tanh(x)**2

# --------- FORWARD PROPAGATIONS FUNCTIONS ---------- #

# Compute the layer propagation before activation
def preforward(x, w, b):
    return np.dot(x, w) + b

# Compute the layer propagation after activation
# This is also equivalent to a predict function once model is trained
def predict(x, W1, B1, W2, B2):
    return sigmoid(preforward(sigmoid(preforward(x , W1, B1)), W2, B2))

# --------- BACKWARD PROPAGATIONS FUNCTIONS ---------- #

# Compute the derivative of the error regarding B2
def derivativeB2(y_pred, y_true, y_h, f_prime):
    return (y_pred - y_true) * f_prime(y_h)

# Compute the derivative of the error regarding W2
def derivativeW2(h, dB2):
    return np.dot(h.T, dB2)

# Compute the derivative of the error regarding B1
def derivativeB1(h_h, dB2, W2, f_prime):
    return np.dot(dB2, W2.T) * f_prime(h_h)

# Compute the derivative of the error regarding W1
def derivativeW1(x, dB1):
    return np.dot(x.T, dB1)

# --------- EVALUATION FUNCTIONS ---------- #

def get_metrics(pred, true):
    cm = multilabel_confusion_matrix(true, pred)
    return (cm)

# Cost function
def sse(y_pred, y_true):
    return 0.5 * np.sum(np.power(y_pred - y_true, 2))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: main_task.py <TRAIN_IMAGE_FILE> <TRAIN_LABEL_FILE> <TEST_IMAGE_FILE> <TEST_LABEL_FILE> <OUTPUT_FILE> ", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="Project")
    sc.setLogLevel("ERROR")

    print('Loading Data...')

    txt_train_images = sc.textFile(sys.argv[1], 1)
    x_train = txt_train_images.map(lambda x : np.fromstring(x, dtype=float, sep=' ')\
                              .reshape(1, 784))\
                              .zipWithIndex()\
                              .map(lambda x: (str(x[1]), x[0]))

    txt_train_labels = sc.textFile(sys.argv[2], 1)
    y_train = txt_train_labels.map(lambda x : np.fromstring(x, dtype=float, sep=' ')\
                              .reshape(1, 10))\
                              .zipWithIndex()\
                              .map(lambda x: (str(x[1]), x[0]))

    txt_test_images = sc.textFile(sys.argv[3], 1)
    x_test = txt_test_images.map(lambda x : np.fromstring(x, dtype=float, sep=' ')\
                            .reshape(1, 784))\
                            .zipWithIndex()\
                            .map(lambda x: (str(x[1]), x[0]))

    txt_test_labels = sc.textFile(sys.argv[4], 1)
    y_test = txt_test_labels.map(lambda x : np.fromstring(x, dtype=float, sep=' ')\
                            .reshape(1, 10))\
                            .zipWithIndex()\
                            .map(lambda x: (str(x[1]), x[0]))

    train_rdd = x_train.join(y_train).map(lambda x: x[1])
    test_rdd = x_test.join(y_test).map(lambda x: x[1])
    train_rdd.cache()

    print('Data Loaded!')


    # Hyperparameters
    num_iteration = 100
    learningRate = 0.1

    input_layer = 784 # number of neurones in the input layer (equal to image size)
    hidden_layer = 64 # number of neurones in the hidden layer (Custom)
    output_layer = 10 # number of neurones in the output layer (equal to the number of possible labels)


    # Paramater Initialization
    W1 = np.random.rand(input_layer, hidden_layer) - 0.5  # Shape (784, 64)
    W2 = np.random.rand(hidden_layer, output_layer) - 0.5 # Shape (64, 10)
    B1 = np.random.rand(1, hidden_layer) - 0.5 # Shape (1, 64)
    B2 = np.random.rand(1, output_layer) - 0.5 # Shape (1, 10)

    # History over epochs
    cost_history = []
    acc_history = []

    # Epoch Loop (mini batch implementation)
    print("Start Training Loop:")

    for i in range(num_iteration):

        # Compute gradients, cost and accuracy over mini batch

        ################## Notations ######################
        # x -> Input Image flatten of shape (1, 784)
        # y* -> One hot label of shape (1, 10)
        # h^ -> Forward prop from Input layer to hidden layer before activation (1, 64) using W1, B1 parm
        # h -> Forward prop from Input layer to hidden layer after tanh activation (1, 64)
        # y^ -> Forward prop from hidden layer to output layer before activation (1, 10) using W2, B2 parm
        # y -> Forward prop from hidden layer to output layer after sigmoid activation (1, 10)
        # E -> Error between y and y* using SSE
        # Acc -> 1 is right prediction 0 otherwise
        # DE/D? -> Partial derivative of the Error regarding parmaters (B2, W2, B1, W1)


        ################# Forward Prop ######################
        # map batch ([x], [y*]) to ([x], [h^],[y*])
        # map batch ([x], [h^],[y*]) to ([x], [h^], [h], [y*])
        # map batch ([x], [h^], [h], [y*]) to ([x], [h^], [h], [y^], [y*])
        # map batch ([x], [h^], [h], [y^], [y*]) to ([x], [h^], [h], [y^], [y], [y*])
        ################# Backward Prop #####################
        # map batch ([x], [h^], [h], [y^], [y], [y*]) to ([x], [h^], [h], [E], [DE/DB2], [Acc])
        # map batch ([x], [h^], [h], [E], [DE/DB2], [Acc]) to ([x], [h^], [E], [DE/DB2], [DE/DW2], [Acc])
        # map batch ([x], [h^], [E], [DE/DB2], [DE/DW2], [Acc]) to ([x], [E], [DE/DB2], [DE/DW2], [DE/DB1], [Acc])
        # map batch ([x], [E], [DE/DB2], [DE/DW2], [DE/DB1], [Acc]) to ([E], [DE/DB2], [DE/DW2], [DE/DB1], [DE/DW1],[Acc])
        ############### Reduce over the mini batch #########


        gradientCostAcc = train_rdd\
                            .sample(False,0.7)\
                            .map(lambda x: (x[0], preforward(x[0], W1, B1), x[1]))\
                            .map(lambda x: (x[0], x[1], activation(x[1], tanh), x[2]))\
                            .map(lambda x: (x[0], x[1], x[2], preforward(x[2], W2, B2), x[3]))\
                            .map(lambda x: (x[0], x[1], x[2], x[3], activation(x[3], sigmoid), x[4]))\
                            .map(lambda x: (x[0], x[1], x[2], sse(x[4], x[5]), derivativeB2(x[4], x[5], x[3], sigmoid_prime), int(np.argmax(x[4]) == np.argmax(x[5]))))\
                            .map(lambda x: (x[0], x[1], x[3], x[4],  derivativeW2(x[2], x[4]) ,x[5]))\
                            .map(lambda x: (x[0], x[2], x[3], x[4],  derivativeB1(x[1],  x[3], W2, tanh_prime) ,x[5]))\
                            .map(lambda x: (x[1], x[2], x[3], x[4], derivativeW1(x[0], x[4]) ,x[5], 1)) \
                            .reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4], x[5] + y[5], x[6] + y[6]))

        # Cost and Accuarcy of the mini batch
        n = gradientCostAcc[-1] # number of images in the mini batch
        cost = gradientCostAcc[0]/n # Cost over the mini batch
        acc = gradientCostAcc[5]/n # Accuarcy over the mini batch

         # Add to history
        cost_history.append(cost)
        acc_history.append(acc)

        # Bold Driver technique to dynamically change the learning rate
        if len(cost_history) > 1:
            if cost_history[i] < cost_history[i-1]:
                learningRate *= 1.05 #Better than last time
            else:
                learningRate *= 0.5 #Worse than last time

        # Extract gradiends
        DB2 = gradientCostAcc[1]/n
        DW2 = gradientCostAcc[2]/n
        DB1 = gradientCostAcc[3]/n
        DW1 = gradientCostAcc[4]/n

        # Update parameter with new learning rate and gradients using Gradient Descent
        B2 -= learningRate * DB2
        W2 -= learningRate * DW2
        B1 -= learningRate * DB1
        W1 -= learningRate * DW1

        # Display performances
        print(f"   Epoch {i+1}/{num_iteration} | Cost: {cost_history[i]} | Acc: {acc_history[i]*100} | Batchsize:{n}")

    print("Training end..")

    # Save cost history
    dataToASingleFile = sc.parallelize(cost_history).coalesce(1)
    dataToASingleFile.saveAsTextFile(sys.argv[-1] + f"/cost_history")

    # Save accuracy history
    dataToASingleFile = sc.parallelize(acc_history).coalesce(1)
    dataToASingleFile.saveAsTextFile(sys.argv[-1] + f"/accuracy_history")


    # Use the trained model over the Testset and get Confusion matrix per class
    metrics = test_rdd.map(lambda x: get_metrics(np.round(predict(x[0], W1, B1, W2, B2)), x[1]))\
                      .reduce(lambda x, y: x + y)

    # For each class give TP, FP, FN, TN and precision, and recall, and F1 score
    save_metrics = []
    for label, label_metrics in enumerate(metrics):

        print(f"\n---- Digit {label} ------\n")
        tn, fp, fn, tp = label_metrics.ravel()
        print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)

        precision = tp / (tp + fp + 0.000001)
        print(f"\nPrecision : {precision}")

        recall = tp / (tp + fn + 0.000001)
        print(f"Recall: {recall}")

        F1 = 2 * (precision * recall) / (precision + recall + 0.000001)
        print(f"F1 score: {F1}")

        save_metrics.append((precision, recall, F1))

    # Save metrics per class
    dataToASingleFile = sc.parallelize(save_metrics).coalesce(1)
    dataToASingleFile.saveAsTextFile(sys.argv[-1] + f"/metrics")

    # Display some Images for checking
    tests = []
    for image_test in test_rdd.map(lambda x: (x[0], predict(x[0], W1, B1, W2, B2), np.argmax(x[1]))).takeSample(False, 15):

        pred = np.argmax(image_test[1])
        tests.append((pred, image_test[2]))
        print(f'pred: {pred}, prob: {round(image_test[1][0][pred], 2)} true: {image_test[2]}')

    # Save test
    dataToASingleFile = sc.parallelize(tests).coalesce(1)
    dataToASingleFile.saveAsTextFile(sys.argv[-1] + "/tests")

    sc.stop()
