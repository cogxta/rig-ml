
from sklearn.datasets import make_moons
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.metrics import classification_report
import sklearn.model_selection as ms
import numpy as np
from math import exp
from random import random
from matplotlib import pyplot as plt
np.random.seed(111)
from tensorflow.keras import models as km
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import joblib, os
from sklearn.neural_network import MLPClassifier
cpath = os.path.abspath(os.getcwd())
models_path = cpath+"/models"
X, y = make_moons(300, noise=0.20)
#X = iris.data, y = iris.target

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def softmax(y):
    exp_scores = np.exp(y)
    y_hat = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return y_hat

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    nn_len = len(network)
    for i in range(nn_len):
        layer = network[i]
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
#            print("activated--->",activation)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    y_hat = np.array(inputs)
    return y_hat, network

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
    
    return network


# Update network weights with error
def update_weights(network, x, l_rate):
    for i in range(len(network)):
        inputs = x
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
    return network


# Train a network for a fixed number of epochs
def train_network(network, x, y, l_rate, n_epoch, n_outputs):
    error_movement=[]
    for epoch in range(n_epoch):
        sum_error = 0
        for i in range(len(x)):
            y_hat, network = forward_propagate(network, x[i])
            expected = [0 for i in range(n_outputs)]
            expected[y[i]] = 1
#            print("y vs y_hat---->",expected, "vs", y_hat)
            sum_error += (1-y_hat[y[i]])**2
#            sum_error += (expected-y_hat)**2
            network = backward_propagate_error(network, expected)
            network = update_weights(network, x[i], l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        error_movement.append(sum_error)
    return network, error_movement


# Make a prediction with a network
def predict(network, data):
    preds=[]
    for row in data:
        y_hat, network = forward_propagate(network, row)
        binary_y_hat = np.argmax(y_hat); #print(binary_y_hat)
        preds.append(binary_y_hat)
#    print(preds)
    return preds


def print_nn(network):
    print("Network Layers:")
    l=0
    for layer in network:
        l=l+1
        if l==1: print("Hidden Layers:")
        if l==2: print("Output Layer:")
        for neuron in layer:
            print(neuron)
        print("")


def view_cr(y_train,y_test,train_pred,test_pred):
    print("Training:")
    print(classification_report(y_train, train_pred))
#    print("roc_auc:", roc_auc_score(y_train, train_pred))
    print("")
#    print("% of Unknown classe @ threshold = "+str(pred_th), " is ", round(len(test_pred[test_pred==-1])/len(test_pred),3))
    print("Testing:")
    print(classification_report(y_test, test_pred))
        

def ann_train_test(X,y,lib,model_name):
    model_file = models_path+"/"+model_name+"_ann_"+lib+".pkl"
    x_train,x_test,y_train,y_test=ms.train_test_split(X,y,test_size=0.2, random_state=111)
    n_inputs = len(x_train[0])
    n_outputs = len(set(list(y))); print("classes--->",n_outputs)
    n_epoch = 500
    
    if lib == 'custom':
        n_hidden = 3
        l_rate = 0.5
        network = initialize_network(n_inputs, n_hidden, n_outputs)
        network, em = train_network(network, x_train, y_train, l_rate, n_epoch, n_outputs)
        print_nn(network)
        plt.plot(em); plt.show()
        joblib.dump(network, model_file)
        network = joblib.load(model_file)
        train_pred = predict(network, x_train)
        test_pred = predict(network, x_test)
        view_cr(y_train,y_test,train_pred,test_pred)



    if lib == 'mlp':
        model = MLPClassifier()
        print(model)
        model.fit(x_train,y_train)
        joblib.dump(model,model_file)
        model = joblib.load(model_file)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        view_cr(y_train, y_test, train_pred, test_pred)



    if lib == 'keras':
#        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
#        y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
        if n_outputs == 2:
            loss_fuction = 'binary_crossentropy'
            last_activation = 'sigmoid'
        elif n_outputs > 2:
            loss_fuction = 'categorical_crossentropy'
            last_activation = 'softmax'
        model = Sequential()
        model.add(Dense(12, input_dim=n_inputs, activation='relu'))
        model.add(Dense(8, activation='sigmoid'))
        model.add(Dense(1, activation=last_activation))
        opt = Adam(learning_rate=0.01)
        model.compile(loss=loss_fuction, optimizer=opt, metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=n_epoch, batch_size=10, shuffle=False)
        model.save(model_file)
        model = km.load_model(model_file)


        if n_outputs == 2:
            train_pred = (model.predict(x_train) > 0.5).astype("int32")
            test_pred = (model.predict(x_test) > 0.5).astype("int32")
        elif n_outputs > 2:
            train_pred = np.argmax(model.predict(x_train), axis=-1)
            test_pred = np.argmax(model.predict(x_test), axis=-1)


        print(y_train[:5]); print(train_pred[:5])
        view_cr(y_train,y_test,train_pred,test_pred)

models_path = "../models"
ann_train_test(X,y,'keras','makemoon')
#MLP = 87%, custom ANN = 90%, keras DNN = 98%