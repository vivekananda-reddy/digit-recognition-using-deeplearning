from data_processing import create_train_test_set

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import math
import pickle

def random_mini_batches(X, Y, mini_batch_size, seed):

    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_x,n_y):
    X=tf.placeholder(tf.float32,[n_x,None],name="X")
    Y=tf.placeholder(tf.float32,[n_y,None],name="Y")

    return X,Y

def initialize_paramters(layers_dim):

    L=len(layers_dim)
    parameters={}

    for i in range(1,L):
        parameters['W'+str(i)]=tf.get_variable("W"+str(i), [layers_dim[i],layers_dim[i-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters['b'+str(i)] = tf.get_variable("b"+str(i), [layers_dim[i], 1], initializer=tf.zeros_initializer())

    return parameters

def forward_propagation(X,parameters):
    L=len(parameters)//2
    A=X
    for i in range(1,L+1):
        A_prev=A
        Z = tf.add(tf.matmul(parameters["W"+str(i)], A_prev), parameters["b"+str(i)])
        A=tf.nn.relu(Z)

    return Z,A

def compute_cost(Z, Y):

    logits= tf.transpose(Z)
    labels=tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,  labels = labels))
    return cost


def final_model(X_data,Y_data,X_test,Y_test,layers_dim,minibatch_size,learning_rate,num_iterations):
    # tf.set_random_seed(1)
    ops.reset_default_graph()
    print(X_data.shape)
    (n_x,m)=X_data.shape
    n_y=Y_data.shape[0]
    seed = 3

    X,Y=create_placeholders(n_x,n_y)

    parameters= initialize_paramters(layers_dim)

    Z,Yhat= forward_propagation(X,parameters)

    cost=compute_cost(Z,Y)

    optimizer =  tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init= tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_iterations):


            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_data, Y_data, minibatch_size,seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            float_formatter = "{:.6f}".format
            np.set_printoptions(formatter={'float_kind': float_formatter})
            if  epoch % 10 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))


        parameters=sess.run(parameters)


        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_data, Y: Y_data}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))


        return parameters

x_train,y_train,x_test,y_test= create_train_test_set()

features=784
layer_dims=[features,512,256,128,10]
parameters=final_model(x_train,y_train,x_test,y_test,layer_dims,64,0.001,100)


with open("mySavedDict.txt", "wb") as myFile:
    pickle.dump(parameters, myFile)