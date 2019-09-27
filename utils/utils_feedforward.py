import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import ipdb
import matplotlib.pyplot as plt
import pandas as pd

"""
Helper functions for feedforward_robust
"""
def get_max_abs(mat):
    return np.max(np.abs(mat))

def get_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_test_ogi = y_test
    x_test_ogi = x_test
    x_train = x_train/255
    x_test = x_test/255
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    x_train_flat, input_shape = flatten_mnist(x_train)
    x_test_flat, _ = flatten_mnist(x_test)
    hidden_sizes = [32,32,32,32,32,32,32]
    dataset = ((x_train_flat, y_train), (x_test_flat, y_test))
    return dataset, input_shape, num_classes

def fully_connected_layer(x, output_dim, scope_name, weight_initializer, bias_initializer, sigma):
    #Reuse = True because everytime we wish to call model(x), we want it to
    #use the most recently learned weights.
    with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE) as scope:
        weight_shape = (x.shape[1], output_dim)
        w = tf.get_variable('weights', shape = weight_shape, initializer = weight_initializer)
        b = tf.get_variable('biases', output_dim, initializer=bias_initializer)
        z = tf.add(tf.matmul(x, w), b)
        a = sigma(z)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", a)
        return a

def model(x, hidden_sizes, num_classes, act_fxn = tf.nn.relu):
    """
    Input: Placeholder x, sizes of hidden layers, num_classes
    Output: Activations of fully connected neural network
    """
    act = x
    activations = []

    initial = tf.contrib.layers.xavier_initializer(dtype = tf.float32)
    #initial = tf.initializers.truncated_normal(mean = -0.001, stddev = 0.9)
    bias_initial = tf.initializers.zeros

    #Compute the prediction placeholder
    for i in range(len(hidden_sizes)):
        scope = 'fc_' + str(i)
        act = fully_connected_layer(act, hidden_sizes[i], scope, initial, bias_initial, act_fxn)
        activations.append(act)

    scope = 'fc_' + str(len(hidden_sizes))
    predictions = fully_connected_layer(act, num_classes, scope, initial, bias_initial, tf.identity)

    return activations, predictions

def op_norm(matrix):
    column_norms = tf.norm(matrix, ord = 1, axis = 0)
    op_inf_inf = tf.reduce_max(column_norms)
    return op_inf_inf

def trace_norm_approx(A):
    """
    Whereas the actual trace norm computes the sums of the singular values of the matrix,
    this norm computes the sums of squares of the singular values
    """
    prod = tf.matmul(A, tf.linalg.transpose(A))
    return tf.linalg.trace(prod)

def get_spectral_norm_elementwise(weights_np):
    norm_lst = []
    for mat in weights_np:
        spectral_norm = np.linalg.norm(mat, ord = 2)
        norm_lst.append(spectral_norm)
    return norm_lst

def get_two_one_norm(M):
    two_norm_lst = []
    for i in range(M.shape[1]):
        column = M[:, i]
        two_norm = np.linalg.norm(column)
        two_norm_lst.append(two_norm)
    two_norm_vec = np.array(two_norm_lst)
    return np.linalg.norm(two_norm_vec, ord = 1)

def get_two_one_norm_elementwise(weights_np):
    norm_lst = []
    for mat in weights_np:
        two_one_norm = get_two_one_norm(mat)
        norm_lst.append(two_one_norm)
    return norm_lst

def get_norms(weights_list):
    norms = []
    for weights in weights_list:
        norms.append(op_norm(weights))
    return norms

def regularize_op_norm(weights_list):
    penalty = 0
    for weights in weights_list:
        penalty += op_norm(weights)
    return penalty

def regularize_trace_norm(weights_list):
    penalty = 0
    for weights in weights_list:
        penalty += trace_norm_approx(weights)
    return penalty

def regularize_trace_norm_first(weights_list):
    penalty = 0
    penalty += trace_norm_approx(weights_list[0])
    return penalty

def regularize_l1_norm(weights_list):
    penalty = 0
    for weights in weights_list:
        penalty += tf.norm(weights, ord = 1)
    return penalty

def regularize_lipschitz_norm(weights_list):
    penalty = 1
    for weights in weights_list:
        penalty *= tf.norm(weights)
    return penalty

def write_to_results_csv(epochs, op_reg, trace_first_reg, trace_all_reg, reg_l1, lr, adv_train_flag, activation, acc_reg, acc_fgsm, acc_pgd,  loss_reg, loss_fgsm, loss_pgd, logdir, weights_path, tb_dir, dist, norms, pgd_train_eps = None, pgd_train_eta = None, pgd_train_num_iter = None):
    df = pd.read_excel("results.xlsx")
    index = len(df)
    df.loc[index] = [index, epochs, op_reg, trace_first_reg, trace_all_reg, reg_l1, lr, adv_train_flag, pgd_train_eps, pgd_train_eta, pgd_train_num_iter, activation, acc_reg, acc_fgsm, acc_pgd, loss_reg, loss_fgsm, loss_pgd, logdir, weights_path, tb_dir, dist, norms]
    df.to_excel("results.xlsx", index = False)
    return True

def flatten_mnist(x):
    n, img_rows, img_cols = x.shape
    D = img_rows * img_cols
    x_flattened = x.reshape(n, D)
    return x_flattened, (D, )
