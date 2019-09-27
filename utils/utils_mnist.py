import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import ipdb
import matplotlib.pyplot as plt
import pandas as pd

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

def flatten_mnist(x):
    n, img_rows, img_cols = x.shape
    D = img_rows * img_cols
    x_flattened = x.reshape(n, D)
    return x_flattened, (D, )
