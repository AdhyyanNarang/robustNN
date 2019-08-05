import logging
import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
sys.path.append('../')
import feedforward_robust as ffr
import ellipsoid
import ipdb
from mnist_corruption import random_perturbation, gaussian_perturbation
import cvxpy as cp
from util import *

"""
TODO:
    Repeat CCA experiments
    TSNE - float plots

    NOW:
"""

#Config
eps_train = 0.1
eps_test = 0.1
tensorboard_dir = "tb/"

#Configuring the logger

#Read the counter
ctr_file = "counter.txt"
f = open(ctr_file, 'r')
counter = f.readline()
f.close()

counter = 1 + int(counter)
f = open(ctr_file,'w') 
f.write('{}'.format(counter))
f.close()
logfile = "logs/results_" + str(counter) + ".log"

logger = logging.getLogger("robustness")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(logfile)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

if __name__ == "__main__":

    #TODO: Replace this whole section with the function from util
    logger.info("Something happened")
    #Setup - Dataset stuff
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

    sess = tf.Session()
    hidden_sizes = [32,32,32,32,32,32,32]
    dataset = ((x_train_flat, y_train), (x_test_flat, y_test))

    scope_name = "model_non_robust"
    with tf.variable_scope(scope_name) as scope:

        logdir = tensorboard_dir + str(counter) + "/non_robust"

        #Create, train and test model
        writer = tf.summary.FileWriter(logdir)
        model = ffr.RobustMLP(sess, input_shape, hidden_sizes, num_classes, dataset, writer = writer, scope = scope_name, logger = logger)
        logger.info("Created model successfully. Now going to train")

        #TODO: Fit the model until convergence before running the distance experiments again
        epochs, reg, lr = 1, 0.0, 3e-3
        model.fit(sess, x_train_flat,y_train, training_epochs = epochs, reg = reg , lr = lr)
        loss_reg, acc_reg = model.evaluate(sess, x_test_flat, y_test)
        loss_adv, acc_adv = model.adv_evaluate(sess, x_test_flat, y_test, eps_test)
        logger.info((loss_reg, acc_reg))
        logger.info((loss_adv, acc_adv))
        writer.add_graph(sess.graph)

        #Distances and norms
        norms = get_norms(model.get_weights())
        norms_np = sess.run(norms)
        logger.info(norms_np)

        overall, overall_std, correct, _, incorrect, _ = model.get_distance(sess, eps_test, x_test_flat, y_test)
        logger.info(overall)
        logger.info(overall_std)

        write_to_results_csv(epochs, reg, lr, False, "sigmoid", acc_reg, acc_adv, loss_reg, loss_adv, logfile, logdir, tuple(overall), tuple(norms_np))
        x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
        dist = model.dist_calculator(sess, x_test_flat[0], x_test_flat_adv[0], order = float("inf"))
        logger.info(dist)

        #TSNE visualization of final layer.
        x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
        metadata_path = os.path.join(logdir, 'metadata.tsv')
        write_metadata(metadata_path, y_test_ogi[0:1000])
        sprite_path = os.path.join(logdir, 'sprite_images.png')
        write_sprite_image(sprite_path, x_test_ogi[0:1000])
        model.visualize_activation_tsne(sess, x_test_flat_adv[0:1000], 'metadata.tsv', 'sprite_images.png', logdir)
