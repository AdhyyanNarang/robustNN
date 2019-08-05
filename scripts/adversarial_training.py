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

    scope_name_rob = "model_robust"
    with tf.variable_scope(scope_name_rob) as scope:
        logdir = tensorboard_dir + str(counter) + "/robust"
        writer_robust = tf.summary.FileWriter(logdir)
        logger.info("Adversarial Training")
        robust_model = ffr.RobustMLP(sess, input_shape, hidden_sizes, num_classes, dataset, writer = writer_robust, scope = scope_name_rob, logger = logger)
        robust_model.adv_fit(sess, x_train_flat, y_train, eps_train, lr = 3e-4, training_epochs = 20)

        epochs, reg, lr = 1, 0.0, 3e-3
        robust_model.adv_fit(sess, x_train_flat,y_train, training_epochs = epochs, lr = lr)
        loss_reg, acc_reg = robust_model.evaluate(sess, x_test_flat, y_test)
        loss_adv, acc_adv = robust_model.adv_evaluate(sess, x_test_flat, y_test, eps_test)
        norms_np = robust_model.get_weight_norms(sess)

        logger.info((loss_reg, acc_reg))
        logger.info((loss_adv, acc_adv))
        logger.info(norms_np)
        overall, _, correct, _, incorrect, _ = robust_model.get_distance(sess, eps_test, x_test_flat, y_test)
        logger.info(overall)
        writer_robust.add_graph(sess.graph)

        #TODO: Test this untested function call.
        write_to_results_csv(epochs, reg, lr, True, "sigmoid", acc_reg, acc_adv, loss_reg, loss_adv, logfile, logdir, tuple(overall), tuple(norms_np))

        ipdb.set_trace()
        #TSNE visualization of final layer.
        x_test_flat_adv = robust_model.fgsm_np(sess, x_test_flat, y_test, eps_test)
        metadata_path = os.path.join(logdir, 'metadata.tsv')
        write_metadata(metadata_path, y_test_ogi[0:1000])
        sprite_path = os.path.join(logdir, 'sprite_images.png')
        write_sprite_image(sprite_path, x_test_ogi[0:1000])
        robust_model.visualize_activation_tsne(sess, x_test_flat[0:1000], 'metadata.tsv', 'sprite_images.png', logdir)
