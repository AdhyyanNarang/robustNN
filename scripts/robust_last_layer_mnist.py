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

#Four different methods to train the model.
non_robust_flag = True
adv_train_flag = False
sampled_flag = False
ellipse_flag = False
cca_flag = False

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

    #Create and train the model
    sess = tf.Session()
    hidden_sizes = [32,32,32,32,32,32,32]
    dataset = ((x_train_flat, y_train), (x_test_flat, y_test))


    activation_non = None
    activation_rob = None
    activation_non_two = None
    activation_rob_two = None
    x_adv_cca = None

    if(non_robust_flag):
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


            #Distances and norms
            norms = get_norms(model.get_weights())
            norms_np = sess.run(norms)
            logger.info(norms_np)

            writer.add_graph(sess.graph)
            overall, overall_std, correct, _, incorrect, _ = model.get_distance(sess, eps_test, x_test_flat, y_test)
            logger.info(overall)
            logger.info(overall_std)

            write_to_results_csv(epochs, reg, lr, False, "sigmoid", acc_reg, acc_adv, loss_reg, loss_adv, logfile, logdir, tuple(overall), tuple(norms_np))
            """
            x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
            dist = model.dist_calculator(sess, x_test_flat[0], x_test_flat_adv[0], order = float("inf"))
            logger.info(dist)
            """

            """
            #TSNE visualization of final layer.
            x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
            metadata_path = os.path.join(logdir, 'metadata.tsv')
            write_metadata(metadata_path, y_test_ogi[0:1000])
            sprite_path = os.path.join(logdir, 'sprite_images.png')
            write_sprite_image(sprite_path, x_test_ogi[0:1000])
            model.visualize_activation_tsne(sess, x_test_flat_adv[0:1000], 'metadata.tsv', 'sprite_images.png', logdir)

            logger.info("-----After slashing-------")
            weights_new = model.slash_weights(sess)
            logger.info(model.evaluate(sess, x_test_flat, y_test))
            logger.info(model.adv_evaluate(sess, x_test_flat, y_test, eps_test))

            #Distances and norms
            logger.info(model.get_weight_norms(sess))
            writer.add_graph(sess.graph)
            overall, correct, incorrect = model.get_distance(sess, eps_test, x_test_flat, y_test)
            logger.info(overall)
            """
            #Activations for CCA
            if(cca_flag):
                x_adv_cca = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
                activation_non = model.get_activation(sess, x_adv_cca)

    if(adv_train_flag):
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

