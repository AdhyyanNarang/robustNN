import logging
import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
sys.path.append('../')
import feedforward_robust as ffr
import ipdb
from util import *

"""
TODO:
    Repeat CCA experiments
    TSNE - float plots

    NOW:
"""

def adv_train(config, logger):
    eps_train = config['eps_train']
    eps_test = config['eps_test']
    tensorboard_dir = config['tensorboard_dir']
    weights_dir = config['weights_dir']
    load_weights =  config['load_weights']
    load_counter = config['load_counter']
    sigma = config['sigma']
    epochs, reg, lr, batch_size = config['epochs'] , config['reg'] , config['lr'] , config['batch_size']
    pgd_eta, pgd_num_iter = config['pgd_eta'] , config['pgd_num_iter']


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
    tf.reset_default_graph()

    sess = tf.Session()
    hidden_sizes = [32,32,32,32,32,32,32]
    dataset = ((x_train_flat, y_train), (x_test_flat, y_test))

    scope_name = "model_robust"
    with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE) as scope:
        logdir = tensorboard_dir + str(counter) + "/robust"

        #Create, train and test model
        writer = tf.summary.FileWriter(logdir)
        model = ffr.RobustMLP(input_shape, hidden_sizes, num_classes, writer = writer, scope = scope_name, logger = logger, sigma = sigma)
        logger.info("Created model successfully. Now going to train")
        sess.run(tf.global_variables_initializer())

        model.pgd_fit(sess, x_train_flat, y_train, eps_train, pgd_eta, pgd_num_iter, lr = lr, training_epochs = epochs, batch_size = batch_size, reg = reg)
        #Save weights
        weights = tf.trainable_variables()
        #weights = model.get_weights()[0] + model.get_weights()[1]
        saver = tf.train.Saver(weights)
        weights_path = saver.save(sess, weights_dir + "model_" + str(counter) + ".ckpt")


        loss_reg, acc_reg = model.evaluate(sess, x_test_flat, y_test)
        loss_fgsm, acc_fgsm = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = False)
        logger.info("----Regular test accuracy and loss ----")
        logger.info((loss_reg, acc_reg))
        logger.info("----FGSM test accuracy and loss ----")
        logger.info((loss_fgsm, acc_fgsm))
        loss_pgd, acc_pgd = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = True, eta=5e-2, num_iter = 100)
        logger.info("----PGD test accuracy and loss ----")
        logger.info((loss_pgd , acc_pgd))

        write_to_results_csv(epochs, reg, lr, "PGD", str(sigma), acc_reg, acc_fgsm, acc_pgd, loss_reg, loss_fgsm, acc_fgsm, logfile, weights_path, logdir, None, None, eps_train, pgd_eta, pgd_num_iter)
        return True

if __name__ == "__main__":

    #Creating the logger
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

    #Hyperparam search config
    etas = [0.001, 0.01]#, 0.1]
    num_iters, eps_trains = [3], [0.1]
    #num_iters = [10, 30, 60]
    #eps_trains = [0.1, 0.05, 0.01]

    for eta in etas:
        for num_iter in num_iters:
            for eps_train in eps_trains:
                #Single Experiment Config
                config = {}
                config['eps_train'] = 0.1
                config['eps_test'] = 0.1
                config['tensorboard_dir'] = "tb/"
                config['weights_dir'] = "weights/"
                config['load_weights'] = False
                config['load_counter'] = 74
                config['sigma'] = tf.nn.relu
                config['epochs'] , config['reg'] , config['lr'] , config['batch_size'] = 2, 0, 3e-3, 32
                config['pgd_eta'] , config['pgd_num_iter'] = 0.1, 3

                success = adv_train(config, logger)
