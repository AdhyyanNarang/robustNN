import os, random, time, sys
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import ipdb
import pandas as pd
import feedforward_robust as ffr
import ipdb
from utils_mnist import *
from utils_feedforward import write_to_results_csv

def regular_training(config_inp):
    #eps_train = config_inp['eps_train']
    eps_test = config_inp['eps_test']
    tensorboard_dir = config_inp['tensorboard_dir']
    weights_dir = config_inp['weights_dir']
    load_counter = config_inp['load_counter']
    sigma = config_inp['sigma']
    scope_name = config_inp['scope_name']
    should_load = config_inp['should_load']
    logger = config_inp['logger']
    counter = config_inp['write_counter']
    epochs = config_inp['epochs']
    lr = config_inp['lr']

    reg_trace_first = config_inp['reg_trace_first']
    reg_op = config_inp['reg_op']
    reg_trace_all = config_inp['reg_trace_all']
    reg_l1 = config_inp['reg_l1']

    #Dataset setup stuff
    dataset, input_shape, num_classes  = get_dataset()
    x_train_flat, y_train = dataset[0]
    x_test_flat, y_test = dataset[1]
    sess = tf.Session()
    hidden_sizes = [32,32,32,32,32,32,32]
    #dataset = ((x_train_flat, y_train), (x_test_flat, y_test))


    with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE) as scope:
        sess = tf.Session()
        logdir = tensorboard_dir + str(counter) + "/non_robust"

        #Create, train and test model
        writer = tf.summary.FileWriter(logdir)
        model = ffr.RobustMLP(input_shape, hidden_sizes, num_classes, writer = writer, scope = scope_name, logger = logger, sigma =sigma)
        logger.info("Created model successfully. Now going to load weights")
        #Restore weights
        weights_path = None
        if should_load:
            weights, biases = model.get_weights()
            weights_exp = weights + biases
            saver = tf.train.Saver(weights_exp)
            weights_path = saver.restore(sess, weights_dir + "model_" + str(load_counter) + ".ckpt")
            logger.info("Restored model from %s"%weights_path)
        else:
            sess.run(tf.global_variables_initializer())
            model.fit(sess, x_train_flat,y_train, training_epochs = epochs, lr = lr, reg_op = reg_op, reg_trace_first = reg_trace_first, reg_trace_all = reg_trace_all, reg_l1 = reg_l1)
            #Save weights
            weights = tf.trainable_variables()
            #weights = model.get_weights()[0] + model.get_weights()[1]
            saver = tf.train.Saver(weights)
            weights_path = saver.save(sess, weights_dir + "model_" + str(counter) + ".ckpt")
            logger.info("Saved model at %s"%weights_path)

        #writer.add_graph(sess.graph)

        loss_reg, acc_reg = model.evaluate(sess, x_test_flat, y_test)
        logger.info("----Regular test accuracy and loss ----")
        logger.info((loss_reg, acc_reg))

        loss_reg, acc_reg = model.evaluate(sess, x_test_flat, y_test)
        logger.info("----Regular test accuracy and loss ----")
        logger.info((loss_reg, acc_reg))

        loss_fgsm, acc_fgsm = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = False)
        logger.info("----FGSM test accuracy and loss ----")
        logger.info((loss_fgsm, acc_fgsm))
        x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)

        write_to_results_csv(epochs, reg_op, reg_trace_first, reg_trace_all, reg_l1, lr, "Regular", str(sigma), acc_reg, acc_fgsm, None, loss_reg, loss_fgsm, None, None, weights_path, None, None, None)


    return model, sess, weights_path

def hybrid_training(config_inp):
    eps_train = config_inp['eps_train']
    eps_test = config_inp['eps_test']
    tensorboard_dir = config_inp['tensorboard_dir']
    weights_dir = config_inp['weights_dir']
    load_counter_non_robust = config_inp['load_counter_non_robust']
    load_counter_robust = config_inp['load_counter_robust']
    sigma = config_inp['sigma']
    scope_name = config_inp['scope_name']
    should_slash = config_inp['should_slash']
    slash_factor = config_inp['slash_factor']

    with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE) as scope:
        sess = tf.Session()
        logdir = tensorboard_dir + str(counter)

        #Create, train and test model
        writer = tf.summary.FileWriter(logdir)
        model = ffr.RobustMLP(input_shape, hidden_sizes, num_classes, writer = writer, scope = scope_name, logger = logger, sigma =sigma)
        feed_dict = {model.x: x_train_flat, model.y: y_train}
        sess.run(tf.global_variables_initializer(), feed_dict = feed_dict)

        logger.info("Created model successfully. Now going to load weights")

        #Restore all weights from non-robust
        weights, biases = model.get_weights()
        weights_exp = weights + biases
        saver = tf.train.Saver(weights_exp)
        weights_path = saver.restore(sess, weights_dir + "model_" + str(load_counter_non_robust) + ".ckpt")

        #Optionally slash weights
        if should_slash:
            model.slash_weights(sess, slash_factor)
            logger.info("Successfully slashed weights")

        logger.info("Successfully restored weights from non rob model.")

        #Restore the first and last layer from the robust model
        weights, biases = model.get_weights()
        selected_vars = [weights[0]] + [weights[-1]] + [biases[0]] + [biases[-1]]
        saver = tf.train.Saver(selected_vars)
        weights_path_rob = saver.restore(sess, weights_dir + "model_" + str(load_counter_robust) + ".ckpt")

        logger.info("Successfully restored weights from rob model.")

        #Run one epoch to try to make the weights more compatible with each other

        model.fit(sess, x_train_flat[0:10000], y_train[0:10000], training_epochs = 1, reg = 0, lr =3e-3)
        logger.info("Trained one epoch to attempt to make weights from different models compatible")

        #writer.add_graph(sess.graph)
        loss_reg, acc_reg = model.evaluate(sess, x_test_flat, y_test)
        logger.info("----Regular test accuracy and loss ----")
        logger.info((loss_reg, acc_reg))

        logger.info("Evaluating on FGSM points now.")
        loss_fgsm, acc_fgsm = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = False)
        logger.info("----FGSM test accuracy and loss ----")
        logger.info((loss_fgsm, acc_fgsm))
        x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)

        logger.info("Evaluating on PGD points now.")
        loss_pgd, acc_pgd = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = True, eta=1e-2, num_iter = 40)
        logger.info("----PGD test accuracy and loss ----")
        logger.info((loss_pgd , acc_pgd))

    return model, sess

def adversarial_training(config):
    eps_train = config['eps_train']
    eps_test = config['eps_test']
    tensorboard_dir = config['tensorboard_dir']
    weights_dir = config['weights_dir']
    load_counter = config['load_counter']
    sigma = config['sigma']
    scope_name_rob = config['scope_name']
    sess = tf.Session()
    hidden_sizes = [32,32,32,32,32,32,32]

    dataset, input_shape, num_classes  = get_dataset()
    x_train_flat, y_train = dataset[0]
    x_test_flat, y_test = dataset[1]
    sess = tf.Session()
    hidden_sizes = [32,32,32,32,32,32,32]
    dataset = ((x_train_flat, y_train), (x_test_flat, y_test))

    with tf.variable_scope(scope_name_rob, reuse = tf.AUTO_REUSE) as scope:
        tmp = set(tf.all_variables())
        logdir = tensorboard_dir + str(counter) + "/robust"
        writer_robust = tf.summary.FileWriter(logdir)
        logger.info("Adversarial Training")
        robust_model = ffr.RobustMLP(input_shape, hidden_sizes, num_classes, writer = writer_robust, scope = scope_name_rob, logger = logger, sigma = sigma)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - tmp))
        robust_model.adv_fit(sess, x_train_flat, y_train, eps_train, lr = 3e-4, training_epochs = 20)

        print(robust_model.evaluate(sess, x_test_flat, y_test))
        print(robust_model.adv_evaluate(sess, x_test_flat, y_test, eps_test))
    return robust_model, sess

def non_robust_model_train(counter, logger):
    config = {}
    config['eps_train'] = 0.1
    config['eps_test'] = 0.1
    config['tensorboard_dir'] = "tb/"
    config['weights_dir'] = "weights/"

    config['load_counter'] = 222
    config['write_counter'] = counter
    config['sigma'] = tf.nn.relu
    config['epochs'] = 30
    config['reg'] = 0.0
    config['lr'] = 3e-4
    config['reg_trace_first'] = 0
    config['reg_op'] = 0
    config['reg_trace_all'] = 0
    config['reg_l1'] = 0


    config['scope_name'] = "model_non_robust"
    config['should_load'] = False
    config['logger'] = logger
    return regular_training(config)

def l1_reg_model_train(counter, logger, reg):
    config = {}
    config['eps_train'] = 0.1
    config['eps_test'] = 0.1
    config['tensorboard_dir'] = "tb/"
    config['weights_dir'] = "weights/"

    config['load_counter'] = 222
    config['write_counter'] = counter
    config['sigma'] = tf.nn.relu
    config['epochs'] = 50
    config['reg'] = 0.0
    config['lr'] = 3e-3
    config['reg_trace_first'] = 0
    config['reg_op'] = 0
    config['reg_trace_all'] = 0
    config['reg_l1'] = reg

    config['scope_name'] = "model_non_robust"
    config['should_load'] = False
    config['logger'] = logger
    return regular_training(config)

def op_reg_model_train(counter, logger, reg):
    config = {}
    config['eps_train'] = 0.1
    config['eps_test'] = 0.1
    config['tensorboard_dir'] = "tb/"
    config['weights_dir'] = "weights/"

    config['load_counter'] = 222
    config['write_counter'] = counter
    config['sigma'] = tf.nn.relu
    config['epochs'] = 100
    config['reg'] = 0.0
    config['lr'] = 3e-4
    config['reg_trace_first'] = 0
    config['reg_op'] = reg 
    config['reg_trace_all'] = 0
    config['reg_l1'] = 0

    config['scope_name'] = "model_non_robust"
    config['should_load'] = False
    config['logger'] = logger
    return regular_training(config)

def trace_reg_model_train(counter, logger, reg):
    config = {}
    config['eps_train'] = 0.1
    config['eps_test'] = 0.1
    config['tensorboard_dir'] = "tb/"
    config['weights_dir'] = "weights/"

    config['load_counter'] = 222
    config['write_counter'] = counter
    config['sigma'] = tf.nn.relu
    config['epochs'] = 30
    config['reg'] = 0.0
    config['lr'] = 3e-4
    config['reg_trace_first'] = 0
    config['reg_op'] = 0
    config['reg_trace_all'] = reg 
    config['reg_l1'] = 0

    config['scope_name'] = "model_non_robust"
    config['should_load'] = False
    config['logger'] = logger
    return regular_training(config)

def trace_first_reg_model_train(counter, logger, reg):
    config = {}
    config['eps_train'] = 0.1
    config['eps_test'] = 0.1
    config['tensorboard_dir'] = "tb/"
    config['weights_dir'] = "weights/"

    config['load_counter'] = 222
    config['write_counter'] = counter
    config['sigma'] = tf.nn.relu
    config['epochs'] = 30
    config['reg'] = 0.0
    config['lr'] = 3e-3
    config['reg_trace_first'] = reg
    config['reg_op'] = 0
    config['reg_trace_all'] = 0
    config['reg_l1'] = 0

    config['scope_name'] = "model_non_robust"
    config['should_load'] = False
    config['logger'] = logger
    return regular_training(config)