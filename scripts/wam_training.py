import logging
import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
sys.path.append('../')
sys.path.append('../utils/')
import feedforward_robust as ffr
import ipdb
from utils.utils_feedforward import *
from utils.utils_visualize import *

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
weights_dir = "weights/"
load_weights = False
load_counter = 234
sigma = tf.nn.relu
epochs, reg, lr = 10, 0.07, 3e-4
pgd_eta, pgd_num_iter = 1e-2, 50
nnw, freq = 6, 5

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
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
fh.setFormatter(formatter)
logger.addHandler(fh)

if __name__ == "__main__":

    #Setup - Dataset stuff
    dataset, input_shape, num_classes  = get_dataset()
    x_train_flat, y_train = dataset[0]
    x_test_flat, y_test = dataset[1]
    sess = tf.Session()
    hidden_sizes = [32,32,32,32,32,32,32]
    dataset = ((x_train_flat, y_train), (x_test_flat, y_test))

    scope_name = "hybrid"
    if not load_weights:
        with tf.variable_scope(scope_name) as scope:

            logdir = tensorboard_dir + str(counter) #+ "/non_robust"

            #Create, train and test model
            writer = tf.summary.FileWriter(logdir)
            model = ffr.RobustMLP(input_shape, hidden_sizes, num_classes, writer = writer, scope = scope_name, logger = logger, sigma = sigma)
            sess.run(tf.global_variables_initializer())
            logger.info("Created model successfully. Now going to train")

            #TODO: Fit the model until convergence before running the distance experiments again
            model.fit_whac_a_mole(sess, x_train_flat,y_train, training_epochs = epochs, lr = lr, num_neurons_whac=nnw, freq_whac_per_epoch=freq)

            #Save weights
            weights = tf.trainable_variables()
            #weights = model.get_weights()[0] + model.get_weights()[1]
            saver = tf.train.Saver(weights)
            weights_path = saver.save(sess, weights_dir + "model_" + str(counter) + ".ckpt")
            logger.info("Saved model at %s"%weights_path)


            loss_reg, acc_reg = model.evaluate(sess, x_test_flat, y_test)
            loss_fgsm, acc_fgsm = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = False)
            logger.info("----Regular test accuracy and loss ----")
            logger.info((loss_reg, acc_reg))
            logger.info("----FGSM test accuracy and loss ----")
            logger.info((loss_fgsm, acc_fgsm))
            loss_pgd, acc_pgd = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = True, eta=pgd_eta, num_iter = pgd_num_iter)
            logger.info("----PGD test accuracy and loss ----")
            logger.info((loss_pgd , acc_pgd))

            writer.add_graph(sess.graph)
