import logging
import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
sys.path.append('../')
sys.path.append('../utils/')
import feedforward_robust as ffr
import ipdb
from utils.util_feedforward import *
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
epochs, reg, lr = 100, 0.07, 3e-4
pgd_eta, pgd_num_iter = 1e-2, 50

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
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') fh.setFormatter(formatter)
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
            model.fit(sess, x_train_flat,y_train, training_epochs = epochs, reg = reg , lr = lr)

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

            #Distances and norms
            norms = get_norms(model.get_weights()[0])
            norms_np = sess.run(norms)
            logger.info(norms_np)

            overall, overall_std, correct, _, incorrect, _ = model.get_distance(sess, eps_test, x_test_flat, y_test)
            logger.info("---Distances----")
            logger.info(overall)
            logger.info("------Std devs on Distances----")
            logger.info(overall_std)

            write_to_results_csv(epochs, reg, lr, "Regular", str(sigma), acc_reg, acc_fgsm, acc_pgd, loss_reg, loss_fgsm, acc_fgsm, logfile, weights_path, logdir, tuple(overall), tuple(norms_np))
            x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)

            #TSNE visualization of final layer.
            x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
            metadata_path = os.path.join(logdir, 'metadata.tsv')
            write_metadata(metadata_path, y_test_ogi[0:1000])
            sprite_path = os.path.join(logdir, 'sprite_images.png')
            write_sprite_image(sprite_path, x_test_ogi[0:1000])
            model.visualize_activation_tsne(sess, x_test_flat_adv[0:1000], 'metadata.tsv', 'sprite_images.png', logdir)


    else:
        with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE) as scope:
            logdir = tensorboard_dir + str(counter) + "/non_robust"

            #Create, train and test model
            writer = tf.summary.FileWriter(logdir)
            model = ffr.RobustMLP(input_shape, hidden_sizes, num_classes, writer = writer, scope = scope_name, logger = logger, sigma =sigma)
            logger.info("Created model successfully. Now going to load weights")

            #Restore weights
            weights = tf.trainable_variables()
            saver = tf.train.Saver(weights)
            weights_path = saver.restore(sess, weights_dir + "model_" + str(load_counter) + ".ckpt")

            writer.add_graph(sess.graph)

            loss_reg, acc_reg = model.evaluate(sess, x_test_flat, y_test)
            logger.info("----Regular test accuracy and loss ----")
            logger.info((loss_reg, acc_reg))
            ipdb.set_trace()

            loss_reg, acc_reg = model.evaluate(sess, x_test_flat, y_test)
            logger.info("----Regular test accuracy and loss ----")
            logger.info((loss_reg, acc_reg))
            feed_dict = {model.x: x_test_flat, model.y: y_test}
            summary = sess.run(model.merged_summary, feed_dict = feed_dict)
            model.writer.add_summary(summary, 0)

            loss_fgsm, acc_fgsm = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = False)
            logger.info("----FGSM test accuracy and loss ----")
            logger.info((loss_fgsm, acc_fgsm))
            x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
            feed_dict = {model.x: x_test_flat_adv, model.y: y_test}
            summary = sess.run(model.merged_summary, feed_dict = feed_dict)
            model.writer.add_summary(summary, 100)
            loss_pgd, acc_pgd = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = True, eta=pgd_eta, num_iter = pgd_num_iter)
            logger.info("----PGD test accuracy and loss ----")
            logger.info((loss_pgd , acc_pgd))

            #Tries again a second time to check out whether the scopes work
            loss_pgd, acc_pgd = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = True, eta=pgd_eta, num_iter = pgd_num_iter)
            logger.info("----PGD test accuracy and loss ----")
            logger.info((loss_pgd , acc_pgd))

            #Distances and norms
            if False:
                norms = get_norms(model.get_weights()[0])
                norms_np = sess.run(norms)
                logger.info(norms_np)
                overall, overall_std, correct, _, incorrect, _ = model.get_distance(sess, eps_test, x_test_flat, y_test)
                logger.info("---Distances----")
                logger.info(overall)
                logger.info("------Std devs on Distances----")
                logger.info(overall_std)


            if True:
                #TSNE visualization of final layer.
                x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
                metadata_path = os.path.join(logdir, 'metadata.tsv')
                write_metadata(metadata_path, y_test_ogi[0:1000])
                sprite_path = os.path.join(logdir, 'sprite_images.png')
                write_sprite_image(sprite_path, x_test_ogi[0:1000])
                model.visualize_activation_tsne(sess, x_test_flat_adv[0:1000], 'metadata.tsv', 'sprite_images.png', logdir)
