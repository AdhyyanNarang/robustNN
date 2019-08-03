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
    More elaborate test for the get_weight_norms function
    Repeat CCA experiments
    TSNE - float plots

    NOW:
"""

#Config

#Four different methods to train the model.
#These flags determine which ones we wish to run
non_robust_flag = True 
adv_train_flag = True 
sampled_flag = False 
ellipse_flag = False
cca_flag = False


num_samples_ellipse = 100
num_samples_perturb = 50
eps_train = 0.1
eps_test = 0.1

if __name__ == "__main__":
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

            logdir = "tmp/5/non_robust"
            #Create, train and test model
            writer = tf.summary.FileWriter(logdir)
            model = ffr.RobustMLP(sess, input_shape, hidden_sizes, num_classes, dataset, writer = writer, scope = scope_name)
            print("Created model successfully. Now going to train")

            #TODO: Fit the model until convergence before running the distance experiments again
            model.fit(sess, x_train_flat, y_train, training_epochs = 6)
            print(model.evaluate(sess, x_test_flat, y_test))
            print(model.adv_evaluate(sess, x_test_flat, y_test, eps_test))

            #Distances and norms
            print(model.get_weight_norms(sess))
            writer.add_graph(sess.graph)
            overall, correct, incorrect = model.get_distance(sess, eps_test, x_test_flat, y_test)
            print(overall)
            x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
            ipdb.set_trace()
            dist = model.dist_calculator(sess, x_test_flat[0], x_test_flat-adv[0])

            """
            #TSNE visualization of final layer.
            x_test_flat_adv = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
            metadata_path = os.path.join(logdir, 'metadata.tsv')
            write_metadata(metadata_path, y_test_ogi[0:1000])
            sprite_path = os.path.join(logdir, 'sprite_images.png')
            write_sprite_image(sprite_path, x_test_ogi[0:1000])
            model.visualize_activation_tsne(sess, x_test_flat_adv[0:1000], 'metadata.tsv', 'sprite_images.png', logdir)

            print("-----After slashing-------")
            weights_new = model.slash_weights(sess)
            print(model.evaluate(sess, x_test_flat, y_test))
            print(model.adv_evaluate(sess, x_test_flat, y_test, eps_test))

            #Distances and norms
            print(model.get_weight_norms(sess))
            writer.add_graph(sess.graph)
            overall, correct, incorrect = model.get_distance(sess, eps_test, x_test_flat, y_test)
            print(overall)
            """
            #Activations for CCA
            if(cca_flag):
                x_adv_cca = model.fgsm_np(sess, x_test_flat, y_test, eps_test)
                activation_non = model.get_activation(sess, x_adv_cca)


    if(adv_train_flag):
        scope_name_rob = "model_robust"
        with tf.variable_scope(scope_name_rob) as scope:
            logdir = "tmp/5/robust"
            writer_robust = tf.summary.FileWriter(logdir)
            print("Adversarial Training")
            robust_model = ffr.RobustMLP(sess, input_shape, hidden_sizes, num_classes, dataset, writer = writer_robust, scope = scope_name_rob)
            robust_model.adv_fit(sess, x_train_flat, y_train, eps_train, lr = 3e-4, training_epochs = 20)
            print(robust_model.evaluate(sess, x_test_flat, y_test))
            print(robust_model.adv_evaluate(sess, x_test_flat, y_test, eps_test))
            print(robust_model.get_weight_norms(sess))
            overall, correct, incorrect = robust_model.get_distance(sess, eps_test, x_test_flat, y_test)
            print(overall)
            writer_robust.add_graph(sess.graph)
            
            ipdb.set_trace()
            #TSNE visualization of final layer.
            x_test_flat_adv = robust_model.fgsm_np(sess, x_test_flat, y_test, eps_test)
            metadata_path = os.path.join(logdir, 'metadata.tsv')
            write_metadata(metadata_path, y_test_ogi[0:1000])
            sprite_path = os.path.join(logdir, 'sprite_images.png')
            write_sprite_image(sprite_path, x_test_ogi[0:1000])
            robust_model.visualize_activation_tsne(sess, x_test_flat[0:1000], 'metadata.tsv', 'sprite_images.png', logdir)

            #Activations for CCA
            if(cca_flag):
                activation_rob = robust_model.get_activation(sess, x_adv_cca)

    if cca_flag:
        #Train new non-robust model
        scope_name = "model_non_robust_two"
        with tf.variable_scope(scope_name) as scope:
            logdir = "tmp/2/non_robust"
            #Create, train and test model
            writer = tf.summary.FileWriter(logdir)
            model = ffr.RobustMLP(sess, input_shape, hidden_sizes, num_classes, dataset, writer = writer, scope = scope_name)
            print("Created model successfully. Now going to train")
            model.fit(sess, x_train_flat, y_train, training_epochs = 25)
            print(model.evaluate(sess, x_test_flat, y_test))
            print(model.adv_evaluate(sess, x_test_flat, y_test, eps_test))
            writer.add_graph(sess.graph)

            x_adv_flat = model.sample_attack_np(sess, x_test_flat, y_test, eps_test * 3)
            print(model.evaluate(sess, x_adv_flat, y_test))
            activation_non_two = model.get_activation(sess, x_adv_cca)

        #Train new robust model
        scope_name_rob = "model_robust_two"
        with tf.variable_scope(scope_name_rob) as scope:
            logdir = "tmp/2/robust"
            writer_robust = tf.summary.FileWriter(logdir)
            print("Adversarial Training")
            robust_model = ffr.RobustMLP(sess, input_shape, hidden_sizes, num_classes, dataset, writer = writer_robust, scope = scope_name_rob)
            robust_model.adv_fit(sess, x_train_flat, y_train, eps_train, lr = 3e-4, training_epochs = 25)
            print(robust_model.evaluate(sess, x_test_flat, y_test))
            print(robust_model.adv_evaluate(sess, x_test_flat, y_test, eps_test))
            writer_robust.add_graph(sess.graph)
            #TSNE visualization of final layer.
            x_test_flat_adv = robust_model.fgsm_np(sess, x_test_flat, y_test, eps_test)
            metadata_path = os.path.join(logdir, 'metadata.tsv')
            write_metadata(metadata_path, y_test_ogi[0:1000])
            sprite_path = os.path.join(logdir, 'sprite_images.png')
            write_sprite_image(sprite_path, x_test_ogi[0:1000])
            robust_model.visualize_activation_tsne(sess, x_test_flat_adv[0:1000], 'metadata.tsv', 'sprite_images.png', logdir)

            #Activations for CCA
            if(cca_flag):
                activation_rob = robust_model.get_activation(sess, x_adv_cca)

        #Get CCA and print them


    X_star = np.copy(x_train_flat)
    #Find X_star: Method 1 by sampling

    if (sampled_flag):
        """
        for (idx,x_i) in enumerate(x_train_flat):
            print(idx)
            perturbed_x = random_perturbation(x_i, eps = eps_train, num_samples = 10)
            y_i = np.array([y_train[idx]])
            y_set = y_i.repeat(repeats = 10, axis = 0)
            x_i_star = find_worst_point_in_set(sess, model, perturbed_x, y_set)
            X_star[idx] = x_i_star
        print("Yes I found some X_star")
        X_star = np.load("X_star.npy")
        loss, accuracy = model.evaluate(sess, X_star, y_train)
        print(accuracy)
        """
        scope_name_sampled = "model_robust"
        with tf.variable_scope(scope_name_sampled) as scope:
            logdir = "tmp/2/sampled"
            writer_robust = tf.summary.FileWriter(logdir)
            print("Final layer training using sampled adversarial attack")

            robust_model = ffr.RobustMLP(sess, input_shape, hidden_sizes, num_classes, dataset, writer = writer_robust, scope = scope_name_sampled)
            robust_model.fit(sess, x_train_flat, y_train, training_epochs = 3)
            print(robust_model.evaluate(sess, x_test_flat, y_test))
            print(robust_model.adv_evaluate(sess, x_test_flat, y_test, eps_test))
            robust_model.fit_robust_final_layer(sess, x_train_flat, y_train, eps = 0.10, training_epochs = 6)
            print(robust_model.evaluate(sess, x_test_flat, y_test))
            print(robust_model.adv_evaluate(sess, x_test_flat, y_test, eps_test))

            #print(robust_model.get_weight_norms(sess))
            #overall, correct, incorrect = robust_model.get_distance(sess, eps_test, x_test_flat, y_test, order = 2)
            #print(overall)
            writer_robust.add_graph(sess.graph)

            """
            #TSNE visualization of final layer.
            x_test_flat_adv = robust_model.fgsm_np(sess, x_test_flat, y_test, eps_test)
            metadata_path = os.path.join(logdir, 'metadata.tsv')
            write_metadata(metadata_path, y_test_ogi[0:1000])
            sprite_path = os.path.join(logdir, 'sprite_images.png')
            write_sprite_image(sprite_path, x_test_ogi[0:1000])
            robust_model.visualize_activation_tsne(sess, x_test_flat_adv[0:1000], 'metadata.tsv', 'sprite_images.png', logdir)
            """


    #Find X_star: Method 2 by Ellipse
    if (ellipse_flag):
        start = time.time()
        #X_featurized_star = calc_X_featurized_star(sess, model, y_train, x_train_flat, num_samples_perturb, num_samples_ellipse, display_step = 200)
        #np.save("X_feat_star_01", X_featurized_star)
        X_featurized_star = np.load("X_feat_star_01.npy")
        X_featurized_regular = model.get_activation(sess, x_train_flat)[-2]

        X_feat_train = X_featurized_star.copy()
        for i in range(len(X_feat_train)):
            rnd = np.random.uniform()
            if rnd < 0:
                X_feat_train[i] = X_featurized_regular[i]

        """
        x_dummy = x_train_flat[0]
        perturbed_dummy = random_perturbation(x_dummy, num_samples =50)
        mapping_perturbed_dummy = model.get_activation(sess, perturbed_dummy)[-2]
        ipdb.set_trace()
        print("Now going to find A, b")
        A, b = learn_constraint_setV2(mapping_perturbed_dummy)
        print("Now going to find x_star on boundary of the ellipse")
        x_dummy_star = solve_inner_opt_problem(sess, model, y_train[0], 100, A, b)
        """
        end = time.time()
        print(end - start)
        x_test_noisy = gaussian_perturbation(x_test_flat)
        print(model.evaluate(sess, x_train_flat, y_train))
        print(model.evaluate(sess, x_test_flat, y_test))
        print(model.evaluate(sess, x_test_noisy, y_test))
        model.fit_robust_final_layer(sess, X_feat_train, y_train, feed_features_flag = True, batch_size = 60000, training_epochs = 400)
        print(model.evaluate(sess, x_train_flat, y_train))
        print(model.evaluate(sess, x_test_flat, y_test))
        print(model.evaluate(sess, x_test_noisy, y_test))
