import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import ipdb
import matplotlib.pyplot as plt
import pandas as pd

"""
Functions that aid with visualization
"""
def write_sprite_image(filename, images, img_h = 28, img_w = 28):
    """
        Create a sprite image consisting of sample images
        :param filename: name of the file to save on disk
        :param shape: tensor of flattened images
    """

    # Invert grayscale image
    images = 1 - images

    # Calculate number of plot
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    # Make the background of sprite image
    sprite_image = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            img_idx = i * n_plots + j
            if img_idx < images.shape[0]:
                img = images[img_idx]
                sprite_image[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = img

    plt.imsave(filename, sprite_image, cmap='gray')
    print('Sprite image saved in {}'.format(filename))
    return True

def write_metadata(filename, labels):
    """
            Create a metadata file image consisting of sample indices and labels
            :param filename: name of the file to save on disk
            :param shape: tensor of labels
    """
    with open(filename, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("{}\t{}\n".format(index, label))

    print('Metadata file saved in {}'.format(filename))
    return True


    """
    Helper functions for sampled attack
    and ellipse attack
    """

    def calc_X_featurized_star(sess, model, y_train, x_train, num_samples_perturb, num_samples_ellipse, display_step = 1):
        A_list = []
        b_list = []
        x_featurized_star = []
        for (idx, x_i) in enumerate(x_train):
            if idx % display_step == 0:
                print("Training point number %d" % idx)
            perturbed_x_i = random_perturbation(x_i, eps = eps_train, num_samples = num_samples_perturb)
            featurized_perturbed_x = model.get_activation(sess, perturbed_x_i)[-2]
            A_i, b_i = learn_constraint_setV2(featurized_perturbed_x)
            A_list.append(A_i)
            b_list.append(b_i)
            x_i_star = solve_inner_opt_problem(sess, model, y_train[idx], num_samples_ellipse, A_i, b_i)
            x_featurized_star.append(x_i_star)
        return np.array(x_featurized_star)



