import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import ipdb
import matplotlib.pyplot as plt
import pandas as pd

"""
Helper functions for feedforward_robust
"""
def fully_connected_layer(x, output_dim, scope_name, weight_initializer, bias_initializer, sigma):
    #Reuse = True because everytime we wish to call model(x), we want it to
    #use the most recently learned weights.
    with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE) as scope:
        weight_shape = (x.shape[1], output_dim)
        w = tf.get_variable('weights', shape = weight_shape, initializer = weight_initializer)
        b = tf.get_variable('biases', output_dim, initializer=bias_initializer)
        z = tf.add(tf.matmul(x, w), b)
        a = sigma(z)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", a)
        return a

def model(x, hidden_sizes, num_classes):
    """
    Input: Placeholder x, sizes of hidden layers, num_classes
    Output: Activations of fully connected neural network
    """
    act = x
    activations = []

    initial = tf.contrib.layers.xavier_initializer(dtype = tf.float32)
    bias_initial = tf.initializers.zeros

    #Compute the prediction placeholder
    for i in range(len(hidden_sizes)):
        scope = 'fc_' + str(i)
        #act_fxn = tf.nn.relu
        act_fxn = tf.math.sigmoid
        act = fully_connected_layer(act, hidden_sizes[i], scope, initial, bias_initial, act_fxn)
        activations.append(act)

    scope = 'fc_' + str(len(hidden_sizes))
    predictions = fully_connected_layer(act, num_classes, scope, initial, bias_initial, tf.identity)

    return activations, predictions

def op_norm(matrix):
    column_norms = tf.norm(matrix, ord = 1, axis = 0)
    op_inf_inf = tf.reduce_max(column_norms)
    return op_inf_inf

def get_norms(weights_list):
    norms = []
    for weights in weights_list:
        norms.append(op_norm(weights))
    return norms

def regularize_op_norm(weights_list):
    penalty = 0
    for weights in weights_list:
        penalty += op_norm(weights)
    return penalty

def write_to_results_csv(epochs, op_reg, lr, adv_train_flag, activation, acc_reg, acc_robust, loss_reg, loss_robust, logdir, tb_dir, dist, norms):
    df = pd.read_excel("results.xlsx")
    index = len(df)
    df.loc[index] = [index, epochs, op_reg, lr, adv_train_flag, activation, acc_reg, acc_robust, loss_reg, loss_robust, logdir, tb_dir, dist, norms]
    df.to_excel("results.xlsx")
    return True

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

"""
Functions for ellipsoid and sampling
"""
def find_worst_point_in_set(sess, model, X, y):
    loss_vector = model.get_loss_vector(sess, X, y)
    idx = np.argmax(loss_vector)
    return X[idx]

def find_worst_featurization_in_set(sess, model, X_featurized, y):
    loss_vector = model.get_loss_vector_from_featurization(sess, X_featurized, y)
    idx = np.argmax(loss_vector)
    return X_featurized[idx]

def objective_function(A):
    return cp.atoms.log_det(A)

def sample_spherical(npoints, ndim=3):
    #Each sampled point is along a column
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def solve_inner_opt_problem(sess, model, y, num_samples, A, b):
    #Sample points from the boundary of the ellipse
    V = sample_spherical(num_samples, len(b))

    A_inverse = np.linalg.pinv(A)
    to_add = A_inverse @ b
    #TODO: Test the dimensions and ensure that adding works properly
    X_feat_sampled_from_U = (A_inverse@V + to_add[:, np.newaxis]).T
    y_two_d = np.array([y])
    y_set = y_two_d.repeat(repeats = num_samples, axis = 0)

    #Find the point that the model receives highest loss on
    x_star = find_worst_featurization_in_set(sess, model, X_feat_sampled_from_U, y_set)
    return x_star

def learn_constraint_setV2(X):
    tool = ellipsoid.EllipsoidTool()
    A, b = tool.getMinVolEllipse(P = X)
    return A, b

def flatten_mnist(x):
    n, img_rows, img_cols = x.shape
    D = img_rows * img_cols
    x_flattened = x.reshape(n, D)
    return x_flattened, (D, )
