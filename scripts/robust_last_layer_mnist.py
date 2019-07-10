import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
sys.path.append('../')
import feedforward_robust as ffr
import ipdb
from mnist_corruption import random_perturbation
import cvxpy as cp

#Config
method_1_flag = False 

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
    print("Going to sample points from the sphere")
    V = sample_spherical(num_samples, len(b))
    ipdb.set_trace()

    print("Random linalg to find the X from the V")
    A_inverse = np.linalg.inv(A)
    to_add = A_inverse @ b
    #TODO: Test the dimensions and ensure that adding works properly
    X_feat_sampled_from_U = (A_inverse@V + to_add[:, np.newaxis]).T
    y_two_d = np.array([y])
    y_set = y_two_d.repeat(repeats = num_samples, axis = 0)

    #Find the point that the model receives highest loss on 
    print("Find the argmax of the sampled points!")
    x_star = find_worst_featurization_in_set(sess, model, X_feat_sampled_from_U, y_set)

def calc_X_featurized_star(sess, x_train_flat, num_samples):
    A_list = []
    b_list = []
    for (idx, x_i) in enumerate(x_train_flat):
        perturbed_x_i = random_perturbation(x_i, num_samples = num_samples)
        featurized_perturbed_x = model.get_activation(sess, perturbed_x_i)
        A_i, b_i = learn_constraint_set(featurized_perturbed_x)
        A_list.append(A_i)
        b_list.append(b_i)
        x_i_star = solve_inner_opt_problem(A_i, b_i)


def learn_constraint_set(sampled_points):
    m,n = sampled_points.shape
    A = cp.Variable(shape = (n,n), symmetric = True)
    b = cp.Variable(n)
    objective = cp.Maximize(objective_function(A))
    constraints = []
    for i in range(m):
        x_i = sampled_points[i].T
        constraints.append(cp.atoms.norm(A*x_i + b) <= 1)
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print(prob.value)
    return A.value, b.value

def flatten_mnist(x):
    n, img_rows, img_cols = x.shape
    D = img_rows * img_cols
    x_flattened = x.reshape(n, D)
    return x_flattened, (D, )

if __name__ == "__main__":

    #Setup - Dataset stuff
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train/255
    x_test = x_test/255
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    x_train_flat, input_shape = flatten_mnist(x_train)
    x_test_flat, _ = flatten_mnist(x_test)


    #Create and train the model
    sess = tf.Session()
    hidden_sizes = [32,32,32,32]
    dataset = ((x_train_flat, y_train), (x_test_flat, y_test))
    model = ffr.RobustMLP(sess, input_shape, hidden_sizes, num_classes, dataset)
    print("Created model successfully. Now going to train")
    model.fit(sess, x_train_flat, y_train, training_epochs = 3)
    print("Trained model successfully. Moving to robustify....")

    X_star = np.copy(x_train_flat)
    #Find X_star: Method 1 by sampling

    if (method_1_flag):
        """
        for (idx,x_i) in enumerate(x_train_flat):
            print(idx)
            perturbed_x = random_perturbation(x_i, num_samples = 10)
            y_i = np.array([y_train[idx]])
            y_set = y_i.repeat(repeats = 10, axis = 0)
            x_i_star = find_worst_point_in_set(sess, model, perturbed_x, y_set)
            X_star[idx] = x_i_star
        print("Yes I found some X_star")
        """
        X_star = np.load("X_star.npy")
        loss, accuracy = model.evaluate(sess, X_star, y_train)
        print(accuracy)
        model.fit_robust_final_layer(sess, X_star, y_train, training_epochs = 100)

    #Find X_star: Method 2 by Ellipse
    else:
        #X_featurized_star = calc_X_featurized_star(sess, x_train_flat, num_samples)
        x_dummy = x_train_flat[0]
        perturbed_dummy = random_perturbation(x_dummy, num_samples = 10)
        mapping_perturbed_dummy = model.get_activation(sess, perturbed_dummy)[-2]
        print("Now going to find A, b")
        A, b = learn_constraint_set(mapping_perturbed_dummy)
        print("Now going to find x_star on boundary of the ellipse")
        x_dummy_star = solve_inner_opt_problem(sess, model, y_train[0], 100, A, b)
