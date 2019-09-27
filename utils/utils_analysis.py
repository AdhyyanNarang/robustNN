import logging
import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import feedforward_robust as ffr
import ipdb
from mnist_corruption import *
from utils_feedforward import *
#TODO: Figure out util imports
#from util_ import *

def plot_singular_values(matrix, idx, title):
    _, sig, V = np.linalg.svd(matrix)
    plt.plot(range(len(sig)), sig, label = "Layer number %d" %idx)
    plt.legend(loc = 'upper right')
    plt.title(title)
    return sig

def activation_distance(activation_list):
    diff = []
    diff_norms = []
    for i in range(len(activation_list) - 1):
        difference = activation_list[i+1] - activation_list[i]
        diff.append(difference)
        diff_norms.append(np.linalg.norm(difference)/np.linalg.norm(activation_list[i]))
    return diff, diff_norms

def make_plot_acc_vs_eps_black_box(model_list, sess_list, eps_list, angle_list, legend_list, scope_list):
    x_test_2d = np.array([vec.reshape((28,28)) for vec in x_test_flat])
    fig = plt.figure(figsize = (10,10))
    ax_gb = fig.add_subplot(2, 2, 1)
    ax_rp = fig.add_subplot(2, 2, 2)
    ax_bw = fig.add_subplot(2, 2, 3)
    ax_rot = fig.add_subplot(2, 2, 4)
    
    for idx, model in enumerate(model_list):
        sess = sess_list[idx]
        legend = legend_list[idx]
        gb_acc = []
        rp_acc = []
        bw_acc = []
        rot_acc = []
        
        scope_name = scope_list[idx]
        
        for (idx, eps_test) in enumerate(eps_list):
            x_test_gaussian_blurred = flatten_mnist(gaussian_blurring(x_test_2d, eps_test))[0]
            x_test_random_pert = flatten_mnist(random_perturbation(x_test_2d, eps_test))[0]
            x_test_black_white = flatten_mnist(random_blackout_whiteout(x_test_2d, eps_test))[0]
            angle = angle_list[idx]
            x_test_rot = flatten_mnist(rotate(x_test_2d, angle))[0]
            
            #ipdb.set_trace()
            gb_acc.append(model.evaluate(sess, x_test_gaussian_blurred, y_test)[1])
            rp_acc.append(model.evaluate(sess, x_test_random_pert, y_test)[1])
            bw_acc.append(model.evaluate(sess, x_test_black_white, y_test)[1])
            rot_acc.append(model.evaluate(sess, x_test_rot, y_test)[1])

        ax_gb.plot(eps_list, gb_acc, label = legend)
        ax_rp.plot(eps_list, rp_acc, label = legend)
        ax_bw.plot(eps_list, bw_acc, label = legend)
        ax_rot.plot(angle_list, rot_acc, label = legend)
 
    ax_gb.set_title("Gaussian blurring")
    ax_rp.set_title("Random perturbation")
    ax_bw.set_title("Blackwhite")
    ax_rot.set_title("Rotation")
    
    plt.legend(loc = (2.5, 0.7))
    plt.show()
    return True

    def make_plot_acc_vs_eps(model_list, sess_list, eps_list, legend_list, scope_list, pgd = False):
        fig = plt.figure()
        x_test_2d = [vec.reshape((28,28)) for vec in x_test_flat]
        
        for idx, model in enumerate(model_list):
            sess = sess_list[idx]
            legend = legend_list[idx]
            y_array = []
            scope_name = scope_list[idx]
            
            for eps_test in eps_list:
                if not pgd:
                    _, acc_fgsm = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = False)
                    y_array.append(acc_fgsm)
                else:
                    with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE) as scope:
                        _, acc_pgd = model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = True, eta=1e-2, num_iter = 50)
                        y_array.append(acc_pgd)
                        
            plt.plot(eps_list, y_array, label = legend)
        
        plt.legend(loc = 1)
        plt.show()
        return True

    def get_stats_all_models(model_list, sess_list, scope_list):
        eps_test = 0.1
        reg_evals = []
        adv_evals = []
        margins = []
        dphi_dxs = []
            
        for idx, model in enumerate(model_list):
            sess = sess_list[idx]
            y_array = []
            scope_name = scope_list[idx]

            with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE) as scope:
                #Get dphi_dx
                dphi_dx_test = model.get_dphi_dx(sess, x_test_flat)[0]
                dphi_dxs.append(np.mean(np.abs(dphi_dx_test)))
                
                #Get reg_evals
                reg_evals.append(model.evaluate(sess, x_test_flat, y_test))
                
                #Get adv_evals
                adv_evals.append(model.adv_evaluate(sess, x_test_flat, y_test, eps_test, pgd = False))
                
                #Get margins
                x_test_feat_pert = model.attack_featurization_space(sess, x_test_flat, y_test, 1.0, 0.1, 200)
                loss_feat_adv, acc_feat_adv = model.evaluate_from_featurizations(sess, x_test_feat_pert, y_test)
                margins.append(acc_feat_adv)
                
        return reg_evals, adv_evals, margins, dphi_dxs


def make_dist_plot_all_models(model_list, sess_list, scope_list, legend_list, order):
    plt.figure()
    
    for idx, model in enumerate(model_list):
        sess = sess_list[idx]
        scope_name = scope_list[idx]
        legend = legend_list[idx]

        with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE) as scope:
            y_array = model.get_distance(sess, 0.10, x_test_flat, y_test, order = order)[0]
            plt.plot(range(len(y_array)), y_array, label = legend)
    
    plt.legend()
    plt.show()
    return True

