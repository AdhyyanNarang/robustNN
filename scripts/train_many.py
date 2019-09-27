#Imports
import logging
import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import defaultdict
sys.path.append('../')
sys.path.append('../utils/')
import feedforward_robust as ffr
import ipdb
from utils.mnist_corruption import *
from utils.utils_models import *
from utils.utils_analysis import *
from utils.utils_feedforward import *

if __name__ == '__main__':
    #Read the counter
    ctr_file = "counter.txt"
    f = open(ctr_file, 'r')
    counter = f.readline()
    f.close()

    counter = 1 + int(counter)
    
    f.close()
    logfile = "logs/results_many_" + str(counter) + ".log"

    logger = logging.getLogger("robustness")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

    model_paths = defaultdict(list)
    print("OK entering loop")
    for i in range(1):
        print("OK entered loop")

        _, _, non_rob_path = non_robust_model_train(counter, logger)
        model_paths['non_robust'].append(non_rob_path)
        counter +=1
        tf.reset_default_graph()

        _, _, l1_path = l1_reg_model_train(counter, logger, 0.0009)
        model_paths['l1'].append(l1_path)
        counter +=1
        tf.reset_default_graph()

        _, _, op_path = op_reg_model_train(counter, logger, 0.09)
        counter +=1 
        model_paths['op'].append(op_path)
        tf.reset_default_graph()

        _, _, trace_first_path = trace_first_reg_model_train(counter, logger, 0.01)
        model_paths['trace_first'].append(trace_first_path)
        counter +=1
        tf.reset_default_graph()

        _, _, trace_reg_path = trace_reg_model_train(counter, logger, 0.02)
        model_paths['trace_all'].append(trace_reg_path)
        counter += 1
        tf.reset_default_graph()

        _, _, fgsm_train_path = fgsm_adv_train(counter, logger)
        model_paths['fgsm'].append(fgsm_train_path)
        counter += 1
        tf.reset_default_graph()


    f = open(ctr_file,'w')
    f.write('{}'.format(counter))
    np.save("paths_regularized.npy", model_paths)

