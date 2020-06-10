import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.tensorboard.plugins import projector
import ipdb
sys.path.append("utils/")
from utils.utils_feedforward import *
from utils.utils_visualize import *
import cvxpy as cp


"""
TODOS:
    Make NN creation more general for flexibility with experiments
    For fisher rao norm, ensure that I'm fetching the right predictions
"""
class RobustMLP(object):

    def __init__(self,input_shape,hidden_sizes,num_classes, writer, scope, logger, sigma, classification = True):

        #Initialize instance variables
        #TODO: Fix this hacky solution.
        input_shape = input_shape[0]
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.writer = writer
        self.scope = scope
        self.logger = logger
        self.sigma = sigma
        self.logger.debug("Initialized instance variables of the robust model class")

        #Creates the graph
        x_shape = [None] + [self.input_shape]
        self.x = tf.placeholder("float", x_shape)
        self.y = tf.placeholder("float", [None, num_classes])
        self.logger.debug("Created placeholders for x and y")

        self.activations, self.predictions = model(self.x, self.hidden_sizes, self.num_classes, sigma)
        self.featurizations = self.activations[-1]
        self.logger.debug("Created layers and tensor for logits")

        self.correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.logger.debug("Added accuracy computation to the graph")
        tf.summary.scalar("accuracy", self.accuracy)

        self.loss = None
        if classification:
            self.loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.y)
            self.loss = tf.reduce_mean(self.loss_vector)
            tf.summary.scalar("loss", self.loss)
            self.logger.debug("Added cross-entropy loss computation to the graph")
        else :
            self.loss = tf.nn.l2_loss(self.predictions - self.y)
            tf.summary.scalar("loss", self.loss)
            self.logger.debug("Added MSE loss computation to the graph")

        self.logger.info("Model graph was created")
        self.merged_summary = tf.summary.merge_all(scope = self.scope)

    def get_activation(self, sess, x_input):
        activations = sess.run(self.activations,
                               feed_dict = {
                                   self.x: x_input,
                               })
        return activations

    def get_prediction(self, sess, x_input):
        prediction = sess.run(self.predictions,
                               feed_dict = {
                                   self.x: x_input,
                               })
        return prediction

    def get_prediction_number(self, sess, x_input):
        prediction_simplex = tf.nn.softmax(self.predictions)
        prediction_simplex_np = sess.run(prediction_simplex,
                               feed_dict = {
                                   self.x: x_input,
                               })

        return np.argmax(prediction_simplex_np, axis = 1)

    def get_featurizations(self, sess, x_input):
        feats_np = sess.run(self.featurizations,
                               feed_dict = {
                                   self.x: x_input,
                               })
        return feats_np

    def get_loss_vector(self, sess, X, y):
        feed_dict = {
            self.x : X,
            self.y : y
        }
        loss_vector = sess.run(self.loss_vector, feed_dict = feed_dict)
        return loss_vector

    def get_loss_vector_from_featurization(self, sess, X_feat, y):
        feed_dict = {
            self.featurizations: X_feat,
            self.y : y
        }
        loss_vector = sess.run(self.loss_vector, feed_dict = feed_dict)
        return loss_vector

    def get_weights(self):
        weights = []
        biases = []
        for i in range(len(self.hidden_sizes) + 1):
            scope_name = 'fc_' + str(i)
            with tf.variable_scope(scope_name, reuse = True):
                w = tf.get_variable('weights')
                b = tf.get_variable('biases')
                weights.append(w)
                biases.append(b)
        return weights, biases

    def get_weights_np(self, sess):
        weights = self.get_weights()[0]
        weights_np = sess.run(weights)
        return weights_np

    def get_weight_norms(self, sess, matrix_norm_fxn = lambda x: np.linalg.norm(x, ord = 1)):
        """
        Getter method for the norms of the weight matrices of
        the network
        """
        model_norms = []
        weights_list = self.get_weights_np(sess)
        for weights in weights_list:
            norm = matrix_norm_fxn(weights)
            model_norms.append(norm)
        return model_norms

    def get_pred_dx(self, sess, x_train):
        """
        Gets gradients of features with respect to input
        """
        dpred_dx = tf.gradients(self.predictions, self.x)
        feed_dict = {self.x: x_train}
        dpred_dx_np = sess.run(dpred_dx, feed_dict = feed_dict)
        return dpred_dx_np

    #TODO: Move this to some utils file
    def slice_tensor_one_d(self, tensor, m):
        sliced = [tensor[:, i] for i in range(m)]
        return sliced

    def get_dphi_dx(self, sess, x_test):
        """
        Gets gradients of features with respect to input
        """
        self.logger.debug("Finding dphi dx")
        n, d = x_test.shape

        #TODO: Remove the hardcoding from here
        featurization_sliced = self.slice_tensor_one_d(self.featurizations, 32)
        num_hidden_last_layer = len(featurization_sliced)
        self.logger.debug("Completed slicing")

        run_list = []
        for i in range(len(featurization_sliced)):
            dphi_i_dx = tf.gradients(featurization_sliced[i], self.x)
            run_list.append(dphi_i_dx)

        self.logger.debug("Found run list")
        feed_dict = {self.x: x_test}
        dphi_dx_np = sess.run(run_list, feed_dict = feed_dict)
        dphi_dx_np = np.array(dphi_dx_np).squeeze()

        self.logger.debug("Found grads")
        #dphi_dx_np = dphi_dx_np.reshape((n, num_hidden_last_layer, d))
        dphi_dx_np = dphi_dx_np.swapaxes(0, 1)
        return dphi_dx_np

    def get_dphi_dx_fast(self, sess, x_train):
        """
        Gets gradients of features with respect to input
        """
        dphi_dx = tf.gradients(self.featurizations, self.x)
        feed_dict = {self.x: x_train}
        dphi_dx_np = sess.run(dphi_dx, feed_dict = feed_dict)
        return dphi_dx_np

    def get_margin_qp_single_class(self, x, w_corr, w_inc, S):
        """
        Find perturbation with smallest weighted norm
        that makes the point classified as the other
        class
        Use CVXPY to solve this QP.
        S_ii: the penalty for using the ith feature. Higher if
        it has lower gradient.
        """
        inf_idxs = []
        for i in range(len(S)):
            if S[i,i] == np.inf:
                S[i,i] = 1
                inf_idxs.append(i)

        delta = cp.Variable(shape = x.shape)
        cost = cp.sum_squares(S@delta)
        constraints = [
            w_corr.T@(x+delta) -  w_inc.T@(x + delta) <=0
        ]

        for j, inf_idx in enumerate(inf_idxs):
            constraints.append(delta[inf_idx] == 0)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver = cp.GUROBI)
        margin_squared = prob.value
        return np.sqrt(margin_squared)

    def get_margin_qp_pointwise(self, x_test, y_test, weights, S):
        """
        Find smallest perturbation to featurized x that throws it across
        the decision boundary
        """
        correct_index = np.where(y_test)[0][0]
        incorrect_indices = [i for i in range(len(y_test)) if i != correct_index]
        w_corr = weights[:, correct_index]

        margins_all_classes = []
        for inc_idx in incorrect_indices:
            w_inc = weights[inc_idx]
            w_inc = weights[:, inc_idx]
            margin_single_class = self.get_margin_qp_single_class(x_test, w_corr, w_inc, S)
            margins_all_classes.append(margin_single_class)
        return np.min(margins_all_classes)

    def get_margin_qp_aggregate(self, sess, x_test, y_test, weighted = False):
        """
        Solve lp to get margin for each test point
        """
        self.logger.info("Finding aggregate margin")
        margins = []
        #Expected shape = (10, 32)
        weights = self.get_weights_np(sess)[-1]
        phi_x_test = self.get_featurizations(sess, x_test)
        preds = self.get_prediction_number(sess, x_test)
        G_all_points = None
        if weighted:
            G_all_points = self.get_dphi_dx(sess, x_test)
        S = np.eye(32)

        for idx, phi_x in enumerate(phi_x_test):
            if idx % 300 == 0:
                self.logger.debug("Point Number:%d" %idx)
            pred = preds[idx]
            if pred != np.where(y_test[idx])[0]:
                #margins.append(0)
                continue
            else:
                if weighted:
                    G_single_point = G_all_points[idx]
                    vector = np.linalg.norm(G_single_point, ord = 1, axis = 1)
                    vector = [1.0/l for l in vector]
                    S = np.diag(vector)
                pw_margin = self.get_margin_qp_pointwise(phi_x, y_test[idx], weights, S)
                margins.append(pw_margin)
        return margins

    def get_margin_native_approx_single_class(self, phi_x, w_corr, w_inc, G):
        """
        Find smallest perturbation that makes the point classified as the other
        class
        Use CVXPY to solve this QP.
        """
        d = G.shape[1]
        delta = cp.Variable(shape = d)
        cost = cp.sum_squares(delta)
        #cost = cp.norm(delta, "inf")
        c = None
        if phi_x@w_corr >= 0:
            #c = 5.5
            c = 1
        else:
            self.logger.debug("Oh really? That's weird")
            c = 0.5

        constraints = [
            delta.T@(c*G.T@w_corr - G.T@w_inc) + phi_x.T@(c*w_corr - w_inc) <=0
        ]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver = cp.GUROBI)
        except:
            self.logger.error("An error occured with solving for point")
            return None
        margin_squared = prob.value
        delta_found = delta.value
        return np.sqrt(margin_squared), delta_found
        #return margin_squared, delta_found

    def get_margin_native_approx_pointwise(self, x_test, y_test, weights, G_one_point):
        """
        Find smallest perturbation to featurized x that throws it across
        the decision boundary
        """
        correct_index = np.where(y_test)[0][0]
        incorrect_indices = [i for i in range(len(y_test)) if i != correct_index]
        w_corr = weights[:, correct_index]

        margins_all_classes = []
        deltas = []
        for inc_idx in incorrect_indices:
            w_inc = weights[inc_idx]
            w_inc = weights[:, inc_idx]
            margin_single_class, delta_class = self.get_margin_native_approx_single_class(x_test, w_corr, w_inc, G_one_point)
            if margin_single_class == None:
                continue
            margins_all_classes.append(margin_single_class)
            deltas.append(delta_class)

        if len(margins_all_classes) == 0:
            return None
        else:
            idx = np.argmin(margins_all_classes)
            return margins_all_classes[idx], deltas[idx]

    def get_margin_native_approx_aggregate(self, sess, x_test, y_test):
        """
        Solve lp to get margin for each test point
        """
        self.logger.info("Finding aggregate margin")
        margins = []
        deltas = []
        indices = []
        #Expected shape = (10, 32)
        weights = self.get_weights_np(sess)[-1]
        phi_x_test = self.get_featurizations(sess, x_test)
        preds = self.get_prediction_number(sess, x_test)
        G_all_points = self.get_dphi_dx(sess, x_test)

        for idx, phi_x in enumerate(phi_x_test):
            if idx % 300 == 0:
                self.logger.debug("Point Number:%d" %idx)
            pred = preds[idx]
            if pred != np.where(y_test[idx])[0]:
                continue
            else:
                G_one_point = G_all_points[idx]
                pw_margin, delta_found = self.get_margin_native_approx_pointwise(phi_x, y_test[idx], weights, G_one_point)
                if pw_margin == None:
                    continue
                margins.append(pw_margin)
                deltas.append(delta_found)
                indices.append(idx)
        return margins, deltas, indices

    def get_dl_df(self, sess, x_test, y_test):
        """
        Helper for Fisher Rao Norm
        """
        dl_df = tf.gradients(self.loss, self.predictions)
        feed_dict = {self.x: x_test, self.y:y_test}
        dl_df_np = sess.run(dl_df, feed_dict = feed_dict)
        dl_df_np = np.array(dl_df_np)
        return dl_df_np.squeeze()

    def get_fisher_rao_norm_squared(self, sess, x_test, y_test):
        """
        Complexity measure proposed by Liang et.al
        """
        pred_np = self.get_prediction(sess, x_test)
        dl_df_np = self.get_dl_df(sess, x_test, y_test)
        prod = pred_np * dl_df_np
        inner_prod_vector = np.sum(prod, axis = 0)
        inner_prod_squared = inner_prod_vector * inner_prod_vector
        return np.mean(inner_prod_squared)

    def get_first_term(self, sess):
        weights_np = self.get_weights_np(sess)
        spectral_norm_vector = get_spectral_norm_elementwise(weights_np)
        return np.prod(spectral_norm_vector), spectral_norm_vector

    def get_second_term(self, sess):
        #Denominator stuff
        spectral_vector_lst = self.get_first_term(sess)[1]
        spectral_vector_lst = [elem ** (2/3) for elem in spectral_vector_lst]

        #Numerator stuff
        weights_np = self.get_weights_np(sess)

        #TODO: Try for some other types of reference matrices
        reference_matrices = [np.zeros((weights_np[i].shape[0], weights_np[i].shape[1])) for i in range(len(weights_np))]
        difference_lst = [weights_np[i] - reference_matrices[i] for i in range(len(weights_np))]
        numerator_lst = get_two_one_norm_elementwise(difference_lst)
        numerator_lst = [elem ** (2/3) for elem in numerator_lst]

        #Putting it together to get the sum
        frac_lst = np.divide(numerator_lst, spectral_vector_lst)
        final_sum = np.sum(frac_lst)
        return final_sum ** (3/2)

    def get_spectral_norm(self, sess):
        """
        Complexity measure of neural networks proposed by Bartlett et. al
        """
        first_term = self.get_first_term(sess)[0]
        second_term = self.get_second_term(sess)
        return first_term * second_term

    def compare_first_order_approx_with_actual(self, sess, x, delta):
        #delta = np.random.normal(scale = 0.01, size = x.shape)
        #delta = np.random.uniform(high = 0.01, shape = x.shape)

        perturbed = x + delta
        original_matrix = x.reshape(1, x.shape[0])
        perturbed_matrix = perturbed.reshape(1, perturbed.shape[0])

        phi_x_tilde = self.get_featurizations(sess, perturbed_matrix)
        phi_x = self.get_featurizations(sess, original_matrix)

        J = self.get_dphi_dx(sess, perturbed_matrix)
        approx = phi_x + delta@J
        ipdb.set_trace()
        return phi_x_tilde, approx, np.linalg.norm(phi_x_tilde - approx)

    def fgsm(self, x, eps):
        #TODO: Remove x as a parameter and change all function calls accordingly
        g = tf.gradients(self.loss, self.x)
        delta = eps * tf.math.sign(g)
        x_adv = self.x + delta
        #x_adv = tf.clip_by_value(x_adv, clip_value_min = 0.0, clip_value_max = float("inf"))
        return tf.squeeze(x_adv)

    def fgsm_np(self, sess, X, y, eps):
        x_adv = self.fgsm(X, eps)
        feed_dict = {
            self.x : X,
            self.y : y
        }
        x_adv_conc = sess.run(x_adv, feed_dict = feed_dict)
        return x_adv_conc

    def pgd_create_adv_graph(self, sess, X, y, eps, eta, scope):
        #with tf.variable_scope(scope, reuse = tf.AUTO_REUSE) as scope:
        #temp = set(tf.all_variables())
        """
        Creates the pgd graph to allow for adversarial training
        """
        init_delta = tf.random_uniform(shape = tf.shape(self.x), minval = -eps, maxval = eps)
        delta = tf.Variable(init_delta, name = "delta", validate_shape = False)
        x_tilde = self.x + delta

        #New predictions and loss - call to model will reuse learned weights
        activations, predictions = model(x_tilde, self.hidden_sizes, self.num_classes, self.sigma)
        loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=self.y)
        loss_tilde = tf.reduce_mean(loss_vector)

        #Gradient step, project step and then assign
        optimization_step = tf.assign(delta, tf.squeeze(tf.clip_by_value(delta + eta * tf.math.sign(tf.gradients(loss_tilde, delta)), clip_value_min = -eps, clip_value_max = eps)))

        #sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        return optimization_step, x_tilde, loss_tilde, delta

    def pgd_optimizer(self, sess, X, y, optimization_step, num_iter, loss, delta, last = False, featurized_X = None):
        """
        Runs the pgd optimization step num_iter times
        """
        feed_dict = None
        if not last:
            feed_dict = {self.x: X, self.y: y}
            sess.run(tf.initialize_variables([delta]), feed_dict = feed_dict)
        else:
            feed_dict = {self.x : X, self.featurizations: featurized_X, self.y: y}
            sess.run(tf.initialize_variables([delta]), feed_dict = feed_dict)
            feed_dict = {self.featurizations: featurized_X, self.y: y}

        for i in range(num_iter):
            print("iteration: %d"%i)
            sess.run(optimization_step, feed_dict = feed_dict)
            loss_adv = sess.run(loss, feed_dict = feed_dict)
            print("loss %f" %loss_adv)
        return True

    def pgd_adam(self, sess, X, y, eps, eta, num_iter, scope_name):
        """
        TODO: This function seems kind of useless. Remove it later
        """
        optimization_step, x_tilde, loss, delta = self.pgd_create_adv_graph(sess, X, y, eps, eta, scope = "test")
        success = self.pgd_optimizer(sess, X, y, optimization_step, num_iter, loss, delta)
        return x_tilde

    def pgd_adam_np(self, sess, x, y, eps, eta, num_iter, scope_name = "Test"):
        x_tilde = self.pgd_adam(sess, x, y, eps, eta, num_iter, scope_name)
        feed_dict = {self.x: x, self.y: y}
        x_tilde_np = sess.run(x_tilde, feed_dict = feed_dict)
        diff = x_tilde_np - x
        self.logger.debug("This is to confirm that attack does not violate constraints")
        self.logger.debug("Should be no more than eps")
        self.logger.debug(np.max(np.abs(diff)))
        return x_tilde_np

    def attack_featurization_space(self, sess, X, y, eps, eta, num_iter):
        featurized_X = self.get_featurizations(sess, X)
        self.logger.debug("Obtained featurizations")
        #Solve the optimization problem
        featurized_X_adv_np = self.attack_featurization_space_helper(sess, X, featurized_X, y, eps, eta, num_iter)
        self.logger.debug("Found adversarial featurizations")
        return featurized_X_adv_np

    def attack_featurization_space_helper(self, sess, X, featurized_X, y, eps, eta, num_iter):
        optimization_step, featurization_perturbed, loss, delta = self.pgd_feat_create_adv_graph(sess, featurized_X, y, eps, eta)
        self.logger.debug("Created feat adv graph")
        success = self.pgd_optimizer(sess, X, y, optimization_step, num_iter, loss, delta, last = True, featurized_X=featurized_X)
        feed_dict = {self.featurizations: featurized_X, self.y: y}
        featurization_perturbed_np = sess.run(featurization_perturbed, feed_dict)
        return featurization_perturbed_np

    def pgd_feat_create_adv_graph(self, sess, feats, y, eps, eta):
        """
        Creates the pgd graph for perturbation in featurization space
        """
        init_delta = tf.random_uniform(shape = tf.shape(self.featurizations), minval = -eps, maxval = eps)
        delta = tf.Variable(init_delta, name = "delta_feat", validate_shape = False)

        #Define new featurizations
        featurizations_perturbed = self.featurizations + delta

        #Find predictions and new loss
        initial = tf.contrib.layers.xavier_initializer(dtype = tf.float32)
        bias_initial = tf.initializers.zeros

        scope = 'fc_' + str(len(self.hidden_sizes))
        predictions = fully_connected_layer(featurizations_perturbed, self.num_classes, scope, initial, bias_initial, tf.identity)

        loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=self.y)
        loss_tilde = tf.reduce_mean(loss_vector)

        optimization_step = tf.assign(delta, tf.squeeze(tf.clip_by_value(delta + eta * tf.math.sign(tf.gradients(loss_tilde, delta)), clip_value_min = -eps, clip_value_max = eps)))

        return optimization_step, featurizations_perturbed, loss_tilde, delta

    def sample_attack(self, eps, num_samples = 100):
        """
        Returns only x_adv in the interest of generalizable code
        """
        #Repeat x num_sample times
        n, d = self.x.shape
        x_ext = tf.keras.backend.repeat(self.x, num_samples)
        big_shape = tf.shape(x_ext)
        x_ext = tf.reshape(x_ext, [-1, d])
        n, num_classes = self.y.shape
        y_ext = tf.keras.backend.repeat(self.y, num_samples)
        y_ext = tf.reshape(y_ext, [-1, num_classes])

        #Perturb x_ext
        x_pert = x_ext + tf.random.uniform(tf.shape(x_ext), minval = -eps, maxval = eps)

        #Get loss for x_pert
        activations, predictions = model(x_pert, self.hidden_sizes, self.num_classes, self.sigma)
        loss_vector_ext = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y_ext)

        #Reshape into desired shapes
        loss_vector_ext = tf.reshape(loss_vector_ext, [-1, num_samples])
        x_three_dim = tf.reshape(x_ext, big_shape)

        #Perform argmax to get indices
        best_indices = tf.argmax(loss_vector_ext, axis = 1, output_type = tf.dtypes.int32)
        n = tf.shape(self.x)[0]
        row_idx = tf.range(n)
        extract_idx = tf.stack([row_idx, best_indices], axis = 1)

        #Return X_adv, loss_adv, acc_adv
        x_adv = tf.gather_nd(x_three_dim, extract_idx)

        #Sample a bunch of points around X
        return x_adv

    def sample_attack_np(self, sess, X, y, eps):
        x_adv = self.sample_attack(eps)
        feed_dict = {
            self.x : X,
            self.y : y
        }
        x_adv_conc = sess.run(x_adv, feed_dict = feed_dict)
        return x_adv_conc

    #TODO: Implement option to get distance for PGD points as well
    def get_distance(self, sess, eps, x_test, y_test, order = float("inf")):
        """
        Getter method to get distances between regular and adversarial points
        at each layer
        x_test: regular test set
        x_test_adv: adversarial test set
        order: order of the distance norm - either 2 or float("inf")
        """
        self.logger.info("Getting activation distances between regular and FGSM data")
        x_test_adv = self.fgsm_np(sess, x_test, y_test, eps)

        pred = self.get_prediction(sess, x_test_adv)

        self.logger.debug("Found correct and incorrect indices")
        incorrect_indices = [i for i,v in enumerate(pred) if np.argmax(v) != np.argmax(y_test[i])]
        correct_indices = [i for i,v in enumerate(pred) if np.argmax(v) == np.argmax(y_test[i])]

        overall, overall_std  = self.dist_average(sess, x_test, x_test_adv, order = order)
        correct_dist, correct_std = self.dist_average(sess, x_test[correct_indices], x_test_adv[correct_indices], order)
        incorrect_dist, incorrect_std = self.dist_average(sess, x_test[incorrect_indices], x_test_adv[incorrect_indices], order)

        return overall, overall_std, correct_dist, correct_std, incorrect_dist, incorrect_std

    def dist_average(self, sess, x_test, x_test_adv, order):
        distances = []
        for i in range(len(x_test)):
            dist = self.dist_calculator(sess, x_test[i], x_test_adv[i], order)
            distances.append(dist)
        self.logger.debug("Found average and standard deviation of distances")
        return np.average(distances, axis = 0), np.std(distances, axis = 0)

    def dist_calculator(self,sess, x, x_adv, order):
        reg_image = x.reshape(1, 784)
        adv_image = x_adv.reshape(1, 784)

        activation_reg = np.array(self.get_activation(sess, reg_image))
        activation_adv = np.array(self.get_activation(sess, adv_image))

        difference = activation_reg - activation_adv
        difference = difference.squeeze()
        L = len(difference)
        if L == 1:
                return np.linalg.norm(difference, ord = order)
        else:
            distances = []
            for i in range(L):
                    distances.append(np.linalg.norm(difference[i], ord = order))
            return distances

    def evaluate(self, sess, X, y):
        loss, accuracy = sess.run([self.loss, self.accuracy],
                                 feed_dict = {
                                     self.x : X,
                                     self.y: y
                                 })
        self.logger.info("Model was evaluated on benign data")
        return loss, accuracy

    def adv_evaluate(self, sess, X, y, eps, pgd = False, eta = 1e-2, num_iter = 500):
        X_adv = None
        if not pgd:
            X_adv = self.fgsm_np(sess, X, y, eps)
            self.logger.info("Model is being evaluated on FGSM data")
        else:
            self.logger.info("Model is being evaluated on PGD points generated using %f learning rate and %d iterations" %(eta, num_iter))
            X_adv = self.pgd_adam_np(sess, X, y, eps, eta, num_iter)

        loss, accuracy = sess.run([self.loss, self.accuracy],
                                 feed_dict = {
                                     self.x : X_adv,
                                     self.y: y
                                 })
        return loss, accuracy

    def evaluate_from_featurizations(self, sess, featurizations, y):
        """
        Evaluates loss and accuracy of model
        But instead of taking X as input,
        it takes phi(X) as input
        """
        feed_dict = {self.featurizations: featurizations, self.y: y}
        loss, acc = sess.run([self.loss, self.accuracy], feed_dict = feed_dict)
        self.logger.info("Model was evaluated from featurizations")
        return loss, acc

    def fit_helper(self, sess, X, y, optimizer, loss, accuracy, lr = 0.003, training_epochs=15, batch_size=32, display_step=1, pgd = False, eps_train = 0.1, whac_a_mole = False, num_neurons_whac = 0, freq_whac_per_epoch = 0):
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(X) / batch_size)
            whac_indices = None
            if whac_a_mole:
                whac_indices = np.linspace(0, total_batch, freq_whac_per_epoch)
                whac_indices = [int(e) for e in whac_indices]
                whac_indices = set(whac_indices)

            x_batches = np.array_split(X, total_batch)
            y_batches = np.array_split(y, total_batch)

            for i in range(total_batch):
                if whac_a_mole and (i in whac_indices):
                    #Remove this hardcode later on
                    X_train_truncated = X[0:10000]
                    neuron_score = self.get_dphi_dx(sess, X_train_truncated)
                    ns = np.sum(neuron_score, axis = 0)
                    ns = np.sum(ns, axis = 1)
                    top_k_indices = np.argsort(-ns)[:num_neurons_whac]
                    self.whac_neuron_k(sess, top_k_indices)
                    self.logger.debug("Whac-ed %d neurons!"%num_neurons_whac)

                batch_x, batch_y = x_batches[i], y_batches[i]
                if pgd:
                    it_num = epoch * total_batch + i
                    sc_name = "train_" + str(it_num)
                    batch_x = self.pgd_adam_np(sess, batch_x, batch_y, eps = eps_train, eta = 5e-1, num_iter = 10, scope_name = sc_name)
                _, c, acc = sess.run([optimizer, loss, accuracy],
                                     feed_dict={
                                         self.x: batch_x,
                                         self.y: batch_y,
                                     })
                avg_cost += c / total_batch
                if i % 100 == 0:
                    feed_dict = {self.x: x_batches[0], self.y: y_batches[0]}
                    summary = sess.run(self.merged_summary, feed_dict = feed_dict)
                    hor = epoch * total_batch + i
                    self.writer.add_summary(summary, hor)

            if epoch % display_step == 0:
                #self.logger.debug("Epoch:", '%04d' % (epoch+1), "cost=", \
                        #"{:.9f}".format(avg_cost))
                self.logger.debug("Epoch: %04d    cost: %.9f " %(epoch+1, avg_cost))
                self.logger.debug("Accuracy on batch: %f" %acc)
        self.logger.debug("Optimization Finished!")

        final_acc, final_loss = sess.run([self.accuracy, self.loss],
                                         feed_dict={
                                             self.x: X,
                                             self.y: y,
                                         }
                                        )
        self.logger.debug("Final Train Loss %f" %final_loss)
        self.logger.debug("Final Train Accuracy %f:" %final_acc)
        return True

    def fit(self, sess, X, y, lr = 0.003, training_epochs=15, batch_size=32, display_step=1, reg_op = 0, reg_trace_first = 0, reg_trace_all = 0, reg_l1 = 0):

        loss = self.loss
        loss += reg_op*regularize_op_norm(self.get_weights()[0])
        loss += reg_trace_all*regularize_trace_norm(self.get_weights()[0])
        loss += reg_l1*regularize_l1_norm(self.get_weights()[0])
        loss += reg_trace_first*regularize_trace_norm_first(self.get_weights()[0])
        #loss += reg*regularize_lipschitz_norm(self.get_weights()[0])

        temp = set(tf.all_variables())
        optimization_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        self.fit_helper(sess, X, y, optimization_step, loss,
            self.accuracy, lr, training_epochs, batch_size, display_step)
        self.logger.info("Model was trained on benign data")
        return True

    def fit_whac_a_mole(self, sess, X, y, lr = 0.003, training_epochs=15, batch_size=32, display_step=1, reg_op = 0, reg_trace_first = 0, reg_trace_all = 0, reg_l1 = 0, num_neurons_whac = 1, freq_whac_per_epoch = 5):

        loss = self.loss
        loss += reg_op*regularize_op_norm(self.get_weights()[0])
        loss += reg_trace_all*regularize_trace_norm(self.get_weights()[0])
        loss += reg_l1*regularize_l1_norm(self.get_weights()[0])
        loss += reg_trace_first*regularize_trace_norm_first(self.get_weights()[0])
        #loss += reg*regularize_lipschitz_norm(self.get_weights()[0])

        temp = set(tf.all_variables())
        optimization_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        self.fit_helper(sess, X, y, optimization_step, loss,
            self.accuracy, lr, training_epochs, batch_size, display_step, whac_a_mole=True, num_neurons_whac=num_neurons_whac, freq_whac_per_epoch=freq_whac_per_epoch)
        self.logger.info("Model was trained on benign data")
        return True

    def adv_fit(self, sess, X, y, eps, lr = 3e-4, training_epochs=15, batch_size=32, display_step=1):
        x_adv = self.fgsm(self.x, eps)
        _, predictions = model(x_adv, self.hidden_sizes, self.num_classes, self.sigma)

        loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=self.y)
        loss_adv = tf.reduce_mean(loss_vector)
        tf.summary.scalar("loss", loss_adv)

        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(self.y, 1))
        accuracy_adv = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        temp = set(tf.all_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_adv)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        self.fit_helper(sess, X, y, optimizer, loss_adv, accuracy_adv,
        lr, training_epochs, batch_size, display_step)

        X_adv = self.fgsm_np(sess, X, y, eps)
        final_acc_adv, final_loss_adv = sess.run([self.accuracy, self.loss],
                                         feed_dict={
                                             self.x: X_adv,
                                             self.y: y,
                                         }
                                        )
        self.logger.debug("Final Train Loss on Adv points %f" %final_loss_adv)
        self.logger.debug("Final Train Accuracy on Adv points %f" %final_acc_adv)
        self.logger.info("Model was trained on adversarial data")
        return True

    def pgd_fit(self, sess, X, y, eps, eta, num_iter_pgd, lr = 0.003, training_epochs=40, batch_size=32, display_step=1, reg = 0.005, early_stop_flag = True, early_stop_threshold = 0.02):
        #Good optimization
        loss = self.loss + reg*regularize_op_norm(self.get_weights()[0])
        temp = set(tf.all_variables())
        optimization_good = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        #batch_size = len(X)

        #Create the pgd graph which we can optimize below
        optimization_pgd, x_tilde, loss_pgd, delta = self.pgd_create_adv_graph(sess, X, y, eps, eta, scope = "train")
        avg_cost_old = np.float("inf")

        #Alternating optimization
        for epoch in range(training_epochs):
            avg_cost = 0.0

            #PGD optimization
            success = self.pgd_optimizer(sess, X, y, optimization_pgd, num_iter_pgd, loss_pgd, delta)
            feed_dict = {self.x: X, self.y: y}
            X_adv = sess.run(x_tilde, feed_dict = feed_dict)

            total_batch = int(len(X_adv) / batch_size)
            x_batches = np.array_split(X_adv, total_batch)
            y_batches = np.array_split(y, total_batch)


            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                #Actual optimization
                _, c, acc = sess.run([optimization_good, loss, self.accuracy],
                                     feed_dict={
                                         self.x: batch_x,
                                         self.y: batch_y,
                                     })
                avg_cost += c / total_batch
                if i % 100 == 0:
                    feed_dict = {self.x: x_batches[0], self.y: y_batches[0]}
                    summary = sess.run(self.merged_summary, feed_dict = feed_dict)
                    hor = epoch * total_batch + i
                    self.writer.add_summary(summary, hor)

            #Early Stopping Criterion
            if early_stop_flag and np.abs(avg_cost - avg_cost_old) < early_stop_threshold:
                self.logger.info("Hit the early stopping criterion, and stopping training")
                break

            avg_cost_old = avg_cost

            if epoch % display_step == 0:
                self.logger.debug("Epoch: %04d    cost: %.9f " %(epoch+1, avg_cost))
                self.logger.debug("Accuracy on batch: %f" %acc)
        self.logger.debug("Optimization Finished!")

        final_acc, final_loss = sess.run([self.accuracy, self.loss],
                                         feed_dict={
                                             self.x: X,
                                             self.y: y,
                                         }
                                        )
        self.logger.debug("Final Train Loss %f" %final_loss)
        self.logger.debug("Final Train Accuracy %f:" %final_acc)
        return True

    def whac_neuron_k(self, sess, k_lst):
        weights = []
        assign_ops = []

        scope_name = 'fc_' + str(len(self.hidden_sizes) - 1)
        with tf.variable_scope(scope_name, reuse = True):
            w = tf.get_variable('weights')
            w_np = sess.run(w)
            for k in k_lst:
                w_np[:, k] = 0
            assign_op = tf.assign(w, w_np)

        sess.run(assign_op)
        return True

    def slash_weights(self, sess, factor):
        assign_ops = []
        for i in range(len(self.hidden_sizes) + 1):
            scope_name = 'fc_' + str(i)
            with tf.variable_scope(scope_name, reuse = True):
                w = tf.get_variable('weights')
                assign_op = tf.assign(w, w * factor)
                assign_ops.append(assign_op)

        sess.run(assign_ops)
        weights_np = sess.run(weights)
        return weights_np

    def visualize_activation_tsne(self, sess, x_input, metadata_path, sprite_path, LOG_DIR, imgh = 28, imgw = 28):
        """
        Activations: To be visualized
        Metadata: Labels/Images
        Logdir: Where to save the activations

        How this works:
        1. Save the data to be visualized to a log file.
        2. Create a projector object that looks at the log file and creates summaries for tensorboard.

        """
        activations_all = self.get_activation(sess, x_input)
        config = projector.ProjectorConfig()
        var_list = []

        for layer_number in range(len(self.activations)):
            activation = activations_all[layer_number]
            tf_data = tf.Variable(activation)
            sess.run(tf_data.initializer)
            var_list.append(tf_data)

            embedding = config.embeddings.add()
            embedding.tensor_name = tf_data.name
            embedding.metadata_path = metadata_path
            embedding.sprite.image_path = sprite_path
            embedding.sprite.single_image_dim.extend([imgh, imgw])


        saver = tf.train.Saver(var_list)
        saver.save(sess, os.path.join(LOG_DIR, 'activations.ckpt'))

        projector.visualize_embeddings(self.writer, config)
        self.logger.info("Created embeddings for visualization")
        return True
