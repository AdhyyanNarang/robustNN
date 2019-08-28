import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.tensorboard.plugins import projector
import ipdb
sys.path.append("util/")
from util import *

"""
Reorganization:
Add documentation for all functions
Create a new function to get loss/acc from predictions
"""

class RobustMLP(object):

    def __init__(self,input_shape,hidden_sizes,num_classes, writer, scope, logger, sigma):

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

        self.loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.y)
        self.loss = tf.reduce_mean(self.loss_vector)
        tf.summary.scalar("loss", self.loss)
        self.logger.debug("Added loss computation to the graph")

        self.correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.logger.debug("Added accuracy computation to the graph")
        tf.summary.scalar("accuracy", self.accuracy)

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

    def get_dphi_dx(self, sess, x_train):
        """
        Gets gradients of features with respect to input
        """
        dphi_dx = tf.gradients(self.featurizations, self.x)
        feed_dict = {self.x: x_train}
        dphi_dx_np = sess.run(dphi_dx, feed_dict = feed_dict)
        return dphi_dx_np

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
        temp = set(tf.all_variables())

        #TODO:This hack needs to change to accept variable shape
        x_ph = tf.placeholder("float", X.shape)
        delta = tf.get_variable("delta", shape = X.shape, initializer = tf.initializers.zeros(dtype = tf.float32))
        x_tilde = x_ph + delta
        y_ph = tf.placeholder("float", y.shape)

        #New predictions and loss - call to model will reuse learned weights
        activations, predictions = model(x_tilde, self.hidden_sizes, self.num_classes, self.sigma)
        loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y_ph)
        loss_tilde = tf.reduce_mean(loss_vector)

        #Optimization
        zeros_delta = tf.zeros_like(delta)
        delta_zero_assign_op = tf.assign(delta, zeros_delta)
        optimization_step = tf.train.AdamOptimizer(learning_rate = eta).minimize(-loss_tilde, var_list = [delta])
        tmp = tf.clip_by_value(delta, clip_value_min = -eps, clip_value_max = eps)
        project_op = tf.assign(delta, tmp)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        return x_ph, y_ph, optimization_step, project_op, x_tilde, delta_zero_assign_op, loss_tilde

    def pgd_optimizer(self, sess, X, y, x_ph, y_ph, optimization_step, project_op, assign_op, num_iter, loss):
        sess.run(assign_op)
        for i in range(num_iter):
            print("iteration: %d"%i)
            feed_dict = {x_ph: X, y_ph: y}
            _, _, loss_adv = sess.run([optimization_step, project_op, loss], feed_dict = feed_dict)
            print("loss %f" %loss_adv)
        return True

    def pgd_adam(self, sess, X, y, eps, eta, num_iter, scope_name):
        x_ph, y_ph, optimization_step, project_op, x_tilde, assign_op, loss = self.pgd_create_adv_graph(sess, X, y, eps, eta, scope = "test")
        success = self.pgd_optimizer(sess, X, y, x_ph, y_ph, optimization_step, project_op, assign_op, num_iter, loss)
        return x_tilde, x_ph, y_ph

    def pgd_adam_np(self, sess, x, y, eps, eta, num_iter, scope_name = "Test"):
        x_tilde, x_ph, y_ph = self.pgd_adam(sess, x, y, eps, eta, num_iter, scope_name)
        feed_dict = {x_ph : x, y_ph: y}
        x_tilde_np = sess.run(x_tilde, feed_dict = feed_dict)
        return x_tilde_np

    def sample_attack(self, eps, num_samples = 100):
        """
        Returns only x_adv in the interest of generalizable code
        """
        #Repeat x num_sample times
        n, d = self.x.shape
        #ipdb.set_trace()
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

    def fit_helper(self, sess, X, y, optimizer, loss, accuracy, lr = 0.003, training_epochs=15, batch_size=32, display_step=1, pgd = False, eps_train = 0.1):
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(X) / batch_size)
            x_batches = np.array_split(X, total_batch)
            y_batches = np.array_split(y, total_batch)

            for i in range(total_batch):
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

    def fit(self, sess, X, y, lr = 0.003, training_epochs=15, batch_size=32, display_step=1, reg = 0.005):

        #loss = self.loss + reg*regularize_op_norm(self.get_weights()[0])
        loss = self.loss + reg*regularize_trace_norm(self.get_weights()[0])
        temp = set(tf.all_variables())
        optimization_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        self.fit_helper(sess, X, y, optimization_step, loss,
            self.accuracy, lr, training_epochs, batch_size, display_step)
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

    def pgd_fit(self, sess, X, y, eps, eta, num_iter_pgd, lr = 0.003, training_epochs=40, batch_size=32, display_step=1, reg = 0.005):
        #Good optimization
        loss = self.loss + reg*regularize_op_norm(self.get_weights()[0])
        temp = set(tf.all_variables())
        optimization_good = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        batch_size = len(X)

        #Define pgd graph
        #TODO: Fix this hack later
        total_batch = int(len(X) / batch_size)
        x_batches = np.array_split(X, total_batch)
        y_batches = np.array_split(y, total_batch)
        x_ph, y_ph, optimization_pgd, project_op, x_tilde, zeros_assign_op, _ = self.pgd_create_adv_graph(sess, x_batches[0], y_batches[0], eps, eta, scope = "train")

        #Alternating optimization
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(X) / batch_size)
            x_batches = np.array_split(X, total_batch)
            y_batches = np.array_split(y, total_batch)

            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                #PGD optimization
                success = self.pgd_optimizer(sess, batch_x, batch_y, x_ph, y_ph, optimization_pgd, project_op, zeros_assign_op, num_iter_pgd)
                feed_dict = {x_ph : batch_x, y_ph: batch_y}
                batch_x_pgd = sess.run(x_tilde, feed_dict = feed_dict)

                #Actual optimization
                _, c, acc = sess.run([optimization_good, loss, self.accuracy],
                                     feed_dict={
                                         self.x: batch_x_pgd,
                                         self.y: batch_y,
                                     })
                avg_cost += c / total_batch
                if i % 100 == 0:
                    feed_dict = {self.x: x_batches[0], self.y: y_batches[0]}
                    summary = sess.run(self.merged_summary, feed_dict = feed_dict)
                    hor = epoch * total_batch + i
                    self.writer.add_summary(summary, hor)

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


        return True


    def slash_weights(self, sess):
        weights = []
        assign_ops = []
        for i in range(len(self.hidden_sizes) + 1):
            scope_name = 'fc_' + str(i)
            with tf.variable_scope(scope_name, reuse = True):
                w = tf.get_variable('weights')
                weights.append(w)
                assign_op = tf.assign(w, w * 0.9)
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
