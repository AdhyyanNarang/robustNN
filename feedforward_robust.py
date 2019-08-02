import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.tensorboard.plugins import projector
import ipdb
from util import *

"""
Reorganization:
Add documentation for all functions
Create a new function to get loss/acc from predictions
"""

class RobustMLP(object):

    def __init__(self,session,input_shape,hidden_sizes,num_classes, dataset, writer, scope):

        #Initialize instance variables
        self.sess = session
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
        self.writer = writer
        self.scope = scope

        #TODO: Fix this hacky solution.
        input_shape = input_shape[0]

        #Placeholders for the data
        x_shape = [None] + [input_shape]
        self.x = tf.placeholder("float", x_shape)
        self.y = tf.placeholder("float", [None, num_classes])

        self.activations, self.predictions = model(self.x, self.hidden_sizes, self.num_classes)
        self.featurizations = self.activations[-1]

        self.loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.y)
        self.loss = tf.reduce_mean(self.loss_vector)
        tf.summary.scalar("loss", self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        tf.summary.scalar("accuracy", self.accuracy)

        self.merged_summary = tf.summary.merge_all(scope = self.scope)
        self.sess.run(tf.global_variables_initializer())

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

    def get_weights_np(self, sess):
        weights = []
        for i in range(len(self.hidden_sizes) + 1):
            scope_name = 'fc_' + str(i)
            with tf.variable_scope(scope_name, reuse = True):
                w = tf.get_variable('weights')
                weights.append(w)

        weights_np = sess.run(weights)
        return weights_np

    def get_weight_norms(self, sess, matrix_norm_fxn = lambda x: np.linalg.norm(x.T, ord = 1)):
        """
        Getter method for the norms of the weight matrices of
        the network
        """
        model_norms = []
        weights_list = self.get_weights_np(sess)
        for weights in weights_list:
            norm = matrix_norm_fxn(weights[0])
            model_norms.append(norm)
        return model_norms

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
        activations, predictions = model(x_pert, self.hidden_sizes, self.num_classes)
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

    def get_distance(self, sess, eps, x_test, y_test, order = float("inf")):
        """
        Getter method to get distances between regular and adversarial points
        at each layer
        x_test: regular test set
        x_test_adv: adversarial test set
        order: order of the distance norm - either 2 or float("inf")
        """
        x_test_adv = self.fgsm_np(sess, x_test, y_test, eps)

        pred = self.get_prediction(sess, x_test_adv)

        incorrect_indices = [i for i,v in enumerate(pred) if np.argmax(v) != np.argmax(y_test[i])]
        correct_indices = [i for i,v in enumerate(pred) if np.argmax(v) == np.argmax(y_test[i])]

        overall = self.dist_average(sess, x_test, x_test_adv, order = order)
        correct_dist = self.dist_average(sess, x_test[correct_indices], x_test_adv[correct_indices], order)
        incorrect_dist = self.dist_average(sess, x_test[incorrect_indices], x_test_adv[incorrect_indices], order)

        return overall, correct_dist, incorrect_dist

    def dist_average(self, sess, x_test, x_test_adv, order):
        distances = []
        for i in range(len(x_test)):
            dist = self.dist_calculator(sess, x_test[i], x_test_adv[i], order)
            distances.append(dist)
        return np.average(distances, axis = 0)

    def dist_calculator(self,sess, x, x_adv, order):
        reg_image = x.reshape(1, 784)
        adv_image = x_adv.reshape(1, 784)

        activation_reg = np.array(self.get_activation(sess, reg_image))
        activation_adv = np.array(self.get_activation(sess, adv_image))

        difference = activation_reg - activation_adv
        L = len(difference)
        if L == 1:
                return np.linalg.norm(activation_reg - activation_adv, ord = order)
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
        return loss, accuracy

    def adv_evaluate(self, sess, X, y, eps):
        X_adv = self.fgsm_np(sess, X, y, eps)
        loss, accuracy = sess.run([self.loss, self.accuracy],
                                 feed_dict = {
                                     self.x : X_adv,
                                     self.y: y
                                 })
        return loss, accuracy

    def fit_helper(self, sess, X, y, optimizer, loss, accuracy, lr = 0.003, training_epochs=15, batch_size=32, display_step=1):
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(X) / batch_size)
            x_batches = np.array_split(X, total_batch)
            y_batches = np.array_split(y, total_batch)

            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
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
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
                print("Accuracy on batch:", acc)
        print("Optimization Finished!")

        final_acc, final_loss = sess.run([self.accuracy, self.loss],
                                         feed_dict={
                                             self.x: X,
                                             self.y: y,
                                         }
                                        )
        print("Final Train Loss", final_loss)
        print("Final Train Accuracy:", final_acc)
        return True

    def fit(self, sess, X, y, lr = 0.003, training_epochs=15, batch_size=32, display_step=1):
        temp = set(tf.all_variables())
        optimization_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        self.fit_helper(sess, X, y, optimization_step, self.loss,
            self.accuracy, lr, training_epochs, batch_size, display_step)
        return True

    def adv_fit(self, sess, X, y, eps, lr = 3e-4, training_epochs=15, batch_size=32, display_step=1):
        x_adv = self.fgsm(self.x, eps)
        _, predictions = model(x_adv, self.hidden_sizes, self.num_classes)

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
        print("Final Train Loss on Adv points", final_loss_adv)
        print("Final Train Accuracy on Adv points", final_acc_adv)
        return True

    def fit_robust_final_layer(self, sess, X, y, eps, feed_features_flag = False, lr = 0.001, training_epochs=15, batch_size=32, display_step=1):
        #Generate new loss and accuracy variables
        #x_adv = self.sample_attack(eps)
        x_adv = self.fgsm(self.x, eps)
        activations, predictions = model(x_adv, self.hidden_sizes, self.num_classes)
        loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=self.y)
        loss_adv = tf.reduce_mean(loss_vector)
        tf.summary.scalar("loss", loss_adv)
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(self.y, 1))
        accuracy_adv = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #Create var list
        var_list = []
        scope_name = 'fc_' + str(len(self.hidden_sizes))
        with tf.variable_scope(scope_name, reuse = True ) as scope:
            w = tf.get_variable('weights')
            b = tf.get_variable('biases')
            var_list.append(w)
            var_list.append(b)

        #Crete optimizer variable
        temp = set(tf.all_variables())
        #optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        objective = 0.9*loss_adv + 0.1*self.loss
        optimization_update = optimizer.minimize(objective, var_list = var_list)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        self.fit_helper(sess, X, y, optimization_update, loss_adv, accuracy_adv, feed_features_flag, lr, training_epochs, batch_size, display_step)
        return True

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
        return True
