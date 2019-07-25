import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import ipdb

def fully_connected_layer(x, output_dim, scope, weight_initializer, bias_initializer, sigma):
    #with tf.variable_scope(scope) as scope:
    w = tf.get_variable('weights' + scope, [x.shape[1], output_dim], initializer = weight_initializer)
    b = tf.get_variable('biases' + scope, [output_dim], initializer=bias_initializer)
    z = tf.add(tf.matmul(x, w), b)
    return sigma(z)

class RobustMLP(object):

    def __init__(self,session,input_shape,hidden_sizes,num_classes, dataset):

        #Initialize instance variables
        self.sess = session
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes + [num_classes]
        self.num_classes = num_classes
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset

        #TODO: Fix this hacky solution.
        input_shape = input_shape[0]

        """
        #Initialize weights and bias optimization variables
        shape = (input_shape, self.hidden_sizes[0])
        initial = tf.contrib.layers.xavier_initializer(dtype = tf.float32)
        self.weights = {}
        self.biases = {}
        self.weights['h1'] = tf.get_variable('h1', shape = shape, initializer=initial)
        self.biases['b1'] = tf.get_variable('b1', shape = hidden_sizes[0], initializer = tf.initializers.zeros)

        for i in range(len(hidden_sizes) - 1):
            key_number = i + 2
            weight_key, bias_key = 'h' + str(key_number), 'b' + str(key_number)
            weight_shape = (hidden_sizes[i], hidden_sizes[i+1])
            self.weights[weight_key] = tf.get_variable(weight_key, shape=weight_shape, initializer=initial)
            self.biases[bias_key] = tf.get_variable(bias_key, shape = hidden_sizes[i+1], initializer = tf.initializers.zeros)

        self.weights['out'] = tf.get_variable('hout', shape = (hidden_sizes[-1], num_classes), initializer=initial)
        self.biases['out'] = tf.get_variable('bout', shape = num_classes, initializer = tf.initializers.zeros)
        """

        initial = tf.contrib.layers.xavier_initializer(dtype = tf.float32)
        bias_initial = tf.initializers.zeros

        #Placeholders for the data
        x_shape = [None] + [input_shape]
        self.x = tf.placeholder("float", x_shape)
        self.y = tf.placeholder("float", [None, num_classes])
        act = self.x
        self.activations = []

        for i in range(len(self.hidden_sizes)):
            scope = 'fc_' + str(i)
            act = fully_connected_layer(act, self.hidden_sizes[i], scope, initial, bias_initial, tf.nn.relu)
            self.activations.append(act)

        #Compute the prediction placeholder
        """
        self.activations = []
        layer_next = self.x
        for i in range(len(hidden_sizes)):
            key_number = i + 1
            weight_key, bias_key = 'h' + str(key_number), 'b' + str(key_number)
            layer_next = tf.add(tf.matmul(layer_next, self.weights[weight_key]), self.biases[bias_key])
            layer_next = tf.nn.relu(layer_next)
            self.activations.append(layer_next)
        """
        #Save featurizations and predictions as instance vars
        self.featurizations = self.activations[-2]
        self.predictions = self.activations[-1] 

        self.loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.y)
        self.loss = tf.reduce_mean(self.loss_vector)

        self.correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        self.sess.run(tf.global_variables_initializer())

    def get_activation(self, sess, x_input):
        activations = sess.run(self.activations,
                               feed_dict = {
                                   self.x: x_input,
                               })
        return activations

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

    def evaluate(self, sess, X, y):
        loss, accuracy = sess.run([self.loss, self.accuracy],
                                 feed_dict = {
                                     self.x : X,
                                     self.y: y
                                 })
        return loss, accuracy

    def fit(self, sess, X, y, lr = 0.001, training_epochs=15, batch_size=32, display_step=1):
        temp = set(tf.all_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))


        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(X) / batch_size)
            x_batches = np.array_split(X, total_batch)
            y_batches = np.array_split(y, total_batch)

            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                _, c, acc = sess.run([optimizer, self.loss, self.accuracy],
                                     feed_dict={
                                         self.x: batch_x,
                                         self.y: batch_y,
                                     })
                avg_cost += c / total_batch
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

    def fit_robust_final_layer(self, sess, X, y, feed_features_flag = False, lr = 0.001, training_epochs=15, batch_size=32, display_step=1):
        #Hacky solution below - Would be nice to fix it
        temp = set(tf.all_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        var_list = [self.weights['out'], self.biases['out']]
        optimization_update = optimizer.minimize(self.loss, var_list = var_list)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(X) / batch_size)
            x_batches = np.array_split(X, total_batch)
            y_batches = np.array_split(y, total_batch)

            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                feed_dict = {}
                if feed_features_flag:
                    feed_dict = {
                        self.featurizations: batch_x,
                        self.y : batch_y
                    }
                else:
                    feed_dict = {
                        self.x : batch_x,
                        self.y : batch_y
                    }
                _, c, acc = sess.run([optimization_update, self.loss, self.accuracy],
                                     feed_dict= feed_dict
                                     )
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
                print("Accuracy on batch:", acc)
        print("Optimization Finished!")

        feed_dict = {}
        feed_dict[self.y] = y
        if feed_features_flag:
            feed_dict[self.featurizations] = X
        else:
            feed_dict[self.x] = X

        final_acc, final_loss = sess.run([self.accuracy, self.loss],
                                         feed_dict= feed_dict
                                        )
        print("Final Train Loss", final_loss)

        print("Final Train Accuracy:", final_acc)
