import os, random, time, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import ipdb

def fully_connected_layer(x, output_dim, scope_name, weight_initializer, bias_initializer, sigma):
    with tf.variable_scope(scope_name) as scope:
        weight_shape = (x.shape[1], output_dim)
        print(weight_shape)
        w = tf.get_variable('weights', shape = weight_shape, initializer = weight_initializer)
        b = tf.get_variable('biases', output_dim, initializer=bias_initializer)
        z = tf.add(tf.matmul(x, w), b)
        a = sigma(z)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", a)
        return a

class RobustMLP(object):

    def __init__(self,session,input_shape,hidden_sizes,num_classes, dataset, writer):

        #Initialize instance variables
        self.sess = session
        self.input_shape = input_shape
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
        self.writer = writer

        #TODO: Fix this hacky solution.
        input_shape = input_shape[0]

        initial = tf.contrib.layers.xavier_initializer(dtype = tf.float32)
        bias_initial = tf.initializers.zeros

        #Placeholders for the data
        x_shape = [None] + [input_shape]
        self.x = tf.placeholder("float", x_shape)
        self.y = tf.placeholder("float", [None, num_classes])
        act = self.x
        self.activations = []

        #Compute the prediction placeholder
        for i in range(len(self.hidden_sizes)):
            scope = 'fc_' + str(i)
            act = fully_connected_layer(act, self.hidden_sizes[i], scope, initial, bias_initial, tf.nn.relu)
            self.activations.append(act)


        #Save featurizations and predictions as instance vars
        self.featurizations = act
        scope = 'fc_' + str(len(self.hidden_sizes))
        self.predictions = fully_connected_layer(act, self.num_classes, scope, initial, bias_initial, tf.identity)

        self.loss_vector = tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.y)
        self.loss = tf.reduce_mean(self.loss_vector)
        tf.summary.scalar("loss", self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        tf.summary.scalar("accuracy", self.accuracy)

        self.merged_summary = tf.summary.merge_all()
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

    def fgsm(self, sess, X, y, eps):
        g = tf.gradients(self.loss, self.x)
        delta = eps * tf.math.sign(g)
        x_adv = self.x + delta
        feed_dict = {
            self.x : X,
            self.y : y
        }
        x_adv_conc = sess.run(x_adv, feed_dict = feed_dict).squeeze()
        return x_adv_conc

    def evaluate(self, sess, X, y):
        loss, accuracy = sess.run([self.loss, self.accuracy],
                                 feed_dict = {
                                     self.x : X,
                                     self.y: y
                                 })
        return loss, accuracy

    def adv_evaluate(self, sess, X, y, eps):
        X_adv = self.fgsm(sess, X, y, eps)
        loss, accuracy = sess.run([self.loss, self.accuracy],
                                 feed_dict = {
                                     self.x : X_adv,
                                     self.y: y
                                 })
        return loss, accuracy


    def fit(self, sess, X, y, lr = 0.003, training_epochs=15, batch_size=32, display_step=1):
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
                if i % 100 == 0:
                    feed_dict = {self.x: x_batches[0], self.y: y_batches[0]}
                    summary = sess.run(self.merged_summary, feed_dict = feed_dict)
                    self.writer.add_summary(summary, i)

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

    def adv_fit(self, sess, X, y, eps, lr = 0.003, training_epochs=15, batch_size=32, display_step=1):
        temp = set(tf.all_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))


        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(X) / batch_size)
            x_batches = np.array_split(X, total_batch)
            y_batches = np.array_split(y, total_batch)

            for i in range(total_batch):
                if i % 10 == 0:
                    print(i)
                batch_x, batch_y = x_batches[i], y_batches[i]

                #Train on the adversarial points instead of regular points.
                batch_x_adv = self.fgsm(sess, batch_x, batch_y, eps)
                _, c, acc = sess.run([optimizer, self.loss, self.accuracy],
                                     feed_dict={
                                         self.x: batch_x_adv,
                                         self.y: batch_y
                                     })
                avg_cost += c / total_batch
                """
                if i % 100 == 0:
                    feed_dict = {self.x: x_batches[0], self.y: y_batches[0]}
                    summary = sess.run(self.merged_summary, feed_dict = feed_dict)
                    self.writer.add_summary(summary, i)
                """

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
        print("Final Train Loss on Regular points", final_loss)
        print("Final Train Accuracy on regular points", final_acc)

        X_adv = self.fgsm(sess, X, y, eps)
        final_acc_adv, final_loss_adv = sess.run([self.accuracy, self.loss],
                                         feed_dict={
                                             self.x: X_adv,
                                             self.y: y,
                                         }
                                        )
        print("Final Train Loss on Adv points", final_loss_adv)
        print("Final Train Accuracy on regular points", final_acc_adv)
        return True



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
