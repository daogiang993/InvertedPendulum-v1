import tensorflow as tf
import numpy as np
from constants import *

LR = LR_ACTOR

class Actor:
    def __init__(self, sess, n_state, n_action):
        self.sess = sess
        self.n_state = n_state
        self.n_action = n_action

        self.online_state_in, self.online_out, self.online_net, self.online_is_training = self._create_online()
        self.target_state_in, self.target_out, self.target_update, self.target_is_training = self._create_target()
        self.q_gradient_in, self.parameters_gradients, self.optimizer = self._create_training()

        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def _create_online(self):
        '''Online Actor Network'''
        state_in = tf.placeholder('float', [None, self.n_state])
        is_training = tf.placeholder('bool')

        # Initialization for ReLU
        W1 = tf.Variable(tf.random_uniform([self.n_state, H1_SIZE], -1./np.sqrt(self.n_state), 1/np.sqrt(self.n_state)))
        b1 = tf.Variable(tf.random_uniform([H1_SIZE], -1./np.sqrt(self.n_state), 1/np.sqrt(self.n_state)))
        W2 = tf.Variable(tf.random_uniform([H1_SIZE, H2_SIZE], -1./np.sqrt(H1_SIZE), 1/np.sqrt(H1_SIZE)))
        b2 = tf.Variable(tf.random_uniform([H2_SIZE], -1./np.sqrt(H1_SIZE), 1/np.sqrt(H1_SIZE)))
        W3 = tf.Variable(tf.random_uniform([H2_SIZE, self.n_action], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([self.n_action], -3e-3, 3e-3))

        # Train graph
        in_bn = self._batch_norm(state_in, is_training, 'batch_norm_in')
        h1 = tf.matmul(in_bn, W1) + b1
        h1_bn = self._batch_norm(h1, is_training, 'batch_norm_h1', tf.nn.relu)
        h2 = tf.matmul(h1_bn, W2) + b2
        h2_bn = self._batch_norm(h2, is_training, 'batch_norm_h2', tf.nn.relu)
        online_out = tf.tanh(tf.matmul(h2_bn, W3) + b3)
        return state_in, online_out, [W1, b1, W2, b2, W3, b3], is_training

    def _create_target(self):
        '''Target Actor Network'''
        state_in = tf.placeholder('float', [None, self.n_state])
        is_training = tf.placeholder('bool')
        # target_new = (1-TAU)*target_old + TAU*online
        _train = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = _train.apply(self.online_net)
        target_net = [_train.average(x) for x in self.online_net]

        in_bn = self._batch_norm(state_in, is_training, 'target_batch_norm_in')
        h1 = tf.matmul(in_bn, target_net[0]) + target_net[1]
        h1_bn = self._batch_norm(h1, is_training, 'target_batch_norm_h1', tf.nn.relu)
        h2 = tf.matmul(h1_bn, target_net[2]) + target_net[3]
        h2_bn = self._batch_norm(h2, is_training, 'target_batch_norm_h2', tf.nn.relu)
        target_out = tf.tanh(tf.matmul(h2_bn, target_net[4]) + target_net[5])
        return state_in, target_out, target_update, is_training

    def _create_training(self):
        '''Training Actor method by applying gradients'''
        q_gradient_in = tf.placeholder('float', [None, self.n_action])
        parameters_gradients = tf.gradients(self.online_out, self.online_net, -q_gradient_in)
        optimizer = tf.train.AdamOptimizer(LR).apply_gradients(zip(parameters_gradients, self.online_net))
        return q_gradient_in, parameters_gradients, optimizer

    def _batch_norm(self, x, training_phase, scope_bn, activation=tf.identity):
        '''Custom Batch Normalization layer'''
        return tf.cond(training_phase,
                       lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                            updates_collections=None, is_training=True, reuse=None, scope=scope_bn, decay=0.9, epsilon=1e-5),
                       lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn, decay=0.9, epsilon=1e-5))

    def update_target(self):
        '''Update Actor Target Network'''
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, state_batch):
        '''Train Actor Online Network'''
        self.sess.run(self.optimizer, feed_dict={self.q_gradient_in:q_gradient_batch, self.online_state_in:state_batch, self.online_is_training:True})

    def actions(self, state_batch):
        '''Get actions batch from Actor Online Network for training'''
        return self.sess.run(self.online_out, feed_dict={self.online_state_in:state_batch, self.online_is_training:True})

    def action(self, state):
        '''Get an action from Actor Online Network for simulation and add data'''
        return self.sess.run(self.online_out, feed_dict={self.online_state_in:[state], self.online_is_training:False})[0]

    def target_actions(self, state_batch):
        '''Get target actions batch from Actor Target Network'''
        return self.sess.run(self.target_out, feed_dict={self.target_state_in: state_batch, self.target_is_training:True})
