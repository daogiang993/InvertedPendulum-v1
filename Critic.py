import tensorflow as tf
import numpy as np
from constants import *

LR = LR_CRITIC

class Critic:
    def __init__(self, sess, n_state, n_action):
        self.time_step = 0
        self.sess = sess
        self.n_state = n_state
        self.n_action = n_action

        self.online_state_in, self.online_action_in, self.online_q_value, self.online_net = self._create_online()
        self.target_state_in, self.target_action_in, self.target_q_value, self.target_update = self._create_target()
        self.q_in, self.cost, self.optimizer, self.action_gradients = self._create_training()

        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def _create_online(self):
        '''Online Critic Network'''
        state_in = tf.placeholder('float', [None, self.n_state])
        action_in = tf.placeholder('float', [None, self.n_action])

        # Initialization for ReLU
        W1 = tf.Variable(tf.random_uniform([self.n_state, H1_SIZE], -1./np.sqrt(self.n_state), 1/np.sqrt(self.n_state)))
        b1 = tf.Variable(tf.random_uniform([H1_SIZE], -1./np.sqrt(self.n_state), 1/np.sqrt(self.n_state)))
        W2 = tf.Variable(tf.random_uniform([H1_SIZE, H2_SIZE], -1./np.sqrt(H1_SIZE+self.n_action), 1/np.sqrt(H1_SIZE+self.n_action)))
        W2_action = tf.Variable(tf.random_uniform([self.n_action, H2_SIZE], -1./np.sqrt(H1_SIZE+self.n_action), 1/np.sqrt(H1_SIZE+self.n_action)))
        b2 = tf.Variable(tf.random_uniform([H2_SIZE], -1./np.sqrt(H1_SIZE+self.n_action), 1/np.sqrt(H1_SIZE+self.n_action)))
        W3 = tf.Variable(tf.random_uniform([H2_SIZE, self.n_action], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([self.n_action], -3e-3, 3e-3))

        # Train graph
        h1 = tf.nn.relu(tf.matmul(state_in, W1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, W2) + tf.matmul(action_in, W2_action) + b2)
        q_value = tf.matmul(h2, W3) + b3

        return state_in, action_in, q_value, [W1, b1, W2, W2_action, b2, W3, b3]

    def _create_target(self):
        '''Target Critic Network'''
        state_in = tf.placeholder('float', [None, self.n_state])
        action_in = tf.placeholder('float', [None, self.n_action])

        _train = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = _train.apply(self.online_net)
        target_net = [_train.average(x) for x in self.online_net]

        h1 = tf.nn.relu(tf.matmul(state_in, target_net[0]) + target_net[1])
        h2 = tf.nn.relu(tf.matmul(h1, target_net[2]) + tf.matmul(action_in, target_net[3]) + target_net[4])
        q_value = tf.matmul(h2, target_net[5]) + target_net[6]

        return state_in, action_in, q_value, target_update

    def _create_training(self):
        '''Train Online Critic Network by minimize q-value loss and get gradient of action'''
        q_in = tf.placeholder('float', [None, 1])
        weight_decay = tf.add_n([L2*tf.nn.l2_loss(x) for x in self.online_net]) # Regularizer
        cost = tf.reduce_mean(tf.square(q_in - self.online_q_value)) + weight_decay
        optimizer = tf.train.AdamOptimizer(LR).minimize(cost)
        action_gradients = tf.gradients(self.online_q_value, self.online_action_in)
        return q_in, cost, optimizer, action_gradients

    def update_target(self):
        '''Update Target Critic Network'''
        self.sess.run(self.target_update)

    def train(self, q_batch, state_batch, action_batch):
        '''Train Online Critic'''
        self.time_step += 1
        self.sess.run(self.optimizer, feed_dict={self.q_in:q_batch, self.online_state_in:state_batch, self.online_action_in:action_batch})

    def gradients(self, state_batch, action_batch):
        '''Get action gradient'''
        return self.sess.run(self.action_gradients, feed_dict={self.online_state_in:state_batch, self.online_action_in:action_batch})[0]

    def target_q(self, state_batch, action_batch):
        '''Get target Q value from Target Critic'''
        return self.sess.run(self.target_q_value, feed_dict={self.target_state_in:state_batch, self.target_action_in:action_batch})

    def q_value(self, state_batch, action_batch):
        '''Get target Q value from Online Critic'''
        return self.sess.run(self.online_q_value, feed_dict={self.online_state_in:state_batch, self.online_action_in:action_batch})
