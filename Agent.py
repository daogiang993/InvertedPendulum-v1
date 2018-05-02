import gym
import tensorflow as tf
import numpy as np
from OU_Noise import OUNoise
from Actor import Actor
from Critic import Critic
from Replay_Buffer import ReplayBuffer
from constants import *

class Agent:
    def __init__(self, env, sess):
        # Environment
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]

        # Neural Networks
        self.sess = sess
        self.actor = Actor(self.sess, self.n_state, self.n_action)
        self.critic = Critic(self.sess, self.n_state, self.n_action)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        # Ornstein-Uhlenbeck Noise
        self.exploration_noise = OUNoise(self.n_action)

    def noise_action(self, state):
        '''Get action with noise'''
        return self.action(state) + self.exploration_noise.noise()

    def action(self, state):
        '''Get action from online actor'''
        return self.actor.action(state)

    def train(self):
        '''Train Networks'''
        # Draw sample from Replay Buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([d[0] for d in minibatch])
        action_batch = np.asarray([d[1] for d in minibatch])
        reward_batch = np.asarray([d[2] for d in minibatch])
        next_state_batch = np.asarray([d[3] for d in minibatch])
        done_batch = np.asarray([d[4] for d in minibatch])

        # Train Critic
        next_action_batch = self.actor.target_actions(next_state_batch)
        target_q_value_batch = self.critic.target_q(next_state_batch, next_action_batch)
        # q = r if done else r+gamma*target_q
        q_batch = reward_batch.reshape((BATCH_SIZE,1)) + (1. - done_batch.reshape(BATCH_SIZE,1).astype(float))*GAMMA*target_q_value_batch
        self.critic.train(q_batch, state_batch, action_batch)

        # Train Actor
        action_batch_grads = self.actor.actions(state_batch)
        q_grads_batch = self.critic.gradients(state_batch, action_batch_grads)
        self.actor.train(q_grads_batch, state_batch)

        # Slowly update Target Networks
        self.actor.update_target()
        self.critic.update_target()

    def perceive(self, state, action, reward, next_state, done):
        '''Add transition to replay buffer and train if there are sufficient amount of transitions'''
        # Add samples
        self.replay_buffer.add(state, action, reward, next_state, done)
        # Train if there are sufficient number of samples
        if self.replay_buffer.count() > REPLAY_START:
            self.train()
        # Reset the noise for next episode
        if done:
            self.exploration_noise.reset()
