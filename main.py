from Env_Normalization import env_norm
from Agent import Agent
import gym
import numpy as np
import tensorflow as tf
import os.path
from constants import *

if __name__ == '__main__':
    env = env_norm(gym.make(ENV_NAME))
    sess = tf.InteractiveSession()

    agent = Agent(env, sess)

    saver = tf.train.Saver()
    tracker = [0.0]
    if MODE == 'TRAIN':
        '''Train'''
        print('Collecting data ...')
        for episode in range(EPISODES):
            state = env.reset()
            for step in range(env.spec.timestep_limit):
                action = agent.noise_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.perceive(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

            if agent.replay_buffer.count() > REPLAY_START:
                # Validation
                state = env.reset()
                for j in range(env.spec.timestep_limit): # env.spec.timestep_limit = 1000
                    if agent.replay_buffer.count() > REPLAY_START and RENDER:
                        env.render()
                    action = agent.action(state)
                    state, reward, done, _ = env.step(action)
                    tracker[-1] += reward
                    if done:
                        # if agent.replay_buffer.count() > REPLAY_START and RENDER:
                        #     env.render(close=True)
                        break
                print('Episode: {}, Episode Validation Reward: {:.2f}'.format(episode, tracker[-1]))
                if np.mean(tracker[-100:]) > 995.0: # Model is stable
                    print('Model is stable!')
                    break
                tracker.append(0.0)

        print('Mean of Current 100 Episodes: {:.2f}'.format(np.mean(tracker[-100:])))
        save_path = saver.save(sess, './out/model.ckpt')
        print('Model is saved at %s' % save_path)
    else:
        '''Test'''
        saver.restore(sess, './out/model.ckpt')
        state = env.reset()
        for j in range(env.spec.timestep_limit): # env.spec.timestep_limit = 1000
            env.render()
            action = agent.action(state)
            state, reward, done, _ = env.step(action)
            tracker[-1] += reward
            if done:
                env.render(close=True)
                break
        print('Reward: {}'.format(tracker[-1]))
