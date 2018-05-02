import numpy as np

class OUNoise:
    def __init__(self, n_action, mu=0, theta=0.15, sigma=0.2):
        self.n_action = n_action
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        # Reset every episode
        self._noise = np.ones(self.n_action)*self.mu

    def noise(self):
        # d_xt = theta*(mu - xt) + sigma*d_Wt
        d_noise = self.theta*(self.mu - self._noise) + self.sigma*np.random.randn(len(self._noise))
        self._noise += d_noise
        return self._noise
