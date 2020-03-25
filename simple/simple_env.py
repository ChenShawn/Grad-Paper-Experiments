import numpy as np
import math
import random
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class SimpleEnv(object):
    """SimpleEnv
    An agent moves on a surface of a 2-dimensional Gaussian with isotropic noise
    """
    state_dim = 2
    action_dim = 2
    action_range = [-0.66, 0.66]
    state_range = [0.0, 50.0]
    epsilon = 0.25
    _max_episode_steps = 200

    def __init__(self, gmean=[20.0, 20.0], gstd=[8.0, 8.0], use_noise=True,
                 noise_ratio=0.1, nmean=[8.0, 8.0], nstd=[1.0, 1.0], 
                 reward_ratio=1e+6, finite_horizon=True):
        self._use_noise = use_noise
        # major Gaussian
        self.gauss_mean = np.array(gmean, dtype=np.float32).reshape((-1, 1))
        self.gauss_var = np.square(np.diag(gstd).astype(np.float32))
        # noise gaussian
        self.noise_mean = np.array(nmean, dtype=np.float32).reshape((-1, 1))
        self.noise_var = np.square(np.diag(nstd).astype(np.float32))
        self.nratio = noise_ratio
        self.rratio = reward_ratio
        self._finite_horizon = finite_horizon
        # running variables
        self.pos = np.random.uniform(self.state_range[0], self.state_range[1], size=[2, 1]).astype(np.float32)
        self.num_steps = 0
        self.trajectory = []


    def _compute_reward(self):
        gauss_ratio = 1.0 / (math.sqrt(2 * math.pi) * math.sqrt(np.linalg.det(self.gauss_var)))
        dist = self.pos - self.gauss_mean
        distnum = np.matmul(np.matmul(dist.T, np.linalg.inv(self.gauss_var)), dist)[0, 0]
        gauss_val = gauss_ratio * math.exp(-0.5 * distnum)
        if self._use_noise:
            noise_det = math.sqrt(np.linalg.det(self.noise_var))
            noise_ratio = 1.0 / (math.sqrt(2.0 * math.pi) * noise_det)
            dist = self.pos - self.noise_mean
            distnum = np.matmul(np.matmul(dist.T, np.linalg.inv(self.noise_var)), dist)[0, 0]
            noise_val = noise_ratio * math.exp(-0.5 * distnum)
        else:
            noise_val = 0.0
        dist = math.sqrt(np.square(self.pos - self.gauss_mean).sum())
        if dist < self.epsilon:
            rend = 20.0
        else:
            rend = 0.0
        return self.rratio * (gauss_val - self.nratio * noise_val) + rend


    def _is_finished(self):
        if self._finite_horizon:
            dist = math.sqrt(np.square(self.pos - self.gauss_mean).sum())
            return dist < self.epsilon or self.num_steps >= self._max_episode_steps
        else:
            return False


    def reset(self):
        # self.pos = np.zeros(self.pos.shape, dtype=self.pos.dtype)
        self.pos = np.random.uniform(self.state_range[0], self.state_range[1], size=[2, 1]).astype(np.float32)
        self.num_steps = 0
        self.trajectory.clear()
        return self.pos.flatten()


    def step(self, action):
        action = np.clip(action, self.action_range[0], self.action_range[1]).reshape((-1, 1))
        assert action.shape[0] == self.action_dim, 'Invalid action dimension'
        prev_height = self._compute_reward()
        self.trajectory.append((self.pos[0, 0], self.pos[1, 0], prev_height))
        self.pos += action
        new_height = self._compute_reward()
        #print(f'prev={prev_height}\t new={new_height}')
        reward = new_height - prev_height
        done = self._is_finished()
        self.num_steps += 1
        if done:
            self.trajectory.append((self.pos[0, 0], self.pos[1, 0], new_height))
        return self.pos.flatten(), reward, done, None


    def sample(self):
        return np.random.uniform(self.action_range[0], self.action_range[1], size=[2])


    def seed(self, es):
        np.random.seed(es)
        random.seed(es)


    def render(self):
        plt.close()
        # regarding height value as z axis
        data = np.array(self.trajectory, dtype=np.float32)
        # draw figures
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title("Trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        figure = ax.plot(data[:, 0], data[:, 1], data[:, 2], marker='x')
        plt.show()
        


if __name__ == '__main__':
    env = SimpleEnv()
    s = env.reset()
    for it in range(30):
        a = env.sample()
        s_next, r, done, info = env.step(a)
        print(f'step={it}\t pos={s_next}\t reward={r}\t done={done}')
    env.render()
    print('\n DONE')
