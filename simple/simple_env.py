import numpy as np
import math
import random
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class SimpleEnv(object):
    """SimpleEnv
    An agent moves on a surface of a 2-dimensional Gaussian with isotropic noise
    Updated: try using 2D orthogonal sine function as noise
    """
    state_dim = 2
    action_dim = 2
    action_range = [-1.0, 1.0]
    state_range = [0.0, 40.0]
    epsilon = 0.25
    _max_episode_steps = 200

    def __init__(self, gmean=[20.0, 20.0], gstd=[8.0, 8.0], use_noise=False,
                 noise_ratio=0.1, nmean=[8.0, 8.0], nstd=[1.0, 1.0], 
                 reward_ratio=1e+3, finite_horizon=True):
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
        pos = self.pos.flatten().tolist()
        gauss = self._compute_gaussian(pos)
        if self._use_noise:
            noise = self._compute_noise(pos)
        else:
            noise = 0.0
        return self.rratio * (gauss - self.nratio * noise)


    def _compute_gaussian(self, pos):
        dist = (pos[0] - self.gauss_mean[0, 0]) ** 2 + (pos[1] - self.gauss_mean[1, 0]) ** 2
        dist = dist / (2.0 * math.sqrt(self.gauss_var[0, 0] * self.gauss_var[1, 1]))
        #scale = math.sqrt(2.0 * math.pi * self.gauss_var[0, 0] * self.gauss_var[1, 1])
        scale = 1.0
        return math.exp(-dist) / scale


    def _compute_noise(self, pos):
        """_compute_noise
        Generate 2D sine noise with period equal to 2.0
        TODO: support more types of noise later
        """
        return math.sin(math.pi * pos[0]) + math.sin(math.pi * pos[1])


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
        fig = plt.figure()
        # draw surface
        xs = np.arange(self.state_range[0], self.state_range[1], 0.2)
        ys = np.arange(self.state_range[0], self.state_range[1], 0.2)
        xs, ys = np.meshgrid(xs, ys)
        zs = np.exp(-((xs - self.gauss_mean[0, 0]) ** 2 + (ys - self.gauss_mean[1, 0]) ** 2) / (2.0 * self.gauss_var[0, 0]))
        if self._use_noise:
            zs = zs - self.nratio * (np.sin(math.pi * xs) + np.sin(math.pi * ys))
        zs *= self.rratio
        ax = Axes3D(fig)
        ax.plot_surface(xs, ys, zs, cmap='viridis')
        # plot line on surface
        ax = fig.gca(projection='3d')
        ax.set_title("Trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        figure = ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=1.5, color='red')
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
