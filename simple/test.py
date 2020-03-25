import gym
import pybullet_envs
from PIL import Image
import argparse
import numpy as np
import torch
import copy
import os
from sklearn.preprocessing import normalize as Normalize

from models import TD3, TD3_adv2


def parse_arguments():
    parser = argparse.ArgumentParser("TESTING")
    parser.add_argument('-p', "--policy", type=str, default='td3', help="td3/adv")
    parser.add_argument('-e', "--env", type=str, default="LunarLanderContinuous-v2", help="env name")
    parser.add_argument('-n', "--n-episodes", type=int, default=10, help="number of episodes")
    parser.add_argument('-m', "--relative-mass", type=float, default=1.0, help="relative-mass")
    parser.add_argument("--noise-scale", type=float, default=0.0, help="relative-mass")
    parser.add_argument("--train-seed", type=int, default=1, help="random seed for training")
    parser.add_argument("--test-seed", type=int, default=1, help="random seed for testing")
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--ensemble", action="store_true", default=False)
    parser.add_argument("--multiple", type=str, default='', help='mass/noise')
    return parser.parse_args()


def get_policy(arglist, kwargs, max_action):
	# Initialize policy
	if arglist.policy == "td3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = 0.0
		kwargs["noise_clip"] = 0.0
		kwargs["policy_freq"] = 2
		policy = TD3.TD3(**kwargs)
	elif arglist.policy == 'adv':
		kwargs['alpha'] = 0.01
		kwargs['adv_epsilon'] = 0.01
		kwargs['logdir'] = f'./tensorboard/{arglist.policy}_{arglist.env}_{arglist.train_seed}/'
		policy = TD3_adv2.TD3(**kwargs)
	else:
		raise NotImplementedError
	return policy


def test(arglist):
    env_name = arglist.env
    random_seed = arglist.test_seed
    n_episodes = arglist.n_episodes
    lr = 0.002
    max_timesteps = 3000
    render = arglist.render
    
    if not arglist.ensemble:
        filename = "{}_{}_{}".format(arglist.policy, env_name, arglist.train_seed)
        directory = "./train/{}".format(env_name)
    else:
        filename = "{}_{}_{}_ensemble".format(arglist.policy, env_name, arglist.train_seed)
        directory = "./train/{}".format(env_name)
    
    #env = gym.make(env_name)
    env = gen_envs(arglist)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Set random seed
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": 0.99,
		"tau": 0.005,
        "policy_noise": 0.001,
        "noise_clip": 1.0,
        "policy_freq": 2
	}
    policy = get_policy(arglist, kwargs, max_action)
    policy.load(os.path.join(directory, filename))
    
    total_reward_list = []
    for ep in range(1, n_episodes+1):
        ep_reward = 0.0
        state = env.reset()
        for t in range(max_timesteps):
            noise = np.random.normal(0.0, 1.0, size=state.shape)
            noise = np.clip(noise, -1.0, 1.0)
            adv_state = state + arglist.noise_scale * noise
            action = policy.select_action(adv_state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if done:
                break
            
        #print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        total_reward_list.append(ep_reward)
        ep_reward = 0.0
    env.close()
    return total_reward_list


if __name__ == '__main__':
    args = parse_arguments()

    if not args.multiple:
        reward_list = test(args)

        reward_array = np.array(reward_list, dtype=np.float32)
        reward_mean = reward_array.mean()
        reward_half_std = reward_array.std() / 2.0
        loginfo = 'policy={} env={} train_seed={} test_seed={} relative_mass={} noise_scale={} result={}±{}'
        print(loginfo.format(args.policy, args.env, args.train_seed, args.test_seed, args.relative_mass, args.noise_scale, reward_mean, reward_half_std))

    elif 'mass' in args.multiple:
        result_list = multi_test_mass(args)
        xs = [info['relative_mass'] for info in result_list]
        ys = [info['reward_mean'] for info in result_list]
        ys_half_std = [info['reward_half_std'] for info in result_list]
        ys_lower = np.array(ys, dtype=np.float32) - np.array(ys_half_std, dtype=np.float32)
        ys_upper = np.array(ys, dtype=np.float32) + np.array(ys_half_std, dtype=np.float32)

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(xs, ys, linewidth=1.5, color='blue')
        plt.fill_between(xs, ys_lower, ys_upper, color='blue', alpha=0.2)
        plt.xlabel('relative mass')
        plt.ylabel('average returns')
        plt.title('Performance under different relative mass values')
        plt.show()

    elif 'noise' in args.multiple:
        result_list = multi_test_mass(args)
        xs = [info['relative_mass'] for info in result_list]
        ys = [info['reward_mean'] for info in result_list]
        ys_half_std = [info['reward_half_std'] for info in result_list]
        ys_lower = np.array(ys, dtype=np.float32) - np.array(ys_half_std, dtype=np.float32)
        ys_upper = np.array(ys, dtype=np.float32) + np.array(ys_half_std, dtype=np.float32)

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(xs, ys, linewidth=1.5, color='blue')
        plt.fill_between(xs, ys_lower, ys_upper, color='blue', alpha=0.2)
        plt.xlabel('relative mass')
        plt.ylabel('average returns')
        plt.title('Performance under different relative mass values')
        plt.show()

    else:
        raise NotImplementedError('invalid argument for [multiple]')
