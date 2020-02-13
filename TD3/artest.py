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
    parser.add_argument("--mode", type=str, default='nr', help="nr (default) / pr")
    parser.add_argument("--train-seed", type=int, default=1, help="random seed for training")
    parser.add_argument("--test-seed", type=int, default=1, help="random seed for testing")
    parser.add_argument("--nr-delta", type=float, default=0.0, help="delta for NR-MDP") 
    parser.add_argument("--pr-prob", type=float, default=0.0, help="prob of PR-MDP")
    parser.add_argument("--render", action="store_true", default=False)
    return parser.parse_args()



def get_policy(arglist, kwargs, max_action):
	# Initialize policy
	if arglist.policy == "td3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = 0.0
		kwargs["noise_clip"] = 0.0
		kwargs["policy_freq"] = 2
		policy = TD3.TD3(**kwargs)
	elif arglist.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif arglist.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)
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
    
    filename = "{}_{}_{}".format(arglist.policy, env_name, arglist.train_seed)
    directory = "./train/{}".format(env_name)
    
    env = gym.make(env_name)
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

            action = policy.select_action(state)
            if arglist.mode == 'nr':
                # use truncated gaussian noise for both nr-mdp and pr-mdp settings
                noise = np.random.normal(0.0, max_action, size=action.shape)
                noise = np.clip(noise, -max_action, max_action)
                adv_action = (1.0 - arglist.nr_delta) * action + arglist.nr_delta * noise
            elif arglist.mode == 'pr':
                adv_action = action
                if np.random.rand() < arglist.pr_prob:
                    adv_action = np.random.normal(0.0, action_dim, size=action.shape)
                    adv_action = np.clip(adv_action, -max_action, max_action)
            else:
                raise NotImplementedError('invalid mode')

            state, reward, done, _ = env.step(adv_action)
            ep_reward += reward
            if render:
                env.render()
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        total_reward_list.append(ep_reward)
        ep_reward = 0.0
    env.close()
    return total_reward_list



if __name__ == '__main__':
    args = parse_arguments()

    reward_list = test(args)

    reward_array = np.array(reward_list, dtype=np.float32)
    reward_mean = reward_array.mean()
    reward_half_std = reward_array.std() / 2.0
    loginfo = 'policy={} env={} load_seed={} random_seed={} mode={} pr-prob={} nr-delta={} result={}±{}'
    print(loginfo.format(args.policy, args.env, args.train_seed, args.test_seed, args.mode, args.pr_prob, args.nr_delta, reward_mean, reward_half_std))

