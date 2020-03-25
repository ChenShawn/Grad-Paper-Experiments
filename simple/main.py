import numpy as np
import torch
import gym
import argparse
import os
import random
import copy

import utils
from simple_env import SimpleEnv
from models import td3, sac, sirtd3, sirsac

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_arguments():
	parser = argparse.ArgumentParser('TRAINING')
	# Global variables
	parser.add_argument("--policy", default="td3")                  # td3/ddpg/adv
	parser.add_argument("--env", default="simple")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start-timesteps", default=1e4, type=int) # Time steps initial random policy is used
	parser.add_argument("--max-timesteps", default=10000, type=int)   # Max time steps to run environment
	# Hyperparameters
	parser.add_argument("--expl-noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch-size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy-noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise-clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy-freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument('--adv-epsilon', type=float, default=0.05, help='ONLY USED FOR ADV MODELS')
	parser.add_argument('--alpha', type=float, default=0.05, help='ONLY USED FOR ADV MODELS')
	# boolean control variables
	parser.add_argument("--save-model", type=int, default=20)		# Save model every xxx episodes
	return parser.parse_args()


def get_policy(arglist, kwargs, max_action):
	# Initialize policy
	if arglist.policy == "td3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = arglist.policy_noise * max_action
		kwargs["noise_clip"] = arglist.noise_clip * max_action
		kwargs["policy_freq"] = arglist.policy_freq
		policy = td3.TD3(**kwargs)
	elif arglist.policy == 'sirtd3':
		kwargs['alpha'] = arglist.alpha
		kwargs['adv_epsilon'] = arglist.adv_epsilon
		kwargs['logdir'] = f'./tensorboard/{arglist.policy}_{arglist.env}_{arglist.seed}/'
		policy = TD3_adv2.TD3(**kwargs)
	else:
		raise NotImplementedError
	return policy


def main():
	args = parse_arguments()

	model_filename = f"train/{args.env}/{args.policy}_{args.env}_{args.seed}"
	log_filename = f'./logs/{args.policy}_{args.env}_{args.seed}.log'
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists('./logs'):
		os.makedirs('./logs')
	if args.save_model > 0 and not os.path.exists("./train"):
		os.makedirs("./train")
	if not os.path.exists(f'./train/{args.env}'):
		os.makedirs(f'./train/{args.env}')

	env = SimpleEnv()
	logfile = open(log_filename, 'w+')

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	state_dim = env.state_dim
	action_dim = env.action_dim
	max_action = float(env.action_range[-1])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	policy = get_policy(args, kwargs, max_action)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(1e+5))
	state, done = env.reset(), False
	episode_reward = 0.0
	episode_timesteps = 0
	episode_num = 0
	print(' [*] Initialization finished... Ready to get started!')

	for t in range(int(args.max_timesteps) + 1 + int(args.start_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			global_step = int(t + 1 - args.start_timesteps)
			loginfo = f"iter={global_step} n_episodes={episode_num+1} episode_len={episode_timesteps} total_reward={episode_reward} final_pos=[{next_state[0]}, {next_state[1]}]"
			print(loginfo)
			if t + 1 - args.start_timesteps >= 0:
				logfile.write(loginfo + '\n')
				logfile.flush()

			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0.0
			episode_timesteps = 0
			episode_num += 1

	logfile.close()
	policy.save(model_filename)
	print(' [*] Models succesfully saved!!')
	env.render()
	print(' [*] Done!!')


if '__main__' == __name__:
	main()
