import argparse
import tensorflow as tf
import datetime
import gym
import numpy as np
import itertools
import torch
import os

from sac import SAC
from adv_sac import ADVSAC
from replay_memory import ReplayMemory

"""
Updated in Feb 9th, 2020:
use target Q-network to estimate the adversarial network value
"""


def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('-e', '--env', default="HalfCheetah-v2", type=str, help='gym env')
    parser.add_argument('-p', '--policy', default="Gaussian", type=str, help='`Gaussian` or `deterministic`')
    parser.add_argument('--eval', type=bool, default=False, help='Evaluates a policy a policy every 10 episode')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount reward')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='target smoothing coefficient (τ)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G', help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G', help='Temperature')
    parser.add_argument('--automatic-entropy-tuning', type=bool, default=False, metavar='G', help='Automatically adjust alpha')
    parser.add_argument('--seed', type=int, default=123456, metavar='N', help='global random seed')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='batch size')
    parser.add_argument('--num-steps', type=int, default=1000001, metavar='N', help='maximum number of steps')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N', help='hidden size')
    parser.add_argument('--updates-per-step', type=int, default=1, metavar='N', help='model updates per simulator step')
    parser.add_argument('--max-episode-len', type=int, default=1000, help='max episode len')
    parser.add_argument('--start-steps', type=int, default=10000, metavar='N', help='Steps sampling random actions')

    # parameters for adversarial training
    parser.add_argument('--adv-epsilon', type=float, default=0.05, help='max perturb region in adv training')
    parser.add_argument('--adv-lambda', type=float, default=0.05, help='coefficient for adv error')

    parser.add_argument('--target-update-interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates')
    parser.add_argument('--replay-size', type=int, default=1000000, metavar='N', help='size of replay buffer')
    parser.add_argument('--use-adv', action="store_true", default=False)
    parser.add_argument('--render', action="store_true", default=False)
    return parser.parse_args()


def get_env_from_args(args):
    """get_env_from_args
        To be updated in future version to add support for ensemble training
    """
    env = gym.make(args.env)
    return env   


"""
if __name__ == '__main__'
    run training experiments
"""

args = get_arguments()
# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = get_env_from_args(args)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# processing log files
if not os.path.exists('./logs'):
    os.makedirs('./logs')
log_dirname  = './logs/{}'.format(args.env)
if not os.path.exists(log_dirname):
    os.makedirs(log_dirname)
log_filename = './logs/{}/{}_{}_{}.log'.format(args.env, args.policy, args.env, args.seed)
log_fd = open(log_filename, 'w+')

# Agent
if not args.use_adv:
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
else:
    agent = ADVSAC(env.observation_space.shape[0], env.action_space, args)

#TesnorboardX
writer = tf.summary.FileWriter('tensorboard/{}_{}_{}'.format(args.policy, args.env, args.seed))

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = int(-args.start_steps)
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    critic_1_loss = None

    while not done:
        if total_numsteps < 0:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size and total_numsteps >= 0:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, adv_err_1, adv_err_2 = agent.update_parameters(memory, args.batch_size, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory.push(state, action, reward, next_state, mask)
        state = next_state

    if total_numsteps > args.num_steps:
        break

    if len(memory) > args.batch_size and total_numsteps >= 0 and critic_1_loss is not None:
        sumstr = tf.Summary(value=[
            tf.Summary.Value(tag='loss/qloss_1', simple_value=critic_1_loss),
            tf.Summary.Value(tag='loss/qloss_2', simple_value=critic_2_loss),
            tf.Summary.Value(tag='loss/pi_loss', simple_value=policy_loss),
            tf.Summary.Value(tag='loss/entropy_loss', simple_value=ent_loss),
            tf.Summary.Value(tag='agent/reward', simple_value=episode_reward),
            tf.Summary.Value(tag='agent/alpha', simple_value=alpha),
            tf.Summary.Value(tag='agent/adv_error_1', simple_value=adv_err_1),
            tf.Summary.Value(tag='agent/adv_error_2', simple_value=adv_err_2),

        ])
        writer.add_summary(sumstr, global_step=total_numsteps)
        loginfo = f'iter={total_numsteps} n_episodes={i_episode} episode_len={episode_steps} total_reward={episode_reward}'
        print(loginfo)
        log_fd.write(loginfo + '\n')
        log_fd.flush()

    if i_episode % 10 == 0 and args.eval == True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

# save models
agent.save_model(args.env, args.seed)

env.close()
log_fd.close()
