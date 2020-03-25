import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Q <- E_pi [ r_t - alpha * max_eta ( Q(s_t, a_t) - Q(s_t - eta, a_t) ) + gamma * Q'(s_t+1, a_t+1) ]
pi <- argmax_pi E_pi { Q }
"""

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1



class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		adv_epsilon=0.1,
		alpha=0.1,
		logdir=None
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		# Q1 networks
		self.critic_1 = Critic(state_dim, action_dim).to(device)
		self.critic_target_1 = copy.deepcopy(self.critic_1)
		self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
		# Q2 networks
		self.critic_2 = Critic(state_dim, action_dim).to(device)
		self.critic_target_2 = copy.deepcopy(self.critic_2)
		self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		# adv parameters
		self.adv_epsilon = adv_epsilon
		self.alpha = alpha
		if logdir is not None:
			self.writer = tf.summary.FileWriter(logdir)

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		state.requires_grad = True

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic_1(state, action), self.critic_2(state, action)
		current_Q1_target = self.critic_target_1(state, action)
		current_Q2_target = self.critic_target_2(state, action)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1 = self.critic_target_1(next_state, next_action)
			target_Q2 = self.critic_target_2(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			
		# get adversarial target Q
		state.requires_grad = True
		adv_loss_1 = current_Q1_target.mean()
		self.critic_target_1.zero_grad()
		adv_loss_1.backward(retain_graph=True)
		adv_perturb_1 = F.normalize(state.grad.data)

		state.requires_grad = True
		adv_loss_2 = current_Q2_target.mean()
		self.critic_target_2.zero_grad()
		adv_loss_2.backward(retain_graph=True)
		adv_perturb_2 = F.normalize(state.grad.data)

		# get Q1 Q2 adversairial estimation
		adv_q1 = self.critic_1(state - self.adv_epsilon * adv_perturb_1, action)
		adv_q2 = self.critic_2(state - self.adv_epsilon * adv_perturb_2, action)
		adv_error_1 = torch.clamp(current_Q1 - adv_q1, 0.0, 1000.0)
		adv_error_2 = torch.clamp(current_Q2 - adv_q2, 0.0, 1000.0)
		target_Q1 = reward - self.alpha * adv_error_1 + not_done * self.discount * target_Q
		target_Q2 = reward - self.alpha * adv_error_2 + not_done * self.discount * target_Q
		target_Q1, target_Q2 = target_Q1.detach(), target_Q2.detach()

		# Compute critic loss
		critic_loss_1 = F.mse_loss(current_Q1, target_Q1)
		critic_loss_2 = F.mse_loss(current_Q2, target_Q2)

		# Optimize critic Q1
		self.critic_optimizer_1.zero_grad()
		critic_loss_1.backward()
		self.critic_optimizer_1.step()
		# Optimize critic Q2
		self.critic_optimizer_2.zero_grad()
		critic_loss_2.backward()
		self.critic_optimizer_2.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor losse
			actor_loss = -self.critic_1(state, self.actor(state)).mean()
			
			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			# [ ONLY FOR DEBUG ]
			if self.total_it % 100 == 2:
				reward_mean = reward.mean().float()
				adv_error_mean = 0.5 * adv_error_1.mean().float() + 0.5 * adv_error_2.mean().float()
				q_loss_mean = 0.5 * critic_loss_1.float() + 0.5 * critic_loss_2.float()
				pi_loss_mean = actor_loss.float()
				sumstr = tf.Summary(value=[
					tf.Summary.Value(tag='agent/reward', simple_value=reward_mean),
					tf.Summary.Value(tag='agent/adv_error', simple_value=adv_error_mean),
					tf.Summary.Value(tag='agent/qloss', simple_value=q_loss_mean),
					tf.Summary.Value(tag='agent/pi_loss', simple_value=pi_loss_mean)
				])
				self.writer.add_summary(sumstr, global_step=self.total_it)


	def save(self, filename):
		torch.save(self.critic_1.state_dict(), filename + "_critic_1.pth")
		torch.save(self.critic_optimizer_1.state_dict(), filename + "_critic_optimizer_1.pth")
		torch.save(self.critic_2.state_dict(), filename + "_critic_2.pth")
		torch.save(self.critic_optimizer_2.state_dict(), filename + "_critic_optimizer_2.pth")
		torch.save(self.actor.state_dict(), filename + "_actor.pth")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pth")


	def load(self, filename):
		self.critic_1.load_state_dict(torch.load(filename + "_critic_1.pth", map_location=lambda storage, loc: storage))
		self.critic_optimizer_1.load_state_dict(torch.load(filename + "_critic_optimizer_1.pth", map_location=lambda storage, loc: storage))
		self.critic_2.load_state_dict(torch.load(filename + "_critic_2.pth", map_location=lambda storage, loc: storage))
		self.critic_optimizer_2.load_state_dict(torch.load(filename + "_critic_optimizer_2.pth", map_location=lambda storage, loc: storage))
		self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=lambda storage, loc: storage))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth", map_location=lambda storage, loc: storage))
