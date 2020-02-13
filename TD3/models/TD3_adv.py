import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TD3 import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Q <- Vanilla Bellman update
Q_adv(s_t, a_t) <- E_pi [r_t + min_{eta <= epsilon} gamma * Q'(s_t+1 + eta, a_t+1)] 
pi <- argmax_pi { alpha * Q + (1 - alpha) * Q_adv }
Does not work yet!!!
"""

class CriticAdv(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(CriticAdv, self).__init__()

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


class TD3_adv(object):
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
		alpha=0.2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)

		# one more adv networks
		self.critic_adv = CriticAdv(state_dim, action_dim).to(device)
		self.critic_target_adv = copy.deepcopy(self.critic_adv)
		critic_params = list(self.critic.parameters()) + list(self.critic_adv.parameters())
		self.critic_optimizer = torch.optim.Adam(critic_params, lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.adv_epsilon = adv_epsilon
		self.alpha = alpha

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		next_state.requires_grad = True

		# Select action according to policy and add clipped noise
		noise = (
			torch.randn_like(action) * self.policy_noise
		).clamp(-self.noise_clip, self.noise_clip)
		
		next_action = (
			self.actor_target(next_state) + noise
		).clamp(-self.max_action, self.max_action)

		# Compute the target Q value
		target_Q1, target_Q2 = self.critic_target(next_state, next_action)
		target_Q = torch.min(target_Q1, target_Q2)
		target_Q = reward + not_done * self.discount * target_Q

		# Compute the fucking adversarial gradient perturbation
		adv_loss = target_Q.mean()
		self.critic_target.zero_grad()
		adv_loss.backward()

		# adversarial states (perturb state/action)
		next_state_grad = F.normalize(next_state.grad.data)
		next_state_adv = next_state - self.adv_epsilon * next_state_grad
		target_Q_adv = self.critic_target_adv(next_state_adv, next_action)
		target_Q_adv = reward + not_done * self.discount * target_Q_adv

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		current_Q_adv = self.critic_adv(state, action)

		# Compute critic loss
		q1_loss = F.mse_loss(current_Q1, target_Q.detach())
		q2_loss = F.mse_loss(current_Q2, target_Q.detach())
		q_adv_loss = F.mse_loss(current_Q_adv, target_Q_adv.detach())
		critic_loss = q1_loss + q2_loss + q_adv_loss

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			# actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			actor_loss_ori = (1.0 - self.alpha) * self.critic.Q1(state, self.actor(state))
			actor_loss_adv = self.alpha * self.critic_adv(state, action)
			actor_loss = -(actor_loss_ori + actor_loss_adv).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.critic_adv.parameters(), self.critic_target_adv.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic.pth")
		torch.save(self.critic_adv.state_dict(), filename + "_critic_adv.pth")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pth")
		torch.save(self.actor.state_dict(), filename + "_actor.pth")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pth")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic.pth"))
		self.critic.load_state_dict(torch.load(filename + "_critic_adv.pth"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pth"))
		self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pth"))
