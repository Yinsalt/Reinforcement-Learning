import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

import numpy as np

import gymnasium as gym

env_id = "FrozenLake-v1"
env_kwargs = dict(map_name="8x8", is_slippery=False)
device = "cpu"
lr_a = 5e-4
lr_c = 1e-3
num_episodes = 10_000
hidden_size = 128
gamma = 0.99
entropy_beta = 0.01
solve_window = 50
solve_score = 0.9
render_episodes = 5

##################################################################################

class Actor(nn.Module):
	def __init__(self, input_size, hidden_size, output_size,lr=lr_a,device=device):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_size,hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size,hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size,output_size)
		)
		self.to(device)
		self.optimizer = optim.Adam(self.parameters(),lr=lr)
		
	def forward(self,x):
		return self.net(x)
		
class Critic(nn.Module):
	def __init__(self, input_size, hidden_size,lr=lr_c,device=device):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_size,hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size,hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size,1)
		)
		self.to(device)
		self.optimizer = optim.Adam(self.parameters(),lr=lr)
		
	def forward(self,x):
		return self.net(x)
		
##################################################################################

def one_hot(s, n, device):
	x = torch.zeros((1,n),dtype=torch.float32,device=device)
	x[0,s] = 1.0
	return x

##################################################################################		
		
def rollout(actor, env,device):
	n_states = env.observation_space.n
	done = False
	obs, _ = env.reset()
	obs = one_hot(obs,n_states,device)
	states = []
	rewards = []
	actions = []
	dones = []
	states_ = []
	entropies = []
	
	while not done:
		logits = actor(obs)
		dist = Categorical(logits=logits)
		action = dist.sample()
		logprob = dist.log_prob(action)
		
		obs_, reward, terminated, truncated, info = env.step(action.item())
		obs_ = one_hot(obs_,n_states,device)
		
		states.append(obs)
		rewards.append(reward)
		actions.append(logprob)
		entropies.append(dist.entropy())
		
		done = truncated or terminated
		
		dones.append(terminated)
		states_.append(obs_)
		obs=obs_
		
	return states,rewards,actions,dones,states_,entropies
		
##################################################################################

def learn(actor,critic,states,rewards,actions,dones,states_,entropies,gamma=0.99):
	states = torch.cat(states)
	states_ = torch.cat(states_)
	rewards = torch.tensor(rewards,dtype=torch.float32,device=states.device)
	dones = torch.tensor(dones,dtype=torch.float32,device=states.device)
	logprobs = torch.cat(actions)
	entropy = torch.cat(entropies).mean()
	
	V = critic(states).squeeze(-1)
	with torch.no_grad():
		V_ = critic(states_).squeeze(-1)
	
	target = rewards + gamma*V_*(1-dones)
	delta = target - V
	
	actor_loss = -(logprobs*delta.detach()).mean() - entropy_beta*entropy
	critic_loss = (delta**2).mean()
	
	actor.optimizer.zero_grad()
	critic.optimizer.zero_grad()
	
	actor_loss.backward()
	critic_loss.backward()
	
	actor.optimizer.step()
	critic.optimizer.step()
	
	return actor_loss.item(), critic_loss.item()

##################################################################################

if __name__ == "__main__":
	env = gym.make(env_id,**env_kwargs)
	n_states = env.observation_space.n
	actor = Actor(n_states,hidden_size,env.action_space.n)
	critic = Critic(n_states,hidden_size)
	scores = deque(maxlen=solve_window)
	first_success = None
	
	for episode in range(num_episodes):
		states,rewards,actions,dones,states_,entropies = rollout(actor,env,device)
		a_loss, c_loss = learn(actor,critic,states,rewards,actions,dones,states_,entropies,gamma)
		
		scores.append(sum(rewards))
		if first_success is None and sum(rewards) > 0:
			first_success = episode
			print(f"First success in episode {episode}")
		avg = np.mean(scores)
		if episode % 100 == 0:
			print(f"Episode {episode:5d} | Avg{solve_window} {avg:5.2f} | Len {len(rewards):3d} | A {a_loss:7.3f} | C {c_loss:7.4f}")
		
		if len(scores) == solve_window and avg > solve_score:
			print(f"Solved in {episode} episodes, Avg{solve_window} = {avg:.2f}")
			break
	
	env.close()
	
	env = gym.make(env_id, render_mode="human",**env_kwargs)
	for ep in range(render_episodes):
		obs, _ = env.reset()
		done, total = False, 0.0
		while not done:
			with torch.no_grad():
				logits = actor(one_hot(obs,n_states,device))
			action = logits.argmax(dim=-1).item()
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			total += reward
		print(f"Render-Episode {ep + 1}: {total:.0f}")
	env.close()
