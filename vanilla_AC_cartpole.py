import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

import numpy as np

import gymnasium as gym

env_id = "CartPole-v1"
device = "cpu"
lr_a = 5e-4
lr_c = 1e-3
num_episodes = 3000
hidden_size = 128
gamma = 0.99
solve_window = 50
solve_score = 470
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
		
def rollout(actor, env,device):
	done = False
	obs, _ = env.reset()
	obs = torch.tensor(np.array([obs],dtype=np.float32),dtype=torch.float32,device=device)
	states = []
	rewards = []
	actions = []
	dones = []
	states_ = []
	
	while not done:
		logits = actor(obs)
		dist = Categorical(logits=logits)
		action = dist.sample()
		logprob = dist.log_prob(action)
		
		obs_, reward, terminated, truncated, info = env.step(action.item())
		obs_ = torch.tensor(np.array([obs_],dtype=np.float32),dtype=torch.float32,device=device)
		
		states.append(obs)
		rewards.append(reward)
		actions.append(logprob)
		
		done = truncated or terminated
		
		dones.append(terminated)
		states_.append(obs_)
		obs=obs_
		
	return states,rewards,actions,dones,states_
		
##################################################################################

def learn(actor,critic,states,rewards,actions,dones,states_,gamma=0.99):
	states = torch.cat(states)
	states_ = torch.cat(states_)
	rewards = torch.tensor(rewards,dtype=torch.float32,device=states.device)
	dones = torch.tensor(dones,dtype=torch.float32,device=states.device)
	logprobs = torch.cat(actions)
	
	V = critic(states).squeeze(-1)
	with torch.no_grad():
		V_ = critic(states_).squeeze(-1)
	
	target = rewards + gamma*V_*(1-dones)
	delta = target - V
	
	actor_loss = -(logprobs*delta.detach()).mean()
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
	env = gym.make(env_id)
	actor = Actor(env.observation_space.shape[0],hidden_size,env.action_space.n)
	critic = Critic(env.observation_space.shape[0],hidden_size)
	scores = deque(maxlen=solve_window)
	
	for episode in range(num_episodes):
		states,rewards,actions,dones,states_ = rollout(actor,env,device)
		a_loss, c_loss = learn(actor,critic,states,rewards,actions,dones,states_,gamma)
		
		scores.append(sum(rewards))
		avg = np.mean(scores)
		if episode % 10 == 0:
			print(f"Episode {episode:4d} | Score {scores[-1]:6.1f} | Avg{solve_window} {avg:6.1f} | A {a_loss:7.3f} | C {c_loss:7.3f}")
		
		if len(scores) == solve_window and avg > solve_score:
			print(f"Solved in {episode} episodes, Avg{solve_window} = {avg:.1f}")
			break
	
	env.close()
	
	env = gym.make(env_id, render_mode="human")
	for ep in range(render_episodes):
		obs, _ = env.reset()
		done, total = False, 0.0
		while not done:
			with torch.no_grad():
				logits = actor(torch.tensor(np.array([obs]),dtype=torch.float32,device=device))
			action = logits.argmax(dim=-1).item()
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			total += reward
		print(f"Render-Episode {ep + 1}: {total:.0f}")
	env.close()
