import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

import numpy as np

import gymnasium as gym

###############################################################################

device = "cpu"
lr = 1e-3
num_steps = 500_000
hidden_size = 128
gamma = 0.99
episodes = 0
solve_window = 50
solve_score = 470
render_episodes = 5

###############################################################################

class REINFORCE_net(nn.Module):
	def __init__(self,input_size, hidden_size, output_size, lr, device):
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
	
###############################################################################

if __name__ == "__main__":
	env = gym.make("CartPole-v1")
	net = REINFORCE_net(env.observation_space.shape[0],hidden_size,env.action_space.n,lr=lr,device=device)
	obs, _ = env.reset()
	action_memory = []
	reward_memory = []
	scores = deque(maxlen=solve_window)
	
	for i in range(num_steps):
		obs = torch.tensor(np.array([obs]),dtype=torch.float32,device=device)
		logits = net(obs)
		dist = Categorical(logits = logits)
		action = dist.sample()
		obs, reward, terminated, truncated, info = env.step(action.item())
		
		reward_memory.append(reward)
		action_memory.append(dist.log_prob(action))
		
		if terminated or truncated:
			net.optimizer.zero_grad()
			discounted_return = np.zeros_like(reward_memory,dtype=np.float32)
			
			for timestep in range(len(reward_memory)):
				discounted_return_timestep = 0
				
				for j in range(timestep,len(reward_memory)):
					discounted_return_timestep += gamma**(j - timestep) * reward_memory[j]
				
				discounted_return[timestep] = discounted_return_timestep
			
			discounted_return = torch.tensor(discounted_return, dtype=torch.float32, device=device)
			discounted_return = (discounted_return - discounted_return.mean()) / (discounted_return.std() + 1e-8)
			loss = 0
			for G, logprob in zip(discounted_return,action_memory):
				loss += -G * logprob
			loss.sum().backward()
			net.optimizer.step()
			
			episodes += 1
			scores.append(sum(reward_memory))
			avg = np.mean(scores)
			if episodes % 10 == 0:
				print(f"Episode {episodes:4d} | Score {scores[-1]:6.1f} | Avg{solve_window} {avg:6.1f}")
			
			if len(scores) == solve_window and avg > solve_score:
				print(f"Solved in {episodes} episodes, avg{solve_window} = {avg:.1f}")
				break
			
			obs, _ = env.reset()
			action_memory = []
			reward_memory = []
	
	env.close()

  ###########################################################################
  
	env = gym.make("CartPole-v1", render_mode="human")
	for ep in range(render_episodes):
		obs, _ = env.reset()
		done, total = False, 0.0
		while not done:
			with torch.no_grad():
				logits = net(torch.tensor(np.array([obs]),dtype=torch.float32,device=device))
			action = logits.argmax(dim=-1).item()
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			total += reward
		print(f"Render-Episode {ep + 1}: {total:.0f}")
	env.close()
