import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

import numpy as np
from collections import deque
import gymnasium as gym

NUM_ENVS = 32
NUM_STEPS = 16
HIDDEN_SIZE = 256
LR_A = 1e-3
LR_C = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 1e-2
MAX_UPDATES = 5000
SOLVE_WINDOW = 50
SOLVE_SCORE = 470
RENDER_EPISODES = 5
ENV_NAME = "CartPole-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Buffer():
	def __init__(self,num_timesteps, num_envs, obs_dim,device=DEVICE):
		self.num_timesteps, self.num_envs = num_timesteps, num_envs
		self.obs_dim = obs_dim
		self.device=device

		self.obs = torch.zeros((self.num_timesteps, self.num_envs, self.obs_dim),dtype=torch.float32).to(self.device)
		self.actions = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.long).to(self.device)
		self.log_probs = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)
		self.advantages = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)
		self.values = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)
		self.returns = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)
		self.dones = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)
		self.rewards = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)

		self.step = 0

	def reset(self):
		self.step = 0

	def append(self, obs, actions, log_probs, values, rewards, dones):
		t = self.step
		self.obs[t] = obs
		self.actions[t] = actions
		self.log_probs[t] = log_probs
		self.values[t] = values
		self.rewards[t] = torch.as_tensor(rewards,dtype=torch.float32,device=self.device)
		self.dones[t] = torch.as_tensor(dones,dtype=torch.float32,device=self.device)
		self.step+=1

	def compute_gae(self,next_value,gamma,gae_lambda):
		last_gae = 0
		for t in reversed(range(self.num_timesteps)):
			next_values = next_value if t==(self.num_timesteps-1) else self.values[t+1]
			nonterminal = 1.0-self.dones[t]
			delta = self.rewards[t] + gamma * nonterminal * next_values - self.values[t]
			last_gae = delta + gamma*gae_lambda*nonterminal*last_gae
			self.advantages[t] = last_gae
		self.returns = self.advantages + self.values


class Actor(nn.Module):
	def __init__(self,input_size,hidden_size,output_size,lr,device):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size)
			)
		self.to(device)
		self.optimizer = optim.Adam(self.parameters(),lr=lr)
	
	def forward(self,x):
		return self.net(x)
		
class Critic(nn.Module):
	def __init__(self,input_size,hidden_size,lr,device):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1)
			)
		self.to(device)
		self.optimizer = optim.Adam(self.parameters(),lr=lr)
	
	def forward(self,x):
		return self.net(x)


class Agent():
	def __init__(self, hidden_size, lr_a,lr_c,device,num_steps, num_envs, env_name):
		self.device = device
		self.envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)],autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
		input_size = self.envs.single_observation_space.shape[0]
		output_size = self.envs.single_action_space.n
		
		self.gamma, self.gae_lambda = GAMMA, GAE_LAMBDA
		self.actor = Actor(input_size,hidden_size,output_size,lr_a,device)
		self.critic = Critic(input_size,hidden_size,lr_c,device)
		self.buffer = Buffer(num_steps,num_envs,input_size,device)
		
		self.ep_returns = np.zeros(num_envs)
		self.finished_returns = []
		
		self.next_state = self.initialize()
		
	def initialize(self):
		self.buffer.reset()
		obs, _ = self.envs.reset()
		return torch.as_tensor(obs,dtype=torch.float32,device=self.device)
		
	def play(self, num_steps):
		self.buffer.reset()
		self.finished_returns = []
		for _ in range(num_steps):
			with torch.no_grad():
				dist = Categorical(logits=self.actor(self.next_state))
				action = dist.sample()
				value = self.critic(self.next_state).squeeze(-1)
			next_obs, reward, terminated, truncated, info = self.envs.step(action.cpu().numpy())
			done = terminated | truncated
			
			self.ep_returns += reward
			for i in np.where(done)[0]:
				self.finished_returns.append(self.ep_returns[i])
				self.ep_returns[i] = 0.0
			
			reward = torch.as_tensor(reward,dtype=torch.float32,device=self.device)
			
			idx = truncated & ~terminated
			if idx.any():
				final = torch.as_tensor(np.stack(info["final_obs"][idx]),dtype=torch.float32,device=self.device)
				with torch.no_grad():
					reward[torch.as_tensor(idx,device=self.device)] += self.gamma*self.critic(final).squeeze(-1)
			
			self.buffer.append(self.next_state,action,dist.log_prob(action),value,reward,done)
			self.next_state = torch.as_tensor(next_obs,dtype=torch.float32,device=self.device)
			
	def learn(self):
		with torch.no_grad():
			next_value = self.critic(self.next_state).squeeze(-1)
		self.buffer.compute_gae(next_value,self.gamma,self.gae_lambda)
		
		T, N = self.buffer.num_timesteps, self.buffer.num_envs
		obs = self.buffer.obs.reshape(T*N,-1)
		actions = self.buffer.actions.reshape(-1)
		adv = self.buffer.advantages.reshape(-1)
		ret = self.buffer.returns.reshape(-1)
		
		dist = Categorical(logits=self.actor(obs))
		log_prob = dist.log_prob(actions)
		entropy = dist.entropy().mean()
		values = self.critic(obs).squeeze(-1)
		
		actor_loss = -(log_prob*adv.detach()).mean() - ENTROPY_COEF*entropy
		critic_loss = 0.5*F.mse_loss(values,ret.detach())
		
		self.actor.optimizer.zero_grad()
		self.critic.optimizer.zero_grad()
		(actor_loss + critic_loss).backward()
		self.actor.optimizer.step()
		self.critic.optimizer.step()
		
		return actor_loss.item(), critic_loss.item(), entropy.item()
		
		
if __name__ == "__main__":
	agent = Agent(HIDDEN_SIZE,LR_A,LR_C,DEVICE,NUM_STEPS,NUM_ENVS,ENV_NAME)
	scores = deque(maxlen=SOLVE_WINDOW)
	episodes = 0
	
	for update in range(1,MAX_UPDATES+1):
		agent.play(NUM_STEPS)
		a_loss, c_loss, ent = agent.learn()
		
		scores.extend(agent.finished_returns)
		episodes += len(agent.finished_returns)
		avg = np.mean(scores) if scores else 0.0
		
		if update % 20 == 0:
			print(f"Update {update:5d} | Steps {update*NUM_STEPS*NUM_ENVS:8d} | Episodes {episodes:5d} | Avg{SOLVE_WINDOW} {avg:6.1f} | A {a_loss:7.3f} | C {c_loss:8.3f} | H {ent:5.3f}")
		
		if len(scores) == SOLVE_WINDOW and avg > SOLVE_SCORE:
			print(f"Solved after {update} updates / {episodes} episodes, Avg{SOLVE_WINDOW} = {avg:.1f}")
			break
	
	agent.envs.close()
	
	env = gym.make(ENV_NAME,render_mode="human")
	for ep in range(RENDER_EPISODES):
		obs, _ = env.reset()
		done, total = False, 0.0
		while not done:
			with torch.no_grad():
				logits = agent.actor(torch.as_tensor(obs,dtype=torch.float32,device=DEVICE).unsqueeze(0))
			action = logits.argmax(dim=-1).item()
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			total += reward
		print(f"Render-Episode {ep+1}: {total:.0f}")
	env.close()
