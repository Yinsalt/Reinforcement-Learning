import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F

import numpy as np
import gymnasium as gym

import time
from collections import deque


BATCH_SIZE = 512
GAMMA = 0.99
TOTAL_STEPS = 1_000_000
WARMUP_STEPS = 10_000
HIDDEN_SIZE = 128
LR_A = 3e-4
LR_C = 3e-4
BUFFER_SIZE = 1_000_000
RENDER_EPISODES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPDATE_EVERY = 50

###########################################################################



def initialize_layer(layer, std=np.sqrt(2),bias=0):
	nn.init.orthogonal_(layer.weight, gain=std)
	nn.init.constant_(layer.bias,bias)
	return layer




class ResBlock(nn.Module):
	def __init__(self, layer_size, factor=4):
		super().__init__()
		self.layer_size = layer_size
		self.LN = nn.LayerNorm(self.layer_size)
		self.hidden = int(layer_size*factor) if layer_size*factor>1 and layer_size > 1 else 1
		self.net = nn.Sequential(
			initialize_layer(nn.Linear(layer_size,self.hidden)),
			nn.ReLU(),
			initialize_layer(nn.Linear(self.hidden,layer_size))
		)
	def forward(self,x):
		return x + self.net(self.LN(x))

		
	


class Actor(nn.Module):
	def __init__(self, input_size, hidden_size, output_size,lr,device,max_action=1.0):
		super().__init__()
		self.max_action = max_action
		self.net = nn.Sequential(
			initialize_layer(nn.Linear(input_size,hidden_size),std=1),
			nn.Tanh(),
			ResBlock(hidden_size,factor=4),
			#nn.LayerNorm(hidden_size),
			initialize_layer(nn.Linear(hidden_size,2*hidden_size),std=1),
			nn.Tanh(),
			ResBlock(2*hidden_size,factor=2),
			nn.LayerNorm(2*hidden_size),
			initialize_layer(nn.Linear(2*hidden_size,output_size),std=0.01),
			nn.Tanh()
		)
		self.to(device)
		self.optimizer = optim.Adam(self.parameters(),lr=lr)
		
	def forward(self,x):
		return self.max_action*self.net(x)




		
class Critic(nn.Module):
	def __init__(self, obs_size, action_size, hidden_size,lr,device):
		super().__init__()
		self.net = nn.Sequential(
			initialize_layer(nn.Linear(obs_size+action_size,hidden_size)),
			nn.ReLU(),
			ResBlock(hidden_size,factor=4),
			nn.LayerNorm(hidden_size),
			initialize_layer(nn.Linear(hidden_size,2*hidden_size)),
			nn.ReLU(),
			ResBlock(2*hidden_size,factor=2),
			nn.LayerNorm(2*hidden_size),
			initialize_layer(nn.Linear(2*hidden_size,1),std=0.1)
		)

		self.to(device)
		self.optimizer = optim.AdamW(self.parameters(),lr=lr,weight_decay=1e-2)
		
	def forward(self,state,action):
		return self.net(torch.cat([state,action],dim=-1))
		
		

###########################################################################



class ReplayBuffer():

	def __init__(self, obs_size, action_size, max_size,device):
		
		self.step = 0
		self.max_size = max_size
		self.obs_size = obs_size
		self.action_size = action_size
		
		self.states = torch.zeros((max_size,obs_size),dtype=torch.float32)
		self.actions = torch.zeros((max_size,action_size),dtype=torch.float32)
		self.rewards = torch.zeros((max_size,1),dtype=torch.float32)
		self.terminated = torch.zeros((max_size,1),dtype=torch.float32)
		self.states_ = torch.zeros((max_size,obs_size),dtype=torch.float32)
		
		self.device = device
		
	def save(self,state,action,reward,state_,terminated):
		t = self.step%self.max_size
		
		self.states[t]=torch.as_tensor(state,dtype=torch.float32)
		self.actions[t]=torch.as_tensor(action,dtype=torch.float32)
		self.rewards[t]=float(reward)
		self.states_[t]=torch.as_tensor(state_,dtype=torch.float32)
		self.terminated[t]=float(terminated)
		self.step +=1
		
	def __len__(self):
		return min(self.max_size,self.step)
		
	def sample(self,batchsize=64):
		idx = torch.randint(0,len(self),(batchsize,))
		return (self.states[idx].to(self.device),
				self.actions[idx].to(self.device),
				self.rewards[idx].to(self.device),
				self.states_[idx].to(self.device),
				self.terminated[idx].to(self.device))



###########################################################################



class OUNoise(nn.Module):
	def __init__(
		self,
		mu: float = 0.0,
		theta: float = 0.15,
		sigma: float = 0.2,
		dt: float = 1.0,
		shape: tuple = (1, 8),
	):
		super().__init__()
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.dt = dt
		self.shape = shape

		self.register_buffer("state", torch.full(shape, float(mu)))

	@torch.no_grad()
	def reset(self):
		self.state.fill_(self.mu)

	@torch.no_grad()
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * (self.dt ** 0.5) * torch.randn_like(self.state)
		self.state.add_(dx)
		return x + self.state


###########################################################################




class Agent():
	def __init__(self, obs_size, hidden_size, act_size, maxlen, max_action, lr_a,lr_c, device):
		
		self.device = device
		self.max_action = max_action
		
		self.actor = Actor(obs_size,hidden_size,act_size,lr_a,device,max_action)
		self.critic = Critic(obs_size,act_size,hidden_size,lr_c,device)
		
		self.target_actor = Actor(obs_size,hidden_size,act_size,lr_a,device,max_action)
		self.target_critic = Critic(obs_size,act_size,hidden_size,lr_c,device)
		
		self.buffer = ReplayBuffer(obs_size,act_size,maxlen,device)
		self.tau = 0.005
		
		self.noise = OUNoise(sigma=0.2*max_action, shape=(1,act_size)).to(device)
		
		self.training = True
		
		self.initialize()
		
	def initialize(self):
		self.target_actor.load_state_dict(self.actor.state_dict())
		self.target_critic.load_state_dict(self.critic.state_dict())
		
		self.target_actor.requires_grad_(False)
		self.target_critic.requires_grad_(False)
		
		self.target_actor.eval()
		self.target_critic.eval()
	
	def set_train(self):
		self.training = True
		
	def set_eval(self):
		self.training = False
		
	def reset_noise(self):
		self.noise.reset() 
		
	def soft_update(self):
		with torch.no_grad():
			for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
				tp.lerp_(p, self.tau)
			for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
				tp.lerp_(p, self.tau)

	def get_action(self,obs):
		obs = torch.as_tensor(obs,dtype=torch.float32,device=self.device).unsqueeze(0)
		with torch.no_grad():
			mu = self.actor(obs)
			if self.training:
				mu = torch.clamp(self.noise(mu),-self.max_action,self.max_action)
		return mu.cpu().numpy()[0]



	def learn(self):
		if len(self.buffer) < BATCH_SIZE:
			return
		
		states, actions, rewards, states_, dones = self.buffer.sample(BATCH_SIZE)
		
		with torch.no_grad():
			tgt_actions = self.target_actor(states_)
			tgt_value = self.target_critic(states_,tgt_actions)
			tgt = rewards + GAMMA*(1-dones)*tgt_value
		
		critic_value = self.critic(states,actions)
		critic_loss = F.mse_loss(critic_value,tgt)
		self.critic.optimizer.zero_grad()
		critic_loss.backward()
		self.critic.optimizer.step()
		
		self.critic.requires_grad_(False)
		actor_loss = -self.critic(states,self.actor(states)).mean()
		self.actor.optimizer.zero_grad()
		actor_loss.backward()
		self.actor.optimizer.step()
		self.critic.requires_grad_(True)
		
		self.soft_update()
		
		return actor_loss.item(), critic_loss.item()










###########################################################################

if __name__ == "__main__":

	
	env = gym.make("HalfCheetah-v5")
	obs_size = env.observation_space.shape[0]
	act_size = env.action_space.shape[0]
	max_action = float(env.action_space.high[0]) 
	
	agent = Agent(obs_size,HIDDEN_SIZE,act_size,BUFFER_SIZE,max_action,LR_A,LR_C,DEVICE)
	
	scores = deque(maxlen=10)
	a_losses = deque(maxlen=1000)
	c_losses = deque(maxlen=1000)
	episode = 0
	ep_return = 0.0
	t0 = time.time()
	solve_score = env.spec.reward_threshold
	
	obs, _ = env.reset()
	agent.reset_noise()
	
	for step in range(TOTAL_STEPS):
		if step < WARMUP_STEPS:
			action = env.action_space.sample()
		else:
			action = agent.get_action(obs)
		
		obs_, reward, terminated, truncated, info = env.step(action)
		agent.buffer.save(obs,action,reward,obs_,terminated)
		ep_return += reward
		obs = obs_
		
		if step >= WARMUP_STEPS and step % UPDATE_EVERY == 0:
			for _ in range(UPDATE_EVERY):
				a_loss, c_loss = agent.learn()
				a_losses.append(a_loss)
				c_losses.append(c_loss)
		
		if terminated or truncated:
			episode += 1
			scores.append(ep_return)
			sps = (step+1)/(time.time()-t0)
			al = np.mean(a_losses) if a_losses else float("nan")
			cl = np.mean(c_losses) if c_losses else float("nan")
			print(f"Ep {episode:4d} | Step {step+1:7d} | Return {ep_return:8.1f} | Avg10 {np.mean(scores):8.1f} | A {al:8.2f} | C {cl:8.3f} | {sps:5.0f} sps")
			if len(scores) == scores.maxlen and np.mean(scores) > solve_score:
				print(f"Solved in {episode} episodes / {step+1} steps, Avg10 = {np.mean(scores):.1f}")
				break
			ep_return = 0.0
			obs, _ = env.reset()
			agent.reset_noise()
	
	env.close()
	torch.save(agent.actor.state_dict(),"ddpg_cheetah_actor.pt")
	
	env = gym.make("HalfCheetah-v5",render_mode="human")
	agent.set_eval()
	for ep in range(RENDER_EPISODES):
		obs, _ = env.reset()
		done, total = False, 0.0
		while not done:
			action = agent.get_action(obs)
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			total += reward
		print(f"Render-Episode {ep+1}: {total:.1f}")
	env.close()


