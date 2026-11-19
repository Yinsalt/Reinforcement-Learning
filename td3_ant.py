import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import gymnasium as gym

import time
from collections import deque


BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_STEPS = 1_000_000
HIDDEN_SIZE = 256
LR_A = 3e-4
LR_C = 3e-4
UPDATE_EVERY = 50
RENDER_EPISODES = 5
ENV_ID = "Ant-v5"
ENV_KWARGS = dict(include_cfrc_ext_in_observation=False)



def initialize_layer(layer, std=np.sqrt(2),bias=0):
	nn.init.orthogonal_(layer.weight, gain=std)
	nn.init.constant_(layer.bias,bias)
	return layer


class ResBlock(nn.Module):
	def __init__(self, layer_size, factor=4):
		super().__init__()
		self.layer_size = layer_size
		self.LN = nn.LayerNorm(self.layer_size)
		self.hidden = max(1,int(layer_size*factor))
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
			initialize_layer(nn.Linear(hidden_size,hidden_size),std=1),
			nn.Tanh(),
			ResBlock(hidden_size,factor=4),
			nn.LayerNorm(hidden_size),
			initialize_layer(nn.Linear(hidden_size,output_size),std=0.01),
			nn.Tanh()
		)
		self.to(device)
		self.optimizer = optim.Adam(self.parameters(),lr=lr)
		
	def forward(self,x):
		return self.max_action*self.net(x)


class Critic(nn.Module):
	def __init__(self, obs_size, action_size, hidden_size, lr, device):
		super().__init__()
		self.net = nn.Sequential(
			initialize_layer(nn.Linear(obs_size+action_size,hidden_size)),
			nn.ReLU(),
			ResBlock(hidden_size,factor=4),
			initialize_layer(nn.Linear(hidden_size,hidden_size)),
			nn.ReLU(),
			ResBlock(hidden_size,factor=4),
			nn.LayerNorm(hidden_size),
			initialize_layer(nn.Linear(hidden_size,1),std=0.1)
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


class Agent():
	def __init__(self, env, num_steps, hidden_size, lr_a, lr_c,gamma=GAMMA, tau=TAU, batch_size = BATCH_SIZE, device = DEVICE):
		self.env = env
		self.num_steps = num_steps
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.device = device
		self.gamma = gamma
		self.tau = tau
		
		self.max_action = float(env.action_space.high[0])
		self.warmup_steps = 25_000
		self.update_count = 0
		self.policy_noise = 0.2*self.max_action
		self.noise_clip = 0.5*self.max_action

		self.policy_delay = 2
		
		self.act_size = env.action_space.shape[0]
		self.obs_size = env.observation_space.shape[0]
		
		self.actor = Actor(self.obs_size, hidden_size, self.act_size,lr_a,device,self.max_action)
		self.critic_1 = Critic(self.obs_size, self.act_size,self.hidden_size, lr_c,device)
		self.critic_2 = Critic(self.obs_size, self.act_size,self.hidden_size, lr_c,device)
		
		self.target_actor = Actor(self.obs_size, hidden_size, self.act_size,lr_a,device,self.max_action)
		self.target_critic_1 = Critic(self.obs_size, self.act_size,self.hidden_size, lr_c,device)
		self.target_critic_2 = Critic(self.obs_size, self.act_size,self.hidden_size, lr_c,device)
		
		self.buffer = ReplayBuffer(self.obs_size, self.act_size,self.num_steps,device)
		
		self.training = False
		
		self.initialize()
		
	def set_train(self):
		self.training = True
		
	def set_eval(self):
		self.training = False
		
	def initialize(self):
		for net, tgt in ((self.actor,self.target_actor),(self.critic_1,self.target_critic_1),(self.critic_2,self.target_critic_2)):
			tgt.load_state_dict(net.state_dict())
			tgt.requires_grad_(False)
			tgt.eval()
		self.set_train()
			
	def soft_update_actor(self):
		with torch.no_grad():
			net, tgt = self.actor,self.target_actor
			for tp, p in zip(tgt.parameters(), net.parameters()):
				tp.lerp_(p, self.tau)
	
	def soft_update_critic_1(self):
		with torch.no_grad():
			net, tgt = self.critic_1,self.target_critic_1
			for tp, p in zip(tgt.parameters(), net.parameters()):
				tp.lerp_(p, self.tau)
					
	def soft_update_critic_2(self):
		with torch.no_grad():
			net, tgt = self.critic_2,self.target_critic_2
			for tp, p in zip(tgt.parameters(), net.parameters()):
				tp.lerp_(p, self.tau)

	
	def get_action(self,obs):
		if self.training and self.buffer.step < self.warmup_steps:
			return self.env.action_space.sample()
		obs = torch.as_tensor(obs,dtype=torch.float32,device=self.device).unsqueeze(0)
		with torch.no_grad():
			action = self.actor(obs)
			if self.training:
				action = action + 0.1*self.max_action*torch.randn_like(action)
				action = action.clamp(-self.max_action,self.max_action)
		return action.cpu().numpy()[0]
			
		
	def learn(self):
		s,a,r,s_,term = self.buffer.sample(self.batch_size)
		
		with torch.no_grad():
			noise = (self.policy_noise*torch.randn_like(a)).clamp(-self.noise_clip,self.noise_clip)
			a_ = (self.target_actor(s_) + noise).clamp(-self.max_action,self.max_action)
			pred_c1 = self.target_critic_1(s_,a_)
			pred_c2 = self.target_critic_2(s_,a_)
			min_pred = torch.min(pred_c1,pred_c2)
			pred = r + self.gamma*(1-term)*min_pred
		
		critic1_loss = F.mse_loss(self.critic_1(s,a),pred)
		critic2_loss = F.mse_loss(self.critic_2(s,a),pred)
		
		self.critic_1.optimizer.zero_grad()
		self.critic_2.optimizer.zero_grad()
		
		critic1_loss.backward()
		critic2_loss.backward()
		
		self.critic_1.optimizer.step()
		self.critic_2.optimizer.step()
		
		self.update_count += 1
		actor_loss = None
		
		if self.update_count % self.policy_delay == 0:
			self.critic_1.requires_grad_(False)
			actor_loss = -self.critic_1(s,self.actor(s)).mean()
			self.actor.optimizer.zero_grad()
			actor_loss.backward()
			self.actor.optimizer.step()
			self.critic_1.requires_grad_(True)
			
			self.soft_update_actor()
			self.soft_update_critic_1()
			self.soft_update_critic_2()
			actor_loss = actor_loss.item()
		
		return actor_loss, critic1_loss.item(), critic2_loss.item()
			
		
		
###########################################################################



if __name__ == "__main__":
	env = gym.make(ENV_ID,**ENV_KWARGS)
	agent = Agent(env,TOTAL_STEPS,HIDDEN_SIZE,LR_A,LR_C)
	
	scores = deque(maxlen=10)
	lengths = deque(maxlen=10)
	a_losses = deque(maxlen=1000)
	c_losses = deque(maxlen=1000)
	episode = 0
	ep_return = 0.0
	ep_len = 0
	t0 = time.time()
	solve_score = env.spec.reward_threshold
	
	obs, _ = env.reset()
	
	for step in range(TOTAL_STEPS):
		action = agent.get_action(obs)
		obs_, reward, terminated, truncated, info = env.step(action)
		agent.buffer.save(obs,action,reward,obs_,terminated)
		ep_return += reward
		ep_len += 1
		obs = obs_
		
		if step >= agent.warmup_steps and step % UPDATE_EVERY == 0:
			for _ in range(UPDATE_EVERY):
				a_loss, c1_loss, c2_loss = agent.learn()
				c_losses.append((c1_loss+c2_loss)/2)
				if a_loss is not None:
					a_losses.append(a_loss)
		
		if terminated or truncated:
			episode += 1
			scores.append(ep_return)
			lengths.append(ep_len)
			sps = (step+1)/(time.time()-t0)
			al = np.mean(a_losses) if a_losses else float("nan")
			cl = np.mean(c_losses) if c_losses else float("nan")
			print(f"Ep {episode:5d} | Step {step+1:7d} | Return {ep_return:8.1f} | Avg10 {np.mean(scores):8.1f} | Len {np.mean(lengths):6.1f} | A {al:8.2f} | C {cl:8.3f} | {sps:5.0f} sps")
			
			if len(scores) == scores.maxlen and np.mean(scores) > solve_score:
				print(f"Solved in {episode} episodes / {step+1} steps, Avg10 = {np.mean(scores):.1f}")
				break
			
			ep_return = 0.0
			ep_len = 0
			obs, _ = env.reset()
	
	env.close()
	torch.save(agent.actor.state_dict(),"td3_ant_actor.pt")
	
	env = gym.make(ENV_ID,render_mode="human",**ENV_KWARGS)
	agent.env = env
	agent.set_eval()
	for ep in range(RENDER_EPISODES):
		obs, _ = env.reset()
		done, total, length = False, 0.0, 0
		while not done:
			action = agent.get_action(obs)
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			total += reward
			length += 1
		print(f"Render-Episode {ep+1}: {total:.1f} ({length} Steps)")
	env.close()

