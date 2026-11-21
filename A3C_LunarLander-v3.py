import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical
import torch.nn.functional as F

import numpy as np
from collections import deque
import gymnasium as gym

NUM_WORKERS = min(32, os.cpu_count())
T_MAX = 20
HIDDEN_SIZE = 256
LR = 1e-3
GAMMA = 0.99
ENTROPY_COEF = 1e-2
MAX_STEPS = 2_000_000
GRAD_CLIP = 40.0
SOLVE_WINDOW = 50
SOLVE_SCORE = 175
RENDER_EPISODES = 5
ENV_NAME = "LunarLander-v3"
SEED = 0


class SharedAdam(optim.Adam):
	def __init__(self, params, lr):
		super().__init__(params, lr=lr, foreach=False)
		for group in self.param_groups:
			for p in group["params"]:
				state = self.state[p]
				state["step"] = torch.tensor(0.0)
				state["exp_avg"] = torch.zeros_like(p.data)
				state["exp_avg_sq"] = torch.zeros_like(p.data)
				
	def share_memory(self):
		for group in self.param_groups:
			for p in group["params"]:
				for v in self.state[p].values():
					v.share_memory_()
		return self


class ActorCritic(nn.Module):
	def __init__(self,input_size,hidden_size,output_size):
		super().__init__()
		self.actor = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size)
			)
		self.critic = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1)
			)
			
	def forward(self,x):
		return self.actor(x), self.critic(x).squeeze(-1)


def worker(rank, global_model, optimizer, counter, lock, queue, stop):
	torch.manual_seed(SEED+rank)
	torch.set_num_threads(1)
	env = gym.make(ENV_NAME)
	obs_size = env.observation_space.shape[0]
	local = ActorCritic(obs_size,HIDDEN_SIZE,env.action_space.n)
	
	obs, _ = env.reset(seed=SEED+rank)
	ep_return = 0.0
	
	while not stop.value and counter.value < MAX_STEPS:
		local.load_state_dict(global_model.state_dict())
		
		log_probs, values, entropies, rewards = [], [], [], []
		terminated = truncated = False
		for _ in range(T_MAX):
			logits, value = local(torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0))
			dist = Categorical(logits=logits)
			action = dist.sample()
			obs, reward, terminated, truncated, _ = env.step(action.item())
			ep_return += reward
			
			log_probs.append(dist.log_prob(action))
			values.append(value)
			entropies.append(dist.entropy())
			rewards.append(reward)
			
			with lock:
				counter.value += 1
			if terminated or truncated:
				queue.put(ep_return)
				ep_return = 0.0
				break
		
		if terminated:
			R = 0.0
		else:
			with torch.no_grad():
				R = local(torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0))[1].item()
		
		returns = []
		for r in reversed(rewards):
			R = r + GAMMA*R
			returns.append(R)
		returns = torch.tensor(returns[::-1],dtype=torch.float32)
		
		log_probs = torch.cat(log_probs)
		values = torch.cat(values)
		entropy = torch.cat(entropies).mean()
		adv = returns - values
		
		actor_loss = -(log_probs*adv.detach()).mean() - ENTROPY_COEF*entropy
		critic_loss = 0.5*adv.pow(2).mean()
		
		local.zero_grad()
		(actor_loss + critic_loss).backward()
		nn.utils.clip_grad_norm_(local.parameters(),GRAD_CLIP)
		
		for lp, gp in zip(local.parameters(), global_model.parameters()):
			gp.grad = lp.grad
		optimizer.step()
		
		if terminated or truncated:
			obs, _ = env.reset()


if __name__ == "__main__":
	mp.set_start_method("spawn")
	
	probe = gym.make(ENV_NAME)
	obs_size, act_size = probe.observation_space.shape[0], probe.action_space.n
	probe.close()
	
	global_model = ActorCritic(obs_size,HIDDEN_SIZE,act_size)
	global_model.share_memory()
	optimizer = SharedAdam(global_model.parameters(),lr=LR).share_memory()
	
	counter = mp.Value("i",0)
	lock = mp.Lock()
	queue = mp.Queue()
	stop = mp.Value("b",False)
	
	processes = [mp.Process(target=worker,args=(rank,global_model,optimizer,counter,lock,queue,stop)) for rank in range(NUM_WORKERS)]
	for p in processes:
		p.start()
	
	scores = deque(maxlen=SOLVE_WINDOW)
	episodes = 0
	while any(p.is_alive() for p in processes):
		try:
			ret = queue.get(timeout=1.0)
		except Exception:
			continue
		scores.append(ret)
		episodes += 1
		avg = np.mean(scores)
		if episodes % 50 == 0:
			print(f"Episodes {episodes:5d} | Steps {counter.value:8d} | Avg{SOLVE_WINDOW} {avg:6.1f}")
		if not stop.value and len(scores) == SOLVE_WINDOW and avg > SOLVE_SCORE:
			print(f"Solved after {episodes} episodes / {counter.value} steps, Avg{SOLVE_WINDOW} = {avg:.1f}")
			stop.value = True
	
	for p in processes:
		p.join()
	
	env = gym.make(ENV_NAME,render_mode="human")
	for ep in range(RENDER_EPISODES):
		obs, _ = env.reset()
		done, total = False, 0.0
		while not done:
			with torch.no_grad():
				logits, _ = global_model(torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0))
			obs, reward, terminated, truncated, _ = env.step(logits.argmax(dim=-1).item())
			done = terminated or truncated
			total += reward
		print(f"Render-Episode {ep+1}: {total:.0f}")
	env.close()
