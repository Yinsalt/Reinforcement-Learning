# RL from Scratch

From-scratch implementations of reinforcement learning algorithms in PyTorch.

## Why

Many of these are built directly from the original papers, not adapted from existing code. Using a library tells you that an algorithm works. Writing it yourself shows you why, and where it breaks.

## What's in here

The repo grows in no particular order. Some algorithms are simple, some are more involved, and I upload whatever I'm currently working on while walking through the reinforcement learning jungle. The same algorithm can show up more than once, on a different environment or with a different action space, because that is often where the interesting problems are.

Every file is self-contained. Run it, it trains, and at the end it renders the trained agent.

| File | Algorithm | Environment | Notes |
|---|---|---|---|
| `reinforce_cartpole.py` | REINFORCE | CartPole-v1 | Monte Carlo policy gradient |
| `reinforce_LunarLander.py` | REINFORCE | LunarLander-v3 | Same algorithm on a harder task |
| `vanilla_AC_cartpole.py` | Actor-Critic | CartPole-v1 | One-step TD advantage, separate actor and critic |
| `entropy_AC_frozenLake.py` | Actor-Critic + entropy bonus | FrozenLake-v1 (8x8) | Sparse reward, entropy term keeps exploration alive |
| `ppo_cartpole_pure.py` | PPO | CartPole-v1 | Minimal version, under 80 lines |
| `ppo_cartpole.py` | PPO | CartPole-v1 | Full version with checkpoints, diagnostics (KL, clip fraction, explained variance) and plots |
| `ppo_discrete_action_space_acrobot.py` | PPO | Acrobot-v1 | Discrete actions |
| `ppo_continuous_lunar_lander.py` | PPO | LunarLander-v3 (continuous) | Gaussian policy |
| `ppo_continuous_action_space_walker.py` | PPO | BipedalWalker-v3 | Continuous actions, running observation normalization |
| `ppo_mixed_action_space_CarRacing.py` | PPO | CarRacing-v3 | Mixed action space (Gaussian steering + categorical pedals), CNN on stacked frames |
| `dqn_cnn_breakout.py` | Double DQN | ALE/Breakout-v5 | CNN, replay buffer, Atari wrappers, TensorBoard logging |
| `ddpg_halfCheetah.py` | DDPG | HalfCheetah-v5 (MuJoCo) | OU noise, soft target updates, residual blocks |

## Roadmap

- [x] REINFORCE
- [x] Actor-Critic
- [x] PPO (discrete, continuous, mixed action space)
- [x] DQN / Double DQN
- [x] DDPG
- [ ] A2C
- [ ] A3C
- [x] TD3
- [ ] TRPO
- [ ] Rainbow
- [ ] D4PG
- [ ] SAC
- [ ] MADDPG

## Tutorials

Once enough algorithms and environments are covered, I will start recording video walkthroughs on my YouTube channel.

## Requirements

- Python 3
- PyTorch
- Gymnasium

Some environments need extras:

```bash
pip install "gymnasium[box2d]"    # LunarLander, BipedalWalker, CarRacing
pip install "gymnasium[mujoco]"   # HalfCheetah
pip install ale-py                # Breakout
pip install torchvision           # CarRacing
pip install tensorboard           # DQN logging
pip install matplotlib            # PPO CartPole plots
```
