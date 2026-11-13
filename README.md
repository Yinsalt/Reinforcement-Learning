# RL from Scratch

From-scratch implementations of reinforcement learning algorithms in PyTorch.

## Why

I wrote these without following tutorials or books. Many of them are built directly from the original papers, not adapted from existing code. Using a library tells you that an algorithm works. Writing it yourself shows you why, and where it breaks.

## What's in here

The repo grows in no particular order. Some algorithms are simple, some are more involved, and I upload whatever I'm currently working on while walking through the reinforcement learning jungle.

Planned so far:

- REINFORCE
- A2C, A3C
- PPO
- TRPO
- DQN, Double DQN
- DDPG, MADDPG
- SAC

The environments vary too. Simple algorithms run on the classic Gymnasium tasks like CartPole or Pendulum. The ones built for continuous control run on MuJoCo tasks.

## Tutorials

Once enough algorithms and environments are covered, I will start recording video walkthroughs for them.

## Requirements

- Python 3
- PyTorch
- Gymnasium
- MuJoCo (only for the continuous control environments)
