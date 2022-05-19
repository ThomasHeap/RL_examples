import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np

eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):

    def __init__(self, state_size, outputs):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, outputs)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return F.softmax(self.layer2(x), dim=0)





def REINFORCE(alpha=0.01, gamma=0.99, num_episodes=100, episode_length=10000, learning_rate=1e-2, render=True):
    env = gym.make('MountainCar-v0')

    def select_action(state):
        probs = policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def finish_episode(rewards,log_probs):
        G = 0
        returns = []
        policy_loss = []
        for r in rewards[::-1]:
            G = r + gamma * G
            returns.insert(0,G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean())/(returns.std() + eps)

        for R, lp in zip(returns,log_probs):
            policy_loss.append(-lp*R)

        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

    policy = Policy(2, 3)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    running_reward = -1000

    for episode in range(num_episodes):
        rewards = []
        log_probs = []
        state = env.reset()
        state = torch.tensor(state)
        ep_reward = -1
        for t in range(1,episode_length):
            action, lp = select_action(state)
            log_probs.append(lp)
            state, reward, done, _ = env.step(action)
            state=torch.tensor(state)
            rewards.append(reward)
            ep_reward += reward
            if render:
                env.render()
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(rewards,log_probs)

        if episode % 3 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    REINFORCE()
