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


class StateValue(nn.Module):

    def __init__(self, state_size, outputs):
        super(StateValue, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, outputs)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)


def REINFORCEwithbaseline(gamma=0.99, num_episodes=100, episode_length=100000, learning_rate_policy=1e-2, render=True):
    env = gym.make('CartPole-v1')

    def select_action(states):
        probs = policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def finish_episode(rewards,log_probs,states):
        G = 0
        returns = []
        policy_loss = []
        for i in range(len(rewards)):
            G = rewards[i] + gamma * G - state_values(states[i])
            returns.insert(0,G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean())/(returns.std() + eps)

        for R, lp in zip(returns,log_probs):
            policy_loss.append(-lp*R)


        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()



        optimizer.step()

    policy = Policy(4, 2)
    state_values = StateValue(4,1)
    optimizer = optim.Adam(list(policy.parameters()) + list(state_values.parameters()), lr=1e-2)

    running_reward = 10

    for episode in range(num_episodes):
        rewards = []
        log_probs = []
        states = []
        state = env.reset()
        state = torch.tensor(state)
        states.append(state)
        ep_reward = -1
        for t in range(1,episode_length):
            action, lp = select_action(states)
            log_probs.append(lp)
            state, reward, done, _ = env.step(action)
            state=torch.tensor(state)
            states.append(state)
            rewards.append(reward)
            ep_reward += reward
            if render:
                env.render()
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(rewards,log_probs,states)

        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
              episode, ep_reward, running_reward))



if __name__ == '__main__':
    REINFORCEwithbaseline()
