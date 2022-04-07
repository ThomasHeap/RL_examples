import numpy as np
import matplotlib.pyplot as plt
from env import Easy21

class MonteCarloControl():
    '''
    Implementation of Monte Carlo Control
    '''

    def __init__(self, env, N_0):
        self.value_function = 0
        self.N_0 = N_0
        self.N_s = np.zeros((11,22))
        self.env = env
        self.N_sa = {action:np.zeros((11,22)) for action in self.env.get_possible_actions()}
        self.Q_sa = {action:np.zeros((11,22)) for action in self.env.get_possible_actions()}

    def get_state_action_value(self, state, action):
        #return self.Q_sa.get(state[0], {state[1]: {action: 0}}).get(state[1], {state[1]:action}).get(action, 0)
        return self.Q_sa[action][state[0],state[1]]

    def get_state_action_count(self, state, action):
        #return self.N_sa.get(state[0], {state[1]: {action: 0}}).get(state[1], {state[1]:action}).get(action, 0)
        return self.N_sa[action][state[0],state[1]]

    def get_state_count(self, state):
        #return self.N_s.get(state[0], {state[1]:0}).get(state[1], 0)
        return self.N_s[state[0],state[1]]


    def policy(self, state):
        actions = self.env.get_possible_actions().copy()
        num_actions = len(actions)
        greedy_action = str(max(actions, key=lambda a: self.get_state_action_value(state,a)))
        actions.remove(greedy_action)
        epsilon = self.N_0 / (self.N_0 + self.get_state_count(state))


        #Different ways of choosing actions
        #return np.random.choice([greedy_action] + actions, p = [epsilon/num_actions + 1 - epsilon, epsilon/num_actions])
        return np.random.choice([greedy_action] + actions, p = [1 - epsilon, epsilon])

    def update(self, states, actions, rewards):
        i = 0
        for state, action in zip(states, actions):


            self.N_sa[action][state[0],state[1]] = self.get_state_action_count(state, action) + 1
            self.N_s[state[0],state[1]] = self.get_state_count(state) + 1
            self.Q_sa[action][state[0],state[1]] = self.get_state_action_value(state, action) \
                                                    + (1/self.get_state_count(state)) \
                                                    * (sum(rewards[i:]) - self.get_state_action_value(state, action))
            i += 1



    def run(self, num_episodes):

        for i in range(num_episodes):
            self.env.reset_state()
            states = []
            actions = []
            rewards = []
            while True:
                current_state = [self.env.state['dealer'], self.env.state['player']]
                action = self.policy(current_state)
                states.append(current_state)
                actions.append(action)
                self.env.step(action)
                rewards.append(self.env.reward)
                if self.env.state['is_terminal']:
                    break
            self.update(states, actions,rewards)


if __name__ == "__main__":


    easy21 = Easy21()
    MC = MonteCarloControl(env=easy21, N_0=100)

    MC.run(num_episodes = 1000000)


    X = np.arange(0, 11).T
    Y = np.arange(0, 22).T
    X, Y = np.meshgrid(X, Y)
    Z = np.maximum(MC.Q_sa['hit'], MC.Q_sa['stick']).T
    np.save("Q_MC.npy", Z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X[1:,:], Y[1:,:], Z[1:,:], rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    ax.set_zlim(-0.6, 0.6)
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("Value")

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
