import numpy as np
import matplotlib.pyplot as plt
from env import Easy21

class SarsaControl():
    """
    Implementation of sarsa(\lambda) control
    """

    def __init__(self, env, N_0, lm):
        self.N_0 = N_0
        self.lm = lm
        self.N_s = np.zeros((11,22))
        self.env = env
        self.N_sa = {action:np.zeros((11,22)) for action in self.env.get_possible_actions()}
        self.Q_sa = {action:np.zeros((11,22)) for action in self.env.get_possible_actions()}
        self.E_sa = {action:np.zeros((11,22)) for action in self.env.get_possible_actions()}

    def get_state_action_value(self, state, action):
        #return self.Q_sa.get(state[0], {state[1]: {action: 0}}).get(state[1], {state[1]:action}).get(action, 0)
        return self.Q_sa[action][state[0],state[1]]

    def get_state_action_count(self, state, action):
        #return self.N_sa.get(state[0], {state[1]: {action: 0}}).get(state[1], {state[1]:action}).get(action, 0)
        return self.N_sa[action][state[0],state[1]]

    def get_eligibility(self, state, action):
        #return self.N_sa.get(state[0], {state[1]: {action: 0}}).get(state[1], {state[1]:action}).get(action, 0)
        return self.E_sa[action][state[0],state[1]]

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
        return np.random.choice([greedy_action] + actions, p = [epsilon/num_actions + 1 - epsilon, epsilon/num_actions])

    def update(self, states, actions):
        for s, a in zip(states, actions):
            # self.E_sa[a][s[0],s[1]] = self.lm * self.get_eligibility(s, a)
            self.Q_sa[a][s[0],s[1]] = self.get_state_action_value(s, a) \
                                    + (1/self.get_state_action_count(s, a)) \
                                    * (self.delta * self.get_eligibility(s, a))




    def run(self, num_episodes):
        mse = []
        Q_true = np.load('Q_MC.npy')
        for i in range(num_episodes):

            mse.append((((Q_true.T - np.maximum(self.Q_sa['hit'], self.Q_sa['stick']))**2).sum() / (21*10*2)))
            self.env.reset_state()
            self.E_sa = {action:np.zeros((11,22)) for action in self.env.get_possible_actions()}
            states = []
            actions = []
            self.delta = 0
            s = [self.env.state['dealer'], self.env.state['player']]
            a = self.policy(s)




            while not self.env.state['is_terminal']:

                self.env.step(a)
                s_ = [self.env.state['dealer'], self.env.state['player']]


                if not self.env.state['is_terminal']:
                    a_ = self.policy(s_)
                    self.delta = self.env.reward + self.get_state_action_value(s_, a_) \
                            - self.get_state_action_value(s, a)
                else:
                    self.delta = self.env.reward - self.get_state_action_value(s, a)

                states.append(s)
                actions.append(a)
                #updating
                self.E_sa[a][s[0],s[1]] += 1
                self.N_sa[a][s[0],s[1]] += 1
                self.N_s[s[0],s[1]] += 1


                self.update(states,actions)
                for a in self.env.get_possible_actions():
                    self.E_sa[a] = self.lm * self.E_sa[a]

                if not self.env.state['is_terminal']:
                    s = s_
                    a = a_




        return mse


if __name__ == "__main__":

    Q_true = np.load('Q_MC.npy')


    errors = []
    for lm in np.arange(0,1.1,0.1):

        easy21 = Easy21()
        sarsa = SarsaControl(env=easy21, N_0=100, lm = lm)
        mse = sarsa.run(num_episodes = 10000)

        print(len(mse))
        errors.append(((Q_true.T - np.maximum(sarsa.Q_sa['hit'], sarsa.Q_sa['stick']))**2).sum() / (21*10*2))

        if lm == 0.1 or lm == 1:
            fig, ax = plt.subplots()
            ax.plot(np.arange(0,10000,1), mse)
            plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.arange(0,1.1,0.1), errors)
    plt.show()
