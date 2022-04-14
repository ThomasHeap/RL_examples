import numpy as np
import matplotlib.pyplot as plt
from env import Easy21

class SarsaLFAControl():
    """
    Implementation of sarsa(\lambda) control
    """

    def __init__(self, env, N_0, lm):
        self.N_0 = N_0
        self.lm = lm
        self.N_s = np.zeros((11,22))
        self.env = env
        self.N_sa = {action:np.zeros((11,22)) for action in self.env.get_possible_actions()}
        self.theta = np.random.randn(36,1)
        self.E_sa = np.zeros_like(self.theta)

    def features(self, state, action):
        f = np.zeros(3*6*2)

        for fi, (lower, upper) in enumerate(zip(range(1,8,3), range(4,11,3))):
            f[fi] = (lower <= state[0] <= upper)

        for fi, (lower, upper) in enumerate(zip(range(1,17,3), range(6,22,3)), start=3):
            f[fi] = (lower <= state[1] <= upper)

        f[-2] = action == 'hit'
        f[-1] = action == 'stick'

        return f.reshape(1,-1)

    def get_state_action_value(self, state, action):
        return self.features(state,action).dot(self.theta)

    def allQ(self):
        allFeatures = np.zeros((22, 11, 2, 3*6*2))
        actions = self.env.get_possible_actions()
        for i in range(1,22):
            for j in range(1,11):
                for a in range(0,2):
                    allFeatures[i-1, j-1, a] = self.features([i,j],actions[a])


        allFeatures = allFeatures.max(axis=2)
        return allFeatures.reshape(-1,3*6*2).dot(self.theta).reshape(-1)


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
        epsilon = 0.05

        #Different ways of choosing actions
        return np.random.choice([greedy_action] + actions, p = [epsilon/num_actions + 1 - epsilon, epsilon/num_actions])

    # def update(self, states, actions):
    #     for s, a in zip(states, actions):
    #         self.E_sa[a][s[0],s[1]] = self.lm * self.get_eligibility(s, a)


    def run(self, num_episodes):
        mse = []
        Q_true = np.load('Q_MC.npy')
        theta = np.random.randn(3*6*2, 1)
        for i in range(num_episodes):

            mse.append(((Q_true.ravel() - self.allQ())**2).sum() / (21*10*2))
            self.env.reset_state()
            self.E_sa = np.zeros_like(self.theta)
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
                    self.delta = self.env.reward + self.get_state_action_value(s_, a_)[0] \
                            - self.get_state_action_value(s, a)[0]
                else:
                    self.delta = self.env.reward - self.get_state_action_value(s, a)[0]

                states.append(s)
                actions.append(a)
                #updating
                self.N_sa[a][s[0],s[1]] += 1
                self.N_s[s[0],s[1]] += 1

                self.E_sa = self.lm * self.E_sa + self.features(s,a).reshape(-1,1)

                #print(self.delta)
                #print(self.E_sa)


                self.theta += (0.01) * self.delta * self.E_sa



                if not self.env.state['is_terminal']:
                    s = s_
                    a = a_




        return mse


if __name__ == "__main__":

    Q_true = np.load('Q_MC.npy')


    errors = []
    for lm in np.arange(0,1.1,0.1):

        easy21 = Easy21()
        lfa = SarsaLFAControl(env=easy21, N_0=100, lm = lm)
        mse = lfa.run(num_episodes = 10000)

        print(len(mse))
        errors.append(((Q_true.ravel() - lfa.allQ())**2).sum() / (21*10*2))

        if lm == 0.1 or lm == 1:
            fig, ax = plt.subplots()
            ax.plot(np.arange(0,10000,1), mse)
            plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.arange(0,1.1,0.1), errors)
    plt.show()
