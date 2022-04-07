import numpy as np
import matplotlib.pyplot as plt

def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

class Easy21():
    """
    Implementation of Easy 21 Game following:
    https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf
    """

    def __init__(self):
        self.valid_actions = ['stick', 'hit']
        self.state = {'dealer':None, 'player':None, 'is_terminal':False, 'dealer_busted':False, 'player_busted':False}
        self.reward = 0

    def reset_state(self):
        self.state['dealer'] = np.random.randint(1,10)
        self.state['player'] = np.random.randint(1,10)
        self.state['is_terminal'] = False
        self.state['dealer_busted'] = False
        self.state['player_busted'] = False
        self.reward = 0

    def get_possible_actions(self):
        return self.valid_actions

    def step(self, action):
        assert action in self.valid_actions, 'Action: {} not a valid action!'.format(action)

        if action == 'stick':
            self.play_out()

        if action == 'hit':
            card_value = np.random.randint(1,10)
            card_colour = np.random.choice(['red', 'black'], p=[1/3, 2/3])
            if card_colour == 'red':
                self.state['player'] -= card_value
            else:
                self.state['player'] += card_value

            if self.state['player'] > 21 or self.state['player'] < 1:
                self.state['player_busted'] = True
                self.state['is_terminal'] = True

        if self.state['is_terminal']:
            if self.state['dealer'] > self.state['player'] or self.state['player_busted']:
                self.reward = -1
                return self.reward
            elif self.state['dealer'] < self.state['player'] or self.state['dealer_busted']:
                self.reward = 1
                return self.reward
            else:
                self.reward = 0
                return self.reward


    def play_out(self):
        while self.state['dealer'] < 17 and self.state['dealer'] > 0:
            card_value = np.random.randint(1,10)
            card_colour = np.random.choice(['red', 'black'], p=[1/3, 2/3])

            if card_colour == 'red':
                self.state['dealer'] -= card_value
            else:
                self.state['dealer'] += card_value


        if self.state['dealer'] < 1 or self.state['dealer'] > 21:
            self.state['dealer_busted'] = True
        self.state['is_terminal'] = True



class MonteCarloControl():
    '''
    Implementation of Monte Carlo Control
    '''

    def __init__(self, env, N_0):
        self.value_function = 0
        self.N_0 = N_0
        self.N_s = np.zeros((10,22))
        self.env = env
        self.N_sa = {action:np.zeros((10,22)) for action in self.env.get_possible_actions()}
        self.Q_sa = {action:np.zeros((10,22)) for action in self.env.get_possible_actions()}

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

        return np.random.choice([greedy_action] + actions, p = [epsilon/num_actions + 1 - epsilon, epsilon/num_actions])

    def update(self, states, actions, rewards):
        for state, action, reward in zip(states, actions, rewards):


            self.N_sa[action][state[0],state[1]] = self.get_state_action_count(state, action) + 1
            self.N_s[state[0],state[1]] = self.get_state_count(state) + 1
            self.Q_sa[action][state[0],state[1]] = self.get_state_action_value(state, action) \
                                                    + (1/self.get_state_count(state)) \
                                                    * (reward - self.get_state_action_value(state, action))



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

    MC.run(num_episodes = 100000)

    for i in ['stick', 'hit']:
        print(np.matrix(MC.Q_sa[i]))


    X = np.arange(0, 10)
    Y = np.arange(0, 22)
    X, Y = np.meshgrid(X, Y)
    Z = np.maximum(MC.Q_sa['hit'], MC.Q_sa['stick']).T
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("Value")

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
