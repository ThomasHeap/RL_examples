import numpy as np

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
        self.state['dealer'] = np.random.randint(1,11)
        self.state['player'] = np.random.randint(1,11)
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
            card_value = np.random.randint(1,11)
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
            card_value = np.random.randint(1,11)
            card_colour = np.random.choice(['red', 'black'], p=[1/3, 2/3])

            if card_colour == 'red':
                self.state['dealer'] -= card_value
            else:
                self.state['dealer'] += card_value


        if self.state['dealer'] < 1 or self.state['dealer'] > 21:
            self.state['dealer_busted'] = True
        self.state['is_terminal'] = True
