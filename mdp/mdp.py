##############################################
# Implementation of MDP inspired (heavily) by https://github.com/yandexdataschool/Practical_RL/blob/master/week02_value_based/mdp.py
# Implementation of graph plotting of MDP taken wholesale
##############################################

import numpy
import graphviz

class MDP():
    '''
    Implementation of MDP
    '''

    def __init__(self, transitions, rewards, initial_state=None, seed=None):
        """
        :param transitions: transitions[s][a][s'] = p(s_t+1 = s' | s_t = s, a_t=a)
            a dict[state->dict] of dicts[action->dict] of dicts[next_state->probability]
        :params rewards: rewards[s][a][s'] = r(s,a,s')
            a dict[state->dict] of dicts[action->rewards] of dicts[next_state->rewards]
        :params initial_state: starting state
        """

        self.transitions = transitions
        self.rewards = rewards
        self.initial_state = initial_state
        self.num_states = len(self.transitions)
        np.seed(seed)
        self.reset()

    def get_all_states(self):

        return tuple(self.transitions.keys())

    def get_possible_actions(self, state):

        return tuple(self.transitions.get(state, {}).keys())

    def get_next_states(self, state, action):

        assert action in self.get_possible_actions(state), "cannot do action {0} in state {1}".format(action, state)
        return self.transitions[state][action]

    def get_rewards(self, state, action, next_state):

        assert action in self.get_possible_actions(state), "cannot do action {0} in state {1}".format(action, state)
        return self.rewards.get(state, {}).get(action, {}).get(next_state, 0.0)

    def get_transition_prob(self, state, action, next_state):

        assert action in self.get_possible_actions(state), "cannot do action {0} in state {1}".format(action, state)
        return self.get_next_states(state, action).get(next_state. 0.0)

    def is_terminal(self, state):

        return len(self.get_possible_actions(state)) == 0

    def reset(self):
        if self.initial_state is None:
            self.current_state = np.random.choice(self.get_all_states())
        elif self.initial_state in self.get_all_states():
            self.current_state = self.initial_state

    def step(self, action):
        possible_states, probs = zip(*self.get_next_states(self.current_state, action).items())
        next_state = possible_states[np.random.choice(np.arange(len(possible_states)),p=probs)]
        reward = self.get_rewards(state, action, next_state)
        is_done = self.is_terminal(next_state)
        self.current_state = next_state

        #next_state, reward, is_done, empty info
        return next_state, reward, is_done, {}


def plot_graph(mdp, graph_size='10,10', s_node_size='1,5',
               a_node_size='0,5', rankdir='LR', ):
    """
    Function for pretty drawing MDP graph with graphviz library.
    Requirements:
    graphviz : https://www.graphviz.org/
    for ubuntu users: sudo apt-get install graphviz
    python library for graphviz
    for pip users: pip install graphviz
    :param mdp:
    :param graph_size: size of graph plot
    :param s_node_size: size of state nodes
    :param a_node_size: size of action nodes
    :param rankdir: order for drawing
    :return: dot object
    """
    s_node_attrs = {'shape': 'doublecircle',
                    'color': '#85ff75',
                    'style': 'filled',
                    'width': str(s_node_size),
                    'height': str(s_node_size),
                    'fontname': 'Arial',
                    'fontsize': '24'}

    a_node_attrs = {'shape': 'circle',
                    'color': 'lightpink',
                    'style': 'filled',
                    'width': str(a_node_size),
                    'height': str(a_node_size),
                    'fontname': 'Arial',
                    'fontsize': '20'}

    s_a_edge_attrs = {'style': 'bold',
                      'color': 'red',
                      'ratio': 'auto'}

    a_s_edge_attrs = {'style': 'dashed',
                      'color': 'blue',
                      'ratio': 'auto',
                      'fontname': 'Arial',
                      'fontsize': '16'}

    graph = Digraph(name='MDP')
    graph.attr(rankdir=rankdir, size=graph_size)
    for state_node in mdp._transition_probs:
        graph.node(state_node, **s_node_attrs)

        for posible_action in mdp.get_possible_actions(state_node):
            action_node = state_node + "-" + posible_action
            graph.node(action_node,
                       label=str(posible_action),
                       **a_node_attrs)
            graph.edge(state_node, state_node + "-" +
                       posible_action, **s_a_edge_attrs)

            for posible_next_state in mdp.get_next_states(state_node,
                                                          posible_action):
                probability = mdp.get_transition_prob(
                    state_node, posible_action, posible_next_state)
                reward = mdp.get_rewards(
                    state_node, posible_action, posible_next_state)

                if reward != 0:
                    label_a_s_edge = 'p = ' + str(probability) + \
                                     '  ' + 'reward =' + str(reward)
                else:
                    label_a_s_edge = 'p = ' + str(probability)

                graph.edge(action_node, posible_next_state,
                           label=label_a_s_edge, **a_s_edge_attrs)
    return graph


def plot_graph_with_state_values(mdp, state_values):
    """ Plot graph with state values"""
    graph = plot_graph(mdp)
    for state_node in mdp._transition_probs:
        value = state_values[state_node]
        graph.node(state_node, label=str(state_node) + '\n' + 'V =' + str(value)[:4])
    return graph


def get_optimal_action_for_plot(mdp, state_values, state, get_action_value, gamma=0.9):
    """ Finds optimal action using formula above. """
    if mdp.is_terminal(state):
        return None
    next_actions = mdp.get_possible_actions(state)
    q_values = [get_action_value(mdp, state_values, state, action, gamma) for action in next_actions]
    optimal_action = next_actions[np.argmax(q_values)]
    return optimal_action


def plot_graph_optimal_strategy_and_state_values(mdp, state_values, get_action_value, gamma=0.9):
    """ Plot graph with state values and """
    graph = plot_graph(mdp)
    opt_s_a_edge_attrs = {'style': 'bold',
                          'color': 'green',
                          'ratio': 'auto',
                          'penwidth': '6'}

    for state_node in mdp._transition_probs:
        value = state_values[state_node]
        graph.node(state_node, label=str(state_node) + '\n' + 'V =' + str(value)[:4])
        for action in mdp.get_possible_actions(state_node):
            if action == get_optimal_action_for_plot(mdp, state_values, state_node, get_action_value, gamma):
                graph.edge(state_node, state_node + "-" + action, **opt_s_a_edge_attrs)
    return graph
