##############################################
# Implementation of mdp Solver inspired by
# https://colab.research.google.com/github/yandexdataschool/Practical_RL/blob/master/week02_value_based/seminar_vi.ipynb
##############################################


import numpy as np

from mdp import MDP
from mdp import plot_graph, plot_graph_with_state_values, plot_graph_optimal_strategy_and_state_values

class Solver():
    """
    Value Iteration solver for MDP
    """
    def __init__(self, mdp, gamma):
        """
        param mdp: Markov Decision Process
        param gamma: 0.0 < gamma =< 1.0
            Discount for mdp
        """
        self.mdp = mdp
        self.gamma = gamma

    def get_action_value(self, state_values, state, action):
        """Compute Q(s,a)"""

        next_states = self.mdp.get_next_states(state, action)

        q = sum([self.mdp.get_transition_prob(state, action, next_state) \
                * (self.mdp.get_rewards(state, action, next_state) + self.gamma * state_values[next_state]) for next_state in next_states])

        return q

    def get_new_state_value(state_values, state):
        """Calculate max_a q(s,a)"""
        if self.mdp.is_terminal(state):
            return 0

        possible_actions = self.mdp.get_possible_actions(state)

        v = max([self.get_action_value(state_values, state, action) for action in possible_actions])

        return v

    def get_optimal_action(state_values, state):
        """ Finds optimal action using formula above. """
        if mdp.is_terminal(state): return None

        optimal_actions = {a: get_action_value(state_values, state, a) for a in self.mdp.get_possible_actions(state)}
        optimal_actions = max(optimal_actions, key=optimal_actions.get)
        return optimal_actions

    def compute_vpi(self, policy):
        """
        Computes V^pi(s) FOR ALL STATES under given policy.
        :param policy: a dict of currently chosen actions {s : a}
        :returns: a dict {state : V^pi(state) for all states}
        """
        states = self.mdp.get_all_states()

        r_pi = np.array([sum([self.mdp.get_rewards(s, policy[s], next_s) for next_s in self.mdp.get_next_states(s, policy[s])]) for s in states])
        p_pi = np.array([[self.mdp.transition(s, policy[s], next_s) for next_s in states] for s in states])
        v_pi = np.linalg.solve((np.eye(len(states)) - self.gamma * p_pi), r_pi.T)


        return {states[i]: v_pi[i] for i in range(len(states))}

    def compute_new_policy(vpi):
        """
        Computes new policy as argmax of state values
        :param vpi: a dict {state : V^pi(state) for all states}
        :returns: a dict {state : optimal action for all states}
        """

        return return {state: get_optimal_action(vpi, state) for state in self.mdp.get_all_states()}

    def solve_vi(num_iters, min_diff):
        """
        Solve MDP using value iteration.

        param num_iters: Number of iterations for solver to run
        param min_diff: Minimum difference between subsequent values
        """

        #init values
        state_values = {s: 0 for s in self.mdp.get_all_states()}

        #display MDP
        plot_graph_with_state_values(mdp, state_values)

        for i in range(num_iters):

            new_state_values = {s: self.get_new_state_value(s) for s in self.mdp.get_all_states()}

            #compute difference
            diff = max(abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states())

            print("iter %4i   |   diff: %6.5f   |   " % (i, diff), end="")
            print('   '.join("V(%s) = %.3f" % (s, v) for s, v in state_values.items()))
            state_values = new_state_values

            if diff < min_difference:
                print("Terminated")
                break

        #return state_values, number of iterations taken, and difference
        return state_values, i, diff

    def solve_pi(policy=None, num_iter=1000, min_difference=1e-5):
        """
        Run the policy iteration loop for num_iter iterations or till difference between V(s) is below min_difference.
        If policy is not given, initialize it at random.
        """

        #init policy and state
        policy = policy if policy is not None else {s: np.random.choice(mdp.get_possible_actions(s)) for s in mdp.get_all_states()}
        state_values = {s: 0 for s in mdp.get_all_states()}

        for i in range(num_iter):

            new_state_values = compute_vpi(mdp, policy, gamma)
            #compute difference
            diff = max(abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states())

            print("iter %4i   |   diff: %6.5f   |   " % (i, diff), end="")
            print('   '.join("V(%s) = %.3f" % (s, v) for s, v in state_values.items()))
            state_values = new_state_values

            if diff < min_difference:
                print("Terminated")
                break
            policy = compute_new_policy(mdp, state_values, gamma)

        #return state_values, policy
        return state_values, policy
