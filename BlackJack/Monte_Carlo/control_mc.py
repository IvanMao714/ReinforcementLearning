"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024-09-28 23:40
 @Author  : Ivan Mao
 @File    : control_mc.py
 @Description : 
"""
import sys
from collections import defaultdict
import gym
import numpy as np

from BlackJack.Monte_Carlo.plot_utils import plot_blackjack_values


class LearningMC(object):
    def __init__(self):
        self.episodes = 50000
        self.env = gym.make('Blackjack-v1')
        self.returns_sum = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.V = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def generate_episode_from_V(self, V, epsilon):
        """ generates an episode from following the epsilon-greedy policy """
        episode = []
        state = self.env.reset()[0]
        while True:
            action = np.random.choice(np.arange(self.env.action_space.n),
                                      p=self.get_probs(V[state], epsilon, self.env.action_space.n)) \
                if state in V else self.env.action_space.sample()
            next_state, reward, done, _, info = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

    def get_probs(self, Q_s, epsilon, nA):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(nA) * epsilon / nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s

    def update_V(self, env, episode, V, alpha, gamma):
        """ updates the action-value function estimate using the most recent episode """
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            old_V = V[state][actions[i]]
            V[state][actions[i]] = old_V + alpha * (sum(rewards[i:] * discounts[:-(1 + i)]) - old_V)
        return V

    def mc_control(self, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
        nA = self.env.action_space.n
        # initialize empty dictionary of arrays
        V = defaultdict(lambda: np.zeros(nA))
        epsilon = eps_start
        # loop over episodes
        for i_episode in range(1, self.episodes + 1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, self.episodes), end="")
                sys.stdout.flush()
            # set the value of epsilon
            epsilon = max(epsilon * eps_decay, eps_min)
            # generate an episode by following epsilon-greedy policy
            episode = self.generate_episode_from_V(V, epsilon)
            # update the action-value function estimate using the episode
            V = self.update_V(self.env, episode, V, alpha, gamma)
        # determine the policy corresponding to the final action-value function estimate
        policy = dict((k, np.argmax(v)) for k, v in V.items())
        return policy, V


if __name__ == '__main__':
    policy, V = LearningMC().mc_control(0.02)
    V = dict((k, np.max(v)) for k, v in V.items())
    plot_blackjack_values(V)
