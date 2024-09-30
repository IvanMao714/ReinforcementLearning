"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024-09-28 22:15
 @Author  : Ivan Mao
 @File    : learning_mc.py
 @Description : 
"""
import sys
from collections import defaultdict

import gym
import numpy as np

from BlackJack.Monte_Carlo.plot_utils import plot_blackjack_values


class LearningMC(object):
    def __init__(self):
        # self.values = {}
        # self.episodes = 100
        self.episodes = 50000
        self.env = gym.make('Blackjack-v1')
        self.returns_sum = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.V = defaultdict(lambda: np.zeros(self.env.action_space.n))

    # generate episode from the environment stochastic policy
    def generate_episode_from_limit_stochastic(self):
        episode = []
        state = self.env.reset()[0]
        # print("state:", state)
        while True:
            # the policy
            probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
            action = np.random.choice(np.arange(2), p=probs)
            next_state, reward, done, _, info = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

    def mc_prediction_v(self, gamma=1.0):
        # loop over episodes
        for i_episode in range(1, self.episodes + 1):
            # monitor progress
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, self.episodes), end="")
                sys.stdout.flush()
            # generate an episode
            episode = self.generate_episode_from_limit_stochastic()
            # obtain the states, actions, and rewards
            states, actions, rewards = zip(*episode)
            # print("States:", states, "Actions:", actions, "Rewards", rewards)
            # prepare for discounting
            discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
            # update the sum of the returns, number of visits, and action-value
            # function estimates for each state-action pair in the episode
            for i, state in enumerate(states):
                # print("state:", state, "i:", i)
                # print("rewards:", rewards, "discounts:", discounts)
                # print("rewards[i:]:", rewards[i:], "discounts[:-(1 + i)]:", discounts[:-(1 + i)])
                self.returns_sum[state][actions[i]] += sum(rewards[i:] * discounts[:-(1 + i)])
                self.N[state][actions[i]] += 1.0
                self.V[state][actions[i]] = self.returns_sum[state][actions[i]] / self.N[state][actions[i]]
                # print('N:', N)
        return self.V

    import sys
    import numpy as np
    from collections import defaultdict

    def mc_prediction_fv(self, gamma=1.0):
        """
        使用首次访问（First-Visit）蒙特卡洛方法来预测状态-动作值函数 V，
        并采用倒序遍历优化回报计算。

        参数:
        - gamma: 折扣因子

        返回:
        - self.V: 更新后的状态-动作值函数
        """
        # 循环遍历所有回合
        for i_episode in range(1, self.episodes + 1):
            # 监控进度
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, self.episodes), end="")
                sys.stdout.flush()

            # 生成一个回合
            episode = self.generate_episode_from_limit_stochastic()

            # 获取状态、动作和奖励
            states, actions, rewards = zip(*episode)

            # 准备折扣因子
            discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])

            # 使用集合来跟踪已经访问过的状态-动作对
            visited_state_actions = set()

            # 累积回报初始化
            G = 0.0

            # 倒序遍历回合
            for i in reversed(range(len(states))):
                state = states[i]
                action = actions[i]
                reward = rewards[i]

                # 累积回报更新
                G = reward + gamma * G

                state_action = (state, action)

                # 如果这个状态-动作对是首次访问（倒序中首次遇到）
                if state_action not in visited_state_actions:
                    # 更新累计回报和访问次数
                    self.returns_sum[state][action] += G
                    self.N[state][action] += 1.0

                    # 更新状态-动作值函数 V
                    self.V[state][action] = self.returns_sum[state][action] / self.N[state][action]

                    # 将该状态-动作对标记为已访问
                    visited_state_actions.add(state_action)

        return self.V


if __name__ == '__main__':
    fmc = LearningMC()
    # for i in range(3):
    #     print(fmc.generate_episode_from_limit_stochastic())
    Q = fmc.mc_prediction_v(0.8)
    # obtain the corresponding state-value function
    V_to_plot = dict((k, (k[0] > 18) * (np.dot([0.8, 0.2], v)) + (k[0] <= 18) * (np.dot([0.2, 0.8], v))) for k, v in Q.items())
    # print("V_to_plot:", V_to_plot)
    # plot the state-value function
    plot_blackjack_values(V_to_plot)

    fQ = fmc.mc_prediction_fv(0.8)
    fV_to_plot = dict((k, (k[0] > 18) * (np.dot([0.8, 0.2], v)) + (k[0] <= 18) * (np.dot([0.2, 0.8], v))) for k, v in fQ.items())
    # print("fV_to_plot:", fV_to_plot)
    plot_blackjack_values(fV_to_plot)

