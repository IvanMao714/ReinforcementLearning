"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024-09-28 22:19
 @Author  : Ivan Mao
 @File    : bg_explore.py
 @Description : 
"""
import gym
import numpy as np

env = gym.make('Blackjack-v1')
print(env.observation_space)
print(env.action_space)

for i_episode in range(3):
    state = env.reset()
    while True:
        # print(state)
        action = env.action_space.sample()
        print("Action is:", action)
        # print(env.step(action))
        # env.step() return 5 values: state, reward, done, False, info
        state, reward, done, _, info = env.step(action)
        if done:
            print('End game! Reward: ', reward)
            print('You won :)\n') if reward > 0 else print('You lost :(\n')
            break
