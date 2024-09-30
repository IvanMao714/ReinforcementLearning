"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024-07-06 11:26
 @Author  : Ivan Mao
 @File    : ll_test.py
 @Description : 
"""
import gym
env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("=====================================")
    print("Observation:", observation, "Reward:", reward, "Terminated:", terminated, "Truncated:", truncated, "Info:", info)
    action = env.action_space.sample()
    env.step(action)
    print("Action:", action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()