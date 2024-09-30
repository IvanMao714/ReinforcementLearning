"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024-07-06 15:43
 @Author  : Ivan Mao
 @File    : bw_explore.py
 @Description : 
"""
import gym
import pygame
import time

from gym.envs.toy_text import BlackjackEnv

game = BlackjackEnv(render_mode="human", natural=False, sab=False)
game.reset()
while True:
    action = game.action_space.sample()
    print("Action:", action)
    print(game.step(action))
    (state, reward, done, _, info) = game.step(action)
    print("state:", state, "reward:", reward, "done:", done)
    if done:
        time.sleep(200000)
    game.render()
