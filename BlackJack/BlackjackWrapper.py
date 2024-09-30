"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024-07-06 22:07
 @Author  : Ivan Mao
 @File    : BlackjackWrapper.py
 @Description : 
"""
import random

import torch
from gym.envs.toy_text import BlackjackEnv

# from BlackJack.Trainer import Trainer


class BlackjackWrapper:
    def __init__(self):
        import gym
        self.env = BlackjackEnv(render_mode="human", natural=False, sab=False)
        self.state = self.env.reset()
        self.done = False

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def step(self, action):
        (self.state, reward, self.done, _, info) = self.env.step(action)

        return self.state, reward, self.done, info

    def show(self):
        self.env.render()

    def close(self):
        self.env.close()

    def play(self, model_actor, show=False):
        state = []
        action = []
        reward = []
        next_state = []
        over = []

        s, _ = self.env.reset()
        o = False
        # print(s)
        while not o:
            # 根据概率采样
            # print(torch.FloatTensor(s).reshape(1, 3))
            prob = model_actor(torch.FloatTensor(s).reshape(1, 3))[0].tolist()
            # print(model_actor(torch.FloatTensor(s).reshape(1, 3)),prob)
            a = random.choices(range(2), weights=prob, k=1)[0]

            (ns, r, o, _, _) = self.env.step(a)

            state.append(s)
            action.append(a)
            reward.append(r)
            next_state.append(ns)
            over.append(o)

            s = ns

            if show:
                self.env.render()

        state = torch.FloatTensor(state).reshape(-1, 3)
        action = torch.LongTensor(action).reshape(-1, 1)
        reward = torch.FloatTensor(reward).reshape(-1, 1)
        next_state = torch.FloatTensor(next_state).reshape(-1, 3)
        over = torch.LongTensor(over).reshape(-1, 1)

        return state, action, reward, next_state, over, reward.sum().item()


# if __name__ == '__main__':
#     env = BlackjackWrapper()
#     trainer = Trainer()
#     # print(env.play(trainer.actor, show=True))
#     trainer.train_actor(*env.play(trainer.actor))
