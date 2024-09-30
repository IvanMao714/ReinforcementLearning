"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024-07-06 22:12
 @Author  : Ivan Mao
 @File    : Trainer.py
 @Description : 
"""
import torch

from BlackJack.BlackjackWrapper import BlackjackWrapper


def requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad_(value)


class Trainer:
    def __init__(self,
                 actor=torch.nn.Sequential(
                     torch.nn.Linear(3, 64),
                     torch.nn.ReLU(),
                     torch.nn.Linear(64, 64),
                     torch.nn.ReLU(),
                     torch.nn.Linear(64, 2),
                     torch.nn.Softmax(dim=1), ),
                 critic=torch.nn.Sequential(
                     torch.nn.Linear(3, 64),
                     torch.nn.ReLU(),
                     torch.nn.Linear(64, 64),
                     torch.nn.ReLU(),
                     torch.nn.Linear(64, 1),
                     torch.nn.Softmax(dim=1), ),
                 critic_delay=torch.nn.Sequential(
                     torch.nn.Linear(3, 64),
                     torch.nn.ReLU(),
                     torch.nn.Linear(64, 64),
                     torch.nn.ReLU(),
                     torch.nn.Linear(64, 1),
                     torch.nn.Softmax(dim=1), ),
                 ):
        self.actor = actor
        self.critic = critic
        self.critic_delay = critic_delay
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-10)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-10)
        self.env = BlackjackWrapper()

    def train_critic(self, state, reward, next_state, over):
        requires_grad(self.actor, False)
        requires_grad(self.critic, True)

        # 计算values和targets
        value = self.critic(state)

        with torch.no_grad():
            target = self.critic_delay(next_state)
        target = target * 0.99 * (1 - over) + reward

        # 时序差分误差,也就是tdloss
        loss = torch.nn.functional.mse_loss(value, target)

        loss.backward()
        self.optimizer_critic.step()
        self.optimizer_critic.zero_grad()

        return (target - value).detach()

    def train_actor(self, state, action, value):
        requires_grad(self.actor, True)
        requires_grad(self.critic, False)

        # 重新计算动作的概率
        prob = self.actor(state)
        prob = prob.gather(dim=1, index=action)
        # print("train_actor:prob:", prob)
        # 根据策略梯度算法的导函数实现
        # 函数中的Q(state,action),这里使用critic模型估算
        prob = (prob + 1e-8).log() * value
        loss = -prob.mean()

        loss.backward()
        self.optimizer_actor.step()
        self.optimizer_actor.zero_grad()

        return loss.item()

    def train(self,):
        self.actor.train()
        self.critic.train()

        # 训练N局
        for epoch in range(1000):

            # 一个epoch最少玩N步
            steps = 0
            for _ in range(10):
                state, action, reward, next_state, over, _ = self.env.play(self.actor)
                steps += len(state)

                # 训练两个模型
                value = self.train_critic(state, reward, next_state, over)
                loss = self.train_actor(state, action, value)
                # print('Epoch:', epoch, 'Step:', steps, 'Loss:', loss)

            # 复制参数
            for param, param_delay in zip(self.critic.parameters(),
                                          self.critic_delay.parameters()):
                value = param_delay.data * 0.7 + param.data * 0.3
                param_delay.data.copy_(value)

            if epoch % 10 == 0:
                test_result = sum([self.env.play(self.actor)[-1] for _ in range(20)]) / 20
                print(epoch, loss, test_result)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
