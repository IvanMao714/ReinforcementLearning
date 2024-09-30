"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024-07-10 16:55
 @Author  : Ivan Mao
 @File    : test.py
 @Description : 
"""
import argparse
import pprint

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed


def run(args):
    # Make environment
    env = rlcard.make(
        args.env,
        config={
            'seed': 42,
        }
    )

    # Seed numpy, torch, random
    set_seed(42)

    # Set agents
    agent = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent for _ in range(env.num_players)])

    # Generate data from the environment
    trajectories, player_wins = env.run(is_training=False)
    # Print out the trajectories
    print('\nTrajectories:')
    print(trajectories)
    print('\nSample raw observation:')
    pprint.pprint(trajectories[0][0]['raw_obs'])
    print('\nSample raw legal_actions:')
    pprint.pprint(trajectories[0][0]['raw_legal_actions'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='blackjack',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )

    args = parser.parse_args()

    run(args)
