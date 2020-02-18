import numpy as np
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios


if __name__ == '__main__':
    scenario = scenarios.load("maze_push.py").Scenario()
    world = scenario.make_world(mode=0)
    done_callback= None

    env = MultiAgentEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        done_callback=done_callback)

    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        act_n.append(np.array([0., 0.]))
        act_n.append(np.array([0., 0.]))

        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)

        # render all agent views
        env.render()

        # # display rewards
        # for agent in env.world.agents:
        #     print(agent.name + " reward: %0.3f" % env._get_reward(agent))
