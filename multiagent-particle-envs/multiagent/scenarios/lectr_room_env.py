import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, mode=0):
        world = World()
        self.mode = mode

        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 2
        world.collaborative = True
        assert num_agents == 2, "This env is only for two agents"

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([1.0, 0.0, 0.0])
            elif i == 1:
                agent.color = np.array([0.0, 1.0, 0.0])
            else:
                raise ValueError()

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.state.p_pos = np.array([-0.75, 0.])
            elif i == 1:
                landmark.state.p_pos = np.array([+0.75, 0.])
            else:
                raise ValueError()
            landmark.state.p_vel = np.zeros(world.dim_p)

        for i, agent in enumerate(world.agents):
            if self.mode == 0:
                if i == 0:
                    agent.state.p_pos = np.array([-0.50, 0.])
                elif i == 1:
                    agent.state.p_pos = np.array([+0.50, 0.])
                else:
                    raise ValueError()
            elif self.mode == 1:
                if i == 0:
                    agent.state.p_pos = np.array([+0.50, 0.])
                elif i == 1:
                    agent.state.p_pos = np.array([-0.50, 0.])
                else:
                    raise ValueError()
            else:
                raise ValueError()

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def reward(self, agent, world):
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
