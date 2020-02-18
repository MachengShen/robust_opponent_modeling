import numpy as np
from multiagent.core import World, Agent, Landmark, Goal
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # add goals (used only for vis)
        world.goals = [Goal() for i in range(1)]
        for i, goal in enumerate(world.goals):
            goal.name = 'goal %d' % i
            goal.collide = False
            goal.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # dk: Check the distance between agent and landmark are initialized
            # at least x distance
            while self.check_distance(world.agents, landmark.state.p_pos):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, goal in enumerate(world.goals):
            goal.state.p_pos = np.zeros(world.dim_p) - 2  # NOTE Initialize outside of the box
            goal.state.p_vel = np.zeros(world.dim_p)
            goal.color = np.array([0.0, 1.0, 0.0])

    def check_distance(self, agents, landmark_p_pos):
        assert len(agents) == 1

        threshold = 1.
            
        for i_agt, agent in enumerate(agents):
            if np.linalg.norm(agent.state.p_pos - landmark_p_pos) < threshold:
                return True
            else:
                return False

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [world.landmarks[0].state.p_pos])
