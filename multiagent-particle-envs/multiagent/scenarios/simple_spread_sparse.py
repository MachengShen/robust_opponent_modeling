import numpy as np
from multiagent.core import World, Agent, Landmark, Goal
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = num_agents
        num_goals = num_agents
        world.collaborative = True

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.20

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # add goals (used only for vis)
        world.goals = [Goal() for i in range(num_goals)]
        for i, goal in enumerate(world.goals):
            goal.name = 'goal %d' % i
            goal.collide = False
            goal.movable = False

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([1.0, 0.0, 0.0])
            elif i == 1:
                agent.color = np.array([0.0, 1.0, 0.0])
            else:
                raise NotImplementedError()

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        for i, goal in enumerate(world.goals):
            if i == 0:
                goal.color = np.array([1.0, 0.0, 0.0])
            elif i == 1:
                goal.color = np.array([0.0, 1.0, 0.0])
            else:
                raise NotImplementedError()

        for agent in world.agents:
            # agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        world.agents[0].state.p_pos = np.array([-0.25, 0.8])
        world.agents[1].state.p_pos = np.array([+0.25, 0.8])

        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        # while self.check_landmark_dist(world, th=agent.size * 2.5) is False:
        #     for i, landmark in enumerate(world.landmarks):
        #         landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #         landmark.state.p_vel = np.zeros(world.dim_p)
        world.landmarks[0].state.p_pos = np.array([-0.8, 0.0])
        world.landmarks[1].state.p_pos = np.array([+0.8, 0.0])

        for i, goal in enumerate(world.goals):
            goal.state.p_pos = np.zeros(world.dim_p) - 2  # NOTE Initialize outside of the box
            goal.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Agents are rewarded based on whether or not all landmarks are reached
        occupied_landmarks = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < agent.size:
                occupied_landmarks += 1

        if occupied_landmarks == len(world.landmarks):
            return 1.
        else:
            return 0.

    def check_landmark_dist(self, world, th):
        for i, landmark_i in enumerate(world.landmarks):
            pos_i = landmark_i.state.p_pos
            for j, landmark_j in enumerate(world.landmarks):
                if i != j:
                    pos_j = landmark_j.state.p_pos
                    dist = np.sqrt(np.sum(np.square(pos_i - pos_j)))
                    if dist < th:
                        return False
        return True

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            # entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_pos.append(entity.state.p_pos)

        other_pos = []
        for other in world.agents:
            if other is agent: 
                continue
            assert np.sum(other.state.c) == 0.  # NOTE dk: Removed comm as not used in this scenario
            # other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_pos.append(other.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
