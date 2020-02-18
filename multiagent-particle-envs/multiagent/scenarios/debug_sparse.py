import numpy as np
from multiagent.core import World, Agent, Landmark, Goal
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, num_agents=1, collide=True):
        world = World()

        # set any world properties first
        world.dim_c = 2
        num_agents = num_agents
        num_landmarks = num_agents
        num_goals = num_agents
        world.collaborative = True

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = collide
            agent.silent = True
            agent.size = 0.50

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
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([1.0, 0.0, 0.0])
            elif i == 1:
                agent.color = np.array([0.0, 1.0, 0.0])
            elif i == 2:
                agent.color = np.array([0.0, 0.0, 1.0])
            else:
                raise NotImplementedError()

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        for i, goal in enumerate(world.goals):
            if i == 0:
                goal.color = np.array([1.0, 0.0, 0.0])
            elif i == 1:
                goal.color = np.array([0.0, 1.0, 0.0])
            elif i == 2:
                goal.color = np.array([0.0, 0.0, 1.0])
            else:
                raise NotImplementedError()

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # world.agents[0].state.p_pos = np.array([-0.1, 0.0])
        # world.agents[1].state.p_pos = np.array([+0.1, 0.0])

        # For landmark, we make sure that their distance is at least some size
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        while self.check_landmark_dist(world, agent.size * 2) is False:
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
        
        # world.landmarks[0].state.p_pos = np.array([-0.8, 0.0])
        # world.landmarks[1].state.p_pos = np.array([+0.8, 0.0])

        for i, goal in enumerate(world.goals):
            goal.state.p_pos = np.zeros(world.dim_p) - 2  # Initialize outside of the box
            goal.state.p_vel = np.zeros(world.dim_p)

    def check_landmark_dist(self, world, size):
        for i, landmark_i in enumerate(world.landmarks):
            pos_i = landmark_i.state.p_pos

            for j, landmark_j in enumerate(world.landmarks):
                if i != j:
                    pos_j = landmark_j.state.p_pos
                    dist = np.sqrt(np.sum(np.square(pos_i - pos_j)))
                    if dist < size:
                        return False

        return True

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            if min(dists) < agent.size:
                occupied_landmarks += 1
        if occupied_landmarks == len(world.landmarks):
            rew = 1
        else:
            rew = -0.01
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

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

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: 
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    def done(self, agent, world):
        done_flag = False

        # Check whether all landmarks are occupied
        occupied_landmarks = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < agent.size:
                occupied_landmarks += 1
        if occupied_landmarks == len(world.landmarks):
            done_flag = True

        # NOTE Check for max_step is done in the main train loop
        return done_flag
