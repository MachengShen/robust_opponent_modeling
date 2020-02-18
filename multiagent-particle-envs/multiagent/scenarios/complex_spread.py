import numpy as np
from multiagent.core import World, Agent, Landmark, Goal
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = num_agents + 1
        num_goals = num_agents
        world.collaborative = True

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            # Add landmarks
            if i == 0 or i == 1:
                landmark.name = 'landmark %d' % i
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.10
            elif i == 2:
                # Add oracle
                landmark.name = "oracle"
                landmark.collide = False
                landmark.movable = False
                landmark.oracle_on = False
                landmark.size = 0.10
            else:
                raise ValueError()

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
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

            if i == 0:
                agent.i_target_landmark = np.random.choice(2)
                agent.color = np.array([1.0, 0.0, 0.0])
                world.landmarks[agent.i_target_landmark].color = \
                    np.array([1.0, 0.0, 0.0])
            elif i == 1:
                assert world.agents[0].i_target_landmark is not None
                agent.i_target_landmark = 1 - world.agents[0].i_target_landmark
                agent.color = np.array([0.0, 1.0, 0.0])
                world.landmarks[agent.i_target_landmark].color = \
                    np.array([0.0, 1.0, 0.0])
            else:
                raise ValueError()

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

            if i == 2:
                landmark.oracle_on = False
                landmark.color = np.array([0.25, 0.25, 0.25])

        while self.check_landmark_dist(world, th=1.5) is False:
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

        for i, goal in enumerate(world.goals):
            goal.state.p_pos = np.zeros(world.dim_p) - 2  # NOTE Initialize outside of the box
            goal.state.p_vel = np.zeros(world.dim_p)

            if i == 0:
                goal.color = np.array([1.0, 0.0, 0.0])
            elif i == 1:
                goal.color = np.array([0.0, 1.0, 0.0])
            else:
                raise NotImplementedError()

        assert world.agents[0].i_target_landmark != world.agents[1].i_target_landmark

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

    def reward(self, agent, world):
        target_landmark = world.landmarks[agent.i_target_landmark]
        rew = np.sum(np.square(agent.state.p_pos - target_landmark.state.p_pos))

        assert ("obstacle" in target_landmark.name) is False

        return -rew

    def observation(self, agent, world):
        # Check oracle turn on
        if world.landmarks[-1].oracle_on is False:
            oracle = world.landmarks[-1]
            num_agents_near_oracle = 0
            for a in world.agents:
                dist_to_oracle = np.sqrt(np.sum(np.square(a.state.p_pos - oracle.state.p_pos)))
                if dist_to_oracle < 0.30:
                    num_agents_near_oracle += 1

            if num_agents_near_oracle >= 1:
                print("Oracle turned on!")
                world.landmarks[-1].oracle_on = True

        landmark_pos = []
        for entity in world.landmarks:
            landmark_pos.append(entity.state.p_pos - agent.state.p_pos)

        other_pos = []
        for other in world.agents:
            if other is agent: 
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if world.landmarks[-1].oracle_on:
            target_obs = [0.] * 2
            target_obs[agent.i_target_landmark] = 1
            # target_obs[-1] = 1
            assert sum(target_obs) == 1
        else:
            target_obs = [0.] * 2
        target_obs = np.asarray(target_obs)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmark_pos + other_pos + [target_obs]) 
