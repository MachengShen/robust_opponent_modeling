import numpy as np
from multiagent.core import World, Agent, Landmark, Goal
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, mode):
        world = World()
        self.mode = mode

        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1

        # add boxes
        self.boxes = [Landmark() for _ in range(2)]
        for i, box in enumerate(self.boxes):
            box.name = 'box %d' % i
            box.size = 0.25 
            box.collide = True
            box.index = i

            # Box movable for pretrain
            if self.mode == 0 and box.index == 1:
                box.movable = False
            elif self.mode == 1 and box.index == 0:
                box.movable = False
            else:
                box.movable = True

            # Different box mass
            if box.index == 0:
                box.initial_mass = 2.
            elif box.index == 1:
                box.initial_mass = 6.
            else:
                raise ValueError()
            world.landmarks.append(box)

        # add targets
        self.targets = [Landmark() for _ in range(2)]
        for i, target in enumerate(self.targets):
            target.name = 'target %d' % i
            target.collide = False
            target.movable = False
            target.size = 0.05
            target.index = i
            world.landmarks.append(target)

        # add goals (used only for vis)
        world.goals = [Goal() for i in range(len(world.agents))]
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
            else:
                raise NotImplementedError()

            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_vel = np.zeros(world.dim_p)

            if "box" in landmark.name and landmark.index == 0:
                landmark.state.p_pos = np.array([-0.30, 0.0])
            elif "box" in landmark.name and landmark.index == 1:
                landmark.state.p_pos = np.array([0.30, 0.0])
            elif "target" in landmark.name and landmark.index == 0:
                landmark.state.p_pos = np.array([-0.85, 0.0])
            elif "target" in landmark.name and landmark.index == 1:
                landmark.state.p_pos = np.array([+0.85, 0.0])
            else:
                raise ValueError()

        # random properties for goals (vis purpose)
        for i, goal in enumerate(world.goals):
            goal.color = world.agents[i].color
            goal.state.p_pos = np.zeros(world.dim_p) - 2  # Initialize outside of the box
            goal.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        for i, landmark in enumerate(world.landmarks):
            if "box" in landmark.name and landmark.index == 0:
                box0 = landmark
            elif "box" in landmark.name and landmark.index == 1:
                box1 = landmark
            elif "target" in landmark.name and landmark.index == 0:
                target0 = landmark
            elif "target" in landmark.name and landmark.index == 1:
                target1 = landmark
            else:
                raise ValueError()

        dist1 = np.sum(np.square(box0.state.p_pos - target0.state.p_pos))
        dist2 = np.sum(np.square(box1.state.p_pos - target1.state.p_pos))
        dist = dist1 + dist2

        return -dist

    def observation(self, agent, world):
        # get positions of all entities
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos)
        assert len(entity_pos) == len(self.boxes) + len(self.targets)

        # Add other agent position
        other_pos = []
        for other in world.agents:
            if other is agent: 
                continue
            other_pos.append(other.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
