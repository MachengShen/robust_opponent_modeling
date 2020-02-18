import numpy as np
from multiagent.core import World, Agent, Landmark, Goal, Border
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, mode):
        """
        - mode0: Pretrain from room2 to room1
        - mode1: Pretrain from room1 to target
        - mode2: Train from room2 to target
        """
        world = World()
        self.mode = mode

        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
            agent.color = np.array([1.0, 0.0, 0.0]) if i == 0 else np.array([0.0, 1.0, 0.0])

        # add boxes
        self.box = Landmark()
        self.box.name = 'box'
        self.box.collide = True
        self.box.movable = True
        self.box.size = 0.25  # Radius
        self.box.initial_mass = 3.
        self.box.color = np.array([0.25, 0.25, 0.25])
        world.landmarks.append(self.box)

        # add targets
        self.target = Landmark()
        self.target.name = 'target'
        self.target.collide = False
        self.target.movable = False
        self.target.size = 0.05
        self.target.color = np.array([0.25, 0.25, 0.25])
        world.landmarks.append(self.target)

        # add borders
        self.add_borders(world)

        # add goals (used only for vis)
        world.goals = [Goal() for i in range(2)]
        for i, goal in enumerate(world.goals):
            goal.name = 'goal %d' % i
            goal.collide = False
            goal.movable = False
            goal.color = world.agents[i].color

        # make initial conditions
        self.reset_world(world)
        
        return world

    def add_borders(self, world):
        """ Adding the center border """
        # Parameters
        self.length = 1.4
        offset = 0.1  # Distance between center to center
        n_border = round(self.length / offset)
        center_border = [Border() for _ in range(n_border)]

        # Add the center border
        x, y = 0, -1 + (offset / 2.)
        for border in center_border:
            border.name = "border"
            border.collide = True
            border.movable = False
            border.size = offset / 2.
            border.shape = [
                [-border.size, -border.size],
                [border.size, -border.size],
                [border.size, border.size],
                [-border.size, border.size]]
            border.color = np.array([0.25, 0.25, 0.25])
            border.state.p_vel = np.zeros(world.dim_p)
            border.state.p_pos = np.asarray([x, y])
            world.borders.append(border)

            x, y = x, y + offset

        # Define room1 and room2
        self.x_room1_from = -1.
        self.x_room1_to = 0. - border.size
        self.y_room1_from = -1.
        self.y_room1_to = +1.

        self.x_room2_from = 0. + border.size
        self.x_room2_to = 1.
        self.y_room2_from = -1.
        self.y_room2_to = +1.

        # Define boundary between room 1 and room 2
        self.boundary_pos = np.array([0., (self.length - 1.) + self.box.size])

    def reset_world(self, world):
        # random properties for agents
        # NOTE Agents always starting inside room 2
        for i, agent in enumerate(world.agents):
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

            agent.state.p_pos = np.zeros(world.dim_p)
            agent.state.p_pos[0] = np.random.uniform(
                low=self.x_room2_from + agent.size * 1.5,
                high=self.x_room2_to - agent.size * 1.5)
            agent.state.p_pos[1] = np.random.uniform(
                low=self.y_room2_from + agent.size * 1.5,
                high=self.y_room2_to - agent.size * 1.5)

        # random properties for box
        # At mode 1, we initialize box at the boundary (with some noise)
        self.box.state.p_vel = np.zeros(world.dim_p)
        if self.mode == 0 or self.mode == 2:
            self.box.state.p_pos = np.array([self.boundary_pos[1], 0.])
        elif self.mode == 1:
            self.box.state.p_pos = np.array([0., self.boundary_pos[1]])

        # reset properties for target
        self.target.state.p_vel = np.zeros(world.dim_p)
        self.target.state.p_pos = np.array([-self.boundary_pos[1], 0.])

        # reset properties for goals (vis purpose)
        for i, goal in enumerate(world.goals):
            goal.state.p_pos = np.zeros(world.dim_p) - 2  # Initialize outside of the box
            goal.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Inside Room 1
        if self.box.state.p_pos[0] < 0.:
            dist = np.sum(np.square(self.box.state.p_pos - self.target.state.p_pos))
        # Inside Room 2
        # NOTE room1_dist refers to maximum distance w.r.t room 1
        elif self.box.state.p_pos[0] >= 0.:
            room1_dist = np.sum(np.square(np.array([0., 1.]) - self.target.state.p_pos))
            room2_dist = np.sum(np.square(self.boundary_pos - self.box.state.p_pos))
            dist = room1_dist + room2_dist
        else:
            raise ValueError()

        return -dist / 10.  # Reward scale

    def observation(self, agent, world):
        # get positions of all entities
        entity_pos = []
        entity_pos.append(self.box.state.p_pos)
        entity_pos.append(self.target.state.p_pos)

        # Add other agent position
        other_pos = []
        for other in world.agents:
            if other is agent: 
                continue
            other_pos.append(other.state.p_pos)

        # border position
        border_pos = []
        border_pos.append(np.array([0., -1.]))
        border_pos.append(np.array([0., (self.length - 1.)]))

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + border_pos)
