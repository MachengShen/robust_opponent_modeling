import numpy as np
#from multiagent.core import World, Agent, Landmark, Goal
from .Entity import Agent, GoodAgent, Landmark
from .World import World
from multiagent.scenario import BaseScenario
import copy, random
from .dqn_agent import DQNAgent
#from .Action_callback import null_policy, dqn_callback
from collections import namedtuple

BeliefTrueType = namedtuple('BeliefTrueType', ['belief', 'true_type'])

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.num_agents = 2
        world.num_good_agents = 1
        world.num_adversaries = 1
        world.agents = []

        for i in range(world.num_agents):
            agent = GoodAgent() if i < world.num_good_agents else Agent()
            world.agents.append(agent)
            agent.name = 'agent %d' % i

        world.landmarks = [Landmark(), Landmark()] if world.dim_p == 2 else [Landmark()]
        # add landmarks
        world.x_range = np.array([0.0, 8.0])
        world.y_range = np.array([0.0, 8.0])
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.reset_time()
        world.set_entity_dim()
        # random properties for agents
        if world.dim_p == 1:
            world.landmarks[0].state.p_pos = np.array([world.x_range[1]])
        else:
            world.landmarks[0].state.p_pos = np.array([world.x_range[1], world.y_range[1]])
            world.landmarks[1].state.p_pos = np.array([world.x_range[0], world.y_range[1]])
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.75])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
                agent.goal = world.landmarks[0]
                agent.state.tagged = False
            if agent.neutral:
                agent.color = np.array([0.25, 0.75, 0.25])
                agent.goal = world.landmarks[1]
                agent.state.tagged = False
        #set agent goal
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.55, 0.25, 0.25])
            if i >= 1:
                landmark.color = np.array([0.25, 0.55, 0.25])
        # set random initial states
        for agent in world.agents:
            if agent.good:
                agent.state.naive_belief = np.array([0.5])
                agent.state.tag_count = np.array([0])
                agent.state.probe_count = np.array([0])
                if agent.naive == False:
                    agent.state.smart_belief = np.array([0.5])
                if world.dim_p == 1:
                    agent.pos_range = [world.x_range / 2.0]
                    agent.state.p_pos = np.array([world.x_range[1]/2.0])
                    agent.action_size = 3 + 2
                else:
                    agent.pos_range = [world.x_range, world.y_range / 2.0]
                    agent.state.p_pos = np.array([world.x_range[1]/2.0, world.y_range[1]/2.0])
                    agent.action_size = 5 + 2
            if agent.adversary or agent.neutral:
                if agent.naive == False:  #the good agent should at least be naive instead of None
                    good_agent = world.good_agents[0]
                    assert good_agent.naive is not None
                    agent.state.naive_belief = good_agent.state.naive_belief
                if world.dim_p == 1:
                    agent.pos_range = [world.x_range]
                    agent.state.p_pos = np.array([0.0])
                    agent.action_size = 3
                else:
                    agent.pos_range = [world.x_range, world.y_range]
                    agent.state.p_pos = np.array([world.x_range[1] / 2.0, 0.0])
                    agent.action_size = 5
            if world.random_pose:
                agent.sample_pos()

    def check_distance(self, agents, landmark_p_pos):
        threshold = 1.
        for i_agt, agent in enumerate(agents):
            if np.linalg.norm(agent.state.p_pos - landmark_p_pos) < threshold:
                return True
            else:
                return False

    def reward(self, agent, world):
        if agent.naive == None and agent.good:  #null policy agent
            return None
        neu_adv = world.unknown_agents[0]
        good_agent = world.good_agents[0]
        good_rew = 0.0
        adv_tag_penalty = 0.0
        adv_belief_maneuver_reward = - good_agent.belief
        TAG_COST = 0.2
        dist = np.linalg.norm(good_agent.state.p_pos - neu_adv.state.p_pos)
        good_rew += - 0.25 * np.power(dist, 0.4)
        #if random.random() < 0.01:
        #    print('discrete_decep line 108, changed distance reward')
        if good_agent.action.tag == True:
            good_rew -= TAG_COST
            if neu_adv.state.tagged:
                good_belief_reward = good_agent.belief * 10.0 - (1.0 - good_agent.belief) * 20.0
                good_rew += good_belief_reward
                good_rew -= (good_agent.state.tag_count - 1.0) * 10.5  #the second time tag is penalized
                adv_tag_penalty += 10.0
                adv_belief_maneuver_reward -= good_belief_reward
            else:
                good_rew -= (good_agent.state.tag_count - 1.0) * 2.0  #also penalize ineffective tags

        if agent.adversary:
            shaped = True
            if shaped:
                dist2 = np.power(np.linalg.norm(agent.state.p_pos - agent.goal.state.p_pos), 0.4)
                #dist2 = np.power(np.linalg.norm(agent.state.p_pos - agent.goal.state.p_pos), 1.0)
                rew = - 0.5 * dist2
            else:
                rew = float(self.check_distance([agent], agent.goal.state.p_pos))
            rew -= adv_tag_penalty
            rew += adv_belief_maneuver_reward
            return float(rew)

        if agent.good:
            rew = 0.0 if not agent.action.probe else -good_agent.state.probe_count * 0.25
            rew += good_rew
            return float(rew)

        if agent.neutral: #neutral reward not really needed
            return None

    def blue_actual_reward(self, agent, world):
        assert agent.good
        original_belief = copy.deepcopy(agent.belief)
        if world.unknown_agents[0].neutral:
            agent.set_belief(np.array([0.0]))
        else:
            assert world.unknown_agents[0].adversary
            agent.set_belief(np.array([1.0]))
        actual_reward = self.reward(agent, world)
        agent.set_belief(original_belief) #recover agent.belief
        return actual_reward

    def belief_and_true_type(self, agent, world):  #return the blue agent belief and the true type of the opponent
        assert agent.good
        true_type = 1 if world.unknown_agents[0].adversary else 0
        return BeliefTrueType(copy.deepcopy(agent.belief), true_type)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        agents_pos = []
        entity_pos = []
        relative_pos = False
        if relative_pos:
            for entity in world.landmarks:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            for all_agent in world.agents:
                agents_pos.append(all_agent.state.p_pos - agent.state.p_pos)
        else:
            for entity in world.landmarks:
                entity_pos.append(entity.state.p_pos)
            for all_agent in world.agents:
                agents_pos.append(all_agent.state.p_pos)

        good_agent = world.good_agents[0]
        tag_probe_state = [good_agent.state.tag_count] + [good_agent.state.probe_count]
        belief_state = good_agent.belief
        if agent.good: assert agent is good_agent;

        if agent.good:
            if agent.naive is None: #should be null policy agent
                return
            assert agent.naive is not None  #good agent with null policy does not need observation
            belief_state = agent.state.naive_belief if agent.naive else agent.state.smart_belief
        if agent.adversary or agent.neutral:
            if world.use_belief:
                belief_state = agent.state.naive_belief if not agent.naive else None  #note only smart adv have naive_belief
            else:
                belief_state = None

        if belief_state is not None:
            assert belief_state.shape[0] == 1
        return np.concatenate([belief_state] + agents_pos + entity_pos + tag_probe_state)\
            if belief_state is not None else np.concatenate(agents_pos + entity_pos + tag_probe_state)

    def done(self, agent, world):
        return world.time >= 50