import gym
import random
#import gym.spaces as spaces
#from gym.envs.registration import EnvSpec
import numpy as np
#from multiagent.multi_discrete import MultiDiscrete
import torch


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentDisActionEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True, blue_actual_reward_callback=None,
                 belief_and_true_type_callback=None):
        self.world = world
        self.agents = self.world.policy_agents  #assign policy agent, important to check policy agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.blue_actual_reward_callback = blue_actual_reward_callback
        self.belief_and_true_type_callback = belief_and_true_type_callback
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.self_play = False  #flag indicating env execute in self-play mode(receiving all agent actions) or single-agent mode(receiving only one action)


        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    @property
    def adversary_opponent(self) -> bool:
        return True if self.world.unknown_agents[0].adversary else False

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def agent_default_policy(self, agent_index):  #this is designed for CERL, note adversary use TD3 policy
        agent = self.world.agents[agent_index]
        if agent.neutral:
            return self.world.sample_neutral()
        else:
            obs = self._get_obs(agent)
            return agent.default_policy(obs)
            """
            if agent.good:
                return agent.dqn_agent_naive.soft_act(obs)
            if agent.adversary:
                gumbel_sample = agent.TD3_actor.Gumbel_softmax_sample_distribution(torch.from_numpy(obs).float().unsqueeze(0))
                return int(agent.TD3_actor.turn_max_into_onehot(gumbel_sample).argmax().numpy().__float__())
            """
        raise Exception("must be either good, or adv, or neutral")
    #this is the policy used for belief update (for red agent, should use the average model)

    #take policy agent's action from the input action_n, returns all the agents' reward, observation
    def step(self, action_n, use_actual_reward=False) -> tuple:
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        assert len(action_n) == len(self.agents)
        if self.self_play:
            assert len(action_n) == self.n
        # set action for each agent
        #for i, agent in enumerate(self.agents):
            #self._set_action(action_n[i], agent, self.action_space[i])
        for i, agent in enumerate(self.agents):
            agent.set_action(action_n[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            if agent.good and use_actual_reward:
                reward_n.append(self.blue_actual_reward())
            else:
                reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
        for agent in self.world.scripted_agents:
            obs_n.append(self._get_obs(agent))
            if agent.good and use_actual_reward:
                reward_n.append(self.blue_actual_reward())
            else:
                reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
        assert len(self.agents) == 1 or self.self_play  #now only treat as single agent environment, later can use multi-agent environment
        return obs_n, reward_n, done_n, [None]*len(self.world.agents)
        #return obs_n, reward_n, done_n, info_n

    def reset(self) -> list:
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents + self.world.scripted_agents:
            obs_n.append(self._get_obs(agent))
        assert len(self.agents) == 1 or self.self_play
        return obs_n
        # return obs_n[0]  # NOTE dkk Required for single agent rl

    # get observation for a particular agent
    def _get_obs(self, agent) -> np.array:
        if self.observation_callback is None:
            agent.obs = None
        else:
            agent.obs = self.observation_callback(agent, self.world)
        return agent.obs

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent) ->float:
        if self.reward_callback is None:
            return None
        return self.reward_callback(agent, self.world)

    def blue_actual_reward(self) ->float:
        if self.blue_actual_reward_callback is None:
            return None
        return self.blue_actual_reward_callback(self.world.good_agents[0], self.world)

    def belief_and_true_type(self):
        if self.belief_and_true_type is None:
            return None
        return self.belief_and_true_type_callback(self.world.good_agents[0], self.world)
    # set env action for a particular agent
    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment

    def close(self):  #override the close method in gym.env (super class method), such that when __del__() invoked,
        pass
        """
        for viewer in self.viewers:
            if viewer is not None:
                viewer.close()
        """
    def render(self, mode='human'):#, close=False): #close is dummy
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.world.goals is not None:
            vis_entities = self.world.entities + self.world.goals
        else:
            vis_entities = self.world.entities

        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []

            for entity in vis_entities:
                if "border" in entity.name:
                    geom = rendering.make_polygon(entity.shape)
                else:
                    geom = rendering.make_circle(entity.size)

                xform = rendering.Transform()
                if 'goal' in entity.name or "target" in entity.name or 'landmark' in entity.name:
                    geom.set_color(*entity.color, alpha=0.3)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = self.world.x_range[1]/2.0 + 0.5
            if self.shared_viewer:
                if self.world.dim_p == 1:
                    pos = np.array([self.world.x_range[1] / 2.0, 0])
                else:
                    pos = np.array([self.world.x_range[1]/2.0, self.world.y_range[1]/2.0])
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(vis_entities):
                if self.world.dim_p == 1:
                    translation = np.concatenate((entity.state.p_pos, np.array([0])), axis=0)
                    self.render_geoms_xform[e].set_translation(*translation)
                else:
                    self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))
        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
