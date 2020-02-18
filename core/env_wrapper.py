# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import gym
#from ...make_envs import make_self_play_env  #. current dir, .. parent dir, ...grandparent dir


class EnvironmentWrapper:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name
	"""
	def __init__(self, env_name, ALGO, env=None, agent_index = None):
		"""
		A base template for all environment wrappers.
		"""
		self.ALGO = ALGO
		if ALGO == 'dis' or ALGO == 'TD3_tennis':  #need to provide the env
			assert env is not None and (agent_index == 0 or agent_index == 1)
			self.env = env
			self.agent_index = agent_index #agent_index 0(blue), 1(red) determines the input and return states
			self.env.reset() #reset world
		else:
			self.env = gym.make(env_name)
			self.action_low = float(self.env.action_space.low[0])
			self.action_high = float(self.env.action_space.high[0])

	def pick_index(self, x_n):  #pick the corresponding state, action etc. according to agent index
		#used as interface between env for world agents and envwrapper for trainers
		if isinstance(x_n, tuple): #e.g. return of env.step
			return tuple([x[self.agent_index] for x in x_n]) if self.ALGO == 'dis' else x_n
		else:#e.g. return of env.reset
			return x_n[self.agent_index] if self.ALGO == 'dis' or 'TD3_tennis' else x_n

	def reset(self, record_red_reward = True, record_actual_blue_reward = True):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		self.record_red_reward = record_red_reward
		if record_red_reward: self.red_reward = 0.0;
		else:
			if hasattr('red_reward', self): del self.red_reward;

		self.record_actual_blue_reward = record_actual_blue_reward
		if record_actual_blue_reward: self.actual_blue_reward = 0.0;
		else:
			if hasattr('actual_blue_reward', self): del self.actual_blue_reward;
		return self.pick_index(self.env.reset())

	#the step take in one action, and return one obs, reward, done according to the agent_index,
	#so this is used for decentralized learning and compatibility with both training methods
	def step(self, action, use_actual_reward=False): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""
		if not self.ALGO == 'dis':
			action = self.action_low + action * (self.action_high - self.action_low)
			action_n = action #to keep the return syntax simple
		else:
		#else we need to augment action to action_n, and enforce env.step returns single state, reward etc.
		#note the env.step takes in action_n and return state_n
			action_n = [None for i in range(self.env.n)]
			for i in range(self.env.n):
				action_n[i] = action if i == self.agent_index else self.env.agent_default_policy(i) #default for unknown agent if neutral

		s_r_d_i_tuple = self.env.step(action_n, use_actual_reward=use_actual_reward)
		if self.record_red_reward and self.env.world.unknown_agents[0].adversary:
			self.red_reward += s_r_d_i_tuple[1][1]
		if self.record_actual_blue_reward: self.actual_blue_reward += self.blue_actual_reward()
		return self.pick_index(s_r_d_i_tuple)

	def blue_actual_reward(self):
		return self.env.blue_actual_reward()

	def belief_and_true_type(self):
		return self.env.belief_and_true_type()

	def randomize_neu_adv(self, neutral_prob=0.5):
		self.env.world._randomize_neu_adv(neutral_prob)

	def try_set_pomdp_adv(self):
		self.env.world._try_set_pomdp_adv()

	def render(self):
		self.env.render()

	def set_TD3_actor(self, red_actor):
		self.env.world.unknown_agents[0].TD3_actor = red_actor

	def get_red_reward(self):
		if self.record_red_reward:
			if self.env.world.unknown_agents[0].adversary:
				return self.red_reward
			else:
				return None

	def get_blue_actual_reward(self):
		if self.record_actual_blue_reward:
			return self.actual_blue_reward
		else:
			return None

