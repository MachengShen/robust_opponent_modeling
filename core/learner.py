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

from core.off_policy_algo import Off_Policy_Algo
import torch
from core.buffer import Buffer




class Learner:
	"""Learner object encapsulating a local learner

		Parameters:
		algo_name (str): Algorithm Identifier
		state_dim (int): State size
		action_dim (int): Action size
		actor_lr (float): Actor learning rate
		critic_lr (float): Critic learning rate
		gamma (float): DIscount rate
		tau (float): Target network sync generate
		init_w (bool): Use kaimling normal to initialize?
		**td3args (**kwargs): arguments for TD3 algo


	"""

	def __init__(self, wwid, algo_name, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, init_w = True, **td3args):
		self.td3args = td3args; self.id = id
		self.wwid = wwid
		self.algo = Off_Policy_Algo(wwid, algo_name, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, init_w)
		self.args = td3args['cerl_args']

		#LEARNER STATISTICS
		self.fitnesses = []
		self.ep_lens = []
		self.value = None
		self.visit_count = 0
		self.private_replay_buffer = Buffer(1000000, self.args.buffer_gpu)   #

	def share_memory(self):
		self.algo.share_memory()

	def act(self, state, eps=None): #eps not used, to have common interface with dqn.act
		return self.algo.act(state)

	def step(self, state, action, reward, next_state, done):  #for training blue agent, add experience to reply buffer, and do one learning iteration
		self.private_replay_buffer.add(state, action, reward, next_state, done)
		self.update_parameters(self.private_replay_buffer, self.args.buffer_gpu, self.args.batch_size, iterations=1)

	def save_net(self, path):
		self.algo.save_net(path)

	def update_parameters(self, replay_buffer, buffer_gpu, batch_size, iterations):
		for _ in range(iterations):
			s, ns, a, r, done = replay_buffer.sample(batch_size)
			if not buffer_gpu and torch.cuda.is_available():
				s = s.cuda(); ns = ns.cuda(); a = a.cuda(); r = r.cuda(); done = done.cuda()
			self.algo.update_parameters(s, ns, a, r, done, 1, **self.td3args)


	def update_stats(self, fitness, ep_len, gamma=0.2):  #ms:fitness is the cum reward each whole episode
		self.visit_count += 1	#ms:visit_count is the number of workers per policy
		self.fitnesses.append(fitness)
		self.ep_lens.append(ep_len)

		if self.value == None: self.value = fitness	#ms: moving avg of fitness as value, value used in ucb for policy reallocation
		else: self.value = gamma * fitness + (1-gamma) * self.value
