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

from core.env_wrapper import EnvironmentWrapper
from core import mod_utils as utils
import numpy as np
import torch
from discreteenv.make_envs import make_self_play_env
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
#from unityTennisEnv import make_tennis_env


# Rollout evaluate an agent in a complete game
def rollout_worker(id, second_pid, task_pipe, result_pipe, is_noise, data_bucket, model_bucket, env_name, noise_std, ALGO, trainers = None, Save_net = True, pomdp_adv = False):
	"""Rollout Worker runs a simulation in the environment to generate experiences and fitness values

		Parameters:
			task_pipe (pipe): Receiver end of the task pipe used to receive signal to start on a task
			result_pipe (pipe): Sender end of the pipe used to report back results
			is_noise (bool): Use noise?
			data_bucket (list of shared object): A list of shared object reference to s,ns,a,r,done (replay buffer) managed by a manager that is used to store experience tuples
			model_bucket (shared list object): A shared list object managed by a manager used to store all the models (actors)
			env_name (str): Environment name?
			noise_std (float): Standard deviation of Gaussian for sampling noise

		Returns:
			None
	"""
	np.random.seed(id)  ###make sure the random seeds across learners are different
	if ALGO == 'dis':	#make env with blue and red policy agent inside,
		assert trainers is not None
		dis_env = make_self_play_env(seed=id, return_policy_agent=False, trainers= trainers)[0]  #trainer if not None, first is the shared DQN agent, second is the best red policy
		env = EnvironmentWrapper(env_name, ALGO, dis_env, 1) #this is for red agent
	elif ALGO == 'TD3_tennis':
		pid = id + second_pid * 20  #should be larger than the number of processes
		tennis_env = make_tennis_env.TennisEnvFactory(seed=id, no_graphics=True, pid=pid).getEnv()[0]
		env = EnvironmentWrapper('Tennis', ALGO, tennis_env, 0)
	else:
		print("ALGO is:", ALGO)
		env = EnvironmentWrapper(env_name, ALGO)


	###LOOP###
	while True:
		identifier, gen = task_pipe.recv()  # Wait until a signal is received  to start rollout
		if identifier == 'TERMINATE':
			print('Process:', os.getpid(), ' killed')
			exit(0) #Kill yourself

		# Get the requisite network
		net = model_bucket[identifier]

		fitness = 0.0;
		total_frame = 0
		state = env.reset();

		if pomdp_adv: env.try_set_pomdp_adv()

		rollout_trajectory = []
		state = utils.to_tensor(np.array(state)).unsqueeze(0)

		while True:  # unless done
			#cannot pass this line
			action = net.Gumbel_softmax_sample_distribution(state, use_cuda=False) if ALGO == 'dis' else net.forward(state)
			#here action is on the simplex, while use the arg_max sample in the env step

			action = utils.to_numpy(action)
			if is_noise and not ALGO == 'dis':
				#action = (action + np.random.normal(0, noise_std, size=env.env.action_space.shape[0])).clip(env.env.action_space.low, env.env.action_space.high)
				action = (action + np.random.normal(0, noise_std, size=env.env.action_space.shape[0])).clip(env.env.action_space.low, env.env.action_space.high)

			next_state, reward, done, info = env.step(int(net.turn_max_into_onehot(action).argmax().numpy().__float__())) if ALGO == 'dis' else env.step(action.flatten())  # Simulate one step in environment

			#if id == 1: env.render()

			next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
			fitness += reward
			#print('line 79')
			# If storing transitions
			if data_bucket != None: #Skip for test set
				rollout_trajectory.append([utils.to_numpy(state), utils.to_numpy(next_state),
										np.float32(action), np.reshape(np.float32(np.array([reward])), (1, 1)),
										   np.reshape(np.float32(np.array([float(done)])), (1, 1))])
			state = next_state
			total_frame += 1

			# DONE FLAG IS Received
			#print(done)
			if done:

				# Push experiences to main
				for entry in rollout_trajectory:
					data_bucket.append(entry)  #ms this data_bucket store experience, is shared accross process (and training thread?)


				break
		# Send back id, fitness, total length and shaped fitness using the result pipe
		#if gen >= 10 and id == 1:
		#	env.env.world.good_agents[0].dqn_agent_naive.save_net('./runner_blue_dqn.pth')
		if gen >= 10 and gen % 5 == 0 and Save_net:
			net.save_net('./pytorch_models/red_' + str(id) + '_net_step_' + str(gen) + '.pth')

		result_pipe.send([identifier, fitness, total_frame, env.get_blue_actual_reward()])
