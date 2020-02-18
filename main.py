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

import numpy as np
import os
import time
import random
import torch
import pickle
import sys
from core.neuroevolution import SSNE
from core.models import Actor
from core import mod_utils as utils
from core.mod_utils import str2bool
from core.ucb import ucb
from core.runner import rollout_worker
from core.portfolio import initialize_portfolio
from torch.multiprocessing import Process, Pipe, Manager
import threading
from core.buffer import Buffer
from core.genealogy import Genealogy
import gym
import argparse
# just used to create dqn blue agent trainer and get state action dim
from discreteenv.make_envs import make_self_play_env
from core.misc import visualize_critic, log_TPR_FPR_TNR_FNR
from core.env_wrapper import EnvironmentWrapper
import copy
from core.average_actor import AverageActor
from core.evaluator import Evaluator
import logging
from core.mutation_schemes import Mutation_Add, Mutation_Delete, Mutation_Exchange
from core.learner import Learner
#from unityTennisEnv import make_tennis_env

parser = argparse.ArgumentParser()
parser.add_argument('-render', type=int,
					help='render or not',  default=1)
parser.add_argument('-pop_size', type=int,
					help='#Policies in the population',  default=6)

parser.add_argument('-use_simulated_annealing', action='store_false', default=True)
parser.add_argument('-random_portfolio', action='store_true', default=False) #this flag make SA into random walk
parser.add_argument('-isolate_pg', action='store_true', default=False) #this flag make SA into random walk
parser.add_argument('-drqn', action='store_true', default=False) #this flag make SA into random walk

parser.add_argument('-seed', type=int, help='Seed',  default=2005)
parser.add_argument('-rollout_size', type=int,
					help='#Policies in rolout size',  default=6)
parser.add_argument(
	'-env', type=str, help='#Environment name',  default='Humanoid-v1')
parser.add_argument('-gradperstep', type=float,
					help='#Gradient step per env step',  default=1.0)
parser.add_argument('-savetag', type=str,
					help='#Tag to append to savefile',  default='')
parser.add_argument('-gpu_id', type=int, help='#GPU ID ',  default=0)
parser.add_argument('-buffer_gpu', type=str2bool,
					help='#Store buffer in GPU?',  default=0)
parser.add_argument('-portfolio', type=int, help='Portfolio ID',  default=10)
parser.add_argument('-total_steps', type=float,
					help='#Total steps in the env in millions ',  default=2)
parser.add_argument('-batchsize', type=int, help='Seed',  default=256)

SA_FLAG = vars(parser.parse_args())['use_simulated_annealing']
RANDOM_WALK = vars(parser.parse_args())['random_portfolio']
if RANDOM_WALK: assert SA_FLAG
RENDER = vars(parser.parse_args())['render']
RENDER = False if RENDER == 0 else True
POP_SIZE = vars(parser.parse_args())['pop_size']
BATCHSIZE = vars(parser.parse_args())['batchsize']
ROLLOUT_SIZE = vars(parser.parse_args())['rollout_size']
ENV_NAME = vars(parser.parse_args())['env']
GRADPERSTEP = vars(parser.parse_args())['gradperstep']
SAVETAG = vars(parser.parse_args())['savetag']
BUFFER_GPU = vars(parser.parse_args())['buffer_gpu']
SEED = vars(parser.parse_args())['seed']
GPU_DEVICE = vars(parser.parse_args())['gpu_id']
PORTFOLIO_ID = vars(parser.parse_args())['portfolio']
if SA_FLAG: assert PORTFOLIO_ID == 10  #need at least several learners
TOTAL_STEPS = int(vars(parser.parse_args())['total_steps'] * 1000000)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)
ISOLATE_PG = vars(parser.parse_args())['isolate_pg']
DRQN = vars(parser.parse_args())['drqn']

if ISOLATE_PG:
	print("Attention, we are using isolated PG")
# ICML EXPERIMENT
"""
if PORTFOLIO_ID == 11 or PORTFOLIO_ID == 12 or PORTFOLIO_ID == 13 or PORTFOLIO_ID == 14 or PORTFOLIO_ID == 101 or PORTFOLIO_ID == 102:
	ISOLATE_PG = True
else:
	ISOLATE_PG = False
"""


"""
if PORTFOLIO_ID == 10:
	POP_SIZE = 10
	ROLLOUT_SIZE = 10
else:
	POP_SIZE = 3
	ROLLOUT_SIZE = 3
"""


ALGO = "dis"
#ALGO = "dis"
SAVE = True
TEST_SIZE = 5


class Parameters:
	def __init__(self):
		"""Parameter class stores all parameters for policy gradient

		Parameters:
			None

		Returns:
			None
		"""

		self.seed = SEED
		self.asynch_frac = 1.0  # Aynchronosity of NeuroEvolution
		self.algo = ALGO
		self.drqn = DRQN
		self.isolate_pg = ISOLATE_PG

		self.render = RENDER
		self.batch_size = BATCHSIZE  # Batch size
		self.noise_std = 0.1  # Gaussian noise exploration std
		self.ucb_coefficient = 0.25  # ms: was 0.9 #Exploration coefficient in UCB
		self.gradperstep = GRADPERSTEP
		self.buffer_gpu = BUFFER_GPU
		self.rollout_size = ROLLOUT_SIZE  # Size of learner rollouts

		# NeuroEvolution stuff
		self.pop_size = POP_SIZE
		self.elite_fraction = 0.2
		self.crossover_prob = 0.15
		self.mutation_prob = 0.90

		#######unused########
		self.extinction_prob = 0.005  # Probability of extinction event
		# Probabilty of extinction for each genome, given an extinction event
		self.extinction_magnituide = 0.5
		self.weight_magnitude_limit = 10000000
		self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform

		# Save Results
		if ALGO == 'dis':
			a = make_self_play_env(trainers=[[], []])
			# actually does not need trainers, only want blue_agent_trainer
			dummy_env, blue_agent_trainer = make_self_play_env(trainers=[[], []], blue_use_drqn=DRQN)
			# blue_agent_trainer this is actually two trainer
			self.blue_trainer = blue_agent_trainer[0]
			self.blue_trainer.share_memory()
			self.action_dim = dummy_env.action_dim
			self.state_dim = dummy_env.state_dim
			self.action_low = 0
			self.action_high = 1
		elif ALGO == 'TD3_tennis':
			no_graphics = not RENDER
			dummy_env, self.action_dim, self.state_dim = make_tennis_env.TennisEnvFactory(seed=SEED, no_graphics=no_graphics, pid=-1).getEnv()
			self.action_low = -1.0
			self.action_high = +1.0   #according to unity document
			td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2,
					   'action_low': self.action_low, 'action_high': self.action_high, 'cerl_args': self}
			self.blue_trainer = Learner(-1, 'TD3', self.state_dim, self.action_dim, actor_lr=5e-5, critic_lr=1e-3, gamma=0.99,
					tau=5e-3,
					init_w=True, **td3args)
			self.blue_trainer.share_memory()
		else:
			dummy_env = gym.make(ENV_NAME)
			self.state_dim = dummy_env.observation_space.shape[
				0]; self.action_dim = dummy_env.action_space.shape[0]
			self.action_low = float(dummy_env.action_space.low[0]); self.action_high = float(
				dummy_env.action_space.high[0])
		self.savefolder = 'Results/'
		if not os.path.exists('Results/'): os.makedirs('Results/')
		if not os.path.exists('pytorch_models/'): os.makedirs('pytorch_models/')
		self.aux_folder = self.savefolder + 'Auxiliary/'
		if not os.path.exists(self.aux_folder): os.makedirs(self.aux_folder)


class CERL_Agent:
	"""Main CERL class containing all methods for CERL

		Parameters:
		args (int): Parameter class with all the parameters

	"""

	def __init__(self, args):  # need to intialize rollout_workers to have blue agent
		self.args = args
		self.evolver = SSNE(self.args)  # this evolver implements neuro-evolution

		# MP TOOLS
		self.manager = Manager()

		self.mutate_algos = [Mutation_Add(self), Mutation_Delete(self), Mutation_Exchange(self)]   #store all the mutate algorithm objects
		# Genealogy tool
		self.genealogy = Genealogy()

		# Init BUFFER
		self.replay_buffer = Buffer(1000000, self.args.buffer_gpu)

		#if SA_FLAG:
		self.metrics = []
		self.last_portfolio = None
		self.T_max = 30
		self.T = self.T_max
		self.T_min = 0.2
		self.decay_rate = 0.975

		# Initialize population
		self.pop = self.manager.list()
		for _ in range(args.pop_size):
			wwid = self.genealogy.new_id('evo')
			if ALGO == 'SAC': self.pop.append(GaussianPolicy(
				args.state_dim, args.action_dim, args.hidden_size, wwid))
			elif ALGO == 'TD3': self.pop.append(Actor(args.state_dim, args.action_dim, wwid, ALGO))
			# use ALGO to distinguish differe net architecture
			elif ALGO == 'dis' or 'TD3_tennis': self.pop.append(Actor(args.state_dim, args.action_dim, wwid, ALGO))
			else: assert False, "invalid algorithm type"

		if ALGO == "SAC": self.best_policy = GaussianPolicy(
			args.state_dim, args.action_dim, args.hidden_size, -1)
		else:
			self.best_policy = Actor(args.state_dim, args.action_dim, -1, ALGO)
			if ALGO == 'dis':
				self.average_policy = AverageActor(args.state_dim, args.action_dim, -2, ALGO, self.pop, self.replay_buffer, args.buffer_gpu, args.batch_size, iterations=10)
				self.average_policy.share_memory()

		self.best_policy.share_memory()


			# added by macheng, share the best policy accross processes (used as internal belief update models for blue)

		# now we assign shared blue_trainer, we should train this agent such that the roll_out workers are also up to date
		# should make sure that self.best_policy (emergent learner) is also shared
		if ALGO == 'dis' or 'TD3_tennis':
			assert hasattr(args, "blue_trainer"), "must have blue_agent trainer to intialize rollout_worker, see line 109, class Parameter definition"
		if ALGO == 'dis':
			trainers = [args.blue_trainer, self.average_policy]
		else:
			trainers = [args.blue_trainer, None] if ALGO == 'TD3_tennis' else []

		self.trainers = trainers

		self.blue_dqn = args.blue_trainer

		# Turn off gradients and put in eval mod
		for actor in self.pop:
			actor = actor.cpu()
			actor.eval()
		# Intialize portfolio of learners
		self.portfolio = []
		self.portfolio = initialize_portfolio(self.portfolio, self.args, self.genealogy, PORTFOLIO_ID)
		self.complement_portfolio = []   #complementary of the portfolio, whatever not in the portfolio should be stored here
		self.total_rollout_bucket = self.manager.list()   #macheng: we use total_rollout_bucket to represents the whole set of rollout models, now rollout_bukcet dynamically resize according to portforlio, for SA
		self.rollout_bucket = self.total_rollout_bucket
		#self.rollout_bucket = self.manager.list()
		#print("rollout_bucker needs to be updated, main.py line 239 ")
		for _ in range(len(self.portfolio)):
			if ALGO == 'SAC': self.rollout_bucket.append(GaussianPolicy(args.state_dim, args.action_dim, args.hidden_size, -1))
			else: self.rollout_bucket.append(Actor(args.state_dim, args.action_dim, -1, ALGO))
		# Initialize shared data bucket
		self.data_bucket = self.replay_buffer.tuples

		############## MULTIPROCESSING TOOLS ###################
		# Evolutionary population Rollout workers
		self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_workers = [Process(target=rollout_worker, args=(id, 0, self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], False, self.data_bucket, self.pop, ENV_NAME, None, ALGO, self.trainers)) for id in range(args.pop_size)]
		for worker in self.evo_workers: worker.start()
		self.evo_flag = [True for _ in range(args.pop_size)]

		# Learner rollout workers
		self.task_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.result_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.workers = [Process(target=rollout_worker, args=(id, 1, self.task_pipes[id][1], self.result_pipes[id][0], True, self.data_bucket, self.rollout_bucket, ENV_NAME, args.noise_std, ALGO, self.trainers)) for id in range(args.rollout_size)]
		for worker in self.workers: worker.start()
		self.roll_flag = [True for _ in range(args.rollout_size)]

		# Test bucket
		self.test_bucket = self.manager.list()
		if ALGO == 'SAC':
			self.test_bucket.append(GaussianPolicy(args.state_dim, args.action_dim, args.hidden_size, -1))
		else:
			self.test_bucket.append(Actor(args.state_dim, args.action_dim, -1, ALGO))

		# 5 Test workers
		self.test_task_pipes = [Pipe() for _ in range(TEST_SIZE)]
		self.test_result_pipes = [Pipe() for _ in range(TEST_SIZE)]
		self.test_workers = [Process(target=rollout_worker, args=(id, 2, self.test_task_pipes[id][1], self.test_result_pipes[id][0], False, None, self.test_bucket, ENV_NAME, args.noise_std, ALGO, self.trainers)) for id in range(TEST_SIZE)]
		for worker in self.test_workers: worker.start()
		self.test_flag = False

		# Meta-learning controller (Resource Distribution)
		self.allocation = [] #Allocation controls the resource allocation across learners
		for i in range(args.rollout_size): self.allocation.append(i % len(self.portfolio)) #Start uniformly (equal resources)
		# self.learner_stats = [{'fitnesses': [], 'ep_lens': [], 'value': 0.0, 'visit_count':0} for _ in range(len(self.portfolio))] #Track node statistsitic (each node is a learner), to compute UCB scores

		# Trackers
		self.best_score = -np.inf; self.gen_frames = 0; self.total_frames = 0; self.best_shaped_score = None; self.test_score = None; self.test_std = None

		# trainer contains the blue_dqn to be trained, and the red model used for belief update, red_actor is the actual red agent trained against
		# id is the actual red agent id

	def _update_SA_temperature(self):
		self.T = max(self.T * self.decay_rate, self.T_min)

	def _get_accept_rate(self):
		if RANDOM_WALK:
			return 1.0
		else:
			if self.metrics[-1] > self.metrics[-2]:
				return 1.0
			else:
				return np.exp((self.metrics[-1] - self.metrics[-2]) / self.T)

	def _mutate(self):
		while True:
			mutate_algo_index = random.choice(range(3))
			if self._try_mutate(mutate_algo_index):
				return

	def _try_mutate(self, algo_index): # 0 for add, 1 for delete, 2 for exchange
		return self.mutate_algos[algo_index].try_mutate()

	def simulated_annealing(self, metric): #take in the current metric
		self.metrics.append(metric)
		if self.last_portfolio:  #has last_portfolio
			accept_rate = self._get_accept_rate() #based on self.metrics[-2:]
			self._update_SA_temperature()
			if np.random.random() > accept_rate: #reject
				self.portfolio = self.last_portfolio
				self.complement_portfolio = self.last_complement_portfolio

		self.last_portfolio = copy.copy(self.portfolio) #maintain a shallow copy as
		self.last_complement_portfolio = copy.copy(self.complement_portfolio)
		self._mutate()  #perturb the portfolio
		# update rollout_bucket size, only the first len(self.portfolio) rollout_buckets are visible
		self.update_rollout_bucket()
		# update allocation, to be compatible with the current portfolio
		self.update_allocation()

	def update_rollout_bucket(self):
		self.rollout_bucket = self.total_rollout_bucket[:len(self.portfolio)]

	def train_blue_dqn(self, trainers, env_name, gen, ALGO = 'dis', pomdp_adv=False):#in this method, rollout and training are done together, opponent sampled from the population
		NUM_EPISODE = 100 #train 100 episodes for the blue to converge to the new best response to red
		EPS_START = max(1.0 * 0.5**(gen - 10), 0.15) if gen >= 10 else 1.0#initial epsilon
		EPS_END = 0.05
		EPS_DECAY = 0.995

		if ALGO == 'dis':  # make env with blue and red policy agent inside,
			assert trainers is not None
			dis_env = make_self_play_env(seed=np.random.choice(np.array(range(len(self.pop)))), return_policy_agent=False, trainers=trainers)[0]# trainer if not None, first is the shared DQN agent, second is the best red policy
			env = EnvironmentWrapper(env_name, ALGO, dis_env, 0)  # the "0" is the index for training blue agent
		elif ALGO == 'TD3_tennis':
			no_graphics = not RENDER
			tennis_env = make_tennis_env.TennisEnvFactory(seed=np.random.choice(np.array(range(len(self.pop)))), no_graphics=no_graphics, pid=-1).getEnv()[0]
			env = EnvironmentWrapper('Tennis', ALGO, tennis_env, 0)
		else:
			env = EnvironmentWrapper(env_name, ALGO)

		blue_dqn = trainers[0]
		average_reward = 0
		eps = EPS_START

		average_red_reward = 0
		red_count = 0
		average_actual_blue_reward = 0
		blue_count = 0

		for it in range(NUM_EPISODE):
			if not pomdp_adv:  #if pomdp_adv, make sure that TD3_actor is never used
				id = np.random.choice(np.array(range(len(self.pop))))
				red_actor = self.pop[id]
				env.set_TD3_actor(red_actor)

			fitness = 0.0;  #here fitness if simplely reward
			total_frame = 0
			state = env.reset()
			env.randomize_neu_adv()

			if pomdp_adv:
				env.try_set_pomdp_adv()   #try to set if opponent to pomdp adv if opponent is adversary, else do nothing

			render_flag = (np.random.random() < 0.05)
			while True:  # unless done

				action = blue_dqn.act(state, eps=eps)
				# action = utils.to_numpy(action)

				next_state, reward, done, info = env.step(copy.deepcopy(action), use_actual_reward=DRQN)  #after calling env.step, evaluator initialized later does not work
				#should be something wrong with the internal red model?
				blue_dqn.step(state, action, reward, next_state, done)

				if render_flag and self.args.render:
					env.render()
				# next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
				state = next_state
				fitness += reward
				total_frame += 1

				# DONE FLAG IS Received
				if done:
					average_red_reward += env.get_red_reward() if env.get_red_reward() is not None else 0
					average_actual_blue_reward += env.get_blue_actual_reward() if env.get_blue_actual_reward() is not None else 0
					red_count += 1 if env.get_red_reward() is not None else 0
					blue_count += 1 if env.get_blue_actual_reward() is not None else 0
					if render_flag: env.env.close();
					break

			average_reward += fitness
			eps = max(EPS_END, EPS_DECAY * eps)

		if gen >= 10 and gen % 5 == 0:
			blue_dqn.save_net('./pytorch_models/train_blue_dqn_step_' + str(gen)+ '.pth')

		average_reward /= NUM_EPISODE
		if red_count != 0:
			average_red_reward /= red_count
		if blue_count != 0:
			average_actual_blue_reward /= blue_count
		return average_reward, average_red_reward, average_actual_blue_reward

	def evaluate_training_fixed_blue(self):  #this evaluate against the training opponent (red pop)
		self.evaluator.pomdp_adv = False
		return self.evaluator.evaluate_fixed_agents(self.trainers[0], self.trainers[1], self.pop)

	def train(self, gen, frame_tracker):
		"""Main training loop to do rollouts, neureoevolution, and policy gradients

			Parameters:
				gen (int): Current epoch of training

			Returns:
				None
		"""
		################ START ROLLOUTS ##############

		# Start Evolution rollouts
		if not ISOLATE_PG:
			for id, actor in enumerate(self.pop):
				if self.evo_flag[id]:
					self.evo_task_pipes[id][0].send((id, gen))
					self.evo_flag[id] = False

		# Sync all learners actor to cpu (rollout) actor
		# (update rollout parameter using the learner parameter, such that rollout worker is up to date)
		for i, learner in enumerate(self.portfolio):  #number of learner
			learner.algo.actor.cpu()
			utils.hard_update(self.rollout_bucket[i], learner.algo.actor)  #rollout bucket is now synchronized with learner to perform rollout for learner actors
			if torch.cuda.is_available(): learner.algo.actor.cuda();

		# Start Learner rollouts
		for rollout_id, learner_id in enumerate(self.allocation):	#number of rollout_size
			if self.roll_flag[rollout_id]:
				self.task_pipes[rollout_id][0].send((learner_id, gen))  #allocation record the id of the learner that bucket should run, so rollout_id is the id of rollout_bucket
				self.roll_flag[rollout_id] = False

		# Start Test rollouts
		if gen % 5 == 0:
			self.test_flag = True
			for pipe in self.test_task_pipes: pipe[0].send((0, gen))


		############# UPDATE PARAMS USING GRADIENT DESCENT ##########
		# main training loop
		if self.replay_buffer.__len__() > self.args.batch_size * 10: ###BURN IN PERIOD
			self.replay_buffer.tensorify()  # Tensorify the buffer for fast sampling

			# Spin up threads for each learner
			threads = [threading.Thread(target=learner.update_parameters, args=(self.replay_buffer, self.args.buffer_gpu, self.args.batch_size, int(self.gen_frames * self.args.gradperstep))) for learner in
					   self.portfolio]  #macheng: do we want to train all the learners?

			# Start threads
			for thread in threads: thread.start()

			# Join threads
			for thread in threads: thread.join()

			# Now update average_policy
			#self.average_policy.cuda()
			if ALGO == 'dis':
				self.average_policy.update()  #update the average_policy parameter with supervised learning

			self.gen_frames = 0

			#########Visualize Learner Critic Function#################
			# if self.replay_buffer.__len__() % 2500 == 0:
			#	visualize_critic(self.portfolio[2], make_self_play_env(trainers=[[],[]])[0], 50)  #arguments: Learner, env, N_GRID


		########## SOFT -JOIN ROLLOUTS FOR EVO POPULATION ############
		if not ISOLATE_PG:
			all_fitness = []; all_net_ids = []; all_eplens = []
			while True:
				for i in range(self.args.pop_size):
					if self.evo_result_pipes[i][1].poll():
						entry = self.evo_result_pipes[i][1].recv()
						all_fitness.append(entry[1]); all_net_ids.append(entry[0]); all_eplens.append(entry[2]); self.gen_frames += entry[2]; self.total_frames += entry[2]
						self.evo_flag[i] = True

				# Soft-join (50%)
				if len(all_fitness) / self.args.pop_size >= self.args.asynch_frac: break

		########## HARD -JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		for i in range(self.args.rollout_size):
			entry = self.result_pipes[i][1].recv()
			learner_id = entry[0]; fitness = entry[1]; num_frames = entry[2]
			self.portfolio[learner_id].update_stats(fitness, num_frames)

			self.gen_frames += num_frames; self.total_frames += num_frames
			if fitness > self.best_score: self.best_score = fitness

			self.roll_flag[i] = True

		# Referesh buffer (housekeeping tasks - pruning to keep under capacity)
		self.replay_buffer.referesh()
		######################### END OF PARALLEL ROLLOUTS ################

		############ PROCESS MAX FITNESS #############
		# ms:best policy is always up to date
		# so here the best learner is saved
		if not ISOLATE_PG:
			champ_index = all_net_ids[all_fitness.index(max(all_fitness))]
			utils.hard_update(self.test_bucket[0], self.pop[champ_index])
			if max(all_fitness) > self.best_score:
				self.best_score = max(all_fitness)
				utils.hard_update(self.best_policy, self.pop[champ_index])
				if SAVE:
					torch.save(self.pop[champ_index].state_dict(), self.args.aux_folder + ENV_NAME +'_best' + SAVETAG)
					print("Best policy saved with score", '%.2f' % max(all_fitness))

		else: #Run PG in isolation
			utils.hard_update(self.test_bucket[0], self.rollout_bucket[0])

		###### TEST SCORE ######
		if self.test_flag:
			self.test_flag = False
			test_scores = []
			for pipe in self.test_result_pipes: #Collect all results
				entry = pipe[1].recv()
				test_scores.append(entry[1])
			test_scores = np.array(test_scores)
			test_mean = np.mean(test_scores); test_std = (np.std(test_scores))

			# Update score to trackers
			frame_tracker.update([test_mean], self.total_frames)
		else:
			test_mean, test_std = None, None


		# NeuroEvolution's probabilistic selection and recombination step
		# ms: this epoch() method implements neuro-evolution
		if not ISOLATE_PG: #seems pop_size and rollout_size must be 10, otherwise this will produce error
			if gen % 5 == 0:
				self.evolver.epoch(gen, self.genealogy, self.pop, all_net_ids, all_fitness, self.rollout_bucket)  #this method also copies learner to evoler
			else:
				self.evolver.epoch(gen, self.genealogy, self.pop, all_net_ids, all_fitness, [])

		# META LEARNING - RESET ALLOCATION USING UCB
		if gen % 1 == 0:
			self.update_allocation()
		# Metrics
		if not ISOLATE_PG:
			champ_len = all_eplens[all_fitness.index(max(all_fitness))]
			champ_wwid = int(self.pop[champ_index].wwid.item())
			max_fit = max(all_fitness)
		else:
			champ_len = num_frames; champ_wwid = int(self.rollout_bucket[0].wwid.item())
			all_fitness = [fitness]; max_fit = fitness; all_eplens = [num_frames]

		return max_fit, champ_len, all_fitness, all_eplens, test_mean, test_std, champ_wwid

	def update_allocation(self):
		self.allocation = ucb(len(self.allocation), self.portfolio, self.args.ucb_coefficient)

	def sim_and_eval_POMDP(self):
		self.evaluator = Evaluator(self, 5, self.trainers, pomdp_adv=True)  # evaluator must be created before train_dqn
		for gen in range(1000000):
			print('gen=', gen)
			blue_score, red_score, actual_blue_score = agent.train_blue_dqn(agent.trainers, ENV_NAME, gen, ALGO='dis', pomdp_adv=True)
			print('Env', ENV_NAME, 'Gen', gen, ", Training average: Blue agent score: ", blue_score, " Red score: ", red_score, " Actual blue score: ", actual_blue_score)
			blue_score, red_score, actual_blue_score = self.evaluator.evaluate()
			print("Evaluation result: Blue agent score: ", blue_score, " Red score: ", red_score, " Actual blue score: ", actual_blue_score)

def test_CERL_Agent():
	args = Parameters()
	"""
	torch.manual_seed(args.seed);
	np.random.seed(args.seed);
	random.seed(args.seed)
	"""
	agent = CERL_Agent(args);
	for i in range(100):
		agent.simulated_annealing(10*len(agent.portfolio))
		print([learner.wwid for learner in agent.portfolio], len(agent.portfolio) + len(agent.complement_portfolio))


if __name__ == "__main__":

	#test_CERL_Agent()
	args = Parameters()  # Create the Parameters class, args also has blue trainer if ALGO == 'dis'
	blue_trainer = args.blue_trainer if (ALGO == 'dis' or ALGO == 'TD3_tennis') else None

	SAVETAG = SAVETAG + '_p' + str(PORTFOLIO_ID)
	SAVETAG = SAVETAG + '_s' + str(SEED)
	if ISOLATE_PG: SAVETAG = SAVETAG + '_pg'

	frame_tracker = utils.Tracker(args.savefolder, ['score_'+ENV_NAME+SAVETAG], '.csv')  #Tracker class to log progress
	max_tracker = utils.Tracker(args.aux_folder, ['pop_max_score_'+ENV_NAME+SAVETAG], '.csv')  #Tracker class to log progress FOR MAX (NOT REPORTED)

	# Set seeds
	torch.manual_seed(args.seed);
	np.random.seed(args.seed);
	random.seed(args.seed)

	# INITIALIZE THE MAIN AGENT CLASS
	agent = CERL_Agent(args) #Initialize the agent

	if ALGO == 'dis':ENV_NAME = 'discrete env'
	print('Running CERL for', ENV_NAME, 'State_dim:', args.state_dim, ' Action_dim:', args.action_dim)
	time_start = time.time()

	#agent.sim_and_eval_POMDP()   #this method evaluate a POMDP blue

	agent.evaluator = Evaluator(agent, 5, agent.trainers)  # evaluator must be created before train_dqn
	#initilize logger file, if using colab, also store in google drive
	from core.misc import initialize_logger
	log_file_name = "scores_id_" + str(PORTFOLIO_ID)
	if SA_FLAG:
		log_file_name += "_SA"
		if RANDOM_WALK:
			log_file_name += "_RW"
	if ISOLATE_PG:
		log_file_name += "_ISO_PG"
	if DRQN:
		log_file_name += "_DRQN"
	if ALGO == 'TD3_tennis':
		log_file_name += '_tennis'

	tpr_file_name = "_".join(log_file_name.split("_")[2:]) + '.p'
	if os.path.exists(tpr_file_name):
		os.remove(tpr_file_name)
	tpr_pickle_file = open(tpr_file_name, 'wb')

	log_file_name += ".log"
	if os.path.exists('/content/drive'):
		if not os.path.exists('/content/drive/My Drive/CERL_colab_logs'):
			os.makedirs('/content/drive/My Drive/CERL_colab_logs')
		initialize_logger('./', log_file_name, '/content/drive/My Drive/CERL_colab_logs/')
	else:
		initialize_logger('./', log_file_name)

	metrics = []

	import subprocess

	start = time.time()

	for gen in range(1, 1000000000): #Infinite generations
		if os.path.exists('/content/drive'):
			log_path = os.path.join('/content/drive/My Drive/CERL_colab_logs/', log_file_name)
			cp_log_path = os.path.join('/content/drive/My Drive/CERL_colab_logs/', 'cp_' + log_file_name)
			subprocess.call(["cp", log_path, cp_log_path])
		# Train one iteration
		# Each train iteration runs a number of (pop_size + rollout_size) = 20 episodes
		best_score, test_len, all_fitness, all_eplen, test_mean, test_std, champ_wwid = agent.train(gen, frame_tracker)  # ms:training of red agent
		# Train blue agent
		if gen % 5 == 0:  #every 5 step
			#print('gen=', gen)
			logging.info('gen = ' + str(gen))
			blue_score, red_score, actual_blue_score = agent.train_blue_dqn(agent.trainers, ENV_NAME, gen, ALGO=ALGO)
			logging.info("Training average: Blue agent score: " + str(blue_score) + " Red score: " + str(red_score) + " Actual blue score: " + str(actual_blue_score))
			blue_score, red_score, actual_blue_score, belief_and_true_type = agent.evaluate_training_fixed_blue()
			logging.info("Training average fixed: Blue agent score: " + str(blue_score) + " Red score: " + str(red_score) + " Actual blue score: " + str(actual_blue_score))
			log_TPR_FPR_TNR_FNR(belief_and_true_type, prefix = "Training average fixed: ")
			pickle.dump(("Training", gen, belief_and_true_type), tpr_pickle_file)
			blue_score, red_score, actual_blue_score, belief_and_true_type = agent.evaluator.evaluate()
			logging.info("Evaluation average fixed: Blue agent score: " + str(blue_score) + ", Red score: " + str(red_score) + " Actual blue score: " + str(actual_blue_score) + '\n')
			log_TPR_FPR_TNR_FNR(belief_and_true_type, prefix = "Evaluation average fixed: ")
			pickle.dump(("Evaluation", gen, belief_and_true_type), tpr_pickle_file)

			if agent.last_portfolio is None:
				logging.info("Current active portfolio: " + str([learner.wwid for learner in agent.portfolio]) + '\n')
			else:
				logging.info("Current active portfolio: " + str([learner.wwid for learner in agent.last_portfolio]) + '\n')
			metrics.append(actual_blue_score - 0.1 * red_score - len(agent.pop))
			if agent.last_portfolio is None:
				logging.info("Portfolio size: " + str(len(agent.portfolio)) + ", Metric: " + str(metrics[-1]) + '\n')
			else:
				logging.info("Portfolio size: " + str(len(agent.last_portfolio)) + ", Metric: " + str(metrics[-1]) + '\n')

			logging.info("Time: " + str(time.time() - start) + '\n')

			if SA_FLAG:
				agent.simulated_annealing(metrics[-1])  #SA has its own copy of metrics record history


		# PRINT PROGRESS
		print('Env', ENV_NAME, 'Gen', gen, 'Frames', agent.total_frames, ' Pop_max/max_ever:','%.2f'%best_score, '/','%.2f'%agent.best_score, ' Avg:','%.2f'%frame_tracker.all_tracker[0][1],
			  ' Frames/sec:','%.2f'%(agent.total_frames/(time.time()-time_start)),
			  ' Champ_len', '%.2f'%test_len, ' Test_score u/std', utils.pprint(test_mean), utils.pprint(test_std), 'savetag', SAVETAG, )
		print("All fitness:", all_fitness)

		# # PRINT MORE DETAILED STATS PERIODICALLY
		if gen % 1 == 0:
			print('Learner Fitness', [utils.pprint(learner.value) for learner in agent.portfolio], 'Sum_stats_resource_allocation', [learner.visit_count for learner in agent.portfolio])
			print('Pop/rollout size', args.pop_size,'/',args.rollout_size, 'gradperstep', args.gradperstep, 'Seed', SEED, 'Portfolio_id', PORTFOLIO_ID)
			try:
				print('Best Policy ever genealogy:', agent.genealogy.tree[int(agent.best_policy.wwid.item())].history)
				print('Champ genealogy:', agent.genealogy.tree[champ_wwid].history)
			except: None
			print()

		max_tracker.update([best_score], agent.total_frames)
		if agent.total_frames > TOTAL_STEPS:
			break

		# Save sum stats
		if PORTFOLIO_ID == 10 or PORTFOLIO_ID == 100:
			visit_counts = np.array([learner.visit_count for learner in agent.portfolio])
			np.savetxt(args.aux_folder + 'allocation_' + ENV_NAME + SAVETAG, visit_counts, fmt='%.3f', delimiter=',')

	# Kill all processes
	try:
		for p in agent.task_pipes: p[0].send('TERMINATE')
		for p in agent.test_task_pipes: p[0].send('TERMINATE')
		for p in agent.evo_task_pipes: p[0].send('TERMINATE')

	except: None


