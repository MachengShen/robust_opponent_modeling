from core.buffer import Buffer
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
import threading
from core.models import Actor
from core import mod_utils as utils
from core.learner import Learner
import numpy as np
from core.env_wrapper import EnvironmentWrapper
from discreteenv.make_envs import make_self_play_env
import random
import copy
#from unityTennisEnv import make_tennis_env

#this class takes in a blue_agent_dqn(with its corresponding internal model) in the CERL_agent and evaluate its performance
#by training a red agent against blue, and record the blue and red score
BUFFER_SIZE = 1000000
TRAIN_ITERATION = 50  #number of iteration for training red agent
#EVALUATION_ITERATION = 10  #number of iterations for evaluation, total # equals # of workers * EVALUATION_ITERATION
class Evaluator(object):
    def __init__(self, CERL_agent, num_workers, trainers, pomdp_adv=False):  #trainers first is the blue agent and second is the red model
        self.num_workers = num_workers
        self.trainers = trainers
        self.pomdp_adv = pomdp_adv
        self.args = CERL_agent.args
        self.drqn = CERL_agent.args.drqn  #denote if blue uses drqn
        if self.pomdp_adv:
            self.trainers = [trainers[0], None] #make sure the red model is never used
        self.buffer_gpu = CERL_agent.args.buffer_gpu
        self.batch_size = CERL_agent.args.batch_size
        self.algo = CERL_agent.args.algo
        self.state_dim = CERL_agent.args.state_dim
        self.action_dim = CERL_agent.args.action_dim
        self.buffer = Buffer(BUFFER_SIZE, self.buffer_gpu)  #initialize own replay buffer
        self.data_bucket = self.buffer.tuples
        self.evo_task_pipes = [Pipe() for _ in range(self.num_workers)]
        self.evo_result_pipes = [Pipe() for _ in range(self.num_workers)]
        self.actual_red_worker = Actor(CERL_agent.args.state_dim, CERL_agent.args.action_dim, -1, 'dis')  #this model is shared accross the workers
        self.actual_red_worker.share_memory()
        self.td3args = {'policy_noise': 0.2, 'policy_noise_clip': 0.5, 'policy_ups_freq': 2, 'action_low': CERL_agent.args.action_low,
                   'action_high': CERL_agent.args.action_high, 'cerl_args': self.args}
        self.renew_learner()#now we are not using new learner for each iteration
        self.rollout_bucket = [self.actual_red_worker for i in range(num_workers)]
        self.workers = [Process(target=rollout_worker, args=(id, 3, self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], False, self.data_bucket, self.rollout_bucket, 'dummy_name', None, 'dis', self.trainers, False, self.pomdp_adv)) for id in range(num_workers)]

        for worker in self.workers: worker.start()
        self.evo_flag = [True for _ in range(self.num_workers)]

    #def initialize(self, actor_in):  #use the given actor parameter to initialize the red actor
    #    utils.hard_update(self.actual_red_actor, actor_in)

    def renew_learner(self):  #create a new learning agent, with randomized initial parameter
        self.learner = Learner(-1, self.algo, self.state_dim, self.action_dim, actor_lr=5e-5, critic_lr=1e-3, gamma=0.99, tau=5e-3, init_w=True, **self.td3args)
        self.actual_red_actor = self.learner.algo.actor

    def collect_trajectory(self):
        utils.hard_update(self.actual_red_worker, self.actual_red_actor)  #first snyc the actor

        #launch rollout_workers
        for id, actor in enumerate(self.rollout_bucket):
            if self.evo_flag[id]:
                self.evo_task_pipes[id][0].send((id, 0))  #second argument in send is dummy
                self.evo_flag[id] = False

        #wait for the rollout to complete and record fitness
        all_fitness = []
        for i in range(self.num_workers):
            entry = self.evo_result_pipes[i][1].recv()
            all_fitness.append(entry[1])
            self.evo_flag[i] = True

        self.buffer.referesh()  #update replay buffer

        return all_fitness

    def train_red(self, training_iterations):  #alternate between collect_trajectory and parameter update
        while self.buffer.__len__() < self.batch_size * 10:  ###BURN IN PERIOD
            self.collect_trajectory()

        for i in range(training_iterations):
            self.collect_trajectory()
            self.buffer.tensorify()  # Tensorify the buffer for fast sampling
            self.learner.update_parameters(self.buffer, self.buffer_gpu, self.batch_size, 2)  #2 update steps

    def evaluate(self):     #evaluate the quality of blue agent policy, by training a red against it, after evaluation, erase the reply buffer and renew learner
        self.train_red(TRAIN_ITERATION)
        self.clear_buffer()
        #self.renew_learner()
        return self.evaluate_fixed_agents(self.trainers[0], self.trainers[1], [self.actual_red_actor])  #calculate the mean and std of the evaluation metric

    def evaluate_fixed_agents(self, blue_dqn, red_model, red_actor_list, num_iterations=25): #evaluate the performance given agents, use random neutral and red agent
        if self.algo == 'dis':  # make env with blue and red policy agent inside,
            dis_env = make_self_play_env(seed=0, return_policy_agent=False, trainers=[blue_dqn, red_model])[0]  # trainer if not None, first is the shared DQN agent, second is the best red policy
            env = EnvironmentWrapper('', self.algo, dis_env, 0)  # the "0" is the index for training blue agent
        elif self.algo == 'TD3_tennis':
            tennis_env = make_tennis_env.TennisEnvFactory(seed=np.random.choice(np.array(range(len(self.pop)))), no_graphics=True, pid=-1).getEnv()[0]
            env = EnvironmentWrapper('Tennis', self.algo, tennis_env, 0)
        else:
            raise Exception("only work for 'dis' envir?")
        average_reward = 0
        eps = 0
        average_red_reward = 0
        red_count = 0
        average_actual_blue_reward = 0
        blue_count = 0
        belief_and_true_type_list = []
        assert len(red_actor_list) is not None, "make sure to input a list of possible red"
        for it in range(num_iterations):
            belief_and_true_type = []
            if not self.pomdp_adv:  # if pomdp_adv, make sure that TD3_actor is never used
                red_actor = random.choice(red_actor_list)
                env.set_TD3_actor(red_actor)
            fitness = 0.0;  # here fitness if simplely reward
            state = env.reset()
            belief_and_true_type.append(env.belief_and_true_type())
            env.randomize_neu_adv()

            if self.pomdp_adv:
                env.try_set_pomdp_adv()  # try to set if opponent to pomdp adv if opponent is adversary, else do nothing

            render_flag = (np.random.random() < 0.05)
            while True:  # unless done
                action = blue_dqn.act(state, eps=eps)
                next_state, reward, done, info = env.step(copy.deepcopy(action), use_actual_reward=self.drqn)
                belief_and_true_type.append(env.belief_and_true_type())
                if render_flag and self.args.render:
                    env.render()

                state = next_state
                fitness += reward

                if done:
                    average_red_reward += env.get_red_reward() if env.get_red_reward() is not None else 0
                    average_actual_blue_reward += env.get_blue_actual_reward() if env.get_blue_actual_reward() is not None else 0
                    red_count += 1 if env.get_red_reward() is not None else 0
                    blue_count += 1 if env.get_blue_actual_reward() is not None else 0
                    if render_flag: env.env.close();
                    break
            belief_and_true_type_list.append(belief_and_true_type)
            average_reward += fitness
        average_reward /= num_iterations
        if red_count != 0:
            average_red_reward /= red_count
        if blue_count != 0:
            average_actual_blue_reward /= blue_count
        return average_reward, average_red_reward, average_actual_blue_reward, belief_and_true_type_list

    def clear_buffer(self):
        self.buffer.clear_buffer_data()  #reinitialize replay buffer

    def kill_processes(self):
        for id, actor in enumerate(self.rollout_bucket):
            self.evo_task_pipes[id][0].send(('TERMINATE', 0))  #second argument in send is dummy

    def __del__(self):
        self.kill_processes()