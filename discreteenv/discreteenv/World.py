import numpy as np

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.goals = None
        # position dimensionality
        self.dim_p = 2
        self.adversary = None
        self.random_pose = True
        self.time_step = None
        self.stats = []
        self.tag_radius = 2.5
        self.use_belief = False   #enable the adv to reason about belief
        self.time = 0

    def reset_time(self):
        self.time = 0

    def new_statistics(self):
        stat = {}
        stat['dis2goal'] = []
        stat['belief'] = []
        stat['tag'] = []
        return stat
    # return all entities in the world
    def set_entity_dim(self):
        for entity in self.entities:
            entity.dim_p = self.dim_p
    @property
    def neutral_policy(self):
        if self.dim_p == 1:
            return np.array([0.45, 0.45, 0.1])
        #stay, forward, fast_forward
        if self.dim_p == 2:
            return np.array([0.1, 0.4, 0.2, 0.05, 0.25])
        # stay, left, right, down, up
        raise Exception('dim_p = 2 neutral policy not defined yet')

    def sample_neutral(self) -> np.array:  #sample neutral action
        neutral_prob = self.neutral_policy
        sample = np.random.choice(list(range(neutral_prob.shape[0])), size=1, p=neutral_prob)[0]
        return sample

    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    @property
    def good_agents(self):
        return [agent for agent in self.agents if agent.good]

    @property
    def adversaries(self):
        return [agent for agent in self.agents if agent.adversary]

    @property
    def neutrals(self):
        return [agent for agent in self.agents if agent.neutral]
    @property
    def unknown_agents(self):
        return [agent for agent in self.agents if not agent.good]
    # update state of the world

    def step(self):
        self.time += 1
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)  #for dqn agent, bind it with dqn action_callback
        p_force = [None] * len(self.agents)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # no need to apply environment force, constraint agent into a finite grid world
        self.integrate_state(p_force)  #constrained implemented
        self.update_agent_internal_state()
        self.update_tag_probe_status()

    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            p_force[i] = agent.action.u
        return p_force

    def integrate_state(self, p_force):
        for i, agent in enumerate(self.agents):
            agent.state.p_pos += p_force[i]
            # constrain into grid world
            self.constrain(agent)
        return

    def constrain(self, agent):  # clip the xy location s.t. within the grid world
        p_pos = agent.state.p_pos
        p_pos[0] = np.clip(p_pos[0], agent.pos_range[0][0], agent.pos_range[0][1])
        if p_pos.shape[0] == 2:
            p_pos[1] = np.clip(p_pos[1], agent.pos_range[1][0], agent.pos_range[1][1])
        else:
            assert p_pos.shape[0] == 1
        return


    def update_agent_internal_state(self):  #obs_action is the action of the opponent
        good_agent = self.good_agents[0]
        #other = self.unknown_agents[0]
        adv_neu = self.unknown_agents[0]
        if good_agent.naive is None and adv_neu.naive == True:
            return #no belief
        likelihood_naive, likelihood_smart = self.get_likelihood()
        self.belief_update(good_agent, likelihood_naive, naive=True) #0: neutral; 1: adv
        if good_agent.naive == False:
            self.belief_update(good_agent, likelihood_smart, naive=False)  # 0: neutral; 1: adv
        #sync naive_belief
        adv_neu.state.naive_belief = good_agent.state.naive_belief

        if good_agent.action.probe == True:
            if adv_neu.adversary:  #if world contains adversary agent
                self.belief_update(good_agent, np.array([0.2, 0.8]), good_agent.naive)
            else:
                assert adv_neu.neutral
                self.belief_update(good_agent, np.array([0.8, 0.2]), good_agent.naive)
        return

    def get_likelihood(self):
        assert len(self.unknown_agents) == 1
        likelihood_naive = None
        likelihood_smart = None
        agent = self.unknown_agents[0]
        naive_adv_policy = agent.naive_policy
        if agent.naive == False and self.use_belief:
            #smart adv scenario if smart adv observes blue belief, then naive_obs
            naive_obs = agent.obs[1:]
        else:
            naive_obs = agent.obs #if not use_belief, smart adv has same obs as naive
        _, likelihood_naive = naive_adv_policy(naive_obs, return_prob=True)
        assert agent.raw_action is not None
        likelihood_naive = likelihood_naive[agent.raw_action]
        likelihood_naive = np.array([self.neutral_policy[agent.raw_action], likelihood_naive])
        if self.good_agents[0].naive == False: #adv should also be smart
            smart_adv_policy = agent.smart_policy
            _, likelihood_smart = smart_adv_policy(agent.obs, return_prob=True)
            assert agent.raw_action is not None
            likelihood_smart = likelihood_smart[agent.raw_action]
            likelihood_smart = np.array([self.neutral_policy[agent.raw_action], likelihood_smart])
        return likelihood_naive, likelihood_smart

    def belief_update(self, agent, likelihood, naive):
        assert agent.good
        if naive:
            belief = agent.state.naive_belief
        else:
            belief = agent.state.smart_belief
        belief = np.array([1-belief[0], belief[0]]) #represented as a vector
        belief = belief * likelihood
        belief = belief/np.sum(belief)
        if naive:
            agent.state.naive_belief = belief[-1:] #belief[1]
            assert agent.state.naive_belief.shape[0] == 1
        else:
            agent.state.smart_belief = belief[-1:]
            assert agent.state.naive_belief.shape[0] == 1
        return

    def update_tag_probe_status(self):  #this is called before the first reward
        good_agent = self.good_agents[0]
        unknown_agent = self.unknown_agents[0]
        if good_agent.action.tag == True:
            good_agent.state.tag_count = min(good_agent.state.tag_count + 1, np.array([2]))
            if np.linalg.norm(good_agent.state.p_pos - unknown_agent.state.p_pos) <= self.tag_radius:
                unknown_agent.state.tagged = True
        else:
            unknown_agent.state.tagged = False
        if good_agent.action.probe == True:
            good_agent.state.probe_count = min(good_agent.state.probe_count + 1, np.array([2]))
        return

    def get_tagged_state(self):
        return self.unknown_agents[0].state.tagged

    def action_n(self):
        actions = []
        for agent in self.policy_agents + self.scripted_agents:
            if agent.raw_action is not None:
                actions.append(agent.action_decoder[agent.raw_action])
            else:
                actions.append(agent.raw_action)
        return actions

    def _randomize_neu_adv(self, neutral_prob = 0.5):
        assert neutral_prob <= 1.0, "prob should less than 1.0"
        if np.random.random() < neutral_prob:
            self.unknown_agents[0].neutral = True
            self.unknown_agents[0].adversary = False
            self.unknown_agents[0].color = np.array([0.25, 0.75, 0.25])
            self.unknown_agents[0].goal = self.landmarks[1] #really important to update goal

        else:
            self.unknown_agents[0].adversary = True
            self.unknown_agents[0].neutral = False
            self.unknown_agents[0].color = np.array([0.75, 0.25, 0.25])
            self.unknown_agents[0].goal = self.landmarks[0]

    def _try_set_pomdp_adv(self):
        if self.unknown_agents[0].adversary:
            self.unknown_agents[0].turn2pomdp_adv()

    def collect_statistics(self, stat):
        unknown_agent = self.unknown_agents[0]
        if unknown_agent.adversary:
            agent_identity = 'adversary'
        else:
            agent_identity = 'neutral'
        p_agent = self.policy_agents[0]
        actions = self.action_n()
        if not p_agent.good:
            actions = [actions[1], actions[0]]

        tag_success = unknown_agent.state.tagged
        naive_belief = self.good_agents[0].state.naive_belief
        naive_belief = naive_belief[0] if naive_belief is not None else naive_belief
        smart_belief = self.good_agents[0].state.smart_belief
        smart_belief = smart_belief[0] if smart_belief is not None else smart_belief
        dis2goal = np.linalg.norm(unknown_agent.state.p_pos - unknown_agent.goal.state.p_pos)

        stat.append_stat(agent_identity, actions[0], actions[1], tag_success, dis2goal, naive_belief, smart_belief)

