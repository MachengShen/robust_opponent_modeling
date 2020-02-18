class Stat(object):
    def __init__(self):
        self.agent_identity = []
        self.good_action = []
        self.adv_action = []
        self.tag_success = []
        self.dis2goal = []
        self.naive_belief = []
        self.smart_belief = []

    def append_stat(self, agent_identity, good_action, adv_action, tag_success, dis2goal, naive_belief, smart_belief):
        self.agent_identity.append(agent_identity)
        self.good_action.append(good_action)
        self.adv_action.append(adv_action)
        self.tag_success.append(tag_success)
        self.dis2goal.append(dis2goal)
        self.naive_belief.append(naive_belief)
        self.smart_belief.append(smart_belief)


