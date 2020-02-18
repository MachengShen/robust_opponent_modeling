import random

class Mutation(object):
    def __init__(self, cerl_agent):
        self.cerl_agent = cerl_agent

    def try_mutate(self) -> bool: #return True if succeed
        raise NotImplementedError

    def move_A_to_B(self, a, b, min_num) -> bool: #move a random element from A to B, return True if succeed
        if len(a) >= min_num:  #minimum number of elements that allowed to excecute this mutation
            random.shuffle(a)
            b.append(a.pop())
            return True
        else:
            return False

class Mutation_Add(Mutation):
    #def __init__(self, cerl_agent):
    #    super().__init__(cerl_agent)
    def try_mutate(self) -> bool:
        return self.move_A_to_B(self.cerl_agent.complement_portfolio, self.cerl_agent.portfolio, 1)

class Mutation_Delete(Mutation): #need to make sure portfolio has at least one element left
    def try_mutate(self) -> bool:
        return self.move_A_to_B(self.cerl_agent.portfolio, self.cerl_agent.complement_portfolio, 2)

class Mutation_Exchange(Mutation):
    def try_mutate(self) -> bool:
        port = self.cerl_agent.portfolio
        comp_port = self.cerl_agent.complement_portfolio
        if len(port) > 0 and len(comp_port) > 0:
            random.shuffle(port)
            random.shuffle(comp_port)
            comp_port.append(port.pop())
            port.append(comp_port.pop(-2))
            return True
        else:
            return False