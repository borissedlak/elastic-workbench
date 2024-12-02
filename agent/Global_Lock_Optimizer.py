from agent.ScalingAgent_v2 import ScalingAgent


class Global_Lock_Optimizer:
    def __init__(self, agents: [ScalingAgent]):
        self.s_agents = agents

    def evaluate_slof(self):
        pass

    def estimate_swapping(self):
        pass

    def swap_core(self):
        pass
