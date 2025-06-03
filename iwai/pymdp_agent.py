import logging
import os
from typing import Dict
import numpy as np
from pymdp import utils as mdp_utils
from pymdp.agent import Agent

import utils
from agent.es_registry import ServiceID, ServiceType
from agent.ScalingAgent import ScalingAgent, EVALUATION_CYCLE_DELAY

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))

logger = logging.getLogger("multiscale")
logger.setLevel(logging.INFO)

ROOT = os.path.dirname(__file__)


def generate_normalized_2d_sq_matrix(rows):
    """
    Generates a matrix of the given size (rows x cols) with random values,
    where each row is normalized so that its sum equals 1.
    """
    matrix = np.ones((rows, rows))  # Create a matrix with all values set to 1
    normalized_matrix = matrix / rows  # Normalize so that each row sums to 1
    return normalized_matrix


class pymdp_Agent(ScalingAgent):

    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path=ROOT + "/../config/slo_config.json",
                 es_registry_path=ROOT + "/../config/es_registry.json",
                 log_experience=None):
        super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
                         log_experience)

        # States
        self.throughput = np.arange(0, 101, 1)
        self.quality = np.arange(128, 352, 32)
        self.model_size = np.arange(1, 6, 1)
        self.cores = np.arange(1, 9, 1)

        self.num_states = [len(self.throughput), len(self.quality), len(self.model_size), len(self.cores)]
        self.num_observations = [len(self.throughput), len(self.quality), len(self.model_size), len(self.cores)]
        self.num_factors = len(self.num_states)

        # Actions (u)
        self.u_quality = np.array([-1, 0, 1])  # 'DECREASE', 'KEEP', 'INCREASE'
        self.u_model_size = np.array([-1, 0, 1])  # 'DECREASE', 'KEEP', 'INCREASE'
        self.u_cores = np.array([-1, 0, 1])  # 'DECREASE', 'KEEP', 'INCREASE'

        self.num_controls = (len(self.u_quality), len(self.u_cores), len(self.u_model_size))

        # Dependencies on other state factors (include itself)
        self.B_factor_list = [[0,1,2,3], [1], [2], [3]]

        # Dependencies of factors wrt. actions
        self.B_factor_control_list = [[0,1,2], [0], [1], [2]]

        # Matrices initialization
        A_shapes = [[o_dim] + self.num_states for o_dim in self.num_observations]
        self.A = mdp_utils.obj_array_zeros(A_shapes)
        self.B = mdp_utils.obj_array(self.num_factors)
        self.C = mdp_utils.obj_array_zeros(self.num_observations)
        self.D = mdp_utils.obj_array_zeros(self.num_states)

    def generate_A(self):
        A_matrices = [np.eye(self.throughput.size), np.eye(self.quality.size), np.eye(self.model_size.size),
                      np.eye(self.cores.size)]
        index_to_A = {0: A_matrices[0], 1: A_matrices[1], 2: A_matrices[2], 3: A_matrices[3]}
        ranges = [len(self.throughput), len(self.quality), len(self.model_size), len(self.cores)]

        for i in range(len(ranges)):
            for ii in range(ranges[0]):
                for jj in range(ranges[1]):
                    for kk in range(ranges[2]):
                        for ll in range(ranges[3]):
                            if i == 0:
                                self.A[i][:, :, jj, kk, ll] = index_to_A[0]
                            if i == 1:
                                self.A[i][:, ii, :, kk, ll] = index_to_A[1]
                            if i == 2:
                                self.A[i][:, ii, jj, :, ll] = index_to_A[2]
                            if i == 3:
                                self.A[i][:, ii, jj, kk, :] = index_to_A[3]

    def generate_B(self):
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # Unknown throughput transition matrices, depends on ALL variables and ALL actions
        for ii, _ in enumerate(self.quality):
            for jj, _ in enumerate(self.model_size):
                for kk, _ in enumerate(self.cores):
                    for ll, _ in enumerate(self.u_quality):
                        for mm, _ in enumerate(self.u_model_size):
                            for nn, _ in enumerate(self.u_cores):
                                self.B[0][:, :, ii, jj, kk, ll, mm, nn] = generate_normalized_2d_sq_matrix(self.num_states[0])

        # Unknown quality transition matrices, depends on itself and its action
        self.B[1][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[1])

        # Unknown model_size transition matrices, depends on itself and its action
        self.B[2][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[2])
        self.B[2][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[2])
        self.B[2][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[2])

        # Unknown cores transition matrices, depends on itself and its action
        self.B[3][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[3])

    def generate_C(self):
        # Preferred outcomes, this could be updated at "runtime"
        # throughput has to be above 5. The other states variables do not have any "SLO"
        self.C[0] = np.zeros(self.num_states[0])
        for idx, _ in enumerate(self.C[0]):
            if idx > 5:
                self.C[0][idx] = 3
        self.C[1] = np.zeros(self.num_states[1])
        self.C[2] = np.zeros(self.num_states[2])
        self.C[3] = np.zeros(self.num_states[3])

    def generate_D(self):
        # Initial states
        self.D[0] = [1]  # Throughput at 1
        self.D[1] = [4]  # Quality at 256
        self.D[2] = [3]  # Model size at 3
        self.D[3] = [2]  # Cores at 2

    def generate_uniform_dirichlet_dist(self):
        pA = mdp_utils.dirichlet_like(self.A)
        pB = mdp_utils.dirichlet_like(self.B)
        return pA, pB

    def generate_agent(self, policy_length, learning_rate):
        # Added only 2 parameters: Policy length: number of steps ahead to compute (high values exponentially affect computing time)
        # learning_rate: How fast the B matrices are learnt.
        # There are more parameters, such as gamma or alpha, but for now with these 2 should be fine.
        self.generate_A()
        self.generate_B()
        self.generate_C()
        self.generate_D()

        pA, pB = self.generate_uniform_dirichlet_dist()

        return Agent(A=self.A, B=self.B, C=self.C, D=self.D, pA=pA, pB=pB, policy_len=policy_length,
                     num_controls=self.num_controls, B_factor_list=self.B_factor_list,
                     B_factor_control_list=self.B_factor_control_list,action_selection='deterministic',
                     inference_algo='VANILLA', lr_pB=learning_rate, use_param_info_gain=True)


    def orchestrate_services_optimally(self, services_m):
        pass


if __name__ == '__main__':
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")
    p_agent = pymdp_Agent(services_monitored=[qr_local], prom_server=ps,
                          evaluation_cycle=EVALUATION_CYCLE_DELAY)

    pymdp_agent = p_agent.generate_agent(policy_length=1, learning_rate=0.1)

    print("done")

