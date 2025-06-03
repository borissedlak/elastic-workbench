import logging
import os
from typing import Dict
import numpy as np
import pandas as pd
import itertools
from pymdp import utils as mdp_utils
from pymdp.agent import Agent
import time

import utils
from agent.es_registry import ServiceID, ServiceType
from agent.ScalingAgent import ScalingAgent, EVALUATION_CYCLE_DELAY
from iwai.dqn_trainer import CV_DATA_QUALITY_STEP, QR_DATA_QUALITY_STEP
from iwai.global_training_env import GlobalTrainingEnv

from iwai.lgbn_training_env import LGBNTrainingEnv

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))

logger = logging.getLogger("multiscale")
logger.setLevel(logging.INFO)

ROOT = os.path.dirname(__file__)


# def generate_normalized_2d_sq_matrix(rows):
#     """
#     Generates a matrix of the given size (rows x cols) with random values,
#     where each row is normalized so that its sum equals 1.
#     """
#     matrix = np.ones((rows, rows))  # Create a matrix with all values set to 1
#     normalized_matrix = matrix / rows  # Normalize so that each row sums to 1
#     return normalized_matrix
# NOT IN USE.

def convert_action(action):
    converted_actions = list()

    # QUALITY action TRANSFORMATION
    if action[0] == 0:
        converted_actions.append(1)
    elif action[0] == 1:
        converted_actions.append(0)
    elif action[0] == 2:
        converted_actions.append(2)

    # Model size action transformation
    if action[1] == 0:
        converted_actions.append(5)
    elif action[1] == 1:
        converted_actions.append(0)
    elif action[1] == 2:
        converted_actions.append(6)

    # Cores action transformation
    if action[2] == 0:
        converted_actions.append(3)
    elif action[2] == 1:
        converted_actions.append(0)
    elif action[2] == 2:
        converted_actions.append(4)

    return converted_actions


class pymdp_Agent(): # ScalingAgent):

    def __init__(self):  #, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
        #          slo_registry_path=ROOT + "/../config/slo_config.json",
        #          es_registry_path=ROOT + "/../config/es_registry.json",
        #          log_experience=None):
        # super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
        #                  log_experience)

        # States
        self.throughput_cv = np.arange(0, 11, 1)
        self.quality_cv = np.arange(128, 352, 32)
        self.model_size = np.arange(1, 6, 1)
        self.cores_cv = np.arange(1, 9, 1)
        self.throughput_qr = np.arange(0, 101, 5)
        self.quality_qr = np.arange(300, 1100, 100)
        self.cores_qr = np.arange(1, 9, 1)

        self.num_states = [len(self.throughput_cv), len(self.quality_cv), len(self.model_size), len(self.cores_cv) ,len(self.throughput_qr),
                           len(self.quality_qr), len(self.cores_qr)]
        self.num_observations = [len(self.throughput_cv), len(self.quality_cv), len(self.model_size), len(self.cores_cv) ,len(self.throughput_qr),
                           len(self.quality_qr), len(self.cores_qr)]
        self.num_factors = len(self.num_states)

        # Actions (u)
        self.u_quality_cv = np.array([-1, 0, 1])  # 'DECREASE', 'KEEP', 'INCREASE'
        self.u_model_size_cv = np.array([-1, 0, 1])  # 'DECREASE', 'KEEP', 'INCREASE'
        self.u_cores_cv = np.array([-1, 0, 1])  # 'DECREASE', 'KEEP', 'INCREASE'
        self.u_quality_qr = np.array([-1, 0, 1])  # 'DECREASE', 'KEEP', 'INCREASE'
        self.u_cores_qr = np.array([-1, 0, 1])  # 'DECREASE', 'KEEP', 'INCREASE'



        self.num_controls = (len(self.u_quality_cv), len(self.u_model_size_cv), len(self.u_cores_cv), len(self.u_quality_qr), len(self.u_cores_qr))

        # Dependencies on other state factors (include itself)
        self.B_factor_list = [[0,1,2,3], [1], [2], [3, 6], [4, 5, 6], [5], [3, 6]]
        # thr_cv, q_cv, model, cores_cv, thr_qr, q_qr, cores_qr, cores

        # Dependencies of factors wrt. actions
        self.B_factor_control_list = [[0,1,2], [0], [1], [2, 4], [3, 4], [3], [2, 4]]

        # Matrices initialization
        A_shapes = [[o_dim] + self.num_states for o_dim in self.num_observations]
        self.A = mdp_utils.obj_array_zeros(A_shapes)
        self.B = mdp_utils.obj_array(self.num_factors)
        self.C = mdp_utils.obj_array_zeros(self.num_observations)
        self.D = mdp_utils.obj_array_zeros(self.num_states)

    def generate_A(self):
        A_matrices = [np.eye(self.throughput_cv.size), np.eye(self.quality_cv.size), np.eye(self.model_size.size), np.eye(self.cores_cv.size),
                      np.eye(self.throughput_qr.size), np.eye(self.quality_qr.size), np.eye(self.cores_qr.size)]
        index_to_A = {0: A_matrices[0], 1: A_matrices[1], 2: A_matrices[2], 3: A_matrices[3], 4: A_matrices[4],
                      5: A_matrices[5], 6: A_matrices[6]}
        ranges = [len(self.throughput_cv), len(self.quality_cv), len(self.model_size), len(self.cores_cv), len(self.throughput_qr),
                  len(self.quality_qr), len(self.cores_qr)]

        for i in range(len(ranges)):
            for ii in range(ranges[0]):
                for jj in range(ranges[1]):
                    for kk in range(ranges[2]):
                        for ll in range(ranges[3]):
                            for mm in range(ranges[4]):
                                for nn in range(ranges[5]):
                                    for oo in range(ranges[6]):
                                        if i == 0:
                                            self.A[i][:, :, jj, kk, ll, mm, nn, oo] = index_to_A[0]
                                        if i == 1:
                                            self.A[i][:, ii, :, kk, ll, mm, nn, oo] = index_to_A[1]
                                        if i == 2:
                                            self.A[i][:, ii, jj, :, ll, mm, nn, oo] = index_to_A[2]
                                        if i == 3:
                                            self.A[i][:, ii, jj, kk, :, mm, nn, oo] = index_to_A[3]
                                        if i == 4:
                                            self.A[i][:, ii, jj, kk, ll, :, nn, oo] = index_to_A[4]
                                        if i == 5:
                                            self.A[i][:, ii, jj, kk, ll, mm, :, oo] = index_to_A[5]
                                        if i == 6:
                                            self.A[i][:, ii, jj, kk, ll, mm, nn, :] = index_to_A[6]

    # def generate_B(self):
    #     for factor in range(self.num_factors):
    #         lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
    #         control_shape = [na for i, na in enumerate(self.num_controls) if i in self.B_factor_control_list[factor]]
    #         factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
    #         self.B[factor] = np.zeros(factor_shape)
    #
    #     # throughput_cv transition matrices
    #     for ii, _ in enumerate(self.quality_cv):
    #         for jj, _ in enumerate(self.model_size):
    #             for kk, _ in enumerate(self.cores_cv):
    #                 for ll, _ in enumerate(self.u_quality_cv):
    #                     for mm, _ in enumerate(self.u_model_size_cv):
    #                         for nn, _ in enumerate(self.u_cores_cv):
    #                             self.B[0][:, :, ii, jj, kk, ll, mm, nn] = generate_normalized_2d_sq_matrix(self.num_states[0])
    #
    #     # quality_cv transition matrices
    #     self.B[1][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
    #     self.B[1][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])
    #     self.B[1][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[1])
    #
    #     # model_size transition matrices
    #     self.B[2][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[2])
    #     self.B[2][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[2])
    #     self.B[2][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[2])
    #
    #     # cores_cv transition matrices
    #     for ii, _ in enumerate(self.cores_qr):
    #         for jj, _ in enumerate(self.u_cores_cv):
    #             for kk, _ in enumerate(self.u_cores_qr):
    #                 self.B[3][:, :, ii, jj, kk] = generate_normalized_2d_sq_matrix(self.num_states[3])
    #
    #     # throughput_qr transition matrices
    #     for ii, _ in enumerate(self.quality_qr):
    #         for jj, _ in enumerate(self.cores_qr):
    #             for kk, _ in enumerate(self.u_quality_qr):
    #                 for ll, _ in enumerate(self.u_cores_qr):
    #                     self.B[4][:, :, ii, jj, kk, ll] = generate_normalized_2d_sq_matrix(self.num_states[4])
    #
    #     # quality_qr transition matrices
    #     self.B[5][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[5])
    #     self.B[5][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[5])
    #     self.B[5][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[5])
    #
    #     # cores_qr transition matrices
    #     for ii, _ in enumerate(self.cores_cv):
    #         for jj, _ in enumerate(self.u_cores_cv):
    #             for kk, _ in enumerate(self.u_cores_qr):
    #                 self.B[6][:, :, ii, jj, kk] = generate_normalized_2d_sq_matrix(self.num_states[6])

    def generate_B(self):
        control_mapping = {
            0: len(self.u_quality_cv),
            1: len(self.u_model_size_cv),
            2: len(self.u_cores_cv),
            3: len(self.u_quality_qr),
            4: len(self.u_cores_qr),
        }

        for factor in range(self.num_factors):
            lagging_shape = [self.num_states[i] for i in self.B_factor_list[factor]]
            control_shape = [control_mapping[i] for i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

            index_shape = lagging_shape + control_shape
            n_states = self.num_states[factor]

            # Uniform probability vector
            uniform_probs = np.full(n_states, 1.0 / n_states)

            for idx in itertools.product(*[range(s) for s in index_shape]):
                full_index = (slice(None),) + idx  # Full slice over to_state axis
                self.B[factor][full_index] = uniform_probs

    def generate_C(self):
        # Preferred outcomes, this could be updated at "runtime"
        # throughput_cv has to be above as high as possible.
        self.C[0] = np.zeros(self.num_states[0])
        for idx, _ in enumerate(self.C[0]):
            if idx == 2:
                self.C[0][idx] = 0.75
            elif idx == 3:
                self.C[0][idx] = 1.5
            elif idx >= 4:
                self.C[0][idx] = 3
        # quality_cv as high as possible, last 3 values are valuable.
        self.C[1] = np.zeros(self.num_states[1])
        self.C[1][-3:] = [0.75, 1.5, 3]
        # model_size as high as possible, last 3 values are valuable.
        self.C[2] = np.zeros(self.num_states[2])
        self.C[2][-3:] = [0.75, 1.5, 3]
        # cores_cv
        self.C[3] = np.zeros(self.num_states[3])
        # Throughput_qr as high as possible
        self.C[4] = np.zeros(self.num_states[4])
        smooth_values = np.linspace(0.75, 3.0, 21 - 8)
        self.C[4][8:21] = smooth_values
        # quality_qr as high as possible
        self.C[5] = np.zeros(self.num_states[5])
        self.C[5][-3:] = [0.75, 1.5, 3]
        # Cores_qr
        self.C[6] = np.zeros(self.num_states[6])

    def generate_D(self):
        # Initial states
        self.D[0] = np.zeros(self.num_states[0])
        self.D[0][1] = 1  # Throughput_cv at 1
        self.D[1] = np.zeros(self.num_states[1])
        self.D[1][4] = 1  # Quality_cv at 256
        self.D[2] = np.zeros(self.num_states[2])
        self.D[2][2] = 1  # Model_size at 3
        self.D[3] = np.zeros(self.num_states[5])
        self.D[3][1] = 1  # Cores at 2
        self.D[4] = np.zeros(self.num_states[4])
        self.D[4][1] = 1  # Throughput_qr at 5
        self.D[5] = np.zeros(self.num_states[5])
        self.D[5][4] = 1  # Quality_qr at 256
        self.D[6] = np.zeros(self.num_states[6])
        self.D[6][1] = 1  # Cores at 2

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

    # def valid_joint_actions(self, cores_cv, cores_qr, max_cores=PHYSICAL_CORES):
    #
    #     joint_action_space = list(itertools.product(
    #         self.u_quality_cv,
    #         self.u_model_size_cv,
    #         self.u_cores_cv,
    #         self.u_quality_qr,
    #         self.u_cores_qr
    #     ))
    #     valid = list()
    #     for joint in joint_action_space:
    #         dq_cv = joint[2]
    #         dq_qr = joint[4]
    #         new_cv = cores_cv + dq_cv
    #         new_qr = cores_qr + dq_qr
    #         new_total = new_cv + new_qr
    #         if 1 <= new_cv <= max_cores and 1 <= new_qr <= max_cores and new_total <= max_cores:
    #             valid.append(joint)
    #     return np.array(valid).reshape(len(valid), 1, 5)

    def generate_valid_policies(self, qs, max_cores=PHYSICAL_CORES):
        # Reduces the amount of candidate policies by removing those actions that try to go beyond the state limits
        # Get MAP indices from belief distributions
        qcv_idx = np.argmax(qs[1])  # quality_cv
        msize_idx = np.argmax(qs[2])  # model_size
        ccv_idx = np.argmax(qs[3])  # cores_cv
        qqr_idx = np.argmax(qs[5])  # quality_qr
        cqr_idx = np.argmax(qs[6])  # cores_qr

        def is_valid_action(idx, action, value_list):
            """Check if the action is valid at a given state index."""
            if action == 1 and idx == len(value_list) - 1:
                return False
            if action == -1 and idx == 0:
                return False
            return True

        valid_policies = []

        for u_qcv in self.u_quality_cv:
            if not is_valid_action(qcv_idx, u_qcv, self.quality_cv):
                continue

            for u_msize in self.u_model_size_cv:
                if not is_valid_action(msize_idx, u_msize, self.model_size):
                    continue

                for u_ccv in self.u_cores_cv:
                    if not is_valid_action(ccv_idx, u_ccv, self.cores_cv):
                        continue

                    for u_qqr in self.u_quality_qr:
                        if not is_valid_action(qqr_idx, u_qqr, self.quality_qr):
                            continue

                        for u_cqr in self.u_cores_qr:
                            if not is_valid_action(cqr_idx, u_cqr, self.cores_qr):
                                continue

                            # Check resource constraints
                            new_cv = self.cores_cv[ccv_idx] + u_ccv
                            new_qr = self.cores_qr[cqr_idx] + u_cqr
                            total = new_cv + new_qr
                            if (1 <= new_cv <= max_cores) and (1 <= new_qr <= max_cores) and (total <= max_cores):
                                valid_policies.append([[u_qcv, u_msize, u_ccv, u_qqr, u_cqr]])

        return np.array(valid_policies)

    def orchestrate_services_optimally(self, services_m):
        pass


if __name__ == '__main__':
    # ps = "http://localhost:9090"
    # qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    # cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")
    start_time = time.time()
    df = pd.read_csv("../share/metrics/LGBN.csv")

    env_qr = LGBNTrainingEnv(ServiceType.QR, step_data_quality=QR_DATA_QUALITY_STEP)
    env_qr.reload_lgbn_model(df)

    env_cv = LGBNTrainingEnv(ServiceType.CV, step_data_quality=CV_DATA_QUALITY_STEP)
    env_cv.reload_lgbn_model(df)

    # Wrap in joint environment
    joint_env = GlobalTrainingEnv(env_qr, env_cv, max_cores=PHYSICAL_CORES)

    init_state_qr, init_state_cv = joint_env.reset()

    pymdp_state_qr = init_state_qr.for_pymdp('qr')
    pymdp_state_cv = init_state_cv.for_pymdp('cv')
    pymdp_state = pymdp_state_cv + pymdp_state_qr

    print("Env ready.")
    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.4f} seconds")

    start_time = time.time()
    p_agent = pymdp_Agent()
    pymdp_agent = p_agent.generate_agent(policy_length=1, learning_rate=1)
    print("Agent ready.")
    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.4f} seconds")

    acc_reward = list()
    joint_reward = 0

    for steps in range(10):
        start_time = time.time()

        a_s = pymdp_agent.infer_states(pymdp_state)
        print("states inferred")
        if steps > 0:
            pymdp_agent.update_B(a_s)
            print("Updated B")

        valid_policies = p_agent.generate_valid_policies(qs = a_s, max_cores=PHYSICAL_CORES)
        pymdp_agent.policies = valid_policies
        print("Policies validated. Candidate policies: " + str(len(valid_policies)))

        q_pi, G, G_sub = pymdp_agent.infer_policies()
        print("Policies inferred")
        chosen_action_id = pymdp_agent.sample_action()
        print("Chosen action")
        actions = convert_action(chosen_action_id)

        for act in actions:
            print("applying action.")
            (next_state_qr, next_state_cv), joint_reward, done = joint_env.step(action_qr=act, action_cv=act)

        print("next state")
        print(next_state_qr)
        print(next_state_cv)
        acc_reward.append(joint_reward)
        print(acc_reward)
        elapsed = time.time() - start_time
        print(f"Loop time: {elapsed:.4f} seconds")


    print("done")

