import logging
import os
from typing import Dict
import numpy as np
import pandas as pd
from pymdp import utils as mdp_utils
from pymdp.agent import Agent
import time
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proj_types import ESServiceAction
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

def load_npz_obj_array(filename, save_path):
    data = np.load(os.path.join(save_path, filename), allow_pickle=True)
    values = [data[key] for key in sorted(data.files)]
    return np.array(values, dtype=object)  # ensures it's a proper obj_array

def save_agent_parameters(agent, save_path="../experiments/iwai/saved_agent"):
    os.makedirs(save_path, exist_ok=True)

    # Save A, B, C, D, pA, pB
    np.savez_compressed(os.path.join(save_path, "A.npz"), *agent.A)
    np.savez_compressed(os.path.join(save_path, "B.npz"), *agent.B)
    np.savez_compressed(os.path.join(save_path, "C.npz"), *agent.C)
    np.savez_compressed(os.path.join(save_path, "D.npz"), *agent.D)

    # These are optional but useful if learning is enabled
    np.savez_compressed(os.path.join(save_path, "pA.npz"), *agent.pA)
    np.savez_compressed(os.path.join(save_path, "pB.npz"), *agent.pB)

    print(f"Agent parameters saved to: {save_path}")

def generate_normalized_2d_sq_matrix(rows):
    """
    Generates a matrix of the given size (rows x cols) with random values,
    where each row is normalized so that its sum equals 1.
    """
    matrix = np.ones((rows, rows))  # Create a matrix with all values set to 1
    normalized_matrix = matrix / rows  # Normalize so that each row sums to 1
    return normalized_matrix

class pymdp_Agent(): # ScalingAgent):

    def __init__(self):  #, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
        #          slo_registry_path=ROOT + "/../config/slo_config.json",
        #          es_registry_path=ROOT + "/../config/es_registry.json",
        #          log_experience=None):
        # super().__init__(prom_server, services_monitored, evaluation_cycle, slo_registry_path, es_registry_path,
        #                  log_experience)

        # States
        self.throughput_cv = np.arange(0, 6, 1)
        self.quality_cv = np.arange(128, 352, 32)
        self.model_size = np.arange(1, 6, 1)
        self.cores_cv = np.arange(1, 9, 1)
        self.throughput_qr = np.arange(0, 101, 20)
        self.quality_qr = np.arange(300, 1100, 100)
        self.cores_qr = np.arange(1, 9, 1)

        self.num_states = [len(self.throughput_cv), len(self.quality_cv), len(self.model_size), len(self.cores_cv) ,len(self.throughput_qr),
                           len(self.quality_qr), len(self.cores_qr)]
        self.num_observations = [len(self.throughput_cv), len(self.quality_cv), len(self.model_size), len(self.cores_cv) ,len(self.throughput_qr),
                           len(self.quality_qr), len(self.cores_qr)]
        self.num_factors = len(self.num_states)

        # Actions (u)
        # Don't act, decrease quality, increase quality, decrease cores, increase cores, decrease model size, increase model size
        self.u_cv = np.array([0,1,2,3,4,5,6])
        # Don't act, decrease quality, increase quality, decrease cores, increase cores
        self.u_qr = np.array([0,1,2,3,4])

        self.num_controls = (len(self.u_cv), len(self.u_qr))

        # Dependencies on other state factors (include itself)
        self.B_factor_list = [[0,1,2,3], [1], [2], [3, 6], [4, 5, 6], [5], [3, 6]]
        # thr_cv, q_cv, model, cores_cv, thr_qr, q_qr, cores_qr

        # Dependencies of factors wrt. actions
        self.B_factor_control_list = [[0], [0], [0], [0, 1], [1], [1], [0, 1]]

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

    def generate_B(self):
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # throughput_cv transition matrices
        for ii, _ in enumerate(self.quality_cv):
            for jj, _ in enumerate(self.model_size):
                for kk, _ in enumerate(self.cores_cv):
                    for ll, _ in enumerate(self.u_cv):
                        self.B[0][:, :, ii, jj, kk, ll] = generate_normalized_2d_sq_matrix(self.num_states[0])
        #print("B0: " + str(self.B[0]))

        # quality_cv transition matrices
        num_qcv = len(self.quality_cv)
        B1 = np.zeros((num_qcv, num_qcv, len(self.u_cv)))
        for a in range(len(self.u_cv)):
            for j in range(num_qcv):
                if a in [0, 3, 4, 5, 6]:  # NO CHANGE
                    B1[j, j, a] = 1.0
                elif a == 1:  # DECREASE
                    if j > 0:
                        B1[j - 1, j, a] = 1.0
                    else:
                        B1[j, j, a] = 1.0  # saturate at min
                elif a == 2:  # INCREASE
                    if j < num_qcv - 1:
                        B1[j + 1, j, a] = 1.0
                    else:
                        B1[j, j, a] = 1.0  # saturate at max
        self.B[1] = B1

        # model_size transition matrices
        num_model = len(self.model_size)
        B2 = np.zeros((num_model, num_model, len(self.u_cv)))
        for a in range(len(self.u_cv)):
            for j in range(num_model):
                if a in [0, 1, 2, 3, 4]:  # NO CHANGE
                    B2[j, j, a] = 1.0
                elif a == 5:  # DECREASE
                    if j > 0:
                        B2[j - 1, j, a] = 1.0
                    else:
                        B2[j, j, a] = 1.0
                elif a == 6:  # INCREASE
                    if j < num_model - 1:
                        B2[j + 1, j, a] = 1.0
                    else:
                        B2[j, j, a] = 1.0
        self.B[2] = B2

        # cores_cv transition matrices
        n_states = len(self.cores_cv)
        n_qr_states = len(self.cores_qr)
        n_actions_cv = len(self.u_cv)
        n_actions_qr = len(self.u_qr)
        # Initialize transition matrix: B[3]
        B3 = np.zeros((n_states, n_states, n_qr_states, n_actions_cv, n_actions_qr))
        # Build transitions
        for from_cv_idx, from_cv in enumerate(self.cores_cv):  # current cores_cv
            for qr_idx, qr in enumerate(self.cores_qr):  # current cores_qr
                for a_cv_idx, a_cv in enumerate(self.u_cv):  # CV action
                    for a_qr_idx, a_qr in enumerate(self.u_qr):  # QR action
                        # Determine action effect on cores
                        delta_cv = 0
                        if a_cv == 3:
                            delta_cv = -1
                        elif a_cv == 4:
                            delta_cv = +1
                        delta_qr = 0
                        if a_qr == 3:
                            delta_qr = -1
                        elif a_qr == 4:
                            delta_qr = +1
                        # Compute proposed new states
                        new_cv = from_cv + delta_cv
                        new_qr = qr + delta_qr
                        # Check if transition is valid
                        if (1 <= new_cv <= 8) and (1 <= new_qr <= 8) and (new_cv + new_qr <= PHYSICAL_CORES):
                            to_cv_idx = new_cv - 1
                        else:
                            # Invalid → stay in same state
                            to_cv_idx = from_cv_idx
                        # Set the deterministic transition
                        B3[to_cv_idx, from_cv_idx, qr_idx, a_cv_idx, a_qr_idx] = 1.0
        #print("B3: " + str(B3))
        self.B[3] = B3

        # throughput_qr transition matrices
        for ii, _ in enumerate(self.quality_qr):
            for jj, _ in enumerate(self.cores_qr):
                for kk, _ in enumerate(self.u_qr):
                        self.B[4][:, :, ii, jj, kk] = generate_normalized_2d_sq_matrix(self.num_states[4])
        #print("B4: " + str(self.B[4]))

        # quality_qr transition matrices
        num_q_qr = len(self.quality_qr)
        B5 = np.zeros((num_q_qr, num_q_qr, len(self.u_qr)))
        for a in range(len(self.u_qr)):
            for j in range(num_q_qr):
                if a in [0, 3, 4]:  # NO CHANGE
                    B5[j, j, a] = 1.0
                elif a == 1:  # DECREASE
                    if j > 0:
                        B5[j - 1, j, a] = 1.0
                    else:
                        B5[j, j, a] = 1.0
                elif a == 2:  # INCREASE
                    if j < num_q_qr - 1:
                        B5[j + 1, j, a] = 1.0
                    else:
                        B5[j, j, a] = 1.0
        self.B[5] = B5

        ### B6 - cores_qr
        n_cv = len(self.cores_cv)
        n_qr = len(self.cores_qr)
        n_acv = len(self.u_cv)
        n_aqr = len(self.u_qr)
        # Initialize B[6]
        B6 = np.zeros((n_qr, n_qr, n_cv, n_acv, n_aqr))
        # Construct transitions
        for from_qr_idx, from_qr in enumerate(self.cores_qr):  # current cores_qr
            for cv_idx, cv in enumerate(self.cores_cv):  # current cores_cv
                for a_cv_idx, a_cv in enumerate(self.u_cv):  # CV action
                    for a_qr_idx, a_qr in enumerate(self.u_qr):  # QR action
                        # Determine action effect
                        delta_cv = -1 if a_cv == 3 else 1 if a_cv == 4 else 0
                        delta_qr = -1 if a_qr == 3 else 1 if a_qr == 4 else 0
                        new_cv = cv + delta_cv
                        new_qr = from_qr + delta_qr
                        # Validate transition
                        if (1 <= new_cv <= 8) and (1 <= new_qr <= 8) and (new_cv + new_qr <= PHYSICAL_CORES):
                            to_qr_idx = new_qr - 1
                        else:
                            to_qr_idx = from_qr_idx  # Stay in same state
                        # Set transition
                        B6[to_qr_idx, from_qr_idx, cv_idx, a_cv_idx, a_qr_idx] = 1.0
        #print("B6: " + str(B6))
        self.B[6] = B6

    def generate_C(self):
        # Preferred outcomes, this could be updated at "runtime"
        # throughput_cv has to be above as high as possible.
        self.C[0] = np.zeros(self.num_states[0])
        smooth_values = np.linspace(0.1, 3.0, self.num_states[0] - 1)
        self.C[0][1:] = smooth_values
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
        smooth_values = np.linspace(0.1, 3.0, self.num_states[4] - 1)
        self.C[4][1:] = smooth_values
        # quality_qr as high as possible
        self.C[5] = np.zeros(self.num_states[5])
        self.C[5][-3:] = [0.75, 1.5, 3]
        # Cores_qr
        self.C[6] = np.zeros(self.num_states[6])
        # print("Preferred Outcomes (C)")
        # print(self.C)

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
                     inference_algo='VANILLA', lr_pB=learning_rate, use_param_info_gain=True, use_states_info_gain=True)

    def load_agent_parameters(self, save_path="../experiments/iwai/saved_agent", policy_len=1, lr_pB=1.0):
        A = load_npz_obj_array("A.npz", save_path)
        B = load_npz_obj_array("B.npz", save_path)
        C = load_npz_obj_array("C.npz", save_path)
        D = load_npz_obj_array("D.npz", save_path)
        pA = load_npz_obj_array("pA.npz", save_path)
        pB = load_npz_obj_array("pB.npz", save_path)

        # Create agent with loaded parameters
        agent = Agent(A=A, B=B, C=C, D=D, pA=pA, pB=pB, policy_len=policy_len, lr_pB=lr_pB, num_controls=self.num_controls,
                      B_factor_list=self.B_factor_list, B_factor_control_list=self.B_factor_control_list,
                      action_selection='deterministic', inference_algo='VANILLA', use_param_info_gain=True,
                      use_states_info_gain=True)

        print(f"Agent successfully loaded from: {save_path}")
        return agent

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

    learning_agent = False

    if learning_agent:
        pymdp_agent = p_agent.generate_agent(policy_length=1, learning_rate=1)
    else:
        pymdp_agent = p_agent.load_agent_parameters(save_path="../experiments/iwai/saved_agent", policy_len=1)
    print("Agent ready.")
    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.4f} seconds")

    joint_reward = 0

    logged_data = list()

    for steps in range(50):
        start_time_loop = time.time()

        a_s = pymdp_agent.infer_states(pymdp_state)
        elapsed = time.time() - start_time_loop
        if steps & learning_agent> 0:
            pymdp_agent.update_B(a_s)

        q_pi, G, G_sub = pymdp_agent.infer_policies()

        chosen_action_id = pymdp_agent.sample_action()

        policy_list = pymdp_agent.policies  # shape: (num_policies, policy_len, num_control_factors)

        # Flatten if policy length is 1
        policy_array = np.array(policy_list)
        flattened_policies = policy_array[:, 0, :]  # shape: (num_policies, num_control_factors)

        # Find the index of the selected policy
        policy_index = next(
            i for i, policy in enumerate(flattened_policies)
            if np.array_equal(policy, chosen_action_id)
        )

        # Extract metrics
        efe = G[policy_index]
        info_gain = G_sub["ig_s"][policy_index]  # usually epistemic value
        pragmatic_value = G_sub["r"][policy_index]  # usually extrinsic value

        action_cv = ESServiceAction(int(chosen_action_id[0]))
        action_qr = ESServiceAction(int(chosen_action_id[1]))

        #print("applying actions.")
        (next_state_qr, next_state_cv), joint_reward, done = joint_env.step(action_qr=action_qr, action_cv=action_cv)

        elapsed = time.time() - start_time_loop
        print(f"Loop time: {elapsed:.4f} seconds")

        timestamp = datetime.now().isoformat()

        logged_data.append({
            "timestamp": timestamp,
            "next_state_qr": str(next_state_qr),
            "next_state_cv": str(next_state_cv),
            "action_qr": action_qr.name if hasattr(action_qr, 'name') else str(action_qr),
            "action_cv": action_cv.name if hasattr(action_cv, 'name') else str(action_cv),
            "reward": joint_reward,
            "efe":efe,
            "info_gain": info_gain,
            "pragmatic_value": pragmatic_value,
        })

    save_agent_parameters(pymdp_agent, save_path="../experiments/iwai/saved_agent")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"../experiments/iwai/{timestamp}_pymdp_service_log.csv"
    df = pd.DataFrame(logged_data)
    df.to_csv(log_path, index=False, mode='a', header=not os.path.exists(log_path))
    log_entries = []  # Clear after saving
    print("done")

