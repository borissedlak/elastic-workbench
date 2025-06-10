import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from pymdp import utils as mdp_utils
from pymdp.agent import Agent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proj_types import ESServiceAction
import utils
from agent.es_registry import ServiceType
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

class OptimizedPymdpAgent():
    """优化版本的PyMDP智能体，专注于矩阵构建的向量化优化"""

    def __init__(self):
        # States
        self.throughput_cv = np.arange(0, 6, 1)
        self.quality_cv = np.arange(128, 352, 32)
        self.model_size = np.arange(1, 6, 1)
        self.cores_cv = np.arange(1, 8, 1)  # Should use 7 cores max per service
        self.throughput_qr = np.arange(0, 101, 20)
        self.quality_qr = np.arange(300, 1100, 100)
        self.cores_qr = np.arange(1, 8, 1)  # Should use 7 cores max per service

        self.num_states = [len(self.throughput_cv), len(self.quality_cv), len(self.model_size), len(self.cores_cv), len(self.throughput_qr),
                           len(self.quality_qr), len(self.cores_qr)]
        self.num_observations = [len(self.throughput_cv), len(self.quality_cv), len(self.model_size), len(self.cores_cv), len(self.throughput_qr),
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

    def generate_A_optimized(self):
        """
        优化版本的A矩阵生成 - 使用向量化操作替代7层嵌套循环
        
        原始版本需要4,321,440次循环操作
        优化版本使用numpy broadcasting和高级索引，大幅减少计算量
        """
        print("Generating A matrices with vectorized operations...")
        start_time = time.perf_counter()
        
        # 预计算所有需要的单位矩阵
        identity_matrices = [
            np.eye(self.throughput_cv.size),    # 6x6
            np.eye(self.quality_cv.size),       # 7x7
            np.eye(self.model_size.size),       # 5x5
            np.eye(self.cores_cv.size),         # 7x7
            np.eye(self.throughput_qr.size),    # 6x6
            np.eye(self.quality_qr.size),       # 8x8
            np.eye(self.cores_qr.size)          # 7x7
        ]
        
        # 为每个状态因子构建A矩阵
        for factor_idx in range(self.num_factors):
            print(f"  Building A[{factor_idx}] for factor {factor_idx}...")
            self.A[factor_idx] = self._build_A_factor_vectorized(factor_idx, identity_matrices[factor_idx])
        
        elapsed = time.perf_counter() - start_time
        print(f"A matrix generation completed in {elapsed:.4f} seconds")
        
    def _build_A_factor_vectorized(self, factor_idx, identity_matrix):
        """
        为单个状态因子构建A矩阵的真正向量化版本
        
        使用numpy广播避免所有显式循环
        """
        # 获取A[factor_idx]的完整形状
        full_shape = [self.num_observations[factor_idx]] + self.num_states
        
        # 创建新的维度排列，将单位矩阵扩展到正确的维度
        # 对于factor_idx，单位矩阵应该在观测维度(0)和状态维度(factor_idx+1)上
        
        # 创建扩展后的单位矩阵形状
        extended_shape = [1] * len(full_shape)
        extended_shape[0] = self.num_observations[factor_idx]  # 观测维度
        extended_shape[factor_idx + 1] = self.num_states[factor_idx]  # 对应的状态维度
        
        # 将单位矩阵扩展到正确的形状
        # 首先将2D单位矩阵reshape到包含所有维度的形状
        identity_extended = np.zeros(extended_shape)
        
        # 使用高级索引填充单位矩阵
        obs_indices = np.arange(self.num_observations[factor_idx])
        state_indices = np.arange(self.num_states[factor_idx])
        
        # 创建索引网格
        obs_grid, state_grid = np.meshgrid(obs_indices, state_indices, indexing='ij')
        
        # 只在对角线位置（obs_idx == state_idx）设置为1
        diagonal_mask = (obs_grid == state_grid)
        
        # 构建完整的索引
        full_indices = [slice(None)] * len(extended_shape)
        full_indices[0] = obs_grid[diagonal_mask]
        full_indices[factor_idx + 1] = state_grid[diagonal_mask]
        
        # 将其他维度设为0（广播时会自动扩展）
        for dim in range(1, len(extended_shape)):
            if dim != factor_idx + 1:
                full_indices[dim] = 0
        
        identity_extended[obs_grid[diagonal_mask], ...] = 1.0
        
        # 等等，这还是太复杂了。让我用更简单的方法
        
        # 直接使用numpy的tile和reshape
        A_factor = np.zeros(full_shape)
        
        # 使用更简单直接的方法：创建正确维度的单位矩阵并广播
        # A[factor_idx]的维度是：[obs_dim, state_0, state_1, ..., state_6]
        # 对于factor_idx，单位矩阵应该在obs_dim和state_factor_idx之间
        
        # 创建广播的形状
        broadcast_shape = [1] * len(full_shape)
        broadcast_shape[0] = self.num_observations[factor_idx]  # 观测维度
        broadcast_shape[factor_idx + 1] = self.num_states[factor_idx]  # 对应状态维度
        
        # 将单位矩阵重塑为可广播的形状
        identity_broadcast = identity_matrix.reshape(broadcast_shape)
        
        # 使用numpy的broadcast_to扩展到完整维度
        A_factor = np.broadcast_to(identity_broadcast, full_shape).copy()
        
        return A_factor

    def generate_A_original(self):
        """
        原始版本的A矩阵生成 - 保留用于性能对比
        """
        print("Generating A matrices with original nested loops...")
        start_time = time.perf_counter()
        
        A_matrices = [np.eye(self.throughput_cv.size), np.eye(self.quality_cv.size), np.eye(self.model_size.size), np.eye(self.cores_cv.size),
                      np.eye(self.throughput_qr.size), np.eye(self.quality_qr.size), np.eye(self.cores_qr.size)]
        index_to_A = {0: A_matrices[0], 1: A_matrices[1], 2: A_matrices[2], 3: A_matrices[3], 4: A_matrices[4],
                      5: A_matrices[5], 6: A_matrices[6]}
        ranges = [len(self.throughput_cv), len(self.quality_cv), len(self.model_size), len(self.cores_cv), len(self.throughput_qr),
                  len(self.quality_qr), len(self.cores_qr)]

        loop_count = 0
        for i in range(len(ranges)):
            for ii in range(ranges[0]):
                for jj in range(ranges[1]):
                    for kk in range(ranges[2]):
                        for ll in range(ranges[3]):
                            for mm in range(ranges[4]):
                                for nn in range(ranges[5]):
                                    for oo in range(ranges[6]):
                                        loop_count += 1
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
        
        elapsed = time.perf_counter() - start_time
        print(f"Original A matrix generation completed in {elapsed:.4f} seconds (loops: {loop_count})")

    def generate_B(self):
        """B矩阵生成 - 暂时保持原始版本"""
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
                        if (1 <= new_cv <= 7) and (1 <= new_qr <= 7) and (new_cv + new_qr <= PHYSICAL_CORES):
                            to_cv_idx = new_cv - 1
                        else:
                            # Invalid → stay in same state
                            to_cv_idx = from_cv_idx
                        # Set the deterministic transition
                        B3[to_cv_idx, from_cv_idx, qr_idx, a_cv_idx, a_qr_idx] = 1.0
        self.B[3] = B3

        # throughput_qr transition matrices
        for ii, _ in enumerate(self.quality_qr):
            for jj, _ in enumerate(self.cores_qr):
                for kk, _ in enumerate(self.u_qr):
                        self.B[4][:, :, ii, jj, kk] = generate_normalized_2d_sq_matrix(self.num_states[4])

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
                        if (1 <= new_cv <= 7) and (1 <= new_qr <= 7) and (new_cv + new_qr <= PHYSICAL_CORES):
                            to_qr_idx = new_qr - 1
                        else:
                            to_qr_idx = from_qr_idx  # Stay in same state
                        # Set transition
                        B6[to_qr_idx, from_qr_idx, cv_idx, a_cv_idx, a_qr_idx] = 1.0
        self.B[6] = B6

    def generate_C(self):
        # Preferred outcomes, this could be updated at "runtime"
        # throughput_cv >= 5
        self.C[0][0] = -5
        self.C[0][1:] = np.linspace(0.1, 4.0, self.num_states[0] - 1)
        # quality_cv as high as possible, last two states best
        self.C[1][:6] = np.linspace(0.1, 1.0, self.num_states[1] - 1)
        self.C[1][6] = 0.5
        # model_size as high as possible, last two states best
        self.C[2] = np.array([1, 2, 3, 1.5, 1])
        # cores_cv
        self.C[3] = np.zeros(self.num_states[3])
        # Throughput_qr >= 60
        self.C[4][0] = -5
        self.C[4][1:] = [1.25, 2.5, 4, 4, 4]
        # quality_qr as high as possible, last two states best
        self.C[5][:7] = np.linspace(0.1, 4.0, self.num_states[5] - 1)
        self.C[5][7] = 4.0
        # Cores_qr
        self.C[6] = np.zeros(self.num_states[6])

    def generate_D(self):
        # Initial states
        self.D[0] = np.zeros(self.num_states[0])
        self.D[0][2] = 0.5  # Throughput_cv at 3
        self.D[0][3] = 0.5  # Throughput_cv at 4
        self.D[1] = np.zeros(self.num_states[1])
        self.D[1][4] = 1  # Quality_cv at 256
        self.D[2] = np.zeros(self.num_states[2])
        self.D[2][1] = 1  # Model_size at 2
        self.D[3] = np.zeros(self.num_states[3])
        self.D[3][1] = 1  # Cores at 2
        self.D[4] = np.zeros(self.num_states[4])
        self.D[4][1] = 1  # Throughput_qr at 5
        self.D[5] = np.zeros(self.num_states[5])
        self.D[5][4] = 1  # Quality_qr at 700
        self.D[6] = np.zeros(self.num_states[6])
        self.D[6][1] = 1  # Cores at 2

    def generate_uniform_dirichlet_dist(self):
        pA = mdp_utils.dirichlet_like(self.A)
        pB = mdp_utils.dirichlet_like(self.B)
        return pA, pB

    def generate_agent(self, policy_length, learning_rate, action_selection, alpha, use_optimized=True):
        """
        生成PyMDP智能体
        
        Args:
            use_optimized: 是否使用优化的A矩阵生成方法
        """
        if use_optimized:
            self.generate_A_optimized()
        else:
            self.generate_A_original()
            
        self.generate_B()
        self.generate_C()
        self.generate_D()

        pA, pB = self.generate_uniform_dirichlet_dist()

        return Agent(A=self.A, B=self.B, C=self.C, D=self.D, pA=pA, pB=pB, policy_len=policy_length,
                     num_controls=self.num_controls, B_factor_list=self.B_factor_list,
                     B_factor_control_list=self.B_factor_control_list,action_selection=action_selection,
                     alpha=alpha,
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
                      action_selection='deterministic', inference_algo='VANILLA')

        print(f"Agent successfully loaded from: {save_path}")
        return agent


def performance_comparison_test():
    """
    性能对比测试函数
    """
    print("=== PyMDP Agent Performance Comparison Test ===")
    
    # 创建智能体实例
    agent = OptimizedPymdpAgent()
    
    print("\n1. Testing Original A Matrix Generation...")
    start_time = time.perf_counter()
    agent.generate_A_original()
    original_time = time.perf_counter() - start_time
    A_original = [a.copy() for a in agent.A]  # 保存原始结果
    
    print(f"Original method time: {original_time:.4f} seconds")
    
    # 重置A矩阵
    A_shapes = [[o_dim] + agent.num_states for o_dim in agent.num_observations]
    agent.A = mdp_utils.obj_array_zeros(A_shapes)
    
    print("\n2. Testing Optimized A Matrix Generation...")
    start_time = time.perf_counter()
    agent.generate_A_optimized()
    optimized_time = time.perf_counter() - start_time
    A_optimized = [a.copy() for a in agent.A]  # 保存优化结果
    
    print(f"Optimized method time: {optimized_time:.4f} seconds")
    
    # 验证结果一致性
    print("\n3. Verifying Result Consistency...")
    all_equal = True
    for i in range(len(A_original)):
        if not np.allclose(A_original[i], A_optimized[i], rtol=1e-10):
            print(f"ERROR: A[{i}] matrices are not equal!")
            all_equal = False
        else:
            print(f"✓ A[{i}] matrices are identical")
    
    if all_equal:
        print("\n✅ All A matrices are identical - optimization is correct!")
    else:
        print("\n❌ Optimization failed - results differ!")
        return False
    
    # 计算性能提升
    speedup = original_time / optimized_time
    improvement = (1 - optimized_time / original_time) * 100
    
    print(f"\n=== Performance Results ===")
    print(f"Original time:   {original_time:.4f} seconds")
    print(f"Optimized time:  {optimized_time:.4f} seconds")
    print(f"Speedup:         {speedup:.2f}x")
    print(f"Improvement:     {improvement:.1f}%")
    
    return True


if __name__ == "__main__":
    # 运行性能对比测试
    performance_comparison_test() 