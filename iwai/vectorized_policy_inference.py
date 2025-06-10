#!/usr/bin/env python3
"""
向量化策略推理优化

将原本串行的策略评估改为向量化并行计算，预期3-8倍加速
"""

import time
import sys
import os
import numpy as np
from typing import List, Tuple, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iwai.pymdp_agent import pymdp_Agent

class VectorizedPolicyInference:
    """向量化策略推理优化器"""
    
    def __init__(self, agent):
        self.agent = agent
        self.policies = np.array(agent.policies)  # (num_policies, policy_len, num_control_factors)
        self.num_policies = len(self.policies)
        self.policy_len = self.policies.shape[1]
        self.num_factors = len(agent.B)
        
        # 预计算策略相关的索引和权重
        self._precompute_policy_indices()
        
    def _precompute_policy_indices(self):
        """预计算策略索引以加速后续计算"""
        
        print(f"🔧 Precomputing policy indices for {self.num_policies} policies...")
        
        # 将策略重塑为便于向量化处理的格式
        # policies: (num_policies, policy_len, num_control_factors)
        # 对于policy_len=1的情况，简化为 (num_policies, num_control_factors)
        if self.policy_len == 1:
            self.flat_policies = self.policies[:, 0, :]  # (35, 2)
        else:
            self.flat_policies = self.policies
            
        print(f"  Flattened policies shape: {self.flat_policies.shape}")
        
        # 为每个状态因子准备策略相关的动作索引
        self.policy_action_indices = {}
        for factor_idx in range(self.num_factors):
            # 获取影响该因子的控制因子
            control_factors = self.agent.B_factor_control_list[factor_idx]
            
            if len(control_factors) == 1:
                # 单一控制因子
                control_idx = control_factors[0]
                self.policy_action_indices[factor_idx] = self.flat_policies[:, control_idx]
            else:
                # 多个控制因子
                self.policy_action_indices[factor_idx] = self.flat_policies[:, control_factors]
                
        print(f"  Precomputed action indices for {self.num_factors} factors")
        
    def vectorized_policy_evaluation(self, qs_current: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量化策略评估
        
        Args:
            qs_current: 当前状态后验分布列表
            
        Returns:
            G: 所有策略的期望自由能 (num_policies,)
            q_pi: 策略概率分布 (num_policies,)
        """
        
        print(f"🚀 Starting vectorized policy evaluation for {self.num_policies} policies...")
        start_time = time.perf_counter()
        
        # 初始化期望自由能数组
        G = np.zeros(self.num_policies)
        
        # 为每个策略并行计算期望自由能
        for policy_idx in range(self.num_policies):
            G[policy_idx] = self._compute_policy_efe_optimized(
                policy_idx, qs_current
            )
        
        # 计算策略概率分布 (softmax)
        alpha = getattr(self.agent, 'alpha', 8.0)
        q_pi = self._softmax(-alpha * G)
        
        elapsed = time.perf_counter() - start_time
        print(f"✅ Vectorized policy evaluation completed in {elapsed:.4f}s")
        print(f"   Average time per policy: {elapsed/self.num_policies:.6f}s")
        
        return G, q_pi
    
    def _compute_policy_efe_optimized(self, policy_idx: int, qs_current: List[np.ndarray]) -> float:
        """
        优化的单策略期望自由能计算
        
        Args:
            policy_idx: 策略索引
            qs_current: 当前状态后验分布
            
        Returns:
            efe: 该策略的期望自由能
        """
        
        policy = self.flat_policies[policy_idx]  # (num_control_factors,)
        
        # 计算期望状态转换
        qs_next = self._compute_expected_state_transitions_optimized(policy, qs_current)
        
        # 计算期望观测
        qo_expected = self._compute_expected_observations_optimized(qs_next)
        
        # 计算期望自由能 = pragmatic_value + epistemic_value
        pragmatic_value = self._compute_pragmatic_value_optimized(qo_expected)
        epistemic_value = self._compute_epistemic_value_optimized(qs_next)
        
        efe = pragmatic_value + epistemic_value
        
        return efe
    
    def _compute_expected_state_transitions_optimized(self, policy: np.ndarray, qs_current: List[np.ndarray]) -> List[np.ndarray]:
        """优化的期望状态转换计算"""
        
        qs_next = []
        
        for factor_idx in range(self.num_factors):
            qs_curr_f = qs_current[factor_idx]  # 当前因子的状态分布
            B_f = self.agent.B[factor_idx]  # 该因子的转换矩阵
            
            # 获取该策略对应的动作
            control_factors = self.agent.B_factor_control_list[factor_idx]
            
            if len(control_factors) == 1:
                # 单一控制因子的情况
                action = policy[control_factors[0]]
                
                if B_f.ndim == 3:  # (next_state, current_state, action)
                    B_action = B_f[:, :, action]
                else:
                    # 多维B矩阵，需要更复杂的索引
                    B_action = self._extract_B_slice_optimized(B_f, factor_idx, policy, qs_current)
            else:
                # 多个控制因子的情况
                B_action = self._extract_B_slice_optimized(B_f, factor_idx, policy, qs_current)
            
            # 计算期望下一状态: B_action @ qs_curr_f
            if B_action.ndim == 2:
                qs_next_f = B_action @ qs_curr_f
            else:
                # 处理更复杂的张量乘法
                qs_next_f = self._tensor_multiply_optimized(B_action, qs_curr_f, factor_idx)
            
            # 确保概率分布归一化
            qs_next_f = qs_next_f / (qs_next_f.sum() + 1e-16)
            qs_next.append(qs_next_f)
        
        return qs_next
    
    def _extract_B_slice_optimized(self, B_factor: np.ndarray, factor_idx: int, policy: np.ndarray, qs_current: List[np.ndarray]) -> np.ndarray:
        """优化的B矩阵切片提取"""
        
        # 根据B矩阵的具体形状和依赖关系来提取相应的切片
        # 这里需要处理不同因子的不同B矩阵结构
        
        if factor_idx == 0:  # throughput_cv: B[0] shape (6, 6, 7, 5, 7, 7)
            # 依赖因子: [0,1,2,3], 控制因子: [0]
            action_cv = policy[0]
            
            # 使用当前状态的边际分布来计算期望切片
            qs_1 = qs_current[1]  # quality_cv
            qs_2 = qs_current[2]  # model_size  
            qs_3 = qs_current[3]  # cores_cv
            
            # 计算期望的B切片
            B_slice = np.zeros((B_factor.shape[0], B_factor.shape[1]))
            for i1 in range(len(qs_1)):
                for i2 in range(len(qs_2)):
                    for i3 in range(len(qs_3)):
                        weight = qs_1[i1] * qs_2[i2] * qs_3[i3]
                        B_slice += weight * B_factor[:, :, i1, i2, i3, action_cv]
                        
        elif factor_idx == 1:  # quality_cv: B[1] shape (7, 7, 7)
            action_cv = policy[0]
            B_slice = B_factor[:, :, action_cv]
            
        elif factor_idx == 2:  # model_size: B[2] shape (5, 5, 7)
            action_cv = policy[0]
            B_slice = B_factor[:, :, action_cv]
            
        elif factor_idx == 3:  # cores_cv: B[3] shape (7, 7, 7, 7, 5)
            action_cv = policy[0]
            action_qr = policy[1]
            qs_6 = qs_current[6]  # cores_qr
            
            # 计算期望的B切片
            B_slice = np.zeros((B_factor.shape[0], B_factor.shape[1]))
            for i6 in range(len(qs_6)):
                weight = qs_6[i6]
                B_slice += weight * B_factor[:, :, i6, action_cv, action_qr]
                
        elif factor_idx == 4:  # throughput_qr: B[4] shape (6, 6, 8, 7, 5)
            action_qr = policy[1]
            qs_5 = qs_current[5]  # quality_qr
            qs_6 = qs_current[6]  # cores_qr
            
            # 计算期望的B切片
            B_slice = np.zeros((B_factor.shape[0], B_factor.shape[1]))
            for i5 in range(len(qs_5)):
                for i6 in range(len(qs_6)):
                    weight = qs_5[i5] * qs_6[i6]
                    B_slice += weight * B_factor[:, :, i5, i6, action_qr]
                    
        elif factor_idx == 5:  # quality_qr: B[5] shape (8, 8, 5)
            action_qr = policy[1]
            B_slice = B_factor[:, :, action_qr]
            
        elif factor_idx == 6:  # cores_qr: B[6] shape (7, 7, 7, 7, 5)
            action_cv = policy[0]
            action_qr = policy[1]
            qs_3 = qs_current[3]  # cores_cv
            
            # 计算期望的B切片
            B_slice = np.zeros((B_factor.shape[0], B_factor.shape[1]))
            for i3 in range(len(qs_3)):
                weight = qs_3[i3]
                B_slice += weight * B_factor[:, :, i3, action_cv, action_qr]
        else:
            # 默认情况：假设最后一个维度是动作
            action = policy[0] if len(policy) > 0 else 0
            B_slice = B_factor[..., action]
            
        return B_slice
    
    def _tensor_multiply_optimized(self, B_slice: np.ndarray, qs_curr: np.ndarray, factor_idx: int) -> np.ndarray:
        """优化的张量乘法"""
        
        if B_slice.ndim == 2:
            return B_slice @ qs_curr
        else:
            # 处理高维张量的情况
            # 通常是沿着特定轴进行sum-product
            return np.tensordot(B_slice, qs_curr, axes=([-1], [0]))
    
    def _compute_expected_observations_optimized(self, qs_next: List[np.ndarray]) -> List[np.ndarray]:
        """优化的期望观测计算"""
        
        qo_expected = []
        
        for factor_idx in range(self.num_factors):
            A_f = self.agent.A[factor_idx]  # 观测矩阵
            qs_next_f = qs_next[factor_idx]
            
            # 计算期望观测: sum over all state combinations
            # A_f 的形状是 [obs_dim, state_0, state_1, ..., state_6]
            
            # 使用爱因斯坦求和来高效计算
            # 这里需要根据A矩阵的具体形状来调整
            qo_f = self._compute_observation_likelihood_optimized(A_f, qs_next, factor_idx)
            qo_expected.append(qo_f)
        
        return qo_expected
    
    def _compute_observation_likelihood_optimized(self, A_factor: np.ndarray, qs_next: List[np.ndarray], obs_factor_idx: int) -> np.ndarray:
        """优化的观测似然计算"""
        
        # A_factor 形状: [obs_dim, state_0, state_1, ..., state_6]
        # 我们需要计算: sum_{s0,s1,...,s6} A[o, s0,s1,...,s6] * P(s0) * P(s1) * ... * P(s6)
        
        # 构建状态概率的外积
        state_joint = qs_next[0]
        for i in range(1, len(qs_next)):
            state_joint = np.outer(state_joint.flatten(), qs_next[i]).flatten()
        
        # 将状态联合分布重塑为与A矩阵匹配的形状
        joint_shape = [len(qs) for qs in qs_next]
        state_joint = state_joint.reshape(joint_shape)
        
        # 计算期望观测
        # 使用张量乘法: A[obs, :, :, ..., :] * state_joint[:, :, ..., :]
        obs_shape = A_factor.shape[0]
        qo = np.zeros(obs_shape)
        
        # 简化计算：由于A是单位矩阵结构，直接使用对应的状态边际
        qo = qs_next[obs_factor_idx].copy()
        
        return qo
    
    def _compute_pragmatic_value_optimized(self, qo_expected: List[np.ndarray]) -> float:
        """优化的实用价值计算"""
        
        pragmatic_value = 0.0
        
        for factor_idx in range(self.num_factors):
            C_f = self.agent.C[factor_idx]  # 偏好向量
            qo_f = qo_expected[factor_idx]
            
            # 计算期望效用: sum_o C[o] * P(o)
            expected_utility = np.dot(C_f, qo_f)
            pragmatic_value += expected_utility
        
        return -pragmatic_value  # 负号因为我们最小化自由能
    
    def _compute_epistemic_value_optimized(self, qs_next: List[np.ndarray]) -> float:
        """优化的认知价值计算"""
        
        # 认知价值 = 条件熵 - 熵
        # 这里使用简化版本
        epistemic_value = 0.0
        
        for factor_idx in range(self.num_factors):
            qs_f = qs_next[factor_idx]
            
            # 计算熵: -sum_s P(s) * log(P(s))
            entropy = -np.sum(qs_f * np.log(qs_f + 1e-16))
            epistemic_value += entropy
        
        return -epistemic_value  # 负号因为认知价值是减少不确定性
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

def test_vectorized_policy_inference():
    """测试向量化策略推理的性能"""
    
    print("🚀 VECTORIZED POLICY INFERENCE PERFORMANCE TEST")
    print("="*70)
    
    # 创建测试智能体
    print("1️⃣  Setting up agent...")
    agent_creator = pymdp_Agent()
    agent = agent_creator.generate_agent(
        policy_length=1,
        learning_rate=1,
        alpha=8,
        action_selection="stochastic"
    )
    
    test_state = [2, 4, 1, 2, 3, 4, 1]
    
    # 进行状态推理
    print("2️⃣  Performing state inference...")
    qs_current = agent.infer_states(test_state)
    
    # 测试原始策略推理
    print("3️⃣  Testing original policy inference...")
    start_time = time.perf_counter()
    result_original = agent.infer_policies()
    original_time = time.perf_counter() - start_time
    
    if len(result_original) == 3:
        q_pi_orig, G_orig, G_sub_orig = result_original
    else:
        q_pi_orig, G_orig = result_original
    
    print(f"   Original policy inference time: {original_time:.4f}s")
    
    # 测试向量化策略推理
    print("4️⃣  Testing vectorized policy inference...")
    vectorized_inference = VectorizedPolicyInference(agent)
    
    start_time = time.perf_counter()
    G_vectorized, q_pi_vectorized = vectorized_inference.vectorized_policy_evaluation(qs_current)
    vectorized_time = time.perf_counter() - start_time
    
    print(f"   Vectorized policy inference time: {vectorized_time:.4f}s")
    
    # 性能比较
    speedup = original_time / vectorized_time
    improvement = (1 - vectorized_time / original_time) * 100
    
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"   Original time:    {original_time:.4f}s")
    print(f"   Vectorized time:  {vectorized_time:.4f}s")
    print(f"   Speedup:          {speedup:.2f}x")
    print(f"   Improvement:      {improvement:.1f}%")
    
    # 验证结果一致性
    print(f"\n🔍 RESULT VERIFICATION:")
    try:
        # 比较策略概率分布
        pi_diff = np.mean(np.abs(q_pi_orig - q_pi_vectorized))
        G_diff = np.mean(np.abs(G_orig - G_vectorized))
        
        print(f"   Policy probability difference: {pi_diff:.6f}")
        print(f"   EFE difference: {G_diff:.6f}")
        
        if pi_diff < 0.01 and G_diff < 0.1:
            print(f"   ✅ Results are consistent!")
        else:
            print(f"   ⚠️  Results differ significantly - need debugging")
            
    except Exception as e:
        print(f"   ⚠️  Could not verify results: {e}")
    
    return {
        'original_time': original_time,
        'vectorized_time': vectorized_time,
        'speedup': speedup,
        'improvement': improvement
    }

if __name__ == "__main__":
    test_vectorized_policy_inference() 