import logging
import os
import time
import sys
import numpy as np
from datetime import datetime
from typing import List, Dict

import pandas as pd

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ScalingAgent import ScalingAgent, convert_action_to_real_ES
from agent.agent_utils import FullStateDQN
from agent.components.es_registry import ServiceID, ServiceType, ESType
from iwai.fast_pymdp_agent import FastPymdpAgent
from iwai.proj_types import ESServiceAction
import utils

logger = logging.getLogger("multiscale")
logger.setLevel(logging.DEBUG)

ROOT = os.path.dirname(__file__)
PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))

class AIF_agent(ScalingAgent):
    def __init__(self, prom_server, services_monitored: list[ServiceID], evaluation_cycle,
                 slo_registry_path=ROOT + "/../config/slo_config.json",
                 es_registry_path=ROOT + "/../config/es_registry.json", 
                 log_experience=None,
                 policy_length=1,
                 learning_rate=1,
                 alpha=8,
                 action_selection="stochastic",
                 motivate_cores=True):
        
        super().__init__(prom_server, services_monitored, evaluation_cycle,
                         slo_registry_path, es_registry_path, log_experience)
        
        # PyMDP智能体参数
        self.policy_length = policy_length
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.action_selection = action_selection
        self.motivate_cores = motivate_cores
        
        # 初始化快速PyMDP智能体（静默模式）
        logger.info("Initializing AIF Agent with Fast PyMDP...")
        start_time = time.perf_counter()
        
        # 临时静默详细输出
        import logging
        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        
        self.fast_agent_creator = FastPymdpAgent()
        self.pymdp_agent = self.fast_agent_creator.generate_agent(
            policy_length=self.policy_length,
            learning_rate=self.learning_rate,
            alpha=self.alpha,
            action_selection=self.action_selection
        )
        
        # 恢复日志级别
        logging.getLogger().setLevel(old_level)
        
        init_time = time.perf_counter() - start_time
        logger.info(f"AIF Agent initialized in {init_time:.2f}s")
        
        # 经验记录
        self.logged_data = []
        self.step_count = 0

    def convert_service_state_to_pymdp(self, qr_state: Dict, cv_state: Dict) -> List[int]:
        """将服务状态转换为PyMDP格式的状态"""
        
        # QR服务状态映射 - 确保都转换为数值类型
        qr_data_quality = float(qr_state.get('data_quality', 700))
        qr_cores = int(qr_state.get('cores', 2))
        
        # CV服务状态映射 - 确保都转换为数值类型
        cv_data_quality = float(cv_state.get('data_quality', 256))
        cv_model_size = int(cv_state.get('model_size', 3))
        cv_cores = int(cv_state.get('cores', 2))
        
        # 获取自由核心数
        free_cores = int(self.get_free_cores())
        
        # 映射到PyMDP状态空间 (假设与训练时的状态空间一致)
        # 确保所有值都是整数
        pymdp_state_cv = [
            int(min(max((cv_data_quality - 200) / 50, 0), 4)),    # CV数据质量离散化
            int(min(max(cv_model_size - 1, 0), 2)),               # CV模型大小
            int(min(max(cv_cores - 1, 0), PHYSICAL_CORES - 2))    # CV核心数
        ]
        
        pymdp_state_qr = [
            int(min(max((qr_data_quality - 500) / 100, 0), 4)),   # QR数据质量离散化
            int(min(max(qr_cores - 1, 0), PHYSICAL_CORES - 2)),   # QR核心数
            int(min(max(free_cores, 0), PHYSICAL_CORES - 2)),     # 自由核心数
            0  # 额外状态维度
        ]
        
        return pymdp_state_cv + pymdp_state_qr

    def orchestrate_services_optimally(self, services_m: List[ServiceID]):
        """使用快速PyMDP智能体进行服务协调"""
        
        try:
            # 获取QR和CV服务状态  
            qr_service = next(s for s in services_m if s.service_type == ServiceType.QR)
            cv_service = next(s for s in services_m if s.service_type == ServiceType.CV)
            
            qr_clients = self.reddis_client.get_assignments_for_service(qr_service)
            cv_clients = self.reddis_client.get_assignments_for_service(cv_service)
            
            qr_state = self.resolve_service_state(qr_service, qr_clients)
            cv_state = self.resolve_service_state(cv_service, cv_clients)
            
            if not qr_state or not cv_state:
                logger.warning("Cannot resolve service states, skipping decision")
                return
                
            # 记录SLO满足情况
            if self.log_experience is not None:
                qr_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(ServiceType.QR, qr_clients)
                cv_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(ServiceType.CV, cv_clients)
                self.evaluate_slos_and_buffer(qr_service, qr_state, qr_slos)
                self.evaluate_slos_and_buffer(cv_service, cv_state, cv_slos)
            
        except Exception as e:
            logger.error(f"Error resolving service states: {e}")
            return
            
        # 转换为PyMDP状态
        pymdp_state = self.convert_service_state_to_pymdp(qr_state, cv_state)
        
        # PyMDP推理过程
        start_inference_time = time.perf_counter()
        
        try:
            # 状态推理
            a_s = self.pymdp_agent.infer_states(pymdp_state)
            
            # 参数学习 (从第二步开始)
            if self.step_count > 0:
                self.pymdp_agent.update_B(a_s)
            
            # 策略推理（静默模式）
            import sys
            import io
            
            # 临时重定向print输出来静默FastPyMDP的详细输出
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                result = self.pymdp_agent.infer_policies()
            finally:
                sys.stdout = old_stdout  # 恢复stdout
            
            if len(result) == 3:  # 兼容老版本
                q_pi, G, G_sub = result
            else:  # 新版本
                q_pi, G = result
                G_sub = {"ig_s": np.zeros_like(G), "r": np.zeros_like(G)}
            
            # 动作采样
            chosen_action_id = self.pymdp_agent.sample_action()
            
            # 提取动作 - 确保转换为整数
            action_cv = ESServiceAction(int(chosen_action_id[0]))
            action_qr = ESServiceAction(int(chosen_action_id[1]))
            
            inference_time = time.perf_counter() - start_inference_time
            
            # 核心激励机制
            if self.motivate_cores:
                free_cores = int(self.get_free_cores())
                cv_cores = int(cv_state.get('cores', 2))
                qr_cores = int(qr_state.get('cores', 2))
                
                # 确保索引在有效范围内
                cv_cores = max(0, min(cv_cores, PHYSICAL_CORES - 1))
                qr_cores = max(0, min(qr_cores, PHYSICAL_CORES - 1))
                
                if free_cores > 1:
                    self.pymdp_agent.C[3][cv_cores:] = 3
                    self.pymdp_agent.C[6][qr_cores:] = 1
                elif free_cores == 1:
                    if cv_cores > qr_cores:
                        self.pymdp_agent.C[6][qr_cores:] = 3
                        self.pymdp_agent.C[3] = np.zeros(PHYSICAL_CORES - 1)
                        if cv_cores - 1 >= 0:
                            self.pymdp_agent.C[3][cv_cores - 1] = 1
                    else:
                        self.pymdp_agent.C[3][cv_cores:] = 3
                        self.pymdp_agent.C[6] = np.zeros(PHYSICAL_CORES - 1)
                        if qr_cores - 1 >= 0:
                            self.pymdp_agent.C[6][qr_cores - 1] = 1
                else:
                    self.pymdp_agent.C[3] = np.zeros(PHYSICAL_CORES - 1)
                    self.pymdp_agent.C[6] = np.zeros(PHYSICAL_CORES - 1)
            
            # 计算期望自由能等指标
            policy_list = self.pymdp_agent.policies
            policy_array = np.array(policy_list)
            flattened_policies = policy_array[:, 0, :]
            
            policy_index = next(
                i for i, policy in enumerate(flattened_policies)
                if np.array_equal(policy, chosen_action_id)
            )
            
            efe = G[policy_index]
            info_gain = G_sub["ig_s"][policy_index]
            pragmatic_value = G_sub["r"][policy_index]
            
            # 计算reward（基于SLO满足度）
            try:
                qr_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(ServiceType.QR, qr_clients)
                cv_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(ServiceType.CV, cv_clients)
                
                from agent.components.SLORegistry import calculate_SLO_F_clients
                qr_reward = calculate_SLO_F_clients(qr_state, qr_slos)
                cv_reward = calculate_SLO_F_clients(cv_state, cv_slos)
                total_reward = (qr_reward + cv_reward) / 2.0  # 平均reward
            except Exception as e:
                logger.warning(f"Failed to calculate reward: {e}")
                total_reward = 0.0
            
            # 执行动作
            self._execute_pymdp_actions(services_m, action_qr, action_cv, qr_state, cv_state)
            
            # 简化日志输出，类似其他agent
            logger.info(f"AIF Agent step {self.step_count}: Reward: {total_reward:.3f} | "
                       f"CV: {action_cv.name}, QR: {action_qr.name} | EFE: {efe:.3f}")
            
            # 记录详细数据（用于分析）
            timestamp = datetime.now().isoformat()
            self.logged_data.append({
                "timestamp": timestamp,
                "step": self.step_count,
                "reward": total_reward,
                "qr_reward": qr_reward if 'qr_reward' in locals() else 0.0,
                "cv_reward": cv_reward if 'cv_reward' in locals() else 0.0,
                "qr_state": str(qr_state),
                "cv_state": str(cv_state),
                "pymdp_state": str(pymdp_state),
                "action_qr": action_qr.name if hasattr(action_qr, 'name') else str(action_qr),
                "action_cv": action_cv.name if hasattr(action_cv, 'name') else str(action_cv),
                "efe": float(efe),
                "info_gain": float(info_gain),
                "pragmatic_value": float(pragmatic_value),
                "inference_time": inference_time,
            })
            
            self.step_count += 1
            
        except Exception as e:
            logger.error(f"PyMDP inference failed: {e}")
            # 回退到随机动作
            self._execute_random_actions(services_m)

    def _execute_pymdp_actions(self, services_m: List[ServiceID], action_qr: ESServiceAction, 
                              action_cv: ESServiceAction, qr_state: Dict, cv_state: Dict):
        """执行PyMDP智能体决策的动作"""
        
        qr_service = next(s for s in services_m if s.service_type == ServiceType.QR)
        cv_service = next(s for s in services_m if s.service_type == ServiceType.CV)
        
        # 转换并执行QR动作
        if action_qr != ESServiceAction.DILLY_DALLY:
            qr_full_state = self._create_full_state_dqn(qr_service, qr_state)
            qr_es, qr_params = convert_action_to_real_ES(
                qr_service, qr_full_state, action_qr.value, self.get_free_cores()
            )
            
            if qr_es != ESType.IDLE:
                # 添加安全检查
                if not self._is_action_safe(qr_service, qr_full_state, action_qr.value):
                    logger.info(f"Preventing unsafe QR action {action_qr} for service {qr_service}")
                else:
                    self.execute_ES(qr_service.host, qr_service, qr_es, qr_params, respect_cooldown=False)
        
        time.sleep(0.01)  # 小延迟避免冲突
        
        # 转换并执行CV动作
        if action_cv != ESServiceAction.DILLY_DALLY:
            cv_full_state = self._create_full_state_dqn(cv_service, cv_state)
            cv_es, cv_params = convert_action_to_real_ES(
                cv_service, cv_full_state, action_cv.value, self.get_free_cores()
            )
            
            if cv_es != ESType.IDLE:
                # 添加安全检查
                if not self._is_action_safe(cv_service, cv_full_state, action_cv.value):
                    logger.info(f"Preventing unsafe CV action {action_cv} for service {cv_service}")
                else:
                    self.execute_ES(cv_service.host, cv_service, cv_es, cv_params, respect_cooldown=False)

    def _create_full_state_dqn(self, service: ServiceID, service_state: Dict) -> FullStateDQN:
        """创建FullStateDQN对象用于动作转换"""
        
        assigned_clients = self.reddis_client.get_assignments_for_service(service)
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(
            service.service_type, assigned_clients
        )
        
        model_size, model_size_t = 1, 1
        if "model_size" in service_state or "model_size" in all_client_slos[0]:
            model_size, model_size_t = (
                service_state["model_size"],
                all_client_slos[0]["model_size"].target,
            )
        
        data_quality_t, throughput_t = (
            all_client_slos[0]["data_quality"].target,
            all_client_slos[0]["throughput"].target,
        )
        
        free_cores = self.get_free_cores()
        boundaries = self.es_registry.get_boundaries_minimalistic(
            service.service_type, PHYSICAL_CORES
        )
        
        return FullStateDQN(
            service_state["data_quality"],
            data_quality_t,
            service_state["throughput"],
            throughput_t,
            model_size,
            model_size_t,
            service_state["cores"],
            free_cores,
            boundaries,
        )

    def _is_action_safe(self, service: ServiceID, state: FullStateDQN, action: int) -> bool:
        """检查动作是否安全，避免危险操作"""
        
        if service.service_type == ServiceType.QR:
            return not (
                (state.cores <= 4 and action == 3) or
                (state.data_quality >= 900 and action == 2) or
                (state.data_quality <= 500 and action == 1)
            )
        elif service.service_type == ServiceType.CV:
            return not (
                (state.cores <= 4 and action == 3) or
                (state.model_size >= 2 and action == 6) or
                (state.data_quality >= 288 and action == 2)
            )
        
        return True

    def _execute_random_actions(self, services_m: List[ServiceID]):
        """执行随机动作作为回退策略"""
        
        import random
        
        for service in services_m:
            if random.random() < 0.3:  # 30%概率执行动作
                all_es = self.es_registry.get_active_ES_for_service(service.service_type)
                if all_es:
                    random_es = random.choice(all_es)
                    max_cores = self.get_max_available_cores(service)
                    param_bounds = self.es_registry.get_parameter_bounds_for_active_ES(
                        service.service_type, max_cores
                    ).get(random_es, {})
                    
                    from agent.agent_utils import get_random_parameter_assignments
                    random_params = get_random_parameter_assignments(param_bounds)
                    self.execute_ES(service.host, service, random_es, random_params, respect_cooldown=False)

    def save_experience_log(self, filename_suffix=""):
        """保存经验日志"""
        
        if not self.logged_data:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = ROOT + f"/../experiments/iwai/{timestamp}_aif_service_log{filename_suffix}.csv"
        
        df = pd.DataFrame(self.logged_data)
        df.to_csv(log_path, index=False)
        
        logger.info(f"AIF Agent experience log saved to: {log_path}")
        return log_path

    def terminate_gracefully(self):
        """优雅终止并保存日志"""
        
        super().terminate_gracefully()
        
        if self.logged_data:
            log_path = self.save_experience_log("_final")
            logger.info(f"AIF Agent terminated after {self.step_count} steps")


if __name__ == "__main__":
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")
    cv_local = ServiceID("172.20.0.10", ServiceType.CV, "elastic-workbench-cv-analyzer-1")

    aif_agent = AIF_agent(
        prom_server=ps,
        services_monitored=[qr_local, cv_local],
        evaluation_cycle=5,
        alpha=8,
        motivate_cores=True
    )
    
    aif_agent.reset_services_states()
    time.sleep(3)
    aif_agent.start()
        
        
        