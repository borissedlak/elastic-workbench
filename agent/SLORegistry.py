import json
import logging
import os
from typing import Dict, Any, List, Tuple, NamedTuple

import numpy as np

from agent.agent_utils import FullStateDQN
from agent.es_registry import ServiceType, ESRegistry

logger = logging.getLogger("multiscale")

ROOT = os.path.dirname(__file__)
es_registry = ESRegistry(ROOT + "/../config/es_registry.json")


class SLO(NamedTuple):
    var: str
    larger: bool
    target: float
    weight: float


def smoothstep(x, x0=0.0, x1=1.0) -> float:
    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return float(t * t * (3 - 2 * t))


def calculate_SLO_F_clients(service_type:ServiceType, full_state, slos_all_clients):
    slo_f_all_clients = 0.0

    for slos_single_client in slos_all_clients:
        # boundaries = es_registry.get_boundaries_minimalistic(service_type, 8)
        # full_state_tensor = FullStateDQN(
        #     full_state["data_quality"],
        #     slos_single_client['data_quality'].target,
        #     full_state["throughput"],
        #     slos_single_client['throughput'].target,
        #     full_state['model_size'] if 'model_size' in slos_single_client else 0,
        #     slos_single_client['model_size'].target if 'model_size' in slos_single_client else 0,
        #     0,  # cores irrelevant for SLO-F
        #     0,  # cores irrelevant for SLO-F
        #     boundaries,
        # )

        slo_f_list = calculate_slo_fulfillment(full_state, slos_single_client)
        normalized_reward = to_normalized_slo_f(slo_f_list, slos_single_client)
        slo_f_all_clients += normalized_reward

    return slo_f_all_clients / len(slos_all_clients)


def to_normalized_slo_f(slof: List[Tuple[str, float]], slos: Dict[str, SLO]) -> float:
    slo_f_single_client = sum(value for _, value in slof)

    max_slo_f_single_client = sum([s.weight for s in slos.values()])
    scaled_reward = slo_f_single_client / max_slo_f_single_client

    return scaled_reward

# TODO: Should again use normalized values
def calculate_slo_fulfillment(
        full_state: Dict[str, Any], slos: Dict[str, SLO]
) -> List[Tuple[str, float]]:
    # if 'model_size' in slos:  # === service_type.CV
    #     quality = full_state["data_quality"] * 0.25 + full_state["model_size"] * 0.75
    #     quality_target = (
    #             full_state["data_quality_target"] * 0.25
    #             + full_state["model_size_target"] * 0.75
    #     )
    # else:  # === service_type.QR
    #     quality = full_state["data_quality"]
    #     quality_target = full_state["data_quality_target"]
    #
    # throughput = full_state["throughput"]
    # throughput_target = full_state["throughput_target"]
    #
    # slo_trace = [
    #     ("quality", smoothstep((quality / quality_target)) if throughput > 0.0 else 0.0),
    #     ("throughput", smoothstep((throughput / throughput_target))),
    # ]
    slo_trace = []
    for slo in slos.values():
        var, larger, target, weight = slo
        value = full_state[var]
        if larger:
            slo_f_single_slo = value / target
        else:
            slo_f_single_slo = 1 - (
                (value - target) / target
            )  # SLO-F is 0 after 2 * t

        slo_f_single_slo = float(smoothstep(slo_f_single_slo) * weight)
        if "throughput" in full_state and full_state["throughput"] <= 0.0:
            slo_f_single_slo = 0  # Heavily penalize if no output

        slo_trace.append((var, slo_f_single_slo))

    return slo_trace


class SLO_Registry:
    def __init__(self, slo_config_path):

        with open(slo_config_path, "r") as f:
            self.slo_lib = json.load(f)

    def get_all_SLOs_for_assigned_clients(
            self, service_type: ServiceType, assigned_clients: Dict[str, int]
    ):
        all_client_slos = []

        for client_id, client_rps in assigned_clients.items():
            client_slos = self.get_SLOs_for_client(client_id, service_type)
            all_client_slos.append(client_slos)

        return all_client_slos

    def get_SLOs_for_client(
            self, client_id, service_type: ServiceType
    ) -> Dict[str, SLO]:
        result = {}
        for entry in self.slo_lib["clientSLOs"]:
            if (
                    entry["client_id"] == client_id
                    and entry["service_type"] == service_type.value
            ):
                for slo in entry["SLOs"]:
                    result = result | {slo["var"]: SLO(**slo)}
        return result
