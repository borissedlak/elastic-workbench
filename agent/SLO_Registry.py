import json
import logging
from typing import Dict, Any, List, Tuple, NamedTuple

import numpy as np

from agent.ES_Registry import ServiceType

logger = logging.getLogger("multiscale")


class SLO(NamedTuple):
    var: str
    larger: bool
    thresh: float
    weight: float


def soft_clip(x, x0=0.0, x1=1.0) -> float:
    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return float(t ** 3 * (t * (6 * t - 15) + 10))


def to_normalized_SLO_F(slof: List[Tuple[str, float]], slos: Dict[str, SLO]) -> float:
    # return sum(value for _, value in slof) / float(len(slof))
    slo_f_single_client = sum(value for _, value in slof)

    max_slo_f_single_client = sum([s.weight for s in slos.values()])
    scaled_reward = slo_f_single_client / max_slo_f_single_client

    return scaled_reward


# TODO: Calculate overall streaming latency and place into state
#  I might also add a flag to use either the soft clip or the hard np.clip
def calculate_slo_fulfillment(state: Dict[str, Any], slos: Dict[str, SLO]) -> List[Tuple[str, float]]:
    slo_trace = []
    # slo_f_single_client = 0.0

    for slo in slos.values():
        var, larger, thresh, weight = slo
        value = state[var]
        if larger:
            slo_f_single_slo = (value / float(thresh))
        else:
            slo_f_single_slo = 1 - ((value - float(thresh)) / float(thresh))  # SLO-F is 0 after 2 * t

        slo_f_single_slo = float(soft_clip(slo_f_single_slo) * weight)
        if 'throughput' in state and state['throughput'] < 1.0:
            slo_f_single_slo *= 0.1  # Heavily penalize if no output

        slo_trace.append((var, slo_f_single_slo))

    return slo_trace


class SLO_Registry:
    def __init__(self, slo_config_path):

        with open(slo_config_path, 'r') as f:
            self.slo_lib = json.load(f)

    def get_all_SLOs_for_assigned_clients(self, service_type: ServiceType, assigned_clients: Dict[str, int]):
        all_client_slos = []

        for client_id, client_rps in assigned_clients.items():
            client_slos = self.get_SLOs_for_client(client_id, service_type)
            all_client_slos.append(client_slos)

        return all_client_slos

    def get_SLOs_for_client(self, client_id, service_type: ServiceType) -> Dict[str, SLO]:
        result = {}
        for entry in self.slo_lib["clientSLOs"]:
            if entry["client_id"] == client_id and entry["service_type"] == service_type.value:
                for slo in entry["SLOs"]:
                    result = result | {slo["var"]: SLO(**slo)}
        return result
