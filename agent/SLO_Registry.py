import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np

from agent.ES_Registry import ServiceType


class SLO_Registry:
    def __init__(self):
        ROOT = os.path.dirname(__file__)
        with open(ROOT + '/conf/slo_config.json', 'r') as f:
            self.slo_lib = json.load(f)

    def get_all_SLOs_for_assigned_clients(self, service_type: ServiceType, assigned_clients: Dict[str, int]):
        all_client_slos = []

        for client_id, client_rps in assigned_clients.items():
            client_slos = self.get_SLOs_for_client(client_id, service_type)
            all_client_slos.append(client_slos)

        return all_client_slos


    def get_SLOs_for_client(self, client_id, service_type: ServiceType):
        result = {}
        for entry in self.slo_lib["clientSLOs"]:
            if entry["client_id"] == client_id and entry["service_type"] == service_type.value:
                for slo in entry["SLOs"]:
                    result = result | {slo["var"]: slo}
        return result

    # TODO: Calculate overall streaming latency and place into state
    #  Ideally I do this in a function that can also be reused for the expected SLO_F
    def calculate_slo_fulfillment(self, state: Dict[str, Any], SLOs):
        fuzzy_slof = []

        for state_var, value in state.items():
            if state_var in SLOs:
                var, larger, thresh, weight = tuple(SLOs[state_var].values())

                if larger:
                    slo_f = (value / float(thresh))
                else:
                    slo_f = 1 - ((value - float(thresh)) / float(thresh))  # SLO-F is 0 after 2 * t

                slo_f = float(np.clip(slo_f, 0.0, 1.0)) * float(weight)
                fuzzy_slof.append((state_var, slo_f))

        return fuzzy_slof


def to_avg_SLO_F(slof: List[Tuple[str, float]]) -> float:
    return sum(value for _, value in slof) / float(len(slof))


if __name__ == '__main__':
    slo_registry = SLO_Registry()
    slos = slo_registry.get_SLOs_for_client("C_1", ServiceType.QR)

    print(slo_registry.calculate_slo_fulfillment({"quality": 10}, slos))
