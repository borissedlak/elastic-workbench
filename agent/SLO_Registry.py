import json
import os
from typing import Dict, Any

import numpy as np

from agent.ES_Registry import ServiceType


class SLO_Registry:
    def __init__(self):
        ROOT = os.path.dirname(__file__)
        with open(ROOT + '/conf/slo_config.json', 'r') as f:
            self.slo_lib = json.load(f)

    def get_SLOs_for_client(self, client_id, service_type: ServiceType):
        result = {}
        for entry in self.slo_lib["clientSLOs"]:
            if entry["client_id"] == client_id and entry["service_type"] == service_type.value:
                for slo in entry["SLOs"]:
                    result = result | {slo["var"]: slo}
        return result

    def calculate_slo_reward(self, state: Dict[str, Any], SLOs):
        fuzzy_slof = []

        for state_var, value in state.items():
            if state_var in SLOs:
                slo = SLOs[state_var]

                if slo["larger"] == "True":
                    slo_f = (value / float(slo["thresh"]))
                else:
                    slo_f = 1 - ((value - float(slo["thresh"])) / float(slo["thresh"]))  # SLO-F is 0 after 2 * t

                slo_f = float(np.clip(slo_f, 0.0, 1.0)) * float(slo["weight"])  # Could allow cap=1.1
                fuzzy_slof.append((state_var, slo_f))

        return fuzzy_slof


if __name__ == '__main__':
    slo_registry = SLO_Registry()
    slos = slo_registry.get_SLOs_for_client("C_1", ServiceType.QR)

    print(slo_registry.calculate_slo_reward({"avg_p_latency": 10}, slos))
