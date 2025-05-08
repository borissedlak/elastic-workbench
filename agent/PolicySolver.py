import random
from typing import Dict

import numpy as np
from pgmpy.factors.continuous import LinearGaussianCPD
from scipy.optimize import minimize

from agent.ES_Registry import EsType


# Exponential decay from linear start
def soft_clip(x, max):
    return x * np.exp(-1.5 * x) + max * (1 - np.exp(-1.5 * x))


def composite_obj(x, parameter_bounds, linear_relations: Dict[str, LinearGaussianCPD], all_SLOs, total_rps):
    variables = {param['name']: val for param, val in zip(parameter_bounds, x)}

    # ---------- Part 1: LGBN Relations ----------

    arguments = {}
    for key, item in linear_relations.items():
        arguments[key] = item.beta[0]
        for i in range(1, len(item.beta)):
            arguments[key] = arguments[key] + (variables[item.evidence[i - 1]] * item.beta[i])

    arguments["throughput"] = (1000 / arguments["avg_p_latency"])
    if linear_relations['avg_p_latency'] is not None and 'cores' not in linear_relations['avg_p_latency'].evidence:
        arguments["throughput"] = arguments["throughput"] * variables["cores"]
    arguments["completion_rate"] = arguments["throughput"] / total_rps

    # ---------- Part 2: Client SLOs ----------

    SLO_F_all_clients = 0
    for client_SLOs in all_SLOs:
        SLO_F_single_client = 0

        for slo in client_SLOs.values():
            var, larger, thresh, weight = slo
            value = (variables | arguments)[var]
            if larger:
                single_slo = (value / float(thresh))
            else:
                single_slo = 1 - ((value - float(thresh)) / float(thresh))

            SLO_F_single_client += soft_clip(single_slo, 1.0) * weight
        client_x_max_slof = sum([s.weight for s in client_SLOs.values()])
        SLO_F_all_clients += (SLO_F_single_client / client_x_max_slof)

    slo_f = SLO_F_all_clients / len(all_SLOs)
    return -slo_f  # because we want to maximize


def solve(parameter_bounds, linear_relations, clients_SLOs, total_rps):
    bounds = [(param["min"], param["max"]) for param in parameter_bounds]  # Shape [(360, 1080), (1, 8)]
    x0 = [random.randint(mini, maxi) for mini, maxi in bounds]  # Initial guess; Shape [520, 4]

    result = minimize(composite_obj, x0, method='L-BFGS-B', bounds=bounds,
                      args=(parameter_bounds, linear_relations, clients_SLOs, total_rps))

    if not result.success:
        raise RuntimeWarning("Policy solver encountered an error: " + result.message)

    es_param_ass = {}
    for index, param in enumerate(parameter_bounds):
        es_param_ass[param["name"]] = int(result.x[index])

    return es_param_ass
