import random
from typing import Dict

import numpy as np
from pgmpy.factors.continuous import LinearGaussianCPD
from scipy.optimize import minimize

from agent.LGBN import calculate_missing_vars


# def soft_clip(x):
#     return x * np.exp(-1.5 * x) + 1.0 * (1 - np.exp(-1.5 * x))

def soft_clip(x, x0=0.0, x1=1.0):
    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return t ** 3 * (t * (6 * t - 15) + 10)


def composite_obj(x, parameter_bounds, linear_relations: Dict[str, LinearGaussianCPD], slos_all_clients, total_rps):
    variables = {list(param.keys())[0]: val for param, val in zip(parameter_bounds.values(), x)}

    # ---------- Part 1: LGBN Relations ----------

    # arguments = {}
    for key, item in linear_relations.items():
        variables[key] = item.beta[0]
        for i in range(1, len(item.beta)):
            variables[key] = variables[key] + (variables[item.evidence[i - 1]] * item.beta[i])

    variables |= calculate_missing_vars(variables, total_rps)
    # arguments["throughput"] = (1000 / arguments["avg_p_latency"])
    # if linear_relations['avg_p_latency'] is not None and 'cores' not in linear_relations['avg_p_latency'].evidence:
    #     arguments["throughput"] = arguments["throughput"] * variables["cores"]
    # arguments["completion_rate"] = arguments["throughput"] / total_rps

    # ---------- Part 2: Client SLOs ----------

    slo_f_all_clients = 0
    for slos_single_client in slos_all_clients:

        slo_f_single_client = 0
        max_slo_f_single_client = sum([s.weight for s in slos_single_client.values()])

        for slo in slos_single_client.values():
            var, larger, thresh, weight = slo
            value = variables[var]
            if larger:
                slo_f_single_slo = (value / float(thresh))
            else:
                slo_f_single_slo = 1 - ((value - float(thresh)) / float(thresh))

            slo_f_single_client += soft_clip(slo_f_single_slo) * weight
            # print("soft_clip(slo_f_single_slo) * weight", soft_clip(slo_f_single_slo) * weight)
        slo_f_all_clients += (slo_f_single_client / max_slo_f_single_client)

    slo_f = slo_f_all_clients / len(slos_all_clients)
    # print(f"Calculated SLO-F for {variables}: {slo_f}")
    return -slo_f  # because we want to maximize


def solve(parameter_bounds, linear_relations, clients_SLOs, total_rps):
    bounds = [(inner["min"], inner["max"]) for param in parameter_bounds.values() for inner in param.values()] # Shape [(360, 1080), (1, 8)]
    x0 = [random.randint(mini, maxi) for mini, maxi in bounds]  # Initial guess; Shape [520, 4]

    result = minimize(composite_obj, x0, method='L-BFGS-B', bounds=bounds,
                      args=(parameter_bounds, linear_relations, clients_SLOs, total_rps))

    # print(result)
    if not result.success:
        raise RuntimeWarning("Policy solver encountered an error: " + result.message)

    es_param_ass = {}
    for index, var_name in enumerate([inner for param in parameter_bounds.values() for inner in param.keys()]):
        es_param_ass[var_name] = int(result.x[index]) # Might need higher precision for resources later

    return es_param_ass
