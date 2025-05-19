import random
from typing import Dict

import numpy as np
from pgmpy.factors.continuous import LinearGaussianCPD
from scipy.optimize import minimize

from agent.LGBN import calculate_missing_vars


# TODO: Next problem is that I should change to quadratic relations and incorporate this to the solver
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

        scaled_reward = slo_f_single_client / max_slo_f_single_client
        if variables['throughput'] < 1.0:
            scaled_reward *= 0.1 # Heavily penalize if no output

        slo_f_all_clients += scaled_reward

    slo_f = slo_f_all_clients / len(slos_all_clients)
    print(f"Calculated SLO-F for {variables}: {slo_f}")
    return -slo_f  # because we want to maximize


def solve(parameter_bounds, linear_relations, clients_SLOs, total_rps):
    bounds = [(inner["min"], inner["max"]) for param in parameter_bounds.values() for inner in
              param.values()]  # Shape [(360, 1080), (1, 8)]
    x0 = [random.randint(mini, maxi) for mini, maxi in bounds]  # Initial guess; Shape [520, 4]

    result = minimize(composite_obj, x0, method='L-BFGS-B', bounds=bounds,
                      args=(parameter_bounds, linear_relations, clients_SLOs, total_rps))

    # print(result)
    if not result.success:
        raise RuntimeWarning("Policy solver encountered an error: " + result.message)

    es_param_ass = {}
    for index, var_name in enumerate([inner for param in parameter_bounds.values() for inner in param.keys()]):
        es_param_ass[var_name] = int(result.x[index])  # Might need higher precision for resources later

    return es_param_ass


def composite_obj_global(x, service_context):
    offset = 0
    total_slo_f = 0

    for parameter_bounds, linear_relations, slos, total_rps in service_context:
        num_params = sum(len(p) for p in parameter_bounds.values())
        x_i = x[offset:offset + num_params]
        offset += num_params

        slo_f_i = composite_obj(x_i, parameter_bounds, linear_relations, slos, total_rps)
        total_slo_f += slo_f_i

    return total_slo_f  # already negative (since composite_obj returns -slo_f)


def constraint_total_cores(x, services, max_total_cores):
    offset = 0
    total_cores_ass = 0

    for parameter_bounds, *_ in services:
        param_names = [name for group in parameter_bounds.values() for name in group]
        num_params = len(param_names)

        for i, name in enumerate(param_names):
            if name == 'cores':
                total_cores_ass += x[offset + i]

        offset += num_params

    return max_total_cores - total_cores_ass  # must be 0


def solve_global(service_context_m, max_cores):
    constraints = [{'type': 'eq', 'fun': constraint_total_cores, 'args': (service_context_m, max_cores)}]
    flat_bounds = []

    x0 = []
    for parameter_bounds, *_ in service_context_m:
        for ES_desc in parameter_bounds.values():
            ES_var = list(ES_desc.keys())[0]
            flat_bounds.append((ES_desc[ES_var]["min"], ES_desc[ES_var]["max"]))

            if ES_var == 'cores':
                x0.append(max_cores / len(service_context_m))
            else:
                x0.append((ES_desc[ES_var]["min"] + ES_desc[ES_var]["max"]) / 2)

    result = minimize(composite_obj_global, x0, method='SLSQP', constraints=constraints,
                      bounds=flat_bounds, args=service_context_m)

    # print(result)
    if not result.success:
        raise RuntimeWarning("Policy solver encountered an error: " + result.message)

    # TODO: need to place into correct structure
    # es_param_ass = {}
    # for index, var_name in enumerate([inner for param in parameter_bounds.values() for inner in param.keys()]):
    #     es_param_ass[var_name] = int(result.x[index]) # Might need higher precision for resources later
    print(result.x)

    return result.x
