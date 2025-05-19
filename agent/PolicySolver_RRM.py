import random

import numpy as np
from scipy.optimize import minimize

from agent.ES_Registry import ServiceType
from agent.LGBN import calculate_missing_vars
from agent.RRM import RRM


def soft_clip(x, x0=0.0, x1=1.0):
    t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return t ** 3 * (t * (6 * t - 15) + 10)


rrm = RRM(show_figures=False)


def solve(service_type, parameter_bounds, clients_SLOs, total_rps):
    bounds = [(inner["min"], inner["max"]) for param in parameter_bounds.values() for inner in
              param.values()]  # Shape [(360, 1080), (1, 8)]
    x0 = [random.randint(mini, maxi) for mini, maxi in bounds]  # Initial guess; Shape [520, 4]

    result = minimize(local_obj, x0, method='L-BFGS-B', bounds=bounds,
                      args=(service_type, parameter_bounds, clients_SLOs, total_rps))

    if not result.success:
        raise RuntimeWarning("Policy solver encountered an error: " + result.message)

    es_param_ass = {}
    for index, var_name in enumerate([inner for param in parameter_bounds.values() for inner in param.keys()]):
        es_param_ass[var_name] = int(result.x[index])  # Might need higher precision for resources later

    return es_param_ass


def local_obj(x, service_type: ServiceType, parameter_bounds, slos_all_clients, total_rps):
    independent_variables = {list(param.keys())[0]: val for param, val in zip(parameter_bounds.values(), x)}

    # ---------- Part 1: LGBN Relations ----------

    dependent_variables = rrm.get_all_dependent_vars_ass(service_type, independent_variables)
    full_state = independent_variables | dependent_variables
    full_state |= calculate_missing_vars(full_state, total_rps)

    # ---------- Part 2: Client SLOs ----------

    slo_f_all_clients = 0
    for slos_single_client in slos_all_clients:

        slo_f_single_client = 0
        max_slo_f_single_client = sum([s.weight for s in slos_single_client.values()])

        for slo in slos_single_client.values():
            var, larger, thresh, weight = slo
            value = full_state[var]
            if larger:
                slo_f_single_slo = (value / float(thresh))
            else:
                slo_f_single_slo = 1 - ((value - float(thresh)) / float(thresh))

            slo_f_single_client += soft_clip(slo_f_single_slo) * weight

        scaled_reward = slo_f_single_client / max_slo_f_single_client
        if full_state['throughput'] < 1.0:
            scaled_reward *= 0.1  # Heavily penalize if no output

        slo_f_all_clients += scaled_reward

    slo_f = slo_f_all_clients / len(slos_all_clients)
    # print(f"Calculated SLO-F for {full_state}: {slo_f}")
    return -slo_f  # because we want to maximize


def composite_obj_global(x, service_context):
    offset = 0
    total_slo_f = 0

    for service_type, parameter_bounds, slos, total_rps in service_context:
        num_params = sum(len(p) for p in parameter_bounds.values())
        x_i = x[offset:offset + num_params]
        offset += num_params

        slo_f_i = local_obj(x_i, service_type, parameter_bounds, slos, total_rps)
        total_slo_f += slo_f_i

    return total_slo_f  # already negative (since composite_obj returns -slo_f)


def constraint_total_cores(x, services, max_total_cores):
    offset = 0
    total_cores_ass = 0

    for _, parameter_bounds, _, _ in services:
        param_names = [name for group in parameter_bounds.values() for name in group]
        num_params = len(param_names)

        for i, name in enumerate(param_names):
            if name == 'cores':
                total_cores_ass += x[offset + i]

        offset += num_params

    return max_total_cores - total_cores_ass  # must be 0


def solve_global(service_contexts_m, max_cores):
    global rrm
    rrm.init_models() # Might not be needed every time we solve the assignment

    constraints = [{'type': 'eq', 'fun': constraint_total_cores, 'args': (service_contexts_m, max_cores)}]
    flat_bounds = []

    x0 = []
    for _, parameter_bounds, _, _ in service_contexts_m:
        for ES_desc in parameter_bounds.values():
            ES_var = list(ES_desc.keys())[0]
            flat_bounds.append((ES_desc[ES_var]["min"], ES_desc[ES_var]["max"]))

            if ES_var == 'cores':
                x0.append(max_cores / len(service_contexts_m))
            else:
                x0.append((ES_desc[ES_var]["min"] + ES_desc[ES_var]["max"]) / 2)

    result = minimize(composite_obj_global, x0, method='SLSQP', constraints=constraints,
                      bounds=flat_bounds, args=service_contexts_m)

    # print(result)
    if not result.success:
        raise RuntimeWarning("Policy solver encountered an error: " + result.message)

    # TODO: need to place into correct structure
    assignments = []
    offset = 0
    for _, parameter_bounds, _, _ in service_contexts_m:
        param_names = [k for group in parameter_bounds.values() for k in group]
        num_params = len(param_names)
        x_i = result.x[offset:offset + num_params]
        assignments.append(dict(zip(param_names, x_i)))
        offset += num_params

    return assignments
