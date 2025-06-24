import random

from scipy.optimize import minimize

from agent.LGBN import calculate_missing_vars
from agent.RASK import RASK
from agent.SLORegistry import calculate_SLO_F_clients
from agent.es_registry import ServiceType


def solve(service_type, parameter_bounds, clients_SLOs, total_rps, rask: RASK):
    bounds = [(inner["min"], inner["max"]) for param in parameter_bounds.values() for inner in
              param.values()]  # Shape [(360, 1080), (1, 8)]
    x0 = [random.randint(mini, maxi) for mini, maxi in bounds]  # Initial guess; Shape [520, 4]

    result = minimize(local_obj, x0, method='L-BFGS-B', bounds=bounds,
                      args=(service_type, parameter_bounds, clients_SLOs, total_rps, rask))

    if not result.success:
        raise RuntimeWarning("Policy solver encountered an error: " + result.message)

    es_param_ass = {}
    for index, var_name in enumerate([inner for param in parameter_bounds.values() for inner in param.keys()]):
        es_param_ass[var_name] = int(result.x[index])  # Might need higher precision for resources later

    return es_param_ass


def local_obj(x, service_type: ServiceType, parameter_bounds, slos_all_clients, total_rps, rask: RASK):
    independent_variables = {list(param.keys())[0]: val for param, val in zip(parameter_bounds.values(), x)}

    # ---------- Part 1: LGBN Relations ----------

    dependent_variables = rask.get_all_dependent_vars_ass(service_type, independent_variables)
    full_state = independent_variables | dependent_variables
    full_state |= calculate_missing_vars(full_state, total_rps)

    # ---------- Part 2: Client SLOs ----------

    slo_f = calculate_SLO_F_clients(full_state, slos_all_clients)
    # print(f"Calculated SLO-F for {full_state}: {slo_f}")
    return -slo_f  # because we want to maximize


def composite_obj_global(x, service_context, rask: RASK):
    offset = 0
    total_slo_f = 0

    for service_type, parameter_bounds, slos, total_rps in service_context:
        num_params = sum(len(p) for p in parameter_bounds.values())
        x_i = x[offset:offset + num_params]
        offset += num_params

        slo_f_i = local_obj(x_i, service_type, parameter_bounds, slos, total_rps, rask)
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


def solve_global(service_contexts_m, max_cores, rask: RASK):

    constraints = [{'type': 'eq', 'fun': constraint_total_cores, 'args': (service_contexts_m, max_cores)}]
    flat_bounds = []

    x0 = []
    for _, parameter_bounds, _, _ in service_contexts_m:
        for ES_desc in parameter_bounds.values():
            ES_var = list(ES_desc.keys())[0]
            flat_bounds.append((ES_desc[ES_var]["min"], ES_desc[ES_var]["max"]))

            # Set default value for starting numerical solver
            if ES_var == 'cores':
                x0.append(max_cores / len(service_contexts_m))
            else:
                x0.append((ES_desc[ES_var]["min"] + ES_desc[ES_var]["max"]) / 2)

    result = minimize(composite_obj_global, x0, method='SLSQP', constraints=constraints,
                      bounds=flat_bounds, args=(service_contexts_m, rask))

    # print(result)
    if not result.success:
        raise RuntimeWarning("Policy solver encountered an error: " + result.message)

    assignments = []
    offset = 0
    for _, parameter_bounds, _, _ in service_contexts_m:
        param_names = [k for group in parameter_bounds.values() for k in group]
        num_params = len(param_names)
        x_i = result.x[offset:offset + num_params]
        assignments.append(dict(zip(param_names, x_i)))
        offset += num_params

    return assignments
