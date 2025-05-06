import random
from typing import Dict

from pgmpy.factors.continuous import LinearGaussianCPD
from scipy.optimize import minimize

from agent.LGBN import calculate_missing_vars


# Exponential decay from linear start
# def soft_clip(x):  # thats a tuned sigmoid
#     sharpness = 10  # make it sharper
#     return 1 / (1 + np.exp(-sharpness * (x - 0.5)))  # center at 0.5

def soft_clip(x, upper_bound=1.0):
    return min(x, 1.0)


def composite_obj(x, parameter_bounds, linear_relations: Dict[str, LinearGaussianCPD], slos_all_clients, total_rps):
    variables = {param['name']: val for param, val in zip(parameter_bounds, x)}

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
    print(f"Calculated SLO-F for {variables}: {slo_f}")
    return -slo_f  # because we want to maximize


def solve(parameter_bounds, linear_relations, clients_SLOs, total_rps):
    bounds = [(param["min"], param["max"]) for param in parameter_bounds]  # Shape [(360, 1080), (1, 8)]
    x0 = [random.randint(mini, maxi) for mini, maxi in bounds]  # Initial guess; Shape [520, 4]

    result = minimize(composite_obj, x0, method='L-BFGS-B', bounds=bounds,
                      args=(parameter_bounds, linear_relations, clients_SLOs, total_rps))

    print(result)
    if not result.success:
        raise RuntimeWarning("Policy solver encountered an error: " + result.message)

    es_param_ass = {}
    for index, param in enumerate(parameter_bounds):
        es_param_ass[param["name"]] = result.x[index]

    return es_param_ass
