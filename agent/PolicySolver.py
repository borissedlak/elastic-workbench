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
            arguments[key] = arguments[key] + (variables[item.evidence[i-1]] * item.beta[i])

    arguments["throughput"] = (1000 / arguments["avg_p_latency"])
    if linear_relations['avg_p_latency'] is not None and 'cores' not in linear_relations['avg_p_latency'].evidence:
         arguments["throughput"] = arguments["throughput"] * variables["cores"]
    arguments["completion_rate"] = arguments["throughput"] / total_rps

    # ---------- Part 2: Client SLOs ----------

    overall_slo_f = 0
    for client_SLOs in all_SLOs:
        client_slof = 0
        for _, item in client_SLOs.items():
            var, larger, thresh, weight = tuple(item.values())
            value = (variables | arguments)[var]
            if larger:
                single_slo = (value / float(thresh))
            else:
                single_slo = 1 - ((value - float(thresh)) / float(thresh))

            client_slof += soft_clip(single_slo, 1.0) * weight
        client_x_max_slof = sum([s['weight'] for s in client_SLOs.values()])
        overall_slo_f += (client_slof / client_x_max_slof)

    slo_f = overall_slo_f / len(all_SLOs)
    return -slo_f  # because we want to maximize


def solve(parameter_bounds, linear_relations, clients_SLOs, total_rps, verify=False):
    bounds = [(param["min"], param["max"]) for param in parameter_bounds] # Shape [(360, 1080), (1, 8)]
    x0 = [random.randint(mini, maxi) for mini, maxi in bounds]  # Initial guess; Shape [520, 4]

    result = minimize(composite_obj, x0, method='L-BFGS-B', bounds=bounds,
                      args=(parameter_bounds, linear_relations, clients_SLOs, total_rps))

    if not result.success:
        raise RuntimeWarning("Policy solver encountered an error: " + result.message)

    # if verify:
    #     # for r in [result, result_2]:
    #     v_1 = round(result.x[0])
    #     v_2 = round(result.x[1])
    #
    #     variables = {"model_size": v_1, "cores": v_2}
    #
    #     arguments = {}
    #     for key, item in linear_relations.items():
    #         # variable_ref, k, d = item[0]
    #         arguments[key] = item.beta[0]
    #         for i in range(1, len(item.beta)):
    #             arguments[key] = arguments[key] + (variables[item.evidence[i - 1]] * item.beta[i])
    #             # (variables[variable_ref] * k) +
    #
    #     arguments["throughput"] = (1000 / arguments["avg_p_latency"]) * variables["cores"]
    #     arguments["completion_rate"] = arguments["throughput"] / total_rps
    #
    #     # client_slos = [(800, 70), (1000, 20)]
    #     verify = []
    #     overall_slo_f = 0
    #     for client_SLOs in clients_SLOs:
    #         client_slof = 0
    #         for _, item in client_SLOs.items():
    #             var, larger, thresh, weight = tuple(item.values())
    #             value = (variables | arguments)[var]
    #             if larger:
    #                 single_slo = (value / float(thresh))
    #             else:
    #                 single_slo = 1 - ((value - float(thresh)) / float(thresh))
    #
    #             client_slof += soft_clip(single_slo, 1.0) * weight
    #             verify.append({"var": var, "thresh": thresh, "weight": weight, "before_clip": single_slo,
    #                            "after_clip_and_weight": soft_clip(single_slo, 1.0) * weight})
    #         client_x_max_slof = sum([s['weight'] for s in client_SLOs.values()])
    #         overall_slo_f += (client_slof / client_x_max_slof)
    #
    #     slo_f = overall_slo_f / len(clients_SLOs)
    #
    #     print("Status:", result.message)
    #     print("Optimal SLO F:", slo_f)
    #     # print("Pixel-based SLO:", slo_pixel)
    #     # print("Latency-based SLO:", slo_latency)
    #     # print("Completion-based SLO:", slo_completion)
    #     print("Optimal pixel (rounded):", v_1)
    #     print("Optimal cores (rounded):", v_2)
    #     # print("Optimal latency:", p_latency)
    #     # print("Estimated throughput:", throughput)

    es_param_ass = {}
    for index, param in enumerate(parameter_bounds):
        es_param_ass[param["name"]] = int(result.x[index])

    return es_param_ass


if __name__ == '__main__':
    # cpd_1 = LinearGaussianCPD(variable='avg_p_latency', evidence=['cores', 'model_size'], beta=[34.512, -7.509, 41.19],
    #                           std=740.654)
    #
    # parameter_bounds = [{'es_type': EsType.RESOURCE_SCALE, 'max': 8, 'min': 1, 'name': 'cores'},
    #                     {'es_type': EsType.MODEL_SCALE, 'max': 2, 'min': 1, 'name': 'model_size'}]
    # linear_relation = {'avg_p_latency': cpd_1}
    # clients_slos = [{'avg_p_latency': {'var': 'avg_p_latency', 'larger': 'False', 'thresh': '30', 'weight': 1.0},
    #                  'completion_rate': {'var': 'completion_rate', 'larger': 'True', 'thresh': '1.0', 'weight': 1.0}},
    #                 {'avg_p_latency': {'var': 'avg_p_latency', 'larger': 'False', 'thresh': '70', 'weight': 1.0},
    #                  'completion_rate': {'var': 'completion_rate', 'larger': 'True', 'thresh': '1.0', 'weight': 1.0}}]

    cpd_1 = LinearGaussianCPD(variable='avg_p_latency', evidence=['quality'], beta=[-6.7, 0.05004630052983987], std=740.654)
    parameter_bounds = [{'es_type': EsType.RESOURCE_SCALE, 'max': 8, 'min': 1, 'name': 'cores'}, {'es_type': EsType.QUALITY_SCALE, 'max': 1080, 'min': 360, 'name': 'quality'}]
    linear_relation = {'avg_p_latency': cpd_1}
    clients_slos = [{'quality': {'var': 'quality', 'larger': 'True', 'thresh': '800', 'weight': 1.0},
                     'avg_p_latency': {'var': 'avg_p_latency', 'larger': 'False', 'thresh': '30', 'weight': 1.0},
                     'completion_rate': {'var': 'completion_rate', 'larger': 'True', 'thresh': '1.0', 'weight': 1.0}},
                    {'quality': {'var': 'quality', 'larger': 'True', 'thresh': '1000', 'weight': 1.0},
                     'avg_p_latency': {'var': 'avg_p_latency', 'larger': 'False', 'thresh': '70', 'weight': 1.0},
                     'completion_rate': {'var': 'completion_rate', 'larger': 'True', 'thresh': '1.0', 'weight': 1.0}}]
    print(solve(parameter_bounds, linear_relation, clients_slos, 1, True))
