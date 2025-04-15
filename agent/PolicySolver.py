import random

import numpy as np
from scipy.optimize import minimize


# Exponential decay from linear start
def soft_clip(x, max):
    return x * np.exp(-1.5 * x) + max * (1 - np.exp(-1.5 * x))


def composite_obj(x, parameter_bounds, linear_relations, all_SLOs, total_rps):
    variables = {param['name']: val for param, val in zip(parameter_bounds, x)}

    arguments = {}
    for key, item in linear_relations.items():
        variable_ref, k, d = item[0]
        arguments[key] = (variables[variable_ref] * k) + d

    arguments["throughput"] = (1000 / arguments["avg_p_latency"]) * variables["cores"]
    arguments["completion_rate"] = arguments["throughput"] / total_rps

    # client_slos = [(800, 70), (1000, 20)]
    overall_slo_f = 0
    client_slof = 0
    for client_SLOs in all_SLOs:
        for _, item in client_SLOs.items():
            var, larger, thresh, weight = tuple(item.values())
            value = (variables | arguments)[var]
            if larger == "True":
                single_slo = (value / float(thresh))
            else:
                single_slo = 1 - ((value - float(thresh)) / float(thresh))

            client_slof += soft_clip(single_slo, 1.0) * weight
        client_x_max_slof = sum([s['weight'] for s in client_SLOs.values()])
        overall_slo_f += client_slof / client_x_max_slof

    slo_f = overall_slo_f / len(all_SLOs)
    return -slo_f  # because we want to maximize


def objective(x):
    quality, cores = x
    p_latency = eval("0.049 * quality - 6.624", {}, {"quality": quality, "cores": cores})
    throughput = (1000 / p_latency) * cores

    client_slos = [(800, 70), (1000, 20)]
    overall_slo_f = 0

    for pixel_t, latency_t in client_slos:
        slo_pixel = soft_clip(quality / pixel_t, 1.0)
        slo_latency = soft_clip(1 - (p_latency - latency_t) / latency_t, 1.0)
        slo_completion = soft_clip(throughput / 50, 1.0)
        client_slo_f = (slo_pixel + slo_latency + slo_completion) / 3
        overall_slo_f += client_slo_f

    slo_f = overall_slo_f / len(client_slos)
    return -slo_f  # because we want to maximize


def solve(parameter_bounds, linear_relations, clients_SLOs, total_rps):
    # bounds = [(360, 1080), (1, 8)]
    bounds = [(param["min"], param["max"]) for param in parameter_bounds]
    # x0 = [520, 4]
    x0 = [random.randint(mini, maxi) for mini, maxi in bounds]  # Initial guess

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    result_2 = minimize(composite_obj, x0, method='L-BFGS-B', bounds=bounds,
                        args=(parameter_bounds, linear_relations, clients_SLOs, total_rps))

    for r in [result, result_2]:
        quality = round(r.x[0])
        cores = round(r.x[1])
        p_latency = 0.049 * quality - 6.624
        throughput = (1000 / p_latency) * cores
        slo_pixel = soft_clip(quality / 800, 1.0)
        slo_latency = soft_clip(1 - (p_latency - 70) / 70, 1.0)
        slo_completion = soft_clip(throughput / 50, 1.0)
        slo_f = (slo_pixel + slo_latency + slo_completion) / 3

        print("Status:", r.message)
        print("Optimal SLO F:", slo_f)
        print("Pixel-based SLO:", slo_pixel)
        print("Latency-based SLO:", slo_latency)
        print("Completion-based SLO:", slo_completion)
        print("Optimal pixel:", quality)
        print("Optimal cores (rounded):", cores)
        print("Optimal latency:", p_latency)
        print("Estimated throughput:", throughput)

    return {}


if __name__ == '__main__':
    parameter_bounds = [{'max': 1080, 'min': 360, 'name': 'quality'}, {'max': 8, 'min': 1, 'name': 'cores'}]
    linear_relation = {'avg_p_latency': [('quality', np.float64(0.05004630052983987), np.float64(-6.797676905942581))]}
    # clients_slos = [{'quality': {'var': 'quality', 'larger': 'True', 'thresh': '800', 'weight': 1.0},
    #                  'avg_p_latency': {'var': 'avg_p_latency', 'larger': 'False', 'thresh': '30', 'weight': 1.0},
    #                  'completion_rate': {'var': 'completion_rate', 'larger': 'True', 'thresh': '100', 'weight': 1.0}},
    #                 {'quality': {'var': 'quality', 'larger': 'True', 'thresh': '1000', 'weight': 1.0},
    #                  'avg_p_latency': {'var': 'avg_p_latency', 'larger': 'False', 'thresh': '70', 'weight': 1.0},
    #                  'completion_rate': {'var': 'completion_rate', 'larger': 'True', 'thresh': '100', 'weight': 1.0}}]
    clients_slos = [{'quality': {'var': 'quality', 'larger': 'True', 'thresh': '800', 'weight': 1.0}}]
    print(solve(parameter_bounds, linear_relation, clients_slos, 70))
