import unittest
from unittest import TestCase

from pgmpy.factors.continuous import LinearGaussianCPD

from agent.ES_Registry import EsType
from agent.PolicySolver import solve
from agent.SLO_Registry import SLO


class TestPolicySolver(TestCase):

    def test_solve_qr(self):
        cpd_qr = LinearGaussianCPD(variable='avg_p_latency', evidence=['quality'], beta=[-6.7, 0.05004630052983987],
                                   std=740.654)
        parameter_bounds = [{'es_type': EsType.RESOURCE_SCALE, 'max': 8, 'min': 1, 'name': 'cores'},
                            {'es_type': EsType.QUALITY_SCALE, 'max': 1080, 'min': 360, 'name': 'quality'}]
        linear_relation = {'avg_p_latency': cpd_qr}
        clients_slos = [{'quality': {'var': 'quality', 'larger': 'True', 'thresh': '800', 'weight': 1.0},
                         'avg_p_latency': {'var': 'avg_p_latency', 'larger': 'False', 'thresh': '30', 'weight': 1.0},
                         'completion_rate': {'var': 'completion_rate', 'larger': 'True', 'thresh': '1.0',
                                             'weight': 1.0}},
                        {'quality': {'var': 'quality', 'larger': 'True', 'thresh': '1000', 'weight': 1.0},
                         'avg_p_latency': {'var': 'avg_p_latency', 'larger': 'False', 'thresh': '70', 'weight': 1.0},
                         'completion_rate': {'var': 'completion_rate', 'larger': 'True', 'thresh': '1.0',
                                             'weight': 1.0}}]

        print(solve(parameter_bounds, linear_relation, clients_slos, 1))



    def test_solve_cv(self):
        cpd_cv = LinearGaussianCPD(variable='avg_p_latency', evidence=['cores', 'model_size'],
                                   beta=[34.512, -7.509, 41.19], std=740.654)

        parameter_bounds = [{'es_type': EsType.RESOURCE_SCALE, 'max': 8, 'min': 1, 'name': 'cores'},
                            {'es_type': EsType.MODEL_SCALE, 'max': 2, 'min': 1, 'name': 'model_size'}]
        linear_relation = {'avg_p_latency': cpd_cv}
        clients_slos = [{'avg_p_latency': SLO(var='avg_p_latency', larger=False, thresh=1000, weight=1.0),
                         'completion_rate': SLO(var='completion_rate', larger=True, thresh=1.0, weight=1.0),
                         'model_size': SLO(var='model_size', larger=True, thresh=2.0, weight=1.0)}]



        print(solve(parameter_bounds, linear_relation, clients_slos, 1))
