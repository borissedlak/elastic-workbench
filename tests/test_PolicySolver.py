from unittest import TestCase

from pgmpy.factors.continuous import LinearGaussianCPD

from agent.ES_Registry import EsType
from agent.PolicySolver import solve, composite_obj
from agent.SLO_Registry import SLO


class TestPolicySolver(TestCase):

    def setUp(self):
        self.cpd_qr = LinearGaussianCPD(variable='avg_p_latency', evidence=['quality'], beta=[-6.7, 0.05], std=740)
        self.cpd_cv = LinearGaussianCPD(variable='avg_p_latency', evidence=['cores'], beta=[34.512, -7.509], std=740)

    def test_composite_obj(self):
        # parameter_bounds = [{'es_type': EsType.RESOURCE_SCALE, 'min': 1, 'max': 8, 'name': 'cores'},
        #                     {'es_type': EsType.QUALITY_SCALE, 'min': 360, 'max': 1080, 'name': 'quality'}]
        parameter_bounds = {EsType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 8}},
                            EsType.QUALITY_SCALE: {'quality': {'min': 360, 'max': 1080}}}
        linear_relation = {'avg_p_latency': self.cpd_qr}  # TODO: I think this needs the throughput relation
        clients_slos = [{'throughput': SLO(var='throughput', larger=True, thresh=100000, weight=1.0),
                         'completion_rate': SLO(var='completion_rate', larger=True, thresh=1.0, weight=1.0),
                         'quality': SLO(var='quality', larger=True, thresh=800, weight=0.7)}]

        calculated_slo_f = composite_obj([6.0, 800.0], parameter_bounds, linear_relation, clients_slos, 1)
        self.assertAlmostEqual((1.7 / 2.7), -calculated_slo_f, delta=0.015)

        calculated_slo_f = composite_obj([6.0, 360.0], parameter_bounds, linear_relation, clients_slos, 1)
        self.assertAlmostEqual((1.0 + ((360 / 800) * 0.7)) / 2.7, -calculated_slo_f, delta=0.015)

        loose_boundary = SLO(var='throughput', larger=True, thresh=1, weight=1.0)
        clients_slos[0]['throughput'] = loose_boundary

        calculated_slo_f = composite_obj([8.0, 800.0], parameter_bounds, linear_relation, clients_slos, 1)
        self.assertAlmostEqual(1.0, -calculated_slo_f, delta=0.015)

    def test_solve_cv_minimal(self):
        parameter_bounds = [{'es_type': EsType.RESOURCE_SCALE, 'min': 7, 'max': 7, 'name': 'cores'}]
        linear_relation = {'avg_p_latency': self.cpd_cv}
        clients_slos_loose = [{'throughput': SLO(var='throughput', larger=True, thresh=50, weight=1.0)}]
        clients_slos_tight = [{'throughput': SLO(var='throughput', larger=True, thresh=500, weight=1.0)}]

        print(solve(parameter_bounds, linear_relation, clients_slos_loose, 10))
        # print(solve(parameter_bounds, linear_relation, clients_slos_tight, 50))
        # print(solve(parameter_bounds, linear_relation, clients_slos_tight, 5000))

        # self.lgbn = LGBN()
        # self.slo_registry = SLO_Registry('./static/slo_config.json')
        #
        # service_state_exp = self.lgbn.get_expected_state({'cores': 4}, ServiceType.CV, 50)
        # client_SLO_F_exp = calculate_slo_fulfillment(service_state_exp, clients_slos_loose[0])
        # print("Expected SLO-F", client_SLO_F_exp)

    def test_solve_qr(self):
        parameter_bounds = [{'es_type': EsType.RESOURCE_SCALE, 'min': 1, 'max': 8, 'name': 'cores'},
                            {'es_type': EsType.QUALITY_SCALE, 'min': 360, 'max': 1080, 'name': 'quality'}]
        linear_relation = {'avg_p_latency': self.cpd_qr}
        clients_slos = [{'avg_p_latency': SLO(var='avg_p_latency', larger=False, thresh=800, weight=1.0),
                         'completion_rate': SLO(var='completion_rate', larger=True, thresh=1.0, weight=1.0),
                         'quality': SLO(var='quality', larger=True, thresh=800, weight=1.0)}]

        print(solve(parameter_bounds, linear_relation, clients_slos, 1))
