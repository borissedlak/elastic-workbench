from unittest import TestCase

from agent.ES_Registry import EsType, ServiceType
from agent.PolicySolver_RRM import solve, local_obj, solve_global
from agent.SLO_Registry import SLO


class TestPolicySolver(TestCase):

    def setUp(self):
        # self.cpd_qr = LinearGaussianCPD(variable='throughput', evidence=['quality', 'cores'],
        #                                 beta=[100.294, -0.085, 5.237], std=246)
        # self.cpd_cv = LinearGaussianCPD(variable='throughput', evidence=['cores', 'model_size'],
        #                                 beta=[32.311, 8.081, -26.618], std=58)
        pass

    def test_composite_obj(self):
        parameter_bounds = {EsType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 8}},
                            EsType.QUALITY_SCALE: {'quality': {'min': 360, 'max': 1080}}}
        clients_slos = [{'throughput': SLO(var='throughput', larger=True, thresh=100000, weight=1.0),
                         'completion_rate': SLO(var='completion_rate', larger=True, thresh=1.0, weight=1.0),
                         'quality': SLO(var='quality', larger=True, thresh=800, weight=0.7)}]

        calculated_slo_f = local_obj([6.0, 800.0], ServiceType.QR, parameter_bounds, clients_slos, 1)
        self.assertAlmostEqual((1.7 / 2.7), -calculated_slo_f, delta=0.015)

        calculated_slo_f = local_obj([6.0, 360.0], ServiceType.QR, parameter_bounds, clients_slos, 1)
        self.assertAlmostEqual((1.0 + ((360 / 800) * 0.7)) / 2.7, -calculated_slo_f, delta=0.015)

        loose_boundary = SLO(var='throughput', larger=True, thresh=1, weight=1.0)
        clients_slos[0]['throughput'] = loose_boundary

        calculated_slo_f = local_obj([8.0, 800.0], ServiceType.QR, parameter_bounds, clients_slos, 1)
        self.assertAlmostEqual(1.0, -calculated_slo_f, delta=0.015)

    def test_solve_cv_minimal(self):
        parameter_bounds_cv = {EsType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 8}},
                               EsType.MODEL_SCALE: {'model_size': {'min': 1, 'max': 2}}}
        clients_slos_cv = [{'throughput': SLO(var='throughput', larger=True, thresh=100000, weight=1.0),
                            'completion_rate': SLO(var='completion_rate', larger=True, thresh=1.0, weight=1.0),
                            'model_size': SLO(var='model_size', larger=True, thresh=2, weight=0.7)}]

        print(solve(ServiceType.CV, parameter_bounds_cv, clients_slos_cv, 10))

    def test_solve_qr(self):
        parameter_bounds = {EsType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 8}},
                            EsType.QUALITY_SCALE: {'quality': {'min': 360, 'max': 1080}}}
        clients_slos = [{'throughput': SLO(var='throughput', larger=True, thresh=100000, weight=1.0),
                         'completion_rate': SLO(var='completion_rate', larger=True, thresh=1.0, weight=1.0),
                         'quality': SLO(var='quality', larger=True, thresh=800, weight=0.7)}]

        print(solve(ServiceType.QR, parameter_bounds, clients_slos, 1))

    def test_solve_global(self):
        max_cores = 8
        service_context = []

        parameter_bounds_qr = {EsType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 8}},
                               EsType.QUALITY_SCALE: {'quality': {'min': 300, 'max': 1000}}}
        clients_slos_qr = [{'throughput': SLO(var='throughput', larger=True, thresh=100, weight=1.0),
                            'quality': SLO(var='quality', larger=True, thresh=800, weight=0.7)}]
        service_context.append((ServiceType.QR, parameter_bounds_qr, clients_slos_qr, 100))
        # service_context.append((parameter_bounds_qr, {'throughput': self.cpd_qr}, clients_slos_qr, 100))

        parameter_bounds_cv = {EsType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 8}},
                               EsType.MODEL_SCALE: {'model_size': {'min': 1, 'max': 5}},
                               EsType.QUALITY_SCALE: {'quality': {'min': 300, 'max': 1000}}}
        clients_slos_cv = [{'throughput': SLO(var='throughput', larger=True, thresh=10, weight=0.6),
                            'model_size': SLO(var='model_size', larger=True, thresh=2, weight=0.2)}]
        service_context.append((ServiceType.CV, parameter_bounds_cv, clients_slos_cv, 100))

        print(solve_global(service_context, max_cores))
