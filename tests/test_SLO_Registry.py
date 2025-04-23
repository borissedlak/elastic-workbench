from unittest import TestCase

from agent.ES_Registry import ServiceType
from agent.SLO_Registry import SLO_Registry, SLO


class TestSLO_Registry(TestCase):

    def setUp(self):
        self.slo_registry = SLO_Registry('./static/slo_config.json')

        self.slos_c_1_qr = self.slo_registry.get_SLOs_for_client("C_1", ServiceType.QR)
        self.slos_c_3_cv = self.slo_registry.get_SLOs_for_client("C_3", ServiceType.CV)

    def test_get_slos_for_client(self):
        t_slos_c_1_qr = {'quality': SLO('quality', True, 800, 1.0),
                         'avg_p_latency': SLO('avg_p_latency', False, 1000, 1.0),
                         'completion_rate': SLO('completion_rate', True, 1.0, 1.0)}

        t_slos_c_3_cv = {'avg_p_latency': {SLO('avg_p_latency', 'larger': False, 'thresh': 100, 'weight': 1.0},
                         'completion_rate': {SLO( 'completion_rate', 'larger': True, 'thresh': 1.0, 'weight': 1.0},
                         'model_size': {SLO('model_size', 'larger': True, 'thresh': 2, 'weight': 1.0}}

        self.assertDictEqual(t_slos_c_1_qr, self.slos_c_1_qr)
        self.assertDictEqual(t_slos_c_3_cv, self.slos_c_3_cv)

    def test_calculate_slo_fulfillment_qr(self):
        # TODO: See in agent if the state commonly looks like this
        service_state_qr = {"quality": 700, "avg_p_latency": 1200, "completion_rate": 0.8, "throughput": 500}
        service_state_qr_2 = {"quality": 900, "avg_p_latency": 200, "completion_rate": 1.5}

        t_slo_f_qr_1 = [("quality", 700 / 800), ("avg_p_latency", 0.8), ("completion_rate", 0.8)]
        slo_f_qr_1 = self.slo_registry.calculate_slo_fulfillment(service_state_qr, self.slos_c_1_qr)
        t_slo_f_qr_2 = [("quality", 1.0), ("avg_p_latency", 1.0), ("completion_rate", 1.0)]
        slo_f_qr_2 = self.slo_registry.calculate_slo_fulfillment(service_state_qr_2, self.slos_c_1_qr)

        self.assertListEqual(t_slo_f_qr_1, slo_f_qr_1)
        self.assertListEqual(t_slo_f_qr_2, slo_f_qr_2)

    def test_calculate_slo_fulfillment_cv(self):
        service_state_cv = {"model_size": 2, "avg_p_latency": 150, "completion_rate": 0.8, "quality": 500}
        service_state_cv_2 = {"model_size": 1, "avg_p_latency": 20, "completion_rate": 1.0, "quality": 500}

        t_slo_f_cv_1 = [("model_size", 1.0), ("avg_p_latency", 0.5), ("completion_rate", 0.8)]
        slo_f_cv_1 = self.slo_registry.calculate_slo_fulfillment(service_state_cv, self.slos_c_3_cv)
        t_slo_f_cv_2 = [("model_size", 0.5), ("avg_p_latency", 1.0), ("completion_rate", 1.0)]
        slo_f_cv_2 = self.slo_registry.calculate_slo_fulfillment(service_state_cv_2, self.slos_c_3_cv)

        self.assertListEqual(t_slo_f_cv_1, slo_f_cv_1)
        self.assertListEqual(t_slo_f_cv_2, slo_f_cv_2)

    # def test_get_all_slos_for_assigned_clients(self):
    #     self.fail()
    #
    # def test_to_avg_slo_f(self):
    #     self.fail()
