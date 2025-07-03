from unittest import TestCase

from agent.components.es_registry import ServiceType
from agent.components.SLORegistry import SLO_Registry, SLO, calculate_slo_fulfillment, smoothstep


class TestSLO_Registry(TestCase):

    def setUp(self):
        self.slo_registry = SLO_Registry('./static/slo_config.json')

        self.slos_c_1_qr = self.slo_registry.get_SLOs_for_client("C_1", ServiceType.QR)
        self.slos_c_3_cv = self.slo_registry.get_SLOs_for_client("C_3", ServiceType.CV)

        self.service_state_qr = {"quality": 700, "avg_p_latency": 1200, "completion_rate": 0.8, "throughput": 500}
        self.service_state_qr_2 = {"quality": 900, "avg_p_latency": 200, "completion_rate": 1.5}

        self.service_state_cv = {"model_size": 2, "avg_p_latency": 150, "completion_rate": 0.8, "quality": 500}
        self.service_state_cv_2 = {"model_size": 1, "avg_p_latency": 20, "completion_rate": 1.0, "quality": 500}

    def test_get_slos_for_client(self):
        t_slos_c_1_qr = {'quality': SLO(**{'var': 'quality', 'larger': True, 'target': 800, 'weight': 1.0}),
                         'avg_p_latency': SLO(
                             **{'var': 'avg_p_latency', 'larger': False, 'target': 1000, 'weight': 1.0}),
                         'completion_rate': SLO(
                             **{'var': 'completion_rate', 'larger': True, 'target': 1.0, 'weight': 1.0})}

        t_slos_c_3_cv = {
            'avg_p_latency': SLO(**{'var': 'avg_p_latency', 'larger': False, 'target': 100, 'weight': 1.0}),
            'completion_rate': SLO(**{'var': 'completion_rate', 'larger': True, 'target': 1.0, 'weight': 1.0}),
            'model_size': SLO(**{'var': 'model_size', 'larger': True, 'target': 2, 'weight': 1.0})}

        self.assertDictEqual(t_slos_c_1_qr, self.slos_c_1_qr)
        self.assertDictEqual(t_slos_c_3_cv, self.slos_c_3_cv)

    def test_calculate_slo_fulfillment_qr(self):
        t_slo_f_qr_1 = [("quality", smoothstep(0.875)), ("avg_p_latency", smoothstep(0.8)), ("completion_rate", smoothstep(0.8))]
        slo_f_qr_1 = calculate_slo_fulfillment(self.service_state_qr, self.slos_c_1_qr)
        t_slo_f_qr_2 = [("quality", 1.0), ("avg_p_latency", 1.0), ("completion_rate", 1.0)]
        slo_f_qr_2 = calculate_slo_fulfillment(self.service_state_qr_2, self.slos_c_1_qr)

        self.assertCountEqual(t_slo_f_qr_1, slo_f_qr_1)
        self.assertCountEqual(t_slo_f_qr_2, slo_f_qr_2)

    def test_calculate_slo_fulfillment_cv(self):
        t_slo_f_cv_1 = [("model_size", 1.0), ("avg_p_latency", 0.5), ("completion_rate", smoothstep(0.8))]
        slo_f_cv_1 = calculate_slo_fulfillment(self.service_state_cv, self.slos_c_3_cv)
        t_slo_f_cv_2 = [("model_size", smoothstep(0.5)), ("avg_p_latency", 1.0), ("completion_rate", 1.0)]
        slo_f_cv_2 = calculate_slo_fulfillment(self.service_state_cv_2, self.slos_c_3_cv)

        self.assertCountEqual(t_slo_f_cv_1, slo_f_cv_1)
        self.assertCountEqual(t_slo_f_cv_2, slo_f_cv_2)

    # TODO: Should fix at some point; the current problem is that I cannot apply soft clip on the left side simply
    #  because currently this is applied to the different factors before summation
    # def test_to_avg_slo_f(self):
    #     slo_f_cv_1 = calculate_slo_fulfillment(self.service_state_cv, self.slos_c_3_cv)
    #     slo_f_cv_2 = calculate_slo_fulfillment(self.service_state_cv_2, self.slos_c_3_cv)
    #
    #     slo_f_qr_1 = calculate_slo_fulfillment(self.service_state_qr, self.slos_c_1_qr)
    #     slo_f_qr_2 = calculate_slo_fulfillment(self.service_state_qr_2, self.slos_c_1_qr)
    #
    #     self.assertEqual(2.475 / 3, to_normalized_SLO_F(slo_f_qr_1, self.slos_c_1_qr))
    #     self.assertEqual(3.0 / 3, to_normalized_SLO_F(slo_f_qr_2, self.slos_c_1_qr))
    #     self.assertEqual(2.3 / 3, to_normalized_SLO_F(slo_f_cv_1, self.slos_c_3_cv))
    #     self.assertEqual(2.5 / 3, to_normalized_SLO_F(slo_f_cv_2, self.slos_c_3_cv))
