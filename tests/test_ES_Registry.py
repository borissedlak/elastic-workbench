from unittest import TestCase

from agent.ES_Registry import ES_Registry, EsType, ServiceType


class TestES_Registry(TestCase):

    def setUp(self):
        self.es_registry = ES_Registry('./static/es_registry.json')

    def test_is_es_supported(self):
        self.assertTrue(self.es_registry.is_ES_supported(ServiceType.QR, EsType.QUALITY_SCALE))
        self.assertTrue(self.es_registry.is_ES_supported(ServiceType.QR, EsType.RESOURCE_SCALE))
        self.assertFalse(self.es_registry.is_ES_supported(ServiceType.QR, EsType.MODEL_SCALE))
        self.assertFalse(self.es_registry.is_ES_supported(ServiceType.QR, EsType.OFFLOADING))

        self.assertFalse(self.es_registry.is_ES_supported(ServiceType.CV, EsType.QUALITY_SCALE))

    def test_get_supported_es_for_s(self):
        self.assertListEqual([], self.es_registry.get_supported_ES_for_service(ServiceType.CV))
        self.assertListEqual([EsType.QUALITY_SCALE, EsType.RESOURCE_SCALE],
                             self.es_registry.get_supported_ES_for_service(ServiceType.QR))

    def test_get_es_cooldown(self):
        self.assertEqual(1500, self.es_registry.get_ES_cooldown(ServiceType.QR, EsType.QUALITY_SCALE))
        self.assertEqual(2000, self.es_registry.get_ES_cooldown(ServiceType.QR, EsType.RESOURCE_SCALE))

        self.assertEqual(3500, self.es_registry.get_ES_cooldown(ServiceType.CV, EsType.STARTUP))
        self.assertEqual(0, self.es_registry.get_ES_cooldown(ServiceType.CV, EsType.RESOURCE_SCALE))

    def test_get_parameter_bounds_for_active_es(self):
        t_param_dict_qr = {EsType.QUALITY_SCALE: {'quality': {'min': 360, 'max': 1080}},
                           EsType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 8}}}
        self.assertDictEqual(t_param_dict_qr, self.es_registry.get_parameter_bounds_for_active_ES(ServiceType.QR))

        t_param_dict_qr = {EsType.QUALITY_SCALE: {'quality': {'min': 360, 'max': 1080}},
                           EsType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 4}}}
        self.assertDictEqual(t_param_dict_qr, self.es_registry.get_parameter_bounds_for_active_ES(ServiceType.QR, 4))

        self.assertDictEqual({}, self.es_registry.get_parameter_bounds_for_active_ES(ServiceType.CV))
