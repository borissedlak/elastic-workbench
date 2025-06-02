from unittest import TestCase

from agent.es_registry import ESRegistry, ESType, ServiceType


class TestES_Registry(TestCase):

    def setUp(self):
        self.es_registry = ESRegistry('./static/es_registry.json')

    def test_is_es_supported(self):
        self.assertTrue(self.es_registry.is_es_supported(ServiceType.QR, ESType.QUALITY_SCALE))
        self.assertTrue(self.es_registry.is_es_supported(ServiceType.QR, ESType.RESOURCE_SCALE))
        self.assertFalse(self.es_registry.is_es_supported(ServiceType.QR, ESType.MODEL_SCALE))
        self.assertFalse(self.es_registry.is_es_supported(ServiceType.QR, ESType.OFFLOADING))

        self.assertFalse(self.es_registry.is_es_supported(ServiceType.CV, ESType.QUALITY_SCALE))

    def test_get_supported_es_for_s(self):
        self.assertListEqual([], self.es_registry.get_supported_ES_for_service(ServiceType.CV))
        self.assertListEqual([ESType.QUALITY_SCALE, ESType.RESOURCE_SCALE],
                             self.es_registry.get_supported_ES_for_service(ServiceType.QR))

    def test_get_es_cooldown(self):
        self.assertEqual(1500, self.es_registry.get_es_cooldown(ServiceType.QR, ESType.QUALITY_SCALE))
        self.assertEqual(2000, self.es_registry.get_es_cooldown(ServiceType.QR, ESType.RESOURCE_SCALE))

        self.assertEqual(3500, self.es_registry.get_es_cooldown(ServiceType.CV, ESType.STARTUP))
        self.assertEqual(0, self.es_registry.get_es_cooldown(ServiceType.CV, ESType.RESOURCE_SCALE))

    def test_get_parameter_bounds_for_active_es(self):
        t_param_dict_qr = {ESType.QUALITY_SCALE: {'quality': {'min': 360, 'max': 1080}},
                           ESType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 8}}}
        self.assertDictEqual(t_param_dict_qr, self.es_registry.get_parameter_bounds_for_active_ES(ServiceType.QR))

        t_param_dict_qr = {ESType.QUALITY_SCALE: {'quality': {'min': 360, 'max': 1080}},
                           ESType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 4}}}
        self.assertDictEqual(t_param_dict_qr, self.es_registry.get_parameter_bounds_for_active_ES(ServiceType.QR, 4))

        self.assertDictEqual({}, self.es_registry.get_parameter_bounds_for_active_ES(ServiceType.CV))

    def test_get_random_ES_and_params(self):
        random_es, random_params= self.es_registry.get_random_ES_and_params(ServiceType.QR)
        self.assertTrue(random_es == ESType.QUALITY_SCALE or random_es == ESType.RESOURCE_SCALE)
        self.assertEqual(1, len(random_params))
