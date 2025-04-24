from unittest import TestCase

from agent.ES_Registry import ES_Registry, EsType, ServiceType


class TestES_Registry(TestCase):

    def setUp(self):
        self.es_registry = ES_Registry('./static/es_registry.json')

    def test_is_es_supported(self):
        self.assertTrue(self.es_registry.is_ES_supported(ServiceType.QR, EsType.QUALITY_SCALE))
        self.assertTrue(self.es_registry.is_ES_supported(ServiceType.QR, EsType.RESOURCE_SCALE))
        self.assertFalse(self.es_registry.is_ES_supported(ServiceType.QR, EsType.MODEL_SCALE))

        self.assertFalse(self.es_registry.is_ES_supported(ServiceType.CV, EsType.QUALITY_SCALE))
        self.assertTrue(self.es_registry.is_ES_supported(ServiceType.CV, EsType.RESOURCE_SCALE))
        self.assertTrue(self.es_registry.is_ES_supported(ServiceType.CV, EsType.MODEL_SCALE))
        self.assertFalse(self.es_registry.is_ES_supported(ServiceType.CV, EsType.OFFLOADING))
