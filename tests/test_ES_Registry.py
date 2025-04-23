from unittest import TestCase

from agent.ES_Registry import ES_Registry


class TestES_Registry(TestCase):

    def setUp(self):
        self.es_registry = ES_Registry('./static/es_registry.json')

    def test_get_parameter_bounds_for_active_es(self):
        self.fail()

    def test_is_es_supported(self):
        self.fail()
