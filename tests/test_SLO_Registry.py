from unittest import TestCase

from agent.ES_Registry import ServiceType
from agent.SLO_Registry import SLO_Registry


class TestSLO_Registry(TestCase):

    def setUp(self):
        self.slo_registry = SLO_Registry('./static/slo_config.json')

    def test_get_slos_for_client(self):

        slos = self.slo_registry.get_SLOs_for_client("C_1", ServiceType.QR)

    def test_calculate_slo_fulfillment(self):

        slos = self.slo_registry.get_SLOs_for_client("C_1", ServiceType.QR)

        print(self.slo_registry.calculate_slo_fulfillment({"quality": 10}, slos))

    # def test_get_all_slos_for_assigned_clients(self):
    #     self.fail()
    #
    # def test_to_avg_slo_f(self):
    #     self.fail()
