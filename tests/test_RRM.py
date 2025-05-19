from unittest import TestCase

import pytest

from agent.ES_Registry import ServiceType
from agent.RRM import RRM


class TestPolicySolver(TestCase):

    def setUp(self):
        pass

    def test_predict_single_sample(self):
        rrm = RRM(show_figures=False)

        sample_base = {'cores': 4, 'quality': 400}
        sample_extra = {'cores': 4, 'quality': 400, 'blabla': 999}
        sample_missing = {'quality': 400}

        # First and second results should be equal
        result1 = rrm.predict_single_sample(ServiceType.QR, 'throughput', sample_base)
        result2 = rrm.predict_single_sample(ServiceType.QR, 'throughput', sample_extra)
        self.assertEqual(result1, result2)

        # Third call should raise a RuntimeWarning due to missing 'cores'
        with pytest.raises(RuntimeWarning):
            rrm.predict_single_sample(ServiceType.QR, 'throughput', sample_missing)
