from unittest import TestCase

import pandas as pd

from agent.components.RASK import RASK
from agent.components.es_registry import ESType, ServiceType
from agent.components.PolicySolverRASK import solve_global
from agent.components.SLORegistry import SLO

MAX_CORES = 8

class TestPolicySolver(TestCase):

    def setUp(self):
        self.rask = RASK()
        df = pd.read_csv("static/metrics_20_0.csv")
        self.rask.init_models(df)


    def test_solve_global_qr(self):
        parameter_bounds_qr = {ESType.RESOURCE_SCALE: {'cores': {'min': 1, 'max': 8}},
                            ESType.QUALITY_SCALE: {'data_quality': {'min': 100, 'max': 1000}}}

        clients_slos_qr = [{'completion_rate': SLO(var='completion_rate', larger=True, target=1.0, weight=1.0),
                         'data_quality': SLO(var='data_quality', larger=True, target=900, weight=0.5)}]

        service_context = [(ServiceType.QR, parameter_bounds_qr, clients_slos_qr, 100)]

        print(solve_global(service_context, MAX_CORES, self.rask, None))
