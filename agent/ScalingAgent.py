import logging
import platform
import time
from threading import Thread
from typing import Dict

import utils
from DockerClient import DockerClient
from HttpClient import HttpClient
from PrometheusClient import PrometheusClient
from RedisClient import RedisClient
from agent import agent_utils, PolicySolver
from agent.ES_Registry import ES_Registry, ServiceID, ServiceType, EsType
from agent.LGBN import LGBN
from agent.SLO_Registry import SLO_Registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multiscale")

PHYSICAL_CORES = int(utils.get_env_param('MAX_CORES', 8))


class ScalingAgent(Thread):
    def __init__(self, prom_server, services_monitored: [ServiceID], evaluation_cycle):
        super().__init__()
        self._running = True
        self._idle = False
        self.evaluation_cycle = evaluation_cycle

        self.services_monitored = services_monitored
        self.prom_client = PrometheusClient(prom_server)
        self.docker_client = DockerClient()
        self.http_client = HttpClient()
        self.es_registry = ES_Registry()
        self.reddis_client = RedisClient()
        self.slo_registry = SLO_Registry()
        self.lgbn = LGBN()

    def resolve_service_state(self, service_id: ServiceID, assigned_clients: Dict[str, int]):
        metric_values = self.prom_client.get_metrics(["avg_p_latency", "throughput"], service_id, period="10s")
        parameter_ass = self.prom_client.get_metrics(["quality", "cores"], service_id)

        target_throughput = utils.to_absolut_rps(assigned_clients)
        completion_rate = metric_values['throughput'] / target_throughput
        return metric_values | parameter_ass | {"completion_rate": completion_rate}

    # WRITE: Add a high-level algorithm of this to the paper
    def run(self):
        while self._running:

            cores_ass = self.get_assigned_cores(self.services_monitored)
            for service_m in self.services_monitored:  # For all monitored services

                service_m: ServiceID = service_m
                assigned_clients = self.reddis_client.get_assignments_for_service(service_m)
                service_state = self.resolve_service_state(service_m, assigned_clients)

                if service_state == {}:
                    logger.warning(f"Cannot find state for service {service_m}")
                    continue

                logger.info(f"Current state for <{service_m.host},{service_m.container_id}>: {service_state}")
                all_client_SLO_F = self.get_clients_SLO_F(service_m, service_state, assigned_clients)

                # TODO: According to this discrepancy, I must adjust the model, or is this done automatically?
                # service_state_exp = self.lgbn.get_expected_state(agent_utils.to_partial(service_state), assigned_clients)
                # client_SLO_F_exp = self.slo_registry.calculate_slo_fulfillment(service_state_exp, client_SLOs)
                # print("Expected SLO-F", client_SLO_F_exp)

                host_fix = "localhost" if platform.system() == "Windows" else service_m.host
                if self.reddis_client.is_under_cooldown(service_m):
                    warning_msg = f"Service <{service_m.host, service_m.container_id}> under cooldown, cannot call ES"
                    logger.warning(warning_msg)
                    continue

                if True:  # random.randint(1, 2) == 1:
                    all_elastic_params = self.get_optimal_local_ES(service_m, assigned_clients, cores_ass)
                    for es_type in self.es_registry.get_active_ES_for_s(service_m.service_type):
                        self.execute_ES(host_fix, service_m.service_type, es_type, all_elastic_params)
                else:
                    self.execute_random_ES(host_fix, service_m.service_type)

            time.sleep(self.evaluation_cycle)

    def get_clients_SLO_F(self, service_m: ServiceID, service_state, assigned_clients):

        all_client_SLO_F = []
        for client_id, client_rps in assigned_clients.items():  # Check the SLO-F of their clients

            client_SLOs = self.slo_registry.get_SLOs_for_client(client_id, service_m.service_type)
            if client_SLOs == {}:
                logger.warning(f"Cannot find SLOs for service {service_m}, client {client_id}")
                continue

            client_SLO_F_emp = self.slo_registry.calculate_slo_fulfillment(service_state, client_SLOs)
            all_client_SLO_F.append((client_id, client_SLO_F_emp))

        print("Actual SLO-F", all_client_SLO_F)
        return all_client_SLO_F

    def execute_random_ES(self, host, service_type: ServiceType):
        rand_ES = self.es_registry.get_random_ES_for_service(service_type)

        if not self.es_registry.is_ES_supported(service_type, rand_ES):
            logger.warning(f"Trying to call unsupported ES for {service_type}, {rand_ES}")
            return

        ES_endpoints = self.es_registry.get_ES_information(service_type, rand_ES)['endpoints']
        for endpoint in ES_endpoints:
            random_params = agent_utils.get_random_parameter_assignments(endpoint['parameters'])
            self.http_client.call_ES_endpoint(host, endpoint['target'], random_params)
            logger.info(f"Calling random ES <{service_type},{rand_ES}> with {random_params}")

    def execute_ES(self, host, service_type: ServiceType, es_type: EsType, all_params):

        if not self.es_registry.is_ES_supported(service_type, es_type):
            logger.warning(f"Trying to call unsupported ES for {service_type}, {es_type}")
            return

        ES_endpoints = self.es_registry.get_ES_information(service_type, es_type)['endpoints']
        for endpoint in ES_endpoints:
            # specific_params = agent_utils.get_random_parameter_assignments(endpoint['parameters'])
            self.http_client.call_ES_endpoint(host, endpoint['target'], all_params)
            logger.info(f"Calling ES <{service_type},{es_type}> with {all_params}")

    def get_optimal_local_ES(self, service: ServiceID, assigned_clients: Dict[str, int], cores_ass):

        free_cores = PHYSICAL_CORES - sum(cores_ass.values())
        max_available_c = free_cores + cores_ass[service.container_id]
        ES_parameter_bounds = self.es_registry.get_parameter_bounds_for_active_ES(service.service_type, max_available_c)

        linear_relations = self.lgbn.get_linear_relations()
        all_client_slos = self.slo_registry.get_all_SLOs_for_assigned_clients(service.service_type, assigned_clients)
        total_rps = utils.to_absolut_rps(assigned_clients)

        return PolicySolver.solve(ES_parameter_bounds, linear_relations, all_client_slos, total_rps)

    def get_assigned_cores(self, service_list: [ServiceID]):
        cores_per_service = {}

        for service_id in service_list:
            s_cores = self.docker_client.get_container_cores(service_id.container_id)
            cores_per_service[service_id.container_id] = s_cores
        return cores_per_service


if __name__ == '__main__':
    ps = "http://localhost:9090"
    qr_local = ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-video-processing-1")
    qr_local_2 = ServiceID("172.20.0.6", ServiceType.QR, "elastic-workbench-video-processing-2")
    ScalingAgent(services_monitored=[qr_local, qr_local_2], prom_server=ps, evaluation_cycle=15).start()
