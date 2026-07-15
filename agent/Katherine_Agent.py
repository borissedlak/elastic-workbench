from HttpClient import HttpClient
from agent.components.es_registry import ESRegistry, ESType, ServiceID, ServiceType
from agent.k8_Agent import SERVICE_HOST

qr_local = ServiceID(SERVICE_HOST, ServiceType.QR, "elastic-workbench-qr-detector-1", port="8080")
cv_local = ServiceID(SERVICE_HOST, ServiceType.CV, "elastic-workbench-cv-analyzer-1", port="8081")
pc_local = ServiceID(SERVICE_HOST, ServiceType.PC, "elastic-workbench-pc-visualizer-1", port="8082")

http_client = HttpClient()
es_registry_path="../config/es_registry.json"
es_registry = ESRegistry(es_registry_path)

QR_RPS_DEFAULT = 300
http_client.update_service_rps(qr_local, QR_RPS_DEFAULT)

params = {'cores': 3} # TODO: Change parameters here (Look Readme)
ES_endpoint = es_registry.get_es_information(qr_local.service_type, ESType.RESOURCE_SCALE)['endpoint'] # TODO: Change service and ESType here
http_client.call_ES_endpoint(qr_local, ES_endpoint, params)

params = {'data_quality': 100}
ES_endpoint = es_registry.get_es_information(qr_local.service_type, ESType.QUALITY_SCALE)['endpoint']
http_client.call_ES_endpoint(qr_local, ES_endpoint, params)

params = {'parallelism': 5}
ES_endpoint = es_registry.get_es_information(qr_local.service_type, ESType.PARALLELISM_SCALE)['endpoint']
http_client.call_ES_endpoint(qr_local, ES_endpoint, params)
