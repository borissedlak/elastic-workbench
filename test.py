# import utils
# from DockerClient import DockerClient
# from HttpClient import HttpClient
#
# DOCKER_PREFIX = utils.get_ENV_PARAM('DOCKER_PREFIX', "unix://")
# DOCKER_SOCKET_PATH = utils.get_ENV_PARAM('DOCKER_SOCKET_PATH', "/home/boris/.docker/desktop/docker.sock")
#
# docker_client = DockerClient(DOCKER_PREFIX + DOCKER_SOCKET_PATH)
# http_client = HttpClient()
#
# THREADS_AND_CORES = 2
#
# docker_client.update_cpu("a780851d157d",THREADS_AND_CORES)
# http_client.change_threads("localhost", THREADS_AND_CORES)

from datetime import datetime, timedelta

from prometheus_api_client import PrometheusConnect

# Connect to Prometheus
prom = PrometheusConnect(url="http://172.18.0.1:9090", disable_ssl=True)

# Define the metric name and the time range
metric_name = "fps"
end_time = datetime.now()
start_time = end_time - timedelta(seconds=1)

# Fetch the data for the metric within the specified time range
metric_data = prom.get_metric_range_data(
    metric_name=metric_name,
    start_time=start_time,
    end_time=end_time
    # step="15s"  # Adjust the step as needed, e.g., 15 seconds
)

# Print the data
print(metric_data[0]['values'])
