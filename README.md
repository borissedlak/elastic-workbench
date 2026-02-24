# Elastic Workbench: Multi-Dimensional Autoscaling for Edge Computing

This repository contains _Elastic Workbench_, a platform designed for the context-aware autoscaling of stream processing
services on resource-constrained Edge devices. Unlike traditional horizontal autoscalers, this workbench enables
_Multi-Dimensional Elasticity_, allowing services to scale not just hardware resources (CPU/RAM) but also
application-specific
parameters (e.g., image resolution, model complexity, or data quality) to ensure Service Level Objectives (SLOs) despite
fluctuating workloads.

The project implements the MUDAP (Multi-Dimensional Autoscaling Platform) architecture, providing a testbed for the
RASK (Regression Analysis of Structural Knowledge) scaling agent to optimize global performance across co-located
services.

For further reference, please check the following publications, where the ideas of this project were developed and
evaluated:

* Sedlak et al., **Multi-Dimensional Autoscaling of Stream Processing Services on Edge Devices** (2025)
  [[ref]](https://arxiv.org/abs/2510.06882)
* Sedlak et al., **Visual Insights into Agentic Optimization of Pervasive Stream Processing Services** (2025)
  [[ref]](https://arxiv.org/abs/2602.17282)

## Installation

### Setup basic requirements

Create a new virtual environment and install dependencies. It was developed and tested with Python3.12

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r ./requirements.txt
```

### Prepare processing services

The processing environment contains three services; while the QR and the PC service
already contain all required data and models, the CV service does not include the YOLOv10 models required
for object detection. Before the CV service can be executed, the models must be loaded and converted through
one of the following way:

1. Install the ultralytics package locally and run the
   script [model_converter.py](iot_services/CvAnalyzer_Yolo/models/model_converter.py)
   so that the YOLOv10 models are finally contained in the [models](iot_services/CvAnalyzer_Yolo/models) directory.
2. Uncommenting the respective lines in
   the [CvAnalyzer_Yolo.Dockerfile](iot_services/CvAnalyzer_Yolo/CvAnalyzer_Yolo.Dockerfile)
   to directly load the ultralytics package and the YOLOv10 models into the image; notice that ultralytics increases the
   size of the Docker image considerable.

With the models in place, the CV service is ready to go.

### Start processing environment

with Docker installed, you can build and start all containers with

```bash
docker compose up -d
```

[//]: # (This starts the following services, which can be accessed from the local machine.)

**Networking**: by default, Services operate on a **fixed subnet** (172.20.0.0/24).

**Mounts**: all IoT services automatically mount [./share](share) for persisting monitoring information,
and /var/run/docker.sock for providing an interface to the Docker execution.

### Service Overview

| Service Name       | Container Name                      | Role                                              | Ports       |
|:-------------------|:------------------------------------|:--------------------------------------------------|:------------|
| **IoT Services**   |                                     |                                                   |             |
| `qr-detector-1`    | `elastic-workbench-qr-detector-1`   | Stream processing service.                        | `8080:8080` |
| `cv-analyzer-1`    | `elastic-workbench-cv-analyzer-1`   | Stream processing service.                        | `8081:8080` |
| `pc-visualizer-1`  | `elastic-workbench-pc-visualizer-1` | Stream processing service.                        | `8082:8080` |
| **Infrastructure** |                                     |                                                   |             |
| `prometheus`       | `prometheus`                        | Time-series DB for storing processing metrics.    | `9090:9090` |
| `grafana`          | `grafana`                           | Dashboard for visualizing real-time metrics.      | `3000:3000` |
| `cadvisor`         | `cadvisor`                          | Access detailed metics from service containers.   | `8090:8080` |
| `redis`            | `redis`                             | Store common configs (e.g., clients per service). | `6379:6379` |

# Structure of Services

### Main Class

In the folder [iot_services](iot_services), there are three different implementations of one base
class: [IoTService.py](iot_services%2FIoTService.py). This class features central functions,
e.g., starting & stopping the service, changing the service configuration, or exporting the processing metrics
to Prometheus and the [metrics.csv](share%2Fmetrics%2Fmetrics.csv). Metrics in Prometheus can be visualized
in Grafana or inspected through the [PrometheusClient.py](PrometheusClient.py);
the [metrics.csv](share%2Fmetrics%2Fmetrics.csv)
simplifies this access if developers quickly want to inspect and process the values.

## Service Instantiations

The workbench features three core stream processing services. Each service is "resource-aware," meaning it can
dynamically adjust its internal logic (e.g., threading) and application parameters in real-time based on the allocated
hardware and configuration.

| Service          | Task Description                                        | Service-Specific Parameters (Quality)                                  | Hardware Parameters (Resources) |
|:-----------------|:--------------------------------------------------------|:-----------------------------------------------------------------------|:--------------------------------|
| **QrDetector**   | Scans video frames for QR codes using OpenCV.           | **Data Quality**: Adjusts the resolution of the input frames.          | **Cores**: Maximum CPU quota.   |
| **CvAnalyzer**   | Performs object detection/classification using YOLOv10. | **Data Quality**: Image resolution.<br>**Model Size**: Swaps DNN size. | **Cores**: Maximum CPU quota.   |
| **PcVisualizer** | Renders LiDAR point clouds from the Kitti dataset.      | **Data Quality**: Adjusts the Lidar detection/rendering radius.        | **Cores**: Maximum CPU quota.   |

## Service Wrapper API

Each service is wrapped in a REST API ([Service_Wrapper.py](iot_services/Service_Wrapper.py)) that exposes endpoints for
scaling the stream processing services without container restarts.

### Resource Scaling (Hardware)

To adjust the physical CPU limits of a container:
`PUT /resource_scaling?cores=4.5`
*This updates the Docker CFS quota and notifies the service to scale its internal worker threads.*

### Application Scaling (Logic)

To adjust application-specific logic:

* **Quality**: `PUT /quality_scaling?quality=200` (Adjusts resolution or range)
* **Model**: `PUT /model_scaling?model_size=3` (Swaps the YOLOv10 model)

### Service Management

To adjust application-specific logic:

* **Start**: `PUT /start_processing` (Starts the processing services, if it was stopped explicitly)
* **Stop**: `PUT /stop_processing` (Stops the )

## Docker Containerization

Services are containerized primarily to enable strict resource isolation. When the scaling agent calls the API, MUDAP
performs two simultaneous actions:

1. **System Level**: It adjusts the `cpu_quota` via the Docker socket to limit physical CPU cycles.
2. **Application Level**: It signals the Python process (e.g., `QrDetector.py`) to increase or decrease its internal
   processing threads to match the new hardware allotment, ensuring optimal efficiency.

# Structure of Agents

### Important Functions

The base class for all agents is `Scaling_Agent.py`, it features multiple functions that are generally useful.
This includes:

* `resolve_service_state()` Getting the current state of a service from Prometheus (i.e., metrics and parameters)
* `execute_ES()` Execute an elasticity strategy on a specific service; the ``ServiceWrapper`` forwards it to Docker and
  the wrapped `IoTService.py` instance.
* `evaluate_slos_and_buffer()` Evaluate the current SLO fulfillment and log to a buffer; to export them, execute
  `agent_utils.export_experience_buffer()`
* `get_max_available_cores()` Gives the maximum number of cores accessible for a service

### Main Loop

These functions can then be included in the main loop `Thread.run()`, where a list of tracked `self.services_monitored`
are
subject to the scaling policy of the respective implementation of the scaling agent. For example, in
`RRM_Global_Agent.py`,
the implementation of `orchestrate_services_optimally()` uses an algebraic solver to get the best configuration for the
processing environment. These functions still need to be implemented for the `DAI_Agent.py` (Alireza) and the
`AIF_Agent.py` (Daniel).

Agents can be executed from the source path, like:

```bash
PYTHONPATH=. python3 ./iwai/AIF_Agent.py
```

The `DQN_Agent.py` is in an intermediary state: while the `DQN_Trainer.py` correctly trains the Q network needed for
scaling, the functions
still need to be included into the `DQN_Agent.py`. Generally, the `DQN_Agent.py` uses a ``gynmasium.env`` for training a
scaling policy.
This environment can equally be used for training or testing the other scaling agents.

### Training Environment

The training environment `LGBN_Training_Env.py` is an instantiation of `gynmasium.env`, which means it offers the
general interface for interacting with
an environment similar to the runtime one. This includes acting on the environment and receiving a reward according to
the subsequent state.
Depending on the `ServiceType`, the environment either supports 5 (`ServiceType.QR`) or 7 (`ServiceType.CV`) actions.
Please have a look at
`LGBN_Training_Env.step()` to see how the agent is rewarded for bringing the environment to states that fulfill the
SLOs, and how it gets
penalized for exceeding the boundaries.

### SLOs and Parameter Boundaries

The SLOs and boundaries are defined globally for all agents, tests, and experiments in one directory [config](config).
If required, we can
include multiple files there to evaluate the system under different configurations. Generally, the idea is to not adjust
SLOs or parameter
thresholds during runtime to save complexity. Also, the `DQN_Agent.py` is only trained for the currently configured
SLOs, this means that
we would need to (re-)train it for different thresholds.

