# IWAI - Comparison of Agents

This project demonstrated the capability of different agents to solve a complex scaling problem.


## Installation

### Setup basic requirements

Create a new virtual environment and install dependencies. It was developed and tested with Python3.12

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r ./requirements.txt
```

## Start Experiments

### Start processing environment 

with Docker installed, start all processing services. This requires you to first install the quite heavy `ultralytics`
dependency and then execute the [model_loader.py](iot_services/CvAnalyzer_Yolo/models/model_loader.py) script in the respective
directory. This fetches and converts the models for the CV service. Afterward, you can build and start all containers with

```bash
docker compose up -d
```

### Start scaling agents

start the experiments, which evaluate one scaling agent type after the other

```bash
PYTHONPATH=. python3 results/IWAI_A1/A1.py
```

# Structure of Services

### Main Class

In the folder [iot_services](iot_services), there are three different implementations of one base class: [IoTService.py](iot_services%2FIoTService.py).
This class features central functions, e.g., starting & stopping the service, changing the service configuration, or exporting the processing metrics to 
Prometheus and the [metrics.csv](share%2Fmetrics%2Fmetrics.csv). This of course assumes that the instance of Prometheus is running, in this case, the
metrics can also be displayed in the Grafana dashboard started alongside. 

### Service Instantiations

The project contains the following 3 services:
1. [QrDetector.py](iot_services%2FQrDetector%2FQrDetector.py) scans an input video for QR codes; it can adjust the quality (i.e., video resolution) and resources (i.e., cores)
2. [CvAnalyzer.py](iot_services%2FCvAnalyzer_Yolo%2FCvAnalyzer.py) runs video inference on an input videos; it can adjust the quality (i.e., video resolution), model size, and resources (i.e., cores)
3. [PcVisualizer.py](iot_services%2FPcVisualizer%2FPcVisualizer.py) should render a point cloud object; still not finished though

### Service Wrapper API

To interact with the services and call the functions of the base class, each service is wrapped in a Rest API. Please have a look at
[Service_Wrapper.py](iot_services%2FService_Wrapper.py) to see the different routes that are available. 

### Docker Containerization

The main reason that Services need to be dockerized is to split the resources between them. If a service is scaled vertically (i.e., ``/resource_scaling?cores=4``), the services both adjust
the time that the Service can be scheduled on the CPU, as well as on the service instance. For example, the [QrDetector.py](iot_services%2FQrDetector%2FQrDetector.py) starts more processing
threads according to the currently configured resources. 

# Structure of Agents

### Important Functions

The base class for all agents is `Scaling_Agent.py`, it features multiple functions that are generally useful.
This includes:

* `resolve_service_state()` Getting the current state of a service from Prometheus (i.e., metrics and parameters)
* `execute_ES()` Execute an elasticity strategy on a specific service; the ``ServiceWrapper`` forwards it to Docker and the wrapped `IoTService.py` instance.
* `evaluate_slos_and_buffer()` Evaluate the current SLO fulfillment and log to a buffer; to export them, execute `agent_utils.export_experience_buffer()`
* `get_max_available_cores()` Gives the maximum number of cores accessible for a service

### Main Loop

These functions can then be included in the main loop `Thread.run()`, where a list of tracked `self.services_monitored` are 
subject to the scaling policy of the respective implementation of the scaling agent. For example, in `RRM_Global_Agent.py`,
the implementation of `orchestrate_services_optimally()` uses an algebraic solver to get the best configuration for the 
processing environment. These functions still need to be implemented for the `DAI_Agent.py` (Alireza) and the `AIF_Agent.py` (Daniel).

Agents can be executed from the source path, like:

```bash
PYTHONPATH=. python3 ./iwai/AIF_Agent.py
```

The `DQN_Agent.py` is in an intermediary state: while the `DQN_Trainer.py` correctly trains the Q network needed for scaling, the functions
still need to be included into the `DQN_Agent.py`. Generally, the `DQN_Agent.py` uses a ``gynmasium.env`` for training a scaling policy.
This environment can equally be used for training or testing the other scaling agents.

### Training Environment

The training environment `LGBN_Training_Env.py` is an instantiation of `gynmasium.env`, which means it offers the general interface for interacting with
an environment similar to the runtime one. This includes acting on the environment and receiving a reward according to the subsequent state.
Depending on the `ServiceType`, the environment either supports 5 (`ServiceType.QR`) or 7 (`ServiceType.CV`) actions. Please have a look at
`LGBN_Training_Env.step()` to see how the agent is rewarded for bringing the environment to states that fulfill the SLOs, and how it gets
penalized for exceeding the boundaries.

### SLOs and Parameter Boundaries

The SLOs and boundaries are defined globally for all agents, tests, and experiments in one directory [config](config). If required, we can
include multiple files there to evaluate the system under different configurations. Generally, the idea is to not adjust SLOs or parameter 
thresholds during runtime to save complexity. Also, the `DQN_Agent.py` is only trained for the currently configured SLOs, this means that
we would need to (re-)train it for different thresholds.

# Practical Information

1. For running the agents on the physical environment, it is necessary to run the containers and agents on Ubuntu to access the local Docker IPs (e.g., 172.20.0.5)
2. However, for solely interacting with the training environment it is sufficient to work on Windows (Daniel)


