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

with Docker installed, start all processing services

```bash
docker compose up -d
```

### Start scaling agents

start the experiments, which evaluate one scaling agent type after the other

```bash
PYTHONPATH=. python3 results/IWAI_A1/A1.py
```

# Structure of Agents

The base class for all agents is `Scaling_Agent.py`, it features multiple functions that are generally useful.
This includes:

* 

```bash
PYTHONPATH=. python3 ./iwai/AIF_Agent.py
```