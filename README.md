# IWAI - Comparison of Agents

This project demonstrated the capability of different agents to solve a complex scaling problem.


## Installation

### Setup basic requirements

Create a new virtual environment and install dependencies

```bash
python3.12 -m venv venv
source venv/bin/activate
python3.12 -m pip install -r ./requirements.txt
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
python3.12 results/IWAI_A1/A1.py
```