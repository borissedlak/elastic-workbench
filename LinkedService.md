## Get started

1. Install Docker & docker compose locally
2. Clone project from GitHub
3. run [docker-compose_chain.yml](docker-compose_chain.yml) script

This runs all the services and tools in the script, including the three instances of the processing services,
the prometheus and grafana instances, and some other stuff.

When the applications are running, you can also at any time check the metrics output in the 
[metrics.csv](share/metrics/metrics.csv), otherwise, in a custom Grafana dashboard.

You can access each of these services through the IP address indicated in the compose file, so for example,
if you want to access the grafana UI, you can do so through http://localhost:3000. If you want to send an
HTTP request to the second service, you have to send it to http://localhost:8081. In the following, I describe
the type of requests relevant for you to perturbate the services.

## Service Perturbations

If you're wondering about the values you can introduce for the different parameters, have a look at 
[es_registry.json](config/es_registry.json). For example, the linked-service allows values for _cores_ 
between 1 and 8. I would also recommend you to store the requests in Postman; for a future version I should
create an OpenAPI description though.

### Application Queue: Created Images

This affects how many image frames the first service in the chain (i.e., port 8080 as in the docker compose)
tries to process every second. Change this to simulate a drop in received images; only for the first service.

```bash
curl -X PUT "http://localhost:8080/change_rps?client_id=buffer&rps=30"
```

### Application Queue: Delay

TODO: Should affect the number of frames arriving from the service through a coefficient. So for example, by
setting it to 0.5, only half of the images are received by the next service in time.

### Service Quality

Allows you to change the quality of the images processed by the service; notice that this can be any of the 
three services, so adjust the port accordingly. In further consequence, this also affects the throughput,
because smaller images can be processed faster.

TODO: Currently, the image size does not affect the next service. Need to find a solution.

```bash
curl -X PUT "http://localhost:8080/quality_scaling?data_quality=400
```

### Service Resources

Allows you to change the maximum amount of cores allocated to the services. By doing so, the application can
process its batch of items faster or slower. Again, this can by applied to all the services in the chain.

```bash
curl -X PUT "http://localhost:8080/resource_scaling?cores=2.5
```

TODO: Add default values for parameters

## Running Experiments


TODO: Describe a bit more

Example of old script, first install requirements locally and put everything in venv

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r ./requirements.txt
```

Start docker script with something like

```bash
docker compose -f docker-compose_chain.yml up -d
```

Start some experiment

```bash
PYTHONPATH=. python3 experiments/tsc/E1/E1.py
```