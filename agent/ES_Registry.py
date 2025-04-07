import json
import logging

logger = logging.getLogger("multiscale")

class ES_Registry:
    _es_activate_default = {'elastic-workbench-video-processing': ['vertical_scaling', 'quality_scaling']}

    def __init__(self):
        self.es_activated = self._es_activate_default
        with open('strategy_registry.json', 'r') as f:
            self.es_api = json.load(f)

    def is_es_supported(self, service_type, es_name):
        strategies = self.es_api.get("strategies", [])
        for strategy in strategies:
            if (
                    strategy.get("service_type") == service_type and
                    strategy.get("ES_name") == es_name
            ):
                if es_name in self.es_activated.get(service_type, []):
                    return True
                else:
                    logger.info(f"Strategy <{service_type},{es_name}> is registered, but not activated")
                    return False

        logger.info("No corresponding strategy registered")
        return False

if __name__ == "__main__":
    es_reg = ES_Registry()
    print(es_reg.is_es_supported("elastic-workbench-video-processing", "vertical_scaling"))
