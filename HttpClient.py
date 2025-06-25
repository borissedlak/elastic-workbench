import threading

import requests

from agent.es_registry import ServiceID


class HttpClient:
    def __init__(self):
        self.SESSION = requests.Session()

    def _call_ES_endpoint(self, service: ServiceID, route, parameter_ass):
        try:
            response = self.SESSION.put(f"http://{service.host}:{service.port}{route}", params=parameter_ass)
            response.raise_for_status()  # Raise an exception for non-2xx status codes
        except requests.RequestException as e:
            print("Request failed:", e)

    def call_ES_endpoint(self, service: ServiceID, route, parameter_ass):
        thread = threading.Thread(target=self._call_ES_endpoint, args=(service, route, parameter_ass))
        thread.start()
