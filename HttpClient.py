import threading
import requests


class HttpClient:
    def __init__(self):
        self.PORT = 8080
        self.SESSION = requests.Session()

    def _call_ES_endpoint(self, host, route, parameter_ass):
        try:
            response = self.SESSION.put(f"http://{host}:{self.PORT}{route}", params=parameter_ass)
            response.raise_for_status()  # Raise an exception for non-2xx status codes
        except requests.RequestException as e:
            print("Request failed:", e)

    def call_ES_endpoint(self, host, route, parameter_ass):
        thread = threading.Thread(target=self._call_ES_endpoint, args=(host, route, parameter_ass))
        thread.start()
