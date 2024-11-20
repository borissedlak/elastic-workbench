import requests


class HttpClient:
    def __init__(self):
        self.PORT = 8080
        self.SESSION = requests.Session()
        self.http_connection = None
        self.CHANGE_THREADS_ROUTE = "/change_threads"
        self.CHANGE_CONFIG_ROUTE = "/change_config"

        print(f"Opening HTTP Connection on port {self.PORT}")

    def change_threads(self, target_route, number):
        query_params = {"thread_number": number}
        response = self.SESSION.put(f"http://{target_route}:{self.PORT}{self.CHANGE_THREADS_ROUTE}",
                                    params=query_params)
        print(response.content)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

    def change_config(self, target_route, config):
        query_params = {"service_description": str(config)}
        response = self.SESSION.put(f"http://{target_route}:{self.PORT}{self.CHANGE_CONFIG_ROUTE}",
                                    params=query_params)
        print(response.content)
        response.raise_for_status()  # Raise an exception for non-2xx status codes
