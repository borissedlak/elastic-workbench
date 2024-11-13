import requests


class HttpClient:
    def __init__(self):
        self.PORT = 8080
        self.SESSION = requests.Session()
        self.http_connection = None
        self.CHANGE_THREADS_ROUTE = "/change_threads"
        # self.MODEL_UPLOAD_PATH = "/model/upload"
        # self.MODEL_UPDATE_PATH = "/model/update"
        # self.ASSIGNMENT_UPDATE_PATH = "/update_service_assignment"

        print(f"Opening HTTP Connection on port {self.PORT}")

    def change_threads(self, target_route, number):
        query_params = {"thread_number": number}
        response = self.SESSION.put(f"http://{target_route}:{self.PORT}{self.CHANGE_THREADS_ROUTE}",
                                     params=query_params)
        print(response.content)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

    # def start_service_remotely(self, s_desc, target_route):
    #     query_params = {
    #         "service_description": str(s_desc)
    #     }
    #     response = self.SESSION.post(f"http://{target_route}:{self.PORT}{self.START_SERVICE_PATH}", params=query_params)
    #     response.raise_for_status()  # Raise an exception for non-2xx status codes
    #
    # def push_files_to_member(self, model_names, target_route):
    #     files = []
    #     for index, m in enumerate(model_names):
    #         files.append((f'file{index + 1}', (m, open("models/" + m, 'rb'), 'application/xml')))
    #
    #     # headers = {'Content-Type': 'application/xml'}  # Set the content type
    #     url = f"http://{target_route}:{self.PORT}{self.MODEL_UPLOAD_PATH}"
    #     response = self.SESSION.post(url, files=files)
    #     response.raise_for_status()  # Raise an exception for non-2xx status codes
    #
    # def push_metrics_retrain(self, service_name, df: pd.DataFrame, target_route, asynchronous=False):
    #     csv_string = df.to_csv(index=False)
    #
    #     headers = {'Content-Type': 'text/csv'}
    #     url = f"http://{target_route}:{self.PORT}{self.MODEL_UPDATE_PATH}/{service_name}"
    #     query_params = {"asynchronous": asynchronous}
    #     response = self.SESSION.post(url, data=csv_string, headers=headers, params=query_params)
    #     response.raise_for_status()  # Raise an exception for non-2xx status codes
    #
    # def update_service_assignment(self, s_desc, s_host, target_route):
    #     query_params = {
    #         "service_description": str(s_desc),
    #         "service_host": s_host
    #     }
    #     url = f"http://{target_route}:{self.PORT}{self.ASSIGNMENT_UPDATE_PATH}"
    #     response = self.SESSION.post(url, params=query_params)
    #     response.raise_for_status()  # Raise an exception for non-2xx status codes
