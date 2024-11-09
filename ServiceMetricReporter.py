from datetime import datetime


# This might actually run as a detached thread, but I think it facilitates the linking of entries
class ServiceMetricReporter:
    def __init__(self, service):
        self.service = service

    def create_metrics(self, time, fps, pixel=None, mode=None, rate=None):
        metrics = {"target": self.service,
                   "metrics": {"service": self.service, "delta": time, "fps": fps, "timestamp": datetime.now()}}
        if pixel:
            metrics['metrics']['pixel'] = pixel
        if mode:
            metrics['metrics']['mode'] = mode
        if rate:
            metrics['metrics']['rate'] = 1.0 if rate > 1.0 else rate

        return metrics
