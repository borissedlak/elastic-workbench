from typing import Dict

import redis

from agent.ES_Registry import ServiceID, ServiceType


class RedisClient:
    def __init__(self, host="localhost", port=6379, db=0):
        self.redis_conn = redis.Redis(host=host, port=port, decode_responses=True, db=db)

    def store_assignment(self, service_id: ServiceID, client_ass: Dict[str, int]):
        key = create_ass_key(service_id)
        self.redis_conn.hset(key, mapping=client_ass)

    def get_assignments_for_service(self, service_id: ServiceID) -> Dict[str, int]:
        key = create_ass_key(service_id)
        return self.redis_conn.hgetall(key)

    def store_cooldown(self, service_id: ServiceID, target_timestamp):  # TODO: THis is the only real int
        key = create_cool_key(service_id)
        self.redis_conn.set(key, target_timestamp)


def create_ass_key(service_id: ServiceID):
    return f"a:{service_id.host}:{service_id.service_type.value}:{service_id.container_id}"


def create_cool_key(service_id: ServiceID):
    return f"f:{service_id.host}:{service_id.container_id}"


if __name__ == '__main__':
    redis = RedisClient()
    redis.store_assignment(ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-video-processing-1"),
                           {'C_X': 50})
    print(redis.get_assignments_for_service(ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-video-processing-1")))
