import datetime
from typing import Dict

import redis

from agent.es_registry import ServiceID, ServiceType, ESType


class RedisClient:
    def __init__(self, host="localhost", port=6379, db=0):
        self.redis_conn = redis.Redis(host=host, port=port, decode_responses=True, db=db)

    def store_assignment(self, service_id: ServiceID, client_ass: Dict[str, int]):
        key = create_ass_key(service_id)
        self.redis_conn.delete(key) # Need to delete, otherwise the old elements stay there
        if client_ass != {}:
            self.redis_conn.hset(key, mapping=client_ass)

    def get_assignments_for_service(self, service_id: ServiceID) -> Dict[str, int]:
        key = create_ass_key(service_id)
        dict_with_str = self.redis_conn.hgetall(key)
        return {service_id: int(rps) for service_id, rps in dict_with_str.items()}

    def store_cooldown(self, service_id: ServiceID, es_type: ESType, cooldown_ms):
        key = create_cool_key(service_id)
        freeze_until = (datetime.datetime.now() + datetime.timedelta(milliseconds=cooldown_ms)).isoformat()
        self.redis_conn.hset(key, mapping={"ES": es_type.value, "unfreeze": freeze_until})

    def get_cooldown(self, service_id: ServiceID):
        key = create_cool_key(service_id)
        item = self.redis_conn.hgetall(key)
        freeze_until = datetime.datetime.fromisoformat(item['unfreeze'])
        return {"ES": item['ES'], "unfreeze": freeze_until}

    def is_under_cooldown(self, service_id: ServiceID):
        key = create_cool_key(service_id)
        if self.redis_conn.exists(key):
            cooldown = self.get_cooldown(service_id)
            return datetime.datetime.now() < cooldown["unfreeze"]
        else:
            return False


def create_ass_key(service_id: ServiceID):
    return f"a:{service_id.service_type.value}:{service_id.container_id}"


def create_cool_key(service_id: ServiceID):
    return f"f:{service_id.service_type.value}:{service_id.container_id}"


if __name__ == '__main__':
    redis = RedisClient("128.131.172.182")

    cv_local = ServiceID("128.131.172.182", ServiceType.CV, "elastic-workbench-cv-analyzer-1")
    qr_local = ServiceID("128.131.172.182", ServiceType.QR, "elastic-workbench-qr-detector-1")
    # nonsense = ServiceID("172.20", ServiceType.QR, "elastic--qr-detector-1")
    redis.store_assignment(cv_local, {"C_10": 50})
    print(redis.get_assignments_for_service(cv_local))

    redis.redis_conn.close()
    # redis.store_assignment(ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1"),
    #                        {'C_X': 50})
    # print(redis.get_assignments_for_service(
    #     ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-qr-detector-1")))
    # print(datetime.datetime.now())
    # redis.store_cooldown(qr_local, EsType.QUALITY_SCALE, 10000)
    # print(redis.is_under_cooldown(qr_local))
    # print(redis.is_under_cooldown(nonsense))

    # redis.store_cooldown(qr_local, EsType.QUALITY_S, 0)
    # print(redis.is_under_cooldown(qr_local))
    # print(redis.is_under_cooldown(nonsense))
