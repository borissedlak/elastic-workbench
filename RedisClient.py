import redis

from agent.ES_Registry import ServiceID, ServiceType


class RedisClient:
    def __init__(self, host="localhost", port=6379, db=0):
        self.redis_conn = redis.Redis(host=host, port=port, decode_responses=True, db=db)

    def store_assignment(self, client_id, location_info):  # TODO: This is also a Hash
        key = create_ass_key(client_id)
        self.redis_conn.set(key, location_info)

    # def store_slos(self, client_id, slo_dict):
    #     key = create_slo_key(client_id)
    #     self.redis_conn.hset(key, mapping=slo_dict)

    def store_cooldown(self, service_id: ServiceID, target_timestamp):  # TODO: THis is the only real int
        key = create_cool_key(service_id)
        self.redis_conn.set(key, target_timestamp)

    # def reset_default_slos(self):
    #     self.store_slos("C_1", str[("pixel", ">", 600)])


def create_ass_key(client_id: str):
    return f"c:{client_id}:ass"


# def create_slo_key(client_id: str):
#     return f"c:{client_id}:slos"


def create_cool_key(service_id: ServiceID):
    return f"f:{service_id.host}:{service_id.container_id}"


if __name__ == '__main__':
    redis = RedisClient()
    redis.store_assignment("C_1", str(ServiceID("172.20.0.5", ServiceType.QR, "elastic-workbench-video-processing-1")))
