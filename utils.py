import logging
import os
import time


def get_ENV_PARAM(var, DEFAULT) -> str:
    ENV = os.environ.get(var)
    if ENV:
        logging.info(f'Found ENV value for {var}: {ENV}')
    else:
        ENV = DEFAULT
        logging.warning(f"Didn't find ENV value for {var}, default to: {DEFAULT}")
    return ENV


# def get_local_ip():
#     interfaces = netifaces.interfaces()
#     for interface in interfaces:
#         ifaddresses = netifaces.ifaddresses(interface)
#         if netifaces.AF_INET in ifaddresses:
#             for addr in ifaddresses[netifaces.AF_INET]:
#                 ip = addr.get('addr')
#                 if ip and ip.startswith('192.168'):
#                     return ip
#     return None

def print_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000.0
        print(f"{func.__name__} took {execution_time_ms:.0f} ms to execute")
        return result

    return wrapper
