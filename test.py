import docker

# Initialize the Docker client
client = docker.DockerClient(base_url='unix:///home/boris/.docker/desktop/docker.sock')

# container = client.containers.get("dd158b94a138")


running_container = client.containers.list()
for name in running_container:
    print(f"Container ID: {name.id}")
    print(f"Container Status: {name.status}")

# # Print container ID and status
# print(f"Container ID: {container.id}")
# print(f"Container Status: {container.status}")

# Optional: Updating resource limits on a running container
# container.update(cpu_quota=25000, mem_limit="256m")  # Adjust limits here if needed

# Stop and remove the container when done
# container.stop()
# container.remove()