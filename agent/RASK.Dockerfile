FROM python:3.12-slim-bookworm

RUN apt-get update
# Not sure if these are actually needed here
#RUN apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get install zbar-tools -y

WORKDIR /src
COPY ./agent/requirements.txt /src/agent/requirements.txt
RUN pip install -r ./agent/requirements.txt

COPY . /src/

EXPOSE 8080
CMD [ "python", "-m", "agent.RASK_Agent" ]