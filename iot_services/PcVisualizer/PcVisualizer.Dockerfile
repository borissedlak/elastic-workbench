FROM python:3.12-slim-bookworm

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install zbar-tools -y

WORKDIR /src
COPY ./iot_services/PcVisualizer/requirements.txt /src/iot_services/PcVisualizer/
RUN pip install -r ./iot_services/PcVisualizer/requirements.txt

COPY . /src/

ENV SERVICE_TYPE PC

EXPOSE 8080
CMD [ "python", "-m", "iot_services.Service_Wrapper" ]