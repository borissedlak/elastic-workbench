FROM python:3.12-slim-bookworm

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install zbar-tools -y

WORKDIR /src
COPY ./iot_services/CvAnalyzer/requirements.txt /src/iot_services/CvAnalyzer/
RUN pip install -r ./iot_services/CvAnalyzer/requirements.txt

COPY ./iot_services /src/iot_services
COPY . /src/iot_services

ENV SERVICE_TYPE CV

EXPOSE 8080
CMD [ "python", "iot_services/Service_Wrapper.py" ]