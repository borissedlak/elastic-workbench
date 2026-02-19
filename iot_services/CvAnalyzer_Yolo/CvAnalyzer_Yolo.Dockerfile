FROM python:3.12-slim-bookworm

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install zbar-tools -y

WORKDIR /src
COPY ./iot_services/CvAnalyzer_Yolo/requirements.txt /src/iot_services/CvAnalyzer_Yolo/
RUN pip install -r ./iot_services/CvAnalyzer_Yolo/requirements.txt

COPY . /src/

# Uncomment this for directly loading the ultralytics package and the YOLOv10 models
#RUN pip install ultralytics
#RUN python ./iot_services/CvAnalyzer_Yolo/models/model_converter.py

ENV SERVICE_TYPE CV

EXPOSE 8080
CMD [ "python", "-m", "iot_services.Service_Wrapper" ]