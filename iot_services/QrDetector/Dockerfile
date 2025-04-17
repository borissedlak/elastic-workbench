FROM python:3.12-slim-bookworm

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install zbar-tools -y

WORKDIR /src
COPY ./iot_services/QrDetector/requirements.txt /src/iot_services/QrDetector/
RUN pip install -r ./iot_services/QrDetector/requirements.txt
RUN pip install torch~=2.3.0 --index-url https://download.pytorch.org/whl/cpu # TODO: THis is also wrong...

# TODO: Should not copy other service types, but I'm incapable :(
COPY ./iot_services /src/iot_services
COPY . /src/iot_services

ENV SERVICE_TYPE QR

EXPOSE 8080
CMD [ "python", "iot_services/Service_Wrapper.py" ]