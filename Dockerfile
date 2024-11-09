FROM python:3.12-slim-bookworm

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install zbar-tools -y

WORKDIR /src
COPY . /src
COPY /data /src

RUN pip install -r ./requirements.txt

EXPOSE 8080
CMD [ "python", "./HttpServer.py" ]