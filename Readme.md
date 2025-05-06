# Project Structure

This project contains a video processing service ([QrDetector.py](../multiScaler/QrDetector.py)) that continuously processed videos stored in [/data](../multiScaler/data). 
The processing service can be scaled in two ways: (1) by changing the quality of the video stream and (2) by increasing the available resources.
To scale resources, the processing service is wrapped in a Docker container. The respective actions are then orchestrated by the agent.