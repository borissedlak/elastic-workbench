#!/bin/sh

echo "Uploading Lidar Files"
rsync -av ./PcVisualizer/data root@128.131.172.182:~/development/elastic-workbench/iot_services/PcVisualizer

echo "Uploading Lidar Files"
rsync -av ./CvAnalyzer_Yolo/models root@128.131.172.182:~/development/elastic-workbench/iot_services/CvAnalyzer_Yolo

echo "Upload completed."
