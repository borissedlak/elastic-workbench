import logging

import cv2
import numpy as np
import onnxruntime as ort

from video_utils import predict, draw_detections_simple

logging.getLogger("onnxruntime").setLevel(logging.WARNING)

class FaceDetector:
    def __init__(self, model_path):
        self.face_detector =  ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def detect_faces(self, frame):
        _image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _image = cv2.resize(_image, (640, 480))
        image_mean = np.array([127, 127, 127])
        _image = (_image - image_mean) / 128
        _image = np.transpose(_image, [2, 0, 1])
        _image = np.expand_dims(_image, axis=0)
        _image = _image.astype(np.float32)

        input_name = self.face_detector.get_inputs()[0].name
        confidences, boxes = self.face_detector.run(None, {input_name: _image})
        boxes, labels, probs = predict(frame.shape[1], frame.shape[0], confidences, boxes, 0.7)

        annotated_frame = draw_detections_simple(frame, boxes)
        return annotated_frame


if __name__ == '__main__':
    model_path = "./models/version-RFB-640.onnx"

    # Initialize YOLOv10 object detector
    detector = FaceDetector(model_path)

    img = cv2.imread("./data/pedestrian.jpg")

    # Detect Objects
    frame = detector.detect_faces(img)

    # print(boxes)

    # Draw detections
    # combined_img = draw_detections(img, boxes, confidences, class_ids)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", frame)
    cv2.waitKey(0)
