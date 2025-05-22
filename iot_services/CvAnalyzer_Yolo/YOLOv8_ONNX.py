import time

import cv2
import numpy as np
import onnxruntime

import video_utils


class YOLOv8:
    def __init__(self, path: str, conf_threshold: float = 0.2):
        self.conf_threshold = conf_threshold
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.detect_objects(image)

    def detect_objects(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # start = time.perf_counter()
        input_tensor = video_utils.prepare_yolo_input(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        # print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return self.process_output(outputs[0], image.shape)

    def process_output(self, output: np.ndarray, original_shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        output = output.squeeze(axis=0)  # shape: [num_detections, 6]
        boxes = output[:, :4]
        scores = output[:, 4]
        class_ids = output[:, 5].astype(int)

        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Rescale boxes to original image size
        h, w = original_shape[:2]
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = boxes.astype(np.int32)

        return class_ids, boxes, scores


if __name__ == '__main__':
    model_path = "./models/yolov8x.onnx"
    detector = YOLOv8(model_path)

    img = cv2.imread("./data/CV_Image.png")
    img = cv2.resize(img, (700, 701))  # match training shape
    class_ids, boxes, confidences = detector(img)
    print(boxes)

    combined_img = video_utils.draw_detections(img, boxes, confidences, class_ids)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
