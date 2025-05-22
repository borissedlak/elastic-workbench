import cv2
import numpy as np
import onnxruntime

import video_utils
from video_utils import draw_detections


class YOLOv8:
    def __init__(self, path: str, conf_threshold: float = 0.2):
        self.conf_threshold = conf_threshold
        # video_utils.check_model(path)
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name  # e.g., images
        self.input_shape = self.session.get_inputs()[0].shape  # e.g., ['batch', 3, 'height', 'width']

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.detect_objects(image)

    def detect_objects(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # start = time.perf_counter()
        input_tensor = video_utils.prepare_yolo_input(image)

        # Perform inference on the image
        outputs = self.session.run(None, {self.input_name: input_tensor})
        # print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")

        return self.process_output(outputs[0])

    def process_output(self, output):
        output = output.squeeze()
        boxes = output[:, :-2]
        confidences = output[:, -2]
        class_ids = output[:, -1].astype(int)

        mask = confidences > self.conf_threshold
        boxes = boxes[mask, :]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        return class_ids, boxes, confidences


if __name__ == '__main__':
    model_path = "./models/yolov10n.onnx"

    # Initialize YOLOv10 object detector
    detector = YOLOv8(model_path)

    img = cv2.imread("./data/CV_Image.png")
    img = cv2.resize(img, (32 * 50, 32 * 50))

    # Detect Objects
    class_ids, boxes, confidences = detector(img)

    # Draw detections
    combined_img = draw_detections(img, boxes, confidences, class_ids)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
