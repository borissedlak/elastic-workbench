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
        # This image is resized to multiples of 32
        input_tensor = video_utils.prepare_yolo_input(image)

        # Get the actual shape of the tensor we are sending in
        # input_tensor.shape is [1, 3, height, width]
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]

        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Pass both the original shape AND the shape the model actually saw
        return self.process_output(outputs[0], image.shape, (input_h, input_w))

    def process_output(self, output: np.ndarray, original_shape, input_shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        output = output.squeeze(axis=0)
        boxes = output[:, :4]
        scores = output[:, 4]
        class_ids = output[:, 5].astype(int)

        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        h_orig, w_orig = original_shape[:2]
        h_input, w_input = input_shape

        # The ratio between where the model saw the box and the real image size
        x_scale = w_orig / w_input
        y_scale = h_orig / h_input

        boxes[:, [0, 2]] *= x_scale
        boxes[:, [1, 3]] *= y_scale

        return class_ids, boxes.astype(np.int32), scores


if __name__ == '__main__':
    model_path = "./models/yolov10n.onnx"
    detector = YOLOv8(model_path)

    img = cv2.imread("./data/CV_Image_2.png")
    # img = cv2.resize(img, (700, 701))  # match training shape
    class_ids, boxes, confidences = detector(img)
    print(boxes)

    combined_img = video_utils.draw_detections(img, boxes, confidences, class_ids)
    # combined_img = video_utils.draw_detections_simple(img, boxes)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
