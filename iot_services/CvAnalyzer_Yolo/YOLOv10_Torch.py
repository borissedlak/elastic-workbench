import logging
import cv2
import numpy as np
import torch

import utils
from video_utils import draw_detections

logger = logging.getLogger("multiscale")


class YOLOv10:
    def __init__(self, path: str, conf_threshold: float = 0.2):
        self.conf_threshold = conf_threshold
        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()  # Set the model to evaluation mode

        # Get model info
        self.get_input_details()

    # @utils.print_execution_time
    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.detect_objects(image)

    def detect_objects(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        return self.process_output(outputs[0])

    def prepare_input(self, image: np.ndarray) -> torch.Tensor:
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # h, w = input_img.shape[:2]
        # new_h, new_w = (h // 32) * 32, (w // 32) * 32
        # input_img = cv2.resize(input_img, (new_w, new_h))

        # TODO: If I keep this, changing the original image size will never have any impact
        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        return input_tensor

    def inference(self, input_tensor: torch.Tensor):
        # Perform inference using PyTorch
        with torch.no_grad():
            outputs = self.model(input_tensor)  # Assuming the model returns raw outputs
        return outputs

    def process_output(self, output):
        # output = output.squeeze(0).cpu().numpy()  # shape: (300, 6)
        #
        # boxes = output[:, :4]  # x1, y1, x2, y2
        # confidences = output[:, 4]
        # class_ids = output[:, 5].astype(int)
        #
        # # Filter detections by confidence threshold
        # conf_threshold = 0.25
        # mask = confidences > conf_threshold
        #
        # boxes = boxes[mask]
        # confidences = confidences[mask]
        # class_ids = class_ids[mask]
        #
        # return class_ids, boxes, confidences
        return [], [], []

    # def rescale_boxes(self, boxes):
    #     input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
    #     boxes = np.divide(boxes, input_shape, dtype=np.float32)
    #     boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
    #     return boxes

    def get_input_details(self):
        # Define input dimensions (based on model architecture)
        self.input_height = 640  # Replace with model-specific height if needed
        self.input_width = 640  # Replace with model-specific width if needed


if __name__ == '__main__':
    model_path = "./models/yolov10n.torchscript"  # Change to PyTorch model file

    # Initialize YOLOv10 object detector
    detector = YOLOv10(model_path)

    img = cv2.imread("./data/CV_Image.png")

    # Detect Objects
    class_ids, boxes, confidences = detector(img)

    # Draw detections
    combined_img = draw_detections(img, boxes, confidences, class_ids)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
