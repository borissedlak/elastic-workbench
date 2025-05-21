import torch
# from ultralytics import YOLO
#
# for size in ["n", "s", "m", "l", "x"]:
#     model = YOLO(f"yolov10{size}.pt")  # load a pretrained model
#     # torch.save(model, f"yolov10{size}.pt")
#     # model.export(format="torchscript")
#     model.export(format="onnx", dynamic=True)  # export the model to ONNX format

# from ultralytics import YOLO
#
# model = YOLO("yolov10n.pt")  # Automatically downloads and loads the model
# results = model("../data/CV_Image.png")  # or pass a NumPy array or torch.Tensor
#
# # Access detections
# for result in results:
#     print(result.boxes.xyxy)  # Bounding boxes
#     print(result.boxes.conf)  # Confidences
#     print(result.boxes.cls)   # Class IDs

import onnxruntime as ort
import numpy as np
import cv2

# Path to your exported ONNX model
onnx_model_path = "yolov10n.onnx"

# Create ONNX Runtime session
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Get input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g., ['batch', 3, 'height', 'width']
print(f"Model input name: {input_name}")
print(f"Model input shape: {input_shape}")

# Sample input sizes (some valid, some invalid)
sizes = [
    (640, 640),     # ✅ Valid
    (384, 1280),    # ✅ Valid
    (300, 500),     # ❌ Invalid (not divisible by 32)
    (512, 512),     # ✅ Valid
    (640, 641),     # ❌ Invalid (width not divisible by 32)
]

# Create a dummy RGB image and preprocess
def preprocess_image(size):
    h, w = size
    # if h % 32 != 0 or w % 32 != 0:
    #     raise ValueError(f"Invalid input size: {size}. Height and width must be divisible by 32.")

    dummy_image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    image = np.transpose(image, (2, 0, 1))     # HWC to CHW
    image = np.expand_dims(image, axis=0)      # Add batch dimension
    return image

# Run inference on all test sizes
for size in sizes:
    try:
        input_tensor = preprocess_image(size)
        print(f"\nRunning inference on input size: {size}")
        outputs = session.run(None, {input_name: input_tensor})
        print(f"Output shapes: {[o.shape for o in outputs]}")
    except Exception as e:
        print(f"⚠️ Skipped input size {size}: {e}")
