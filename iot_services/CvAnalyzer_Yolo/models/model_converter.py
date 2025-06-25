import os

from ultralytics import YOLO

for size in ["n", "s", "m", "l", "x"]:
    model_path = f"yolov8{size}.pt"
    model_path_onnx = f"yolov8{size}.onnx"

    if not os.path.exists(model_path_onnx):
        model = YOLO(model_path)  # load a pretrained model
        model.export(format="onnx", dynamic=True)  # export the model to ONNX format
        os.remove(model_path)
