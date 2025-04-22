import torch
from ultralytics import YOLO

for size in ["n", "s", "m", "l", "x"]:
    model = YOLO(f"yolov10{size}.pt")  # load a pretrained model
    # torch.save(model, f"yolov10{size}.pt")
    model.export(format="torchscript")
    # path = model.export(format="onnx")  # export the model to ONNX format
