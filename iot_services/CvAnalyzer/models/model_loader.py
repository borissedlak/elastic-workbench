from ultralytics import YOLO

for size in ["n", "s", "m", "l", "x"]:
    model = YOLO(f"yolov10{size}.pt")  # load a pretrained model
    path = model.export(format="onnx")  # export the model to ONNX format
