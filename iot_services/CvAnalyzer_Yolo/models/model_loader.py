from ultralytics import YOLO

for size in ["n", "s", "m", "l", "x"]:
    model = YOLO(f"yolov8{size}.pt")  # load a pretrained model
    # torch.save(model, f"yolov10{size}.pt")
    # model.export(format="torchscript", dynamic=True)
    model.export(format="onnx", dynamic=True)  # export the model to ONNX format
