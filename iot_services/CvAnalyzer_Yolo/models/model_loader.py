import torch
from ultralytics import YOLO

for size in ["n", "s", "m", "l", "x"]:
    model = YOLO(f"yolov10{size}.pt")  # load a pretrained model
    # torch.save(model, f"yolov10{size}.pt")
    # model.export(format="torchscript")
    model.export(format="onnx", dynamic=True)  # export the model to ONNX format

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