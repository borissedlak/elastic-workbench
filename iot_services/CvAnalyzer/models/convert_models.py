import onnx
from onnx import version_converter, helper

# Load the original model
model = onnx.load("old_model.onnx")

# Optionally check what's the current opset version
original_opset = None
for opset in model.opset_import:
    if opset.domain == "":
        original_opset = opset.version
print(f"Original opset: {original_opset}")

# Upgrade the opset version to 19 (or latest you want)
target_opset = 19
converted_model = version_converter.convert_version(model, target_opset)

# Save the upgraded model
onnx.save(converted_model, "upgraded_model.onnx")