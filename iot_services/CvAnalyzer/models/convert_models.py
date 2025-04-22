import onnx
from onnx import helper, numpy_helper, ModelProto

def eliminate_unused_initializers(model: ModelProto) -> ModelProto:
    graph = model.graph
    # Find all used initializer names
    used_tensor_names = set()
    for node in graph.node:
        used_tensor_names.update(node.input)

    # Keep only initializers that are used
    used_initializers = [init for init in graph.initializer if init.name in used_tensor_names]
    graph.ClearField("initializer")
    graph.initializer.extend(used_initializers)
    return model

def extract_constants_from_graph_inputs(model: ModelProto) -> ModelProto:
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializer_names]
    graph.ClearField("input")
    graph.input.extend(inputs)
    return model

def optimize_model(input,output):
    # Load the model
    # onnxfile = "your_model.onnx"
    model = onnx.load(input)

    # Apply passes manually
    model = extract_constants_from_graph_inputs(model)
    model = eliminate_unused_initializers(model)

    # Save the optimized model
    onnx.save(model, output)


optimize_model("version-RFB-320.onnx", "version-RFB-320.onnx")
# optimize_model("version-RFB-640.onnx", "version-RFB-640.onnx")