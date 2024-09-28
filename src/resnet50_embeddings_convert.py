import torch
from torchvision import models

# Load and modify the model as you've done
model = models.resnet50(pretrained=True)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define an example input tensor
dummy_input = torch.randn(
    1, 3, 224, 224
)  # Batch size 1, 3 color channels, 224x224 image

# Define the output file name
output_path = "resnet50_embeddings.onnx"

# Export the model to ONNX format
torch.onnx.export(
    model,  # model being run
    dummy_input,  # model input (or a tuple for multiple inputs)
    output_path,  # where to save the model
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=11,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
)

print(f"Model saved to {output_path}")
