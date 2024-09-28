import onnxruntime
from PIL import Image
from torchvision import transforms

flatten = True


def load_and_preprocess_image(image_path):
    # Define the same preprocessing as used in training
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Open the image file
    img = Image.open(image_path)

    # Preprocess the image
    img_preprocessed = preprocess(img)

    # Add batch dimension
    return img_preprocessed.unsqueeze(0).numpy()


# Path to your exported ONNX file
onnx_model_path = "resnet50_embeddings.onnx"

# Create an ONNX Runtime session
session = onnxruntime.InferenceSession(onnx_model_path)

# Get the input name of the model
input_name = session.get_inputs()[0].name

# Load and preprocess an image (replace with your image path)
image_path = "data/images/1975706280196685130.jpg"
input_data = load_and_preprocess_image(image_path)

# Run inference
outputs = session.run(None, {input_name: input_data})

# The output should be a single tensor (the embeddings)
embeddings = outputs[0]

# Print the original shape of the embeddings
print(f"Original embeddings shape: {embeddings.shape}")

# Flatten the embeddings
if flatten:
    expected_shape = (1, 2048)

    embeddings = embeddings.reshape(embeddings.shape[0], -1)

    # Print the shape of the flattened embeddings
    print(f"Flattened embeddings shape: {embeddings.shape}")
else:
    expected_shape = (1, 2048, 1, 1)


if embeddings.shape == expected_shape:
    print("ONNX model exported and loaded successfully!")
else:
    print(f"Unexpected output shape. Expected {expected_shape}, got {embeddings.shape}")
