import io
import onnxruntime
from typing import Annotated

from typing import List

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

app = FastAPI()

# Define image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Embedding(BaseModel):
    embedding: List[float]


@app.post("/predict", response_model=Embedding)
async def predict(image: Annotated[bytes, File()]):
    # Read and preprocess the image
    image = Image.open(io.BytesIO(image)).convert("RGB")
    img_preprocessed = preprocess(image)
    img_tensor = img_preprocessed.unsqueeze(0).numpy()

    # Path to your exported ONNX file
    onnx_model_path = "resnet50_embeddings.onnx"

    # Create an ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_model_path)

    # Get the input name of the model
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: img_tensor})

    # The output should be a single tensor (the embeddings)
    embeddings = outputs[0]

    # Print the original shape of the embeddings
    print(f"Original embeddings shape: {embeddings.shape}")

    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    # Print the shape of the flattened embeddings
    print(f"Flattened embeddings shape: {embeddings.shape}")

    # Convert to list for JSON serialization
    # embedding_list = [float(x) for x in embeddings]
    embedding_list = embeddings.flatten().tolist()

    # embedding_list = []

    return {"embedding": embedding_list}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8800)
