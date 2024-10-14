import asyncio
import io
import os
from typing import Annotated, List

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms
from tqdm.asyncio import tqdm_asyncio


class ResNet50Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode_image(self, image):
        with torch.no_grad():
            return self.model(image).squeeze()


def get_model():
    model = models.resnet50(pretrained=True)
    model.eval()  # Set to evaluation mode

    # Remove the final classification layer to get embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # Wrap the model with our custom wrapper
    model = ResNet50Wrapper(model)

    def preprocess(image_binary):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image_tensor = transform(image_binary).unsqueeze(0)

        return image_tensor

    def postprocess(embedding_tensor):
        return embedding_tensor.squeeze().numpy()

    def combine(features):
        features = [f if f.dim() == 2 else f.unsqueeze(0) for f in features]
        features = torch.cat(features, dim=0)
        features = F.normalize(features, p=2, dim=1)
        return features

    return model, preprocess, postprocess, combine


def get_embedding_filename(image_path):
    return os.path.splitext(os.path.basename(image_path))[0] + ".pt"


async def load_and_process_image(
    model,
    preprocess,
    postprocess,
    image_path,
    embeddings_dir,
):
    try:
        embedding_file = os.path.join(
            embeddings_dir, get_embedding_filename(image_path)
        )

        if os.path.exists(embedding_file):
            return torch.load(embedding_file)

        embedding = await get_image_embeddings(
            image_path,
            preprocess,
            model,
            postprocess,
        )
        torch.save(embedding, embedding_file)
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


async def get_image_embeddings(
    image_path,
    preprocess,
    model,
    postprocess,
):
    image = preprocess(image_path)
    embedding = await asyncio.to_thread(model.encode_image, image)
    embedding = postprocess(embedding)
    return embedding


async def load_dataset(folder_path, embeddings_dir):
    model, preprocess, postprocess, combine = get_model()

    os.makedirs(embeddings_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg"))
    ]

    print("Processing images...")
    tasks = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        tasks.append(
            load_and_process_image(
                model, preprocess, postprocess, img_path, embeddings_dir
            )
        )

    results = await tqdm_asyncio.gather(*tasks)

    features = []
    valid_files = []
    for img_file, feature in zip(image_files, results):
        if feature is not None:
            features.append(feature)
            valid_files.append(img_file)

    features = combine(features)

    return features, valid_files


async def compare_new_image_binary(image, top_k=5):
    image_tensor = preprocess(image)
    embedding = await asyncio.to_thread(model.encode_image, image_tensor)
    new_feature = postprocess(embedding)

    new_feature = torch.from_numpy(new_feature)
    new_feature = new_feature.unsqueeze(0)

    similarities = torch.mm(new_feature, features.t()).squeeze()
    similarities_np = similarities.cpu().detach().numpy()

    top_indices = np.argsort(similarities_np)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        similarity_percentage = similarities_np[idx]
        results.append(
            {
                "rank": rank,
                "filename": valid_files[idx],
                "similarity_percentage": similarity_percentage,
            }
        )

    return results


class SimilarImage(BaseModel):
    rank: int
    filename: str
    similarity_percentage: float


class PredictionResponse(BaseModel):
    similar_images: List[SimilarImage]


app = FastAPI()

model, preprocess, postprocess, combine = get_model()

# Load dataset once at startup
embeddings_dir = "data/embeddings_resnet50"
features, valid_files = asyncio.run(load_dataset("data/images", embeddings_dir))


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: Annotated[bytes, File()]):
    image = Image.open(io.BytesIO(image)).convert("RGB")

    similar_images = await compare_new_image_binary(image, top_k=5)

    return {"similar_images": similar_images}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8800)
