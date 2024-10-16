from contextlib import asynccontextmanager
import asyncio
import io
import os
from typing import List

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi_structlog import LogSettings, setup_logger, BaseSettingsModel
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

os.environ["LOG__JSON_LOGS"] = "False"


class Settings(BaseSettingsModel):
    log: LogSettings


settings = Settings()
logger = structlog.get_logger()


class SimilarImage(BaseModel):
    rank: int
    filename: str
    similarity_percentage: float


class PredictionResponse(BaseModel):
    similar_images: List[SimilarImage]


class ImageSimilarityModel:
    def __init__(self):
        self.model = self._get_model()
        self.preprocess = self._get_preprocess()
        self.postprocess = self._get_postprocess()
        self.combine = self._get_combine()
        self.features = None
        self.valid_files = None

    def _get_model(self):
        model_path = "resnet50_model.pth"
        if not os.path.exists(model_path):
            model = models.resnet50(pretrained=True)
            torch.save(model.state_dict(), model_path)
        else:
            model = models.resnet50()
            model.load_state_dict(torch.load(model_path))
        model.eval()
        model = torch.nn.Sequential(*list(model.children())[:-1])
        return model

    def _get_preprocess(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _get_postprocess(self):
        return lambda embedding_tensor: embedding_tensor.squeeze().numpy()

    def _get_combine(self):
        def combine(features):
            features = [f if f.dim() == 2 else f.unsqueeze(0) for f in features]
            features = torch.cat(features, dim=0)
            return F.normalize(features, p=2, dim=1)

        return combine

    async def load_dataset(self, folder_path: str, embeddings_dir: str):
        os.makedirs(embeddings_dir, exist_ok=True)
        image_files = [
            f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg"))
        ]

        tasks = [
            self.load_and_process_image(
                os.path.join(folder_path, img_file), embeddings_dir
            )
            for img_file in image_files
        ]
        results = await asyncio.gather(*tasks)

        features = []
        valid_files = []
        for img_file, feature in zip(image_files, results):
            if feature is not None:
                features.append(feature)
                valid_files.append(img_file)

        self.features = self.combine(features)
        self.valid_files = valid_files

    async def load_and_process_image(self, image_path: str, embeddings_dir: str):
        embedding_file = os.path.join(
            embeddings_dir, self.get_embedding_filename(image_path)
        )

        if os.path.exists(embedding_file):
            return torch.load(embedding_file)

        try:
            embedding = await self.get_image_embeddings(image_path)
            torch.save(embedding, embedding_file)
            return embedding
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def get_embedding_filename(self, image_path: str):
        return os.path.splitext(os.path.basename(image_path))[0] + ".pt"

    async def get_image_embeddings(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            embedding = await asyncio.to_thread(self.model, image_tensor)
            return self.postprocess(embedding)

    async def compare_new_image(self, image: Image.Image, top_k: int = 5):
        image_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            embedding = await asyncio.to_thread(self.model, image_tensor)
            new_feature = torch.from_numpy(self.postprocess(embedding)).unsqueeze(0)

        similarities = torch.mm(new_feature, self.features.t()).squeeze()
        similarities_np = similarities.cpu().numpy()

        top_indices = np.argsort(similarities_np)[::-1][:top_k]

        return [
            {
                "rank": rank,
                "filename": self.valid_files[idx],
                "similarity_percentage": float(similarities_np[idx]),
            }
            for rank, idx in enumerate(top_indices, start=1)
        ]


# Global variable to store the model
image_similarity_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global image_similarity_model
    image_similarity_model = ImageSimilarityModel()
    await image_similarity_model.load_dataset("data/images", "data/embeddings_resnet50")
    yield
    # Clean up the ML models and release the resources
    image_similarity_model = None


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    logger.info("Loading image...")
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    similar_images = await image_similarity_model.compare_new_image(img, top_k=5)
    return {"similar_images": similar_images}


if __name__ == "__main__":
    import uvicorn

    setup_logger(settings.log)

    port = int(os.getenv("PORT", 8800))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        access_log=False,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
        },
    )
