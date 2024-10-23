import asyncio
import concurrent.futures
import io
import os
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, List, Optional, Tuple

import boto3
import numpy as np
import structlog
import torch
import torch.nn.functional as F
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi_structlog import BaseSettingsModel, LogSettings, setup_logger
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

os.environ["LOG__JSON_LOGS"] = "False"


def get_client(
    region_name: str,
    profile_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> Any:
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        return session.client("s3", region_name=region_name)
    elif aws_access_key_id and aws_secret_access_key:
        return boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    else:
        raise ValueError("neither authentication method provided")


class Settings(BaseSettingsModel):
    log: LogSettings


settings = Settings()
logger = structlog.get_logger()


class SimilarImage(BaseModel):
    rank: int
    filename: str
    similarity_percentage: float
    image_base64: str


class PredictionResponse(BaseModel):
    similar_images: List[SimilarImage]


class Source(Enum):
    local = "local"
    remote = "remote"


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

    async def load_dataset(self, source: Source):
        match source:
            case Source.local:
                logger.info("loading local dataset")
                await self.load_dataset_local()
                logger.info("loaded local dataset")
            case Source.remote:
                logger.info("loading remote dataset")
                await self.load_dataset_remote()
                logger.info("loaded remote dataset")

    async def load_dataset_local(self):
        image_dir = "data/images"
        embeddings_dir = "data/embeddings_resnet50"
        os.makedirs(embeddings_dir, exist_ok=True)
        image_files = [
            f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg"))
        ]

        tasks = [
            self.load_and_process_image(
                os.path.join(image_dir, img_file), embeddings_dir
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

    async def load_dataset_remote(self):
        region_name: Optional[str] = os.getenv("REGION_NAME")
        if region_name is None:
            raise ValueError("missing region name")
        bucket_name: Optional[str] = os.getenv("BUCKET_NAME")
        if bucket_name is None:
            raise ValueError("missing bucket name")

        profile_name: Optional[str] = os.getenv("PROFILE_NAME")
        aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")

        if profile_name:
            logger.info("Using AWS profile for authentication")
            s3_client = get_client(region_name, profile_name=profile_name)
        elif aws_access_key_id and aws_secret_access_key:
            logger.info("Using AWS access key and secret for authentication")
            s3_client = get_client(
                region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        else:
            raise ValueError("neither authentication method provided")

        local_image_dir = "data/images"
        local_embeddings_dir = "data/embeddings_resnet50"
        os.makedirs(local_image_dir, exist_ok=True)
        os.makedirs(local_embeddings_dir, exist_ok=True)

        # Download images
        logger.info("Downloading images from S3...")
        self.download_s3_folder(
            s3_client,
            bucket_name,
            "images/",
            local_image_dir,
        )
        logger.info("Downloaded images from S3...")

        # Download embeddings
        logger.info("Downloading embeddings from S3...")
        self.download_s3_folder(
            s3_client,
            bucket_name,
            "embeddings_resnet50/",
            local_embeddings_dir,
        )
        logger.info("Downloaded embeddings from S3...")

        # Load the dataset from local files
        await self.load_dataset_local()

    def download_file(self, args: Tuple[Any, str, str, str]) -> str:
        s3_client, bucket_name, s3_file, local_file = args
        try:
            s3_client.download_file(bucket_name, s3_file, local_file)
            return f"Successfully downloaded {s3_file} to {local_file}"
        except Exception as e:
            result = f"Error downloading {s3_file}: {str(e)}"
            logger.info(result)
            return result

    def upload_candidate_image(self, image_data: bytes, filename: str) -> None:
        region_name: Optional[str] = os.getenv("REGION_NAME")
        if region_name is None:
            raise ValueError("missing region name")
        bucket_name: Optional[str] = os.getenv("BUCKET_NAME")
        if bucket_name is None:
            raise ValueError("missing bucket name")

        profile_name: Optional[str] = os.getenv("PROFILE_NAME")
        aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")

        if profile_name:
            logger.info("Using AWS profile for authentication")
            s3_client = get_client(region_name, profile_name=profile_name)
        elif aws_access_key_id and aws_secret_access_key:
            logger.info("Using AWS access key and secret for authentication")
            s3_client = get_client(
                region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
        else:
            raise ValueError("neither authentication method provided")

        s3_key = f"candidates/{filename}"
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=image_data,
                ContentType="image/jpeg",
            )
            logger.info(f"Uploaded candidate image to s3://{bucket_name}/{s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload candidate image: {str(e)}")
            raise

    def download_s3_folder(
        self, s3_client, bucket_name, s3_folder, local_dir, max_workers: int = 10
    ):
        paginator = s3_client.get_paginator("list_objects_v2")
        download_args = []
        for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
            if "Contents" in result:
                for file in result["Contents"]:
                    s3_file = file["Key"]
                    local_file = os.path.join(local_dir, os.path.basename(s3_file))
                    download_args.append((s3_client, bucket_name, s3_file, local_file))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self.download_file, download_args))

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

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            filename = self.valid_files[idx]
            image_path = os.path.join("data/images", filename)
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

            results.append(
                {
                    "rank": rank,
                    "filename": filename,
                    "similarity_percentage": float(similarities_np[idx]),
                    "image_base64": image_base64,
                }
            )

        return results


# Global variable to store the model
image_similarity_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global image_similarity_model
    image_similarity_model = ImageSimilarityModel()
    source = Source(os.getenv("SOURCE"))
    await image_similarity_model.load_dataset(source)
    yield
    # Clean up the ML models and release the resources
    image_similarity_model = None


app = FastAPI(lifespan=lifespan)

app.mount("/images", StaticFiles(directory="data/images"), name="images")


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)):
    logger.info("Loading image...")
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    similar_images = await image_similarity_model.compare_new_image(img, top_k=5)

    try:
        await asyncio.to_thread(
            image_similarity_model.upload_candidate_image,
            contents,
            image.filename,
        )
    except Exception as e:
        logger.error(f"Failed to upload candidate image: {str(e)}")

    return {"similar_images": similar_images}


if __name__ == "__main__":
    import uvicorn

    setup_logger(settings.log)

    port = int(os.getenv("PORT", 8800))
    host = os.getenv("HOST", "::")  # or IPv4: 0.0.0.0

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
