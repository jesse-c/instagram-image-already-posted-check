import asyncio
import os
from enum import Enum

import matplotlib.pyplot as plt
import mobileclip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm.asyncio import tqdm_asyncio


class Model(Enum):
    MobileCLIP = "mobileclip"
    ResNet50 = "resnet50"


class ResNet50Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode_image(self, image):
        with torch.no_grad():
            return self.model(image).squeeze()


def get_model(model):
    match model:
        case Model.MobileCLIP:
            model, _, preprocess = mobileclip.create_model_and_transforms(
                "mobileclip_s2",
                pretrained=os.getenv("PRETRAINED_MOBILECLIP"),
            )

            def preprocess(image_path):
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0)

                return image_tensor

            def identity(x):
                return x

            def combine(features):
                features = torch.cat(features)
                features = F.normalize(features, p=2, dim=1)
                return features

            return model, preprocess, identity, combine
        case Model.ResNet50:
            model = models.resnet50(pretrained=True)
            model.eval()  # Set to evaluation mode

            # Remove the final classification layer to get embeddings
            model = torch.nn.Sequential(*list(model.children())[:-1])

            # Wrap the model with our custom wrapper
            model = ResNet50Wrapper(model)

            def preprocess(image_path):
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

                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0)

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

        embedding = get_image_embeddings(
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


async def load_dataset(model, folder_path, embeddings_dir):
    model, preprocess, postprocess, combine = get_model(model)

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


def analyse_dataset(features, valid_files, model):
    print(f"\nAnalyzing dataset using {model}...")
    similarity_matrix = torch.mm(features, features.t())

    similarity_matrix_np = similarity_matrix.cpu().detach().numpy()

    np.fill_diagonal(similarity_matrix_np, 0)
    similarities = similarity_matrix_np.flatten()
    similarities = similarities[similarities != 0]

    avg_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)

    print(f"\nDataset Analysis Results for {model}:")
    print(f"Number of images: {len(valid_files)}")
    print(f"Average similarity: {avg_similarity:.4f}")
    print(f"Median similarity: {median_similarity:.4f}")
    print(f"Min similarity: {min_similarity:.4f}")
    print(f"Max similarity: {max_similarity:.4f}")

    most_similar_idx = np.unravel_index(
        np.argmax(similarity_matrix_np), similarity_matrix_np.shape
    )
    least_similar_idx = np.unravel_index(
        np.argmin(similarity_matrix_np), similarity_matrix_np.shape
    )

    print(f"\nMost similar pair:")
    print(
        f"  {valid_files[most_similar_idx[0]]} and {valid_files[most_similar_idx[1]]}"
    )
    print(f"  Similarity: {similarity_matrix_np[most_similar_idx]:.4f}")

    print(f"\nLeast similar pair:")
    print(
        f"  {valid_files[least_similar_idx[0]]} and {valid_files[least_similar_idx[1]]}"
    )
    print(f"  Similarity: {similarity_matrix_np[least_similar_idx]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, edgecolor="black")
    plt.title(f"Distribution of Image Similarities ({model})")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.savefig(f"similarity_distribution_{model}.png")
    print(
        f"\nSimilarity distribution histogram saved as 'similarity_distribution_{model}.png'"
    )

    return {
        "avg_similarity": avg_similarity,
        "median_similarity": median_similarity,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "most_similar_pair": (
            valid_files[most_similar_idx[0]],
            valid_files[most_similar_idx[1]],
            similarity_matrix_np[most_similar_idx],
        ),
        "least_similar_pair": (
            valid_files[least_similar_idx[0]],
            valid_files[least_similar_idx[1]],
            similarity_matrix_np[least_similar_idx],
        ),
    }


async def analyze_model(model):
    embeddings_dir = f"data/embeddings_{model.value}"
    features, valid_files = await load_dataset(
        model,
        "data/images",
        embeddings_dir,
    )
    print(f"Processed {len(valid_files)} images using {model.value}")
    return analyse_dataset(features, valid_files, model.value)


async def compare_new_image(
    model,
    new_image_path,
):
    embeddings_dir = f"data/embeddings_{model.value}"
    features, valid_files = await load_dataset(
        model,
        "data/images",
        embeddings_dir,
    )

    encoder, preprocess, postprocess, combine = get_model(model)

    new_feature = await get_image_embeddings(
        new_image_path,
        preprocess,
        encoder,
        postprocess,
    )

    new_feature = torch.from_numpy(new_feature)
    new_feature = new_feature.unsqueeze(0)

    similarities = torch.mm(new_feature, features.t()).squeeze()
    similarities_np = similarities.cpu().detach().numpy()

    top_k = 5
    top_indices = np.argsort(similarities_np)[::-1][:top_k]

    print(f"\nTop {top_k} similar images to {os.path.basename(new_image_path)}:")
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. {valid_files[idx]} (Similarity: {similarities_np[idx]:.4f})")


async def main():
    results = {}
    for model in Model:
        results[model.value] = await analyze_model(model)

    print("\nComparison of Models:")
    for metric in [
        "avg_similarity",
        "median_similarity",
        "min_similarity",
        "max_similarity",
    ]:
        print(f"\n{metric.capitalize()}:")
        for model, data in results.items():
            print(f"  {model}: {data[metric]:.4f}")

    print("\nMost Similar Pairs:")
    for model, data in results.items():
        print(
            f"  {model}: {data['most_similar_pair'][0]} and {data['most_similar_pair'][1]} (Similarity: {data['most_similar_pair'][2]:.4f})"
        )

    print("\nLeast Similar Pairs:")
    for model, data in results.items():
        print(
            f"  {model}: {data['least_similar_pair'][0]} and {data['least_similar_pair'][1]} (Similarity: {data['least_similar_pair'][2]:.4f})"
        )

    next_folder = "next"
    new_image_files = [
        f for f in os.listdir(next_folder) if f.lower().endswith(".jpeg")
    ]

    for new_image_file in new_image_files:
        new_image_path = os.path.join(next_folder, new_image_file)
        await compare_new_image(
            Model.ResNet50,
            new_image_path,
        )


if __name__ == "__main__":
    asyncio.run(main())
