import asyncio
import os

import matplotlib.pyplot as plt
import mobileclip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

# Load the model
model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s2",
    pretrained="/Users/jesse/.cache/huggingface/hub/models--pcuenq--MobileCLIP-S2/snapshots/22fec9d0b9b614cb270f3322bbb80bf98a3f9c2f/mobileclip_s2.pt",
)


def get_embedding_filename(image_path):
    return os.path.splitext(os.path.basename(image_path))[0] + ".pt"


async def load_and_process_image(image_path, embeddings_dir):
    try:
        embedding_file = os.path.join(
            embeddings_dir, get_embedding_filename(image_path)
        )

        if os.path.exists(embedding_file):
            return torch.load(embedding_file)

        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
        embedding = await asyncio.to_thread(model.encode_image, image)
        torch.save(embedding, embedding_file)
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


async def load_dataset(folder_path, embeddings_dir):
    os.makedirs(embeddings_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg"))
    ]

    print("Processing images...")
    tasks = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        tasks.append(load_and_process_image(img_path, embeddings_dir))

    results = await tqdm_asyncio.gather(*tasks)

    features = []
    valid_files = []
    for img_file, feature in zip(image_files, results):
        if feature is not None:
            features.append(feature)
            valid_files.append(img_file)

    return features, valid_files


def analyse_dataset(features, valid_files):
    features = torch.cat(features)
    features = F.normalize(features, p=2, dim=1)

    print("Computing similarity matrix...")
    similarity_matrix = torch.mm(features, features.t())

    similarity_matrix_np = similarity_matrix.cpu().detach().numpy()

    np.fill_diagonal(similarity_matrix_np, 0)
    similarities = similarity_matrix_np.flatten()
    similarities = similarities[similarities != 0]

    avg_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)

    print(f"\nDataset Analysis Results:")
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
    plt.title("Distribution of Image Similarities")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.savefig("similarity_distribution.png")
    print("\nSimilarity distribution histogram saved as 'similarity_distribution.png'")

    return similarity_matrix_np, valid_files


async def compare_new_image(new_image_path, features, valid_files):
    new_feature = await load_and_process_image(new_image_path, "data/embeddings")
    if new_feature is None:
        print(f"Error processing new image: {new_image_path}")
        return

    new_feature = F.normalize(new_feature, p=2, dim=1)
    similarities = torch.mm(new_feature, features.t()).squeeze()
    similarities_np = similarities.cpu().detach().numpy()

    top_k = 5
    top_indices = np.argsort(similarities_np)[::-1][:top_k]

    print(f"\nTop {top_k} similar images to {os.path.basename(new_image_path)}:")
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. {valid_files[idx]} (Similarity: {similarities_np[idx]:.4f})")


async def main():
    embeddings_dir = "data/embeddings"
    features, valid_files = await load_dataset("data/images", embeddings_dir)
    print(f"Processed {len(valid_files)} images")
    similarity_matrix_np, valid_files = analyse_dataset(features, valid_files)

    # Example usage of compare_new_image
    new_image_path = "data/images/1165588029506164864.jpg"
    await compare_new_image(new_image_path, torch.cat(features), valid_files)


if __name__ == "__main__":
    asyncio.run(main())
