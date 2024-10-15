import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image
import os


class OneToManySimilarityModel(nn.Module):
    def __init__(self, num_reference_images):
        super().__init__()
        # Initialize ResNet-50 equivalent (simplified for this example)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # ... (more layers would be added here to match ResNet-50 architecture)
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(2048, 512)
        self.reference_embeddings = mx.random.normal((num_reference_images, 512))

        self.similarity_to_percentage = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def __call__(self, x):
        x = self.features(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)

        # Calculate cosine similarity
        similarities = mx.cosine_similarity(
            x[:, None, :], self.reference_embeddings[None, :, :], axis=2
        )

        # Convert similarities to percentages
        percentages = (
            self.similarity_to_percentage(similarities[:, :, None]).squeeze(-1) * 100
        )

        return percentages


def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return mx.array(image.transpose(2, 0, 1))


def create_dataset(folder_path):
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg"))
    ]
    num_reference_images = len(image_paths)
    target_percentages = np.random.rand(len(image_paths), num_reference_images) * 100

    images = mx.stack([load_and_preprocess_image(path) for path in image_paths])
    targets = mx.array(target_percentages, dtype=mx.float32)

    return images, targets, num_reference_images


def train_step(model, optimizer, images, target_percentages):
    def loss_fn(model):
        predicted_percentages = model(images)
        return mx.mean((predicted_percentages - target_percentages) ** 2)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train(model, images, targets, optimizer, num_epochs, batch_size):
    num_samples = images.shape[0]

    for epoch in range(num_epochs):
        print(f"Starting epoch [{epoch+1}/{num_epochs}]")
        total_loss = 0

        for i in range(0, num_samples, batch_size):
            batch_images = images[i : i + batch_size]
            batch_targets = targets[i : i + batch_size]

            loss = train_step(model, optimizer, batch_images, batch_targets)
            total_loss += loss

        avg_loss = total_loss / (num_samples // batch_size)
        print(f"Finished epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training completed.")


def main():
    folder_path = "data/images"
    print("creating dataset")
    images, targets, num_reference_images = create_dataset(folder_path)
    print("created dataset")

    model = OneToManySimilarityModel(num_reference_images)
    optimizer = optim.Adam(learning_rate=0.001)

    num_epochs = 10
    batch_size = 32
    train(model, images, targets, optimizer, num_epochs, batch_size)

    # Save the trained model (Note: MLX doesn't have a direct equivalent to torch.save)
    # You might need to implement custom serialization for your model


if __name__ == "__main__":
    main()
