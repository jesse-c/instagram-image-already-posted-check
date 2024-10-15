import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class OneToManySimilarityModel(nn.Module):
    def __init__(self, num_reference_images, pretrained=True):
        super(OneToManySimilarityModel, self).__init__()
        # Load pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add new layers
        self.fc = nn.Linear(2048, 512)

        # Learnable embeddings for reference images
        self.reference_embeddings = nn.Parameter(torch.randn(num_reference_images, 512))

        # Similarity to percentage conversion
        self.similarity_to_percentage = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features from input image
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Calculate cosine similarity with all reference embeddings
        similarities = F.cosine_similarity(
            x.unsqueeze(1), self.reference_embeddings.unsqueeze(0), dim=2
        )

        # Convert similarities to percentages
        percentages = (
            self.similarity_to_percentage(similarities.unsqueeze(-1)).squeeze(-1) * 100
        )

        return percentages


class SimilarityDataset(Dataset):
    def __init__(self, image_paths, target_percentages, transform=None):
        self.image_paths = image_paths
        # self.target_percentages = target_percentages
        self.target_percentages = torch.tensor(target_percentages, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.target_percentages[idx]


# Define transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def train():
    # Create dataset and dataloader
    folder_path = "data/images"
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg"))
    ]

    num_reference_images = len(image_paths)

    target_percentages = (
        np.random.rand(len(image_paths), num_reference_images) * 100
    )  # Replace with actual target percentages
    dataset = SimilarityDataset(image_paths, target_percentages, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = OneToManySimilarityModel(num_reference_images)
    model.to(torch.float32)  # Ensure model is in float32
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    def train_step(model, optimizer, images, target_percentages):
        model.train()
        optimizer.zero_grad()
        predicted_percentages = model(images)
        loss = criterion(predicted_percentages, target_percentages)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(model, dataloader, optimizer, num_epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(num_epochs):
            print(f"Starting epoch [{epoch+1}/{num_epochs}]")
            total_loss = 0
            for images, targets in dataloader:
                images = images.to(device)
                # targets = targets.to(device)
                targets = targets.to(device).float()  # Ensure targets are float32

                loss = train_step(model, optimizer, images, targets)
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            print(f"Finished epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        print("Training completed.")

    # Run the training
    num_epochs = 10
    train(model, dataloader, optimizer, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), "similarity_model.pth")


def infer():
    folder_path = "data/images"
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg"))
    ]

    num_reference_images = len(image_paths)

    # Initialize your model architecture
    model = OneToManySimilarityModel(num_reference_images)
    # Load the saved state dict
    model.load_state_dict(torch.load("similarity_model.pth"))
    # Set the model to evaluation mode
    model.eval()

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # For inference after training
    def get_similarity_percentages(model, image_path):
        model.eval()
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        device = next(model.parameters()).device  # Get the device of the model
        image = image.to(device)

        with torch.no_grad():
            percentages = model(image)

        return percentages.squeeze(0).cpu().numpy()

    # Example usage of inference
    # test_image_path = "next/DSC01605.jpeg"
    test_image_path = "prev/3219230429193591551_3219230422960777245.jpg"
    similarity_percentages = get_similarity_percentages(model, test_image_path)

    # Print top 5 similar images
    top_5_indices = similarity_percentages.argsort()[-5:][::-1]
    for i, idx in enumerate(top_5_indices, 1):
        print(
            f"Top {i}: Image index {idx}, Image path: {image_paths[idx]}, Similarity: {similarity_percentages[idx]:.2f}%"
        )


if __name__ == "__main__":
    infer()
