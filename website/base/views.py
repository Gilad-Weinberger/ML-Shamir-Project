import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
from django.apps import apps
from django_tqdm import BaseCommand
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# A simple CNN-based regressor for grape leaf white percentage prediction.
class GrapeLeafRegressor(nn.Module):
    """
    A Convolutional Neural Network (CNN) model for predicting the white percentage
    on grape leaves. The model consists of convolutional layers for feature extraction
    and fully connected layers for regression.
    """
    def __init__(self):
        super(GrapeLeafRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 32 x 112 x 112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 64 x 56 x 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 128 x 28 x 28
        )
        self.regressor = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Single regression output (white percentage)
        )

    """
    Forward pass through the model.
    Args:
        x (torch.Tensor): Input image tensor.
    Returns:
        torch.Tensor: Predicted white percentage.
    """
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# Custom Dataset for grape leaf images stored in the data/final_images folder.
class GrapeLeafDataset(Dataset):
    """
    A custom PyTorch Dataset for loading grape leaf images and their corresponding
    white percentage labels. The labels are extracted from the filenames.
    """
    def __init__(self, images_folder='data/final_images', transform=None):
        """
        Initialize the dataset.
        Args:
            images_folder (str): Path to the folder containing images.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.images_folder = images_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(os.path.join(images_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding label by index.
        Args:
            idx (int): Index of the image.
        Returns:
            tuple: (image, label) where image is a transformed image tensor
                   and label is the white percentage as a tensor.
        """
        image_name = self.image_files[idx]
        # print(image_name)
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
        if self.transform:
            image = self.transform(image)
        # Extract the ground-truth white percentage from the filename.
        try:
            percentage_str = image_name.split('-')[1]
            percentage = float(percentage_str.split('.')[0]) if '.' in percentage_str else float(percentage_str)
        except Exception as e:
            print(f"Error parsing percentage from filename {image_name}: {e}")
            percentage = 0.0  # Fallback if parsing fails
        # Convert the percentage to a torch tensor of type float32.
        percentage_tensor = torch.tensor(percentage, dtype=torch.float32)
        return image, percentage_tensor

# Function to visualize predictions
def visualize_predictions(model, dataloader, device, num_images=5):
    """
    Visualize the model's predictions on a batch of images.
    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
        num_images (int): Number of images to visualize.
    """
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 10))
    with torch.no_grad():
        for images, true_percentages in dataloader:
            images = images.to(device)
            outputs = model(images).cpu().squeeze()
            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                image = images[i].cpu().permute(1, 2, 0).numpy()
                true_percentage = true_percentages[i].item()
                predicted_percentage = outputs[i].item()
                plt.subplot(1, num_images, images_shown + 1)
                plt.imshow(image)
                plt.title(f"True: {true_percentage:.2f}%\nPredicted: {predicted_percentage:.2f}%")
                plt.axis('off')
                images_shown += 1
            if images_shown >= num_images:
                break
    plt.show()

# Training function for the grape leaf model.
def train_grape_leaf_model(device):
    """
    Train the grape leaf regression model.
    Args:
        device (torch.device): Device to train the model on (CPU or GPU).
    Returns:
        tuple: (model, dataloader) where model is the trained model and
               dataloader is the DataLoader for the training dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = GrapeLeafDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Adjust batch size as needed

    model = GrapeLeafRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 15  # Adjust the number of epochs as needed

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device).unsqueeze(1)  # Make shape (batch_size, 1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                pbar.update(1)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return model, dataloader

# View to handle image upload and evaluation
def Home(request):
    """
    Django view to handle image upload and evaluate the model on the uploaded image.
    Args:
        request (HttpRequest): The HTTP request object.
    Returns:
        HttpResponse: Rendered HTML page with the prediction result.
    """
    prediction = None
    uploaded_image_url = None
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_image_url = fs.url(filename)

            # Load and preprocess the uploaded image
            image_path = fs.path(filename)
            pil_image = Image.open(image_path).convert('RGB')
            pil_image = pil_image.resize((224, 224))
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(pil_image).unsqueeze(0)

            # Retrieve the pre-trained model from the app config
            app_config = apps.get_app_config('base')  # Replace with your app's name
            model = app_config.model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_tensor = image_tensor.to(device)

            # Evaluate the model on the uploaded image
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)
                predicted_percentage = output.item()
            prediction = round(predicted_percentage, 1)
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'prediction': prediction,
        'uploaded_image_url': uploaded_image_url,
    }
    return render(request, 'base/home.html', context)
