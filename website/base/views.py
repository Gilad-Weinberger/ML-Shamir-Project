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

# Neural network model for MNIST classification
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 output classes for digits 0-9
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

# Function to train the model on the MNIST dataset
def train_mnist_model(device):
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = MNIST(root=os.path.join('data'), train=True, download=True, transform=transform)
    mnist_valid = MNIST(root=os.path.join('data'), train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
    valid_loader = DataLoader(mnist_valid, batch_size=32, shuffle=False)

    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 4  # Adjust as needed
    train_losses, val_losses = [], []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Initialize tqdm progress bar
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                pbar.update(1)  # Update progress bar

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
        
        val_loss = running_loss / len(valid_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return model

# View to handle image upload and evaluation
def Home(request):
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
            pil_image = Image.open(image_path).convert('L')
            pil_image = pil_image.resize((28, 28))
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
                predicted_class = output.argmax(dim=1).item()
            prediction = predicted_class
    else:
        form = ImageUploadForm()

    context = {
        'form': form,
        'prediction': prediction,
        'uploaded_image_url': uploaded_image_url,
    }
    return render(request, 'base/home.html', context)
