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
        # Define the feature extraction layers - a series of convolution and pooling operations
        # that progressively reduce spatial dimensions while increasing feature depth
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Input: RGB image (3 channels)
            nn.ReLU(),  # Applies non-linearity to introduce complex patterns
            nn.MaxPool2d(kernel_size=2),  # 32 x 112 x 112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Increases feature depth
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 64 x 56 x 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Further increases feature depth
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),  # 128 x 28 x 28
        )
        # Regression head that takes the extracted features and predicts a single continuous value
        self.regressor = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),  # Flatten and reduce dimensions
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
        x = self.features(x)  # Extract features using CNN layers
        x = x.view(x.size(0), -1)  # Flatten the feature maps for the fully connected layer
        x = self.regressor(x)  # Produce the final regression prediction
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
        # Find all image files in the specified directory with common image extensions
        self.image_files = [f for f in os.listdir(os.path.join(images_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        # Return the total number of images in the dataset
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
            image = self.transform(image)  # Apply transformations like resize and normalization
            
        # Extract the ground-truth white percentage from the filename.
        # Assumes filename format includes "-X" where X is the percentage
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
    # Set model to evaluation mode to disable dropout and other training-specific behaviors
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 10))  # Create a matplotlib figure for visualization
    
    # Disable gradient calculation for inference (saves memory and computation)
    with torch.no_grad():
        for images, true_percentages in dataloader:
            images = images.to(device)  # Move images to the appropriate device (CPU/GPU)
            outputs = model(images).cpu().squeeze()  # Run model and get predictions
            
            # Loop through each image in the batch
            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                    
                # Convert tensor to numpy array for matplotlib visualization
                image = images[i].cpu().permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC format
                true_percentage = true_percentages[i].item()
                predicted_percentage = outputs[i].item()
                
                # Plot the image with true and predicted percentages as title
                plt.subplot(1, num_images, images_shown + 1)
                plt.imshow(image)
                plt.title(f"True: {true_percentage:.2f}%\nPredicted: {predicted_percentage:.2f}%")
                plt.axis('off')
                images_shown += 1
                
            if images_shown >= num_images:
                break
    plt.show()  # Display the visualization

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
    # Define image transformations for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (standard input size for many models)
        transforms.ToTensor(),  # Convert PIL images to tensors and normalize to [0,1]
    ])
    
    # Create dataset and dataloader for efficient batch processing
    dataset = GrapeLeafDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Shuffle data for better training

    # Initialize model and move it to the appropriate device (CPU/GPU)
    model = GrapeLeafRegressor().to(device)
    
    # Define loss function (Mean Squared Error is standard for regression problems)
    criterion = nn.MSELoss()
    
    # Define optimizer with learning rate (Adam is a popular choice that adapts learning rates)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 30  # Training will run for 30 full passes through the dataset

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        # Use tqdm for a progress bar during training
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, targets in dataloader:
                # Move data to the appropriate device (CPU/GPU)
                images = images.to(device)
                targets = targets.to(device).unsqueeze(1)  # Add dimension to match model output shape

                # Standard training step: zero gradients, forward pass, compute loss, backward pass, update weights
                optimizer.zero_grad()  # Clear previous gradients
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, targets)  # Calculate loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update model parameters
                
                # Accumulate loss for monitoring
                running_loss += loss.item() * images.size(0)
                pbar.update(1)  # Update progress bar

        # Calculate and display average loss for the epoch
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
    
    # Check if form was submitted (POST request)
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_image_url = fs.url(filename)

            # Prepare the image for model input - load, convert to RGB, resize, and transform to tensor
            image_path = fs.path(filename)
            pil_image = Image.open(image_path).convert('RGB')
            pil_image = pil_image.resize((224, 224))  # Match the input size expected by the model
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

            # Retrieve the pre-trained model from the app config where it was stored during app initialization
            app_config = apps.get_app_config('base')  # Replace with your app's name
            model = app_config.model
            
            # Determine if GPU is available, otherwise use CPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_tensor = image_tensor.to(device)

            # Run inference on the uploaded image
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():  # Disable gradient calculation for inference
                output = model(image_tensor)
                predicted_percentage = output.item()
            
            # Ensure prediction is between 0 and 100
            predicted_percentage = max(0, min(100, predicted_percentage))
            prediction = round(predicted_percentage, 1)  # Round to 1 decimal place for display
    else:
        # For GET requests, just display the empty form
        form = ImageUploadForm()

    # Prepare context for template rendering
    context = {
        'form': form,
        'prediction': prediction,
        'uploaded_image_url': uploaded_image_url,
    }
    return render(request, 'base/home.html', context)
