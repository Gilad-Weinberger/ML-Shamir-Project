from django.apps import AppConfig
import torch
import os
from .views import train_grape_leaf_model, GrapeLeafRegressor

class BaseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'base'
    model = None  

    def ready(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join(os.path.dirname(__file__), "grape_leaf_model.pth")

        if os.path.exists(model_path):
            print("Loading pre-trained model...")
            model = GrapeLeafRegressor().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            BaseConfig.model = model
        else:
            print("No pre-trained model found. Training new model...")
            BaseConfig.model, _ = train_grape_leaf_model(device)
            torch.save(BaseConfig.model.state_dict(), model_path)  # Save the trained model
