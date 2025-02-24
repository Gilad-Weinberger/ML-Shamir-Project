from django.apps import AppConfig
import torch
from .views import train_mnist_model

class BaseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'base'
    model = None 

    def ready(self):
        if not BaseConfig.model:  # Check if the model is already trained
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            BaseConfig.model = train_mnist_model(device)
