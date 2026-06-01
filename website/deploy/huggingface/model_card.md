---
license: mit
tags:
  - image-regression
  - agriculture
  - grape-leaves
  - downy-mildew
library_name: pytorch
---

# Grape Leaf Downy Mildew Regressor (5deg variant)

Convolutional neural network that estimates the **percentage of downy mildew** on grape leaf images (0–100%).

## Model details

| Property | Value |
|----------|-------|
| Task | Image regression |
| Variant | `5deg` (5-degree rotation augmentation) |
| Input | RGB image, resized to 224×224 |
| Output | Single scalar (affected area %) |
| Architecture | 3-layer CNN + 2-layer MLP regressor |

## Preprocessing

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

No ImageNet normalization is applied.

## Usage (local PyTorch)

```python
import torch
from PIL import Image
from torchvision import transforms

# Copy model.py from this repo or from the Django app ml/model.py
from model import GrapeLeafRegressor

model = GrapeLeafRegressor()
model.load_state_dict(torch.load("grape_leaf_model_5deg.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open("leaf.jpg").convert("RGB")
tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    prediction = model(tensor).item()

prediction = max(0, min(100, prediction))
print(f"Estimated downy mildew: {prediction:.1f}%")
```

## Inference API (HF Space)

Deploy the companion Space from `deploy/huggingface/space/` for a Gradio HTTP API used by the Vercel-hosted Django app.

## Training

- Dataset: `data/final_images_5deg` (train / validation / test split)
- Loss: MSE, Optimizer: Adam (lr=0.001)
- Early stopping on validation loss (patience 5)
- Train locally or in Google Colab: `python train_model.py --variant 5deg`

## Limitations

- Trained on augmented lab/field leaf images; performance may drop on very different lighting or species.
- Filename-based labels during training; not suitable for unlabeled arbitrary images without retraining.

## Citation

Machine Learning Project — Shamir Institute (Dr. Lior Gur), with Daniel Kusai.
