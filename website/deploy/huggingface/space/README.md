---
title: Grape Leaf Downy Mildew Inference
emoji: 🍇
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
python_version: 3.11
---

# Grape Leaf Inference Space

Gradio API for the `5deg` grape leaf regressor. Used by the Django app on Vercel when `HF_INFERENCE_URL` is set.

## Space secrets

| Secret          | Example                                      |
| --------------- | -------------------------------------------- |
| `HF_MODEL_REPO` | `YOUR_USERNAME/grape-leaf-5deg`              |
| `HF_MODEL_FILE` | `grape_leaf_model_5deg.pth`                  |
| `HF_TOKEN`      | (required if model repo or Space is private) |

## API

After the Space is running, set in your Django / Vercel env:

```
HF_INFERENCE_URL=https://YOUR_USERNAME-grape-leaf-inference.hf.space
```

The `/predict` API returns a number, e.g. `12.3`.
