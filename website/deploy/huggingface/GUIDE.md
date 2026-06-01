# Hugging Face Model Upload & Inference Guide

Step-by-step guide to publish the **5deg** grape leaf model to Hugging Face Hub, deploy a Gradio inference Space, and connect it to the Django app.

---

## Prerequisites

- Hugging Face account: [huggingface.co/join](https://huggingface.co/join)
- Trained weights at `website/base/grape_leaf_model_5deg.pth`  
  (train with `python train_model.py --variant 5deg` if missing)
- Hugging Face CLI: `pip install huggingface_hub` then `hf auth login`

### `hf auth login` prompts

1. **Token** — paste a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **Write** access (needed for uploads).
2. **Add token as git credential?** — choose **Yes**.

   Say **Yes** so `git push` / `git clone` to Hugging Face repos (model + Space) works without re-entering the token. The token is stored in your system’s git credential helper — it is **not** committed to this project.

   Say **No** only if you prefer entering the token manually each time you use git with HF.

Verify login:

```bash
hf auth whoami
```

---

## Step 1 — Create a model repository

1. Go to [huggingface.co/new](https://huggingface.co/new)
2. Choose **Model**
3. Name it e.g. `grape-leaf-5deg` → full id `YOUR_USERNAME/grape-leaf-5deg`
4. Set visibility (Public recommended for free Space access)

---

## Step 2 — Upload weights and model card

From the `website/` directory:

```bash
pip install huggingface_hub
hf auth login
python deploy/huggingface/upload_model.py --repo YOUR_USERNAME/grape-leaf-5deg
```

This uploads:

- `grape_leaf_model_5deg.pth`
- `config.json` (model metadata)
- `README.md` (from `model_card.md`)

Verify at `https://huggingface.co/YOUR_USERNAME/grape-leaf-5deg`.

---

## Step 3 — Create the inference Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name: e.g. `grape-leaf-inference`
3. SDK: **Gradio**
4. Create the Space

Upload the files from `website/deploy/huggingface/space/`:

| File               | Purpose                           |
| ------------------ | --------------------------------- |
| `app.py`           | Gradio app + `/predict` API       |
| `model.py`         | `GrapeLeafRegressor` architecture |
| `requirements.txt` | Python dependencies               |
| `README.md`        | Space description                 |

Or clone and push:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/grape-leaf-inference
cp deploy/huggingface/space/* grape-leaf-inference/
cd grape-leaf-inference
git add . && git commit -m "Add inference app" && git push
```

---

## Step 4 — Configure Space secrets

In the Space → **Settings** → **Repository secrets**, add:

| Secret          | Value                                                 |
| --------------- | ----------------------------------------------------- |
| `HF_MODEL_REPO` | `YOUR_USERNAME/grape-leaf-5deg`                       |
| `HF_MODEL_FILE` | `grape_leaf_model_5deg.pth`                           |
| `HF_TOKEN`      | Your HF token (required if model repo is **private**) |

Wait for the Space to build and show **Running**.

---

## Step 5 — Get the inference URL

The Django app uses `HF_INFERENCE_URL` as the **Space base URL** (no trailing path):

```
https://YOUR_USERNAME-grape-leaf-inference.hf.space
```

The app calls the Gradio `/predict` API under that base URL.

Test in the Space UI: upload a leaf image and confirm JSON like `{"prediction": 12.3}`.

---

## Step 6 — Connect the Django app (local)

Copy env template:

```bash
cp deploy/huggingface/.env.example .env
```

Edit `.env`:

```env
HF_MODEL_REPO=YOUR_USERNAME/grape-leaf-5deg
HF_MODEL_FILE=grape_leaf_model_5deg.pth
HF_INFERENCE_URL=https://YOUR_USERNAME-grape-leaf-inference.hf.space
HF_TOKEN=hf_...   # only if repo/Space is private
```

**Local inference modes:**

| Mode             | Config                                   | Behavior                             |
| ---------------- | ---------------------------------------- | ------------------------------------ |
| Local PyTorch    | `.pth` in `base/`, no `HF_INFERENCE_URL` | In-process torch (default dev)       |
| Remote HF        | `HF_INFERENCE_URL` set                   | Calls Space API (same as production) |
| Download weights | `HF_MODEL_REPO` set, no local `.pth`     | Downloads from Hub on startup        |

Run the app:

```bash
pip install -r requirements.txt
py manage.py runserver
```

Upload a leaf image and verify a prediction appears.

---

## Environment variables reference

| Variable           | Required                                 | Description                                |
| ------------------ | ---------------------------------------- | ------------------------------------------ |
| `HF_MODEL_REPO`    | For Hub download                         | Model repo id, e.g. `user/grape-leaf-5deg` |
| `HF_MODEL_FILE`    | No (default `grape_leaf_model_5deg.pth`) | Weights filename in repo                   |
| `HF_INFERENCE_URL` | For remote inference                     | Space base URL                             |
| `HF_TOKEN`         | Private repos/Spaces                     | HF access token                            |

---

## Troubleshooting

| Issue                                     | Fix                                                                                                                     |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `No module named 'audioop'` / `pyaudioop` | Space README must include `python_version: 3.11`. Re-upload `space/README.md` and rebuild.                              |
| `cannot import name 'HfFolder'`           | Pin `huggingface_hub>=0.23.0,<0.26.0` in Space `requirements.txt`. Re-upload and rebuild. |
| `TypeError: argument of type 'bool' is not iterable` | Re-upload latest `app.py` (includes gradio_client schema patch) and pin `gradio-client==1.7.0` in `requirements.txt`. |
| Exit code 0 / Space stops immediately     | Add `demo.launch(server_name="0.0.0.0", server_port=7860)` at the end of `app.py` (Gradio 5 + Blocks). |
| `share=True` / localhost launch error     | Set `GRADIO_SERVER_NAME=0.0.0.0` before importing gradio (included in latest `app.py`). |
| Space build fails                         | Check `requirements.txt` and Space logs                                                                                 |
| Model download 401                        | Add `HF_TOKEN` secret to Space and app                                                                                  |
| Slow first request                        | HF Space cold start (30–60s after idle) — normal on free tier                                                           |
| `No trained model is loaded`              | Set `HF_INFERENCE_URL` **or** place `.pth` in `base/`                                                                   |
| Upload script fails                       | Run `hf auth login`; confirm `.pth` exists                                                                              |

---

## Verification checklist

- [ ] Model repo shows `grape_leaf_model_5deg.pth` and README
- [ ] Space builds and returns predictions in the UI
- [ ] Local Django works with `HF_INFERENCE_URL` set (remote path)
- [ ] Local Django works with local `.pth` and no `HF_INFERENCE_URL`

---

## Next step

Deploy the Django UI to Vercel with Supabase image storage: see [`../vercel/GUIDE.md`](../vercel/GUIDE.md).
