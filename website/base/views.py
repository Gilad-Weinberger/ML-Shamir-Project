import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

from django.apps import apps
from django.shortcuts import render
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .forms import ImageUploadForm
from .inference import is_remote_inference_configured, predict
from .supabase_storage import is_supabase_configured, upload_image

WEBSITE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WEBSITE_ROOT not in sys.path:
    sys.path.insert(0, WEBSITE_ROOT)


def visualize_predictions(model, dataloader, device, num_images=5):
    """
    Visualize the model's predictions on a batch of images.
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
                plt.axis("off")
                images_shown += 1

            if images_shown >= num_images:
                break
    plt.show()


def evaluate_model_performance(model, dataloader, device, threshold=5.0, output_dir="."):
    """
    Evaluates the model on test data and prints detailed accuracy metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    predictions = []
    targets = []

    total_batches = len(dataloader)
    total_images = len(dataloader.dataset)
    print(f"\n--- Starting Evaluation (Acceptable Margin: ±{threshold}%) ---", flush=True)
    print(f"Device: {device} | Images: {total_images} | Batches: {total_batches} (batch size {dataloader.batch_size})", flush=True)
    print("Running inference...", flush=True)

    inference_start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (images, true_percents) in enumerate(dataloader, start=1):
            batch_start = time.perf_counter()
            images = images.to(device)
            outputs = model(images).cpu().squeeze()

            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)

            predictions.extend(outputs.numpy())
            targets.extend(true_percents.numpy())

            images_done = len(predictions)
            batch_elapsed = time.perf_counter() - batch_start
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
                elapsed = time.perf_counter() - inference_start
                rate = images_done / elapsed if elapsed > 0 else 0
                print(
                    f"  Batch {batch_idx}/{total_batches} — {images_done}/{total_images} images "
                    f"({batch_elapsed:.2f}s this batch, {rate:.1f} img/s overall)",
                    flush=True,
                )

    inference_elapsed = time.perf_counter() - inference_start
    print(f"Inference finished in {inference_elapsed:.2f}s ({total_images / inference_elapsed:.1f} img/s)", flush=True)

    predictions = np.array(predictions)
    targets = np.array(targets)
    predictions = np.clip(predictions, 0, 100)

    absolute_errors = np.abs(predictions - targets)
    mae = np.mean(absolute_errors)

    n = len(targets)
    accurate_count = np.sum(absolute_errors <= threshold)
    accuracy = (accurate_count / n) * 100.0 if n > 0 else 0

    print(f"\nResults on {n} Test Images:", flush=True)
    if n < 10:
        print("  (With so few test images, accuracy % is unstable; consider adding more test data.)", flush=True)
    print(f"Mean Absolute Error (MAE): {mae:.2f}% (average error per leaf)", flush=True)
    print(f"Accuracy (Within ±{threshold}%): {accuracy:.2f}% ({accurate_count}/{n} images)", flush=True)
    accurate_7 = np.sum(absolute_errors <= 7.0)
    accuracy_7 = (accurate_7 / n) * 100.0 if n > 0 else 0
    print(f"Accuracy (Within ±7%): {accuracy_7:.2f}% ({accurate_7}/{n} images)", flush=True)
    for margin in (5.0, 10.0, 15.0):
        if margin != threshold:
            count = np.sum(absolute_errors <= margin)
            pct = (count / n) * 100.0 if n > 0 else 0
            print(f"  Within ±{margin:.0f}%: {pct:.1f}% ({count}/{n})", flush=True)
    print("\nPer-image: True % → Predicted % (error):", flush=True)
    if n > 50:
        print(f"  (showing first 10 and last 5 of {n} images; full list omitted for large sets)", flush=True)
        indices = list(range(10)) + list(range(max(10, n - 5), n))
    else:
        indices = range(n)
    for i in indices:
        err = absolute_errors[i]
        mark = "✓" if err <= threshold else "✗"
        print(f"  {mark}  {targets[i]:.1f}% → {predictions[i]:.1f}% (error {err:.1f}%)", flush=True)

    chart_dpi = 150
    print("\nGenerating evaluation charts...", flush=True)
    title_font_size = 22
    label_font_size = 18
    tick_font_size = 14
    legend_font_size = 14
    point_label_font_size = 15

    plt.figure(figsize=(14, 9), dpi=chart_dpi)
    plt.scatter(targets, predictions, alpha=0.6, color="blue", label="Predictions", s=70)
    plt.plot([0, 100], [0, 100], "r--", linewidth=2.5, label="Perfect Prediction")
    x_plot = np.linspace(0, 100, 2)
    plt.plot(x_plot, np.clip(x_plot + 7, 0, 100), "g--", alpha=0.8, linewidth=2.5, label="±7% margin")
    plt.plot(x_plot, np.clip(x_plot - 7, 0, 100), "g--", alpha=0.8, linewidth=2.5, label="_nolegend_")
    plt.xlabel("True White Percentage", fontsize=label_font_size)
    plt.ylabel("Predicted White Percentage", fontsize=label_font_size)
    plt.title(f"Model Performance (Accuracy within ±7%: {accuracy_7:.2f}%)", fontsize=title_font_size, pad=18)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.grid(True, alpha=0.3)
    results_chart_path = os.path.join(output_dir, "evaluation_results.png")
    plt.savefig(results_chart_path, dpi=chart_dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter chart: '{results_chart_path}'", flush=True)

    margins = np.arange(0, 9, dtype=float)
    accuracies_at_margin = [
        (np.sum(absolute_errors <= m) / n) * 100.0 if n > 0 else 0.0
        for m in margins
    ]
    plt.figure(figsize=(14, 9), dpi=chart_dpi)
    plt.plot(margins, accuracies_at_margin, color="blue", linewidth=3, zorder=2)
    plt.scatter(
        margins,
        accuracies_at_margin,
        color="blue",
        edgecolors="white",
        linewidths=2,
        s=150,
        zorder=3,
    )
    for x, y in zip(margins, accuracies_at_margin):
        label = f"{int(round(y))}%" if y == int(y) else f"{y:.1f}%"
        plt.text(x, min(y + 3, 102), label, ha="center", va="bottom", fontsize=point_label_font_size)
    plt.xlabel("Acceptable margin (±%)", fontsize=label_font_size)
    plt.ylabel("Accuracy (%)", fontsize=label_font_size)
    plt.title("Test accuracy vs acceptable error margin (±0% to ±8%)", fontsize=title_font_size, pad=18)
    plt.xticks(margins, fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.ylim(-5, 110)
    plt.grid(True, alpha=0.3)
    margin_chart_path = os.path.join(output_dir, "evaluation_accuracy_by_margin.png")
    plt.savefig(margin_chart_path, dpi=chart_dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved margin chart: '{margin_chart_path}'", flush=True)
    print("Evaluation complete.", flush=True)

    return mae, accuracy


def Home(request):
    """
    Django view to handle image upload and evaluate the model on the uploaded image.
    """
    prediction = None
    uploaded_image_url = None
    uploaded_image_name = None
    model_error = None

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES["image"]
            image_bytes = uploaded_file.read()
            uploaded_image_name = uploaded_file.name

            if not is_supabase_configured():
                model_error = (
                    "Supabase Storage is not configured. Set SUPABASE_URL, "
                    "SUPABASE_SERVICE_ROLE_KEY, and SUPABASE_STORAGE_BUCKET "
                    "(see website/deploy/vercel/GUIDE.md)."
                )
            else:
                try:
                    buffer = BytesIO(image_bytes)
                    buffer.name = uploaded_file.name
                    buffer.content_type = uploaded_file.content_type or "image/jpeg"
                    uploaded_image_url, _ = upload_image(buffer, uploaded_file.name)
                except Exception as exc:
                    model_error = f"Failed to upload image to Supabase: {exc}"

            if model_error is None:
                app_config = apps.get_app_config("base")
                model = app_config.model

                if model is None and not is_remote_inference_configured():
                    model_error = (
                        "No trained model is loaded. Train with "
                        "'python train_model.py --variant 5deg', place the .pth file "
                        "in website/base/, set HF_MODEL_REPO to download from Hugging Face, "
                        "or set HF_INFERENCE_URL to use a Hugging Face Space."
                    )
                else:
                    try:
                        prediction = predict(image_bytes, model=model)
                    except Exception as exc:
                        model_error = f"Prediction failed: {exc}"
    else:
        form = ImageUploadForm()

    context = {
        "form": form,
        "prediction": prediction,
        "uploaded_image_url": uploaded_image_url,
        "uploaded_image_name": uploaded_image_name,
        "model_error": model_error,
    }
    return render(request, "base/home.html", context)
