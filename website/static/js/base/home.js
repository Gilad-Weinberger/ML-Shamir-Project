(function () {
  const form = document.getElementById("upload-form");
  const fileInput = form.querySelector('input[type="file"]');
  const resultsStatus = document.getElementById("results-status");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const previewFilename = document.getElementById("preview-filename");
  const csrfInput = form.querySelector("[name=csrfmiddlewaretoken]");

  let previewObjectUrl = null;
  let analysisGeneration = 0;

  function getCsrfToken() {
    return csrfInput ? csrfInput.value : "";
  }

  function revokePreviewUrl() {
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }
  }

  function setPreview(file) {
    revokePreviewUrl();
    if (!file) {
      previewContainer.hidden = true;
      previewContainer.classList.add("hidden");
      previewImage.removeAttribute("src");
      previewFilename.textContent = "";
      previewFilename.removeAttribute("title");
      return;
    }

    previewObjectUrl = URL.createObjectURL(file);
    previewImage.src = previewObjectUrl;
    previewImage.alt = "Selected leaf sample: " + file.name;
    previewFilename.textContent = file.name;
    previewFilename.title = file.name;
    previewContainer.hidden = false;
    previewContainer.classList.remove("hidden");
  }

  function resetResultsPlaceholder() {
    resultsStatus.className = "empty-state";
    resultsStatus.innerHTML =
      '<span class="empty-icon" aria-hidden="true"></span>' +
      "<p>Results will appear here after analysis.</p>";
  }

  function setLoading(isLoading) {
    fileInput.disabled = isLoading;
    if (isLoading) {
      resultsStatus.className = "empty-state is-loading";
      resultsStatus.innerHTML =
        '<span class="spinner" aria-hidden="true"></span>' +
        "<p>Analyzing image…</p>";
    }
  }

  function showError(message) {
    resultsStatus.className = "result-card result-card-error";
    resultsStatus.innerHTML =
      '<span class="result-label">Analysis unavailable</span>' +
      "<p>" + escapeHtml(message) + "</p>";
  }

  function showPrediction(value) {
    resultsStatus.className = "result-card";
    resultsStatus.innerHTML =
      '<span class="result-label">Estimated percentage of Downy Mildew in the leaf</span>' +
      '<strong class="result-value">' +
      escapeHtml(String(value)) +
      "%</strong>";
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function uploadInBackground(file, generation) {
    const body = new FormData();
    body.append("csrfmiddlewaretoken", getCsrfToken());
    body.append("image", file);

    fetch("/api/upload/", {
      method: "POST",
      body: body,
      headers: { "X-CSRFToken": getCsrfToken() },
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, data: data };
        });
      })
      .then(function (result) {
        if (generation !== analysisGeneration) {
          return;
        }
        if (result.ok && result.data.url) {
          revokePreviewUrl();
          previewImage.src = result.data.url;
          previewImage.alt =
            "Uploaded leaf sample" +
            (result.data.filename ? ": " + result.data.filename : "");
        }
      })
      .catch(function () {
        /* Preview stays on blob URL; upload failure is non-blocking */
      });
  }

  function runAnalysis(file, generation) {
    setLoading(true);

    const body = new FormData();
    body.append("csrfmiddlewaretoken", getCsrfToken());
    body.append("image", file);

    fetch("/api/predict/", {
      method: "POST",
      body: body,
      headers: { "X-CSRFToken": getCsrfToken() },
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, data: data };
        });
      })
      .then(function (result) {
        if (generation !== analysisGeneration) {
          return;
        }
        if (!result.ok) {
          showError(result.data.error || "Prediction failed.");
          return;
        }
        showPrediction(result.data.prediction);
        uploadInBackground(file, generation);
      })
      .catch(function () {
        if (generation !== analysisGeneration) {
          return;
        }
        showError("Network error. Please try again.");
      })
      .finally(function () {
        if (generation === analysisGeneration) {
          setLoading(false);
        }
      });
  }

  fileInput.addEventListener("change", function () {
    const file = fileInput.files && fileInput.files[0];
    setPreview(file || null);

    if (!file) {
      resetResultsPlaceholder();
      return;
    }

    analysisGeneration += 1;
    const generation = analysisGeneration;
    resetResultsPlaceholder();
    runAnalysis(file, generation);
  });

  window.addEventListener("pagehide", revokePreviewUrl);
})();
