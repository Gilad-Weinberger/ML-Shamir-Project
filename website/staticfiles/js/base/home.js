(function () {
  const form = document.getElementById("upload-form");
  const dropZone = document.getElementById("drop-zone");
  const fileInput = form.querySelector('input[type="file"]');
  const dropIdle = document.getElementById("drop-idle");
  const dropResult = document.getElementById("drop-result");
  const resultsStatus = document.getElementById("results-status");
  const previewImage = document.getElementById("preview-image");
  const previewFilename = document.getElementById("preview-filename");
  const retryBtn = document.getElementById("retry-btn");
  const csrfInput = form.querySelector("[name=csrfmiddlewaretoken]");

  let previewObjectUrl = null;
  let analysisGeneration = 0;
  let dragDepth = 0;
  let lastStableState = "idle";

  function getCsrfToken() {
    return csrfInput ? csrfInput.value : "";
  }

  function revokePreviewUrl() {
    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = null;
    }
  }

  function setZoneState(state) {
    dropZone.className = "drop-zone is-" + state;
    if (state !== "dragover") {
      lastStableState = state;
    }

    const isIdle = state === "idle" || state === "dragover";
    const isResult = state === "result" || state === "error";

    dropIdle.hidden = !isIdle;
    dropIdle.classList.toggle("hidden", !isIdle);
    dropResult.hidden = !isResult;
    dropResult.classList.toggle("hidden", !isResult);
  }

  function setPreview(file) {
    revokePreviewUrl();
    if (!file) {
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
  }

  function isResultView() {
    return lastStableState === "result" || lastStableState === "error";
  }

  function setAnalyzing(isAnalyzing, isReplacing) {
    fileInput.disabled = isAnalyzing;
    retryBtn.disabled = isAnalyzing;
    if (isReplacing) {
      resultsStatus.classList.toggle("is-reanalyzing", isAnalyzing);
    }
  }

  function showResultLoading() {
    setZoneState("result");
    resultsStatus.className = "result-card is-loading";
    resultsStatus.innerHTML =
      '<span class="result-label">Estimated percentage of Downy Mildew in the leaf</span>' +
      '<span class="spinner spinner-in-card" aria-hidden="true"></span>' +
      '<span class="result-loading-text">Analyzing image…</span>';
  }

  function showError(message) {
    setZoneState("error");
    resultsStatus.className = "result-card result-card-error";
    resultsStatus.innerHTML =
      '<span class="result-label">Analysis unavailable</span>' +
      "<p>" + escapeHtml(message) + "</p>";
  }

  function showPrediction(value) {
    setZoneState("result");
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

  function isImageFile(file) {
    return file && file.type.startsWith("image/");
  }

  function canAcceptUpload() {
    return !fileInput.disabled;
  }

  function processFile(file) {
    if (!canAcceptUpload()) {
      return;
    }

    const isReplacing = isResultView();

    if (!isImageFile(file)) {
      if (!isReplacing) {
        setPreview(null);
      }
      showError("Please choose a valid image file.");
      return;
    }

    analysisGeneration += 1;
    const generation = analysisGeneration;

    if (isReplacing) {
      setAnalyzing(true, true);
    } else {
      setPreview(file);
      showResultLoading();
    }

    runAnalysis(file, generation, isReplacing);
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

  function runAnalysis(file, generation, isReplacing) {
    if (!isReplacing) {
      setAnalyzing(true, false);
    }

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
        setPreview(file);
        previewFilename.textContent = file.name;
        previewFilename.title = file.name;
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
          setAnalyzing(false, isReplacing);
        }
      });
  }

  fileInput.addEventListener("change", function () {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      return;
    }
    processFile(file);
  });

  retryBtn.addEventListener("click", function () {
    fileInput.value = "";
    fileInput.click();
  });

  ["dragenter", "dragover"].forEach(function (eventName) {
    dropZone.addEventListener(eventName, function (event) {
      event.preventDefault();
      if (!canAcceptUpload()) {
        return;
      }
      dragDepth += 1;
      if (
        dropZone.classList.contains("is-idle") ||
        dropZone.classList.contains("is-dragover") ||
        dropZone.classList.contains("is-result") ||
        dropZone.classList.contains("is-error")
      ) {
        setZoneState("dragover");
      }
    });
  });

  dropZone.addEventListener("dragleave", function (event) {
    event.preventDefault();
    dragDepth -= 1;
    if (dragDepth <= 0) {
      dragDepth = 0;
      if (dropZone.classList.contains("is-dragover")) {
        setZoneState(lastStableState);
      }
    }
  });

  dropZone.addEventListener("drop", function (event) {
    event.preventDefault();
    dragDepth = 0;

    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (!file) {
      setZoneState(lastStableState);
      return;
    }

    processFile(file);
  });

  window.addEventListener("pagehide", revokePreviewUrl);
})();
