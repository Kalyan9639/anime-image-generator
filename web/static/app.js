/**
 * ASCII to Image Generator - Interactive Web Frontend
 */

document.addEventListener("DOMContentLoaded", () => {
  const asciiInput = document.getElementById("asciiInput");
  const sampleSelect = document.getElementById("sampleSelect");
  const modeSelect = document.getElementById("modeSelect");
  const themeSelect = document.getElementById("themeSelect");
  const scaleSelect = document.getElementById("scaleSelect");
  const denoiseSlider = document.getElementById("denoiseSlider");
  const denoiseVal = document.getElementById("denoiseVal");
  const btnConvert = document.getElementById("btnConvert");

  const detectionPill = document.getElementById("detectionPill");
  const detectionConf = document.getElementById("detectionConf");
  const detectionExplanation = document.getElementById("detectionExplanation");

  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  const emptyPlaceholder = document.getElementById("emptyPlaceholder");
  const previewImage = document.getElementById("previewImage");
  const previewMeta = document.getElementById("previewMeta");
  const metaDimensions = document.getElementById("metaDimensions");
  const metaSize = document.getElementById("metaSize");
  const metaMode = document.getElementById("metaMode");
  const svgCodeViewer = document.getElementById("svgCodeViewer");

  const btnDownloadSvg = document.getElementById("btnDownloadSvg");
  const btnDownloadPng = document.getElementById("btnDownloadPng");

  let currentSvg = "";
  let currentPngBlob = null;
  let sampleCache = {};
  let debounceTimer = null;

  // 1. Load Built-In Samples
  fetch("/api/samples")
    .then((r) => r.json())
    .then((data) => {
      if (data.samples && data.samples.length > 0) {
        data.samples.forEach((s) => {
          sampleCache[s.id] = s.text;
          const opt = document.createElement("option");
          opt.value = s.id;
          opt.textContent = s.name;
          sampleSelect.appendChild(opt);
        });
      }
    })
    .catch((err) => console.error("Could not load samples:", err));

  sampleSelect.addEventListener("change", (e) => {
    const id = e.target.value;
    if (id && sampleCache[id]) {
      asciiInput.value = sampleCache[id];
      triggerDetection();
      triggerConversion();
    }
  });

  // 2. Real-Time Debounced Detection
  asciiInput.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(triggerDetection, 300);
  });

  denoiseSlider.addEventListener("input", (e) => {
    denoiseVal.textContent = e.target.value;
  });

  function triggerDetection() {
    const text = asciiInput.value.trim();
    if (!text) {
      detectionPill.textContent = "Type: Empty";
      detectionPill.style.backgroundColor = "#4b5563";
      detectionConf.textContent = "-";
      detectionExplanation.textContent = "Paste ASCII text or load a sample.";
      return;
    }

    fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    })
      .then((r) => r.json())
      .then((data) => {
        const isDiagram = data.category === "diagram";
        detectionPill.textContent = `${data.category.toUpperCase()} [${data.subtype}]`;
        detectionPill.style.backgroundColor = isDiagram ? "#0284c7" : "#7c3aed";
        detectionConf.textContent = `${(data.confidence * 100).toFixed(1)}% Confidence`;
        detectionExplanation.textContent = data.explanation;
      })
      .catch((err) => console.error("Detection error:", err));
  }

  // 3. Tab Switching
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });

  // 4. Conversion Trigger
  btnConvert.addEventListener("click", triggerConversion);

  function triggerConversion() {
    const text = asciiInput.value;
    if (!text.trim()) {
      alert("Please enter or paste some ASCII text first.");
      return;
    }

    btnConvert.disabled = true;
    btnConvert.innerHTML = "⏳ Processing...";

    const payload = {
      text,
      mode: modeSelect.value,
      theme: themeSelect.value,
      scale: parseFloat(scaleSelect.value),
      denoise: parseFloat(denoiseSlider.value),
      width: 1200,
    };

    fetch("/api/convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        btnConvert.disabled = false;
        btnConvert.innerHTML = '<span class="btn-icon">⚡</span> Convert to Image';

        if (data.status === "error") {
          alert("Conversion error: " + data.message);
          return;
        }

        // Display PNG preview
        if (data.png_b64) {
          previewImage.src = "data:image/png;base64," + data.png_b64;
          previewImage.style.display = "block";
          emptyPlaceholder.style.display = "none";
          previewMeta.style.display = "flex";

          previewImage.onload = () => {
            metaDimensions.textContent = `Dimensions: ${previewImage.naturalWidth} × ${previewImage.naturalHeight} px`;
          };

          metaSize.textContent = `PNG Size: ${(data.png_size_bytes / 1024).toFixed(1)} KB`;
          metaMode.textContent = `Mode: ${data.effective_mode} (${data.effective_theme})`;

          // Convert b64 to Blob for download
          const byteChars = atob(data.png_b64);
          const byteNums = new Array(byteChars.length);
          for (let i = 0; i < byteChars.length; i++) {
            byteNums[i] = byteChars.charCodeAt(i);
          }
          currentPngBlob = new Blob([new Uint8Array(byteNums)], { type: "image/png" });
          btnDownloadPng.disabled = false;
        }

        // Store & Display SVG source
        if (data.svg) {
          currentSvg = data.svg;
          svgCodeViewer.textContent = data.svg;
          btnDownloadSvg.disabled = false;
        } else {
          currentSvg = "";
          svgCodeViewer.textContent = "// SVG is not available in photographic reconstruction mode.";
          btnDownloadSvg.disabled = true;
        }
      })
      .catch((err) => {
        btnConvert.disabled = false;
        btnConvert.innerHTML = '<span class="btn-icon">⚡</span> Convert to Image';
        alert("Server request failed: " + err);
      });
  }

  // 5. Downloads
  btnDownloadSvg.addEventListener("click", () => {
    if (!currentSvg) return;
    const blob = new Blob([currentSvg], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ascii_diagram.svg";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  btnDownloadPng.addEventListener("click", () => {
    if (!currentPngBlob) return;
    const url = URL.createObjectURL(currentPngBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ascii_render.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
});
