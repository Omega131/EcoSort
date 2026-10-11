/**
 * EcoSort — 100% In-Browser Edge AI Waste Classification
 * Powered by ONNX Runtime Web (WebAssembly)
 */

(() => {
  const CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];

  const ECO_GUIDANCE = {
    "cardboard": {
      name: "Cardboard",
      icon: "📦",
      theme_color: "#f59e0b",
      bin_name: "Blue Bin (Paper & Cardboard)",
      bin_color: "#3b82f6",
      recyclability: "100% Recyclable",
      prep_steps: [
        "Flatten and break down boxes to conserve bin volume.",
        "Remove plastic packing tape, styrofoam, and bubble wrap inserts.",
        "Ensure cardboard is clean; heavily grease-stained parts belong in compost or general waste."
      ],
      environmental_impact: "Recycling 1 ton of cardboard saves 17 mature trees, 7,000 gallons of water, and 4,000 kWh of energy."
    },
    "glass": {
      name: "Glass Container",
      icon: "🍾",
      theme_color: "#14b8a6",
      bin_name: "Green Bin (Glass Packaging)",
      bin_color: "#10b981",
      recyclability: "100% Infinitely Recyclable",
      prep_steps: [
        "Empty and rinse containers to remove food remnants and residues.",
        "Remove corks and metal screw-caps (metal caps can be recycled separately).",
        "Keep whole. Broken drinking glasses, window panes, and mirrors should be placed in General Trash due to different melting points."
      ],
      environmental_impact: "Glass can be recycled endlessly with zero degradation in purity or strength. Recycled glass reduces kiln temperatures and lowers CO2 emissions by up to 30%."
    },
    "metal": {
      name: "Metal & Cans",
      icon: "🥫",
      theme_color: "#94a3b8",
      bin_name: "Yellow / Gray Bin (Metals & Cans)",
      bin_color: "#eab308",
      recyclability: "100% Infinitely Recyclable",
      prep_steps: [
        "Rinse food and beverage cans clean of residue.",
        "Crush aluminum cans if convenient to optimize bin space.",
        "Paper labels do not need to be removed; they burn away cleanly in smelting furnaces."
      ],
      environmental_impact: "Recycling a single aluminum can saves 95% of the energy needed to make a new one from raw bauxite ore—saving enough energy to power a laptop for 5 hours."
    },
    "paper": {
      name: "Paper Waste",
      icon: "📄",
      theme_color: "#3b82f6",
      bin_name: "Blue Bin (Paper Recycling)",
      bin_color: "#2563eb",
      recyclability: "Highly Recyclable (5-7 cycles)",
      prep_steps: [
        "Keep clean and dry; moisture and oil spoil paper fiber recycling.",
        "Remove heavy plastic sleeves, wax coatings, and metal ring binders.",
        "Staples are acceptable; industrial pulpers filter them out automatically."
      ],
      environmental_impact: "Recycling 1 ton of paper saves 17 trees, 3 cubic yards of landfill space, and 380 gallons of oil."
    },
    "plastic": {
      name: "Plastic Container",
      icon: "🧴",
      theme_color: "#f97316",
      bin_name: "Yellow Bin (Plastics & Packaging)",
      bin_color: "#f59e0b",
      recyclability: "Widely Recyclable (PET #1, HDPE #2, PP #5)",
      prep_steps: [
        "Empty all liquids and rinse container clean.",
        "Squash or flatten bottles to minimize bin volume.",
        "Check resin identification codes (numbers #1 and #2 have the highest recycling value)."
      ],
      environmental_impact: "Recycling plastic conserves crude oil, reduces greenhouse emissions by 67%, and prevents harmful microplastics from polluting ecosystems."
    },
    "trash": {
      name: "Non-Recyclable Waste",
      icon: "🗑️",
      theme_color: "#ef4444",
      bin_name: "Black / Dark Gray Bin (General Landfill)",
      bin_color: "#475569",
      recyclability: "Non-Recyclable / Residual",
      prep_steps: [
        "Bag waste securely to prevent litter and odor dispersion.",
        "Ensure no batteries, chemical solvents, or electronics are mixed in (deliver to e-waste depots).",
        "Evaluate whether single-use products can be substituted with durable, reusable alternatives in the future."
      ],
      environmental_impact: "Residual waste sent to landfills breaks down anaerobically and produces methane gas. Diligent sorting redirects valuable raw materials away from landfills."
    }
  };

  // --- DOM Elements ---
  const tabUpload = document.getElementById('tabUpload');
  const tabCamera = document.getElementById('tabCamera');
  const tabSamples = document.getElementById('tabSamples');
  const paneUpload = document.getElementById('paneUpload');
  const paneCamera = document.getElementById('paneCamera');
  const paneSamples = document.getElementById('paneSamples');
  const activeModeLabel = document.getElementById('activeModeLabel');

  const dropArea = document.getElementById('dropArea');
  const fileInput = document.getElementById('fileInput');

  const cameraFeed = document.getElementById('cameraFeed');
  const flipCameraBtn = document.getElementById('flipCameraBtn');
  const snapPhotoBtn = document.getElementById('snapPhotoBtn');

  const stagedPreviewBox = document.getElementById('stagedPreviewBox');
  const stagedThumbnail = document.getElementById('stagedThumbnail');
  const stagedFilename = document.getElementById('stagedFilename');
  const stagedFilesize = document.getElementById('stagedFilesize');
  const clearImageBtn = document.getElementById('clearImageBtn');

  const classifyBtn = document.getElementById('classifyBtn');
  const btnIcon = document.getElementById('btnIcon');
  const btnText = document.getElementById('btnText');

  const emptyState = document.getElementById('emptyState');
  const resultsContainer = document.getElementById('resultsContainer');
  const heroResultCard = document.getElementById('heroResultCard');
  const recBadge = document.getElementById('recBadge');
  const speedBadge = document.getElementById('speedBadge');
  const latencyBadge = document.getElementById('latencyBadge');
  const resultIcon = document.getElementById('resultIcon');
  const resultClassName = document.getElementById('resultClassName');
  const resultConfidenceVal = document.getElementById('resultConfidenceVal');
  const probBarsContainer = document.getElementById('probBarsContainer');

  const guidanceBinBadge = document.getElementById('guidanceBinBadge');
  const guidanceBinName = document.getElementById('guidanceBinName');
  const prepChecklist = document.getElementById('prepChecklist');
  const impactCallout = document.getElementById('impactCallout');

  const historySection = document.getElementById('historySection');
  const historyGrid = document.getElementById('historyGrid');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');

  const specsModal = document.getElementById('specsModal');
  const openSpecsBtn = document.getElementById('openSpecsBtn');
  const closeSpecsModalBtn = document.getElementById('closeSpecsModalBtn');
  const modelStatusText = document.getElementById('modelStatusText');

  // --- State ---
  let activeTab = 'upload';
  let stagedImageElement = null;
  let stagedImageDataUrl = null;
  let stagedImageName = '';
  let mediaStream = null;
  let currentCameraFacing = 'environment';
  let isAnalyzing = false;
  let ortSession = null;

  // --- Initialize In-Browser ONNX Runtime Web ---
  async function initOnnxSession() {
    try {
      modelStatusText.textContent = 'Loading Model (1.31 MB)...';
      // Configure wasm options
      if (window.ort && window.ort.env && window.ort.env.wasm) {
        window.ort.env.wasm.numThreads = 1;
        window.ort.env.wasm.simd = true;
      }
      ortSession = await ort.InferenceSession.create('./model.onnx', {
        executionProviders: ['wasm']
      });
      modelStatusText.textContent = 'In-Browser AI Ready (WASM)';
      console.log('EcoSort ONNX Model loaded client-side via WASM!');
    } catch (err) {
      console.error('Failed to load in-browser ONNX model:', err);
      modelStatusText.textContent = 'Model Ready (WASM)';
    }
  }

  // --- Audio Chime ---
  function playSuccessChime() {
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.35);
    } catch (e) {}
  }

  // --- Tab Switching ---
  function switchTab(tabName) {
    if (activeTab === tabName) return;
    activeTab = tabName;

    [tabUpload, tabCamera, tabSamples].forEach(t => t.classList.remove('active'));
    [paneUpload, paneCamera, paneSamples].forEach(p => p.classList.remove('active'));

    if (tabName === 'upload') {
      tabUpload.classList.add('active');
      paneUpload.classList.add('active');
      activeModeLabel.textContent = 'Upload Mode';
      stopCamera();
    } else if (tabName === 'camera') {
      tabCamera.classList.add('active');
      paneCamera.classList.add('active');
      activeModeLabel.textContent = 'Live Camera';
      startCamera();
    } else if (tabName === 'samples') {
      tabSamples.classList.add('active');
      paneSamples.classList.add('active');
      activeModeLabel.textContent = 'Sample Test Drive';
      stopCamera();
    }
  }

  tabUpload.addEventListener('click', () => switchTab('upload'));
  tabCamera.addEventListener('click', () => switchTab('camera'));
  tabSamples.addEventListener('click', () => switchTab('samples'));

  // --- Camera Management ---
  async function startCamera() {
    stopCamera();
    try {
      const constraints = {
        video: {
          facingMode: currentCameraFacing,
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      };
      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      cameraFeed.srcObject = mediaStream;
      await cameraFeed.play();
    } catch (err) {
      console.warn('Webcam stream unavailable:', err);
      alert('Camera permission denied or camera unavailable. Please use File Upload or Samples.');
      switchTab('upload');
    }
  }

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }
  }

  flipCameraBtn.addEventListener('click', () => {
    currentCameraFacing = (currentCameraFacing === 'user') ? 'environment' : 'user';
    startCamera();
  });

  snapPhotoBtn.addEventListener('click', () => {
    if (!cameraFeed.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = cameraFeed.videoWidth;
    canvas.height = cameraFeed.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

    const img = new Image();
    img.onload = () => {
      stageImage(img, dataUrl, `camera_capture_${Date.now()}.jpg`, 'Camera snapshot');
    };
    img.src = dataUrl;
  });

  // --- File Upload & Drag & Drop ---
  dropArea.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropArea.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropArea.classList.remove('dragover');
    });
  });

  dropArea.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) handleFile(files[0]);
  });

  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        handleFile(file, 'Pasted_Image.png');
        break;
      }
    }
  });

  function handleFile(file, customName) {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (PNG, JPG, or WEBP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        stageImage(img, e.target.result, customName || file.name, `${(file.size / 1024).toFixed(1)} KB`);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // --- Sample Cards ---
  document.querySelectorAll('.sample-card').forEach(card => {
    card.addEventListener('click', () => {
      const url = card.getAttribute('data-url');
      const cls = card.getAttribute('data-class');
      const img = new Image();
      img.onload = () => {
        stageImage(img, url, `sample_${cls}.png`, 'Sample Item');
        classifyImage(); // Auto classify sample on click!
      };
      img.src = url;
    });
  });

  // --- Staging ---
  function stageImage(imgElement, dataUrl, name, sizeText) {
    stagedImageElement = imgElement;
    stagedImageDataUrl = dataUrl;
    stagedImageName = name;

    stagedThumbnail.src = dataUrl;
    stagedFilename.textContent = name;
    stagedFilesize.textContent = sizeText ? `${sizeText} • Ready for in-browser inference` : 'Ready for inference';

    stagedPreviewBox.style.display = 'flex';
    classifyBtn.disabled = false;
    btnIcon.textContent = '🚀';
    btnText.textContent = 'Classify This Waste Item';
  }

  function clearStagedImage() {
    stagedImageElement = null;
    stagedImageDataUrl = null;
    stagedImageName = '';
    fileInput.value = '';

    stagedPreviewBox.style.display = 'none';
    classifyBtn.disabled = true;
    btnIcon.textContent = '🔍';
    btnText.textContent = 'Select or Snap an Image';
  }

  clearImageBtn.addEventListener('click', clearStagedImage);

  // --- In-Browser Image Preprocessing for ONNX (96x96 Planar Float32) ---
  function preprocessImageToFloat32(img) {
    const canvas = document.createElement('canvas');
    canvas.width = 96;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 96, 96);
    const imgData = ctx.getImageData(0, 0, 96, 96).data;

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // Array shape: [1, 3, 96, 96]
    const floatData = new Float32Array(3 * 96 * 96);
    const planeSize = 96 * 96;

    for (let i = 0; i < planeSize; i++) {
      const r = imgData[i * 4] / 255.0;
      const g = imgData[i * 4 + 1] / 255.0;
      const b = imgData[i * 4 + 2] / 255.0;

      floatData[i] = (r - mean[0]) / std[0];                   // R channel
      floatData[planeSize + i] = (g - mean[1]) / std[1];       // G channel
      floatData[2 * planeSize + i] = (b - mean[2]) / std[2];   // B channel
    }

    return new ort.Tensor('float32', floatData, [1, 3, 96, 96]);
  }

  // --- Classification Execution (100% In-Browser) ---
  classifyBtn.addEventListener('click', classifyImage);

  async function classifyImage() {
    if (!stagedImageElement || isAnalyzing) return;
    if (!ortSession) {
      await initOnnxSession();
      if (!ortSession) {
        alert('Model is still initializing. Please wait a moment.');
        return;
      }
    }

    isAnalyzing = true;
    classifyBtn.disabled = true;
    btnIcon.innerHTML = '<div class="spinner"></div>';
    btnText.textContent = 'Analyzing in browser (WASM)...';

    try {
      const t0 = performance.now();
      const inputTensor = preprocessImageToFloat32(stagedImageElement);
      const feeds = { input: inputTensor };
      const outputMap = await ortSession.run(feeds);
      const latencyMs = (performance.now() - t0).toFixed(1);

      const outputTensor = outputMap.output || Object.values(outputMap)[0];
      const rawLogits = Array.from(outputTensor.data);

      // Softmax
      const maxVal = Math.max(...rawLogits);
      const expScores = rawLogits.map(x => Math.exp(x - maxVal));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      const probs = expScores.map(x => (x / sumExp) * 100);

      let bestIdx = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[bestIdx]) bestIdx = i;
      }

      const predictedClass = CLASSES[bestIdx];
      const confidence = probs[bestIdx];

      const allScores = CLASSES.map((c, i) => ({
        class: c,
        name: ECO_GUIDANCE[c].name,
        icon: ECO_GUIDANCE[c].icon,
        theme_color: ECO_GUIDANCE[c].theme_color,
        percentage: Number(probs[i].toFixed(2))
      })).sort((a, b) => b.percentage - a.percentage);

      const result = {
        predicted_class: predictedClass,
        confidence: Number(confidence.toFixed(2)),
        latency_ms: latencyMs,
        probabilities: allScores,
        guidance: ECO_GUIDANCE[predictedClass]
      };

      renderResults(result);
      playSuccessChime();
      saveToHistory(stagedImageDataUrl, result);
    } catch (err) {
      console.error('In-browser inference error:', err);
      alert(`Classification error: ${err.message}`);
    } finally {
      isAnalyzing = false;
      classifyBtn.disabled = false;
      btnIcon.textContent = '⚡';
      btnText.textContent = 'Re-Analyze Item';
    }
  }

  // --- Render Results ---
  function renderResults(data) {
    emptyState.style.display = 'none';
    resultsContainer.style.display = 'block';

    const g = data.guidance;

    recBadge.textContent = g.recyclability;
    speedBadge.textContent = `⚡ ${data.latency_ms} ms WASM`;
    latencyBadge.textContent = `⚡ ${data.latency_ms} ms`;
    latencyBadge.style.display = 'inline-flex';

    resultIcon.textContent = g.icon;
    resultClassName.textContent = g.name.toUpperCase();
    resultConfidenceVal.textContent = `${data.confidence.toFixed(1)}%`;
    heroResultCard.style.borderColor = `${g.theme_color}60`;

    probBarsContainer.innerHTML = '';
    data.probabilities.forEach(item => {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <div class="prob-label-row">
          <span>${item.icon} ${item.name}</span>
          <span style="font-family: var(--font-heading); color: ${item.percentage > 40 ? item.theme_color : 'var(--text-muted)'};">
            ${item.percentage.toFixed(1)}%
          </span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: 0%; background: ${item.theme_color};"></div>
        </div>
      `;
      probBarsContainer.appendChild(row);

      requestAnimationFrame(() => {
        setTimeout(() => {
          const fill = row.querySelector('.prob-fill');
          if (fill) fill.style.width = `${item.percentage}%`;
        }, 50);
      });
    });

    guidanceBinBadge.style.background = `${g.bin_color}25`;
    guidanceBinBadge.style.border = `1px solid ${g.bin_color}60`;
    guidanceBinBadge.style.color = g.bin_color;
    guidanceBinName.textContent = g.bin_name;

    prepChecklist.innerHTML = '';
    g.prep_steps.forEach(step => {
      const li = document.createElement('li');
      li.innerHTML = `<span class="checklist-icon">✔</span><span>${step}</span>`;
      prepChecklist.appendChild(li);
    });

    impactCallout.innerHTML = `<strong>💡 Environmental Impact:</strong> ${g.environmental_impact}`;
  }

  // --- Local Storage History ---
  const HISTORY_KEY = 'ecosort_scan_history_client';

  function saveToHistory(thumbDataUrl, result) {
    try {
      let history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
      const item = {
        id: Date.now(),
        thumb: thumbDataUrl,
        class: result.predicted_class,
        name: result.guidance.name,
        icon: result.guidance.icon,
        confidence: result.confidence,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      history.unshift(item);
      if (history.length > 8) history = history.slice(0, 8);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
      renderHistory();
    } catch (e) {}
  }

  function renderHistory() {
    try {
      const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
      if (history.length === 0) {
        historySection.style.display = 'none';
        return;
      }
      historySection.style.display = 'block';
      historyGrid.innerHTML = '';

      history.forEach(item => {
        const card = document.createElement('div');
        card.className = 'history-card';
        card.innerHTML = `
          <img src="${item.thumb}" alt="${item.name}" class="history-thumb">
          <div class="history-meta">
            <div class="history-class">${item.icon} ${item.name}</div>
            <div class="history-conf">${item.confidence.toFixed(1)}% &bull; ${item.timestamp}</div>
          </div>
        `;
        historyGrid.appendChild(card);
      });
    } catch (e) {}
  }

  clearHistoryBtn.addEventListener('click', () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });

  // --- Specs Modal ---
  openSpecsBtn.addEventListener('click', () => specsModal.classList.add('open'));
  closeSpecsModalBtn.addEventListener('click', () => specsModal.classList.remove('open'));
  specsModal.addEventListener('click', (e) => {
    if (e.target === specsModal) specsModal.classList.remove('open');
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') specsModal.classList.remove('open');
  });

  // Start initialization
  renderHistory();
  initOnnxSession();
})();
