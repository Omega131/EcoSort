/**
 * EcoSort — Interactive Client Application
 * Handles Drag & Drop, Live Camera, Quick Samples, API Inference, and Audio Feedback
 */

(() => {
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

  // --- State ---
  let activeTab = 'upload';
  let stagedImageBlob = null;
  let stagedImageDataUrl = null;
  let stagedImageName = '';
  let mediaStream = null;
  let currentCameraFacing = 'environment'; // default to back camera on phones
  let isAnalyzing = false;

  // --- Web Audio Chime ---
  function playSuccessChime() {
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
      osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 0.12); // A5
      gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.35);
    } catch (e) {
      // AudioContext unavailable or blocked
    }
  }

  // --- Tab Switching Logic ---
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
      alert('Camera access could not be initialized or permission was denied. You can still use File Upload or Samples!');
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
    
    canvas.toBlob(blob => {
      stageImage(blob, dataUrl, `camera_capture_${Date.now()}.jpg`);
    }, 'image/jpeg', 0.92);
  });

  // --- File Upload & Dropzone Handling ---
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

  // --- Clipboard Paste Support ---
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
      stageImage(file, e.target.result, customName || file.name);
    };
    reader.readAsDataURL(file);
  }

  // --- Sample Card Selection ---
  document.querySelectorAll('.sample-card').forEach(card => {
    card.addEventListener('click', async () => {
      const url = card.getAttribute('data-url');
      const cls = card.getAttribute('data-class');
      try {
        const resp = await fetch(url);
        const blob = await resp.blob();
        const reader = new FileReader();
        reader.onload = (e) => {
          stageImage(blob, e.target.result, `sample_${cls}.png`);
          classifyImage(); // Auto trigger classification for instant satisfaction
        };
        reader.readAsDataURL(blob);
      } catch (err) {
        console.error('Failed to load sample image:', err);
      }
    });
  });

  // --- Staging Logic ---
  function stageImage(blob, dataUrl, name) {
    stagedImageBlob = blob;
    stagedImageDataUrl = dataUrl;
    stagedImageName = name;

    stagedThumbnail.src = dataUrl;
    stagedFilename.textContent = name;
    
    const kb = (blob.size / 1024).toFixed(1);
    stagedFilesize.textContent = `${kb} KB &bull; Ready for inference`;

    stagedPreviewBox.style.display = 'flex';
    classifyBtn.disabled = false;
    btnIcon.textContent = '🚀';
    btnText.textContent = 'Classify This Waste Item';
  }

  function clearStagedImage() {
    stagedImageBlob = null;
    stagedImageDataUrl = null;
    stagedImageName = '';
    fileInput.value = '';

    stagedPreviewBox.style.display = 'none';
    classifyBtn.disabled = true;
    btnIcon.textContent = '🔍';
    btnText.textContent = 'Select or Snap an Image';
  }

  clearImageBtn.addEventListener('click', clearStagedImage);

  // --- Classification Execution ---
  classifyBtn.addEventListener('click', classifyImage);

  async function classifyImage() {
    if (!stagedImageBlob || isAnalyzing) return;
    isAnalyzing = true;

    // Set UI to loading state
    classifyBtn.disabled = true;
    btnIcon.innerHTML = '<div class="spinner"></div>';
    btnText.textContent = 'Analyzing with MobileNet ONNX...';

    const formData = new FormData();
    formData.append('file', stagedImageBlob, stagedImageName || 'upload.png');

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Inference request failed');
      }

      const result = await response.json();
      renderResults(result);
      playSuccessChime();
      saveToHistory(stagedImageDataUrl, result);
    } catch (err) {
      console.error('Prediction error:', err);
      alert(`Classification failed: ${err.message}`);
    } finally {
      isAnalyzing = false;
      classifyBtn.disabled = false;
      btnIcon.textContent = '⚡';
      btnText.textContent = 'Re-Analyze Item';
    }
  }

  // --- Result Rendering ---
  function renderResults(data) {
    emptyState.style.display = 'none';
    resultsContainer.style.display = 'block';

    const g = data.guidance;

    // Header Badges
    recBadge.textContent = g.recyclability;
    speedBadge.textContent = `⚡ ${data.latency_ms} ms CPU`;
    latencyBadge.textContent = `⚡ ${data.latency_ms} ms`;
    latencyBadge.style.display = 'inline-flex';

    // Hero Result Card
    resultIcon.textContent = g.icon;
    resultClassName.textContent = g.name.toUpperCase();
    resultConfidenceVal.textContent = `${data.confidence.toFixed(1)}%`;
    heroResultCard.style.borderColor = `${g.theme_color}60`;

    // Probability Bars
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

      // Trigger width animation on next frame
      requestAnimationFrame(() => {
        setTimeout(() => {
          const fill = row.querySelector('.prob-fill');
          if (fill) fill.style.width = `${item.percentage}%`;
        }, 50);
      });
    });

    // Guidance Section
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
  const HISTORY_KEY = 'ecosort_scan_history';

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
      if (history.length > 8) history = history.slice(0, 8); // Keep last 8 scans
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
      renderHistory();
    } catch (e) {
      console.warn('LocalStorage error:', e);
    }
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
    } catch (e) {
      // Ignore
    }
  }

  clearHistoryBtn.addEventListener('click', () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });

  // --- Edge Specs Modal ---
  openSpecsBtn.addEventListener('click', () => specsModal.classList.add('open'));
  closeSpecsModalBtn.addEventListener('click', () => specsModal.classList.remove('open'));
  specsModal.addEventListener('click', (e) => {
    if (e.target === specsModal) specsModal.classList.remove('open');
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') specsModal.classList.remove('open');
  });

  // Initial load
  renderHistory();

  // Check health on boot
  fetch('/api/health')
    .then(r => r.json())
    .then(data => {
      const txt = document.getElementById('modelStatusText');
      if (txt && data.status === 'healthy') {
        txt.textContent = `Model Online (${data.metadata.file_size_mb} MB)`;
      }
    })
    .catch(() => {});
})();
