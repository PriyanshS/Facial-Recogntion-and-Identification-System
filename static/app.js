/**
 * VisionIQ — Classical Face Recognition Studio Client Application
 * Vanilla JavaScript implementation handling pipeline inspection,
 * real-time matching, threshold recalculation, and canvas charts.
 */

// Global Application State
const state = {
  currentFile: null,
  currentDataUrl: null,
  currentSampleKey: 'alice_query_genuine',
  threshold: 0.46,
  lastResult: null,
  enrolledUsers: [],
  cameraStream: null,
};

// DOM Elements Cache
const elements = {
  // Tabs
  tabButtons: document.querySelectorAll('.nav-tab'),
  tabPanels: document.querySelectorAll('.tab-panel'),

  // Header
  dbUserCount: document.getElementById('db-user-count'),
  btnSeedDemo: document.getElementById('btn-seed-demo'),

  // Identification Tab
  sampleButtonsContainer: document.getElementById('sample-buttons-container'),
  dropzone: document.getElementById('image-dropzone'),
  fileInput: document.getElementById('file-input'),
  btnToggleCamera: document.getElementById('btn-toggle-camera'),
  cameraModal: document.getElementById('camera-modal'),
  webcamVideo: document.getElementById('webcam-video'),
  cameraCanvas: document.getElementById('camera-canvas'),
  btnSnapPhoto: document.getElementById('btn-snap-photo'),
  btnCloseCamera: document.getElementById('btn-close-camera'),
  queryPreviewImg: document.getElementById('query-preview-img'),
  previewFilename: document.getElementById('preview-filename'),
  scanLine: document.getElementById('scan-line'),
  thresholdSlider: document.getElementById('threshold-slider'),
  thresholdVal: document.getElementById('threshold-val'),
  btnRunIdentify: document.getElementById('btn-run-identify'),

  // Verdict & Metrics
  verdictBox: document.getElementById('match-verdict-box'),
  verdictIcon: document.getElementById('verdict-icon'),
  verdictLabel: document.getElementById('verdict-label'),
  verdictName: document.getElementById('verdict-name'),
  verdictDesc: document.getElementById('verdict-desc'),
  metricDistance: document.getElementById('metric-distance'),
  metricDistanceHint: document.getElementById('metric-distance-hint'),
  metricSimilarity: document.getElementById('metric-similarity'),
  metricDim: document.getElementById('metric-dim'),
  metricStatus: document.getElementById('metric-status'),
  metricThresholdHint: document.getElementById('metric-threshold-hint'),
  executionTimeBadge: document.getElementById('execution-time-badge'),
  candidatesTbody: document.getElementById('candidates-tbody'),

  // Mini-strip
  stripInput: document.getElementById('strip-input'),
  stripGray: document.getElementById('strip-gray'),
  stripClahe: document.getElementById('strip-clahe'),
  stripBlur: document.getElementById('strip-blur'),
  stripSobel: document.getElementById('strip-sobel'),
  stripCanny: document.getElementById('strip-canny'),
  stripRoi: document.getElementById('strip-roi'),
  linkGotoPipeline: document.getElementById('link-goto-pipeline'),

  // Pipeline Inspector
  pipeAnnotated: document.getElementById('pipe-annotated'),
  pipeGrayscale: document.getElementById('pipe-grayscale'),
  pipeYuv: document.getElementById('pipe-yuv'),
  pipeHomomorphic: document.getElementById('pipe-homomorphic'),
  pipeHisteq: document.getElementById('pipe-histeq'),
  pipeClahe: document.getElementById('pipe-clahe'),
  pipeGauss: document.getElementById('pipe-gauss'),
  pipeMedian: document.getElementById('pipe-median'),
  pipeSobel: document.getElementById('pipe-sobel'),
  pipeCanny: document.getElementById('pipe-canny'),
  pipeRoi: document.getElementById('pipe-roi'),
  pipeBboxMeta: document.getElementById('pipe-bbox-meta'),

  // Charts & Features
  chartLbp: document.getElementById('chart-lbp'),
  chartGrad: document.getElementById('chart-grad'),
  hogSparkline: document.getElementById('hog-sparkline'),

  // Enrollment Tab
  enrollForm: document.getElementById('enroll-form'),
  enrollName: document.getElementById('enroll-name'),
  enrollDropzone: document.getElementById('enroll-dropzone'),
  enrollFileInput: document.getElementById('enroll-file-input'),
  enrollPreviewBox: document.getElementById('enroll-preview-box'),
  enrollPreviewImg: document.getElementById('enroll-preview-img'),
  usersRosterContainer: document.getElementById('users-roster-container'),
  btnClearDb: document.getElementById('btn-clear-db'),

  // Toast
  toast: document.getElementById('toast'),
  toastMessage: document.getElementById('toast-message'),
  toastIcon: document.getElementById('toast-icon'),
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  setupSamplePicker();
  setupDropzone();
  setupCamera();
  setupThresholdSlider();
  setupEnrollment();
  await refreshDatabaseUsers();

  // If no users, prompt to seed demo or auto-seed
  if (state.enrolledUsers.length === 0) {
    await seedDemoDatabase(false);
  }

  // Run initial test query on Alice Query sample
  await executeIdentification();
});

// ---------------------------------------------------------------------------
// Tab Management
// ---------------------------------------------------------------------------

function setupTabs() {
  elements.tabButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });

  if (elements.linkGotoPipeline) {
    elements.linkGotoPipeline.addEventListener('click', (e) => {
      e.preventDefault();
      switchTab('tab-pipeline');
    });
  }
}

function switchTab(tabId) {
  elements.tabButtons.forEach((b) => b.classList.remove('active'));
  elements.tabPanels.forEach((p) => p.classList.remove('active'));

  const activeBtn = document.querySelector(`[data-tab="${tabId}"]`);
  const activePanel = document.getElementById(tabId);

  if (activeBtn) activeBtn.classList.add('active');
  if (activePanel) activePanel.classList.add('active');

  // Trigger redraw of canvas charts if switching to features tab
  if (tabId === 'tab-features' && state.lastResult) {
    drawLbpChart(state.lastResult.pipeline.lbp_hist);
    drawGradChart(state.lastResult.pipeline.grad_hist);
  }
}

// ---------------------------------------------------------------------------
// Toast Notification
// ---------------------------------------------------------------------------

let toastTimeout = null;
function showToast(message, isSuccess = true) {
  if (toastTimeout) clearTimeout(toastTimeout);

  elements.toastMessage.textContent = message;
  elements.toastIcon.textContent = isSuccess ? '✓' : '⚠';
  elements.toastIcon.style.color = isSuccess ? 'var(--accent-emerald)' : 'var(--accent-rose)';
  elements.toast.style.borderColor = isSuccess ? 'var(--border-emerald)' : 'var(--border-rose)';

  elements.toast.classList.remove('hidden');

  toastTimeout = setTimeout(() => {
    elements.toast.classList.add('hidden');
  }, 3200);
}

// ---------------------------------------------------------------------------
// Sample Test Picker
// ---------------------------------------------------------------------------

async function setupSamplePicker() {
  try {
    const res = await fetch('/api/samples');
    const samples = await res.json();

    elements.sampleButtonsContainer.innerHTML = '';
    samples.forEach((s) => {
      const card = document.createElement('div');
      card.className = `sample-card ${s.key === state.currentSampleKey ? 'active' : ''}`;
      card.id = `sample-${s.key}`;
      card.innerHTML = `
        <img src="${s.image_url}" class="sample-thumb" alt="${s.name}">
        <span class="sample-name">${s.name}</span>
        <span class="sample-tag">${s.expected}</span>
      `;

      card.addEventListener('click', () => {
        document.querySelectorAll('.sample-card').forEach((c) => c.classList.remove('active'));
        card.classList.add('active');

        state.currentSampleKey = s.key;
        state.currentFile = null;
        state.currentDataUrl = null;

        elements.queryPreviewImg.src = s.image_url;
        elements.previewFilename.textContent = s.image_url;

        executeIdentification();
      });

      elements.sampleButtonsContainer.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load samples:', err);
  }
}

// ---------------------------------------------------------------------------
// Drag & Drop / File Input
// ---------------------------------------------------------------------------

function setupDropzone() {
  const dropzone = elements.dropzone;
  const fileInput = elements.fileInput;

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });
}

function handleSelectedFile(file) {
  state.currentFile = file;
  state.currentDataUrl = null;
  state.currentSampleKey = null;

  document.querySelectorAll('.sample-card').forEach((c) => c.classList.remove('active'));

  elements.previewFilename.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

  const reader = new FileReader();
  reader.onload = (e) => {
    elements.queryPreviewImg.src = e.target.result;
    executeIdentification();
  };
  reader.readAsDataURL(file);
}

// ---------------------------------------------------------------------------
// Camera Stream & Snapshot
// ---------------------------------------------------------------------------

function setupCamera() {
  elements.btnToggleCamera.addEventListener('click', async () => {
    if (state.cameraStream) {
      stopCamera();
    } else {
      await startCamera();
    }
  });

  elements.btnCloseCamera.addEventListener('click', stopCamera);

  elements.btnSnapPhoto.addEventListener('click', () => {
    if (!state.cameraStream) return;

    const video = elements.webcamVideo;
    const canvas = elements.cameraCanvas;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

    state.currentDataUrl = dataUrl;
    state.currentFile = null;
    state.currentSampleKey = null;

    document.querySelectorAll('.sample-card').forEach((c) => c.classList.remove('active'));

    elements.queryPreviewImg.src = dataUrl;
    elements.previewFilename.textContent = `Webcam Snapshot (${canvas.width}x${canvas.height})`;

    stopCamera();
    executeIdentification();
  });
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    });
    state.cameraStream = stream;
    elements.webcamVideo.srcObject = stream;
    elements.cameraModal.classList.remove('hidden');
    elements.btnToggleCamera.querySelector('span').textContent = 'Stop Camera';
  } catch (err) {
    showToast('Camera access denied or unavailable: ' + err.message, false);
  }
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  elements.cameraModal.classList.add('hidden');
  elements.btnToggleCamera.querySelector('span').textContent = 'Use Webcam';
}

// ---------------------------------------------------------------------------
// Threshold Slider
// ---------------------------------------------------------------------------

function setupThresholdSlider() {
  elements.thresholdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    state.threshold = val;
    elements.thresholdVal.textContent = val.toFixed(2);
    elements.metricThresholdHint.textContent = `Threshold: ${val.toFixed(2)}`;

    // Real-time recalculation of candidate decision without server roundtrip!
    if (state.lastResult && state.lastResult.candidates) {
      recalculateThresholdLocally(val);
    }
  });

  elements.btnRunIdentify.addEventListener('click', () => {
    executeIdentification();
  });
}

function recalculateThresholdLocally(newThreshold) {
  const candidates = state.lastResult.candidates;
  if (!candidates || candidates.length === 0) return;

  candidates.forEach((c) => {
    c.is_within_threshold = c.distance < newThreshold;
  });

  const best = candidates[0];
  const matched = best.distance < newThreshold;

  const matchData = {
    user_id: matched ? best.user_id : null,
    name: matched ? best.name : 'Unknown',
    closest_name: best.name,
    distance: best.distance,
    similarity: best.similarity,
    matched: matched,
  };

  updateVerdictUI(matchData, newThreshold);
  renderCandidatesTable(candidates, newThreshold);
}

// ---------------------------------------------------------------------------
// Core Identification Execution
// ---------------------------------------------------------------------------

async function executeIdentification() {
  elements.scanLine.classList.remove('hidden');
  elements.btnRunIdentify.disabled = true;

  const formData = new FormData();
  formData.append('threshold', state.threshold.toString());

  if (state.currentFile) {
    formData.append('file', state.currentFile);
  } else if (state.currentDataUrl) {
    formData.append('data_url', state.currentDataUrl);
  } else if (state.currentSampleKey) {
    formData.append('sample_key', state.currentSampleKey);
  } else {
    formData.append('sample_key', 'alice_query');
  }

  try {
    const res = await fetch('/api/identify', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Identification failed.');
    }

    state.lastResult = data;

    // Update Verdict & Metrics
    updateVerdictUI(data.match, data.threshold);
    elements.executionTimeBadge.textContent = `Elapsed: ${data.elapsed_ms} ms`;
    elements.metricDistance.textContent = data.match.distance !== Infinity ? data.match.distance.toFixed(4) : '∞';
    elements.metricSimilarity.textContent = data.match.similarity.toFixed(4);
    elements.metricDim.textContent = data.pipeline.template_len.toLocaleString();

    // Render Candidates Table
    renderCandidatesTable(data.candidates, data.threshold);

    // Update Mini-Strip
    elements.stripInput.src = data.pipeline.annotated_input;
    elements.stripGray.src = data.pipeline.grayscale;
    elements.stripClahe.src = data.pipeline.clahe;
    elements.stripBlur.src = data.pipeline.gaussian_blur;
    elements.stripSobel.src = data.pipeline.sobel_edges;
    elements.stripCanny.src = data.pipeline.canny_edges;
    elements.stripRoi.src = data.pipeline.normalized_roi;

    // Update Full Pipeline Inspector (Tab 2)
    updatePipelineTab(data.pipeline);

    // Update Charts (Tab 3)
    drawLbpChart(data.pipeline.lbp_hist);
    drawGradChart(data.pipeline.grad_hist);
    renderHogSparkline(data.pipeline.hog_sample);

    showToast(`Processed face in ${data.elapsed_ms}ms`, true);
  } catch (err) {
    showToast(err.message, false);
    console.error('Identification error:', err);
  } finally {
    elements.scanLine.classList.add('hidden');
    elements.btnRunIdentify.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// UI Rendering: Verdict & Candidates
// ---------------------------------------------------------------------------

function updateVerdictUI(match, threshold) {
  elements.verdictBox.classList.remove('verdict-idle', 'verdict-match', 'verdict-unknown');

  if (match.matched) {
    elements.verdictBox.classList.add('verdict-match');
    elements.verdictIcon.innerHTML = `
      <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
      </svg>
    `;
    elements.verdictLabel.textContent = 'Identity Verified';
    elements.verdictName.textContent = `MATCHED: ${match.name}`;
    elements.verdictDesc.textContent = `Euclidean distance (${match.distance.toFixed(3)}) is below acceptance threshold (${threshold.toFixed(2)}). Positive match confirmed.`;
    elements.metricStatus.textContent = 'MATCH CONFIRMED';
    elements.metricStatus.style.color = 'var(--accent-emerald)';
  } else {
    elements.verdictBox.classList.add('verdict-unknown');
    elements.verdictIcon.innerHTML = `
      <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
      </svg>
    `;
    elements.verdictLabel.textContent = 'Unrecognized Subject';
    elements.verdictName.textContent = 'UNKNOWN PERSON';
    elements.verdictDesc.textContent =
      match.closest_name !== 'None'
        ? `Closest candidate is ${match.closest_name} (dist=${match.distance.toFixed(3)}), but exceeds threshold (${threshold.toFixed(2)}). Rejected.`
        : 'No matching user found in database or database is empty.';
    elements.metricStatus.textContent = 'FLAGGED AS UNKNOWN';
    elements.metricStatus.style.color = 'var(--accent-rose)';
  }
}

function renderCandidatesTable(candidates, threshold) {
  if (!candidates || candidates.length === 0) {
    elements.candidatesTbody.innerHTML = `
      <tr>
        <td colspan="6" class="text-center text-muted py-8">
          No registered templates found in database.
        </td>
      </tr>
    `;
    return;
  }

  elements.candidatesTbody.innerHTML = '';
  candidates.forEach((c, index) => {
    const isWithin = c.distance < threshold;
    const barWidth = Math.min(100, Math.max(5, (c.distance / 6.0) * 100));
    const barColor = isWithin ? 'var(--accent-emerald)' : 'var(--accent-rose)';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>#${index + 1}</strong></td>
      <td>
        <span style="font-weight:700; color: ${isWithin ? 'var(--accent-emerald)' : 'var(--text-main)'}">
          ${c.name}
        </span>
        <span class="text-xs text-muted"> (ID: ${c.user_id})</span>
      </td>
      <td style="font-family: var(--font-mono); font-weight:600;">${c.distance.toFixed(4)}</td>
      <td style="font-family: var(--font-mono);">${c.similarity.toFixed(4)}</td>
      <td>
        <div class="dist-bar-track">
          <div class="dist-bar-fill" style="width: ${barWidth}%; background: ${barColor}"></div>
        </div>
      </td>
      <td>
        <span class="badge ${isWithin ? 'badge-accent' : 'badge-subtle'}" 
              style="color:${isWithin ? 'var(--accent-emerald)' : 'var(--accent-rose)'}; border-color:${isWithin ? 'var(--border-emerald)' : 'var(--border-rose)'}">
          ${isWithin ? 'MATCH' : 'REJECT'}
        </span>
      </td>
    `;
    elements.candidatesTbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------------------
// Pipeline Stages Inspector (Tab 2)
// ---------------------------------------------------------------------------

function updatePipelineTab(pipeline) {
  elements.pipeAnnotated.src = pipeline.annotated_input;
  elements.pipeGrayscale.src = pipeline.grayscale;
  elements.pipeYuv.src = pipeline.yuv_y;
  if (elements.pipeHomomorphic && pipeline.homomorphic) {
    elements.pipeHomomorphic.src = pipeline.homomorphic;
  }
  elements.pipeHisteq.src = pipeline.hist_equalized;
  elements.pipeClahe.src = pipeline.clahe;
  elements.pipeGauss.src = pipeline.gaussian_blur;
  elements.pipeMedian.src = pipeline.median_filter;
  elements.pipeSobel.src = pipeline.sobel_edges;
  elements.pipeCanny.src = pipeline.canny_edges;
  elements.pipeRoi.src = pipeline.normalized_roi;

  if (pipeline.bbox) {
    elements.pipeBboxMeta.textContent = `Bounding Box: [x=${pipeline.bbox.x}, y=${pipeline.bbox.y}, w=${pipeline.bbox.w}, h=${pipeline.bbox.h}]`;
  }
}

// ---------------------------------------------------------------------------
// Canvas Histograms Rendering (Tab 3)
// ---------------------------------------------------------------------------

function drawLbpChart(data) {
  if (!data || !elements.chartLbp) return;
  const canvas = elements.chartLbp;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  const maxVal = Math.max(...data, 0.001);
  const barWidth = w / data.length;

  // Background grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    const y = (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw histogram bars
  const gradient = ctx.createLinearGradient(0, h, 0, 0);
  gradient.addColorStop(0, '#8b5cf6');
  gradient.addColorStop(1, '#06b6d4');

  ctx.fillStyle = gradient;
  for (let i = 0; i < data.length; i++) {
    const barHeight = (data[i] / maxVal) * (h - 10);
    ctx.fillRect(i * barWidth, h - barHeight, barWidth - 0.5, barHeight);
  }
}

function drawGradChart(data) {
  if (!data || !elements.chartGrad) return;
  const canvas = elements.chartGrad;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  const maxVal = Math.max(...data, 0.001);
  const barWidth = w / data.length;

  // Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    const y = (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Bars
  const gradient = ctx.createLinearGradient(0, h, 0, 0);
  gradient.addColorStop(0, '#10b981');
  gradient.addColorStop(1, '#3b82f6');

  ctx.fillStyle = gradient;
  for (let i = 0; i < data.length; i++) {
    const barHeight = (data[i] / maxVal) * (h - 15);
    ctx.fillRect(i * barWidth + 2, h - barHeight, barWidth - 4, barHeight);
  }
}

function renderHogSparkline(hogSample) {
  if (!hogSample || !elements.hogSparkline) return;
  elements.hogSparkline.innerHTML = '';
  const maxVal = Math.max(...hogSample, 0.01);

  hogSample.forEach((val) => {
    const bar = document.createElement('div');
    bar.className = 'spark-bar';
    const heightPercent = Math.max(3, (val / maxVal) * 100);
    bar.style.height = `${heightPercent}%`;
    elements.hogSparkline.appendChild(bar);
  });
}

// ---------------------------------------------------------------------------
// Database & User Enrollment (Tab 4)
// ---------------------------------------------------------------------------

function setupEnrollment() {
  let enrollFile = null;

  elements.enrollDropzone.addEventListener('click', () => elements.enrollFileInput.click());

  elements.enrollFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      enrollFile = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (ev) => {
        elements.enrollPreviewImg.src = ev.target.result;
        elements.enrollPreviewBox.classList.remove('hidden');
      };
      reader.readAsDataURL(enrollFile);
    }
  });

  elements.enrollForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = elements.enrollName.value.trim();
    if (!name) return;

    if (!enrollFile) {
      showToast('Please select or upload a face portrait image.', false);
      return;
    }

    const formData = new FormData();
    formData.append('name', name);
    formData.append('file', enrollFile);

    try {
      const res = await fetch('/api/enroll', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Enrollment failed.');

      showToast(`User "${data.name}" enrolled successfully with ID=${data.user_id}!`);
      elements.enrollForm.reset();
      enrollFile = null;
      elements.enrollPreviewBox.classList.add('hidden');

      await refreshDatabaseUsers();
      executeIdentification();
    } catch (err) {
      showToast(err.message, false);
    }
  });

  elements.btnSeedDemo.addEventListener('click', async () => {
    await seedDemoDatabase(true);
  });

  elements.btnClearDb.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear all enrolled users from the SQLite database?')) return;
    try {
      const res = await fetch('/api/users/clear', { method: 'POST' });
      const data = await res.json();
      showToast(`Cleared ${data.cleared_count} users from database.`);
      await refreshDatabaseUsers();
      executeIdentification();
    } catch (err) {
      showToast('Failed to clear database.', false);
    }
  });
}

async function seedDemoDatabase(notify = true) {
  try {
    const res = await fetch('/api/seed_demo', { method: 'POST' });
    const data = await res.json();
    if (notify) showToast('Database seeded with Alice and Bob portraits.');
    await refreshDatabaseUsers();
    await executeIdentification();
  } catch (err) {
    showToast('Failed to seed demo database: ' + err.message, false);
  }
}

async function refreshDatabaseUsers() {
  try {
    const res = await fetch('/api/users');
    const users = await res.json();
    state.enrolledUsers = users;

    elements.dbUserCount.textContent = users.length;

    if (!elements.usersRosterContainer) return;

    if (users.length === 0) {
      elements.usersRosterContainer.innerHTML = `
        <div class="text-center text-muted py-8">
          No subjects enrolled yet. Use the form above or click "Seed Demo DB".
        </div>
      `;
      return;
    }

    elements.usersRosterContainer.innerHTML = '';
    users.forEach((u) => {
      const card = document.createElement('div');
      card.className = 'roster-card';
      card.innerHTML = `
        <div class="roster-user-info">
          <img src="/static/thumb_user_${u.id}.png" class="roster-avatar" onerror="this.src='/sample_data/alice_enroll.jpg'" alt="${u.name}">
          <div>
            <div class="roster-name">${u.name}</div>
            <div class="roster-meta">ID: ${u.id} | Template: ${u.template_len.toLocaleString()} dims | Enrolled: ${new Date(u.enrolled_at).toLocaleDateString()}</div>
          </div>
        </div>
        <button class="btn-delete-user" data-id="${u.id}" title="Delete User">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
        </button>
      `;

      card.querySelector('.btn-delete-user').addEventListener('click', async () => {
        if (!confirm(`Delete user "${u.name}" (ID=${u.id})?`)) return;
        try {
          const delRes = await fetch(`/api/users/${u.id}`, { method: 'DELETE' });
          if (delRes.ok) {
            showToast(`User ${u.name} deleted.`);
            await refreshDatabaseUsers();
            executeIdentification();
          }
        } catch (err) {
          showToast('Failed to delete user.', false);
        }
      });

      elements.usersRosterContainer.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load users:', err);
  }
}
