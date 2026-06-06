// MATHIR Vision Testing UI v4
// - Chat history persistence via localStorage + API
// - Camera redesign with clear controls
// - SVG-first icons (no emoji)
// - All 6 views: chat, camera, models, memory, accuracy, settings

const API = '';

// ============ STATE ============
const state = {
  activeView: 'chat',
  models: [],
  activeModel: null,
  mediaRecorder: null,
  audioChunks: [],
  isRecording: false,
  cameraRunning: false,
  chatAttachments: [],
  testsData: null,
  systemOk: false,
  chatHistory: [],      // persisted to localStorage
  lastCameraFrame: null, // base64 last frame for chat context
};

// ============ LOCAL STORAGE: CHAT HISTORY ============
const HISTORY_KEY = 'mathir_chat_history';
const MAX_HISTORY = 50;

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveHistory(history) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch { /* storage full */ }
}

function addToHistory(role, text, attachments = []) {
  state.chatHistory.push({ role, text, attachments, ts: Date.now() });
  if (state.chatHistory.length > MAX_HISTORY) {
    state.chatHistory = state.chatHistory.slice(-MAX_HISTORY);
  }
  saveHistory(state.chatHistory);
}

// ============ UTILITIES ============
async function api(path, options = {}) {
  const opts = {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  };
  if (opts.body && typeof opts.body !== 'string') {
    opts.body = JSON.stringify(opts.body);
  }
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let err = r.statusText;
    try { err = (await r.json()).error || err; } catch (_) {}
    throw new Error(err);
  }
  return r.json();
}

function $(id) { return document.getElementById(id); }
function $$(sel) { return document.querySelectorAll(sel); }

function escapeHtml(s) {
  return String(s).replace(/[<>&"']/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'}[c]));
}

function toast(msg, type = 'info', duration = 4000) {
  const c = $('toast-container');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

function formatTs(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Markdown-lite: bold, italic, code, code blocks, line breaks
function renderMd(text) {
  if (!text) return '';
  text = escapeHtml(text);
  // Code blocks (``` ... ```)
  text = text.replace(/```[\s\S]*?```/g, m => {
    const code = m.slice(3, -3).replace(/```\w*\n?/, '').trim();
    return `<pre style="background:var(--bg-primary);border:1px solid var(--border);border-radius:6px;padding:10px;overflow-x:auto;font-size:12px;margin:6px 0">${code}</pre>`;
  });
  // Inline code (`code`)
  text = text.replace(/`([^`]+)`/g, '<code style="background:var(--bg-primary);padding:1px 5px;border-radius:4px;font-size:12px">$1</code>');
  // Bold (**text**)
  text = text.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  // Italic (*text*)
  text = text.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  // Line breaks
  text = text.replace(/\n/g, '<br>');
  return text;
}

// ============ VIEW NAVIGATION ============
function switchView(view) {
  state.activeView = view;
  $$('.view').forEach(v => v.classList.remove('active'));
  $$('.nav-item').forEach(n => n.classList.remove('active'));
  const vEl = $(`view-${view}`);
  const nEl = $(`main-nav`).querySelector(`[data-view="${view}"]`);
  if (vEl) vEl.classList.add('active');
  if (nEl) nEl.classList.add('active');
  if (view === 'memory') loadMemoryStats();
  if (view === 'accuracy') loadAccuracy();
  if (view === 'models') loadModels();
}

// ============ INIT ============
async function init() {
  // Load persisted chat history
  state.chatHistory = loadHistory();
  renderChatHistory();

  // Navigation
  $$('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  // Keyboard shortcuts 1-6
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const map = { '1':'chat','2':'camera','3':'models','4':'memory','5':'accuracy','6':'settings' };
    if (map[e.key]) switchView(map[e.key]);
  });

  // Global keyboard: Ctrl+K = focus chat
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      switchView('chat');
      $('chat-input')?.focus();
    }
    if (e.key === 'Escape') {
      // Unfocus any focused input
      document.activeElement?.blur();
    }
  });

  setupChat();
  setupCameraView();
  setupModelsView();
  setupMemoryView();
  setupAccuracyView();
  setupSettingsView();
  setupModal();

  // System check
  checkSystem();
  setInterval(checkSystem, 30000);
}

// ============ SYSTEM CHECK ============
async function checkSystem() {
  try {
    // /api/system/context returns {available_models, active_model, platform, date, model_info}
    const ctx = await api('/api/system/context');
    state.systemOk = true;
    $('system-status').textContent = 'System ready';
    $('system-status').style.color = 'var(--accent-green)';

    // Populate model select
    const models = ctx.available_models || [];
    state.models = models;
    populateModelSelect(models, ctx.active_model || '');

    // System info in settings — use /api/system/info
    try {
      const info = await api('/api/system/info');
      const si = $('system-info');
      if (si) {
        si.innerHTML = `
          <p>Platform: ${escapeHtml(info.platform || '?')}</p>
          <p>Python: ${escapeHtml(info.python || '?')}</p>
          <p>Active model: ${escapeHtml(info.active_model || 'none')}</p>
          <p>Camera running: ${info.camera_running ? 'Yes' : 'No'}</p>
          <p>Config: ${escapeHtml(info.config_path || '?')}</p>
          <p>Memory DB: ${escapeHtml(info.memory_db_path || '?')}</p>
        `;
      }
    } catch { /* ignore info errors */ }
  } catch {
    state.systemOk = false;
    $('system-status').textContent = 'System offline';
    $('system-status').style.color = 'var(--accent-red)';
  }
}

function populateModelSelect(models, active) {
  const sel = $('model-select');
  if (!sel) return;
  sel.innerHTML = '';
  if (!models || models.length === 0) {
    sel.innerHTML = '<option value="">No models</option>';
    return;
  }
  models.forEach(m => {
    const o = document.createElement('option');
    o.value = m.name;
    o.textContent = m.name;
    if (m.name === active) o.selected = true;
    sel.appendChild(o);
  });
  state.activeModel = active;
  sel.onchange = () => selectModel(sel.value);

  // Badges
  const badges = $('active-model-badges');
  if (badges) {
    badges.innerHTML = '';
    models.filter(m => m.loaded).forEach(m => {
      const b = document.createElement('span');
      b.className = 'model-badge';
      b.textContent = m.name.split('_')[0] || m.name;
      badges.appendChild(b);
    });
  }
}

async function selectModel(name) {
  try {
    await api('/api/models/switch', { method: 'POST', body: { name } });
    state.activeModel = name;
    toast(`Switched to ${name}`, 'success');
  } catch (e) {
    toast('Failed to switch model: ' + e.message, 'error');
  }
}

// ============ CHAT ============
function setupChat() {
  const input = $('chat-input');
  const sendBtn = $('send-btn');
  const talkBtn = $('talk-btn');
  const attachBtn = $('attach-image-btn');
  const clearBtn = $('clear-chat');
  const chatCamStart = $('chat-start-cam');
  const chatCamStop = $('chat-stop-cam');

  // Auto-resize textarea
  input?.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 140) + 'px';
  });

  // Send on Enter (not Shift+Enter)
  input?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  sendBtn?.addEventListener('click', sendMessage);
  clearBtn?.addEventListener('click', clearChat);

  // Attach image
  attachBtn?.addEventListener('click', () => {
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = 'image/*';
    inp.onchange = () => {
      if (inp.files[0]) addAttachment(inp.files[0]);
    };
    inp.click();
  });

  // Camera in chat mode
  chatCamStart?.addEventListener('click', startCameraInChat);
  chatCamStop?.addEventListener('click', stopCameraInChat);

  // Talk button (hold to record)
  talkBtn?.addEventListener('mousedown', startRecording);
  talkBtn?.addEventListener('mouseup', stopRecording);
  talkBtn?.addEventListener('mouseleave', stopRecording);
  talkBtn?.addEventListener('touchstart', e => { e.preventDefault(); startRecording(); });
  talkBtn?.addEventListener('touchend', stopRecording);
}

function renderChatHistory() {
  const box = $('chat-messages');
  if (!box) return;
  box.innerHTML = '';

  if (state.chatHistory.length === 0) {
    box.innerHTML = `<div class="message system"><div class="message-text">Welcome. Type a message, attach an image, or click the mic to talk.</div></div>`;
    return;
  }

  state.chatHistory.forEach(msg => {
    appendMessage(msg.role, msg.text, msg.attachments, false);
  });
  box.scrollTop = box.scrollHeight;
}

function appendMessage(role, text, attachments = [], save = true) {
  const box = $('chat-messages');
  if (!box) return;

  // Remove welcome message if present
  const welcome = box.querySelector('.message.system .message-text');
  if (welcome && welcome.textContent.includes('Welcome')) {
    box.innerHTML = '';
  }

  const div = document.createElement('div');
  div.className = `message ${role}`;

  const roleLabel = role === 'user' ? 'You' : role === 'assistant' ? 'Model' : 'System';
  let html = `<div class="message-role">${roleLabel}</div>`;
  html += `<div class="message-text">${renderMd(text)}</div>`;

  if (attachments && attachments.length > 0) {
    attachments.forEach(att => {
      if (att.type === 'image' && att.data) {
        html += `<img class="message-img" src="${att.data}" alt="attachment" onclick="window.open(this.src)">`;
      }
    });
  }

  div.innerHTML = html;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;

  if (save) {
    addToHistory(role, text, attachments);
  }
}

function appendStatus(text, type = 'info') {
  const box = $('chat-messages');
  if (!box) return;
  const div = document.createElement('div');
  div.className = `message system`;
  div.innerHTML = `<div class="message-text" style="color:var(--text-muted);font-size:12px">${escapeHtml(text)}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  $('chat-status').textContent = text;
}

function clearChat() {
  state.chatHistory = [];
  saveHistory([]);
  $('chat-messages').innerHTML = `<div class="message system"><div class="message-text">Welcome. Type a message, attach an image, or click the mic to talk.</div></div>`;
  $('chat-attachments').innerHTML = '';
  $('chat-attachments').hidden = true;
  state.chatAttachments = [];
  toast('Chat cleared', 'info');
}

function addAttachment(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const data = e.target.result;
    state.chatAttachments.push({ type: 'image', data, name: file.name });
    renderAttachmentStrip();
  };
  reader.readAsDataURL(file);
}

function renderAttachmentStrip() {
  const strip = $('chat-attachments');
  if (!strip) return;
  if (state.chatAttachments.length === 0) {
    strip.innerHTML = '';
    strip.hidden = true;
    return;
  }
  strip.hidden = false;
  strip.innerHTML = '';
  state.chatAttachments.forEach((att, i) => {
    const item = document.createElement('div');
    item.className = 'attachment-item';
    if (att.type === 'image') {
      item.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
        <span>${escapeHtml(att.name || 'image')}</span>
        <button class="attachment-remove" data-idx="${i}">&times;</button>
      `;
    }
    strip.appendChild(item);
  });
  strip.querySelectorAll('.attachment-remove').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = parseInt(btn.dataset.idx);
      state.chatAttachments.splice(idx, 1);
      renderAttachmentStrip();
    });
  });
}

async function sendMessage() {
  const input = $('chat-input');
  if (!input) return;
  const text = input.value.trim();
  if (!text && state.chatAttachments.length === 0) return;

  const sendBtn = $('send-btn');
  sendBtn.disabled = true;
  input.value = '';
  input.style.height = 'auto';

  const userText = text;
  const attachments = [...state.chatAttachments];

  // Show user message
  appendMessage('user', userText, attachments);
  $('chat-status').textContent = 'Sending...';

  try {
    const body = { message: userText };
    // Backend expects `image` (base64) and `audio` (base64), not attachments array
    if (attachments.length > 0) {
      const imgAtt = attachments.find(a => a.type === 'image');
      if (imgAtt && imgAtt.data) {
        // Strip data:image/xxx;base64, prefix if present
        body.image = imgAtt.data.includes(',') ? imgAtt.data.split(',')[1] : imgAtt.data;
      }
    }
    if (state.lastCameraFrame) {
      body.image = state.lastCameraFrame.includes(',') ? state.lastCameraFrame.split(',')[1] : state.lastCameraFrame;
    }

    const data = await api('/api/chat', {
      method: 'POST',
      body,
    });

    appendMessage('assistant', data.response || 'No response.');
    $('chat-status').textContent = '';

    // TTS
    if (data.tts_audio) {
      const audio = $('tts-audio');
      if (audio) { audio.src = data.tts_audio; audio.play().catch(() => {}); }
    }
  } catch (e) {
    appendMessage('assistant', 'Error: ' + e.message, []);
    $('chat-status').textContent = '';
  } finally {
    sendBtn.disabled = false;
    state.chatAttachments = [];
    renderAttachmentStrip();
  }
}

// ============ CAMERA IN CHAT MODE ============
async function startCameraInChat() {
  try {
    // Start backend camera if not already running
    if (!state.cameraRunning) {
      await api('/api/camera/start', { method: 'POST', body: {} });
      state.cameraRunning = true;
    }

    const strip = $('camera-preview-strip');
    const img = $('chat-camera-preview');
    if (strip && img) {
      strip.hidden = false;
      img.src = '/api/camera/frame?' + Date.now();
      $('chat-start-cam').hidden = true;
      $('chat-stop-cam').hidden = false;
      $('chat-cam-status').textContent = 'Camera active';

      // Refresh preview every 2s and capture for chat context
      state._chatCamInterval = setInterval(() => {
        if (!state.cameraRunning) { stopCameraInChat(); return; }
        img.src = '/api/camera/frame?' + Date.now();
        captureBackendFrame().then(b64 => { if (b64) state.lastCameraFrame = 'data:image/jpeg;base64,' + b64; });
      }, 2000);

      // Initial capture
      captureBackendFrame().then(b64 => { if (b64) state.lastCameraFrame = 'data:image/jpeg;base64,' + b64; });
    }
  } catch (e) {
    toast('Camera error: ' + e.message, 'error');
  }
}

function stopCameraInChat() {
  if (state._chatCamInterval) { clearInterval(state._chatCamInterval); state._chatCamInterval = null; }
  state.lastCameraFrame = null;
  $('camera-preview-strip').hidden = true;
  $('chat-start-cam').hidden = false;
  $('chat-stop-cam').hidden = true;
  $('chat-cam-status').textContent = '';
}

// ============ AUDIO RECORDING ============
async function startRecording() {
  if (state.isRecording) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    state.audioChunks = [];
    state.mediaRecorder.ondataavailable = e => { if (e.data.size > 0) state.audioChunks.push(e.data); };
    state.mediaRecorder.start();
    state.isRecording = true;
    $('talk-btn')?.classList.add('recording');
    $('chat-status').textContent = 'Recording...';
  } catch (e) {
    toast('Mic error: ' + e.message, 'error');
  }
}

async function stopRecording() {
  if (!state.isRecording || !state.mediaRecorder) return;
  state.mediaRecorder.stop();
  state.mediaRecorder.stream.getTracks().forEach(t => t.stop());
  state.isRecording = false;
  $('talk-btn')?.classList.remove('recording');
  $('chat-status').textContent = 'Processing...';

  const blob = new Blob(state.audioChunks, { type: 'audio/webm' });
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const data = await api('/api/chat', {
        method: 'POST',
        body: { message: '', audio: reader.result },
      });
      if (data.response) {
        appendMessage('assistant', data.response);
        $('chat-status').textContent = '';
      } else {
        $('chat-status').textContent = '';
        toast('No response from audio', 'warning');
      }
    } catch (e) {
      $('chat-status').textContent = '';
      toast('Audio processing failed: ' + e.message, 'error');
    }
  };
  reader.readAsDataURL(blob);
}

// ============ CAMERA VIEW ============
function setupCameraView() {
  const camToggle = $('camera-toggle');
  const camSnapshot = $('cam-snapshot');
  const camTalk = $('cam-talk-btn');
  const camDescribe = $('cam-describe');
  const camAsk = $('cam-ask');
  const camAskInput = $('cam-ask-input');

  camToggle?.addEventListener('click', toggleMainCamera);
  camSnapshot?.addEventListener('click', takeSnapshot);
  camDescribe?.addEventListener('click', describeScene);
  camAsk?.addEventListener('click', () => askCamera(camAskInput.value));
  camAskInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); camAsk?.click(); }
  });

  camTalk?.addEventListener('mousedown', () => startCamRecording(camTalk));
  camTalk?.addEventListener('mouseup', () => stopCamRecording());
  camTalk?.addEventListener('mouseleave', () => stopCamRecording());
}

async function toggleMainCamera() {
  if (state.cameraRunning) {
    stopMainCamera();
  } else {
    await startMainCamera();
  }
}

async function startMainCamera() {
  try {
    const btn = $('camera-toggle');
    btn.disabled = true;
    btn.textContent = 'Starting...';

    // Start backend camera via API
    await api('/api/camera/start', { method: 'POST', body: {} });

    state.cameraRunning = true;
    const img = $('camera-stream');
    const placeholder = $('camera-placeholder');

    // Use MJPEG stream from backend
    if (img) {
      img.src = '/api/camera/stream?' + Date.now(); // cache-bust
      img.style.display = 'block';
      if (placeholder) placeholder.style.display = 'none';
    }

    if (btn) {
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg> Stop Camera`;
      btn.disabled = false;
    }

    // Enable action buttons
    $('cam-snapshot').disabled = false;
    $('cam-talk-btn').disabled = false;
    $('cam-describe').disabled = false;
    $('cam-ask').disabled = false;

    // Also update chat camera preview
    startChatCamPreview();

    toast('Camera started', 'success');
  } catch (e) {
    toast('Camera error: ' + e.message, 'error');
    const btn = $('camera-toggle');
    if (btn) {
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Camera`;
      btn.disabled = false;
    }
  }
}

function stopMainCamera() {
  // Stop backend camera
  api('/api/camera/stop', { method: 'POST' }).catch(() => {});

  state.cameraRunning = false;
  const img = $('camera-stream');
  const placeholder = $('camera-placeholder');
  if (img) { img.src = ''; img.style.display = 'none'; }
  if (placeholder) placeholder.style.display = 'flex';

  const btn = $('camera-toggle');
  if (btn) {
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Camera`;
  }

  $('cam-snapshot').disabled = true;
  $('cam-talk-btn').disabled = true;
  $('cam-describe').disabled = true;
  $('cam-ask').disabled = true;

  stopChatCamPreview();
  toast('Camera stopped', 'info');
}

async function takeSnapshot() {
  try {
    const resp = await fetch('/api/camera/frame');
    if (!resp.ok) throw new Error('No frame available');
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mathir_snapshot_${Date.now()}.jpg`;
    a.click();
    URL.revokeObjectURL(url);
    toast('Snapshot saved', 'success');
  } catch (e) {
    toast('Snapshot failed: ' + e.message, 'error');
  }
}

async function describeScene() {
  $('camera-response').innerHTML = `<div class="empty-state"><p>Analyzing scene...</p></div>`;
  try {
    const frame = await captureBackendFrame();
    if (!frame) throw new Error('No frame available');
    const data = await api('/api/chat', {
      method: 'POST',
      body: { message: 'Describe what you see in this image in detail.', image: frame },
    });
    $('camera-response').innerHTML = `<div class="response-text">${renderMd(data.response || 'No response.')}</div>`;
  } catch (e) {
    $('camera-response').innerHTML = `<div class="response-error">Error: ${escapeHtml(e.message)}</div>`;
  }
}

async function askCamera(question) {
  if (!question.trim()) return;
  $('camera-response').innerHTML = `<div class="empty-state"><p>Thinking...</p></div>`;
  try {
    const frame = await captureBackendFrame();
    const body = { message: question };
    if (frame) body.image = frame;
    const data = await api('/api/chat', { method: 'POST', body });
    $('camera-response').innerHTML = `<div class="response-text">${renderMd(data.response || 'No response.')}</div>`;
    $('cam-ask-input').value = '';
  } catch (e) {
    $('camera-response').innerHTML = `<div class="response-error">Error: ${escapeHtml(e.message)}</div>`;
  }
}

// Capture a frame from the backend camera as base64
async function captureBackendFrame() {
  try {
    const resp = await fetch('/api/camera/frame');
    if (!resp.ok) return null;
    const blob = await resp.blob();
    return new Promise(resolve => {
      const reader = new FileReader();
      reader.onload = () => {
        const b64 = reader.result.split(',')[1];
        resolve(b64);
      };
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(blob);
    });
  } catch { return null; }
}

// Chat camera preview — polls backend camera frames
function startChatCamPreview() {
  if (!state.cameraRunning) return;
  const strip = $('camera-preview-strip');
  const img = $('chat-camera-preview');
  if (strip && img) {
    strip.hidden = false;
    img.src = '/api/camera/frame?' + Date.now();
    $('chat-start-cam').hidden = true;
    $('chat-stop-cam').hidden = false;
    $('chat-cam-status').textContent = 'Camera active';

    // Refresh preview every 2s
    state._chatCamInterval = setInterval(() => {
      if (!state.cameraRunning) { stopChatCamPreview(); return; }
      img.src = '/api/camera/frame?' + Date.now();
      // Also store frame for chat context
      captureBackendFrame().then(b64 => { if (b64) state.lastCameraFrame = 'data:image/jpeg;base64,' + b64; });
    }, 2000);

    // Initial capture for chat context
    captureBackendFrame().then(b64 => { if (b64) state.lastCameraFrame = 'data:image/jpeg;base64,' + b64; });
  }
}

function stopChatCamPreview() {
  if (state._chatCamInterval) { clearInterval(state._chatCamInterval); state._chatCamInterval = null; }
  state.lastCameraFrame = null;
  $('camera-preview-strip').hidden = true;
  $('chat-start-cam').hidden = false;
  $('chat-stop-cam').hidden = true;
  $('chat-cam-status').textContent = '';
}

function startCamRecording(btn) {
  if (!state.cameraRunning) { toast('Start camera first', 'warning'); return; }
  startRecording();
  btn?.classList.add('recording');
}

function stopCamRecording() {
  stopRecording();
  $('cam-talk-btn')?.classList.remove('recording');
}

// Copy response button
$('cam-copy-response')?.addEventListener('click', () => {
  const text = $('camera-response')?.textContent || '';
  navigator.clipboard.writeText(text).then(() => toast('Copied!', 'success')).catch(() => toast('Copy failed', 'error'));
});

// ============ MODELS VIEW ============
async function loadModels() {
  const list = $('models-list');
  if (!list) return;
  list.innerHTML = '<p class="muted">Loading...</p>';
  try {
    const data = await api('/api/models');
    const models = data.models || [];
    if (models.length === 0) {
      list.innerHTML = '<p class="muted">No models found. Add models via HuggingFace.</p>';
      return;
    }
    list.innerHTML = '';
    models.forEach(m => {
      const card = document.createElement('div');
      card.className = 'model-card';
      const icon = m.supports_vision ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`
                  : m.supports_audio ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>`
                  : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>`;
      const sizeLabel = m.size_mb ? `${(m.size_mb / 1024).toFixed(1)}GB` : m.vram_mb ? `${(m.vram_mb / 1024).toFixed(1)}GB VRAM` : '';
      card.innerHTML = `
        <div class="model-card-icon">${icon}</div>
        <div class="model-card-body">
          <h3>${escapeHtml(m.display_name || m.name)}</h3>
          <p>${escapeHtml(m.type || 'unknown')} · ${escapeHtml(sizeLabel)} · ${escapeHtml((m.modalities || []).join(', '))}</p>
          <p class="muted">${m.active ? '<span class="status-dot"></span> Active' : m.enabled ? '<span class="status-dot" style="background:var(--accent-amber)"></span> Enabled' : '<span class="status-dot" style="background:var(--text-muted)"></span> Disabled'}</p>
        </div>
        <div class="model-card-actions">
          ${!m.active && m.enabled ? `<button class="btn btn-small" onclick="selectModel('${escapeHtml(m.name)}')">Activate</button>` : ''}
          ${!m.enabled ? `<button class="btn btn-small" onclick="loadModel('${escapeHtml(m.name)}')">Enable</button>` : ''}
          ${m.active ? `<button class="btn btn-small" disabled>Active</button>` : ''}
        </div>
      `;
      list.appendChild(card);
    });
  } catch (e) {
    list.innerHTML = `<p class="muted">Error loading models: ${escapeHtml(e.message)}</p>`;
  }
}

async function loadModel(name) {
  try {
    toast(`Enabling ${name}...`, 'info');
    await api('/api/models/toggle', { method: 'POST', body: { name, enabled: true } });
    toast(`${name} enabled`, 'success');
    loadModels();
    checkSystem();
  } catch (e) {
    toast('Enable failed: ' + e.message, 'error');
  }
}

function setupModelsView() {
  $('add-hf-btn')?.addEventListener('click', () => {
    $('hf-modal').classList.remove('hidden');
    $('hf-url').value = '';
    $('hf-name').value = '';
    $('hf-status').textContent = '';
  });
}

// ============ MEMORY VIEW ============
function setupMemoryView() {
  $('memory-recall-btn')?.addEventListener('click', doMemoryRecall);
  $('memory-query-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') doMemoryRecall();
  });
  $('memory-stats-btn')?.addEventListener('click', loadMemoryStats);
}

async function doMemoryRecall() {
  const query = $('memory-query-input')?.value.trim();
  if (!query) return;
  const results = $('memory-results');
  if (!results) return;
  results.innerHTML = '<div class="empty-state-small">Searching...</div>';
  try {
    const data = await api('/api/memory/recall', {
      method: 'POST',
      body: { query },
    });
    if (!data.results || data.results.length === 0) {
      results.innerHTML = `<div class="empty-state"><p>No memories found for "${escapeHtml(query)}"</p></div>`;
      return;
    }
    results.innerHTML = '';
    data.results.forEach(r => {
      const item = document.createElement('div');
      item.className = 'memory-result-item';
      item.innerHTML = `
        <div class="mem-type">${escapeHtml(r.model || 'memory')}</div>
        <div class="mem-content">${renderMd(r.text || '')}</div>
        <div class="mem-meta">Score: ${(r.score || 0).toFixed(3)} ${r.timestamp ? '· ' + formatTs(r.timestamp) : ''}</div>
      `;
      results.appendChild(item);
    });
  } catch (e) {
    results.innerHTML = `<div class="empty-state"><p>Error: ${escapeHtml(e.message)}</p></div>`;
  }
}

async function loadMemoryStats() {
  const stats = $('memory-stats');
  if (!stats) return;
  try {
    const data = await api('/api/memory/stats');
    stats.innerHTML = `<p>Total entries: ${data.count || 0}</p>`;
  } catch {
    stats.innerHTML = '<p>Could not load memory stats.</p>';
  }
}

// ============ ACCURACY VIEW ============
function setupAccuracyView() {
  // Tab switching
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      $$('.tab-btn').forEach(b => b.classList.remove('active'));
      $$('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      $(`accuracy-tab-${tab}`)?.classList.add('active');
    });
  });

  $('accuracy-run')?.addEventListener('click', runAccuracy);
  $('accuracy-refresh')?.addEventListener('click', loadAccuracy);
}

async function loadAccuracy() {
  try {
    // Load results + catalog in parallel
    const [resultsResp, catalogResp] = await Promise.all([
      api('/api/accuracy/results').catch(() => ({ models: {} })),
      api('/api/accuracy/tests').catch(() => ({ images: [] })),
    ]);

    // Backend returns {models: {modelName: {per_test: [...], overall_score, ...}}}
    // Flatten to array for frontend consumption
    const flatResults = [];
    const models = resultsResp.models || {};
    for (const [modelName, modelData] of Object.entries(models)) {
      const perTest = modelData.per_test || [];
      perTest.forEach(t => {
        flatResults.push({
          model: modelName,
          test_name: t.test_name || t.name || t.image || 'unknown',
          type: t.category || t.type || 'unknown',
          score: t.score != null ? t.score : (t.correct ? 1 : 0),
          response: t.response || t.model_response || '',
          expected: t.expected || t.ground_truth || '',
        });
      });
    }

    // Build summary
    const modelNames = Object.keys(models);
    const allScores = flatResults.map(r => r.score).filter(s => s != null);
    const summary = {
      avg_score: allScores.length ? allScores.reduce((a, b) => a + b, 0) / allScores.length : null,
      tests_run: flatResults.length,
      passed: flatResults.filter(r => r.score >= 0.5).length,
      models_tested: modelNames.length,
    };

    state.testsData = {
      results: flatResults,
      catalog: catalogResp.images || [],
      summary,
    };
    renderAccuracySummary(state.testsData);
    renderAccuracyCompare(state.testsData);
    renderAccuracyMatrix(state.testsData);
    renderAccuracyCatalog(state.testsData);
    populateAccuracyModelSelect(state.testsData);
  } catch (e) {
    $('accuracy-summary').innerHTML = `<div class="empty-state-small">Error loading: ${escapeHtml(e.message)}</div>`;
  }
}

async function runAccuracy() {
  const btn = $('accuracy-run');
  if (!btn) return;
  btn.disabled = true;
  $('accuracy-status').textContent = 'Running...';
  try {
    // /api/accuracy/test runs in background thread, returns immediately
    await api('/api/accuracy/test', { method: 'POST', body: {} });
    toast('Accuracy battery started. This may take a few minutes...', 'info');
    // Track model count before run
    const startResp = await api('/api/accuracy/results').catch(() => ({ models: {} }));
    const startModelCount = Object.keys(startResp.models || {}).length;
    const poll = setInterval(async () => {
      try {
        const current = await api('/api/accuracy/results');
        const currentModelCount = Object.keys(current.models || {}).length;
        $('accuracy-status').textContent = `Running... (${currentModelCount} models tested)`;
        // If new model results appeared, consider done
        if (currentModelCount > startModelCount) {
          clearInterval(poll);
          $('accuracy-status').textContent = 'Done';
          btn.disabled = false;
          loadAccuracy();
        }
      } catch { clearInterval(poll); btn.disabled = false; $('accuracy-status').textContent = 'Error'; }
    }, 5000);
    // Safety timeout after 10 min
    setTimeout(() => { clearInterval(poll); btn.disabled = false; $('accuracy-status').textContent = 'Timeout'; }, 600000);
  } catch (e) {
    $('accuracy-status').textContent = 'Error';
    btn.disabled = false;
    toast('Accuracy run failed: ' + e.message, 'error');
  }
}

function renderAccuracySummary(data) {
  const el = $('accuracy-summary');
  if (!el || !data?.summary) return;
  const s = data.summary;
  const avgScore = s.avg_score != null ? s.avg_score : (s.average !== undefined ? s.average : null);
  el.innerHTML = `
    <div class="summary-card">
      <h4>Overall Score</h4>
      <div class="summary-val ${scoreClass(avgScore)}">${avgScore != null ? avgScore.toFixed(2) : 'N/A'}</div>
    </div>
    <div class="summary-card">
      <h4>Tests Run</h4>
      <div class="summary-val">${s.tests_run || s.total || 0}</div>
    </div>
    <div class="summary-card">
      <h4>Tests Passed</h4>
      <div class="summary-val ${scoreClass(s.passed / (s.tests_run || 1))}">${s.passed || 0}</div>
    </div>
    <div class="summary-card">
      <h4>Models</h4>
      <div class="summary-val">${s.models_tested || 0}</div>
    </div>
  `;
}

function scoreClass(score) {
  if (score == null) return '';
  if (score >= 0.8) return 'good';
  if (score >= 0.5) return 'ok';
  return 'bad';
}

function renderAccuracyCompare(data) {
  const wrap = $('chart-accuracy-overall');
  if (!wrap || !data?.results) return;
  const models = {};
  data.results.forEach(r => {
    if (!models[r.model]) models[r.model] = [];
    models[r.model].push(r);
  });
  const modelNames = Object.keys(models);
  if (modelNames.length === 0) {
    wrap.innerHTML = '<div class="empty-state-small">No data yet</div>';
    return;
  }
  const averages = modelNames.map(m => {
    const scores = models[m].map(r => r.score != null ? r.score : (r.correct ? 1 : 0));
    return scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
  });
  const maxAvg = Math.max(...averages, 0.01);
  wrap.innerHTML = `<div class="bar-chart">${modelNames.map((m, i) => `
    <div class="bar-col" style="flex:${averages[i]}">
      <div class="bar-val">${averages[i].toFixed(2)}</div>
      <div class="bar-fill" style="height:${(averages[i] / maxAvg * 100).toFixed(1)}%;background:var(--accent-blue)"></div>
      <div class="bar-label">${escapeHtml(m.split('_')[0])}</div>
    </div>
  `).join('')}</div>`;
}

function renderAccuracyMatrix(data) {
  const el = $('accuracy-matrix');
  if (!el || !data?.results) return;
  const tests = [...new Set(data.results.map(r => r.test_name || r.name || 'unknown'))];
  const models = [...new Set(data.results.map(r => r.model || 'unknown'))];
  if (tests.length === 0 || models.length === 0) {
    el.innerHTML = '<div class="empty-state-small">No data yet</div>';
    return;
  }
  let html = '<table><thead><tr><th>Test</th>';
  models.forEach(m => { html += `<th>${escapeHtml(m.split('_')[0])}</th>`; });
  html += '</tr></thead><tbody>';
  tests.forEach(t => {
    html += `<tr><td>${escapeHtml(t)}</td>`;
    models.forEach(m => {
      const r = data.results.find(x => (x.test_name || x.name) === t && x.model === m);
      const score = r?.score ?? r?.correct;
      if (score == null) { html += '<td><span class="score-cell na">—</span></td>'; }
      else {
        const cls = score >= 0.8 ? 'v-high' : score >= 0.6 ? 'high' : score >= 0.4 ? 'medium' : score >= 0.2 ? 'low' : 'v-low';
        html += `<td><span class="score-cell ${cls}">${(score * 100).toFixed(0)}%</span></td>`;
      }
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

function renderAccuracyCatalog(data) {
  const el = $('accuracy-catalog');
  if (!el) return;
  const tests = data?.catalog || data?.tests || [];
  if (tests.length === 0) {
    el.innerHTML = '<div class="empty-state-small">No test catalog available</div>';
    return;
  }
  el.innerHTML = tests.map(t => {
    const colorMap = { 'color': 'cat-blue', 'count': 'cat-green', 'shape': 'cat-gold', 'ocr': 'cat-purple', 'scene': 'cat-pink' };
    const color = colorMap[t.type] || 'cat-blue';
    return `
      <div class="test-catalog-item">
        <div class="tci-type" style="background:var(--${color});color:#000">${escapeHtml(t.type || 'general')}</div>
        <div class="tci-name">${escapeHtml(t.name || t.test_name || 'Unnamed')}</div>
        <p class="muted" style="font-size:11px">${escapeHtml(t.prompt?.substring(0, 80) || '')}</p>
      </div>
    `;
  }).join('');
}

function populateAccuracyModelSelect(data) {
  const models = [...new Set((data?.results || []).map(r => r.model).filter(Boolean))];
  const sel = $('accuracy-permodel-select');
  const typeSel = $('accuracy-permodel-type');
  if (sel) {
    sel.innerHTML = models.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join('');
    sel.onchange = () => renderPerModelResults(data);
  }
  if (typeSel) {
    const types = [...new Set((data?.results || []).map(r => r.type).filter(Boolean))];
    typeSel.innerHTML = '<option value="">All types</option>' + types.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
    typeSel.onchange = () => renderPerModelResults(data);
  }
  if (models.length > 0) renderPerModelResults(data);
}

function renderPerModelResults(data) {
  const el = $('accuracy-permodel-results');
  const model = $('accuracy-permodel-select')?.value;
  const type = $('accuracy-permodel-type')?.value;
  if (!el || !model || !data?.results) return;
  let results = data.results.filter(r => r.model === model);
  if (type) results = results.filter(r => r.type === type);
  if (results.length === 0) {
    el.innerHTML = '<div class="empty-state"><p>No results for this model</p></div>';
    return;
  }
  el.innerHTML = results.map(r => `
    <div class="permodel-result-item">
      <div class="pri"><strong>${escapeHtml(r.test_name || r.name || 'Unnamed')}</strong> · ${escapeHtml(r.type || '')}</div>
      <div class="resp">${escapeHtml(r.response || r.model_response || '').substring(0, 200)}</div>
      <div class="truth">Expected: ${escapeHtml(r.expected || r.ground_truth || '')}</div>
      <div class="score">Score: <span class="${scoreClass(r.score)}">${r.score != null ? r.score.toFixed(2) : 'N/A'}</span></div>
    </div>
  `).join('');
}

// ============ SETTINGS VIEW ============
function setupSettingsView() {
  // Nothing to init yet — settings are read directly from DOM
}

// ============ MODAL (HuggingFace) ============
function setupModal() {
  $('hf-cancel')?.addEventListener('click', () => $('hf-modal').classList.add('hidden'));
  $('hf-modal')?.addEventListener('click', e => {
    if (e.target === $('hf-modal')) $('hf-modal').classList.add('hidden');
  });
  $('hf-submit')?.addEventListener('click', submitHfModel);
}

async function submitHfModel() {
  const url = $('hf-url')?.value.trim();
  const name = $('hf-name')?.value.trim();
  if (!url) { $('hf-status').textContent = 'URL is required'; return; }
  $('hf-status').textContent = 'Adding model...';
  $('hf-submit').disabled = true;
  try {
    const data = await api('/api/models/add-from-hf', {
      method: 'POST',
      body: { hf_url: url, name },
    });
    $('hf-status').textContent = 'Model added successfully!';
    toast('Model added: ' + (name || url), 'success');
    setTimeout(() => $('hf-modal').classList.add('hidden'), 1000);
    loadModels();
    checkSystem();
  } catch (e) {
    $('hf-status').textContent = 'Error: ' + e.message;
  } finally {
    $('hf-submit').disabled = false;
  }
}

// ============ START ============
document.addEventListener('DOMContentLoaded', init);
