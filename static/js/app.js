const API = "/api/v1";

const state = {
  corpusReady: false,
  lastIngest: null,
  dashboardHtml: null,
  dashboardFilename: "voc_dashboard.html",
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const corpusStatus = document.getElementById("corpusStatus");
const corpusStatusText = document.getElementById("corpusStatusText");
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");

const reviewsInput = document.getElementById("reviewsInput");
const fileInput = document.getElementById("fileInput");
const ingestBtn = document.getElementById("ingestBtn");
const ingestResult = document.getElementById("ingestResult");
const ingestVersion = document.getElementById("ingestVersion");
const ingestDateRange = document.getElementById("ingestDateRange");

const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");
const chatVersionFilter = document.getElementById("chatVersionFilter");
const chatTopK = document.getElementById("chatTopK");

const generateDashBtn = document.getElementById("generateDashBtn");
const dashVersion = document.getElementById("dashVersion");
const dashDateRange = document.getElementById("dashDateRange");
const dashTopK = document.getElementById("dashTopK");
const dashResult = document.getElementById("dashResult");
const dashProgress = document.getElementById("dashProgress");
const dashPreviewWrap = document.getElementById("dashPreviewWrap");
const dashPreview = document.getElementById("dashPreview");
const openDashBtn = document.getElementById("openDashBtn");
const downloadDashBtn = document.getElementById("downloadDashBtn");

const toast = document.getElementById("toast");

// ── Tabs ─────────────────────────────────────────────────────────────────────
tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const name = tab.dataset.tab;
    tabs.forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
    panels.forEach((p) => {
      const isActive = p.id === `panel-${name}`;
      p.classList.toggle("active", isActive);
      p.hidden = !isActive;
    });
  });
});

// ── Helpers ──────────────────────────────────────────────────────────────────
function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.remove("hidden");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => toast.classList.add("hidden"), 4000);
}

function setLoading(btn, loading) {
  btn.disabled = loading;
  btn.querySelector(".btn-label").classList.toggle("hidden", loading);
  btn.querySelector(".spinner")?.classList.toggle("hidden", !loading);
}

async function apiPost(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    let msg;
    if (data.error) {
      msg = data.detail ? `${data.error}: ${data.detail}` : data.error;
    } else if (typeof data.detail === "string") {
      msg = data.detail;
    } else {
      msg = JSON.stringify(data.detail || data);
    }
    throw new Error(msg);
  }
  return data;
}

function updateCorpusStatus(ready, text) {
  state.corpusReady = ready;
  corpusStatus.classList.toggle("ready", ready);
  corpusStatusText.textContent = text;
}

function syncVersionFields(version) {
  dashVersion.value = version;
  chatVersionFilter.placeholder = version;
}

// ── Ingest ───────────────────────────────────────────────────────────────────
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const name = file.name.toLowerCase();
  if (name.endsWith(".xlsx") || name.endsWith(".xls")) {
    showToast("Excel not supported — export CSV from Play Console", true);
    fileInput.value = "";
    return;
  }
  try {
    const text = await file.text();
    if (text.includes("\x00") || (text.length > 50 && text.slice(0, 2) === "PK")) {
      showToast("Binary file detected — use .txt or .csv export", true);
      fileInput.value = "";
      return;
    }
    reviewsInput.value = text;
    showToast(`Loaded ${file.name} (${text.length.toLocaleString()} chars)`);
  } catch {
    showToast("Could not read file", true);
  }
  fileInput.value = "";
});

document.getElementById("clearReviewsBtn").addEventListener("click", () => {
  reviewsInput.value = "";
});

ingestBtn.addEventListener("click", async () => {
  const reviews = reviewsInput.value.trim();
  if (!reviews) {
    showToast("Paste or upload reviews first", true);
    return;
  }

  setLoading(ingestBtn, true);
  ingestResult.classList.add("hidden");

  try {
    const data = await apiPost("/ingest", {
      reviews,
      app_version: ingestVersion.value.trim() || "30.XX",
      date_range: ingestDateRange.value.trim(),
    });

    state.lastIngest = data;
    syncVersionFields(ingestVersion.value.trim() || "30.XX");
    if (ingestDateRange.value.trim()) {
      dashDateRange.value = ingestDateRange.value.trim();
    }

    if (data.reviews_parsed === 0) {
      throw new Error(
        "No reviews parsed. Check format: author, ★ rating line, and review text (15+ chars)."
      );
    }

    updateCorpusStatus(
      true,
      `${data.reviews_parsed} reviews · ${data.chunks_stored} chunks in corpus`
    );

    const topHtml =
      data.top_upvoted?.length > 0
        ? `<p><strong>Top upvoted:</strong> ${data.top_upvoted[0].text.slice(0, 80)}… (${data.top_upvoted[0].upvotes} ↑)</p>`
        : "";

    ingestResult.className = "result-box success";
    ingestResult.innerHTML = `
      <strong>Ingest successful</strong>
      ${data.message ? `<p>${escapeHtml(data.message)}</p>` : ""}
      <div class="result-stats">
        <div class="stat"><strong>${data.reviews_parsed}</strong> reviews parsed</div>
        <div class="stat"><strong>${data.chunks_stored}</strong> chunks stored</div>
        <div class="stat"><strong>${data.avg_rating}</strong> avg rating</div>
      </div>
      ${topHtml}
    `;
    ingestResult.classList.remove("hidden");
    showToast(`Ingested ${data.reviews_parsed} reviews`);
  } catch (err) {
    ingestResult.className = "result-box error";
    ingestResult.textContent = err.message;
    ingestResult.classList.remove("hidden");
    showToast(err.message, true);
  } finally {
    setLoading(ingestBtn, false);
  }
});

// ── Chat ─────────────────────────────────────────────────────────────────────
function appendMessage(role, content, meta = "") {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  if (role === "assistant") {
    div.innerHTML = `<div class="answer-body">${escapeHtml(content)}</div>${meta ? `<div class="meta">${meta}</div>` : ""}`;
  } else {
    div.textContent = content;
  }
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

document.getElementById("clearChatBtn").addEventListener("click", () => {
  chatMessages.innerHTML = `
    <div class="message assistant welcome">
      <p><strong>Conversation cleared.</strong> Ask a new question about your review corpus.</p>
    </div>`;
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;

  if (!state.corpusReady) {
    showToast("Ingest reviews first (tab ①)", true);
    return;
  }

  appendMessage("user", question);
  chatInput.value = "";

  const sendBtn = document.getElementById("chatSendBtn");
  setLoading(sendBtn, true);

  try {
    const data = await apiPost("/chat", {
      question,
      version_filter: chatVersionFilter.value.trim(),
      top_k: parseInt(chatTopK.value, 10) || 20,
    });

    const meta = `${data.retrieved_chunks} chunks · avg ${data.avg_rating_in_context}/5 · ${data.high_upvote_count} high-upvote`;
    appendMessage("assistant", data.answer, meta);
  } catch (err) {
    appendMessage("assistant", `Error: ${err.message}`, "");
    const last = chatMessages.lastElementChild;
    if (last) last.classList.add("error");
    showToast(err.message, true);
  } finally {
    setLoading(sendBtn, false);
  }
});

// ── Dashboard ────────────────────────────────────────────────────────────────
generateDashBtn.addEventListener("click", async () => {
  if (!state.corpusReady) {
    showToast("Ingest reviews first (tab ①)", true);
    return;
  }

  setLoading(generateDashBtn, true);
  dashProgress.classList.remove("hidden");
  dashResult.classList.add("hidden");
  dashPreviewWrap.classList.add("hidden");
  openDashBtn.classList.add("hidden");
  downloadDashBtn.classList.add("hidden");
  state.dashboardHtml = null;

  try {
    const data = await apiPost("/dashboard", {
      app_version: dashVersion.value.trim() || "30.XX",
      date_range: dashDateRange.value.trim() || "Recent window",
      top_k: parseInt(dashTopK.value, 10) || 100,
    });

    state.dashboardHtml = data.html;
    state.dashboardFilename = data.filename || "voc_dashboard.html";

    dashResult.className = "result-box success";
    dashResult.innerHTML = `
      <strong>Dashboard generated</strong>
      <p>${data.total_reviews} reviews · ${data.date_range} · ${data.generated_at}</p>
    `;
    dashResult.classList.remove("hidden");

    const blob = new Blob([data.html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    dashPreview.src = url;
    dashPreviewWrap.classList.remove("hidden");
    openDashBtn.classList.remove("hidden");
    downloadDashBtn.classList.remove("hidden");

    openDashBtn.onclick = () => window.open(url, "_blank");
    downloadDashBtn.onclick = () => {
      const a = document.createElement("a");
      a.href = url;
      a.download = state.dashboardFilename;
      a.click();
    };

    showToast("Dashboard ready — preview below");
  } catch (err) {
    dashResult.className = "result-box error";
    dashResult.textContent = err.message;
    dashResult.classList.remove("hidden");
    showToast(err.message, true);
  } finally {
    setLoading(generateDashBtn, false);
    dashProgress.classList.add("hidden");
  }
});

// ── Init: check if corpus exists from prior session ──────────────────────────
(async function init() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("API unavailable");
    const health = await res.json();
    console.log("API:", health.service);
  } catch {
    showToast("API not reachable — is uvicorn running?", true);
  }
})();
