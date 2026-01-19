const state = {
  models: [],
  selectedModel: null,
  run: null,
  logOffset: 0,
  lastRunId: null,
};

const statusPill = document.getElementById("status-pill");
const modelList = document.getElementById("model-list");
const modelTitle = document.getElementById("model-title");
const modelDesc = document.getElementById("model-desc");
const configInput = document.getElementById("config-path");
const runButton = document.getElementById("run-btn");
const queueButton = document.getElementById("queue-btn");
const runId = document.getElementById("run-id");
const runModel = document.getElementById("run-model");
const runStatus = document.getElementById("run-status");
const logOutput = document.getElementById("log-output");
const clearLog = document.getElementById("clear-log");

queueButton.addEventListener("click", (event) => event.preventDefault());

clearLog.addEventListener("click", () => {
  logOutput.textContent = "";
  state.logOffset = 0;
});

runButton.addEventListener("click", async () => {
  if (!state.selectedModel) {
    return;
  }
  runButton.disabled = true;
  const payload = {
    config_path: configInput.value.trim() || null,
  };
  try {
    const response = await fetch(`/api/run/${state.selectedModel.id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      alert(data.detail || "Failed to start run.");
      runButton.disabled = false;
      return;
    }
    state.run = data.run;
    state.logOffset = 0;
    logOutput.textContent = "";
    renderRunMeta(data.run);
  } catch (error) {
    alert("Failed to start run.");
    runButton.disabled = false;
  }
});

async function loadModels() {
  const response = await fetch("/api/models");
  const data = await response.json();
  state.models = data.models || [];
  renderModels();
  if (state.models.length > 0) {
    selectModel(state.models[0].id);
  }
}

function renderModels() {
  modelList.innerHTML = "";
  if (state.models.length === 0) {
    modelList.innerHTML = '<div class="model-placeholder">No models configured.</div>';
    return;
  }
  state.models.forEach((model) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-card";
    button.dataset.modelId = model.id;
    button.innerHTML = `
      <div class="model-name">${model.name}</div>
      <p class="model-desc">${model.description || ""}</p>
    `;
    button.addEventListener("click", () => selectModel(model.id));
    modelList.appendChild(button);
  });
}

function selectModel(modelId) {
  const selected = state.models.find((model) => model.id === modelId);
  if (!selected) {
    return;
  }
  state.selectedModel = selected;
  modelTitle.textContent = selected.name;
  modelDesc.textContent = selected.description || "";
  configInput.value = selected.default_config || "";

  document.querySelectorAll(".model-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.modelId === modelId);
  });
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    updateStatusUI(data);
  } catch (error) {
    statusPill.textContent = "Offline";
    statusPill.className = "status-pill failed";
  }
}

function updateStatusUI(data) {
  const status = data.status || "idle";
  statusPill.textContent = status.toUpperCase();
  statusPill.className = `status-pill ${status}`;

  if (data.run) {
    state.run = data.run;
    renderRunMeta(data.run);
  } else {
    runId.textContent = "-";
    runModel.textContent = "-";
    runStatus.textContent = "Idle";
    state.run = null;
  }

  if (status === "running") {
    runButton.disabled = true;
  } else {
    runButton.disabled = false;
  }

  if (state.run && state.run.run_id !== state.lastRunId) {
    state.lastRunId = state.run.run_id;
    state.logOffset = 0;
    logOutput.textContent = "";
  }
}

function renderRunMeta(run) {
  runId.textContent = run.run_id || "-";
  runModel.textContent = run.model_id || "-";
  runStatus.textContent = run.status || "-";
}

async function pollLogs() {
  if (!state.run) {
    return;
  }
  const params = new URLSearchParams({
    run_id: state.run.run_id,
    offset: state.logOffset.toString(),
  });
  try {
    const response = await fetch(`/api/logs?${params.toString()}`);
    const data = await response.json();
    if (data.data) {
      logOutput.textContent += data.data;
      logOutput.scrollTop = logOutput.scrollHeight;
    }
    state.logOffset = data.offset;
  } catch (error) {
    // Ignore log polling errors.
  }
}

loadModels();
pollStatus();
pollLogs();
setInterval(pollStatus, 2000);
setInterval(pollLogs, 2000);
