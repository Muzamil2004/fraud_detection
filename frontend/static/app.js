function toNumber(value) {
  return value === "" ? null : Number(value);
}

function formatPct(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function badgeClass(prob, threshold) {
  if (prob >= threshold) {
    return "bad";
  }
  if (prob >= threshold * 0.7) {
    return "warn";
  }
  return "ok";
}

function setLoading(buttonId, loadingText, idleText, isLoading) {
  const btn = document.getElementById(buttonId);
  if (!btn) return;
  btn.disabled = isLoading;
  btn.textContent = isLoading ? loadingText : idleText;
}

async function callApi(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data = text;

  try {
    data = JSON.parse(text);
  } catch (err) {
    // use plain response text
  }

  if (!response.ok) {
    const message = typeof data === "string" ? data : (data.detail || JSON.stringify(data));
    throw new Error(message);
  }
  return data;
}

function renderError(container, err) {
  container.className = "result";
  container.innerHTML = `
    <p class="result-title">Request Failed</p>
    <p class="result-text">${escapeHtml(err.message || String(err))}</p>
  `;
}

async function loadHealth() {
  try {
    const data = await callApi("/health", { method: "GET" });
    const info = await callApi("/model-info", { method: "GET" });

    document.getElementById("healthStatus").textContent = data.status || "-";
    document.getElementById("healthModel").textContent = data.model || "-";
    document.getElementById("healthThreshold").textContent = data.threshold ?? "-";
    document.getElementById("healthPath").textContent = data.model_path || "-";

    document.getElementById("modelTarget").textContent = info.target_col || "-";
    const metricKeys = Object.keys(info.available_metrics || {});
    document.getElementById("metricCount").textContent = `${metricKeys.length} model(s)`;

    const modelInfoBox = document.getElementById("modelInfoBox");
    if (metricKeys.length === 0) {
      modelInfoBox.className = "result empty";
      modelInfoBox.textContent = "No metrics available in artifact.";
      return;
    }
    const preview = metricKeys.slice(0, 3).map((name) => {
      const m = info.available_metrics[name] || {};
      return `<tr><td>${escapeHtml(name)}</td><td>${Number(m.f1 || 0).toFixed(4)}</td><td>${Number(m.pr_auc || 0).toFixed(4)}</td></tr>`;
    }).join("");
    modelInfoBox.className = "result";
    modelInfoBox.innerHTML = `
      <p class="result-title">Training Snapshot</p>
      <div class="mini-table">
        <table>
          <thead><tr><th>Model</th><th>F1</th><th>PR-AUC</th></tr></thead>
          <tbody>${preview}</tbody>
        </table>
      </div>
    `;
  } catch (err) {
    document.getElementById("healthStatus").textContent = "Error";
    document.getElementById("healthModel").textContent = "-";
    document.getElementById("healthThreshold").textContent = "-";
    document.getElementById("healthPath").textContent = err.message || String(err);
    document.getElementById("modelTarget").textContent = "-";
    document.getElementById("metricCount").textContent = "-";
    const modelInfoBox = document.getElementById("modelInfoBox");
    modelInfoBox.className = "result";
    modelInfoBox.innerHTML = `<p class="result-title">Metadata Load Failed</p><p class="result-text">${escapeHtml(err.message || String(err))}</p>`;
  }
}

async function handlePredict(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const output = document.getElementById("predictOutput");
  setLoading("predictBtn", "Scoring...", "Score Transaction", true);
  output.className = "result";
  output.innerHTML = "<p class='result-text'>Scoring transaction...</p>";

  const payload = {
    amount: Number(form.amount.value),
    transaction_hour: Number(form.transaction_hour.value),
    merchant_category: form.merchant_category.value,
    foreign_transaction: Number(form.foreign_transaction.value),
    location_mismatch: Number(form.location_mismatch.value),
    device_trust_score: Number(form.device_trust_score.value),
    velocity_last_24h: Number(form.velocity_last_24h.value),
    cardholder_age: Number(form.cardholder_age.value)
  };

  const threshold = toNumber(form.custom_threshold.value);
  const query = threshold === null ? "" : `?custom_threshold=${encodeURIComponent(threshold)}`;

  try {
    const data = await callApi(`/predict${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const bClass = badgeClass(data.fraud_probability, data.threshold_used);
    const decision = data.is_fraud === 1 ? "Flagged as Fraud" : "Likely Legitimate";

    output.innerHTML = `
      <p class="result-title">${decision}</p>
      <p class="result-text">Fraud probability: <strong>${formatPct(data.fraud_probability)}</strong></p>
      <p class="result-text">Threshold used: <strong>${Number(data.threshold_used).toFixed(2)}</strong> | Model: <strong>${escapeHtml(data.model)}</strong></p>
      <span class="badge ${bClass}">${data.is_fraud === 1 ? "HIGH RISK" : "LOW RISK"}</span>
    `;
  } catch (err) {
    renderError(output, err);
  } finally {
    setLoading("predictBtn", "Scoring...", "Score Transaction", false);
  }
}

async function handleBatch(event) {
  event.preventDefault();
  const output = document.getElementById("batchOutput");
  setLoading("batchBtn", "Running...", "Run Batch", true);
  output.className = "result";
  output.innerHTML = "<p class='result-text'>Running batch scoring...</p>";

  const raw = document.getElementById("batchInput").value;
  const threshold = toNumber(document.getElementById("batchThreshold").value);
  const query = threshold === null ? "" : `?custom_threshold=${encodeURIComponent(threshold)}`;

  let payload;
  try {
    payload = JSON.parse(raw);
  } catch (err) {
    renderError(output, new Error(`Invalid JSON payload: ${err.message}`));
    setLoading("batchBtn", "Running...", "Run Batch", false);
    return;
  }

  try {
    const data = await callApi(`/predict/batch${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const predictions = Array.isArray(data.predictions) ? data.predictions : [];
    const flagged = predictions.filter((item) => Number(item.is_fraud) === 1).length;
    const avgProb = predictions.length
      ? predictions.reduce((acc, item) => acc + Number(item.fraud_probability), 0) / predictions.length
      : 0;
    const thresholdUsed = predictions.length ? Number(predictions[0].threshold_used) : 0;
    const bClass = badgeClass(avgProb, thresholdUsed || 0.5);

    const rows = predictions
      .slice(0, 12)
      .map((item, idx) => {
        return `<tr>
          <td>${idx + 1}</td>
          <td>${formatPct(item.fraud_probability)}</td>
          <td>${Number(item.is_fraud) === 1 ? "Fraud" : "Legit"}</td>
        </tr>`;
      })
      .join("");

    output.innerHTML = `
      <p class="result-title">Batch Completed</p>
      <p class="result-text">Scored <strong>${predictions.length}</strong> records. Flagged: <strong>${flagged}</strong></p>
      <p class="result-text">Average fraud probability: <strong>${formatPct(avgProb)}</strong></p>
      <span class="badge ${bClass}">${flagged > 0 ? "ACTION NEEDED" : "STABLE BATCH"}</span>
      <div class="mini-table">
        <table>
          <thead>
            <tr><th>#</th><th>Probability</th><th>Decision</th></tr>
          </thead>
          <tbody>${rows || "<tr><td colspan='3'>No predictions returned.</td></tr>"}</tbody>
        </table>
      </div>
    `;
  } catch (err) {
    renderError(output, err);
  } finally {
    setLoading("batchBtn", "Running...", "Run Batch", false);
  }
}

document.getElementById("refreshHealth").addEventListener("click", loadHealth);
document.getElementById("predictForm").addEventListener("submit", handlePredict);
document.getElementById("batchForm").addEventListener("submit", handleBatch);

loadHealth();
