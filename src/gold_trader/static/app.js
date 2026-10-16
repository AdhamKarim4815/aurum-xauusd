const tickClock = () => {
  const element = document.getElementById("utc-clock");
  if (element) element.textContent = new Date().toISOString().slice(11, 19) + " UTC";
};
tickClock();
setInterval(tickClock, 1000);

document.querySelectorAll("form[data-confirm]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    if (!window.confirm(form.dataset.confirm)) event.preventDefault();
  });
});

function renderEquityChart(history, currentEquity) {
  const canvas = document.getElementById("equity-chart");
  const empty = document.getElementById("equity-empty");
  if (!canvas) return;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);

  const points = (history || []).map((point) => Number(point.equity));
  if (points.length < 2) {
    canvas.hidden = true;
    if (empty) empty.hidden = false;
    return;
  }
  canvas.hidden = false;
  if (empty) empty.hidden = true;

  context.strokeStyle = "rgba(219, 180, 84, .16)";
  context.lineWidth = 1;
  for (let y = 30; y < height; y += 45) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  const padding = 16;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const span = Math.max(max - min, 1);
  const startedUp = points[points.length - 1] >= points[0];
  context.strokeStyle = startedUp ? "#7fcaa2" : "#f07d73";
  context.lineWidth = 2.5;
  context.beginPath();
  points.forEach((value, index) => {
    const x = (index / (points.length - 1)) * width;
    const y = height - padding - ((value - min) / span) * (height - padding * 2);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();

  const equity = document.getElementById("paper-equity");
  if (equity) equity.textContent = `$${Number(currentEquity ?? points[points.length - 1]).toFixed(2)}`;
}

function quoteAgeClass(ageSeconds, staleAfterSeconds) {
  if (ageSeconds === undefined || ageSeconds === null) return "";
  if (!staleAfterSeconds || ageSeconds < staleAfterSeconds * 0.5) return "ok";
  if (ageSeconds < staleAfterSeconds) return "warn";
  return "stale";
}

function updateProviderChain(providerChain, activeProvider) {
  const container = document.getElementById("provider-chain");
  if (!container) return;
  container.querySelectorAll(".provider-chip").forEach((chip) => {
    const name = chip.dataset.provider;
    const entry = (providerChain || []).find((item) => item.name === name);
    const configured = entry ? entry.configured : true;
    chip.classList.toggle("active", name === activeProvider);
    chip.classList.toggle("unset", !configured);
    chip.textContent = configured ? name : `${name} · no key`;
  });
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { headers: { "Accept": "application/json" } });
    if (!response.ok) return;
    const data = await response.json();
    const mapping = {
      "paper-status": data.paper.status || "stopped",
      "feed-status": data.paper.feed_status || "unconfigured",
      "sleep-status": data.paper.sleep_lease ? "held" : "released",
      "approval-status": data.approval ? data.approval.candidate_hash.slice(0, 10) : "required",
      "mode-status": data.paper.mode || "none",
    };
    if (!document.querySelector("[data-paper-lab]")) {
      mapping["header-status"] = data.paper.status || "stopped";
    }
    for (const [id, value] of Object.entries(mapping)) {
      const element = document.getElementById(id);
      if (element) element.textContent = value;
    }
    const trades = document.getElementById("paper-trades");
    if (trades) trades.textContent = String(data.paper.observation_trade_count || 0);
    const quoteAge = document.getElementById("quote-age");
    if (quoteAge) {
      const age = data.paper.quote_age_seconds;
      quoteAge.textContent = age === undefined ? "—" : `${Number(age).toFixed(0)} sec`;
      quoteAge.classList.remove("ok", "warn", "stale");
      const health = quoteAgeClass(age, data.live_paper_stale_after_seconds);
      if (health) quoteAge.classList.add(health);
    }
    updateProviderChain(data.provider_chain, data.paper.live_provider);
    renderEquityChart(data.equity_history, data.paper.equity);
    if (data.generation) {
      const summary = data.generation.summary || {};
      const generationMapping = {
        "generation-status": data.generation.status,
        "generation-completed": summary.completed_trials || 0,
        "generation-total": summary.total_trials || 0,
        "generation-shortlisted": summary.shortlisted || 0,
        "generation-batches": summary.batches || 0,
        "generation-pbo": Number(summary.probability_of_backtest_overfit ?? 1).toFixed(2),
      };
      for (const [id, value] of Object.entries(generationMapping)) {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
      }
    }
  } catch (_) { /* The dashboard stays useful when the worker or network is interrupted. */ }
}
refreshStatus();
setInterval(refreshStatus, 3000);

async function refreshValidations() {
  const cards = document.querySelectorAll("[data-validation-job]");
  if (!cards.length) return;
  try {
    const response = await fetch("/api/validations", { headers: { "Accept": "application/json" } });
    if (!response.ok) return;
    const data = await response.json();
    const jobs = data.jobs || [];
    let finished = false;
    cards.forEach((card) => {
      const job = jobs.find((item) => item.trial_hash === card.dataset.trialHash);
      if (!job) return;
      const progress = Number(job.progress || 0);
      const status = card.querySelector("[data-validation-status]");
      const percent = card.querySelector("[data-validation-percent]");
      const bar = card.querySelector("[data-validation-bar]");
      const stage = card.querySelector("[data-validation-stage]");
      if (status) status.textContent = job.status.charAt(0).toUpperCase() + job.status.slice(1);
      if (percent) percent.textContent = `${progress}%`;
      if (bar) bar.style.width = `${progress}%`;
      if (stage) stage.textContent = job.stage || "Final validation";
      if (["completed", "failed"].includes(job.status)) finished = true;
    });
    if (finished) window.location.reload();
  } catch (_) { /* Durable state is shown again after connectivity returns. */ }
}
refreshValidations();
setInterval(refreshValidations, 2000);

function formatLabPosition(position) {
  if (!position) return "flat";
  const lots = Number(position.quantity_lots ?? position.quantity ?? 0).toFixed(2);
  const entry = Number(position.entry_price ?? 0).toFixed(2);
  return `${position.side || "open"} ${lots} lots @ ${entry}`;
}

function updateLabField(card, name, value) {
  const element = card.querySelector(`[data-lab-field="${name}"]`);
  if (element) element.textContent = value;
}

async function refreshPaperLab() {
  const container = document.querySelector("[data-paper-lab]");
  if (!container) return;
  try {
    const response = await fetch("/api/paper-lab", {
      headers: { "Accept": "application/json" },
    });
    if (!response.ok) return;
    const data = await response.json();
    const status = document.getElementById("lab-status");
    if (status) {
      status.textContent = data.status || "stopped";
      status.className = `status-pill ${data.status || "stopped"}`;
    }
    const headerStatus = document.getElementById("header-status");
    if (headerStatus) {
      headerStatus.textContent = data.status || "stopped";
      headerStatus.className = `status-pill ${data.status || "stopped"}`;
    }
    const summary = {
      "lab-connection": data.connection_status || "not started",
      "lab-open-positions": `${Number(data.open_positions || 0)} / 4`,
      "lab-closed-trades": String(data.closed_trades || 0),
      "lab-sleep": data.sleep_lease ? "held" : "released",
      "lab-last-poll": data.last_poll_at
        ? new Date(data.last_poll_at).toISOString().replace("T", " ").slice(0, 19) + " UTC"
        : "not started",
    };
    Object.entries(summary).forEach(([id, value]) => {
      const element = document.getElementById(id);
      if (element) element.textContent = value;
    });
    const networkMessage = document.getElementById("lab-network-message");
    if (networkMessage) {
      networkMessage.hidden = data.status !== "paused-network";
      if (data.status === "paused-network") {
        const retry = data.next_retry_at
          ? ` Next retry: ${new Date(data.next_retry_at).toISOString().replace("T", " ").slice(0, 19)} UTC.`
          : "";
        networkMessage.textContent =
          "Feed connection is unavailable. New entries are paused and existing paper positions are preserved." +
          retry +
          " Recovery checks missing bars and never creates retroactive trades.";
      }
    }
    (data.accounts || []).forEach((account) => {
      const card = document.querySelector(`[data-lab-account="${account.candidate_hash}"]`);
      if (!card) return;
      const equity = Number(account.equity ?? 10000);
      updateLabField(card, "status", account.status || "stopped");
      updateLabField(card, "equity", `$${equity.toFixed(2)}`);
      updateLabField(card, "return", `${((equity / 10000 - 1) * 100).toFixed(2)}%`);
      updateLabField(card, "trades", String(account.observation_trade_count || 0));
      updateLabField(card, "feed", account.feed_status || "not started");
      updateLabField(card, "position", formatLabPosition(account.position));
      updateLabField(card, "entries", account.entries_allowed ? "allowed" : "blocked");
      updateLabField(
        card,
        "quote-age",
        account.quote_age_seconds === undefined
          ? "—"
          : `${Number(account.quote_age_seconds).toFixed(0)} sec`,
      );
      updateLabField(
        card,
        "risk",
        account.risk_halt_reason || account.last_feed_error || "clear",
      );
      updateLabField(card, "decision", account.last_decision_bar || "waiting");
    });
  } catch (_) { /* Durable lab state remains visible after connectivity returns. */ }
}
refreshPaperLab();
setInterval(refreshPaperLab, 3000);
