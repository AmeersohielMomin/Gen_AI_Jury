"use strict";

const state = {
  overview: null,
  juryLab: null,
  samples: [],
  quizPanels: [],
  quizGuesses: new Map(),
  quizResult: null,
  roadmap: null,
  selectedModality: "dermoscopy",
  selectedSampleId: null,
  uploadedFile: null,
  lastAnalysis: null,
  lastReport: null,
  sampleFilter: "all",
};

const ARM_COLORS = {
  real_only: "#0f766e",
  repeated_real_scc: "#2563eb",
  cgan_scc: "#d97706",
};

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (v !== undefined && v !== null) node.setAttribute(k, String(v));
  }
  for (const child of children) {
    if (typeof child === "string") node.appendChild(document.createTextNode(child));
    else if (child) node.appendChild(child);
  }
  return node;
}

function announce(message) {
  const region = document.getElementById("live-announcer");
  if (region) region.textContent = message;
}

async function fetchJSON(url, options = {}) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(errText || `HTTP ${resp.status}`);
  }
  return resp.json();
}

/* ---------------- Theme Toggle (light/dark/system) ---------------- */
function initThemeToggle() {
  const meta = document.querySelector('meta[name="color-scheme"]');
  const toggleBtn = document.getElementById("theme-toggle");
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

  function resolvedScheme() {
    const c = meta.content;
    if (c === "light" || c === "dark") return c;
    return systemDark.matches ? "dark" : "light";
  }

  toggleBtn.addEventListener("click", () => {
    const target = resolvedScheme() === "dark" ? "light" : "dark";
    const sys = systemDark.matches ? "dark" : "light";
    try {
      if (target === sys) localStorage.removeItem("color-scheme");
      else localStorage.setItem("color-scheme", target);
    } catch (_) {}
    meta.content = target === sys ? "light dark" : target;
    announce(`Color theme switched to ${target} mode`);
  });

  window.addEventListener("storage", (ev) => {
    if (ev.key === "color-scheme") {
      meta.content = ev.newValue === "light" || ev.newValue === "dark" ? ev.newValue : "light dark";
    }
  });
}

/* ---------------- Primary View Routing ---------------- */
const VIEW_TITLES = {
  overview: "Overview | GenAI Jury — Medical Imaging & cGAN Studio",
  analyzer: "Live Analyzer & Grad-CAM | GenAI Jury Studio",
  "jury-lab": "Interactive Jury Lab | GenAI Jury Studio",
  roadmap: "Future Roadmap | GenAI Jury Studio",
};

function switchView(viewId) {
  document.querySelectorAll(".app-view").forEach((sec) => {
    sec.hidden = sec.id !== `view-${viewId}`;
  });
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    if (btn.dataset.view === viewId) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  });
  document.title = VIEW_TITLES[viewId] || VIEW_TITLES.overview;
  if (viewId === "analyzer" && !state.lastAnalysis && state.selectedSampleId) {
    runAnalysis({ sampleId: state.selectedSampleId });
  }
  announce(`Opened ${viewId.replace("-", " ")} view`);
}

function initNavigation() {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });
  document.querySelectorAll("[data-jump-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchView(btn.dataset.jumpView);
      if (btn.dataset.jumpTab) {
        const tab = document.getElementById(btn.dataset.jumpTab);
        if (tab) selectLabTab(tab);
      }
    });
  });
}

/* ---------------- ARIA Tabs inside Jury Lab ---------------- */
const labTabs = Array.from(document.querySelectorAll('.tablist [role="tab"]'));
const labPanels = Array.from(document.querySelectorAll('.tabs-shell [role="tabpanel"]'));

function selectLabTab(selectedTab, focusTab = false) {
  labTabs.forEach((tab, idx) => {
    const isSel = tab === selectedTab;
    tab.setAttribute("aria-selected", String(isSel));
    tab.tabIndex = isSel ? 0 : -1;
    const panel = labPanels[idx];
    if (panel) {
      if (isSel) {
        panel.removeAttribute("hidden");
        panel.tabIndex = 0;
      } else {
        panel.setAttribute("hidden", "until-found");
        panel.removeAttribute("tabindex");
      }
    }
  });
  if (focusTab) selectedTab.focus();
  selectedTab.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function initLabTabs() {
  labTabs.forEach((tab, idx) => {
    tab.addEventListener("click", () => selectLabTab(tab));
    tab.addEventListener("keydown", (ev) => {
      let next = null;
      if (ev.key === "ArrowRight") next = labTabs[(idx + 1) % labTabs.length];
      else if (ev.key === "ArrowLeft") next = labTabs[(idx - 1 + labTabs.length) % labTabs.length];
      else if (ev.key === "Home") next = labTabs[0];
      else if (ev.key === "End") next = labTabs[labTabs.length - 1];
      if (next) {
        ev.preventDefault();
        selectLabTab(next, true);
      }
    });
  });
  labPanels.forEach((panel, idx) => {
    panel.addEventListener("beforematch", () => {
      if (labTabs[idx]) selectLabTab(labTabs[idx]);
    });
  });
}

/* ---------------- Native SVG Chart Helpers ---------------- */
function svgEl(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) {
    node.setAttribute(k, String(v));
  }
  return node;
}

function renderImbalanceChart(overview) {
  const host = document.getElementById("chart-imbalance");
  const counts = overview.splits.train.counts;
  const items = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const maxVal = Math.max(...items.map((x) => x[1]));

  const width = 560;
  const height = 250;
  const pad = { top: 20, right: 20, bottom: 42, left: 56 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Bar chart of ISIC 2019 training set class counts showing severe imbalance with SCC at 429 images",
  });

  const barStep = plotW / items.length;
  const barW = barStep * 0.62;

  for (let t = 0; t <= 4; t++) {
    const val = Math.round((maxVal * t) / 4);
    const y = pad.top + plotH - (plotH * t) / 4;
    svg.appendChild(svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y, y2: y, stroke: "currentColor", "stroke-opacity": 0.12 }));
    const lbl = svgEl("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end", fill: "currentColor", "font-size": 10, opacity: 0.72 });
    lbl.textContent = val.toLocaleString();
    svg.appendChild(lbl);
  }

  items.forEach(([diag, count], idx) => {
    const barH = (count / maxVal) * plotH;
    const x = pad.left + idx * barStep + (barStep - barW) / 2;
    const y = pad.top + plotH - barH;
    const isTarget = diag === overview.target_class;
    svg.appendChild(svgEl("rect", {
      x, y, width: barW, height: Math.max(2, barH), rx: 4,
      fill: isTarget ? "#d97706" : "#0f766e",
    }));
    const valLbl = svgEl("text", { x: x + barW / 2, y: y - 5, "text-anchor": "middle", fill: "currentColor", "font-size": 10, "font-weight": isTarget ? 700 : 500 });
    valLbl.textContent = count.toLocaleString();
    svg.appendChild(valLbl);

    const xLbl = svgEl("text", { x: x + barW / 2, y: height - 18, "text-anchor": "middle", fill: "currentColor", "font-size": 11, "font-weight": isTarget ? 800 : 600 });
    xLbl.textContent = diag;
    svg.appendChild(xLbl);
  });

  host.replaceChildren(svg);
}

function renderPRCurveChart(juryLab) {
  const host = document.getElementById("chart-pr-curves");
  const width = 540;
  const height = 310;
  const pad = { top: 20, right: 20, bottom: 44, left: 50 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "SCC Precision-Recall curves across real_only, repeated_real_scc, and cgan_scc",
  });

  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    const x = pad.left + v * plotW;
    const y = pad.top + plotH - v * plotH;
    svg.appendChild(svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y, y2: y, stroke: "currentColor", "stroke-opacity": 0.12 }));
    svg.appendChild(svgEl("line", { x1: x, x2: x, y1: pad.top, y2: pad.top + plotH, stroke: "currentColor", "stroke-opacity": 0.12 }));
    const yLbl = svgEl("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end", fill: "currentColor", "font-size": 10, opacity: 0.72 });
    yLbl.textContent = v.toFixed(1);
    svg.appendChild(yLbl);
    const xLbl = svgEl("text", { x, y: pad.top + plotH + 16, "text-anchor": "middle", fill: "currentColor", "font-size": 10, opacity: 0.72 });
    xLbl.textContent = v.toFixed(1);
    svg.appendChild(xLbl);
  }

  const xTitle = svgEl("text", { x: pad.left + plotW / 2, y: height - 6, "text-anchor": "middle", fill: "currentColor", "font-size": 11, "font-weight": 600 });
  xTitle.textContent = "SCC Recall";
  svg.appendChild(xTitle);

  const yTitle = svgEl("text", { x: 14, y: pad.top + plotH / 2, transform: `rotate(-90 14 ${pad.top + plotH / 2})`, "text-anchor": "middle", fill: "currentColor", "font-size": 11, "font-weight": 600 });
  yTitle.textContent = "SCC Precision";
  svg.appendChild(yTitle);

  juryLab.arm_order.forEach((arm, idx) => {
    const pts = juryLab.arms[arm].pr_curve;
    const d = pts
      .map((p, i) => `${i === 0 ? "M" : "L"}${(pad.left + p.recall * plotW).toFixed(1)},${(pad.top + plotH - p.precision * plotH).toFixed(1)}`)
      .join(" ");
    svg.appendChild(svgEl("path", { d, fill: "none", stroke: ARM_COLORS[arm], "stroke-width": 2.6 }));

    const ly = pad.top + 14 + idx * 18;
    svg.appendChild(svgEl("rect", { x: width - 205, y: ly - 9, width: 14, height: 4, rx: 2, fill: ARM_COLORS[arm] }));
    const lt = svgEl("text", { x: width - 184, y: ly - 4, fill: "currentColor", "font-size": 10.5, "font-weight": 600 });
    lt.textContent = `${juryLab.arms[arm].short} (AP=${juryLab.arms[arm].metrics.target_pr_auc.toFixed(3)})`;
    svg.appendChild(lt);
  });

  host.replaceChildren(svg);
}

function renderBootstrapCIChart(juryLab) {
  const host = document.getElementById("chart-bootstrap-ci");
  const width = 540;
  const height = 310;
  const pad = { top: 24, right: 20, bottom: 44, left: 50 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const yMax = 0.55;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Target SCC F1 scores with 95% group-bootstrap confidence intervals",
  });

  for (let i = 0; i <= 5; i++) {
    const val = (yMax * i) / 5;
    const y = pad.top + plotH - (val / yMax) * plotH;
    svg.appendChild(svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y, y2: y, stroke: "currentColor", "stroke-opacity": 0.12 }));
    const lbl = svgEl("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end", fill: "currentColor", "font-size": 10, opacity: 0.72 });
    lbl.textContent = val.toFixed(2);
    svg.appendChild(lbl);
  }

  const arms = juryLab.arm_order;
  const step = plotW / arms.length;
  const barW = Math.min(82, step * 0.5);

  arms.forEach((arm, idx) => {
    const m = juryLab.arms[arm].metrics;
    const cx = pad.left + idx * step + step / 2;
    const barH = (m.target_f1 / yMax) * plotH;
    const yTop = pad.top + plotH - barH;
    const yLo = pad.top + plotH - (m.target_f1_ci_low / yMax) * plotH;
    const yHi = pad.top + plotH - (m.target_f1_ci_high / yMax) * plotH;

    svg.appendChild(svgEl("rect", { x: cx - barW / 2, y: yTop, width: barW, height: barH, rx: 5, fill: ARM_COLORS[arm], "fill-opacity": 0.85 }));
    svg.appendChild(svgEl("line", { x1: cx, x2: cx, y1: yHi, y2: yLo, stroke: "currentColor", "stroke-width": 2 }));
    svg.appendChild(svgEl("line", { x1: cx - 10, x2: cx + 10, y1: yHi, y2: yHi, stroke: "currentColor", "stroke-width": 2 }));
    svg.appendChild(svgEl("line", { x1: cx - 10, x2: cx + 10, y1: yLo, y2: yLo, stroke: "currentColor", "stroke-width": 2 }));

    const valLbl = svgEl("text", { x: cx, y: yHi - 7, "text-anchor": "middle", fill: "currentColor", "font-size": 11, "font-weight": 700 });
    valLbl.textContent = `${m.target_f1.toFixed(3)} [${m.target_f1_ci_low.toFixed(2)}–${m.target_f1_ci_high.toFixed(2)}]`;
    svg.appendChild(valLbl);

    const xLbl = svgEl("text", { x: cx, y: height - 16, "text-anchor": "middle", fill: "currentColor", "font-size": 11, "font-weight": 700 });
    xLbl.textContent = juryLab.arms[arm].short;
    svg.appendChild(xLbl);
  });

  host.replaceChildren(svg);
}

function renderValHistoryChart(juryLab) {
  const host = document.getElementById("chart-val-history");
  const width = 540;
  const height = 270;
  const pad = { top: 20, right: 20, bottom: 38, left: 48 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Validation Macro-F1 across 12 training epochs",
  });

  const yMin = 0.44;
  const yMax = 0.58;
  for (let i = 0; i <= 4; i++) {
    const val = yMin + ((yMax - yMin) * i) / 4;
    const y = pad.top + plotH - ((val - yMin) / (yMax - yMin)) * plotH;
    svg.appendChild(svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y, y2: y, stroke: "currentColor", "stroke-opacity": 0.12 }));
    const lbl = svgEl("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end", fill: "currentColor", "font-size": 10, opacity: 0.72 });
    lbl.textContent = val.toFixed(2);
    svg.appendChild(lbl);
  }

  juryLab.arm_order.forEach((arm, idx) => {
    const rows = juryLab.arms[arm].training_history;
    if (!rows.length) return;
    const d = rows
      .map((r, i) => {
        const x = pad.left + ((r.epoch - 1) / 11) * plotW;
        const y = pad.top + plotH - ((r.macro_f1 - yMin) / (yMax - yMin)) * plotH;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    svg.appendChild(svgEl("path", { d, fill: "none", stroke: ARM_COLORS[arm], "stroke-width": 2.4 }));

    const lx = pad.left + idx * 145;
    svg.appendChild(svgEl("rect", { x: lx, y: height - 13, width: 12, height: 4, fill: ARM_COLORS[arm] }));
    const lt = svgEl("text", { x: lx + 17, y: height - 9, fill: "currentColor", "font-size": 10, "font-weight": 600 });
    lt.textContent = juryLab.arms[arm].short;
    svg.appendChild(lt);
  });

  host.replaceChildren(svg);
}

function renderCGANLossChart(audit) {
  const host = document.getElementById("chart-cgan-loss");
  const rows = audit.loss_history;
  const width = 540;
  const height = 270;
  const pad = { top: 22, right: 20, bottom: 38, left: 48 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const yMax = 2.2;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "cGAN generator and discriminator loss over 80 epochs showing discriminator loss pinned at 2.0",
  });

  for (let i = 0; i <= 4; i++) {
    const v = (yMax * i) / 4;
    const y = pad.top + plotH - (v / yMax) * plotH;
    svg.appendChild(svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y, y2: y, stroke: "currentColor", "stroke-opacity": 0.12 }));
    const lbl = svgEl("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end", fill: "currentColor", "font-size": 10, opacity: 0.72 });
    lbl.textContent = v.toFixed(2);
    svg.appendChild(lbl);
  }

  const dPath = rows
    .map((r, i) => {
      const x = pad.left + ((r.epoch - 1) / (rows.length - 1)) * plotW;
      const y = pad.top + plotH - (Math.min(yMax, r.discriminator_loss) / yMax) * plotH;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const gPath = rows
    .map((r, i) => {
      const x = pad.left + ((r.epoch - 1) / (rows.length - 1)) * plotW;
      const y = pad.top + plotH - (Math.min(yMax, Math.max(0, r.generator_loss)) / yMax) * plotH;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  svg.appendChild(svgEl("path", { d: dPath, fill: "none", stroke: "#dc2626", "stroke-width": 2.4 }));
  svg.appendChild(svgEl("path", { d: gPath, fill: "none", stroke: "#0f766e", "stroke-width": 2.4 }));

  const legend = [
    ["Discriminator Loss (pinned ~2.000)", "#dc2626", 0],
    ["Generator Loss", "#0f766e", 215],
  ];
  legend.forEach(([txt, col, dx]) => {
    svg.appendChild(svgEl("rect", { x: pad.left + dx, y: height - 14, width: 12, height: 4, fill: col }));
    const t = svgEl("text", { x: pad.left + dx + 17, y: height - 10, fill: "currentColor", "font-size": 10, "font-weight": 600 });
    t.textContent = txt;
    svg.appendChild(t);
  });

  host.replaceChildren(svg);
}

function renderPhashHistChart(audit) {
  const host = document.getElementById("chart-phash-hist");
  const bins = audit.real_phash_hist;
  if (!bins.length) return;
  const maxC = Math.max(...bins.map((b) => b.count));

  const width = 520;
  const height = 220;
  const pad = { top: 18, right: 16, bottom: 34, left: 46 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Histogram of perceptual hash distance from cGAN candidates to nearest real SCC image",
  });

  const step = plotW / bins.length;
  const bw = step * 0.72;
  bins.forEach((b, idx) => {
    const bh = (b.count / maxC) * plotH;
    const x = pad.left + idx * step + (step - bw) / 2;
    const y = pad.top + plotH - bh;
    svg.appendChild(svgEl("rect", { x, y, width: bw, height: Math.max(2, bh), rx: 3, fill: b.end <= 4 ? "#dc2626" : "#0f766e" }));
    const xl = svgEl("text", { x: x + bw / 2, y: height - 14, "text-anchor": "middle", fill: "currentColor", "font-size": 9 });
    xl.textContent = b.label;
    svg.appendChild(xl);
  });

  host.replaceChildren(svg);
}

/* ---------------- Overview & Protocol ---------------- */
function populateOverview(overview) {
  document.getElementById("kpi-total-images").textContent = (
    overview.splits.train.images + overview.splits.val.images + overview.splits.test.images
  ).toLocaleString();
  document.getElementById("kpi-split-sub").textContent =
    `${overview.splits.train.images.toLocaleString()} train · ${overview.splits.val.images.toLocaleString()} val · ${overview.splits.test.images.toLocaleString()} test (0 leakage)`;

  const base = overview.arm_summaries.real_only.metrics;
  const cgan = overview.arm_summaries.cgan_scc.metrics;
  const rep = overview.arm_summaries.repeated_real_scc.metrics;

  document.getElementById("kpi-baseline-f1").textContent = base.target_f1.toFixed(3);
  document.getElementById("kpi-cgan-f1").textContent = cgan.target_f1.toFixed(3);
  document.getElementById("kpi-detector-auc").textContent = overview.cgan_audit_summary.synthetic_detector_auc.toFixed(3);

  const stack = document.getElementById("arm-protocol-cards");
  stack.replaceChildren(
    ...overview.arm_order.map((arm) => {
      const info = overview.arm_summaries[arm];
      return el("div", { class: "protocol-item" }, [
        el("div", { class: "protocol-item-head" }, [
          el("span", { text: info.label }),
          el("span", { class: "metric-pill", text: `SCC F1 ${info.metrics.target_f1.toFixed(3)} · PR-AUC ${info.metrics.target_pr_auc.toFixed(3)}` }),
        ]),
        el("p", { class: "muted", text: info.description }),
      ]);
    })
  );

  renderImbalanceChart(overview);
}

/* ---------------- Jury Lab Population ---------------- */
function renderArmDetails(armKey) {
  const juryLab = state.juryLab;
  const arm = juryLab.arms[armKey];
  const names = juryLab.class_names;

  // Confusion matrix
  const cmHost = document.getElementById("confusion-matrix-host");
  const table = el("table", { class: "cm-table" });
  const thead = el("thead", {}, [
    el("tr", {}, [
      el("th", { scope: "col", text: "True \\ Pred" }),
      ...names.map((n) => el("th", { scope: "col", text: n })),
    ]),
  ]);
  const tbody = el("tbody");
  arm.confusion_matrix.forEach((rowVals, rIdx) => {
    const rowSum = Math.max(1, rowVals.reduce((a, b) => a + b, 0));
    const tr = el("tr", {}, [
      el("th", { scope: "row", text: names[rIdx] }),
      ...rowVals.map((val, cIdx) => {
        const ratio = val / rowSum;
        const alpha = Math.min(0.65, 0.06 + ratio * 0.72).toFixed(2);
        const bg = rIdx === cIdx ? `rgba(15, 118, 110, ${alpha})` : val > 0 ? `rgba(217, 119, 6, ${Math.min(0.45, ratio * 0.6).toFixed(2)})` : "transparent";
        return el("td", {
          text: String(val),
          style: `background:${bg};`,
          title: `True ${names[rIdx]} → Pred ${names[cIdx]}: ${val}`,
        });
      }),
    ]);
    tbody.appendChild(tr);
  });
  table.appendChild(thead);
  table.appendChild(tbody);
  cmHost.replaceChildren(table);

  // Per-class table
  const perTbody = document.getElementById("per-class-tbody");
  perTbody.replaceChildren(
    ...arm.per_class.map((row) =>
      el("tr", { class: row.diagnosis === "SCC" ? "is-target-row" : "" }, [
        el("th", { scope: "row", text: row.diagnosis }),
        el("td", { text: row.full_name }),
        el("td", { text: row.precision.toFixed(3) }),
        el("td", { text: row.recall.toFixed(3) }),
        el("td", { text: row.f1.toFixed(3) }),
        el("td", { text: row.support.toLocaleString() }),
      ])
    )
  );
}

function populateJuryLab(juryLab) {
  renderPRCurveChart(juryLab);
  renderBootstrapCIChart(juryLab);
  renderValHistoryChart(juryLab);
  renderArmDetails("real_only");

  document.getElementById("cm-arm-select").addEventListener("change", (ev) => {
    renderArmDetails(ev.target.value);
  });

  // Epoch scrubber
  const epochs = juryLab.cgan_audit.valid_sample_epochs;
  const slider = document.getElementById("epoch-slider");
  slider.max = String(epochs.length - 1);
  slider.value = String(epochs.length - 1);
  const badge = document.getElementById("epoch-badge");
  const img = document.getElementById("epoch-grid-img");

  slider.addEventListener("input", () => {
    const epoch = epochs[Number(slider.value)] || 80;
    badge.textContent = `Epoch ${epoch}`;
    img.src = `/api/media/epoch/${epoch}`;
    img.alt = `cGAN generated sample grid at epoch ${epoch}`;
  });

  renderCGANLossChart(juryLab.cgan_audit);
  renderPhashHistChart(juryLab.cgan_audit);
}

/* ---------------- Live cGAN Sampler ---------------- */
function initCGANSampler() {
  const form = document.getElementById("cgan-sample-form");
  const grid = document.getElementById("cgan-live-grid");
  const btn = document.getElementById("cgan-generate-btn");

  async function generate() {
    const diagnosis = document.getElementById("cgan-diag-select").value;
    const seed = Number(document.getElementById("cgan-seed-input").value || 146);
    btn.disabled = true;
    btn.textContent = "Sampling...";
    try {
      const data = await fetchJSON("/api/cgan/sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ diagnosis, seed, count: 4 }),
      });
      grid.replaceChildren(
        ...data.samples.map((s) =>
          el("div", { class: "cgan-tile" }, [
            el("img", { src: s.image_png, alt: `Live cGAN ${data.diagnosis} sample ${s.index}` }),
            el("div", { class: "cgan-tile-meta", text: `#${s.index} · edge Δ=${s.checkerboard_gradient_score.toFixed(3)}` }),
          ])
        )
      );
      announce(`Generated 4 synthetic ${data.diagnosis} samples`);
    } catch (err) {
      grid.replaceChildren(el("p", { class: "muted", text: `Error: ${err.message}` }));
    } finally {
      btn.disabled = false;
      btn.textContent = "Generate 4 Samples";
    }
  }

  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    generate();
  });
  generate();
}

/* ---------------- Blinded 24-Panel Quiz ---------------- */
function updateQuizProgress() {
  const txt = document.getElementById("quiz-progress-text");
  txt.textContent = `${state.quizGuesses.size} / ${state.quizPanels.length} answered`;
}

function renderQuizGrid() {
  const host = document.getElementById("quiz-grid");
  const revealMap = new Map((state.quizResult?.revealed || []).map((r) => [r.panel_id, r]));

  host.replaceChildren(
    ...state.quizPanels.map((p) => {
      const guess = state.quizGuesses.get(p.panel_id);
      const rev = revealMap.get(p.panel_id);
      let cls = "quiz-card";
      if (rev) cls += rev.correct ? " is-correct" : " is-wrong";

      const btnRow = el("div", { class: "quiz-btn-row" }, [
        ...["real", "synthetic", "unsure"].map((choice) => {
          const btn = el("button", {
            type: "button",
            class: "quiz-choice-btn",
            "aria-pressed": String(guess === choice),
            text: choice === "real" ? "Real" : choice === "synthetic" ? "Synthetic" : "Unsure",
          });
          btn.addEventListener("click", () => {
            state.quizGuesses.set(p.panel_id, choice);
            state.quizResult = null;
            document.getElementById("quiz-score-banner").hidden = true;
            updateQuizProgress();
            renderQuizGrid();
          });
          return btn;
        }),
      ]);

      const children = [
        el("img", { src: p.image_url, alt: `Blinded skin lesion panel ${p.index}`, loading: "lazy" }),
        el("strong", { text: `Panel #${String(p.index).padStart(2, "0")}` }),
        btnRow,
      ];
      if (rev) {
        children.push(
          el("span", {
            class: "cgan-tile-meta",
            text: `${rev.correct ? "✓ Correct" : "✕ Actual"}: ${rev.actual_source.toUpperCase()}`,
          })
        );
      }
      return el("div", { class: cls }, children);
    })
  );
}

function initQuiz(quizData) {
  state.quizPanels = quizData.panels;
  updateQuizProgress();
  renderQuizGrid();

  document.getElementById("quiz-reset-btn").addEventListener("click", () => {
    state.quizGuesses.clear();
    state.quizResult = null;
    document.getElementById("quiz-score-banner").hidden = true;
    updateQuizProgress();
    renderQuizGrid();
  });

  document.getElementById("quiz-submit-btn").addEventListener("click", async () => {
    const guesses = state.quizPanels.map((p) => ({
      panel_id: p.panel_id,
      choice: state.quizGuesses.get(p.panel_id) || "unsure",
    }));
    const result = await fetchJSON("/api/quiz/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ guesses }),
    });
    state.quizResult = result;
    const banner = document.getElementById("quiz-score-banner");
    banner.hidden = false;
    banner.replaceChildren(
      el("strong", {
        text: `Jury Score: ${result.correct_count} / ${result.total_panels} (${(result.overall_accuracy * 100).toFixed(1)}%) — `,
      }),
      el("span", {
        text: `Real identified: ${result.real_correct}/${result.real_total} · Synthetic spotted: ${result.synthetic_correct}/${result.synthetic_total} · Automated pixel-stats detector AUC: ${result.statistical_detector_auc.toFixed(3)}`,
      })
    );
    renderQuizGrid();
    announce(`Quiz scored: ${result.correct_count} of ${result.total_panels} correct`);
  });
}

/* ---------------- Live Analyzer & Grad-CAM ---------------- */
function renderSampleGallery() {
  const host = document.getElementById("sample-gallery");
  const filtered = state.samples.filter((s) => {
    if (state.sampleFilter === "all") return true;
    if (state.sampleFilter === "SCC") return s.source === "real_test" && s.ground_truth === "SCC";
    if (state.sampleFilter === "other") return s.source === "real_test" && s.ground_truth !== "SCC";
    if (state.sampleFilter === "synthetic") return s.source === "synthetic_cgan";
    return true;
  });

  host.replaceChildren(
    ...filtered.map((s) => {
      const btn = el("button", {
        type: "button",
        class: `sample-tile${state.selectedSampleId === s.sample_id ? " active" : ""}`,
      }, [
        el("img", { src: s.image_url, alt: `${s.category}: ${s.image_name}`, loading: "lazy" }),
        el("span", { class: "sample-tile-title", text: s.category }),
        el("span", { class: "sample-tile-sub", text: `${s.ground_truth} · ${s.image_name}` }),
      ]);
      btn.addEventListener("click", () => {
        state.selectedSampleId = s.sample_id;
        state.uploadedFile = null;
        document.getElementById("run-upload-btn").disabled = true;
        renderSampleGallery();
        runAnalysis({ sampleId: s.sample_id });
      });
      return btn;
    })
  );
}

function renderAnalysisOutput() {
  const data = state.lastAnalysis;
  const shell = document.getElementById("analysis-results-shell");
  if (!data) {
    shell.hidden = true;
    return;
  }
  shell.hidden = false;
  document.getElementById("open-report-btn").disabled = false;

  const camLayer = document.getElementById("cam-view-select").value;
  const meta = data.sample_metadata;
  const gtLabel = meta ? `Ground Truth: ${meta.ground_truth}` : "Custom Uploaded Lesion";
  const consensusTxt = data.arm_consensus
    ? `All 3 arms agree: ${data.consensus_diagnosis}`
    : "Cross-arm prediction disagreement";

  const bar = document.getElementById("analysis-summary-bar");
  bar.replaceChildren(
    el("div", {}, [
      el("strong", { text: `${data.case_label} · ${gtLabel}` }),
      el("p", { class: "muted", text: meta ? meta.note : `Running on ${data.device.toUpperCase()} backend` }),
    ]),
    el("div", { class: "quiz-actions" }, [
      el("span", { class: "metric-pill", text: consensusTxt }),
      el("span", { class: "metric-pill", text: `Peak SCC Prob: ${(data.max_scc_probability * 100).toFixed(1)}%` }),
    ])
  );

  const grid = document.getElementById("analysis-arm-grid");
  grid.replaceChildren(
    ...Object.values(data.arms).map((arm) => {
      const imgSrc = camLayer === "original" ? data.original_png : camLayer === "heatmap" ? arm.heatmap_png : arm.overlay_png;
      const top4 = arm.probabilities.slice(0, 4);
      return el("article", { class: "arm-card" }, [
        el("div", { class: "arm-card-head" }, [
          el("strong", { text: arm.label }),
          el("span", { class: "metric-pill", text: `${arm.predicted_diagnosis} ${(arm.predicted_confidence * 100).toFixed(1)}%` }),
        ]),
        el("div", { class: "cam-image-box" }, [
          el("img", { src: imgSrc, alt: `${arm.short} Grad-CAM view targeting ${arm.gradcam_target}` }),
        ]),
        el("div", { class: "cgan-tile-meta", text: `Grad-CAM target: ${arm.gradcam_target} · SCC prob: ${(arm.scc_probability * 100).toFixed(1)}%` }),
        el("div", { class: "prob-list" }, top4.map((p) =>
          el("div", { class: "prob-row" }, [
            el("strong", { text: p.diagnosis }),
            el("div", { class: "prob-track" }, [
              el("div", { class: `prob-fill${p.diagnosis === "SCC" ? " is-scc" : ""}`, style: `width:${Math.max(2, p.probability * 100).toFixed(1)}%` }),
            ]),
            el("span", { text: `${(p.probability * 100).toFixed(1)}%` }),
          ])
        )),
      ]);
    })
  );
}

async function runAnalysis({ sampleId = null, file = null } = {}) {
  const form = new FormData();
  form.append("gradcam_mode", document.getElementById("gradcam-mode-select").value);
  if (file) form.append("file", file);
  else if (sampleId) form.append("sample_id", sampleId);
  else return;

  const summary = document.getElementById("analysis-summary-bar");
  document.getElementById("analysis-results-shell").hidden = false;
  summary.replaceChildren(el("strong", { text: "Running 3-arm EfficientNet-B0 inference & Grad-CAM..." }));

  try {
    const result = await fetchJSON("/api/analyze", { method: "POST", body: form });
    state.lastAnalysis = result;
    renderAnalysisOutput();
    announce(`Analyzed ${result.case_label}`);
  } catch (err) {
    summary.replaceChildren(el("strong", { text: `Analysis error: ${err.message}` }));
  }
}

function initAnalyzer(samplesData) {
  state.samples = samplesData.samples;
  renderSampleGallery();

  document.querySelectorAll("[data-sample-filter]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-sample-filter]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.sampleFilter = btn.dataset.sampleFilter;
      renderSampleGallery();
    });
  });

  const fileInput = document.getElementById("lesion-file-input");
  const runUploadBtn = document.getElementById("run-upload-btn");
  const dropzone = document.getElementById("upload-dropzone");

  fileInput.addEventListener("change", () => {
    state.uploadedFile = fileInput.files?.[0] || null;
    state.selectedSampleId = null;
    runUploadBtn.disabled = !state.uploadedFile;
    renderSampleGallery();
  });

  ["dragenter", "dragover"].forEach((evName) => {
    dropzone.addEventListener(evName, (ev) => {
      ev.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evName) => {
    dropzone.addEventListener(evName, (ev) => {
      ev.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (ev) => {
    const dropped = ev.dataTransfer?.files?.[0];
    if (dropped) {
      state.uploadedFile = dropped;
      state.selectedSampleId = null;
      runUploadBtn.disabled = false;
      runAnalysis({ file: dropped });
    }
  });

  runUploadBtn.addEventListener("click", () => {
    if (state.uploadedFile) runAnalysis({ file: state.uploadedFile });
  });

  document.getElementById("cam-view-select").addEventListener("change", () => renderAnalysisOutput());
  document.getElementById("gradcam-mode-select").addEventListener("change", () => {
    if (state.uploadedFile) runAnalysis({ file: state.uploadedFile });
    else if (state.selectedSampleId) runAnalysis({ sampleId: state.selectedSampleId });
  });

  if (state.samples.length) {
    state.selectedSampleId = state.samples[0].sample_id;
    renderSampleGallery();
  }
}

/* ---------------- Report Dialog (with Safari closedby fallback) ---------------- */
function initReportDialog() {
  const dialog = document.getElementById("report-dialog");
  const openBtn = document.getElementById("open-report-btn");
  const closeBtn = document.getElementById("close-report-dialog");
  const refreshBtn = document.getElementById("refresh-report-btn");
  const downloadBtn = document.getElementById("download-md-btn");
  const printBtn = document.getElementById("print-report-btn");
  const frame = document.getElementById("report-preview-frame");

  if (!("closedBy" in HTMLDialogElement.prototype)) {
    dialog.addEventListener("click", (ev) => {
      if (ev.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      const inside = rect.top <= ev.clientY && ev.clientY <= rect.top + rect.height &&
                     rect.left <= ev.clientX && ev.clientX <= rect.left + rect.width;
      if (!inside) dialog.close();
    });
  }

  async function buildReport() {
    if (!state.lastAnalysis) return;
    const notes = document.getElementById("reviewer-notes-input").value;
    const res = await fetchJSON("/api/report/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        case_label: state.lastAnalysis.case_label,
        reviewer_notes: notes,
        analysis: state.lastAnalysis,
      }),
    });
    state.lastReport = res;
    frame.srcdoc = res.html;
  }

  openBtn.addEventListener("click", async () => {
    await buildReport();
    dialog.showModal();
  });
  closeBtn.addEventListener("click", () => dialog.close());
  refreshBtn.addEventListener("click", buildReport);
  printBtn.addEventListener("click", () => frame.contentWindow?.print());
  downloadBtn.addEventListener("click", () => {
    if (!state.lastReport) return;
    const blob = new Blob([state.lastReport.markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `genai-jury-report-${(state.lastAnalysis?.case_label || "case").replace(/\W+/g, "-")}.md`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

/* ---------------- Future Roadmap Hub ---------------- */
function renderSelectedModality(roadmap) {
  const mod = roadmap.modalities.find((m) => m.id === state.selectedModality) || roadmap.modalities[0];
  const host = document.getElementById("modality-detail");
  host.replaceChildren(
    el("div", { class: "card-header" }, [
      el("div", {}, [
        el("h3", { text: mod.name }),
        el("p", { class: "muted", text: mod.status }),
      ]),
      el("span", { class: "metric-pill", text: mod.badge }),
    ]),
    el("div", { class: "modality-kv" }, [
      el("div", {}, [el("strong", { text: "Accepted Formats: " }), el("span", { text: mod.formats.join(", ") })]),
      el("div", {}, [el("strong", { text: "Preprocessing Stack: " }), el("span", { text: mod.preprocessing })]),
      el("div", {}, [el("strong", { text: "Model Architecture: " }), el("span", { text: mod.models })]),
      el("div", {}, [el("strong", { text: "Explainability Output: " }), el("span", { text: mod.explainability })]),
      el("div", {}, [el("strong", { text: "Target Metrics: " }), el("span", { text: mod.target_metrics })]),
      el("div", {}, [el("strong", { text: "Next Milestone: " }), el("span", { text: mod.next_milestone })]),
    ])
  );
}

function populateRoadmap(roadmap) {
  state.roadmap = roadmap;
  document.getElementById("roadmap-cgan-summary").textContent = roadmap.cgan_upgrade_diff.summary;

  document.getElementById("roadmap-diff-grid").replaceChildren(
    ...roadmap.cgan_upgrade_diff.comparisons.map((c) =>
      el("article", { class: "diff-card" }, [
        el("h3", { text: c.dimension }),
        el("p", { class: "muted" }, [el("strong", { text: "v1 (Trained): " }), c.v1_current]),
        el("p", { class: "muted" }, [el("strong", { text: "v2 (Proposed): " }), c.v2_proposed]),
        el("p", { class: "cgan-tile-meta", text: `Impact: ${c.expected_impact}` }),
      ])
    )
  );

  const selector = document.getElementById("modality-selector");
  function drawPills() {
    selector.replaceChildren(
      ...roadmap.modalities.map((m) => {
        const btn = el("button", {
          type: "button",
          class: `pill-btn${m.id === state.selectedModality ? " active" : ""}`,
          text: `${m.name} (${m.badge})`,
        });
        btn.addEventListener("click", () => {
          state.selectedModality = m.id;
          drawPills();
          renderSelectedModality(roadmap);
        });
        return btn;
      })
    );
  }
  drawPills();
  renderSelectedModality(roadmap);

  document.getElementById("hardware-tiers-list").replaceChildren(
    ...roadmap.hardware_tiers.map((h) =>
      el("div", { class: "stack-item" }, [
        el("strong", { text: h.tier }),
        el("p", { class: "muted", text: `${h.specs} — ${h.capabilities}` }),
        el("span", { class: "cgan-tile-meta", text: `Constraint: ${h.limitations}` }),
      ])
    )
  );

  document.getElementById("rai-controls-list").replaceChildren(
    ...roadmap.responsible_ai_controls.map((r) =>
      el("div", { class: "stack-item" }, [
        el("div", { class: "protocol-item-head" }, [
          el("strong", { text: r.area }),
          el("span", { class: "metric-pill", text: r.status }),
        ]),
        el("span", { class: "muted", text: r.detail }),
      ])
    )
  );

  document.getElementById("methodology-timeline").replaceChildren(
    ...roadmap.methodology_phases.map((p) =>
      el("div", { class: "phase-card" }, [
        el("strong", { text: p.phase }),
        el("p", { class: "muted", text: p.activity }),
        el("span", { class: "metric-pill", text: p.artifact }),
      ])
    )
  );
}

/* ---------------- Bootstrap Application ---------------- */
async function bootstrap() {
  initThemeToggle();
  initNavigation();
  initLabTabs();
  initReportDialog();

  const [overview, juryLab, samples, quizData, roadmap] = await Promise.all([
    fetchJSON("/api/overview"),
    fetchJSON("/api/jury-lab"),
    fetchJSON("/api/samples"),
    fetchJSON("/api/quiz/panels"),
    fetchJSON("/api/roadmap"),
  ]);

  state.overview = overview;
  state.juryLab = juryLab;

  populateOverview(overview);
  initAnalyzer(samples);
  populateJuryLab(juryLab);
  initCGANSampler();
  initQuiz(quizData);
  populateRoadmap(roadmap);
}

window.addEventListener("DOMContentLoaded", () => {
  bootstrap().catch((err) => {
    console.error("Failed to bootstrap GenAI Jury UI:", err);
  });
});
