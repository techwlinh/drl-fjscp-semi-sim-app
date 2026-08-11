// State Management
let datasetData = null;
let currentZoom = 1.0;
let selectedArea = "ALL";
let selectedProduct = "ALL";
let selectedPriority = "ALL";
let groupMode = "ws"; // 'ws' (Group by Workstation), 'wsg' (Group by WS Group), 'tool' (Flat Tool)
let detailLevel = "full"; // 'full' or 'compact'
const collapsedIds = new Set();

const TIME_SCALE_BASE = 0.5; // 0.5 pixels per minute at 1.0x zoom

// Product Color Palette Generator
const PRODUCT_PALETTE = {
  Product_A: { bg: "linear-gradient(135deg, #3b82f6, #1d4ed8)", border: "#60a5fa", color: "#3b82f6" },
  Product_B: { bg: "linear-gradient(135deg, #10b981, #047857)", border: "#34d399", color: "#10b981" },
  Product_C: { bg: "linear-gradient(135deg, #8b5cf6, #6d28d9)", border: "#a78bfa", color: "#8b5cf6" },
  Product_D: { bg: "linear-gradient(135deg, #f59e0b, #b45309)", border: "#fbbf24", color: "#f59e0b" },
  Product_E: { bg: "linear-gradient(135deg, #ec4899, #be185d)", border: "#f472b6", color: "#ec4899" },
  Product_F: { bg: "linear-gradient(135deg, #06b6d4, #0e7490)", border: "#67e8f9", color: "#06b6d4" },
  Product_G: { bg: "linear-gradient(135deg, #84cc16, #4d7c0f)", border: "#a3e635", color: "#84cc16" },
};

function getProductStyle(productName) {
  if (PRODUCT_PALETTE[productName]) return PRODUCT_PALETTE[productName];
  // Deterministic HSL generator for dynamic product types
  let hash = 0;
  for (let i = 0; i < (productName || "").length; i++) {
    hash = productName.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return {
    bg: `linear-gradient(135deg, hsl(${hue}, 75%, 45%), hsl(${hue}, 85%, 32%))`,
    border: `hsl(${hue}, 80%, 65%)`,
    color: `hsl(${hue}, 75%, 45%)`,
  };
}

// DOM Elements
const kpiMakespan = document.getElementById("kpi-makespan");
const kpiMakespanHours = document.getElementById("kpi-makespan-hours");
const kpiTardiness = document.getElementById("kpi-tardiness");
const kpiTardyCount = document.getElementById("kpi-tardy-count");
const kpiOntime = document.getElementById("kpi-ontime");
const kpiTotalJobs = document.getElementById("kpi-total-jobs");
const kpiSetup = document.getElementById("kpi-setup");
const kpiUtilization = document.getElementById("kpi-utilization");
const kpiTotalTasks = document.getElementById("kpi-total-tasks");

const selectExperiment = document.getElementById("select-experiment");
const filterArea = document.getElementById("filter-area");
const filterProduct = document.getElementById("filter-product");
const filterJob = document.getElementById("filter-job");
const filterPriority = document.getElementById("filter-priority");
const selectGroupMode = document.getElementById("group-mode");
const selectDetailLevel = document.getElementById("detail-level");
const searchJob = document.getElementById("search-job");
const zoomSlider = document.getElementById("zoom-slider");
const zoomVal = document.getElementById("zoom-val");

const btnResetHighlight = document.getElementById("btn-reset-highlight");
let selectedJob = "ALL";
let highlightedJob = null;



const dynamicLegend = document.getElementById("dynamic-legend");
const btnResetProductFilter = document.getElementById("btn-reset-product-filter");

const sidebarRows = document.getElementById("sidebar-rows");
const timelineRuler = document.getElementById("timeline-ruler");
const timelineBody = document.getElementById("timeline-body");
const tooltip = document.getElementById("tooltip");
const btnReload = document.getElementById("btn-reload");
const fileInput = document.getElementById("file-input");
const btnThemeToggle = document.getElementById("btn-theme-toggle");
const btnToggleCollapse = document.getElementById("btn-toggle-collapse");

// Tab Navigation
const tabButtons = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const targetTab = btn.getAttribute("data-tab");
    tabButtons.forEach((b) => b.classList.remove("active"));
    tabContents.forEach((c) => c.classList.remove("active"));

    btn.classList.add("active");
    document.getElementById(targetTab).classList.add("active");
  });
});

const selectModel = document.getElementById("select-model");

// Load Experiments Manifest for Dropdown Selector
async function loadExperimentsManifest() {
  // Default model options pointing to experiments subfolders
  if (selectModel) {
    selectModel.innerHTML = `
      <option value="/experiments/ppo/schedule.json">🤖 PPO Deep Reinforcement Learning</option>
      <option value="/experiments/ga/schedule.json">🧬 GA Metaheuristic</option>
    `;
  }

  try {
    const res = await fetch("/experiments/manifest.json");
    if (!res.ok) return;
    const manifest = await res.json();
    if (!manifest.experiments || !Array.isArray(manifest.experiments)) return;

    // Rebuild model selector from manifest
    if (selectModel) {
      selectModel.innerHTML = "";
      manifest.experiments.forEach((exp) => {
        const opt = document.createElement("option");
        opt.value = exp.rel_path || `/experiments/${exp.algorithm}/schedule.json`;
        opt.textContent = `${exp.title || exp.algorithm.toUpperCase()} (Fit: ${Number(exp.fitness).toFixed(0)})`;
        selectModel.appendChild(opt);
      });
    }

    if (selectExperiment) {
      selectExperiment.innerHTML = "";
      manifest.experiments.forEach((exp) => {
        const opt = document.createElement("option");
        opt.value = exp.rel_path || `/experiments/${exp.algorithm}/schedule.json`;
        opt.textContent = `${exp.title || exp.algorithm.toUpperCase()}`;
        selectExperiment.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn("Could not load experiments manifest:", err);
  }
}

// Load Initial / Selected Data
async function loadData(url = null) {
  const targetUrl = url || (selectModel ? selectModel.value : "/experiments/ppo/schedule.json");
  try {
    const res = await fetch(targetUrl);
    if (!res.ok) throw new Error("Dataset file not found");
    datasetData = await res.json();
    initDashboard();
  } catch (err) {
    console.error("Failed to load dataset:", err);
    sidebarRows.innerHTML = `<div class="sidebar-row" style="color:#ef4444; padding:12px;">Failed to load dataset: ${targetUrl}</div>`;
  }
}


function initDashboard() {
  if (!datasetData) return;

  renderKPIs(datasetData.kpis);
  populateAreaFilter(datasetData.factory_hierarchy);
  populateProductFilter();
  populateJobFilter();
  renderGanttChart();
  renderProgressSection(datasetData.history);
  renderBenchmarkSection(datasetData.heuristic_comparisons);
}

// ----------------------------------------------------------------------
// Job / Lot Filter Setup
// ----------------------------------------------------------------------
function populateJobFilter() {
  if (!filterJob || !datasetData || !datasetData.tasks) return;
  const currentVal = filterJob.value || "ALL";

  const jobSet = new Set();
  datasetData.tasks.forEach((t) => jobSet.add(t.job_id));

  const sortedJobs = Array.from(jobSet).sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" })
  );

  filterJob.innerHTML = `<option value="ALL">All Jobs (${sortedJobs.length} lots)</option>`;
  sortedJobs.forEach((jId) => {
    const opt = document.createElement("option");
    opt.value = jId;
    opt.textContent = `Job/Lot: ${jId}`;
    filterJob.appendChild(opt);
  });

  if (jobSet.has(currentVal)) {
    filterJob.value = currentVal;
  } else {
    filterJob.value = "ALL";
  }
}


// ----------------------------------------------------------------------
// Render KPI Cards
// ----------------------------------------------------------------------
function renderKPIs(kpis) {
  if (!kpis) return;

  const makespanMins = kpis.makespan || 0;
  const makespanHours = (makespanMins / 60).toFixed(1);

  kpiMakespan.textContent = `${makespanMins.toLocaleString()}m`;
  kpiMakespanHours.textContent = `${makespanHours} hours`;

  kpiTardiness.textContent = (kpis.total_weighted_tardiness || 0).toLocaleString();
  kpiTardyCount.textContent = `${kpis.tardy_jobs || 0} tardy jobs`;

  kpiOntime.textContent = `${kpis.on_time_rate_percent || 0}%`;
  kpiTotalJobs.textContent = `${kpis.total_jobs || 0} total lots`;

  kpiSetup.textContent = `${(kpis.total_setup_time || 0).toLocaleString()} min`;
  kpiUtilization.textContent = `${kpis.avg_tool_utilization_percent || 0}%`;
  kpiTotalTasks.textContent = `${kpis.total_scheduled_tasks || 0} scheduled tasks`;
}

// ----------------------------------------------------------------------
// Area Filter Setup
// ----------------------------------------------------------------------
function populateAreaFilter(factoryHierarchy) {
  if (!factoryHierarchy) return;
  filterArea.innerHTML = `<option value="ALL">All Areas</option>`;

  factoryHierarchy.forEach((area) => {
    const opt = document.createElement("option");
    opt.value = area.area_id;
    opt.textContent = `Area: ${area.area_id}`;
    filterArea.appendChild(opt);
  });
}

// ----------------------------------------------------------------------
// Product Stats & Dynamic Product Legend Filter
// ----------------------------------------------------------------------
function getProductStats() {
  if (!datasetData || !datasetData.tasks) return {};
  const stats = {};
  datasetData.tasks.forEach((t) => {
    const p = t.product_type || "Generic";
    if (!stats[p]) {
      stats[p] = { product_type: p, lots: new Set(), total_tasks: 0 };
    }
    stats[p].lots.add(t.job_id);
    stats[p].total_tasks++;
  });
  return stats;
}

function populateProductFilter() {
  const stats = getProductStats();
  const productKeys = Object.keys(stats).sort();

  // Populate Dropdown
  filterProduct.innerHTML = `<option value="ALL">All Products (${datasetData.kpis.total_jobs || 0} lots)</option>`;
  productKeys.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = `${p} (${stats[p].lots.size} lots, ${stats[p].total_tasks} tasks)`;
    if (p === selectedProduct) opt.selected = true;
    filterProduct.appendChild(opt);
  });

  // Populate Dynamic Legend Chips
  dynamicLegend.innerHTML = "";
  productKeys.forEach((p) => {
    const style = getProductStyle(p);
    const chip = document.createElement("div");
    const isActive = selectedProduct === p;
    chip.className = `legend-chip ${isActive ? "active" : ""}`;
    chip.setAttribute("data-product", p);

    chip.innerHTML = `
      <span class="chip-color" style="background:${style.bg}; border-color:${style.border}"></span>
      <span class="chip-label">${p}</span>
      <span class="chip-count">${stats[p].lots.size} lots</span>
    `;

    chip.addEventListener("click", () => {
      if (selectedProduct === p) {
        selectedProduct = "ALL";
        filterProduct.value = "ALL";
      } else {
        selectedProduct = p;
        filterProduct.value = p;
      }
      updateProductChipStates();
      renderGanttChart();
    });

    dynamicLegend.appendChild(chip);
  });

  btnResetProductFilter.classList.toggle("hidden", selectedProduct === "ALL");
}

function updateProductChipStates() {
  const chips = dynamicLegend.querySelectorAll(".legend-chip");
  chips.forEach((c) => {
    const p = c.getAttribute("data-product");
    if (selectedProduct === "ALL") {
      c.classList.remove("active", "dimmed");
    } else if (p === selectedProduct) {
      c.classList.add("active");
      c.classList.remove("dimmed");
    } else {
      c.classList.remove("active");
      c.classList.add("dimmed");
    }
  });
  btnResetProductFilter.classList.toggle("hidden", selectedProduct === "ALL");
}

// ----------------------------------------------------------------------
// Render Gantt Chart & Workstation Grouping
// ----------------------------------------------------------------------
function renderGanttChart() {
  if (!datasetData) return;

  selectedArea = filterArea.value;
  selectedProduct = filterProduct.value;
  selectedJob = filterJob ? filterJob.value : "ALL";
  selectedPriority = filterPriority.value;
  groupMode = selectGroupMode.value;
  detailLevel = selectDetailLevel.value;

  const searchTerm = searchJob.value.trim().toLowerCase();
  const pxPerMin = TIME_SCALE_BASE * currentZoom;
  const maxTime = datasetData.kpis.makespan || 1000;

  updateProductChipStates();

  // Filter Tasks
  const filteredTasks = datasetData.tasks.filter((t) => {
    if (selectedArea !== "ALL" && t.area_id !== selectedArea) return false;
    if (selectedProduct !== "ALL" && t.product_type !== selectedProduct) return false;
    if (selectedJob !== "ALL" && t.job_id !== selectedJob) return false;
    if (selectedPriority !== "ALL" && t.priority !== selectedPriority) return false;
    if (searchTerm) {
      const matchJob = t.job_id.toLowerCase().includes(searchTerm);
      const matchProd = t.product_type.toLowerCase().includes(searchTerm);
      const matchTool = t.tool_id.toLowerCase().includes(searchTerm);
      const matchWs = t.ws_id.toLowerCase().includes(searchTerm);
      if (!matchJob && !matchProd && !matchTool && !matchWs) return false;
    }
    return true;
  });

  // Track tools containing operations of filtered tasks
  const activeToolIds = new Set(filteredTasks.map((t) => t.tool_id));
  const isFilterActive =
    selectedArea !== "ALL" ||
    selectedProduct !== "ALL" ||
    selectedJob !== "ALL" ||
    selectedPriority !== "ALL" ||
    searchTerm !== "";

  const shouldRenderTool = (toolId) => {
    if (isFilterActive) {
      return activeToolIds.has(toolId);
    }
    return true;
  };

  // Extract Visible Hierarchy Rows based on groupMode
  const displayRows = [];

  datasetData.factory_hierarchy.forEach((area) => {
    if (selectedArea !== "ALL" && area.area_id !== selectedArea) return;

    const areaId = area.area_id;
    const isAreaCollapsed = collapsedIds.has(`area_${areaId}`);

    const areaRowIndex = displayRows.length;
    displayRows.push({
      type: "area",
      id: `area_${areaId}`,
      label: `🏭 Area: ${areaId}`,
      area_id: areaId,
      isCollapsed: isAreaCollapsed,
    });

    let areaChildCount = 0;

    area.workstation_groups.forEach((wsg) => {
      const wsgId = wsg.wsg_id;
      const isWsgCollapsed = isAreaCollapsed || collapsedIds.has(`wsg_${wsgId}`);

      if (groupMode === "wsg") {
        // Grouping by WS Group
        let wsgToolsCount = 0;
        wsg.workstations.forEach((ws) => {
          ws.tools.forEach((toolId) => {
            if (shouldRenderTool(toolId)) {
              wsgToolsCount++;
              if (!isWsgCollapsed) {
                displayRows.push({
                  type: "tool",
                  id: toolId,
                  tool_id: toolId,
                  ws_id: ws.ws_id,
                  wsg_id: wsgId,
                  area_id: areaId,
                  level: 1,
                });
              }
            }
          });
        });
        if (wsgToolsCount > 0) areaChildCount++;
      } else if (groupMode === "ws") {
        // Grouping by Workstation (Recommended)
        if (!isAreaCollapsed) {
          let wsgMatchingToolsCount = 0;
          const wsgRowIndex = displayRows.length;
          displayRows.push({
            type: "wsg",
            id: `wsg_${wsgId}`,
            label: `📁 Group: ${wsgId}`,
            wsg_id: wsgId,
            area_id: areaId,
            isCollapsed: isWsgCollapsed,
            level: 1,
          });

          wsg.workstations.forEach((ws) => {
            const wsId = ws.ws_id;
            const matchingTools = ws.tools.filter(shouldRenderTool);

            if (matchingTools.length > 0) {
              wsgMatchingToolsCount += matchingTools.length;
              const isWsCollapsed = isWsgCollapsed || collapsedIds.has(`ws_${wsId}`);

              if (!isWsgCollapsed) {
                displayRows.push({
                  type: "ws",
                  id: `ws_${wsId}`,
                  label: `⚙️ WS: ${wsId}`,
                  ws_id: wsId,
                  wsg_id: wsgId,
                  area_id: areaId,
                  isCollapsed: isWsCollapsed,
                  toolCount: matchingTools.length,
                  level: 2,
                });

                if (!isWsCollapsed) {
                  matchingTools.forEach((toolId) => {
                    displayRows.push({
                      type: "tool",
                      id: toolId,
                      tool_id: toolId,
                      ws_id: wsId,
                      wsg_id: wsgId,
                      area_id: areaId,
                      level: 3,
                    });
                  });
                }
              }
            }
          });

          if (wsgMatchingToolsCount === 0) {
            displayRows.splice(wsgRowIndex, 1);
          } else {
            areaChildCount++;
          }
        }
      } else {
        // Flat Tool View
        wsg.workstations.forEach((ws) => {
          ws.tools.forEach((toolId) => {
            if (shouldRenderTool(toolId)) {
              if (!isAreaCollapsed) {
                displayRows.push({
                  type: "tool",
                  id: toolId,
                  tool_id: toolId,
                  ws_id: ws.ws_id,
                  wsg_id: wsgId,
                  area_id: areaId,
                  level: 1,
                });
              }
              areaChildCount++;
            }
          });
        });
      }
    });

    if (areaChildCount === 0) {
      displayRows.splice(areaRowIndex, 1);
    }
  });


  // 1. Render Sidebar Rows
  sidebarRows.innerHTML = "";
  displayRows.forEach((row) => {
    const div = document.createElement("div");
    let rowClass = "sidebar-row ";

    if (row.type === "area") {
      rowClass += "area-header";
      div.innerHTML = `
        <span class="chevron">${row.isCollapsed ? "▶" : "▼"}</span>
        <span>${row.label}</span>
      `;
      div.addEventListener("click", () => toggleCollapse(row.id));
    } else if (row.type === "wsg") {
      rowClass += "wsg-header level-1";
      div.innerHTML = `
        <span class="chevron">${row.isCollapsed ? "▶" : "▼"}</span>
        <span>${row.label}</span>
      `;
      div.addEventListener("click", () => toggleCollapse(row.id));
    } else if (row.type === "ws") {
      rowClass += "ws-header level-2";
      div.innerHTML = `
        <span class="chevron">${row.isCollapsed ? "▶" : "▼"}</span>
        <span>${row.label} <small>(${row.toolCount} tools)</small></span>
      `;
      div.addEventListener("click", () => toggleCollapse(row.id));
    } else {
      rowClass += `tool-item level-${row.level || 1}`;
      div.innerHTML = `<span>🔧 ${row.tool_id}</span>`;
    }

    div.className = rowClass;
    sidebarRows.appendChild(div);
  });

  // 2. Render Timeline Ruler with Adaptive Non-Overlapping Tick Interval
  timelineRuler.innerHTML = "";
  const totalWidthPx = maxTime * pxPerMin + 250;
  timelineRuler.style.width = `${totalWidthPx}px`;

  // Ensure ruler tick labels have at least 160px spacing to prevent overlap
  const MIN_TICK_PIXELS = 160;
  const candidateIntervals = [30, 60, 120, 180, 240, 360, 480, 720, 1440, 2880, 5760, 10080];
  let tickIntervalMins = 360;
  for (const candidate of candidateIntervals) {
    if (candidate * pxPerMin >= MIN_TICK_PIXELS) {
      tickIntervalMins = candidate;
      break;
    }
    tickIntervalMins = candidate;
  }

  const tickWidthPx = tickIntervalMins * pxPerMin;
  for (let t = 0; t <= maxTime + tickIntervalMins; t += tickIntervalMins) {
    const tick = document.createElement("div");
    tick.className = "ruler-tick";
    tick.style.width = `${tickWidthPx}px`;
    const hours = (t / 60).toFixed(0);
    tick.textContent = t === 0 ? "0m" : `${t}m (${hours}h)`;
    timelineRuler.appendChild(tick);
  }

  // 3. Render Timeline Body & Tasks
  timelineBody.innerHTML = "";
  timelineBody.style.width = `${totalWidthPx}px`;

  // Draw Vertical Grid Lines
  const gridContainer = document.createElement("div");
  gridContainer.className = "timeline-grid-bg";
  gridContainer.style.width = `${totalWidthPx}px`;
  for (let t = 0; t <= maxTime + tickIntervalMins; t += tickIntervalMins) {
    const line = document.createElement("div");
    line.className = "grid-line";
    line.style.left = `${t * pxPerMin}px`;
    gridContainer.appendChild(line);
  }
  timelineBody.appendChild(gridContainer);

  const tasksByTool = {};
  filteredTasks.forEach((t) => {
    if (!tasksByTool[t.tool_id]) tasksByTool[t.tool_id] = [];
    tasksByTool[t.tool_id].push(t);
  });

  if (btnResetHighlight) {
    if (highlightedJob) {
      btnResetHighlight.textContent = `✨ Clear Highlight (${highlightedJob})`;
      btnResetHighlight.classList.remove("hidden");
    } else {
      btnResetHighlight.classList.add("hidden");
    }
  }

  const applyHighlightClass = (blockElement, jobId) => {
    if (highlightedJob) {
      if (jobId === highlightedJob) {
        blockElement.classList.add("highlighted");
        blockElement.classList.remove("dimmed");
      } else {
        blockElement.classList.add("dimmed");
        blockElement.classList.remove("highlighted");
      }
    }
  };

  const bindBlockHighlightClick = (blockElement, jobId) => {
    blockElement.addEventListener("click", (e) => {
      e.stopPropagation();
      if (highlightedJob === jobId) {
        highlightedJob = null;
      } else {
        highlightedJob = jobId;
      }
      renderGanttChart();
    });
  };

  displayRows.forEach((row) => {
    const rowDiv = document.createElement("div");
    let rowClass = "timeline-row";
    if (row.type !== "tool") rowClass += " header-row";
    rowDiv.className = rowClass;

    if (row.type === "tool") {
      const toolTasks = tasksByTool[row.tool_id] || [];

      toolTasks.forEach((t) => {
        // AMHS Transport Block (Only in full detail mode)
        if (detailLevel === "full" && t.transport_end > t.transport_start) {
          const transBlock = document.createElement("div");
          transBlock.className = "gantt-block transport";
          const left = t.transport_start * pxPerMin;
          const width = Math.max(2, (t.transport_end - t.transport_start) * pxPerMin);
          transBlock.style.left = `${left}px`;
          transBlock.style.width = `${width}px`;
          transBlock.textContent = width > 28 ? "🚚" : "";
          applyHighlightClass(transBlock, t.job_id);
          bindBlockHighlightClick(transBlock, t.job_id);
          attachTooltip(transBlock, t, "transport");
          rowDiv.appendChild(transBlock);
        }

        // Setup SDST Block (Only in full detail mode)
        if (detailLevel === "full" && t.setup_end > t.setup_start) {
          const setupBlock = document.createElement("div");
          setupBlock.className = "gantt-block setup";
          const left = t.setup_start * pxPerMin;
          const width = Math.max(2, (t.setup_end - t.setup_start) * pxPerMin);
          setupBlock.style.left = `${left}px`;
          setupBlock.style.width = `${width}px`;
          setupBlock.textContent = width > 28 ? "⚙️" : "";
          applyHighlightClass(setupBlock, t.job_id);
          bindBlockHighlightClick(setupBlock, t.job_id);
          attachTooltip(setupBlock, t, "setup");
          rowDiv.appendChild(setupBlock);
        }

        // Processing Task Block
        const procBlock = document.createElement("div");
        const style = getProductStyle(t.product_type);

        let procClass = "gantt-block proc-block";
        if (t.priority === "Super_Hot") procClass += " is-super-hot";
        else if (t.priority === "Hot") procClass += " is-hot";

        procBlock.className = procClass;
        procBlock.style.background = style.bg;
        procBlock.style.borderColor = style.border;

        const left = t.proc_start * pxPerMin;
        const width = Math.max(3, (t.proc_end - t.proc_start) * pxPerMin);
        procBlock.style.left = `${left}px`;
        procBlock.style.width = `${width}px`;

        // Adaptive Label Logic to Prevent Overlapping
        if (width >= 90) {
          procBlock.textContent = `${t.job_id} | S${t.step_id}`;
        } else if (width >= 55) {
          procBlock.textContent = `${t.job_id}`;
        } else if (width >= 32) {
          procBlock.textContent = `S${t.step_id}`;
        } else {
          procBlock.textContent = "";
        }

        applyHighlightClass(procBlock, t.job_id);
        bindBlockHighlightClick(procBlock, t.job_id);
        attachTooltip(procBlock, t, "proc");
        rowDiv.appendChild(procBlock);
      });
    }

    timelineBody.appendChild(rowDiv);
  });

}

function toggleCollapse(id) {
  if (collapsedIds.has(id)) {
    collapsedIds.delete(id);
  } else {
    collapsedIds.add(id);
  }
  renderGanttChart();
}

// ----------------------------------------------------------------------
// Tooltip Handler
// ----------------------------------------------------------------------
function attachTooltip(element, task, type) {
  element.addEventListener("mouseenter", (e) => {
    let title = `${task.job_id} (Step ${task.step_id})`;
    let duration = (task.proc_end - task.proc_start).toFixed(1);
    let typeLabel = "Processing Operation";

    if (type === "transport") {
      title = `${task.job_id} - AMHS Transport`;
      duration = (task.transport_end - task.transport_start).toFixed(1);
      typeLabel = "AMHS Lot Transport";
    } else if (type === "setup") {
      title = `${task.job_id} - SDST Tool Setup`;
      duration = (task.setup_end - task.setup_start).toFixed(1);
      typeLabel = "Sequence-Dependent Setup";
    }

    const style = getProductStyle(task.product_type);

    tooltip.innerHTML = `
      <h4>
        <span>${title}</span>
        <span class="t-badge" style="background:${style.color}; color:#fff;">${task.product_type}</span>
      </h4>
      <div class="tooltip-row"><span>Type:</span> <span>${typeLabel}</span></div>
      <div class="tooltip-row"><span>Priority:</span> <span>${task.priority} (weight=${task.priority_weight})</span></div>
      <div class="tooltip-row"><span>Workstation:</span> <span>${task.ws_id} (${task.wsg_id})</span></div>
      <div class="tooltip-row"><span>Tool ID:</span> <span>${task.tool_id}</span></div>
      <div class="tooltip-row"><span>Route:</span> <span>${task.from_location} ➔ ${task.to_location}</span></div>
      <div class="tooltip-row"><span>Start Time:</span> <span>${type === "transport" ? task.transport_start : type === "setup" ? task.setup_start : task.proc_start} min</span></div>
      <div class="tooltip-row"><span>End Time:</span> <span>${type === "transport" ? task.transport_end : type === "setup" ? task.setup_end : task.proc_end} min</span></div>
      <div class="tooltip-row"><span>Duration:</span> <span>${duration} mins</span></div>
      <div class="tooltip-row"><span>Due Date:</span> <span>${task.due_date} min</span></div>
      <div class="tooltip-row"><span>Tardiness:</span> <span style="color:${task.tardiness > 0 ? "#ef4444" : "#10b981"}; font-weight:700;">${task.tardiness} min</span></div>
    `;

    tooltip.classList.remove("hidden");
    positionTooltip(e);
  });

  element.addEventListener("mousemove", positionTooltip);

  element.addEventListener("mouseleave", () => {
    tooltip.classList.add("hidden");
  });
}

function positionTooltip(e) {
  const x = e.clientX + 15;
  const y = e.clientY + 15;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

// ----------------------------------------------------------------------
// TAB 2: Render Progress Section SVG Charts (Dynamic Multi-Strategy)
// ----------------------------------------------------------------------
const STRATEGY_COLORS = {
  ga: "#8b5cf6",
  ppo_continuous_tardiness: "#10b981",
  ppo_pbrs: "#06b6d4",
  ppo_baseline: "#f59e0b",
  ppo_milestone_progress: "#ec4899",
  ppo_workload_balance: "#3b82f6",
  ppo: "#10b981",
};

async function renderProgressSection(currentHistory) {
  const seriesFitness = [];
  const seriesMakespan = [];
  const seriesReward = [];
  let globalBestFit = null;
  let allHistories = [];

  try {
    const manifestRes = await fetch("/experiments/manifest.json");
    if (manifestRes.ok) {
      const manifestData = await manifestRes.json();
      const experiments = manifestData.experiments || [];

      for (const item of experiments) {
        try {
          const res = await fetch(item.rel_path);
          if (!res.ok) continue;
          const data = await res.json();
          const hist = data.history || [];

          if (hist.length > 0) {
            const algKey = item.algorithm.toLowerCase();
            const color = STRATEGY_COLORS[algKey] || getProductStyle(item.algorithm).color;
            const label = item.title || item.algorithm.toUpperCase();

            const hasBestFitness = hist.some(d => d.best_fitness !== undefined && d.best_fitness !== null);

            if (hasBestFitness) {
              seriesFitness.push({ label: label + " - Best", data: hist, key: "best_fitness", color, style: "solid" });
              seriesFitness.push({ label: label + " - Episode", data: hist, key: "fitness", color, style: "dashed" });
              seriesReward.push({ label: label + " - Reward", data: hist, key: "total_reward", color, style: "solid" });
            } else {
              seriesFitness.push({ label, data: hist, key: "fitness", color, style: "solid" });
            }

            seriesMakespan.push({ label, data: hist, key: "makespan", color, style: "solid" });

            const lastFit = hasBestFitness ? hist[hist.length - 1].best_fitness : hist[hist.length - 1].fitness;
            if (globalBestFit === null || lastFit < globalBestFit) {
              globalBestFit = lastFit;
            }
            allHistories.push(`${item.algorithm.toUpperCase()}: ${hist.length}`);
          }
        } catch (err) {
          console.warn("Could not load history for", item.rel_path, err);
        }
      }
    }
  } catch (e) {
    console.warn("Failed to load experiments manifest for progress section", e);
  }

  // Fallback to passed currentHistory if manifest had no history items
  if (seriesFitness.length === 0 && currentHistory && currentHistory.length > 0) {
    const hasBestFitness = currentHistory.some(d => d.best_fitness !== undefined && d.best_fitness !== null);
    if (hasBestFitness) {
      seriesFitness.push({ label: "Current Run - Best", data: currentHistory, key: "best_fitness", color: "#10b981", style: "solid" });
      seriesFitness.push({ label: "Current Run - Episode", data: currentHistory, key: "fitness", color: "#10b981", style: "dashed" });
      seriesReward.push({ label: "Current Run - Reward", data: currentHistory, key: "total_reward", color: "#10b981", style: "solid" });
      globalBestFit = currentHistory[currentHistory.length - 1].best_fitness;
    } else {
      seriesFitness.push({ label: "Current Run Progress", data: currentHistory, key: "fitness", color: "#10b981", style: "solid" });
      globalBestFit = currentHistory[currentHistory.length - 1].fitness;
    }
    seriesMakespan.push({ label: "Current Run Makespan", data: currentHistory, key: "makespan", color: "#3b82f6", style: "solid" });
  }

  const initFit = seriesFitness.length > 0 && seriesFitness[0].data.length > 0 ? 
    (seriesFitness[0].key === "best_fitness" ? seriesFitness[0].data[0].best_fitness : seriesFitness[0].data[0].fitness) : "--";

  document.getElementById("prog-init-fitness").textContent = typeof initFit === "number" ? initFit.toLocaleString() : "--";
  document.getElementById("prog-best-fitness").textContent = typeof globalBestFit === "number" ? globalBestFit.toLocaleString() : "--";
  document.getElementById("prog-improvement").textContent = seriesFitness.length ? `${seriesFitness.length} Strategies` : "--";
  document.getElementById("prog-generations").textContent = allHistories.length ? allHistories.join(" | ") : "No History Data";

  renderMultiLineChart("chart-fitness", seriesFitness, "Fitness Convergence");
  renderMultiLineChart("chart-makespan-tardiness", seriesMakespan, "Makespan Progression (mins)");
  renderMultiLineChart("chart-reward", seriesReward, "Episode Reward");
}

function renderMultiLineChart(containerId, seriesList, labelName) {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (!seriesList || seriesList.length === 0) {
    container.innerHTML = `<div style="padding:40px; text-align:center; color:var(--text-muted);">No optimization progress data available.</div>`;
    return;
  }

  const width = container.clientWidth || 500;
  const height = 260;
  const padding = 40;

  let allValues = [];
  seriesList.forEach((s) => {
    allValues = allValues.concat(s.data.map((d) => d[s.key]).filter(v => v !== null && v !== undefined && !isNaN(v)));
  });

  if (allValues.length === 0) {
    container.innerHTML = `<div style="padding:40px; text-align:center; color:var(--text-muted);">No valid data points available.</div>`;
    return;
  }

  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);
  const range = maxVal - minVal || 1;

  let legendHTML = `<g transform="translate(${width - 220}, 15)">`;
  seriesList.forEach((s, idx) => {
    const dashAttr = s.style === "dashed" ? `stroke-dasharray="4,2"` : "";
    legendHTML += `
      <line x1="0" y1="${idx * 16 + 6}" x2="12" y2="${idx * 16 + 6}" stroke="${s.color}" stroke-width="2.5" ${dashAttr} />
      <text x="18" y="${idx * 16 + 10}" font-size="11" fill="var(--text-main)" font-weight="500">${s.label}</text>
    `;
  });
  legendHTML += `</g>`;

  let pathsHTML = "";
  seriesList.forEach((s) => {
    const validPoints = s.data
      .map((d, i) => ({ val: d[s.key], index: i }))
      .filter(p => p.val !== null && p.val !== undefined && !isNaN(p.val));

    if (validPoints.length === 0) return;

    const points = validPoints
      .map((p) => {
        const x = padding + (p.index / (s.data.length - 1 || 1)) * (width - 2 * padding);
        const y = height - padding - ((p.val - minVal) / range) * (height - 2 * padding);
        return `${x},${y}`;
      })
      .join(" ");

    const dashAttr = s.style === "dashed" ? `stroke-dasharray="6,4"` : "";
    pathsHTML += `<polyline fill="none" stroke="${s.color}" stroke-width="2.5" points="${points}" stroke-linejoin="round" ${dashAttr} />`;

    const stepInterval = Math.max(1, Math.floor(s.data.length / 15));
    validPoints.forEach((p) => {
      if (p.index % stepInterval === 0 || p.index === s.data.length - 1) {
        const x = padding + (p.index / (s.data.length - 1 || 1)) * (width - 2 * padding);
        const y = height - padding - ((p.val - minVal) / range) * (height - 2 * padding);
        pathsHTML += `<circle cx="${x}" cy="${y}" r="3" fill="${s.color}" stroke="#fff" stroke-width="1" />`;
      }
    });
  });

  const svgHTML = `
    <svg viewBox="0 0 ${width} ${height}">
      <!-- Axes -->
      <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="var(--border-glass-bright)" stroke-width="1.5" />
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="var(--border-glass-bright)" stroke-width="1.5" />

      <!-- Grid & Y Labels -->
      <text x="${padding - 8}" y="${padding + 5}" font-size="10" fill="var(--text-muted)" text-anchor="end">${maxVal.toFixed(0)}</text>
      <text x="${padding - 8}" y="${height - padding}" font-size="10" fill="var(--text-muted)" text-anchor="end">${minVal.toFixed(0)}</text>

      <!-- Legend -->
      ${legendHTML}

      <!-- Polylines -->
      ${pathsHTML}
    </svg>
  `;

  container.innerHTML = svgHTML;
}

// ----------------------------------------------------------------------
// TAB 3: Render Benchmark Section Table & Bar Cards
// ----------------------------------------------------------------------
// TAB 3: Render Benchmark Section Table & Bar Cards (Dynamic Multi-Algorithm)
// ----------------------------------------------------------------------
async function renderBenchmarkSection(heuristicComparisons) {
  const tbody = document.getElementById("benchmark-table-body");
  const cardsGrid = document.getElementById("comparison-cards");

  let mergedMap = { ...(heuristicComparisons || {}) };

  try {
    const manifestRes = await fetch("/experiments/manifest.json");
    if (manifestRes.ok) {
      const manifestData = await manifestRes.json();
      const experiments = manifestData.experiments || [];

      for (const item of experiments) {
        const key = item.algorithm;
        mergedMap[key] = {
          name: item.title || item.algorithm.toUpperCase(),
          fitness: Number(item.fitness || 0),
          makespan: Number(item.makespan || 0),
          total_weighted_tardiness: Number(item.tardiness || 0),
          total_setup_time: Number(item.setup_time || 0),
          on_time_rate_percent: item.on_time_rate_percent !== undefined ? item.on_time_rate_percent : 0,
          avg_tool_utilization_percent: item.avg_tool_utilization_percent !== undefined ? item.avg_tool_utilization_percent : 0,
        };
      }
    }
  } catch (e) {
    console.warn("Could not load experiments manifest for benchmark tab", e);
  }

  if (Object.keys(mergedMap).length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;">No benchmark data available. Run run_reward_comparison.py.</td></tr>`;
    return;
  }

  const entries = Object.entries(mergedMap).map(([key, data]) => ({ key, ...data }));
  entries.sort((a, b) => a.fitness - b.fitness);

  const minFitness = Math.min(...entries.map((e) => e.fitness));
  const minMakespan = Math.min(...entries.map((e) => e.makespan));

  // Render Table Rows
  tbody.innerHTML = entries
    .map((e, idx) => {
      const isBest = e.fitness === minFitness;
      let rankBadge = `<span class="rank-badge standard">Rank ${idx + 1}</span>`;
      if (idx === 0) rankBadge = `<span class="rank-badge best">🥇 Rank 1 (Best)</span>`;
      else if (idx === 1) rankBadge = `<span class="rank-badge runner-up">🥈 Rank 2</span>`;
      else if (idx === 2) rankBadge = `<span class="rank-badge third">🥉 Rank 3</span>`;

      return `
      <tr>
        <td><strong>${e.name}</strong></td>
        <td><strong style="color:${isBest ? "#10b981" : "inherit"};">${e.fitness.toLocaleString()}</strong></td>
        <td>${e.makespan.toLocaleString()}m</td>
        <td>${e.total_weighted_tardiness.toLocaleString()}m</td>
        <td>${e.total_setup_time.toLocaleString()}m</td>
        <td>${e.on_time_rate_percent !== undefined ? e.on_time_rate_percent : 0}%</td>
        <td>${e.avg_tool_utilization_percent !== undefined ? e.avg_tool_utilization_percent : 0}%</td>
        <td>${rankBadge}</td>
      </tr>
    `;
    })
    .join("");

  // Render Bar Charts Cards
  const maxFitness = Math.max(...entries.map((e) => e.fitness));
  const maxMakespan = Math.max(...entries.map((e) => e.makespan));

  cardsGrid.innerHTML = `
    <div class="metric-bar-card">
      <h4>🏆 Objective Fitness Score (Lower is Better)</h4>
      ${entries
        .map((e) => {
          const pct = Math.max(15, (e.fitness / maxFitness) * 100);
          const isBest = e.fitness === minFitness;
          return `
          <div class="bar-item">
            <div class="bar-info"><span>${e.name}</span><span>${e.fitness.toLocaleString()}</span></div>
            <div class="bar-track">
              <div class="bar-fill ${isBest ? "best-fill" : ""}" style="width: ${pct}%;"></div>
            </div>
          </div>
        `;
        })
        .join("")}
    </div>

    <div class="metric-bar-card">
      <h4>⏱️ Makespan C<sub>max</sub> (Mins - Lower is Better)</h4>
      ${entries
        .map((e) => {
          const pct = Math.max(15, (e.makespan / maxMakespan) * 100);
          const isBest = e.makespan === minMakespan;
          return `
          <div class="bar-item">
            <div class="bar-info"><span>${e.name}</span><span>${e.makespan.toLocaleString()}m</span></div>
            <div class="bar-track">
              <div class="bar-fill ${isBest ? "best-fill" : ""}" style="width: ${pct}%;"></div>
            </div>
          </div>
        `;
        })
        .join("")}
    </div>
  `;
}

// ----------------------------------------------------------------------
// Event Listeners
// ----------------------------------------------------------------------
filterArea.addEventListener("change", renderGanttChart);

if (filterJob) {
  filterJob.addEventListener("change", renderGanttChart);
}

filterProduct.addEventListener("change", (e) => {
  selectedProduct = e.target.value;
  updateProductChipStates();
  renderGanttChart();
});


filterPriority.addEventListener("change", renderGanttChart);
selectGroupMode.addEventListener("change", renderGanttChart);
selectDetailLevel.addEventListener("change", renderGanttChart);
searchJob.addEventListener("input", renderGanttChart);

if (btnResetHighlight) {
  btnResetHighlight.addEventListener("click", () => {
    highlightedJob = null;
    renderGanttChart();
  });
}


btnResetProductFilter.addEventListener("click", () => {
  selectedProduct = "ALL";
  filterProduct.value = "ALL";
  updateProductChipStates();
  renderGanttChart();
});

zoomSlider.addEventListener("input", (e) => {
  currentZoom = parseFloat(e.target.value);
  zoomVal.textContent = `${currentZoom.toFixed(1)}x`;
  renderGanttChart();
});

btnToggleCollapse.addEventListener("click", () => {
  if (!datasetData) return;
  // If some are collapsed, expand all; otherwise collapse all
  if (collapsedIds.size > 0) {
    collapsedIds.clear();
  } else {
    datasetData.factory_hierarchy.forEach((area) => {
      collapsedIds.add(`area_${area.area_id}`);
      area.workstation_groups.forEach((wsg) => {
        collapsedIds.add(`wsg_${wsg.wsg_id}`);
        wsg.workstations.forEach((ws) => {
          collapsedIds.add(`ws_${ws.ws_id}`);
        });
      });
    });
  }
  renderGanttChart();
});

if (selectModel) {
  selectModel.addEventListener("change", () => {
    loadData(selectModel.value);
  });
}

if (selectExperiment) {
  selectExperiment.addEventListener("change", () => {
    loadData(selectExperiment.value);
  });
}

btnReload.addEventListener("click", () => {
  loadExperimentsManifest().then(() => loadData());
});

btnThemeToggle.addEventListener("click", () => {
  const currentTheme = document.documentElement.getAttribute("data-theme");
  if (currentTheme === "dark") {
    document.documentElement.removeAttribute("data-theme");
    btnThemeToggle.textContent = "🌙 Dark Mode";
  } else {
    document.documentElement.setAttribute("data-theme", "dark");
    btnThemeToggle.textContent = "☀️ Light Mode";
  }
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    try {
      datasetData = JSON.parse(event.target.result);
      initDashboard();
    } catch (err) {
      alert("Invalid JSON dataset file!");
    }
  };
  reader.readAsText(file);
});

// Initialize on page load
loadExperimentsManifest().then(() => loadData());

