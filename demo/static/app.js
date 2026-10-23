/**
 * TESLA MACRO FSD // ASSUMPTION MONITOR OS v12 CONTROLLER (demo/static/app.js)
 */
(function () {
  "use strict";

  const TOPICS = [
    "fuel",
    "currency",
    "demand",
    "labor",
    "rates",
    "supplier",
    "regulation",
    "other",
  ];

  const state = {
    activeTab: "command",
    docSource: "plan1",
    asOf: "2026-10-16",
    mode: "hybrid",
    threshold: 0.45,
    overrideOil: null,
    overrideFx: null,
    customText: null,
    uploadedB64: null,
    uploadedFilename: null,

    matrixFilter: "all",
    matrixTopic: "all",
    matrixSearch: "",
    highlightedSentenceId: null,

    horizonYears: 3,
    seriesData: null,
    modelLabData: null,
    analysis: null,
    activeDrawerSentence: null,
  };

  let toastTimer = null;
  function showToast(message) {
    const toast = document.getElementById("telemetry-toast");
    const msgEl = document.getElementById("toast-message");
    if (!toast || !msgEl) return;
    msgEl.textContent = message;
    toast.classList.remove("hidden");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.add("hidden");
    }, 3400);
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // =========================================================================
  // THREE.JS 3D TESLA FULL SELF-DRIVING (FSD) TELEMETRY STAGE
  // =========================================================================
  const threeCtrl = {
    scene: null,
    camera: null,
    renderer: null,
    egoCore: null,
    shieldMesh: null,
    laneGroup: null,
    nodeGroup: null,
    gridHelper: null,
    satelliteMeshes: [],
    raycaster: null,
    mouse: null,
    isPaused: false,
    wireframeOnly: false,
    camMode: "fsd", // 'fsd' | 'orbit' | 'radar'
    userYaw: 0,
    userPitch: 0.34,
    zoomDist: 10.5,
    isDragging: false,
    prevX: 0,
    prevY: 0,
    clockStart: performance.now(),
  };

  function initThreeScene() {
    const container = document.getElementById("three-container");
    const canvas = document.getElementById("three-canvas");
    if (!container || !canvas || typeof window.THREE === "undefined") return;

    const THREE = window.THREE;
    try {
      const renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: true,
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(container.clientWidth || 540, container.clientHeight || 420);
      threeCtrl.renderer = renderer;
    } catch (e) {
      return;
    }

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x060709, 0.035);
    threeCtrl.scene = scene;

    const camera = new THREE.PerspectiveCamera(
      48,
      (container.clientWidth || 540) / (container.clientHeight || 420),
      0.1,
      120
    );
    threeCtrl.camera = camera;

    scene.add(new THREE.AmbientLight(0x94a3b8, 0.85));
    const pLight1 = new THREE.PointLight(0x3e6ae1, 2.2, 40);
    pLight1.position.set(0, 8, 6);
    scene.add(pLight1);

    const pLight2 = new THREE.PointLight(0x00e5ff, 1.6, 35);
    pLight2.position.set(-5, 4, -8);
    scene.add(pLight2);

    // FSD Ground Grid
    const gridHelper = new THREE.GridHelper(44, 44, 0x3e6ae1, 0x161f33);
    gridHelper.position.y = -1.6;
    scene.add(gridHelper);
    threeCtrl.gridHelper = gridHelper;

    // FSD Glowing Autopilot Corridor Lanes
    const laneGroup = new THREE.Group();
    const laneOffsets = [-3.2, -1.1, 1.1, 3.2];
    laneOffsets.forEach((xPos, idx) => {
      const isCenter = idx === 1 || idx === 2;
      const laneGeo = new THREE.PlaneGeometry(isCenter ? 0.09 : 0.05, 38);
      laneGeo.rotateX(-Math.PI / 2);
      const laneMat = new THREE.MeshBasicMaterial({
        color: isCenter ? 0x3e6ae1 : 0x283552,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: isCenter ? 0.85 : 0.45,
      });
      const laneMesh = new THREE.Mesh(laneGeo, laneMat);
      laneMesh.position.set(xPos, -1.58, -6);
      laneGroup.add(laneMesh);
    });
    scene.add(laneGroup);
    threeCtrl.laneGroup = laneGroup;

    // Foreground Cyber-Ego Telemetry Polyhedron
    const egoGeo = new THREE.OctahedronGeometry(0.95, 1);
    const egoMat = new THREE.MeshStandardMaterial({
      color: 0x3e6ae1,
      emissive: 0x0d2154,
      roughness: 0.2,
      metalness: 0.8,
      flatShading: true,
    });
    const egoCore = new THREE.Mesh(egoGeo, egoMat);
    egoCore.position.set(0, -0.35, 3.2);
    scene.add(egoCore);
    threeCtrl.egoCore = egoCore;

    // Wireframe Halo around Ego Core
    const shieldGeo = new THREE.IcosahedronGeometry(1.3, 1);
    const shieldMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      wireframe: true,
      transparent: true,
      opacity: 0.3,
    });
    const shieldMesh = new THREE.Mesh(shieldGeo, shieldMat);
    shieldMesh.position.copy(egoCore.position);
    scene.add(shieldMesh);
    threeCtrl.shieldMesh = shieldMesh;

    // Group for 3D Assumption Nodes
    const nodeGroup = new THREE.Group();
    scene.add(nodeGroup);
    threeCtrl.nodeGroup = nodeGroup;

    threeCtrl.raycaster = new THREE.Raycaster();
    threeCtrl.mouse = new THREE.Vector2();

    // Mouse Drag & Click Raycasting
    canvas.addEventListener("mousedown", (e) => {
      threeCtrl.isDragging = true;
      threeCtrl.prevX = e.clientX;
      threeCtrl.prevY = e.clientY;
    });
    window.addEventListener("mouseup", () => {
      threeCtrl.isDragging = false;
    });
    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      if (threeCtrl.isDragging) {
        const dx = e.clientX - threeCtrl.prevX;
        const dy = e.clientY - threeCtrl.prevY;
        threeCtrl.userYaw += dx * 0.007;
        threeCtrl.userPitch = Math.max(0.05, Math.min(1.45, threeCtrl.userPitch + dy * 0.007));
        threeCtrl.prevX = e.clientX;
        threeCtrl.prevY = e.clientY;
        return;
      }
      threeCtrl.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      threeCtrl.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      checkNodeHit(false);
    });
    canvas.addEventListener("click", (e) => {
      const rect = canvas.getBoundingClientRect();
      threeCtrl.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      threeCtrl.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      checkNodeHit(true);
    });
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        threeCtrl.zoomDist = Math.max(5.5, Math.min(18, threeCtrl.zoomDist + e.deltaY * 0.006));
      },
      { passive: false }
    );
    window.addEventListener("resize", () => {
      if (!threeCtrl.renderer || !threeCtrl.camera) return;
      const w = container.clientWidth || 540;
      const h = container.clientHeight || 420;
      threeCtrl.camera.aspect = w / h;
      threeCtrl.camera.updateProjectionMatrix();
      threeCtrl.renderer.setSize(w, h);
    });

    requestAnimationFrame(animateThreeScene);
  }

  function updateThreeNodes(analysis) {
    if (!analysis) return;
    const sentences = analysis.sentences || [];
    const assumptions = sentences.filter((s) => s.is_assumption);

    let cBreach = 0;
    let cWatch = 0;
    let cOk = 0;
    let cInfo = 0;

    assumptions.forEach((s) => {
      const st = s.claim ? s.claim.status : "unmapped";
      if (st === "breached") cBreach++;
      else if (st === "warning") cWatch++;
      else if (st === "holding") cOk++;
      else cInfo++;
    });

    document.getElementById("3d-count-breach").textContent = String(cBreach);
    document.getElementById("3d-count-watch").textContent = String(cWatch);
    document.getElementById("3d-count-ok").textContent = String(cOk);
    document.getElementById("3d-count-info").textContent = String(cInfo);

    if (!threeCtrl.scene || !threeCtrl.nodeGroup || typeof window.THREE === "undefined") return;
    const THREE = window.THREE;

    while (threeCtrl.nodeGroup.children.length > 0) {
      threeCtrl.nodeGroup.remove(threeCtrl.nodeGroup.children[0]);
    }
    threeCtrl.satelliteMeshes = [];

    if (threeCtrl.egoCore) {
      if (cBreach > 0) {
        threeCtrl.egoCore.material.color.setHex(0xe82127);
        threeCtrl.egoCore.material.emissive.setHex(0x4c070a);
      } else if (cWatch > 0) {
        threeCtrl.egoCore.material.color.setHex(0xf59e0b);
        threeCtrl.egoCore.material.emissive.setHex(0x452804);
      } else {
        threeCtrl.egoCore.material.color.setHex(0x3e6ae1);
        threeCtrl.egoCore.material.emissive.setHex(0x0d2154);
      }
    }

    const lanesX = [-2.15, 0, 2.15];
    assumptions.forEach((sent, idx) => {
      const st = sent.claim ? sent.claim.status : "unmapped";
      let hex = 0x3e6ae1;
      if (st === "breached") hex = 0xe82127;
      else if (st === "warning") hex = 0xf59e0b;
      else if (st === "holding") hex = 0x12b981;

      const geo = new THREE.BoxGeometry(0.62, 0.42, 0.82);
      const mat = new THREE.MeshStandardMaterial({
        color: hex,
        emissive: hex,
        emissiveIntensity: 0.45,
        roughness: 0.25,
        metalness: 0.65,
        wireframe: threeCtrl.wireframeOnly,
      });
      const mesh = new THREE.Mesh(geo, mat);

      const laneX = lanesX[idx % lanesX.length];
      const zPos = 0.5 - Math.floor(idx / 3) * 2.4 - (idx % 3) * 0.55;
      mesh.position.set(laneX, -1.1, zPos);

      mesh.userData = {
        sentenceObj: sent,
        baseX: laneX,
        baseY: -1.1,
        baseZ: zPos,
        orbitAngle: (idx / Math.max(assumptions.length, 1)) * Math.PI * 2,
        orbitRadius: idx % 2 === 0 ? 2.8 : 3.9,
      };

      threeCtrl.nodeGroup.add(mesh);
      threeCtrl.satelliteMeshes.push(mesh);
    });
  }

  function checkNodeHit(openOnHit) {
    if (!threeCtrl.raycaster || !threeCtrl.camera || threeCtrl.satelliteMeshes.length === 0) return;
    threeCtrl.raycaster.setFromCamera(threeCtrl.mouse, threeCtrl.camera);
    const hits = threeCtrl.raycaster.intersectObjects(threeCtrl.satelliteMeshes);
    const readoutEl = document.getElementById("three-node-text");
    if (hits.length > 0) {
      const sent = hits[0].object.userData.sentenceObj;
      if (sent && readoutEl) {
        const st = sent.claim ? sent.claim.status.toUpperCase() : "UNMAPPED";
        readoutEl.innerHTML = `<strong>[${escapeHtml(sent.sentence_id)} · ${escapeHtml(st)}]</strong> (${escapeHtml(sent.topic || "other")}) — ${escapeHtml(sent.sentence)}`;
      }
      if (openOnHit && sent) {
        openInspectorDrawer(sent);
      }
    }
  }

  function animateThreeScene(now) {
    requestAnimationFrame(animateThreeScene);
    if (!threeCtrl.renderer || !threeCtrl.scene || !threeCtrl.camera) return;

    const elapsed = (now - threeCtrl.clockStart) * 0.001;

    if (!threeCtrl.isPaused) {
      if (threeCtrl.egoCore) {
        threeCtrl.egoCore.rotation.y = elapsed * 0.55;
      }
      if (threeCtrl.shieldMesh) {
        threeCtrl.shieldMesh.rotation.y = -elapsed * 0.35;
        threeCtrl.shieldMesh.rotation.z = elapsed * 0.2;
      }
      if (threeCtrl.gridHelper) {
        threeCtrl.gridHelper.position.z = (elapsed * 1.8) % 1.0;
      }

      threeCtrl.satelliteMeshes.forEach((m, idx) => {
        const u = m.userData;
        if (threeCtrl.camMode === "orbit") {
          const a = u.orbitAngle + elapsed * 0.35;
          m.position.set(
            Math.cos(a) * u.orbitRadius,
            Math.sin(a * 2 + idx) * 0.35,
            Math.sin(a) * u.orbitRadius
          );
          m.rotation.y = -a;
        } else {
          m.position.set(u.baseX, u.baseY + Math.sin(elapsed * 2 + idx) * 0.08, u.baseZ);
          m.rotation.y = 0;
        }
      });
    }

    if (threeCtrl.camMode === "fsd") {
      const yaw = threeCtrl.userYaw * 0.5;
      threeCtrl.camera.position.set(Math.sin(yaw) * 3.5, 2.5 + threeCtrl.userPitch * 1.5, 7.6);
      threeCtrl.camera.lookAt(0, -0.6, -3.5);
    } else if (threeCtrl.camMode === "radar") {
      threeCtrl.camera.position.set(0, 12.5, 0.01);
      threeCtrl.camera.lookAt(0, 0, -2.0);
    } else {
      const yaw = threeCtrl.userYaw + (threeCtrl.isPaused ? 0 : elapsed * 0.12);
      const pitch = threeCtrl.userPitch;
      const d = threeCtrl.zoomDist;
      threeCtrl.camera.position.set(
        Math.sin(yaw) * Math.cos(pitch) * d,
        Math.sin(pitch) * d,
        Math.cos(yaw) * Math.cos(pitch) * d
      );
      threeCtrl.camera.lookAt(0, 0, 0);
    }

    threeCtrl.renderer.render(threeCtrl.scene, threeCtrl.camera);
  }

  // =========================================================================
  // CORE API CALLS
  // =========================================================================
  async function fetchAnalysis() {
    const payload = {
      doc_source: state.docSource,
      as_of: state.asOf,
      mode: state.mode,
      threshold: state.threshold,
      override_oil: state.overrideOil,
      override_fx: state.overrideFx,
      custom_text: state.docSource === "custom" ? state.customText : null,
      uploaded_b64: state.docSource === "uploaded" ? state.uploadedB64 : null,
      uploaded_filename: state.docSource === "uploaded" ? state.uploadedFilename : null,
    };

    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) {
      showToast("Error: " + data.error);
      return;
    }
    state.analysis = data;
    renderAllViews();
  }

  async function fetchSeriesData() {
    const res = await fetch(`/api/series?years=${state.horizonYears}`);
    const data = await res.json();
    state.seriesData = data;
    renderHorizonCharts();
  }

  async function fetchModelLab(retrain = false) {
    const res = await fetch(retrain ? "/api/retrain" : "/api/model-lab", {
      method: retrain ? "POST" : "GET",
      headers: { "Content-Type": "application/json" },
      body: retrain ? JSON.stringify({}) : undefined,
    });
    const data = await res.json();
    state.modelLabData = data;
    renderModelLab();
    if (retrain) {
      const systems = data.cv_comparison || [];
      const best = systems[systems.length - 1] || systems[0];
      showToast(
        `Retrained Model A! F1=${best ? (best.f1 * 100).toFixed(1) + "%" : "—"} (${data.ui_reviews_count || 0} human reviews applied)`
      );
    }
  }

  // =========================================================================
  // RENDER ALL VIEWS FROM state.analysis
  // =========================================================================
  function renderAllViews() {
    const a = state.analysis;
    if (!a) return;

    const s = a.summary;
    const fin = a.financial_telemetry || {};

    document.getElementById("active-doc-title").textContent = a.doc_title;
    document.getElementById("nav-clause-count").textContent = String(s.total_sentences);

    const dlBtn = document.getElementById("btn-download-active-doc");
    if (dlBtn && ["plan1", "plan2", "plan3"].includes(state.docSource)) {
      dlBtn.href = `/api/download-pdf?doc=${encodeURIComponent(state.docSource)}`;
    }

    // FSD Speedometer & Road Signs
    document.getElementById("kpi-viability-score").textContent = String(fin.viability_index ?? 85);
    document.getElementById("hud-regime-kicker").textContent = fin.regime_label || "AUTOPILOT ACTIVE";
    document.getElementById("kpi-total-sentences").textContent = String(s.total_sentences);
    document.getElementById("kpi-assumptions-detected").textContent = String(s.assumptions_detected);
    document.getElementById("kpi-breached-claims").textContent = String(s.breached_claims);
    document.getElementById("kpi-holding-claims").textContent = String(s.holding_claims);
    document.getElementById("kpi-warning-claims").textContent = String(s.warning_claims);
    document.getElementById("kpi-monitored-total").textContent = String(s.monitored_claims);
    document.getElementById("kpi-reviewed-count").textContent = String(s.human_reviewed || 0);

    // Financial Margin Strip
    document.getElementById("fin-base-margin").textContent = `${fin.base_margin_pct ?? 9.2}%`;
    document.getElementById("fin-proj-margin").textContent = `${fin.projected_margin_pct ?? 5.9}%`;
    const bps = fin.margin_delta_bps ?? 0;
    document.getElementById("fin-margin-bps").textContent = `${bps >= 0 ? "+" : ""}${bps} bps`;
    document.getElementById("financial-narrative-text").textContent = fin.narrative || "";

    // Sync sliders & Bottom Dock Climate Steppers
    const oilSnap = a.indicators ? a.indicators.DCOILWTICO : null;
    const fxSnap = a.indicators ? a.indicators.DEXINUS : null;
    if (oilSnap) {
      document.getElementById("sim-oil-slider").value = oilSnap.value;
      document.getElementById("sim-oil-val").textContent = `$${oilSnap.value.toFixed(2)}`;
      document.getElementById("dock-oil-val").textContent = `$${oilSnap.value.toFixed(1)}`;
    }
    if (fxSnap) {
      document.getElementById("sim-fx-slider").value = fxSnap.value;
      document.getElementById("sim-fx-val").textContent = `₹${fxSnap.value.toFixed(2)}`;
      document.getElementById("dock-fx-val").textContent = `₹${fxSnap.value.toFixed(1)}`;
    }

    renderIndicatorCards(a.indicators || {});
    renderMonitoredClaims(a.monitored_claims || []);
    renderAuditLog(a.audit_log || [], s.audit_counts || {});
    renderClauseMatrix(a.sentences || []);
    updateThreeNodes(a);
    if (state.seriesData) renderHorizonCharts();
  }

  function renderIndicatorCards(indicators) {
    const wrap = document.getElementById("indicator-snapshot-list");
    if (!wrap) return;
    const list = Object.values(indicators);
    wrap.innerHTML = list
      .map((ind) => {
        const valFmt =
          ind.series_id === "DCOILWTICO"
            ? `$${ind.value.toFixed(2)} / bbl`
            : `₹${ind.value.toFixed(2)} / USD`;
        return `
          <div class="ind-snap-item">
            <div class="text-cyan">${escapeHtml(ind.series_id)} ${ind.is_overridden ? "(OVERRIDDEN)" : ""}</div>
            <div style="font-size:13.5px;font-weight:700;margin:2px 0;">${escapeHtml(valFmt)}</div>
            <div class="text-muted">Avail: ${escapeHtml(ind.available_at)} · ${ind.change_4w_pct >= 0 ? "+" : ""}${ind.change_4w_pct}% (4w)</div>
          </div>
        `;
      })
      .join("");
  }

  function renderMonitoredClaims(claims) {
    const grid = document.getElementById("monitored-claims-grid");
    if (!grid) return;

    if (claims.length === 0) {
      grid.innerHTML = `<div class="claim-telemetry-box">No quantitative FRED claims detected under current settings.</div>`;
      return;
    }

    grid.innerHTML = claims
      .map((c) => {
        return `
          <article class="claim-card status-${escapeHtml(c.status)}">
            <div>
              <div class="claim-card-top">
                <div class="claim-id-group">
                  <span>${escapeHtml(c.sentence_id)}</span>
                  <span>·</span>
                  <span class="text-cyan">${escapeHtml(c.series_id || "MACRO")}</span>
                  <span>·</span>
                  <span>${escapeHtml(c.mapping_kind)}</span>
                </div>
                <span class="status-badge status-badge-${escapeHtml(c.status)}">${escapeHtml(c.status)}</span>
              </div>
              <p class="claim-sentence" style="margin:6px 0;">${escapeHtml(c.sentence)}</p>
              <div class="claim-telemetry-box">
                <div><strong>Assumed:</strong> ${escapeHtml(c.assumed_label)} | <strong>Actual:</strong> ${escapeHtml(c.actual_label)}</div>
                <div style="margin-top:3px;">${escapeHtml(c.explanation)}</div>
              </div>
            </div>
            <div class="claim-card-actions">
              <button class="tesla-btn tesla-btn-subtle btn-inspect-claim" style="padding:4px 9px;font-size:10px;" data-sentence-id="${escapeHtml(c.sentence_id)}">
                🔍 X-RAY &amp; REVIEW
              </button>
              <button class="tesla-btn tesla-btn-dark btn-view-series" style="padding:4px 9px;font-size:10px;" data-series="${escapeHtml(c.series_id || "DCOILWTICO")}">
                📈 FRED CHART →
              </button>
            </div>
          </article>
        `;
      })
      .join("");

    grid.querySelectorAll(".btn-inspect-claim").forEach((btn) => {
      btn.addEventListener("click", () => {
        const sid = btn.getAttribute("data-sentence-id");
        const sent = (state.analysis?.sentences || []).find((x) => x.sentence_id === sid);
        if (sent) openInspectorDrawer(sent);
      });
    });

    grid.querySelectorAll(".btn-view-series").forEach((btn) => {
      btn.addEventListener("click", () => {
        const sid = btn.getAttribute("data-series");
        switchWorkspaceTab("horizon");
        const el = document.getElementById(`chart-card-${sid}`);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    });
  }

  function renderAuditLog(auditRecords, auditCounts) {
    document.getElementById("audit-removed-badge").textContent = `${auditRecords.length} EXCISED`;
    const pillsRow = document.getElementById("audit-metrics-row");
    const entries = Object.entries(auditCounts);
    pillsRow.innerHTML =
      entries.length === 0
        ? `<span class="audit-pill">0 boilerplate lines removed</span>`
        : entries
            .map(([reason, cnt]) => `<span class="audit-pill">${escapeHtml(reason)}: <strong>${cnt}</strong></span>`)
            .join("");

    const listEl = document.getElementById("audit-removed-list");
    if (auditRecords.length === 0) {
      listEl.innerHTML = `<div class="audit-line-item"><span>Document clean — no noisy lines excised.</span></div>`;
      return;
    }
    listEl.innerHTML = auditRecords
      .map(
        (r) => `
        <div class="audit-line-item">
          <span>P${escapeHtml(r.page)}: "${escapeHtml(r.text)}"</span>
          <span class="audit-reason-tag">${escapeHtml(r.reason)}</span>
        </div>
      `
      )
      .join("");
  }

  // =========================================================================
  // RENDER CLAUSE MATRIX & HUMAN-IN-THE-LOOP REVIEW WORKBENCH
  // =========================================================================
  function renderClauseMatrix(sentences) {
    const tbody = document.getElementById("clause-matrix-tbody");
    if (!tbody) return;

    const q = state.matrixSearch.trim().toLowerCase();
    const filtered = sentences.filter((s) => {
      if (state.matrixFilter === "assumptions" && !s.is_assumption) return false;
      if (state.matrixFilter === "needs_review" && !s.needs_review) return false;
      if (state.matrixFilter === "reviewed" && !s.is_human_reviewed) return false;
      if (state.matrixFilter === "non_assumptions" && (s.is_assumption || s.needs_review)) return false;
      if (state.matrixTopic !== "all" && (s.topic || s.guessed_topic || "other") !== state.matrixTopic) return false;
      if (q) {
        const hay = `${s.sentence_id} ${s.section} ${s.sentence} ${s.topic}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    if (filtered.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:20px;" class="text-muted">No clauses match current filter criteria.</td></tr>`;
      return;
    }

    tbody.innerHTML = filtered
      .map((s) => {
        const statusBadge = s.is_assumption
          ? `<span class="status-badge status-badge-holding">ASSUMPTION (1)</span>`
          : s.needs_review
          ? `<span class="status-badge status-badge-warning">NEEDS REVIEW</span>`
          : `<span class="tesla-badge-mono">FACTUAL (0)</span>`;

        const revBadge = s.is_human_reviewed
          ? `<span class="status-badge status-badge-unmapped">HUMAN ✓</span>`
          : "";

        const pct = Math.round((s.model_prob || 0) * 100);
        const claimStatus =
          s.claim && s.claim.monitored
            ? `<span class="status-badge status-badge-${escapeHtml(s.claim.status)}">${escapeHtml(s.claim.status)}</span>`
            : `<span class="text-muted">—</span>`;

        const activeApprove = s.is_assumption ? "active-approve" : "";
        const activeReject = !s.is_assumption && !s.needs_review ? "active-reject" : "";
        const activeFlag = s.needs_review ? "active-flag" : "";
        const currentTopic = s.topic || s.guessed_topic || "other";

        const topicOptions = TOPICS.map(
          (t) => `<option value="${t}" ${t === currentTopic ? "selected" : ""}>${t}</option>`
        ).join("");

        return `
          <tr data-sid="${escapeHtml(s.sentence_id)}" class="${state.highlightedSentenceId === s.sentence_id ? "row-highlighted" : ""}">
            <td>
              <div style="font-family:var(--font-mono);font-weight:700;color:var(--tesla-blue);">${escapeHtml(s.sentence_id)}</div>
              <div class="text-muted" style="font-size:10.5px;">${escapeHtml(s.section || "General")}</div>
            </td>
            <td class="clause-text-cell" data-open-sid="${escapeHtml(s.sentence_id)}">
              ${escapeHtml(s.sentence)}
            </td>
            <td>
              <div style="display:flex;gap:4px;flex-wrap:wrap;align-items:center;">
                ${statusBadge}
                ${revBadge}
              </div>
            </td>
            <td>
              <div class="prob-bar-wrap">
                <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${pct}%"></div></div>
                <span>${(s.model_prob || 0).toFixed(2)}</span>
              </div>
            </td>
            <td style="font-family:var(--font-mono);font-size:10.5px;">
              ${escapeHtml(s.rule_is_assumption || "0")} (${(s.rule_confidence || 0).toFixed(2)})
            </td>
            <td>
              <select class="inline-topic-select" data-topic-sid="${escapeHtml(s.sentence_id)}" aria-label="Topic">
                ${topicOptions}
              </select>
            </td>
            <td>${claimStatus}</td>
            <td>
              <div class="review-actions-cell">
                <button class="btn-rev-inline ${activeApprove}" data-rev-sid="${escapeHtml(s.sentence_id)}" data-rev-val="1">✓ ASSUMP</button>
                <button class="btn-rev-inline ${activeReject}" data-rev-sid="${escapeHtml(s.sentence_id)}" data-rev-val="0">✕ REJECT</button>
                <button class="btn-rev-inline ${activeFlag}" data-rev-sid="${escapeHtml(s.sentence_id)}" data-rev-val="needs_review">? FLAG</button>
                <button class="btn-rev-inline" data-open-sid="${escapeHtml(s.sentence_id)}">X-RAY</button>
              </div>
            </td>
          </tr>
        `;
      })
      .join("");

    tbody.querySelectorAll("[data-open-sid]").forEach((el) => {
      el.addEventListener("click", () => {
        const sid = el.getAttribute("data-open-sid");
        const sent = (state.analysis?.sentences || []).find((x) => x.sentence_id === sid);
        if (sent) openInspectorDrawer(sent);
      });
    });

    tbody.querySelectorAll("[data-rev-sid]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const sid = btn.getAttribute("data-rev-sid");
        const decision = btn.getAttribute("data-rev-val");
        const sent = (state.analysis?.sentences || []).find((x) => x.sentence_id === sid);
        if (!sent) return;
        const topicSel = tbody.querySelector(`[data-topic-sid="${sid}"]`);
        const chosenTopic = topicSel ? topicSel.value : sent.topic || sent.guessed_topic || "other";
        await submitSentenceReview(sent, decision, chosenTopic);
      });
    });

    tbody.querySelectorAll("[data-topic-sid]").forEach((sel) => {
      sel.addEventListener("change", async () => {
        const sid = sel.getAttribute("data-topic-sid");
        const sent = (state.analysis?.sentences || []).find((x) => x.sentence_id === sid);
        if (!sent) return;
        await submitSentenceReview(sent, "1", sel.value);
      });
    });
  }

  async function submitSentenceReview(sentObj, decision, topic) {
    const isAssumpVal = decision === "needs_review" ? "" : decision;
    const revStatus = decision === "needs_review" ? "needs_review" : "reviewed";
    const payload = {
      sentence_id: sentObj.sentence_id,
      doc_id: state.analysis?.doc_id || state.docSource,
      sentence: sentObj.sentence,
      is_assumption: isAssumpVal,
      topic: isAssumpVal === "1" ? topic || "other" : "",
      review_status: revStatus,
    };

    const res = await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) {
      showToast("Review error: " + data.error);
      return;
    }
    showToast(`Saved Human Override for ${sentObj.sentence_id} (${revStatus})`);
    await fetchAnalysis();
    if (state.activeDrawerSentence && state.activeDrawerSentence.sentence_id === sentObj.sentence_id) {
      const updated = (state.analysis?.sentences || []).find((x) => x.sentence_id === sentObj.sentence_id);
      if (updated) openInspectorDrawer(updated);
    }
  }

  // =========================================================================
  // SLIDE-OVER NEURAL X-RAY & REVIEW DRAWER
  // =========================================================================
  function openInspectorDrawer(sent) {
    state.activeDrawerSentence = sent;
    document.getElementById("inspector-drawer").classList.remove("hidden");
    document.getElementById("drawer-sentence-id").textContent = `${sent.sentence_id} · ${sent.section || "Clause"}`;
    document.getElementById("drawer-sentence-text").textContent = sent.sentence;
    document.getElementById("drawer-review-status").textContent = sent.is_human_reviewed
      ? "HUMAN REVIEWED ✓"
      : sent.decision_source;

    const btnApprove = document.getElementById("drawer-btn-approve");
    const btnReject = document.getElementById("drawer-btn-reject");
    const btnFlag = document.getElementById("drawer-btn-flag");
    [btnApprove, btnReject, btnFlag].forEach((b) => b.classList.remove("active"));
    if (sent.is_assumption) btnApprove.classList.add("active");
    else if (sent.needs_review) btnFlag.classList.add("active");
    else btnReject.classList.add("active");

    document.getElementById("drawer-topic-select").value = sent.topic || sent.guessed_topic || "other";

    document.getElementById("drawer-dual-grid").innerHTML = `
      <div class="ind-snap-item">
        <div class="text-cyan">MODEL A (TF-IDF + LR)</div>
        <div style="font-size:15px;font-weight:700;margin:2px 0;">P = ${(sent.model_prob || 0).toFixed(3)}</div>
        <div class="text-muted">Source: ${escapeHtml(sent.decision_source)}</div>
      </div>
      <div class="ind-snap-item">
        <div class="text-cyan">RULES V1</div>
        <div style="font-size:15px;font-weight:700;margin:2px 0;">Label: ${escapeHtml(sent.rule_is_assumption || "0")}</div>
        <div class="text-muted">${escapeHtml(sent.rule_reason || "")}</div>
      </div>
    `;

    const tokens = [
      ...(sent.model_top_positive || []),
      ...(sent.model_top_negative || []),
    ].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    const tokenBars = document.getElementById("drawer-token-bars");
    if (tokens.length === 0) {
      tokenBars.innerHTML = `<div class="text-muted">No matched TF-IDF vocabulary n-grams.</div>`;
    } else {
      const maxAbs = Math.max(...tokens.map((t) => Math.abs(t.contribution)), 0.1);
      tokenBars.innerHTML = tokens
        .slice(0, 8)
        .map((t) => {
          const w = Math.min(100, Math.round((Math.abs(t.contribution) / maxAbs) * 100));
          const isPos = t.contribution >= 0;
          return `
            <div class="weight-row">
              <span>${escapeHtml(t.ngram)}</span>
              <div class="weight-bar-bg"><div class="${isPos ? "weight-bar-fill-pos" : "weight-bar-fill-neg"}" style="width:${w}%"></div></div>
              <span class="${isPos ? "text-cyan" : "text-muted"}">${isPos ? "+" : ""}${t.contribution.toFixed(3)}</span>
            </div>
          `;
        })
        .join("");
    }

    const claimBox = document.getElementById("drawer-claim-box");
    const c = sent.claim;
    if (!c || c.status === "not_assumption") {
      claimBox.innerHTML = `<div class="text-muted">Clause is not an active quantitative assumption.</div>`;
    } else {
      claimBox.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
          <span class="series-tag">${escapeHtml(c.series_id || "UNMAPPED")}</span>
          <span class="status-badge status-badge-${escapeHtml(c.status)}">${escapeHtml(c.status)}</span>
        </div>
        <div>${escapeHtml(c.explanation)}</div>
      `;
    }
  }

  // =========================================================================
  // RENDER WORKSPACE 03 // FRED MACRO HORIZON SVG CHARTS
  // =========================================================================
  function renderHorizonCharts() {
    const sObj = state.seriesData?.series;
    if (!sObj) return;

    renderSingleSvgChart(
      "DCOILWTICO",
      (sObj.DCOILWTICO?.points || []).filter((p) => p.value != null),
      [
        { value: 90.0, label: "Margin Trigger ($90/bbl)", color: "#e82127" },
        { value: 78.0, label: "Corridor High ($78/bbl)", color: "#f59e0b" },
        { value: 68.0, label: "Corridor Low ($68/bbl)", color: "#12b981" },
      ],
      (v) => `$${v.toFixed(1)}`
    );

    renderSingleSvgChart(
      "DEXINUS",
      (sObj.DEXINUS?.points || []).filter((p) => p.value != null),
      [
        { value: 87.0, label: "Import Trigger (₹87/USD)", color: "#e82127" },
        { value: 84.0, label: "Plan Target (₹84/USD)", color: "#3e6ae1" },
      ],
      (v) => `₹${v.toFixed(2)}`
    );
  }

  function renderSingleSvgChart(seriesId, points, thresholds, fmtVal) {
    const wrap = document.getElementById(`svg-wrap-${seriesId}`);
    const statsEl = document.getElementById(`stats-${seriesId}`);
    if (!wrap || points.length === 0) return;

    const vals = points.map((p) => p.value);
    const minVal = Math.min(...vals, ...thresholds.map((t) => t.value)) * 0.96;
    const maxVal = Math.max(...vals, ...thresholds.map((t) => t.value)) * 1.04;
    const latest = points[points.length - 1];

    if (statsEl) {
      statsEl.innerHTML = `
        <span class="tesla-badge-mono">LATEST: ${fmtVal(latest.value)} (${latest.observation_date})</span>
        <span class="tesla-badge-mono">MIN: ${fmtVal(Math.min(...vals))}</span>
        <span class="tesla-badge-mono">MAX: ${fmtVal(Math.max(...vals))}</span>
      `;
    }

    const W = 860;
    const H = 215;
    const padL = 56;
    const padR = 20;
    const padT = 16;
    const padB = 24;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const xForIdx = (i) => padL + (i / Math.max(points.length - 1, 1)) * plotW;
    const yForVal = (v) => padT + plotH - ((v - minVal) / Math.max(maxVal - minVal, 1e-6)) * plotH;

    const pathCoords = points
      .map((p, i) => `${i === 0 ? "M" : "L"}${xForIdx(i).toFixed(1)},${yForVal(p.value).toFixed(1)}`)
      .join(" ");
    const areaCoords = `${pathCoords} L${xForIdx(points.length - 1).toFixed(1)},${padT + plotH} L${padL},${padT + plotH} Z`;

    let asOfIdx = points.findIndex((p) => p.available_at >= state.asOf);
    if (asOfIdx < 0) asOfIdx = points.length - 1;
    const asOfX = xForIdx(asOfIdx);

    const threshLines = thresholds
      .map((t) => {
        const ty = yForVal(t.value);
        return `
          <line x1="${padL}" y1="${ty.toFixed(1)}" x2="${W - padR}" y2="${ty.toFixed(1)}" stroke="${t.color}" stroke-width="1.3" stroke-dasharray="5,4" />
          <text x="${W - padR - 4}" y="${(ty - 4).toFixed(1)}" fill="${t.color}" font-family="JetBrains Mono" font-size="9.5" text-anchor="end">${escapeHtml(t.label)}</text>
        `;
      })
      .join("");

    wrap.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" preserveAspectRatio="none">
        <defs>
          <linearGradient id="grad-${seriesId}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#3e6ae1" stop-opacity="0.32"/>
            <stop offset="100%" stop-color="#3e6ae1" stop-opacity="0.0"/>
          </linearGradient>
        </defs>
        <text x="${padL - 8}" y="${padT + 10}" fill="#8a8d93" font-family="JetBrains Mono" font-size="9.5" text-anchor="end">${fmtVal(maxVal)}</text>
        <text x="${padL - 8}" y="${padT + plotH}" fill="#8a8d93" font-family="JetBrains Mono" font-size="9.5" text-anchor="end">${fmtVal(minVal)}</text>
        <path d="${areaCoords}" fill="url(#grad-${seriesId})" />
        <path d="${pathCoords}" fill="none" stroke="#3e6ae1" stroke-width="2" />
        ${threshLines}
        <line x1="${asOfX.toFixed(1)}" y1="${padT}" x2="${asOfX.toFixed(1)}" y2="${padT + plotH}" stroke="#00e5ff" stroke-width="1.8" />
        <circle cx="${asOfX.toFixed(1)}" cy="${yForVal(points[asOfIdx].value).toFixed(1)}" r="4" fill="#00e5ff" />
      </svg>
    `;
  }

  // =========================================================================
  // RENDER WORKSPACE 04 // NEURAL MODEL LAB & SANDBOX
  // =========================================================================
  function renderModelLab() {
    const lab = state.modelLabData;
    if (!lab) return;

    const systems = lab.cv_comparison || [];
    const primary = systems[systems.length - 1] || systems[0] || { f1: 0, precision: 0, recall: 0, n: 0, tp: 0, tn: 0, fp: 0, fn: 0 };

    document.getElementById("cv-summary-grid").innerHTML = `
      <div class="cv-stat-box">
        <div class="cv-stat-val">${(primary.f1 * 100).toFixed(1)}%</div>
        <div class="cv-stat-lbl">GROUPKFOLD F1</div>
      </div>
      <div class="cv-stat-box">
        <div class="cv-stat-val">${(primary.precision * 100).toFixed(1)}%</div>
        <div class="cv-stat-lbl">PRECISION</div>
      </div>
      <div class="cv-stat-box">
        <div class="cv-stat-val">${(primary.recall * 100).toFixed(1)}%</div>
        <div class="cv-stat-lbl">RECALL</div>
      </div>
      <div class="cv-stat-box">
        <div class="cv-stat-val">${lab.ui_reviews_count || 0}</div>
        <div class="cv-stat-lbl">HUMAN OVERRIDES</div>
      </div>
    `;

    document.getElementById("cv-folds-wrap").innerHTML = `
      <div style="display:grid;grid-template-columns:1fr;gap:6px;">
        ${systems
          .map(
            (sys) => `
          <div class="ind-snap-item" style="display:flex;justify-content:space-between;align-items:center;">
            <span><strong>${escapeHtml(sys.system)}</strong> (N=${sys.n})</span>
            <span>F1: <strong class="text-cyan">${(sys.f1 * 100).toFixed(1)}%</strong> · P: ${(sys.precision * 100).toFixed(1)}% · R: ${(sys.recall * 100).toFixed(1)}% (TP:${sys.tp} FP:${sys.fp} FN:${sys.fn} TN:${sys.tn})</span>
          </div>
        `
          )
          .join("")}
      </div>
    `;

    const posWords = lab.top_words?.positive_for_assumption || [];
    const negWords = lab.top_words?.negative_for_assumption || [];
    const maxPos = Math.max(...posWords.map((w) => Math.abs(w[1])), 0.1);
    const maxNeg = Math.max(...negWords.map((w) => Math.abs(w[1])), 0.1);

    document.getElementById("top-pos-weights").innerHTML = posWords
      .slice(0, 10)
      .map(([word, coef]) => {
        const pct = Math.round((Math.abs(coef) / maxPos) * 100);
        return `
          <div class="weight-row">
            <span>${escapeHtml(word)}</span>
            <div class="weight-bar-bg"><div class="weight-bar-fill-pos" style="width:${pct}%"></div></div>
            <span class="text-cyan">+${Number(coef).toFixed(2)}</span>
          </div>
        `;
      })
      .join("");

    document.getElementById("top-neg-weights").innerHTML = negWords
      .slice(0, 10)
      .map(([word, coef]) => {
        const pct = Math.round((Math.abs(coef) / maxNeg) * 100);
        return `
          <div class="weight-row">
            <span>${escapeHtml(word)}</span>
            <div class="weight-bar-bg"><div class="weight-bar-fill-neg" style="width:${pct}%"></div></div>
            <span class="text-muted">${Number(coef).toFixed(2)}</span>
          </div>
        `;
      })
      .join("");
  }

  // =========================================================================
  // LIVE SENTENCE PREDICTION & CLAUSE INJECTION
  // =========================================================================
  async function runQuickPredictOnly(sentence) {
    if (!sentence.trim()) {
      showToast("Enter an assumption sentence first.");
      return;
    }
    const res = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence: sentence, as_of: state.asOf }),
    });
    const data = await res.json();
    if (data.error) {
      showToast(data.error);
      return;
    }

    const preview = document.getElementById("quick-prediction-preview");
    const badge = document.getElementById("quick-pred-badge");
    const meta = document.getElementById("quick-pred-meta");
    const claimBadge = document.getElementById("quick-pred-claim-badge");
    const expl = document.getElementById("quick-pred-explanation");

    preview.classList.remove("hidden");
    const isAssump = data.rules_v1.is_assumption === "1" || data.model_a.probability >= state.threshold;
    badge.className = `status-badge ${isAssump ? "status-badge-holding" : "status-badge-unmapped"}`;
    badge.textContent = isAssump ? "PREDICTED ASSUMPTION (1)" : "FACTUAL (0)";
    meta.textContent = `Model A Prob: ${data.model_a.probability.toFixed(3)} | Rules v1: ${data.rules_v1.is_assumption} (${data.rules_v1.reason}) | Topic: ${data.rules_v1.topic || "other"}`;

    if (data.claim && data.claim.monitored) {
      claimBadge.className = `status-badge status-badge-${data.claim.status}`;
      claimBadge.textContent = `FRED: ${data.claim.status.toUpperCase()}`;
      expl.textContent = data.claim.explanation;
    } else {
      claimBadge.className = "tesla-badge-mono";
      claimBadge.textContent = data.claim ? data.claim.status.toUpperCase() : "UNMAPPED";
      expl.textContent = data.claim ? data.claim.explanation : "";
    }
    showToast(`Predicted: P=${data.model_a.probability.toFixed(3)} | FRED=${data.claim ? data.claim.status.toUpperCase() : "N/A"}`);
  }

  async function injectNewAssumption(sentence, section = "5. Custom Strategic Assumptions") {
    if (!sentence.trim()) {
      showToast("Enter an assumption sentence to inject.");
      return;
    }
    const res = await fetch("/api/add-assumption", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        doc_source: state.docSource,
        sentence: sentence,
        section: section,
        as_of: state.asOf,
      }),
    });
    const added = await res.json();
    if (added.error) {
      showToast("Error: " + added.error);
      return;
    }
    const newId = added.sentence_id || added.injected?.sentence_id;
    state.highlightedSentenceId = newId;
    showToast(
      `Injected ${newId} into ${state.docSource.toUpperCase()}! FRED Status: ${added.claim ? added.claim.status.toUpperCase() : "ADDED"}`
    );
    await runQuickPredictOnly(sentence);
    await fetchAnalysis();
  }

  async function runSandboxExplain(sentence) {
    if (!sentence.trim()) return;
    const res = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence: sentence, as_of: state.asOf }),
    });
    const data = await res.json();
    if (data.error) return;

    const tokens = [
      ...(data.model_a.top_positive || []),
      ...(data.model_a.top_negative || []),
    ];
    const maxAbs = Math.max(...tokens.map((t) => Math.abs(t.contribution)), 0.1);

    document.getElementById("sandbox-results-grid").innerHTML = `
      <div class="sandbox-box">
        <span class="tesla-kicker">MODEL A PROBABILITY</span>
        <div style="font-size:24px;font-weight:700;color:var(--tesla-blue);margin:4px 0;">
          ${(data.model_a.probability * 100).toFixed(1)}%
        </div>
        <div class="text-muted">Intercept: ${data.model_a.intercept} · Matched N-Grams: ${data.model_a.all_Matched_ngrams}</div>
      </div>
      <div class="sandbox-box">
        <span class="tesla-kicker">RULES V1 &amp; FRED CHECK</span>
        <div style="margin:6px 0;">
          <span class="status-badge status-badge-${escapeHtml(data.claim ? data.claim.status : "unmapped")}">
            ${escapeHtml(data.claim ? data.claim.status.toUpperCase() : "UNMAPPED")}
          </span>
          <span class="tesla-badge-mono" style="margin-left:6px;">Topic: ${escapeHtml(data.rules_v1.topic || "other")}</span>
        </div>
        <div style="font-size:11.5px;">${escapeHtml(data.claim ? data.claim.explanation : "")}</div>
      </div>
      <div class="sandbox-box">
        <span class="tesla-kicker">TOKEN LOG-ODDS ATTRIBUTION</span>
        <div class="weights-bars" style="margin-top:6px;">
          ${
            tokens.length === 0
              ? `<div class="text-muted">No matching TF-IDF tokens.</div>`
              : tokens
                  .slice(0, 5)
                  .map((t) => {
                    const w = Math.round((Math.abs(t.contribution) / maxAbs) * 100);
                    return `
                      <div class="weight-row">
                        <span>${escapeHtml(t.ngram)}</span>
                        <div class="weight-bar-bg"><div class="weight-bar-fill-pos" style="width:${w}%"></div></div>
                        <span class="text-cyan">${t.contribution >= 0 ? "+" : ""}${t.contribution.toFixed(3)}</span>
                      </div>
                    `;
                  })
                  .join("")
          }
        </div>
      </div>
    `;
  }

  // =========================================================================
  // NAVIGATION & EVENT BINDINGS
  // =========================================================================
  function switchWorkspaceTab(tabName) {
    state.activeTab = tabName;
    document.querySelectorAll(".dock-app-btn[data-tab]").forEach((btn) => {
      const isMatch = btn.getAttribute("data-tab") === tabName;
      btn.classList.toggle("active", isMatch);
      btn.setAttribute("aria-selected", isMatch ? "true" : "false");
    });
    document.querySelectorAll(".workspace-panel").forEach((p) => {
      p.classList.toggle("active", p.id === `panel-${tabName}`);
    });
    if (tabName === "horizon" && !state.seriesData) fetchSeriesData();
    if (tabName === "modellab" && !state.modelLabData) fetchModelLab(false);
  }

  function setActiveDocGear(docId) {
    document.querySelectorAll("#doc-selector-group .prnd-gear").forEach((g) => {
      g.classList.toggle("active", g.getAttribute("data-doc") === docId);
    });
  }

  function bindEvents() {
    // Bottom Dock App Buttons
    document.querySelectorAll(".dock-app-btn[data-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        switchWorkspaceTab(btn.getAttribute("data-tab"));
      });
    });

    // Jump links
    document.querySelectorAll("[data-jump-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-jump-tab");
        const filter = btn.getAttribute("data-jump-filter");
        if (filter) {
          state.matrixFilter = filter;
          document.querySelectorAll("#matrix-filter-group .seg-item").forEach((p) => {
            p.classList.toggle("active", p.getAttribute("data-filter") === filter);
          });
          if (state.analysis) renderClauseMatrix(state.analysis.sentences || []);
        }
        switchWorkspaceTab(targetTab);
      });
    });

    // PRND Plan Selector
    document.querySelectorAll("#doc-selector-group .prnd-gear").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.docSource = btn.getAttribute("data-doc");
        state.overrideOil = null;
        state.overrideFx = null;
        setActiveDocGear(state.docSource);
        fetchAnalysis();
        showToast(`Engaged Plan Gear: ${state.docSource.toUpperCase()}`);
      });
    });

    // Date Horizon Presets & Picker
    document.querySelectorAll("#date-preset-group .seg-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.asOf = btn.getAttribute("data-date");
        document.getElementById("as-of-date-input").value = state.asOf;
        document.querySelectorAll("#date-preset-group .seg-item").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        fetchAnalysis();
      });
    });

    document.getElementById("as-of-date-input").addEventListener("change", (e) => {
      state.asOf = e.target.value;
      fetchAnalysis();
    });

    // Detector Engine Selector
    document.querySelectorAll("#mode-selector-group .seg-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.mode = btn.getAttribute("data-mode");
        document.querySelectorAll("#mode-selector-group .seg-item").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        fetchAnalysis();
      });
    });

    // ML Gate Slider
    const threshSlider = document.getElementById("threshold-slider");
    threshSlider.addEventListener("input", () => {
      state.threshold = parseFloat(threshSlider.value);
      document.getElementById("threshold-display").textContent = state.threshold.toFixed(2);
    });
    threshSlider.addEventListener("change", () => fetchAnalysis());

    // Theme Toggle
    document.getElementById("btn-theme-toggle").addEventListener("click", () => {
      const html = document.documentElement;
      const next = html.getAttribute("data-theme") === "light" ? "dark" : "light";
      html.setAttribute("data-theme", next);
      document.getElementById("theme-icon").textContent = next === "dark" ? "☀" : "☾";
    });

    // 3D Camera Controls
    document.querySelectorAll("[data-cam]").forEach((btn) => {
      btn.addEventListener("click", () => {
        threeCtrl.camMode = btn.getAttribute("data-cam");
        document.querySelectorAll("[data-cam]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
      });
    });

    document.getElementById("cam-btn-wireframe").addEventListener("click", (e) => {
      threeCtrl.wireframeOnly = !threeCtrl.wireframeOnly;
      e.currentTarget.classList.toggle("active", threeCtrl.wireframeOnly);
      if (threeCtrl.egoCore) threeCtrl.egoCore.material.wireframe = threeCtrl.wireframeOnly;
      threeCtrl.satelliteMeshes.forEach((m) => {
        m.material.wireframe = threeCtrl.wireframeOnly;
      });
    });

    document.getElementById("cam-btn-pause").addEventListener("click", (e) => {
      threeCtrl.isPaused = !threeCtrl.isPaused;
      e.currentTarget.textContent = threeCtrl.isPaused ? "PLAY" : "PAUSE";
      e.currentTarget.classList.toggle("active", threeCtrl.isPaused);
    });

    document.getElementById("cam-btn-reset").addEventListener("click", () => {
      threeCtrl.camMode = "fsd";
      threeCtrl.userYaw = 0;
      threeCtrl.userPitch = 0.34;
      threeCtrl.zoomDist = 10.5;
      document.querySelectorAll("[data-cam]").forEach((b) => {
        b.classList.toggle("active", b.getAttribute("data-cam") === "fsd");
      });
    });

    // Quick Predict & Inject Bar
    document.querySelectorAll(".quick-predict-card .tesla-pill-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const sample = btn.getAttribute("data-sample");
        document.getElementById("quick-assumption-input").value = sample;
        runQuickPredictOnly(sample);
      });
    });

    document.getElementById("btn-quick-predict").addEventListener("click", () => {
      runQuickPredictOnly(document.getElementById("quick-assumption-input").value);
    });

    document.getElementById("btn-quick-inject").addEventListener("click", () => {
      injectNewAssumption(document.getElementById("quick-assumption-input").value);
    });

    // Stress Sliders & Bottom Dock Climate Steppers
    const oilSlider = document.getElementById("sim-oil-slider");
    const fxSlider = document.getElementById("sim-fx-slider");
    oilSlider.addEventListener("input", () => {
      state.overrideOil = parseFloat(oilSlider.value);
      document.getElementById("sim-oil-val").textContent = `$${state.overrideOil.toFixed(2)}`;
      document.getElementById("dock-oil-val").textContent = `$${state.overrideOil.toFixed(1)}`;
    });
    oilSlider.addEventListener("change", () => fetchAnalysis());

    fxSlider.addEventListener("input", () => {
      state.overrideFx = parseFloat(fxSlider.value);
      document.getElementById("sim-fx-val").textContent = `₹${state.overrideFx.toFixed(2)}`;
      document.getElementById("dock-fx-val").textContent = `₹${state.overrideFx.toFixed(1)}`;
    });
    fxSlider.addEventListener("change", () => fetchAnalysis());

    document.getElementById("dock-oil-down").addEventListener("click", () => {
      const cur = parseFloat(oilSlider.value || "95");
      state.overrideOil = Math.max(50, cur - 2.5);
      oilSlider.value = state.overrideOil;
      fetchAnalysis();
    });
    document.getElementById("dock-oil-up").addEventListener("click", () => {
      const cur = parseFloat(oilSlider.value || "95");
      state.overrideOil = Math.min(130, cur + 2.5);
      oilSlider.value = state.overrideOil;
      fetchAnalysis();
    });
    document.getElementById("dock-fx-down").addEventListener("click", () => {
      const cur = parseFloat(fxSlider.value || "92");
      state.overrideFx = Math.max(78, cur - 0.5);
      fxSlider.value = state.overrideFx;
      fetchAnalysis();
    });
    document.getElementById("dock-fx-up").addEventListener("click", () => {
      const cur = parseFloat(fxSlider.value || "92");
      state.overrideFx = Math.min(98, cur + 0.5);
      fxSlider.value = state.overrideFx;
      fetchAnalysis();
    });

    document.getElementById("btn-reset-sim").addEventListener("click", () => {
      state.overrideOil = null;
      state.overrideFx = null;
      fetchAnalysis();
      showToast("Restored live point-in-time FRED readings.");
    });

    // Matrix Search, Filter & Append
    document.getElementById("matrix-search-input").addEventListener("input", (e) => {
      state.matrixSearch = e.target.value;
      if (state.analysis) renderClauseMatrix(state.analysis.sentences || []);
    });

    document.getElementById("matrix-topic-select").addEventListener("change", (e) => {
      state.matrixTopic = e.target.value;
      if (state.analysis) renderClauseMatrix(state.analysis.sentences || []);
    });

    document.querySelectorAll("#matrix-filter-group .seg-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.matrixFilter = btn.getAttribute("data-filter");
        document.querySelectorAll("#matrix-filter-group .seg-item").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        if (state.analysis) renderClauseMatrix(state.analysis.sentences || []);
      });
    });

    document.getElementById("btn-matrix-add").addEventListener("click", async () => {
      const txt = document.getElementById("matrix-add-input").value;
      const sec = document.getElementById("matrix-add-section").value || "5. Custom Strategic Assumptions";
      await injectNewAssumption(txt, sec);
      document.getElementById("matrix-add-input").value = "";
    });

    // Horizon Lookback Window
    document.querySelectorAll("#horizon-years-group .seg-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.horizonYears = parseInt(btn.getAttribute("data-years"), 10);
        document.querySelectorAll("#horizon-years-group .seg-item").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        fetchSeriesData();
      });
    });

    // Sandbox & Retrain
    document.querySelectorAll(".sandbox-preset").forEach((btn) => {
      btn.addEventListener("click", () => {
        const txt = btn.getAttribute("data-text");
        document.getElementById("sandbox-sentence-input").value = txt;
        runSandboxExplain(txt);
      });
    });

    document.getElementById("btn-sandbox-explain").addEventListener("click", () => {
      runSandboxExplain(document.getElementById("sandbox-sentence-input").value);
    });

    document.getElementById("btn-sandbox-inject").addEventListener("click", () => {
      injectNewAssumption(document.getElementById("sandbox-sentence-input").value);
    });

    document.getElementById("btn-retrain-model").addEventListener("click", () => {
      fetchModelLab(true);
    });

    // Inspector Drawer
    document.getElementById("btn-close-drawer").addEventListener("click", () => {
      document.getElementById("inspector-drawer").classList.add("hidden");
    });

    ["drawer-btn-approve", "drawer-btn-reject", "drawer-btn-flag"].forEach((id) => {
      document.getElementById(id).addEventListener("click", async (e) => {
        if (!state.activeDrawerSentence) return;
        const decision = e.currentTarget.getAttribute("data-decision");
        const topic = document.getElementById("drawer-topic-select").value;
        await submitSentenceReview(state.activeDrawerSentence, decision, topic);
      });
    });

    document.getElementById("drawer-topic-select").addEventListener("change", async (e) => {
      if (!state.activeDrawerSentence) return;
      await submitSentenceReview(state.activeDrawerSentence, "1", e.target.value);
    });

    // Reset Session
    document.getElementById("btn-reset-session").addEventListener("click", async () => {
      await fetch("/api/reset-session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      state.overrideOil = null;
      state.overrideFx = null;
      state.highlightedSentenceId = null;
      await fetchAnalysis();
      showToast("Session overrides and custom injected clauses reset.");
    });

    // Ingest PDF Modal
    const modalBackdrop = document.getElementById("pdf-modal-backdrop");
    const openModal = () => modalBackdrop.classList.remove("hidden");
    document.getElementById("btn-open-pdf-modal").addEventListener("click", openModal);
    document.getElementById("dock-btn-ingest").addEventListener("click", openModal);
    document.getElementById("btn-close-pdf-modal").addEventListener("click", () => {
      modalBackdrop.classList.add("hidden");
    });

    document.querySelectorAll("#ingest-mode-tabs .seg-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.getAttribute("data-ingest-tab");
        document.querySelectorAll("#ingest-mode-tabs .seg-item").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        document.querySelectorAll(".ingest-tab-panel").forEach((p) => {
          p.classList.toggle("active", p.id === `ingest-panel-${tab}`);
        });
      });
    });

    // Generate & Ingest PDF (PyMuPDF -> plan3.pdf)
    document.getElementById("btn-run-generate-pdf").addEventListener("click", async () => {
      const payload = {
        company_name: document.getElementById("gen-company-name").value,
        crude_low: parseFloat(document.getElementById("gen-crude-low").value || "68"),
        crude_high: parseFloat(document.getElementById("gen-crude-high").value || "82"),
        diesel_cap: parseFloat(document.getElementById("gen-diesel-cap").value || "94"),
        fx_target: parseFloat(document.getElementById("gen-fx-target").value || "83.5"),
        rate_target: parseFloat(document.getElementById("gen-rate-target").value || "9.8"),
        demand_growth: parseFloat(document.getElementById("gen-demand-growth").value || "11.0"),
        as_of: state.asOf,
        mode: state.mode,
        threshold: state.threshold,
      };
      const res = await fetch("/api/generate-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.error) {
        showToast("PDF Generation Error: " + data.error);
        return;
      }
      state.docSource = "plan3";
      setActiveDocGear("plan3");
      state.analysis = data.analysis;
      renderAllViews();
      modalBackdrop.classList.add("hidden");
      showToast(
        `Generated & Ingested plan3.pdf (${data.generated.size_bytes} bytes): ${data.analysis.summary.assumptions_detected} assumptions!`
      );
    });

    // File Upload
    const fileInput = document.getElementById("file-upload-input");
    const submitUploadBtn = document.getElementById("btn-submit-upload");
    document.getElementById("btn-Browse-file").addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      document.getElementById("upload-filename-display").textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
      const reader = new FileReader();
      reader.onload = () => {
        const str = String(reader.result || "");
        const idx = str.indexOf(",");
        state.uploadedB64 = idx >= 0 ? str.slice(idx + 1) : str;
        state.uploadedFilename = file.name;
        submitUploadBtn.disabled = false;
      };
      reader.readAsDataURL(file);
    });

    submitUploadBtn.addEventListener("click", async () => {
      if (!state.uploadedB64) return;
      state.docSource = "uploaded";
      setActiveDocGear("");
      modalBackdrop.classList.add("hidden");
      await fetchAnalysis();
      showToast(`Ingested uploaded file: ${state.uploadedFilename}`);
    });

    // Paste Raw Text
    document.getElementById("btn-submit-raw-text").addEventListener("click", async () => {
      state.customText = document.getElementById("raw-text-input").value;
      state.docSource = "custom";
      setActiveDocGear("");
      modalBackdrop.classList.add("hidden");
      await fetchAnalysis();
      showToast("Analyzed custom pasted business plan.");
    });
  }

  document.addEventListener("DOMContentLoaded", async () => {
    bindEvents();
    initThreeScene();
    await fetchAnalysis();
    fetchSeriesData();
    fetchModelLab(false);
    runSandboxExplain(document.getElementById("sandbox-sentence-input").value);
  });
})();
