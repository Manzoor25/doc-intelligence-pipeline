const state = {
  filter: "needs_review",
  documents: [],
  selectedDoc: null,
  currentPage: 1,
  zoomed: false,
};

function esc(val) {
  return String(val ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function confClass(score) {
  if (score >= 0.85) return "conf-high";
  if (score >= 0.65) return "conf-mid";
  return "conf-low";
}

async function loadCapabilities() {
  try {
    const res = await fetch("/api/v1/capabilities");
    const cap = await res.json();
    const banner = document.getElementById("capabilityBanner");
    banner.textContent =
      `model=${cap.model} | vision=${cap.supports_vision} | json_schema=${cap.supports_json_schema} | path=${cap.chosen_extraction_path}`;
  } catch {
    document.getElementById("capabilityBanner").textContent = "Endpoint probe unavailable";
  }
}

async function refreshQueue(preserveSelection = true) {
  const qs = state.filter ? `?status=${encodeURIComponent(state.filter)}&limit=150` : "?limit=150";
  const res = await fetch(`/api/v1/documents${qs}`);
  const payload = await res.json();
  const sum = payload.summary;

  document.getElementById("kpiTotal").textContent = sum.total_documents;
  document.getElementById("kpiApproved").textContent = sum.approved;
  document.getElementById("kpiReview").textContent = sum.needs_review;
  document.getElementById("kpiQuarantined").textContent = sum.quarantined_injections;
  document.getElementById("kpiTelemetry").textContent =
    `$${Number(sum.avg_cost_usd).toFixed(5)} / ${Math.round(sum.avg_latency_ms)} ms`;
  document.getElementById("kpiExports").textContent =
    `${sum.webhooks_delivered} / ${sum.eval_feedback_samples}`;

  state.documents = payload.documents;
  renderDocList();

  if (!preserveSelection || !state.selectedDoc) {
    const urlDocId = new URLSearchParams(window.location.search).get("doc");
    const urlMatch = urlDocId
      ? state.documents.find((d) => d.id === urlDocId || d.id.startsWith(urlDocId))
      : null;
    const defaultDoc =
      urlMatch ||
      state.documents.find((d) => Boolean(d.document_number)) ||
      state.documents[0];
    if (defaultDoc) {
      await selectDocument(defaultDoc.id);
    } else if (state.filter === "needs_review" && sum.total_documents > 0) {
      // Switch to All automatically if review queue is empty
      setActiveFilter("");
    }
  }
}

function renderDocList() {
  const container = document.getElementById("docList");
  if (!state.documents.length) {
    container.innerHTML = `<div class="empty-note">No documents matching filter "${esc(state.filter || "all")}".</div>`;
    return;
  }

  container.innerHTML = state.documents
    .map((d) => {
      const selected = state.selectedDoc && state.selectedDoc.id === d.id ? "selected" : "";
      const confPct = d.overall_confidence != null ? `${Math.round(d.overall_confidence * 100)}%` : "—";
      const amt = d.total_amount != null ? `${esc(d.currency || "")} ${Number(d.total_amount).toFixed(2)}` : "Pending";
      const injBadge = d.injection_detected
        ? `<span class="badge badge-quarantine">INJECTION</span>`
        : `<span class="badge badge-${esc(d.status)}">${esc(d.status.replace("_", " "))}</span>`;
      return `
        <div class="doc-card ${selected}" data-doc-id="${esc(d.id)}">
          <div class="doc-card-top">
            <span class="doc-num">${esc(d.document_number || d.file_name)}</span>
            ${injBadge}
          </div>
          <div class="doc-supplier">${esc(d.supplier_name || d.file_name)}</div>
          <div class="doc-card-sub">
            <span>${esc((d.document_type || "doc").toUpperCase())} &middot; ${esc((d.language || "en").toUpperCase())}</span>
            <span><strong>${amt}</strong> &middot; Conf ${confPct}</span>
          </div>
        </div>
      `;
    })
    .join("");

  container.querySelectorAll(".doc-card").forEach((el) => {
    el.addEventListener("click", () => selectDocument(el.dataset.docId));
  });
}

async function selectDocument(docId) {
  const res = await fetch(`/api/v1/documents/${encodeURIComponent(docId)}`);
  if (!res.ok) return;
  const doc = await res.json();
  state.selectedDoc = doc;
  state.currentPage = 1;
  renderDocList();
  renderViewer(doc);
  renderEditor(doc);
}

function renderViewer(doc) {
  document.getElementById("viewerTitle").textContent =
    `${doc.file_name} (${doc.extraction_path || "queued"} · via ${doc.source_channel})`;
  document.getElementById("rawFileLink").href = `/api/v1/documents/${encodeURIComponent(doc.id)}/raw`;

  const safety = doc.safety_report || {};
  const alertBox = document.getElementById("safetyAlertBox");
  if (safety.injection_detected && (safety.signals || []).length > 0) {
    const snippets = safety.signals.map((s) => `[${esc(s.pattern_name)}] "${esc(s.matched_excerpt || s.matched_snippet)}"`).join(" ; ");
    alertBox.innerHTML =
      `<strong>SECURITY QUARANTINE — Embedded Prompt Injection Treated as Untrusted Data:</strong><br/>${snippets}`;
    alertBox.classList.remove("hidden");
  } else {
    alertBox.classList.add("hidden");
  }

  updatePageImage();
}

function updatePageImage() {
  const doc = state.selectedDoc;
  const img = document.getElementById("pageImage");
  const empty = document.getElementById("emptyViewer");
  if (!doc || !doc.page_count) {
    img.classList.add("hidden");
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  img.classList.remove("hidden");
  img.src = `/api/v1/documents/${encodeURIComponent(doc.id)}/pages/${state.currentPage}`;
  document.getElementById("pageIndicator").textContent = `Page ${state.currentPage} / ${doc.page_count}`;
  document.getElementById("prevPageBtn").disabled = state.currentPage <= 1;
  document.getElementById("nextPageBtn").disabled = state.currentPage >= doc.page_count;
}

function addLineItemRow(item = { sku: "", description: "", quantity: 1, unit_price: 0, line_total: 0 }) {
  const tbody = document.getElementById("lineItemsBody");
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input type="text" class="li-sku" value="${esc(item.sku || "")}" placeholder="SKU" /></td>
    <td><input type="text" class="li-desc" value="${esc(item.description || "")}" required /></td>
    <td style="width:68px"><input type="number" step="0.01" min="0.01" class="li-qty" value="${Number(item.quantity ?? 1)}" required /></td>
    <td style="width:88px"><input type="number" step="0.01" min="0" class="li-unit" value="${Number(item.unit_price ?? 0).toFixed(2)}" required /></td>
    <td style="width:92px"><input type="number" step="0.01" min="0" class="li-total" value="${Number(item.line_total ?? 0).toFixed(2)}" required /></td>
    <td style="width:28px"><button type="button" class="btn-sm remove-line-btn">&times;</button></td>
  `;
  const recalcRow = () => {
    const q = parseFloat(tr.querySelector(".li-qty").value) || 0;
    const u = parseFloat(tr.querySelector(".li-unit").value) || 0;
    tr.querySelector(".li-total").value = (q * u).toFixed(2);
    updateLineSumHint();
  };
  tr.querySelector(".li-qty").addEventListener("input", recalcRow);
  tr.querySelector(".li-unit").addEventListener("input", recalcRow);
  tr.querySelector(".li-total").addEventListener("input", updateLineSumHint);
  tr.querySelector(".remove-line-btn").addEventListener("click", () => {
    if (tbody.children.length > 1) {
      tr.remove();
      updateLineSumHint();
    }
  });
  tbody.appendChild(tr);
  updateLineSumHint();
}

function updateLineSumHint() {
  let sum = 0;
  document.querySelectorAll("#lineItemsBody .li-total").forEach((el) => {
    sum += parseFloat(el.value) || 0;
  });
  document.getElementById("lineSumHint").textContent = `Lines Sum: ${sum.toFixed(2)}`;
  return sum;
}

function renderEditor(doc) {
  document.getElementById("editorEmpty").classList.add("hidden");
  const form = document.getElementById("reviewForm");
  form.classList.remove("hidden");
  document.getElementById("toastBanner").classList.add("hidden");

  const confOverall = doc.overall_confidence != null ? `${Math.round(doc.overall_confidence * 100)}%` : "N/A";
  document.getElementById("docStatusHeader").innerHTML = `
    <div>
      <strong>Status:</strong> <span class="badge badge-${esc(doc.status)}">${esc(doc.status)}</span>
      &nbsp;|&nbsp; <strong>Calibrated Confidence:</strong> ${confOverall}
      &nbsp;|&nbsp; <strong>Repairs:</strong> ${doc.repair_attempts || 0}
    </div>
    <div>
      Cost: <code>$${Number(doc.estimated_cost_usd || 0).toFixed(5)}</code> &middot;
      ${Math.round(doc.processing_latency_ms || 0)} ms
    </div>
  `;

  // Routing reasons
  const reasons = (doc.confidence_report || {}).review_reasons || (doc.error_message ? [doc.error_message] : []);
  const reasonsBox = document.getElementById("reasonsBox");
  if (reasons.length > 0 && doc.status === "needs_review") {
    reasonsBox.innerHTML =
      `<strong>Routed to Human Review (${reasons.length} trigger${reasons.length > 1 ? "s" : ""}):</strong>` +
      `<ul>${reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`;
    reasonsBox.classList.remove("hidden");
  } else {
    reasonsBox.classList.add("hidden");
  }

  // Deterministic validation checks
  const checks = ((doc.validation_report || {}).results) || ((doc.validation_report || {}).checks) || [];
  document.getElementById("validationChecks").innerHTML = checks.length
    ? checks
        .map((c) => {
          const cls = c.passed ? "check-pass" : c.severity === "warning" ? "check-warn" : "check-fail";
          const icon = c.passed ? "PASS" : c.severity === "warning" ? "WARN" : "FAIL";
          const label = (c.rule_code || c.rule_name || "RULE").replaceAll("_", " ");
          return `<div class="check-item ${cls}" title="${esc(c.message)}">
            <span>${esc(label)}</span>
            <strong>${icon}</strong>
          </div>`;
        })
        .join("")
    : `<div class="check-item">Pending validation</div>`;

  // Per-field confidence pills
  const fieldConfs = ((doc.confidence_report || {}).fields) || {};
  document.querySelectorAll("[data-conf]").forEach((pill) => {
    const key = pill.dataset.conf;
    const fc = fieldConfs[key];
    if (fc) {
      const scoreVal = fc.score ?? fc.confidence ?? 0;
      const pct = Math.round(scoreVal * 100);
      pill.className = `conf-pill ${confClass(scoreVal)}`;
      pill.textContent = `${pct}%`;
      pill.title = (fc.reasons || []).join(" | ") || `Pass1/2 Agree=${fc.two_pass_agreed}`;
    } else {
      pill.className = "conf-pill";
      pill.textContent = "";
    }
  });

  const ex = doc.extracted_data || {
    document_type: doc.document_type || "invoice",
    language: doc.language || "en",
    document_number: "",
    supplier_name: "",
    supplier_tax_id: "",
    issue_date: "2026-03-15",
    due_date: "",
    currency: "USD",
    subtotal: 0,
    tax_amount: 0,
    total_amount: 0,
    handwritten_notes: "",
    line_items: [{ sku: "", description: "", quantity: 1, unit_price: 0, line_total: 0 }],
  };

  document.getElementById("f_document_type").value = ex.document_type || "invoice";
  document.getElementById("f_language").value = ex.language || "en";
  document.getElementById("f_document_number").value = ex.document_number || "";
  document.getElementById("f_currency").value = ex.currency || "USD";
  document.getElementById("f_supplier_name").value = ex.supplier_name || "";
  document.getElementById("f_supplier_tax_id").value = ex.supplier_tax_id || "";
  document.getElementById("f_issue_date").value = ex.issue_date || "";
  document.getElementById("f_due_date").value = ex.due_date || "";
  document.getElementById("f_subtotal").value = Number(ex.subtotal ?? 0).toFixed(2);
  document.getElementById("f_tax_amount").value = Number(ex.tax_amount ?? 0).toFixed(2);
  document.getElementById("f_total_amount").value = Number(ex.total_amount ?? 0).toFixed(2);
  document.getElementById("f_handwritten_notes").value = ex.handwritten_notes || "";
  document.getElementById("f_reviewer_notes").value = doc.reviewer_notes || "";

  const tbody = document.getElementById("lineItemsBody");
  tbody.innerHTML = "";
  (ex.line_items && ex.line_items.length ? ex.line_items : [{ description: "", quantity: 1, unit_price: 0, line_total: 0 }])
    .forEach((item) => addLineItemRow(item));

  // Audit logs
  const audits = doc.audit_logs || [];
  document.getElementById("auditTrail").innerHTML = audits
    .map(
      (a) => `<div class="audit-row">
        <code>${esc(a.event_type)}</code> by <strong>${esc(a.actor)}</strong>: ${esc(a.summary)}
      </div>`
    )
    .join("");
}

function collectReviewPayload(action) {
  const exOld = (state.selectedDoc && state.selectedDoc.extracted_data) || {};
  const lineItems = [];
  document.querySelectorAll("#lineItemsBody tr").forEach((tr) => {
    const sku = tr.querySelector(".li-sku").value.trim();
    lineItems.push({
      sku: sku || null,
      description: tr.querySelector(".li-desc").value.trim(),
      quantity: parseFloat(tr.querySelector(".li-qty").value),
      unit_price: parseFloat(tr.querySelector(".li-unit").value),
      line_total: parseFloat(tr.querySelector(".li-total").value),
    });
  });

  const subtotal = parseFloat(document.getElementById("f_subtotal").value);
  const taxAmount = parseFloat(document.getElementById("f_tax_amount").value);
  const dueVal = document.getElementById("f_due_date").value.trim();
  const taxIdVal = document.getElementById("f_supplier_tax_id").value.trim();
  const notesVal = document.getElementById("f_handwritten_notes").value.trim();

  return {
    action,
    reviewer: document.getElementById("f_reviewer").value.trim() || "ap-specialist@acme.example",
    reviewer_notes: document.getElementById("f_reviewer_notes").value.trim() || null,
    extracted_data: {
      document_type: document.getElementById("f_document_type").value,
      language: document.getElementById("f_language").value,
      document_number: document.getElementById("f_document_number").value.trim(),
      supplier_name: document.getElementById("f_supplier_name").value.trim(),
      supplier_tax_id: taxIdVal || null,
      supplier_address: exOld.supplier_address || null,
      issue_date: document.getElementById("f_issue_date").value,
      due_date: dueVal || null,
      currency: document.getElementById("f_currency").value,
      subtotal,
      tax_rate: subtotal > 0 ? Number((taxAmount / subtotal).toFixed(4)) : 0.0,
      tax_amount: taxAmount,
      total_amount: parseFloat(document.getElementById("f_total_amount").value),
      line_items: lineItems,
      handwritten_notes: notesVal || null,
    },
  };
}

async function submitReview(action) {
  if (!state.selectedDoc) return;
  const payload = collectReviewPayload(action);
  const res = await fetch(`/api/v1/documents/${encodeURIComponent(state.selectedDoc.id)}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    alert(`Validation error: ${JSON.stringify(err.detail)}`);
    return;
  }
  const out = await res.json();
  state.selectedDoc = out.document;
  renderEditor(out.document);
  const toast = document.getElementById("toastBanner");
  toast.textContent =
    action === "approve"
      ? "Saved! Exported to PostgreSQL + CSV + Finance ERP Webhook & added to Evaluation Gold Dataset."
      : "Document rejected and recorded in immutable audit trail.";
  toast.classList.remove("hidden");
  await refreshQueue(true);
}

function setActiveFilter(filterVal) {
  state.filter = filterVal;
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.filter === filterVal);
  });
  refreshQueue(false);
}

document.addEventListener("DOMContentLoaded", () => {
  loadCapabilities();
  refreshQueue(false);

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => setActiveFilter(btn.dataset.filter));
  });

  document.getElementById("prevPageBtn").addEventListener("click", () => {
    if (state.currentPage > 1) {
      state.currentPage -= 1;
      updatePageImage();
    }
  });
  document.getElementById("nextPageBtn").addEventListener("click", () => {
    if (state.selectedDoc && state.currentPage < state.selectedDoc.page_count) {
      state.currentPage += 1;
      updatePageImage();
    }
  });
  document.getElementById("zoomToggleBtn").addEventListener("click", (e) => {
    state.zoomed = !state.zoomed;
    document.getElementById("pageImage").classList.toggle("zoomed", state.zoomed);
    e.target.textContent = state.zoomed ? "Fit Width" : "Zoom 140%";
  });

  document.getElementById("addLineBtn").addEventListener("click", () => addLineItemRow());
  document.getElementById("recalcTotalsBtn").addEventListener("click", () => {
    const sub = updateLineSumHint();
    document.getElementById("f_subtotal").value = sub.toFixed(2);
    const tax = parseFloat(document.getElementById("f_tax_amount").value) || 0;
    document.getElementById("f_total_amount").value = (sub + tax).toFixed(2);
  });

  document.getElementById("reviewForm").addEventListener("submit", (e) => {
    e.preventDefault();
    submitReview("approve");
  });
  document.getElementById("rejectBtn").addEventListener("click", () => submitReview("reject"));

  document.getElementById("uploadInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("sync", "true");
    const res = await fetch("/api/v1/documents/upload", { method: "POST", body: fd });
    if (res.ok) {
      const out = await res.json();
      setActiveFilter("");
      await selectDocument(out.document.id);
    }
    e.target.value = "";
  });

  document.getElementById("scanInboxBtn").addEventListener("click", async () => {
    await fetch("/api/v1/documents/scan-inbox", { method: "POST" });
    setTimeout(() => refreshQueue(true), 1200);
  });
});
