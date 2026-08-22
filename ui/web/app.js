/* VERITY web UI client.
 *
 * Talks to the FastAPI backend it is served from: POST /verify for a check,
 * POST /report for the PDF, GET /health for the header indicator.
 *
 * Everything rendered from the response is built with createElement +
 * textContent rather than innerHTML. Evidence snippets, source names and
 * URLs are third-party text pulled off the open web by the retrieval agent -
 * interpolating them into markup would be a stored-XSS path straight through
 * our own verdict card.
 */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var STAGES = {
    text:  ["Reading input", "Extracting claims", "Gathering evidence", "Building verdict"],
    url:   ["Fetching the page", "Extracting claims", "Gathering evidence", "Building verdict"],
    image: ["Reading image", "Running forensics", "Extracting claims", "Gathering evidence", "Building verdict"]
  };

  // Banner treatment per verdict, matching the design: the two "bad" labels
  // invert to accent, the two "good" ones to ink, everything else stays a
  // plain surface. MIXED/UNVERIFIABLE deliberately get no dramatic colour -
  // UNVERIFIABLE is a neutral outcome here, not a failure.
  var TONE = { FALSE: "accent", MISLEADING: "accent", TRUE: "ink", MOSTLY_TRUE: "ink" };

  var TAG = {
    TRUE: "tag tag-neutral", MOSTLY_TRUE: "tag tag-neutral",
    MIXED: "tag tag-outline", UNVERIFIABLE: "tag tag-outline", SATIRE_OPINION: "tag tag-outline",
    MISLEADING: "tag tag-accent", FALSE: "tag tag-accent"
  };

  var state = { tab: "text", file: null, verdict: null, timer: null, open: null };

  /* ───────────────────────────── theme ───────────────────────────── */

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    $("theme-light").setAttribute("aria-pressed", String(theme !== "dark"));
    $("theme-dark").setAttribute("aria-pressed", String(theme === "dark"));
    try { localStorage.setItem("verity-theme", theme); } catch (e) { /* private mode */ }
  }

  function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem("verity-theme"); } catch (e) { /* private mode */ }
    if (!saved) {
      saved = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    applyTheme(saved);
  }

  /* ───────────────────────────── tabs ────────────────────────────── */

  function pickTab(tab) {
    state.tab = tab;
    ["text", "url", "image"].forEach(function (t) {
      $("tab-" + t).setAttribute("aria-selected", String(t === tab));
      $("panel-" + t).hidden = t !== tab;
    });
    hideError();
  }

  /* ──────────────────────────── helpers ──────────────────────────── */

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function showError(message) {
    var box = $("error-box");
    box.textContent = message;
    box.hidden = false;
  }

  function hideError() { $("error-box").hidden = true; }

  function pct(x) { return Math.round((Number(x) || 0) * 100) + "%"; }

  /* ─────────────────────────── progress ──────────────────────────── */
  /* The backend answers one blocking request, so these stages are an
     optimistic timeline, not live telemetry - the same approach the Streamlit
     UI took. The authoritative record of what actually ran is the
     "What we checked" panel, which is built from the real checks_performed. */

  function startStages() {
    var labels = STAGES[state.tab];
    var host = $("stages");
    clear(host);
    labels.forEach(function (label, i) {
      var row = el("div", "stage-row");
      row.setAttribute("data-state", i === 0 ? "active" : "pending");
      row.appendChild(el("span", "stage-mark", i === 0 ? "→" : "→"));
      row.appendChild(el("span", null, label));
      host.appendChild(row);
    });
    $("running").hidden = false;

    var i = 0;
    state.timer = setInterval(function () {
      var rows = host.children;
      if (i < rows.length - 1) {
        rows[i].setAttribute("data-state", "done");
        rows[i].firstChild.textContent = "──";
        i += 1;
        rows[i].setAttribute("data-state", "active");
      }
    }, 2500);
  }

  function stopStages() {
    clearInterval(state.timer);
    state.timer = null;
    $("running").hidden = true;
  }

  /* ──────────────────────────── rendering ────────────────────────── */

  // The API's Verdict has no prose summary, so derive an honest one from the
  // numbers it does return. If a `summary` field is ever added server-side
  // this picks it up automatically and stops synthesising.
  function summaryFor(v) {
    if (v.summary) return v.summary;
    var claims = (v.per_claim || []).length;
    var sup = 0, ref = 0;
    (v.per_claim || []).forEach(function (c) { sup += c.supporting || 0; ref += c.refuting || 0; });
    var sources = (v.evidence_citations || []).length;
    if (!claims) return "No checkable claims were extracted, so there is nothing to verify against sources.";
    if (!sources) return claims + (claims === 1 ? " claim" : " claims") +
      " extracted, but no source in the index confirms or contradicts them.";
    return claims + (claims === 1 ? " claim" : " claims") + " checked against " + sources +
      (sources === 1 ? " citation" : " citations") + " — " + sup + " supporting, " + ref + " refuting.";
  }

  function renderBanner(v) {
    var tone = TONE[v.label] || "none";
    $("banner").setAttribute("data-tone", tone);
    $("verdict-label").textContent = String(v.label || "").replace(/_/g, " ");
    $("conf-text").textContent = pct(v.confidence);
    $("bar-fill").style.width = pct(v.confidence);
    $("summary").textContent = summaryFor(v);
  }

  function renderEvidence(host, items) {
    items.forEach(function (e) {
      var box = el("div", "evidence");
      box.appendChild(el("p", "evidence-snippet", e.snippet || ""));

      var meta = el("p", "evidence-meta");
      var link = el("a", null, e.source_name || "source");
      // Only http(s) becomes a link - a javascript: URL from a scraped page
      // must never become a clickable anchor.
      if (/^https?:\/\//i.test(e.url || "")) {
        link.href = e.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
      meta.appendChild(link);

      var bits = [];
      if (e.source_tier) bits.push("Tier " + e.source_tier);
      if (e.published_date) bits.push(String(e.published_date).slice(0, 10));
      if (bits.length) meta.appendChild(el("span", "text-muted", bits.join(" · ")));

      box.appendChild(meta);
      host.appendChild(box);
    });
  }

  function renderClaims(v) {
    var host = $("claims");
    clear(host);
    var claims = v.per_claim || [];
    var citations = v.evidence_citations || [];

    $("claim-count").textContent = claims.length
      ? claims.length + (claims.length === 1 ? " claim extracted" : " claims extracted") +
        " — tap one to read the evidence behind it."
      : "No claims were extracted from this input.";

    claims.forEach(function (c, i) {
      var mine = citations.filter(function (e) { return e.claim_id === c.claim_id; });
      var isOpen = state.open === null ? i === 0 : state.open === c.claim_id;

      var wrap = el("div", "claim");
      var head = el("button", "claim-head");
      head.type = "button";
      head.setAttribute("aria-expanded", String(isOpen));

      head.appendChild(el("span", "claim-num", String(i + 1).padStart(2, "0")));

      var body = el("div", "claim-body");
      // per_claim carries no claim text today - fall back to the id so the row
      // is still identifiable, and light up automatically if `text` is added.
      body.appendChild(el("span", "claim-text", c.text || ("Claim " + c.claim_id)));

      var meta = el("span", "claim-meta");
      meta.appendChild(el("span", TAG[c.label] || "tag tag-outline", String(c.label || "").replace(/_/g, " ")));
      meta.appendChild(el("span", "text-muted", (c.supporting || 0) + " supporting · " + (c.refuting || 0) + " refuting"));
      body.appendChild(meta);
      head.appendChild(body);

      head.appendChild(el("span", "claim-chevron", isOpen ? "−" : "+"));
      wrap.appendChild(head);

      var detail = el("div", "claim-detail");
      detail.hidden = !isOpen;
      if (c.rationale) detail.appendChild(el("p", null, c.rationale));
      if (mine.length) renderEvidence(detail, mine);
      else detail.appendChild(el("p", "text-muted no-evidence",
        "No source in the index confirms or contradicts this claim."));
      wrap.appendChild(detail);

      head.addEventListener("click", function () {
        var nowOpen = detail.hidden;
        state.open = nowOpen ? c.claim_id : "";
        detail.hidden = !nowOpen;
        head.setAttribute("aria-expanded", String(nowOpen));
        head.lastChild.textContent = nowOpen ? "−" : "+";
      });

      host.appendChild(wrap);
    });
  }

  function renderSignals(v) {
    var host = $("signals");
    clear(host);
    (v.signals || []).forEach(function (s) {
      var tr = el("tr");
      var td = el("td");
      td.style.paddingBlock = "10px";
      var name = el("span", null, s.name);
      name.style.display = "block";
      name.style.fontWeight = "600";
      td.appendChild(name);
      if (s.note) td.appendChild(el("span", "text-muted", s.note));
      td.lastChild.style.fontSize = "12px";
      tr.appendChild(td);
      tr.appendChild(el("td", "tabular", Number(s.score).toFixed(2)));
      tr.appendChild(el("td", "text-muted tabular", Number(s.weight).toFixed(2)));
      host.appendChild(tr);
    });
  }

  function renderChecks(v) {
    var host = $("checks");
    clear(host);
    (v.checks_performed || []).forEach(function (k) {
      var row = el("div", "check");
      row.setAttribute("data-status", k.status);
      row.appendChild(el("span", "check-dot"));
      row.appendChild(el("span", "check-agent", k.agent_name));
      row.appendChild(el("span", "text-muted check-note", k.note || k.status));
      row.appendChild(el("span", "text-muted check-ms", (k.duration_ms || 0) + " ms"));
      host.appendChild(row);
    });
  }

  function renderCaveats(v) {
    var host = $("caveats");
    clear(host);
    var list = v.caveats || [];
    $("caveats-block").hidden = list.length === 0;
    list.forEach(function (text) { host.appendChild(el("p", "caveat", text)); });
  }

  function renderOcr(v) {
    // OCRResult is not part of the Verdict contract today, so this panel stays
    // hidden. It renders the moment an `ocr` object appears on the response.
    var ocr = v.ocr;
    $("ocr-block").hidden = !ocr;
    if (!ocr) return;
    $("ocr-text").textContent = ocr.full_text || "";
    var regions = (ocr.regions || []).length;
    $("ocr-meta").textContent = regions + (regions === 1 ? " region" : " regions") +
      " · read on " + (ocr.device_used || "CPU") + " in " + (ocr.duration_ms || 0) + " ms";
  }

  function render(v) {
    state.verdict = v;
    state.open = null;
    renderBanner(v);
    renderClaims(v);
    renderSignals(v);
    renderChecks(v);
    renderCaveats(v);
    renderOcr(v);
    $("results").hidden = false;
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ───────────────────────────── actions ─────────────────────────── */

  function buildBody() {
    var fd = new FormData();
    if (state.tab === "text") {
      var text = $("v-text").value.trim();
      if (!text) return { error: "Paste some article text first." };
      fd.append("text", text);
    } else if (state.tab === "url") {
      var url = $("v-url").value.trim();
      if (!url) return { error: "Paste a URL first." };
      if (!/^https?:\/\//i.test(url)) return { error: "The URL needs to start with http:// or https://" };
      fd.append("url", url);
    } else {
      if (!state.file) return { error: "Choose a clipping photo first." };
      fd.append("image", state.file, state.file.name);
    }
    return { body: fd };
  }

  function run() {
    hideError();
    var built = buildBody();
    if (built.error) { showError(built.error); return; }

    $("results").hidden = true;
    $("btn-run").disabled = true;
    $("btn-run").textContent = "Checking…";
    startStages();

    fetch("/verify", { method: "POST", body: built.body })
      .then(function (res) {
        return res.text().then(function (raw) {
          if (!res.ok) {
            var detail = raw;
            try { detail = JSON.parse(raw).detail || raw; } catch (e) { /* not JSON */ }
            throw new Error("Backend returned " + res.status + ": " + detail);
          }
          return JSON.parse(raw);
        });
      })
      .then(render)
      .catch(function (err) {
        showError(err.message === "Failed to fetch"
          ? "Could not reach the backend. Is it running on this port?"
          : err.message);
      })
      .finally(function () {
        stopStages();
        $("btn-run").disabled = false;
        $("btn-run").textContent = "Check this";
      });
  }

  function reset() {
    stopStages();
    hideError();
    $("v-text").value = "";
    $("v-url").value = "";
    $("v-file").value = "";
    $("file-label").textContent = "No file chosen";
    state.file = null;
    state.verdict = null;
    state.open = null;
    $("results").hidden = true;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function downloadPdf() {
    if (!state.verdict) return;
    var btn = $("btn-pdf");
    btn.disabled = true;
    fetch("/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.verdict)
    })
      .then(function (res) {
        if (!res.ok) throw new Error("Report failed (" + res.status + ")");
        return res.blob();
      })
      .then(function (blob) {
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = "verity_report.pdf";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      })
      .catch(function (err) { showError(err.message); })
      .finally(function () { btn.disabled = false; });
  }

  function setFile(file) {
    state.file = file || null;
    $("file-label").textContent = file ? file.name : "No file chosen";
  }

  function checkHealth() {
    fetch("/health")
      .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
      .then(function () {
        $("api-dot").setAttribute("data-state", "up");
        $("api-state-text").textContent = "API READY";
      })
      .catch(function () {
        $("api-dot").setAttribute("data-state", "down");
        $("api-state-text").textContent = "API UNREACHABLE";
      });
  }

  /* ───────────────────────────── wiring ──────────────────────────── */

  document.addEventListener("DOMContentLoaded", function () {
    initTheme();
    $("theme-light").addEventListener("click", function () { applyTheme("light"); });
    $("theme-dark").addEventListener("click", function () { applyTheme("dark"); });

    ["text", "url", "image"].forEach(function (t) {
      $("tab-" + t).addEventListener("click", function () { pickTab(t); });
    });

    $("btn-run").addEventListener("click", run);
    $("btn-clear").addEventListener("click", reset);
    $("btn-again").addEventListener("click", reset);
    $("btn-pdf").addEventListener("click", downloadPdf);

    $("v-file").addEventListener("change", function (e) { setFile(e.target.files && e.target.files[0]); });

    var dz = $("dropzone");
    ["dragenter", "dragover"].forEach(function (evt) {
      dz.addEventListener(evt, function (e) { e.preventDefault(); dz.classList.add("is-over"); });
    });
    ["dragleave", "drop"].forEach(function (evt) {
      dz.addEventListener(evt, function (e) { e.preventDefault(); dz.classList.remove("is-over"); });
    });
    dz.addEventListener("drop", function (e) {
      var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) setFile(f);
    });

    // Ctrl/Cmd+Enter submits from the textarea, the way a review tool should.
    $("v-text").addEventListener("keydown", function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
    });
    $("v-url").addEventListener("keydown", function (e) { if (e.key === "Enter") run(); });

    checkHealth();
  });
})();
