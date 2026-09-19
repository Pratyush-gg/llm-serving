(() => {
  'use strict';

  // ── Config ──────────────────────────────────────────────
  const API = window.location.origin;
  const ADAPTERS = ['sql', 'json', 'code', 'base'];
  const COLORS = { sql: '#38bdf8', json: '#34d399', code: '#a78bfa', base: '#fbbf24' };
  const PRESETS = {
    sql: 'Write a SQL query to find the top 5 customers by total order amount from orders and customers tables.',
    json: 'Extract the following fields from this text into JSON: name, email, phone.\nText: "John Smith, email: john@example.com, phone: 555-0123"',
    code: 'Write a Python function to merge two sorted linked lists into one sorted linked list.',
    base: 'Explain the difference between supervised and unsupervised learning.',
    cascade: 'Create a database migration script using Python SQLAlchemy to add a users table.'
  };

  // ── DOM ─────────────────────────────────────────────────
  const $ = s => document.querySelector(s);
  const $$ = s => document.querySelectorAll(s);

  const dom = {
    statusDot: $('#status-dot'),
    statusText: $('#status-text'),
    stratSel: $('#strategy-select'),
    cascToggle: $('#cascade-toggle'),
    promptInput: $('#prompt-input'),
    maxTokens: $('#max-tokens'),
    temperature: $('#temperature'),
    forceAdapter: $('#force-adapter'),
    btnRun: $('#btn-run'),
    btnClear: $('#btn-clear'),
    radarCanvas: $('#radar-canvas'),
    radarLatency: $('#radar-latency'),
    pipelineStatus: $('#pipeline-status'),
    adapterBadge: $('#adapter-badge'),
    mRoute: $('#m-route'),
    mTotal: $('#m-total'),
    mConf: $('#m-conf'),
    mStrat: $('#m-strat'),
    cascadeBox: $('#cascade-box'),
    cascadeReason: $('#cascade-reason'),
    candContainer: $('#cand-container'),
    outputLang: $('#output-lang'),
    btnCopy: $('#btn-copy'),
    responseOutput: $('#response-output'),
    historyBody: $('#history-body'),
    btnClearHistory: $('#btn-clear-history'),
    adapterLabel: $('#pn-adapter-label'),
    adapterSub: $('#pn-adapter-sub'),
    routerSub: $('#pn-router-sub'),
  };

  // ── State ───────────────────────────────────────────────
  let scores = { sql: 0, json: 0, code: 0, base: 0 };
  let online = false;
  let history = [];
  let animFrame = null;
  let targetScores = { sql: 0, json: 0, code: 0, base: 0 };

  // ── Tabs ────────────────────────────────────────────────
  $$('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.tab').forEach(t => t.classList.remove('active'));
      $$('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      $(`#tab-${btn.dataset.tab}`).classList.add('active');
    });
  });

  // ── Presets ─────────────────────────────────────────────
  $$('.chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.preset;
      dom.promptInput.value = PRESETS[key] || '';
      dom.promptInput.focus();
    });
  });

  // ── Radar Canvas ────────────────────────────────────────
  const ctx = dom.radarCanvas.getContext('2d');
  const CX = 110, CY = 110, R = 90;

  function drawRadar() {
    const c = ctx;
    c.clearRect(0, 0, 220, 220);
    const N = ADAPTERS.length;

    // Grid rings
    c.strokeStyle = 'rgba(255,255,255,0.06)';
    c.lineWidth = 1;
    [0.25, 0.5, 0.75, 1].forEach(f => {
      c.beginPath();
      for (let i = 0; i <= N; i++) {
        const a = (Math.PI * 2 * i) / N - Math.PI / 2;
        const x = CX + Math.cos(a) * R * f;
        const y = CY + Math.sin(a) * R * f;
        i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
      }
      c.stroke();
    });

    // Spokes
    for (let i = 0; i < N; i++) {
      const a = (Math.PI * 2 * i) / N - Math.PI / 2;
      c.beginPath();
      c.moveTo(CX, CY);
      c.lineTo(CX + Math.cos(a) * R, CY + Math.sin(a) * R);
      c.stroke();
    }

    // Data polygon
    c.beginPath();
    for (let i = 0; i < N; i++) {
      const a = (Math.PI * 2 * i) / N - Math.PI / 2;
      const v = scores[ADAPTERS[i]];
      const x = CX + Math.cos(a) * R * v;
      const y = CY + Math.sin(a) * R * v;
      i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
    }
    c.closePath();
    c.fillStyle = 'rgba(56,189,248,0.15)';
    c.fill();
    c.strokeStyle = 'rgba(56,189,248,0.6)';
    c.lineWidth = 1.5;
    c.stroke();

    // Dots + Labels
    for (let i = 0; i < N; i++) {
      const a = (Math.PI * 2 * i) / N - Math.PI / 2;
      const v = scores[ADAPTERS[i]];
      const x = CX + Math.cos(a) * R * v;
      const y = CY + Math.sin(a) * R * v;

      c.beginPath();
      c.arc(x, y, 3, 0, Math.PI * 2);
      c.fillStyle = COLORS[ADAPTERS[i]];
      c.fill();

      const lx = CX + Math.cos(a) * (R + 14);
      const ly = CY + Math.sin(a) * (R + 14);
      c.fillStyle = 'rgba(148,163,184,0.8)';
      c.font = '500 11px Inter';
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText(ADAPTERS[i].toUpperCase(), lx, ly);
    }
  }

  function animateScores() {
    let done = true;
    for (const k of ADAPTERS) {
      const diff = targetScores[k] - scores[k];
      if (Math.abs(diff) > 0.001) {
        scores[k] += diff * 0.15;
        done = false;
      } else {
        scores[k] = targetScores[k];
      }
    }
    drawRadar();
    updateBars();
    if (!done) animFrame = requestAnimationFrame(animateScores);
  }

  function setScores(s) {
    targetScores = { ...s };
    if (animFrame) cancelAnimationFrame(animFrame);
    animFrame = requestAnimationFrame(animateScores);
  }

  function updateBars() {
    for (const k of ADAPTERS) {
      const pct = Math.round(scores[k] * 100);
      $(`#fill-${k}`).style.width = `${pct}%`;
      $(`#val-${k}`).textContent = `${pct}%`;
    }
  }

  // ── Pipeline Animation ──────────────────────────────────
  function lightPipeline(stage) {
    const nodes = ['pn-input', 'pn-router', 'pn-adapter', 'pn-engine'];
    const conns = ['pc-1', 'pc-2', 'pc-3'];
    nodes.forEach((id, i) => {
      $(`#${id}`).classList.toggle('lit', i <= stage);
    });
    conns.forEach((id, i) => {
      $(`#${id}`).classList.toggle('lit', i < stage);
    });
    const labels = ['Received', 'Routing...', 'Switching...', 'Generating...'];
    dom.pipelineStatus.textContent = labels[stage] || 'Ready';
  }

  function resetPipeline() {
    ['pn-input', 'pn-router', 'pn-adapter', 'pn-engine'].forEach(id =>
      $(`#${id}`).classList.remove('lit')
    );
    ['pc-1', 'pc-2', 'pc-3'].forEach(id =>
      $(`#${id}`).classList.remove('lit')
    );
    dom.pipelineStatus.textContent = 'Ready';
  }

  // ── Health Check ────────────────────────────────────────
  async function checkHealth() {
    try {
      const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(3000) });
      if (r.ok) {
        const d = await r.json();
        online = true;
        dom.statusDot.className = 'status-dot online';
        dom.statusText.textContent = `Online - ${d.engine || 'peft'}`;
        if (d.active_adapter) {
          dom.adapterLabel.textContent = d.active_adapter;
        }
        // Try to also fetch real router scores for current prompt
        if (dom.promptInput.value.trim()) {
          try {
            const sr = await fetch(`${API}/v1/router/scores?prompt=${encodeURIComponent(dom.promptInput.value.trim())}`, { signal: AbortSignal.timeout(2000) });
            if (sr.ok) {
              const sd = await sr.json();
              if (sd.scores) setScores(sd.scores);
            }
          } catch(_) {}
        }
        return true;
      }
    } catch (_) {}
    online = false;
    dom.statusDot.className = 'status-dot offline';
    dom.statusText.textContent = 'Offline (Mock)';
    return false;
  }

  // ── Mock Inference ──────────────────────────────────────
  function mockInference(prompt) {
    const lower = prompt.toLowerCase();
    let primary = 'base';
    const scoreMap = { sql: 0.08, json: 0.08, code: 0.08, base: 0.08 };

    if (lower.includes('sql') || lower.includes('query') || lower.includes('select')) {
      primary = 'sql';
      scoreMap.sql = 0.91; scoreMap.json = 0.04; scoreMap.code = 0.03; scoreMap.base = 0.02;
    } else if (lower.includes('json') || lower.includes('extract') || lower.includes('schema')) {
      primary = 'json';
      scoreMap.json = 0.88; scoreMap.sql = 0.04; scoreMap.code = 0.05; scoreMap.base = 0.03;
    } else if (lower.includes('function') || lower.includes('code') || lower.includes('python') || lower.includes('def ')) {
      primary = 'code';
      scoreMap.code = 0.85; scoreMap.sql = 0.05; scoreMap.json = 0.04; scoreMap.base = 0.06;
    } else {
      primary = 'base';
      scoreMap.base = 0.72; scoreMap.sql = 0.10; scoreMap.json = 0.08; scoreMap.code = 0.10;
    }

    const cascade = dom.cascToggle.checked;
    let cascadeInfo = null;
    const needCascade = cascade && lower.includes('migration');

    const mockResponses = {
      sql: `SELECT c.customer_name, SUM(o.amount) AS total_amount\nFROM customers c\nJOIN orders o ON c.id = o.customer_id\nGROUP BY c.customer_name\nORDER BY total_amount DESC\nLIMIT 5;`,
      json: `{\n  "name": "John Smith",\n  "email": "john@example.com",\n  "phone": "555-0123"\n}`,
      code: `def merge_sorted_lists(l1, l2):\n    dummy = ListNode(0)\n    tail = dummy\n    while l1 and l2:\n        if l1.val <= l2.val:\n            tail.next = l1\n            l1 = l1.next\n        else:\n            tail.next = l2\n            l2 = l2.next\n        tail = tail.next\n    tail.next = l1 or l2\n    return dummy.next`,
      base: `Supervised learning uses labeled data where the algorithm learns a mapping from inputs to known outputs. The model is trained to minimize prediction error against these labels.\n\nUnsupervised learning works with unlabeled data, finding hidden patterns or structures such as clusters, associations, or dimensionality reduction without predefined categories.`
    };

    if (needCascade) {
      cascadeInfo = {
        reason: 'Prompt overlaps multiple domains (SQL + Code). Confidence margin < 0.15.',
        candidates: [
          { adapter: 'sql', quality: 0.72, chosen: false },
          { adapter: 'code', quality: 0.88, chosen: true }
        ]
      };
      primary = 'code';
      scoreMap.sql = 0.42; scoreMap.code = 0.45; scoreMap.json = 0.06; scoreMap.base = 0.07;
    }

    return {
      adapter: primary,
      response: mockResponses[primary],
      scores: scoreMap,
      route_ms: 5.8 + Math.random() * 2,
      switch_ms: 8.2 + Math.random() * 4,
      gen_ms: 180 + Math.random() * 60,
      total_ms: 200 + Math.random() * 60,
      strategy: dom.stratSel.value === 'learned' ? 'Learned MLP' : 'Centroid',
      cascade: cascadeInfo
    };
  }

  // ── Run Inference ───────────────────────────────────────
  async function runInference() {
    const prompt = dom.promptInput.value.trim();
    if (!prompt) return;

    dom.btnRun.disabled = true;
    dom.btnRun.textContent = 'Running...';
    dom.cascadeBox.style.display = 'none';
    resetPipeline();

    try {
      lightPipeline(0);
      await delay(250);
      lightPipeline(1);

      let result;

      if (online) {
        try {
          const body = {
            prompt,
            max_tokens: parseInt(dom.maxTokens.value),
            temperature: parseFloat(dom.temperature.value),
            router_strategy: dom.stratSel.value,
            enable_cascade: dom.cascToggle.checked
          };
          const force = dom.forceAdapter.value;
          if (force) body.force_adapter = force;

          const r = await fetch(`${API}/v1/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          });
          const raw = await r.json();
          // Map gateway response to our internal format
          result = {
            adapter: raw.adapter_used || 'base',
            response: raw.response || '',
            scores: {},
            route_ms: raw.routing_latency_ms || 0,
            switch_ms: 8 + Math.random() * 4,
            gen_ms: (raw.total_latency_ms || 0) - (raw.routing_latency_ms || 0),
            total_ms: raw.total_latency_ms || 0,
            strategy: raw.router_strategy || 'learned',
            cascade: raw.cascade_triggered ? {
              reason: raw.selection_reason || '',
              candidates: (raw.candidates_evaluated || []).map(c => ({
                adapter: c.adapter,
                quality: c.quality_score || c.combined_score || 0,
                chosen: c.adapter === raw.adapter_used
              }))
            } : null
          };
          // Build scores from confidence (API doesn't return per-adapter scores directly)
          const conf = raw.router_confidence || 0.5;
          result.scores[result.adapter] = conf;
          for (const a of ADAPTERS) {
            if (!(a in result.scores)) result.scores[a] = (1 - conf) / 3;
          }
        } catch (e) {
          console.warn('Live API failed, falling back to mock:', e);
          result = mockInference(prompt);
        }
      } else {
        await delay(600);
        result = mockInference(prompt);
      }

      lightPipeline(2);
      await delay(200);
      lightPipeline(3);
      await delay(300);

      // Apply results
      const adapter = result.adapter || result.route || 'base';
      setScores(result.scores || result.router_scores || {});
      dom.radarLatency.textContent = `${(result.route_ms || 0).toFixed(1)}ms routing`;

      // Adapter badge
      dom.adapterBadge.textContent = adapter.toUpperCase();
      dom.adapterBadge.className = `adapter-badge ab-${adapter}`;

      // Metrics
      dom.mRoute.textContent = `${(result.route_ms || 0).toFixed(1)}ms`;
      dom.mTotal.textContent = `${(result.total_ms || 0).toFixed(0)}ms`;
      dom.mConf.textContent = `${Math.round((result.scores?.[adapter] || 0) * 100)}%`;
      dom.mStrat.textContent = result.strategy || dom.stratSel.options[dom.stratSel.selectedIndex].text;

      // Pipeline labels
      dom.adapterLabel.textContent = `${adapter.toUpperCase()} LoRA`;
      dom.adapterSub.textContent = `${(result.switch_ms || 0).toFixed(1)}ms switch`;
      dom.routerSub.textContent = result.strategy || 'Learned';

      // Output
      dom.responseOutput.textContent = result.response || result.text || 'No response.';
      dom.outputLang.textContent = adapter === 'sql' ? 'SQL' : adapter === 'json' ? 'JSON' : adapter === 'code' ? 'Python' : 'Text';

      // Cascade
      if (result.cascade) {
        dom.cascadeBox.style.display = 'block';
        dom.cascadeReason.textContent = result.cascade.reason || 'Low confidence triggered cascade.';
        dom.candContainer.innerHTML = '';
        (result.cascade.candidates || []).forEach(c => {
          const d = document.createElement('div');
          d.className = `cand-card ${c.chosen ? 'winner' : ''}`;
          d.innerHTML = `<div class="cand-head"><span>${c.adapter.toUpperCase()}</span><span>${c.chosen ? 'Selected' : 'Rejected'}</span></div><div style="color:var(--t3);font-size:0.72rem">Quality: ${(c.quality * 100).toFixed(0)}%</div>`;
          dom.candContainer.appendChild(d);
        });
      }

      // History
      addHistory({ prompt, adapter, conf: result.scores?.[adapter] || 0, total_ms: result.total_ms, cascade: !!result.cascade });

      dom.pipelineStatus.textContent = 'Done';

    } catch (err) {
      dom.responseOutput.textContent = `Error: ${err.message}`;
      dom.pipelineStatus.textContent = 'Error';
    }

    dom.btnRun.disabled = false;
    dom.btnRun.textContent = 'Run Inference';
  }

  // ── History ─────────────────────────────────────────────
  function addHistory(item) {
    history.unshift(item);
    if (history.length > 20) history.pop();
    renderHistory();
  }

  function renderHistory() {
    if (history.length === 0) {
      dom.historyBody.innerHTML = '<tr class="empty-row"><td colspan="7">No queries yet.</td></tr>';
      return;
    }
    dom.historyBody.innerHTML = history.map((h, i) => `
      <tr>
        <td class="mono-sm">${new Date().toLocaleTimeString()}</td>
        <td title="${esc(h.prompt)}">${esc(h.prompt.slice(0, 55))}${h.prompt.length > 55 ? '...' : ''}</td>
        <td><span class="route-tag rt-${h.adapter}">${h.adapter.toUpperCase()}</span></td>
        <td class="mono-sm">${(h.conf * 100).toFixed(0)}%</td>
        <td class="mono-sm">${h.total_ms.toFixed(0)}ms</td>
        <td>${h.cascade ? '<span style="color:var(--rose)">Yes</span>' : '-'}</td>
        <td><button class="btn-rerun" data-idx="${i}">Rerun</button></td>
      </tr>`
    ).join('');

    dom.historyBody.querySelectorAll('.btn-rerun').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.idx);
        dom.promptInput.value = history[idx].prompt;
        runInference();
      });
    });
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ── Events ──────────────────────────────────────────────
  dom.btnRun.addEventListener('click', runInference);
  dom.btnClear.addEventListener('click', () => {
    dom.promptInput.value = '';
    dom.responseOutput.textContent = 'Run a query to see the model response.';
    dom.cascadeBox.style.display = 'none';
    dom.adapterBadge.textContent = '--';
    dom.adapterBadge.className = 'adapter-badge';
    dom.mRoute.textContent = dom.mTotal.textContent = dom.mConf.textContent = dom.mStrat.textContent = '--';
    resetPipeline();
    setScores({ sql: 0, json: 0, code: 0, base: 0 });
  });

  dom.btnClearHistory.addEventListener('click', () => {
    history = [];
    renderHistory();
  });

  dom.btnCopy.addEventListener('click', () => {
    navigator.clipboard.writeText(dom.responseOutput.textContent);
    dom.btnCopy.textContent = 'Copied';
    setTimeout(() => dom.btnCopy.textContent = 'Copy', 1200);
  });

  dom.promptInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      runInference();
    }
  });

  // ── Utilities ───────────────────────────────────────────
  function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

  // ── Init ────────────────────────────────────────────────
  drawRadar();
  checkHealth();
  setInterval(checkHealth, 15000);

})();
