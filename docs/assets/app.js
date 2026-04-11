// research-everything viz — vanilla JS, no build step.
// Reads ./data.json, renders DAG + leaderboard + timeline + orbit detail.
// Hash router: #/ → index, #/orbit/<name> → detail.

const app = document.getElementById('app');

(async function main() {
  let data;
  try {
    const res = await fetch('./data.json', { cache: 'no-store' });
    data = await res.json();
  } catch (e) {
    app.innerHTML = `<p class="empty">No data.json found. Run <code>/publish</code>.</p>`;
    return;
  }
  document.title = data.campaign?.title || 'Campaign';
  cytoscape.use(cytoscapeDagre);
  window.addEventListener('hashchange', () => render(data));
  render(data);
})();

function render(data) {
  // Single-page: always show the index with inline detail panel on node click
  app.innerHTML = renderIndex(data);
  mountDag(data);
}

function renderIndex(data) {
  const c = data.campaign || {};
  const orbits = data.orbits || [];
  const counts = tallyStatus(orbits);
  const best = c.best;
  const stats = [
    best ? `<div class="stat"><small>best metric</small><span class="value">${fmt(best.metric)}</span></div>` : '',
    `<div class="stat"><small>eval</small><span class="value">${esc(c.eval_version || '—')}</span></div>`,
    `<div class="stat"><small>orbits</small><span class="value">${orbits.length}</span></div>`,
  ].filter(Boolean).join('');

  const sorted = [...orbits].sort((a, b) => metricRank(a, c) - metricRank(b, c));

  let heroHtml = '';
  if (c.teaser_image) {
    heroHtml += `<img src="${esc(c.teaser_image)}" class="teaser-hero" style="max-width:100%;border-radius:8px;margin:16px 0;" />`;
  }
  if (c.background) {
    heroHtml += `<details class="background-section" style="margin:16px 0;">
        <summary style="cursor:pointer;font-weight:bold;">Research Background</summary>
        <div style="padding:8px 0;">${renderMarkdown(c.background)}</div>
    </details>`;
  }

  return `
    <header class="hero">
      <small class="meta">campaign</small>
      <h1>${esc(c.title || 'Untitled')}</h1>
      ${heroHtml}
      <div class="problem">${renderMarkdown(c.problem || '')}</div>
      <div class="stats">${stats}</div>
    </header>
    ${c.metric_description || c.eval_methodology ? renderMethodology(c) : ''}
    <h2>Research map</h2>
    ${orbits.length ? `<div id="dag"></div>` : `<p class="empty">0 orbits yet.</p>`}
    <h2>Leaderboard</h2>
    ${orbits.length ? renderLeaderboard(sorted) : ''}
    ${data.campaign?.timeline?.length ? `<h2>Timeline</h2>${renderTimeline(data.campaign.timeline)}` : ''}
    ${c.references?.length ? renderReferences(c.references) : ''}
    ${renderFooter(counts)}
  `;
}

function renderLeaderboard(orbits) {
  const rows = orbits.map(o => `
    <tr class="${o.status || ''}" onclick="location.hash='#/orbit/${encodeURIComponent(o.name)}'">
      <td>${esc(o.name)}</td>
      <td><small>${esc(o.strategy || '')}</small></td>
      <td><small>${esc(o.status || '')}</small></td>
      <td class="metric">${fmt(o.metric)}</td>
    </tr>
  `).join('');
  return `<table class="leaderboard">
    <thead><tr><th>orbit</th><th>strategy</th><th>status</th><th style="text-align:right">metric</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderMethodology(c) {
  const ev = c.eval_methodology || {};
  const stages = (ev.stages || []).map(s => `<li>${esc(s)}</li>`).join('');
  return `
    <h2>Evaluation</h2>
    <div class="methodology">
      ${c.metric_description ? `<p class="metric-desc"><strong>Metric:</strong> ${esc(c.metric_description)}</p>` : ''}
      ${ev.summary ? `<p>${esc(ev.summary)}</p>` : ''}
      ${stages ? `<ul class="stages">${stages}</ul>` : ''}
      ${ev.baseline ? `<p class="baseline"><strong>Baseline:</strong> ${esc(ev.baseline)}</p>` : ''}
      ${ev.params ? `<p class="eval-params"><small class="mono">${esc(ev.params)}</small></p>` : ''}
    </div>
  `;
}

function renderReferences(refs) {
  const items = refs.map(r => `
    <li>
      <span class="ref-cite">${esc(r.cite || '')}</span>
      ${r.note ? `<span class="ref-note"> — ${esc(r.note)}</span>` : ''}
    </li>
  `).join('');
  return `<h2>References</h2><ul class="references">${items}</ul>`;
}

function renderIssueComments(comments) {
  const items = comments.map(c => {
    const imgs = (c.images || []).map(url =>
      `<img src="${esc(url)}" alt="" loading="lazy">`
    ).join('');
    return `
      <div class="issue-comment">
        <div class="comment-meta">
          <span class="comment-author">${esc(c.author || '')}</span>
          <span class="comment-date">${esc(c.date || '')}</span>
        </div>
        <div class="comment-body">${renderMarkdown(c.body || c.excerpt || '')}</div>
        ${imgs ? `<div class="comment-images">${imgs}</div>` : ''}
      </div>`;
  }).join('');
  return `
    <div class="issue-comments">
      <h4>Issue activity</h4>
      ${items}
    </div>`;
}

function renderMarkdown(text) {
  // Use marked.js for full GitHub-flavored markdown rendering.
  // Strip markdown images first (rendered separately in the images section).
  let clean = text.replace(/!\[[^\]]*\]\([^)]+\)/g, '');
  try {
    return marked.parse(clean, { breaks: true, gfm: true });
  } catch (e) {
    // Fallback: basic escaping
    return esc(clean).replace(/\n/g, '<br>');
  }
}

function renderTimeline(entries) {
  const items = entries.map(e => {
    // Format date nicely
    const raw = e.at || '';
    const date = raw.length >= 10 ? raw.slice(0, 10) : raw;
    // Detect entry type from content
    const body = e.text || '';
    const isDebate = body.includes('🗳️ Debate') || body.includes('debate-agent');
    const isMilestone = body.includes('## Round') || body.includes('## Milestone') || body.includes('Leaderboard');
    const isOrbitComplete = body.includes('complete') && body.includes('orbit/');
    const tag = isDebate ? 'debate' : isMilestone ? 'milestone' : isOrbitComplete ? 'result' : 'update';
    const imgs = (e.images || []).map(url =>
      `<img src="${esc(url)}" loading="lazy" class="timeline-image" style="max-width:100%;margin:8px 0;border-radius:4px;" />`
    ).join('');
    return `
      <div class="tl-entry" data-tag="${tag}">
        <div class="tl-header">
          <span class="tl-date">${esc(date)}</span>
          <span class="tl-tag tl-tag-${tag}">${tag}</span>
        </div>
        <div class="tl-body">${renderMarkdown(body)}</div>
        ${imgs ? `<div class="tl-images">${imgs}</div>` : ''}
      </div>`;
  }).join('');
  return `<div class="timeline-log">${items}</div>`;
}

function renderOrbit(data, o) {
  const figs = (o.figures || []).map(f => `
    <figure>
      <img src="${esc(f.url || f.path)}" alt="">
      <figcaption>${esc(f.caption || '')}${f.author ? ` — ${esc(f.author)}` : ''}</figcaption>
    </figure>
  `).join('');
  const links = (o.links || []).map(l => `
    <li><a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.title || l.url)}</a>${l.author ? ` <small>— ${esc(l.author)}</small>` : ''}</li>
  `).join('');
  const parents = (o.parents || []).map(p =>
    `<a href="#/orbit/${encodeURIComponent(p)}">${esc(p)}</a>`
  ).join(', ') || '—';

  return `
    <p class="back"><a href="#/">← back</a></p>
    <small class="meta">orbit</small>
    <h1>${esc(o.name)}</h1>
    <p class="problem">${esc(o.strategy || '')}</p>
    <dl class="frontmatter">
      <dt>status</dt><dd>${esc(o.status || '—')}</dd>
      <dt>metric</dt><dd class="mono">${fmt(o.metric)}</dd>
      <dt>parents</dt><dd>${parents}</dd>
      ${o.issue ? `<dt>issue</dt><dd><a href="${esc(issueUrl(data, o.issue))}" target="_blank">#${o.issue}</a></dd>` : ''}
      ${o.log_url ? `<dt>log</dt><dd><a href="${esc(o.log_url)}" target="_blank">log.md</a></dd>` : ''}
    </dl>
    ${figs ? `<h2>Figures</h2><div class="figures">${figs}</div>` : ''}
    ${links ? `<h2>References</h2><ul class="links">${links}</ul>` : ''}
  `;
}

function mountDag(data) {
  const el = document.getElementById('dag');
  if (!el) return;
  const orbits = data.orbits || [];

  // Compute metric range for node sizing
  const metrics = orbits.map(o => o.metric).filter(m => m != null && typeof m === 'number');
  const metricMin = Math.min(...metrics), metricMax = Math.max(...metrics);
  const dir = data.campaign?.best?.direction === 'max' ? 'max' : 'min';

  function nodeSize(o) {
    if (o.metric == null) return 20;
    if (metricMin === metricMax) return 24;
    const norm = dir === 'min'
      ? 1 - (o.metric - metricMin) / (metricMax - metricMin)
      : (o.metric - metricMin) / (metricMax - metricMin);
    return 18 + norm * 20; // 18..38px
  }

  // Connected Papers-style: node fill = metric quality gradient
  // Compute a 0..1 "quality" score per node, store as data for mapData()
  function nodeQuality(o) {
    if (o.metric == null) return 0;
    if (metricMin === metricMax) return 0.5;
    return dir === 'min'
      ? 1 - (o.metric - metricMin) / (metricMax - metricMin)
      : (o.metric - metricMin) / (metricMax - metricMin);
  }

  function truncLabel(s, n) { return s && s.length > n ? s.slice(0, n) + '…' : s || ''; }

  const nodes = orbits.map(o => ({
    data: {
      id: o.name,
      label: truncLabel(o.strategy || o.name, 24),
      shortName: o.name,
      status: o.status || 'exploring',
      metric: o.metric,
      strategy: o.strategy || '',
      size: nodeSize(o),
      quality: nodeQuality(o),  // 0..1, used for fill gradient
    },
  }));
  const edges = [];
  for (const o of orbits) {
    for (const p of (o.parents || [])) {
      if (orbits.find(x => x.name === p)) edges.push({ data: { source: p, target: o.name } });
    }
  }
  const cs = getComputedStyle(document.body);
  const fg = cs.getPropertyValue('--foreground').trim();
  const bg = cs.getPropertyValue('--background').trim();
  const muted = cs.getPropertyValue('--muted').trim();
  const border = cs.getPropertyValue('--border').trim();
  const destructive = cs.getPropertyValue('--destructive').trim();

  const cy = cytoscape({
    container: el,
    elements: [...nodes, ...edges],
    wheelSensitivity: 0.2,
    pixelRatio: 'auto',
    layout: {
      name: 'dagre', rankDir: 'TB',
      nodeSep: 80, rankSep: 100, edgeSep: 25,
      ranker: 'tight-tree',
    },
    style: [
      // Connected Papers-inspired: filled nodes, thin border, quality → fill intensity
      {
        selector: 'node',
        style: {
          'background-color': 'mapData(quality, 0, 1, ' + bg + ', ' + fg + ')',
          'background-opacity': 0.18,
          'border-width': 0.75,
          'border-color': fg,
          'border-opacity': 0.3,
          'label': 'data(label)',
          'font-family': "'JetBrains Mono', ui-monospace, monospace",
          'font-size': 10,
          'font-weight': 400,
          'color': fg,
          'text-valign': 'bottom',
          'text-halign': 'center',
          'text-margin-y': 8,
          'text-max-width': '120px',
          'text-wrap': 'ellipsis',
          'text-opacity': 0.8,
          'width': 'data(size)', 'height': 'data(size)',
          'shape': 'ellipse',
          'transition-property': 'border-width, width, height, opacity, background-opacity, border-opacity',
          'transition-duration': 150,
        },
      },
      // Better metric → more saturated fill
      { selector: 'node[quality > 0.3]', style: { 'background-opacity': 0.28, 'border-opacity': 0.45 } },
      { selector: 'node[quality > 0.6]', style: { 'background-opacity': 0.38, 'border-opacity': 0.55, 'font-weight': 600 } },
      { selector: 'node[quality > 0.85]', style: { 'background-opacity': 0.48, 'border-opacity': 0.7, 'border-width': 1.25 } },
      { selector: 'node[status = "dead-end"]', style: {
          'opacity': 0.35,
          'background-opacity': 0.08,
          'border-style': 'dashed',
          'border-opacity': 0.2,
          'font-size': 9,
          'text-opacity': 0.5,
        }
      },
      { selector: 'node[status = "malformed"]', style: {
          'opacity': 0.2,
          'background-opacity': 0.05,
          'border-style': 'dotted',
          'font-size': 9,
        }
      },
      { selector: 'node[status = "promising"]', style: {
          'border-width': 1.25,
          'border-opacity': 0.6,
        }
      },
      { selector: 'node[status = "exploring"]', style: {
          'border-style': 'dotted',
          'border-width': 0.75,
        }
      },
      { selector: 'node[status = "winner"], node[status = "graduated"]', style: {
          'background-color': destructive,
          'background-opacity': 0.65,
          'border-color': destructive,
          'border-width': 2,
          'border-opacity': 1,
          'color': fg,
          'font-weight': 600,
          'font-size': 11,
          'text-opacity': 1,
        }
      },
      { selector: 'node:active', style: { 'overlay-opacity': 0.04, 'overlay-color': fg } },
      // Edges: thin, quiet, letting nodes be the star (Connected Papers style)
      {
        selector: 'edge',
        style: {
          'width': 1,
          'line-color': border,
          'curve-style': 'bezier',
          'control-point-step-size': 55,
          'target-arrow-shape': 'none',
          'opacity': 0.5,
        },
      },
      ...multiParentEdgeStyles(orbits, fg),
      ...edgeConnectednessStyles(orbits),
      // Selection classes — applied on node tap
      { selector: '.dimmed', style: { 'opacity': 0.12, 'transition-duration': 200 } },
      { selector: '.highlighted-edge', style: {
          'line-color': destructive, 'opacity': 0.8, 'width': 2.5,
          'line-style': 'dashed', 'line-dash-pattern': [8, 4],
          'transition-duration': 200,
        }
      },
      { selector: '.ancestor-node', style: {
          'border-color': destructive, 'border-opacity': 0.6, 'border-width': 2,
          'transition-duration': 200,
        }
      },
    ],
  });

  // Highlight winner ancestry path with animated dashes (flow animation)
  const ancestry = winnerAncestryPath(orbits);
  if (ancestry.size > 1) {
    // Edges on the winner's ancestry path
    cy.edges().forEach(e => {
      const src = e.data('source'), tgt = e.data('target');
      if (ancestry.has(src) && ancestry.has(tgt)) {
        e.style({
          'line-color': destructive,
          'opacity': 0.7,
          'line-style': 'dashed',
          'line-dash-pattern': [8, 4],
        });
      }
    });
    // Animate the dash offset to create flow effect
    let offset = 0;
    function animateFlow() {
      offset = (offset + 0.5) % 12;
      cy.edges().forEach(e => {
        const src = e.data('source'), tgt = e.data('target');
        if (ancestry.has(src) && ancestry.has(tgt)) {
          e.style('line-dash-offset', -offset);
        }
      });
      requestAnimationFrame(animateFlow);
    }
    requestAnimationFrame(animateFlow);

    // Ancestry nodes get a subtle ring
    cy.nodes().forEach(n => {
      if (ancestry.has(n.id()) && n.data('status') !== 'winner') {
        n.style({
          'border-color': destructive,
          'border-width': 1.75,
          'border-opacity': 0.45,
        });
      }
    });
  }

  // Tooltip
  const tip = document.createElement('div');
  tip.className = 'dag-tooltip';
  el.appendChild(tip);

  cy.on('mouseover', 'node', (evt) => {
    const n = evt.target;
    const d = n.data();
    const pos = n.renderedPosition();
    const metricStr = d.metric != null ? fmt(d.metric) : '—';
    const parents = (data.orbits.find(o => o.name === d.id) || {}).parents || [];
    tip.innerHTML = `<strong>${esc(d.id)}</strong>
      <br><span class="tip-strategy">${esc(d.strategy)}</span>
      <br><span class="tip-metric">${d.status} · ${metricStr}</span>
      ${parents.length ? '<br><span class="tip-parents">← ' + parents.map(esc).join(', ') + '</span>' : ''}`;
    tip.style.left = pos.x + 'px';
    tip.style.top = (pos.y - 12) + 'px';
    tip.classList.add('visible');
    // Connected Papers hover: node grows, border intensifies, subtle glow
    n.style({
      'border-width': 2,
      'border-opacity': 1,
      'background-opacity': Math.min(0.7, (parseFloat(n.style('background-opacity')) || 0.18) + 0.25),
      'z-index': 999,
      'width': n.data('size') * 1.2,
      'height': n.data('size') * 1.2,
    });
  });

  cy.on('mouseout', 'node', (evt) => {
    tip.classList.remove('visible');
    evt.target.removeStyle('border-width');
    evt.target.removeStyle('border-opacity');
    evt.target.removeStyle('background-opacity');
    evt.target.removeStyle('z-index');
    evt.target.removeStyle('width');
    evt.target.removeStyle('height');
  });

  // Fade in
  el.style.opacity = '0';
  cy.ready(() => { el.style.transition = 'opacity 350ms ease'; el.style.opacity = '1'; });

  // Click node → highlight ancestry + slide in side panel
  let selectedNode = null;

  function selectOrbit(name) {
    if (!name) { deselectNode(); return; }
    if (selectedNode === name) { deselectNode(); return; } // toggle off
    selectedNode = name;
    const orbit = (data.orbits || []).find(o => o.name === name);

    // Reset + highlight ancestry
    cy.elements().removeClass('dimmed highlighted-edge ancestor-node');
    const ancestors = traceAncestry(name, data.orbits);
    cy.nodes().forEach(n => {
      if (!ancestors.has(n.id())) n.addClass('dimmed');
      else n.addClass('ancestor-node');
    });
    cy.edges().forEach(e => {
      if (ancestors.has(e.data('source')) && ancestors.has(e.data('target'))) e.addClass('highlighted-edge');
      else e.addClass('dimmed');
    });

    // Slide in side panel
    showDetailPanel(data, orbit, el, selectOrbit);
  }

  function deselectNode() {
    selectedNode = null;
    cy.elements().removeClass('dimmed highlighted-edge ancestor-node');
    hideDetailPanel();
  }

  cy.on('tap', 'node', (evt) => {
    tip.classList.remove('visible');
    selectOrbit(evt.target.id());
  });

  cy.on('tap', function(evt) {
    if (evt.target === cy) deselectNode();
  });
}

function traceAncestry(id, orbits) {
  // BFS backward from id through parents; return set of all ancestor names + self
  const byName = Object.fromEntries(orbits.map(o => [o.name, o]));
  const visited = new Set();
  const queue = [id];
  while (queue.length) {
    const cur = queue.shift();
    if (visited.has(cur)) continue;
    visited.add(cur);
    for (const p of (byName[cur]?.parents || [])) {
      if (byName[p]) queue.push(p);
    }
  }
  return visited;
}

function buildAncestryPath(orbitName, orbits) {
  // Build ordered path(s) from roots to this orbit
  const byName = Object.fromEntries(orbits.map(o => [o.name, o]));
  const paths = [];
  function walk(name, path) {
    const o = byName[name];
    if (!o) return;
    const parents = o.parents || [];
    if (parents.length === 0) { paths.push([name, ...path]); return; }
    for (const p of parents) walk(p, [name, ...path]);
  }
  walk(orbitName, []);
  return paths;
}

function showDetailPanel(data, orbit, dagEl, onSelectOrbit) {
  // Reuse existing panel or create new
  let panel = document.body.querySelector('.detail-panel');
  if (!panel) {
    panel = document.createElement('div');
    panel.className = 'detail-panel';
    document.body.appendChild(panel);
    // Trigger open animation on next frame
    requestAnimationFrame(() => requestAnimationFrame(() => panel.classList.add('open')));
  } else {
    panel.classList.add('open');
    panel.classList.remove('closing');
  }
  if (!orbit) return;

  const parents = (orbit.parents || []).map(p =>
    `<a href="javascript:void(0)" class="parent-link" data-orbit="${esc(p)}">${esc(p)}</a>`
  ).join(', ') || 'none (root)';
  const figs = (orbit.figures || []).map(f =>
    `<img src="${esc(f.url || f.path)}" alt="${esc(f.caption || '')}" title="${esc(f.caption || '')}">`
  ).join('');
  const links = (orbit.links || []).map(l =>
    `<a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.title || l.url)}</a>`
  ).join('');
  const repo = data.campaign?.repo || '';
  const issueUrl = orbit.issue && repo ? `https://github.com/${repo}/issues/${orbit.issue}` : '';

  // Build ancestry path visualization
  const ancestryPaths = buildAncestryPath(orbit.name, data.orbits || []);
  let ancestryHtml = '';
  if (ancestryPaths.length > 0 && ancestryPaths[0].length > 1) {
    const pathLines = ancestryPaths.slice(0, 3).map(path =>
      path.map((name, i) => {
        const isCurrent = name === orbit.name;
        return `<span class="step ${isCurrent ? 'current' : ''}"><a href="javascript:void(0)" class="parent-link" data-orbit="${esc(name)}">${esc(name)}</a></span>${i < path.length - 1 ? '<span class="arrow">→</span>' : ''}`;
      }).join('')
    ).join('<br>');
    ancestryHtml = `
      <div class="detail-ancestry">
        <h4>Lineage</h4>
        <div class="path">${pathLines}</div>
      </div>`;
  }

  panel.innerHTML = `
    <button class="detail-close" title="Close">&times;</button>
    <div class="detail-header">
      <small class="meta">orbit</small>
      <h3>${esc(orbit.name)}</h3>
    </div>
    <p class="detail-strategy">${esc(orbit.strategy || '')}</p>
    <div class="detail-grid">
      <span class="detail-label">status</span><span>${esc(orbit.status || '—')}</span>
      <span class="detail-label">metric</span><span class="mono">${fmt(orbit.metric)}</span>
      <span class="detail-label">parents</span><span>${parents}</span>
      ${issueUrl ? `<span class="detail-label">issue</span><span><a href="${esc(issueUrl)}" target="_blank">#${orbit.issue}</a></span>` : ''}
      ${repo ? `<span class="detail-label">branch</span><span><a href="https://github.com/${esc(repo)}/tree/orbit/${esc(orbit.name)}" target="_blank">orbit/${esc(orbit.name)}</a></span>` : ''}
      ${repo ? `<span class="detail-label">code</span><span><a href="https://github.com/${esc(repo)}/tree/orbit/${esc(orbit.name)}/orbits/${esc(orbit.name)}" target="_blank">orbits/${esc(orbit.name)}/</a></span>` : ''}
      ${orbit.log_url ? `<span class="detail-label">log</span><span><a href="${esc(orbit.log_url)}" target="_blank">log.md</a></span>`
        : repo ? `<span class="detail-label">log</span><span><a href="https://github.com/${esc(repo)}/blob/orbit/${esc(orbit.name)}/orbits/${esc(orbit.name)}/log.md" target="_blank">log.md</a></span>` : ''}
    </div>
    ${figs ? `<div class="detail-figs">${figs}</div>` : ''}
    ${links ? `<div class="detail-links">${links}</div>` : ''}
    ${(orbit.issue_comments || []).length ? renderIssueComments(orbit.issue_comments) : ''}
    ${ancestryHtml}
  `;

  // Scroll panel to top
  panel.scrollTop = 0;

  // Wire close
  panel.querySelector('.detail-close').addEventListener('click', () => {
    hideDetailPanel();
    if (onSelectOrbit) onSelectOrbit(null);
  });

  // Wire all orbit links (parents + ancestry path)
  panel.querySelectorAll('.parent-link').forEach(a => {
    a.addEventListener('click', () => {
      const name = a.dataset.orbit;
      if (onSelectOrbit) onSelectOrbit(name);
    });
  });
}

function hideDetailPanel() {
  const panel = document.body.querySelector('.detail-panel');
  if (!panel) return;
  panel.classList.remove('open');
  panel.classList.add('closing');
  setTimeout(() => { if (panel.parentNode) panel.remove(); }, 300);
}

function multiParentEdgeStyles(orbits, fg) {
  // Edges into multi-parent (cross) nodes get heavier + darker so polygeny
  // reads distinctly from single-parent lineage.
  const styles = [];
  for (const o of orbits) {
    if ((o.parents || []).length > 1) {
      styles.push({
        selector: `edge[target = "${o.name}"]`,
        style: { 'width': 2.25, 'line-color': fg, 'opacity': 0.5 },
      });
    }
  }
  return styles;
}

function edgeConnectednessStyles(orbits) {
  // Thicker edges for nodes with more connections (higher degree = more
  // influence in the research map). Also highlight the winner's ancestry path.
  const degree = {};
  for (const o of orbits) {
    degree[o.name] = (degree[o.name] || 0);
    for (const p of (o.parents || [])) {
      degree[p] = (degree[p] || 0) + 1;  // out-degree of parent
      degree[o.name] = (degree[o.name] || 0) + 1;  // in-degree of child
    }
  }
  const maxDeg = Math.max(1, ...Object.values(degree));
  const styles = [];
  for (const o of orbits) {
    for (const p of (o.parents || [])) {
      const avgDeg = ((degree[p] || 0) + (degree[o.name] || 0)) / 2;
      const w = 1.25 + (avgDeg / maxDeg) * 1.75; // 1.25..3px
      styles.push({
        selector: `edge[source = "${p}"][target = "${o.name}"]`,
        style: { 'width': w },
      });
    }
  }
  return styles;
}

function winnerAncestryPath(orbits) {
  // Trace back from winner to all ancestors; return set of orbit names on the path.
  const winner = orbits.find(o => o.status === 'winner');
  if (!winner) return new Set();
  const byName = Object.fromEntries(orbits.map(o => [o.name, o]));
  const visited = new Set();
  const queue = [winner.name];
  while (queue.length) {
    const cur = queue.shift();
    if (visited.has(cur)) continue;
    visited.add(cur);
    for (const p of (byName[cur]?.parents || [])) {
      if (byName[p]) queue.push(p);
    }
  }
  return visited;
}

function tallyStatus(orbits) {
  const t = { total: orbits.length, graduated: 0, 'dead-end': 0, active: 0 };
  for (const o of orbits) {
    if (o.status === 'graduated' || o.status === 'winner') t.graduated++;
    else if (o.status === 'dead-end') t['dead-end']++;
    else t.active++;
  }
  return t;
}

function renderFooter(t) {
  return `<footer class="site">
    <span>${t.total} orbits explored · ${t.graduated} graduated · ${t['dead-end']} dead-ends · ${t.active} active</span>
    <span>research-everything</span>
  </footer>`;
}

function metricRank(o, c) {
  if (o.metric == null) return Infinity;
  return c.best && c.best.direction === 'max' ? -o.metric : o.metric;
}

function fmt(v) {
  if (v == null) return '—';
  if (typeof v === 'number') return v.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
  if (typeof v === 'object' && 'value' in v) return fmt(v.value);
  return String(v);
}

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
}

function issueUrl(data, n) {
  const r = data.campaign?.repo;
  return r ? `https://github.com/${r}/issues/${n}` : '#';
}

function notFound(name) {
  return `<p class="back"><a href="#/">← back</a></p><h1>orbit not found</h1><p class="meta">No orbit named <code>${esc(name)}</code> in data.json.</p>`;
}
