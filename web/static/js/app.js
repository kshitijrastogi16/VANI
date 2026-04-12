'use strict';

// Minimal, safe HTML escaping
function escapeHTML(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Format internal verse_id to display citation: K.S.V or K.S.V1-V2
function formatCitation(verseId) {
  if (!verseId && verseId !== 0) return 'Unknown';
  const id = String(verseId);
  const parts = id.split('-');
  if (parts.length === 3) return `${parts[0]}.${parts[1]}.${parts[2]}`;
  if (parts.length === 4) return `${parts[0]}.${parts[1]}.${parts[2]}-${parts[3]}`;
  return id.replace(/-/g, '.');
}

class VANIInterface {
  constructor() {
    this.initializeElements();
    this.bindEvents();
  }

  initializeElements() {
    // Form elements
    this.queryForm = document.getElementById('query-form');
    this.questionInput = document.getElementById('question-input');
    this.askButton = document.getElementById('ask-button');
  // Optional UI controls — some templates may omit them. Provide safe defaults when missing.
  this.includeContext = document.getElementById('include-context') || { checked: true };
  this.maxResults = document.getElementById('max-results') || { value: '10' };
  this.advancedToggle = document.getElementById('advanced-toggle') || null;
  this.advancedPanel = document.getElementById('advanced-panel') || null;

    // Display sections
    this.answerSection = document.getElementById('answer-section');
    this.sourcesSection = document.getElementById('sources-section');
    this.errorSection = document.getElementById('error-section');

    // Content elements
    this.answerText = document.getElementById('answer-text');
    this.queryType = document.getElementById('query-type');
    this.responseTime = document.getElementById('response-time');
    this.confidenceScore = document.getElementById('confidence-score');
    this.sourcesList = document.getElementById('sources-list');
    this.errorMessage = document.getElementById('error-message');

    // UI elements
    this.buttonText = document.querySelector('.button-text');
    this.loadingSpinner = document.querySelector('.loading-spinner');
  }

  bindEvents() {
    // Form submission
    this.queryForm.addEventListener('submit', (e) => this.handleSubmit(e));

    // Suggestion buttons
    document.querySelectorAll('.suggestion-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        this.questionInput.value = btn.dataset.query;
        this.handleSubmit();
      });
    });

    // Advanced panel toggle
    if (this.advancedToggle && this.advancedPanel) {
      this.advancedToggle.addEventListener('click', () => {
        const isHidden = this.advancedPanel.classList.contains('hidden');
        this.advancedPanel.classList.toggle('hidden', !isHidden);
        // Flip caret
        this.advancedToggle.textContent = isHidden ? 'Advanced ▴' : 'Advanced ▾';
      });
    }

    // Enter key in textarea
    this.questionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleSubmit();
      }
    });
  }

  async handleSubmit(e) {
    if (e) e.preventDefault();

    const question = this.questionInput.value.trim();
    if (!question) {
      this.showError('Please enter a question about the Ramayana.');
      return;
    }

    this.setLoadingState(true);
    this.hideAllSections();

    try {
      const response = await this.fetchAnswer(question);

      if (response.success) {
        this.displayAnswer(response);
      } else {
        // Prefer server-provided answer if available, else error, else fallback
        if (response.answer) {
          this.displayAnswer(response);
        } else if (response.error) {
          this.showError(response.error);
        } else {
          this.showError('An error occurred while processing your question.');
        }
      }
    } catch (error) {
      console.error('Query error:', error);
      // Surface the server-provided message when available to aid debugging
      const msg = (error && error.message) ? String(error.message) : 'Failed to connect to VANI server. Please try again.';
      this.showError(msg);
    } finally {
      this.setLoadingState(false);
    }
  }

  async fetchAnswer(question) {
    const payload = {
      question: question,
      include_context: this.includeContext.checked,
      max_results: parseInt(this.maxResults.value),
    };

    const response = await fetch('/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      // Try to read response body for a helpful error message
      let body = '';
      try {
        body = await response.text();
      } catch (e) {
        body = response.statusText || '';
      }
      throw new Error(`HTTP ${response.status}: ${body}`);
    }

    return await response.json();
  }

  displayAnswer(data) {
    // Display answer
    this.answerText.innerHTML = this.formatAnswer(data.answer);
    this.queryType.textContent = data.query_type || 'general';
    // API provides seconds; display in milliseconds
    this.responseTime.textContent = Math.round((data.response_time || 0) * 1000);
    this.confidenceScore.textContent = data.confidence
      ? (data.confidence * 100).toFixed(1) + '%'
      : 'N/A';
    // Show abstention indicator when applicable
    if (
      data.query_type === 'insufficient_evidence' ||
      data.query_type === 'no_results'
    ) {
      const meta = document.getElementById('answer-metadata');
      if (meta && !document.getElementById('abstention-indicator')) {
        const span = document.createElement('span');
        span.className = 'metadata-item';
        span.id = 'abstention-indicator';
        span.innerHTML = 'Status: <span class="status-abstained">Abstained</span>';
        meta.appendChild(span);
      }
    } else {
      const existing = document.getElementById('abstention-indicator');
      if (existing) existing.remove();
    }

    this.answerSection.classList.remove('hidden');

    // Display sources if available
    if (data.sources && data.sources.length > 0) {
      // Dynamic title for citation queries
      const titleEl = document.getElementById('sources-title');
      if (titleEl) {
        titleEl.textContent =
          data.query_type === 'citation' ? '📚 Context Verses' : '📚 Sources & Citations';
      }
      this.displaySources(data.sources, data.citations);
      this.sourcesSection.classList.remove('hidden');
    }

    // Smooth scroll to answer
    this.answerSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  displaySources(sources, citations) {
    const sourcesHtml = sources
      .map((source, index) => {
        const citationFormat = formatCitation(source.verse_id);
        const isCited = citations && citations.includes(citationFormat);
        const kandaName = escapeHTML(source.kanda_name || 'Unknown Kanda');
        const sargaNum = escapeHTML(String(source.sarga_num || '?'));

        // Titles handling: tooltip and optional expand
        let titlesBlock = '';
        if (Array.isArray(source.sarga_titles) && source.sarga_titles.length > 0) {
          const firstTitle = escapeHTML(source.sarga_titles[0] || '');
          const allTitlesTooltip = escapeHTML(source.sarga_titles.join(' | '));
          let moreBlock = '';
          if (source.sarga_titles.length > 1) {
            const moreId = `titles-more-${index}`;
            const allTitlesItems = source.sarga_titles
              .map((t) => `<div class="title-item">${escapeHTML(t)}</div>`) 
              .join('');
            moreBlock = ` <button class="titles-toggle" data-target="${moreId}">+${
              source.sarga_titles.length - 1
            } more</button>
                <div class="titles-more hidden mt-2" id="${moreId}">${allTitlesItems}</div>`;
          }
          titlesBlock = `<div class="source-context" title="${allTitlesTooltip}">${firstTitle}${moreBlock}</div>`;
        }

        // Safe verse content
        const skt = source.shloka_sanskrit
          ? `<div class="sanskrit"><strong>Sanskrit:</strong> ${escapeHTML(
              source.shloka_sanskrit
            )}</div>`
          : '';
        const iast = source.shloka_iast
          ? `<div class="sanskrit"><strong>Sanskrit (IAST):</strong> ${escapeHTML(
              source.shloka_iast
            )}</div>`
          : '';
        const en = source.translation_en
          ? `<div class="translation en"><strong>English:</strong> ${escapeHTML(
              source.translation_en
            )}</div>`
          : '';
        const hi = source.translation_hi
          ? `<div class="translation hi"><strong>Hindi:</strong> ${escapeHTML(
              source.translation_hi
            )}</div>`
          : '';

        return `
            <div class="source-item ${isCited ? 'cited' : ''}">
                <div class="source-header">
                    <span class="verse-id">${citationFormat}</span>
                    <span class="source-location">${kandaName}, Sarga ${sargaNum}</span>
                    ${source.search_score ? `<span class="relevance-score">Score: ${(
                      source.search_score * 100
                    ).toFixed(1)}%</span>` : ''}
                </div>
                ${titlesBlock}
                <div class="verse-content">
                    ${skt}${iast}${en}${hi}
                </div>
            </div>
        `;
      })
      .join('');

    this.sourcesList.innerHTML = sourcesHtml;

    // Attach toggle handlers for titles
    this.attachTitlesToggleHandlers();

    // Show global controls if any item has expandable titles
    const anyExpandable =
      Array.isArray(sources) && sources.some((s) => Array.isArray(s.sarga_titles) && s.sarga_titles.length > 1);
    this.ensureGlobalTitlesControls(anyExpandable);
    if (anyExpandable) {
      this.attachGlobalTitlesControls();
    }
  }

  formatAnswer(text) {
    if (!text) return '';
    // Escape all HTML first to prevent injection
    let safe = escapeHTML(text);
    // Transform special lines into styled blocks and preserve other lines
    const lines = safe.split('\n');
    let html = lines
      .map((line) => {
        let m = line.match(/^Context Summary:\s*(.*)$/);
        if (m) {
          return `<div class="context-summary"><span class="label">Context Summary:</span> ${m[1]}</div>`;
        }
        m = line.match(/^Context:\s*(.*)$/);
        if (m) {
          return `<div class="context-note"><span class="label">Context:</span> ${m[1]}</div>`;
        }
        return line;
      })
      .join('<br>');
    // Highlight verse citations like K.S.V or K.S.V1-V2
    html = html.replace(/\b(\d+\.\d+\.\d+(?:-\d+)?)\b/g, '<span class="verse-citation">$1</span>');
    // Highlight Sanskrit/IAST terms (basic heuristic)
    html = html.replace(
      /\b([A-Za-z]*[āīūṛṝḷĕŏāīūṛṝḷṃḥṅñṇṭḍśṣ][A-Za-z]*)\b/g,
      '<span class="sanskrit-term">$1</span>'
    );
    return html;
  }

  attachTitlesToggleHandlers() {
    document.querySelectorAll('.titles-toggle').forEach((btn) => {
      if (!btn.dataset.initial) {
        btn.dataset.initial = btn.textContent;
      }
      btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-target');
        const panel = document.getElementById(targetId);
        if (!panel) return;
        const isHidden = panel.classList.contains('hidden');
        panel.classList.toggle('hidden', !isHidden);
        btn.textContent = isHidden ? 'Hide titles' : btn.dataset.initial;
      });
    });
  }

  ensureGlobalTitlesControls(anyExpandable) {
    const container = document.querySelector('.sources-container');
    if (!container) return;
    let controls = container.querySelector('.titles-controls');
    if (!controls) {
      controls = document.createElement('div');
      controls.className = 'titles-controls';
      controls.innerHTML = `
        <button class="titles-control-btn" id="expand-all-titles">Expand all titles</button>
        <button class="titles-control-btn secondary" id="collapse-all-titles">Collapse all titles</button>
      `;
      container.insertBefore(controls, this.sourcesList);
    }
    controls.style.display = anyExpandable ? 'flex' : 'none';
  }

  attachGlobalTitlesControls() {
    const expandBtn = document.getElementById('expand-all-titles');
    const collapseBtn = document.getElementById('collapse-all-titles');
    if (expandBtn) {
      expandBtn.onclick = () => {
        document.querySelectorAll('.titles-more').forEach((panel) => {
          panel.classList.remove('hidden');
        });
        document.querySelectorAll('.titles-toggle').forEach((btn) => {
          if (!btn.dataset.initial) btn.dataset.initial = btn.textContent;
          btn.textContent = 'Hide titles';
        });
      };
    }
    if (collapseBtn) {
      collapseBtn.onclick = () => {
        document.querySelectorAll('.titles-more').forEach((panel) => {
          panel.classList.add('hidden');
        });
        document.querySelectorAll('.titles-toggle').forEach((btn) => {
          btn.textContent = btn.dataset.initial || btn.textContent;
        });
      };
    }
  }

  showError(message) {
    this.errorMessage.textContent = message;
    this.errorSection.classList.remove('hidden');
    this.errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  hideAllSections() {
    this.answerSection.classList.add('hidden');
    this.sourcesSection.classList.add('hidden');
    this.errorSection.classList.add('hidden');
  }

  setLoadingState(loading) {
    this.askButton.disabled = loading;

    if (loading) {
      this.buttonText.classList.add('hidden');
      this.loadingSpinner.classList.remove('hidden');
      this.askButton.classList.add('loading');
    } else {
      this.buttonText.classList.remove('hidden');
      this.loadingSpinner.classList.add('hidden');
      this.askButton.classList.remove('loading');
    }
  }
}

// Initialize VANI interface when DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
  new VANIInterface();
});

// Service worker registration for offline support (optional)
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('/static/sw.js')
      .then((registration) => {
        console.log('SW registered: ', registration);
      })
      .catch((registrationError) => {
        console.log('SW registration failed: ', registrationError);
      });
  });
}
