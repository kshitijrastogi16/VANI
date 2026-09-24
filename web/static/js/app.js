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

function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return '—';
  if (value < 1) return `${Math.round(value * 1000)}ms`;
  if (value < 60) return `${value.toFixed(1).replace(/\.0$/, '')}s`;

  const minutes = Math.floor(value / 60);
  const remainingSeconds = Math.round(value % 60);
  return remainingSeconds ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
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

    // About modal (optional)
    this.aboutToggle = document.getElementById('about-toggle') || null;
    this.aboutModal = document.getElementById('about-modal') || null;

    // Display sections
    this.answerSection = document.getElementById('answer-section');
    this.sourcesSection = document.getElementById('sources-section');
    this.errorSection = document.getElementById('error-section');

    // Content elements
    this.answerQuestion = document.getElementById('answer-question') || null;
    this.answerQuestionWrap = document.getElementById('answer-question-wrap') || null;
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

    // Advanced panel toggle — a simple on/off switch
    if (this.advancedToggle && this.advancedPanel) {
      this.advancedToggle.addEventListener('change', () => {
        this.advancedPanel.classList.toggle('hidden', !this.advancedToggle.checked);
      });
    }

    // Enter key in textarea
    this.questionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleSubmit();
      }
    });

    // Auto-grow the single-line search field as the person types
    this.questionInput.addEventListener('input', () => {
      this.questionInput.style.height = 'auto';
      this.questionInput.style.height = Math.min(this.questionInput.scrollHeight, 130) + 'px';
    });

    // About modal
    this.bindAboutModal();
  }

  bindAboutModal() {
    if (!this.aboutToggle || !this.aboutModal) return;

    const openModal = () => {
      this.aboutModal.classList.remove('hidden');
      this.aboutToggle.setAttribute('aria-expanded', 'true');
      const closeBtn = this.aboutModal.querySelector('.about-modal__close');
      if (closeBtn) closeBtn.focus();
    };

    const closeModal = () => {
      this.aboutModal.classList.add('hidden');
      this.aboutToggle.setAttribute('aria-expanded', 'false');
      this.aboutToggle.focus();
    };

    this.aboutToggle.addEventListener('click', openModal);

    this.aboutModal.querySelectorAll('[data-about-close]').forEach((el) => {
      el.addEventListener('click', closeModal);
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !this.aboutModal.classList.contains('hidden')) {
        closeModal();
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

    this.lastQuestion = question;

    // Treat submission as a fresh query: clear the input immediately so the
    // previous question never remains in the search control while results load.
    this.questionInput.value = '';
    this.questionInput.style.height = '20px';

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
    // Echo the question that was asked
    if (this.answerQuestion) {
      if (this.lastQuestion) {
        this.answerQuestion.textContent = this.lastQuestion;
        this.answerQuestion.classList.remove('hidden');
        if (this.answerQuestionWrap) this.answerQuestionWrap.classList.remove('hidden');
      } else {
        this.answerQuestion.classList.add('hidden');
        if (this.answerQuestionWrap) this.answerQuestionWrap.classList.add('hidden');
      }
    }

    // Display answer
    this.answerText.innerHTML = this.formatAnswer(data.answer);
    this.queryType.textContent = data.query_type || 'general';
    // API provides seconds; show a human-friendly duration for end users.
    this.responseTime.textContent = formatDuration(data.response_time || 0);
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
    document.body.classList.add('has-result');

    // Display sources if available
    if (data.sources && data.sources.length > 0) {
      // Dynamic title for citation queries
      const titleEl = document.getElementById('sources-title');
      if (titleEl) {
        titleEl.textContent =
          data.query_type === 'citation' ? 'Context Verses' : 'Sources';
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

        // Safe verse content — quiet, labeled lines rather than colored blocks
        const skt = source.shloka_sanskrit
          ? `<p class="verse-line sanskrit"><span class="field-label">Sanskrit</span>${escapeHTML(source.shloka_sanskrit)}</p>`
          : '';
        const iast = source.shloka_iast
          ? `<p class="verse-line"><span class="field-label">IAST</span>${escapeHTML(source.shloka_iast)}</p>`
          : '';
        const en = source.translation_en
          ? `<p class="verse-line"><span class="field-label">English</span>${escapeHTML(source.translation_en)}</p>`
          : '';
        const hi = source.translation_hi
          ? `<p class="verse-line"><span class="field-label">Hindi</span>${escapeHTML(source.translation_hi)}</p>`
          : '';

        return `
            <div class="source-item ${isCited ? 'cited' : ''}">
                <div class="source-header">
                    <span class="verse-id">${citationFormat}</span>
                    <span class="source-location">${kandaName}, Sarga ${sargaNum}</span>
                    ${source.search_score ? `<span class="relevance-score">${(
                      source.search_score * 100
                    ).toFixed(1)}% match</span>` : ''}
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

    // Escape HTML first to keep generated answer content safe.
    const safe = escapeHTML(text);
    const lines = safe.split('\n');
    const blocks = [];

    const highlightTerms = (value) => {
      let html = value;
      html = html.replace(/\b(\d+\.\d+\.\d+(?:-\d+)?)\b/g, '<span class="verse-citation">$1</span>');
      html = html.replace(
        /\b([A-Za-z]*[āīūṛṝḷĕŏṃḥṅñṇṭḍśṣ][A-Za-z]*)\b/g,
        '<span class="sanskrit-term">$1</span>'
      );
      return html;
    };

    lines.forEach((rawLine) => {
      const line = rawLine.trim();

      if (!line) {
        blocks.push('<div class="answer-spacer" aria-hidden="true"></div>');
        return;
      }

      let m = line.match(/^Context Summary:\s*(.*)$/);
      if (m) {
        blocks.push(`<div class="context-summary"><span class="label">Context Summary</span>${highlightTerms(m[1])}</div>`);
        return;
      }

      m = line.match(/^Context:\s*(.*)$/);
      if (m) {
        blocks.push(`<div class="context-note"><span class="label">Context</span>${highlightTerms(m[1])}</div>`);
        return;
      }

      // Citation answers often contain clearly structured Sanskrit / IAST / English fields.
      // Render these as distinct, quiet sections instead of one oversized paragraph.
      m = line.match(/^Sanskrit:\s*(.*)$/i);
      if (m) {
        blocks.push(
          `<div class="answer-verse-field answer-verse-field--sanskrit">` +
          `<span class="answer-field-label">Sanskrit</span>` +
          `<div class="answer-field-content">${m[1]}</div></div>`
        );
        return;
      }

      m = line.match(/^IAST:\s*(.*)$/i);
      if (m) {
        blocks.push(
          `<div class="answer-verse-field answer-verse-field--iast">` +
          `<span class="answer-field-label">IAST</span>` +
          `<div class="answer-field-content">${m[1]}</div></div>`
        );
        return;
      }

      m = line.match(/^English:\s*(.*)$/i);
      if (m) {
        blocks.push(
          `<div class="answer-verse-field">` +
          `<span class="answer-field-label">English</span>` +
          `<div class="answer-field-content">${m[1]}</div></div>`
        );
        return;
      }

      m = line.match(/^Hindi:\s*(.*)$/i);
      if (m) {
        blocks.push(
          `<div class="answer-verse-field answer-verse-field--hindi">` +
          `<span class="answer-field-label">Hindi</span>` +
          `<div class="answer-field-content">${m[1]}</div></div>`
        );
        return;
      }

      blocks.push(`<div class="answer-line">${highlightTerms(line)}</div>`);
    });

    return blocks.join('');
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
    const container = document.querySelector('#sources-section .result-card');
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
    document.body.classList.add('has-result');
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
