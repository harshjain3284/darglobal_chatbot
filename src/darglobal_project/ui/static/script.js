// DarGlobal & Wasalt Real Estate Assistant Client Script

// Configure marked.js options
if (typeof marked !== "undefined") {
  marked.setOptions({
    gfm: true,
    breaks: true,
  });
}

// Configure DOMPurify hook to ensure all links open securely in a new tab
if (typeof DOMPurify !== "undefined") {
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A") {
      const href = node.getAttribute("href") || "";
      // Restrict strictly to safe protocols (http and https only)
      if (
        !href.startsWith("http://") &&
        !href.startsWith("https://")
      ) {
        node.removeAttribute("href");
        node.removeAttribute("target");
        node.removeAttribute("rel");
      } else {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener noreferrer");
      }
    }
  });
}

/**
 * Safely parse Markdown and sanitize rendered HTML.
 * Strips script tags, unsafe protocols (e.g. javascript, data), and dangerous attributes.
 */
function renderMarkdown(text) {
  if (!text) return "";
  if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
    const rawHtml = marked.parse(text);
    return DOMPurify.sanitize(rawHtml, {
      ALLOWED_TAGS: [
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "br", "hr",
        "strong", "b", "em", "i", "u", "s", "strike",
        "ul", "ol", "li",
        "table", "thead", "tbody", "tr", "th", "td",
        "blockquote", "code", "pre",
        "a", "span",
      ],
      ALLOWED_ATTR: ["href", "title", "target", "rel", "class", "align"],
      ALLOWED_URI_REGEXP: /^https?:\/\//i,
    });
  }
  // Safe plain-text fallback
  const fallbackDiv = document.createElement("div");
  fallbackDiv.textContent = text;
  return fallbackDiv.innerHTML;
}

document.addEventListener("DOMContentLoaded", () => {
  const chatContainer = document.getElementById("chat-container");
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const typingIndicator = document.getElementById("typing-indicator");
  const errorBanner = document.getElementById("error-banner");
  const statusText = document.getElementById("status-text");

  // Check backend health on initial load
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        statusText.textContent = `Live (${data.records_indexed} properties indexed)`;
      } else {
        statusText.textContent = "Service degraded";
      }
    } catch {
      statusText.textContent = "Offline";
    }
  }
  checkHealth();

  // Scroll to bottom
  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  // Hide or show error
  function showError(msg) {
    if (msg) {
      errorBanner.textContent = msg;
      errorBanner.style.display = "block";
    } else {
      errorBanner.textContent = "";
      errorBanner.style.display = "none";
    }
  }

  // Append user message
  function appendUserMessage(text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message user-message";

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.textContent = text;

    msgDiv.appendChild(contentDiv);
    chatContainer.appendChild(msgDiv);
    scrollToBottom();
  }

  // Format assistant response with citations
  function appendAssistantMessage(answer, sources) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant-message";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🤖";

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";

    const markdownContainer = document.createElement("div");
    markdownContainer.className = "markdown-body";
    markdownContainer.innerHTML = renderMarkdown(answer);
    contentDiv.appendChild(markdownContainer);

    // Render source citations if available
    if (sources && sources.length > 0) {
      const sourcesSec = document.createElement("div");
      sourcesSec.className = "sources-section";

      const header = document.createElement("div");
      header.className = "sources-header";
      header.textContent = "Grounding Sources & Evidence:";
      sourcesSec.appendChild(header);

      const cardsContainer = document.createElement("div");
      cardsContainer.className = "source-cards";

      sources.forEach((src) => {
        const card = document.createElement("div");
        card.className = "source-card";

        const topRow = document.createElement("div");
        topRow.className = "source-card-top";

        const link = document.createElement("a");
        link.className = "source-title";
        link.href = src.source_url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = src.title || "View Property";

        const badge = document.createElement("span");
        const platform = (src.source_platform || "Unknown").toLowerCase();
        badge.className = `platform-badge badge-${platform}`;
        badge.textContent = src.source_platform || "Source";

        topRow.appendChild(link);
        topRow.appendChild(badge);
        card.appendChild(topRow);

        const metaRow = document.createElement("div");
        metaRow.className = "source-meta";

        if (src.location) {
          const locSpan = document.createElement("span");
          locSpan.textContent = `📍 ${src.location}`;
          metaRow.appendChild(locSpan);
        }

        if (src.price) {
          const priceSpan = document.createElement("span");
          priceSpan.textContent = `💰 ${src.price}`;
          metaRow.appendChild(priceSpan);
        }

        if (metaRow.children.length > 0) {
          card.appendChild(metaRow);
        }

        cardsContainer.appendChild(card);
      });

      sourcesSec.appendChild(cardsContainer);
      contentDiv.appendChild(sourcesSec);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);
    chatContainer.appendChild(msgDiv);
    scrollToBottom();
  }

  // Handle message submission
  async function handleSubmit(question) {
    const query = (question || "").trim();
    if (!query) return;

    showError(null);
    appendUserMessage(query);
    userInput.value = "";
    userInput.disabled = true;
    sendBtn.disabled = true;
    typingIndicator.style.display = "flex";
    scrollToBottom();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: query }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Server error" }));
        throw new Error(errData.detail || `Server responded with ${response.status}`);
      }

      const data = await response.json();
      appendAssistantMessage(data.answer, data.sources || []);
    } catch (err) {
      console.error("Chat error:", err);
      showError(`Unable to process request: ${err.message}`);
      appendAssistantMessage(
        "I encountered an error connecting to the retrieval service. Please ensure API credentials are configured properly.",
        []
      );
    } finally {
      typingIndicator.style.display = "none";
      userInput.disabled = false;
      sendBtn.disabled = false;
      userInput.focus();
      scrollToBottom();
    }
  }

  // Form submit listener
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSubmit(userInput.value);
  });

  // Sample query chips listener
  document.addEventListener("click", (e) => {
    if (e.target.classList.contains("chip")) {
      const q = e.target.getAttribute("data-query");
      if (q) handleSubmit(q);
    }
  });
});
