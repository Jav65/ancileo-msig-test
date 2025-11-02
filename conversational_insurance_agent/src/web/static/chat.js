document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const feed = document.getElementById("message-feed");
  const statusEl = document.getElementById("status");

  if (!form || !input || !feed) {
    return;
  }

  const scrollToBottom = () => {
    feed.scrollTo({ top: feed.scrollHeight, behavior: "smooth" });
  };

  const createMessage = (role, text) => {
    const container = document.createElement("div");
    container.classList.add("message", role === "user" ? "user" : "assistant");

    const meta = document.createElement("div");
    meta.classList.add("meta");
    meta.textContent = role === "user" ? "You" : "Aurora";

    const bubble = document.createElement("div");
    bubble.classList.add("bubble");
    bubble.textContent = text;

    container.appendChild(meta);
    container.appendChild(bubble);
    return container;
  };

  const appendMessage = (role, text) => {
    if (!text) {
      return;
    }
    const msg = createMessage(role, text);
    feed.appendChild(msg);
    scrollToBottom();
  };

  const appendToolRuns = (toolRuns) => {
    if (!Array.isArray(toolRuns) || toolRuns.length === 0) {
      return;
    }

    const details = document.createElement("details");
    details.classList.add("message", "assistant");

    const summary = document.createElement("summary");
    summary.textContent = `Tools executed (${toolRuns.length})`;
    details.appendChild(summary);

    const list = document.createElement("div");
    list.classList.add("bubble");

    toolRuns.forEach((run) => {
      const item = document.createElement("div");
      item.classList.add("tool-run");
      const name = run?.name || run?.tool;
      const input = run?.input ? JSON.stringify(run.input, null, 2) : "{}";
      const result = run?.result ? JSON.stringify(run.result, null, 2) : "{}";
      item.innerHTML = `
        <strong>${name || "unknown"}</strong>
        <pre>input: ${input}</pre>
        <pre>result: ${result}</pre>
      `;
      list.appendChild(item);
    });

    details.appendChild(list);
    feed.appendChild(details);
    scrollToBottom();
  };

  const setStatus = (text) => {
    if (statusEl) {
      statusEl.textContent = text || "";
    }
  };

  setStatus("Ready");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) {
      setStatus("Ask something about coverage or next steps.");
      return;
    }

    setStatus("Sending...");
    appendMessage("user", text);
    input.value = "";

    try {
      const response = await fetch("/integration/chat/send", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        const detail = errorPayload?.detail || "Something went wrong.";
        appendMessage("assistant", `?? ${detail}`);
        setStatus("Failed to send message");
        return;
      }

      const data = await response.json();
      appendMessage("assistant", data.reply || "");
      appendToolRuns(data.tool_runs);
      setStatus("Ready");
    } catch (error) {
      console.error(error);
      appendMessage("assistant", "?? Unable to reach the assistant. Please try again.");
      setStatus("Connection issue");
    }
  });
});
