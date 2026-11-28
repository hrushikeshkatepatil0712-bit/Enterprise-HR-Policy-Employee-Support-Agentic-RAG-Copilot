
const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const question = document.getElementById("question");
const trace = document.getElementById("trace");
const sourceUsed = document.getElementById("sourceUsed");

let isProcessing = false;

function escapeHtml(value = "") {
    return String(value).replace(/[&<>'"]/g, c => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "'": "&#39;",
        '"': "&quot;"
    }[c]));
}

function formatText(value = "") {
    return escapeHtml(value).replace(/\n/g, "<br>");
}

function addMessage(role, text, source = "", citations = []) {
    const wrap = document.createElement("div");
    wrap.className = `message ${role}`;

    const citationHtml = citations.length
        ? `<div class="citations">
            <strong>Sources</strong><br>
            ${citations.map(c => {
                const title = c.title || c.source || "Unknown source";

                if (c.url) {
                    const url = String(c.url);
                    if (/^https?:\/\//i.test(url)) {
                        return `<a href="${escapeHtml(url)}"
                            target="_blank"
                            rel="noopener noreferrer">
                            ${escapeHtml(title)}
                        </a>`;
                    }
                }

                return escapeHtml(title);
            }).join("<br>")}
        </div>`
        : "";

    wrap.innerHTML = `
        <div class="avatar">AI</div>
        <div class="bubble">
            ${formatText(text)}
            ${source
                ? `<div class="answer-source">
                    Source: ${escapeHtml(source)}
                   </div>`
                : ""}
            ${citationHtml}
        </div>
    `;

    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
}

function renderTrace(items = []) {
    trace.innerHTML = items.length
        ? items.map(item =>
            `<div class="trace-item">
                ${escapeHtml(item)}
            </div>`
        ).join("")
        : '<div class="empty">No trace.</div>';
}

async function askAgent(q) {
    if (isProcessing) return;

    isProcessing = true;

    addMessage("user", q);
    question.value = "";

    renderTrace(["Running LangGraph workflow..."]);
    sourceUsed.textContent = "Running";

    const btn = form.querySelector("button");
    btn.disabled = true;

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: q
            })
        });

        const raw = await response.text();

        let data;

        try {
            data = JSON.parse(raw);
        } catch {
            throw new Error(
                `Server returned invalid JSON (HTTP ${response.status})`
            );
        }

        if (!response.ok) {
            const detail = data.detail;

            const message = typeof detail === "string"
                ? detail
                : JSON.stringify(detail || data);

            throw new Error(
                `HTTP ${response.status}: ${message}`
            );
        }

        addMessage(
            "assistant",
            data.answer || "No answer returned.",
            data.source_used || "",
            data.citations || []
        );

        renderTrace(data.trace || []);

        sourceUsed.textContent =
            data.source_used || "Unknown";

    } catch (error) {
        console.error("HR Copilot request error:", error);

        addMessage(
            "assistant",
            `Error: ${error.message}`
        );

        renderTrace([
            "Request failed. Check browser console and FastAPI logs."
        ]);

        sourceUsed.textContent = "Error";

    } finally {
        btn.disabled = false;
        isProcessing = false;
    }
}

form.addEventListener("submit", event => {
    event.preventDefault();

    const q = question.value.trim();

    if (q) {
        askAgent(q);
    }
});

document.querySelectorAll(".example").forEach(button => {
    button.addEventListener("click", () => {
        askAgent(button.textContent.trim());
    });
});

const modal = document.getElementById("uploadModal");

document.getElementById("openUpload").onclick = () => {
    modal.classList.remove("hidden");
};

document.getElementById("closeUpload").onclick = () => {
    modal.classList.add("hidden");
};

document.getElementById("uploadBtn").onclick = async () => {
    const file = document.getElementById("fileInput").files[0];
    const key = document.getElementById("adminKey").value;
    const status = document.getElementById("uploadStatus");

    if (!file) {
        status.textContent = "Choose a file first.";
        return;
    }

    status.textContent = "Indexing document...";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/api/ingest", {
            method: "POST",
            headers: {
                "X-Admin-Key": key
            },
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                typeof data.detail === "string"
                    ? data.detail
                    : JSON.stringify(data.detail || "Upload failed")
            );
        }

        status.textContent =
            `Indexed ${data.file}: ${data.chunks} chunks.`;

    } catch (error) {
        console.error("Upload error:", error);
        status.textContent = `Error: ${error.message}`;
    }
};
