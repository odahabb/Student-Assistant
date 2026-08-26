// Study Assistant — web interface.
// Plain JavaScript, no build step and no third-party code: the page talks only
// to the local server in backend/api.py.

"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  settings: null,
  subjects: [],
  current: null,          // selected subject name
  view: "ask",
  status: null,           // index status of the current subject
  chats: {},              // subject -> messages
  asking: {},             // subject -> true while an answer is streaming
  quiz: {},               // subject -> {question, result, selected, topic}
  topics: {},             // subject -> topic list
  pollTimer: null,
};

// ---------------------------------------------------------------- helpers

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "style") node.style.cssText = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function icon(name, cls) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  if (cls) svg.setAttribute("class", cls);
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

async function api(path, options = {}) {
  const opts = { ...options };
  if (opts.json !== undefined) {
    opts.body = JSON.stringify(opts.json);
    opts.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
    delete opts.json;
  }
  const response = await fetch(path, opts);
  if (!response.ok) {
    let detail = response.statusText;
    let body = null;
    try { body = await response.json(); detail = body.detail || detail; } catch (e) {}
    const error = new Error(typeof detail === "string" ? detail : "Request failed");
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

const subjectUrl = (name, rest = "") => `/api/subjects/${encodeURIComponent(name)}${rest}`;
const docUrl = (name, file, page) =>
  subjectUrl(name, `/documents/${encodeURIComponent(file)}`) + (page ? `#page=${page}` : "");

function toast(message, kind = "") {
  const node = el("div", { class: `toast ${kind}`, text: message });
  $("#toasts").append(node);
  setTimeout(() => node.remove(), kind === "error" ? 6000 : 3200);
}

function fail(error) {
  console.error(error);
  toast(error.message || String(error), "error");
}

function pct(x) { return `${Math.round(x * 100)}%`; }

function ago(seconds) {
  const diff = Date.now() / 1000 - seconds;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  const days = Math.floor(diff / 86400);
  return days === 1 ? "yesterday" : `${days} days ago`;
}

function hue(name) {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.codePointAt(0)) % 360;
  return h;
}

function typeIcon(filename) {
  const ext = filename.split(".").pop().toLowerCase();
  if (["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif"].includes(ext)) return "image";
  if (["mp3", "wav", "m4a", "flac", "ogg", "mp4", "webm"].includes(ext)) return "audio";
  return "pdf";
}

function sourceLabel(src) {
  const parts = [src.file || "Unknown file"];
  // Slides pack several to a chunk and recordings have no pages at all, so a
  // source says "slides 12-15" or "12:03-15:40" rather than always "p. 4".
  const many = src.pages && src.pages.includes("-");
  if (src.timecode) parts.push(src.timecode);
  else if (src.kind === "slide") parts.push(many ? `slides ${src.pages}` : `slide ${src.page}`);
  else if (src.page) parts.push(`p. ${src.page}`);
  return parts.join(" · ");
}

function confirmDialog(title, text, okLabel = "Delete") {
  const dialog = $("#confirm");
  $("#confirm-title").textContent = title;
  $("#confirm-text").textContent = text;
  $("#confirm-ok").textContent = okLabel;
  dialog.returnValue = "";
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok"), { once: true });
  });
}

// Mark every occurrence of `needle` (case-insensitive) inside `text`.
function highlighted(text, needle) {
  const frag = document.createDocumentFragment();
  const n = (needle || "").trim();
  if (n.length < 2) { frag.append(text); return frag; }
  const lower = text.toLowerCase();
  const target = n.toLowerCase();
  let at = 0;
  let found = lower.indexOf(target);
  while (found !== -1) {
    frag.append(text.slice(at, found), el("mark", { text: text.slice(found, found + n.length) }));
    at = found + n.length;
    found = lower.indexOf(target, at);
  }
  frag.append(text.slice(at));
  return frag;
}

// ---------------------------------------------------------------- routing

function setHash() {
  const hash = state.current ? `#/${encodeURIComponent(state.current)}/${state.view}` : "#/";
  if (location.hash !== hash) history.replaceState(null, "", hash);
}

function readHash() {
  const [, name, view] = location.hash.split("/");
  return { name: name ? decodeURIComponent(name) : null, view: view || "ask" };
}

// ---------------------------------------------------------------- sidebar

async function loadSubjects() {
  state.subjects = await api("/api/subjects");
  renderSidebar();
}

function currentSubject() {
  return state.subjects.find((s) => s.name === state.current) || null;
}

function renderSidebar() {
  const list = $("#subject-list");
  list.replaceChildren(...state.subjects.map((s) => el("button", {
    class: "subject",
    "aria-current": s.name === state.current ? "true" : "false",
    onclick: () => { selectSubject(s.name); closeNav(); },
  },
    el("span", { class: "swatch", style: `background: hsl(${hue(s.name)} 38% 52%)` }),
    el("span", { class: "name", text: s.name }),
    el("span", { class: "n", text: s.documents.length || "" }),
  )));
  if (!state.subjects.length) {
    list.append(el("p", { class: "meta", style: "margin: 0 8px", text: "No subjects yet." }));
  }

  const subject = currentSubject();
  $("#materials").hidden = !subject;
  if (!subject) return;

  const failures = new Map((state.status?.failures || []).map((f) => [f.name, f.error]));
  const perDoc = new Map((state.status?.documents || []).map((d) => [d.name, d]));
  const indexing = state.status?.state === "indexing";
  $("#doc-count").textContent = subject.documents.length || "";
  $("#doc-list").replaceChildren(...subject.documents.map((d) => {
    const info = perDoc.get(d.name);
    let status = null;
    if (failures.has(d.name)) {
      status = el("span", { class: "state fail", title: failures.get(d.name), text: "couldn't read" });
    } else if (info) {
      status = el("span", { class: `state${info.note ? " warn" : ""}`,
        title: info.note ? `${info.chunks} passages — ${info.note}` : `${info.chunks} passages`,
        text: info.pages ? `${info.pages} p` : `${info.chunks} ¶` });
    } else if (indexing) {
      status = el("span", { class: "state", text: state.status.current === d.name ? "reading…" : "queued" });
    }
    return el("li", { class: "doc" },
      icon(typeIcon(d.name)),
      el("a", { href: docUrl(subject.name, d.name), target: "_blank", rel: "noopener", title: d.name, text: d.name }),
      status,
      el("button", { class: "icon-btn remove", "aria-label": `Remove ${d.name}`,
        onclick: () => removeDocument(d.name) }, icon("trash")),
    );
  }));
  $("#use-samples").hidden = !(subject.documents.length === 0 && state.settings?.samples);
}

function renderFooter() {
  const s = state.settings;
  if (!s) return;
  $("#side-foot").replaceChildren(
    el("div", {}, "Embeddings ", el("b", { text: s.embedder }), " · ", el("b", { text: `${s.chunking} chunks` })),
    el("div", {}, "Retrieval ", el("b", { text: s.hybrid ? "hybrid (BM25 + dense)" : "dense" }), ` · top ${s.top_k}`),
    el("div", {}, "Answers by ", el("b", { text: "flan-t5-large" }), " on ", el("b", { text: s.device.toUpperCase() })),
  );
  $("#dropzone-hint").textContent = "PDF, images, audio or text";
  $("#file-input").accept = s.extensions.map((e) => `.${e}`).join(",");
}

async function createSubject(event) {
  event.preventDefault();
  const input = $("#new-subject-name");
  const name = input.value.trim();
  if (!name) { input.focus(); return; }
  try {
    const created = await api("/api/subjects", { method: "POST", json: { name } });
    input.value = "";
    await loadSubjects();
    await selectSubject(created.name);
    toast(`Created “${created.name}”. Add some documents to it.`);
  } catch (e) { fail(e); }
}

async function uploadFiles(files) {
  if (!state.current || !files.length) return;
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  const zone = $("#dropzone");
  zone.classList.add("over");
  try {
    const result = await api(subjectUrl(state.current, "/documents"), { method: "POST", body: form });
    if (result.added.length) toast(`Added ${result.added.length} document${result.added.length > 1 ? "s" : ""}.`);
    if (result.skipped.length) toast(`Already here: ${result.skipped.join(", ")}`);
    await refreshSubject(result.index);
  } catch (e) { fail(e); }
  finally { zone.classList.remove("over"); $("#file-input").value = ""; }
}

async function removeDocument(filename) {
  const ok = await confirmDialog("Remove this document?",
    `“${filename}” will be deleted from this subject's folder and the index rebuilt.`, "Remove");
  if (!ok) return;
  try {
    const result = await api(subjectUrl(state.current, `/documents/${encodeURIComponent(filename)}`), { method: "DELETE" });
    await refreshSubject(result.index);
  } catch (e) { fail(e); }
}

async function useSamples() {
  try {
    const result = await api(subjectUrl(state.current, "/samples"), { method: "POST" });
    toast(`Added ${result.added} sample papers.`);
    await refreshSubject(result.index);
  } catch (e) { fail(e); }
}

function openNav() { $("#shell").classList.add("nav-open"); }
function closeNav() { $("#shell").classList.remove("nav-open"); }

// ---------------------------------------------------------------- subject

async function selectSubject(name, view) {
  state.current = name;
  if (view) state.view = view;
  state.status = null;
  closeDrawer();
  setHash();
  renderSidebar();
  renderMain();
  if (!name) return;
  try {
    const subject = await api(subjectUrl(name));
    if (state.current !== name) return;
    await refreshSubject(subject.index, false);
  } catch (e) {
    if (e.status === 404) { state.current = null; setHash(); renderMain(); renderSidebar(); }
    fail(e);
  }
}

async function refreshSubject(status, reloadList = true) {
  if (reloadList) await loadSubjects();
  const previous = state.status?.state;
  const previousChunks = state.status?.chunks;
  state.status = status;
  if (status.state !== "ready") { delete state.topics[state.current]; }
  renderSidebar();
  renderMain();
  schedulePoll();
  if (previous === "indexing" && status.state === "ready") {
    toast(`Ready — ${status.chunks} passages indexed in ${status.seconds}s.`
          + (status.enriching ? " Pictures are still being read." : ""));
  }
  if (previousChunks && status.state === "ready" && !status.enriching
      && status.chunks !== previousChunks) {
    toast(`Finished reading the pictures — ${status.chunks} passages now indexed.`);
    delete state.topics[state.current];
  }
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  if (state.status?.state !== "indexing" && !state.status?.enriching) return;
  const name = state.current;
  state.pollTimer = setTimeout(async () => {
    try {
      const status = await api(subjectUrl(name, "/status"));
      if (state.current === name) await refreshSubject(status, false);
    } catch (e) { fail(e); }
  }, 1200);
}

function renderIndexing() {
  const box = $("#indexing");
  const s = state.status;
  const failures = s?.failures || [];
  const broken = s && (s.state === "error" || s.state === "unreadable" || failures.length);
  if (s?.enriching && !broken) {
    // Answers already work; this is the slow second pass over the pictures.
    const e = s.enriching;
    const p = e.pictures;
    box.hidden = false;
    box.classList.remove("error");
    box.replaceChildren(
      el("div", { class: "spinner small" }),
      el("strong", { text: "Answers are ready — still reading the pictures" }),
      el("p", { class: "meta", text: e.current
        ? `${e.current}${p ? ` — picture ${Math.min(p.done + 1, p.total)} of ${p.total}`
                              + (p.page ? ` (page ${p.page})` : "") : ""}`
        : "Diagrams and picture-only slides are being read in the background." }));
    return;
  }
  if (!s || (s.state !== "indexing" && !broken)) { box.hidden = true; return; }
  box.hidden = false;
  box.classList.toggle("error", !!broken);
  if (s.state === "unreadable" || (failures.length && s.state !== "indexing")) {
    box.replaceChildren(icon("cross"), el("div", {},
      el("strong", { text: s.state === "unreadable"
        ? "None of these documents could be read."
        : `Couldn't read ${failures.length} of ${(s.documents || []).length + failures.length} documents.` }),
      ...failures.map((f) => el("p", { class: "meta", text: `${f.name} — ${f.error}` }))));
    return;
  }
  if (s.state === "error") {
    box.replaceChildren(icon("cross"), el("div", {},
      el("strong", { text: "Couldn't build the index." }), el("p", { class: "meta", text: s.error })));
    return;
  }
  const done = s.done || 0;
  const total = s.total || 1;
  let line;
  if (s.stage === "pictures" && s.pictures) {
    const p = s.pictures;
    line = `Reading pictures in ${s.current} — ${Math.min(p.done + 1, p.total)} of ${p.total}`
      + (p.page ? ` (page ${p.page})` : "");
  }
  else if (s.stage === "embedding") line = "Embedding passages and building the search index…";
  else if (s.current) line = `Reading ${s.current} (${Math.min(done + 1, total)} of ${total})`;
  else line = "Getting ready…";
  const slowNote = s.stage === "pictures"
    ? "Pages whose text is thin or missing are read as pictures: text recognition first, the vision model only when that finds nothing."
    : s.slow?.length
    ? "Images and audio go through a vision or speech model first, which can take a minute or more."
    : "The first time also loads the models, so it takes a little longer.";
  const progress = s.stage === "embedding" ? 0.92 : (done / total) * 0.9;
  box.replaceChildren(
    el("div", { class: "spinner" }),
    el("strong", { text: line }),
    el("p", { class: "meta", text: slowNote }),
    el("div", { class: "bar" }, el("i", { style: `width: ${Math.max(4, progress * 100)}%` })),
  );
}

function renderMain() {
  const subject = currentSubject();
  const title = $("#subject-title");
  const meta = $("#subject-meta");
  const hasDocs = subject && subject.documents.length > 0;
  const ready = state.status?.state === "ready";

  title.textContent = subject ? subject.name : "Study Assistant";
  document.title = subject ? `${subject.name} · Study Assistant` : "Study Assistant";
  if (!subject) meta.textContent = "";
  else if (!hasDocs) meta.textContent = "No documents yet";
  else if (ready) meta.replaceChildren(`${subject.documents.length} document${subject.documents.length > 1 ? "s" : ""} · ${state.status.chunks} passages`,
    el("span", { class: "wide-only", text: " · answers draw on all of them" }));
  else meta.textContent = `${subject.documents.length} document${subject.documents.length > 1 ? "s" : ""}`;

  $("#tabs").hidden = !hasDocs;
  for (const tab of document.querySelectorAll("#tabs button")) {
    tab.setAttribute("aria-selected", String(tab.dataset.view === state.view));
  }
  renderIndexing();

  const views = { welcome: $("#view-welcome"), ask: $("#view-ask"), quiz: $("#view-quiz"), progress: $("#view-progress") };
  const show = hasDocs ? state.view : "welcome";
  for (const [key, node] of Object.entries(views)) node.hidden = key !== show;

  if (!hasDocs) {
    if (subject) {
      $("#welcome-title").textContent = `Add material to ${subject.name}`;
      $("#welcome-text").textContent = "Drop lecture notes, papers, slides, photos of the board or recorded lectures into the panel on the left. Everything is read and searched on this computer.";
    } else {
      $("#welcome-title").textContent = "A quiet place to study your own notes";
      $("#welcome-text").textContent = "Create a subject, add your lecture notes, papers, slides, photos or recordings, then ask questions about them or let the assistant quiz you.";
    }
    return;
  }
  if (show === "ask") renderAsk();
  if (show === "quiz") renderQuiz();
  if (show === "progress") renderProgress();
}

function switchView(view) {
  state.view = view;
  setHash();
  closeDrawer();
  renderMain();
}

// ---------------------------------------------------------------- ask

async function renderAsk() {
  const name = state.current;
  const ready = state.status?.state === "ready";
  $("#question").disabled = !ready;
  $("#question").placeholder = ready ? `Ask about ${name}…`
    : state.status?.state === "unreadable" ? "Nothing readable in this subject yet"
    : "Waiting for the documents to be read…";
  updateSend();
  if (!(name in state.chats)) {
    state.chats[name] = [];
    try { state.chats[name] = await api(subjectUrl(name, "/chat")); } catch (e) { fail(e); }
    if (state.current !== name || state.view !== "ask") return;
  }
  drawConversation();
}

function drawConversation() {
  const name = state.current;
  const box = $("#conversation");
  const messages = state.chats[name] || [];
  if (!messages.length) {
    box.replaceChildren(emptyChat());
    return;
  }
  box.replaceChildren(...messages.map((m) => m.role === "user" ? userMessage(m.content) : answerMessage(m)));
  box.scrollTop = box.scrollHeight;
}

function emptyChat() {
  const node = el("div", { class: "empty-chat" },
    el("h2", { text: "What would you like to know?" }),
    el("p", { text: "Ask about anything in this subject's documents. Each answer is written from the three passages that match your question best, and shows which ones they were." }),
  );
  const prompts = el("div", { class: "prompts" });
  node.append(prompts);
  if (state.status?.state === "ready") {
    api(subjectUrl(state.current, "/progress")).then((p) => {
      for (const r of p.recommendations.slice(0, 3)) {
        prompts.append(el("button", { onclick: () => practise(r.topic) },
          el("span", { class: "eyebrow", text: "Quiz me on" }), el("br"), r.label.split(" — ").pop()));
      }
    }).catch(() => {});
  }
  return node;
}

function userMessage(text) {
  return el("div", { class: "msg user" }, el("div", { class: "bubble", text }));
}

function answerMessage(m) {
  const text = el("div", { class: "answer-text" + (m.pending ? " streaming" : "") });
  const card = el("div", { class: "answer" }, text);
  const wrap = el("div", { class: "msg assistant" }, card);
  fillAnswer(wrap, m);
  return wrap;
}

// (Re)draw the parts of an answer card that depend on the message.
function fillAnswer(wrap, m) {
  const card = wrap.querySelector(".answer");
  const text = card.querySelector(".answer-text");
  card.querySelectorAll(".sources, .answer-foot, .thinking").forEach((n) => n.remove());

  if (m.pending && !m.content) {
    text.hidden = true;
    card.prepend(el("div", { class: "thinking" },
      el("span", { class: "dots" }, el("i"), el("i"), el("i")),
      m.sources ? "Writing an answer…" : "Searching your materials…"));
  } else {
    text.hidden = false;
    text.textContent = m.content;
  }
  text.classList.toggle("streaming", !!m.pending && !!m.content);
  const noAnswer = !m.pending && m.content.startsWith("I couldn't find an answer");
  text.classList.toggle("empty", noAnswer);

  const sources = m.sources || [];
  if (sources.length && !m.pending) {
    const cites = el("span", { class: "cites" }, sources.map((s, i) =>
      el("button", { class: "cite", title: sourceLabel(s), "aria-label": `Source ${i + 1}`,
        onclick: (e) => openSource(s, i, m.content, e.currentTarget) }, String(i + 1))));
    text.append(cites);
  }
  if (sources.length) {
    card.append(el("div", { class: "sources" },
      el("span", { class: "label", text: "From" }),
      sources.map((s, i) => el("button", { class: "source-chip", title: s.section ? `${sourceLabel(s)} — ${s.section}` : sourceLabel(s),
        onclick: (e) => openSource(s, i, m.content, e.currentTarget) },
        el("b", { text: i + 1 }), el("span", { text: sourceLabel(s) }))),
    ));
  }
  if (!m.pending && m.seconds !== undefined) {
    card.append(el("div", { class: "answer-foot" },
      el("span", { text: m.time ? ago(m.time) : "" }),
      el("span", { text: `answered in ${m.seconds}s` })));
  }
}

function updateSend() {
  const busy = !!state.asking[state.current];
  const ready = state.status?.state === "ready";
  $("#send").disabled = busy || !ready || !$("#question").value.trim();
}

function autoGrow() {
  const q = $("#question");
  q.style.height = "auto";
  q.style.height = `${Math.min(q.scrollHeight, 180)}px`;
  updateSend();
}

async function ask(event) {
  event?.preventDefault();
  const name = state.current;
  const input = $("#question");
  const question = input.value.trim();
  if (!question || state.asking[name] || state.status?.state !== "ready") return;

  input.value = "";
  autoGrow();
  state.asking[name] = true;
  updateSend();
  const messages = state.chats[name] || (state.chats[name] = []);
  messages.push({ role: "user", content: question });
  const pending = { role: "assistant", content: "", pending: true, sources: null };
  messages.push(pending);
  drawConversation();
  const box = $("#conversation");
  const node = () => (state.current === name ? box.lastElementChild : null);

  try {
    const response = await fetch(subjectUrl(name, "/ask"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).detail; } catch (e) {}
      throw new Error(detail);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const data = block.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
        if (!data) continue;
        const payload = JSON.parse(data);
        if (payload.type === "sources") pending.sources = payload.sources;
        else if (payload.type === "token") pending.content += payload.text;
        else if (payload.type === "done") {
          Object.assign(pending, { content: payload.answer, seconds: payload.seconds, time: Date.now() / 1000, pending: false });
        } else if (payload.type === "error") throw new Error(payload.detail);
        const wrap = node();
        if (wrap) {
          fillAnswer(wrap, pending);
          box.scrollTop = box.scrollHeight;
        }
      }
    }
    if (pending.pending) throw new Error("The answer was cut off.");
  } catch (e) {
    messages.splice(messages.indexOf(pending), 1);
    messages.pop();
    if (state.current === name) { input.value = question; autoGrow(); drawConversation(); }
    fail(e);
  } finally {
    state.asking[name] = false;
    if (state.current === name) updateSend();
  }
}

async function clearChat() {
  if (!state.current) return;
  const ok = await confirmDialog("Clear this conversation?", "The questions and answers for this subject will be deleted. Your quiz progress is kept.", "Clear");
  if (!ok) return;
  try {
    await api(subjectUrl(state.current, "/chat"), { method: "DELETE" });
    state.chats[state.current] = [];
    drawConversation();
  } catch (e) { fail(e); }
}

// ---------------------------------------------------------------- drawer

let activeCite = null;

function openSource(src, i, highlight, trigger, eyebrow) {
  $("#drawer-eyebrow").textContent = eyebrow || `Source ${i + 1}`;
  $("#drawer-title").textContent = src.section || src.file || "Passage";
  const where = src.timecode ? `at ${src.timecode}`
    : src.kind === "slide"
      ? (src.pages && src.pages.includes("-") ? `slides ${src.pages}` : `slide ${src.page}`)
    : src.page ? `page ${src.page}` : null;
  $("#drawer-meta").replaceChildren(
    [src.file, where].filter(Boolean).join(" · "),
    src.from_image ? el("span", { class: "from-image", title:
      "This text was read out of a picture by the vision model, so it may be approximate" },
      icon("image"), "read from a picture") : null);
  $("#drawer-text").replaceChildren(highlighted(src.text, highlight));
  const link = $("#drawer-open");
  link.hidden = !src.file;
  if (src.file) link.href = docUrl(state.current, src.file, src.page);
  link.lastChild.textContent = src.page ? ` Open the document at page ${src.page}` : " Open the file";
  document.querySelectorAll(".cite.active, .source-chip.active").forEach((n) => n.classList.remove("active"));
  if (trigger) trigger.classList.add("active");
  activeCite = trigger;
  const drawer = $("#drawer");
  drawer.classList.add("open");
  $("#shell").classList.add("drawer-open");
  drawer.setAttribute("aria-hidden", "false");
  $("#drawer-close").focus({ preventScroll: true });
}

function closeDrawer() {
  const drawer = $("#drawer");
  if (!drawer.classList.contains("open")) return;
  drawer.classList.remove("open");
  $("#shell").classList.remove("drawer-open");
  drawer.setAttribute("aria-hidden", "true");
  if (activeCite) { activeCite.classList.remove("active"); activeCite.focus({ preventScroll: true }); }
  activeCite = null;
}

// ---------------------------------------------------------------- quiz

function quizState() {
  return state.quiz[state.current] || (state.quiz[state.current] = { question: null, result: null, selected: null, topic: "", loading: false });
}

async function loadTopics(name) {
  if (!state.topics[name]) state.topics[name] = await api(subjectUrl(name, "/topics"));
  return state.topics[name];
}

async function renderQuiz() {
  const name = state.current;
  const q = quizState();
  const select = $("#topic-select");
  const ready = state.status?.state === "ready";
  $("#new-question").disabled = !ready || q.loading;
  select.disabled = !ready;
  if (!ready) {
    select.replaceChildren(el("option", { text: "Waiting for the documents…" }));
    drawCard();
    return;
  }
  let topics;
  try { topics = await loadTopics(name); } catch (e) { fail(e); return; }
  if (state.current !== name || state.view !== "quiz") return;

  const groups = new Map();
  for (const t of topics) {
    if (!groups.has(t.document)) groups.set(t.document, []);
    groups.get(t.document).push(t);
  }
  select.replaceChildren(
    el("option", { value: "", text: "✦ Recommended for me" }),
    ...[...groups].map(([doc, list]) => el("optgroup", { label: doc },
      list.map((t) => el("option", { value: t.id, text: t.section })))),
  );
  select.value = q.topic || "";
  if (!topics.length) {
    $("#new-question").disabled = true;
    $("#quiz-stage").replaceChildren(el("div", { class: "quiz-empty" },
      el("h2", { text: "Not enough text to quiz on" }),
      el("p", { text: "Quiz questions are written from passages of at least 40 words. Add some notes or papers to this subject." })));
    return;
  }
  drawCard();
}

async function newQuestion() {
  const name = state.current;
  const q = quizState();
  if (q.loading) return;
  q.topic = $("#topic-select").value;
  q.loading = true;
  q.question = null;
  q.result = null;
  q.selected = null;
  $("#new-question").disabled = true;
  drawCard();
  try {
    q.question = await api(subjectUrl(name, "/quiz"), { method: "POST", json: { topic: q.topic || null } });
  } catch (e) {
    fail(e);
  } finally {
    q.loading = false;
    if (state.current === name && state.view === "quiz") {
      $("#new-question").disabled = false;
      drawCard();
    }
  }
}

function levelBadge(level, name) {
  return el("span", { class: `level l${level}`, title: "Difficulty adapts to your answers" },
    el("span", { class: "pips" }, [1, 2, 3].map((n) => el("i", { class: n <= level ? "on" : "" }))),
    name);
}

function drawCard() {
  const stage = $("#quiz-stage");
  const q = quizState();
  const levels = state.settings?.levels || {};

  if (q.loading) {
    stage.replaceChildren(el("div", { class: "writing" },
      el("div", { class: "pencil" }, el("i"), el("i"), el("i")),
      el("strong", { text: "Writing a question…" }),
      el("span", { text: "The model reads a passage, writes a question, then checks it can answer it through normal search. This can take a minute." })));
    return;
  }
  const item = q.question;
  if (!item) {
    stage.replaceChildren(el("div", { class: "quiz-empty" },
      el("div", { class: "deck" }, el("i"), el("i"), el("i")),
      el("h2", { text: "Ready when you are" }),
      el("p", { text: "Pick a topic, or let the recommendations choose the one you know least, then deal a card. Questions get harder as you get them right." }),
      el("p", { class: "shortcuts" }, el("kbd", { text: "N" }), " new question · ", el("kbd", { text: "1" }), "–", el("kbd", { text: "4" }), " choose · ", el("kbd", { text: "Enter" }), " check")));
    return;
  }

  const result = q.result;
  // Deal the card in only when it is a new question, not on every redraw.
  const fresh = stage.dataset.dealt !== item.id;
  stage.dataset.dealt = item.id;
  const card = el("article", { class: `index-card${fresh ? " deal" : ""}`, "aria-label": "Quiz question" });
  card.append(el("header", { class: "card-head" },
    levelBadge(item.level, item.level_name || levels[item.level]),
    el("span", { class: "card-topic", title: item.topic_label, text: item.topic_label })));

  const body = el("div", { class: "card-body" }, el("h2", { class: "card-q", text: item.question }));
  if (item.hint) {
    const where = [item.hint.file, item.hint.page ? `page ${item.hint.page}` : null].filter(Boolean).join(", ");
    body.append(el("div", { class: "hint" }, icon("bulb"),
      el("span", {}, "Look in ", el("em", { text: item.hint.section || item.hint.file }), where ? ` (${where})` : "")));
  }

  if (item.options) {
    const options = el("div", { class: "options", role: "group", "aria-label": "Answer options" });
    item.options.forEach((opt, i) => {
      let cls = "option";
      if (result) {
        if (norm(opt) === norm(result.reference)) cls += " right";
        else if (opt === q.selected) cls += " wrong";
      }
      options.append(el("button", {
        class: cls, disabled: !!result, "aria-pressed": String(opt === q.selected),
        onclick: () => { q.selected = opt; drawCard(); },
      }, el("span", { class: "key", text: String.fromCharCode(65 + i) }), el("span", { text: opt })));
    });
    body.append(options);
  } else {
    const input = el("input", { class: "written", placeholder: "Write your answer…", "aria-label": "Your answer",
      autocomplete: "off", disabled: !!result, value: q.selected || "" });
    input.addEventListener("input", () => { q.selected = input.value; actions.querySelector(".btn.primary").disabled = !input.value.trim(); });
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); checkAnswer(); } });
    body.append(input);
    setTimeout(() => { if (!result) input.focus(); }, 30);
  }

  const actions = el("div", { class: "card-actions" });
  if (!result) {
    actions.append(el("button", { class: "btn primary", disabled: !(q.selected || "").trim(), onclick: checkAnswer },
      icon("check"), "Check answer"));
    actions.append(el("p", { class: "meta", text: item.practised
      ? `Mastery of this topic so far: ${pct(item.mastery)}` : "First question on this topic" }));
  }
  body.append(actions);
  card.append(body);

  if (result) {
    card.append(el("div", { class: `verdict ${result.correct ? "ok" : "no"}` },
      el("span", { class: "badge" }, icon(result.correct ? "check" : "cross")),
      el("strong", { text: result.correct ? "Correct" : "Not quite" }),
      el("span", { class: "ref" }, "Answer from the text: ", el("mark", { text: result.reference })),
    ));
    const meter = el("div", { class: "meter" }, el("i", { style: `width: ${result.mastery_before * 100}%` }));
    const change = result.mastery - result.mastery_before;
    card.append(el("div", { class: "mastery-line" },
      el("span", { text: "Mastery" }), meter,
      el("strong", { text: `${pct(result.mastery)} ${change >= 0 ? "▲" : "▼"} ${Math.abs(Math.round(change * 100))}` }),
    ));
    requestAnimationFrame(() => requestAnimationFrame(() => {
      meter.firstChild.style.width = `${result.mastery * 100}%`;
      meter.className = `meter ${result.mastery < 0.4 ? "low" : result.mastery < 0.65 ? "mid" : ""}`;
    }));
    const src = result.source;
    const details = el("details", { class: "card-source" },
      el("summary", {}, `Where this came from — ${sourceLabel(src)}`),
      el("blockquote", { class: "passage" }, highlighted(src.text, result.reference)));
    card.append(details);
    const next = el("button", { class: "btn primary", onclick: newQuestion }, "Next question →");
    card.append(el("div", { class: "card-actions", style: "margin: 18px 28px 0 76px" }, next,
      el("button", { class: "btn ghost small", onclick: () => openSource(src, 0, result.reference, null, "Question source") },
        icon("external"), "Show passage"),
      el("span", { class: "meta", text: `Next time: ${levels[result.next_level] || ""}` })));
    setTimeout(() => next.focus({ preventScroll: true }), 30);
  }
  stage.replaceChildren(card);
}

function norm(s) { return (s || "").toLowerCase().replace(/[^a-z0-9.]+/g, " ").trim(); }

async function checkAnswer() {
  const name = state.current;
  const q = quizState();
  const answer = (q.selected || "").trim();
  if (!q.question || q.result || !answer) return;
  try {
    q.result = await api(subjectUrl(name, "/quiz/answer"), { method: "POST", json: { id: q.question.id, answer } });
    if (state.current === name) drawCard();
  } catch (e) {
    fail(e);
    if (e.status === 404) { q.question = null; drawCard(); }
  }
}

function practise(topicId) {
  const q = quizState();
  q.topic = topicId;
  q.question = null;
  q.result = null;
  switchView("quiz");
  setTimeout(newQuestion, 0);
}

function quizKeys(event) {
  if (state.view !== "quiz" || $("#view-quiz").hidden) return;
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  const q = quizState();
  const key = event.key.toLowerCase();
  if (!typing && key === "n" && !q.loading && state.status?.state === "ready") {
    event.preventDefault();
    newQuestion();
    return;
  }
  if (typing || !q.question || q.result) return;
  const options = q.question.options;
  if (options) {
    const index = "1234".indexOf(key) !== -1 ? "1234".indexOf(key) : "abcd".indexOf(key);
    if (index !== -1 && index < options.length) {
      event.preventDefault();
      q.selected = options[index];
      drawCard();
    } else if (key === "enter" && q.selected) {
      event.preventDefault();
      checkAnswer();
    }
  }
}

// ---------------------------------------------------------------- progress

async function renderProgress() {
  const name = state.current;
  const body = $("#progress-body");
  if (state.status?.state !== "ready") {
    body.replaceChildren(el("div", { class: "quiz-empty" }, el("h2", { text: "Still reading your documents" }),
      el("p", { text: "Progress appears here once the subject is indexed and you've answered a few quiz questions." })));
    return;
  }
  let p;
  try { p = await api(subjectUrl(name, "/progress")); } catch (e) { fail(e); return; }
  if (state.current !== name || state.view !== "progress") return;

  if (!p.answered) {
    body.replaceChildren(el("div", { class: "quiz-empty" },
      el("div", { class: "deck" }, el("i"), el("i"), el("i")),
      el("h2", { text: "No answers yet" }),
      el("p", { text: `This subject has ${p.topics.length} topics. Answer a few quiz questions and your estimated mastery of each will show up here.` }),
      el("p", {}, el("button", { class: "btn primary", style: "margin-top: 18px", onclick: () => practise("") }, icon("card"), "Start a quiz"))));
    return;
  }

  const stat = (v, k, small) => el("div", { class: "stat" },
    el("div", { class: "v" }, v, small ? el("small", { text: small }) : null), el("div", { class: "k", text: k }));

  const revise = el("div", { class: "revise" }, p.recommendations.map((r) => {
    const [doc, section] = r.label.split(" — ");
    const ring = r.attempts
      ? el("div", { class: "ring", style: `--p: ${Math.round(r.mastery * 100)}` }, el("span", { text: pct(r.mastery) }))
      : el("div", { class: "ring none" }, el("span", { text: "new" }));
    return el("div", { class: "revise-card" },
      el("div", { class: "top" }, ring, el("div", { style: "min-width: 0" },
        el("h3", { text: section || r.label }), el("div", { class: "doc-name", text: doc }))),
      el("div", { class: "reason", text: r.reason }),
      el("button", { class: "btn small primary", style: "align-self: flex-start", onclick: () => practise(r.topic) }, icon("card"), "Practise"));
  }));

  const groups = new Map();
  for (const t of p.topics) {
    if (!groups.has(t.document)) groups.set(t.document, []);
    groups.get(t.document).push(t);
  }
  const bars = [];
  const topicGroups = [...groups].map(([doc, rows]) => el("section", { class: "topic-group" },
    el("header", {}, icon(typeIcon(doc || "")), el("h3", { text: doc, title: doc }),
      el("span", { class: "meta", text: `${rows.filter((r) => r.answered).length} of ${rows.length} practised` })),
    rows.map((t) => {
      let mastery;
      if (t.mastery === null) {
        mastery = el("div", { class: "meter-label none", text: "not practised" });
      } else {
        const meter = el("div", { class: `meter ${t.mastery < 0.4 ? "low" : t.mastery < 0.65 ? "mid" : ""}` }, el("i", { style: "width: 0" }));
        bars.push([meter.firstChild, t.mastery]);
        mastery = el("div", { class: "meter-label" }, meter, el("span", { text: pct(t.mastery) }));
      }
      return el("div", { class: "topic-row" },
        el("span", { class: "t", title: t.section, text: t.section }),
        mastery,
        el("span", { class: "n", text: t.answered ? `${t.correct}/${t.answered} right` : "" }),
        el("span", { class: "lvl-text", text: `next: ${t.next_level_name.toLowerCase()}` }),
        el("button", { class: "btn ghost", onclick: () => practise(t.id) }, "Practise"));
    })));

  const recent = el("ul", { class: "recent" }, p.recent.map((a) => el("li", {},
    el("span", { class: `mark ${a.correct ? "ok" : "no"}` }, icon(a.correct ? "check" : "cross")),
    el("span", { class: "q", title: a.question, text: a.question }),
    levelBadge(a.level, (state.settings?.levels || {})[a.level] || ""),
    el("span", { class: "when", text: a.time ? ago(a.time) : "" }))));

  body.replaceChildren(
    el("div", { class: "stats" },
      stat(p.answered, "questions answered"),
      stat(pct(p.accuracy), "answered correctly"),
      stat(p.practised, "topics practised", ` / ${p.topics.length}`)),
    el("div", { class: "section-title" }, el("h2", { text: "Revise next" }),
      el("span", { class: "meta", text: "Weakest topics first; new topics count as 50%." })),
    revise,
    el("div", { class: "section-title" }, el("h2", { text: "Recent answers" })),
    recent,
    el("div", { class: "section-title" }, el("h2", { text: "All topics" }),
      el("span", { class: "meta", text: "Estimated chance of answering a short-answer question right" })),
    ...topicGroups,
    el("p", { class: "danger-zone" }, el("button", { class: "link-btn", onclick: resetProgress, text: "Reset progress for this subject" })),
  );
  requestAnimationFrame(() => requestAnimationFrame(() => {
    for (const [bar, value] of bars) bar.style.width = `${value * 100}%`;
  }));
}

async function resetProgress() {
  const ok = await confirmDialog("Reset your progress?", "Every recorded quiz answer for this subject will be deleted. Written questions are kept.", "Reset");
  if (!ok) return;
  try {
    await api(subjectUrl(state.current, "/progress"), { method: "DELETE" });
    renderProgress();
  } catch (e) { fail(e); }
}

// ---------------------------------------------------------------- theme

function currentTheme() {
  const set = document.documentElement.dataset.theme;
  if (set) return set;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function drawThemeIcon() {
  const dark = currentTheme() === "dark";
  const button = $("#theme-toggle");
  button.replaceChildren(icon(dark ? "sun" : "moon"));
  button.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
}

function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("sa-theme", next); } catch (e) {}
  drawThemeIcon();
}

// ---------------------------------------------------------------- start

function wire() {
  $("#new-subject").addEventListener("submit", createSubject);
  $("#file-input").addEventListener("change", (e) => uploadFiles([...e.target.files]));
  const zone = $("#dropzone");
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file-input").click(); } });
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("over"); });
  }
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("over"); uploadFiles([...e.dataTransfer.files]); });
  $("#use-samples").addEventListener("click", useSamples);

  for (const tab of document.querySelectorAll("#tabs button")) {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
  }
  $("#composer").addEventListener("submit", ask);
  const q = $("#question");
  q.addEventListener("input", autoGrow);
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(); }
  });
  $("#clear-chat").addEventListener("click", clearChat);

  $("#new-question").addEventListener("click", newQuestion);
  $("#topic-select").addEventListener("change", (e) => { quizState().topic = e.target.value; });

  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#theme-toggle").addEventListener("click", toggleTheme);
  $("#open-sidebar").addEventListener("click", openNav);
  $("#close-sidebar").addEventListener("click", closeNav);
  $("#scrim").addEventListener("click", closeNav);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeDrawer(); closeNav(); }
    quizKeys(e);
  });
  window.addEventListener("hashchange", () => {
    const { name, view } = readHash();
    if (name !== state.current || view !== state.view) selectSubject(name, view);
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawThemeIcon);
}

async function start() {
  wire();
  drawThemeIcon();
  try {
    [state.settings] = await Promise.all([api("/api/settings"), loadSubjects()]);
    renderFooter();
  } catch (e) { fail(e); return; }
  const { name, view } = readHash();
  const known = state.subjects.map((s) => s.name);
  const pick = known.includes(name) ? name : known[0] || null;
  await selectSubject(pick, ["ask", "quiz", "progress"].includes(view) ? view : "ask");
}

start();
