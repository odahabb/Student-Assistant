// Study Assistant — web interface.
// Plain JavaScript, no build step and no third-party code: the page talks only
// to the local server in backend/api.py.
//
// Four screens, addressed by the location hash:
//   #/                      the subjects grid
//   #/s/<subject>           one subject: ask something, or pick up a conversation
//   #/s/<subject>/c/<id>    one conversation
//   #/s/<subject>/quiz      quiz cards     #/s/<subject>/progress   mastery

"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  settings: null,
  subjects: [],           // [{name, documents, chats, updated, state, chunks}]
  recents: [],            // recent conversations across subjects
  route: { view: "subjects", subject: null, chat: null },
  status: {},             // subject -> index status
  chatList: {},           // subject -> [{id, title, updated, exchanges}]
  messages: {},           // subject -> chat id (or "new") -> messages
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
  if (!seconds) return "";
  const diff = Date.now() / 1000 - seconds;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  const days = Math.floor(diff / 86400);
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  return new Date(seconds * 1000).toLocaleDateString(undefined,
    { day: "numeric", month: "short" });
}

function hue(name) {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.codePointAt(0)) % 360;
  return h;
}

function typeIcon(filename) {
  const ext = (filename || "").split(".").pop().toLowerCase();
  if (["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif"].includes(ext)) return "image";
  if (["mp3", "wav", "m4a", "flac", "ogg", "mp4", "webm"].includes(ext)) return "audio";
  return "pdf";
}

function bytes(n) {
  if (!n && n !== 0) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
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

// A small popup menu anchored under a button, as the ⋮ rows use.
function openMenu(anchor, items) {
  closeMenu();
  const menu = el("div", { class: "popup" }, items.map((item) =>
    el("button", { class: `popup-item${item.danger ? " danger" : ""}`,
      onclick: () => { closeMenu(); item.run(); } },
      icon(item.icon), item.label)));
  document.body.append(menu);
  const box = anchor.getBoundingClientRect();
  menu.style.top = `${Math.min(box.bottom + 6, window.innerHeight - menu.offsetHeight - 10)}px`;
  menu.style.left = `${Math.min(box.left, window.innerWidth - menu.offsetWidth - 10)}px`;
  setTimeout(() => document.addEventListener("click", closeMenu, { once: true }), 0);
}

function closeMenu() {
  document.querySelector(".popup")?.remove();
}

// replaceChildren stringifies null into the text "null", unlike el() above,
// so anything conditional has to be filtered out first.
function fill(node, ...children) {
  node.replaceChildren(...children.flat()
    .filter((c) => c !== null && c !== undefined && c !== false));
}

// ---------------------------------------------------------------- routing

function hashFor(route) {
  if (!route.subject) return "#/";
  const base = `#/s/${encodeURIComponent(route.subject)}`;
  if (route.view === "chat") return `${base}/c/${route.chat}`;
  if (route.view === "quiz" || route.view === "progress") return `${base}/${route.view}`;
  return base;
}

function readHash() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] !== "s" || !parts[1]) return { view: "subjects", subject: null, chat: null };
  const subject = decodeURIComponent(parts[1]);
  if (parts[2] === "c" && parts[3]) return { view: "chat", subject, chat: parts[3] };
  if (parts[2] === "quiz" || parts[2] === "progress") {
    return { view: parts[2], subject, chat: null };
  }
  return { view: "subject", subject, chat: null };
}

function go(route) {
  const hash = hashFor(route);
  if (location.hash === hash) render();
  else location.hash = hash;             // hashchange triggers render()
}

const current = () => state.route.subject;

// ---------------------------------------------------------------- loading

async function loadSubjects() {
  state.subjects = await api("/api/subjects");
}

async function loadRecents() {
  try { state.recents = await api("/api/chats?limit=12"); } catch (e) { state.recents = []; }
}

function subjectCard(name) {
  return state.subjects.find((s) => s.name === name) || null;
}

async function loadStatus(name) {
  try {
    state.status[name] = await api(subjectUrl(name, "/status"));
  } catch (e) {
    if (e.status === 404) state.status[name] = { state: "gone" };
    else fail(e);
  }
  return state.status[name];
}

async function loadChats(name) {
  state.chatList[name] = await api(subjectUrl(name, "/chats"));
  return state.chatList[name];
}

// ---------------------------------------------------------------- chrome

function renderSidebar() {
  const list = $("#subject-list");
  list.replaceChildren(...state.subjects.map((s) => el("div", {
    class: "subject" + (s.name === current() ? " current" : ""),
  },
    el("button", { class: "subject-open",
      onclick: () => { go({ view: "subject", subject: s.name }); closeNav(); } },
      el("span", { class: "swatch", style: `background: hsl(${hue(s.name)} 38% 52%)` }),
      el("span", { class: "name", text: s.name }),
      el("span", { class: "n", text: s.documents.length || "" })),
    el("button", { class: "icon-btn remove", "aria-label": `Delete ${s.name}`,
      onclick: (e) => { e.stopPropagation(); deleteSubject(s); } }, icon("trash")),
  )));
  if (!state.subjects.length) {
    list.append(el("p", { class: "meta", style: "margin: 2px 10px", text: "No subjects yet." }));
  }

  const recents = $("#recent-list");
  recents.replaceChildren(...state.recents.map((c) => el("button", {
    class: "recent" + (c.id === state.route.chat ? " current" : ""),
    title: `${c.title} — ${c.subject}`,
    onclick: () => { go({ view: "chat", subject: c.subject, chat: c.id }); closeNav(); },
  },
    el("span", { class: "recent-title", text: c.title }),
    el("span", { class: "recent-sub", text: c.subject }),
  )));
  if (!state.recents.length) {
    recents.append(el("p", { class: "meta", style: "margin: 2px 10px",
                            text: "Conversations appear here." }));
  }
}

function renderFooter() {
  const s = state.settings;
  if (!s) return;
  fill($("#side-foot"),
    el("div", {}, "Embeddings ", el("b", { text: s.embedder }), " · ",
       el("b", { text: `${s.chunking} chunks` })),
    el("div", {}, "Retrieval ", el("b", { text: s.hybrid ? "hybrid (BM25 + dense)" : "dense" }),
       ` · top ${s.top_k}`),
    el("div", {}, "Answers by ", el("b", { text: s.answer_model || "Qwen2.5-1.5B-Instruct" }),
       " on ", el("b", { text: (s.device || "").toUpperCase() })),
    s.answer_style === "explain"
      ? el("div", {}, "Quiz answers by ", el("b", { text: s.quiz_model })) : null,
  );
  $("#file-input").accept = (s.extensions || []).map((e) => `.${e}`).join(",");
}

function renderCrumbs() {
  const { view, subject } = state.route;
  const crumbs = $("#crumbs");
  if (!subject) {
    crumbs.replaceChildren(el("span", { class: "crumb here", text: "Subjects" }));
    return;
  }
  const parts = [
    el("button", { class: "crumb", onclick: () => go({ view: "subjects" }), text: "Subjects" }),
    el("span", { class: "crumb-sep", text: "/" }),
  ];
  if (view === "subject") {
    parts.push(el("span", { class: "crumb here", text: subject }));
  } else {
    parts.push(el("button", { class: "crumb",
      onclick: () => go({ view: "subject", subject }), text: subject }));
    parts.push(el("span", { class: "crumb-sep", text: "/" }));
    const label = view === "chat" ? (chatTitle() || "Conversation")
      : view === "quiz" ? "Quiz" : "Progress";
    parts.push(el("span", { class: "crumb here", text: label }));
  }
  crumbs.replaceChildren(...parts);
}

function chatTitle() {
  const found = (state.chatList[current()] || []).find((c) => c.id === state.route.chat);
  return found?.title;
}

// ---------------------------------------------------------------- render

const VIEWS = {
  subjects: "#view-subjects",
  subject: "#view-subject",
  chat: "#view-chat",
  quiz: "#view-quiz",
  progress: "#view-progress",
};

async function render() {
  state.route = readHash();
  const { view, subject } = state.route;
  closeMenu();
  closeDrawer();
  renderCrumbs();
  renderSidebar();
  for (const [key, sel] of Object.entries(VIEWS)) $(sel).hidden = key !== view;
  document.title = subject ? `${subject} · Study Assistant` : "Study Assistant";

  if (!subject) { renderIndexing(); renderGrid(); return; }
  if (!subjectCard(subject)) {
    await loadSubjects();
    renderSidebar();
    if (!subjectCard(subject)) {        // deleted, or a stale link
      toast(`No subject called “${subject}”.`, "error");
      go({ view: "subjects" });
      return;
    }
  }
  if (state.status[subject]?.state !== "ready") await loadStatus(subject);
  renderIndexing();
  schedulePoll();

  if (view === "subject") await renderSubject();
  if (view === "chat") await renderChat();
  if (view === "quiz") await renderQuiz();
  if (view === "progress") await renderProgress();
}

// ---------------------------------------------------------------- grid

function renderGrid() {
  const grid = $("#subject-grid");
  grid.replaceChildren(...state.subjects.map((s) => {
    const docs = s.documents.length;
    return el("article", { class: "card",
      onclick: () => go({ view: "subject", subject: s.name }) },
      el("header", {},
        el("span", { class: "swatch", style: `background: hsl(${hue(s.name)} 38% 52%)` }),
        el("h2", { text: s.name }),
        el("button", { class: "icon-btn card-menu", "aria-label": `More for ${s.name}`,
          onclick: (e) => {
            e.stopPropagation();
            openMenu(e.currentTarget, [
              { icon: "chat", label: "Open", run: () => go({ view: "subject", subject: s.name }) },
              { icon: "card", label: "Quiz", run: () => go({ view: "quiz", subject: s.name }) },
              { icon: "chart", label: "Progress", run: () => go({ view: "progress", subject: s.name }) },
              { icon: "trash", label: "Delete subject", danger: true, run: () => deleteSubject(s) },
            ]);
          } }, icon("more"))),
      el("p", { class: "card-body", text: docs
        ? `${docs} document${docs === 1 ? "" : "s"}`
          + (s.chunks ? ` · ${s.chunks} passages` : "")
        : "No documents yet — add some to start asking." }),
      el("footer", {},
        el("span", { text: s.chats
          ? `${s.chats} conversation${s.chats === 1 ? "" : "s"}` : "No conversations" }),
        el("span", { text: ago(s.updated) })),
    );
  }));
  if (!state.subjects.length) {
    grid.append(el("div", { class: "empty-grid" },
      el("div", { class: "welcome-mark" }, icon("mark")),
      el("h2", { text: "A quiet place to study your own notes" }),
      el("p", { text: "Create a subject above, add your lecture notes, slides, "
                    + "papers, photos or recordings, then ask questions about them "
                    + "or let the assistant quiz you." })));
  }
}

// ---------------------------------------------------------------- subject

async function renderSubject() {
  const name = current();
  const card = subjectCard(name);
  const status = state.status[name] || {};
  const ready = status.state === "ready";
  $("#subject-name").textContent = name;
  const docs = card.documents.length;
  $("#subject-meta").textContent = docs
    ? `${docs} document${docs === 1 ? "" : "s"}`
      + (status.chunks ? ` · ${status.chunks} passages · answers draw on all of them` : "")
    : "No documents yet";

  const question = $("#question");
  question.disabled = !ready;
  question.placeholder = ready ? `Ask about ${name}…`
    : status.state === "unreadable" ? "Nothing readable in this subject yet"
    : docs ? "Reading your documents…" : "Add a document first";
  $("#send").disabled = !ready || !question.value.trim();

  // With nothing uploaded there is no question to ask, so the page leads with
  // the upload instead of a composer the reader cannot type into.
  renderOnboard(docs, status);
  renderRail();
  if (!state.chatList[name]) {
    try { await loadChats(name); } catch (e) { fail(e); }
    if (current() !== name) return;
  }
  renderChatRows();
}

// The subject page has three states and only one of them is a place to ask a
// question. Showing the composer first in the other two puts the disabled
// control above the thing that would enable it.
function renderOnboard(docs, status) {
  const name = current();
  const onboard = $("#subject-onboard");
  const empty = docs === 0;
  onboard.hidden = !empty;
  $("#composer").hidden = empty;
  $("#recents-title").hidden = empty;
  $("#chat-rows").hidden = empty;
  if (!empty) {
    // Indexing is the one state worth narrating: the composer is disabled and
    // the reason is not otherwise on the page.
    const note = $("#composer .composer-note");
    if (status.state === "indexing") {
      fill(note, "Reading your documents — you can ask as soon as this finishes");
    } else if (status.state === "unreadable") {
      fill(note, "None of these documents held readable text");
    } else {
      fill(note, "Answers come only from this subject's documents · ",
           el("kbd", { text: "Enter" }), " to ask");
    }
    return;
  }
  fill(onboard,
    el("div", { class: "onboard-mark" }, icon("files")),
    el("h2", { text: "Add what you are studying" }),
    el("p", { text: "Lecture notes, slides, papers, photos of a whiteboard or a "
                  + "recording of a lecture. Everything after this — questions, "
                  + "quizzes, progress — is written from what you add here." }),
    el("div", { class: "onboard-actions" },
      el("button", { class: "btn primary", onclick: () => $("#file-input").click(),
                     text: "Add documents" }),
      state.settings?.samples
        ? el("button", { class: "link-btn", onclick: useSamples,
                         text: "Or try the sample papers" }) : null),
    el("p", { class: "meta onboard-formats",
              text: "PDF, PowerPoint, Word, images, audio and plain text." }));
}

function renderChatRows() {
  const name = current();
  const rows = $("#chat-rows");
  const chats = state.chatList[name] || [];
  rows.replaceChildren(...chats.map((c) => el("div", { class: "chat-row",
    onclick: () => go({ view: "chat", subject: name, chat: c.id }) },
    el("span", { class: "chat-row-title", text: c.title }),
    el("span", { class: "chat-row-when", text: ago(c.updated) }),
    el("button", { class: "icon-btn row-menu", "aria-label": `More for ${c.title}`,
      onclick: (e) => {
        e.stopPropagation();
        openMenu(e.currentTarget, [
          { icon: "chat", label: "Open", run: () => go({ view: "chat", subject: name, chat: c.id }) },
          { icon: "trash", label: "Delete", danger: true, run: () => deleteChat(c) },
        ]);
      } }, icon("more")),
  )));
  if (!chats.length) {
    rows.append(el("p", { class: "meta", style: "padding: 10px 2px",
      text: "No conversations yet. Ask something above to start one." }));
  }
}

function railEntry(name, label, note, action) {
  return el("div", { class: "rail-row" },
    icon(name),
    el("div", { class: "rail-label" },
      el("span", { text: label }),
      note ? el("span", { class: "rail-note", text: note }) : null),
    action);
}

function renderRail() {
  const name = current();
  const card = subjectCard(name);
  const status = state.status[name] || {};
  const docs = card.documents;
  const size = docs.reduce((total, d) => total + (d.size || 0), 0);
  const perDoc = new Map((status.documents || []).map((d) => [d.name, d]));
  const failures = new Map((status.failures || []).map((f) => [f.name, f.error]));

  fill($("#rail"),
    railEntry("files", "Materials",
      docs.length ? `${docs.length} file${docs.length === 1 ? "" : "s"} · ${bytes(size)}`
                  : "Nothing uploaded yet",
      el("button", { class: "link-btn", onclick: () => $("#file-input").click(), text: "Add" })),
    el("ul", { class: "doc-list" }, docs.map((d) => {
      const info = perDoc.get(d.name);
      return el("li", { class: "doc" },
        icon(typeIcon(d.name)),
        el("a", { href: docUrl(name, d.name), target: "_blank", rel: "noopener",
                  title: d.name, text: d.name }),
        failures.has(d.name)
          ? el("span", { class: "state fail", title: failures.get(d.name), text: "unreadable" })
          : info ? el("span", { class: `state${info.note ? " warn" : ""}`,
              title: info.note ? `${info.chunks} passages — ${info.note}` : `${info.chunks} passages`,
              text: info.pages ? `${info.pages} p` : `${info.chunks} ¶` })
          : null,
        el("button", { class: "icon-btn remove", "aria-label": `Remove ${d.name}`,
          onclick: () => removeDocument(d.name) }, icon("trash")));
    })),
    // The sample papers used to be offered here as well; the empty subject now
    // offers them in the middle of the page, and twice is once too many.
    railEntry("card", "Quiz", "Questions written from your documents",
      el("button", { class: "link-btn", onclick: () => go({ view: "quiz", subject: name }),
                     text: "Open" })),
    railEntry("chart", "Progress", "Mastery per topic, and what to revise",
      el("button", { class: "link-btn", onclick: () => go({ view: "progress", subject: name }),
                     text: "Open" })),
    railEntry("folder", "Index",
      status.state === "ready" ? `${status.chunks} passages · built in ${status.seconds}s`
      : status.state === "indexing" ? "Reading your documents…"
      : status.state === "unreadable" ? "Nothing readable"
      : "Not built yet", null),
    el("p", { class: "privacy" }, icon("lock"),
       " Private — documents and questions stay on this computer."),
  );
}

// ---------------------------------------------------------------- uploads

async function uploadFiles(files) {
  const name = current();
  if (!name || !files.length) return;
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  try {
    const result = await api(subjectUrl(name, "/documents"), { method: "POST", body: form });
    if (result.added.length) toast(`Added ${result.added.length} document(s).`);
    if (result.skipped.length) toast(`Already here: ${result.skipped.join(", ")}`);
    state.status[name] = result.index;
    await loadSubjects();
    render();
  } catch (e) { fail(e); }
  finally { $("#file-input").value = ""; }
}

async function removeDocument(filename) {
  const name = current();
  const ok = await confirmDialog("Remove this document?",
    `“${filename}” will be deleted from this subject's folder and the index rebuilt.`,
    "Remove");
  if (!ok) return;
  try {
    const result = await api(subjectUrl(name, `/documents/${encodeURIComponent(filename)}`),
                             { method: "DELETE" });
    state.status[name] = result.index;
    await loadSubjects();
    render();
  } catch (e) { fail(e); }
}

async function useSamples() {
  const name = current();
  try {
    const result = await api(subjectUrl(name, "/samples"), { method: "POST" });
    toast(`Added ${result.added} sample papers.`);
    state.status[name] = result.index;
    await loadSubjects();
    render();
  } catch (e) { fail(e); }
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
    go({ view: "subject", subject: created.name });
    toast(`Created “${created.name}”. Add some documents to it.`);
  } catch (e) { fail(e); }
}

async function deleteSubject(subject) {
  const documents = subject.documents.length;
  const ok = await confirmDialog(`Delete “${subject.name}”?`,
    `This deletes the subject with ${documents} document${documents === 1 ? "" : "s"}, `
    + "its conversations, its quiz questions and its progress. It cannot be undone.",
    "Delete subject");
  if (!ok) return;
  try {
    await api(subjectUrl(subject.name), { method: "DELETE" });
    for (const store of [state.status, state.chatList, state.messages, state.quiz,
                         state.topics, state.asking]) {
      delete store[subject.name];
    }
    await Promise.all([loadSubjects(), loadRecents()]);
    toast(`Deleted “${subject.name}”.`);
    if (current() === subject.name) go({ view: "subjects" });
    else render();
  } catch (e) { fail(e); }
}

// ---------------------------------------------------------------- indexing

function renderIndexing() {
  const box = $("#indexing");
  const s = state.status[current()];
  const failures = s?.failures || [];
  const broken = s && (s.state === "error" || s.state === "unreadable" || failures.length);
  if (!current() || (!s?.enriching && s?.state !== "indexing" && !broken)) {
    box.hidden = true;
    return;
  }
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
      el("strong", { text: "Couldn't build the index." }),
      el("p", { class: "meta", text: s.error })));
    return;
  }
  if (s.enriching) {
    const p = s.enriching.pictures;
    box.replaceChildren(
      el("div", { class: "spinner small" }),
      el("div", {},
        el("strong", { text: "Answers are ready — still reading the pictures" }),
        el("p", { class: "meta", text: s.enriching.current
          ? `${s.enriching.current}${p ? ` — picture ${Math.min(p.done + 1, p.total)} of ${p.total}` : ""}`
          : "Diagrams and picture-only slides are being read in the background." })));
    return;
  }

  const done = s.done || 0;
  const total = s.total || 1;
  const line = s.stage === "pictures" && s.pictures
    ? `Reading pictures in ${s.current} — ${Math.min(s.pictures.done + 1, s.pictures.total)} of ${s.pictures.total}`
    : s.stage === "embedding" ? "Embedding passages and building the search index…"
    : s.current ? `Reading ${s.current} (${Math.min(done + 1, total)} of ${total})`
    : "Getting ready…";
  box.replaceChildren(
    el("div", { class: "spinner" }),
    el("div", {}, el("strong", { text: line }),
      el("p", { class: "meta", text: s.slow?.length
        ? "Images and audio go through a vision or speech model first, which can take a minute or more."
        : "The first time also loads the models, so it takes a little longer." })),
    el("div", { class: "bar" }, el("i", {
      style: `width: ${Math.max(4, (s.stage === "embedding" ? 0.92 : done / total * 0.9) * 100)}%` })),
  );
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  const name = current();
  const s = state.status[name];
  if (!s || (s.state !== "indexing" && !s.enriching)) return;
  state.pollTimer = setTimeout(async () => {
    const before = s.chunks;
    const status = await loadStatus(name);
    if (current() !== name) return;
    if (status.state === "ready" && before !== status.chunks) {
      delete state.topics[name];
      await loadSubjects();
    }
    renderIndexing();
    if (state.route.view === "subject") renderSubject();
    // The quiz screen says it will catch up when indexing finishes, so it has
    // to be one of the views this poll re-renders.
    else if (state.route.view === "quiz") renderQuiz();
    else if (state.route.view === "progress") renderProgress();
    else updateSend();
    schedulePoll();
  }, 1500);
}

// ---------------------------------------------------------------- asking

function messagesFor(name, chatId) {
  const perChat = state.messages[name] || (state.messages[name] = {});
  const key = chatId || "new";
  return perChat[key] || (perChat[key] = []);
}

async function renderChat() {
  const name = current();
  const chatId = state.route.chat;
  const ready = state.status[name]?.state === "ready";
  $("#thread-question").disabled = !ready;
  updateSend();
  if (!state.chatList[name]) {
    try { await loadChats(name); } catch (e) { fail(e); }
    renderCrumbs();
  }
  const perChat = state.messages[name] || (state.messages[name] = {});
  if (chatId !== "new" && !perChat[chatId] && !state.asking[name]) {
    try {
      perChat[chatId] = (await api(subjectUrl(name, `/chats/${chatId}`))).messages;
    } catch (e) {
      if (e.status === 404) {
        toast("That conversation is gone.", "error");
        go({ view: "subject", subject: name });
        return;
      }
      fail(e);
    }
    if (state.route.chat !== chatId) return;
  }
  drawConversation();
  setTimeout(() => $("#thread-question").focus({ preventScroll: true }), 30);
}

function drawConversation() {
  const box = $("#conversation");
  const messages = messagesFor(current(), state.route.chat);
  box.replaceChildren(...messages.map((m) =>
    m.role === "user" ? userMessage(m.content) : answerMessage(m)));
  if (!messages.length) {
    box.append(el("div", { class: "empty-chat" },
      el("h2", { text: "What would you like to know?" }),
      el("p", { text: "Ask about anything in this subject's documents. Each answer is "
                    + "written from the three passages that match your question best, "
                    + "and shows which ones they were." })));
  }
  box.scrollTop = box.scrollHeight;
}

function userMessage(text) {
  return el("div", { class: "msg user" }, el("div", { class: "bubble", text }));
}

function answerMessage(m) {
  const text = el("div", { class: "answer-text" + (m.pending ? " streaming" : "") });
  const wrap = el("div", { class: "msg assistant" }, el("div", { class: "answer" }, text));
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
    // What the server is doing depends on the kind of turn it decided this
    // was, so say that rather than always claiming to search.
    const waiting = m.sources ? "Writing an answer…"
      : m.turn === "followup" ? "Answering from this conversation…"
      : m.lookup ? `Looking up “${m.lookup}”…`
      : m.turn ? "Searching your materials…"
      : "Reading your question…";
    card.prepend(el("div", { class: "thinking" },
      el("span", { class: "dots" }, el("i"), el("i"), el("i")), waiting));
  } else {
    text.hidden = false;
    text.textContent = m.content;
  }
  text.classList.toggle("streaming", !!m.pending && !!m.content);
  text.classList.toggle("empty", !m.pending && m.content.startsWith("I couldn't find an answer"));

  const sources = m.sources || [];
  if (sources.length && !m.pending) {
    text.append(el("span", { class: "cites" }, sources.map((s, i) =>
      el("button", { class: "cite", title: sourceLabel(s), "aria-label": `Source ${i + 1}`,
        onclick: (e) => openSource(s, i, m.content, e.currentTarget) }, String(i + 1)))));
  }
  if (sources.length) {
    card.append(el("div", { class: "sources" },
      el("span", { class: "label", text: "From" }),
      sources.map((s, i) => el("button", { class: "source-chip",
        title: s.section ? `${sourceLabel(s)} — ${s.section}` : sourceLabel(s),
        onclick: (e) => openSource(s, i, m.content, e.currentTarget) },
        el("b", { text: i + 1 }), el("span", { text: sourceLabel(s) }))),
    ));
  }
  // A follow-up has no sources because none were fetched, not because the
  // search came back empty. Without saying so it reads as a failure.
  if (!m.pending && !sources.length && m.turn === "followup") {
    card.append(el("div", { class: "sources from-conversation" },
      icon("chat"), el("span", { text: "Answered from this conversation" })));
  }
  if (!m.pending && m.lookup) {
    card.append(el("div", { class: "sources looked-up" },
      icon("search"),
      el("span", {}, "Looked up as ", el("b", { text: m.lookup }))));
  }
  if (!m.pending && m.seconds !== undefined) {
    card.append(el("div", { class: "answer-foot" },
      el("span", { text: m.time ? ago(m.time) : "" }),
      el("span", { text: `answered in ${m.seconds}s` })));
  }
}

function composerFields() {
  return state.route.view === "chat"
    ? { input: $("#thread-question"), send: $("#thread-send") }
    : { input: $("#question"), send: $("#send") };
}

function updateSend() {
  const { input, send } = composerFields();
  const ready = state.status[current()]?.state === "ready";
  send.disabled = !!state.asking[current()] || !ready || !input.value.trim();
}

function autoGrow(input) {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  updateSend();
}

async function ask(event) {
  event?.preventDefault();
  const name = current();
  const { input } = composerFields();
  const question = input.value.trim();
  if (!question || state.asking[name] || state.status[name]?.state !== "ready") return;

  // Asking from the subject page opens a new conversation thread.
  const startingNew = state.route.view !== "chat" || state.route.chat === "new";
  const chatId = startingNew ? null : state.route.chat;
  input.value = "";
  autoGrow(input);

  const messages = messagesFor(name, chatId);
  messages.push({ role: "user", content: question });
  const pending = { role: "assistant", content: "", pending: true, sources: null };
  messages.push(pending);
  state.asking[name] = true;

  if (startingNew) go({ view: "chat", subject: name, chat: "new" });
  else drawConversation();
  updateSend();

  const box = $("#conversation");
  const node = () => (current() === name ? box.lastElementChild : null);
  try {
    const response = await fetch(subjectUrl(name, "/ask"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, chat: chatId }),
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
        const data = block.split("\n").filter((l) => l.startsWith("data: "))
                          .map((l) => l.slice(6)).join("\n");
        if (!data) continue;
        const payload = JSON.parse(data);
        if (payload.type === "turn") {
          pending.turn = payload.kind;
          pending.lookup = payload.retrieved_for || null;
        } else if (payload.type === "sources") pending.sources = payload.sources;
        else if (payload.type === "token") pending.content += payload.text;
        else if (payload.type === "done") {
          Object.assign(pending, { content: payload.answer, seconds: payload.seconds,
                                   time: Date.now() / 1000, pending: false });
          if (payload.chat) await settleChat(name, payload.chat, messages);
        } else if (payload.type === "error") throw new Error(payload.detail);
        const wrap = node();
        if (wrap) { fillAnswer(wrap, pending); box.scrollTop = box.scrollHeight; }
      }
    }
    if (pending.pending) throw new Error("The answer was cut off.");
  } catch (e) {
    messages.splice(messages.indexOf(pending), 1);
    messages.pop();
    if (current() === name) { input.value = question; autoGrow(input); drawConversation(); }
    fail(e);
  } finally {
    state.asking[name] = false;
    if (current() === name) updateSend();
  }
}

// The server names a new conversation and gives it an id; adopt both.
async function settleChat(name, chat, messages) {
  const list = state.chatList[name] || (state.chatList[name] = []);
  const existing = list.find((c) => c.id === chat.id);
  if (existing) Object.assign(existing, chat, { exchanges: (existing.exchanges || 0) + 1 });
  else list.unshift({ ...chat, exchanges: 1 });
  list.sort((a, b) => (b.updated || 0) - (a.updated || 0));

  const perChat = state.messages[name] || (state.messages[name] = {});
  perChat[chat.id] = messages;
  if (state.route.view === "chat" && state.route.chat === "new") {
    delete perChat["new"];
    state.route.chat = chat.id;
    history.replaceState(null, "", hashFor(state.route));
  }
  renderCrumbs();
  await Promise.all([loadRecents(), loadSubjects()]);
  renderSidebar();
}

async function deleteChat(chat) {
  const name = current();
  const ok = await confirmDialog("Delete this conversation?",
    `“${chat.title}” and its ${chat.exchanges} question(s) will be deleted. `
    + "Your documents and quiz progress are kept.", "Delete");
  if (!ok) return;
  try {
    await api(subjectUrl(name, `/chats/${chat.id}`), { method: "DELETE" });
    state.chatList[name] = (state.chatList[name] || []).filter((c) => c.id !== chat.id);
    delete (state.messages[name] || {})[chat.id];
    await Promise.all([loadRecents(), loadSubjects()]);
    if (state.route.view === "chat" && state.route.chat === chat.id) {
      go({ view: "subject", subject: name });
    } else {
      renderSidebar();
      renderChatRows();
    }
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
  fill($("#drawer-meta"),
    [src.file, where].filter(Boolean).join(" · "),
    src.from_image ? el("span", { class: "from-image", title:
      "This text was read out of a picture by the vision model, so it may be approximate" },
      icon("image"), "read from a picture") : null);
  $("#drawer-text").replaceChildren(highlighted(src.text, highlight));
  const link = $("#drawer-open");
  link.hidden = !src.file;
  if (src.file) link.href = docUrl(current(), src.file, src.page);
  link.lastChild.textContent = src.page ? ` Open the document at page ${src.page}` : " Open the file";
  document.querySelectorAll(".cite.active, .source-chip.active")
          .forEach((n) => n.classList.remove("active"));
  if (trigger) trigger.classList.add("active");
  activeCite = trigger;
  const drawer = $("#drawer");
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  $("#shell").classList.add("drawer-open");
  $("#drawer-close").focus({ preventScroll: true });
}

function closeDrawer() {
  const drawer = $("#drawer");
  if (!drawer.classList.contains("open")) return;
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  $("#shell").classList.remove("drawer-open");
  if (activeCite) { activeCite.classList.remove("active"); activeCite.focus({ preventScroll: true }); }
  activeCite = null;
}

// ---------------------------------------------------------------- quiz

function quizState() {
  const name = current();
  return state.quiz[name] || (state.quiz[name] = {
    question: null, result: null, selected: null, topic: "", loading: false });
}

async function renderQuiz() {
  const name = current();
  const q = quizState();
  const select = $("#topic-select");
  const ready = state.status[name]?.state === "ready";
  $("#new-question").disabled = !ready || q.loading;
  select.disabled = !ready;
  // A topic picker over no topics and a deal button that cannot deal are
  // furniture; take them away until there is something behind them.
  $("#quiz-controls").hidden = !ready;
  if (!ready) {
    // A quiz cannot be dealt from documents that are not there. Say which of
    // the three reasons it is, and offer the way out of it, rather than
    // showing "Ready when you are" beside an empty dropdown.
    const status = state.status[name] || {};
    const docs = subjectCard(name).documents.length;
    $("#quiz-stage").replaceChildren(el("div", { class: "quiz-empty" },
      el("div", { class: "deck" }, el("i"), el("i"), el("i")),
      el("h2", { text: !docs ? "Nothing to quiz on yet"
                 : status.state === "unreadable" ? "Nothing readable to quiz on"
                 : "Reading your documents…" }),
      el("p", { text: !docs
        ? `Quiz questions are written from your own material. Add some to ${name} and the cards write themselves.`
        : status.state === "unreadable"
        ? "None of these documents held readable text, so there is nothing to write questions from."
        : "Questions are written from the passages being indexed now. This page will catch up on its own." }),
      !docs ? el("button", { class: "btn primary quiz-empty-go",
        onclick: () => go({ view: "subject", subject: name }),
        text: `Add documents to ${name}` }) : null));
    return;
  }
  let topics = state.topics[name];
  if (!topics) {
    try { topics = state.topics[name] = await api(subjectUrl(name, "/topics")); }
    catch (e) { fail(e); return; }
    if (current() !== name || state.route.view !== "quiz") return;
  }

  const groups = new Map();
  for (const t of topics) {
    if (!groups.has(t.document)) groups.set(t.document, []);
    groups.get(t.document).push(t);
  }
  // How many questions a topic is worth is decided from how much material it
  // holds, so say it here: picking a topic is a decision about how long you
  // are about to sit there.
  select.replaceChildren(
    el("option", { value: "", text: "✦ Recommended for me" }),
    ...[...groups].map(([doc, list]) => el("optgroup", { label: doc },
      list.map((t) => el("option", { value: t.id,
        text: t.questions ? `${t.section} — up to ${t.questions} questions`
                          : t.section })))),
  );
  select.value = q.topic || "";
  if (!topics.length) {
    $("#new-question").disabled = true;
    $("#quiz-controls").hidden = true;
    $("#quiz-stage").replaceChildren(el("div", { class: "quiz-empty" },
      el("h2", { text: "Not enough text to quiz on" }),
      el("p", { text: "Quiz questions are written from passages of at least 40 words. "
                    + "Add some notes or papers to this subject." })));
    return;
  }
  drawCard();
}

async function newQuestion() {
  const name = current();
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
    q.question = await api(subjectUrl(name, "/quiz"),
                           { method: "POST", json: { topic: q.topic || null } });
  } catch (e) {
    fail(e);
  } finally {
    q.loading = false;
    if (current() === name && state.route.view === "quiz") {
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
      el("span", { text: "The model reads a passage, writes a question, then checks it can "
                       + "answer it through normal search. This can take a minute." })));
    return;
  }
  const item = q.question;
  if (!item) {
    stage.replaceChildren(el("div", { class: "quiz-empty" },
      el("div", { class: "deck" }, el("i"), el("i"), el("i")),
      el("h2", { text: "Ready when you are" }),
      el("p", { text: "Pick a topic, or let the recommendations choose the one you know "
                    + "least, then deal a card. Questions get harder as you get them right." }),
      el("p", { class: "shortcuts" }, el("kbd", { text: "N" }), " new question · ",
        el("kbd", { text: "1" }), "–", el("kbd", { text: "4" }), " choose · ",
        el("kbd", { text: "Enter" }), " check")));
    return;
  }

  const result = q.result;
  const fresh = stage.dataset.dealt !== item.id;
  stage.dataset.dealt = item.id;
  const card = el("article", { class: `index-card${fresh ? " deal" : ""}`,
                               "aria-label": "Quiz question" });
  card.append(el("header", { class: "card-head" },
    levelBadge(item.level, item.level_name || levels[item.level]),
    el("span", { class: "card-topic", title: item.topic_label, text: item.topic_label })));

  const body = el("div", { class: "card-body" }, el("h2", { class: "card-q", text: item.question }));
  if (item.hint) {
    const where = [item.hint.file, item.hint.page ? `page ${item.hint.page}` : null]
      .filter(Boolean).join(", ");
    body.append(el("div", { class: "hint" }, icon("bulb"),
      el("span", {}, "Look in ", el("em", { text: item.hint.section || item.hint.file }),
        where ? ` (${where})` : "")));
  }

  const actions = el("div", { class: "card-actions" });
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
    const input = el("input", { class: "written", placeholder: "Write your answer…",
      "aria-label": "Your answer", autocomplete: "off", disabled: !!result,
      value: q.selected || "" });
    input.addEventListener("input", () => {
      q.selected = input.value;
      const check = actions.querySelector(".btn.primary");
      if (check) check.disabled = !input.value.trim();
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); checkAnswer(); }
    });
    body.append(input);
    setTimeout(() => { if (!result) input.focus(); }, 30);
  }

  if (!result) {
    actions.append(el("button", { class: "btn primary", disabled: !(q.selected || "").trim(),
                                  onclick: checkAnswer }, icon("check"), "Check answer"));
    actions.append(el("p", { class: "meta", text: item.practised
      ? `Mastery of this topic so far: ${pct(item.mastery)}` : "First question on this topic" }));
  }
  body.append(actions);
  card.append(body);

  if (result) {
    card.append(el("div", { class: `verdict ${result.correct ? "ok" : "no"}` },
      el("span", { class: "badge" }, icon(result.correct ? "check" : "cross")),
      el("strong", { text: result.correct ? "Correct" : "Not quite" }),
      el("span", { class: "ref" }, "Answer from the text: ",
        el("mark", { text: result.reference }))));
    const meter = el("div", { class: "meter" },
                     el("i", { style: `width: ${result.mastery_before * 100}%` }));
    const change = result.mastery - result.mastery_before;
    card.append(el("div", { class: "mastery-line" },
      el("span", { text: "Mastery" }), meter,
      el("strong", { text: `${pct(result.mastery)} ${change >= 0 ? "▲" : "▼"} `
                         + `${Math.abs(Math.round(change * 100))}` })));
    requestAnimationFrame(() => requestAnimationFrame(() => {
      meter.firstChild.style.width = `${result.mastery * 100}%`;
      meter.className = `meter ${result.mastery < 0.4 ? "low" : result.mastery < 0.65 ? "mid" : ""}`;
    }));
    const src = result.source;
    card.append(el("details", { class: "card-source" },
      el("summary", {}, `Where this came from — ${sourceLabel(src)}`),
      el("blockquote", { class: "passage" }, highlighted(src.text, result.reference))));
    const next = el("button", { class: "btn primary", onclick: newQuestion }, "Next question →");
    card.append(el("div", { class: "card-actions card-footer" }, next,
      el("button", { class: "btn ghost small",
        onclick: () => openSource(src, 0, result.reference, null, "Question source") },
        icon("external"), "Show passage"),
      el("span", { class: "meta", text: `Next time: ${levels[result.next_level] || ""}` })));
    setTimeout(() => next.focus({ preventScroll: true }), 30);
  }
  stage.replaceChildren(card);
}

function norm(s) { return (s || "").toLowerCase().replace(/[^a-z0-9.]+/g, " ").trim(); }

async function checkAnswer() {
  const name = current();
  const q = quizState();
  const answer = (q.selected || "").trim();
  if (!q.question || q.result || !answer) return;
  try {
    q.result = await api(subjectUrl(name, "/quiz/answer"),
                         { method: "POST", json: { id: q.question.id, answer } });
    if (current() === name) drawCard();
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
  go({ view: "quiz", subject: current() });
  setTimeout(newQuestion, 0);
}

function quizKeys(event) {
  if (state.route.view !== "quiz") return;
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  const q = quizState();
  const key = event.key.toLowerCase();
  if (!typing && key === "n" && !q.loading && state.status[current()]?.state === "ready") {
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
  const name = current();
  const body = $("#progress-body");
  const status = state.status[name] || {};
  if (status.state !== "ready") {
    // "Still reading your documents" was shown even for a subject that has
    // none, which reads as a stuck page rather than an empty one.
    const docs = subjectCard(name).documents.length;
    body.replaceChildren(el("div", { class: "quiz-empty" },
      el("h2", { text: !docs ? "Nothing to track yet"
                 : status.state === "unreadable" ? "Nothing readable to track"
                 : "Still reading your documents" }),
      el("p", { text: !docs
        ? "Progress follows how you answer quiz questions, and those are written "
          + "from your own material. Add some to get started."
        : status.state === "unreadable"
        ? "None of these documents held readable text, so there are no topics to track."
        : "Progress appears here once the subject is indexed and you've "
          + "answered a few quiz questions." }),
      !docs ? el("button", { class: "btn primary quiz-empty-go",
        onclick: () => go({ view: "subject", subject: name }),
        text: `Add documents to ${name}` }) : null));
    return;
  }
  let p;
  try { p = await api(subjectUrl(name, "/progress")); } catch (e) { fail(e); return; }
  if (current() !== name || state.route.view !== "progress") return;

  if (!p.answered) {
    body.replaceChildren(el("div", { class: "quiz-empty" },
      el("div", { class: "deck" }, el("i"), el("i"), el("i")),
      el("h2", { text: "No answers yet" }),
      el("p", { text: `This subject has ${p.topics.length} topics. Answer a few quiz `
                    + "questions and your estimated mastery of each will show up here." }),
      el("p", {}, el("button", { class: "btn primary", style: "margin-top: 18px",
        onclick: () => practise("") }, icon("card"), "Start a quiz"))));
    return;
  }

  const stat = (v, k, small) => el("div", { class: "stat" },
    el("div", { class: "v" }, v, small ? el("small", { text: small }) : null),
    el("div", { class: "k", text: k }));

  const revise = el("div", { class: "revise" }, p.recommendations.map((r) => {
    const [doc, section] = r.label.split(" — ");
    const ring = r.attempts
      ? el("div", { class: "ring", style: `--p: ${Math.round(r.mastery * 100)}` },
           el("span", { text: pct(r.mastery) }))
      : el("div", { class: "ring none" }, el("span", { text: "new" }));
    return el("div", { class: "revise-card" },
      el("div", { class: "top" }, ring, el("div", { style: "min-width: 0" },
        el("h3", { text: section || r.label }), el("div", { class: "doc-name", text: doc }))),
      el("div", { class: "reason", text: r.reason }),
      el("button", { class: "btn small primary", style: "align-self: flex-start",
        onclick: () => practise(r.topic) }, icon("card"), "Practise"));
  }));

  const groups = new Map();
  for (const t of p.topics) {
    if (!groups.has(t.document)) groups.set(t.document, []);
    groups.get(t.document).push(t);
  }
  const bars = [];
  const topicGroups = [...groups].map(([doc, rows]) => el("section", { class: "topic-group" },
    el("header", {}, icon(typeIcon(doc || "")), el("h3", { text: doc, title: doc }),
      el("span", { class: "meta",
        text: `${rows.filter((r) => r.answered).length} of ${rows.length} practised` })),
    rows.map((t) => {
      let mastery;
      if (t.mastery === null) {
        mastery = el("div", { class: "meter-label none", text: "not practised" });
      } else {
        const meter = el("div", {
          class: `meter ${t.mastery < 0.4 ? "low" : t.mastery < 0.65 ? "mid" : ""}` },
          el("i", { style: "width: 0" }));
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

  const recent = el("ul", { class: "recent-answers" }, p.recent.map((a) => el("li", {},
    el("span", { class: `mark ${a.correct ? "ok" : "no"}` }, icon(a.correct ? "check" : "cross")),
    el("span", { class: "q", title: a.question, text: a.question }),
    levelBadge(a.level, (state.settings?.levels || {})[a.level] || ""),
    el("span", { class: "when", text: ago(a.time) }))));

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
      el("span", { class: "meta",
        text: "Estimated chance of answering a short-answer question right" })),
    ...topicGroups,
    el("p", { class: "danger-zone" }, el("button", { class: "link-btn",
      onclick: resetProgress, text: "Reset progress for this subject" })),
  );
  requestAnimationFrame(() => requestAnimationFrame(() => {
    for (const [bar, value] of bars) bar.style.width = `${value * 100}%`;
  }));
}

async function resetProgress() {
  const ok = await confirmDialog("Reset your progress?",
    "Every recorded quiz answer for this subject will be deleted. Written questions are kept.",
    "Reset");
  if (!ok) return;
  try {
    await api(subjectUrl(current(), "/progress"), { method: "DELETE" });
    renderProgress();
  } catch (e) { fail(e); }
}

// ---------------------------------------------------------------- theme

function currentTheme() {
  return document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
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

function openNav() { $("#shell").classList.add("nav-open"); }
function closeNav() { $("#shell").classList.remove("nav-open"); }

// ---------------------------------------------------------------- start

function wire() {
  $("#new-subject").addEventListener("submit", createSubject);
  $("#file-input").addEventListener("change", (e) => uploadFiles([...e.target.files]));
  $("#nav-subjects").addEventListener("click", () => { go({ view: "subjects" }); closeNav(); });
  $("#nav-new").addEventListener("click", () => {
    const name = current() || state.subjects[0]?.name;
    if (!name) { go({ view: "subjects" }); $("#new-subject-name").focus(); return; }
    delete (state.messages[name] || {})["new"];
    go({ view: "chat", subject: name, chat: "new" });
    closeNav();
  });

  for (const [form, input] of [["#composer", "#question"],
                               ["#thread-composer", "#thread-question"]]) {
    $(form).addEventListener("submit", ask);
    $(input).addEventListener("input", (e) => autoGrow(e.target));
    $(input).addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(); }
    });
  }

  $("#new-question").addEventListener("click", newQuestion);
  $("#topic-select").addEventListener("change", (e) => { quizState().topic = e.target.value; });
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#theme-toggle").addEventListener("click", toggleTheme);
  $("#open-sidebar").addEventListener("click", openNav);
  $("#close-sidebar").addEventListener("click", closeNav);
  $("#scrim").addEventListener("click", closeNav);

  // Dropping files anywhere inside a subject uploads them.
  for (const type of ["dragenter", "dragover"]) {
    document.addEventListener(type, (e) => {
      if (!current() || !e.dataTransfer?.types.includes("Files")) return;
      e.preventDefault();
      $("#main").classList.add("dropping");
    });
  }
  document.addEventListener("dragleave", (e) => {
    if (e.relatedTarget === null) $("#main").classList.remove("dropping");
  });
  document.addEventListener("drop", (e) => {
    if (!current() || !e.dataTransfer?.files.length) return;
    e.preventDefault();
    $("#main").classList.remove("dropping");
    uploadFiles([...e.dataTransfer.files]);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeDrawer(); closeNav(); closeMenu(); }
    quizKeys(e);
  });
  window.addEventListener("hashchange", render);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawThemeIcon);
}

async function start() {
  wire();
  drawThemeIcon();
  try {
    const [settings] = await Promise.all([api("/api/settings"), loadSubjects(), loadRecents()]);
    state.settings = settings;
    renderFooter();
  } catch (e) { fail(e); return; }
  render();
}

start();
