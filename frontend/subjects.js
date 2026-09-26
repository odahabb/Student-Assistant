// The subjects grid, and one subject's own page: its documents, the upload
// path that fills them, and the banner shown while they are being read.

import {
  api,
  docUrl,
  fail,
  loadChats,
  loadRecents,
  loadStatus,
  loadSubjects,
  subjectCard,
  subjectUrl,
  toast,
} from "./api.js";
import { deleteChat, updateSend } from "./chat.js";
import {
  $,
  ago,
  bytes,
  confirmDialog,
  el,
  fill,
  hue,
  icon,
  nameDialog,
  openMenu,
  typeIcon,
} from "./dom.js";
import { renderProgress } from "./progress.js";
import { renderQuiz } from "./quiz.js";
import { current, go, hashFor, render } from "./router.js";
import { state } from "./state.js";

// ---------------------------------------------------------- grid

export function renderGrid() {
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
              { icon: "pencil", label: "Rename", run: () => renameSubject(s) },
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

// ---------------------------------------------------------- subject

export async function renderSubject() {
  const name = current();
  const card = subjectCard(name);
  const status = state.status[name] || {};
  const ready = status.state === "ready";
  fill($("#subject-name"), name,
    el("button", { class: "icon-btn rename", "aria-label": `Rename ${name}`,
                   title: "Rename this subject",
                   onclick: () => renameSubject(card) }, icon("pencil")));
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

  // An empty subject shows the upload panel in place of the composer.
  renderOnboard(docs, status);
  renderRail();
  if (!state.chatList[name]) {
    try { await loadChats(name); } catch (e) { fail(e); }
    if (current() !== name) return;
  }
  renderChatRows();
}

// The subject page in one of its three states: empty, so the upload panel
// replaces the composer; indexing, so the composer is disabled and says why;
// or ready.
function renderOnboard(docs, status) {
  const name = current();
  const onboard = $("#subject-onboard");
  const empty = docs === 0;
  onboard.hidden = !empty;
  $("#composer").hidden = empty;
  $("#recents-title").hidden = empty;
  $("#chat-rows").hidden = empty;
  if (!empty) {
    // The line under the composer, which says why it is disabled when it is.
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

export function renderChatRows() {
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

// ---------------------------------------------------------- uploads

export async function uploadFiles(files) {
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

export async function createSubject() {
  const name = await nameDialog({
    title: "New subject",
    text: "A subject holds its own documents, conversations and quiz. "
        + "One per module or topic works well.",
    placeholder: "Machine Learning",
    okLabel: "Create",
  });
  if (!name) return;
  try {
    const created = await api("/api/subjects", { method: "POST", json: { name } });
    await loadSubjects();
    go({ view: "subject", subject: created.name });
    toast(`Created “${created.name}”. Add some documents to it.`);
  } catch (e) { fail(e); }
}

async function renameSubject(subject) {
  const old = subject.name;
  const name = await nameDialog({
    title: "Rename subject",
    text: "The documents, conversations, quiz questions and progress all stay "
        + "with it.",
    value: old,
    okLabel: "Rename",
  });
  if (!name || name === old) return;
  try {
    const renamed = await api(subjectUrl(old), { method: "PATCH", json: { name } });
    // Everything the page holds is keyed by subject name, so it is moved
    // across under the new one rather than reloaded.
    for (const store of [state.status, state.chatList, state.messages, state.quiz,
                         state.topics, state.asking]) {
      if (old in store) { store[renamed.name] = store[old]; delete store[old]; }
    }
    state.status[renamed.name] = renamed.index;
    await Promise.all([loadSubjects(), loadRecents()]);
    if (current() === old) {
      history.replaceState(null, "", hashFor({ ...state.route, subject: renamed.name }));
      state.route.subject = renamed.name;
    }
    render();
    toast(`Renamed to “${renamed.name}”.`);
  } catch (e) { fail(e); }
}

export async function deleteSubject(subject) {
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

// ---------------------------------------------------------- indexing

export function renderIndexing() {
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

export function schedulePoll() {
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
    // The quiz and progress screens also say they are waiting on indexing,
    // so this poll redraws whichever screen is open.
    else if (state.route.view === "quiz") renderQuiz();
    else if (state.route.view === "progress") renderProgress();
    else updateSend();
    schedulePoll();
  }, 1500);
}
