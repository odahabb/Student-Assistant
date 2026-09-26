// Asking a question and showing the answer: the streamed reply, the answer
// card with its citations and sources, and the drawer a source opens into.

import {
  api,
  docUrl,
  fail,
  loadChats,
  loadRecents,
  loadSubjects,
  subjectUrl,
  toast,
} from "./api.js";
import { renderCrumbs, renderSidebar } from "./chrome.js";
import {
  $,
  ago,
  confirmDialog,
  el,
  fill,
  highlighted,
  icon,
  sourceLabel,
} from "./dom.js";
import { current, go, hashFor } from "./router.js";
import { state } from "./state.js";
import { renderChatRows } from "./subjects.js";

// ---------------------------------------------------------- asking

function messagesFor(name, chatId) {
  const perChat = state.messages[name] || (state.messages[name] = {});
  const key = chatId || "new";
  return perChat[key] || (perChat[key] = []);
}

export async function renderChat() {
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

// (Re)draw the parts of an answer card that depend on the message: the text
// or the waiting line, the citations, the sources, and the footer.
function fillAnswer(wrap, m) {
  const card = wrap.querySelector(".answer");
  const text = card.querySelector(".answer-text");
  card.querySelectorAll(".sources, .answer-foot, .thinking").forEach((n) => n.remove());

  if (m.pending && !m.content) {
    text.hidden = true;
    // The waiting line follows how far the server has got: the turn kind, the
    // question it is looking up, then the sources it found.
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
  // A follow-up retrieves nothing, so it says where its answer came from in
  // place of the sources row.
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

export function updateSend() {
  const { input, send } = composerFields();
  const ready = state.status[current()]?.state === "ready";
  send.disabled = !!state.asking[current()] || !ready || !input.value.trim();
}

export function autoGrow(input) {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  updateSend();
}

export async function ask(event) {
  event?.preventDefault();
  const name = current();
  const { input } = composerFields();
  const question = input.value.trim();
  if (!question || state.asking[name] || state.status[name]?.state !== "ready") return;

  // Asking from anywhere but an open conversation starts a new one.
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

// Take on the id and title the server gave a conversation, and move the
// messages held under "new" to that id.
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

export async function deleteChat(chat) {
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

// ---------------------------------------------------------- drawer

let activeCite = null;

export function openSource(src, i, highlight, trigger, eyebrow) {
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

export function closeDrawer() {
  const drawer = $("#drawer");
  if (!drawer.classList.contains("open")) return;
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  $("#shell").classList.remove("drawer-open");
  if (activeCite) { activeCite.classList.remove("active"); activeCite.focus({ preventScroll: true }); }
  activeCite = null;
}
