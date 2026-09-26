// The frame around a screen: breadcrumbs, the sidebar and its two
// behaviours, the footer, and the light/dark theme.

import { $, el, fill, hue, icon } from "./dom.js";
import { current, go } from "./router.js";
import { state } from "./state.js";
import { deleteSubject } from "./subjects.js";

// ---------------------------------------------------------- chrome

export function renderSidebar() {
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

export function renderFooter() {
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

export function renderCrumbs() {
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

// ---------------------------------------------------------- theme

function currentTheme() {
  return document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}

export function drawThemeIcon() {
  const dark = currentTheme() === "dark";
  const button = $("#theme-toggle");
  button.replaceChildren(icon(dark ? "sun" : "moon"));
  button.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
}

export function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("sa-theme", next); } catch (e) {}
  drawThemeIcon();
}

// The sidebar has two behaviours. On a narrow screen it slides over the page
// and `nav-open` shows it; on a wide one it is a column of the grid and
// `nav-hidden` takes that column away. Only one class ever applies.
const narrow = () => matchMedia("(max-width: 860px)").matches;

function openNav() { $("#shell").classList.add("nav-open"); }
export function closeNav() { $("#shell").classList.remove("nav-open"); }

function setNavHidden(hidden) {
  $("#shell").classList.toggle("nav-hidden", hidden);
  try { localStorage.setItem("sa-nav-hidden", hidden ? "1" : ""); } catch (e) {}
}

export function toggleNav() {
  if (narrow()) {
    $("#shell").classList.toggle("nav-open");
    return;
  }
  setNavHidden(!$("#shell").classList.contains("nav-hidden"));
}

export function hideNav() {
  if (narrow()) closeNav();
  else setNavHidden(true);
}
