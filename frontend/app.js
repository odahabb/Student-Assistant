// Study Assistant — web interface.
//
// Plain JavaScript as ES modules: no build step and no third-party code. The
// page talks only to the local server in backend/api.py.
//
//   state.js     what every screen reads and redraws from
//   dom.js       building and formatting the page
//   api.js       talking to the server
//   router.js    the location hash, and the render() hub
//   chrome.js    breadcrumbs, sidebar, theme
//   subjects.js  the grid, one subject, uploads, indexing
//   chat.js      asking, answers, the source drawer
//   quiz.js      the quiz screen
//   progress.js  the progress screen
//
// This file is the entry: it attaches every listener, then starts.

import { api, fail, loadRecents, loadSubjects } from "./api.js";
import { ask, autoGrow, closeDrawer } from "./chat.js";
import {
  closeNav,
  drawThemeIcon,
  hideNav,
  renderFooter,
  toggleNav,
  toggleTheme,
} from "./chrome.js";
import { $, closeMenu } from "./dom.js";
import { newQuestion, quizKeys, quizState } from "./quiz.js";
import { current, go, render } from "./router.js";
import { state } from "./state.js";
import { createSubject, uploadFiles } from "./subjects.js";

function wire() {
  $("#new-subject").addEventListener("click", createSubject);
  $("#file-input").addEventListener("change", (e) => uploadFiles([...e.target.files]));
  $("#nav-subjects").addEventListener("click", () => { go({ view: "subjects" }); closeNav(); });
  $("#nav-new").addEventListener("click", () => {
    const name = current() || state.subjects[0]?.name;
    if (!name) { go({ view: "subjects" }); createSubject(); return; }
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
  $("#open-sidebar").addEventListener("click", toggleNav);
  $("#close-sidebar").addEventListener("click", hideNav);
  $("#scrim").addEventListener("click", closeNav);

  // Files dropped anywhere inside a subject are uploaded to it.
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
    // Ctrl/Cmd + \ shows or hides the sidebar.
    if (e.key === "\\" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); toggleNav(); }
    quizKeys(e);
  });
  window.addEventListener("hashchange", render);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawThemeIcon);
}

async function start() {
  wire();
  drawThemeIcon();
  try {
    if (localStorage.getItem("sa-nav-hidden")) $("#shell").classList.add("nav-hidden");
  } catch (e) {}
  try {
    const [settings] = await Promise.all([api("/api/settings"), loadSubjects(), loadRecents()]);
    state.settings = settings;
    renderFooter();
  } catch (e) { fail(e); return; }
  render();
}

start();
