// The location hash names a screen, and render() draws whichever one it
// names:
//
//   #/                        the subjects grid
//   #/s/<subject>             one subject: ask something, or pick up a thread
//   #/s/<subject>/c/<id>      one conversation
//   #/s/<subject>/quiz        quiz cards
//   #/s/<subject>/progress    mastery per topic
//
// render() is the hub every screen is drawn from; go() is how any screen
// moves to another.

import { loadStatus, loadSubjects, subjectCard, toast } from "./api.js";
import { closeDrawer, renderChat } from "./chat.js";
import { renderCrumbs, renderSidebar } from "./chrome.js";
import { $, closeMenu } from "./dom.js";
import { renderProgress } from "./progress.js";
import { renderQuiz } from "./quiz.js";
import { state } from "./state.js";
import {
  renderGrid,
  renderIndexing,
  renderSubject,
  schedulePoll,
} from "./subjects.js";

// ---------------------------------------------------------- routing

export function hashFor(route) {
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

export function go(route) {
  const hash = hashFor(route);
  if (location.hash === hash) render();
  else location.hash = hash;             // hashchange triggers render()
}

export const current = () => state.route.subject;

// ---------------------------------------------------------- render

const VIEWS = {
  subjects: "#view-subjects",
  subject: "#view-subject",
  chat: "#view-chat",
  quiz: "#view-quiz",
  progress: "#view-progress",
};

export async function render() {
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
