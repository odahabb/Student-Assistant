// The progress screen: mastery per topic, what to revise next, and the
// most recent answers.

import { api, fail, subjectCard, subjectUrl } from "./api.js";
import { $, ago, confirmDialog, el, icon, pct, typeIcon } from "./dom.js";
import { levelBadge, practise } from "./quiz.js";
import { current, go } from "./router.js";
import { state } from "./state.js";

export async function renderProgress() {
  const name = current();
  const body = $("#progress-body");
  const status = state.status[name] || {};
  if (status.state !== "ready") {
    // The same three reasons as the quiz screen: no documents, nothing
    // readable, or indexing still running.
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
