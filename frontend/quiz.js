// The quiz screen: picking a topic, dealing a question, answering it, and
// the keyboard shortcuts that drive the card.

import { api, fail, subjectCard, subjectUrl } from "./api.js";
import { openSource } from "./chat.js";
import { $, el, highlighted, icon, pct, sourceLabel } from "./dom.js";
import { current, go } from "./router.js";
import { state } from "./state.js";

export function quizState() {
  const name = current();
  return state.quiz[name] || (state.quiz[name] = {
    question: null, result: null, selected: null, topic: "", loading: false });
}

export async function renderQuiz() {
  const name = current();
  const q = quizState();
  const select = $("#topic-select");
  const ready = state.status[name]?.state === "ready";
  $("#new-question").disabled = !ready || q.loading;
  select.disabled = !ready;
  // The controls are hidden until there is something to quiz on.
  $("#quiz-controls").hidden = !ready;
  if (!ready) {
    // Which of the three reasons there is nothing to quiz on: no documents,
    // nothing readable in them, or indexing still running.
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
  // Topics grouped by document, each labelled with how many questions it is
  // worth, above the recommender's own pick.
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

export async function newQuestion() {
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

export function levelBadge(level, name) {
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

export function practise(topicId) {
  const q = quizState();
  q.topic = topicId;
  q.question = null;
  q.result = null;
  go({ view: "quiz", subject: current() });
  setTimeout(newQuestion, 0);
}

export function quizKeys(event) {
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
