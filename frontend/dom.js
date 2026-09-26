// Building and formatting the page: element construction, icons, dialogs,
// popup menus, and the formatters the screens share.
//
// Nothing here knows about subjects, routes or the server.

// ---------------------------------------------------------- header

export const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------- helpers

export function el(tag, attrs = {}, ...children) {
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

export function icon(name, cls) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  if (cls) svg.setAttribute("class", cls);
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

export function pct(x) { return `${Math.round(x * 100)}%`; }

export function ago(seconds) {
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

export function hue(name) {
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.codePointAt(0)) % 360;
  return h;
}

export function typeIcon(filename) {
  const ext = (filename || "").split(".").pop().toLowerCase();
  if (["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif"].includes(ext)) return "image";
  if (["mp3", "wav", "m4a", "flac", "ogg", "mp4", "webm"].includes(ext)) return "audio";
  return "pdf";
}

export function bytes(n) {
  if (!n && n !== 0) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

// How a source is cited: "notes.pdf · p. 4", "deck.pdf · slides 12-15", or
// "lecture.m4a · 12:03-15:40" for a recording, which has no pages.
export function sourceLabel(src) {
  const parts = [src.file || "Unknown file"];
  const many = src.pages && src.pages.includes("-");
  if (src.timecode) parts.push(src.timecode);
  else if (src.kind === "slide") parts.push(many ? `slides ${src.pages}` : `slide ${src.page}`);
  else if (src.page) parts.push(`p. ${src.page}`);
  return parts.join(" · ");
}

// Which button closed a <dialog>, as a promise.
//
// A form with method="dialog" should close its dialog and fire `close`
// carrying the pressed button's value, but in the desktop shell's Chromium
// the dialog closes and `close` never arrives. The answer is therefore taken
// from whichever of `submit`, `cancel` and `close` turns up first, and the
// rest are unsubscribed.
function dialogResult(dialog) {
  const form = dialog.querySelector("form");
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      dialog.removeEventListener("close", onClose);
      dialog.removeEventListener("cancel", onCancel);
      form.removeEventListener("submit", onSubmit);
      if (dialog.open) dialog.close();
      resolve(value);
    };
    function onClose() { finish(dialog.returnValue); }
    function onCancel() { finish("cancel"); }
    // returnValue is only set once the dialog has closed, so on submit the
    // value is read off the button that was pressed.
    function onSubmit(e) { finish(e.submitter ? e.submitter.value : dialog.returnValue); }
    dialog.addEventListener("close", onClose);
    dialog.addEventListener("cancel", onCancel);
    form.addEventListener("submit", onSubmit);
  });
}

// A yes/no dialog; resolves true when the confirming button was pressed.
export async function confirmDialog(title, text, okLabel = "Delete") {
  const dialog = $("#confirm");
  $("#confirm-title").textContent = title;
  $("#confirm-text").textContent = text;
  $("#confirm-ok").textContent = okLabel;
  dialog.returnValue = "";
  dialog.showModal();
  return await dialogResult(dialog) === "ok";
}

// A one-field dialog, used both to name a new subject and to rename one.
// Resolves to the trimmed name, or null if it was cancelled or left empty.
export async function nameDialog({ title, text, value = "", okLabel, placeholder }) {
  const dialog = $("#name-dialog");
  const input = $("#name-input");
  $("#name-title").textContent = title;
  $("#name-text").textContent = text;
  $("#name-ok").textContent = okLabel;
  input.value = value;
  input.placeholder = placeholder || "";
  dialog.returnValue = "";
  dialog.showModal();
  // The caret goes to the end of the current name rather than selecting it,
  // so typing edits the name instead of replacing it.
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
  const pressed = await dialogResult(dialog);
  const typed = input.value.trim();
  return pressed === "ok" && typed ? typed : null;
}

// `text` as a fragment with every case-insensitive occurrence of `needle`
// wrapped in <mark>. Needles under two characters are not marked.
export function highlighted(text, needle) {
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

// A popup menu anchored under `anchor`, kept inside the window, closing on
// the next click anywhere. Used by the ⋮ buttons.
export function openMenu(anchor, items) {
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

export function closeMenu() {
  document.querySelector(".popup")?.remove();
}

// replaceChildren, but with null, undefined and false dropped first, which
// it would otherwise render as the text "null".
export function fill(node, ...children) {
  node.replaceChildren(...children.flat()
    .filter((c) => c !== null && c !== undefined && c !== false));
}
