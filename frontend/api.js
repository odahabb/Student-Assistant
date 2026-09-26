// Talking to the server in backend/api.py: the fetch wrapper, the URL
// builders, the loaders that put what comes back into `state`, and the
// toast shown when a request fails.

import { $, el } from "./dom.js";
import { state } from "./state.js";

// ---------------------------------------------------------- helpers

export async function api(path, options = {}) {
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

export const subjectUrl = (name, rest = "") => `/api/subjects/${encodeURIComponent(name)}${rest}`;
export const docUrl = (name, file, page) =>
  subjectUrl(name, `/documents/${encodeURIComponent(file)}`) + (page ? `#page=${page}` : "");

export function toast(message, kind = "") {
  const node = el("div", { class: `toast ${kind}`, text: message });
  $("#toasts").append(node);
  setTimeout(() => node.remove(), kind === "error" ? 6000 : 3200);
}

export function fail(error) {
  console.error(error);
  toast(error.message || String(error), "error");
}

// ---------------------------------------------------------- loading

export async function loadSubjects() {
  state.subjects = await api("/api/subjects");
}

export async function loadRecents() {
  try { state.recents = await api("/api/chats?limit=12"); } catch (e) { state.recents = []; }
}

export function subjectCard(name) {
  return state.subjects.find((s) => s.name === name) || null;
}

export async function loadStatus(name) {
  try {
    state.status[name] = await api(subjectUrl(name, "/status"));
  } catch (e) {
    if (e.status === 404) state.status[name] = { state: "gone" };
    else fail(e);
  }
  return state.status[name];
}

export async function loadChats(name) {
  state.chatList[name] = await api(subjectUrl(name, "/chats"));
  return state.chatList[name];
}
