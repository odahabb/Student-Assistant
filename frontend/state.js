// Study Assistant — the state every screen reads and redraws from.
//
// One object, keyed by subject name. A render function reads it and redraws
// its screen. Nothing here touches the DOM or the server, and this module
// imports nothing, so every other module can depend on it.

export const state = {
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
