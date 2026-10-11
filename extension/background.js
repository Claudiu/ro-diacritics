// Loads the model once and answers "restore" requests from content scripts. Chrome runs this
// as a service worker that it stops when idle, so the first request after a pause pays the
// load again (a few hundred ms); Firefox runs it as an event page with the same shape.

import init, { Diacritics } from "./pkg/diacritics_wasm.js";

let loading = null;

async function load() {
  await init();
  const [config, weights] = await Promise.all([
    fetch(chrome.runtime.getURL("model/config.json")).then((r) => r.text()),
    fetch(chrome.runtime.getURL("model/model.safetensors")).then((r) => r.arrayBuffer()),
  ]);
  return new Diacritics(config, new Uint8Array(weights));
}

chrome.runtime.onMessage.addListener((msg, _sender, respond) => {
  if (msg.type !== "restore") return false;
  (loading ??= load()).then(
    (model) => respond({ text: model.restore(msg.text) }),
    (err) => {
      loading = null; // try again next time rather than cache the failure
      respond({ error: String(err) });
    },
  );
  return true; // the reply is asynchronous
});

// The keyboard shortcut and the toolbar icon both mean "fix the field I am in".
function fix(tab) {
  if (!tab?.id) return;
  try {
    chrome.tabs.sendMessage(tab.id, { type: "fix" })?.catch?.(() => {});
  } catch {
    // no content script in this tab (browser pages, store pages)
  }
}

chrome.commands.onCommand.addListener((command, tab) => command === "fix" && fix(tab));
chrome.action.onClicked.addListener(fix);
