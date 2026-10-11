// A small button in the corner of the focused text field. One click restores the diacritics
// in that field (or in the selected part of it) through the model in the background script.
// Nothing happens while typing: the writer asks, the writer can undo (Ctrl+Z).
//
// Plain script: content scripts cannot be modules. The button lives in a closed shadow root,
// so the page's CSS cannot reach it and ours cannot leak out.
(() => {
  const TITLE = "Adaugă diacritice (Alt+Shift+D)";
  const MAX_CHARS = 20_000; // the model's limit per request (crates/model MAX_CHARS)
  const MARKED = /[ăâîșțĂÂÎȘȚşţŞŢ]/;
  const CEDILLA = { ş: "ș", ţ: "ț", Ş: "Ș", Ţ: "Ț" };
  const INPUT_TYPES = new Set(["text", "search"]);
  const SIZE = 26;
  const ICON = `<svg viewBox="0 0 64 64" aria-hidden="true"><path fill="currentColor" d="M31.66 51Q28.56 51 26.83 49.38Q25.1 47.76 25.1 44.77Q25.1 42.39 26.33 40.91Q27.56 39.44 30.42 38.65Q33.28 37.86 38.17 37.62V31.77Q38.17 29.96 37.75 28.87Q37.32 27.78 36.39 27.28Q35.47 26.78 33.97 26.78Q32.63 26.78 31.16 27.25Q29.69 27.72 28.94 28.47Q30.05 29.7 30.6 30.52Q31.15 31.34 31.34 31.87Q31.54 32.4 31.54 32.83Q31.54 33.76 30.76 34.46Q29.97 35.17 28.52 35.17Q26.92 35.17 26.15 34.38Q25.38 33.59 25.38 32.29Q25.38 30.83 26.59 29.37Q27.8 27.91 30 26.95Q32.2 26 35.11 26Q38.22 26 40.39 26.87Q42.57 27.73 43.71 29.69Q44.84 31.64 44.84 34.9V46.5Q44.84 47.71 45.33 48.23Q45.81 48.76 46.61 48.76Q47.14 48.76 47.63 48.63Q48.13 48.5 48.67 48.23L48.9 48.57Q47.7 49.99 46.3 50.5Q44.91 51 43.62 51Q41.25 51 40.06 49.85Q38.86 48.7 38.51 46.6Q37.84 48.17 36.9 49.14Q35.96 50.12 34.68 50.56Q33.4 51 31.66 51ZM34.72 48.62Q35.81 48.62 36.57 48.09Q37.34 47.55 37.76 46.59Q38.17 45.63 38.17 44.39V38.3Q35.8 38.37 34.38 38.91Q32.96 39.45 32.36 40.67Q31.75 41.89 31.75 44.03Q31.75 46.39 32.5 47.5Q33.25 48.62 34.72 48.62Z"/><path d="M28.5 13.5C31 21.5 42 22.5 45.5 14.5" fill="none" stroke="#fa023c" stroke-width="5" stroke-linecap="round"/></svg>`;

  // --- which fields get the button ---------------------------------------------------------

  /** The element whose text we would restore, or null when `el` is not a text field we handle. */
  function fieldOf(el) {
    if (!el || el.disabled || el.readOnly) return null;
    if (el instanceof HTMLTextAreaElement) return el;
    if (el instanceof HTMLInputElement) return INPUT_TYPES.has(el.type) ? el : null;
    if (el.isContentEditable) return el;
    return null;
  }

  const isForm = (el) => el instanceof HTMLTextAreaElement || el instanceof HTMLInputElement;
  const hasText = (el) => (isForm(el) ? el.value : el.textContent).trim().length > 1;

  // --- talking to the model ----------------------------------------------------------------

  function send(text) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({ type: "restore", text }, (reply) => {
        if (reply && reply.text !== undefined) resolve(reply.text);
        else reject(new Error(reply?.error || chrome.runtime.lastError?.message || "modelul nu a răspuns"));
      });
    });
  }

  /** Pieces of at most MAX_CHARS, cut at a line break or a space when there is one. */
  function chunks(text) {
    const out = [];
    while (text.length > MAX_CHARS) {
      let cut = text.lastIndexOf("\n", MAX_CHARS);
      if (cut < MAX_CHARS / 2) cut = text.lastIndexOf(" ", MAX_CHARS);
      if (cut < MAX_CHARS / 2) cut = MAX_CHARS;
      if (/[\uD800-\uDBFF]/.test(text[cut - 1])) cut--; // never split a surrogate pair
      out.push(text.slice(0, cut));
      text = text.slice(cut);
    }
    out.push(text);
    return out;
  }

  /** The model's output, except where the writer already marked a letter: that stays (cedillas
      become commas). Letters map one to one, so positions line up. */
  function merge(original, restored) {
    const o = Array.from(original);
    const r = Array.from(restored);
    if (o.length !== r.length) return restored; // ponytail: cannot happen by construction; trust the model if it does
    return o.map((ch, i) => (MARKED.test(ch) ? CEDILLA[ch] || ch : r[i])).join("");
  }

  async function restore(text) {
    if (!/[aistAIST]/.test(text)) return text;
    const pieces = [];
    for (const piece of chunks(text)) pieces.push(await send(piece));
    return merge(text, pieces.join(""));
  }

  // --- reading and writing the field --------------------------------------------------------

  /** Text nodes under `host` in document order; only those touching the selection when the
      writer selected something inside the host. A partly selected node is restored whole. */
  function textNodes(host) {
    const sel = host.ownerDocument.getSelection();
    const partial = sel.rangeCount && !sel.isCollapsed && host.contains(sel.anchorNode) && host.contains(sel.focusNode);
    const range = partial ? sel.getRangeAt(0) : null;
    const walker = host.ownerDocument.createTreeWalker(host, NodeFilter.SHOW_TEXT);
    const nodes = [];
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (!range || range.intersectsNode(n)) nodes.push(n);
    }
    return nodes;
  }

  /** execCommand keeps the native undo stack and fires the input events frameworks listen to.
      It is deprecated, but nothing else does both, in either browser. */
  const insert = (text) => document.execCommand("insertText", false, text);

  function writeForm(field, from, to, text, caret) {
    field.focus();
    field.setSelectionRange(from, to);
    if (!insert(text)) {
      field.setRangeText(text, from, to, "preserve");
      field.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
    }
    if (caret === null) field.setSelectionRange(from, to);
    else field.setSelectionRange(caret, caret);
  }

  function writeNodes(host, nodes, texts) {
    const sel = host.ownerDocument.getSelection();
    const saved = sel.rangeCount ? sel.getRangeAt(0).cloneRange() : null;
    // Last node first: an editor that re-renders after an insert does not disturb earlier nodes.
    // Only the changed span is replaced: a node emptied by the replacement is dropped by the
    // browser, and the insertion then lands in the neighbouring element.
    for (let i = nodes.length - 1; i >= 0; i--) {
      const node = nodes[i];
      const text = texts[i];
      const old = node.data;
      if (text === old || !node.isConnected) continue;
      let from = 0;
      while (old[from] === text[from]) from++;
      let to = old.length;
      while (to > from && old[to - 1] === text[to - 1]) to--;
      const whole = from === 0 && to === old.length;
      sel.setBaseAndExtent(node, from, node, to);
      if (whole || !insert(text.slice(from, to))) {
        node.data = text; // ponytail: no native undo for a node changed end to end (a one-letter word in its own element)
        host.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: text }));
      }
    }
    // ponytail: the caret returns where it was when its node survived the edit, else to the end.
    sel.removeAllRanges();
    if (saved && saved.startContainer.isConnected && saved.endContainer.isConnected) {
      sel.addRange(saved);
    } else {
      const end = host.ownerDocument.createRange();
      end.selectNodeContents(host);
      end.collapse(false);
      sel.addRange(end);
    }
  }

  /** Restores the field (or its selection) in place. Returns how many letters changed. */
  async function fix(field) {
    let parts, write;
    if (isForm(field)) {
      const whole = field.selectionStart === field.selectionEnd;
      const from = whole ? 0 : field.selectionStart;
      const to = whole ? field.value.length : field.selectionEnd;
      const caret = whole ? field.selectionStart : null;
      parts = [field.value.slice(from, to)];
      write = ([text]) => writeForm(field, from, to, text, caret);
    } else {
      const nodes = textNodes(field);
      parts = nodes.map((n) => n.data);
      write = (texts) => writeNodes(field, nodes, texts);
    }

    // One request for the whole field, so every word gets its context; pieces come back the
    // same length they went in.
    const joined = parts.join("\n");
    const out = await restore(joined);
    const texts = [];
    let at = 0;
    for (const part of parts) {
      texts.push(out.slice(at, at + part.length));
      at += part.length + 1;
    }
    let changed = 0;
    for (let i = 0; i < joined.length; i++) if (joined[i] !== out[i]) changed++;
    if (changed) write(texts);
    return changed;
  }

  // --- the button --------------------------------------------------------------------------

  const host = document.createElement("div");
  host.dataset.diacritice = "";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `<style>
    :host { all: initial; }
    button { position: fixed; top: 0; left: 0; z-index: 2147483647; display: flex; align-items: center;
      gap: 3px; height: ${SIZE}px; min-width: ${SIZE}px; padding: 0 3px; box-sizing: border-box; margin: 0;
      border: 1px solid #413d3d; border-radius: 7px; background: #1c1919; color: #ece8e8;
      font: 600 12px/1 system-ui, sans-serif; cursor: pointer; opacity: .55;
      box-shadow: 0 1px 2px rgb(0 0 0 / .4); transition: opacity .15s; }
    button:hover, button.busy, button.done, button.error { opacity: 1; }
    button[hidden] { display: none; }
    svg { height: calc(100% - 6px); width: auto; display: block; }
    .busy svg { animation: pulse .7s ease-in-out infinite alternate; }
    .label:empty { display: none; }
    .label { color: #c8ff00; padding: 0 2px; white-space: nowrap; }
    .error .label { color: #fa023c; }
    @keyframes pulse { from { opacity: .25; } to { opacity: 1; } }
  </style><button type="button" tabindex="-1" hidden>${ICON}<span class="label"></span></button>`;
  const button = root.querySelector("button");
  const label = root.querySelector(".label");
  button.title = TITLE;
  button.setAttribute("aria-label", TITLE);

  let field = null;
  let raf = 0;
  let busy = false;
  let timer = 0;

  function place() {
    raf = 0;
    if (!field) return;
    if (!field.isConnected) return detach();
    const r = field.getBoundingClientRect();
    const offscreen = r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth;
    if (r.width < 120 || r.height < 18 || offscreen || (!busy && !label.textContent && !hasText(field))) {
      button.hidden = true;
      return;
    }
    button.hidden = false;
    // Keep clear of a textarea's vertical scrollbar; shrink to fit a one-line input.
    const cs = getComputedStyle(field);
    const scrollbar = Math.max(0, field.offsetWidth - field.clientWidth - parseFloat(cs.borderLeftWidth) - parseFloat(cs.borderRightWidth));
    const single = field instanceof HTMLInputElement || r.height < 2 * SIZE;
    const size = single ? Math.max(18, Math.min(SIZE, Math.round(r.height) - 4)) : SIZE;
    button.style.height = button.style.minWidth = `${size}px`;
    const grip = field instanceof HTMLTextAreaElement && cs.resize !== "none" ? 12 : 0; // the resize handle stays reachable
    const x = r.right - scrollbar - grip - button.offsetWidth - 4;
    const y = single ? r.top + (r.height - size) / 2 : r.bottom - size - 4;
    button.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  }

  const schedule = () => {
    if (!raf) raf = requestAnimationFrame(place);
  };
  const watcher = new ResizeObserver(schedule);

  function reset() {
    clearTimeout(timer);
    label.textContent = "";
    button.title = TITLE;
    button.classList.remove("done", "error");
  }

  function attach(el) {
    if (field === el) return;
    if (field) watcher.unobserve(field);
    reset(); // a result shown on the previous field does not follow the focus
    field = el;
    watcher.observe(el);
    if (!host.isConnected) (document.body || document.documentElement).append(host);
    place();
  }

  function detach() {
    if (field) watcher.unobserve(field);
    field = null;
    button.hidden = true;
  }

  function flash(text, title, error = false) {
    clearTimeout(timer);
    label.textContent = text;
    button.title = title;
    button.classList.toggle("error", error);
    button.classList.add("done");
    place();
    timer = setTimeout(() => {
      reset();
      place();
    }, error ? 4000 : 1800);
  }

  async function run(target) {
    if (busy) return;
    busy = true;
    button.classList.add("busy");
    place();
    try {
      const n = await fix(target);
      const letters = n === 1 ? "o literă schimbată" : `${n} ${n < 20 ? "" : "de "}litere schimbate`;
      flash(`✓ ${n}`, n ? `${letters[0].toUpperCase()}${letters.slice(1)}. Ctrl+Z anulează.` : "Nicio literă de schimbat.");
    } catch (err) {
      flash("!", `Nu a mers: ${err.message || err}`, true);
    } finally {
      busy = false;
      button.classList.remove("busy");
    }
  }

  button.addEventListener("mousedown", (e) => e.preventDefault()); // keep the focus in the field
  button.addEventListener("click", () => field && run(field));

  document.addEventListener("focusin", (e) => {
    const f = fieldOf(e.target);
    if (f) attach(f);
    else detach();
  });
  document.addEventListener("focusout", () => {
    setTimeout(() => {
      if (document.activeElement !== field) detach();
    }, 0);
  });
  document.addEventListener("input", (e) => e.target === field && schedule());
  addEventListener("scroll", schedule, { capture: true, passive: true });
  addEventListener("resize", schedule, { passive: true });

  // The shortcut and the toolbar icon arrive as a message from the background script.
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type !== "fix" || !document.hasFocus()) return;
    const f = fieldOf(document.activeElement);
    if (f) {
      attach(f);
      run(f);
    }
  });

  const focused = fieldOf(document.activeElement);
  if (focused) attach(focused);
})();
