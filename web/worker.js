// Runs the model off the main thread. Expects scripts/build-wasm.sh to have produced
// ./pkg/ (wasm-pack) and ./model/ (config.json + model.safetensors from the export).

import init, { Diacritics } from "./pkg/diacritics_wasm.js";

let model = null;

// Reads the body in chunks to report download progress. GitHub Pages gzips the weights: then
// Content-Length counts compressed bytes while the stream yields decompressed ones, so the
// total is only an estimate (exact: false) and the page shows just the MB received.
async function download(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("Content-Length"));
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const exact = !res.headers.get("Content-Encoding");
  const chunks = [];
  let loaded = 0;
  for (const reader = res.body.getReader(); ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    postMessage({ progress: { loaded, total: Math.max(total, loaded), exact } });
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length; }
  return bytes;
}

async function load() {
  const started = performance.now();
  await init();
  const [config, weights] = await Promise.all([
    fetch("./model/config.json").then((r) => r.text()),
    download("./model/model.safetensors"),
  ]);
  model = new Diacritics(config, weights);
  postMessage({ ready: true, ms: Math.round(performance.now() - started), maxChars: model.max_chars() });
}

onmessage = ({ data }) => {
  if (!model) return;
  const started = performance.now();
  try {
    const text = model.restore(data.text);
    postMessage({ text, chars: data.text.length, ms: performance.now() - started });
  } catch (err) {
    postMessage({ error: String(err) });
  }
};

load().catch((err) => postMessage({ error: `Model load failed: ${err}` }));
