// Runs the model off the main thread. Expects scripts/build-wasm.sh to have produced
// ./pkg/ (wasm-pack) and ./model/ (config.json + model.safetensors from the export).

import init, { Diacritics } from "./pkg/diacritics_wasm.js";

let model = null;

async function load() {
  const started = performance.now();
  await init();
  const [config, weights] = await Promise.all([
    fetch("./model/config.json").then((r) => r.text()),
    fetch("./model/model.safetensors").then((r) => r.arrayBuffer()),
  ]);
  model = new Diacritics(config, new Uint8Array(weights));
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
