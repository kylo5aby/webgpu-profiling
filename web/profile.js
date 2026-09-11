// Browser side of profile_model_web.mjs.
//
// Mirrors profile_model.py: create an onnxruntime-web session on the WebGPU EP with profiling
// enabled, build feeds (real .npy tensors or scaled-random data), run N measured iterations,
// call endProfiling() and hand the ORT Chrome-tracing JSON back to the Node driver.
//
// The JSON itself is produced by the same C++ Profiler as in the native/Python build. In the
// WebAssembly build it is streamed to std::cout -> console.log, which index.html intercepts into
// `window.__ortStdout` before this module (and ORT) is loaded.

import { parseNpy, castTensorData, randomTensorData, resolveShape, numElements } from './npy.js';

// onnxruntime-web is imported dynamically from the entry chosen by the driver (--ort-dist /
// --ort-entry), so a custom build (e.g. JSPI) can be swapped in without touching this file.
let ort;

const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
const hasDriver = typeof window.__report === 'function';

const report = (kind, payload) => (hasDriver ? window.__report(kind, payload) : Promise.resolve());

function log(msg) {
  logEl.textContent += msg + '\n';
  void report('log', msg);
}

function setStatus(s) {
  statusEl.textContent = s;
  log(`[status] ${s}`);
}

function fmtMeta(m) {
  return m.isTensor ? `${m.name} ${m.type} [${m.shape.join(',')}]` : `${m.name} (non-tensor)`;
}

async function describeAdapter() {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const info = adapter.info || {};
  return {
    vendor: info.vendor || '',
    architecture: info.architecture || '',
    device: info.device || '',
    description: info.description || '',
    features: Array.from(adapter.features || []),
  };
}

async function buildFeeds(session, inputUrls) {
  const feeds = {};
  const unmatched = new Set(Object.keys(inputUrls));
  for (const meta of session.inputMetadata) {
    if (!meta.isTensor) {
      throw new Error(`Input '${meta.name}' is not a tensor; only tensor inputs are supported.`);
    }
    const url = inputUrls[meta.name];
    if (url) {
      unmatched.delete(meta.name);
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Failed to fetch input '${meta.name}' from ${url}: ${resp.status}`);
      const npy = parseNpy(await resp.arrayBuffer());
      log(`  input '${meta.name}': loaded ${url} shape=[${npy.shape.join(',')}] dtype=${npy.type}`);
      let data = npy.data;
      if (npy.type !== meta.type) {
        log(`    casting ${npy.type} -> ${meta.type} to match model's declared input type`);
        data = castTensorData(data, npy.type, meta.type);
      }
      feeds[meta.name] = new ort.Tensor(meta.type, data, npy.shape);
    } else {
      const shape = resolveShape(meta.shape);
      const data = randomTensorData(meta.type, numElements(shape), 0.5);
      log(
        `  input '${meta.name}': WARNING no --input binding; using scaled-random data ` +
          `shape=[${shape.join(',')}] dtype=${meta.type}`,
      );
      feeds[meta.name] = new ort.Tensor(meta.type, data, shape);
    }
  }
  for (const name of unmatched) {
    log(`WARNING: --input '${name}=...' did not match any model input; ignored.`);
  }
  return feeds;
}

function collectProfileLines() {
  // Keep only what the ORT profiler emitted; other console.log traffic (if any) is dropped.
  return window.__ortStdout.filter((l) => l === '[' || l === ']' || l.startsWith('{"cat"'));
}

async function sendProfile(lines) {
  const CHUNK = 1 << 20; // ~1 MB per message keeps CDP payloads small
  const text = lines.join('\n') + '\n';
  for (let off = 0; off < text.length; off += CHUNK) {
    await report('profile-chunk', text.slice(off, off + CHUNK));
  }
  return text;
}

function offerDownload(text, filename) {
  const blob = new Blob([text], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.textContent = `Download ${filename}`;
  a.style.display = 'block';
  a.style.margin = '8px 0';
  statusEl.after(a);
}

async function main() {
  const cfg = await (await fetch('/config.json')).json();

  setStatus(`loading onnxruntime-web from ${cfg.ortEntryUrl} ...`);
  ort = await import(cfg.ortEntryUrl);
  if (/jspi/i.test(cfg.ortEntryUrl) && typeof WebAssembly.Suspending !== 'function') {
    throw new Error(
      'This onnxruntime-web build needs WebAssembly JSPI, which this browser does not expose. ' +
        'Use Chrome >= 137 or launch with --chrome-arg=--enable-experimental-webassembly-jspi.',
    );
  }

  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.proxy = false; // stdout must come from this thread so the console hook sees it
  if (cfg.numThreads) ort.env.wasm.numThreads = cfg.numThreads;
  ort.env.logLevel = cfg.logLevel || 'warning';

  log(`onnxruntime-web ${ort.env.versions.web}`);
  const adapter = await describeAdapter();
  if (!adapter) throw new Error('WebGPU is not available in this browser (navigator.gpu / requestAdapter failed).');
  log(`GPU adapter: ${JSON.stringify(adapter)}`);
  if (!adapter.features.includes('timestamp-query')) {
    log('WARNING: adapter lacks "timestamp-query"; GPU-side "Api" events will be missing from the profile.');
  }

  const ep = { name: 'webgpu' };
  if (cfg.layout) ep.preferredLayout = cfg.layout;
  const sessionOptions = {
    executionProviders: [ep],
    enableProfiling: true,
    graphOptimizationLevel: 'all',
  };
  if (cfg.externalData && cfg.externalData.length) {
    sessionOptions.externalData = cfg.externalData;
    for (const e of cfg.externalData) log(`  external data: ${e.path} <- ${e.data}`);
  }

  setStatus(`creating session for ${cfg.modelUrl} ...`);
  const tCreate = performance.now();
  const session = await ort.InferenceSession.create(cfg.modelUrl, sessionOptions);
  log(`Session created in ${(performance.now() - tCreate).toFixed(0)} ms`);
  for (const m of session.inputMetadata) log(`  input : ${fmtMeta(m)}`);
  for (const m of session.outputMetadata) log(`  output: ${fmtMeta(m)}`);

  setStatus('building feeds ...');
  const feeds = await buildFeeds(session, cfg.inputs || {});

  if (cfg.warmup > 0) {
    setStatus(`Warmup: ${cfg.warmup} runs ...`);
    for (let i = 0; i < cfg.warmup; i++) {
      const t = performance.now();
      await session.run(feeds);
      log(`  warmup ${i + 1}/${cfg.warmup}: ${(performance.now() - t).toFixed(2)} ms`);
    }
  }

  setStatus(`Measured: ${cfg.iters} runs ...`);
  const tStart = performance.now();
  for (let i = 0; i < cfg.iters; i++) {
    const t = performance.now();
    // Outputs default to CPU tensors, so run() only resolves after the GPU->CPU readback:
    // the wall time is real, same as returning numpy in the Python script.
    await session.run(feeds);
    log(`  run ${i + 1}/${cfg.iters}: ${(performance.now() - t).toFixed(2)} ms`);
  }
  const wallMsPerRun = (performance.now() - tStart) / cfg.iters;
  log(`Wall-clock per run: ${wallMsPerRun.toFixed(2)} ms`);

  setStatus('ending profiling ...');
  session.endProfiling();
  await new Promise((r) => setTimeout(r, 50)); // let Emscripten flush stdout
  const lines = collectProfileLines();
  log(`Captured ${lines.length} profiler lines from stdout`);
  if (lines.length === 0) {
    throw new Error('No profiler output captured. Was the console.log hook installed before ORT loaded?');
  }

  setStatus(hasDriver ? 'sending profile to driver ...' : 'preparing download ...');
  const text = await sendProfile(lines);
  await session.release();

  const summary = { wallMsPerRun, iters: cfg.iters, warmup: cfg.warmup || 0, adapter, ortVersion: ort.env.versions.web };
  if (!hasDriver) offerDownload(text, cfg.outputName || 'model_prof_web.json');
  setStatus('done');
  await report('done', summary);
}

main().catch(async (e) => {
  const msg = e && e.stack ? e.stack : String(e);
  setStatus(`ERROR: ${e && e.message ? e.message : e}`);
  logEl.textContent += msg + '\n';
  await report('error', msg);
});
