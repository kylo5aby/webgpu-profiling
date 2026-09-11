#!/usr/bin/env node
/**
 * Browser counterpart of profile_model.py: operator-level profiling of an ONNX model with
 * onnxruntime-web on the WebGPU EP, inside whatever browser you choose to open the URL in.
 *
 * This script only runs a local HTTP server. It serves the page (web/), the onnxruntime-web
 * build, the model and the .npy inputs, prints a URL, and waits. Open that URL in the browser
 * you want to measure. The page (web/profile.js) creates the session with `enableProfiling`,
 * runs N iterations and calls endProfiling(). In the WASM build ORT's C++ Profiler writes the
 * Chrome-tracing JSON to stdout -> console.log; the page intercepts those lines and POSTs them
 * back here, where they are written verbatim to outputs/<name>.json -- the same format as the
 * Python/native run. The server exits once a result (or an error) has been received.
 *
 * Usage:
 *   node profile_model_web.mjs Z-Image-Turbo-webnn/onnx/vae_decoder_model_f16.onnx \
 *       --input latent_sample=real_latent.npy --iters 20 -o vae_decoder_prof_web.json \
 *       --ort-dist /path/to/onnxruntime/js/web/dist --ort-entry ort.jspi.min.mjs
 *
 * No npm dependencies are required when --ort-dist points at your own onnxruntime-web build.
 */

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.join(HERE, 'web');
const DEFAULT_ORT_DIST = path.join(HERE, 'node_modules', 'onnxruntime-web', 'dist');
const DEFAULT_ORT_ENTRY = 'ort.webgpu.min.mjs';
const OUTPUT_DIR = path.join(HERE, 'outputs');

// ---------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------

const HELP = `Usage: node profile_model_web.mjs MODEL.onnx [options]

Start a local server for profiling an ONNX model's operators with onnxruntime-web (WebGPU EP).
Open the printed URL in the browser you want to measure; the ORT Chrome-tracing JSON is written
to outputs/ when the page finishes, and the server exits.

Options:
  --iters N               Measured runs (default 20). All of them are in the profile.
  --warmup N              Extra un-timed runs before the measured ones (default 0). They are
                          still recorded in the profile (profiling starts at session creation).
  -o, --output NAME       Output JSON filename under outputs/ (default model_prof_web.json).
  --input NAME=PATH       Bind a model input to a real .npy tensor (repeatable). Unbound inputs
                          get scaled-random data matching their declared shape/dtype.
  --external-data PATH    Extra external-data file for the model (repeatable). Files named
                          <model>.data / <model>_data / <stem>.data / <stem>_data / <stem>.onnx.data
                          next to the model are picked up automatically.
  --ort-dist DIR          Directory with the onnxruntime-web build to use (the .mjs entry, the
                          ort-wasm-simd-threaded.*.mjs glue and the .wasm). Default:
                          node_modules/onnxruntime-web/dist. Point it at js/web/dist of your
                          own build to profile a custom (e.g. JSPI) onnxruntime-web.
  --ort-entry FILE        ES-module entry inside --ort-dist (default ort.webgpu.min.mjs; use
                          ort.jspi.min.mjs / ort.jspi.bundle.min.mjs for the JSPI build).
  --layout NHWC|NCHW      WebGPU EP preferredLayout. Default: leave it to ORT (matches Python).
  --threads N             ort.env.wasm.numThreads (default: ORT decides).
  --log-level LEVEL       ort.env.logLevel: verbose|info|warning|error|fatal (default warning).
  --port N                Local server port (default 8787; 0 = random free port).
  --timeout MIN           Exit with an error if no result arrives within MIN minutes (default 0
                          = wait forever).
  --keep-serving          Do not exit after the first result; every page load re-runs the
                          profile and overwrites the output. Ctrl+C to stop.
  -h, --help              Show this help.

Browser tips:
  * WebGPU needs a secure context; http://127.0.0.1 qualifies, so any Chromium-based browser on
    this machine works. Just open the URL there.
  * For sub-100us GPU timings enable chrome://flags/#enable-webgpu-developer-features (or start
    the browser with --enable-webgpu-developer-features); otherwise Chrome quantizes the
    timestamp-query values used for the "Api" events to 100 us.
  * A JSPI build needs Chrome >= 137 (or chrome://flags/#enable-experimental-webassembly-jspi).
`;

function parseCli(argv) {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      iters: { type: 'string', default: '20' },
      warmup: { type: 'string', default: '0' },
      output: { type: 'string', short: 'o', default: 'model_prof_web.json' },
      input: { type: 'string', multiple: true, default: [] },
      'external-data': { type: 'string', multiple: true, default: [] },
      'ort-dist': { type: 'string', default: DEFAULT_ORT_DIST },
      'ort-entry': { type: 'string', default: DEFAULT_ORT_ENTRY },
      layout: { type: 'string' },
      threads: { type: 'string' },
      'log-level': { type: 'string', default: 'warning' },
      port: { type: 'string', default: '8787' },
      timeout: { type: 'string', default: '0' },
      'keep-serving': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });
  if (values.help) {
    process.stdout.write(HELP);
    process.exit(0);
  }
  if (positionals.length !== 1) {
    process.stderr.write(HELP);
    throw new Error('Expected exactly one positional argument: the ONNX model path.');
  }
  const intArg = (name, v) => {
    const n = Number.parseInt(v, 10);
    if (!Number.isInteger(n) || n < 0) throw new Error(`--${name} must be a non-negative integer, got ${v}`);
    return n;
  };
  if (values.layout && !['NHWC', 'NCHW'].includes(values.layout)) {
    throw new Error(`--layout must be NHWC or NCHW, got ${values.layout}`);
  }
  const inputBindings = {};
  for (const pair of values.input) {
    const eq = pair.indexOf('=');
    if (eq < 0) throw new Error(`--input must be NAME=PATH, got: ${pair}`);
    inputBindings[pair.slice(0, eq)] = path.resolve(pair.slice(eq + 1));
  }
  return {
    model: path.resolve(positionals[0]),
    iters: intArg('iters', values.iters),
    warmup: intArg('warmup', values.warmup),
    output: values.output,
    inputBindings,
    externalData: values['external-data'].map((p) => path.resolve(p)),
    ortDist: path.resolve(values['ort-dist']),
    ortEntry: values['ort-entry'],
    layout: values.layout,
    threads: values.threads ? intArg('threads', values.threads) : undefined,
    logLevel: values['log-level'],
    port: intArg('port', values.port),
    timeoutMin: intArg('timeout', values.timeout),
    keepServing: values['keep-serving'],
  };
}

// ---------------------------------------------------------------------------------------------
// Model / input files
// ---------------------------------------------------------------------------------------------

function collectModelFiles(modelPath, explicitExternal) {
  if (!fs.existsSync(modelPath)) throw new Error(`Model not found: ${modelPath}`);
  const dir = path.dirname(modelPath);
  const file = path.basename(modelPath);
  const stem = file.replace(/\.[^.]+$/, '');

  const files = new Map(); // served basename -> absolute path
  files.set(file, modelPath);

  const external = [];
  const addExternal = (abs) => {
    const name = path.basename(abs);
    if (files.has(name)) return;
    files.set(name, abs);
    external.push(name);
  };
  for (const cand of [`${file}.data`, `${file}_data`, `${stem}.data`, `${stem}_data`, `${stem}.onnx.data`, `${stem}.onnx_data`]) {
    const abs = path.join(dir, cand);
    if (fs.existsSync(abs) && fs.statSync(abs).isFile()) addExternal(abs);
  }
  for (const p of explicitExternal) {
    if (!fs.existsSync(p)) throw new Error(`--external-data file not found: ${p}`);
    addExternal(p);
  }
  return { files, external };
}

// ---------------------------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------------------------

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
};

function sendStatus(res, code, text = '') {
  res.writeHead(code, { 'Content-Type': 'text/plain; charset=utf-8' });
  res.end(text);
}

function sendFile(res, absPath) {
  let stat;
  try {
    stat = fs.statSync(absPath);
  } catch {
    return sendStatus(res, 404, 'not found');
  }
  if (!stat.isFile()) return sendStatus(res, 404, 'not found');
  res.writeHead(200, {
    'Content-Type': MIME[path.extname(absPath).toLowerCase()] || 'application/octet-stream',
    'Content-Length': stat.size,
    'Cache-Control': 'no-store',
  });
  fs.createReadStream(absPath).pipe(res);
}

function serveFromDir(res, dir, rel) {
  const abs = path.resolve(dir, rel);
  if (!abs.startsWith(path.resolve(dir) + path.sep)) return sendStatus(res, 403, 'forbidden');
  sendFile(res, abs);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

// ---------------------------------------------------------------------------------------------
// Profile post-processing
// ---------------------------------------------------------------------------------------------

function summarizeProfile(text) {
  const events = JSON.parse(text); // throws if the capture is not a valid tracing file
  const byCat = {};
  for (const ev of events) byCat[ev.cat] = (byCat[ev.cat] || 0) + 1;
  return { count: events.length, byCat };
}

function writeResult(outputName, text, summary) {
  const stats = summarizeProfile(text);
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const outPath = path.join(OUTPUT_DIR, outputName);
  fs.writeFileSync(outPath, text);

  console.log(`Wall-clock per run: ${summary.wallMsPerRun.toFixed(2)} ms`);
  console.log(
    `Profile events: ${stats.count} (${Object.entries(stats.byCat)
      .map(([k, v]) => `${k}=${v}`)
      .join(', ')})`,
  );
  if (!stats.byCat.Api) {
    console.warn(
      'WARNING: no "Api" (GPU timestamp) events in the profile. The WebGPU device did not get ' +
        '"timestamp-query"; check the adapter features printed above.',
    );
  }
  console.log(`Profile written to: ${outPath}`);
  return outPath;
}

// ---------------------------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------------------------

async function main() {
  const args = parseCli(process.argv.slice(2));

  const { files: modelFiles, external } = collectModelFiles(args.model, args.externalData);
  const modelName = path.basename(args.model);

  const ortEntryPath = path.join(args.ortDist, args.ortEntry);
  if (!fs.existsSync(ortEntryPath)) {
    throw new Error(`onnxruntime-web entry not found: ${ortEntryPath} (check --ort-dist / --ort-entry)`);
  }

  const inputFiles = new Map();
  for (const [name, p] of Object.entries(args.inputBindings)) {
    if (!fs.existsSync(p)) throw new Error(`--input ${name}: file not found: ${p}`);
    inputFiles.set(name, p);
  }

  const config = {
    ortEntryUrl: `/ort/${encodeURIComponent(args.ortEntry)}`,
    modelUrl: `/model/${encodeURIComponent(modelName)}`,
    externalData: external.map((name) => ({ path: name, data: `/model/${encodeURIComponent(name)}` })),
    inputs: Object.fromEntries([...inputFiles.keys()].map((n) => [n, `/input/${encodeURIComponent(n)}`])),
    iters: args.iters,
    warmup: args.warmup,
    layout: args.layout || null,
    numThreads: args.threads || 0,
    logLevel: args.logLevel,
    outputName: args.output,
  };

  // The process ends on the first /report/done or /report/error (unless --keep-serving),
  // on --timeout, or on Ctrl+C.
  let resolveExit;
  const exitCodePromise = new Promise((resolve) => {
    resolveExit = resolve;
  });
  const onResult = (code) => {
    if (args.keepServing) {
      console.log('Still serving; reload the page to run again (Ctrl+C to stop).');
      return;
    }
    resolveExit(code);
  };

  const server = http.createServer(async (req, res) => {
    // Cross-origin isolation is required for SharedArrayBuffer (multi-threaded WASM).
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

    const url = new URL(req.url, 'http://localhost');
    const p = decodeURIComponent(url.pathname);

    try {
      if (req.method === 'POST' && p.startsWith('/report/')) {
        const kind = p.slice('/report/'.length);
        const body = await readBody(req);
        switch (kind) {
          case 'log':
            console.log(body);
            break;
          case 'done': {
            const { summary, profile } = JSON.parse(body);
            try {
              writeResult(args.output, profile, summary);
              onResult(0);
            } catch (e) {
              console.error(`ERROR: invalid profile received: ${e.message}`);
              onResult(1);
            }
            break;
          }
          case 'error':
            console.error(`ERROR: page reported an error:\n${body}`);
            onResult(1);
            break;
          default:
            return sendStatus(res, 404, 'unknown report kind');
        }
        return sendStatus(res, 204);
      }

      if (req.method !== 'GET' && req.method !== 'HEAD') return sendStatus(res, 405, 'method not allowed');
      if (p === '/' || p === '/index.html') return sendFile(res, path.join(WEB_DIR, 'index.html'));
      if (p === '/favicon.ico') return sendStatus(res, 204);
      if (p === '/config.json') {
        res.writeHead(200, { 'Content-Type': MIME['.json'], 'Cache-Control': 'no-store' });
        return res.end(JSON.stringify(config));
      }
      if (p.startsWith('/web/')) return serveFromDir(res, WEB_DIR, p.slice('/web/'.length));
      if (p.startsWith('/ort/')) return serveFromDir(res, args.ortDist, p.slice('/ort/'.length));
      if (p.startsWith('/model/')) {
        const abs = modelFiles.get(p.slice('/model/'.length));
        if (abs) return sendFile(res, abs);
      }
      if (p.startsWith('/input/')) {
        const abs = inputFiles.get(p.slice('/input/'.length));
        if (abs) return sendFile(res, abs);
      }
      sendStatus(res, 404, 'not found');
    } catch (e) {
      console.error(`ERROR handling ${req.method} ${p}: ${e.message}`);
      sendStatus(res, 500, e.message);
    }
  });

  await new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(args.port, '127.0.0.1', resolve);
  });
  const url = `http://127.0.0.1:${server.address().port}/`;

  console.log(`ORT   : ${ortEntryPath}`);
  console.log(`Model : ${args.model}${external.length ? `  (+ external data: ${external.join(', ')})` : ''}`);
  for (const [name, p] of inputFiles) console.log(`Input : ${name} <- ${p}`);
  console.log(`Output: ${path.join(OUTPUT_DIR, args.output)}`);
  console.log('');
  console.log(`Open this URL in the browser you want to profile:\n\n    ${url}\n`);
  console.log(
    args.keepServing
      ? 'Waiting for results (--keep-serving: every page load re-runs the profile; Ctrl+C to stop) ...'
      : 'Waiting for the page to finish (Ctrl+C to abort) ...',
  );

  if (args.timeoutMin > 0) {
    setTimeout(() => {
      console.error(`ERROR: no result within ${args.timeoutMin} minutes.`);
      resolveExit(1);
    }, args.timeoutMin * 60_000).unref();
  }
  process.once('SIGINT', () => {
    console.log('\nInterrupted.');
    resolveExit(130);
  });

  const exitCode = await exitCodePromise;
  process.exitCode = exitCode;
  server.close();
  // Give the in-flight 204 response a moment to flush, then exit even if a keep-alive
  // connection from the browser is still open.
  setTimeout(() => process.exit(exitCode), 200);
}

main().catch((e) => {
  console.error(`ERROR: ${e.message}`);
  process.exit(1);
});
