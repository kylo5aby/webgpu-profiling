#!/usr/bin/env node
/**
 * Browser counterpart of profile_model.py: operator-level profiling of an ONNX model with
 * onnxruntime-web on the WebGPU EP, *inside a real Chromium browser*.
 *
 * It starts a local static server (page, onnxruntime-web dist, model, .npy inputs), launches
 * Chrome/Edge through puppeteer-core with the WebGPU flags, and the page (web/profile.js) creates
 * the session with `enableProfiling`, runs N iterations and calls endProfiling(). In the WASM
 * build ORT's C++ Profiler writes the Chrome-tracing JSON to stdout -> console.log; the page
 * intercepts those lines and streams them back here, where they are written verbatim to
 * outputs/<name>.json in exactly the same format as the Python/native run.
 *
 * Usage:
 *   node profile_model_web.mjs Z-Image-Turbo-webnn/onnx/vae_decoder_model_f16.onnx \
 *       --input latent_sample=real_latent.npy --iters 20 -o vae_decoder_prof_web.json
 *
 *   node profile_model_web.mjs model.onnx --serve-only     # open the printed URL yourself
 */

import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import puppeteer from 'puppeteer-core';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.join(HERE, 'web');
const DEFAULT_ORT_DIST = path.join(HERE, 'node_modules', 'onnxruntime-web', 'dist');
const DEFAULT_ORT_ENTRY = 'ort.webgpu.min.mjs';
const OUTPUT_DIR = path.join(HERE, 'outputs');

// ---------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------

const HELP = `Usage: node profile_model_web.mjs MODEL.onnx [options]

Profile an ONNX model's operators with onnxruntime-web (WebGPU EP) in a real browser and write
the ORT Chrome-tracing JSON to outputs/.

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
  --browser PATH          Chrome/Chromium/Edge executable (default: auto-detect, or $CHROME_PATH).
  --chrome-arg ARG        Extra Chromium command-line switch (repeatable).
  --headless              Run the browser headless. WebGPU in headless mode may fall back to a
                          software adapter on some platforms; default is a visible window.
  --port N                Local server port (default: random free port).
  --timeout MIN           Abort if the run takes longer than MIN minutes (default 0 = no limit).
  --keep-open             Leave the browser open after finishing (or failing) for inspection.
  --serve-only            Only start the server and print the URL; open it in any browser
                          yourself. The page then offers the JSON as a download.
  -h, --help              Show this help.
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
      browser: { type: 'string' },
      'chrome-arg': { type: 'string', multiple: true, default: [] },
      headless: { type: 'boolean', default: false },
      port: { type: 'string', default: '0' },
      timeout: { type: 'string', default: '0' },
      'keep-open': { type: 'boolean', default: false },
      'serve-only': { type: 'boolean', default: false },
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
    browser: values.browser,
    chromeArgs: values['chrome-arg'],
    headless: values.headless,
    port: intArg('port', values.port),
    timeoutMin: intArg('timeout', values.timeout),
    keepOpen: values['keep-open'],
    serveOnly: values['serve-only'],
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
// Static server
// ---------------------------------------------------------------------------------------------

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
};

function sendFile(res, absPath) {
  let stat;
  try {
    stat = fs.statSync(absPath);
  } catch {
    res.writeHead(404);
    res.end('not found');
    return;
  }
  if (!stat.isFile()) {
    res.writeHead(404);
    res.end('not found');
    return;
  }
  res.writeHead(200, {
    'Content-Type': MIME[path.extname(absPath).toLowerCase()] || 'application/octet-stream',
    'Content-Length': stat.size,
    'Cache-Control': 'no-store',
  });
  fs.createReadStream(absPath).pipe(res);
}

function serveFromDir(res, dir, rel) {
  const abs = path.resolve(dir, rel);
  if (!abs.startsWith(path.resolve(dir) + path.sep)) {
    res.writeHead(403);
    res.end('forbidden');
    return;
  }
  sendFile(res, abs);
}

function startServer({ port, config, modelFiles, inputFiles, ortDist }) {
  const server = http.createServer((req, res) => {
    // Cross-origin isolation is required for SharedArrayBuffer (multi-threaded WASM).
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

    const url = new URL(req.url, 'http://localhost');
    const p = decodeURIComponent(url.pathname);
    if (p === '/' || p === '/index.html') return sendFile(res, path.join(WEB_DIR, 'index.html'));
    if (p === '/favicon.ico') {
      res.writeHead(204);
      return res.end();
    }
    if (p === '/config.json') {
      const body = JSON.stringify(config);
      res.writeHead(200, { 'Content-Type': MIME['.json'], 'Cache-Control': 'no-store' });
      return res.end(body);
    }
    if (p.startsWith('/web/')) return serveFromDir(res, WEB_DIR, p.slice('/web/'.length));
    if (p.startsWith('/ort/')) return serveFromDir(res, ortDist, p.slice('/ort/'.length));
    if (p.startsWith('/model/')) {
      const abs = modelFiles.get(p.slice('/model/'.length));
      if (abs) return sendFile(res, abs);
    }
    if (p.startsWith('/input/')) {
      const abs = inputFiles.get(p.slice('/input/'.length));
      if (abs) return sendFile(res, abs);
    }
    res.writeHead(404);
    res.end('not found');
  });
  return new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(port, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

// ---------------------------------------------------------------------------------------------
// Browser
// ---------------------------------------------------------------------------------------------

function findBrowser(explicit) {
  const candidates = [];
  if (explicit) candidates.push(explicit);
  if (process.env.CHROME_PATH) candidates.push(process.env.CHROME_PATH);
  if (process.platform === 'win32') {
    const pf = process.env['ProgramFiles'] || 'C:\\Program Files';
    const pf86 = process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)';
    const local = process.env.LOCALAPPDATA || '';
    candidates.push(
      path.join(pf, 'Google', 'Chrome', 'Application', 'chrome.exe'),
      path.join(pf86, 'Google', 'Chrome', 'Application', 'chrome.exe'),
      path.join(local, 'Google', 'Chrome', 'Application', 'chrome.exe'),
      path.join(local, 'Google', 'Chrome SxS', 'Application', 'chrome.exe'), // Canary
      path.join(pf, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
      path.join(pf86, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    );
  } else if (process.platform === 'darwin') {
    candidates.push(
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
      '/Applications/Chromium.app/Contents/MacOS/Chromium',
      '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    );
  } else {
    for (const name of ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser', 'microsoft-edge']) {
      try {
        candidates.push(execFileSync('which', [name], { encoding: 'utf8' }).trim());
      } catch {
        /* not installed */
      }
    }
  }
  const found = candidates.find((c) => c && fs.existsSync(c));
  if (!found) {
    throw new Error(
      'No Chrome/Chromium/Edge found. Pass --browser PATH or set CHROME_PATH.\n  Tried:\n  ' + candidates.join('\n  '),
    );
  }
  return found;
}

function chromeArgs(extra) {
  const args = [
    '--enable-unsafe-webgpu',
    // Unquantized timestamp queries; without it Chrome rounds GPU timestamps to 100us.
    '--enable-webgpu-developer-features',
    '--enable-dawn-features=allow_unsafe_apis',
    '--ignore-gpu-blocklist',
    '--no-first-run',
    '--no-default-browser-check',
    '--window-size=1100,800',
  ];
  if (process.platform === 'linux') args.push('--enable-features=Vulkan');
  return args.concat(extra);
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

  const { server, port } = await startServer({ port: args.port, config, modelFiles, inputFiles, ortDist: args.ortDist });
  const url = `http://127.0.0.1:${port}/`;
  console.log(`ORT   : ${ortEntryPath}`);
  console.log(`Model : ${args.model}${external.length ? `  (+ external data: ${external.join(', ')})` : ''}`);
  for (const [name, p] of inputFiles) console.log(`Input : ${name} <- ${p}`);
  console.log(`Server: ${url}`);

  if (args.serveOnly) {
    console.log('Serve-only mode: open the URL above in a WebGPU-capable browser; Ctrl+C to stop.');
    await new Promise((resolve) => process.once('SIGINT', resolve));
    server.close();
    return;
  }

  const executablePath = findBrowser(args.browser);
  console.log(`Browser: ${executablePath}${args.headless ? ' (headless)' : ''}`);
  const browser = await puppeteer.launch({
    executablePath,
    headless: args.headless,
    args: chromeArgs(args.chromeArgs),
    ignoreDefaultArgs: ['--disable-gpu'],
    protocolTimeout: 2 ** 31 - 1, // page work can take many minutes; never let CDP time out
  });

  const chunks = [];
  let finished = false;
  const outcome = new Promise((resolve, reject) => {
    const fail = (e) => {
      if (!finished) {
        finished = true;
        reject(e instanceof Error ? e : new Error(String(e)));
      }
    };
    browser.on('disconnected', () =>
      fail('Browser disconnected before the run finished (GPU process crash / device lost / window closed?).'),
    );
    if (args.timeoutMin > 0) {
      setTimeout(() => fail(`Timed out after ${args.timeoutMin} minutes.`), args.timeoutMin * 60_000).unref();
    }

    (async () => {
      const page = await browser.newPage();
      page.on('pageerror', (e) => console.error(`[page error] ${e.message}`));
      page.on('console', (msg) => {
        const t = msg.type();
        if (t === 'error' || t === 'warning') console.error(`[browser ${t}] ${msg.text()}`);
      });
      await page.exposeFunction('__report', (kind, payload) => {
        switch (kind) {
          case 'log':
            console.log(payload);
            break;
          case 'profile-chunk':
            chunks.push(payload);
            break;
          case 'done':
            if (!finished) {
              finished = true;
              resolve(payload);
            }
            break;
          case 'error':
            fail(`Page reported an error:\n${payload}`);
            break;
          default:
            console.warn(`unknown report kind: ${kind}`);
        }
      });
      await page.goto(url, { waitUntil: 'load' });
    })().catch(fail);
  });

  let exitCode = 0;
  try {
    const summary = await outcome;
    const text = chunks.join('');
    const stats = summarizeProfile(text);

    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    const outPath = path.join(OUTPUT_DIR, args.output);
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
          '"timestamp-query"; check the adapter features printed above and the Chrome flags.',
      );
    }
    console.log(`Profile written to: ${outPath}`);
  } catch (e) {
    exitCode = 1;
    console.error(`ERROR: ${e.message}`);
  } finally {
    if (args.keepOpen && browser.connected) {
      console.log('--keep-open: browser left running; close the window to exit.');
      await new Promise((resolve) => browser.once('disconnected', resolve));
    } else if (browser.connected) {
      await browser.close().catch(() => {});
    }
    server.close();
  }
  process.exit(exitCode);
}

main().catch((e) => {
  console.error(`ERROR: ${e.message}`);
  process.exit(1);
});
