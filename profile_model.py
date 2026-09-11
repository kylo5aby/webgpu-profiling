"""Generic operator-level profiler for ONNX models on WebGPU (or CPU).

Binds an execution provider, runs measured iterations with ONNX Runtime profiling enabled,
and writes the raw ORT tracing JSON to outputs/ for later analysis.

Most models are static-shape (no data-dependent ops), so timings depend only on shape/dtype --
random data is fine for pure timing. But some models are sensitive to NaN/denormal artifacts in
random noise, so you can bind specific inputs to real .npy tensors with --input NAME=PATH (e.g.
the VAE decoder's `latent_sample`, dumped by dump_latent.py). Any input not bound this way falls
back to scaled-random data matching its declared shape/dtype.

Usage:
    python profile_model.py Z-Image-Turbo-webnn/onnx/vae_decoder_model_f16.onnx \
        --ep WebGPU --input latent_sample=real_latent.npy --iters 20 -o vae_decoder_prof.json
"""

import argparse
import os
import shutil
import time

import numpy as np
import onnxruntime as ort

OUTPUT_DIR = "outputs"

_DTYPE_MAP = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


def _np_dtype(onnx_type: str) -> np.dtype:
    return _DTYPE_MAP.get(onnx_type, np.float32)


def _resolve_shape(shape) -> list:
    return [d if isinstance(d, int) and d > 0 else 1 for d in shape]


def build_session(model_path: str, ep: str):
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = os.path.splitext(os.path.basename(model_path))[0] + "_prof"
    if ep == "WebGPU":
        providers = ["WebGpuExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, so, providers=providers)
    actual = sess.get_providers()
    print(f"Requested EP: {ep}  ->  session providers: {actual}")
    if ep == "WebGPU" and "WebGpuExecutionProvider" not in actual:
        raise RuntimeError("WebGPU requested but not active in this onnxruntime build.")
    return sess


def build_feeds(sess, input_bindings: dict) -> dict:
    feeds = {}
    unmatched = set(input_bindings)
    for inp in sess.get_inputs():
        dtype = _np_dtype(inp.type)
        path = input_bindings.get(inp.name)
        if path is not None:
            unmatched.discard(inp.name)
            data = np.load(path)
            print(f"  input '{inp.name}': loaded {path} shape={data.shape} dtype={data.dtype}")
            if data.dtype != dtype:
                print(f"    casting {data.dtype} -> {dtype} to match model's declared input type")
                data = data.astype(dtype)
            feeds[inp.name] = data
        else:
            shape = _resolve_shape(inp.shape)
            data = (np.random.randn(*shape) * 0.5).astype(dtype)
            print(
                f"  input '{inp.name}': WARNING no --input binding; using scaled-random "
                f"data shape={shape} dtype={dtype}"
            )
            feeds[inp.name] = data

    for name in unmatched:
        print(f"WARNING: --input '{name}=...' did not match any model input; ignored.")

    return feeds


def parse_input_bindings(pairs) -> dict:
    bindings = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--input must be NAME=PATH, got: {pair!r}")
        name, path = pair.split("=", 1)
        bindings[name] = path
    return bindings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Path to an ONNX model.")
    ap.add_argument("--ep", default="WebGPU", choices=["WebGPU", "CPU"])
    ap.add_argument("--iters", type=int, default=20, help="Measured runs.")
    ap.add_argument("-o", "--output", default="model_prof.json", help="Output ORT tracing JSON filename.")
    ap.add_argument(
        "--input",
        action="append",
        metavar="NAME=PATH",
        help="Bind a model input by name to a real .npy tensor (repeatable). "
        "Unbound inputs get scaled-random data matching their declared shape/dtype.",
    )
    args = ap.parse_args()

    sess = build_session(args.model, args.ep)
    for i in sess.get_inputs():
        print(f"  input : {i.name} {i.type} {i.shape}")
    for o in sess.get_outputs():
        print(f"  output: {o.name} {o.type} {o.shape}")

    input_bindings = parse_input_bindings(args.input)
    feeds = build_feeds(sess, input_bindings)

    print(f"Measured: {args.iters} runs ...")
    start = time.perf_counter()
    for _ in range(args.iters):
        sess.run(None, feeds)  # returning numpy forces the GPU sync, so wall time is real
    wall_ms_per_run = (time.perf_counter() - start) * 1000.0 / args.iters
    print(f"Wall-clock per run: {wall_ms_per_run:.2f} ms")

    prof_file = sess.end_profiling()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, args.output)
    shutil.move(prof_file, out_path)
    print(f"Profile written to: {out_path}")


if __name__ == "__main__":
    main()
