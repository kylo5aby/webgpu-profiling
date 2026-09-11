"""Dump a *real* VAE-decoder input latent from the full Z-Image-Turbo pipeline.

Runs the real pipeline (text_encoder -> transformer x N steps) on WebGPU and saves the exact
`latent_sample` tensor that would be fed into the VAE decoder (i.e. after `apply_vae_scaling`)
to a .npy file, so `profile_vae_decoder.py` can profile the decoder on realistic data instead
of random noise.

Usage:
    python dump_latent.py Z-Image-Turbo-webnn --ep WebGPU -n 4 --height 1024 --width 1024 \
        -o real_latent.npy
"""

import argparse
import sys

import numpy as np

import run_z_image_turbo as rz


class DumpPipeline(rz.ZImagePipeline):
    """Same pipeline, but intercepts the VAE-decoder input and writes it to disk."""

    def __init__(self, *args, latent_out="real_latent.npy", **kwargs):
        super().__init__(*args, **kwargs)
        self._latent_out = latent_out

    def run_vae_decoder(self) -> bool:
        # Reconstruct exactly what run_z_image_turbo feeds to the decoder (see its run_vae_decoder).
        latents = np.squeeze(self.latents_current_, axis=2)
        scaled = rz.apply_vae_scaling(
            latents, self.vae_scaling_factor_, self.vae_shift_factor_
        )
        latent_sample = scaled.astype(self.vae_dtype_)
        np.save(self._latent_out, latent_sample)
        print(
            f"[dump] saved real latent_sample shape={latent_sample.shape} "
            f"dtype={latent_sample.dtype} -> {self._latent_out}"
        )
        return super().run_vae_decoder()


def main():
    # Windows console can't encode the non-Latin-1 chars in the default prompt otherwise.
    sys.stdout.reconfigure(errors="backslashreplace")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="Model dir containing onnx/ and tokenizer/ (e.g. Z-Image-Turbo-webnn).")
    ap.add_argument("--ep", default="WebGPU", choices=["WebGPU", "CPU"], help="Execution provider.")
    ap.add_argument(
        "--prompt",
        default="A photograph of a red fox sitting in a snowy forest at sunrise, highly detailed.",
        help="Prompt (only affects latent values, not its shape).",
    )
    ap.add_argument("-n", "--num_inference_steps", type=int, default=4)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument(
        "--transformer",
        default="",
        metavar="PATH",
        help=(
            "Path to a 4D 'dev z-transformer' model.onnx (4D hidden_states, cap_seq_len "
            "encoder states). Required for the latest-webnn export; switches the pipeline to "
            "the 4D code path (squeeze num_frames + pad caption to a multiple of 32)."
        ),
    )
    ap.add_argument("-o", "--latent_out", default="real_latent.npy", help="Output .npy path.")
    args = ap.parse_args()

    pipe = DumpPipeline(
        args.model,
        args.ep,
        args.num_inference_steps,
        args.height,
        args.width,
        dev_transformer_path=args.transformer,
        latent_out=args.latent_out,
    )
    if not pipe.initialize():
        sys.exit("Pipeline initialization failed.")
    # The decoded image is a throwaway here; we only care about the dumped latent.
    if not pipe.run(args.prompt, "dump_latent_preview.png"):
        sys.exit("Pipeline run failed.")
    print("[dump] done.")


if __name__ == "__main__":
    main()
