"""
Generate video clips from a list of text prompts using CogVideoX-2B.
Supports loading a LoRA checkpoint on top of the base model.

Usage:
    python scripts/generate_videos.py \
        --prompts data/prompts.txt \
        --output_dir data/generated \
        [--lora_checkpoint checkpoints/lora] \
        [--num_videos_per_prompt 1] \
        [--num_inference_steps 50] \
        [--guidance_scale 6.0] \
        [--seed 42]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Generate videos with CogVideoX-2B")
    p.add_argument("--prompts", required=True, help="Path to text file with one prompt per line")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lora_checkpoint", default=None, help="Path to LoRA adapter weights")
    p.add_argument("--base_model", default="THUDM/CogVideoX-2b")
    p.add_argument("--num_videos_per_prompt", type=int, default=1)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def load_pipeline(args):
    from diffusers import CogVideoXPipeline
    from peft import PeftModel

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading base model: {args.base_model}")
    pipe = CogVideoXPipeline.from_pretrained(args.base_model, torch_dtype=dtype)

    if args.lora_checkpoint is not None:
        print(f"Loading LoRA weights from: {args.lora_checkpoint}")
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_checkpoint)
        pipe.transformer = pipe.transformer.merge_and_unload()  # merge for faster inference

    pipe = pipe.to(args.device)
    pipe.enable_model_cpu_offload()  # offload to CPU when not in use

    return pipe


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(prompts)} prompts. Generating {args.num_videos_per_prompt} video(s) each.")

    pipe = load_pipeline(args)
    generator = torch.Generator(device=args.device).manual_seed(args.seed) if args.seed else None

    manifest = {}
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] '{prompt}'")
        for j in range(args.num_videos_per_prompt):
            out_name = f"gen_{i:04d}_{chr(65+j)}.mp4"  # gen_0001_A.mp4, gen_0001_B.mp4, ...
            out_path = os.path.join(args.output_dir, out_name)

            if os.path.exists(out_path):
                print(f"  Skipping {out_name} (exists)")
            else:
                result = pipe(
                    prompt=prompt,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                )
                video_frames = result.frames[0]  # list of PIL images
                import imageio
                imageio.mimsave(out_path, video_frames, fps=8, codec="libx264")
                print(f"  Saved {out_name}")

            manifest.setdefault(prompt, []).append(out_path)

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
