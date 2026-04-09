"""
Automated preference dataset collection.

For each prompt, generates two videos (A and B), computes quality metrics for
each, and records the preference as a JSONL record. The preferred video is the
one with the higher composite reward score.

Usage:
    python scripts/collect_preferences.py \
        --model_path checkpoints/lora \
        --prompts data/train_prompts.txt \
        --output data/prefs_auto.jsonl \
        [--num_pairs 500] \
        [--reward_weights '{"motion": 0.2, "temporal": 0.3, "clip": 0.5}']
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.video_dataset import load_video_frames
from data.video_metrics import compute_video_metrics
from data.preference_dataset import save_preference_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None, help="LoRA checkpoint path (None = base model)")
    p.add_argument("--base_model", default="THUDM/CogVideoX-2b")
    p.add_argument("--prompts", required=True, help="Path to prompts file (one per line)")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--video_dir", default="data/generated_pairs", help="Dir to save generated videos")
    p.add_argument("--num_pairs", type=int, default=500)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--reward_weights", type=json.loads,
                   default={"motion": 0.2, "temporal": 0.3, "clip": 0.5})
    p.add_argument("--min_score_gap", type=float, default=0.05,
                   help="Discard pairs where score gap < this threshold (noise filter)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def generate_pair(pipe, prompt, out_dir, pair_idx, args, generator=None):
    """Generate two videos for the same prompt and return their paths."""
    import imageio

    paths = []
    for variant in ["A", "B"]:
        out_path = os.path.join(out_dir, f"pair_{pair_idx:05d}_{variant}.mp4")
        if not os.path.exists(out_path):
            result = pipe(
                prompt=prompt,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            imageio.mimsave(out_path, result.frames[0], fps=8, codec="libx264")
        paths.append(out_path)
    return paths[0], paths[1]


def main():
    args = parse_args()
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    with open(args.prompts) as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Subsample if we need fewer pairs than prompts
    import random
    random.shuffle(prompts)
    prompts = (prompts * (args.num_pairs // len(prompts) + 1))[:args.num_pairs]

    # Load pipeline
    from diffusers import CogVideoXPipeline
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading model...")
    pipe = CogVideoXPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    if args.model_path:
        from peft import PeftModel
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.model_path)
        pipe.transformer = pipe.transformer.merge_and_unload()
    pipe = pipe.to(args.device)
    pipe.enable_model_cpu_offload()

    records = []
    skipped = 0

    print(f"Generating {args.num_pairs} preference pairs...")
    for idx, prompt in enumerate(prompts):
        print(f"  [{idx+1}/{args.num_pairs}] '{prompt[:60]}'")

        path_a, path_b = generate_pair(pipe, prompt, args.video_dir, idx, args)

        # Load frames and compute metrics
        frames_a = load_video_frames(path_a, args.num_frames, args.height, args.width)
        frames_b = load_video_frames(path_b, args.num_frames, args.height, args.width)

        metrics_a = compute_video_metrics(frames_a, prompt, device=args.device,
                                          weights=args.reward_weights)
        metrics_b = compute_video_metrics(frames_b, prompt, device=args.device,
                                          weights=args.reward_weights)

        gap = abs(metrics_a.composite - metrics_b.composite)
        if gap < args.min_score_gap:
            skipped += 1
            print(f"    Skipped (score gap {gap:.3f} < {args.min_score_gap})")
            continue

        if metrics_a.composite >= metrics_b.composite:
            chosen, rejected = path_a, path_b
            score_c, score_r = metrics_a.composite, metrics_b.composite
        else:
            chosen, rejected = path_b, path_a
            score_c, score_r = metrics_b.composite, metrics_a.composite

        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "automated",
            "scores": {"chosen": round(score_c, 4), "rejected": round(score_r, 4)},
            "metrics_chosen": metrics_a.to_dict() if chosen == path_a else metrics_b.to_dict(),
            "metrics_rejected": metrics_b.to_dict() if chosen == path_a else metrics_a.to_dict(),
        })

    save_preference_jsonl(records, args.output)
    print(f"\nSaved {len(records)} preference pairs to {args.output}")
    print(f"Skipped {skipped} pairs (score gap too small).")


if __name__ == "__main__":
    main()
