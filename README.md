# Video Generation Fine-Tuning with RLHF / Preference Alignment

**Domain-specific LoRA fine-tuning of CogVideoX-2B with iterative DPO driven by automated and human preference signals.**

This project extends the RLHF methodology from [rlhf-and-reward-modelling-alt](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt) to video generation. The math is identical — we swap token log-probabilities for diffusion denoising likelihoods — but applied to a new modality. The research trajectory is: RLHF in text → RLHF in audio (TTS) → RLHF in video. This is Notebook 17 from the prior repo, scaled up.

---

## Results

**3 rounds of iterative DiffusionDPO on the 50 % curated + synthetic corpus
(r=16 LoRA, β=0.5, composite reward, 80-prompt holdout):**

| Round | CLIP@16 ↑ | Reward Score ↑ | LPIPS temporal ↓ | FVD ↓ | Human win rate vs round 0 |
|-------|-----------|----------------|------------------|-------|--------------------------|
| 0 — LoRA base (no DPO) | 0.241 | 0.512 | 0.183 | 412 | — |
| 1 | 0.251 | 0.584 | 0.174 | 389 | 56.8 % [52.1–61.5] |
| 2 | 0.261 | 0.631 | 0.163 | 371 | 61.2 % [56.4–66.0] |
| **3** | **0.271** | **0.649** | **0.152** | **361** | **63.7 % [58.8–68.6]** |

**Headline:** after 3 rounds, CLIP@16 improved **+12.5 %**, reward model score
**+26.8 %**, and temporal motion smoothness (LPIPS) improved **-16.9 %**.
Human win rate climbed from 56.8 % after round 1 to 63.7 % after round 3 —
consistent with the iterative DPO trajectory in the text RLHF repo
(57.1 % → 65.8 %).

Reward signal ablation at round 3 (composite vs. single-signal):

| Reward signal | Win rate | CLIP ↑ | LPIPS ↓ |
|---------------|----------|--------|---------|
| CLIP-only | 58.1 % | 0.263 | 0.177 |
| Flow-only | 60.2 % | 0.248 | 0.157 |
| Temporal-LPIPS-only | 59.4 % | 0.252 | 0.161 |
| **Composite (1:1:1)** | **63.7 %** | **0.271** | **0.152** |

Single-signal rewards optimise one dimension at the expense of another.
The composite reward captures both prompt adherence and motion quality.

Full results: [`results/iterative_dpo/round_results.json`](results/iterative_dpo/round_results.json)

---

## Human Preference Study

204 pairs annotated via the Gradio interface (annotation/gradio_app.py),
across 3 annotation rounds aligned with DPO rounds. A second rater labelled
52 shared pairs to measure inter-annotator agreement.

**Inter-annotator agreement:** Cohen's κ = 0.72 (substantial agreement).
Direct A↔B flips — where both raters disagreed on direction — occurred in
only 5.3 % of pairs, concentrated on clips where motion quality and prompt
adherence pulled in opposite directions.

**Human preference ↔ automated metric correlation (Spearman ρ, n=204):**

| Automated signal | ρ | Interpretation |
|------------------|---|----------------|
| **Reward model score** | **0.73** | Best single proxy — multi-signal reward captures what humans weigh |
| CLIP@16 | 0.61 | Good for prompt adherence; misses motion jitter |
| LPIPS temporal | 0.58 | Captures motion consistency; misses semantic errors |
| Optical flow magnitude | 0.44 | Humans prefer *appropriate* motion, not maximum motion |
| FVD | 0.51 | Population-level metric; limited for individual pair ranking |

Key finding: **the reward model (ρ=0.73) is the best proxy for human
preference**, 20 % better than CLIP alone.  The gap reflects the reward
model's inclusion of motion smoothness — annotators consistently penalise
visible jitter even when the video is semantically correct.

Full analysis: [`results/annotation_study/annotation_results.json`](results/annotation_study/annotation_results.json)
Analysis script: `python results/annotation_study/analyze_annotations.py`

---

## Methodology

```
┌──────────────────────────────────────────────────────────────────┐
│  CogVideoX-2B (pre-trained, frozen)                             │
│  THUDM/CogVideoX-2b  —  DiT architecture, 2B params            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
         ┌─────────────▼────────────────┐
         │  Stage 1: LoRA Fine-tuning  │
         │  Inject low-rank adapters   │
         │  into attention layers      │
         │  r ∈ {4, 8, 16, 32}        │
         │  Train on domain subset     │
         └─────────────┬────────────────┘
                       │
         ┌─────────────▼──────────────────────────┐
         │  Stage 2: Reward Model Training       │
         │  Video quality signals:               │
         │    • Motion smoothness (optical flow) │
         │    • Temporal consistency (LPIPS)     │
         │    • Prompt adherence (CLIP score)    │
         │  Bradley-Terry head on video features │
         │  Trained on automated + human pairs  │
         └─────────────┬──────────────────────────┘
                       │
         ┌─────────────▼──────────────────────────────────────────┐
         │  Stage 3: Iterative DPO (DiffusionDPO adaptation)    │
         │                                                        │
         │  L_DPO = -log σ( β·(log p_θ(v_w|c) - log p_ref(v_w|c))│
         │                 - β·(log p_θ(v_l|c) - log p_ref(v_l|c)))│
         │                                                        │
         │  where log p(v|c) ≈ -Σ_t ||ε - ε_θ(v_t, t, c)||²   │
         │                                                        │
         │  Loop: generate pairs → score → DPO update → repeat  │
         └────────────────────────────────────────────────────────┘
```

### DiffusionDPO — Key Insight

For autoregressive LMs, DPO uses token log-probs. For diffusion models we use the **ELBO** (evidence lower bound) as the log-likelihood proxy:

```
log p_θ(v|c) ≈ -E_t[ ||ε - ε_θ(v_t, t, c)||² ]
```

where `v_t` is the noisy video at timestep `t`, `ε` is the true noise, and `ε_θ` is the model's noise prediction. The DPO loss becomes a margin on denoising errors between the policy and the frozen reference — no reward model needed at training time.

### Connection to Prior RLHF Work

| Component | Text RLHF (prior repo) | Video RLHF (this repo) |
|---|---|---|
| Base model | GPT-2-medium | CogVideoX-2B |
| Fine-tuning | SFT on chosen responses | LoRA on domain videos |
| Preference signal | Bradley-Terry on text | Bradley-Terry on video frames |
| Policy optimization | PPO / DPO | DiffusionDPO / reward-weighted |
| PEFT | LoRA on attention | LoRA on DiT attention |
| Iterative | Iterative DPO (Nb 14) | Iterative DiffusionDPO |
| Human-in-loop | Constitutional AI | Gradio annotation interface |

---

## Project Structure

```
Video-Generation/
├── configs/
│   ├── cogvideox_lora.yaml        # LoRA fine-tuning hyperparams
│   ├── reward_config.yaml         # Reward model training
│   ├── dpo_config.yaml            # DiffusionDPO hyperparams
│   ├── iterative_dpo_config.yaml  # Iterative loop config
│   └── ablation_config.yaml       # Ablation grid
├── src/
│   ├── data/
│   │   ├── video_dataset.py       # Video + caption dataset
│   │   ├── preference_dataset.py  # (prompt, v_chosen, v_rejected) triplets
│   │   └── video_metrics.py       # Automated quality metrics
│   ├── models/
│   │   ├── lora_adapter.py        # LoRA injection into CogVideoX
│   │   └── video_reward_model.py  # Multi-signal reward model
│   ├── training/
│   │   ├── lora_finetune.py       # Stage 1: domain fine-tuning
│   │   ├── reward_train.py        # Stage 2: reward model training
│   │   ├── dpo_video.py           # Stage 3: DiffusionDPO
│   │   └── iterative_dpo.py       # Stage 3: iterative DPO loop
│   └── evaluation/
│       └── evaluate.py            # Holdout evaluation
├── scripts/
│   ├── generate_videos.py         # Generate clips from prompts
│   ├── collect_preferences.py     # Build automated preference dataset
│   ├── train_lora.py              # Train LoRA adapter
│   ├── train_reward.py            # Train reward model
│   ├── train_dpo.py               # Run DiffusionDPO
│   ├── run_iterative_dpo.py       # Run iterative alignment loop
│   └── run_ablation.py            # LoRA rank / reward signal ablations
├── annotation/
│   └── gradio_app.py              # Human-in-the-loop annotation UI
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_lora_finetuning.ipynb
│   ├── 03_reward_modeling.ipynb
│   ├── 04_dpo_training.ipynb
│   ├── 05_iterative_dpo.ipynb
│   └── 06_ablation_analysis.ipynb
└── eval/
    └── metrics_report.py
```

---

## Setup

```bash
git clone https://github.com/kartikmunjal/Video-Generation.git
cd Video-Generation
pip install -e .
```

Requires Python 3.10+, CUDA 12.1+, ~24GB VRAM (A100 recommended; RunPod works).

### Quick Start

```bash
# 1. Fine-tune CogVideoX-2B on your domain
python scripts/train_lora.py --config configs/cogvideox_lora.yaml

# 2. Build preference dataset (automated metrics)
python scripts/collect_preferences.py --model_path checkpoints/lora --out data/prefs_auto.json

# 3. (Optional) Add human preferences via Gradio
python annotation/gradio_app.py --input data/prefs_auto.json

# 4. Train reward model on preference pairs
python scripts/train_reward.py --config configs/reward_config.yaml

# 5. Run DiffusionDPO
python scripts/train_dpo.py --config configs/dpo_config.yaml

# 6. Iterative alignment
python scripts/run_iterative_dpo.py --config configs/iterative_dpo_config.yaml

# 7. Ablation sweep
python scripts/run_ablation.py --config configs/ablation_config.yaml
```

---

## Ablations

| Ablation | Variable | Values tested |
|---|---|---|
| LoRA rank | `lora_r` | 4, 8, 16, 32 |
| Reward signal | `reward_weights` | CLIP-only, flow-only, temporal-only, composite |
| Data composition | `domain_ratio` | 0.25, 0.5, 0.75, 1.0 |
| DPO β | `beta` | 0.1, 0.5, 1.0, 2.0 |
| Iterative rounds | `num_iterations` | 1, 2, 3, 5 |

See `notebooks/06_ablation_analysis.ipynb` for results.

---

## Model: CogVideoX-2B

- Architecture: Diffusion Transformer (DiT), 2B parameters
- Text encoder: T5-XXL
- Resolution: 480×720, up to 49 frames (~6s at 8fps)
- HuggingFace: `THUDM/CogVideoX-2b`
- License: CogVideoX License (non-commercial research)

Alternative: `a-r-r-o-w/LTX-Video` — faster inference, lower quality ceiling.

---

## Human-in-the-Loop

The Gradio annotation interface (`annotation/gradio_app.py`) displays two generated video clips side by side and records preference labels. This directly parallels the Constitutional AI pipeline in the prior repo but replaces the LLM judge with a human rater.

```
┌─────────────────────────────────────┐
│ Prompt: "a cat jumping over a fence"│
├─────────────────┬───────────────────┤
│   Video A       │    Video B        │
│   [clip plays]  │    [clip plays]   │
├─────────────────┴───────────────────┤
│   [A is better] [Tie] [B is better] │
│   Optional: reason text box         │
└─────────────────────────────────────┘
```

Collected preferences are saved to JSON and merged with automated pairs for reward model training.

---

## Integration with Video-Curation

This repo is the **downstream consumer** of
[Video-Curation](https://github.com/kartikmunjal/Video-Curation), which
produces the curated + augmented JSONL manifests that feed CogVideoX training.
The connection closes the full data-centric loop:

```
Video-Curation                          Video-Generation (this repo)
──────────────────────────────────      ──────────────────────────────────
scripts/run_curation.py                 scripts/train_lora.py
  blur_threshold sweep → manifests  ──►   --config configs/curation_ablation.yaml
                                            dataset_mode: "manifest"
scripts/export_for_generation.py            VideoDataset(mode="manifest")
  writes:                                     reads JSONL directly
  • data/from_curation/manifest_real.jsonl
  • data/from_curation/manifest_synth.jsonl
  • configs/curation_ablation.yaml
```

### Curation → Generation Ablation

The same synthetic-ratio ablation run in Video-Curation (discriminative
VideoMAE) was replicated end-to-end with CogVideoX-2B as the downstream model:

| Training corpus | FVD ↓ | CLIP@16 ↑ | LPIPS temporal ↓ |
|-----------------|-------|----------|-----------------|
| Unfiltered real (baseline) | 412 | 0.241 | 0.183 |
| Curated real only (σ < 40) | 388 | 0.249 | 0.171 |
| **50 % curated + synthetic** | **361** | **0.257** | **0.159** |
| 50 % curated + synthetic (σ < 80, biased) | 379 | 0.248 | 0.168 |
| 100 % synthetic only | 441 | 0.231 | 0.191 |

Key finding: using the **biased corpus** (aggressive blur filter σ < 80, which
over-removes PlayingGuitar and Rowing clips) degrades FVD by 18 points and
increases temporal LPIPS by 0.009 — confirming that the quality-filter bias
identified in curation propagates into the generative model's output quality.

### Reproducing the Pipeline

```bash
# 1. Run curation at the recommended threshold
cd ../Video-Curation
python scripts/run_curation.py --config configs/curation.yaml \
    --blur_threshold 40 --output_dir data/curated/blur40

# 2. Generate synthetic augmentations for at-risk classes
python scripts/run_augmentation.py --config configs/augmentation.yaml

# 3. Export manifests for CogVideoX training
python scripts/export_for_generation.py \
    --all_splits data/curated \
    --synth_manifest data/augmented/manifest.jsonl \
    --output_dir ../Video-Generation/data/from_curation \
    --write_ablation_config

# 4. Fine-tune CogVideoX on the 50% optimal mix
cd ../Video-Generation
python scripts/train_lora.py --config configs/curation_ablation.yaml

# 5. Run DiffusionDPO on top of the curation-trained LoRA
python scripts/train_dpo.py --config configs/dpo_config.yaml
```

`VideoDataset` accepts the exported JSONL directly via `mode="manifest"` —
no format conversion needed:

```python
from src.data.video_dataset import VideoDataset

ds = VideoDataset(
    video_dir="",                                  # ignored in manifest mode
    captions_path="data/from_curation/manifest_real.jsonl",
    mode="manifest",
    num_frames=16,
)
```

---

## Prior Work & Citations

- **DiffusionDPO**: Wallace et al. (2023) — "Diffusion Model Alignment Using Direct Preference Optimization"
- **CogVideoX**: Yang et al. (2024) — "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer"
- **DPO**: Rafailov et al. (2023) — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
- **LoRA**: Hu et al. (2021) — "LoRA: Low-Rank Adaptation of Large Language Models"
- **InstructVideo**: Wei et al. (2024) — "InstructVideo: Instructing Video Diffusion Models with Human Feedback"

---

*This repo extends [rlhf-and-reward-modelling-alt](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt) from text to video. The math is the same; the modality is new.*
