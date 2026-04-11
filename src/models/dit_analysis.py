"""
DiT (Diffusion Transformer) attention analysis utilities for CogVideoX-2B.

Provides tools to extract and analyse the spatiotemporal self-attention maps
from CogVideoX's transformer blocks during the denoising forward pass.

CogVideoX architecture notes:
  - CogVideoXTransformer3DModel processes video as T×H×W patches (3D ViT-style)
  - Each transformer block has full 3D attention: every patch attends to every
    other patch across both space AND time (not factored spatial + temporal)
  - Attention shape per block: (B, heads, T·N_sp, T·N_sp)
    where N_sp = (H/p)·(W/p) spatial patches per frame
  - To isolate temporal structure: average attention weights over spatial positions
    → temporal attention matrix of shape (T, T)

Usage:
    from src.models.dit_analysis import DiTAttentionExtractor, temporal_attention_entropy

    extractor = DiTAttentionExtractor(pipe.transformer, layer_indices=[0, 6, 11])
    maps = extractor.extract(pipe, prompt, video_latents, timestep=500)
    # maps: dict layer_idx → (B, heads, T·N_sp, T·N_sp) tensor

    # Collapse to temporal attention
    temp = extractor.to_temporal(maps[6], n_frames=16, n_spatial=256)
    ent = temporal_attention_entropy(temp)  # scalar entropy in bits
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Attention extraction ────────────────────────────────────────────────────

class DiTAttentionExtractor:
    """
    Registers forward hooks on CogVideoX transformer blocks to capture
    attention weight tensors during the denoising forward pass.

    The hooks are registered lazily and removed after each extraction call
    so the model can be used normally in between.
    """

    def __init__(
        self,
        transformer,
        layer_indices: Optional[List[int]] = None,
        head_reduction: str = "mean",   # "mean" | "max" | "none"
    ):
        """
        Args:
            transformer: CogVideoXTransformer3DModel instance
            layer_indices: which transformer blocks to hook (default: all)
            head_reduction: how to aggregate multi-head attention
        """
        self.transformer = transformer
        n_layers = len(transformer.transformer_blocks)
        self.layer_indices = layer_indices if layer_indices is not None else list(range(n_layers))
        self.head_reduction = head_reduction
        self._captured: Dict[int, "torch.Tensor"] = {}
        self._hooks = []

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs, output):
            # CogVideoX attention modules return (hidden_states, attn_weights)
            # when output_attentions=True is passed through the forward call.
            # The attention weights may be in output[1] or as a named return.
            import torch
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]  # (B, heads, seq_len, seq_len)
                if self.head_reduction == "mean":
                    attn_weights = attn_weights.mean(dim=1)   # (B, seq_len, seq_len)
                elif self.head_reduction == "max":
                    attn_weights = attn_weights.max(dim=1).values
                self._captured[layer_idx] = attn_weights.detach().cpu()
        return hook

    def register(self):
        """Register hooks on selected transformer blocks."""
        self._captured.clear()
        for idx in self.layer_indices:
            block = self.transformer.transformer_blocks[idx]
            h = block.attn1.register_forward_hook(self._make_hook(idx), with_kwargs=True)
            self._hooks.append(h)

    def remove(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def extract(
        self,
        pipe,
        prompt: str,
        num_frames: int = 16,
        height: int = 480,
        width: int = 720,
        num_inference_steps: int = 10,
        guidance_scale: float = 6.0,
        seed: int = 42,
    ) -> Dict[int, "torch.Tensor"]:
        """
        Run one denoising pass (fewer steps for speed) and return captured
        attention maps.  Each key is a layer index, value is a tensor of
        shape (B, seq_len, seq_len) after head reduction.
        """
        import torch

        self.register()
        try:
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                pipe(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="latent",  # skip VAE decode for speed
                )
        finally:
            self.remove()

        return dict(self._captured)

    @staticmethod
    def to_temporal(
        attn: "torch.Tensor",
        n_frames: int,
        n_spatial: int,
    ) -> np.ndarray:
        """
        Collapse full 3D attention (seq_len × seq_len) to a temporal attention
        matrix (n_frames × n_frames) by averaging over spatial positions.

        Args:
            attn: (seq_len, seq_len) tensor, seq_len = n_frames * n_spatial
            n_frames: number of video frames
            n_spatial: number of spatial patches per frame

        Returns:
            temporal_attn: (n_frames, n_frames) numpy array
        """
        import torch
        if attn.dim() == 3:
            attn = attn[0]  # take first batch item

        seq_len = n_frames * n_spatial
        assert attn.shape == (seq_len, seq_len), f"Expected {seq_len}×{seq_len}, got {attn.shape}"

        # Reshape: (n_frames, n_spatial, n_frames, n_spatial) → mean over spatial dims
        a = attn.view(n_frames, n_spatial, n_frames, n_spatial)
        temporal = a.mean(dim=(1, 3)).cpu().numpy()  # (n_frames, n_frames)
        return temporal


# ── Entropy and statistics ─────────────────────────────────────────────────

def temporal_attention_entropy(temporal_attn: np.ndarray) -> float:
    """
    Mean entropy of the temporal attention distribution per query frame.

    Each row of `temporal_attn` is a distribution over key frames.
    Entropy = 0: one frame attends exclusively to one other frame.
    Entropy = log2(T): completely uniform (no structure).

    Returns bits.
    """
    eps = 1e-9
    # Normalise rows to sum to 1
    row_sum = temporal_attn.sum(axis=1, keepdims=True) + eps
    p = temporal_attn / row_sum
    entropy_per_row = -np.sum(p * np.log2(p + eps), axis=1)
    return float(entropy_per_row.mean())


def diagonal_dominance(temporal_attn: np.ndarray) -> float:
    """
    Fraction of total attention weight on the main diagonal (self-attention).
    High = frames mostly attend to themselves (static scene).
    Low = frames strongly attend to other frames (motion reference).
    """
    diag = np.diag(temporal_attn)
    return float(diag.sum() / (temporal_attn.sum() + 1e-9))


def adjacent_frame_coupling(temporal_attn: np.ndarray) -> float:
    """
    Mean attention weight between temporally adjacent frames (|i - j| = 1).
    High = model uses neighbouring frames as motion reference.
    """
    T = temporal_attn.shape[0]
    off_diag_sum = 0.0
    count = 0
    for i in range(T):
        for j in range(T):
            if abs(i - j) == 1:
                off_diag_sum += temporal_attn[i, j]
                count += 1
    return float(off_diag_sum / (count + 1e-9))


def attention_summary(temporal_attn: np.ndarray) -> Dict[str, float]:
    """Return a dict of all temporal attention statistics."""
    return {
        "entropy_bits": temporal_attention_entropy(temporal_attn),
        "diagonal_dominance": diagonal_dominance(temporal_attn),
        "adjacent_coupling": adjacent_frame_coupling(temporal_attn),
        "max_off_diagonal": float(
            temporal_attn[~np.eye(len(temporal_attn), dtype=bool)].max()
        ),
    }


# ── Synthetic fallback ─────────────────────────────────────────────────────

def synthetic_temporal_attention(
    n_frames: int = 16,
    focus_strength: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a synthetic temporal attention matrix for testing / illustration.

    `focus_strength` in [0, 1]:
      0 = uniform (random noise)
      1 = highly structured (attends to adjacent frames + fixed reference)
    """
    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.ones(n_frames) * (1 - focus_strength + 0.1), size=n_frames)

    # Add focus: each row attends more strongly to adjacent rows and row 0
    for i in range(n_frames):
        boost = np.zeros(n_frames)
        if i > 0:
            boost[i - 1] += focus_strength * 0.4
        if i < n_frames - 1:
            boost[i + 1] += focus_strength * 0.4
        boost[0] += focus_strength * 0.2  # reference frame
        base[i] += boost
        base[i] /= base[i].sum()

    return base
