"""
Camera controllability evaluation for video diffusion models.

Measures how faithfully a generated video follows an intended camera motion
(zoom, pan, tilt, orbital), using optical-flow-based homography estimation
as a lightweight substitute for full COLMAP reconstruction.

Pipeline:
  1. Generate videos from camera-motion prompts (e.g. "a slow zoom in on...")
  2. Extract camera motion from generated frames via sparse optical flow + homography
  3. Classify estimated motion type (zoom / pan-left / pan-right / tilt / static)
  4. Compute controllability score = % of clips where estimated motion type
     matches the intended type from the prompt

Used in:
  notebooks/08_camera_control_conditioning.ipynb
  — comparing controllability before and after DiffusionDPO alignment
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Camera motion taxonomy ─────────────────────────────────────────────────

class MotionType(str, Enum):
    ZOOM_IN   = "zoom_in"
    ZOOM_OUT  = "zoom_out"
    PAN_LEFT  = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP   = "tilt_up"
    TILT_DOWN = "tilt_down"
    STATIC    = "static"
    COMPLEX   = "complex"   # multiple simultaneous motions


# Prompt templates that should elicit each motion type
MOTION_PROMPTS: Dict[MotionType, List[str]] = {
    MotionType.ZOOM_IN: [
        "a cinematic slow zoom-in on a mountain peak at sunrise",
        "the camera slowly zooms in on a flower in a field",
        "a steady zoom-in shot approaching a historic building entrance",
    ],
    MotionType.ZOOM_OUT: [
        "the camera slowly zooms out revealing a vast ocean coastline",
        "a cinematic pull-back shot showing a city skyline from above",
        "the camera zooms out from a close-up of a face to reveal a crowd",
    ],
    MotionType.PAN_LEFT: [
        "a smooth left-panning shot along a mountain range",
        "the camera pans left following a running horse in a meadow",
        "a slow pan left revealing a long dining table",
    ],
    MotionType.PAN_RIGHT: [
        "a smooth right-panning shot along a river bank",
        "the camera pans right following a cyclist on a road",
        "a slow pan right across a gallery of paintings",
    ],
    MotionType.TILT_UP: [
        "the camera tilts up from the ground to reveal a tall skyscraper",
        "a slow tilt up from flowers to a cloudy sky",
    ],
    MotionType.TILT_DOWN: [
        "the camera tilts down from the sky to reveal a busy street below",
        "a slow tilt down from the summit to the valley",
    ],
    MotionType.STATIC: [
        "a static shot of a forest lake at sunrise, no camera movement",
        "a locked-off camera shot of a busy café interior",
    ],
}


# ── Homography-based camera motion estimation ──────────────────────────────

@dataclass
class CameraMotionEstimate:
    translation_x: float    # pixels, positive = right
    translation_y: float    # pixels, positive = down
    scale: float            # >1 = zoom in, <1 = zoom out
    rotation_deg: float     # degrees
    motion_type: MotionType
    confidence: float       # [0, 1], based on inlier ratio


def estimate_homography_motion(
    frame1: np.ndarray,
    frame2: np.ndarray,
    max_features: int = 500,
) -> Optional[CameraMotionEstimate]:
    """
    Estimate the camera motion between two frames using sparse optical flow
    (Shi-Tomasi corners + Lucas-Kanade) and a homography RANSAC fit.

    Returns None if fewer than 8 feature matches are found (degenerate case).
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect features in frame 1
    corners = cv2.goodFeaturesToTrack(
        gray1, maxCorners=max_features, qualityLevel=0.01, minDistance=10
    )
    if corners is None or len(corners) < 8:
        return None

    # Track to frame 2
    pts2, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, corners, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    good1 = corners[status.ravel() == 1]
    good2 = pts2[status.ravel() == 1]

    if len(good1) < 8:
        return None

    # Fit homography with RANSAC
    H, inlier_mask = cv2.findHomography(good1, good2, cv2.RANSAC, 3.0)
    if H is None:
        return None

    inlier_ratio = float(inlier_mask.sum()) / len(good1)

    # Decompose homography into translation, scale, rotation
    h, w = gray1.shape
    center = np.array([[w / 2, h / 2, 1]], dtype=float)
    moved = (H @ center.T).T[0]
    tx = moved[0] / moved[2] - w / 2
    ty = moved[1] / moved[2] - h / 2

    # Scale: average diagonal of the 2×2 rotation-scale sub-matrix
    sx = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
    sy = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
    scale = float((sx + sy) / 2)

    # Rotation (approximate)
    angle = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))

    motion_type = classify_motion(tx, ty, scale, angle)

    return CameraMotionEstimate(
        translation_x=float(tx),
        translation_y=float(ty),
        scale=scale,
        rotation_deg=angle,
        motion_type=motion_type,
        confidence=inlier_ratio,
    )


def classify_motion(
    tx: float,
    ty: float,
    scale: float,
    angle: float,
    tx_thresh: float = 5.0,
    ty_thresh: float = 5.0,
    scale_thresh: float = 0.02,
    angle_thresh: float = 1.0,
) -> MotionType:
    """
    Rule-based classification of dominant camera motion from homography components.
    Thresholds are in pixels (tx/ty) and fractional scale.
    """
    has_zoom = abs(scale - 1.0) > scale_thresh
    has_pan  = abs(tx) > tx_thresh
    has_tilt = abs(ty) > ty_thresh

    active = sum([has_zoom, has_pan, has_tilt])

    if active == 0:
        return MotionType.STATIC
    if active > 1:
        return MotionType.COMPLEX

    if has_zoom:
        return MotionType.ZOOM_IN if scale > 1.0 else MotionType.ZOOM_OUT
    if has_pan:
        return MotionType.PAN_RIGHT if tx > 0 else MotionType.PAN_LEFT
    if has_tilt:
        return MotionType.TILT_DOWN if ty > 0 else MotionType.TILT_UP

    return MotionType.STATIC


def estimate_video_motion(
    frames: List[np.ndarray],
    stride: int = 1,
) -> List[CameraMotionEstimate]:
    """
    Estimate camera motion for consecutive frame pairs in a video.

    Returns a list of estimates (length = len(frames) - stride).
    """
    estimates = []
    for i in range(0, len(frames) - stride, stride):
        est = estimate_homography_motion(frames[i], frames[i + stride])
        if est is not None:
            estimates.append(est)
    return estimates


def dominant_motion_type(estimates: List[CameraMotionEstimate]) -> MotionType:
    """
    Return the most common non-STATIC, non-COMPLEX motion type across estimates.
    Falls back to STATIC if all are static.
    """
    from collections import Counter
    counts = Counter(e.motion_type for e in estimates)
    # Prefer specific types over STATIC/COMPLEX
    for mt in [MotionType.ZOOM_IN, MotionType.ZOOM_OUT,
               MotionType.PAN_LEFT, MotionType.PAN_RIGHT,
               MotionType.TILT_UP, MotionType.TILT_DOWN]:
        if mt in counts:
            return mt
    return MotionType.STATIC


# ── Controllability scoring ────────────────────────────────────────────────

@dataclass
class ControllabilityResult:
    prompt: str
    intended_motion: MotionType
    estimated_motion: MotionType
    correct: bool
    mean_confidence: float
    mean_tx: float
    mean_ty: float
    mean_scale: float


def score_controllability(
    frames: List[np.ndarray],
    intended_motion: MotionType,
    prompt: str,
) -> ControllabilityResult:
    """
    Score how well a generated video follows the intended camera motion.
    """
    estimates = estimate_video_motion(frames)

    if not estimates:
        return ControllabilityResult(
            prompt=prompt,
            intended_motion=intended_motion,
            estimated_motion=MotionType.STATIC,
            correct=False,
            mean_confidence=0.0,
            mean_tx=0.0, mean_ty=0.0, mean_scale=1.0,
        )

    est_type = dominant_motion_type(estimates)
    correct = (est_type == intended_motion)

    return ControllabilityResult(
        prompt=prompt,
        intended_motion=intended_motion,
        estimated_motion=est_type,
        correct=correct,
        mean_confidence=float(np.mean([e.confidence for e in estimates])),
        mean_tx=float(np.mean([e.translation_x for e in estimates])),
        mean_ty=float(np.mean([e.translation_y for e in estimates])),
        mean_scale=float(np.mean([e.scale for e in estimates])),
    )


def controllability_report(results: List[ControllabilityResult]) -> Dict:
    """
    Aggregate controllability results across multiple prompts and motion types.
    """
    from collections import defaultdict
    per_type: Dict[MotionType, List[bool]] = defaultdict(list)
    for r in results:
        per_type[r.intended_motion].append(r.correct)

    overall_acc = float(np.mean([r.correct for r in results]))
    per_type_acc = {mt.value: float(np.mean(corr)) for mt, corr in per_type.items()}

    return {
        "overall_accuracy": overall_acc,
        "n_clips": len(results),
        "per_type_accuracy": per_type_acc,
        "mean_confidence": float(np.mean([r.mean_confidence for r in results])),
    }
