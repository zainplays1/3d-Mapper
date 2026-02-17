#!/usr/bin/env python3
"""
Warp a transparent handwriting overlay onto a crumpled-paper background using
MiDaS depth estimation + OpenCV displacement mapping, then composite with
multiply blending.

Example:
    python midas_displacement_composite.py \
        --background paper.jpg \
        --overlay handwriting.png \
        --output composited.png \
        --offset-x 120 \
        --offset-y 80 \
        --warp-strength 24.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map a transparent overlay onto a background by estimating depth "
            "with MiDaS, warping via displacement, and compositing with "
            "multiply blend."
        )
    )
    parser.add_argument("--background", required=True, help="Path to background image.")
    parser.add_argument(
        "--overlay",
        required=True,
        help="Path to transparent PNG overlay (RGBA expected).",
    )
    parser.add_argument("--output", required=True, help="Path to save final image.")
    parser.add_argument(
        "--offset-x",
        type=int,
        default=0,
        help=(
            "Horizontal placement of overlay in pixels. Positive moves right, "
            "negative moves left."
        ),
    )
    parser.add_argument(
        "--offset-y",
        type=int,
        default=0,
        help=(
            "Vertical placement of overlay in pixels. Positive moves down, "
            "negative moves up."
        ),
    )
    parser.add_argument(
        "--warp-strength",
        type=float,
        default=20.0,
        help=(
            "Multiplier for displacement intensity derived from depth gradients. "
            "Increase for stronger fold-following deformation; decrease for "
            "subtler warping."
        ),
    )
    parser.add_argument(
        "--midas-model",
        default="Intel/dpt-large",
        help="Hugging Face model id for depth estimation.",
    )
    return parser.parse_args()


def load_images(background_path: str, overlay_path: str) -> tuple[np.ndarray, np.ndarray]:
    background_bgr = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background_bgr is None:
        raise FileNotFoundError(f"Could not read background image: {background_path}")

    overlay_rgba = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay_rgba is None:
        raise FileNotFoundError(f"Could not read overlay image: {overlay_path}")
    if overlay_rgba.ndim != 3 or overlay_rgba.shape[2] != 4:
        raise ValueError("Overlay must have 4 channels (RGBA or BGRA with alpha).")

    return background_bgr, overlay_rgba


def place_overlay_on_canvas(
    overlay_rgba: np.ndarray,
    target_h: int,
    target_w: int,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    """Return an RGBA canvas of size target_h x target_w with overlay placed at offsets."""
    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)

    ov_h, ov_w = overlay_rgba.shape[:2]

    dst_x0 = max(0, offset_x)
    dst_y0 = max(0, offset_y)
    dst_x1 = min(target_w, offset_x + ov_w)
    dst_y1 = min(target_h, offset_y + ov_h)

    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return canvas

    src_x0 = max(0, -offset_x)
    src_y0 = max(0, -offset_y)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = overlay_rgba[src_y0:src_y1, src_x0:src_x1]
    return canvas


def estimate_depth_map(background_bgr: np.ndarray, model_id: str) -> np.ndarray:
    """
    Estimate depth via Hugging Face transformers pipeline.
    Returns normalized depth in range [0, 1], shape HxW.
    """
    depth_estimator = pipeline(task="depth-estimation", model=model_id)

    rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    depth_result = depth_estimator(pil_img)
    depth_pil = depth_result["depth"]
    depth = np.array(depth_pil).astype(np.float32)

    depth_resized = cv2.resize(
        depth,
        (background_bgr.shape[1], background_bgr.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    min_val, max_val = depth_resized.min(), depth_resized.max()
    depth_norm = (depth_resized - min_val) / (max_val - min_val + 1e-8)
    return depth_norm


def warp_overlay_with_depth(
    overlay_rgba_canvas: np.ndarray,
    depth_norm: np.ndarray,
    warp_strength: float,
) -> np.ndarray:
    """
    Convert depth map gradients into x/y displacement and warp RGBA overlay.

    np.gradient returns [d/dy, d/dx]. We scale gradients with warp_strength,
    then remap each overlay pixel from source coords to displaced coords.
    """
    grad_y, grad_x = np.gradient(depth_norm)

    h, w = depth_norm.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # Inverse-mapping for cv2.remap: sample source from displaced location.
    # Tune `warp_strength` from CLI:
    #   larger value -> stronger bend, smaller value -> more subtle deformation.
    map_x = (grid_x + grad_x.astype(np.float32) * warp_strength).astype(np.float32)
    map_y = (grid_y + grad_y.astype(np.float32) * warp_strength).astype(np.float32)

    warped = cv2.remap(
        overlay_rgba_canvas,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def multiply_composite(background_bgr: np.ndarray, warped_overlay_rgba: np.ndarray) -> np.ndarray:
    """Multiply blend overlay RGB onto background using overlay alpha as mask."""
    bg = background_bgr.astype(np.float32) / 255.0
    ov_rgb = warped_overlay_rgba[:, :, :3].astype(np.float32) / 255.0
    alpha = warped_overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0

    multiplied = bg * ov_rgb
    composed = bg * (1.0 - alpha) + multiplied * alpha

    return np.clip(composed * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()

    background_bgr, overlay_rgba = load_images(args.background, args.overlay)
    h, w = background_bgr.shape[:2]

    # Position transparent handwriting with user-controlled offsets.
    overlay_canvas = place_overlay_on_canvas(
        overlay_rgba=overlay_rgba,
        target_h=h,
        target_w=w,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
    )

    depth_norm = estimate_depth_map(background_bgr, args.midas_model)

    warped_overlay = warp_overlay_with_depth(
        overlay_rgba_canvas=overlay_canvas,
        depth_norm=depth_norm,
        warp_strength=args.warp_strength,
    )

    final_bgr = multiply_composite(background_bgr, warped_overlay)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), final_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save output image: {output_path}")

    print(f"Saved composited output to: {output_path}")


if __name__ == "__main__":
    main()
