import argparse
from scipy.ndimage import distance_transform_edt
import numpy as np
from PIL import Image
import math
import sys
import time

def load_binary_mask(filepath: str, threshold: float = 0.5) -> np.ndarray:
    """Load image and convert to binary mask (True = Inside)"""
    img = Image.open(filepath).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    binary = arr > threshold
    print(f"Loaded {filepath}: {binary.shape} pixels, {binary.sum()} inside")
    return binary

def generate_sdf_vectorized(
        mask_highres: np.ndarray,
        out_width: int,
        out_height: int,
        spread: float = 6.0,
        search_margin: float = 1.1,
) -> np.ndarray:
    if mask_highres.ndim != 2:
        raise ValueError("Inpput mask must be 2D")

    high_h, high_w = mask_highres.shape

    scale_x = high_w / out_width
    scale_y = high_h / out_height

    max_dist_high = spread * max(scale_x, scale_y)

    # Signed Euclidean distance field (high-res)
    dist_to_bg = distance_transform_edt(~mask_highres.astype(bool))
    dist_to_fg = distance_transform_edt( mask_highres.astype(bool))

    signed_high = np.where(mask_highres, -dist_to_fg, dist_to_bg)

    # Sample at low-res texel centers
    half_x = scale_x / 2
    half_y = scale_y / 2
    yy, xx = np.mgrid[0:out_height, 0:out_width]
    cy = (yy * scale_y + half_y).astype(int)
    cx = (xx * scale_x + half_x).astype(int)
    cy = np.clip(cy, 0, high_h - 1)
    cx = np.clip(cx, 0, high_w - 1)

    signed_low = signed_high[cy, cx]

    # Normalize + clamp like Valve
    value = 0.5 - signed_low / (2.0 * max_dist_high)
    value = np.clip(value, 0.0, 1.0)

    return value

def save_sdf_as_png(sdf: np.ndarray, output_path: str):
    """Save as 8-bit grayscale PNG"""
    img_data = (sdf * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img_data, mode="L").save(output_path)
    print(f"Saved to {output_path} ({sdf.shape[0]}x{sdf.shape[1]})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Valve-style SDF texture (rectangular support)"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input high-resolution image")
    parser.add_argument("--width", type=int, required=True,
                        help="Output width (pixels)")
    parser.add_argument("--height", type=int, required=True,
                        help="Output height (pixels)")
    parser.add_argument("--spread", type=float, default=6.0,
                        help="Spread factor in low-res texels (default: 6)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: <input>_sdf_<w>x<h>.png)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for inside/outside (default 0.5)")


    args = parser.parse_args()

    if args.width < 8 or args.width > 2048 or args.height < 8 or args.height > 2048:
        print("Width and height should be 8–2048", file=sys.stderr)
        return 1

    start = time.time()

    mask = load_binary_mask(args.input, threshold=args.threshold)
    sdf = generate_sdf_vectorized(
        mask,
        out_width=args.width,
        out_height=args.height,
        spread=args.spread
    )

    if args.output:
        out_path = args.output
    else:
        import os
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}_sdf_{args.width}x{args.height}.png"

    save_sdf_as_png(sdf, out_path)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f} seconds")

if __name__ == '__main__':
    sys.exit(main() or 0)
