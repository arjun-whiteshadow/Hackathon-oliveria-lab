#!/usr/bin/env python

"""
Microscopy Image Registration - Stage Drift Correction (VS Code / Script Version)

Supports:
- Multi-page TIF stacks (.tif, .tiff)
- ND2 files (.nd2)
- ZIP datasets containing TIF/ND2 files (.zip)
- Directories of TIF images

Usage examples (from terminal):

    python microscopy_registration.py --path stack.tif
    python microscopy_registration.py --path data.nd2
    python microscopy_registration.py --path images.zip
    python microscopy_registration.py --path ./images_folder --type directory
"""

import argparse
import os
from pathlib import Path
import zipfile
import warnings

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform

import nd2  # pip install nd2
import cv2  # pip install opencv-python-headless

warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

class Config:
    """Configuration for image registration."""
    def __init__(self):
        self.reference_frame = 0   # Index of reference frame
        self.upsample_factor = 10  # Subpixel precision (10–100)
        self.show_progress = True  # Print progress info
        self.max_frames = None     # Limit number of frames (None = all)


config = Config()


# -------------------------------------------------------------------------
# Data loading functions
# -------------------------------------------------------------------------

def load_zip_dataset(zip_path, extract_to=None):
    """Extract zip file containing microscopy images and return extract path."""
    zip_path = Path(zip_path)
    if extract_to is None:
        extract_to = zip_path.parent / f"{zip_path.stem}_extracted"

    extract_to = Path(extract_to)
    print(f"📦 Extracting {zip_path} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    print(f"✓ Files extracted to: {extract_to}")
    return extract_to


def load_tif_stack(file_path):
    """Load multi-page TIF file as an image stack."""
    file_path = str(file_path)
    print(f"📂 Loading TIF stack: {file_path}")
    img = io.imread(file_path)

    # Handle different TIF formats
    if img.ndim == 2:
        img = img[np.newaxis, ...]  # Single frame
    elif img.ndim == 3 and img.shape[-1] <= 4:
        # RGB/RGBA - convert to grayscale and treat as 1 frame
        img = np.mean(img, axis=-1)[np.newaxis, ...]

    if config.max_frames is not None and img.shape[0] > config.max_frames:
        img = img[:config.max_frames]

    print(f"✓ Loaded {img.shape[0]} frames of size {img.shape[1]} x {img.shape[2]}")
    return img.astype(np.float32)


def load_nd2_data(file_path):
    """Load ND2 microscopy file."""
    file_path = str(file_path)
    print(f"📂 Loading ND2 file: {file_path}")

    with nd2.ND2File(file_path) as nd2_file:
        images = nd2_file.asarray()

    # Handle different ND2 formats
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    elif images.ndim == 4:
        # Assume (frames, channels, y, x) or similar
        if images.shape[1] <= 4:
            images = images[:, 0, :, :]      # take first channel
        else:
            images = np.mean(images, axis=1) # average channels

    if config.max_frames is not None and images.shape[0] > config.max_frames:
        images = images[:config.max_frames]

    print(f"✓ Loaded {images.shape[0]} frames of size {images.shape[1]} x {images.shape[2]}")
    return images.astype(np.float32)


def load_image_directory(dir_path, pattern="*.tif"):
    """Load multiple TIF images from a directory as a stack."""
    dir_path = Path(dir_path)
    print(f"📂 Loading images from directory: {dir_path}")
    files = sorted(dir_path.glob(pattern))

    if not files:
        raise ValueError(f"No files matching {pattern} found in {dir_path}")

    images = []
    for f in files:
        img = io.imread(str(f))
        if img.ndim == 3:
            img = np.mean(img, axis=-1)
        images.append(img)

    stack = np.array(images, dtype=np.float32)
    if config.max_frames is not None and stack.shape[0] > config.max_frames:
        stack = stack[:config.max_frames]

    print(f"✓ Loaded {stack.shape[0]} images of size {stack.shape[1]} x {stack.shape[2]}")
    return stack


# -------------------------------------------------------------------------
# Registration functions
# -------------------------------------------------------------------------

def calculate_shifts(image_stack, reference_idx=0, upsample_factor=10):
    """Calculate shift for each frame relative to reference frame."""
    n_frames = image_stack.shape[0]
    reference = image_stack[reference_idx]
    shifts = np.zeros((n_frames, 2))

    print("\n🔍 Calculating frame shifts...")
    for i in range(n_frames):
        if i == reference_idx:
            if config.show_progress:
                print(f"  Frame {i:3d}/{n_frames}: Reference frame (0.00, 0.00)")
            continue

        shift, error, diffphase = phase_cross_correlation(
            reference,
            image_stack[i],
            upsample_factor=upsample_factor,
        )
        shifts[i] = shift

        if config.show_progress and (i % 10 == 0 or i == n_frames - 1):
            print(f"  Frame {i:3d}/{n_frames}: shift = ({shift[0]:6.2f}, {shift[1]:6.2f}) pixels")

    print("✓ Shift calculation complete!")
    return shifts


def apply_registration(image_stack, shifts):
    """Apply calculated shifts to register images."""
    n_frames, height, width = image_stack.shape
    registered = np.zeros_like(image_stack)

    print("\n🔧 Applying registration...")
    for i in range(n_frames):
        tform = AffineTransform(translation=(-shifts[i, 1], -shifts[i, 0]))

        registered[i] = warp(
            image_stack[i],
            tform.inverse,
            output_shape=(height, width),
            preserve_range=True,
        )

        if config.show_progress and (i % 20 == 0 or i == n_frames - 1):
            print(f"  Processed {i+1}/{n_frames} frames")

    print("✓ Registration complete!")
    return registered


def register_image_stack(image_stack, reference_idx=0, upsample_factor=10):
    """Complete registration pipeline."""
    shifts = calculate_shifts(image_stack, reference_idx, upsample_factor)
    registered = apply_registration(image_stack, shifts)
    return registered, shifts


# -------------------------------------------------------------------------
# Visualization functions
# -------------------------------------------------------------------------

def visualize_shifts(shifts):
    """Visualize the calculated shifts."""
    frames = np.arange(len(shifts))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Time series
    axes[0].plot(frames, shifts[:, 0], label="Y shift")
    axes[0].plot(frames, shifts[:, 1], label="X shift")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Shift (pixels)")
    axes[0].set_title("Stage Drift Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Trajectory
    axes[1].plot(shifts[:, 1], shifts[:, 0], marker="o")
    axes[1].set_xlabel("X shift (pixels)")
    axes[1].set_ylabel("Y shift (pixels)")
    axes[1].set_title("Drift Trajectory")
    axes[1].grid(True, alpha=0.3)
    axes[1].axis("equal")

    plt.tight_layout()
    plt.show()

    # Stats
    print("\n" + "=" * 60)
    print("SHIFT STATISTICS")
    print("=" * 60)
    print(f"  Max X shift:  {np.max(np.abs(shifts[:, 1])):8.2f} pixels")
    print(f"  Max Y shift:  {np.max(np.abs(shifts[:, 0])):8.2f} pixels")
    print(f"  Mean X shift: {np.mean(np.abs(shifts[:, 1])):8.2f} pixels")
    print(f"  Mean Y shift: {np.mean(np.abs(shifts[:, 0])):8.2f} pixels")
    print(f"  Total drift:  {np.linalg.norm(shifts[-1] - shifts[0]):8.2f} pixels")
    print("=" * 60)


def visualize_before_after(original, registered, frame_idx=None):
    """Compare original and registered images (single frame + projections)."""
    if frame_idx is None:
        frame_idx = len(original) // 2

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Image Registration Results", fontsize=14)

    # Single-frame
    vmin, vmax = np.percentile(original[frame_idx], [1, 99])

    im1 = axes[0, 0].imshow(original[frame_idx], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"Original Frame {frame_idx}")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(registered[frame_idx], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"Registered Frame {frame_idx}")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    diff = np.abs(original[frame_idx] - registered[frame_idx])
    im3 = axes[0, 2].imshow(diff, cmap="hot")
    axes[0, 2].set_title("Absolute Difference")
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Projections
    max_orig = np.max(original, axis=0)
    max_reg = np.max(registered, axis=0)
    vmin_p, vmax_p = np.percentile(max_orig, [1, 99])

    im4 = axes[1, 0].imshow(max_orig, cmap="gray", vmin=vmin_p, vmax=vmax_p)
    axes[1, 0].set_title("Original - Max Projection")
    axes[1, 0].axis("off")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im5 = axes[1, 1].imshow(max_reg, cmap="gray", vmin=vmin_p, vmax=vmax_p)
    axes[1, 1].set_title("Registered - Max Projection")
    axes[1, 1].axis("off")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    std_orig = np.std(original, axis=0)
    std_reg = np.std(registered, axis=0)
    std_diff = std_orig - std_reg

    im6 = axes[1, 2].imshow(std_diff, cmap="RdBu_r",
                            vmin=-np.max(np.abs(std_diff)),
                            vmax=np.max(np.abs(std_diff)))
    axes[1, 2].set_title("Stability Improvement (Blue = Better)")
    axes[1, 2].axis("off")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    improvement = (np.mean(std_orig) - np.mean(std_reg)) / np.mean(std_orig) * 100
    print("\n📊 Registration Quality:")
    print(f"  Stability improvement: {improvement:.1f}%")
    print(f"  Original std dev:      {np.mean(std_orig):.2f}")
    print(f"  Registered std dev:    {np.mean(std_reg):.2f}")


# -------------------------------------------------------------------------
# Extra analysis & saving
# -------------------------------------------------------------------------

def analyze_drift_velocity(shifts):
    """Calculate and plot drift velocity between consecutive frames."""
    velocity = np.diff(shifts, axis=0)
    vmag = np.linalg.norm(velocity, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(velocity[:, 0], label="Y velocity")
    axes[0].plot(velocity[:, 1], label="X velocity")
    axes[0].set_xlabel("Frame interval")
    axes[0].set_ylabel("Velocity (pixels/frame)")
    axes[0].set_title("Drift Velocity Components")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(vmag, label="Speed")
    axes[1].axhline(np.mean(vmag), linestyle="--",
                    label=f"Mean: {np.mean(vmag):.2f}")
    axes[1].set_xlabel("Frame interval")
    axes[1].set_ylabel("Speed (pixels/frame)")
    axes[1].set_title("Drift Speed")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n📊 Drift Velocity Stats:")
    print(f"  Mean speed: {np.mean(vmag):.3f} pixels/frame")
    print(f"  Max speed:  {np.max(vmag):.3f} pixels/frame")
    print(f"  Min speed:  {np.min(vmag):.3f pixels/frame}")
    print(f"  Std speed:  {np.std(vmag):.3f pixels/frame}")


def plot_overlay_comparison(original, registered, frame_idx=None):
    """Create RGB overlay of original (red) and registered (green)."""
    if frame_idx is None:
        frame_idx = len(original) // 2

    o = original[frame_idx]
    r = registered[frame_idx]

    o_norm = (o - o.min()) / (o.ptp() + 1e-9)
    r_norm = (r - r.min()) / (r.ptp() + 1e-9)

    overlay = np.zeros((*o_norm.shape, 3), dtype=float)
    overlay[..., 0] = o_norm
    overlay[..., 1] = r_norm

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(o, cmap="gray")
    axes[0].set_title(f"Original Frame {frame_idx}")
    axes[0].axis("off")

    axes[1].imshow(r, cmap="gray")
    axes[1].set_title(f"Registered Frame {frame_idx}")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Red=Original, Green=Registered)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def save_results(registered, shifts, output_dir="./output"):
    """Save registered images and shifts to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_path = out_dir / "registered_stack.tif"
    io.imsave(str(tif_path), registered.astype(np.float32))
    print(f"✓ Saved TIF stack: {tif_path}")

    npy_path = out_dir / "shifts.npy"
    np.save(npy_path, shifts)
    print(f"✓ Saved shifts (npy): {npy_path}")

    csv_path = out_dir / "shifts.csv"
    np.savetxt(csv_path, shifts, delimiter=",", header="shift_y,shift_x", comments="")
    print(f"✓ Saved shifts CSV: {csv_path}")

    print(f"\n✓ All results saved to: {out_dir.resolve()}")


# -------------------------------------------------------------------------
# Main pipeline wrapper
# -------------------------------------------------------------------------

def process_microscopy_data(data_path, file_type="auto"):
    """
    High-level function to:
    - Detect file type (if auto)
    - Load data
    - Register stack
    - Visualize results
    """
    data_path = Path(data_path)

    # Detect type if 'auto'
    is_zip = False
    if file_type == "auto":
        if data_path.is_dir():
            file_type = "directory"
        else:
            ext = data_path.suffix.lower()
            if ext in (".tif", ".tiff"):
                file_type = "tif"
            elif ext == ".nd2":
                file_type = "nd2"
            elif ext == ".zip":
                file_type = "zip"
                is_zip = True
            else:
                raise ValueError(f"Unknown extension: {ext}")
    elif file_type == "zip":
        is_zip = True

    print("\n" + "=" * 60)
    print("MICROSCOPY IMAGE REGISTRATION PIPELINE")
    print("=" * 60)
    print(f"Input path:  {data_path}")
    print(f"File type:   {file_type}")
    print(f"Is ZIP:      {is_zip}")
    print("=" * 60)

    # Handle ZIP specially
    if is_zip:
        extract_path = load_zip_dataset(data_path)
        # Try to find ND2 first, then TIF
        nd2_files = list(Path(extract_path).rglob("*.nd2"))
        tif_files = list(Path(extract_path).rglob("*.tif")) + list(Path(extract_path).rglob("*.tiff"))

        if nd2_files:
            inner_path = nd2_files[0]
            inner_type = "nd2"
            print(f"📌 Using ND2 file inside ZIP: {inner_path}")
        elif tif_files:
            if len(tif_files) == 1:
                inner_path = tif_files[0]
                inner_type = "tif"
                print(f"📌 Using TIF stack inside ZIP: {inner_path}")
            else:
                inner_path = tif_files[0].parent
                inner_type = "directory"
                print(f"📌 Using directory of TIFs: {inner_path}")
        else:
            raise ValueError("No ND2/TIF files found inside ZIP.")
        data_path = inner_path
        file_type = inner_type

    # Load stack
    if file_type == "tif":
        image_stack = load_tif_stack(data_path)
    elif file_type == "nd2":
        image_stack = load_nd2_data(data_path)
    elif file_type == "directory":
        image_stack = load_image_directory(data_path)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")

    # Register
    print("\n" + "=" * 60)
    print("PERFORMING IMAGE REGISTRATION")
    print("=" * 60)

    registered_stack, shifts = register_image_stack(
        image_stack,
        reference_idx=config.reference_frame,
        upsample_factor=config.upsample_factor,
    )

    # Visualize
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    visualize_shifts(shifts)
    visualize_before_after(image_stack, registered_stack)

    print("\n" + "=" * 60)
    print("✓ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Original shape:   {image_stack.shape}")
    print(f"Registered shape: {registered_stack.shape}")
    print(f"Shifts shape:     {shifts.shape}")

    return image_stack, registered_stack, shifts


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Microscopy image registration (stage drift correction)."
    )
    parser.add_argument(
        "--path",
        "-p",
        required=True,
        help="Path to TIF / ND2 / ZIP / directory of TIF images.",
    )
    parser.add_argument(
        "--type",
        "-t",
        default="auto",
        choices=["auto", "tif", "nd2", "zip", "directory"],
        help="Force file type (default: auto-detect).",
    )
    parser.add_argument(
        "--ref",
        type=int,
        default=0,
        help="Reference frame index (default: 0).",
    )
    parser.add_argument(
        "--upsample",
        type=int,
        default=10,
        help="Upsample factor for subpixel registration (default: 10).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit maximum number of frames to process (default: all).",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save registered stack and shifts to ./output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply config
    config.reference_frame = args.ref
    config.upsample_factor = args.upsample
    config.max_frames = args.max_frames

    original, registered, shifts = process_microscopy_data(
        args.path,
        file_type=args.type,
    )

    # Example extra analysis (uncomment if you want)
    # analyze_drift_velocity(shifts)
    # plot_overlay_comparison(original, registered, frame_idx=10)

    if args.save_output:
        save_results(registered, shifts, output_dir="./output")


if __name__ == "__main__":
    main()
