

# Run this once at the top of the runtime
#pip install -q tifffile scikit-image matplotlib numpy scipy

"""## Step 2 – Imports and basic configuration"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import zipfile
from pathlib import Path

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, ball, closing
from scipy.ndimage import gaussian_filter

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

import warnings
warnings.filterwarnings("ignore")

# --- Default voxel size in micrometers (edit to your metadata) ---
voxel_size_xy = 0.2   # µm per pixel in x/y
voxel_size_z  = 0.5   # µm per slice in z

print("Imports complete.")
print(f"Default voxel sizes: xy = {voxel_size_xy} µm, z = {voxel_size_z} µm")

"""## Step 3 – Helper functions to load a 3D stack"""

def load_ome_tif_stack(path):
    """Load a 3D OME-TIF and reduce to (z, y, x)."""
    path = Path(path)
    print("Loading OME-TIF:", path)
    arr = tiff.imread(path)
    print("  Raw shape:", arr.shape)

    # Common shapes: (z,y,x), (t,z,y,x), (z,y,x,c), (t,c,z,y,x), (t,z,y,x,c)
    if arr.ndim == 3:
        stack = arr
    elif arr.ndim == 4:
        # could be (t,z,y,x) or (z,y,x,c)
        if arr.shape[-1] <= 4:      # (z,y,x,c)
            stack = arr[..., 0]
        else:                       # assume (t,z,y,x)
            stack = arr[0]
    elif arr.ndim == 5:
        # assume (t,c,z,y,x) or (t,z,y,x,c)
        if arr.shape[1] <= 4:       # (t,c,z,y,x)
            stack = arr[0, 0]
        else:                       # (t,z,y,x,c)
            stack = arr[0, ..., 0]
    else:
        raise ValueError(f"Unsupported OME-TIF shape: {arr.shape}")

    stack = stack.astype(np.float32)
    print("  Using stack shape (z,y,x):", stack.shape)
    return stack


def load_tif_directory(dir_path):
    """Load a directory of TIF slices into a 3D stack (z,y,x)."""
    dir_path = Path(dir_path)
    files = sorted(list(dir_path.glob("*.tif")) + list(dir_path.glob("*.tiff")))
    if not files:
        raise ValueError(f"No .tif/.tiff files found in directory {dir_path}")
    print(f"Loading {len(files)} TIF slices from {dir_path} ...")
    imgs = []
    for f in files:
        img = tiff.imread(f).astype(np.float32)
        if img.ndim == 2:
            imgs.append(img)
        elif img.ndim == 3 and img.shape[-1] <= 4:
            imgs.append(np.mean(img, axis=-1))
        else:
            raise ValueError(f"Unexpected TIF shape {img.shape} for file {f}")
    stack = np.stack(imgs, axis=0)
    print("  Stack shape (z,y,x):", stack.shape)
    return stack


def load_stack_from_path(path):
    """Load a 3D stack from either a .zip or a .ome.tif file.

    - If .zip:
        * extract to a folder
        * first search for *.ome.tif
        * if none found, load directory of TIF slices
    - If .ome.tif directly: load that file.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".zip":
        # Extract
        extract_dir = path.with_suffix("")  # folder next to zip
        print("Extracting ZIP to:", extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(extract_dir)
        print("  Extraction done.")

        # Prefer OME-TIF
        ome_files = list(extract_dir.rglob("*.ome.tif")) + list(extract_dir.rglob("*.ome.tiff"))
        if ome_files:
            print("  Found OME-TIF inside ZIP:", ome_files[0])
            return load_ome_tif_stack(ome_files[0])

        # Fall back to directory of TIF slices
        tif_files = list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.tiff"))
        if not tif_files:
            raise ValueError("ZIP contained no OME-TIF or TIF slices.")
        tif_dir = tif_files[0].parent
        print("  Using directory of TIF slices:", tif_dir)
        return load_tif_directory(tif_dir)

    elif ext in [".tif", ".tiff"] and "ome" in path.stem.lower():
        # Single OME-TIF
        return load_ome_tif_stack(path)
    else:
        raise ValueError("Please provide either a .zip or a .ome.tif file.")

"""## Step 4 – Segment biofilm and compute metrics"""

from skimage.morphology import remove_small_objects, ball, closing
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

def segment_biofilm(stack, smooth_sigma=1.0, min_size=100):
    """Segment biofilm from background in 3D."""
    print("Segmenting biofilm ...")
    smoothed = gaussian_filter(stack, sigma=smooth_sigma)
    th = threshold_otsu(smoothed)
    print("  Otsu threshold:", th)
    mask = smoothed > th

    mask = remove_small_objects(mask, min_size=min_size)
    mask = closing(mask, ball(1))

    print("  Foreground voxels:", int(mask.sum()))
    return mask, th


def compute_metrics(stack, mask, voxel_xy, voxel_z):
    """Compute biovolume, mean intensity, and thickness stats."""
    print("Computing metrics ...")
    voxel_volume = voxel_xy**2 * voxel_z

    biovolume = mask.sum() * voxel_volume
    mean_intensity = float(stack[mask].mean()) if mask.sum() > 0 else float("nan")

    thickness_slices = mask.sum(axis=0)         # (y,x)
    thickness_um = thickness_slices * voxel_z

    valid = thickness_slices > 0
    if np.any(valid):
        mean_thickness = float(thickness_um[valid].mean())
        max_thickness = float(thickness_um[valid].max())
    else:
        mean_thickness = float("nan")
        max_thickness = float("nan")

    metrics = {
        "biovolume_um3": biovolume,
        "mean_intensity": mean_intensity,
        "mean_thickness_um": mean_thickness,
        "max_thickness_um": max_thickness,
    }

    print("  Biovolume:      {:.2f} µm³".format(biovolume))
    print("  Mean intensity: {:.2f}".format(mean_intensity))
    print("  Mean thickness: {:.2f} µm".format(mean_thickness))
    print("  Max thickness:  {:.2f} µm".format(max_thickness))

    return metrics, thickness_um

"""## Step 5 – Visualization helpers"""

def show_z_projection(stack, mask=None, title_prefix="Biofilm"):
    mip = stack.max(axis=0)
    vmin, vmax = np.percentile(mip, [1, 99])

    plt.figure(figsize=(6, 5))
    plt.imshow(mip, cmap="gray", vmin=vmin, vmax=vmax)
    if mask is not None:
        mip_mask = mask.max(axis=0)
        plt.contour(mip_mask, colors="red", linewidths=0.5)
    plt.title(f"{title_prefix} – Max Intensity Projection")
    plt.axis("off")
    plt.show()


def plot_thickness_map(thickness_um):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(thickness_um, cmap="viridis")
    plt.title("Thickness map (µm)")
    plt.colorbar(im, label="Thickness (µm)")
    plt.axis("off")
    plt.show()


def plot_3d_point_cloud(stack, mask, thickness_um, voxel_xy, voxel_z,
                        color_mode="intensity", max_points=100_000):
    z_idx, y_idx, x_idx = np.nonzero(mask)
    if len(z_idx) == 0:
        print("No biofilm voxels to plot.")
        return

    # Downsample
    if len(z_idx) > max_points:
        idx = np.random.choice(len(z_idx), max_points, replace=False)
        z_idx, y_idx, x_idx = z_idx[idx], y_idx[idx], x_idx[idx]

    if color_mode == "intensity":
        colors = stack[z_idx, y_idx, x_idx]
    elif color_mode == "thickness":
        colors = thickness_um[y_idx, x_idx]
    else:
        raise ValueError("color_mode must be 'intensity' or 'thickness'")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x_idx * voxel_xy,
        y_idx * voxel_xy,
        z_idx * voxel_z,
        c=colors,
        cmap="viridis",
        s=1,
    )

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    ax.set_title(f"3D biofilm – color = {color_mode}")

    fig.colorbar(sc, label=color_mode)
    plt.tight_layout()
    plt.show()


# Run this cell in Google Colab to upload your data file

from google.colab import files

print("Please choose a .zip or .ome.tif file from your computer...")
uploaded = files.upload()

if not uploaded:
    raise RuntimeError("No file uploaded.")

uploaded_name = next(iter(uploaded.keys()))
print("Uploaded file:", uploaded_name)

# Save the uploaded bytes to disk so that tifffile/zipfile can read it
with open(uploaded_name, "wb") as f:
    f.write(uploaded[uploaded_name])

data_path = uploaded_name
print("Saved to local path:", data_path)


# This cell uses the `data_path` saved above.
# It will work for both .zip and .ome.tif uploads.

stack = load_stack_from_path(data_path)
mask, th = segment_biofilm(stack, smooth_sigma=1.0, min_size=100)
metrics, thickness_um = compute_metrics(stack, mask, voxel_size_xy, voxel_size_z)

show_z_projection(stack, mask=mask)
plot_thickness_map(thickness_um)
plot_3d_point_cloud(stack, mask, thickness_um,
                    voxel_size_xy, voxel_size_z,
                    color_mode="intensity")  # or "thickness"

print("Metrics dictionary:")
print(metrics)


