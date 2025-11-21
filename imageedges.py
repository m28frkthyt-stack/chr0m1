import io
import math

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# ───────────────────────────────────────────────────────────────
# Streamlit config
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Metal Spike Lab",
    page_icon="✨",
    layout="centered"
)

st.markdown(
    """
    <style>
    body { background: #020617; }
    .main { background: radial-gradient(circle at 10% 20%, #020617, #0f172a 70%); }
    .stApp { color: #e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚙️ Metal Spike Lab")
st.caption("Thin chrome edges + long thorny spikes + warps + glitches + background tricks (auto-pop).")

# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def hex_to_rgb(hex_color: str):
    """#RRGGBB -> (R,G,B) ints 0–255"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def apply_bg_filter(img_rgb: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply filter to background image (RGB uint8).
    Modes: 'None', 'Black & white', 'Soft desaturate', 'Blurred'
    """
    if mode == "None":
        return img_rgb

    if mode == "Black & white":
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return np.dstack([gray, gray, gray])

    if mode == "Soft desaturate":
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray3 = np.dstack([gray, gray, gray])
        out = img_rgb.astype(np.float32) * 0.4 + gray3 * 0.6
        return np.clip(out, 0, 255).astype(np.uint8)

    if mode == "Blurred":
        return cv2.GaussianBlur(img_rgb, (0, 0), 5)

    return img_rgb


# ───────────────────────────────────────────────────────────────
# Edge extraction (always thin)
# ───────────────────────────────────────────────────────────────
def filter_small_edges(edges: np.ndarray, min_edge_area: int) -> np.ndarray:
    """
    Remove tiny edge fragments using connected components.
    min_edge_area in pixels. 0 = no filtering.
    """
    if min_edge_area <= 0:
        return edges

    edges_bin = (edges > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges_bin, connectivity=8
    )

    filtered = np.zeros_like(edges_bin)
    for label in range(1, num_labels):  # 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_edge_area:
            filtered[labels == label] = 1

    filtered = (filtered * 255).astype(np.uint8)
    return filtered


def extract_edges_thin(
    img_bgr: np.ndarray,
    low_thresh: int,
    high_thresh: int,
    pre_blur_sigma: float,
    min_edge_area: int
) -> np.ndarray:
    """
    Thin Canny edges (no dilation). Always stays ~1px-ish.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if pre_blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), pre_blur_sigma)

    edges = cv2.Canny(gray, low_thresh, high_thresh)
    edges = filter_small_edges(edges, min_edge_area=min_edge_area)
    return edges


# ───────────────────────────────────────────────────────────────
# Thorny spikes (long, random, sparse seeds → separated rays)
# ───────────────────────────────────────────────────────────────
def crystal_spike_field_thorny(
    edges_f: np.ndarray,
    spike_length: float,
    decay_unused: float,     # kept for signature compatibility, ignored
    randomness: float,
    source_density: float,
    seed: int
) -> np.ndarray:
    """
    Thorny spikes via directional shifting, from a sparse set of seed points.
    edges_f: float 0–1, thin edges map.

    - source_density: controls how many seeds along edges (0–1).
      Lower = fewer, more isolated thorns.
    - spike_length: how far rays extend.
    - randomness: adds jitter and more directions.

    This version:
    - Samples explicit seed points from edges (image-size aware).
    - Casts long rays with constant strength, then shapes into thorns.
    """
    if spike_length <= 0 or source_density <= 0:
        return np.zeros_like(edges_f, dtype=np.float32)

    h, w = edges_f.shape
    rng = np.random.default_rng(seed)

    # ── 1) Pick sparse seed pixels along edges ─────────────────
    edges_bin = (edges_f > 0.1).astype(np.uint8)
    coords = np.argwhere(edges_bin > 0)
    if coords.size == 0:
        return np.zeros_like(edges_f, dtype=np.float32)

    total_area = float(h * w)
    # target seeds per megapixel, scaled by source_density
    #  e.g. 10–350 per MP
    scale = total_area / 1_000_000.0
    base_n = int((10 + 340 * source_density) * max(scale, 0.2))
    base_n = max(3, min(base_n, len(coords)))

    idx = rng.choice(len(coords), size=base_n, replace=False)
    seeds = np.zeros_like(edges_f, dtype=np.float32)
    for (y, x) in coords[idx]:
        seeds[y, x] = 1.0

    # ── 2) Cast rays from seeds in many directions ─────────────
    L = max(5, int(spike_length))  # enforce at least some length
    randomness = float(np.clip(randomness, 0.0, 1.0))

    # Base cardinal + diagonal directions
    base_dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1),
    ]

    # Extra random ray directions → more bramble-ish
    max_extra_dirs = 48
    extra_count = int(round(randomness * max_extra_dirs))
    extra_dirs = []
    for _ in range(extra_count):
        angle = rng.uniform(0, 2 * math.pi)
        dx = int(round(math.cos(angle)))
        dy = int(round(math.sin(angle)))
        if dx == 0 and dy == 0:
            continue
        extra_dirs.append((dx, dy))

    directions = base_dirs + extra_dirs
    spike = np.zeros_like(edges_f, dtype=np.float32)

    for dx, dy in directions:
        acc = np.zeros_like(edges_f, dtype=np.float32)
        for step in range(1, L + 1):
            sx = step * dx
            sy = step * dy

            # Jitter to break perfect straight lines
            if randomness > 0:
                jitter_amp = 4
                jx = rng.integers(-jitter_amp, jitter_amp + 1)
                jy = rng.integers(-jitter_amp, jitter_amp + 1)
                sx += int(jx * randomness)
                sy += int(jy * randomness)

            shifted = np.roll(seeds, shift=sy, axis=0)
            shifted = np.roll(shifted, shift=sx, axis=1)

            # constant weight (no distance decay → strong rays all along)
            acc = np.maximum(acc, shifted)
        spike = np.maximum(spike, acc)

    # ── 3) Shape + smooth → thorny rays, not blobs ─────────────
    # First smooth a bit to connect, then normalize
    spike = cv2.GaussianBlur(spike, (0, 0), 1.0)
    max_val = spike.max()
    if max_val > 1e-6:
        spike = spike / max_val

    # Emphasize rays and cut weak haze
    spike = np.power(spike, 1.2)
    spike[spike < 0.45] = 0.0
    spike = cv2.GaussianBlur(spike, (0, 0), 0.7)

    return np.clip(spike, 0.0, 1.0)


# ───────────────────────────────────────────────────────────────
# Chrome shading (metal look) with arbitrary edge color
# ───────────────────────────────────────────────────────────────
def chrome_shading_from_edges(
    edges: np.ndarray,
    metal_depth: float,
    halo_radius: float,
    light_angle_deg: float,
    specular_strength: float,
    shininess: float,
    glow_radius: float,
    glow_intensity: float,
    brightness: float,
    contrast: float,
    edge_color_rgb: tuple,
    spike_length: float,
    spike_decay: float,
    spike_strength: float,
    spike_randomness: float,
    spike_source_density: float,
    spike_thickness: float,
    seed: int
) -> (np.ndarray, np.ndarray):
    """
    Build a metal look from thin edges with thorny spikes.
    Auto-contrast so it always "pops".
    """
    edges_f = edges.astype(np.float32) / 255.0

    # Height map from edges, plus halo
    sigma = max(0.1, halo_radius) if halo_radius > 0 else 1.0
    height = cv2.GaussianBlur(edges_f, (0, 0), sigma) * metal_depth

    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)

    nz = np.ones_like(gx)
    nx = -gx
    ny = -gy
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx /= norm
    ny /= norm
    nz /= norm

    theta = math.radians(light_angle_deg)
    lx = math.cos(theta)
    ly = math.sin(theta)
    lz = 0.7
    ln = math.sqrt(lx * lx + ly * ly + lz * lz) + 1e-6
    lx /= ln
    ly /= ln
    lz /= ln

    diffuse = nx * lx + ny * ly + nz * lz
    diffuse = np.clip(diffuse, 0.0, 1.0)

    spec = np.power(np.clip(diffuse, 0.0, 1.0), shininess) * specular_strength

    # Thorny spikes from sparse seeds along thin edges
    spikes = crystal_spike_field_thorny(
        edges_f,
        spike_length=spike_length,
        decay_unused=spike_decay,
        randomness=spike_randomness,
        source_density=spike_source_density,
        seed=seed + 123
    )

    # Optional thickness growth (kept subtle, default small)
    if spike_thickness > 0:
        k = max(1, int(spike_thickness))
        ksize = 2 * k + 1
        kernel = np.ones((ksize, ksize), np.uint8)
        spikes_u8 = (spikes * 255).astype(np.uint8)
        spikes_u8 = cv2.dilate(spikes_u8, kernel, iterations=1)
        spikes = spikes_u8.astype(np.float32) / 255.0
        spikes = cv2.GaussianBlur(spikes, (0, 0), 0.8)

    spikes_scaled = spikes * spike_strength

    # ── Combine lighting & auto-pop ────────────────────────────
    ambient = 0.12          # base light
    kd = 1.0                # diffuse factor
    intensity = ambient + kd * diffuse + spec + spikes_scaled
    intensity = np.clip(intensity, 0.0, 1.0)

    if glow_radius > 0 and glow_intensity > 0:
        glow = cv2.GaussianBlur(intensity, (0, 0), glow_radius)
        intensity = np.clip(intensity + glow * glow_intensity, 0.0, 1.0)

    # Auto-contrast so it always uses the full 0–1 range
    min_i = float(intensity.min())
    max_i = float(intensity.max())
    if max_i - min_i > 1e-6:
        intensity = (intensity - min_i) / (max_i - min_i)

    # Strong highlight curve (brighter whites, punchy mids)
    intensity = np.power(intensity, 0.6)

    # Color edges according to chosen edge color
    edge_col = np.array(edge_color_rgb, dtype=np.float32).reshape(1, 1, 3)
    color = intensity[..., None] * edge_col
    color = np.clip(color * 255.0, 0, 255).astype(np.uint8)

    # Contrast & brightness on colored image
    color_f = color.astype(np.float32)
    color_f = (color_f - 127.5) * contrast + 127.5
    color_f = color_f * brightness
    color = np.clip(color_f, 0, 255).astype(np.uint8)

    # Mask background using edges + spikes
    mask = np.clip(edges_f * 1.2 + spikes * 0.9, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
    mask = np.expand_dims(mask, axis=2)
    metal_dark_bg = (color.astype(np.float32) * mask).astype(np.uint8)

    metal_bgr = cv2.cvtColor(metal_dark_bg, cv2.COLOR_RGB2BGR)
    return metal_bgr, spikes


# ───────────────────────────────────────────────────────────────
# Extra chaos: warp, kaleido, sparks, echo, scanlines, hue, grain
# ───────────────────────────────────────────────────────────────
def warp_image(img: np.ndarray, amount: float, seed: int) -> np.ndarray:
    if amount <= 0:
        return img

    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    noise_x = rng.standard_normal((h, w)).astype(np.float32)
    noise_y = rng.standard_normal((h, w)).astype(np.float32)

    noise_x = cv2.GaussianBlur(noise_x, (0, 0), 40)
    noise_y = cv2.GaussianBlur(noise_y, (0, 0), 40)

    max_disp = 40 * amount
    noise_x *= max_disp
    noise_y *= max_disp

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x.astype(np.float32) + noise_x).clip(0, w - 1)
    map_y = (grid_y.astype(np.float32) + noise_y).clip(0, h - 1)

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    return warped


def kaleido_mix(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img

    flip_h = cv2.flip(img, 1)
    flip_v = cv2.flip(img, 0)
    flip_hv = cv2.flip(img, -1)

    kaleido = (img.astype(np.float32) +
               flip_h.astype(np.float32) +
               flip_v.astype(np.float32) +
               flip_hv.astype(np.float32)) / 4.0
    kaleido = kaleido.astype(np.uint8)

    out = cv2.addWeighted(img, 1.0 - strength, kaleido, strength, 0)
    return out


def add_sparks(img: np.ndarray,
               spikes: np.ndarray,
               edges: np.ndarray,
               density: float,
               seed: int) -> np.ndarray:
    if density <= 0:
        return img

    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    spikes_f = spikes.astype(np.float32)
    edges_f = (edges.astype(np.float32) / 255.0)

    base = np.clip(spikes_f * 0.8 + edges_f * 0.4, 0.0, 1.0)
    noise = rng.random((h, w)).astype(np.float32)

    field = base * noise
    nonzero = field[field > 0]
    if nonzero.size == 0:
        return img

    keep_fraction = float(np.clip(density, 0.01, 0.99))
    thresh = np.quantile(nonzero, 1.0 - keep_fraction)
    mask = (field >= thresh).astype(np.float32)

    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
    mask = np.clip(mask * 1.8, 0.0, 1.0)

    spark = (mask * 255.0).astype(np.uint8)
    spark_b = spark
    spark_g = spark
    spark_r = np.clip(spark * 0.9, 0, 255).astype(np.uint8)
    spark_img = cv2.merge([spark_b, spark_g, spark_r])

    out = cv2.addWeighted(img, 1.0, spark_img, 1.2, 0)
    return out


def add_edge_echo(img: np.ndarray, amount: float, seed: int) -> np.ndarray:
    """
    Offset copies of the image to create echo trails.
    amount ~ 0–1
    """
    if amount <= 0:
        return img

    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)

    layers = int(2 + amount * 6)
    max_shift = int(40 * amount)
    alpha_step = 0.6 / max(layers, 1)

    for _ in range(layers):
        dx = rng.integers(-max_shift, max_shift + 1)
        dy = rng.integers(-max_shift, max_shift + 1)
        shifted = np.roll(img, shift=dy, axis=0)
        shifted = np.roll(shifted, shift=dx, axis=1)
        out = cv2.addWeighted(out, 1.0, shifted.astype(np.float32),
                              alpha_step, 0)

    return np.clip(out, 0, 255).astype(np.uint8)


def add_scanlines(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Horizontal scanlines overlay.
    strength ~ 0–1
    """
    if strength <= 0:
        return img

    h, w = img.shape[:2]
    overlay = img.astype(np.float32).copy()

    for y in range(0, h, 2):
        overlay[y:y+1, :, :] *= (1.0 - 0.4 * strength)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def shift_hue(img: np.ndarray, shift: float) -> np.ndarray:
    """
    Shift hue of RGB image by shift in degrees (0–360).
    """
    if abs(shift) < 1e-3:
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    # OpenCV hue is 0–180, map 0–360 → 0–180
    h = (h + (shift / 2.0)) % 180.0
    hsv_shifted = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2RGB)


def add_grain(img: np.ndarray, amount: float, seed: int) -> np.ndarray:
    """
    Add fine grain / noise overlay.
    amount ~ 0–1
    """
    if amount <= 0:
        return img

    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    # Generate single-channel noise, blur it, then ensure shape (H,W,1)
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 1.0)
    if noise.ndim == 2:
        noise = noise[..., None]  # (H,W) → (H,W,1)

    noise = noise / (np.max(np.abs(noise)) + 1e-6)

    scale = 40 * amount
    noisy = img.astype(np.float32) + noise * scale  # (H,W,1) broadcasts over (H,W,3)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ───────────────────────────────────────────────────────────────
# Glitch
# ───────────────────────────────────────────────────────────────
def apply_band_glitch(img: np.ndarray,
                      intensity: float,
                      seed: int) -> np.ndarray:
    if intensity <= 0:
        return img

    h, w, _ = img.shape
    rng = np.random.default_rng(seed)
    out = img.copy()

    num_bands = int(2 + intensity * 16)
    max_band_height = max(3, int(0.12 * h))
    max_shift = max(1, int(0.1 * w * intensity))

    for _ in range(num_bands):
        y = rng.integers(0, h - 1)
        band_h = rng.integers(2, max_band_height)
        y2 = min(h, y + band_h)
        shift = rng.integers(-max_shift, max_shift + 1)

        band = out[y:y2, :, :].copy()
        out[y:y2, :, :] = np.roll(band, shift, axis=1)

    return out


def apply_chromatic_aberration(img: np.ndarray,
                               amount: float) -> np.ndarray:
    if amount <= 0:
        return img

    h, w, _ = img.shape
    shift = int(amount * 10)

    b, g, r = cv2.split(img)

    M_r = np.float32([[1, 0, shift], [0, 1, 0]])
    r_shifted = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)

    M_b = np.float32([[1, 0, -shift], [0, 1, 0]])
    b_shifted = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)

    merged = cv2.merge([b_shifted, g, r_shifted])
    return merged


# ───────────────────────────────────────────────────────────────
# Full pipeline
# ───────────────────────────────────────────────────────────────
def process_image(
    img_bgr: np.ndarray,
    low_thresh: int,
    high_thresh: int,
    pre_blur_sigma: float,
    min_edge_area: int,
    metal_depth: float,
    halo_radius: float,
    light_angle_deg: float,
    specular_strength: float,
    shininess: float,
    glow_radius: float,
    glow_intensity: float,
    brightness: float,
    contrast: float,
    edge_color_rgb: tuple,
    spike_length: float,
    spike_decay: float,
    spike_strength: float,
    spike_randomness: float,
    spike_source_density: float,
    spike_thickness: float,
    warp_amount: float,
    kaleido_strength: float,
    spark_density: float,
    echo_amount: float,
    scanline_strength: float,
    hue_shift_deg: float,
    grain_amount: float,
    glitch_intensity: float,
    aberration_amount: float,
    seed: int
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Full pipeline -> final artwork + edges + spike mask.
    Returns: (BGR image, edges, spikes)
    """
    edges = extract_edges_thin(
        img_bgr,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        pre_blur_sigma=pre_blur_sigma,
        min_edge_area=min_edge_area
    )

    metal_bgr, spikes = chrome_shading_from_edges(
        edges,
        metal_depth=metal_depth,
        halo_radius=halo_radius,
        light_angle_deg=light_angle_deg,
        specular_strength=specular_strength,
        shininess=shininess,
        glow_radius=glow_radius,
        glow_intensity=glow_intensity,
        brightness=brightness,
        contrast=contrast,
        edge_color_rgb=edge_color_rgb,
        spike_length=spike_length,
        spike_decay=spike_decay,
        spike_strength=spike_strength,
        spike_randomness=spike_randomness,
        spike_source_density=spike_source_density,
        spike_thickness=spike_thickness,
        seed=seed
    )

    warped = warp_image(metal_bgr, amount=warp_amount, seed=seed + 1000)
    warped = kaleido_mix(warped, strength=kaleido_strength)
    warped = add_sparks(warped, spikes=spikes, edges=edges,
                        density=spark_density, seed=seed + 2000)
    warped = add_edge_echo(warped, amount=echo_amount, seed=seed + 2500)
    warped = add_scanlines(warped, strength=scanline_strength)

    # convert to RGB for hue + grain
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    warped_rgb = shift_hue(warped_rgb, hue_shift_deg)
    warped_rgb = add_grain(warped_rgb, amount=grain_amount, seed=seed + 2600)

    # back to BGR
    warped = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR)

    glitched = apply_band_glitch(warped, intensity=glitch_intensity, seed=seed + 3000)
    final = apply_chromatic_aberration(glitched, amount=aberration_amount)

    return final, edges, spikes


# ───────────────────────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image (PNG/JPEG)",
    type=["png", "jpg", "jpeg"],
)

with st.sidebar:
    st.header("🧵 Edge (thin core)")
    low_thresh = st.slider("Canny low threshold", 0, 255, 70)
    high_thresh = st.slider("Canny high threshold", 0, 255, 160)
    pre_blur_sigma = st.slider("Detail smoothing (pre-blur)", 0.0, 10.0, 1.0, step=0.1)
    min_edge_area = st.slider("Min edge area (px)", 0, 10000, 200, step=50)

    st.header("🎨 Edge Color")
    edge_color_hex = st.color_picker("Edge color", "#d0e8ff")
    edge_r, edge_g, edge_b = hex_to_rgb(edge_color_hex)
    edge_color_rgb = (edge_r / 255.0, edge_g / 255.0, edge_b / 255.0)

    st.header("🪙 Metal Shading")
    metal_depth = st.slider("Metal depth", 0.1, 5.0, 2.0, step=0.1)
    halo_radius = st.slider("Halo radius", 0.0, 30.0, 6.0, step=0.5)
    light_angle_deg = st.slider("Light angle (deg)", 0.0, 360.0, 40.0, step=1.0)
    specular_strength = st.slider("Specular strength", 0.0, 6.0, 2.0, step=0.1)
    shininess = st.slider("Shininess", 1.0, 120.0, 14.0, step=1.0)
    glow_radius = st.slider("Glow radius", 0.0, 30.0, 8.0, step=0.5)
    glow_intensity = st.slider("Glow intensity", 0.0, 6.0, 3.0, step=0.1)
    brightness = st.slider("Brightness", 0.5, 2.0, 1.3, step=0.05)
    contrast = st.slider("Contrast", 0.5, 3.0, 1.8, step=0.05)

    st.header("🌵 Thorny Spikes")
    spike_length = st.slider("Spike length (px)", 0.0, 300.0, 160.0, step=5.0)
    spike_decay = st.slider("Spike decay (legacy, minor effect)", 0.01, 0.99, 0.9, step=0.02)
    spike_strength = st.slider("Spike strength", 0.0, 8.0, 3.0, step=0.1)
    spike_randomness = st.slider("Spike randomness", 0.0, 1.0, 0.8, step=0.05)
    spike_source_density = st.slider("Spike source density (how many seeds)", 0.0, 1.0, 0.35, step=0.05)
    spike_thickness = st.slider("Spike thickness (subtle widening)", 0.0, 10.0, 2.0, step=0.5)

    st.header("🧪 Chaos")
    warp_amount = st.slider("Warp amount", 0.0, 1.0, 0.35, step=0.05)
    kaleido_strength = st.slider("Kaleido strength", 0.0, 1.0, 0.35, step=0.05)
    spark_density = st.slider("Spark density", 0.0, 1.0, 0.6, step=0.05)
    echo_amount = st.slider("Edge echo amount", 0.0, 1.0, 0.3, step=0.05)
    scanline_strength = st.slider("Scanlines strength", 0.0, 1.0, 0.3, step=0.05)
    hue_shift_deg = st.slider("Hue shift (deg)", 0.0, 360.0, 0.0, step=5.0)
    grain_amount = st.slider("Grain amount", 0.0, 1.0, 0.4, step=0.05)

    st.header("🎨 Background")
    bg_mode = st.selectbox(
        "Background mode",
        ["Transparent PNG", "Solid color", "Original image (filtered)"]
    )

    bg_color_hex = "#000000"
    bg_filter_mode = "None"
    if bg_mode == "Solid color":
        bg_color_hex = st.color_picker("Background color", "#000000")
    elif bg_mode == "Original image (filtered)":
        bg_filter_mode = st.selectbox(
            "Background filter",
            ["None", "Black & white", "Soft desaturate", "Blurred"]
        )

    # Overall effect opacity on top of background
    effect_opacity = st.slider("Overall effect opacity", 0.0, 1.0, 1.0, step=0.05)

    st.header("🧨 Glitch & Seed")
    glitch_intensity = st.slider("Glitchiness", 0.0, 1.0, 0.25, step=0.05)
    aberration_amount = st.slider("Chromatic aberration", 0.0, 1.0, 0.18, step=0.05)
    seed = st.slider("Random seed", 0, 9999, 42)

if uploaded_file is None:
    st.info("👆 Upload an image to start.")
else:
    pil_img = Image.open(uploaded_file).convert("RGB")
    cv2_img = pil_to_cv2(pil_img)

    max_side = 1000
    h, w, _ = cv2_img.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        cv2_img = cv2.resize(
            cv2_img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    orig_rgb_resized = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    artwork_bgr, edges, spikes = process_image(
        cv2_img,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        pre_blur_sigma=pre_blur_sigma,
        min_edge_area=min_edge_area,
        metal_depth=metal_depth,
        halo_radius=halo_radius,
        light_angle_deg=light_angle_deg,
        specular_strength=specular_strength,
        shininess=shininess,
        glow_radius=glow_radius,
        glow_intensity=glow_intensity,
        brightness=brightness,
        contrast=contrast,
        edge_color_rgb=edge_color_rgb,
        spike_length=spike_length,
        spike_decay=spike_decay,
        spike_strength=spike_strength,
        spike_randomness=spike_randomness,
        spike_source_density=spike_source_density,
        spike_thickness=spike_thickness,
        warp_amount=warp_amount,
        kaleido_strength=kaleido_strength,
        spark_density=spark_density,
        echo_amount=echo_amount,
        scanline_strength=scanline_strength,
        hue_shift_deg=hue_shift_deg,
        grain_amount=grain_amount,
        glitch_intensity=glitch_intensity,
        aberration_amount=aberration_amount,
        seed=seed
    )

    artwork_rgb = cv2.cvtColor(artwork_bgr, cv2.COLOR_BGR2RGB)

    # Alpha field from edges + spikes + brightness
    gray = cv2.cvtColor(artwork_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    edges_f = edges.astype(np.float32) / 255.0
    alpha_f = np.clip(gray * 0.4 + edges_f * 0.9 + spikes * 1.1, 0.0, 1.0)
    alpha_f = cv2.GaussianBlur(alpha_f, (0, 0), 1.5)
    alpha_f = np.clip(alpha_f * 2.5, 0.0, 1.0)

    # Apply overall effect opacity
    alpha_f = np.clip(alpha_f * effect_opacity, 0.0, 1.0)

    if bg_mode == "Transparent PNG":
        rgb_float = artwork_rgb.astype(np.float32) / 255.0
        rgb_premult = rgb_float * alpha_f[..., None]

        cutoff = 0.02
        rgb_premult[alpha_f < cutoff] = 0.0
        alpha_f[alpha_f < cutoff] = 0.0

        rgb_out = np.clip(rgb_premult * 255.0, 0, 255).astype(np.uint8)
        alpha = np.clip(alpha_f * 255.0, 0, 255).astype(np.uint8)

        artwork_rgba = np.dstack([rgb_out, alpha])
        display_img = Image.fromarray(artwork_rgba)

    elif bg_mode == "Solid color":
        bg_r, bg_g, bg_b = hex_to_rgb(bg_color_hex)
        h_img, w_img = artwork_rgb.shape[:2]
        bg = np.zeros_like(artwork_rgb, dtype=np.uint8)
        bg[:, :] = [bg_r, bg_g, bg_b]

        alpha_3 = alpha_f[..., None]
        comp_float = artwork_rgb.astype(np.float32) * alpha_3 + \
                     bg.astype(np.float32) * (1.0 - alpha_3)
        comp = np.clip(comp_float, 0, 255).astype(np.uint8)
        display_img = Image.fromarray(comp)

    else:
        bg_filtered = apply_bg_filter(orig_rgb_resized, bg_filter_mode)
        alpha_3 = alpha_f[..., None]
        comp_float = artwork_rgb.astype(np.float32) * alpha_3 + \
                     bg_filtered.astype(np.float32) * (1.0 - alpha_3)
        comp = np.clip(comp_float, 0, 255).astype(np.uint8)
        display_img = Image.fromarray(comp)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_img, use_column_width=True)
    with col2:
        st.subheader("Metal spike art")
        st.image(display_img, use_column_width=True)

    buf = io.BytesIO()
    display_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="💾 Download artwork (PNG)",
        data=byte_im,
        file_name="metal_spike_lab.png",
        mime="image/png"
    )