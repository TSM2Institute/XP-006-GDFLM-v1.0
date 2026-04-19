"""
kappa_obs_reconstruction.py — XP-006 GDFLM v1.0

Reconstructs the 200×200 kappa_obs convergence map from the digitised
9-contour CSV (data/digitised/clowe2006_kappa.csv).

Logic adapted byte-for-byte from XP-005.1 src/module5_comparison.py
(parse_digitised_csv, transform_to_radec, radec_to_pixel,
kappa_value_from_name, reconstruct_kappa_obs).

Calibration anchors and centre coordinates are inherited from XP-005.1
without modification. Polygon fill via matplotlib.path.Path.

Output: 200×200 numpy array, normalised to [0, max] = [0, max/max] = [0,1]
where max is the highest kappa contour level present (kappa_0.37 in
practice). Comparison region is kappa_obs > 0 (≈7,487 pixels).

This module is the canonical regeneration path for kappa_obs from the
in-repo CSV. It is invoked by run_xp006.py and produces a reference
.npy snapshot at data/processed/kappa_obs.npy committed to git for
audit purposes.
"""

import csv
import os

import numpy as np
from matplotlib.path import Path as MplPath


# Calibration anchors for digitised CSV → RA/Dec (inherited from XP-005.1)
X1_DIG, RA1  = -5.6078e-3, 104.675
X2_DIG, RA2  = -4.9944,    104.550
Y1_DIG, DEC1 = -2.8039e-2, -55.9667
Y2_DIG, DEC2 = -5.5508e-2, -55.9333

# Standard Bullet Cluster centre (midpoint of Clowe figure)
RA_CENTRE  = 104.625
DEC_CENTRE = -55.950


def parse_digitised_csv(filepath):
    """Parse WebPlotDigitizer CSV with paired X,Y columns per contour."""
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 3:
        raise ValueError("Digitised CSV has fewer than 3 rows; cannot parse.")

    contour_header = rows[0]
    n_cols = len(contour_header)

    # Build contour name list (each contour spans 2 columns: X, Y)
    contours = []
    col = 0
    while col < n_cols:
        name = contour_header[col].strip()
        if not name:
            col += 1
            continue
        contours.append({"name": name, "x_col": col, "y_col": col + 1, "points": []})
        col += 2

    # Parse data rows
    for r in rows[2:]:
        for c in contours:
            if c["x_col"] >= len(r) or c["y_col"] >= len(r):
                continue
            xs = r[c["x_col"]].strip()
            ys = r[c["y_col"]].strip()
            if xs == "" or ys == "":
                continue
            try:
                x = float(xs)
                y = float(ys)
                c["points"].append((x, y))
            except ValueError:
                continue

    return contours


def transform_to_radec(x_dig, y_dig):
    """Linear interpolation from digitised CSV coordinates to RA/Dec."""
    ra  = RA1  + (x_dig - X1_DIG) * (RA2  - RA1)  / (X2_DIG - X1_DIG)
    dec = DEC1 + (y_dig - Y1_DIG) * (DEC2 - DEC1) / (Y2_DIG - Y1_DIG)
    return ra, dec


def radec_to_pixel(ra, dec, grid_params):
    """Convert RA/Dec to pixel coordinates on the 200×200 grid."""
    pixel_scale = grid_params["pixel_scale_arcsec"]
    cx, cy = grid_params["centre_pixel"]
    cos_dec = np.cos(np.radians(DEC_CENTRE))
    col = cx + (RA_CENTRE - ra) * 3600.0 * cos_dec / pixel_scale
    row = cy - (dec - DEC_CENTRE) * 3600.0 / pixel_scale
    return col, row


def kappa_value_from_name(name):
    """Extract numerical kappa level from contour name like 'kappa_0.23_main'."""
    parts = name.split("_")
    for p in parts[1:]:
        try:
            return float(p)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse kappa value from contour name: {name}")


def reconstruct_kappa_obs_unnormalised(contours, grid_params):
    """Fill polygon interiors with their corresponding kappa contour levels.

    Higher levels overwrite lower levels (so each pixel ends at the highest
    contour it is enclosed by). Returns the unnormalised array.
    """
    grid_size = grid_params["grid_size"]
    kappa_obs = np.zeros((grid_size, grid_size))

    levels = {}
    for c in contours:
        if not c["points"]:
            continue
        level = kappa_value_from_name(c["name"])
        levels.setdefault(level, []).append(c)

    inside_counts = []

    for level in sorted(levels.keys()):
        for c in levels[level]:
            radec_pts = [transform_to_radec(x, y) for (x, y) in c["points"]]
            pixel_pts = [radec_to_pixel(ra, dec, grid_params) for (ra, dec) in radec_pts]
            cols = np.array([p[0] for p in pixel_pts])
            rows = np.array([p[1] for p in pixel_pts])

            inside = (
                (cols >= 0) & (cols < grid_size) &
                (rows >= 0) & (rows < grid_size)
            )
            inside_counts.append((c["name"], int(np.sum(inside)), int(np.sum(~inside))))

            if len(pixel_pts) >= 3:
                first = pixel_pts[0]
                last = pixel_pts[-1]
                gap = float(np.hypot(first[0] - last[0], first[1] - last[1]))
                if gap > 5.0:
                    print(f"  WARNING: contour '{c['name']}' not closed "
                          f"(gap={gap:.2f} px); closing manually")
                pixel_pts_closed = pixel_pts + [pixel_pts[0]]

                poly = MplPath(pixel_pts_closed)
                yy, xx = np.mgrid[0:grid_size, 0:grid_size]
                pts_grid = np.column_stack([xx.ravel(), yy.ravel()])
                inside_mask = poly.contains_points(pts_grid).reshape(grid_size, grid_size)

                kappa_obs[inside_mask] = level

    return kappa_obs, inside_counts


def reconstruct_kappa_obs(csv_path, grid_params):
    """Top-level: parse CSV, reconstruct, normalise to peak=1.

    Returns (kappa_obs_normalised, kappa_obs_unnormalised, inside_counts).
    """
    contours = parse_digitised_csv(csv_path)
    kappa_unnorm, inside_counts = reconstruct_kappa_obs_unnormalised(contours, grid_params)
    max_unnorm = float(np.max(kappa_unnorm))
    if max_unnorm == 0:
        raise RuntimeError("kappa_obs is all zero after reconstruction.")
    kappa_obs = kappa_unnorm / max_unnorm
    return kappa_obs, kappa_unnorm, inside_counts


if __name__ == "__main__":
    # Standalone regeneration entry point — produces and saves the snapshot.
    base_dir   = os.path.join(os.path.dirname(__file__), "..")
    csv_path   = os.path.join(base_dir, "data", "digitised", "clowe2006_kappa.csv")
    grid_path  = os.path.join(base_dir, "data", "processed", "grid_params.json")
    out_path   = os.path.join(base_dir, "data", "processed", "kappa_obs.npy")
    out_unnorm = os.path.join(base_dir, "data", "processed", "kappa_obs_unnorm.npy")

    import json
    with open(grid_path) as f:
        grid_params = json.load(f)

    print(f"Reconstructing kappa_obs from {csv_path}")
    kappa_obs, kappa_unnorm, inside_counts = reconstruct_kappa_obs(csv_path, grid_params)
    print(f"\n  Contour | inside_pixels | outside_pixels")
    for name, ins, outs in inside_counts:
        print(f"  {name:<25s} | {ins:>6d}        | {outs:>6d}")
    print(f"\n  Unnormalised: min={kappa_unnorm.min():.4f}  max={kappa_unnorm.max():.4f}")
    print(f"  Normalised:   min={kappa_obs.min():.4f}  max={kappa_obs.max():.4f}")
    print(f"  Non-zero pixels: {int((kappa_obs > 0).sum())}")

    np.save(out_path,   kappa_obs)
    np.save(out_unnorm, kappa_unnorm)
    import hashlib
    with open(out_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"\n  Saved {out_path}")
    print(f"  Saved {out_unnorm}")
    print(f"  SHA-256 of kappa_obs.npy: {sha}")
