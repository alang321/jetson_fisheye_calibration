#!/usr/bin/env python3
"""
lens_shape_plotter_outlier.py
-----------------------------

Plots fisheye "lens shape" curves from calibration JSONs
with group-based coloring and outlier highlighting.

Usage:
  python3 lens_shape_plotter_outlier.py --folder ./calibrations --group anton kalibr --save
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List


# ==========================================================
# --- Fisheye model utilities -------------------------------
# ==========================================================
def fisheye_theta_curve(k, n=300, max_theta_deg=100):
    theta = np.linspace(0, np.radians(max_theta_deg), n)
    theta2 = theta ** 2
    theta4 = theta2 ** 2
    theta6 = theta4 * theta2
    theta8 = theta4 ** 2
    theta_d = theta * (1 + k[0]*theta2 + k[1]*theta4 + k[2]*theta6 + k[3]*theta8)
    return np.degrees(theta), np.degrees(theta_d)


def fisheye_side_profile(k, n=300, max_theta_deg=100):
    theta = np.linspace(0, np.radians(max_theta_deg), n)
    theta2 = theta**2
    theta4 = theta2**2
    theta6 = theta4*theta2
    theta8 = theta4**2
    theta_d = theta * (1 + k[0]*theta2 + k[1]*theta4 + k[2]*theta6 + k[3]*theta8)
    theta_d = np.clip(theta_d, 0, np.radians(89.9))  # clamp to avoid tan() explosion
    f = 1.0
    r = f * np.tan(theta_d)
    x = f * np.cos(theta)
    y = r
    return x, y


# ==========================================================
# --- Color assignment -------------------------------------
# ==========================================================
MANUAL_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"
]

def assign_color(filename: str, groups: List[str]):
    if not groups:
        return "#999999"
    for i, g in enumerate(groups):
        if g.lower() in filename.lower():
            return MANUAL_COLORS[i % len(MANUAL_COLORS)]
    return "#999999"


# ==========================================================
# --- Main plotting logic ----------------------------------
# ==========================================================
def plot_fisheye_lens_shapes(folder: str, group_patterns: List[str] = None, save_fig: bool = True):
    json_files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
    if not json_files:
        print(f"No JSON files found in {folder}")
        return

    group_patterns = group_patterns or []

    # --- Step 1: collect all fisheye curves ---
    theta_common = np.linspace(0, 100, 200)
    thd_all, filenames, k_all = [], [], []

    for fname in json_files:
        path = os.path.join(folder, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            intr = data.get("intrinsics", {})
            k = intr.get("dist_coeff_drone") or data.get("D")
            k = np.array(k, dtype=float).flatten()
            if len(k) < 4:
                continue
            th, thd = fisheye_theta_curve(k)
            thd_interp = np.interp(theta_common, th, thd)
            thd_all.append(thd_interp)
            filenames.append(fname)
            k_all.append(k)
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    if not thd_all:
        print("No valid fisheye calibrations found.")
        return

    thd_all = np.vstack(thd_all)
    mean_curve = np.mean(thd_all, axis=0)
    std_curve = np.std(thd_all, axis=0)
    rms_errors = np.sqrt(np.mean((thd_all - mean_curve)**2, axis=1))
    threshold = np.percentile(rms_errors, 90)
    outlier_ids = np.where(rms_errors > threshold)[0]

    # --- Step 2: plot combined figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=False)

    for i, fname in enumerate(filenames):
        color = assign_color(fname, group_patterns)
        if not any(g.lower() in fname.lower() for g in group_patterns):
            continue
        lw = 1.4
        alpha = 0.9
        if i in outlier_ids:
            lw = 2.5
            alpha = 1.0
        ax1.plot(theta_common, thd_all[i], color=color, linewidth=lw, alpha=alpha)
        x, y = fisheye_side_profile(k_all[i])
        ax2.plot(x, y, color=color, linewidth=lw, alpha=alpha)

    # Mean + σ band
    ax1.plot(theta_common, mean_curve, "k", lw=2.0, label="Mean")
    ax1.fill_between(theta_common, mean_curve-std_curve, mean_curve+std_curve,
                     color="k", alpha=0.1, label="±1σ")

    # --- Top plot: angular mapping
    ax1.set_title("Angular Mapping (θd vs θ)")
    ax1.set_xlabel("Incident angle θ [°]")
    ax1.set_ylabel("Distorted angle θd [°]")
    ax1.grid(True)

    # --- Bottom plot: side profile
    ax2.set_title("Normalized Lens Side Profile")
    ax2.set_xlabel("Optical axis distance (normalized)")
    ax2.set_ylabel("Radial offset (normalized)")
    ax2.grid(True)
    ax2.set_aspect("equal", "box")
    ax2.set_ylim(-2, 2)

    # --- Group legend (color map) ---
    legend_patches = []
    if group_patterns:
        for i, g in enumerate(group_patterns):
            c = MANUAL_COLORS[i % len(MANUAL_COLORS)]
            legend_patches.append(Patch(color=c, label=f"group: {g}"))
    legend_patches.append(Patch(color="#999999", label="unmatched files"))

    fig.legend(handles=legend_patches, loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 0.82, 1])

    if save_fig:
        out_path = os.path.join(folder, "fisheye_lens_shapes_outliers.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved combined plot to {out_path}")
    
    plt.show()

    # --- Step 3: print summary table ---
    print("\nCalibration deviations (RMSE vs mean):")
    print("─" * 60)
    for i, (fname, err) in enumerate(sorted(zip(filenames, rms_errors),
                                           key=lambda x: -x[1])):
        flag = "OUTLIER" if err > threshold else ""
        print(f"{fname:45s}  {err:6.3f}°  {flag}")
    print("─" * 60)
    print(f"Mean RMSE = {np.mean(rms_errors):.3f}°, threshold = {threshold:.3f}°")


# ==========================================================
# --- CLI Entry Point --------------------------------------
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Plot fisheye lens shapes from calibration JSONs with outlier detection and group legend.")
    parser.add_argument("--folder", type=str, default="./calibrations",
                        help="Folder containing calibration JSONs")
    parser.add_argument("--save", action="store_true", help="Save combined plot to PNG file")
    parser.add_argument("--group", nargs="*", default=[],
                        help="Strings for grouping (e.g., --group anton kalibr)")
    args = parser.parse_args()

    print(f"Reading JSONs from: {args.folder}")
    if args.group:
        print(f"Grouping patterns: {args.group}")

    plot_fisheye_lens_shapes(args.folder, args.group, save_fig=args.save)


if __name__ == "__main__":
    main()
