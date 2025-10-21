import numpy as np
import matplotlib.pyplot as plt

def visualize_asymmetric_pattern(cols: int, rows: int, spacing_mm: float, circle_radius_mm: float = 5.0):
    """
    Visualizes the asymmetric circle grid pattern and shows key dimensions.

    Args:
        cols: Number of circles per row (horizontally)
        rows: Number of rows (vertically)
        spacing_mm: Distance between neighboring circle centers in mm
        circle_radius_mm: Visual radius of each circle for plotting (default 5 mm)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # --- Generate object points (same logic as in generate_object_points) ---
    objp = np.zeros((cols * rows, 3), np.float32)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            row_offset = (r % 2) * (spacing_mm / 2.0)
            objp[idx, 0] = c * spacing_mm + row_offset
            objp[idx, 1] = r * spacing_mm

    xs = objp[:, 0]
    ys = objp[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.set_title(f"Asymmetric Circle Grid ({cols}×{rows}), Spacing = {spacing_mm} mm")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    # --- Draw circles ---
    for x, y in zip(xs, ys):
        circle = patches.Circle((x, y), radius=circle_radius_mm, color='lightblue', ec='k', lw=0.8)
        ax.add_patch(circle)

    # --- Draw example distance indicators ---
    if cols > 1:
        ax.plot([xs[0], xs[1]], [ys[0], ys[1]], 'r--', lw=1)
        mid_x = (xs[0] + xs[1]) / 2
        mid_y = (ys[0] + ys[1]) / 2
        distance = np.sqrt((xs[0] - xs[1])**2 + (ys[0] - ys[1])**2)
        ax.text(mid_x, mid_y + 0.5 * spacing_mm, f"{distance:.1f} mm", color='r', ha='center')

    if rows > 1:
        ax.plot([xs[0], xs[cols]], [ys[0], ys[cols]], 'g--', lw=1)
        mid_x = (xs[0] + xs[cols]) / 2
        mid_y = (ys[0] + ys[cols]) / 2
        distance = np.sqrt((xs[0] - xs[cols])**2 + (ys[0] - ys[cols])**2)
        ax.text(mid_x - 0.5 * spacing_mm, mid_y, f"{distance:.1f} mm", color='g', va='center', rotation=90)

    # --- Draw bounding box ---
    ax.set_xlim(min(xs) - spacing_mm, max(xs) + spacing_mm)
    ax.set_ylim(max(ys) + spacing_mm, min(ys) - spacing_mm)  # invert Y for camera-like orientation
    ax.invert_yaxis()

    # --- Annotate key stats ---
    total_width = max(xs) - min(xs)
    total_height = max(ys) - min(ys)
    ax.text(min(xs), min(ys) - 0.5 * spacing_mm,
            f"Pattern size: {total_width:.1f} × {total_height:.1f} mm "
            f"({cols}×{rows} circles)",
            fontsize=10, color='black', ha='left')

    plt.tight_layout()
    plt.show()
