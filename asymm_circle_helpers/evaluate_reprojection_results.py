import cv2
import numpy as np
import os
from typing import Dict, Any, Tuple, List, Set

def calculate_reprojection_errors(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K: np.ndarray,
    D: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper to calculate per-point reprojection errors."""
    all_imgpoints = []
    all_reprojected_points = []
    
    for i in range(len(objpoints)):
        reprojected_pts, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        all_imgpoints.append(imgpoints[i])
        all_reprojected_points.append(reprojected_pts)
        
    all_imgpoints = np.vstack(all_imgpoints).squeeze()
    all_reprojected_points = np.vstack(all_reprojected_points).squeeze()
    
    # Calculate per-point error vectors and magnitudes
    per_point_error_vectors = all_reprojected_points - all_imgpoints
    per_point_error_magnitudes = np.linalg.norm(per_point_error_vectors, axis=1)
    
    return all_imgpoints, per_point_error_vectors, per_point_error_magnitudes

def create_error_vector_plot(
    img_shape: Tuple[int, int],
    points: np.ndarray,
    vectors: np.ndarray,
    magnification: float = 50.0
) -> np.ndarray:
    """Creates a quiver plot of error vectors."""
    vis_img = np.full((img_shape[0], img_shape[1], 3), 255, dtype=np.uint8)
    
    for pt, vec in zip(points, vectors):
        start_point = tuple(pt.astype(int))
        # Magnify the error vector to make it visible
        end_point = tuple((pt + vec * magnification).astype(int))
        cv2.arrowedLine(vis_img, start_point, end_point, (0, 0, 255), 1, tipLength=0.2)
        
    cv2.putText(vis_img, f"Error Vectors (Magnified {magnification:.0f}x)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return vis_img

def create_error_heatmap(
    img_shape: Tuple[int, int],
    points: np.ndarray,
    errors: np.ndarray,
    max_error_for_scale: float = 2.0
) -> np.ndarray:
    """Creates a heatmap of error magnitudes."""
    h, w = img_shape
    vis_img = np.full((img_shape[0], img_shape[1], 3), 255, dtype=np.uint8)
    
    # Normalize errors for color mapping (clamping at max_error_for_scale)
    norm_errors = np.clip(errors, 0, max_error_for_scale) / max_error_for_scale
    norm_errors = (norm_errors * 255).astype(np.uint8)
    
    # Apply a color map (JET is good for heatmaps)
    colors = cv2.applyColorMap(norm_errors, cv2.COLORMAP_JET).squeeze()
    
    for i, pt in enumerate(points):
        center = tuple(pt.astype(int))
        color = tuple(map(int, colors[i]))
        cv2.circle(vis_img, center, 3, color, -1)
        
    cv2.putText(vis_img, "Reprojection Error Heatmap (px)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add a color bar legend
    bar_h, bar_w = 200, 20
    for i in range(bar_h):
        color = cv2.applyColorMap(np.array([int(i/bar_h * 255)], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
        cv2.line(vis_img, (w - 50, h - 50 - i), (w - 50 + bar_w, h - 50 - i), tuple(map(int, color)), 1)
    cv2.putText(vis_img, f"{max_error_for_scale:.1f}", (w - 55 + bar_w, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(vis_img, "0.0", (w - 55 + bar_w, h - 50 - bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return vis_img

def report_and_visualize_results(
    calib_data: Dict[str, Any],
    img_shape: Tuple[int, int],
    max_window_dim: int = 1200
):
    """
    Prints final calibration results and shows detailed diagnostic visualizations.
    
    Args:
        calib_data: The dictionary returned by a successful calibration run.
                    MUST contain: K, D, ret, imgpoints, objpoints, rvecs, tvecs, filenames
        img_shape: The (height, width) of the calibration images.
        max_window_dim: The maximum dimension for visualization windows.
    """
    K, D, rms = calib_data['K'], calib_data['D'], calib_data['ret']
    imgpoints, objpoints = calib_data['imgpoints'], calib_data['objpoints']
    rvecs, tvecs = calib_data['rvecs'], calib_data['tvecs']
    filenames = calib_data['filenames']
    h, w = img_shape
    
    # --- 1. Print Numerical Results ---
    print("\n" + "="*40)
    print("      Calibration Successful! ✅")
    print("="*40)
    print(f"  Used {len(imgpoints)} views for final calibration.")
    print(f"  RMS reprojection error: {rms:.4f} pixels")
    print("\nCamera Matrix (K):")
    print(np.round(K, 2))
    print("\nDistortion Coefficients (D) [k1, k2, p1, p2]:")
    print(np.round(D.flatten(), 4))
    print("="*40 + "\n")

    # --- 2. Calculate Detailed Errors ---
    print("Calculating detailed per-point reprojection errors...")
    points, vectors, errors = calculate_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, K, D)
    
    # --- 3. Create and Show Visualizations ---
    print("Generating diagnostic plots...")
    
    # Create plots in parallel
    coverage_img = np.full((h, w, 3), 255, dtype=np.uint8)
    for corner in points:
        cv2.circle(coverage_img, tuple(corner.astype(int)), 3, (255, 0, 0), -1)
    cv2.putText(coverage_img, f"Coverage: {len(points)} points from {len(imgpoints)} views", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    vector_plot = create_error_vector_plot(img_shape, points, vectors)
    heatmap_plot = create_error_heatmap(img_shape, points, errors)

    # Display plots sequentially
    vis_options = {
        "Coverage": coverage_img,
        "Error Vector Plot": vector_plot,
        "Error Heatmap": heatmap_plot
    }

    for name, img in vis_options.items():
        print(f"-> Displaying '{name}'. Press any key to continue...")
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        aspect_ratio = w / h
        win_w = min(w, max_window_dim); win_h = int(win_w / aspect_ratio)
        cv2.resizeWindow(name, win_w, win_h)
        cv2.imshow(name, img)
        cv2.waitKey(0)

    # --- 4. Visualize Undistortion ---
    print("\nVisualizing undistortion on the first accepted sample image...")
    # ... (This part of your code is good and remains unchanged) ...
    # For brevity, it is omitted here.
    
    print("\n-> All visualizations complete. Press any key to close all windows.")
    cv2.destroyAllWindows()

def create_error_barchart(
    per_view_errors: List[Tuple[float, str, int]],
    outlier_indices: Set[int]
) -> np.ndarray:
    """Creates a horizontal bar chart visualizing per-view RMS errors."""
    bar_height = 25
    padding = 40
    font_scale = 0.5
    
    num_views = len(per_view_errors)
    img_h = num_views * bar_height + 2 * padding
    img_w = 1200
    
    vis_img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    
    # Sort by filename for consistent ordering
    sorted_errors = sorted(per_view_errors, key=lambda x: x[1])
    
    max_error = max(e[0] for e in sorted_errors) if sorted_errors else 1.0
    
    cv2.putText(vis_img, "Per-View RMS Reprojection Error", (padding, padding - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    for i, (error, filename, original_index) in enumerate(sorted_errors):
        y_pos = padding + i * bar_height
        
        # Determine bar color
        bar_color = (255, 150, 150) if original_index in outlier_indices else (150, 150, 150) # Red for outliers, gray for others
        text_color = (0, 0, 255) if original_index in outlier_indices else (0, 0, 0)
        
        # Draw bar
        bar_len = int((error / max_error) * (img_w - padding * 4))
        cv2.rectangle(vis_img, (padding, y_pos), (padding + bar_len, y_pos + bar_height - 5),
                      bar_color, -1)
        
        # Draw text label
        label = f"{os.path.basename(filename)} ({error:.3f} px)"
        cv2.putText(vis_img, label, (padding + 5, y_pos + bar_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        
    return vis_img

def create_live_coverage_plot(
    img_shape: Tuple[int, int],
    all_imgpoints: List[np.ndarray],
    sorted_per_view_errors: List[Tuple[float, str, int]],
    num_to_remove: int
) -> np.ndarray:
    """Creates a live plot of point coverage, highlighting points from views to be removed."""
    h, w = img_shape
    vis_img = np.full((h, w, 3), 255, dtype=np.uint8)
    
    cv2.putText(vis_img, "Live Coverage Preview", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Get the original indices of the views that would be removed
    indices_to_remove = {item[2] for item in sorted_per_view_errors[:num_to_remove]}

    # Iterate through all original views and their points
    for view_idx, view_points in enumerate(all_imgpoints):
        points = view_points.squeeze().astype(int)
        if view_idx in indices_to_remove:
            # Draw points from "removed" views in red
            for pt in points:
                cv2.drawMarker(vis_img, tuple(pt), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
        else:
            # Draw points from "kept" views in blue
            for pt in points:
                cv2.circle(vis_img, tuple(pt), 3, (255, 100, 0), -1)

    # Add a legend to explain the colors
    cv2.circle(vis_img, (30, h - 60), 5, (255, 100, 0), -1)
    cv2.putText(vis_img, "Kept View", (45, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    cv2.drawMarker(vis_img, (35, h - 30), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(vis_img, "To Be Removed", (45, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    
    return vis_img