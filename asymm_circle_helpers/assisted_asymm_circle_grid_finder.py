
import cv2
import numpy as np
from scipy.spatial import distance # For efficient nearest neighbor searches
import math # For checking NaN
from typing import List, Dict, Tuple, Optional


def outer_corner_assisted_local_vector_walk(
    image_to_select_on: np.ndarray,
    detected_keypoints: List[cv2.KeyPoint],
    objp: np.ndarray,
    pattern_size: Tuple[int, int],
    visualize: bool = True
) -> Optional[np.ndarray]:
    """
    Orchestrates the calibration grid detection process.

    This function first prompts the user to select four corners, then estimates a
    homography, and finally walks the grid using local vectors to find all points.

    Args:
        image_to_select_on: The color image to process.
        detected_keypoints: A list of cv2.KeyPoint found by a blob detector.
        objp: The (N, 3) array of ideal object points.
        pattern_size: A tuple (cols, rows) defining the grid dimensions.
        visualize: If True, show debugging visualizations for the grid walk.

    Returns:
        A numpy array of shape (rows*cols, 1, 2) with the float32 corner
        coordinates, or None if the process fails or is aborted.
    """
    cols, rows = pattern_size
    num_expected_points = cols * rows
    print("\n--- Four-Corner + Local Vector NN Walk ---")

    kp_coords = np.array([kp.pt for kp in detected_keypoints], dtype=np.float32)
    if len(kp_coords) < num_expected_points:
        print(f"  Error: Not enough blobs detected (< {num_expected_points}).")
        return None

    # 1. Get 4 Corner Clicks from the user
    corner_labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]

    while True:
        clicked_indices = _get_four_corners_from_user(image_to_select_on, kp_coords, corner_labels)
        if not clicked_indices:
            print("Corner selection aborted.")
            return None

        # 2. Estimate Transform and Predict all points for vector calculation
        ideal_corner_indices = [0, cols - 1, (rows - 1) * cols, num_expected_points - 1]
        label_to_ideal_map = dict(zip(corner_labels, ideal_corner_indices))

        img_pts_corners = np.array([kp_coords[clicked_indices[lbl]] for lbl in corner_labels], dtype=np.float32)
        obj_pts_corners = np.array([objp[label_to_ideal_map[lbl], :2] for lbl in corner_labels], dtype=np.float32)

        try:
            H, _ = cv2.findHomography(obj_pts_corners, img_pts_corners, cv2.RANSAC, 3.0)
            if H is None: raise ValueError("Homography matrix is None.")
            predicted_img_pts_all = cv2.perspectiveTransform(objp[:, :2].reshape(-1, 1, 2), H)
            if predicted_img_pts_all is None: raise ValueError("perspectiveTransform returned None.")
            predicted_img_pts_all = predicted_img_pts_all.reshape(-1, 2)
        except Exception as e:
            print(f"  Error during perspective transform: {e}")
            return None
        print("  Transform estimated and all grid points predicted.")

        # 3. Perform the serpentine grid walk
        final_corners = _walk_grid_with_local_vectors(
            image=image_to_select_on,
            all_kp_coords=kp_coords,
            clicked_indices=clicked_indices,
            predicted_pts=predicted_img_pts_all,
            pattern_size=pattern_size,
            label_to_ideal_map=label_to_ideal_map,
            visualize=visualize
        )

        # 4. Final check and return
        if final_corners is None:
            print("Grid walk failed or was aborted, reselect corners or press 'q' to abort this image.")
            continue

        final_matched_count = num_expected_points - np.count_nonzero(np.isnan(final_corners[:, 0, 0]))
        print(f"Finished. Matched {final_matched_count}/{num_expected_points} points.")

        if final_matched_count != num_expected_points:
            print("  Warning: Failed to match all points. Result is incomplete, reselect corners or press 'q' to abort this image.")
            continue

        print("  Successfully matched all points.")

        return final_corners
    

def _get_four_corners_from_user(
    image: np.ndarray,
    kp_coords: np.ndarray,
    corner_labels: List[str]
) -> Optional[Dict[str, int]]:
    """

    Handles the GUI interaction for selecting four corner keypoints.

    Args:
        image: The image to display.
        kp_coords: Coordinates of all detected keypoints.
        corner_labels: A list of four strings, e.g., ["Top-Left", ...].

    Returns:
        A dictionary mapping corner labels to their keypoint indices,
        or None if the user quits.
    """
    clicked_kp_indices: Dict[str, int] = {}
    current_click_idx = 0
    
    win_name = f"Click {len(corner_labels)} Corners ({', '.join(lbl for lbl in corner_labels)})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1000, 700)

    # Create a local display copy to draw on
    display_img = image.copy()
    
    def _redraw_display():
        """Helper to reset and draw the current state on the display image."""
        display_img[:] = image[:]
        # Draw all available keypoints
        for i, pt in enumerate(kp_coords):
            if i not in clicked_kp_indices.values():
                 cv2.circle(display_img, tuple(pt.astype(int)), 3, (200, 200, 200), 1)

        # Draw selected keypoints with more emphasis
        colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
        for i, label in enumerate(corner_labels):
            if label in clicked_kp_indices:
                kp_idx = clicked_kp_indices[label]
                pt_int = tuple(kp_coords[kp_idx].astype(int))
                cv2.circle(display_img, pt_int, 6, colors[i], 2)
                cv2.circle(display_img, pt_int, 1, (0, 0, 0), -1)
                cv2.putText(display_img, label[:2], (pt_int[0] + 8, pt_int[1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_click_idx
        if current_click_idx >= len(corner_labels) or event != cv2.EVENT_LBUTTONDOWN:
            return

        target_label = corner_labels[current_click_idx]
        distances = distance.cdist([(x, y)], kp_coords)[0]
        nearest_kp_idx = np.argmin(distances)

        if nearest_kp_idx in clicked_kp_indices.values():
            print(f"  Warning: Blob {nearest_kp_idx} already selected. Click a different blob for {target_label}.")
            return
        
        clicked_kp_indices[target_label] = nearest_kp_idx
        pt_clicked = kp_coords[nearest_kp_idx]
        print(f"  Clicked {target_label} near blob {nearest_kp_idx} at ({pt_clicked[0]:.0f}, {pt_clicked[1]:.0f})")
        
        _redraw_display()
        current_click_idx += 1

        if current_click_idx < len(corner_labels):
            print(f"-> Now click: {corner_labels[current_click_idx]} or press 'r' to restart or 'q' to quit.")
        else:
            print("-> All 4 corners clicked. Press 'y' to proceed, 'r' to restart, 'q' to quit.")

    cv2.setMouseCallback(win_name, mouse_callback)
    _redraw_display()
    print(f"-> Click the blob for: {corner_labels[current_click_idx]}")

    while True:
        cv2.imshow(win_name, display_img)
        key = cv2.waitKey(20) & 0xFF
        
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1 or key == ord('q'):
            cv2.destroyWindow(win_name)
            return None
        
        if key == ord('r'):
            print("Restarting corner selection...")
            clicked_kp_indices.clear()
            current_click_idx = 0
            _redraw_display()
            print(f"-> Click the blob for: {corner_labels[current_click_idx]}")
        
        if current_click_idx == len(corner_labels):
            cv2.destroyWindow(win_name)
            return clicked_kp_indices

# --- Helper Function 2: Grid Walk Algorithm ---

def _find_nearest_in_radius(
    search_point: np.ndarray,
    available_coords: np.ndarray,
    available_indices: List[int],
    max_dist_sq: float
) -> Tuple[Optional[int], Optional[float]]:
    """Finds the nearest keypoint within a squared distance."""
    if len(available_indices) == 0:
        return None, None

    distances_sq = distance.cdist([search_point], available_coords, 'sqeuclidean')[0]
    min_idx_in_available = np.argmin(distances_sq)
    
    if distances_sq[min_idx_in_available] <= max_dist_sq:
        # Return the original keypoint index, not the index within the available list
        return available_indices[min_idx_in_available], distances_sq[min_idx_in_available]
    
    return None, None

def _walk_grid_with_local_vectors(
    image: np.ndarray,
    all_kp_coords: np.ndarray,
    clicked_indices: Dict[str, int],
    predicted_pts: np.ndarray,
    pattern_size: Tuple[int, int],
    label_to_ideal_map: Dict[str, int],
    visualize: bool
) -> Optional[np.ndarray]:
    """
    Performs the serpentine grid walk using local vector predictions.
    (This version restores the multiple search point logic for robustness).
    """
    # ... (initialization code is identical to the previous version) ...
    cols, rows = pattern_size
    num_expected_points = cols * rows
    final_corners = np.full((num_expected_points, 1, 2), np.nan, dtype=np.float32)

    # 1. Initialize available points pool
    available_indices = list(range(len(all_kp_coords)))
    for label, kp_idx in clicked_indices.items():
        grid_idx = label_to_ideal_map[label]
        final_corners[grid_idx, 0, :] = all_kp_coords[kp_idx]
        if kp_idx in available_indices:
            available_indices.remove(kp_idx)
    available_coords = all_kp_coords[available_indices]

    # 2. Estimate search radius
    tl = final_corners[label_to_ideal_map["Top-Left"], 0, :]
    br = final_corners[label_to_ideal_map["Bottom-Right"], 0, :]
    tr = final_corners[label_to_ideal_map["Top-Right"], 0, :]
    bl = final_corners[label_to_ideal_map["Bottom-Left"], 0, :]
    avg_diag = (np.linalg.norm(tl - br) + np.linalg.norm(tr - bl)) / 2.0
    grid_diag = np.sqrt((cols - 1)**2 + (rows - 1)**2) if cols > 1 and rows > 1 else 1
    approx_spacing = avg_diag / grid_diag if grid_diag > 0 else 50.0
    search_radius = approx_spacing * 0.9
    max_dist_sq = search_radius**2
    print(f"  Approx pixel spacing: {approx_spacing:.1f}px, Search Radius: {search_radius:.1f}px")

    # 3. Generate serpentine walk order
    walk_order = []
    for r in range(rows):
        path = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in path:
            walk_order.append(r * cols + c)
    
    # 4. Perform the walk
    win_name = "Local Vector NN Walk"
    if visualize:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL); cv2.resizeWindow(win_name, 1000, 700)
    
    run_mode = False
    prev_grid_idx = walk_order[0]

    for current_grid_idx in walk_order:
        if not np.isnan(final_corners[current_grid_idx, 0, 0]):
            prev_grid_idx = current_grid_idx
            continue

        if not available_indices:
            print("  Error: Ran out of available blobs to match."); break

        last_found_pt = final_corners[prev_grid_idx, 0, :]
        
        # A. Define multiple search points for robustness (RESTORED LOGIC)
        vector_pred = predicted_pts[current_grid_idx] - predicted_pts[prev_grid_idx]
        search_points = [
            last_found_pt + vector_pred,        # Full vector
            last_found_pt + 0.5 * vector_pred   # Half vector
        ]

        # B. Find the best candidate match across all search points
        best_match = {'kp_idx': None, 'dist_sq': float('inf'), 'search_pt': None}
        for sp in search_points:
            found_kp_idx, dist_sq = _find_nearest_in_radius(
                sp, available_coords, available_indices, max_dist_sq
            )
            if found_kp_idx is not None and dist_sq < best_match['dist_sq']:
                best_match.update(kp_idx=found_kp_idx, dist_sq=dist_sq, search_pt=sp)

        # C. Process the best match found (if any)
        match_status = f"Grid {prev_grid_idx}->{current_grid_idx}"
        
        # --- Visualization for this step ---
        vis_step = None
        if visualize:
            vis_step = image.copy()
            # ... (drawing logic for available/found points is the same) ...
            for idx in range(len(all_kp_coords)):
                pt_int = tuple(all_kp_coords[idx].astype(int))
                is_available = idx in available_indices
                # Check if this point is already in final_corners (within small tolerance)
                is_found = np.any(np.all(np.isclose(final_corners[:, 0, :], all_kp_coords[idx], atol=1e-3), axis=1))
                color = (180, 180, 180) if is_available else (0, 180, 0)
                size = 3 if is_available else 5
                cv2.circle(vis_step, pt_int, size, color, -1 if is_found else 1)
            
            # Draw vectors and search areas for all search points
            last_pt_int = tuple(last_found_pt.astype(int))
            cv2.circle(vis_step, last_pt_int, 7, (0, 255, 0), 2)
            for sp in search_points:
                sp_int = tuple(sp.astype(int))
                cv2.line(vis_step, last_pt_int, sp_int, (255, 255, 0), 1)
                cv2.drawMarker(vis_step, sp_int, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
                cv2.circle(vis_step, sp_int, int(search_radius), (255, 0, 0), 1)
        # --- End Visualization ---

        if best_match['kp_idx'] is not None:
            found_kp_idx = best_match['kp_idx']
            found_pt = all_kp_coords[found_kp_idx]
            final_corners[current_grid_idx, 0, :] = found_pt
            prev_grid_idx = current_grid_idx

            match_status += f" -> MATCH Blob {found_kp_idx} (Dist: {math.sqrt(best_match['dist_sq']):.1f})"
            idx_in_available = available_indices.index(found_kp_idx)
            available_indices.pop(idx_in_available)
            available_coords = np.delete(available_coords, idx_in_available, axis=0)

            if visualize:
                cv2.circle(vis_step, tuple(found_pt.astype(int)), 6, (0, 255, 255), 2)
        else:
            match_status += " -> NO MATCH"
            if visualize:
                for sp in search_points:
                    cv2.circle(vis_step, tuple(sp.astype(int)), int(search_radius), (0, 0, 255), 2)

        if visualize:
            cv2.putText(vis_step, match_status, (10, vis_step.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win_name, vis_step)
            key = cv2.waitKey(1 if run_mode else 0) & 0xFF
            if key == ord('q'): return None
            if key == ord('r'): run_mode = True

    if visualize:
        cv2.destroyAllWindows()
        
    return final_corners