import cv2
import numpy as np
import os
import glob
import argparse
import sys
from scipy.spatial import distance, KDTree # For efficient nearest neighbor searches
from sklearn.neighbors import NearestNeighbors
import math # For checking NaN
from typing import List, Dict, Tuple, Optional, Any, Set
from sklearn.cluster import DBSCAN
from sklearn.neighbors import radius_neighbors_graph

# --- Main Public Function (Controller) ---

def run_four_corner_local_vector_walk(
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
        auto_grid = find_grid_adaptive(
            keypoints=detected_keypoints,
            pattern_size=(cols, rows),
            visualize=True
        )

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
    
# _find_best_row_in_pool remains the same as the previous version.
# The visualization for its output is handled in the main function.

# This function is unchanged as it contains no visualization code.
def _find_best_row_in_pool(
    pool_coords: np.ndarray,
    pool_indices: np.ndarray,
    num_cols_expected: int,
    nn_search_k: int = 7,
    search_radius_factor: float = 0.6,
    consistency_weight: float = 0.5,
    visualize: bool = False,
) -> Optional[np.ndarray]:
    """
    Searches a pool of points to find the most consistent, straightest possible row.

    Args:
        pool_coords: Coordinates of points to search within.
        pool_indices: Original indices corresponding to pool_coords.
        num_cols_expected: The target length of the row.
        nn_search_k: Number of nearest neighbors to consider for starting a chain.
        search_radius_factor: Multiplier for local spacing to define search radius.
        consistency_weight: Weight for scoring angle vs. distance consistency.

    Returns:
        An array of original indices representing the best row found, or None.
    """
    if len(pool_coords) < num_cols_expected:
        return None

    kdtree = KDTree(pool_coords)
    found_rows = []

    for i, p1_coord in enumerate(pool_coords):
        # Find k nearest neighbors to start a chain
        distances, neighbor_indices = kdtree.query(p1_coord, k=nn_search_k)
        
        for j, p2_coord in zip(neighbor_indices[1:], pool_coords[neighbor_indices[1:]]):
            # Start a new chain
            current_chain_indices = [i, j]
            
            while len(current_chain_indices) < num_cols_expected:
                # Predict the next point's position
                p_prev_idx = current_chain_indices[-2]
                p_curr_idx = current_chain_indices[-1]
                
                p_prev, p_curr = pool_coords[p_prev_idx], pool_coords[p_curr_idx]
                
                step_vector = p_curr - p_prev
                predicted_coord = p_curr + step_vector
                
                # Search for the best match near the prediction
                step_dist = np.linalg.norm(step_vector)
                search_radius = step_dist * search_radius_factor
                
                possible_next_indices = kdtree.query_ball_point(predicted_coord, r=search_radius)
                
                best_next_idx = -1
                min_dist_to_pred = float('inf')
                
                for next_idx in possible_next_indices:
                    if next_idx not in current_chain_indices:
                        dist = np.linalg.norm(pool_coords[next_idx] - predicted_coord)
                        if dist < min_dist_to_pred:
                            min_dist_to_pred = dist
                            best_next_idx = next_idx
                
                if best_next_idx != -1:
                    current_chain_indices.append(best_next_idx)
                else:
                    break # Chain broken
            
            if len(current_chain_indices) == num_cols_expected:
                found_rows.append(current_chain_indices)

    if not found_rows:
        return None
    

    # Score the found rows based on consistency of angle and distance
    best_row = None
    min_score = float('inf')

    for row_local_indices in found_rows:
        row_coords = pool_coords[row_local_indices]
        vectors = np.diff(row_coords, axis=0)
        distances = np.linalg.norm(vectors, axis=1)
        
        # Normalize vectors to get angles
        norm_vectors = vectors / distances[:, np.newaxis]
        # Dot product of adjacent normalized vectors gives cosine of angle change
        angle_cosines = np.sum(norm_vectors[:-1] * norm_vectors[1:], axis=1)
        
        # Score is a mix of distance variation and angle variation
        dist_score = np.std(distances) / np.mean(distances)
        angle_score = np.mean(1 - angle_cosines) # smaller angle change is better
        
        score = dist_score + angle_score * consistency_weight
        
        if score < min_score:
            min_score = score
            best_row = row_local_indices

    # visualise all rows in opencv window and highlight the best row if it exists
    if visualize:
        cv2.namedWindow("Best Row Visualization", cv2.WINDOW_NORMAL)
        # Create an empty canvas sized to include all pool points
        h = int(np.max(pool_coords[:,1])) + 50
        w = int(np.max(pool_coords[:,0])) + 50
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)

        # Generate a distinct color for each candidate row using HSV hues
        num_rows = len(found_rows)
        row_colors = []
        for i in range(num_rows):
            hue = int(180.0 * i / max(1, num_rows-1))  # spread hues 0–180
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            row_colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))

        # Draw each row in its unique color
        for i, row_indices in enumerate(found_rows):
            color = row_colors[i]
            # draw points
            for idx in row_indices:
                pt = tuple(pool_coords[idx].astype(int))
                cv2.circle(vis_img, pt, 4, color, -1)
            # connect in sequence
            pts = [tuple(pool_coords[idx].astype(int)) for idx in row_indices]
            for j in range(len(pts) - 1):
                cv2.line(vis_img, pts[j], pts[j+1], color, 1)

        # Highlight the best row in red on top
        for idx in best_row:
            pt = tuple(pool_coords[idx].astype(int))
            cv2.circle(vis_img, pt, 6, (0, 0, 255), 2)

        cv2.imshow("Best Row Visualization", vis_img)
        cv2.waitKey(0)

    return pool_indices[best_row]


# This function is unchanged. It uses the window created by the main controller.
def _propagate_grid_asymmetric(
    row0_indices: np.ndarray,
    all_coords: np.ndarray,
    kdtree_all: KDTree,
    available_mask: np.ndarray,
    pattern_size: Tuple[int, int],
    visualize: bool = False,
    vis_img_base: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    Propagates a grid assuming an asymmetric (hexagonal/staggered) layout.
    """
    num_cols, num_rows = pattern_size
    grid = np.full((num_rows, num_cols), -1, dtype=int)
    grid[0, :] = row0_indices
    available_mask[row0_indices] = False
    
    vis_img = None
    if visualize and vis_img_base is not None:
        vis_img = vis_img_base.copy()

    # --- Step 1: Find the second row robustly ---
    print("\n--- Stage 2: Finding Second Row (Asymmetric) ---")
    row1_indices = np.full(num_cols, -1, dtype=int)
    for c in range(num_cols):
        p0_idx = grid[0, c]
        p0_coord = all_coords[p0_idx]
        
        # Find nearest neighbor not in the seed row
        _, neighbor_indices = kdtree_all.query(p0_coord, k=5)
        best_match = -1
        for n_idx in neighbor_indices:
            if available_mask[n_idx]:
                best_match = n_idx
                break
        if best_match == -1:
            print(f"  [ propagate FAIL ] Could not find a free neighbor for Row 1, Col {c+1}.")
            return None
        row1_indices[c] = best_match
    
    # Check consistency of Row 1 before accepting
    row0_vectors = np.diff(all_coords[grid[0, :]], axis=0)
    row1_vectors = np.diff(all_coords[row1_indices], axis=0)
    if not np.allclose(np.linalg.norm(row0_vectors, axis=1), np.linalg.norm(row1_vectors, axis=1), rtol=0.3):
         print("  [ propagate FAIL ] Found second row is not geometrically consistent with the first.")
         return None

    grid[1, :] = row1_indices
    available_mask[row1_indices] = False
    if visualize:
        for c in range(num_cols):
            p0_coord = all_coords[grid[0, c]]
            p1_coord = all_coords[grid[1, c]]
            cv2.circle(vis_img, tuple(p1_coord.astype(int)), 5, (50, 255, 150), -1)
            cv2.line(vis_img, tuple(p0_coord.astype(int)), tuple(p1_coord.astype(int)), (50, 255, 150), 1)
        cv2.imshow("Propagation Debug", vis_img)
        cv2.waitKey(0)

    # --- Step 2: Determine Stagger Direction ---
    x_offsets = all_coords[grid[1, :], 0] - all_coords[grid[0, :], 0]
    avg_x_offset = np.mean(x_offsets)
    step_dist_x = np.mean(np.abs(row0_vectors[:, 0]))
    
    stagger_dir = 0
    if avg_x_offset > step_dist_x * 0.25: stagger_dir = -1  # Row 1 is shifted right, so P(r,c) depends on P(r-1, c-1)
    elif avg_x_offset < -step_dist_x * 0.25: stagger_dir = 1 # Row 1 is shifted left, so P(r,c) depends on P(r-1, c+1)
    print(f"  [ INFO ] Detected stagger direction: {stagger_dir}")
    if stagger_dir == 0:
        print("  [ propagate FAIL ] Could not determine clear stagger direction.")
        return None

    # --- Step 3: Propagate using the Parallelogram Rule ---
    if visualize: print("\n--- Stage 3: Propagating Full Grid (Asymmetric) ---")

    for r in range(2, num_rows):
        for c in range(num_cols):
            # Check for boundary conditions for the stagger index
            if not (0 <= c + stagger_dir < num_cols):
                print(f"  [ propagate FAIL ] Stagger calculation out of bounds at Row {r+1}, Col {c+1}.")
                return None
                
            p_ref1 = all_coords[grid[r-1, c]]
            p_ref2 = all_coords[grid[r-1, c + stagger_dir]]
            p_ref3 = all_coords[grid[r-2, c + stagger_dir]]
            
            # The vector from the previous step in the zig-zag column
            local_vector = p_ref2 - p_ref3
            pred_pt = p_ref1 + local_vector
            
            search_radius = np.linalg.norm(local_vector) * 0.7
            
            if visualize:
                print(f"  > Searching for Row {r+1}, Col {c+1}/{num_cols}...")
                step_vis_img = vis_img.copy()
                # Draw the parallelogram
                cv2.line(step_vis_img, tuple(p_ref3.astype(int)), tuple(p_ref2.astype(int)), (255,255,0), 2)
                cv2.line(step_vis_img, tuple(p_ref1.astype(int)), tuple(pred_pt.astype(int)), (255,255,0), 1)
                cv2.line(step_vis_img, tuple(p_ref2.astype(int)), tuple(pred_pt.astype(int)), (150,150,0), 1)
                cv2.drawMarker(step_vis_img, tuple(pred_pt.astype(int)), (255,0,255), cv2.MARKER_CROSS, 12, 1)
                cv2.circle(step_vis_img, tuple(pred_pt.astype(int)), int(search_radius), (255, 0, 255), 1)

            possible_indices = kdtree_all.query_ball_point(pred_pt, r=search_radius)
            best_match, min_dist = -1, float('inf')
            for p_idx in possible_indices:
                if available_mask[p_idx]:
                    d = np.linalg.norm(all_coords[p_idx] - pred_pt)
                    if d < min_dist:
                        min_dist, best_match = d, p_idx

            if best_match == -1:
                print(f"  [ propagate FAIL ] Could not find match for Row {r+1}, Col {c+1}.")
                if visualize:
                    cv2.circle(step_vis_img, tuple(pred_pt.astype(int)), int(search_radius), (0, 0, 255), 2)
                    cv2.imshow("Propagation Debug", step_vis_img)
                    cv2.waitKey(0)
                return None
            
            grid[r, c] = best_match
            available_mask[best_match] = False
            if visualize:
                match_coord = all_coords[best_match]
                cv2.circle(vis_img, tuple(match_coord.astype(int)), 5, (50, 255, 150), -1)
                cv2.line(vis_img, tuple(p_ref1.astype(int)), tuple(match_coord.astype(int)), (50, 255, 150), 1)
                cv2.imshow("Propagation Debug", vis_img)
                cv2.waitKey(50)
            
    return all_coords[grid].reshape(num_rows * num_cols, 1, 2)

# --- Main Controller Function (Modified) ---
def find_grid_adaptive(
    keypoints: List[cv2.KeyPoint],
    pattern_size: Tuple[int, int],
    visualize: bool = False
) -> Optional[np.ndarray]:
    """
    (Docstring is the same as before)
    """
    num_cols, num_rows = pattern_size
    min_points = max(20, num_cols * num_rows // 2)
    if len(keypoints) < min_points:
        return None

    all_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    # ... (DBSCAN pre-filtering logic is unchanged) ...
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=4).fit(all_coords)
    distances, _ = nn.kneighbors(all_coords)
    median_dist = np.median(distances[:, -1])
    eps = median_dist * 2.0
    db = DBSCAN(eps=eps, min_samples=5).fit(all_coords)
    labels = db.labels_
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0: return None
    main_cluster_label = unique_labels[np.argmax(counts)]
    pool_mask = (labels == main_cluster_label)
    pool_indices = np.where(pool_mask)[0]
    pool_coords = all_coords[pool_indices]

    # Step 1: Find the first, most confident row
    print("--- Stage 1: Finding Seed Row ---")
    # Note: _find_best_row_in_pool is unchanged from the previous version
    row0_indices = _find_best_row_in_pool(pool_coords, pool_indices, num_cols, visualize=visualize)
    
    if row0_indices is None:
        print("  [ FAIL ] Could not find a confident seed row.")
        return None
    print(f"  [ OK ] Found seed row with {len(row0_indices)} points.")

    # --- Setup for Visualization ---
    vis_img_base = None
    if visualize:
        # MODIFICATION: Create a named, resizable window before first use.
        cv2.namedWindow("Propagation Debug", cv2.WINDOW_NORMAL)
        
        vis_img_base = np.zeros((int(np.max(all_coords[:,1]))+50, int(np.max(all_coords[:,0]))+50, 3), dtype=np.uint8)
        # Draw all points faintly
        for pt in all_coords: cv2.circle(vis_img_base, tuple(pt.astype(int)), 3, (70,70,70), -1)
        # Draw the candidate pool points
        for pt in pool_coords: cv2.circle(vis_img_base, tuple(pt.astype(int)), 4, (150,100,0), -1)
        # Highlight the found seed row
        for idx in row0_indices:
            pt = all_coords[idx]
            cv2.circle(vis_img_base, tuple(pt.astype(int)), 6, (0,200,0), 2)
        # Connect seed row points
        for i in range(len(row0_indices) - 1):
            pt1 = all_coords[row0_indices[i]]
            pt2 = all_coords[row0_indices[i+1]]
            cv2.line(vis_img_base, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0,255,0), 1)
        cv2.imshow("Propagation Debug", vis_img_base)
        cv2.waitKey(0)

    # Step 2 & 3: Propagate grid from the seed row
    kdtree_all = KDTree(all_coords)
    available_mask = np.ones(len(all_coords), dtype=bool)
    
    # Try propagating from the row as found
    final_grid = _propagate_grid_asymmetric(row0_indices, all_coords, kdtree_all, available_mask.copy(), pattern_size, visualize, vis_img_base)
    
    # If that fails, try propagating from the reversed row
    if final_grid is None:
        print("\nPropagation failed, trying reversed seed row...")
        final_grid = _propagate_grid_asymmetric(row0_indices[::-1], all_coords, kdtree_all, available_mask.copy(), pattern_size, visualize, vis_img_base)
        
    # ... (Rest of the function for handling failure/success and final visualization is the same) ...
    # ...
    if final_grid is None:
        print("\nGrid propagation failed in both directions.")
        if visualize: cv2.destroyAllWindows()
        return None

    print("\nSuccessfully propagated the full grid.")
    # (final visualization logic)
    return final_grid



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
    search_radius = approx_spacing * 0.75
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
                is_found = not np.isnan(final_corners[:, 0, 0])[np.all(final_corners[:, 0, :] == all_kp_coords[idx], axis=1)]
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

# --- Configuration ---
DEFAULT_IMAGE_DIR = './calibration_images/'
DEFAULT_GRID_COLS = 4
DEFAULT_GRID_ROWS = 11
DEFAULT_SPACING_MM = 39.0
DEFAULT_OUTPUT_FILE = 'fisheye_calibration_data.npz'

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Fisheye Camera Calibration using Asymmetric Circle Grid.')
parser.add_argument('--dir', type=str, default=DEFAULT_IMAGE_DIR,
                    help=f'Directory containing calibration images (default: {DEFAULT_IMAGE_DIR})')
parser.add_argument('--cols', type=int, default=DEFAULT_GRID_COLS,
                    help=f'Number of circles horizontally in the grid (default: {DEFAULT_GRID_COLS})')
parser.add_argument('--rows', type=int, default=DEFAULT_GRID_ROWS,
                    help=f'Number of circles vertically in the grid (default: {DEFAULT_GRID_ROWS})')
parser.add_argument('--spacing', type=float, default=DEFAULT_SPACING_MM,
                    help=f'Distance between adjacent circle centers in real-world units (e.g., mm) (default: {DEFAULT_SPACING_MM})')
parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE,
                    help=f'Output file path for calibration data (.npz) (default: {DEFAULT_OUTPUT_FILE})')
parser.add_argument('--ext', type=str, nargs='+', default=['jpg', 'png', 'bmp', 'tif', 'jpeg'],
                    help='Image file extensions to process (default: jpg png bmp tif jpeg)')
parser.add_argument('--visualize_serpentine', action='store_true',
                    help='Visualize serpentine grid detection.')

args = parser.parse_args()

# --- Setup ---
image_dir = args.dir
grid_cols = args.cols
grid_rows = args.rows
spacing = args.spacing
output_file = args.output
image_extensions = args.ext

pattern_size = (grid_cols, grid_rows) # OpenCV format: (cols, rows)
min_images_for_calib = 10 # Minimum number of good views required

# --- Prepare Object Points ---
# Create the 3D coordinates of the grid points in real-world space (e.g., mm)
# Z=0 because the grid is planar.
# The order must match the order circle centers are detected by findCirclesGrid.
# THIS SECTION IS MODIFIED to match the generator script's logic.

print(f"\nGenerating object points for {pattern_size} grid with spacing={spacing} mm...")
objp = np.zeros((grid_cols * grid_rows, 3), np.float32)

# Use the real-world spacing value provided via the --spacing argument
real_world_spacing = spacing # Alias for clarity

for r in range(grid_rows):
    for c in range(grid_cols):
        idx = r * grid_cols + c

        # --- NEW objp Calculation ---
        # Matches the logic from generate_asymmetric_circles_grid.py
        # Origin (0,0,0) is the center of the circle at r=0, c=0.

        # Y coordinate increases linearly by row
        objp[idx, 1] = r * real_world_spacing

        # X coordinate increases linearly by column, with a half-spacing offset for odd rows
        row_offset = (r % 2) * (real_world_spacing / 2.0)
        objp[idx, 0] = c * real_world_spacing + row_offset

        # Z coordinate is always 0 for a planar target
        objp[idx, 2] = 0
        # --- End NEW objp Calculation ---

# draw
# Check if objp was actually populated
if objp.shape[0] > 0:
    # Find the bounds of the points to determine canvas size
    min_x = np.min(objp[:, 0])
    max_x = np.max(objp[:, 0])
    min_y = np.min(objp[:, 1])
    max_y = np.max(objp[:, 1])

    # Add a margin for visualization
    vis_margin = 50 # Pixels margin around the pattern
    point_radius = 5 # Radius of points to draw in visualization

    # Calculate canvas dimensions (using ceiling to ensure points fit)
    canvas_width = int(np.ceil(max_x - min_x)) + 2 * vis_margin
    canvas_height = int(np.ceil(max_y - min_y)) + 2 * vis_margin

    # Create a white canvas (BGR format for colored points)
    vis_image = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    print(f"  Visualization canvas size: {canvas_width}x{canvas_height}")
    print(f"  Object point range: X [{min_x:.2f}, {max_x:.2f}], Y [{min_y:.2f}, {max_y:.2f}]")

    # Draw each point onto the canvas
    for i in range(objp.shape[0]):
        # Get the original x, y coords
        x_orig = objp[i, 0]
        y_orig = objp[i, 1]

        # Translate coordinates so min_x, min_y maps near the top-left margin
        x_vis = int(round(x_orig - min_x + vis_margin))
        y_vis = int(round(y_orig - min_y + vis_margin))

        # Draw a blue circle at the translated coordinate
        cv2.circle(vis_image, (x_vis, y_vis), point_radius, (255, 0, 0), -1) # Blue circle, filled

        # Optional: Draw index number near the point
        # cv2.putText(vis_image, str(i), (x_vis + point_radius, y_vis + point_radius),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    # draw dimensions
    # Draw lines to indicate the grid spacing dimensions on the canvas

    # Calculate the start point (corresponding to the origin of objp)
    origin = (int(round(vis_margin)), int(round(vis_margin)))

    # Horizontal measurement: from the first point to where the next point should be in x-direction
    pt_horizontal = (int(round(real_world_spacing + vis_margin)), int(round(vis_margin)))
    cv2.line(vis_image, origin, pt_horizontal, (0, 0, 0), 2)
    # Label the horizontal spacing
    mid_horizontal = ((origin[0] + pt_horizontal[0]) // 2, origin[1] - 10)
    cv2.putText(vis_image, f"{real_world_spacing} mm", mid_horizontal, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # Vertical measurement: from the first point to where the next point should be in y-direction
    pt_vertical = (int(round(vis_margin)), int(round(real_world_spacing + vis_margin)))
    cv2.line(vis_image, origin, pt_vertical, (0, 0, 0), 2)
    # Label the vertical spacing
    mid_vertical = (origin[0] + 20, (origin[1] + pt_vertical[1]) // 2)
    cv2.putText(vis_image, f"{real_world_spacing} mm", mid_vertical, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # Display the visualization
    cv2.imshow("Object Point Layout (Calculated)", vis_image)
    print("-> Displaying calculated object point layout. Press any key in the window to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow("Object Point Layout (Calculated)") # Close only this window
else:
    print("  Warning: objp array is empty, cannot visualize layout.")


# Arrays to store object points and image points from all accepted images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

# --- Find Image Files ---
image_files = []
for ext in image_extensions:
    # Ensure case-insensitivity if needed, though glob might handle it on some OS
    search_path_lower = os.path.join(image_dir, f'*.{ext.lower()}')
    search_path_upper = os.path.join(image_dir, f'*.{ext.upper()}')
    found_files = glob.glob(search_path_lower) + glob.glob(search_path_upper)
    # Remove duplicates if both upper and lower case found
    found_files = list(set(found_files))
    print(f"Searching: {os.path.join(image_dir, f'*.{ext}')}, Found: {len(found_files)} files")
    image_files.extend(found_files)

image_files = sorted(list(set(image_files))) # Ensure unique and sorted list

if not image_files:
    print(f"Error: No images found in directory '{image_dir}' with extensions {image_extensions}")
    sys.exit(1)

print(f"\nFound {len(image_files)} potential calibration images.")

# --- Initialize Blob Detector ---
params = cv2.SimpleBlobDetector_Params()

# --- Initial Parameter Values (will be overridden by sliders in debug mode) ---
initial_min_area = 50     # Default starting value
initial_max_area = 10000  # Default starting value (adjust max range on slider if needed)
max_area_slider_limit = 20000 # Upper limit for the Max Area slider

# --- Tune these parameters as needed ---
params.filterByArea = True
params.minArea = initial_min_area
params.maxArea = initial_max_area

params.filterByCircularity = True
params.minCircularity = 0.5 # Lower for fisheye/perspective distortion

params.filterByConvexity = True
params.minConvexity = 0.80

params.filterByInertia = True
params.minInertiaRatio = 0.1 # Lower allows more elongation

# --- Create detector - its parameters might be updated in the loop if debugging ---
blob_detector = cv2.SimpleBlobDetector_create(params)

print("\nInitial Blob Detector parameters (can be tuned in debug mode):")
print(f"  Filter by Area: {params.filterByArea} ({params.minArea} - {params.maxArea})")
print(f"  Filter by Circularity: {params.filterByCircularity} (> {params.minCircularity})")
print(f"  Filter by Convexity: {params.filterByConvexity} (> {params.minConvexity})")
print(f"  Filter by Inertia: {params.filterByInertia} (> {params.minInertiaRatio})")


# --- Globals for Trackbars ---
current_min_area = initial_min_area
current_max_area = initial_max_area
manual_thresh_value = 127 # Default starting value for manual threshold slider

# --- Trackbar Callbacks (for Debug Mode) ---
def set_min_area(val):
    global current_min_area
    # Ensure min area is at least 1 and less than max area
    current_min_area = max(1, val)
    # Optional: Also ensure min < max here if needed, although the detector handles it
    # current_min_area = min(current_min_area, current_max_area - 1)

def set_max_area(val):
    global current_max_area
    # Ensure max area is greater than min area
    current_max_area = max(val, current_min_area + 1)

def set_thresh(val):
    global manual_thresh_value
    manual_thresh_value = val

# --- Process Images ---
img_shape = None # Store image shape (height, width)

print("\nProcessing images...")
print("Press 'y' to accept detection, 'n' to reject, 'q' to quit during final view.")

print("Debug View Controls:")
print("  'm': Switch Preprocessing (CLAHE -> None -> Manual -> Adaptive)")
print("  't': Toggle Threshold Type (Binary / Binary Inv) for Manual/Adaptive")
print("  ' ' (Space): Process this image with current settings & try findCirclesGrid")
print("  's': Skip this image")
print("  'q': Quit application")
print("  Use sliders in 'Controls' window to adjust thresholds and blob area.")
print("\nFinal Confirmation View Controls (after finding grid in Debug mode):")
print("  'y': Accept detection for calibration")
print("  'n': Reject detection")
print("  'q': Quit application")

# Create windows for debug mode
cv2.namedWindow('Debug View', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Debug View', 1200, 400) # Adjust size (W, H) - assuming 3 horizontal panels
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED) # Added flag for better layout
cv2.resizeWindow('Controls', 400, 200) # Adjust size

# Create Trackbars
cv2.createTrackbar('Manual Thresh', 'Controls', manual_thresh_value, 255, set_thresh)
cv2.createTrackbar('Min Area', 'Controls', current_min_area, 5000, set_min_area) # Adjust range 0-5000 as needed
cv2.createTrackbar('Max Area', 'Controls', current_max_area, max_area_slider_limit, set_max_area)

# Initialize debug state variables
debug_preprocess_mode = 0 # 0: None, 1: Manual, 2: Adaptive, 3: CLAHE
debug_thresh_type = cv2.THRESH_BINARY # Start with Binary
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

processed_count = 0
accepted_count = 0
processed_image_filenames = [] # Keep track of filenames corresponding to objpoints/imgpoints

for i, fname in enumerate(image_files):
    print(f"\n[{processed_count+1}/{len(image_files)}] Processing: {os.path.basename(fname)}")
    img_color = cv2.imread(fname)
    if img_color is None:
        print("  Warning: Could not read image.")
        continue

    # Check image shape consistency
    current_img_shape = img_color.shape[:2] # (height, width)
    if img_shape is None:
        img_shape = current_img_shape
        print(f"  Detected image size: {img_shape[1]}x{img_shape[0]} (WxH)")
    elif current_img_shape != img_shape:
         print(f"  Warning: Image size {current_img_shape} differs from first image {img_shape}. Skipping.")
         continue

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    processed_gray = gray # Default: use original gray image if not debugging or if mode is 'None'

    # --- DEBUG MODE: Interactive Preprocessing & Visualization ---
    while True: # Loop for interactive debugging adjustments
        vis_list = []

        # --- 1. Original Color Image (scaled) ---
        h_orig_color, w_orig_color = img_color.shape[:2]
        max_h_vis = 350 # Max height for visualization panels
        scale = max_h_vis / h_orig_color if h_orig_color > 0 else 1
        vis_width = int(w_orig_color * scale)
        vis_height = int(h_orig_color * scale)
        vis_dim = (vis_width, vis_height)

        vis_orig = cv2.resize(img_color, vis_dim, interpolation=cv2.INTER_AREA)
        cv2.putText(vis_orig, 'Original', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        vis_list.append(vis_orig)

        # --- 2. Apply Selected Preprocessing ---
        preprocess_desc = ""
        # Apply preprocessing to a temporary variable for visualization
        if debug_preprocess_mode == 2: # Manual Threshold
            _, processed_gray_dbg = cv2.threshold(gray, manual_thresh_value, 255, debug_thresh_type)
            preprocess_desc = f"Manual Th: {manual_thresh_value}, T:{'BIN' if debug_thresh_type==cv2.THRESH_BINARY else 'INV'}"
        elif debug_preprocess_mode == 3: # Adaptive Threshold
                # Block size must be odd and >= 3
                # Link trackbar crudely for block size adjustment (example)
                block_size = 11 + 2 * (manual_thresh_value // 32)
                block_size = max(3, block_size | 1) # Ensure odd >= 3
                C = 2 # Constant subtracted
                processed_gray_dbg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                debug_thresh_type, block_size, C)
                preprocess_desc = f"Adaptive G Th: B:{block_size}, C:{C}, T:{'BIN' if debug_thresh_type==cv2.THRESH_BINARY else 'INV'}"
        elif debug_preprocess_mode == 0: # CLAHE
            processed_gray_dbg = clahe.apply(gray)
            preprocess_desc = "CLAHE Contrast"
        elif debug_preprocess_mode == 1: # Mode 0: None (Use original grayscale)
            processed_gray_dbg = gray.copy()
            preprocess_desc = "Raw Grayscale"

        # Prepare processed image for visualization (convert to BGR)
        vis_proc = cv2.cvtColor(processed_gray_dbg, cv2.COLOR_GRAY2BGR)
        vis_proc = cv2.resize(vis_proc, vis_dim, interpolation=cv2.INTER_NEAREST) # Nearest to see pixels if needed
        cv2.putText(vis_proc, preprocess_desc, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        vis_list.append(vis_proc)

        # --- Update Blob Detector Parameters from Sliders ---
        # Ensure min < max from sliders
        if current_min_area >= current_max_area:
                current_max_area = current_min_area + 1
                # Optional: Update slider position visually (might flicker)
                # cv2.setTrackbarPos('Max Area', 'Controls', current_max_area)

        params.minArea = current_min_area
        params.maxArea = current_max_area
        # Re-create detector with updated area params
        blob_detector = cv2.SimpleBlobDetector_create(params)

        # --- 3. Detect Blobs on the full-resolution processed image ---
        keypoints = blob_detector.detect(processed_gray_dbg) # Detect on full-res DBG image

        # --- Create scaled keypoints specifically for visualization ---
        scaled_keypoints = []
        h_detect, w_detect = processed_gray_dbg.shape[:2] # Dimensions blobs were detected on
        h_vis_draw, w_vis_draw = vis_orig.shape[:2]      # Dimensions of the image to draw on

        if h_detect > 0 and w_detect > 0:
            scale_x = w_vis_draw / w_detect
            scale_y = h_vis_draw / h_detect
            vis_scale = (scale_x + scale_y) / 2.0 # Average scale for size

            if keypoints: # Check if keypoints list is not empty
                for kp in keypoints:
                    # Scale the coordinates
                    scaled_pt_x = kp.pt[0] * scale_x
                    scaled_pt_y = kp.pt[1] * scale_y
                    # Scale the size
                    scaled_size = kp.size * vis_scale
                    # Create a new KeyPoint object with scaled values
                    scaled_kp = cv2.KeyPoint(x=scaled_pt_x, y=scaled_pt_y, size=max(1, scaled_size), # Ensure size > 0
                                                angle=kp.angle, response=kp.response, octave=kp.octave,
                                                class_id=kp.class_id)
                    scaled_keypoints.append(scaled_kp)

        # --- Draw the SCALED keypoints onto the SCALED visualization image ---
        vis_blobs = vis_orig.copy() # Draw on a fresh copy of the scaled original
        vis_blobs = cv2.drawKeypoints(
            vis_blobs,          # The scaled image to draw on
            scaled_keypoints,   # The list of *scaled* keypoints
            np.array([]),
            (0, 0, 255),        # Red blobs
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS # Draws circles proportional to size
        )
        blob_text = f'Blobs: {len(keypoints)} (Area:{current_min_area}-{current_max_area})'
        cv2.putText(vis_blobs, blob_text, (10, vis_blobs.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        vis_list.append(vis_blobs)

        # --- Display Debug Views ---
        # Ensure all images have the same height before stacking
        final_h = max(img.shape[0] for img in vis_list)
        vis_list_resized = [cv2.resize(img, (int(img.shape[1] * final_h / img.shape[0]), final_h)) if img.shape[0] != final_h else img for img in vis_list]
        debug_vis = np.hstack(vis_list_resized)
        cv2.imshow('Debug View', debug_vis)
        key = cv2.waitKey(10) & 0xFF # Use waitKey(10) for responsiveness to sliders

        # --- Handle Debug Keystrokes ---
        if key == ord(' '): # Space: Proceed with findCirclesGrid using current settings
            # Set the processed_gray that will be used by findCirclesGrid
            processed_gray = processed_gray_dbg
            print(f"  DEBUG: Proceeding with Preproc: {preprocess_desc}, Area: {current_min_area}-{current_max_area}")
            break # Exit debug adjustment loop, proceed to findCirclesGrid
        elif key == ord('m'): # Switch preprocessing mode
            debug_preprocess_mode = (debug_preprocess_mode + 1) % 4 # Cycle through 0, 1, 2, 3
            print(f"  DEBUG: Switched Preprocessing Mode to {debug_preprocess_mode}")
        elif key == ord('t'): # Toggle threshold type
            debug_thresh_type = cv2.THRESH_BINARY if debug_thresh_type == cv2.THRESH_BINARY_INV else cv2.THRESH_BINARY
            print(f"  DEBUG: Switched Threshold Type to {'BINARY' if debug_thresh_type==cv2.THRESH_BINARY else 'BINARY_INV'}")
        elif key == ord('s'): # Skip image
            print("  Skipped (Debug).")
            processed_gray = None # Signal to skip findCirclesGrid
            break
        elif key == ord('q'): # Quit
            print("\nQuitting application.")
            cv2.destroyAllWindows()
            sys.exit(0)
            # Else: trackbar change or other key, loop again to update view

    # --- End of Debug adjustment loop ---
    if processed_gray is None: # Image was skipped in debug mode
        processed_count += 1
        continue # Go to next image

    # --- Find the circle grid centers ---
    # Use the 'processed_gray' which was determined above (either original gray or from debug step)
    # Use the 'blob_detector' which potentially has updated parameters from debug sliders

    # Consider adding CALIB_CB_CLUSTERING if needed
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING

    # Ensure the detector passed has the latest parameters if in debug mode
    # (It should, due to recreation in the debug loop)
    ret, corners = cv2.findCirclesGrid(
        processed_gray,      # Use the potentially preprocessed image
        pattern_size,
        flags=flags,
        blobDetector=blob_detector # Pass the explicitly created/updated detector
    )

    processed_count += 1 # Increment here, after attempt is made

    if not ret:

            # Inside the 'if not ret:' block, when key == ord('m')
        keypoints = blob_detector.detect(processed_gray) # Make sure keypoints are detected
        if keypoints:
            corners_manual = run_four_corner_local_vector_walk(img_color, keypoints, objp, pattern_size, args.visualize_serpentine)
            if corners_manual is not None:
                ret = True
                corners = corners_manual
                print("  Assisted matching successful!")
            else:
                print("  Assisted matching failed.")
                ret = False
        else:
            print("  Error: No blobs detected, cannot attempt assisted matching.")
            ret = False

    # --- Show Final Detection Result and Ask for Confirmation ---
    if ret:
        print(f"  Grid detected! ({pattern_size[0]}x{pattern_size[1]})")
        # corners are the detected circle centers (N, 1, 2)

        # Draw the corners on the image
        vis_img = img_color.copy()
        # corners should be float32 for drawChessboardCorners
        if corners is not None:
            corners = corners.astype(np.float32)
            cv2.drawChessboardCorners(vis_img, pattern_size, corners, ret) # Works for circles too

        # Display the image with detected corners for confirmation
        cv2.namedWindow('Detection Confirmation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection Confirmation', 800, 600) # Adjust size as needed
        cv2.imshow('Detection Confirmation', vis_img)

        while True:
            key = cv2.waitKey(0) & 0xFF # Wait indefinitely for user input
            if key == ord('y'):
                print("  Accepted.")
                objpoints.append(objp.reshape(-1, 1, 3))
                imgpoints.append(corners) # Append the detected corners
                accepted_count += 1
                processed_image_filenames.append(fname) # Store filename for accepted view
                break
            elif key == ord('n') or key == ord('s'): # Allow 's' for skip here too
                print("  Rejected.")
                break
            elif key == ord('q'):
                print("\nQuitting detection phase.")
                cv2.destroyAllWindows()
                sys.exit(0)
            else:
                print("  Invalid key. Press 'y' (accept), 'n' (reject), or 'q' (quit).")
        cv2.destroyWindow('Detection Confirmation') # Close confirmation window
    else:
        print("  Grid not detected. Skipping this image.")
        # Optionally: Draw keypoints on the image for debugging


# --- End of Image Processing Loop ---
cv2.destroyWindow('Debug View')
cv2.destroyWindow('Controls')
cv2.destroyAllWindows() # Close any other remaining OpenCV windows

# --- Perform Calibration ---
print(f"\nCollected {accepted_count} valid views out of {processed_count} processed images.")

if accepted_count < min_images_for_calib:
    print(f"Error: Insufficient number of valid views ({accepted_count}). Need at least {min_images_for_calib}.")
    if processed_count > 0 and accepted_count == 0:
        print("Possible issues:")
        print("- Blob detector parameters might be wrong (check min/max Area especially in debug mode).")
        print("- Preprocessing might be needed (try debug mode with thresholding/CLAHE).")
        print("- Grid parameters (--cols, --rows) might not match the physical grid.")
        print("- Lighting conditions might be poor (shadows, glare).")
        print("- Image quality might be low (blur, noise).")
        print("- Grid geometry might be too distorted in all views for internal checks.")
    sys.exit(1)

if img_shape is None:
    print("Error: Could not determine image shape (no images processed successfully?).")
    sys.exit(1)

print(f"\nRunning fisheye calibration with image size: {img_shape[::-1]} (width, height)...")

# Calibration flags
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
# Consider adding: cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT if center is known reliably
# Consider adding: cv2.fisheye.CALIB_FIX_K[1,2,3,4] if optimization is unstable

# NOTE: cv2.fisheye.calibrate requires image_size as (WIDTH, HEIGHT).
image_size_wh = (img_shape[1], img_shape[0])

# Termination criteria for the optimization process
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
import random # <-- Add this import at the top of your script

# --- (Keep all the code above the Perform Calibration section the same) ---

# --- Perform Calibration ---
print(f"\nCollected {accepted_count} valid views out of {processed_count} processed images.")

if accepted_count < min_images_for_calib:
    print(f"Error: Insufficient number of valid views ({accepted_count}). Need at least {min_images_for_calib}.")
    # ... (keep existing error message suggestions) ...
    sys.exit(1)

if img_shape is None:
    print("Error: Could not determine image shape (no images processed successfully?).")
    sys.exit(1)

print(f"\nRunning fisheye calibration with image size: {img_shape[::-1]} (width, height)...")

# Prepare for fisheye calibration
K_init = np.eye(3)
D_init = np.zeros((4, 1))
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
image_size_wh = (img_shape[1], img_shape[0])
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# --- Variables to hold the final successful calibration results ---
final_ret = None
final_K = None
final_D = None
final_rvecs = None
final_tvecs = None
final_objpoints = objpoints # Start with all accepted points
final_imgpoints = imgpoints
final_filenames = processed_image_filenames
final_accepted_count = accepted_count
calibration_succeeded = False # Flag to track success

# --- Configuration for Randomized Removal ---
max_random_attempts = 5 # How many times to try the random removal process
print(f"Will attempt randomized removal up to {max_random_attempts} times if initial calibration fails.")

try:
    print(f"\nAttempting initial calibration with {len(final_objpoints)} views...")
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        final_objpoints,
        final_imgpoints,
        image_size_wh,
        K_init,
        D_init,
        flags=calib_flags,
        criteria=criteria
    )
    # If successful on the first try:
    final_ret = ret
    final_K = K
    final_D = D
    final_rvecs = rvecs
    final_tvecs = tvecs
    calibration_succeeded = True
    print("Initial calibration attempt successful.")

except cv2.error as e:
    print(f"\n!!! Initial OpenCV Error during calibration: {e}")
    # --- START: Randomized Iterative Removal Logic ---
    if "CALIB_CHECK_COND" in str(e) and accepted_count >= min_images_for_calib:
        print(f"\nAttempting randomized iterative removal (up to {max_random_attempts} attempts)...")

        for attempt in range(max_random_attempts):
            print(f"\n--- Random Removal Attempt {attempt + 1}/{max_random_attempts} ---")

            # --- Reset state for this attempt ---
            # Work on copies so we don't modify the original full lists *across attempts*
            objpoints_copy = list(objpoints)
            imgpoints_copy = list(imgpoints)
            filenames_copy = list(processed_image_filenames)
            current_view_count = accepted_count
            removed_in_this_attempt = [] # Keep track of removals in this specific attempt

            # --- Inner loop: Randomly remove images until success or minimum count reached ---
            while current_view_count >= min_images_for_calib:
                try:
                    print(f"  Attempt {attempt+1}: Retrying calibration with {current_view_count} views...")
                    ret_iter, K_iter, D_iter, rvecs_iter, tvecs_iter = cv2.fisheye.calibrate(
                        objpoints_copy,
                        imgpoints_copy,
                        image_size_wh,
                        K_init, # Re-use initial K guess
                        D_init, # Re-use initial D guess
                        flags=calib_flags,
                        criteria=criteria
                    )
                    # SUCCESS!
                    print(f"  Calibration succeeded in attempt {attempt + 1} after removing {accepted_count - current_view_count} view(s).")
                    print(f"  Views removed in this successful attempt: {removed_in_this_attempt}")
                    final_ret = ret_iter
                    final_K = K_iter
                    final_D = D_iter
                    final_rvecs = rvecs_iter
                    final_tvecs = tvecs_iter
                    final_objpoints = objpoints_copy # Keep the successful subset
                    final_imgpoints = imgpoints_copy
                    final_filenames = filenames_copy
                    final_accepted_count = current_view_count # Update the count
                    calibration_succeeded = True
                    break # Exit the inner while loop (this attempt was successful)

                except cv2.error as inner_e:
                    # Still failing, remove a RANDOM image and try again
                    if current_view_count > min_images_for_calib:
                        # --- Random Selection ---
                        idx_to_remove = random.randrange(current_view_count)
                        removed_filename = os.path.basename(filenames_copy[idx_to_remove])
                        print(f"    Attempt {attempt+1}: Calibration failed again ({inner_e}). Randomly removing view {idx_to_remove + 1}/{current_view_count}: {removed_filename}")

                        # Remove from copies using the random index
                        objpoints_copy.pop(idx_to_remove)
                        imgpoints_copy.pop(idx_to_remove)
                        filenames_copy.pop(idx_to_remove)
                        removed_in_this_attempt.append(removed_filename) # Track removal for this attempt
                        current_view_count -= 1
                        # Loop continues to retry calibration
                    else:
                        # Reached minimum images and still failed *within this attempt*
                        print(f"    Attempt {attempt+1}: Calibration failed with minimum required views ({min_images_for_calib}). Stopping this removal path.")
                        break # Exit the inner while loop for this attempt

                except Exception as general_exception:
                     print(f"\n!!! An unexpected error occurred during attempt {attempt+1}'s calibration: {general_exception}")
                     import traceback
                     traceback.print_exc()
                     break # Stop this attempt's inner loop on unexpected errors

            # --- Check if this attempt succeeded ---
            if calibration_succeeded:
                break # Exit the outer for loop (attempts loop) because we found a working set

        # --- After all attempts ---
        if not calibration_succeeded:
            print(f"\nRandomized iterative removal failed after {max_random_attempts} attempts.")
            # Optional: Keep the original error message 'e' available if needed
            print("Original error likely persisted or occurred with minimum views across multiple random paths.")

    else:
        # Error was not CALIB_CHECK_COND or not enough images to start removing
        print("Cannot attempt iterative removal (error not CALIB_CHECK_COND or insufficient initial views).")
        print("Consider checking input points, view variation, or objp definition.")
    # --- END: Randomized Iterative Removal Logic ---

except Exception as e:
     print(f"\n!!! An unexpected GENERAL error occurred during initial calibration: {e}")
     import traceback
     traceback.print_exc()
     # calibration_succeeded remains False


# --- Results ---
# Now, base the rest of the script on the 'calibration_succeeded' flag
# and use the 'final_' variables which hold the results either from the
# initial attempt or the successful randomized iterative attempt.

if calibration_succeeded:
    print("\nCalibration successful!")
    print(f"  Used {final_accepted_count} views for final calibration.") # Report the number used
    print(f"  RMS reprojection error: {final_ret:.4f} pixels")
    print("\nCamera Matrix (K):")
    print(final_K)
    print("\nDistortion Coefficients (D) [k1, k2, k3, k4]:")
    print(final_D.flatten())

    # --- Save Results ---
    print(f"\nSaving calibration data to: {output_file}")
    try:
        # Save the results from the SUCCESSFUL calibration
        np.savez(output_file, K=final_K, D=final_D, img_shape=img_shape, rms=final_ret,
                 objpoints=np.array(final_objpoints, dtype=object), # Save points ACTUALLY used
                 imgpoints=np.array(final_imgpoints, dtype=object),
                 filenames=np.array(final_filenames)) # Save corresponding filenames
        print("Data saved.")
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")

    # --- Visualize Calibration Point Coverage ---
    # (Code remains the same, using final_imgpoints)
    print("\nVisualizing overall calibration point coverage...")
    if img_shape is not None and len(final_imgpoints) > 0:
        coverage_img = np.full((img_shape[0], img_shape[1], 3), 255, dtype=np.uint8)
        total_points_drawn = 0
        point_color = (0, 0, 255); point_radius = 2

        for view_corners in final_imgpoints: # Iterate through points from each USED view
            for corner in view_corners:
                center = tuple(corner[0].astype(int))
                cv2.circle(coverage_img, center, point_radius, point_color, -1)
                total_points_drawn += 1

        cv2.putText(coverage_img, f"Coverage: {total_points_drawn} points from {len(final_imgpoints)} views",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)

        cv2.namedWindow("Calibration Point Coverage", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration Point Coverage", 800, int(800 * img_shape[0] / img_shape[1]))
        cv2.imshow("Calibration Point Coverage", coverage_img)
        print(f"  Displayed coverage map with {total_points_drawn} points.")
        print("  Press any key in the 'Calibration Point Coverage' window to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow("Calibration Point Coverage")
    else:
        print("  Skipping coverage visualization (no image shape or no accepted points).")


    # --- Optional: Calculate Reprojection Error Manually ---
    # (Code remains the same, using final_ variables)
    print("\nCalculating reprojection errors per image (for the successful set)...")
    total_error = 0
    per_view_errors = []
    if len(final_objpoints) > 0 and len(final_rvecs) == len(final_objpoints) and len(final_tvecs) == len(final_objpoints):
        for i in range(len(final_objpoints)):
            try:
                imgpoints2, _ = cv2.fisheye.projectPoints(final_objpoints[i], final_rvecs[i], final_tvecs[i], final_K, final_D)
                error = cv2.norm(final_imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                per_view_errors.append(error)
                total_error += error
                # Optional: Print per-view error using final_filenames
                # print(f"  Image {i+1} ({os.path.basename(final_filenames[i])}): {error:.4f} pixels")
            except cv2.error as proj_err:
                print(f"  Warning: Could not project points for image {i} ({os.path.basename(final_filenames[i])}). Error: {proj_err}")
                per_view_errors.append(np.inf)

        mean_error = total_error / len(final_objpoints) if len(final_objpoints) > 0 else 0
        print(f"\nAverage reprojection error (calculated manually): {mean_error:.4f} pixels")
    else:
         print("\nSkipping manual reprojection error calculation (missing points or extrinsics).")


    # --- Visualize Undistortion ---
    # (Code remains the same, using final_ variables)
    print("\nVisualizing undistortion on the first accepted sample image...")
    if final_accepted_count > 0 and final_filenames:
        first_accepted_img_path = final_filenames[0] # Use the first from the FINAL successful set
        img_distorted = cv2.imread(first_accepted_img_path)

        if img_distorted is not None:
            h, w = img_distorted.shape[:2]
            balance = 0.0
            # Use final K and D for undistortion
            Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(final_K, final_D, image_size_wh, np.eye(3), balance=balance)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(final_K, final_D, np.eye(3), Knew, image_size_wh, cv2.CV_16SC2)
            img_undistorted = cv2.remap(img_distorted, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            vis_compare = np.hstack((img_distorted, img_undistorted))
            cv2.namedWindow('Distorted vs Undistorted', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Distorted vs Undistorted', 1200, 600)
            cv2.imshow('Distorted vs Undistorted', vis_compare)
            print(f"Displaying undistortion result for {os.path.basename(first_accepted_img_path)}. Press any key to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Warning: Could not reload sample image {first_accepted_img_path} for visualization.")
    else:
         print("No valid images were accepted, cannot visualize undistortion.")


else: # calibration_succeeded is False
    print("\nCalibration failed. The optimization could not converge, even after attempting randomized removal.")
    print("Possible reasons include:")
    print("- Persistently poor quality detections (high initial reprojection error).")
    print("- Insufficient number of *good* views or lack of variation even after removal.")
    print("- Incorrect grid parameters (--cols, --rows, --spacing).")
    print("- Severe numerical instability not resolved by removing views.")

print("\nCalibration process finished.")