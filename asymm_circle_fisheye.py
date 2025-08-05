import cv2
import numpy as np
import os
import glob
import argparse
import sys
from scipy.spatial import distance, KDTree # For efficient nearest neighbor searches
import math # For checking NaN
from typing import List, Dict, Set, Tuple, Optional
import collections

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
        auto_grid = find_grid_by_hexagon_symmetry(
            keypoints=detected_keypoints,
            pattern_size=(cols, rows),
            visualize=True
        )

        if auto_grid is not None:
            print("  Found an automatic grid, but you can still select corners manually.")
            print("  Press 'y' to skip manual selection and use the automatic grid.")

            tmp_img = image_to_select_on.copy()
            cv2.drawChessboardCorners(tmp_img, pattern_size, auto_grid, True)

            cv2.imshow("Automatic Grid", tmp_img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                print("  Skipping manual corner selection, using automatic grid.")
                return auto_grid.reshape(-1, 1, 2).astype(np.float32)
        else:
            print("  No automatic grid found, please select corners manually.")

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
    
def inspect_hexagon_scores_interactive(
    all_coords: np.ndarray,
    scores: np.ndarray,
    best_seed_idx: int,
    kdtree: KDTree,
    max_window_dim: int = 1200
):
    """
    Creates an interactive window to inspect the hexagon symmetry of any point.

    - Displays a heatmap of scores (Blue=Good, Red=Bad).
    - Left-click a point to draw its 6-neighbor hexagon and see its score components.
    - Right-click to clear the inspection drawing.
    - Press 'q' to close the window and proceed.
    """
    print("\n--- Interactive Score Inspector ---")
    print("  - Left-click a point to inspect its neighborhood symmetry.")
    print("  - Right-click to clear.")
    print("  - Press 'q' to close and continue.")

    # --- State for Interaction ---
    base_img = None
    display_img = None
    inspected_idx = -1  # Original index of the point being inspected

    # --- Scaling Logic ---
    max_x = np.max(all_coords[:, 0]); max_y = np.max(all_coords[:, 1])
    scale = min(1.0, max_window_dim / max(max_x, max_y))

    def _redraw():
        """Redraws the visualization, including any inspection details."""
        nonlocal display_img
        display_img = base_img.copy()

        if inspected_idx != -1:
            # --- Draw the hexagon and scores for the inspected point ---
            pt_coord = all_coords[inspected_idx]
            
            # Find 6 neighbors
            try:
                distances, indices = kdtree.query(pt_coord, k=7)
                if len(indices) < 7: return
            except Exception:
                return

            neighbor_indices = indices[1:]
            neighbor_coords = all_coords[neighbor_indices]

            # Draw lines from center to neighbors
            for n_coord in neighbor_coords:
                cv2.line(display_img, tuple((pt_coord * scale).astype(int)), tuple((n_coord * scale).astype(int)), (255, 255, 0), 1)

            # Draw the hexagon outline by connecting neighbors in angular order
            vectors = neighbor_coords - pt_coord
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            sorted_neighbor_coords = neighbor_coords[np.argsort(angles)]
            for i in range(6):
                p1 = sorted_neighbor_coords[i]
                p2 = sorted_neighbor_coords[(i + 1) % 6]
                cv2.line(display_img, tuple((p1 * scale).astype(int)), tuple((p2 * scale).astype(int)), (0, 255, 255), 2)

            # Recalculate scores to display them
            score_dist, score_angle, score_centroid = _calculate_hexagon_score(inspected_idx, all_coords, kdtree, return_components=True)
            
            # Display text with scores
            text_y = int(pt_coord[1] * scale) + 20
            cv2.putText(display_img, f"Pt {inspected_idx}", (int(pt_coord[0] * scale), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(display_img, f" D_std: {score_dist:.3f}", (int(pt_coord[0] * scale), text_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(display_img, f" A_std: {score_angle:.3f}", (int(pt_coord[0] * scale), text_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(display_img, f" C_off: {score_centroid:.3f}", (int(pt_coord[0] * scale), text_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    def on_mouse(event, x, y, flags, param):
        """Handles user clicks to select points for inspection."""
        nonlocal inspected_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            unscaled_click = np.array([x, y]) / scale
            distances = np.linalg.norm(all_coords - unscaled_click, axis=1)
            closest_idx = np.argmin(distances)
            
            # Check if the clicked point can be scored
            if np.isfinite(scores[closest_idx]):
                inspected_idx = closest_idx
                _redraw()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            inspected_idx = -1
            _redraw()

    # --- Main Visualization Setup ---
    vis_w = int((max_x + 50) * scale); vis_h = int((max_y + 50) * scale)
    base_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)

    # Draw heatmap of scores
    valid_scores = scores[np.isfinite(scores)]
    min_s, max_s = np.min(valid_scores), np.max(valid_scores)
    for i, pt in enumerate(all_coords):
        score = scores[i]
        pt_scaled = tuple((pt * scale).astype(int))
        if np.isfinite(score):
            norm_score = (score - min_s) / (max_s - min_s + 1e-9)
            color_val = int(255 * (1 - norm_score))
            color = cv2.applyColorMap(np.uint8([[color_val]]), cv2.COLORMAP_JET)[0][0].tolist()
            cv2.circle(base_img, pt_scaled, int(5*scale), color, -1)

    # Highlight the automatically chosen seed point
    seed_coord_scaled = tuple((all_coords[best_seed_idx] * scale).astype(int))
    cv2.circle(base_img, seed_coord_scaled, int(12*scale), (255, 255, 255), 2)
    
    _redraw() # Initial draw

    WIN_NAME = "Interactive Score Inspector"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, vis_w, vis_h)
    cv2.setMouseCallback(WIN_NAME, on_mouse)
    
    while True:
        cv2.imshow(WIN_NAME, display_img)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    
def _calculate_hexagon_score(
    point_idx: int,
    all_coords: np.ndarray,
    kdtree: KDTree,
    weights: Tuple[float, float, float] = (1.0, 0.5, 1.5),
    return_components: bool = False
) -> float:
    # ... (Calculation logic is the same as before) ...
    try:
        distances, indices = kdtree.query(all_coords[point_idx], k=7)
    except Exception:
        return (float('inf'), float('inf'), float('inf')) if return_components else float('inf')

    if len(indices) < 7:
        return (float('inf'), float('inf'), float('inf')) if return_components else float('inf')

    center_coord = all_coords[point_idx]
    neighbor_indices = indices[1:]
    neighbor_coords = all_coords[neighbor_indices]
    neighbor_distances = distances[1:]
    mean_dist = np.mean(neighbor_distances)
    if mean_dist < 1e-6: return (float('inf'), float('inf'), float('inf')) if return_components else float('inf')
    
    score_dist = np.std(neighbor_distances) / mean_dist
    vectors = neighbor_coords - center_coord
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_angles = np.sort(angles)
    angle_diffs = np.diff(np.append(sorted_angles, sorted_angles[0] + 2 * np.pi))
    score_angle = np.std(angle_diffs)
    centroid = np.mean(neighbor_coords, axis=0)
    score_centroid = np.linalg.norm(center_coord - centroid) / mean_dist
    
    if return_components:
        return score_dist, score_angle, score_centroid

    w_dist, w_angle, w_centroid = weights
    final_score = (w_dist * score_dist + w_angle * score_angle + w_centroid * score_centroid)
    return final_score

# --- Helper 2: Propagate the Grid from a Seed Point ---
def _propagate_from_seed_wavefront(
    seed_idx: int,
    all_coords: np.ndarray,
    kdtree: KDTree,
    pattern_size: Tuple[int, int],
    visualize: bool = False,
    max_window_dim: int = 1200
) -> Optional[Dict[Tuple[int, int], int]]:
    """
    Grows the grid outwards from a seed point, with detailed step-by-step visualization.
    """
    # ... (Initial setup logic: find neighbors, sort, get v_col, v_row is the same) ...
    _, neighbor_indices = kdtree.query(all_coords[seed_idx], k=7)
    neighbor_indices = neighbor_indices[1:]
    seed_coord = all_coords[seed_idx]
    vectors = all_coords[neighbor_indices] - seed_coord
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    sorted_neighbor_indices = neighbor_indices[np.argsort(angles)]

    grid_map: Dict[Tuple[int, int], int] = {(0, 0): seed_idx}
    queue = collections.deque([((0, 0), seed_idx)])
    visited_original_indices = {seed_idx}

    seed_v_row = all_coords[sorted_neighbor_indices[3]] - seed_coord # (0, 1)
    seed_v_row_m = all_coords[sorted_neighbor_indices[0]] - seed_coord # (0, -1)
    seed_v_col = all_coords[sorted_neighbor_indices[1]] - seed_coord # (1, 0)
    seed_v_col_m = all_coords[sorted_neighbor_indices[4]] - seed_coord # (-1, 0)
    seed_v_diag = all_coords[sorted_neighbor_indices[2]] - seed_coord # (1, 1)
    seed_v_diag_m = all_coords[sorted_neighbor_indices[5]] - seed_coord # (-1, -1)

    local_direction_vectors = {seed_idx: np.array([seed_v_row, seed_v_diag, seed_v_col, seed_v_row_m, seed_v_diag_m, seed_v_col_m])} # direction vectors for the seed point, last item is a boolean indicating if the vectors are already updated or a propagated guess
    direction_vectors_index_increments = [(0, 1), (1, 1), (1, 0), (0, -1), (-1, -1), (-1, 0)] # (r,c) increments for the 6 hexagonal directions

    # definition if direction vector is not yet known
    unknown_direction_vector = np.array([np.nan, np.nan], dtype=np.float32)
    unknown_direction_vectors = np.array([unknown_direction_vector] * 6, dtype=np.float32) # Placeholder for unknown vectors

    # hexagon opposite side mapping
    # This is used to find the opposite side of the hexagon for the parallelogram
    opposite_side_idx_map = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}

    # --- Visualization Setup ---
    vis_img = None
    if visualize:
        print("\n--- Stage 2: Propagating Grid (Animated) ---")
        print("  - Controls: Press 'p' to pause/play, 'q' to skip.")
        
        vis_img = np.zeros((int(np.max(all_coords[:,1]))+50, int(np.max(all_coords[:,0]))+50, 3), dtype=np.uint8)
        for pt in all_coords: cv2.circle(vis_img, tuple(pt.astype(int)), 3, (70,70,70), -1)
        cv2.circle(vis_img, tuple(seed_coord.astype(int)), 6, (0,255,0), -1)

        WIN_NAME = "Propagation Animator"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        h, w, _ = vis_img.shape
        aspect_ratio = w / h
        win_w = min(w, max_window_dim); win_h = int(win_w / aspect_ratio)
        cv2.resizeWindow(WIN_NAME, win_w, win_h)

        # draw the row and column vectors
        cv2.arrowedLine(vis_img, tuple(seed_coord.astype(int)), tuple((seed_coord + seed_v_row).astype(int)), (255, 0, 0), 2, tipLength=0.2)
        cv2.arrowedLine(vis_img, tuple(seed_coord.astype(int)), tuple((seed_coord + seed_v_col).astype(int)), (0, 0, 255), 2, tipLength=0.2)

        #label them
        cv2.putText(vis_img, "Row Vector", tuple((seed_coord + seed_v_row * 0.5).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis_img, "Col Vector", tuple((seed_coord + seed_v_col * 0.5).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, vis_img)
        cv2.waitKey(0)
        is_paused = True

    # --- Main Propagation Loop ---
    while queue:
        (r, c), p_idx = queue.popleft()
        p_coord = all_coords[p_idx]

        if visualize:
            # Highlight the point currently being processed
            cv2.circle(vis_img, tuple(p_coord.astype(int)), 8, (255,100,0), 2)

        # local_direction_vectors[p_idx]

        for i in range(6):
            prediction_grid_pos = (r + direction_vectors_index_increments[i][0], c + direction_vectors_index_increments[i][1])
            prediction_vector = local_direction_vectors[p_idx][i]

            if np.isnan(prediction_vector).any():
                # see if the opposite side is already known
                opposite_side_idx = opposite_side_idx_map[i]
                if not np.isnan(local_direction_vectors[p_idx][opposite_side_idx]).any():
                    prediction_vector = -local_direction_vectors[p_idx][opposite_side_idx].copy()
                else:
                    continue

            pred_coord = p_coord + prediction_vector

            search_radius = np.linalg.norm(prediction_vector) * 0.3 # Adaptive radius

            if prediction_grid_pos in grid_map:
                continue

            if visualize:
                step_vis_img = vis_img.copy()
                cv2.circle(step_vis_img, tuple(p_coord.astype(int)), 10, (255,255,255), 4)
                cv2.drawMarker(step_vis_img, tuple(pred_coord.astype(int)), (0,255,255), cv2.MARKER_CROSS, 12, 1)
                cv2.circle(step_vis_img, tuple(pred_coord.astype(int)), int(search_radius), (0, 200, 200), 1)

                #draw all queued points in pink
                for (qr, qc), q_idx in queue:
                    q_coord = all_coords[q_idx]
                    cv2.circle(step_vis_img, tuple(q_coord.astype(int)), 10, (255,0,255), 3)

                # draw all known local direction vectors for current point
                for j in range(6):
                    if not np.isnan(local_direction_vectors[p_idx][j]).any():
                        target_coord = p_coord + local_direction_vectors[p_idx][j]
                        cv2.arrowedLine(step_vis_img, tuple(p_coord.astype(int)), tuple(target_coord.astype(int)), (255,0,255), 1, tipLength=0.2)
            
            possible_indices = kdtree.query_ball_point(pred_coord, r=search_radius)
            best_match, min_dist = -1, float('inf')
            
            for next_idx in possible_indices:
                if next_idx not in visited_original_indices:
                    dist = np.linalg.norm(all_coords[next_idx] - pred_coord)
                    if dist < min_dist:
                        min_dist, best_match = dist, next_idx
            
            if best_match != -1:
                grid_map[prediction_grid_pos] = best_match
                visited_original_indices.add(best_match)
                queue.append((prediction_grid_pos, best_match))

                if visualize:
                    cv2.circle(vis_img, tuple(all_coords[best_match].astype(int)), 5, (0,255,0), -1)

                # give an initial guess for the new point's vectors
                local_direction_vectors[best_match] = unknown_direction_vectors.copy()

                for j in range(6):
                    grid_pos_query = (prediction_grid_pos[0] + direction_vectors_index_increments[j][0], prediction_grid_pos[1] + direction_vectors_index_increments[j][1])

                    if grid_pos_query in grid_map:
                        local_direction_vectors[best_match][j] = all_coords[grid_map[grid_pos_query]] - all_coords[best_match]
                        local_direction_vectors[grid_map[grid_pos_query]][opposite_side_idx_map[j]] = -local_direction_vectors[best_match][j].copy()

                        if visualize:
                            cv2.line(vis_img, tuple(all_coords[best_match].astype(int)), tuple(all_coords[grid_map[grid_pos_query]].astype(int)), (255,255,255), 1)

            if visualize:
                cv2.imshow(WIN_NAME, step_vis_img)
                delay = 1 if not is_paused else 0
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q'): # Quit visualization
                    visualize = False 
                elif key == ord('p'): # Pause/play
                    is_paused = not is_paused
    
    if visualize:
        print("  [ INFO ] Propagation finished.")
        print ("  [ INFO ] Found grid map with {} points out of expected {}.".format(len(grid_map), pattern_size[0] * pattern_size[1]))

        #draw the final grid map with grid coordinates
        for (r, c), idx in grid_map.items():
            pt = all_coords[idx]
            cv2.circle(vis_img, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(vis_img, f"({r},{c})", tuple(pt.astype(int) + np.array([8, 8], dtype=int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(WIN_NAME, vis_img)   
        cv2.waitKey(0)

        #grid_points = set(grid_map.keys())

        #_draw_hex_grid(grid_points)

    return grid_map

def _generate_target_hex_grid(
    width: int,
    height: int,
) -> Set[Tuple[int, int]]:
    """
    Generates a hexagonal grid of r and c coordinates within the specified width and height.
    """
    hex_grid = set()

    for r in range(height):
        for c in range(width):
            # Calculate the offset for odd rows
            offset = ((r + 1) // 2)
            hex_grid.add((r, int(c + offset)))

    return hex_grid

def _draw_hex_grid(
    grid_points: Set[Tuple[int, int]],
) -> Optional[np.ndarray]:
    # make a second visulisation
    max_row = max(r for r, c in grid_points)
    max_col = max(c for r, c in grid_points)
    min_row = min(r for r, c in grid_points)
    min_col = min(c for r, c in grid_points)

    range_rows = max_row - min_row + 1
    range_cols = max_col - min_col + 1

    # create a grid image with cols and rows labeled
    grid_img = np.zeros((range_rows * 10, range_cols * 10, 3), dtype=np.uint8)

    #fill the grid with white
    grid_img.fill(255)
    
    # draw the grid lines
    for r in range(range_rows + 1):
        cv2.line(grid_img, (0, r * 10), (range_cols * 10, r * 10), (200, 200, 200), 1)

    for c in range(range_cols + 1):
        cv2.line(grid_img, (c * 10, 0), (c * 10, range_rows * 10), (200, 200, 200), 1)

    # loop through grid_map and fill the found points with black
    for r, c in grid_points:
        pt_row = r - min_row
        pt_col = c - min_col
        
        # draw a filled box at the grid position
        cv2.rectangle(grid_img, (pt_col * 10 + 1, pt_row * 10 + 1),
                    (pt_col * 10 + 9, pt_row * 10 + 9), (0, 0, 0), -1)

    # draw the row and column numbers
    for r in range(range_rows):
        cv2.putText(grid_img, str(r + min_row), (5, r * 10 + 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (120, 120, 120), 1, cv2.LINE_AA)
    for c in range(range_cols):
        cv2.putText(grid_img, str(c + min_col), (c * 10 + 10, 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (120, 120, 120), 1, cv2.LINE_AA)
        
    cv2.imshow("Grid Map", grid_img)
    cv2.waitKey(0)

def rotate_hex_grid(
    grid_points: Set[Tuple[int, int]],
    steps: int
) -> Set[Tuple[int, int]]:
    """
    Rotates a set of hexagonal grid points by a multiple of 60 degrees.

    This function uses an axial coordinate system for input/output and temporarily
    converts to a cube coordinate system for rotation, which simplifies the math.

    Args:
        grid_points: A set of tuples, where each tuple is a (q, r) axial coordinate.
                     In our case, this will be the (r, c) keys from your grid_map.
        steps: The number of 60-degree clockwise rotations to perform.
               For example, steps=1 is a 60-degree rotation, steps=3 is 180 degrees.

    Returns:
        A new set of tuples representing the rotated (q, r) coordinates.
    """
    rotated_points = set()
    
    # Normalize steps to be within 0-5 range
    steps = steps % 6

    for q, r in grid_points:
        # 1. Convert axial (q, r) to cube (x, y, z) coordinates
        # In a cube system for a hex grid, x + y + z always equals 0.
        x = q
        z = r
        y = -x - z

        # 2. Perform rotation by shuffling cube coordinates
        # A 60-degree clockwise rotation is equivalent to shifting the cube coordinates.
        for _ in range(steps):
            x, y, z = -z, -x, -y
        
        # 3. Convert back from cube to axial coordinates for the output
        new_q = x
        new_r = z
        rotated_points.add((new_q, new_r))
        
    return rotated_points

def _translate_hex_grid(
    grid_points: Set[Tuple[int, int]],
    translation: Tuple[int, int]
) -> Set[Tuple[int, int]]:
    """
    Translates a set of hexagonal grid points by a given (r, c) offset.

    Args:
        grid_points: A set of tuples, where each tuple is a (r, c) coordinate.
        translation: A tuple (dr, dc) representing the translation in row and column.

    Returns:
        A new set of tuples representing the translated (r, c) coordinates.
    """
    translated_points = set()
    dr, dc = translation

    for r, c in grid_points:
        translated_points.add((r + dr, c + dc))

    return translated_points

def _is_hex_grid_covered(
    grid_map: Set[Tuple[int, int]],
    ideal_grid: Set[Tuple[int, int]]
) -> Tuple[int, int, float]:
    """
    Calculates the coverage of the grid_map against the ideal hexagonal grid.

    Args:
        grid_map: A dictionary mapping (r, c) coordinates to original indices.
        ideal_grid: A set of (r, c) coordinates representing the ideal hexagonal grid.

    Returns:
        A tuple containing:
        - wether all points in the ideal grid are covered by the grid_map
    """
    if ideal_grid.issubset(grid_map):
        return True
    return False

def _match_and_finalize_grid(
    grid_map: Dict[Tuple[int, int], int],
    all_coords: np.ndarray,
    pattern_size: Tuple[int, int],
    visualize: bool = False,
    max_window_dim: int = 1200
) -> Optional[np.ndarray]:
    """
    Finds the best affine transformation to map the arbitrary (r,c) coordinates
    from the grid_map to a final, ordered (row, col) system.
    """
    print("\n--- Stage 3: Matching Grid with Affine Transform ---")
    num_cols, num_rows = pattern_size

    target_hex_grid = _generate_target_hex_grid(num_cols, num_rows)

    hex_grid = set(grid_map.keys())

    # Iterate through every point as a potential anchor for the transformation
    r_final, c_final = -1, -1
    rotation_steps_final = -1

    for r_start, c_start in hex_grid:
        # move hex grid so that the current point is at (0, 0)
        temp_grid = _translate_hex_grid(
            hex_grid,
            (-r_start, -c_start)
        )
        
        for rotation_steps in range(6):
            rotated_grid = rotate_hex_grid(temp_grid, rotation_steps)

            # Calculate the coverage of the rotated grid against the ideal hex grid
            if _is_hex_grid_covered(rotated_grid, target_hex_grid):
                # If the rotated grid covers the ideal grid, we can calculate the affine transform
                r_final, c_final = r_start, c_start
                rotation_steps_final = rotation_steps
                break

    # build the final grid with the found coordinates
    if r_final == -1 or c_final == -1:
        print("  [ FAIL ] No valid affine transformation found that covers the ideal grid.")
        if visualize: cv2.destroyAllWindows()
        return None
    
    print(f"  [ OK ] Found affine transformation with r={r_final}, c={c_final}, rotation={rotation_steps_final} steps.")
    
    final_grid = []

    # transform grid map with the found affine transformation
    new_grid_map = {}
    for (r, c) in grid_map.keys():
        idx = grid_map[(r, c)]
        # Translate the grid point to the new origin
        translated_r = r - r_final
        translated_c = c - c_final

        x = translated_r
        z = translated_c
        y = -x - z

        # 2. Perform rotation by shuffling cube coordinates
        # A 60-degree clockwise rotation is equivalent to shifting the cube coordinates.
        for _ in range(rotation_steps_final):
            x, y, z = -z, -x, -y
        
        # 3. Convert back from cube to axial coordinates for the output
        new_r = x
        new_c = z
        new_grid_map[(new_r, new_c)] = idx

    for i in range(num_rows):
        for j in range(num_cols):
            r = num_rows - 1 - i  # Reverse the row order
            offset = ((r + 1) // 2)
            c = j + offset

            if (r, c) in new_grid_map:
                idx = new_grid_map[(r, c)]
                final_grid.append(all_coords[idx])
            else:
                # If a point is missing, fill with NaN
                print(f"  [ WARN ] Missing point at ({r}, {c}).")
                return None
            
    final_grid = np.array(final_grid, dtype=np.float32)

    return final_grid

def find_grid_by_hexagon_symmetry(
    keypoints: List[cv2.KeyPoint],
    pattern_size: Tuple[int, int],
    visualize: bool = False
) -> Optional[np.ndarray]:
    """
    (Docstring updated to mention wavefront propagation)
    """
    # ... (Scoring and seed finding is exactly the same as before) ...
    num_cols, num_rows = pattern_size
    if len(keypoints) < num_cols * num_rows * 0.5: return None
    all_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    kdtree = KDTree(all_coords)

    print("--- Stage 1: Scoring all points for neighborhood symmetry ---")
    scores = np.array([_calculate_hexagon_score(i, all_coords, kdtree) for i in range(len(all_coords))])
    
    valid_scores = scores[np.isfinite(scores)]
    if len(valid_scores) == 0: return None

    best_seed_idx = np.argmin(scores)
    min_score = scores[best_seed_idx]
    
    if visualize:
        # The interactive inspector is still very useful for checking the seed
        inspect_hexagon_scores_interactive(all_coords, scores, best_seed_idx, kdtree)

    score_threshold = 2.0 
    if min_score > score_threshold:
        print(f"  [ FAIL ] Best score ({min_score:.2f}) is above threshold.")
        return None
    
    print(f"  [ OK ] Found seed point {best_seed_idx} with score {min_score:.3f}.")

    # --- Step 3: Propagate using the new WAVEFRONT method ---
    print("\n--- Stage 2: Propagating grid from the seed point using wavefront method ---")
    grid_map = _propagate_from_seed_wavefront(best_seed_idx, all_coords, kdtree, pattern_size, visualize)

    if grid_map is None:
        print("  [ FAIL ] Grid propagation from seed point failed.")
        if visualize: cv2.destroyAllWindows()
        return None

    # --- NEW: Call the matching and finalization function ---
    final_grid = _match_and_finalize_grid(grid_map, all_coords, pattern_size, visualize)
    
    if final_grid is None:
        print("  [ FAIL ] Could not match the found points to the ideal grid pattern.")
        if visualize: cv2.destroyAllWindows()
        return None

    print("\nSuccessfully found and ordered the final grid.")
    if visualize: cv2.destroyAllWindows()

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