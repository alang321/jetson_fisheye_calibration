import cv2
import numpy as np
from scipy.spatial import KDTree # For efficient nearest neighbor searches
from typing import List, Dict, Set, Tuple, Optional
import collections
from typing import Optional, List, Tuple, Set

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

def _propagate_from_seed_wavefront(
    img: np.ndarray,
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

    seed_v_row = all_coords[sorted_neighbor_indices[3]] - seed_coord # (1, 0)
    seed_v_row_m = all_coords[sorted_neighbor_indices[0]] - seed_coord # (-1, 0)
    seed_v_col = all_coords[sorted_neighbor_indices[2]] - seed_coord # (0, 1)
    seed_v_col_m = all_coords[sorted_neighbor_indices[5]] - seed_coord # (0, -1)
    seed_v_diag = all_coords[sorted_neighbor_indices[1]] - seed_coord # (-1, 1)
    seed_v_diag_m = all_coords[sorted_neighbor_indices[4]] - seed_coord # (1, -1)

    local_direction_vectors = {seed_idx: np.array([seed_v_row, seed_v_diag, seed_v_col, seed_v_row_m, seed_v_diag_m, seed_v_col_m])} # direction vectors for the seed point, last item is a boolean indicating if the vectors are already updated or a propagated guess
    direction_vectors_index_increments = [(1, 0), (-1, 1), (0, 1), (-1, 0), (1, -1), (0, -1)] # (r,c) increments for the 6 hexagonal directions

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
        
        vis_img = img.copy()
        for pt in all_coords: cv2.circle(vis_img, tuple(pt.astype(int)), 3, (0,0,255), -1)
        cv2.circle(vis_img, tuple(seed_coord.astype(int)), 6, (0,255,0), -1)

        WIN_NAME = "Propagation Animator"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        h, w, _ = vis_img.shape
        aspect_ratio = w / h
        win_w = min(w, max_window_dim); win_h = int(win_w / aspect_ratio)
        cv2.resizeWindow(WIN_NAME, win_w, win_h)

        # draw the row and column vectors
        cv2.arrowedLine(vis_img, tuple(seed_coord.astype(int)), tuple((seed_coord + seed_v_row).astype(int)), (255, 0, 0), 5, tipLength=0.2)
        cv2.arrowedLine(vis_img, tuple(seed_coord.astype(int)), tuple((seed_coord + seed_v_col).astype(int)), (0, 0, 255), 5, tipLength=0.2)

        #label them
        cv2.putText(vis_img, "Row Vector", tuple((seed_coord + seed_v_row * 0.5).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis_img, "Col Vector", tuple((seed_coord + seed_v_col * 0.5).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

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
                        cv2.arrowedLine(step_vis_img, tuple(p_coord.astype(int)), tuple(target_coord.astype(int)), (255,0,255), 3, tipLength=0.2)
            
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
                            cv2.line(vis_img, tuple(all_coords[best_match].astype(int)), tuple(all_coords[grid_map[grid_pos_query]].astype(int)), (0,255,0), 2)

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
            offset = -(r // 2)
            hex_grid.add((r, int(c + offset)))

    return hex_grid

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
    num_cols, num_rows = pattern_size

    target_hex_grid = _generate_target_hex_grid(num_cols, num_rows)

    hex_grid = set(grid_map.keys())

    # Iterate through every point as a potential anchor for the transformation
    r_final, c_final = None, None
    rotation_steps_final = -1

    for r_start, c_start in hex_grid:
        # move hex grid so that the current point is at (0, 0)
        temp_grid = _translate_hex_grid(
            hex_grid,
            (-r_start, -c_start)
        )
            
        rotated_grids = [rotate_hex_grid(temp_grid, i) for i in range(6)]

        for rotation_steps in range(6):
            # Calculate the coverage of the rotated grid against the ideal hex grid
            if target_hex_grid.issubset(rotated_grids[rotation_steps]):
                # If the rotated grid covers the ideal grid, we can calculate the affine transform
                r_final, c_final = r_start, c_start
                rotation_steps_final = rotation_steps
                break

    # build the final grid with the found coordinates
    if r_final is None or c_final is None:
        print("  [ FAIL ] No valid affine transformation found that covers the ideal grid.")


        rotated_hex_grids = [rotate_hex_grid(hex_grid, i) for i in range(6)]

        if visualize:
            print("  [ INFO ] Visualizing all rotated hex grids and ideal grid for comparison...")
            all_grids = [target_hex_grid] + rotated_hex_grids

            labels = ['Ideal Grid'] + [f'Rotation {i * 60}°' for i in range(6)]

            _draw_hex_grids(*all_grids, labels=labels)

        if visualize: cv2.destroyAllWindows()
        return None
    if visualize:
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
            r = i  # Reverse the row order
            offset = -(r // 2)
            c = j + offset

            if (r, c) in new_grid_map:
                idx = new_grid_map[(r, c)]
                final_grid.append(all_coords[idx])
            else:
                # If a point is missing, fill with NaN
                print(f"  [ WARN ] Missing point at ({r}, {c}).")
                return None
            
    final_grid = np.array(final_grid, dtype=np.float32)
    final_grid = np.expand_dims(final_grid, axis=1)

    return final_grid

def _draw_hex_grids(
        *grid_point_sets: Set[Tuple[int, int]],
        labels: Optional[List[str]] = None,
        cell_size: int = 20
    ) -> Optional[np.ndarray]:
        """
        Draws one or more hex‐grid point sets side by side for comparison.
        Each set is rendered as a small grid with filled cells at the found points.
        labels: optional list of titles for each grid (one per set).
        cell_size: the size in pixels of each grid cell (default=20 for higher resolution).
        Returns the concatenated image.
        """
        if not grid_point_sets:
            return None

        imgs = []
        num_grids = len(grid_point_sets)
        # Prepare labels
        if labels is None or len(labels) != num_grids:
            labels = [''] * num_grids

        label_height = cell_size  # reserve one cell height for label

        for grid_points, label in zip(grid_point_sets, labels):
            # compute bounding box for this set
            max_row = max(r for r, c in grid_points)
            min_row = min(r for r, c in grid_points)
            max_col = max(c for r, c in grid_points)
            min_col = min(c for r, c in grid_points)

            range_rows = max_row - min_row + 1
            range_cols = max_col - min_col + 1

            # create image with extra space for the label
            h = label_height * 2 + range_rows * cell_size
            w = range_cols * cell_size
            grid_img = np.ones((h, w, 3), dtype=np.uint8) * 255

            # draw label if provided
            if label:
                cv2.putText(
                    grid_img,
                    label,
                    (5, label_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    cell_size / 40.0,  # scale relative to cell_size
                    (50, 50, 50),
                    1,
                    cv2.LINE_AA
                )

            # draw grid lines (shifted down by label_height * 2)
            for r in range(range_rows + 1):
                y = label_height * 2 + r * cell_size
                cv2.line(grid_img, (0, y), (w, y), (200, 200, 200), 1)
            for c in range(range_cols + 1):
                x = c * cell_size
                cv2.line(grid_img, (x, label_height * 2), (x, label_height * 2 + range_rows * cell_size), (200, 200, 200), 1)

            # fill found points
            for r, c in grid_points:
                pr = r - min_row
                pc = c - min_col
                x1 = pc * cell_size + 1
                y1 = label_height * 2 + pr * cell_size + 1
                x2 = (pc + 1) * cell_size - 1
                y2 = label_height * 2 + (pr + 1) * cell_size - 1
                cv2.rectangle(grid_img, (x1, y1), (x2, y2), (0, 0, 0), -1)

            # draw row/col labels inside the grid
            for r in range(range_rows):
                y = label_height * 2 + r * cell_size + int(cell_size * 0.7)
                cv2.putText(
                    grid_img,
                    str(r + min_row),
                    (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    cell_size / 50.0,
                    (120, 120, 120),
                    1,
                    cv2.LINE_AA
                )
            for c in range(range_cols):
                x = c * cell_size + int(cell_size * 0.4)
                cv2.putText(
                    grid_img,
                    str(c + min_col),
                    (x, label_height * 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    cell_size / 50.0,
                    (120, 120, 120),
                    1,
                    cv2.LINE_AA
                )

            imgs.append(grid_img)

        # pad all images to the same height for horizontal concatenation
        max_h = max(img.shape[0] for img in imgs)
        padded = []
        for img in imgs:
            h, w = img.shape[:2]
            if h < max_h:
                pad = np.ones((max_h - h, w, 3), dtype=np.uint8) * 255
                img = np.vstack((pad, img))
            padded.append(img)

        combined = cv2.hconcat(padded)
        cv2.imshow("Hex Grid Comparison", combined)
        cv2.waitKey(0)
        return combined

def auto_asymm_cricle_hexagon_matching(
    img: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    pattern_size: Tuple[int, int],
    visualize: bool = False
) -> Optional[np.ndarray]:
    
    # ... (Scoring and seed finding is exactly the same as before) ...
    num_cols, num_rows = pattern_size
    if len(keypoints) < num_cols * num_rows * 0.5: return None
    all_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    kdtree = KDTree(all_coords)

    scores = np.array([_calculate_hexagon_score(i, all_coords, kdtree) for i in range(len(all_coords))])
    
    valid_scores = scores[np.isfinite(scores)]
    if len(valid_scores) == 0: return None

    best_seed_idx = np.argmin(scores)
    min_score = scores[best_seed_idx]


    score_threshold = 2.0 
    if min_score > score_threshold:
        print(f"  [ FAIL ] Best score ({min_score:.2f}) is above threshold.")
        return None

    # --- Step 3: Propagate using the new WAVEFRONT method ---
    grid_map = _propagate_from_seed_wavefront(img, best_seed_idx, all_coords, kdtree, pattern_size, visualize)

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

    if visualize: cv2.destroyAllWindows()

    return final_grid
