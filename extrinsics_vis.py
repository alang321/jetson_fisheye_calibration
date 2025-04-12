import cv2
import numpy as np
import os
import glob
import argparse
import sys
from scipy.spatial import distance # For efficient nearest neighbor searches
import math # For checking NaN
import random # For potential future use if adding RANSAC/removal

# --- Matplotlib for 3D Plotting ---
# Ensure you have matplotlib installed: pip install matplotlib
import matplotlib.pyplot as plt
# The next line is sometimes needed for 3D plotting, though often implicitly handled
# from mpl_toolkits.mplot3d import Axes3D # No longer explicitly required for basic 3D

# --- Circle Grid Detection Logic (run_four_corner_local_vector_walk and helpers) ---
# <<< PASTE the run_four_corner_local_vector_walk function and find_nearest_available function here >>>
# --- [From the previous code blocks] ---
def find_nearest_available(target_pt, available_coords, available_indices, max_dist_sq=np.inf):
    """Finds the nearest point in available_coords to target_pt."""
    if not available_indices:
        return -1, np.inf # No points available

    # Calculate squared distances (faster than sqrt)
    # Ensure target_pt is 2D for cdist
    target_pt_2d = np.array(target_pt).reshape(1, 2)
    dist_sq = distance.cdist(target_pt_2d, available_coords)[0]**2 # Use available_coords here

    best_dist_sq = np.inf
    best_idx_in_available = -1

    # Find the minimum distance within the threshold
    valid_indices_local = np.where(dist_sq < max_dist_sq)[0] # Indices within available_coords

    if len(valid_indices_local) > 0:
         min_local_idx = np.argmin(dist_sq[valid_indices_local]) # Index within the valid subset
         best_idx_in_available = valid_indices_local[min_local_idx] # Index within the 'available_coords' array
         best_dist_sq = dist_sq[best_idx_in_available]
    else:
         return -1, np.inf # No point found within max_dist_sq

    # Map back to the original index from detected_keypoints
    original_kp_index = available_indices[best_idx_in_available]
    return original_kp_index, best_dist_sq

def rotation_matrix_to_euler_zyx(R):
    """
    Converts a 3x3 rotation matrix to Euler angles (yaw, pitch, roll)
    corresponding to the ZYX intrinsic convention (R = Rz * Ry * Rx).

    Args:
        R (np.array): A 3x3 rotation matrix.

    Returns:
        tuple: (yaw, pitch, roll) angles in radians.
               Returns None if the matrix is invalid.
               Handles gimbal lock.
    """
    assert R.shape == (3, 3), "Input must be a 3x3 matrix"

    sy_thresh = 1.0 - 1e-6 # Threshold for singularity check

    if abs(R[2, 0]) < sy_thresh:
        # Non-singular case (cos(pitch) is not close to 0)
        # pitch = -asin(R[2, 0])
        # Use atan2 for potentially better numerical stability getting pitch
        pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2))

        # yaw = atan2(R[1, 0] / cos(pitch), R[0, 0] / cos(pitch))
        yaw = math.atan2(R[1, 0], R[0, 0])

        # roll = atan2(R[2, 1] / cos(pitch), R[2, 2] / cos(pitch))
        roll = math.atan2(R[2, 1], R[2, 2])

    else:
        # Singular case: Gimbal Lock (|cos(pitch)| is close to 0)
        # R[2, 0] is close to -1 (pitch = +pi/2) or +1 (pitch = -pi/2)
        print("Warning: Gimbal lock detected (pitch is close to +/- 90 degrees).")
        print("         Roll angle set to 0, Yaw adjusted accordingly.")

        pitch = math.pi / 2.0 if R[2, 0] < 0 else -math.pi / 2.0
        roll = 0.0 # Conventionally set roll to 0 in gimbal lock
        # yaw = atan2( yaw_component_y , yaw_component_x ) derived from R elements
        # If pitch = +pi/2: R[0,1] = sin(yaw-roll), R[0,2] = cos(yaw-roll) -> yaw - roll = atan2(R[0,1], R[0,2])
        # If pitch = -pi/2: R[0,1] = -sin(yaw+roll), R[0,2] = -cos(yaw+roll) -> yaw + roll = atan2(-R[0,1], -R[0,2])
        yaw = math.atan2(-R[0, 1], -R[0, 2]) if pitch < 0 else math.atan2(R[0, 1], R[0, 2])


    return yaw, pitch, roll # Radians

def run_four_corner_local_vector_walk(image_to_select_on, detected_keypoints, objp, pattern_size, visualize=True):
    """
    Uses 4 corner clicks, predicts all points via transform to get vectors,
    then walks the grid using vectors applied to the previously found point,
    matching via nearest neighbor search at each step.

    Args:
        image_to_select_on: Color image.
        detected_keypoints: List of cv2.KeyPoint found by SimpleBlobDetector.
        objp: The (N, 3) array of ideal object points.
        pattern_size: Tuple (cols, rows).
        visualize: Boolean flag to enable step-by-step visualization.

    Returns:
        A numpy array of shape (num_points, 1, 2) float32 corners, or None.
    """
    cols, rows = pattern_size
    num_expected_points = cols * rows
    print("\n--- Starting Four-Corner + Local Vector NN Walk ---")
    if not detected_keypoints:
        print("  Error: No keypoints provided to run_four_corner_local_vector_walk.")
        return None
    print("Detected Blobs:", len(detected_keypoints))
    if len(detected_keypoints) < 4: print("  Error: Not enough blobs detected (< 4)."); return None
    if len(detected_keypoints) < num_expected_points * 0.8: print(f"  Warning: Significantly fewer blobs detected ({len(detected_keypoints)}) than expected ({num_expected_points}).")

    # --- 1. Get 4 Corner Clicks ---
    img_display_corners = image_to_select_on.copy()
    kp_coords = np.array([kp.pt for kp in detected_keypoints], dtype=np.float32)
    for pt in kp_coords: cv2.circle(img_display_corners, tuple(pt.astype(int)), 3, (200, 200, 200), 1)
    window_name_corners = "Click 4 Corners (TL, TR, BL, BR)"
    # Use WINDOW_NORMAL for resizability, WINDOW_AUTOSIZE might be too small/large
    cv2.namedWindow(window_name_corners, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_corners, 1000, 700) # Suggest a reasonable size
    corner_labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    ideal_corner_indices = [0, cols - 1, (rows - 1) * cols, num_expected_points - 1]
    label_to_ideal_idx = dict(zip(corner_labels, ideal_corner_indices))
    clicked_kp_indices = {}
    current_click_idx = 0

    # Using a dictionary to pass multiple parameters to the callback
    callback_params = {
        'kp_coords': kp_coords,
        'image_to_select_on': image_to_select_on,
        'img_display_corners': img_display_corners,
        'corner_labels': corner_labels,
        'clicked_kp_indices': clicked_kp_indices,
        'current_click_idx_ref': [current_click_idx] # Use list to pass by reference
    }

    def four_corner_click_callback(event, x, y, flags, param):
        # Unpack parameters
        kp_coords_cb = param['kp_coords']
        image_to_select_on_cb = param['image_to_select_on']
        img_display_corners_cb = param['img_display_corners']
        corner_labels_cb = param['corner_labels']
        clicked_kp_indices_cb = param['clicked_kp_indices']
        current_click_idx_ref_cb = param['current_click_idx_ref']
        current_click_idx_val = current_click_idx_ref_cb[0] # Get current value

        if current_click_idx_val >= len(corner_labels_cb): return
        target_label = corner_labels_cb[current_click_idx_val]

        if event == cv2.EVENT_LBUTTONDOWN:
            distances = distance.cdist([(x, y)], kp_coords_cb)[0]; nearest_kp_idx = np.argmin(distances)
            if nearest_kp_idx in clicked_kp_indices_cb.values():
                print(f"  Warning: Blob {nearest_kp_idx} already selected. Click different blob for {target_label}.")
                return

            clicked_kp_indices_cb[target_label] = nearest_kp_idx
            pt_clicked_coords = kp_coords_cb[nearest_kp_idx]
            print(f"  Clicked {target_label} near blob {nearest_kp_idx} at ({pt_clicked_coords[0]:.0f}, {pt_clicked_coords[1]:.0f})")

            # Update display (draw on a fresh copy each time)
            temp_display = image_to_select_on_cb.copy()
            for i, pt in enumerate(kp_coords_cb):
                 pt_int = tuple(pt.astype(int)); cv2.circle(temp_display, pt_int, 3, (200, 200, 200), 1)

            corner_colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)] # TL, TR, BL, BR
            for i, lbl in enumerate(corner_labels_cb):
                if lbl in clicked_kp_indices_cb:
                    kp_idx = clicked_kp_indices_cb[lbl]; pt = kp_coords_cb[kp_idx]; pt_int = tuple(pt.astype(int))
                    cv2.circle(temp_display, pt_int, 6, corner_colors[i], 2); cv2.circle(temp_display, pt_int, 1, (0,0,0), -1)
                    text_org = (pt_int[0] + 8, pt_int[1] + 8); cv2.putText(temp_display, lbl[:2], text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            img_display_corners_cb[:] = temp_display[:] # Update the display image in-place

            current_click_idx_ref_cb[0] += 1 # Increment via reference
            current_click_idx_val = current_click_idx_ref_cb[0] # Update local value after increment

            if current_click_idx_val < len(corner_labels_cb):
                print(f"-> Now click: {corner_labels_cb[current_click_idx_val]}")
            else:
                print("-> All 4 corners clicked. Press 'y' to proceed, 'r' to restart, 'q' to quit.")

    cv2.setMouseCallback(window_name_corners, four_corner_click_callback, callback_params)
    print(f"-> Click the blob for: {corner_labels[current_click_idx]}")

    while True:
        cv2.imshow(window_name_corners, img_display_corners)
        key = cv2.waitKey(20) & 0xFF

        current_click_idx = callback_params['current_click_idx_ref'][0] # Get latest value

        if current_click_idx == len(corner_labels) and key == ord('y'):
            break
        if key == ord('r'):
             print("Restarting corner selection...")
             callback_params['current_click_idx_ref'][0] = 0 # Reset counter
             callback_params['clicked_kp_indices'].clear() # Clear clicked points
             # Reset display image
             img_display_corners[:] = image_to_select_on.copy()
             [cv2.circle(img_display_corners, tuple(pt.astype(int)), 3, (200, 200, 200), 1) for pt in kp_coords]
             print(f"-> Click the blob for: {corner_labels[0]}") # Prompt for first corner again
        if key == ord('q'):
            print("Quit during corner selection.")
            cv2.destroyWindow(window_name_corners)
            return None
        # Check if window was closed manually
        # Use getWindowProperty for robustness; 0 means visible, -1 means destroyed/not visible
        if cv2.getWindowProperty(window_name_corners, cv2.WND_PROP_VISIBLE) < 1:
             print("Corner selection window closed by user.")
             try: cv2.destroyWindow(window_name_corners) # Try to close explicitly
             except: pass
             return None # Or handle as quit

    try: cv2.destroyWindow(window_name_corners)
    except: pass
    clicked_kp_indices = callback_params['clicked_kp_indices'] # Get final dictionary

    # --- 2. Estimate Transform and Predict ALL Points ---
    print("Estimating transform and predicting all points (for vector calculation)...")
    if len(clicked_kp_indices) != 4:
        print(f"  Error: Did not collect 4 corner points (collected {len(clicked_kp_indices)}).")
        return None

    img_pts_corners_list = []; obj_pts_corners_list = []
    for label in corner_labels:
        if label not in clicked_kp_indices:
            print(f"  Error: Corner '{label}' was not clicked.")
            return None
        kp_idx = clicked_kp_indices[label]; ideal_idx = label_to_ideal_idx[label]
        img_pts_corners_list.append(kp_coords[kp_idx])
        obj_pts_corners_list.append(objp[ideal_idx, :2]) # Use only X, Y for homography

    img_pts_corners = np.array(img_pts_corners_list, dtype=np.float32)
    obj_pts_corners = np.array(obj_pts_corners_list, dtype=np.float32)

    try:
        H, mask = cv2.findHomography(obj_pts_corners, img_pts_corners, cv2.RANSAC, 5.0) # Increased RANSAC threshold slightly
        if H is None or H.shape != (3, 3):
            raise ValueError("Transform estimation failed (findHomography returned None or invalid shape)")
        # Check mask if needed: num_inliers = np.sum(mask)
    except Exception as e:
        print(f"  Error estimating perspective transform: {e}.")
        return None

    obj_pts_all_xy = objp[:, :2].astype(np.float32).reshape(-1, 1, 2)
    try:
        predicted_img_pts_all = cv2.perspectiveTransform(obj_pts_all_xy, H)
        if predicted_img_pts_all is None:
            raise ValueError("perspectiveTransform returned None")
        predicted_img_pts_all = predicted_img_pts_all.reshape(-1, 2) # Shape (N, 2)
    except Exception as e:
        print(f"  Error applying perspective transform: {e}")
        return None
    print("  Transform estimated and all points predicted.")

    # --- 3. Initialize for NN Walk ---
    print("Starting Local Vector Nearest-Neighbor walk...")
    final_corners = np.full((num_expected_points, 1, 2), np.nan, dtype=np.float32)
    # Create a list of (original_index, coordinate_tuple) for available points
    available_map = {idx: tuple(coord) for idx, coord in enumerate(kp_coords)}
    available_indices = list(available_map.keys()) # Original indices [0, 1, ..., N-1]
    available_coords_list = list(available_map.values()) # Coordinates

    # --- Assign corners and remove from available pool ---
    indices_to_remove_from_available = [] # Store original indices to remove
    for label in corner_labels:
        grid_idx = label_to_ideal_idx[label]
        kp_idx = clicked_kp_indices[label] # Original index of the matched keypoint
        final_corners[grid_idx, 0, :] = kp_coords[kp_idx]
        if kp_idx in available_map:
            indices_to_remove_from_available.append(kp_idx)
            del available_map[kp_idx] # Remove from map

    available_indices = list(available_map.keys()) # Update list of available original indices
    available_coords = np.array(list(available_map.values()), dtype=np.float32) # Update numpy array of available coordinates
    matched_count = len(indices_to_remove_from_available)
    print(f"  Assigned {matched_count} corners initially.")

    # Estimate max search distance (using pixel coords from corners)
    tl_pt = final_corners[label_to_ideal_idx["Top-Left"], 0, :]
    tr_pt = final_corners[label_to_ideal_idx["Top-Right"], 0, :]
    bl_pt = final_corners[label_to_ideal_idx["Bottom-Left"], 0, :]
    br_pt = final_corners[label_to_ideal_idx["Bottom-Right"], 0, :]

    # Check if corner points are valid before calculating distances
    if np.isnan(tl_pt).any() or np.isnan(tr_pt).any() or np.isnan(bl_pt).any() or np.isnan(br_pt).any():
        print("  Error: One or more corner points are NaN, cannot estimate spacing.")
        # Fallback to a default? Or just fail? Let's use a default guess.
        approx_spacing_px = 50.0 # Default guess
        print("  Warning: Using default approx pixel spacing: 50.0")
    else:
        diag1_dist = np.linalg.norm(tl_pt - br_pt)
        diag2_dist = np.linalg.norm(tr_pt - bl_pt)
        avg_diag = (diag1_dist + diag2_dist) / 2.0
        if cols > 1 and rows > 1:
            approx_spacing_px = avg_diag / np.sqrt((cols - 1)**2 + (rows - 1)**2)
        elif cols > 1:
            approx_spacing_px = np.linalg.norm(tl_pt - tr_pt) / (cols - 1)
        elif rows > 1:
            approx_spacing_px = np.linalg.norm(tl_pt - bl_pt) / (rows - 1)
        else:
            approx_spacing_px = 50.0 # Fallback for 1x1 grid (or error)

    max_dist_sq = (approx_spacing_px * 0.9)**2 # Search radius slightly LARGER than before (0.9 instead of 0.75)
    search_radius = int(round(math.sqrt(max_dist_sq)))
    print(f"  Approx pixel spacing estimate: {approx_spacing_px:.1f}, Max Search Dist Sq: {max_dist_sq:.1f} (Radius ~{search_radius}px)")

    # --- 4. Perform the Serpentine Grid Walk ---
    debug_walk_win = None
    if visualize:
        debug_walk_win = "Local Vector NN Walk"
        cv2.namedWindow(debug_walk_win, cv2.WINDOW_NORMAL); cv2.resizeWindow(debug_walk_win, 1000, 700)
        run_mode = False # Start in step mode

    # Initialize walker state
    # Start from TL, which is guaranteed to be filled
    prev_idx = label_to_ideal_idx["Top-Left"]
    prev_found_pt = final_corners[prev_idx, 0, :]

    # Serpentine walk order (same as object point generation)
    walk_indices = []
    for r in range(rows):
        if r % 2 == 0: # Even rows: left to right
            for c in range(cols): walk_indices.append(r * cols + c)
        else:          # Odd rows: right to left
            for c in range(cols - 1, -1, -1): walk_indices.append(r * cols + c)

    for current_grid_idx in walk_indices:
        # Skip if already filled (corners or previously matched in walk)
        if not np.isnan(final_corners[current_grid_idx, 0, 0]):
            # Update previous state *if* this point was successfully filled in a *previous* iteration
            # This ensures we always use the *actual* location of the previously processed grid point
            if current_grid_idx != prev_idx: # Check avoids self-update if corners are consecutive in walk order
                prev_idx = current_grid_idx
                prev_found_pt = final_corners[current_grid_idx, 0, :]
            continue # Already done, move to next in walk order

        # --- Start processing for current_grid_idx ---
        if visualize:
            vis_step = image_to_select_on.copy()
            # Draw all available blobs faintly (using original kp_coords and available_indices)
            current_available_coords_tuples = [tuple(kp_coords[k_idx].astype(int)) for k_idx in available_indices]
            for pt_int in current_available_coords_tuples:
                cv2.circle(vis_step, pt_int, 3, (180, 180, 180), 1)
            # Draw already matched points (green filled circles)
            for m_idx in range(num_expected_points):
                if not np.isnan(final_corners[m_idx, 0, 0]):
                    pt_int = tuple(final_corners[m_idx,0,:].astype(int))
                    cv2.circle(vis_step, pt_int, 5, (0,180,0), -1) # Filled green
            # Highlight the previous successfully found point (bright green outline)
            cv2.circle(vis_step, tuple(prev_found_pt.astype(int)), 7, (0, 255, 0), 2)

        # Get predicted locations from global transform for current and previous ideal points
        pred_pt_current_ideal = predicted_img_pts_all[current_grid_idx]
        pred_pt_prev_ideal = predicted_img_pts_all[prev_idx]

        # Calculate the *predicted vector* between these ideal locations
        vector_pred = pred_pt_current_ideal - pred_pt_prev_ideal

        # Calculate the primary *search point* by applying the vector to the *last found* point
        search_point = prev_found_pt + vector_pred
        search_point_int = tuple(search_point.astype(int))

        # --- Draw prediction details on vis_step ---
        if visualize:
            cv2.line(vis_step, tuple(prev_found_pt.astype(int)), search_point_int, (255, 255, 0), 1) # Cyan vector line
            cv2.drawMarker(vis_step, search_point_int, (0, 0, 255), cv2.MARKER_CROSS, 12, 2) # Bold Red Cross at search point
            cv2.circle(vis_step, search_point_int, search_radius, (255, 0, 0), 1) # Blue search circle

        match_status_text = f"Grid {current_grid_idx} (from {prev_idx}): Search @ ({search_point[0]:.0f}, {search_point[1]:.0f})"
        matched_kp_idx = -1 # Reset

        # --- Perform NN Search ---
        if not available_indices: # Check if the list is empty
            match_status_text += " -> FAIL (No blobs left!)"
        else:
            # Make sure available_coords is up-to-date and not empty
            if available_coords.shape[0] == 0:
                 match_status_text += " -> FAIL (Avail coords array empty!)"
            else:
                # Use the currently available coordinates and their original indices
                found_kp_idx, dist_sq = find_nearest_available(
                    search_point, available_coords, available_indices, max_dist_sq
                )

                if found_kp_idx != -1: # Match found! found_kp_idx is the ORIGINAL index
                    # Assign the matched point using the original index
                    found_pt = kp_coords[found_kp_idx]
                    final_corners[current_grid_idx, 0, :] = found_pt
                    matched_kp_idx = found_kp_idx
                    match_status_text += f" -> MATCH Blob {found_kp_idx} (Dist: {math.sqrt(dist_sq):.1f})"

                    # Draw match details before updating state
                    match_pt_int = tuple(found_pt.astype(int))
                    if visualize:
                        cv2.circle(vis_step, match_pt_int, 6, (0, 255, 255), 2) # Yellow highlight
                        cv2.line(vis_step, search_point_int, match_pt_int, (0, 255, 0), 1) # Green line to match

                    # ---- CRITICAL: Update state for the NEXT iteration ----
                    prev_found_pt = found_pt       # Use the newly found point as the base for the next vector
                    prev_idx = current_grid_idx    # The current index becomes the previous index for the next step

                    # Remove the matched point from the available pool
                    try:
                        idx_in_list = available_indices.index(found_kp_idx) # Find where it is in the current list
                        available_indices.pop(idx_in_list) # Remove original index from list
                        available_coords = np.delete(available_coords, idx_in_list, axis=0) # Remove coordinate from array
                    except ValueError:
                        print(f"  Warning: Matched kp_idx {found_kp_idx} not found in available_indices list during removal.")
                        # This shouldn't happen if logic is correct
                    matched_count += 1

                else: # No match found within radius
                    match_status_text += f" -> NO MATCH (Radius {search_radius}px)"
                    # Highlight search point/circle in bright red if no match
                    if visualize:
                        cv2.circle(vis_step, search_point_int, search_radius, (0, 0, 255), 2) # Thicker red circle
                    # CRITICAL: Do NOT update prev_found_pt or prev_idx if no match was found!
                    # The next iteration should still originate from the last *successfully* found point.

        # --- Display step visualization and handle input ---
        if visualize:
            cv2.putText(vis_step, match_status_text, (10, vis_step.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(vis_step, f"Matched: {matched_count}/{num_expected_points}", (10, vis_step.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow(debug_walk_win, vis_step)

            # --- Wait for user input ---
            if not run_mode:
                key = cv2.waitKey(0) & 0xFF # Wait indefinitely in step mode
            else:
                key = cv2.waitKey(1) & 0xFF # Very short delay in run mode for viz update

            if key == ord('r'): print("Switching to Run mode"); run_mode = True
            if key == ord('q'):
                print("Quit during vector walk.")
                try: cv2.destroyWindow(debug_walk_win)
                except: pass
                return None
            # Check if window closed
            if cv2.getWindowProperty(debug_walk_win, cv2.WND_PROP_VISIBLE) < 1:
                 print("Vector walk window closed by user.")
                 return None

    # --- 5. Final Check and Cleanup ---
    if visualize and debug_walk_win is not None:
        try: cv2.destroyWindow(debug_walk_win)
        except cv2.error: pass # Ignore if already closed

    final_matched_count = num_expected_points - np.count_nonzero(np.isnan(final_corners[:,0,0]))
    print(f"Finished local vector walk. Matched {final_matched_count}/{num_expected_points} points.")

    if final_matched_count < num_expected_points: # Use '<' as we might have missed some points
        print("  Warning: Failed to match all points. Result is incomplete.")
        # Decide whether to return partial results or None. Returning None is safer.
        # If partial results are sometimes okay, modify this.
        # Let's check *which* points are missing for more info
        missing_indices = np.where(np.isnan(final_corners[:, 0, 0]))[0]
        print(f"  Missing grid indices: {missing_indices}")
        return None
    elif np.isnan(final_corners).any():
        # This case should be caught above, but as a safety check
        print("  Warning: NaN values detected in final corners despite matching count. Result is invalid.")
        return None
    else:
        print("  Successfully matched all points via Local Vector NN Walk.")
        return final_corners.astype(np.float32) # Ensure float32


# --- Utility Functions ---
def compose_T(R, t):
    """Compose 4x4 homogeneous transformation matrix T from R (3x3) and t (3x1 or 1x3)."""
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.flatten()
    return T

def decompose_T(T):
    """Decompose 4x4 homogeneous transformation matrix T into R (3x3) and t (3x1)."""
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    return R, t

def invert_T(T):
    """Invert 4x4 homogeneous transformation matrix T."""
    R, t = decompose_T(T)
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = compose_T(R_inv, t_inv)
    return T_inv

# --- 3D Plotting Function ---
def plot_3d_setup(objp, ground_ref_point_world, front_center_world, T_imu_world, T_world_cam):
    """
    Visualizes the calibration setup in 3D using Matplotlib.

    Args:
        objp (np.array): Nx3 array of grid object points (mm, World frame).
        T_world_cam (np.array): 4x4 matrix, Camera pose in World frame (mm).
        T_world_body (np.array): 4x4 matrix, IMU/Body pose in World frame (mm).
        t_world_F (np.array): 3x1 vector, Front Center position in World frame (mm).
    """
    print("\nGenerating 3D visualization...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')


    # --- Plot Grid Points ---
    ax.scatter(objp[:, 0], objp[:, 1], objp[:, 2], c='black', marker='.', label='Grid Points (World XY Plane)')

    # --- Plot Origins ---
    ax.scatter(0, 0, 0, c='red', s=100, marker='o', label='World Origin (Top-Left Grid Pt)')

    # plot intermediat points
    ax.scatter(ground_ref_point_world[0], ground_ref_point_world[1], ground_ref_point_world[2], c='blue', s=100, marker='o', label='Ground Reference Point')
    ax.scatter(front_center_world[0], front_center_world[1], front_center_world[2], c='green', s=100, marker='o', label='Drone Front Center')

    # --- Plot Coordinate Frames ---
    axis_length = max(np.ptp(objp[:,0]), np.ptp(objp[:,1])) * 0.2 # Scale axis length based on grid size
    if axis_length < 50: axis_length = 50 # Minimum length
    print(f"Plotting axis length: {axis_length:.1f} mm")

    # cam frame plotting
    t_world_cam = T_world_cam[0:3, 3] # Camera origin in world frame
    R_world_cam = T_world_cam[0:3, 0:3] # Camera rotation in world frame
    cam_origin = t_world_cam.flatten()
    cam_x_axis = R_world_cam[:, 0] # Camera X in World coords
    cam_y_axis = R_world_cam[:, 1] # Camera Y in World coords
    cam_z_axis = R_world_cam[:, 2] # Camera Z in World coords
    ax.quiver(cam_origin[0], cam_origin[1], cam_origin[2], cam_x_axis[0], cam_x_axis[1], cam_x_axis[2], length=axis_length, color='red', label='Cam X (Right)', arrow_length_ratio=0.1)
    ax.quiver(cam_origin[0], cam_origin[1], cam_origin[2], cam_y_axis[0], cam_y_axis[1], cam_y_axis[2], length=axis_length, color='green', label='Cam Y (Down)', arrow_length_ratio=0.1)
    ax.quiver(cam_origin[0], cam_origin[1], cam_origin[2], cam_z_axis[0], cam_z_axis[1], cam_z_axis[2], length=axis_length, color='blue', label='Cam Z (Fwd)', arrow_length_ratio=0.1)

    # plt imu aka drone frame
    origin = T_imu_world[0:3, 3] # IMU origin in world frame
    ax.quiver(origin[0], origin[1], origin[2],
               T_imu_world[0, 0], T_imu_world[1, 0], T_imu_world[2, 0],
               length=axis_length, color='red', label='Drone X (Forward)', arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2],
               T_imu_world[0, 1], T_imu_world[1, 1], T_imu_world[2, 1],
               length=axis_length, color='green', label='Drone Y (Right)', arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2],
               T_imu_world[0, 2], T_imu_world[1, 2], T_imu_world[2, 2],
               length=axis_length, color='blue', label='Drone Z (Down)', arrow_length_ratio=0.1)

    # --- Set Plot Limits and Aspect Ratio ---
    # Collect all key points to determine plot range
    all_points = np.vstack([
        objp,
        ground_ref_point_world.T,
        front_center_world.T
    ])

    max_range = np.array([all_points[:,0].max()-all_points[:,0].min(),
                          all_points[:,1].max()-all_points[:,1].min(),
                          all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0

    mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
    mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
    mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- Labels and Legend ---
    ax.set_xlabel('World X (Right, mm)')
    ax.set_ylabel('World Y (Up, mm)')
    ax.set_zlabel('World Z (into Wall, mm)')
    ax.set_title('3D Visualization of Calibration Setup')
    ax.legend(fontsize='small') # Adjust legend size if needed
    plt.tight_layout() # Adjust layout
    print("Showing plot. Close the plot window to continue...")
    plt.show()


# --- Configuration and Argument Parsing ---
# ... (Argument parsing remains the same) ...
DEFAULT_GRID_COLS = 4
DEFAULT_GRID_ROWS = 11
DEFAULT_SPACING_MM = 20.0
DEFAULT_DRONE_WALL_DIST_M = 1.5 # Distance wall-to-drone-front-center
DEFAULT_FRONT_CENTER_TO_IMU_M = "0.0,0.0,0.0" # Vector from front-center TO imu in drone body frame

parser = argparse.ArgumentParser(
    description='Camera-to-IMU Extrinsic Calibration using a single view and geometric constraints.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image', type=str, required=True,
                    help='Path to the single calibration image.')
parser.add_argument('--intrinsics', type=str, required=True,
                    help='Path to the camera intrinsics file (.npz format, containing K and D).')
parser.add_argument('--cols', type=int, default=DEFAULT_GRID_COLS,
                    help='Number of circles horizontally in the grid.')
parser.add_argument('--rows', type=int, default=DEFAULT_GRID_ROWS,
                    help='Number of circles vertically in the grid.')
parser.add_argument('--spacing', type=float, default=DEFAULT_SPACING_MM,
                    help='Distance between adjacent circle centers in millimeters (mm).')
parser.add_argument('--drone_wall_dist', type=float, default=DEFAULT_DRONE_WALL_DIST_M,
                    help='Distance from wall (grid plane Z=0) to drone\'s "Front Center" reference point, in meters (m). Assumes Front Center is at World X=0, Y=0.')
parser.add_argument('--pattern_height', type=float, default=DEFAULT_DRONE_WALL_DIST_M,
                    help='Distance from floor to Bottom center point row of pattern.')
parser.add_argument('--front_center_to_imu', type=str, default=DEFAULT_FRONT_CENTER_TO_IMU_M,
                    help='Translation vector [x,y,z] from drone "Front Center" TO IMU origin, in meters (m), expressed in Drone Body Frame (FRD). Example: "-0.15,0.0, -0.1"')
parser.add_argument('--output', type=str, default='camera_imu_extrinsics.npz',
                    help='Output file path for Camera-to-IMU extrinsic data (.npz).')
# Detection parameters
parser.add_argument('--visualize_serpentine', action='store_true',
                    help='Enable step-by-step visualization during the serpentine grid walk.')
parser.add_argument('--blob_min_area', type=float, default=25.0, help='Initial Min Area for blob detection.')
parser.add_argument('--blob_max_area', type=float, default=5000.0, help='Initial Max Area for blob detection.')
parser.add_argument('--blob_min_circ', type=float, default=0.6, help='Min Circularity for blob detection.')
parser.add_argument('--blob_min_conv', type=float, default=0.8, help='Min Convexity for blob detection.')
parser.add_argument('--blob_min_inertia', type=float, default=0.1, help='Min Inertia Ratio for blob detection.')
parser.add_argument('--preprocess', type=str, default='clahe', choices=['none', 'clahe', 'thresh_bin', 'thresh_inv', 'adapt_bin', 'adapt_inv'],
                    help='Preprocessing method for blob detection.')
parser.add_argument('--thresh_val', type=int, default=127, help='Threshold value for manual thresholding (0-255).')
parser.add_argument('--adapt_block', type=int, default=11, help='Block size for adaptive threshold (odd number >= 3).')
parser.add_argument('--adapt_c', type=int, default=2, help='Constant C for adaptive threshold.')
parser.add_argument('--no_plot', action='store_true', help='Skip generating the 3D plot.')


args = parser.parse_args()

# --- Print Assumptions ---
# ... (remains the same) ...
print("\n" + "="*30)
print("IMPORTANT ASSUMPTIONS:")
print("1. World Frame (W): Origin at center of top-left grid circle. X-right(wall), Y-up(wall), Z-out(perp. to wall).")
print("2. Drone Body Frame (B): Origin at IMU. X-forward(out), Y-right, Z-down.")
print("3. Drone Orientation: Assumed PERFECTLY level (Body XY plane parallel to ground) and normal to wall (Body X || World Z).")
print("4. Drone Position: Assumed 'Front Center' point is at World X=0, Y=0, Z=drone_wall_dist.")
print("5. Input Vector: 'front_center_to_imu' is from Front Center TO IMU in Body frame coordinates.")
print("ACCURACY DEPENDS CRITICALLY ON THESE ASSUMPTIONS HOLDING TRUE!")
print("="*30 + "\n")


# --- Setup ---
pattern_size = (args.cols, args.rows)
spacing_mm = args.spacing
drone_wall_dist_mm = args.drone_wall_dist * 1000.0

try:
    fci_list = [float(x.strip()) for x in args.front_center_to_imu.split(',')]
    if len(fci_list) != 3: raise ValueError("Must have 3 elements")
    front_center_to_imu_mm = np.array(fci_list) * 1000.0 # Convert to mm
    print(f"Parsed front_center_to_imu (mm, Body Frame): {front_center_to_imu_mm}")
except Exception as e:
    print(f"Error parsing --front_center_to_imu '{args.front_center_to_imu}': {e}")
    sys.exit(1)

# --- Load Intrinsics ---
# ... (remains the same) ...
print(f"Loading intrinsics from: {args.intrinsics}")
if not os.path.exists(args.intrinsics):
    print(f"Error: Intrinsics file not found at {args.intrinsics}")
    sys.exit(1)
try:
    with np.load(args.intrinsics) as data:
        K = data['K']
        D = data['D']
    print("Intrinsics loaded successfully.")
    print("K (Camera Matrix):\n", K)
    print("D (Distortion Coefficients):\n", D.flatten())
except Exception as e:
    print(f"Error loading intrinsics: {e}")
    sys.exit(1)

# --- Prepare Object Points (World Frame, origin at top-left circle) ---
# ... (remains the same) ...
print(f"\nGenerating object points for {pattern_size} grid with spacing={spacing_mm} mm (World: X-Right, Y-Down, Z-Out)...")
objp = np.zeros((args.cols * args.rows, 3), np.float32)
for r in range(args.rows):
    for c in range(args.cols):
        idx = r * args.cols + c

        # --- CORRECTED Y Calculation (Y increases downwards) ---
        # Y=0 is at the level of the top row (r=0) origin
        # As r increases (moving down physically), Y increases
        objp[idx, 1] = r * spacing_mm
        # ---

        # X coordinate increases rightwards by column (World X is right)
        row_offset = (r % 2) * (spacing_mm / 2.0) # Offset is still based on row index
        objp[idx, 0] = c * spacing_mm + row_offset

        # Z coordinate is 0 (Grid is on the Z=0 plane of World frame)
        objp[idx, 2] = 0
print("Object points generated (units: mm).")


# --- Load Image & Detect Grid ---
# ... (image loading, preprocessing, blob detection, corner finding - remains the same) ...
print(f"\nLoading image: {args.image}")
if not os.path.exists(args.image):
    print(f"Error: Image file not found at {args.image}")
    sys.exit(1)
img_color = cv2.imread(args.image)
if img_color is None:
    print(f"Error: Could not read image file {args.image}")
    sys.exit(1)
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

print(f"Applying preprocessing: {args.preprocess}")
processed_gray = gray # Default to none
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
if args.preprocess == 'clahe':
    processed_gray = clahe.apply(gray)
elif args.preprocess == 'thresh_bin':
    _, processed_gray = cv2.threshold(gray, args.thresh_val, 255, cv2.THRESH_BINARY)
elif args.preprocess == 'thresh_inv':
     _, processed_gray = cv2.threshold(gray, args.thresh_val, 255, cv2.THRESH_BINARY_INV)
elif args.preprocess == 'adapt_bin':
    block = max(3, args.adapt_block | 1) # Ensure odd >= 3
    processed_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, block, args.adapt_c)
elif args.preprocess == 'adapt_inv':
    block = max(3, args.adapt_block | 1) # Ensure odd >= 3
    processed_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, block, args.adapt_c)

print("\nInitializing Blob Detector...")
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True; params.minArea = args.blob_min_area; params.maxArea = args.blob_max_area
params.filterByCircularity = True; params.minCircularity = args.blob_min_circ
params.filterByConvexity = True; params.minConvexity = args.blob_min_conv
params.filterByInertia = True; params.minInertiaRatio = args.blob_min_inertia
blob_detector = cv2.SimpleBlobDetector_create(params)

keypoints = blob_detector.detect(processed_gray)
print(f"Detected {len(keypoints)} blobs initially.")
if not keypoints:
    print("Error: No blobs detected. Check image quality, lighting, and blob detector parameters.")
    sys.exit(1)

print("\nAttempting to find grid corners using 4-corner walk...")
corners = run_four_corner_local_vector_walk(img_color, keypoints, objp, pattern_size, args.visualize_serpentine)

if corners is None:
    print("\nError: Failed to find grid corners using the assisted method.")
    sys.exit(1)

print("\nGrid corners found successfully.")
cv2.destroyAllWindows() # Close OpenCV windows before showing matplotlib plot


# --- Step 1: Calculate Camera Pose in World Frame (T_world_cam) ---
print("\nCalculating Camera Pose in World Frame (T_world_cam) via solvePnP...")
T_world_cam = None # Initialize

if corners is None: # Check if corners detection failed
    print("\nError: Grid corners were not found (corners is None). Cannot proceed.")
    sys.exit(1)
if not isinstance(corners, np.ndarray):
     print(f"\nError: Grid corners are not a numpy array (type: {type(corners)}). Cannot proceed.")
     sys.exit(1)
print(f"Debug: corners type received: {corners.dtype}, shape: {corners.shape}")


if corners.shape[0] != objp.shape[0]:
     print(f"Error: Mismatch between object points ({objp.shape[0]}) and detected image points ({corners.shape[0]}).")
     sys.exit(1)

try:
    print("Undistorting detected corner points...")
    # 1. Reshape
    corners_reshaped = corners
    print(f"Debug: corners_reshaped shape: {corners_reshaped.shape}, dtype: {corners_reshaped.dtype}")

    # Pass corners_float64 and ensure P=K_cv is also float64
    corners_undistorted_norm = cv2.fisheye.undistortPoints(corners_reshaped, K, D, P=K)

    # Convert result back to float32 if needed downstream (solvePnP often takes float32)
    corners_undistorted = corners_undistorted_norm.reshape(-1, 1, 2).astype(np.float32)
    print("Undistortion successful.")

    print("Running solvePnP on undistorted points...")
    # Ensure objp is float32 for solvePnP
    objp_float32 = objp.astype(np.float32)
    # corners_undistorted is already float32 now

    # Pass K as float32 to solvePnP (it's generally more flexible than fisheye)
    K_pnp = K.astype(np.float32)

    print(f"Debug Check -> Type of objp for solvePnP: {objp_float32.dtype}")
    print(f"Debug Check -> Type of corners for solvePnP: {corners_undistorted.dtype}")
    print(f"Debug Check -> Type of K for solvePnP: {K_pnp.dtype}")

    # Pass None for distortion as points are undistorted
    ret, rvec_cam_world, tvec_cam_world = cv2.solvePnP(objp_float32, corners_undistorted, K_pnp, None, flags=cv2.SOLVEPNP_ITERATIVE)

    if not ret:
        print("Error: solvePnP failed to compute camera pose relative to world.")
        sys.exit(1)

    # Convert to Camera Pose in World Frame (T_world_cam)
    R_cam_world, _ = cv2.Rodrigues(rvec_cam_world)
    R_world_cam = R_cam_world.T
    t_world_cam = -R_world_cam @ tvec_cam_world

    T_world_cam = compose_T(R_world_cam, t_world_cam)

    print("Calculated T_world_cam (Camera Pose in World Frame, mm):")
    np.set_printoptions(suppress=True, precision=4)
    print(T_world_cam)
    np.set_printoptions()

except cv2.error as e:
    # Update error handling to show the types passed during the error
    print(f"\n!!! OpenCV Error during solvePnP/undistortion !!!")
    print(f"    Error Code: {e.code}")
    print(f"    Function:   {e.func}")
    print(f"    Message:    {e.err}")
    print(f"    Line:       {e.line}")
    print(f"    Input types at time of error (approx):")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during T_world_cam calculation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

T_world_cam_rdf = T_world_cam.copy()
T_world_cam[0:3, 1] = T_world_cam_rdf[0:3, 0]
T_world_cam[0:3, 2] = T_world_cam_rdf[0:3, 1]
T_world_cam[0:3, 0] = T_world_cam_rdf[0:3, 2]





# Corrected R_world_body based on World Y-Down and Body Z-Down
# Columns are Body axes [X_b, Y_b, Z_b] expressed in World coords
R_world_body = np.array([
    [0.0, 1.0, 0.0],  # World X = Body Y
    [0.0, 0.0, 1.0],  # World Y = Body Z
    [1.0, 0.0, 0.0]   # World Z = Body X
])

# pattern width and height in mm
pattern_width_mm = (args.cols - 1) * spacing_mm + 0.5 * spacing_mm
pattern_height_mm = (args.rows - 1) * spacing_mm

pattern_distance_floor_mm = args.pattern_height * 1000.0 # Convert to mm

ground_ref_point_world = np.array([pattern_width_mm / 2.0, pattern_height_mm + pattern_distance_floor_mm, 0.0])

drone_wall_dist_mm = args.drone_wall_dist * 1000.0 # Convert to mm
# Calculate the front center point in world coordinates
front_center_world = ground_ref_point_world + np.array([0.0, 0.0, -drone_wall_dist_mm])

front_center_to_imu_mm_world = R_world_body @ front_center_to_imu_mm.reshape(3, 1)

print(front_center_to_imu_mm_world)

# Calculate the IMU position in world coordinates
imu_world = front_center_world + front_center_to_imu_mm_world.flatten()

T_world_imu = compose_T(R_world_body, imu_world.reshape(3, 1))


# estimate the transformation from camera to IMU
T_cam_imu = invert_T(T_world_cam) @ T_world_imu
print("\nEstimated T_cam_imu (Camera Pose in IMU Frame):")
np.set_printoptions(suppress=True, precision=4)
print(T_cam_imu)
np.set_printoptions()
# Save the extrinsic parameters
extrinsic_data = {
    'T_cam_imu': T_cam_imu,
    'T_world_cam': T_world_cam,
    'T_world_imu': T_world_imu,
    'ground_ref_point_world': ground_ref_point_world,
    'front_center_world': front_center_world
}
np.savez(args.output, **extrinsic_data)
print(f"Extrinsic parameters saved to {args.output}")

print("\n--- Euler Angles for IMU-to-Camera Rotation (R_cam_body) ---")
print("Convention: ZYX Intrinsic (Yaw, Pitch, Roll applied in this order)")


R_cam_imu = T_cam_imu[0:3, 0:3]
t_cam_imu = T_cam_imu[0:3, 3].reshape(3, 1)
try:
    # Convert R_cam_body to Euler angles
    yaw_rad, pitch_rad, roll_rad = rotation_matrix_to_euler_zyx(R_cam_imu.T)

    # Convert to degrees for easier interpretation
    yaw_deg = math.degrees(yaw_rad)
    pitch_deg = math.degrees(pitch_rad)
    roll_deg = math.degrees(roll_rad)

    print(f"Roll  (X-axis rotation): {roll_deg:.4f} degrees")
    print(f"Pitch (Y-axis rotation): {pitch_deg:.4f} degrees")
    print(f"Yaw   (Z-axis rotation): {yaw_deg:.4f} degrees")

    # --- Print Corresponding Translation ---
    # This is the IMU origin's position in Camera coordinates
    print("\n--- Translation for IMU-to-Camera (t_cam_body) ---")
    print("Represents: IMU Origin Position in Camera Coordinates")
    print(f"Translation Vector (mm): {(-t_cam_imu).flatten()}")
    # Optionally print in meters if that's what the target system expects
    # print(f"Translation Vector (m):  {t_cam_body.flatten() / 1000.0}")

except Exception as e:
    print(f"Error converting rotation matrix to Euler angles: {e}")
    # Set angles to NaN or None if conversion fails
    roll_deg, pitch_deg, yaw_deg = float('nan'), float('nan'), float('nan')


# --- 3D Visualization ---
plot_3d_setup(objp, ground_ref_point_world, front_center_world, T_world_imu, T_world_cam)


print("\nCalibration process finished.")