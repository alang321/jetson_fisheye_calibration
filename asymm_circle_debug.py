import cv2
import numpy as np
import os
import glob
import argparse
import sys
from scipy.spatial import distance # For efficient nearest neighbor searches
import math # For checking NaN

# --- Replace previous manual/corner functions with this ---

def find_nearest_available(target_pt, available_coords, available_indices, max_dist_sq=np.inf):
    """Finds the nearest point in available_coords to target_pt."""
    if not available_indices:
        return -1, np.inf # No points available

    # Calculate squared distances (faster than sqrt)
    dist_sq = distance.cdist([target_pt], available_coords)[0]**2

    best_dist_sq = np.inf
    best_idx_in_available = -1

    # Find the minimum distance within the threshold
    valid_indices = np.where(dist_sq < max_dist_sq)[0]
    if len(valid_indices) > 0:
         min_local_idx = np.argmin(dist_sq[valid_indices]) # Index within the valid subset
         best_idx_in_available = valid_indices[min_local_idx] # Index within the 'available_coords' array
         best_dist_sq = dist_sq[best_idx_in_available]
    else:
         return -1, np.inf # No point found within max_dist_sq


    # Map back to the original index from detected_keypoints
    original_kp_index = available_indices[best_idx_in_available]
    return original_kp_index, best_dist_sq


# --- Replace the run_two_point_assisted_matching function with this ---

# --- Replace previous manual/corner functions with this ---

def run_four_corner_assisted_nn_walk(image_to_select_on, detected_keypoints, objp, pattern_size):
    """
    Uses 4 corner clicks (TL, TR, BL, BR) and then nearest-neighbor walking
    based on the ideal grid structure to order detected blobs.

    Args:
        image_to_select_on: Color image.
        detected_keypoints: List of cv2.KeyPoint found by SimpleBlobDetector.
        objp: The (N, 3) array of ideal object points.
        pattern_size: Tuple (cols, rows).

    Returns:
        A numpy array of shape (num_points, 1, 2) float32 corners, or None.
    """
    cols, rows = pattern_size
    num_expected_points = cols * rows
    print("\n--- Four-Corner + NN Walk Assisted Matching ---")
    print("Detected Blobs:", len(detected_keypoints))
    if not detected_keypoints or len(detected_keypoints) < 4:
         print("  Error: Not enough blobs detected (< 4).")
         return None
    if len(detected_keypoints) < num_expected_points * 0.8:
         print(f"  Warning: Significantly fewer blobs detected ({len(detected_keypoints)}) than expected ({num_expected_points}). Matching likely to fail.")
         # Optional: return None here if too few blobs are detected

    # --- 1. Get 4 Corner Clicks (Reuse code from previous 4-corner method) ---
    img_display = image_to_select_on.copy()
    kp_coords = np.array([kp.pt for kp in detected_keypoints], dtype=np.float32)
    for i, pt in enumerate(kp_coords):
        pt_int = tuple(pt.astype(int))
        cv2.circle(img_display, pt_int, 4, (0, 165, 255), 1)

    window_name = "Click 4 Corners (TL, TR, BL, BR)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 700)

    corner_labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
    ideal_corner_indices = [0, cols - 1, (rows - 1) * cols, num_expected_points - 1]
    label_to_ideal_idx = dict(zip(corner_labels, ideal_corner_indices))
    clicked_kp_indices = {}
    current_click_idx = 0

    # (Keep the four_corner_click_callback function exactly as in the previous 4-corner implementation)
    def four_corner_click_callback(event, x, y, flags, param):
        nonlocal current_click_idx # Modify outer scope variable
        if current_click_idx >= len(corner_labels): return
        target_label = corner_labels[current_click_idx]
        if event == cv2.EVENT_LBUTTONDOWN:
            distances = distance.cdist([(x, y)], kp_coords)[0]
            nearest_kp_idx = np.argmin(distances)
            if nearest_kp_idx in clicked_kp_indices.values():
                 print(f"  Warning: Blob {nearest_kp_idx} already selected. Click a different blob for {target_label}.")
                 return
            clicked_kp_indices[target_label] = nearest_kp_idx
            pt_clicked_coords = kp_coords[nearest_kp_idx]
            print(f"  Clicked {target_label} near blob {nearest_kp_idx} at ({pt_clicked_coords[0]:.0f}, {pt_clicked_coords[1]:.0f})")
            # --- Redraw feedback ---
            temp_display = image_to_select_on.copy()
            for i, pt in enumerate(kp_coords): pt_int = tuple(pt.astype(int)); cv2.circle(temp_display, pt_int, 4, (0, 165, 255), 1)
            corner_colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
            for i, lbl in enumerate(corner_labels):
                if lbl in clicked_kp_indices:
                    kp_idx = clicked_kp_indices[lbl]; pt = kp_coords[kp_idx]; pt_int = tuple(pt.astype(int))
                    cv2.circle(temp_display, pt_int, 6, corner_colors[i], 2)
                    text_org = (pt_int[0] + 8, pt_int[1] + 8)
                    cv2.putText(temp_display, lbl[:2], text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            img_display[:] = temp_display[:]
            # --- End Redraw ---
            current_click_idx += 1
            if current_click_idx < len(corner_labels): print(f"-> Now click: {corner_labels[current_click_idx]}")
            else: print("-> All 4 corners clicked. Press 'y' to proceed, 'r' to restart, 'q' to quit.")

    cv2.setMouseCallback(window_name, four_corner_click_callback)
    print(f"-> Click the blob for: {corner_labels[current_click_idx]}")

    # (Keep the User Interaction Loop exactly as in the previous 4-corner implementation)
    while True:
        cv2.imshow(window_name, img_display)
        key = cv2.waitKey(20) & 0xFF
        if current_click_idx == len(corner_labels) and key == ord('y'): break
        if key == ord('r'):
             print("Restarting corner selection..."); current_click_idx = 0; clicked_kp_indices = {}
             img_display = image_to_select_on.copy()
             for i, pt in enumerate(kp_coords): pt_int = tuple(pt.astype(int)); cv2.circle(img_display, pt_int, 4, (0, 165, 255), 1)
             print(f"-> Click the blob for: {corner_labels[current_click_idx]}")
        if key == ord('q'): print("Quit during corner selection."); cv2.destroyWindow(window_name); return None
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: return None
    cv2.destroyWindow(window_name)

    # --- 2. Initialize for NN Walk ---
    print("Initializing Nearest-Neighbor Walk...")
    if len(clicked_kp_indices) != 4: print("  Error: Did not collect 4 corner points."); return None

    # Final corners array (stores image coordinates)
    final_corners = np.full((num_expected_points, 1, 2), np.nan, dtype=np.float32) # Init with NaN
    # List of indices of available keypoints (blobs)
    available_indices = list(range(len(kp_coords)))
    # Corresponding coordinates (for distance calculation) - keep as numpy array
    available_coords = kp_coords.copy()

    matched_count = 0

    # Assign the 4 known corners and remove them from the available pool
    indices_to_remove_from_available = []
    coords_to_remove_indices = [] # Indices within the *current* available_coords array
    
    for label in corner_labels:
        grid_idx = label_to_ideal_idx[label]
        kp_idx = clicked_kp_indices[label] # Index in original kp_coords
        
        final_corners[grid_idx, 0, :] = kp_coords[kp_idx]
        matched_count += 1
        
        # Find where this kp_idx currently is in the available_indices list
        try:
             list_idx = available_indices.index(kp_idx)
             # Mark for removal (don't modify list while iterating indirectly)
             indices_to_remove_from_available.append(list_idx)
        except ValueError:
             print(f"Warning: Corner blob index {kp_idx} already removed? Skipping.")


    # Remove assigned corners from available pool (in reverse index order to avoid shifting issues)
    indices_to_remove_from_available.sort(reverse=True)
    for list_idx in indices_to_remove_from_available:
         available_indices.pop(list_idx)
    available_coords = np.delete(available_coords, indices_to_remove_from_available, axis=0)


    # --- 3. Estimate Scale and Search Distance ---
    # Get pixel coords of corners
    tl_px = final_corners[label_to_ideal_idx["Top-Left"], 0, :]
    tr_px = final_corners[label_to_ideal_idx["Top-Right"], 0, :]
    bl_px = final_corners[label_to_ideal_idx["Bottom-Left"], 0, :]

    # Estimate pixel distance for one unit of object spacing
    # Use top edge for horizontal, left edge for vertical
    obj_spacing_x = np.linalg.norm(objp[1,:2] - objp[0,:2]) # Should be 'spacing' param value
    obj_spacing_y = np.linalg.norm(objp[cols,:2] - objp[0,:2]) # Vertical step

    pixel_dist_x = np.linalg.norm(tr_px - tl_px) / (cols - 1) if cols > 1 else 50.0 # Avg px per horizontal step
    pixel_dist_y = np.linalg.norm(bl_px - tl_px) / (rows - 1) if rows > 1 else 50.0 # Avg px per vertical step

    # Average scale (pixels per object unit) - use separate scales might be better
    scale_x = pixel_dist_x / obj_spacing_x if obj_spacing_x > 1e-6 else 1.0
    scale_y = pixel_dist_y / obj_spacing_y if obj_spacing_y > 1e-6 else 1.0
    scale_factor = np.array([scale_x, scale_y])

    # Max distance for nearest neighbor search (squared)
    # Allow searching within ~75% of the estimated average pixel spacing
    avg_pixel_spacing = (pixel_dist_x + pixel_dist_y) / 2.0
    max_dist_sq = (avg_pixel_spacing * 0.75)**2
    print(f"  Estimated Avg Px Spacing H:{pixel_dist_x:.1f}, V:{pixel_dist_y:.1f}. Max Search Dist Sq: {max_dist_sq:.1f}")

    # --- 4. Perform the Grid Walk ---
    print("  Starting grid walk...")
    walk_order = [(r, c) for r in range(rows) for c in range(cols)] # Standard row-major order

    for r, c in walk_order:
        grid_idx = r * cols + c

        # Skip if already filled (e.g., the corners)
        if not np.isnan(final_corners[grid_idx, 0, 0]):
            continue

        # Find a valid, already-filled neighbor to predict from
        # Priority: Left (c-1), then Above (r-1)
        predictor_grid_idx = -1
        vec_objp_step = None

        # Try predicting from Left
        if c > 0:
            prev_idx = grid_idx - 1
            if not np.isnan(final_corners[prev_idx, 0, 0]):
                predictor_grid_idx = prev_idx
                vec_objp_step = objp[grid_idx] - objp[predictor_grid_idx]

        # Try predicting from Above (if left didn't work or wasn't possible)
        if predictor_grid_idx == -1 and r > 0:
            prev_idx = grid_idx - cols
            if not np.isnan(final_corners[prev_idx, 0, 0]):
                predictor_grid_idx = prev_idx
                vec_objp_step = objp[grid_idx] - objp[predictor_grid_idx]

        # If we couldn't find a predictor, we cannot proceed for this point
        if predictor_grid_idx == -1:
            print(f"  Error: Cannot find valid neighbor to predict from for grid point ({r},{c}) index {grid_idx}")
            continue # Skip to next point in walk_order, leave this one as NaN

        # Predict the image location
        predictor_img_pt = final_corners[predictor_grid_idx, 0, :]
        # Scale the object space vector by our estimated scale factor
        predicted_pixel_step = vec_objp_step[:2] * scale_factor
        pred_img_pt = predictor_img_pt + predicted_pixel_step

        # Find the nearest *available* blob to the prediction
        if not available_indices:
            print(f"  Error: Ran out of available blobs before reaching grid point ({r},{c})")
            break # Stop the walk

        matched_kp_idx, dist_sq = find_nearest_available(
            pred_img_pt, available_coords, available_indices, max_dist_sq
        )

        # If a match was found within the threshold
        if matched_kp_idx != -1:
            # Assign the matched point's coordinates
            final_corners[grid_idx, 0, :] = kp_coords[matched_kp_idx]
            matched_count += 1

            # Remove the matched point from the available pool
            try:
                idx_in_available = available_indices.index(matched_kp_idx)
                available_indices.pop(idx_in_available)
                available_coords = np.delete(available_coords, idx_in_available, axis=0)
            except ValueError:
                 print(f"Error: Matched kp index {matched_kp_idx} not found in available list?")


        else:
            print(f"  Warning: No suitable match found for grid point ({r},{c}) index {grid_idx} near {pred_img_pt.round(1)}")
            # Leave final_corners[grid_idx] as NaN

    # --- 5. Final Check ---
    print(f"  Finished grid walk. Matched {matched_count}/{num_expected_points} points.")
    if matched_count != num_expected_points or np.isnan(final_corners).any():
        print("  Warning: Failed to match all points cleanly. Result is incomplete/incorrect.")
        # Optionally visualize the partial result for debugging
        # vis_partial(image_to_select_on, final_corners, kp_coords)
        return None

    print("  Successfully matched all points via NN walk.")
    return final_corners

# --- Helper function find_nearest_available (already defined in Method 2, ensure it's present) ---
# (Make sure the find_nearest_available function from the previous implementation is still included
#  somewhere before this new function, or copy it here if needed)
# def find_nearest_available(target_pt, available_coords, available_indices, max_dist_sq=np.inf):
#    ... (implementation as before) ...

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
parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode: visualize preprocessing and blob detection.')
args = parser.parse_args()

# --- Setup ---
image_dir = args.dir
grid_cols = args.cols
grid_rows = args.rows
spacing = args.spacing
output_file = args.output
image_extensions = args.ext
DEBUG_MODE = args.debug

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

    # Display the visualization
    cv2.imshow("Object Point Layout (Calculated)", vis_image)
    print("-> Displaying calculated object point layout. Press any key in the window to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow("Object Point Layout (Calculated)") # Close only this window
else:
    print("  Warning: objp array is empty, cannot visualize layout.")


# Optional: Print first few points to verify
# print("Sample Object Points (real-world coords, e.g., mm):")
# for i in range(min(5, len(objp))):
#    print(f"  Point {i}: {objp[i]}")


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
if not DEBUG_MODE:
    print("Press 'y' to accept detection, 'n' to reject, 'q' to quit during final view.")

if DEBUG_MODE:
    print("--- DEBUG MODE ENABLED ---")
    print("Debug View Controls:")
    print("  'm': Switch Preprocessing (None -> Manual -> Adaptive -> CLAHE)")
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
    if DEBUG_MODE:
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
            if debug_preprocess_mode == 1: # Manual Threshold
                _, processed_gray_dbg = cv2.threshold(gray, manual_thresh_value, 255, debug_thresh_type)
                preprocess_desc = f"Manual Th: {manual_thresh_value}, T:{'BIN' if debug_thresh_type==cv2.THRESH_BINARY else 'INV'}"
            elif debug_preprocess_mode == 2: # Adaptive Threshold
                 # Block size must be odd and >= 3
                 # Link trackbar crudely for block size adjustment (example)
                 block_size = 11 + 2 * (manual_thresh_value // 32)
                 block_size = max(3, block_size | 1) # Ensure odd >= 3
                 C = 2 # Constant subtracted
                 processed_gray_dbg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    debug_thresh_type, block_size, C)
                 preprocess_desc = f"Adaptive G Th: B:{block_size}, C:{C}, T:{'BIN' if debug_thresh_type==cv2.THRESH_BINARY else 'INV'}"
            elif debug_preprocess_mode == 3: # CLAHE
                processed_gray_dbg = clahe.apply(gray)
                preprocess_desc = "CLAHE Contrast"
            else: # Mode 0: None (Use original grayscale)
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

    # --- End of DEBUG_MODE block ---

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
            corners_manual = run_four_corner_assisted_nn_walk(img_color, keypoints, objp, pattern_size)
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
                objpoints.append(objp)
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

# Clean up debug windows if they were opened
if DEBUG_MODE:
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

# Prepare for fisheye calibration
# Initialize K and D matrices (will be estimated)
K_init = np.eye(3)
D_init = np.zeros((4, 1)) # Fisheye model uses 4 coefficients (k1, k2, k3, k4)

# Calibration flags
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
# Consider adding: cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT if center is known reliably
# Consider adding: cv2.fisheye.CALIB_FIX_K[1,2,3,4] if optimization is unstable

# NOTE: cv2.fisheye.calibrate requires image_size as (WIDTH, HEIGHT).
image_size_wh = (img_shape[1], img_shape[0])

# Termination criteria for the optimization process
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

try:
    # IMPORTANT: OpenCV expects lists of Numpy arrays for points
    # Ensure objpoints is List[Nx3 float32], imgpoints is List[Nx1x2 float32]
    # Our current lists should match this format.

    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,      # List of (N, 3) objp arrays
        imgpoints,      # List of (N, 1, 2) corner arrays
        image_size_wh,  # Image size (width, height)
        K_init,         # Initial guess for K
        D_init,         # Initial guess for D (zeros)
        flags=calib_flags,
        criteria=criteria
    )
except cv2.error as e:
     print(f"\n!!! OpenCV Error during calibration: {e}")
     print("This might be due to insufficient points, very poor detections (high initial error),")
     print("numerical instability, or potentially incorrect object points (check grid shape and spacing).")
     print("Ensure views have sufficient variation and accurate detections.")
     sys.exit(1)
except Exception as e:
     print(f"\n!!! An unexpected error occurred during calibration: {e}")
     import traceback
     traceback.print_exc()
     sys.exit(1)

# --- Results ---
if ret:
    print("\nCalibration successful!")
    print(f"  RMS reprojection error: {ret:.4f} pixels")
    print("\nCamera Matrix (K):")
    print(K)
    print("\nDistortion Coefficients (D) [k1, k2, k3, k4]:")
    print(D.flatten()) # D is usually returned as a column vector

    # --- Save Results ---
    print(f"\nSaving calibration data to: {output_file}")
    try:
        # Save K, D, image shape, RMS error, and optionally the points used
        np.savez(output_file, K=K, D=D, img_shape=img_shape, rms=ret,
                 objpoints=np.array(objpoints, dtype=object), # Save points if needed later
                 imgpoints=np.array(imgpoints, dtype=object),
                 filenames=np.array(processed_image_filenames)) # Save corresponding filenames
        print("Data saved.")
    except Exception as e:
        print(f"Error saving data to {output_file}: {e}")


    # --- Optional: Calculate Reprojection Error Manually (for verification) ---
    print("\nCalculating reprojection errors per image...")
    total_error = 0
    per_view_errors = []
    if len(objpoints) > 0 and len(rvecs) == len(objpoints) and len(tvecs) == len(objpoints):
        for i in range(len(objpoints)):
            try:
                # Project object points back into image plane using calibrated parameters
                imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
                # Calculate error (Euclidean distance between detected and reprojected points)
                # Use NORM_L2 per point, then average
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                per_view_errors.append(error)
                total_error += error
                # print(f"  Image {i+1} ({os.path.basename(processed_image_filenames[i])}): {error:.4f} pixels")
            except cv2.error as proj_err:
                print(f"  Warning: Could not project points for image {i}. Error: {proj_err}")
                per_view_errors.append(np.inf) # Indicate an error for this view

        mean_error = total_error / len(objpoints)
        print(f"\nAverage reprojection error (calculated manually): {mean_error:.4f} pixels")
        # You could analyze per_view_errors further to identify bad images
    else:
         print("\nSkipping manual reprojection error calculation (missing points or extrinsics).")


    # --- Visualize Undistortion (Optional but Recommended) ---
    print("\nVisualizing undistortion on the first accepted sample image...")
    if accepted_count > 0 and processed_image_filenames:
        first_accepted_img_path = processed_image_filenames[0]
        img_distorted = cv2.imread(first_accepted_img_path)

        if img_distorted is not None:
            h, w = img_distorted.shape[:2]
            # Option 1: Use fisheye.undistortImage (simpler, may crop)
            # Knew = K.copy() # Use original K, or estimate new one
            # You might want to compute an optimal new camera matrix if you want to scale/crop less aggressively
            balance = 0.0 # 0: crops significantly to remove black areas, 1: keeps all pixels, showing black areas
            Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, image_size_wh, np.eye(3), balance=balance)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, image_size_wh, cv2.CV_16SC2)
            img_undistorted = cv2.remap(img_distorted, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # Option 2: Direct undistortion (simpler call, behavior similar to balance=0?)
            # img_undistorted = cv2.fisheye.undistortImage(img_distorted, K, D, Knew=K)

            # Display side-by-side
            vis_compare = np.hstack((img_distorted, img_undistorted))
            cv2.namedWindow('Distorted vs Undistorted', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Distorted vs Undistorted', 1200, 600) # Adjust size
            cv2.imshow('Distorted vs Undistorted', vis_compare)
            print(f"Displaying undistortion result for {os.path.basename(first_accepted_img_path)}. Press any key to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Warning: Could not reload sample image {first_accepted_img_path} for visualization.")
    else:
         print("No valid images were accepted, cannot visualize undistortion.")

else:
    print("\nCalibration failed. The optimization could not converge.")
    print("Possible reasons include:")
    print("- Poor quality detections (high initial reprojection error).")
    print("- Insufficient number of views or lack of variation in views.")
    print("- Incorrect grid parameters (--cols, --rows, --spacing).")
    print("- Numerical instability (check CALIB_CHECK_COND flag was used).")

print("\nCalibration process finished.")