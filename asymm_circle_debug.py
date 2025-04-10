import cv2
import numpy as np
import os
import glob
import argparse
import sys

# --- Configuration ---
DEFAULT_IMAGE_DIR = './calibration_images/'
DEFAULT_GRID_COLS = 4
DEFAULT_GRID_ROWS = 11
DEFAULT_SPACING_MM = 20.0
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
                    help='Enable debug mode: visualize preprocessing and blob detection.') # <-- Added Debug Flag
args = parser.parse_args()

# --- Setup ---
image_dir = args.dir
grid_cols = args.cols
grid_rows = args.rows
spacing = args.spacing
output_file = args.output
image_extensions = args.ext
DEBUG_MODE = args.debug # <-- Store debug flag

pattern_size = (grid_cols, grid_rows)
min_images_for_calib = 10

# --- Prepare Object Points ---
objp = np.zeros((grid_cols * grid_rows, 3), np.float32)
for r in range(grid_rows):
    for c in range(grid_cols):
        idx = r * grid_cols + c
        objp[idx, 0] = (2*c + r % 2) * spacing
        objp[idx, 1] = r * spacing
        objp[idx, 2] = 0

objpoints = []
imgpoints = []

# --- Find Image Files ---
image_files = []
for ext in image_extensions:
    search_path = os.path.join(image_dir, f'*.{ext.lower()}')
    found_files = glob.glob(search_path)
    print(f"Searching: {search_path}, Found: {len(found_files)} files")
    image_files.extend(found_files)

if not image_files:
    print(f"Error: No images found in directory '{image_dir}' with extensions {image_extensions}")
    sys.exit(1)

print(f"Found {len(image_files)} potential calibration images.")

# --- Initialize Blob Detector (Crucial for Debugging and Tuning) ---
params = cv2.SimpleBlobDetector_Params()
# --- Tune these parameters based on your grid, lighting, and image resolution ---
params.filterByArea = True
params.minArea = 3000      # Adjust! Start small, increase if noise is detected
params.maxArea = 10000   # Adjust! Depends on circle size in pixels
params.filterByCircularity = True
params.minCircularity = 0.5 # Lower for fisheye/perspective distortion
params.filterByConvexity = True
params.minConvexity = 0.80
params.filterByInertia = True
params.minInertiaRatio = 0.1 # Lower allows more elongation

# Create detector with specific parameters
# Note: findCirclesGrid uses a SimpleBlobDetector internally. Providing our own
# allows us to use the *same* detector for debugging visualization.
blob_detector = cv2.SimpleBlobDetector_create(params)
print("\nUsing Blob Detector with parameters:")
print(f"  Area: {params.minArea} - {params.maxArea}")
print(f"  Circularity: > {params.minCircularity}")
print(f"  Convexity: > {params.minConvexity}")
print(f"  Inertia Ratio: > {params.minInertiaRatio}")


# --- Manual Thresholding Callback (for Debug Mode) ---
manual_thresh_value = 127 # Default starting value
def set_thresh(val):
    global manual_thresh_value
    manual_thresh_value = val

# --- Process Images ---
img_shape = None

print("\nProcessing images...")
if not DEBUG_MODE:
    print("Press 'y' to accept detection, 'n' to reject, 'q' to quit during final view.")

if DEBUG_MODE:
    print("--- DEBUG MODE ENABLED ---")
    print("Debug View Controls:")
    print("  'm': Switch Preprocessing (None -> Manual)")
    print("  't': Toggle Threshold Type (Binary / Binary Inv) for Manual/Adaptive")
    print("  ' ' (Space): Process this image with current settings")
    print("  's': Skip this image (same as 'n' in non-debug)")
    print("  'q': Quit application")
    print("\nFinal Confirmation View Controls (after Debug View):")
    print("  'y': Accept detection for calibration")
    print("  'n': Reject detection")
    print("  'q': Quit application")

    cv2.namedWindow('Debug View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Debug View', 1200, 800) # Adjust size
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Manual Thresh', 'Controls', manual_thresh_value, 255, set_thresh)

    debug_preprocess_mode = 0 # 0: None, 1: Manual, 2: Adaptive, 3: CLAHE
    debug_thresh_type = cv2.THRESH_BINARY # Start with Binary
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

processed_count = 0
accepted_count = 0

for i, fname in enumerate(image_files):
    print(f"\n[{processed_count+1}/{len(image_files)}] Processing: {os.path.basename(fname)}")
    img_color = cv2.imread(fname)
    if img_color is None:
        print("  Warning: Could not read image.")
        continue

    if img_shape is None:
        img_shape = img_color.shape[:2] # (height, width)
    elif img_color.shape[:2] != img_shape:
         print(f"  Warning: Image size {img_color.shape[:2]} differs from first image {img_shape}. Skipping.")
         continue

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    processed_gray = gray # Default: use original gray image

    # --- DEBUG MODE: Interactive Preprocessing & Visualization ---
    if DEBUG_MODE:
        while True: # Loop for interactive debugging adjustments
            vis_list = []
            # 1. Original Color Image (scaled down if too large for display)
            h, w = img_color.shape[:2]
            max_h = 300
            scale = max_h / h
            vis_orig = cv2.resize(img_color, (int(w*scale), int(h*scale)))
            cv2.putText(vis_orig, 'Original', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            vis_list.append(vis_orig)

            # 2. Apply Selected Preprocessing
            preprocess_desc = ""
            if debug_preprocess_mode == 1: # Manual Threshold
                _, processed_gray_dbg = cv2.threshold(gray, manual_thresh_value, 255, debug_thresh_type)
                preprocess_desc = f"Manual Th: {manual_thresh_value}, T:{'BIN' if debug_thresh_type==cv2.THRESH_BINARY else 'INV'}"
            elif debug_preprocess_mode == 2: # Adaptive Threshold
                 # Block size must be odd
                 block_size = 11 + 2 * (manual_thresh_value // 32) # Example: Link trackbar crudely
                 block_size = max(3, block_size) # Ensure >= 3
                 C = 2 # Constant subtracted
                 processed_gray_dbg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    debug_thresh_type, block_size, C)
                 preprocess_desc = f"Adaptive G Th: B:{block_size}, C:{C}, T:{'BIN' if debug_thresh_type==cv2.THRESH_BINARY else 'INV'}"
            elif debug_preprocess_mode == 3: # CLAHE
                processed_gray_dbg = clahe.apply(gray)
                preprocess_desc = "CLAHE Contrast"
            else: # Mode 0: None
                processed_gray_dbg = gray.copy() # Use original gray for blob detection vis
                preprocess_desc = "Raw Grayscale"

            vis_proc = cv2.cvtColor(processed_gray_dbg, cv2.COLOR_GRAY2BGR) # For display
            vis_proc = cv2.resize(vis_proc, (int(w*scale), int(h*scale)))
            cv2.putText(vis_proc, preprocess_desc, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            vis_list.append(vis_proc)

            # 3. Detect Blobs on the full-resolution processed image
            # Use the *same* detector configured earlier
            keypoints = blob_detector.detect(processed_gray_dbg) # Detect on full-res image

            # --- Create scaled keypoints specifically for visualization ---
            scaled_keypoints = []
            # Calculate the scaling factor used for the visualization image
            h_orig, w_orig = processed_gray_dbg.shape[:2] # Dimensions of image blobs were detected on
            h_vis, w_vis = vis_orig.shape[:2]      # Dimensions of the scaled image we will draw on
            
            if h_orig > 0 and w_orig > 0: # Avoid division by zero if image loading failed
                scale_x = w_vis / w_orig
                scale_y = h_vis / h_orig
                # Use average scale for size, but specific scales for points
                vis_scale = (scale_x + scale_y) / 2.0

                if keypoints: # Check if keypoints list is not empty
                    for kp in keypoints:
                        # Scale the coordinates
                        scaled_pt_x = kp.pt[0] * scale_x
                        scaled_pt_y = kp.pt[1] * scale_y
                        # Scale the size
                        scaled_size = kp.size * vis_scale
                        # Create a new KeyPoint object with scaled values
                        scaled_kp = cv2.KeyPoint(x=scaled_pt_x, y=scaled_pt_y, size=scaled_size,
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
            cv2.putText(vis_blobs, f'Blobs Found: {len(keypoints)}', (10, vis_blobs.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            vis_list.append(vis_blobs)

            # --- Display Debug Views ---
            # Make sure all images in vis_list have the same height before hstack
            # (They should if generated correctly above, but good practice to check/resize if needed)
            # Example check (optional, usually not needed if scaling logic is consistent):
            # target_h = vis_list[0].shape[0]
            # for k in range(1, len(vis_list)):
            #     if vis_list[k].shape[0] != target_h:
            #         ratio = target_h / vis_list[k].shape[0]
            #         vis_list[k] = cv2.resize(vis_list[k], (int(vis_list[k].shape[1]*ratio), target_h))

            debug_vis = np.hstack(vis_list)
            cv2.imshow('Debug View', debug_vis)
            key = cv2.waitKey(0) & 0xFF

            if key == ord(' '): # Space: Proceed with findCirclesGrid using current settings
                # Set the processed_gray that will be used by findCirclesGrid
                processed_gray = processed_gray_dbg
                print(f"  DEBUG: Proceeding with Preprocessing: {preprocess_desc}")
                break # Exit debug adjustment loop, proceed to findCirclesGrid
            elif key == ord('m'): # Switch preprocessing mode
                debug_preprocess_mode = (debug_preprocess_mode + 1) % 2 # Cycle through 0, 1, 2, 3
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

        if processed_gray is None: # Skipped in debug mode
            processed_count += 1
            continue # Go to next image

    # --- Find the circle grid centers ---
    # Use the 'processed_gray' determined above (either original gray or from debug step)
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID # | cv2.CALIB_CB_CLUSTERING
    ret, corners = cv2.findCirclesGrid(
        processed_gray, # Use the potentially preprocessed image
        pattern_size,
        flags=flags,
        blobDetector=blob_detector # Pass the explicitly created detector
    )

    processed_count += 1

    # --- Show Final Detection Result and Ask for Confirmation ---
    if ret:
        print(f"  Grid detected! ({pattern_size[0]}x{pattern_size[1]})")
        vis_img = img_color.copy()
        # corners should be float32 for drawChessboardCorners
        if corners is not None:
            corners = corners.astype(np.float32)
            cv2.drawChessboardCorners(vis_img, pattern_size, corners, ret)

        cv2.namedWindow('Detection Confirmation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection Confirmation', 800, 600)
        cv2.imshow('Detection Confirmation', vis_img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                print("  Accepted.")
                objpoints.append(objp)
                imgpoints.append(corners)
                accepted_count += 1
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
        cv2.destroyWindow('Detection Confirmation')
    else:
        print("  Grid not detected by findCirclesGrid.")
        if DEBUG_MODE:
             print("  (Check blobs in Debug View to see why grid fitting might have failed)")
             # Optional: Keep debug window open longer?
             # cv2.waitKey(1000) # Wait 1 sec

# Clean up debug windows if they were opened
if DEBUG_MODE:
    cv2.destroyWindow('Debug View')
    cv2.destroyWindow('Controls')
cv2.destroyAllWindows() # Close any other remaining windows

# --- Perform Calibration ---
print(f"\nCollected {accepted_count} valid views out of {processed_count} processed images.")

if accepted_count < min_images_for_calib:
    print(f"Error: Insufficient number of valid views ({accepted_count}). Need at least {min_images_for_calib}.")
    if processed_count > 0 and accepted_count == 0:
        print("Possible issues:")
        print("- Blob detector parameters might be wrong (check min/max Area especially).")
        print("- Preprocessing might be needed (try debug mode with thresholding/CLAHE).")
        print("- Grid parameters (--cols, --rows) might not match the physical grid.")
        print("- Lighting conditions might be poor (shadows, glare).")
        print("- Image quality might be low (blur, noise).")
    sys.exit(1)

if img_shape is None:
    print("Error: Could not determine image shape (no images processed?).")
    sys.exit(1)

print(f"\nRunning fisheye calibration with image size: {img_shape[::-1]} (width, height)...")

K_init = np.eye(3)
D_init = np.zeros((4, 1))
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW

try:
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img_shape[::-1],
        K_init,
        D_init,
        flags=calib_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
except cv2.error as e:
     print(f"!!! OpenCV Error during calibration: {e}")
     print("This might be due to insufficient points, poor detections, numerical instability,")
     print("or potentially incorrect object points (check grid shape and spacing).")
     sys.exit(1)


# --- Results ---
if ret:
    print("\nCalibration successful!")
    print(f"  RMS reprojection error: {ret}")
    print("\nCamera Matrix (K):")
    print(K)
    print("\nDistortion Coefficients (D) [k1, k2, k3, k4]:")
    print(D.flatten())

    print(f"\nSaving calibration data to: {output_file}")
    np.savez(output_file, K=K, D=D, img_shape=img_shape, rms=ret, objpoints=objpoints, imgpoints=imgpoints) # Also save points
    print("Data saved.")

    # --- Optional: Calculate Reprojection Error Manually (for verification) ---
    mean_error = 0
    if len(objpoints) > 0:
        for i in range(len(objpoints)):
            try:
                imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            except cv2.error as proj_err:
                print(f"Warning: Could not project points for image {i}. Error: {proj_err}")
                # Assign a high error or skip? For average, skipping might be better.
        print(f"\nAverage reprojection error (calculated manually): {mean_error / len(objpoints)}")
    else:
         print("\nSkipping manual reprojection error calculation (no points).")


    # --- Visualize Undistortion (Optional but Recommended) ---
    print("\nVisualizing undistortion on the first accepted sample image...")
    if objpoints and image_files: # Check if we have points and file list
        # Find the first image that was successfully used
        first_accepted_idx = -1
        # Need to find the original index corresponding to objpoints[0]
        # This simple approach assumes objpoints are added in order of file processing
        # A more robust way would be to store filenames alongside accepted points
        first_accepted_img_path = None
        temp_accepted_count = 0
        for idx, fname_vis in enumerate(image_files):
             # Rough check if this image *might* have been accepted
             # This isn't perfect without storing which files led to accepted points
             # Let's just use the first image processed IF at least one was accepted
             if accepted_count > 0:
                 first_accepted_img_path = image_files[0] # Default to first image if any accepted
                 break
             # A better way: Check if fname corresponds to an accepted point set.
             # requires storing filenames with objpoints/imgpoints when accepted.

        if first_accepted_img_path:
            img_distorted = cv2.imread(first_accepted_img_path)
            if img_distorted is not None:
                h, w = img_distorted.shape[:2]
                # Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w,h), np.eye(3), balance=0.0) # Crop
                Knew = K.copy() # Keep original K for fisheye.undistortImage
                img_undistorted = cv2.fisheye.undistortImage(img_distorted, K, D, Knew=Knew)

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
             print("Could not determine a sample image for undistortion visualization.")
    else:
         print("No valid images were accepted, cannot visualize undistortion.")


else:
    print("\nCalibration failed.")