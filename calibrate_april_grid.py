import cv2
import cv2.aruco as aruco
import numpy as np
import os
import glob
import argparse
import sys
import time

# --- Configuration ---
# These can be overridden by command-line arguments
DEFAULT_IMAGE_DIR = './calibration_images_april/' # Directory containing calibration images
DEFAULT_MARKERS_X = 5               # Number of markers horizontally
DEFAULT_MARKERS_Y = 7               # Number of markers vertically
DEFAULT_MARKER_LENGTH_MM = 25       # Size of the black square in mm
DEFAULT_MARKER_SEP_MM = 30          # Separation between markers in mm
DEFAULT_DICT_NAME = 'DICT_APRILTAG_36h11' # ArUco/AprilTag dictionary name
DEFAULT_OUTPUT_FILE = 'fisheye_calibration_aprilgrid_data.npz' # Output file
DEFAULT_MIN_MARKERS = 10            # Min detected markers in a view to accept it

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Fisheye Camera Calibration using AprilGrid/ArUco Grid.')
parser.add_argument('--dir', type=str, default=DEFAULT_IMAGE_DIR,
                    help=f'Directory containing calibration images (default: {DEFAULT_IMAGE_DIR})')
parser.add_argument('--markers_x', type=int, default=DEFAULT_MARKERS_X,
                    help=f'Number of markers horizontally (default: {DEFAULT_MARKERS_X})')
parser.add_argument('--markers_y', type=int, default=DEFAULT_MARKERS_Y,
                    help=f'Number of markers vertically (default: {DEFAULT_MARKERS_Y})')
parser.add_argument('--marker_len', type=float, default=DEFAULT_MARKER_LENGTH_MM,
                    help=f'Marker square size in mm (default: {DEFAULT_MARKER_LENGTH_MM})')
parser.add_argument('--marker_sep', type=float, default=DEFAULT_MARKER_SEP_MM,
                    help=f'Marker separation in mm (default: {DEFAULT_MARKER_SEP_MM})')
parser.add_argument('--dict', type=str, default=DEFAULT_DICT_NAME,
                    help=f'ArUco dictionary name (e.g., DICT_APRILTAG_36h11, DICT_6X6_250) (default: {DEFAULT_DICT_NAME})')
parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE,
                    help=f'Output file path for calibration data (.npz) (default: {DEFAULT_OUTPUT_FILE})')
parser.add_argument('--ext', type=str, nargs='+', default=['jpg', 'png', 'bmp', 'tif', 'jpeg'],
                    help='Image file extensions to process (default: jpg png bmp tif jpeg)')
parser.add_argument('--min_markers', type=int, default=DEFAULT_MIN_MARKERS,
                     help=f'Minimum detected markers required per view (default: {DEFAULT_MIN_MARKERS})')

args = parser.parse_args()

# --- Setup ---
image_dir = args.dir
markers_x = args.markers_x
markers_y = args.markers_y
marker_length_mm = args.marker_len
marker_separation_mm = args.marker_sep
aruco_dict_name = args.dict
output_file = args.output
image_extensions = args.ext
min_markers_for_view = args.min_markers

min_views_for_calib = 10 # Minimum number of good views required for final calibration

# Convert mm to meters for ArUco functions
marker_length_m = marker_length_mm / 1000.0
marker_separation_m = marker_separation_mm / 1000.0

# --- Initialize ArUco ---
try:
    # Get the predefined dictionary
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, aruco_dict_name))
except AttributeError:
    print(f"Error: Invalid ArUco dictionary name: {aruco_dict_name}")
    print("Available dictionaries:")
    for name in dir(aruco):
        if name.startswith("DICT_"):
            print(f"  - {name}")
    sys.exit(1)

# Create the grid board object
# Note: The board defines the 3D locations of the marker corners
board = aruco.GridBoard(
    size=(markers_x, markers_y),
    markerLength=marker_length_m,
    markerSeparation=marker_separation_m,
    dictionary=dictionary
)

# ArUco detector parameters
# parameters = aruco.DetectorParameters_create() # Older OpenCV
parameters = aruco.DetectorParameters() # Newer OpenCV >= 4.7?
# Consider tuning parameters if detection is poor:
# parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
# parameters.adaptiveThreshWinSizeMin = 3
# parameters.adaptiveThreshWinSizeMax = 23
# parameters.adaptiveThreshWinSizeStep = 10
# parameters.adaptiveThreshConstant = 7
detector = aruco.ArucoDetector(dictionary, parameters)

# Arrays to store object points and image points from all accepted images
# IMPORTANT: For ArUco calibration, we store points *per view*.
# The object points correspond *only* to the markers detected in that view.
all_objpoints = [] # List of 3D points for detected corners in each valid view
all_imgpoints = [] # List of 2D points for detected corners in each valid view

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
print(f"Using ArUco dictionary: {aruco_dict_name}")
print(f"Grid: {markers_x}x{markers_y}, Marker Size: {marker_length_mm}mm, Separation: {marker_separation_mm}mm")

# --- Process Images ---
img_shape = None # Store image shape (height, width)

print(f"\nProcessing images. Need at least {min_markers_for_view} markers per view.")
print("Press 'y' to accept detection, 'n' to reject, 'q' to quit.")

start_time = time.time()
accepted_count = 0

for i, fname in enumerate(image_files):
    print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(fname)}")
    img_color = cv2.imread(fname)
    if img_color is None:
        print("  Warning: Could not read image.")
        continue

    # Check image shape consistency
    current_shape = img_color.shape[:2] # (height, width)
    if img_shape is None:
        img_shape = current_shape
        print(f"  Image shape detected: {img_shape} (height, width)")
    elif current_shape != img_shape:
         print(f"  Warning: Image size {current_shape} differs from first image {img_shape}. Skipping.")
         continue

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # Draw markers for visualization
    vis_img = img_color.copy()
    aruco.drawDetectedMarkers(vis_img, corners, ids)

    if ids is not None and len(ids) > 0:
        print(f"  Detected {len(ids)} markers.")

        # Refine detected corners (optional but recommended)
        # Note: Use `board` for context. K, D are not needed here.
        # We refine based on the image itself.
        corners, ids, rejected, recovered_ids = detector.refineDetectedMarkers(
             image=gray,
             board=board,
             detectedCorners=corners,
             detectedIds=ids,
             rejectedCorners=rejected
             # cameraMatrix=K_init_guess, # Not needed for refinement itself
             # distCoeffs=D_init_guess
        )
        print(f"  Refined detection: {len(ids)} markers retained.")
        # Re-draw after refinement for display
        vis_img = img_color.copy()
        aruco.drawDetectedMarkers(vis_img, corners, ids)


        # Check if enough markers were detected for this view
        if len(ids) >= min_markers_for_view:
            print(f"  Sufficient markers detected ({len(ids)} >= {min_markers_for_view}).")

            # Get the 3D object points and 2D image points for the *detected* markers
            # objPoints_view: (N, 1, 3), imgPoints_view: (N, 1, 2) where N is total corners
            objPoints_view, imgPoints_view = board.matchImagePoints(corners, ids)

            if objPoints_view is not None and imgPoints_view is not None and len(objPoints_view) > 4:
                # Draw the matched board axes (optional, needs pose estimation)
                # try:
                #     ret_pose, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, K_placeholder, D_placeholder) # Needs dummy K, D or previous estimate
                #     if ret_pose:
                #         cv2.drawFrameAxes(vis_img, K_placeholder, D_placeholder, rvec, tvec, marker_length_m * 1.5)
                # except: pass # Ignore errors if K/D aren't ready

                # Display the image with detected markers
                cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Detection', 800, 600) # Adjust size as needed
                cv2.imshow('Detection', vis_img)

                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('y'):
                        print("  Accepted.")
                        # Reshape for fisheye.calibrate: (N, 3) and (N, 1, 2)
                        all_objpoints.append(objPoints_view.reshape(-1, 3))
                        all_imgpoints.append(imgPoints_view.reshape(-1, 1, 2))
                        accepted_count += 1
                        break
                    elif key == ord('n'):
                        print("  Rejected.")
                        break
                    elif key == ord('q'):
                        print("\nQuitting detection phase.")
                        cv2.destroyAllWindows()
                        end_time = time.time()
                        print(f"Detection phase took {end_time - start_time:.2f} seconds.")
                        sys.exit(0)
                    else:
                        print("  Invalid key. Press 'y' (accept), 'n' (reject), or 'q' (quit).")
            else:
                 print("  Warning: Failed to match enough points with the board definition after detection.")
                 # Optionally display the image anyway for debugging
                 cv2.imshow('Detection', vis_img)
                 cv2.waitKey(500) # Show for 0.5 seconds

        else:
            print(f"  Insufficient markers detected ({len(ids)} < {min_markers_for_view}). Skipping view.")
            # Optionally display the image anyway for debugging
            # cv2.imshow('Detection', vis_img)
            # cv2.waitKey(500) # Show for 0.5 seconds
    else:
        print("  No markers detected.")
        # Optionally display the image anyway for debugging
        # cv2.imshow('Detection', vis_img)
        # cv2.waitKey(500) # Show for 0.5 seconds

cv2.destroyAllWindows()
end_time = time.time()
print(f"\nDetection phase complete. Took {end_time - start_time:.2f} seconds.")

# --- Perform Calibration ---
print(f"\nCollected {len(all_objpoints)} valid views for calibration.")

if len(all_objpoints) < min_views_for_calib:
    print(f"Error: Insufficient number of valid views ({len(all_objpoints)}). Need at least {min_views_for_calib}.")
    sys.exit(1)

if img_shape is None:
    print("Error: Could not determine image shape (no images processed successfully?).")
    sys.exit(1)

print(f"\nRunning fisheye calibration with image size: {img_shape[::-1]} (width, height)...")

# Prepare for fisheye calibration
K_init = np.eye(3)  # Initial guess for K
D_init = np.zeros((4, 1)) # Initial guess for D (k1, k2, k3, k4)

# Fisheye calibration flags
# CALIB_RECOMPUTE_EXTRINSIC: Good practice
# CALIB_CHECK_COND: Helps catch numerical issues
# CALIB_FIX_SKEW: Assumes pixels are square and axes orthogonal
# CALIB_FIX_K1, K2, K3, K4: Can be used to fix specific distortion coeffs if needed (rarely)
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
# Consider adding cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT if the center is known accurately

# Termination criteria for the optimization process
term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# Check data shapes (debugging)
# print(f"Number of object point sets: {len(all_objpoints)}")
# print(f"Number of image point sets: {len(all_imgpoints)}")
# if all_objpoints:
#     print(f"Shape of first objpoints set: {all_objpoints[0].shape}, dtype: {all_objpoints[0].dtype}")
#     print(f"Shape of first imgpoints set: {all_imgpoints[0].shape}, dtype: {all_imgpoints[0].dtype}")
# print(f"Image shape for calibration: {img_shape[::-1]}") # Width, Height

start_calib_time = time.time()
try:
    # Note: cv2.fisheye.calibrate expects image size as (WIDTH, HEIGHT)
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        all_objpoints,   # List of (N, 3) arrays, float32
        all_imgpoints,   # List of (N, 1, 2) arrays, float32
        img_shape[::-1], # (width, height)
        K_init,
        D_init,
        flags=calib_flags,
        criteria=term_criteria
    )
except cv2.error as e:
     print(f"!!! OpenCV Error during fisheye calibration: {e}")
     print("This might be due to:")
     print("  - Insufficient number of views or points per view.")
     print("  - Poor quality detections (check marker size, lighting, focus).")
     print("  - Incorrect board parameters (marker size, separation, layout).")
     print("  - Highly non-planar board or extreme distortion making optimization difficult.")
     print("  - Numerical instability (try adding more diverse views).")
     sys.exit(1)
except Exception as e:
    print(f"!!! An unexpected error occurred during calibration: {e}")
    sys.exit(1)

end_calib_time = time.time()
print(f"Calibration process took {end_calib_time - start_calib_time:.2f} seconds.")

# --- Results ---
if ret:
    print("\nCalibration successful!")
    print(f"  RMS reprojection error: {ret:.4f}") # RMS error returned by calibrate
    print("\nCamera Matrix (K):")
    print(K)
    print("\nDistortion Coefficients (D) [k1, k2, k3, k4]:")
    print(D.flatten())

    # --- Save Results ---
    print(f"\nSaving calibration data to: {output_file}")
    np.savez(output_file, K=K, D=D, img_shape=img_shape, rms=ret,
             markers_x=markers_x, markers_y=markers_y,
             marker_length_mm=marker_length_mm, marker_separation_mm=marker_separation_mm,
             aruco_dict_name=aruco_dict_name)
    print("Data saved.")

    # --- Optional: Calculate Reprojection Error Manually (for verification) ---
    mean_error_manual = 0
    total_points = 0
    for i in range(len(all_objpoints)):
        # Project the 3D points back into the image plane using the calibration results
        imgpoints2, _ = cv2.fisheye.projectPoints(
            all_objpoints[i].reshape(-1, 1, 3), # Needs shape (N, 1, 3) for projectPoints
            rvecs[i],
            tvecs[i],
            K,
            D
        )

        # Calculate the error for this view
        # Compare imgpoints2 (N, 1, 2) with all_imgpoints[i] (N, 1, 2)
        error = cv2.norm(all_imgpoints[i], imgpoints2, cv2.NORM_L2)

        # Accumulate error and points
        mean_error_manual += error*error # Sum of squared errors
        total_points += len(all_objpoints[i])

    # Calculate Root Mean Square Error (RMSE) manually
    rmse_manual = np.sqrt(mean_error_manual / total_points)
    print(f"\nAverage reprojection error (RMSE calculated manually): {rmse_manual:.4f} pixels")
    print(f"(Note: This manual calculation should be close to the RMS returned by calibrate)")

else:
    print("\nCalibration failed. The optimization algorithm did not converge.")
    print("Consider:")
    print("  - Adding more calibration images with diverse views.")
    print("  - Ensuring accurate marker detection in the accepted views.")
    print("  - Verifying the board parameters (size, separation, dictionary).")