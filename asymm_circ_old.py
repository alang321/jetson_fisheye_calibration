import cv2
import numpy as np
import os
import glob
import argparse
import sys

# --- Configuration ---
# These can be overridden by command-line arguments
DEFAULT_IMAGE_DIR = './calibration_images/' # Directory containing calibration images
DEFAULT_GRID_COLS = 4               # Number of circles horizontally
DEFAULT_GRID_ROWS = 11              # Number of circles vertically (total circles = cols * rows)
DEFAULT_SPACING_MM = 20             # Distance between adjacent circle centers in mm
DEFAULT_OUTPUT_FILE = 'fisheye_calibration_data.npz' # Output file for calibration data

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
# For asymmetric grids, points alternate rows with an offset.
objp = np.zeros((grid_cols * grid_rows, 3), np.float32)

for r in range(grid_rows):
    for c in range(grid_cols):
        idx = r * grid_cols + c
        # Asymmetric grid: X alternates offset, Y increments by row
        objp[idx, 0] = (2*c + r % 2) * spacing
        objp[idx, 1] = r * spacing
        objp[idx, 2] = 0 # Z coordinate is always 0 for a planar target

# Arrays to store object points and image points from all accepted images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane

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

# --- Process Images ---
img_shape = None # Store image shape (height, width)

print("\nProcessing images. Press 'y' to accept detection, 'n' to reject, 'q' to quit.")

for i, fname in enumerate(image_files):
    print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(fname)}")
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
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0) # Optional blur to reduce noise first

    params = cv2.SimpleBlobDetector_Params()

    # --- Tune these parameters based on your grid, lighting, and image resolution ---

    # Filter by Area.
    params.filterByArea = True
    # Estimate min/max area: Area = pi * (radius_pixels)^2
    # Measure expected circle radius in pixels in typical images. Be generous.
    params.minArea = 100  # Adjust based on your images (pixels^2)
    params.maxArea = 70000 # Adjust based on your images (pixels^2)

    # Filter by Circularity (1 is a perfect circle)
    params.filterByCircularity = True
    params.minCircularity = 0.6 # Allow some imperfection/perspective distortion

    # Filter by Convexity (1 is perfectly convex)
    params.filterByConvexity = True
    params.minConvexity = 0.85

    # Filter by Inertia Ratio (closer to 0 for elongated, 1 for circle)
    params.filterByInertia = True
    params.minInertiaRatio = 0.1 # Allow some elongation due to perspective

    # Distance between blobs (tune if circles are close)
    # params.minDistBetweenBlobs = 10 # Adjust if needed

    # Thresholds (less critical if using adaptive thresholding before)
    # params.minThreshold = 10
    # params.maxThreshold = 200

    blob_detector = cv2.SimpleBlobDetector_create(params)

    # --- Process Images ---
    # ... inside the loop ...
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # --- Optional Preprocessing (as described above) ---
    # gray_processed = ... apply adaptiveThreshold or CLAHE ...

    # Find the circle grid centers using the custom detector
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID # | cv2.CALIB_CB_CLUSTERING
    ret, corners = cv2.findCirclesGrid(
        gray, # or gray_processed if you preprocessed
        pattern_size,
        flags=flags,
        blobDetector=blob_detector # <--- Pass the custom detector!
    )

    # If found, draw corners and display
    if ret:
        print("  Grid detected!")
        # corners are the detected circle centers (N, 1, 2)

        # Draw the corners on the image
        vis_img = img_color.copy()
        cv2.drawChessboardCorners(vis_img, pattern_size, corners, ret) # Works for circles too

        # Display the image with detected corners
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL) # Allow resizing
        cv2.resizeWindow('Detection', 800, 600) # Adjust size as needed
        cv2.imshow('Detection', vis_img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                print("  Accepted.")
                objpoints.append(objp)
                imgpoints.append(corners) # Append the detected corners
                break
            elif key == ord('n'):
                print("  Rejected.")
                break
            elif key == ord('q'):
                print("\nQuitting detection phase.")
                cv2.destroyAllWindows()
                sys.exit(0)
            else:
                print("  Invalid key. Press 'y' (accept), 'n' (reject), or 'q' (quit).")
    else:
        print("  Grid not detected.")
        # Display the image that failed detection for debugging
        cv2.namedWindow('Detection Failed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection Failed', 800, 600)
        # Choose which image to show (raw gray, thresholded, CLAHE, etc.)
        cv2.imshow('Detection Failed', gray_blur) # Or show thresh, cl1 etc.
        print("  (Displaying image where detection failed. Press any key to continue)")
        cv2.waitKey(0) # Wait indefinitely until a key is pressed
        cv2.destroyWindow('Detection Failed') # Clean up the specific window

cv2.destroyAllWindows()

# --- Perform Calibration ---
print(f"\nCollected {len(objpoints)} valid views for calibration.")

if len(objpoints) < min_images_for_calib:
    print(f"Error: Insufficient number of valid views ({len(objpoints)}). Need at least {min_images_for_calib}.")
    sys.exit(1)

if img_shape is None:
    print("Error: Could not determine image shape (no images processed?).")
    sys.exit(1)

print(f"\nRunning fisheye calibration with image size: {img_shape[::-1]} (width, height)...") # OpenCV uses (width, height)

# Prepare for fisheye calibration
# Initialize K and D matrices (will be estimated)
K_init = np.eye(3)
D_init = np.zeros((4, 1)) # Fisheye model uses 4 coefficients (k1, k2, k3, k4)

# Calibration flags
# CALIB_RECOMPUTE_EXTRINSIC: Recalculates extrinsics after each iteration. Good.
# CALIB_CHECK_COND: Checks validity of condition number.
# CALIB_FIX_SKEW: Assumes zero skew (alpha=gamma). Usually safe.
# Consider CALIB_FIX_PRINCIPAL_POINT if you have a good reason, otherwise let it optimize.
calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW

# NOTE: cv2.fisheye.calibrate requires object points as a list of N x 3 arrays,
#       image points as a list of N x 1 x 2 arrays, and image_size as (WIDTH, HEIGHT).
try:
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img_shape[::-1], # image size (width, height)
        K_init,
        D_init,
        flags=calib_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6) # Termination criteria
    )
except cv2.error as e:
     print(f"!!! OpenCV Error during calibration: {e}")
     print("This might be due to insufficient points, poor detections, or numerical instability.")
     print("Try adding more diverse views, ensure grid detection is accurate, or check grid parameters.")
     sys.exit(1)


# --- Results ---
if ret:
    print("\nCalibration successful!")
    print(f"  RMS reprojection error: {ret}")
    print("\nCamera Matrix (K):")
    print(K)
    print("\nDistortion Coefficients (D) [k1, k2, k3, k4]:")
    print(D.flatten()) # D is usually returned as a column vector

    # --- Save Results ---
    print(f"\nSaving calibration data to: {output_file}")
    np.savez(output_file, K=K, D=D, img_shape=img_shape, rms=ret)
    print("Data saved.")

    # --- Optional: Calculate Reprojection Error Manually (for verification) ---
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"\nAverage reprojection error (calculated manually): {mean_error / len(objpoints)}")

else:
    print("\nCalibration failed.")