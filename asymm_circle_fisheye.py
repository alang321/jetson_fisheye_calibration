import cv2
import numpy as np
import os
import glob
import argparse
import sys
import random
import collections
from typing import Optional, List, Tuple, Dict, Any

# Assuming your custom finder functions are in these files/modules
from asymm_circle_grid_pattern_finders.auto_asymm_circle_grid_finder import auto_asymm_cricle_hexagon_matching
from asymm_circle_grid_pattern_finders.assisted_asymm_circle_grid_finder import outer_corner_assisted_local_vector_walk

# --- Configuration Constants ---
DEFAULT_IMAGE_DIR = './calibration_images/'
DEFAULT_GRID_COLS = 4
DEFAULT_GRID_ROWS = 11
DEFAULT_SPACING_MM = 39.0
DEFAULT_OUTPUT_FILE = 'fisheye_calibration_data.npz'
MIN_IMAGES_FOR_CALIB = 10
class InteractiveTuner:
    """A class to encapsulate the state and logic for the interactive debug GUI."""
    def __init__(self, initial_params: cv2.SimpleBlobDetector_Params):
        self.params = initial_params
        self.min_area = initial_params.minArea
        self.max_area = initial_params.maxArea
        self.manual_thresh = 127
        self.preprocess_mode = 0  # 0: CLAHE, 1: Raw Gray, 2: Manual, 3: Adaptive
        self.thresh_type = cv2.THRESH_BINARY
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _callback_min_area(self, val): self.min_area = max(1, val)
    def _callback_max_area(self, val): self.max_area = val
    def _callback_thresh(self, val): self.manual_thresh = val

    def setup_windows(self):
        cv2.namedWindow('Debug View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug View', 1200, 400)
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Controls', 400, 200)

        # --- FIX IS HERE: Cast self.min_area and self.max_area to int() ---
        cv2.createTrackbar('Manual Thresh', 'Controls', self.manual_thresh, 255, self._callback_thresh)
        cv2.createTrackbar('Min Area', 'Controls', int(self.min_area), 5000, self._callback_min_area)
        cv2.createTrackbar('Max Area', 'Controls', int(self.max_area), 20000, self._callback_max_area)
        # --- END FIX ---

    def get_processed_image(self, gray_img: np.ndarray) -> Tuple[np.ndarray, str]:
        """Applies the currently selected preprocessing filter."""
        mode_map = {0: "CLAHE", 1: "Raw Gray", 2: "Manual Thr", 3: "Adaptive Thr"}
        if self.preprocess_mode == 0:
            return self.clahe.apply(gray_img), mode_map[0]
        if self.preprocess_mode == 1:
            return gray_img, mode_map[1]
        if self.preprocess_mode == 2:
            _, processed = cv2.threshold(gray_img, self.manual_thresh, 255, self.thresh_type)
            return processed, f"{mode_map[2]} ({self.manual_thresh})"
        if self.preprocess_mode == 3:
            block_size = 11 + 2 * (self.manual_thresh // 32); block_size = max(3, block_size | 1)
            processed = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, self.thresh_type, block_size, 2)
            return processed, f"{mode_map[3]} (Block: {block_size})"
        return gray_img, "Unknown"

    def run_debug_loop(self, color_img: np.ndarray, gray_img: np.ndarray) -> Tuple[Optional[np.ndarray], cv2.SimpleBlobDetector]:
        """Runs the interactive tuning loop for a single image."""
        h, w = color_img.shape[:2]
        vis_h = 350
        vis_w = int(w * vis_h / h) if h > 0 else 0
        
        while True:
            processed_gray, desc = self.get_processed_image(gray_img)
            
            # Update blob detector params
            self.params.minArea = self.min_area
            self.params.maxArea = max(self.min_area + 1, self.max_area)
            detector = cv2.SimpleBlobDetector_create(self.params)
            keypoints = detector.detect(processed_gray)
            
            # Create visualizations
            vis_orig = cv2.resize(color_img, (vis_w, vis_h))
            vis_proc = cv2.resize(cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR), (vis_w, vis_h))
            
            # Scale keypoints for visualization on the resized image
            scaled_keypoints = [cv2.KeyPoint(kp.pt[0]*vis_w/w, kp.pt[1]*vis_h/h, kp.size*vis_w/w) for kp in keypoints]
            
            vis_blobs = cv2.drawKeypoints(vis_orig.copy(), scaled_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv2.putText(vis_blobs, f'Blobs: {len(keypoints)}', (10, vis_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(vis_proc, desc, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            debug_vis = np.hstack((vis_orig, vis_proc, vis_blobs))
            cv2.imshow('Debug View', debug_vis)
            
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'): sys.exit(0)
            if key == ord('s'): return None, detector
            if key == ord(' '): return processed_gray, detector
            if key == ord('m'): self.preprocess_mode = (self.preprocess_mode + 1) % 4
            if key == ord('t'): self.thresh_type = cv2.THRESH_BINARY if self.thresh_type == cv2.THRESH_BINARY_INV else cv2.THRESH_BINARY

def parse_arguments() -> argparse.Namespace:
    """Sets up and parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Fisheye Camera Calibration using Asymmetric Circle Grid.')
    parser.add_argument('--dir', type=str, default=DEFAULT_IMAGE_DIR, help=f'Directory with calibration images (default: {DEFAULT_IMAGE_DIR})')
    parser.add_argument('--cols', type=int, default=DEFAULT_GRID_COLS, help=f'Number of circles horizontally (default: {DEFAULT_GRID_COLS})')
    parser.add_argument('--rows', type=int, default=DEFAULT_GRID_ROWS, help=f'Number of circles vertically (default: {DEFAULT_GRID_ROWS})')
    parser.add_argument('--spacing', type=float, default=DEFAULT_SPACING_MM, help=f'Distance between circle centers in mm (default: {DEFAULT_SPACING_MM})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE, help=f'Output file for calibration data (.npz) (default: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('--ext', type=str, nargs='+', default=['jpg', 'png'], help='Image file extensions (default: jpg png)')
    parser.add_argument('--no_confirm', action='store_true', dest='no_confirm', help='Do not ask for confirmation before accepting detected grids.')
    parser.add_argument('--no_manual_backup', action='store_true', dest='no_manual_backup', help='Do not save a backup of the original images.')
    parser.add_argument('--debug', action='store_true', help='Run in interactive debug mode to tune blob detector parameters.')
    parser.add_argument('--visualize_serpentine', action='store_true', help='Visualize serpentine grid detection.')
    parser.add_argument('--visualize_hex_grid', action='store_true', help='Visualize hexagonal auto grid detection.')
    return parser.parse_args()

def report_and_visualize_results(
    calib_data: Dict[str, Any],
    img_shape: Tuple[int, int],
    max_window_dim: int = 1200
):
    """
    Prints final calibration results and shows coverage and undistortion visualizations.

    Args:
        calib_data: The dictionary returned by a successful calibration run.
        img_shape: The (height, width) of the calibration images.
        max_window_dim: The maximum dimension for visualization windows.
    """
    K = calib_data['K']
    D = calib_data['D']
    rms = calib_data['ret']
    imgpoints = calib_data['imgpoints']
    filenames = calib_data['filenames']
    
    # --- 1. Print Numerical Results ---
    print("\n" + "="*40)
    print("      Calibration Successful!")
    print("="*40)
    print(f"  Used {len(imgpoints)} views for final calibration.")
    print(f"  RMS reprojection error: {rms:.4f} pixels")
    print("\nCamera Matrix (K):")
    print(K)
    print("\nDistortion Coefficients (D) [k1, k2, k3, k4]:")
    print(D.flatten())
    print("="*40 + "\n")

    # --- 2. Visualize Calibration Point Coverage ---
    print("Visualizing overall calibration point coverage...")
    coverage_img = np.full((img_shape[0], img_shape[1], 3), 255, dtype=np.uint8)
    total_points = sum(len(view) for view in imgpoints)
    
    for view_corners in imgpoints:
        for corner in view_corners:
            center = tuple(corner[0].astype(int))
            cv2.circle(coverage_img, center, 3, (255, 0, 0), -1)

    cv2.putText(coverage_img, f"Coverage: {total_points} points from {len(imgpoints)} views",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    WIN_NAME_COVERAGE = "Calibration Point Coverage"
    cv2.namedWindow(WIN_NAME_COVERAGE, cv2.WINDOW_NORMAL)
    h, w = img_shape
    aspect_ratio = w / h
    win_w = min(w, max_window_dim); win_h = int(win_w / aspect_ratio)
    cv2.resizeWindow(WIN_NAME_COVERAGE, win_w, win_h)
    cv2.imshow(WIN_NAME_COVERAGE, coverage_img)
    print("-> Displaying coverage map. Press any key to continue...")
    cv2.waitKey(0)

    # --- 3. Visualize Undistortion ---
    print("\nVisualizing undistortion on the first accepted sample image...")
    if not filenames:
        print("  No filenames recorded, cannot show undistortion.")
        return

    sample_img_path = filenames[0]
    img_distorted = cv2.imread(sample_img_path)

    if img_distorted is None:
        print(f"  Warning: Could not reload sample image {sample_img_path}")
        return

    h, w = img_distorted.shape[:2]
    image_size_wh = (w, h)
    
    # Generate the undistortion map
    # Using balance=0.0 shows all source pixels (with black borders)
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, image_size_wh, np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, image_size_wh, cv2.CV_16SC2)
    img_undistorted = cv2.remap(img_distorted, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Create a side-by-side comparison
    vis_compare = np.hstack((img_distorted, img_undistorted))

    WIN_NAME_UNDISTORT = 'Distorted vs Undistorted'
    cv2.namedWindow(WIN_NAME_UNDISTORT, cv2.WINDOW_NORMAL)
    # Make the window wide enough for two images
    h_vis, w_vis = vis_compare.shape[:2]
    aspect_ratio = w_vis / h_vis
    win_w = min(w_vis, max_window_dim * 2)
    win_h = int(win_w / aspect_ratio)
    cv2.resizeWindow(WIN_NAME_UNDISTORT, win_w, win_h)
    
    cv2.imshow(WIN_NAME_UNDISTORT, vis_compare)
    print(f"-> Displaying undistortion result for {os.path.basename(sample_img_path)}. Press any key to exit.")
    cv2.waitKey(0)



def generate_object_points(pattern_size: Tuple[int, int], spacing: float) -> np.ndarray:
    """Generates the 3D real-world coordinates for the asymmetric grid."""
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            row_offset = (r % 2) * (spacing / 2.0)
            objp[idx, 0] = c * spacing + row_offset
            objp[idx, 1] = r * spacing
    return objp

def find_image_files(directory: str, extensions: List[str]) -> List[str]:
    """Finds all image files in a directory with the given extensions."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f'*.{ext.lower()}')))
        files.extend(glob.glob(os.path.join(directory, f'*.{ext.upper()}')))
    if not files:
        print(f"Error: No images found in '{directory}' with extensions {extensions}")
        sys.exit(1)
    return sorted(list(set(files)))

def find_grid_in_image(img: np.ndarray, gray: np.ndarray, detector: cv2.SimpleBlobDetector, objp: np.ndarray, pattern_size: Tuple[int, int], args: argparse.Namespace) -> Optional[np.ndarray]:
    """Tries a sequence of methods to find the grid in an image."""
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING
    ret, corners = cv2.findCirclesGrid(gray, pattern_size, flags=flags, blobDetector=detector)
    if ret:
        print("  cv2.findCirclesGrid successful.")
        return corners

    keypoints = detector.detect(gray)
    if not keypoints:
        print("  No blobs detected, cannot attempt custom finders.")
        return None
    
    print("  findCirclesGrid failed. Trying custom hexagonal finder...")
    corners = auto_asymm_cricle_hexagon_matching(keypoints, pattern_size, visualize=args.visualize_hex_grid)
    if corners is not None:
        print("  Hexagonal auto-finder successful.")
        return corners

    print("  Hexagonal auto-finder failed. Trying assisted serpentine finder...")

    if not args.no_manual_backup:
        corners = outer_corner_assisted_local_vector_walk(img, keypoints, objp, pattern_size, args.visualize_serpentine)
        if corners is not None:
            print("  Assisted serpentine finder successful.")
            return corners
        
    print("  All finders failed.")
    return None

def run_calibration(objpoints: List[np.ndarray], imgpoints: List[np.ndarray], image_size: Tuple[int, int], filenames: List[str]) -> Optional[Dict[str, Any]]:
    """Performs fisheye calibration, with robust iterative removal of bad views."""
    print(f"\nRunning fisheye calibration with {len(objpoints)} views...")
    K_init, D_init = np.eye(3), np.zeros((4, 1))
    calib_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    for attempt in range(5): # Try up to 5 removal attempts
        if len(objpoints) < MIN_IMAGES_FOR_CALIB:
            print(f"  Stopping removal attempts. View count ({len(objpoints)}) is below minimum ({MIN_IMAGES_FOR_CALIB}).")
            break
        try:
            print(f"  Attempting calibration with {len(objpoints)} views...")
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, image_size, K_init, D_init, flags=calib_flags, criteria=criteria)
            print("  Calibration successful!")
            return {'ret': ret, 'K': K, 'D': D, 'rvecs': rvecs, 'tvecs': tvecs, 'objpoints': objpoints, 'imgpoints': imgpoints, 'filenames': filenames}
        except cv2.error as e:
            print(f"  OpenCV error: {e}. Removing a random view and retrying...")
            if "CALIB_CHECK_COND" in str(e):
                idx_to_remove = random.randrange(len(objpoints))
                removed_file = os.path.basename(filenames.pop(idx_to_remove))
                objpoints.pop(idx_to_remove)
                imgpoints.pop(idx_to_remove)
                print(f"  Randomly removed: {removed_file}")
            else:
                print("  Error is not CALIB_CHECK_COND. Cannot recover. Aborting.")
                return None
    
    print("Calibration failed after multiple removal attempts.")
    return None

def save_calibration_results(filepath: str, img_shape: Tuple[int, int], calib_data: Dict[str, Any]):
    """Saves final calibration data to an .npz file."""
    print(f"\nSaving calibration data to: {filepath}")
    try:
        np.savez(filepath, K=calib_data['K'], D=calib_data['D'], img_shape=img_shape, rms=calib_data['ret'],
                 objpoints=np.array(calib_data['objpoints'], dtype=object),
                 imgpoints=np.array(calib_data['imgpoints'], dtype=object),
                 filenames=np.array(calib_data['filenames']))
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    pattern_size = (args.cols, args.rows)
    objp = generate_object_points(pattern_size, args.spacing)
    image_files = find_image_files(args.dir, args.ext)
    
    # Setup Blob Detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True; params.minArea = 30; params.maxArea = 10000
    params.filterByCircularity = True; params.minCircularity = 0.5
    params.filterByConvexity = True; params.minConvexity = 0.80
    params.filterByInertia = True; params.minInertiaRatio = 0.1
    blob_detector = cv2.SimpleBlobDetector_create(params)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    tuner = InteractiveTuner(params) if args.debug else None
    if tuner: tuner.setup_windows()

    objpoints_all, imgpoints_all, filenames_all = [], [], []
    img_shape = None

    for i, fname in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {os.path.basename(fname)}")
        img_color = cv2.imread(fname)
        if img_color is None:
            print("  Warning: Could not read image.")
            continue
        
        if img_shape is None: img_shape = img_color.shape[:2]
        if img_color.shape[:2] != img_shape:
            print(f"  Warning: Image size mismatch. Skipping.")
            continue

        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        if tuner:
            processed_gray, blob_detector = tuner.run_debug_loop(img_color, gray)
            if processed_gray is None:
                print("  Skipped in debug mode.")
                continue
        else:
            print("  Applying CLAHE preprocessing...")
            processed_gray = clahe.apply(gray)
            
        corners = find_grid_in_image(img_color, processed_gray, blob_detector, objp, pattern_size, args)

        ask_confirm = args.no_confirm is not True

        if corners is not None:
            if ask_confirm:
                print("  Grid found. Please confirm.")
                vis_confirm = img_color.copy()
                cv2.drawChessboardCorners(vis_confirm, pattern_size, corners, True)
                # Resize confirmation window for smaller display
                h, w = vis_confirm.shape[:2]
                vis_h = 700
                vis_w = int(w * vis_h / h)
                vis_small = cv2.resize(vis_confirm, (vis_w, vis_h))
                cv2.namedWindow('Detection Confirmation', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Detection Confirmation', vis_w, vis_h)
                cv2.imshow('Detection Confirmation', vis_small)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    print("  Accepted.")
                    objpoints_all.append(objp.reshape(-1,1,3))
                    imgpoints_all.append(corners.reshape(-1,1,2))
                    filenames_all.append(fname)
                elif key == ord('q'):
                    print("Quitting.")
                    break
                else:
                    print("  Rejected.")
                cv2.destroyWindow('Detection Confirmation')
            else:
                # auto-accept without prompt
                print("  Grid found. Auto-accepting (ask_confirm=False).")
                objpoints_all.append(objp.reshape(-1,1,3))
                imgpoints_all.append(corners.reshape(-1,1,2))
                filenames_all.append(fname)

    cv2.destroyAllWindows()

    if len(objpoints_all) < MIN_IMAGES_FOR_CALIB:
        print(f"\nError: Insufficient views for calibration. Found {len(objpoints_all)}, need {MIN_IMAGES_FOR_CALIB}.")
        return

    image_size_wh = (img_shape[1], img_shape[0])
    calib_data = run_calibration(objpoints_all, imgpoints_all, image_size_wh, filenames_all)

    if calib_data:
        # --- NEW: Call the reporting and visualization function here ---
        report_and_visualize_results(calib_data, img_shape)
        
        # Then save the results as before
        save_calibration_results(args.output, img_shape, calib_data)


if __name__ == '__main__':
    main()