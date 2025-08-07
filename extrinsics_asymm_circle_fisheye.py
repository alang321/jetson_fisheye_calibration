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

from typing import Optional, Tuple
from asymm_circle_helpers.auto_asymm_circle_grid_finder import auto_asymm_cricle_hexagon_matching
from asymm_circle_helpers.assisted_asymm_circle_grid_finder import outer_corner_assisted_local_vector_walk

# --- Grid Detection Logic ---
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
    try_recovery = getattr(args, 'try_recover_missing', False)
    visualize_hex = getattr(args, 'visualize_hex_grid', False)
    corners = auto_asymm_cricle_hexagon_matching(img, keypoints, pattern_size, try_recovery=try_recovery, visualize=visualize_hex)
    if corners is not None:
        print("  Hexagonal auto-finder successful.")
        return corners

    print("  Hexagonal auto-finder failed. Trying assisted serpentine finder...")

    if not getattr(args, 'no_assisted', False):
        corners = outer_corner_assisted_local_vector_walk(img, keypoints, objp, pattern_size, args.visualize_serpentine)
        if corners is not None:
            print("  Assisted serpentine finder successful.")
            return corners
        
    print("  No grid found. All enabled finders failed.")
    return None

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
parser.add_argument('--visualize_hex_grid', action='store_true', help='Visualize hexagonal auto grid detection.')
parser.add_argument('--try_recover_missing', action='store_true', help='Attempt to recover missing grid points during hex search.')
parser.add_argument('--no_assisted', action='store_true', help='Do not fall back to the assisted four-corner grid finder.')
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

print("\nAttempting to find grid corners...")
corners = find_grid_in_image(img_color, processed_gray, blob_detector, objp, pattern_size, args)

if corners is None:
    print("\nError: Failed to find grid corners.")
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