#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import os
import glob # To find image files

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Camera calibration using asymmetric circle grids from a folder of images.')
parser.add_argument('-d', '--dir', type=str, required=True, help='Path to the directory containing calibration images.')
parser.add_argument('-r', '--rows', type=int, required=True, help='Number of circles along the grid height.')
parser.add_argument('-c', '--cols', type=int, required=True, help='Number of circles along the grid width.')
parser.add_argument('-s', '--size', type=float, default=1.0, help='Spacing unit between circle centers (e.g., mm, cm, inches, or just 1.0 for relative units). Used for object points.')
parser.add_argument('-e', '--ext', type=str, default='jpg,png,jpeg,bmp,tif,tiff', help='Comma-separated list of image file extensions to process.')
parser.add_argument('-o', '--output', type=str, default='calibration_data_folder.npz', help='Output file name for calibration data (camera matrix, distortion coeffs).')
parser.add_argument('--min_frames', type=int, default=10, help='Minimum number of accepted frames required for calibration.')
parser.add_argument('--subpix', action='store_true', help='Enable sub-pixel refinement for detected corners.')
parser.add_argument('--show_rejected', action='store_true', help='Briefly show images where the grid was not found.')
# --- New Argument ---
parser.add_argument('--adaptive_thresh', action='store_true', help='Apply adaptive thresholding before grid detection.')


args = parser.parse_args()

# --- Configuration ---
image_folder = args.dir
pattern_size = (args.cols, args.rows) # Number of INNER corners (circles) per row and column
square_size = args.size               # Size of the spacing unit
output_file = args.output
min_accepted_frames = args.min_frames
use_subpix_refinement = args.subpix
show_rejected_frames = args.show_rejected
use_adaptive_thresh = args.adaptive_thresh # Store the new argument
extensions = [ext.strip().lower() for ext in args.ext.split(',')]

print("--- Configuration ---")
print(f"Image Folder: {image_folder}")
print(f"Grid Size (cols x rows): {pattern_size[0]} x {pattern_size[1]}")
print(f"Square Size: {square_size}")
print(f"Image Extensions: {extensions}")
print(f"Minimum Frames: {min_accepted_frames}")
print(f"Output File: {output_file}")
print(f"Sub-pixel Refinement: {use_subpix_refinement}")
print(f"Show Not Found: {show_rejected_frames}")
print(f"Use Adaptive Thresholding: {use_adaptive_thresh}") # Print new setting
print("---------------------\n")

# --- Validate Input Folder ---
if not os.path.isdir(image_folder):
    print(f"Error: Input directory not found: {image_folder}")
    exit()

# --- Find Image Files ---
image_files = []
for ext in extensions:
    search_pattern = os.path.join(image_folder, f'*.{ext}')
    found_files = glob.glob(search_pattern)
    # On some systems, glob might be case-sensitive, add uppercase check
    found_files_upper = glob.glob(os.path.join(image_folder, f'*.{ext.upper()}'))
    image_files.extend(found_files)
    image_files.extend(f for f in found_files_upper if f not in image_files)


if not image_files:
    print(f"Error: No images with extensions {extensions} found in {image_folder}")
    exit()

print(f"Found {len(image_files)} potential image files.")

# --- Prepare Object Points ---
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
for i in range(pattern_size[1]): # rows (height)
    for j in range(pattern_size[0]): # cols (width)
        objp[i * pattern_size[0] + j, :2] = ( (2*j + i%2)*square_size, i*square_size )

# Arrays to store object points and image points from all accepted images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
frame_size = None # Will store (width, height) from the first successful image

# Subpixel refinement criteria
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- Process Images ---
print("\nProcessing images...")
print("Press 'a' to ACCEPT the detection.")
print("Press 'r' to REJECT the detection.")
print("Press 'q' to QUIT processing.")
print("-" * 20)

accepted_count = 0
processed_count = 0

for fname in image_files:
    print(f"Processing: {os.path.basename(fname)} ... ", end='')
    processed_count += 1
    img = cv2.imread(fname)

    if img is None:
        print("Failed to read image.")
        continue

    if frame_size is None:
        frame_size = (img.shape[1], img.shape[0])
        print(f"Determined image size: {frame_size[0]}x{frame_size[1]}")
    elif (img.shape[1], img.shape[0]) != frame_size:
         print(f"Warning: Image size {img.shape[1]}x{img.shape[0]} differs from first image {frame_size[0]}x{frame_size[1]}. Skipping.")
         continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    display_img = img.copy()

    # --- Apply Adaptive Thresholding if requested ---
    processed_img_for_detection = gray # Start with grayscale
    if use_adaptive_thresh:
        print("[Thresh] ", end='')
        # Parameters for adaptiveThreshold:
        #   maxValue: 255 (standard for binary images)
        #   adaptiveMethod: ADAPTIVE_THRESH_GAUSSIAN_C is often better than MEAN_C
        #   thresholdType: THRESH_BINARY or THRESH_BINARY_INV. Since circles are black on white, THRESH_BINARY should make circles 0 and background 255.
        #   blockSize: Size of the neighborhood area (must be odd). Needs tuning. Start with 11 or higher.
        #   C: Constant subtracted from the mean or weighted sum. Fine-tunes the threshold. Start with 2.
        processed_img_for_detection = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                            cv2.THRESH_BINARY, 15, 2) # ** TUNABLE PARAMETERS **

        # --- Optional: Visualization of thresholding ---
        # cv2.imshow('Thresholded', processed_img_for_detection)
        # cv2.waitKey(100) # Show briefly
        # --- End Optional Visualization ---


    # --- Find Circle Grid ---
    # Use the (potentially thresholded) image for detection
    ret_find, centers = cv2.findCirclesGrid(processed_img_for_detection, pattern_size,
                                            flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)

    key = -1

    if ret_find:
        print("Grid found!")

        # --- Sub-pixel Refinement (Optional) ---
        # *** IMPORTANT: Always run subpix on the ORIGINAL grayscale image ***
        if use_subpix_refinement:
            centers_subpix = cv2.cornerSubPix(gray, centers, (11, 11), (-1, -1), subpix_criteria) # Use original gray here
            if centers_subpix is not None and len(centers_subpix) == len(centers):
                 centers = centers_subpix
            else:
                 print("Warning: Sub-pixel refinement failed for this frame.")
        else:
             centers_subpix = centers

        # Draw the detected grid on the *original color* image copy
        cv2.drawChessboardCorners(display_img, pattern_size, centers, ret_find)
        status_text = f"Detected! Accepted: {accepted_count}. Press 'a' Accept, 'r' Reject, 'q' Quit"
        cv2.putText(display_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow('Calibration Image', display_img)
        key = cv2.waitKey(0)

        if key == ord('a') or key == ord('A'):
            print(f" -> Accepted ({accepted_count + 1})")
            objpoints.append(objp)
            imgpoints.append(centers) # Use the potentially refined centers
            accepted_count += 1
            cv2.putText(display_img, "ACCEPTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Calibration Image', display_img)
            cv2.waitKey(300)

        elif key == ord('r') or key == ord('R'):
            print(" -> Rejected")
            cv2.putText(display_img, "REJECTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow('Calibration Image', display_img)
            cv2.waitKey(300)

        elif key == ord('q') or key == ord('Q'):
            print(" -> Quit requested during review.")
            break

    else:
        print("Grid not found.")
        if show_rejected_frames:
            status_text = f"Grid not found. Accepted: {accepted_count}. Press 'q' Quit"
            cv2.putText(display_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Calibration Image', display_img)
            key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):
        print(" -> Quit requested.")
        break

cv2.destroyAllWindows()


# --- Perform Calibration ---
if frame_size is None:
     print("\nError: No valid images processed, cannot determine image size for calibration.")
elif accepted_count >= min_accepted_frames:
    print(f"\nAttempting calibration using {accepted_count} accepted frames...")

    ret_calib, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None)

    if ret_calib:
        print("Calibration successful!")
        print("\nCamera Matrix (mtx):")
        print(camera_matrix)
        print("\nDistortion Coefficients (dist):")
        print(dist_coeffs)

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            imgpoints2 = imgpoints2.reshape(-1, 2)
            imgpoints_i = imgpoints[i].reshape(-1, 2)
            error = cv2.norm(imgpoints_i, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        reprojection_error = mean_error / len(objpoints)
        print(f"\nTotal Mean Reprojection Error: {reprojection_error:.4f} pixels")
        if reprojection_error < 1.0:
             print("This reprojection error is generally considered good.")
        else:
             print("This reprojection error might be high. Check grid quality, lighting, image focus, and frame selection.")

        print(f"\nSaving calibration data to '{output_file}'...")
        try:
            np.savez(output_file, mtx=camera_matrix, dist=dist_coeffs, rvecs=rvecs, tvecs=tvecs, reproj_error=reprojection_error, frame_size=frame_size)
            print("Data saved successfully.")
        except Exception as e:
            print(f"Error saving data: {e}")
    else:
        print("Calibration failed.")

elif accepted_count < min_accepted_frames:
     print(f"\nCalibration not performed. Only {accepted_count} frames were accepted (minimum required: {min_accepted_frames}).")
else:
    print("\nCalibration not performed (Quit early or not enough frames accepted).")


# --- Cleanup ---
print("\nDone.")