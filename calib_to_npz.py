#!/usr/bin/env python3
import json
import numpy as np

def main():
    # Load the JSON file
    with open('/home/anton/Git/dronerace/orin/CameraCalibrations/imx219_champion3_anton_intr_11-04.json', 'r') as f:
        data = json.load(f)
    
    # Extract camera matrix and distortion coefficients
    K = np.array(data["intrinsics"]["cam_matrix_drone"])
    D = np.array(data["intrinsics"]["dist_coeff_drone"])
    
    # Save the arrays into an .npz file with variable names 'K' and 'D'
    np.savez('DJ3.npz', K=K, D=D)
    print("Saved camera matrix (K) and distortion coefficients (D) into calibration.npz")
    print("K:", K)
    print("D:", D)

if __name__ == '__main__':
    main()
