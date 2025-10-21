# jetson_fisheye_calibration

## 1. Capturing Images

1. Connect both Drone and Laptop to same local network.

1. **Start the Video Stream from the Drone**  
   Copy `server.py` to the drone and run it to start streaming the camera feed.

3. **Capture Images**  
   Run `control.py` on your laptop. A window will pop up—click on it and press `'c'` to capture an image.

4. **Show the Calibration Target**  
   Display `asymmetric_circles_grid.png` on a screen or print it out. Measure and note the real-world spacing between horizontal points (needed for accurate calibration).

5. **Capture Diverse Views**  
   Capture images from various angles and distances to ensure good calibration coverage.

6. **Transfer Images**  
   Copy the captured images into the `calibration_images` folder on your laptop.

---

## 2. Calibration

1. **Run the Calibration Script**  
   Execute:
   ```bash
   python3 asymm_circle_fisheye.py --dir ./calibration_images/ --cols 11 --rows 7 --spacing 39
   ```
   or as another example:
   ```bash
   python3 intrinsics_asymm_circle_fisheye.py --dir /home/anton/Downloads/dj6 --spacing 41.31428286 --debug --visualize_serpentine --visualize_hex_grid --visualize_asymmetric --cols 11 --rows 7
   ```

2. **Check Blob Detections**  
   Make sure the asymmetric circle grid is detected properly in each image.

3. **Follow the Prompts**  
   If automatic detection fails for any image, follow the terminal instructions for manual steps.
