# jetson_fisheye_calibration

## 1. Capturing Images

1. Connect both Drone and Laptop to a fast local network.

2.

1. **Start the Video Stream from the Drone**  
   Copy `server.py` to the drone and run it to start streaming the camera feed.

2. **Open the Video Stream on Your Laptop**  
   On your laptop, use the following GStreamer command to view the stream:

   ```bash
   gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" \
     ! rtph264depay ! decodebin ! videoconvert ! autovideosink sync=false
   ```

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
   or with pretty manual detection visualization:
   ```bash
   python3 asymm_circle_fisheye.py --dir ./calibration_images/ --cols 11 --rows 7 --spacing 39 --visualize_serpentine
   ```

2. **Check Blob Detections**  
   Make sure the asymmetric circle grid is detected properly in each image.

3. **Follow the Prompts**  
   If automatic detection fails for any image, follow the terminal instructions for manual steps.
