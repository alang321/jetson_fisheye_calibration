import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import cv2
import numpy as np
import socket
import threading
import time
import os
import sys
# --- Configuration ---
CAPTURE_WIDTH = 1640
CAPTURE_HEIGHT = 1232
FRAMERATE = 20

STREAM_WIDTH = 320
STREAM_HEIGHT = 240
STREAM_BITRATE = 400000  # 400 kbps

LAPTOP_IP = '10.42.0.246'
VIDEO_PORT = 5000
COMMAND_PORT = 5001

SAVE_DIR = "calibration_images"
IMAGE_PREFIX = "calib_"
IMAGE_FORMAT = ".png"
# --- End Configuration ---


# Global flags and variables
capture_requested = False
capture_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()   # <-- Add this line
frame_count = 0
# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)
def build_pipeline_string():
    source_pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){CAPTURE_WIDTH}, height=(int){CAPTURE_HEIGHT}, framerate=(fraction){FRAMERATE}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, format=(string)I420, width=(int){CAPTURE_WIDTH}, height=(int){CAPTURE_HEIGHT}"
    )

    tee_pipeline = "tee name=t"

    # Stream branch → MPEG-TS + UDP
    stream_pipeline = (
        f"t. ! queue ! "
        f"videoscale ! "
        f"video/x-raw, width=(int){STREAM_WIDTH}, height=(int){STREAM_HEIGHT} ! "
        f"nvvidconv ! "
        f"video/x-raw(memory:NVMM), format=(string)NV12 ! "
        f"nvv4l2h264enc bitrate={STREAM_BITRATE} preset-level=UltraFastPreset insert-sps-pps=true ! "
        f"h264parse ! mpegtsmux ! "
        f"udpsink host={LAPTOP_IP} port={VIDEO_PORT} sync=false async=false"
    )

    # Capture branch for stills
    capture_pipeline = (
        f"t. ! queue ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR, width=(int){CAPTURE_WIDTH}, height=(int){CAPTURE_HEIGHT} ! "
        f"appsink name=mysink emit-signals=True max-buffers=1 drop=True"
    )

    full_pipeline = f"{source_pipeline} ! {tee_pipeline} {stream_pipeline} {capture_pipeline}"
    print("--- GStreamer Pipeline ---")
    print(full_pipeline)
    print("--------------------------")
    return full_pipeline


# Function to handle incoming capture commands
def command_listener():
    global capture_requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('0.0.0.0', COMMAND_PORT))
            s.listen()
            print(f"Command listener waiting on port {COMMAND_PORT}")
            while True:
                conn, addr = s.accept()
                with conn:
                    print(f"Command connection from {addr}")
                    try:
                        while True:
                            data = conn.recv(1024)
                            if not data:
                                print(f"Connection closed by {addr}")
                                break
                            command = data.decode('utf-8')
                            if command == 'CAPTURE':
                                print("Capture command received!")
                                with capture_lock:
                                    capture_requested = True
                                conn.sendall(b'ACK')
                            else:
                                print(f"Unknown command: {command}")
                                conn.sendall(b'UNKNOWN')
                    except ConnectionResetError:
                        print(f"Connection reset by {addr}")
                    except Exception as e:
                        print(f"Error during command handling with {addr}: {e}")
                print(f"Finished with connection from {addr}")
        except Exception as e:
            print(f"Error in command listener setup: {e}")
            GLib.idle_add(Gst.main_quit)

# Callback function for appsink to get frames
def on_new_sample(appsink):
    global latest_frame
    sample = appsink.emit('pull-sample')
    if not sample:
        return Gst.FlowReturn.OK

    buf = sample.get_buffer()
    caps = sample.get_caps()
    if not buf or not caps:
        print("ERROR:on_new_sample: Could not get buffer or caps")
        return Gst.FlowReturn.ERROR

    structure = caps.get_structure(0)
    height_res, height = structure.get_int("height")
    width_res, width = structure.get_int("width")

    if not height_res or not width_res or width != CAPTURE_WIDTH or height != CAPTURE_HEIGHT:
        print(f"ERROR:on_new_sample: Unexpected frame dimensions. Expected {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}, got {width}x{height}")
        return Gst.FlowReturn.ERROR

    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        print("ERROR:on_new_sample: Failed to map buffer")
        return Gst.FlowReturn.ERROR

    try:
        # Buffer is already BGR, so just wrap it
        frame = np.ndarray(
            (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3),
            buffer=map_info.data,
            dtype=np.uint8
        )
        with frame_lock:
            latest_frame = frame.copy()
    except Exception as e:
        print(f"ERROR:on_new_sample: Failed to create NumPy array: {e}")
    finally:
        buf.unmap(map_info)

    return Gst.FlowReturn.OK

# Main execution
def main():
    global capture_requested, frame_count
    Gst.init(None)

    listener_thread = threading.Thread(target=command_listener, daemon=True)
    listener_thread.start()

    pipeline_str = build_pipeline_string()
    pipeline = Gst.parse_launch(pipeline_str)

    appsink = pipeline.get_by_name('mysink')
    if not appsink:
        print("Error: Could not find appsink element 'mysink'")
        sys.exit(1)

    appsink.set_property("emit-signals", True)
    appsink.connect("new-sample", on_new_sample)

    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline playing...")

    loop = GLib.MainLoop()
    
    def check_capture_request():
        global capture_requested, latest_frame, frame_count
        perform_capture = False
        with capture_lock:
            if capture_requested:
                perform_capture = True
                capture_requested = False

        if perform_capture:
            print("Processing capture request...")
            captured_frame_copy = None
            with frame_lock:
                if latest_frame is not None:
                    captured_frame_copy = latest_frame.copy() 

            if captured_frame_copy is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(SAVE_DIR, f"{IMAGE_PREFIX}{timestamp}_{frame_count:04d}{IMAGE_FORMAT}")
                try:
                    cv2.imwrite(filename, captured_frame_copy)
                    print(f"Successfully saved: {filename}")
                    frame_count += 1
                except Exception as e:
                    print(f"Error saving image: {e}")
            else:
                print("Capture requested, but no frame available yet.")
        
        return True # Keep running

    GLib.timeout_add(100, check_capture_request) 

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Ctrl+C pressed, exiting.")
    finally:
        print("Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped.")

if __name__ == "__main__":
    main()