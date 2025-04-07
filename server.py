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
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
FRAMERATE = 30
STREAM_BITRATE = 2000000 # Bitrate for streaming (adjust as needed)

LAPTOP_IP = '172.27.0.35' # !!! REPLACE WITH YOUR LAPTOP'S ACTUAL IP ADDRESS !!!
VIDEO_PORT = 5000
COMMAND_PORT = 5001

SAVE_DIR = "calibration_images" # Directory to save images on the drone
IMAGE_PREFIX = "calib_"
IMAGE_FORMAT = ".png"
# --- End Configuration ---

# Global flag and lock for signaling capture
capture_requested = False
capture_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()
frame_count = 0

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# GStreamer Pipeline Construction
def build_pipeline_string():
    # Source pipeline - remove trailing ' !' if present
    source_pipeline = (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){CAPTURE_WIDTH}, height=(int){CAPTURE_HEIGHT}, framerate=(fraction){FRAMERATE}/1 ! "
        f"nvvidconv flip-method=0 ! "
        # Output NV12 which is commonly accepted by encoders and videoconvert
        f"video/x-raw(memory:NVMM), width=(int){CAPTURE_WIDTH}, height=(int){CAPTURE_HEIGHT}, format=(string)NV12"
        # Removed the second nvvidconv and the conversion to I420 here for potential simplification
        # If you specifically need I420 before the tee, add it back:
        # f" ! nvvidconv ! video/x-raw(memory:NVMM), format=(string)I420"
    )

    # Tee element - no trailing space needed if concatenated correctly
    tee_pipeline = "tee name=t"

    # Branch 1: Streaming to Laptop (using hardware encoder)
    # Removed the redundant nvvidconv before the encoder
    stream_pipeline = (
        f"t. ! queue ! nvv4l2h264enc "
        f"bitrate={STREAM_BITRATE} preset-level=UltraFastPreset insert-sps-pps=true ! "
        f"h264parse ! rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={LAPTOP_IP} port={VIDEO_PORT} sync=false async=false"
    )

    # Branch 2: Local capture via Appsink (converting NV12 to BGR for OpenCV)
    capture_pipeline = (
        f"t. ! queue ! nvvidconv ! "   # Let nvvidconv handle NVMM->System transfer.
                                    # It will likely output a common format like I420 or NV12 in system memory.
        # REMOVED: video/x-raw, format=(string)BGR !  <-- This was the error
        f"videoconvert ! "             # videoconvert takes the system memory format (e.g., I420/NV12)
                                    # and converts it to BGR.
        f"video/x-raw, format=(string)BGR ! " # Specify BGR *output* for appsink
        f"appsink name=mysink emit-signals=True max-buffers=1 drop=True"
    )

    # Correct concatenation: Link source TO tee, then link FROM tee branches
    # Note the explicit '!' linking source_pipeline to tee_pipeline.
    # The spaces between the branch definitions are important for readability but not syntax.
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
            while True: # Keep accepting new connections
                conn, addr = s.accept()
                with conn: # Keep connection open for multiple commands
                    print(f"Command connection from {addr}")
                    try:
                        while True: # Loop to receive multiple commands
                            data = conn.recv(1024)
                            if not data: # Client closed connection
                                print(f"Connection closed by {addr}")
                                break
                            command = data.decode('utf-8')
                            if command == 'CAPTURE':
                                print("Capture command received!")
                                with capture_lock:
                                    capture_requested = True
                                conn.sendall(b'ACK')
                            # Add other commands or an exit condition if needed
                            # elif command == 'QUIT':
                            #     conn.sendall(b'BYE')
                            #     break
                            else:
                                print(f"Unknown command: {command}")
                                conn.sendall(b'UNKNOWN')
                    except ConnectionResetError:
                        print(f"Connection reset by {addr}")
                    except Exception as e:
                        print(f"Error during command handling with {addr}: {e}")
                # Connection automatically closed by 'with conn:' exiting
                print(f"Finished with connection from {addr}")

        except Exception as e:
            print(f"Error in command listener setup: {e}")
            # Consider if Gst.main_quit() is still appropriate here
            GLib.idle_add(Gst.main_quit) # Safer way to quit from thread

# Callback function for appsink to get frames
# Callback function for appsink to get frames
def on_new_sample(appsink):
    global latest_frame
    # Emit the 'pull-sample' action signal to retrieve the sample
    sample = appsink.emit('pull-sample') # <--- CORRECT WAY
    if sample:
        buf = sample.get_buffer()
        if buf is None:
             print("Error: Could not get buffer from sample")
             return Gst.FlowReturn.ERROR # Or OK depending on desired behaviour

        caps = sample.get_caps()
        if caps is None:
             print("Error: Could not get caps from sample")
             return Gst.FlowReturn.ERROR # Or OK

        # Extract frame details
        structure = caps.get_structure(0)
        if structure is None:
            print("Error: Could not get structure from caps")
            return Gst.FlowReturn.ERROR # Or OK

        # Use get_value which is safer than direct access if field might be missing
        height_res, height = structure.get_int("height")
        width_res, width = structure.get_int("width")

        if not height_res or not width_res:
             print("Error: Could not get height/width from caps structure")
             return Gst.FlowReturn.ERROR # Or OK

        # Map buffer to numpy array
        success, map_info = buf.map(Gst.MapFlags.READ)
        if success:
            frame = np.ndarray(
                (height, width, 3), # Assuming BGR format from videoconvert
                buffer=map_info.data,
                dtype=np.uint8
            )
            with frame_lock:
                # Make a copy as the buffer will be unmapped
                latest_frame = frame.copy()
            buf.unmap(map_info)
            # Sample is automatically unreffed by GStreamer after the signal emission returns
            return Gst.FlowReturn.OK # Indicate success
        else:
            print("Error: Failed to map buffer")
            # Sample is automatically unreffed
            return Gst.FlowReturn.ERROR # Indicate failure

    # If emit('pull-sample') returns None (shouldn't happen if signal fired, but good practice)
    return Gst.FlowReturn.ERROR # Or OK if no sample is not critical


# Main execution
def main():
    global capture_requested, frame_count

    # Start GStreamer
    Gst.init(None)

    # Start command listener thread
    listener_thread = threading.Thread(target=command_listener, daemon=True)
    listener_thread.start()

    # Build and launch the pipeline
    pipeline_str = build_pipeline_string()
    pipeline = Gst.parse_launch(pipeline_str)

    # Get the appsink element
    appsink = pipeline.get_by_name('mysink')
    if not appsink:
        print("Error: Could not find appsink element 'mysink'")
        sys.exit(1)

    # Set appsink properties
    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers", 1) # Store only the latest frame
    appsink.set_property("drop", True) # Drop old frames if buffer is full
    appsink.connect("new-sample", on_new_sample)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline playing...")

    # GLib Main Loop to handle GStreamer events and signals
    loop = GLib.MainLoop()
    
    # Add a periodic check for capture requests within the GLib loop
    def check_capture_request():
        global capture_requested, latest_frame, frame_count
        perform_capture = False
        with capture_lock:
            if capture_requested:
                perform_capture = True
                capture_requested = False # Reset flag

        if perform_capture:
            print("Processing capture request...")
            captured_frame_copy = None
            with frame_lock:
                if latest_frame is not None:
                    # Make another copy for saving to avoid race conditions
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
        
        return True # Keep the timeout source running

    # Check for capture requests every 100ms
    GLib.timeout_add(100, check_capture_request) 

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Ctrl+C pressed, exiting.")
    finally:
        # Cleanup
        print("Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped.")
        # Loop should exit automatically on Gst.main_quit() or Ctrl+C

if __name__ == "__main__":
    main()