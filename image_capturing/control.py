import cv2
print(cv2.getBuildInformation()) # Good for checking GStreamer support
import socket
import sys
import numpy as np


# --- Configuration ---
DRONE_IP = '10.42.0.1'   # Jetson’s IP when it’s the hotspot
VIDEO_PORT = 5000
COMMAND_PORT = 5001

# --- Working MPEG-TS pipeline ---
RECEIVER_PIPELINE = (
    f"udpsrc port={VIDEO_PORT} ! "
    f"tsdemux ! h264parse ! avdec_h264 ! videoconvert ! appsink drop=true max-buffers=1 sync=false"
)

print("--- Receiver Pipeline ---")
print(RECEIVER_PIPELINE)
print("-------------------------")

# Function to send capture command
def send_capture_command(sock):
    try:
        print("Sending CAPTURE command...")
        sock.sendall(b'CAPTURE')
        response = sock.recv(1024)
        print(f"Drone response: {response.decode('utf-8')}")
    except Exception as e:
        print(f"Error sending command: {e}")

# Setup Command Connection
command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    print(f"Connecting to drone command server at {DRONE_IP}:{COMMAND_PORT}...")
    command_socket.connect((DRONE_IP, COMMAND_PORT))
    print("Command connection established.")
except Exception as e:
    print(f"Error connecting command socket: {e}")
    sys.exit(1)


print("\n--- Controls ---")
print("Press 'c' to capture image on drone.")
print("Press 'q' to quit.")
print("----------------\n")

# --- FIXED VIDEO CAPTURE ---
print("Attempting to open GStreamer pipeline...")
cap = cv2.VideoCapture(RECEIVER_PIPELINE, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    print("Check that GStreamer is installed and cv2 was compiled with it (see build info above).")
    print("Also check that the server is running and IPs are correct.")
    command_socket.close()
    sys.exit(1)

print("Pipeline opened successfully. Waiting for video feed...")

window_name = "Drone Live View (Press 'c' to capture, 'q' to quit)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Dummy image for when stream is down
# We don't know the resolution yet, so start with a standard size
# It will be resized once the first frame arrives
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(dummy_image, "Connecting...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
frame_received = False

while True:
    ret, frame = cap.read() # Read frame from GStreamer pipeline

    if ret:
        if not frame_received:
            print("First frame received!")
            frame_received = True
        
        # Once we get a frame, use it
        frame_to_show = frame
        
        # Update dummy image in case connection drops
        if dummy_image.shape != frame.shape:
            dummy_image = np.zeros_like(frame)
            cv2.putText(dummy_image, "Connection Lost...", (50, int(frame.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    else:
        # If read fails, show the dummy image
        frame_to_show = dummy_image
        if frame_received:
            print("Warning: Failed to grab frame. Stream may be down.")

    cv2.imshow(window_name, frame_to_show)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quit key pressed.")
        break
    elif key == ord('c'):
        send_capture_command(command_socket)

# Cleanup
print("Cleaning up...")
cap.release() # Release the video capture
cv2.destroyAllWindows()
command_socket.close()
print("Exited.")