import cv2
import socket
import sys
import gi # Optional: Only if OpenCV doesn't handle the pipeline directly
try:
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None) # Initialize GStreamer if using it directly
    GST_AVAILABLE = True
except (ImportError, ValueError):
    GST_AVAILABLE = False
    print("PyGObject/GStreamer not found. OpenCV will try to handle the pipeline.")
    print("Ensure OpenCV was built with GStreamer support.")


# --- Configuration ---
DRONE_IP = '172.27.0.46' # !!! REPLACE WITH YOUR DRONE'S ACTUAL IP ADDRESS !!!
VIDEO_PORT = 5000
COMMAND_PORT = 5001

RECEIVER_PIPELINE = (
    f"udpsrc port={VIDEO_PORT} "
    f"caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96\" ! "
    f"rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 emit-signals=true sync=false"
    # Alternative simpler pipeline if decodebin works well:
    # f"udpsrc port={VIDEO_PORT} caps=\"application/x-rtp...\" ! rtph264depay ! decodebin ! videoconvert ! appsink ..."
    # Or specify decoder:
    # f"udpsrc port={VIDEO_PORT} caps=\"application/x-rtp...\" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink ..."
)
# --- End Configuration ---

print("--- Receiver Pipeline ---")
print(RECEIVER_PIPELINE)
print("-------------------------")

# Function to send capture command
def send_capture_command(sock):
    try:
        print("Sending CAPTURE command...")
        sock.sendall(b'CAPTURE')
        response = sock.recv(1024) # Wait for ACK
        print(f"Drone response: {response.decode('utf-8')}")
    except Exception as e:
        print(f"Error sending command: {e}")
        # Consider trying to reconnect here if needed

# --- Main Execution ---
# Setup Video Capture
cap = cv2.VideoCapture(RECEIVER_PIPELINE, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    print("Troubleshooting:")
    print("- Check if drone_streamer.py is running on the Jetson.")
    print("- Verify DRONE_IP and LAPTOP_IP are correct in both scripts.")
    print("- Check network connectivity (ping the drone from laptop and vice-versa).")
    print("- Ensure GStreamer is correctly installed on the laptop OR OpenCV has GStreamer support.")
    print(f"- Check if port {VIDEO_PORT} (UDP) is blocked by a firewall.")
    sys.exit(1)
else:
    print("Video stream opened successfully.")

# Setup Command Connection
command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    print(f"Connecting to drone command server at {DRONE_IP}:{COMMAND_PORT}...")
    command_socket.connect((DRONE_IP, COMMAND_PORT))
    print("Command connection established.")
except ConnectionRefusedError:
    print(f"Error: Connection refused. Is the drone_streamer.py running and listening on port {COMMAND_PORT}?")
    cap.release()
    sys.exit(1)
except socket.timeout:
    print(f"Error: Connection timed out. Check DRONE_IP and network.")
    cap.release()
    sys.exit(1)
except Exception as e:
    print(f"Error connecting command socket: {e}")
    cap.release()
    sys.exit(1)


print("\n--- Controls ---")
print("Press 'c' to capture image on drone.")
print("Press 'q' to quit.")
print("----------------\n")

window_name = "Drone Live View (Press 'c' to capture, 'q' to quit)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame. Stream might have ended.")
        time.sleep(0.5) # Avoid busy-looping if stream stalls
        # You might want to attempt reconnection or exit here
        continue

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quit key pressed.")
        break
    elif key == ord('c'):
        send_capture_command(command_socket)

# Cleanup
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
command_socket.close()
print("Exited.")