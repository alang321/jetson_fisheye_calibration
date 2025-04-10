import cv2
print(cv2.getBuildInformation())
import socket
import sys
import numpy as np


# --- Configuration ---
DRONE_IP = '172.27.0.47' # !!! REPLACE WITH YOUR DRONE'S ACTUAL IP ADDRESS !!!
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

RECEIVER_PIPELINE = (
    f"udpsrc port={VIDEO_PORT} caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96\" ! "
    f"rtph264depay ! h264parse ! decodebin ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
)

RECEIVER_PIPELINE = (
    f"udpsrc port={VIDEO_PORT} "
    f"caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96\" ! "
    f"rtph264depay ! decodebin ! videoconvert ! appsink drop=true max-buffers=1 sync=false" # Removed explicit video/x-raw caps
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

# Setup Command Connection
command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    print(f"Connecting to drone command server at {DRONE_IP}:{COMMAND_PORT}...")
    command_socket.connect((DRONE_IP, COMMAND_PORT))
    print("Command connection established.")
except ConnectionRefusedError:
    print(f"Error: Connection refused. Is the drone_streamer.py running and listening on port {COMMAND_PORT}?")
    sys.exit(1)
except socket.timeout:
    print(f"Error: Connection timed out. Check DRONE_IP and network.")
    sys.exit(1)
except Exception as e:
    print(f"Error connecting command socket: {e}")
    sys.exit(1)


print("\n--- Controls ---")
print("Press 'c' to capture image on drone.")
print("Press 'q' to quit.")
print("----------------\n")

window_name = "Drone Live View (Press 'c' to capture, 'q' to quit)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

dummy_image = np.zeros((480, 640, 3), dtype=np.uint8) # Dummy image for initial display
while True:
    cv2.imshow(window_name, dummy_image) # Display dummy image
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quit key pressed.")
        break
    elif key == ord('c'):
        send_capture_command(command_socket)

# Cleanup
print("Cleaning up...")
cv2.destroyAllWindows()
command_socket.close()
print("Exited.")