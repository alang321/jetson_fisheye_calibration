#!/usr/bin/env python3
import cv2, socket, sys, os, time, argparse, struct, numpy as np
import time

parser = argparse.ArgumentParser(description="Drone Stream + Capture Client")
parser.add_argument("--drone-ip", type=str, default="10.42.0.1", help="IP of the Jetson/drone")
parser.add_argument("--video-port", type=int, default=5000, help="UDP port for video stream")
parser.add_argument("--command-port", type=int, default=5001, help="TCP port for commands")
args = parser.parse_args()

RECEIVER_PIPELINE = (
    f"udpsrc port={args.video_port} ! "
    f"tsdemux ! h264parse ! avdec_h264 ! videoconvert ! appsink drop=true max-buffers=1 sync=false"
)

SAVE_DIR = "received_images"
os.makedirs(SAVE_DIR, exist_ok=True)


def receive_image(sock):
    """Receives one image and saves it."""
    try:
        header = sock.recv(8)
        if not header:
            print("Connection closed while waiting for image size.")
            return
        size = struct.unpack("<Q", header)[0]
        print(f"Receiving image ({size} bytes)...")

        data = b""
        while len(data) < size:
            chunk = sock.recv(4096)
            if not chunk:
                print("Connection closed during image transfer.")
                break
            data += chunk
        
        if len(data) == size:
            filename = os.path.join(SAVE_DIR, f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png")
            with open(filename, "wb") as f:
                f.write(data)
            print(f"Saved {filename}")
        else:
            print("Image transfer incomplete.")
            
    except Exception as e:
        print(f"Error in receive_image: {e}")


def main():
    print("Connecting to drone...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            sock.connect((args.drone_ip, args.command_port))
            print("Command connection established.")
            break
        except ConnectionRefusedError:
            print("Server not ready yet, retrying in 2 seconds...")
            time.sleep(2)
        except OSError as e:
            # Handles temporary network errors
            print(f"Connection attempt failed ({e}), retrying in 2 seconds...")
            time.sleep(2)

    print("Opening video stream...")
    cap = cv2.VideoCapture(RECEIVER_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Failed to open video stream")
        sys.exit(1)

    cv2.namedWindow("Drone Live View", cv2.WINDOW_NORMAL)
    print("\nControls: [C] Capture | [Q] Quit\n")

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Drone Live View", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            print("Requesting capture...")
            sock.sendall(b"CAPTURE")
            
            # --- Modified Logic ---
            # Wait for ACK or ERR
            resp = sock.recv(3) 
            if resp == b"ACK":
                receive_image(sock)
            elif resp == b"ERR":
                print("Server: Capture failed (timeout or no frame).")
            else:
                print(f"Server: Unknown response: {resp}")
            # --- End Modified Logic ---

    cap.release()
    sock.close()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()