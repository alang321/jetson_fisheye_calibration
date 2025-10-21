#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import cv2, numpy as np, socket, threading, time, os, sys, argparse, struct

# ================= Configuration via CLI =================
parser = argparse.ArgumentParser(description="Jetson Drone Video + Capture Server")
parser.add_argument("--laptop-ip", type=str, default="10.42.0.246")
parser.add_argument("--video-port", type=int, default=5000)
parser.add_argument("--command-port", type=int, default=5001)
parser.add_argument("--lowres", action="store_true", help="Low-res live stream")
parser.add_argument("--no-denoise", action="store_true", help="Disable Argus denoise and sharpening")
parser.add_argument("--capture-width", type=int, default=1640, help="Capture width")
parser.add_argument("--capture-height", type=int, default=1232, help="Capture height")
args = parser.parse_args()
# =========================================================

CAPTURE_WIDTH = args.capture_width
CAPTURE_HEIGHT = args.capture_height
FRAMERATE = 10

STREAM_WIDTH = 320 if args.lowres else 1280
STREAM_HEIGHT = 240 if args.lowres else 720
STREAM_BITRATE = 400000 if args.lowres else 2000000

SAVE_DIR = "calibration_images"
IMAGE_PREFIX = "calib_"
IMAGE_FORMAT = ".png"

# =========================================================
capture_requested = False
capture_lock = threading.Lock()
latest_frame = None
frame_lock = threading.Lock()
frame_count = 0
os.makedirs(SAVE_DIR, exist_ok=True)


# ========== Build GStreamer pipeline ==========
def build_pipeline_string():
    # 1. Capture branch (full resolution)
    denoise_flags = "tnr-mode=0 ee-mode=0" if args.no_denoise else ""
    source = (
        f"nvarguscamerasrc {denoise_flags} ! "
        f"video/x-raw(memory:NVMM), width=(int){CAPTURE_WIDTH}, height=(int){CAPTURE_HEIGHT}, "
        f"framerate=(fraction){FRAMERATE}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw(memory:NVMM), format=(string)NV12"
    )

    tee = "tee name=t"

    stream = (
        f"t. ! queue ! "
        f"nvvidconv ! video/x-raw(memory:NVMM), width=(int){STREAM_WIDTH}, height=(int){STREAM_HEIGHT}, format=(string)NV12 ! "
        f"nvv4l2h264enc bitrate={STREAM_BITRATE} insert-sps-pps=true ! "
        f"h264parse ! mpegtsmux ! udpsink host={args.laptop_ip} port={args.video_port} sync=false async=false"
    )

    # Capture branch (two nvvidconv stages for safety)
    capture = (
        f"t. ! queue ! "
        f"nvvidconv ! video/x-raw(memory:NVMM), width=(int){CAPTURE_WIDTH}, height=(int){CAPTURE_HEIGHT}, format=(string)NV12 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink name=mysink emit-signals=True max-buffers=1 drop=True"
    )

    return f"{source} ! {tee} {stream} {capture}"



# ========== Send image over socket ==========
def send_image(conn, image_path):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        size = len(data)
        conn.sendall(struct.pack("<Q", size))
        conn.sendall(data)
        print(f"Sent {size} bytes ({os.path.basename(image_path)})")
    except Exception as e:
        print(f"Error sending image: {e}")


# ========== Command listener ==========
def command_listener():
    global capture_requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", args.command_port))
        s.listen()
        print(f"Command listener on port {args.command_port}")

        while True:
            conn, addr = s.accept()
            print(f"Connected to {addr}")
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    cmd = data.decode().strip().upper()
                    if cmd == "CAPTURE":
                        print("Capture command received!")
                        with capture_lock:
                            capture_requested = True
                        conn.sendall(b"ACK")
                        # Wait for image to be saved and send it
                        time.sleep(0.3)
                        last_file = sorted(os.listdir(SAVE_DIR))[-1]
                        send_image(conn, os.path.join(SAVE_DIR, last_file))
                    else:
                        conn.sendall(b"UNKNOWN")
                print(f"Disconnected from {addr}")


# ========== Appsink callback ==========
def on_new_sample(appsink):
    global latest_frame
    sample = appsink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK

    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)

    width = structure.get_value("width")
    height = structure.get_value("height")
    fmt = structure.get_value("format")

    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        print("ERROR: could not map buffer")
        return Gst.FlowReturn.ERROR

    try:
        # Handle BGR and other common formats correctly
        if fmt == "BGR":
            frame = np.frombuffer(map_info.data, np.uint8).reshape((height, width, 3))
        elif fmt == "RGB":
            frame = np.frombuffer(map_info.data, np.uint8).reshape((height, width, 3))[:, :, ::-1]
        elif fmt == "GRAY8":
            frame = np.frombuffer(map_info.data, np.uint8).reshape((height, width, 1))
        else:
            print(f"Unsupported format {fmt}, skipping frame")
            buf.unmap(map_info)
            return Gst.FlowReturn.OK

        with frame_lock:
            latest_frame = frame.copy()

    except Exception as e:
        print(f"ERROR: Failed to parse frame: {e}")

    finally:
        buf.unmap(map_info)

    return Gst.FlowReturn.OK



# ========== Main loop ==========
def main():
    Gst.init(None)
    threading.Thread(target=command_listener, daemon=True).start()

    pipeline_str = build_pipeline_string()
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name("mysink")
    appsink.connect("new-sample", on_new_sample)

    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline started.")

    loop = GLib.MainLoop()

    def check_capture():
        global capture_requested, latest_frame, frame_count
        if capture_requested:
            with capture_lock:
                capture_requested = False
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            if frame is not None:
                fname = os.path.join(SAVE_DIR, f"{IMAGE_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}_{frame_count:04d}{IMAGE_FORMAT}")
                cv2.imwrite(fname, frame)
                print(f"Saved {fname}")
                frame_count += 1
        return True

    GLib.timeout_add(100, check_capture)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
