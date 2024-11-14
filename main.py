import argparse
import cv2
import io
import numpy as np
import shutil
import subprocess
from pathlib import Path
from tkinter import filedialog


__version__ = "0.021.20240921"

DEFAULT_DISPLAY_WIDTH = 700
DEFAULT_DISPLAY_HEIGHT = 700
DEFAULT_BLOCK_SIZE = 16
DEFAULT_MAX_CORNERS = 1000
DEFAULT_QUALITY_LEVEL = 1
DEFAULT_MIN_DISTANCE = 100
DEFAULT_ACCENT_VALUE = 0
DEFAULT_SCALE = 4
UNDO_LIMIT = 10

circle_size = 3
crop_top = 0
crop_bottom = 0
crop_left = 0
crop_right = 0
detected_corners = []
selected_corners = []
display_width = DEFAULT_DISPLAY_WIDTH
display_height = DEFAULT_DISPLAY_HEIGHT
undo_stack = []


def get_unique_path(path: Path) -> Path:
    """Returns new unique & unused path with suffix."""
    if not path.exists():
        return path

    base = path.stem
    ext = path.suffix

    counter = 1

    while path.exists():
        new_name = f"{base}_{counter}{ext}"
        path = path.with_name(new_name)
        counter += 1

    return path


def image_resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize image while maintaining aspect ratio."""
    dim = None
    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r_w = width / float(w)
        r_h = height / float(h)

        if r_w < r_h:
            dim = (width, int(h * r_w))
        else:
            dim = (int(w * r_h), height)

    resized = cv2.resize(img, dim, interpolation=inter)
    return resized, dim


def closest_node(node, nodes):
    """Find closest item to `node` in `nodes`."""
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


def sort_vertices(vertices):
    """Sort `vertices` in UL-UR-DL-DR order."""
    vertices = np.array(vertices)

    sorted_by_y = vertices[np.argsort(vertices[:, 1])]

    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    top_sorted = top_two[np.argsort(top_two[:, 0])]
    top_left = top_sorted[0]
    top_right = top_sorted[1]

    bottom_sorted = bottom_two[np.argsort(bottom_two[:, 0])]
    bottom_left = bottom_sorted[0]
    bottom_right = bottom_sorted[1]

    return np.array([top_left, top_right, bottom_left, bottom_right])


def click_handler(event, x, y, flags, param):
    """Handle clicks on CV2 dialog."""
    global img, display_width, display_height, warped

    use_custom_corners = bool(cv2.getTrackbarPos("Custom", "Select four corners"))
    
    if use_custom_corners:
        selected_corner = (round(x / display_width * image_width), round(y / display_height * image_height))
    else:
        closest_node_index = closest_node(
            (round(x / display_width * image_width), round(y / display_height * image_height)),
            detected_corners
        )
        selected_corner = detected_corners[closest_node_index]

    if event == cv2.EVENT_MOUSEMOVE:
        window_title = f"Select four corners ({len(selected_corners) + 1} of 4)"
        cv2.setWindowTitle("Select four corners", window_title)

    if event != cv2.EVENT_LBUTTONUP:
        return

    selected_corners.append(selected_corner)

    if len(selected_corners) >= 4:
        pp_corners = sort_vertices(list(selected_corners))
        vertices = np.array(pp_corners)

        side_lengths = [
            np.linalg.norm(vertices[0] - vertices[1]),
            np.linalg.norm(vertices[0] - vertices[2]),
            np.linalg.norm(vertices[1] - vertices[3]),
            np.linalg.norm(vertices[0] - vertices[3]),
        ]

        width = int(min(side_lengths[0], side_lengths[3]))
        height = int(min(side_lengths[1], side_lengths[2]))

        warped = unwarp_image(
            img, pp_corners, [(0, 0), (width, 0), (0, height), (width, height)]
        )

        warped = warped[0:height, 0:width]
        cv2.destroyAllWindows()

        resized, dim = image_resize(warped, DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)
        display_width, display_height = dim

        cv2.namedWindow("Cropped image")
        cv2.setWindowTitle("Cropped image", "Cropped image (WASD to crop, ER to rotate, Q to save)")
        cv2.imshow("Cropped image", resized)


def unwarp_image(img, src_points, dst_points):
    """Perform perspective warp operation on input `img`."""
    h, w = img.shape[:2]

    src = np.float32(src_points)
    dst = np.float32(dst_points)

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped


def detect_corners(img, max_corners, quality_level, min_distance):
    """Detect corners on given `img` using Shi-Tomasi corner detection."""
    global detected_corners

    try:
        accent_value = cv2.getTrackbarPos("Accent", "Select four corners")
    except:
        accent_value = DEFAULT_ACCENT_VALUE
    img = apply_accent_adjustment(img, accent_value)

    scale = DEFAULT_SCALE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gray_h, gray_w = gray.shape[:2]
    gray_resize = cv2.resize(gray, (gray_w // scale, gray_h // scale), interpolation=cv2.INTER_AREA)

    # Detect corners on the resized image
    corners = cv2.goodFeaturesToTrack(
        gray_resize,
        maxCorners=max_corners,
        qualityLevel=quality_level / 100.0,
        minDistance=min_distance // scale,
        blockSize=DEFAULT_BLOCK_SIZE
    )

    if corners is not None:
        # Scale corners back to the original size
        scaled_corners = (corners * scale).reshape(-1, 2)
        detected_corners = np.int32(scaled_corners)
    else:
        detected_corners = []

    return detected_corners


def apply_accent_adjustment(img, accent_value, *args):
    """Apply accent adjustment to image."""
    adjusted = img.copy()
    adjusted = np.clip(abs(adjusted + accent_value), 0, 255).astype(np.uint8)
    return adjusted


def rotate_image(flip_code):
    """Rotate image by 90 degrees and display."""
    global warped, undo_stack, display_width, display_height

    if len(undo_stack) >= UNDO_LIMIT:
        undo_stack.pop(0)
    undo_stack.append(warped.copy())

    rotated = cv2.transpose(warped)
    rotated = cv2.flip(rotated, flipCode=flip_code)
    warped = rotated

    resized, dim = image_resize(warped, width=DEFAULT_DISPLAY_WIDTH, height=DEFAULT_DISPLAY_HEIGHT)
    display_width, display_height = dim

    cv2.namedWindow("Cropped image")
    cv2.imshow("Cropped image", resized)


def crop_image(direction):
    """Crop image by one row/column of pixels and display."""
    global warped, undo_stack, crop_top, crop_bottom, crop_left, crop_right, display_width, display_height

    if len(undo_stack) >= UNDO_LIMIT:
        undo_stack.pop(0)
    undo_stack.append(warped.copy())

    if direction == "top":
        if crop_top < warped.shape[0] - 1:
            crop_top += 1
            warped = warped[3:, :]
    elif direction == "bottom":
        if crop_bottom < warped.shape[0] - 1:
            crop_bottom += 1
            warped = warped[:-3, :]
    elif direction == "left":
        if crop_left < warped.shape[1] - 1:
            crop_left += 1
            warped = warped[:, 3:]
    elif direction == "right":
        if crop_right < warped.shape[1] - 1:
            crop_right += 1
            warped = warped[:, :-3]

    resized, dim = image_resize(warped, width=DEFAULT_DISPLAY_WIDTH, height=DEFAULT_DISPLAY_HEIGHT)
    display_width, display_height = dim

    cv2.imshow("Cropped image", resized)


def undo_image(*args):
    """Revert image to last state."""
    global warped, undo_stack

    if not undo_stack:
        return

    warped = undo_stack.pop()

    resized, dim = image_resize(warped, width=DEFAULT_DISPLAY_WIDTH, height=DEFAULT_DISPLAY_HEIGHT)
    cv2.imshow("Cropped image", resized)


def update(*args):
    """Update image display based on trackbar settings."""
    global img, circle_size, detected_corners, display_width, display_height

    max_corners = cv2.getTrackbarPos("Max Corners", "Select four corners")
    quality_level = cv2.getTrackbarPos("Quality Level", "Select four corners")
    min_distance = cv2.getTrackbarPos("Min Distance", "Select four corners")
    circle_size = cv2.getTrackbarPos("Circles", "Select four corners")
    accent_value = cv2.getTrackbarPos("Accent", "Select four corners")

    if max_corners == 0:
        max_corners = 1  # Ensure at least 1 corner is detected

    if quality_level == 0:
        quality_level == 1  # A quality level of 0 is invalid

    detected_corners = detect_corners(img, max_corners, quality_level, min_distance)
    tone_adjusted = apply_accent_adjustment(img, accent_value)
    resized, dim = image_resize(tone_adjusted, DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)
    display_width, display_height = dim

    for corner in detected_corners:
        x, y = corner.ravel().astype(int)

        # White circle shadow
        cv2.circle(
            resized,
            (round(x / (image_width / display_width)), round(y / (image_height / display_height))),
            circle_size + 1,
            (255, 255, 255),
            -1,
        )

        # Red circle
        cv2.circle(
            resized,
            (round(x / (image_width / display_width)), round(y / (image_height / display_height))),
            circle_size,
            (0, 0, 255),
            -1,
        )

    cv2.imshow("Select four corners", resized)


def save_image(img):
    """Save image as file."""
    global input_file

    unique_path = get_unique_path(input_file.parent / input_file.stem)
    output_file = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*"),
        ],
        initialdir=unique_path.parent,
        initialfile=unique_path.name,
    )
    
    output_file = Path(output_file)
    if output_file and str(output_file) != ".":
        buffer = io.BytesIO()

        success, encoded = cv2.imencode(output_file.suffix, img)

        if success:
            buffer.write(encoded.tobytes())
            buffer.seek(0)

            with open(str(output_file), "wb") as f:
                f.write(buffer.getvalue())
        else:
            raise RuntimeError("Error encoding image.")
    
        cv2.destroyAllWindows()
        raise SystemExit(0)


def main():
    global img, image_height, image_width, input_file, output_dir, detected_corners

    parser = argparse.ArgumentParser(description="Quick scan cropping application")
    parser.add_argument("input_file", type=str, help="image file path", nargs="?")
    args = parser.parse_args()

    input_file = args.input_file
    if not input_file:
        input_file = filedialog.askopenfilename(
            filetypes=[("Image files", "*.tif *.jpg *.png *.bmp *.jxl")]
        )

    if not args.input_file and not input_file:
        raise RuntimeError("no file selected")

    input_file = Path(input_file)
    if input_file.suffix.lower() == ".jxl":
        ffmpeg_path = shutil.which("ffmpeg")    
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found on PATH")

        result = subprocess.run(
            [ffmpeg_path, "-i", str(input_file), "-f", "image2pipe", "-vcodec", "png", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        image_bytes = result.stdout
    else:
        with open(input_file, "rb") as f:
            image_bytes = f.read()
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot load image from path {input_file}")

    image_height, image_width = img.shape[:2]

    detected_corners = detect_corners(img, DEFAULT_MAX_CORNERS, DEFAULT_QUALITY_LEVEL, DEFAULT_MIN_DISTANCE)
    tone_adjusted = apply_accent_adjustment(img, DEFAULT_ACCENT_VALUE)
    resized, dim = image_resize(tone_adjusted, DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)

    cv2.namedWindow("Select four corners")
    cv2.setWindowProperty("Select four corners", cv2.WND_PROP_TOPMOST, 1)

    cv2.createTrackbar("Max Corners", "Select four corners", DEFAULT_MAX_CORNERS, 1000, update)
    cv2.createTrackbar("Quality Level", "Select four corners", DEFAULT_QUALITY_LEVEL, 100, update)
    cv2.createTrackbar("Min Distance", "Select four corners", DEFAULT_MIN_DISTANCE, 200, update)
    cv2.createTrackbar("Circles", "Select four corners", circle_size, 20, update)
    cv2.createTrackbar("Accent", "Select four corners", DEFAULT_ACCENT_VALUE, 30, update)
    cv2.createTrackbar("Custom", "Select four corners", 0, 1, update)


    cv2.imshow("Select four corners", resized)
    cv2.setMouseCallback("Select four corners", click_handler)
    update()

    while True:
        if (cv2.getWindowProperty("Select four corners", cv2.WND_PROP_VISIBLE) < 1 and
            cv2.getWindowProperty("Cropped image", cv2.WND_PROP_VISIBLE) < 1): 
            break

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord("r"):  # R to rotate CW
            rotate_image(1)
        elif key == ord("e"):  # E to rotate CCW
            rotate_image(0)
        elif key == ord("w"):  # Crop top
            crop_image("top")
        elif key == ord("s"):  # Crop bottom
            crop_image("bottom")
        elif key == ord("a"):  # Crop left
            crop_image("left")
        elif key == ord("d"):  # Crop right
            crop_image("right")
        elif key == ord("q"):  # Q to save
            save_image(warped)
        elif key == 9:  # Tab to undo
            undo_image()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
