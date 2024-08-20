import argparse
import cv2
import io
import numpy as np
import shutil
import subprocess
from pathlib import Path
from tkinter import filedialog


__version__ = "0.01.20240820"

DEFAULT_DISPLAY_WIDTH = 800
DEFAULT_DISPLAY_HEIGHT = 800
DEFAULT_BLOCK_SIZE = 16
DEFAULT_APERTURE_SIZE = 25
DEFAULT_K = 12
DEFAULT_ACCENT_VALUE = 0

circle_size = 3
crop_top = 0
crop_bottom = 0
crop_left = 0
crop_right = 0
detected_corners = []
selected_corners = []
display_width = DEFAULT_DISPLAY_WIDTH
display_height = DEFAULT_DISPLAY_HEIGHT
undo_state = None


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


def detect_corners(img, block_size, ksize, k):
    """Detect corners on given `img`."""
    global detected_corners

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, ksize, k / 100.0)

    ret, dst = cv2.threshold(dst, 0.001 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    detected_corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    return detected_corners


def apply_accent_adjustment(img, accent_value, *args):
    """Apply accent adjustment to image."""
    adjusted = img.copy()
    adjusted = np.clip(adjusted + accent_value, 0, 255).astype(np.uint8)
    return adjusted


def rotate_image(flip_code):
    """Rotate image by 90 degrees and display."""
    global warped, last_state, display_width, display_height

    last_state = warped

    rotated = cv2.transpose(warped)
    rotated = cv2.flip(rotated, flipCode=flip_code)
    warped = rotated

    resized, dim = image_resize(warped, width=DEFAULT_DISPLAY_WIDTH, height=DEFAULT_DISPLAY_HEIGHT)
    display_width, display_height = dim

    cv2.namedWindow("Cropped image")
    cv2.imshow("Cropped image", resized)


def crop_image(direction):
    """Crop image by one row/column of pixels and display."""
    global warped, last_state, crop_top, crop_bottom, crop_left, crop_right, display_width, display_height

    last_state = warped

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
    global warped, last_state

    if last_state is None:
        return
    
    warped = last_state
    last_state = None

    resized, dim = image_resize(warped, width=DEFAULT_DISPLAY_WIDTH, height=DEFAULT_DISPLAY_HEIGHT)

    cv2.imshow("Cropped image", resized)


def update(*args):
    """Update image display based on trackbar settings."""
    global img, circle_size, detected_corners, display_width, display_height

    block_size = cv2.getTrackbarPos("Block Size", "Select four corners")
    ksize = cv2.getTrackbarPos("Aperture", "Select four corners")
    k = cv2.getTrackbarPos("K", "Select four corners")
    circle_size = cv2.getTrackbarPos("Circles", "Select four corners")
    accent_value = cv2.getTrackbarPos("Accent", "Select four corners")

    if block_size % 2 == 0:
        block_size += 1
    if ksize % 2 == 0:
        ksize += 1

    detected_corners = detect_corners(img, block_size, ksize, k)
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

    detected_corners = detect_corners(img, DEFAULT_BLOCK_SIZE, DEFAULT_APERTURE_SIZE, DEFAULT_K)
    tone_adjusted = apply_accent_adjustment(img, DEFAULT_ACCENT_VALUE)
    resized, dim = image_resize(tone_adjusted, DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)

    cv2.namedWindow("Select four corners")
    cv2.setWindowProperty("Select four corners", cv2.WND_PROP_TOPMOST, 1)

    cv2.createTrackbar("Block Size", "Select four corners", DEFAULT_BLOCK_SIZE, 50, update)
    cv2.createTrackbar("Aperture", "Select four corners", DEFAULT_APERTURE_SIZE, 50, update)
    cv2.createTrackbar("K", "Select four corners", DEFAULT_K, 100, update)
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
