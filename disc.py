import argparse
import cv2
import io
import numpy as np
import shutil
import subprocess
from pathlib import Path
from tkinter import filedialog

background_color = (255, 255, 255)
crop = None
crop_ready = False
drawing = False
drawing_complete = False
feathered_full_res_crop = None
hover_pos = (0, 0)
resize_factor = 8
rotated_crop = None
rotation_angle = 0
start_point = (0, 0)


def rotate_image(image, angle):
    global background_color

    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform affine transformation with the current background color
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_color,  # Use the global background color
    )
    return rotated


def apply_circular_mask_with_feather(crop, center, radius, feather_size=15):
    global background_color  # Use the current background color

    h, w = crop.shape[:2]
    background = np.full((h, w, 3), background_color, dtype=np.uint8)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    mask = cv2.GaussianBlur(mask, (2 * feather_size + 1, 2 * feather_size + 1), feather_size)
    
    alpha = mask.astype(float) / 255.0

    feathered_crop = (crop * alpha[..., None] + background * (1 - alpha[..., None])).astype(np.uint8)
    return feathered_crop


def draw_circle(event, x, y, flags, param):
    global crop, crop_ready, drawing, drawing_complete, feathered_full_res_crop
    global full_res_center, full_res_crop, img_resized, img, orig_radius
    global rotation_angle, rotated_crop, start_point, temp_img, hover_pos

    hover_pos = (x, y)  # Update the hover position

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_img = img_resized.copy()
        radius = int(((start_point[0] - x) ** 2 + (start_point[1] - y) ** 2) ** 0.5)
        center = (x, y)
        cv2.circle(temp_img, center, radius, (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        radius = int(((start_point[0] - x) ** 2 + (start_point[1] - y) ** 2) ** 0.5)
        center = (x, y)

        orig_center_x = int(center[0] * resize_factor)
        orig_center_y = int(center[1] * resize_factor)
        orig_radius = int(radius * resize_factor)

        x1 = max(0, orig_center_x - orig_radius - 15)
        y1 = max(0, orig_center_y - orig_radius - 15)
        x2 = min(img.shape[1], orig_center_x + orig_radius + 15)
        y2 = min(img.shape[0], orig_center_y + orig_radius + 15)

        full_res_crop = img[y1:y2, x1:x2].copy()

        full_res_center = (orig_center_x - x1, orig_center_y - y1)
        feathered_full_res_crop = apply_circular_mask_with_feather(
            full_res_crop, full_res_center, orig_radius, feather_size=15
        )
        rotated_crop = feathered_full_res_crop

        resized_crop = cv2.resize(
            feathered_full_res_crop,
            (
                feathered_full_res_crop.shape[1] // resize_factor,
                feathered_full_res_crop.shape[0] // resize_factor,
            ),
            interpolation=cv2.INTER_AREA,
        )
        crop_ready = True
        rotation_angle = 0
        cv2.imshow("Cropped image", resized_crop)

        drawing_complete = True
        cv2.destroyWindow("Draw circle around disc")


def update_rotate():
    global feathered_full_res_crop, full_res_crop, full_res_center, orig_radius, resize_factor, resized_rotated_crop, rotated_crop, rotation_angle, background_color

    feathered_full_res_crop = apply_circular_mask_with_feather(
        full_res_crop, full_res_center, orig_radius, feather_size=15
    )

    rotated_crop = rotate_image(feathered_full_res_crop, rotation_angle)

    resized_rotated_crop = cv2.resize(
        rotated_crop,
        (
            rotated_crop.shape[1] // resize_factor,
            rotated_crop.shape[0] // resize_factor,
        ),
        interpolation=cv2.INTER_AREA,
    )

    cv2.imshow("Cropped image", resized_rotated_crop)


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
    global background_color, crop_ready, detected_corners, full_res_center, img, img_resized, image_height, image_width, input_file, output_dir, orig_radius, rotated_crop, rotation_angle, temp_img

    parser = argparse.ArgumentParser(description="Quick disc cropping application")
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
            [
                ffmpeg_path,
                "-i",
                str(input_file),
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
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

    new_width = img.shape[1] // resize_factor
    new_height = img.shape[0] // resize_factor
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    temp_img = img_resized.copy()

    cv2.namedWindow("Draw circle around disc")
    cv2.setMouseCallback("Draw circle around disc", draw_circle)

    while True:
        if not drawing_complete:
            cv2.imshow("Draw circle around disc", temp_img)

        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC to exit
            break
        elif crop_ready:
            if key == ord("w"):  # W to rotate 45°
                rotation_angle += 45
                update_rotate()
            elif key == ord("e"):  # E to rotate -1°
                rotation_angle += 1
                update_rotate()
            elif key == ord("r"):  # R to rotate 1°
                rotation_angle -= 1
                update_rotate()
            elif key == ord("t"):  # T to rotate -45°
                rotation_angle -= 45
                update_rotate()
            elif key == ord("y"):  # Y to set the background to the hovered pixel color
                if 0 <= hover_pos[1] < img_resized.shape[0] and 0 <= hover_pos[0] < img_resized.shape[1]:
                    pixel_color = img_resized[hover_pos[1], hover_pos[0]]
                    background_color = tuple(map(int, pixel_color))
                    feathered_full_res_crop = apply_circular_mask_with_feather(
                        full_res_crop, full_res_center, orig_radius, feather_size=15
                    )
                    rotated_crop = feathered_full_res_crop
                    update_rotate()
            elif key == ord("q"):  # Q to save
                save_image(rotated_crop)
                crop_ready = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
