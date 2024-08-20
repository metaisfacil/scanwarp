# ScanWarp

Quick scan cropping application. Intended as a workflow improvement for processing scans of album inserts.

## Requirements

- Python 3.6+
- OpenCV (`cv2`)
- FFmpeg (`.jxl` support)

## Installation

To run this script, you need to have Python and the required libraries installed:

```bash
pip install opencv-python
```

## Usage

1. On execution, you will be prompted for an input image.

3. A preview of the selected scan will appear.

   - Use the **mouse** to click on the corners of the insert you wish to crop.
   - Use the **trackbars** to adjust block size, aperture, `k` value for corner detection, circle sizes, and tonal value.
   - Disable corner detection by increasing `Custom`.

4. **Cropping & Rotating**: Once the corners are selected, the image will be cropped and displayed. You can further crop or rotate the image using the following keys:
   
   - **WASD**: Crop the image (Top/Bottom/Left/Right).
   - **E/R**: Rotate the image 90 degrees (CCW/CW).
   - **Q**: Save the final image.

Please note Ctrl+S will save only the low-resolution preview. This is a hard-coded limitation of CV2 image previews.

## Notes

- `.jxl` images require FFmpeg for conversion. Ensure FFmpeg is installed and accessible via PATH.
- The script provides a unique filename for saving the image to prevent overwriting.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.