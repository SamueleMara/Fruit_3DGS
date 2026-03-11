import os
import cv2
from pathlib import Path

class ImageResizer:
    """
    Resize images in a folder to multiple downscaled versions.
    """

    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
        self.images_folder = self.source_path / "images"
        self.scales = [1, 2, 4, 8]  # downscale factors
        self.dest_folders = {
            scale: self.source_path / f"images_{scale}" for scale in self.scales
        }
        for folder in self.dest_folders.values():
            folder.mkdir(parents=True, exist_ok=True)

    def resize_all(self):
        files = [f for f in os.listdir(self.images_folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for file in files:
            src_file = self.images_folder / file
            img = cv2.imread(str(src_file))
            if img is None:
                print(f"[WARNING] Failed to read {src_file}, skipping.")
                continue

            for scale in self.scales:
                dst_file = self.dest_folders[scale] / file
                if scale == 1:
                    resized = img.copy()
                else:
                    new_w = img.shape[1] // scale
                    new_h = img.shape[0] // scale
                    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst_file), resized)
                print(f"[INFO] Saved {dst_file} ({resized.shape[1]}x{resized.shape[0]})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Resize images in a folder to multiple resolutions")
    parser.add_argument("source_path", type=str, help="Path to folder containing 'images' subfolder")
    args = parser.parse_args()

    resizer = ImageResizer(args.source_path)
    resizer.resize_all()
    print("[INFO] All images resized.")


if __name__ == "__main__":
    main()
