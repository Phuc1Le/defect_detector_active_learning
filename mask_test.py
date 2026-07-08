import sys
from pathlib import Path

import cv2

sys.path.append("src")
from synth_defects import get_mask


def main():
    image_dir = Path("data/raw/bottle/train/good")
    image_paths = sorted(image_dir.glob("*.png"))[:5]

    if not image_paths:
        raise FileNotFoundError(f"No sample images found in {image_dir}")

    print("Testing get_mask on sample bottle images...")
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        mask = get_mask(img)
        overlay = img.copy()
        overlay[mask == 0] = [255, 0, 0]
        cv2.imwrite("mask_preview.png", overlay)
        mask_pixels = int(mask.sum())
        print(f"{path.name}: mask_pixels={mask_pixels}, shape={mask.shape}")

        if mask_pixels <= 0:
            raise ValueError(f"Mask is empty for {path.name}")

        if mask_pixels >= img.shape[0] * img.shape[1]:
            raise ValueError(f"Mask covers the whole image for {path.name}")

    print("All get_mask checks passed.")


if __name__ == "__main__":
    main()
