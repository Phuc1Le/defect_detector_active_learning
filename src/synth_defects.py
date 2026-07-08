import argparse, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_mask(img):
    """Return a binary mask for the main foreground object in an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.01 * img.shape[0] * img.shape[1]:
        return np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return (mask > 0).astype(np.uint8)


def add_scratch(img, mask=None):
    h, w = img.shape[:2]
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)

    valid_y, valid_x = np.where(mask > 0)
    if len(valid_y) == 0:
        return img.copy()

    start_idx = random.randrange(len(valid_y))
    x, y = int(valid_x[start_idx]), int(valid_y[start_idx])
    overlay = img.copy()
    pts = [(x, y)]
    for _ in range(random.randint(2, 5)):
        x = int(np.clip(x + random.randint(-w // 4, w // 4), 0, w - 1))
        y = int(np.clip(y + random.randint(-h // 4, h // 4), 0, h - 1))
        if mask[y, x] == 0:
            # fall back to a nearby valid point if the random step leaves the mask
            nearby = np.argwhere(mask > 0)
            if len(nearby) == 0:
                break
            choice = nearby[random.randrange(len(nearby))]
            x, y = int(choice[1]), int(choice[0])
        pts.append((x, y))

    pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    local = img[y, x].astype(np.float32)
    if random.random() < 0.5:
        color = np.clip(local * random.uniform(0.75, 0.95), 0, 255)
    else:
        color = np.clip(local * random.uniform(1.05, 1.25), 0, 255)
    color = tuple(int(c) for c in color)
    cv2.polylines(
        overlay,
        [pts],
        False,
        color,
        thickness=random.choice([1, 1, 1, 2]),
        lineType=cv2.LINE_AA,
    )
    alpha = random.uniform(0.2, 0.4)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def add_blob(img, mask=None):
    h, w = img.shape[:2]
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)

    valid_y, valid_x = np.where(mask > 0)
    if len(valid_y) == 0:
        return img.copy()

    start_idx = random.randrange(len(valid_y))
    cx, cy = int(valid_x[start_idx]), int(valid_y[start_idx])

    blob_mask = np.zeros((h, w), dtype=np.float32)
    axes = (random.randint(5, max(6, w // 10)), random.randint(5, max(6, h // 10)))
    cv2.ellipse(blob_mask, (cx, cy), axes, random.randint(0, 360), 0, 360, 1.0, -1)
    blob_mask = cv2.GaussianBlur(blob_mask, (0, 0), sigmaX=random.uniform(2, 6))
    if blob_mask.max() > 0:
        blob_mask = (blob_mask / blob_mask.max()) * random.uniform(0.4, 0.8)

    blob_mask *= mask.astype(np.float32)
    if blob_mask.max() > 0:
        blob_mask = (blob_mask / blob_mask.max()) * random.uniform(0.4, 0.8)
    
    local = img[cy, cx].astype(np.float32)

    defect_type = random.choices(
        ["dirt", "scuff", "discoloration"],
        weights=[0.5, 0.3, 0.2],
    )[0]

    if defect_type == "dirt":
        factor = random.uniform(0.4, 0.8)
        color = local * factor

    elif defect_type == "scuff":
        factor = random.uniform(1.2, 1.5)
        color = np.clip(local * factor, 0, 255)

    else:  # discoloration
        delta = random.randint(-30, 30)
        color = np.clip(local + delta, 0, 255)

    m3 = blob_mask[..., None]
    out = img.astype(np.float32) * (1 - m3) + color * m3
    return out.astype(np.uint8)


def make_defect(img, mask=None):
    return random.choice([add_scratch, add_blob])(img, mask=mask)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out_dir", default="data/synthetic")
    ap.add_argument("--out_manifest", default="synthetic_manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    src = df[df.split == "synth_source"]

    rows = []
    for _, r in tqdm(src.iterrows(), total=len(src), desc="generating defects"):
        img = cv2.imread(r.filepath)
        if img is None:
            continue
        # keep the clean original as a clean training example (label 0)
        rows.append({"filepath": r.filepath, "label": 0})
        # and make a synthetic defective version (label 1)
        mask = get_mask(img)
        defective = make_defect(img, mask=mask)
        name = Path(r.filepath).stem + "_defect.png"
        dst = out_dir / name
        cv2.imwrite(str(dst), defective)
        rows.append({"filepath": str(dst), "label": 1})

    pd.DataFrame(rows).to_csv(args.out_manifest, index=False)
    print(f"Wrote {len(rows)} rows to {args.out_manifest} "
          f"({sum(r['label']==1 for r in rows)} synthetic defects)")

if __name__ == "__main__":
    main()