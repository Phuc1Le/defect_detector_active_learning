import argparse, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
DEBUG = False

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
    x_box, y_box, w_box, h_box = cv2.boundingRect(pts)
    local = img[y, x].astype(np.float32)
    brightness = local.mean()

    if brightness < 100:
        # dark area -> make scratch lighter
        color = np.clip(local + random.randint(50, 90), 0, 255)
    else:
        # bright area -> make scratch darker
        color = np.clip(local - random.randint(40, 80), 0, 255)

    color = tuple(int(c) for c in color)
    overlay = cv2.GaussianBlur(overlay, (3,3), 0)
    cv2.polylines(
        overlay,
        [pts],
        False,
        color,
        thickness=random.choice([1, 1, 1, 2]),
        lineType=cv2.LINE_AA,
    )
    alpha = random.uniform(0.4, 0.7)
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    if DEBUG:
        cv2.rectangle(
            result,
            (x_box, y_box),
            (x_box + w_box, y_box + h_box),
            (0, 0, 255),   # red
            1,
        )
    return result


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
    if DEBUG:
        cv2.circle(out, (cx, cy), 4, (0,0,255), -1)
    return out.astype(np.uint8)

def add_crack(img, mask=None):
    h, w = img.shape[:2]
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img.copy()

    idx = random.randrange(len(xs))
    x = int(xs[idx])
    y = int(ys[idx])

    overlay = img.copy()

    angle = random.uniform(0, 2 * np.pi)
    pts = [(x, y)]

    n_segments = random.randint(12, 25)

    for _ in range(n_segments):
        angle += random.uniform(-0.4, 0.4)

        step = random.randint(4, 8)

        x = int(np.clip(x + step * np.cos(angle), 0, w - 1))
        y = int(np.clip(y + step * np.sin(angle), 0, h - 1))

        if mask[y, x] == 0:
            break

        pts.append((x, y))

    pts = np.array(pts, np.int32).reshape(-1, 1, 2)
    local = img[pts[0, 0, 1], pts[0, 0, 0]].astype(np.float32)
    brightness = local.mean()

    if brightness < 100:
        # dark area -> make scratch lighter
        color = np.clip(local + random.randint(50, 90), 0, 255)
    else:
        # bright area -> make scratch darker
        color = np.clip(local - random.randint(40, 80), 0, 255)

    color = tuple(int(c) for c in color)
    overlay = cv2.GaussianBlur(overlay, (3,3), 0)
    cv2.polylines(
        overlay,
        [pts],
        False,
        color,
        thickness=random.choice([1,2,2]),
        lineType=cv2.LINE_AA,
    )

    # Optional tiny branch
    if len(pts) > 8 and random.random() < 0.5:
        branch_start = pts[random.randint(3, len(pts) - 4)][0]
        angle += random.uniform(-1.0, 1.0)
        bx, by = int(branch_start[0]), int(branch_start[1])
        branch = [(bx, by)]
        for _ in range(random.randint(4, 8)):
            angle += random.uniform(-0.5, 0.5)
            bx = int(np.clip(bx + 5 * np.cos(angle), 0, w - 1))
            by = int(np.clip(by + 5 * np.sin(angle), 0, h - 1))
            if mask[by, bx] == 0:
                break
            branch.append((bx, by))
        if len(branch) > 2:
            branch = np.array(branch, np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                overlay,
                [branch],
                False,
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
    alpha = random.uniform(0.5, 0.8)
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    x_box, y_box, w_box, h_box = cv2.boundingRect(pts)
    if DEBUG:
        cv2.rectangle(
            result,
            (x_box, y_box),
            (x_box + w_box, y_box + h_box),
            (0, 0, 255),   # red
            1,
        )
    return result

def add_scuff(img, mask=None):
    h, w = img.shape[:2]
    if mask is None:
        mask = np.ones((h, w), dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img.copy()
    idx = random.randrange(len(xs))
    cx = int(xs[idx])
    cy = int(ys[idx])
    overlay = img.copy()
    rx = random.randint(15, 40)
    ry = random.randint(10, 25)
    local = img[cy, cx].astype(np.float32)
    brightness = local.mean()

    if brightness < 100:
        # dark area -> make scratch lighter
        color = np.clip(local + random.randint(50, 90), 0, 255)
    else:
        # bright area -> make scratch darker
        color = np.clip(local - random.randint(40, 80), 0, 255)

    color = tuple(int(c) for c in color)

    for _ in range(random.randint(50, 120)):
        x1 = cx + random.randint(-rx, rx)
        y1 = cy + random.randint(-ry, ry)
        x2 = x1 + random.randint(-6, 6)
        y2 = y1 + random.randint(-6, 6)
        if (
            0 <= x1 < w and
            0 <= y1 < h and
            mask[y1, x1]
        ):
            cv2.line(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                1,
                cv2.LINE_AA,
            )
    overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
    alpha = random.uniform(0.3, 0.7)
    result = cv2.addWeighted(
        overlay,
        alpha,
        img,
        1 - alpha,
        0,
    )
    if DEBUG:
        cv2.rectangle(
            result,
            (cx - rx, cy - ry),
            (cx + rx, cy + ry),
            (0, 255, 255),   # yellow for scuff
            1,
        )
    return result

def make_defect(img, mask=None):
    return random.choice([add_scratch, add_blob, add_crack, add_scuff])(img, mask=mask)


DEFECTS = [
    add_scratch,
    add_blob,
    add_crack,
    add_scuff,
]

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
        mask = get_mask(img)
        N_SYNTH_PER_IMAGE = 3

        for i in range(N_SYNTH_PER_IMAGE):
            defective = img.copy()
            n_defects = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1], k=1)[0]

            for defect_fn in random.sample(DEFECTS, k=n_defects):
                defective = defect_fn(defective, mask=mask)
            name = f"{Path(r.filepath).stem}_defect_{i}.png"
            dst = out_dir / name
            cv2.imwrite(str(dst), defective)

            rows.append({
                "filepath": str(dst),
                "label": 1,
            })

    pd.DataFrame(rows).to_csv(args.out_manifest, index=False)
    print(f"Wrote {len(rows)} rows to {args.out_manifest} "
          f"({sum(r['label']==1 for r in rows)} synthetic defects)")

if __name__ == "__main__":
    main()