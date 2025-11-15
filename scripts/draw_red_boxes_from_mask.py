import os
import argparse
import cv2
import numpy as np
import csv

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("can't read image: " + path)
    # if has alpha, composite on white
    if img.shape[2] == 4:
        alpha = img[:, :, 3].astype(float) / 255.0
        bgr = img[:, :, :3].astype(float)
        bg = 255.0 * np.ones_like(bgr)
        comp = (bgr * alpha[:, :, None] + bg * (1 - alpha[:, :, None])).astype(np.uint8)
        return comp
    return img

def estimate_border_bg(img, border_px=10):
    h, w = img.shape[:2]
    top = img[0:border_px, :, :].reshape(-1,3)
    bot = img[h-border_px:h, :, :].reshape(-1,3)
    left = img[:, 0:border_px, :].reshape(-1,3)
    right = img[:, w-border_px:w, :].reshape(-1,3)
    samples = np.vstack([top, bot, left, right])
    # median is robust to small colored highlights at border
    return np.median(samples, axis=0).astype(int)

def color_dist_sq(a, b):
    d = a.astype(int) - b.astype(int)
    return (d*d).sum(axis=-1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", required=True, help="mask PNG with highlights")
    p.add_argument("--out-img", help="output annotated image", default=None)
    p.add_argument("--out-csv", help="output CSV", default=None)
    p.add_argument("--sat", type=int, default=30, help="HSV saturation threshold")
    p.add_argument("--val", type=int, default=100, help="HSV value (brightness) threshold")
    p.add_argument("--close", type=int, default=5, help="morphological close kernel size (0=off)")
    p.add_argument("--dilate", type=int, default=3, help="dilate to merge nearby pixels (0=off)")
    p.add_argument("--min-area", type=int, default=50, help="ignore tiny regions")
    p.add_argument("--bg-tol", type=int, default=30, help="background color tolerance (Euclidean in RGB)")
    p.add_argument("--max-area-frac", type=float, default=0.4, help="ignore regions larger than this fraction of image area")
    args = p.parse_args()

    mask_path = args.mask
    base = os.path.splitext(os.path.basename(mask_path))[0]
    out_img = args.out_img or os.path.join(os.path.dirname(mask_path), base + "_boxed.png")
    out_csv = args.out_csv or os.path.join(os.path.dirname(mask_path), base + "_boxes.csv")
    debug_mask_out = os.path.splitext(out_img)[0] + "_debug_mask.png"

    img_bgr = load_img(mask_path)           # BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    h, w = s.shape
    img_area = h * w

    # estimate background from borders
    bg_col = estimate_border_bg(img_bgr, border_px=max(4, min(20, int(min(h,w)*0.02))))
    # build initial highlight mask by HSV thresholds
    mask_hsv = (s > args.sat) & (v > args.val)

    # exclude pixels similar to background
    dist2 = color_dist_sq(img_rgb, bg_col)
    bg_sim = (dist2 <= (args.bg_tol * args.bg_tol))
    mask = np.logical_and(mask_hsv, ~bg_sim).astype('uint8') * 255

    # morphology
    if args.close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.close, args.close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if args.dilate > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
        mask = cv2.dilate(mask, k2, iterations=1)

    # debug mask save (so you can inspect)
    cv2.imwrite(debug_mask_out, mask)
    print("Wrote debug mask:", debug_mask_out)

    # find contours (external)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_out = img_bgr.copy()
    rows = []
    cid = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < args.min_area:
            continue
        # discard giant region (likely background or page box)
        if area > args.max_area_frac * img_area:
            # skip overly-large contour
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        cid += 1
        # draw red box (B,G,R)
        cv2.rectangle(img_out, (x, y), (x+ww, y+hh), (0,0,255), 2)
        rows.append((cid, x, y, ww, hh, int(area)))

    # save outputs
    os.makedirs(os.path.dirname(out_img) or ".", exist_ok=True)
    cv2.imwrite(out_img, img_out)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","x","y","width","height","area"])
        writer.writerows(rows)

    print("Wrote:", out_img, out_csv, "regions:", len(rows))
    # helpful summary
    print(f"image area={img_area}, contours found={len(contours)}, kept regions={len(rows)}")
    print("background color guessed (RGB):", tuple(int(c) for c in bg_col))

if __name__ == "__main__":
    main()