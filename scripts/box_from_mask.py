import os
import argparse
from PIL import Image
import numpy as np
import cv2
import csv

def load_rgb(path):
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)  # H,W,4
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3].copy()
        rgb[alpha == 0] = [0, 0, 0]
    else:
        rgb = arr[:, :, :3].copy()
    return rgb

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", required=True)
    p.add_argument("--out-img", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--min-area", type=int, default=20)
    p.add_argument("--tol", type=int, default=8, help="color tolerance for background")
    p.add_argument("--close", type=int, default=0, help="morphological close kernel size (0=off)")
    p.add_argument("--dilate", type=int, default=0, help="dilate pixels to merge nearby ink (0=off)")
    args = p.parse_args()

    mask_path = args.mask
    base = os.path.splitext(os.path.basename(mask_path))[0]
    out_img = args.out_img or os.path.join(os.path.dirname(mask_path), base + "_boxed.png")
    out_csv = args.out_csv or os.path.join(os.path.dirname(mask_path), base + "_boxes.csv")

    rgb = load_rgb(mask_path)
    h, w = rgb.shape[:2]

    # binary mask: pixel is foreground if any channel > tol
    nonbg = np.any(rgb > [args.tol, args.tol, args.tol], axis=2).astype('uint8') * 255

    # optional morphological close to fill holes
    if args.close > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (args.close, args.close))
        nonbg = cv2.morphologyEx(nonbg, cv2.MORPH_CLOSE, kern)

    # optional dilation to ensure separated glyphs join
    if args.dilate > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (args.dilate, args.dilate))
        nonbg = cv2.dilate(nonbg, kern, iterations=1)

    # find external contours on processed mask (original resolution)
    contours, _ = cv2.findContours(nonbg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_draw = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    rows = []
    cid = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < args.min_area:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        cid += 1
        # draw red box
        cv2.rectangle(img_draw, (x, y), (x + ww, y + hh), (0, 0, 255), 2)
        cv2.putText(img_draw, str(cid), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        rows.append((cid, x, y, ww, hh, int(area)))

    os.makedirs(os.path.dirname(out_img) or ".", exist_ok=True)
    cv2.imwrite(out_img, img_draw)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","x","y","width","height","area"])
        for r in rows:
            writer.writerow(r)

    print("Wrote:", out_img, out_csv, "regions:", len(rows))

if __name__ == "__main__":
    main()