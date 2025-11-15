import os, argparse, csv
import cv2
import numpy as np

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("can't read: " + path)
    # composite alpha on white if present
    if img.shape[2] == 4:
        alpha = img[:,:,3].astype(float)/255.0
        bgr = img[:,:,:3].astype(float)
        bg = 255.0 * np.ones_like(bgr)
        img = (bgr * alpha[:,:,None] + bg * (1-alpha[:,:,None])).astype(np.uint8)
    return img

def color_mask_rgb(img_rgb, targets, thresholds):
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for (r,g,b), thr in zip(targets, thresholds):
        tgt = np.array([r,g,b], dtype=np.int16)
        diff = img_rgb.astype(np.int16) - tgt[None,None,:]
        dist2 = np.sum(diff*diff, axis=2)
        mask = np.logical_or(mask, dist2 <= thr*thr)
    return (mask.astype(np.uint8) * 255)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", required=True, help="mask PNG with highlights")
    p.add_argument("--out-img", help="output annotated image", default=None)
    p.add_argument("--orig-image", help="optional original drawing to draw boxes on (PNG/JPG/PDF->convert)", default=None)
    p.add_argument("--out-csv", help="output CSV", default=None)
    p.add_argument("--close", type=int, default=7, help="morph close kernel")
    p.add_argument("--dilate", type=int, default=4, help="dilate kernel")
    p.add_argument("--min-area", type=int, default=60, help="ignore tiny regions")
    p.add_argument("--thr-yellow", type=int, default=60, help="threshold for pale yellow")
    p.add_argument("--thr-dkyellow", type=int, default=50, help="threshold for darker yellow")
    p.add_argument("--thr-blue", type=int, default=70, help="threshold for light blue")
    p.add_argument("--debug-dir", default=None, help="write debug mask files here")
    args = p.parse_args()

    img_bgr = load_img(args.mask)          # mask image (colored/highlight)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # prepare drawing background: prefer user-provided original image if given
    if args.orig_image:
        bg = cv2.imread(args.orig_image, cv2.IMREAD_UNCHANGED)
        if bg is None:
            print("warning: can't read orig-image, falling back to mask image as background:", args.orig_image)
            img_out = img_bgr.copy()
        else:
            # composite alpha if present
            if bg.ndim == 2:
                img_out = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            elif bg.shape[2] == 4:
                alpha = bg[:, :, 3].astype(float) / 255.0
                bgr = bg[:, :, :3].astype(float)
                bg_white = 255.0 * np.ones_like(bgr)
                img_out = (bgr * alpha[:, :, None] + bg_white * (1 - alpha[:, :, None])).astype(np.uint8)
            else:
                img_out = bg.copy()
    else:
        # use the mask image RGB as background (so outlines appear over colored highlights)
        img_out = img_bgr.copy()

    h,w = img_rgb.shape[:2]

    # target colors (RGB)
    targets = [
        (247,242,212),  # pale yellow/beige
        (187,185,110),  # darker yellow
        (200,206,228),  # light bluish
    ]
    thresholds = [args.thr_yellow, args.thr_dkyellow, args.thr_blue]

    # build combined mask and per-target masks for debugging
    per_masks = []
    for (r,g,b), thr in zip(targets, thresholds):
        tgt = np.array([r,g,b], dtype=np.int16)
        diff = img_rgb.astype(np.int16) - tgt[None,None,:]
        dist2 = np.sum(diff*diff, axis=2)
        per_masks.append((dist2 <= thr*thr).astype(np.uint8)*255)
    mask = np.zeros_like(per_masks[0])
    for m in per_masks:
        mask = cv2.bitwise_or(mask, m)

    # debug: save per-target masks and combined mask
    debug_dir = args.debug_dir or os.path.dirname(args.mask)
    os.makedirs(debug_dir or ".", exist_ok=True)
    for i,m in enumerate(per_masks):
        cv2.imwrite(os.path.join(debug_dir, f"debug_mask_target{i+1}.png"), m)
    cv2.imwrite(os.path.join(debug_dir, "debug_mask_combined.png"), mask)
    print("Wrote debug masks to", debug_dir)
    print("white px per target:", [int((m>0).sum()) for m in per_masks], "combined:", int((mask>0).sum()))

    # morphology to join highlight fragments
    if args.close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.close, args.close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if args.dilate > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
        mask = cv2.dilate(mask, k2, iterations=1)

    # save post-morph mask for inspection
    cv2.imwrite(os.path.join(debug_dir, "debug_mask_post_morph.png"), mask)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("contours found:", len(contours))

    rows = []
    cid = 0
    img_area = h * w
    areas = []
    for c in contours:
        area = cv2.contourArea(c)
        areas.append(area)
    areas_sorted = sorted(areas, reverse=True)[:10]
    print("top contour areas:", areas_sorted)

    for c in contours:
        area = cv2.contourArea(c)
        if area < args.min_area:
            continue
        # skip huge page artifacts
        if area > 0.9 * img_area:
            continue
        x,y,ww,hh = cv2.boundingRect(c)
        cid += 1
        # draw outline only (thickness > 0). Do NOT use thickness=-1 (filled).
        cv2.rectangle(img_out, (x,y), (x+ww, y+hh), (0,0,255), 2)  # red outline (B,G,R)
        rows.append((cid, x, y, ww, hh, int(area)))

    out_img = args.out_img or os.path.splitext(args.mask)[0] + "_colorboxed.png"
    out_csv = args.out_csv or os.path.splitext(args.mask)[0] + "_color_boxes.csv"
    os.makedirs(os.path.dirname(out_img) or ".", exist_ok=True)
    cv2.imwrite(out_img, img_out)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["id","x","y","width","height","area"])
        wcsv.writerows(rows)

    print("Wrote:", out_img, out_csv, "regions:", len(rows))

if __name__ == "__main__":
    main()