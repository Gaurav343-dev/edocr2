import os
import csv
import argparse
import numpy as np
from PIL import Image
import cv2
import math
import statistics

# If PDF support needed:
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

# optional pytesseract (preferred for tight boxes)
try:
    import pytesseract
    HAVE_TESSERACT = True
except Exception:
    HAVE_TESSERACT = False

def load_image(path, dpi=300):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        if convert_from_path is None:
            raise RuntimeError("pdf2image not available; install pdf2image and poppler")
        pages = convert_from_path(path, dpi=dpi)
        pil_img = pages[0]
    else:
        pil_img = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def binary_image(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Otsu then light morphology
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw

def get_tesseract_word_boxes(img):
    # returns list of dicts: {text, x, y, w, h, cx, cy}
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    boxes = []
    n = len(data['level'])
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt:
            continue
        x = int(data['left'][i]); y = int(data['top'][i])
        w = int(data['width'][i]); h = int(data['height'][i])
        boxes.append({'text': txt, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + w/2.0, 'cy': y + h/2.0})
    return boxes

def find_nearest_tess_box(tess_boxes, x, y, max_dist=200):
    # choose box by distance between centers; prefer smaller area on tie
    best = None
    best_score = float('inf')
    for b in tess_boxes:
        dx = b['cx'] - x; dy = b['cy'] - y
        dist2 = dx*dx + dy*dy
        area = b['w'] * b['h']
        score = dist2 + area * 0.2
        if score < best_score:
            best_score = score
            best = b
    if best is None or math.sqrt(best_score) > max_dist:
        return None
    return best

def find_component_bbox_refined(bw, x, y, search_radius=40):
    h, w = bw.shape
    x = int(round(x)); y = int(round(y))
    x = max(0, min(w-1, x)); y = max(0, min(h-1, y))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    lbl = labels[y, x]
    def tight(lbl_idx):
        mask = (labels == lbl_idx).astype('uint8') * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x0, y0, ww, hh = cv2.boundingRect(c)
            return x0, y0, ww, hh
        x0, y0, ww, hh, _ = stats[lbl_idx]
        return x0, y0, ww, hh
    if lbl != 0:
        return tight(lbl)
    x0 = max(0, x - search_radius); x1 = min(w, x + search_radius + 1)
    y0 = max(0, y - search_radius); y1 = min(h, y + search_radius + 1)
    local = labels[y0:y1, x0:x1]
    uniq = np.unique(local)
    uniq = uniq[uniq != 0]
    if uniq.size:
        best_lbl = None; best_score = float('inf')
        for ul in uniq:
            cx, cy = centroids[ul]
            area = stats[ul, cv2.CC_STAT_AREA]
            dist = (cx - x)**2 + (cy - y)**2
            score = dist + area * 0.5
            if score < best_score:
                best_score = score; best_lbl = int(ul)
        return tight(best_lbl)
    # expand search
    for r in (search_radius*2, search_radius*4):
        x0 = max(0, x - r); x1 = min(w, x + r + 1)
        y0 = max(0, y - r); y1 = min(h, y + r + 1)
        local = labels[y0:y1, x0:x1]
        uniq = np.unique(local)
        uniq = uniq[uniq != 0]
        if uniq.size:
            best_lbl = None; best_score = float('inf')
            for ul in uniq:
                cx, cy = centroids[ul]
                area = stats[ul, cv2.CC_STAT_AREA]
                dist = (cx - x)**2 + (cy - y)**2
                score = dist + area * 0.5
                if score < best_score:
                    best_score = score; best_lbl = int(ul)
            return tight(best_lbl)
    # fallback
    fw = 10; fh = 10
    xx = max(0, x - fw//2); yy = max(0, y - fh//2)
    return xx, yy, fw, fh

def estimate_linear_transform(pairs):
    # pairs: list of (csv_x, csv_y, tess_cx, tess_cy)
    if len(pairs) < 3:
        return None
    csv_x = np.array([p[0] for p in pairs], dtype=np.float64)
    csv_y = np.array([p[1] for p in pairs], dtype=np.float64)
    tess_x = np.array([p[2] for p in pairs], dtype=np.float64)
    tess_y = np.array([p[3] for p in pairs], dtype=np.float64)
    # fit linear: tess = a * csv + b
    ax, bx = np.polyfit(csv_x, tess_x, 1)
    ay, by = np.polyfit(csv_y, tess_y, 1)
    return (ax, bx, ay, by)

def process(csv_in, image_path, csv_out=None, dpi=300, search_radius=40):
    if csv_out is None:
        base, ext = os.path.splitext(csv_in)
        csv_out = base + "_with_wh.csv"
    img = load_image(image_path, dpi=dpi)
    h_img, w_img = img.shape[:2]
    bw = binary_image(img)
    tess_boxes = None
    if HAVE_TESSERACT:
        try:
            tess_boxes = get_tesseract_word_boxes(img)
        except Exception:
            tess_boxes = None

    # read csv and parse rows early so we can estimate transform
    with open(csv_in, newline='', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if len(lines) >= 2 and lines[1].strip().lower().startswith("text"):
        header = lines[1].split(',')
        data_lines = lines[2:]
        table_name = lines[0]
    else:
        header = lines[0].split(',')
        data_lines = lines[1:]
        table_name = None

    parsed_rows = []
    for line in data_lines:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3: continue
        try:
            x = float(parts[1]); y = float(parts[2])
        except Exception:
            continue
        parsed_rows.append((parts, x, y))

    # auto-estimate linear transform between CSV coords and Tesseract centers (if available)
    transform = None
    if tess_boxes and len(tess_boxes) > 0:
        pairs = []
        # build pairs by matching CSV points to nearest tess box center
        for parts, cx, cy in parsed_rows:
            # find nearest tess box center
            best = None; best_d = float('inf')
            for tb in tess_boxes:
                dx = tb['cx'] - cx; dy = tb['cy'] - cy
                d2 = dx*dx + dy*dy
                if d2 < best_d:
                    best_d = d2; best = tb
            if best is None: continue
            # only accept reasonable matches (tunable threshold)
            if best_d < (400.0**2):
                pairs.append((cx, cy, best['cx'], best['cy']))
            if len(pairs) >= 200:
                break
        transform = estimate_linear_transform(pairs)

    fallback_count = 0
    rows = []
    # now process rows, applying transform if available
    for parts, orig_x, orig_y in parsed_rows:
        text = parts[0]
        x = orig_x; y = orig_y
        if transform is not None:
            ax, bx, ay, by = transform
            x = ax * orig_x + bx
            y = ay * orig_y + by

        # prefer tesseract
        bbox = None
        if tess_boxes:
            tb = find_nearest_tess_box(tess_boxes, x, y, max_dist=300)
            if tb is not None:
                bbox = (int(tb['x']), int(tb['y']), int(tb['w']), int(tb['h']))
        if bbox is None:
            bx, by, bwid, bheight = find_component_bbox_refined(bw, x, y, search_radius=search_radius)
            bbox = (bx, by, bwid, bheight)
            fallback_count += 1
        # clamp
        bx, by, bwid, bheight = bbox
        bx = max(0, min(int(bx), w_img-1)); by = max(0, min(int(by), h_img-1))
        bwid = max(1, min(int(bwid), w_img-bx)); bheight = max(1, min(int(bheight), h_img-by))

        # refine width/height by computing tight bbox of foreground pixels
        try:
            mask_region = bw[by:by+bheight, bx:bx+bwid]
            # find non-zero pixels
            ys, xs = np.where(mask_region > 0)
            if ys.size:
                x_min = int(xs.min()); x_max = int(xs.max())
                y_min = int(ys.min()); y_max = int(ys.max())
                tight_w = x_max - x_min + 1
                tight_h = y_max - y_min + 1
                # replace width/height with tight values
                bwid, bheight = tight_w, tight_h
        except Exception:
            # keep original if anything fails
            pass

        # do not include BBox_X/BBox_Y in output — only Width/Height
        rows.append((text, int(orig_x), int(orig_y), bwid, bheight) + tuple(parts[3:]))

    # write out without BBox_X/BBox_Y
    out_header = ['Text','X','Y','Width','Height'] + (header[3:] if len(header)>3 else [])
    os.makedirs(os.path.dirname(csv_out) or '.', exist_ok=True)
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if table_name:
            w.writerow([table_name])
        w.writerow(out_header)
        for r in rows:
            w.writerow(r)
    print("Wrote:", csv_out, "| used fallback component method for", fallback_count, "items")
    return csv_out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--out', help='output CSV path (optional)')
    p.add_argument('--dpi', type=int, default=300, help='PDF->image dpi (must match OCR run dpi)')
    p.add_argument('--search-radius', type=int, default=40, help='local search radius for connected components')
    args = p.parse_args()
    process(args.csv, args.image, args.out, dpi=args.dpi, search_radius=args.search_radius)

if __name__ == '__main__':
    main()