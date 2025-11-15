import os
import csv
import argparse
from PIL import Image
import numpy as np
import cv2

# optional PDF -> image
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

def load_image(path, dpi=300):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        if convert_from_path is None:
            raise RuntimeError("pdf2image not installed; install pdf2image and poppler")
        pages = convert_from_path(path, dpi=dpi)
        pil = pages[0]
    else:
        pil = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def parse_csv(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if not lines:
        return []
    # skip possible "Table_X" name line
    if lines[0].strip().startswith("Table_") and len(lines) > 1:
        header = [h.strip() for h in lines[1].split(',')]
        data = lines[2:]
    else:
        header = [h.strip() for h in lines[0].split(',')]
        data = lines[1:]
    rows = []
    for ln in data:
        if not ln.strip(): 
            continue
        parts = [p.strip() for p in ln.split(',')]
        # map by header name where possible
        d = {}
        for i, h in enumerate(header):
            if i < len(parts):
                d[h] = parts[i]
        rows.append(d)
    return rows

def draw_boxes(image, rows, label=True, color=(0,255,0), thickness=2):
    img = image.copy()
    h_img, w_img = img.shape[:2]
    for r in rows:
        try:
            bx = int(float(r.get('BBox_X') or r.get('BBox x') or r.get('BBoxX') or ''))
            by = int(float(r.get('BBox_Y') or r.get('BBox y') or r.get('BBoxY') or ''))
            w = int(float(r.get('Width') or r.get('W') or ''))
            hh = int(float(r.get('Height') or r.get('H') or ''))
        except Exception:
            # fallback to X,Y and use small box
            try:
                x = int(float(r.get('X') or r.get('X Coordinate') or r.get('x') or 0))
                y = int(float(r.get('Y') or r.get('Y Coordinate') or r.get('y') or 0))
            except Exception:
                continue
            bx, by, w, hh = max(0, x-5), max(0, y-5), 10, 10
        # clamp
        bx = max(0, min(bx, w_img-1))
        by = max(0, min(by, h_img-1))
        x2 = max(0, min(bx + max(1, w), w_img-1))
        y2 = max(0, min(by + max(1, hh), h_img-1))
        cv2.rectangle(img, (bx, by), (x2, y2), color, thickness)
        if label and 'Text' in r:
            text = r.get('Text')[:40]
            txt_pos = (bx, max(12, by-6))
            cv2.putText(img, text, txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--out', help='output image path (png)')
    args = p.parse_args()

    rows = parse_csv(args.csv)
    if not rows:
        print("No rows parsed from CSV")
        return

    img = load_image(args.image)
    boxed = draw_boxes(img, rows)
    out_path = args.out or os.path.splitext(args.csv)[0] + "_boxed.png"
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    cv2.imwrite(out_path, boxed)
    print("Wrote:", out_path)

if __name__ == '__main__':
    main()