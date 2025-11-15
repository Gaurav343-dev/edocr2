import os, sys, argparse, math, csv
from PIL import Image
import numpy as np
import cv2
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    import pytesseract
    HAVE_TESSERACT = True
except Exception:
    HAVE_TESSERACT = False

def load_image(path, dpi=300):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        if convert_from_path is None:
            raise RuntimeError("pdf2image missing; install pdf2image + poppler")
        pages = convert_from_path(path, dpi=dpi)
        pil = pages[0]
    else:
        pil = Image.open(path).convert('RGB')
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def binary_image(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    return bw

def get_tess_words(img):
    if not HAVE_TESSERACT:
        return []
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    boxes = []
    n = len(data['level'])
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt: continue
        x = int(data['left'][i]); y = int(data['top'][i]); w = int(data['width'][i]); h = int(data['height'][i])
        boxes.append({'text': txt, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + w/2.0, 'cy': y + h/2.0})
    return boxes

def find_nearest_tess(tess_boxes, x, y, max_dist=400):
    best=None; best_score=1e12
    for b in tess_boxes:
        dx=b['cx']-x; dy=b['cy']-y
        dist2=dx*dx+dy*dy; area=b['w']*b['h']
        score=dist2 + area*0.2
        if score<best_score:
            best_score=score; best=b
    if best is None or math.sqrt(best_score)>max_dist:
        return None
    return best

def cc_tight_box(bw, x, y, search_radius=40):
    h,w = bw.shape
    x=int(round(x)); y=int(round(y))
    x=max(0,min(w-1,x)); y=max(0,min(h-1,y))
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    lbl = labels[y,x]
    def tight(lbl_idx):
        mask = (labels==lbl_idx).astype('uint8')*255
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            rx,ry,rw,rh = cv2.boundingRect(c)
            return rx,ry,rw,rh
        rx,ry,rw,rh,_ = stats[lbl_idx]
        return rx,ry,rw,rh
    if lbl!=0:
        return tight(lbl)
    x0=max(0,x-search_radius); x1=min(w,x+search_radius+1)
    y0=max(0,y-search_radius); y1=min(h,y+search_radius+1)
    local = labels[y0:y1, x0:x1]
    uniq = np.unique(local); uniq = uniq[uniq!=0]
    if uniq.size:
        best=None; best_score=1e12
        for ul in uniq:
            cx,cy = cents[ul]
            area = stats[ul, cv2.CC_STAT_AREA]
            score = (cx-x)**2 + (cy-y)**2 + area*0.5
            if score<best_score:
                best_score=score; best=int(ul)
        return tight(best)
    return max(0,x-5), max(0,y-5), 10, 10

def parse_csv(csv_in):
    with open(csv_in, newline='', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if not lines: return [], None
    if lines[0].strip().startswith("Table_") and len(lines)>1:
        header = [h.strip() for h in lines[1].split(',')]
        data = lines[2:]
        table_name = lines[0]
    else:
        header = [h.strip() for h in lines[0].split(',')]
        data = lines[1:]
        table_name = None
    rows=[]
    for ln in data:
        if not ln.strip(): continue
        parts=[p.strip() for p in ln.split(',')]
        rows.append(parts)
    return rows, header

def draw_debug(img, row, header, tess_match, cc_box, save_full, save_crop, pad=40):
    out = img.copy()
    # draw original point
    x=int(float(row[1])); y=int(float(row[2]))
    cv2.drawMarker(out, (x,y), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    # draw tess
    if tess_match:
        cv2.rectangle(out, (tess_match['x'], tess_match['y']), (tess_match['x']+tess_match['w'], tess_match['y']+tess_match['h']), (255,0,0), 2)
    # draw cc
    bx,by,bw,bh = cc_box
    cv2.rectangle(out, (bx,by), (bx+bw,by+bh), (0,255,0), 2)
    # label
    txt = row[0][:60]
    cv2.putText(out, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(save_full, out)
    # crop
    cx0 = max(0, min(x-pad, out.shape[1]-1))
    cy0 = max(0, min(y-pad, out.shape[0]-1))
    cx1 = min(out.shape[1], x+pad); cy1 = min(out.shape[0], y+pad)
    crop = out[cy0:cy1, cx0:cx1].copy()
    cv2.imwrite(save_crop, crop)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--row', type=int, required=True, help='row index (0-based in CSV data section)')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--search-radius', type=int, default=40)
    args = p.parse_args()
    rows, header = parse_csv(args.csv)
    if args.row<0 or args.row>=len(rows):
        print("row out of range", len(rows)); return
    row = rows[args.row]
    img = load_image(args.image, dpi=args.dpi)
    bw = binary_image(img)
    tess_boxes = get_tess_words(img) if HAVE_TESSERACT else []
    x=float(row[1]); y=float(row[2])
    tess_match = find_nearest_tess(tess_boxes, x, y) if tess_boxes else None
    cc_box = cc_tight_box(bw, x, y, search_radius=args.search_radius)
    base = os.path.splitext(os.path.basename(args.csv))[0]
    save_full = base + f"_debug_row{args.row}_dpi{args.dpi}.png"
    save_crop = base + f"_debug_row{args.row}_crop.png"
    draw_debug(img, row, header, tess_match, cc_box, save_full, save_crop)
    print("Wrote:", save_full, save_crop)
    if tess_match:
        print("tess box:", tess_match)
    print("cc box:", cc_box)
    print("original point:", (x,y))

if __name__ == '__main__':
    import argparse
    main()