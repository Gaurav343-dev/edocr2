from PIL import Image, ImageDraw
import csv, os, math
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    import pytesseract
    HAVE_TESS = True
except Exception:
    HAVE_TESS = False

# Config
CSV_PATH = "/Users/gauravpreetsingh/edocr2/C405777-REV-A/table_results.csv"
IMAGE_PATH = "/Users/gauravpreetsingh/edocr2/tests/test_samples/C405777-REV-A.pdf"  # or .png
OUT_PATH = "/Users/gauravpreetsingh/edocr2/C405777-REV-A/annotated_boxes.png"
DPI = 200          # set to same DPI used when creating the CSV / running OCR
MAX_MATCH_DIST = 400  # pixels

def load_image(path, dpi=DPI):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if convert_from_path is None:
            raise RuntimeError("pdf2image required for PDF input")
        pages = convert_from_path(path, dpi=dpi)
        pil = pages[0]
    else:
        pil = Image.open(path)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
    return pil

def tess_word_boxes(pil_img):
    if not HAVE_TESS:
        return []
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    boxes = []
    n = len(data['level'])
    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue
        x = int(data['left'][i]); y = int(data['top'][i])
        w = int(data['width'][i]); h = int(data['height'][i])
        boxes.append({'text': text, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': x + w/2.0, 'cy': y + h/2.0})
    return boxes

def read_csv_rows(csv_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = list(reader)
    # detect and skip optional "Table_" name line
    start = 0
    if lines and lines[0] and lines[0][0].startswith("Table_") and len(lines) > 1:
        start = 1
    # If next line is header like "Text,X,Y" skip it
    if len(lines) > start and any(h.lower().startswith('text') for h in lines[start]):
        start += 1
    for ln in lines[start:]:
        if not ln or len(ln) < 3:
            continue
        text = ln[0].strip()
        try:
            x = float(ln[1]); y = float(ln[2])
        except Exception:
            continue
        w = int(float(ln[3])) if len(ln) > 3 and ln[3] != '' else None
        h = int(float(ln[4])) if len(ln) > 4 and ln[4] != '' else None
        rows.append({'text': text, 'x': x, 'y': y, 'w': w, 'h': h, 'orig': ln})
    return rows

def find_nearest_tess(tess_boxes, x, y, max_dist=MAX_MATCH_DIST):
    best = None; best_d = float('inf')
    for b in tess_boxes:
        dx = b['cx'] - x; dy = b['cy'] - y
        d2 = dx*dx + dy*dy
        if d2 < best_d:
            best_d = d2; best = b
    if best is None or math.sqrt(best_d) > max_dist:
        return None
    return best

def main():
    pil = load_image(IMAGE_PATH, dpi=DPI)
    img_w, img_h = pil.size
    tess_boxes = tess_word_boxes(pil) if HAVE_TESS else []
    rows = read_csv_rows(CSV_PATH)
    draw = ImageDraw.Draw(pil)
    total = len(rows); matched = 0; fallback = 0
    unmatched = []
    for r in rows:
        x = r['x']; y = r['y']
        match = find_nearest_tess(tess_boxes, x, y) if tess_boxes else None
        if match:
            bx, by, bw, bh = match['x'], match['y'], match['w'], match['h']
            matched += 1
        else:
            # fallback drawing so every CSV row is drawn
            fallback += 1
            unmatched.append(r)
            if r['w'] and r['h']:
                bx = int(r['x']); by = int(r['y']); bw = int(r['w']); bh = int(r['h'])
            else:
                bw, bh = 50, 20
                bx = int(r['x'] - bw//2); by = int(r['y'] - bh//2)
        bx = max(0, min(bx, img_w-1)); by = max(0, min(by, img_h-1))
        bw = max(1, min(bw, img_w-bx)); bh = max(1, min(bh, img_h-by))
        draw.rectangle([bx, by, bx+bw, by+bh], outline="red", width=2)
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    pil.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}  total_rows={total} matched={matched} fallback={fallback}")
    if unmatched:
        ufn = os.path.splitext(OUT_PATH)[0] + "_unmatched.csv"
        with open(ufn, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Text','X','Y','orig...'])
            for u in unmatched[:200]:
                w.writerow([u['text'], u['x'], u['y'], '|'.join(u['orig'])])
        print("Wrote unmatched rows sample:", ufn)

if __name__ == "__main__":
    main()