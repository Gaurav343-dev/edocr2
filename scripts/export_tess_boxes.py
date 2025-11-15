import os, csv, math
from PIL import Image
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    import pytesseract
    HAVE_TESS = True
except Exception:
    HAVE_TESS = False

CSV_IN = "/Users/gauravpreetsingh/edocr2/C405777-REV-A/table_results_with_wh.csv"
IMAGE = "/Users/gauravpreetsingh/edocr2/tests/test_samples/C405777-REV-A.pdf"
CSV_OUT = "/Users/gauravpreetsingh/edocr2/C405777-REV-A/table_results_tess_simple.csv"
DPI = 200
MAX_MATCH_DIST = 400  # pixels

def load_image(path, dpi=DPI):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if convert_from_path is None:
            raise RuntimeError("pdf2image required for PDF input")
        pages = convert_from_path(path, dpi=dpi)
        return pages[0]
    return Image.open(path).convert("RGB")

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
        cx = x + w/2.0; cy = y + h/2.0
        boxes.append({'text': text, 'x': x, 'y': y, 'w': w, 'h': h, 'cx': cx, 'cy': cy})
    return boxes

def read_csv_rows(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if len(lines) >= 2 and lines[1].strip().lower().startswith("text"):
        data_lines = lines[2:]
    else:
        data_lines = lines[1:]
    rows = []
    for ln in data_lines:
        if not ln.strip(): continue
        parts = [p.strip() for p in ln.split(',')]
        if len(parts) < 3: continue
        text = parts[0]
        try:
            x = float(parts[1]); y = float(parts[2])
        except:
            continue
        rows.append({'text': text, 'x': x, 'y': y})
    return rows

def find_nearest_tess(tess_boxes, x, y, max_dist=MAX_MATCH_DIST):
    best = None; best_d = float('inf')
    for b in tess_boxes:
        dx = b['cx'] - x; dy = b['cy'] - y
        d2 = dx*dx + dy*dy
        if d2 < best_d:
            best_d = d2; best = b
    if best is None:
        return None
    if math.sqrt(best_d) > max_dist:
        return None
    return best

def main():
    pil = load_image(IMAGE, dpi=DPI)
    tess_boxes = tess_word_boxes(pil) if HAVE_TESS else []
    rows = read_csv_rows(CSV_IN)
    os.makedirs(os.path.dirname(CSV_OUT) or ".", exist_ok=True)
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.writer(outf)
        writer.writerow(['Text','Tess_X','Tess_Y','Tess_W','Tess_H'])
        for r in rows:
            match = None
            if tess_boxes:
                match = find_nearest_tess(tess_boxes, r['x'], r['y'])
            if match:
                writer.writerow([r['text'], match['x'], match['y'], match['w'], match['h']])
            else:
                writer.writerow([r['text'], '', '', '', ''])
    print("Wrote:", CSV_OUT)

if __name__ == "__main__":
    main()