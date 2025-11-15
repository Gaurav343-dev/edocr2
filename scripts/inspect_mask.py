import cv2, numpy as np, os, csv, argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", required=True, help="path to debug mask (black/white) or original mask png")
    p.add_argument("--out", help="overlay output image", default=None)
    p.add_argument("--list", type=int, default=20, help="how many top contours to list")
    args = p.parse_args()

    m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit("cannot read: " + args.mask)
    h,w = m.shape
    total = h*w
    white = int((m>0).sum())
    print("mask:", args.mask, "size:", w, "x", h, "pixels:", total, "white:", white, "ratio:", white/total)

    # find contours and areas
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("contours found:", len(contours))
    areas = [(i, int(cv2.contourArea(c)), cv2.boundingRect(c)) for i,c in enumerate(contours)]
    areas_sorted = sorted(areas, key=lambda x: x[1], reverse=True)
    print("Top contours (idx, area, bbox x,y,w,h):")
    for entry in areas_sorted[:args.list]:
        print(entry)

    # make overlay: draw all contours in yellow, top N in green, and huge ones in red
    overlay = cv2.cvtColor(cv2.imread(args.mask, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if overlay is None:
        overlay = cv2.cvtColor(np.stack([m]*3, axis=2), cv2.COLOR_BGR2RGB)
    max_area = total * 0.5
    for i,(idx,area,(x,y,wc,hc)) in enumerate(areas_sorted):
        color = (255,0,0)  # red for default
        if area > max_area:
            color = (255,0,0)  # red = too large
        if i < args.list:
            color = (0,255,0)  # green for top-listed
        cv2.rectangle(overlay, (x,y), (x+wc, y+hc), color, 2)
    outp = args.out or os.path.splitext(args.mask)[0] + "_overlay.png"
    cv2.imwrite(outp, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Wrote overlay:", outp)

if __name__ == "__main__":
    main()