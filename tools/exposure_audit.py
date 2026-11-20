import argparse
import csv
import glob
import os


import cv2
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("images")
    p.add_argument("--out", default="exposure_audit")
    p.add_argument("--jump", type=float, default=0.08, help="flag realtive brightness change between neighbour frames")
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, "*.png")) + glob.glob(os.path.join(a.images, "*.jpg")))

    rows = []

    for f in files:
        im = cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
        im = cv2.resize(im, (160, 90), interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
        c = im[h//4: 3 * h // 4, w // 4: 3* w //4].reshape(-1, 3).mean(0)
        m = im.reshape(-1, 3).mean(0)
        rows.append([os.path.basename(f), *m, *c])
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "exposure.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["image", "r", "g", "b", "cr", "cg", "cb"]] + rows)
    arr = np.array([r[1:] for r in rows])
    lum = arr[:, :3].mean(1)
    rel = np.abs(np.diff(lum)) / np.maximum(lum[:-1], 1e-3)
    jumps = np.where(rel > a.jump)[0]
    gb = arr[:, 1] / np.maximum(arr[:, 2], 1e-3)
    print(f"{len(rows)} images  mean brightness {lum.mean():.3f}  range {lum.min():.3f}-{lum.max():.3f}")
    print(f"green/blue ratio {gb.mean():.3f} ± {gb.std():.3f}")
    print(f"{len(jumps)} neighbour jumps above {a.jump:.0%}: " + ", ".join(rows[i + 1][0] for i in jumps[:20]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        for k, col in enumerate("rgb"):
            ax[0].plot(arr[:, k], color=col, lw=1)
        ax[0].set_ylabel("mean value")
        ax[1].plot(gb, color="k", lw=1)
        ax[1].set_ylabel("G/B ratio")
        ax[1].set_xlabel("frame")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "exposure.png"), dpi=120)
    except ImportError:
        pass

if __name__ == "__main__":
    main()
