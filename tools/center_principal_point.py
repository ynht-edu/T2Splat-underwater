import argparse
import glob
import os
import shutil
import struct

import cv2
import numpy as np

from check_projection import read_cameras, read_images


def crop_amounts(c, size):
    d = int(round(2 * (c - size / 2)))
    return (d, 0) if d > 0 else (0, -d)


def main():
    p = argparse.ArgumentParser(description="crop images so the principal point is the image centre (what T2-Splat assumes)")
    p.add_argument("src", help="dataset with images/, sparse/0 (PINHOLE)")
    p.add_argument("dst", help="new dataset folder")
    a = p.parse_args()
    sp = os.path.join(a.src, "sparse", "0")
    cams, imgs = read_cameras(f"{sp}/cameras.bin"), read_images(f"{sp}/images.bin")
    crop = {}
    out_sp = os.path.join(a.dst, "sparse", "0")
    os.makedirs(out_sp, exist_ok=True)
    with open(os.path.join(out_sp, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, (model, w, h, prm) in cams.items():
            assert model == 1, "expects PINHOLE cameras"
            fx, fy, cx, cy = prm
            l, r = crop_amounts(cx, w)
            t, b = crop_amounts(cy, h)
            W, H, ncx, ncy = w - l - r, h - t - b, cx - l, cy - t
            crop[cid] = (l, t, W, H)
            f.write(struct.pack("<iiQQ4d", cid, 1, W, H, fx, fy, ncx, ncy))
            print(f"camera {cid}: {w}x{h} -> {W}x{H}, crop left {l} right {r} top {t} bottom {b}, "
                  f"offset now ({ncx - W / 2:+.2f}, {ncy - H / 2:+.2f}) px")
    with open(os.path.join(out_sp, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for iid, (name, q, t, cid, kp) in imgs.items():
            l, tp, _, _ = crop[cid]
            f.write(struct.pack("<i4d3di", iid, *q, *t, cid))
            f.write(name.encode() + b"\0")
            f.write(struct.pack("<Q", len(kp)))
            out = np.empty(len(kp), dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            out["x"], out["y"], out["id"] = kp["x"] - l, kp["y"] - tp, kp["id"]
            f.write(out.tobytes())
    for fn in ("points3D.bin", "points3D.ply"):
        if os.path.exists(os.path.join(sp, fn)):
            shutil.copy2(os.path.join(sp, fn), out_sp)
    for sub in ("images", "masks"):
        if not os.path.isdir(os.path.join(a.src, sub)):
            continue
        os.makedirs(os.path.join(a.dst, sub), exist_ok=True)
        cid_of = {os.path.basename(name).split(".")[0]: cid for name, _, _, cid, _ in imgs.values()}
        for fpath in glob.glob(os.path.join(a.src, sub, "*")):
            name = os.path.basename(fpath)
            cid = cid_of.get(name.split(".")[0])
            if cid is None:
                continue
            l, tp, W, H = crop[cid]
            im = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(a.dst, sub, name), im[tp: tp + H, l: l + W])
    print("written", a.dst)


if __name__ == "__main__":
    main()
