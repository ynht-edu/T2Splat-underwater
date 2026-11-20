import argparse
import os
import struct

import numpy as np

NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        for _ in range(rd(f, "Q")[0]):
            cid, model, w, h = rd(f, "iiQQ")
            cams[cid] = (model, w, h, np.array(rd(f, "d" * NUM_PARAMS[model])))
    return cams


def read_images(path):
    imgs = {}
    with open(path, "rb") as f:
        for _ in range(rd(f, "Q")[0]):
            iid, *rest = rd(f, "idddddddi")
            q, t, cid = np.array(rest[:4]), np.array(rest[4:7]), rest[7]
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            n = rd(f, "Q")[0]
            data = np.frombuffer(f.read(24 * n), dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            imgs[iid] = (name.decode(), q, t, cid, data)
    return imgs


def read_points(path):
    pts = {}
    with open(path, "rb") as f:
        for _ in range(rd(f, "Q")[0]):
            pid, x, y, z, r, g, b, err, n = rd(f, "QdddBBBdQ")
            f.read(8 * n)
            pts[pid] = (x, y, z)
    return pts


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--images", type=int, default=50)
    a = p.parse_args()
    sp = os.path.join(a.dataset, "sparse", "0")
    cams, imgs, pts = read_cameras(f"{sp}/cameras.bin"), read_images(f"{sp}/images.bin"), read_points(f"{sp}/points3D.bin")
    ids = sorted(imgs)[:: max(1, len(imgs) // a.images)]
    err_k, err_c, offs = [], [], []
    for iid in ids:
        name, q, t, cid, kp = imgs[iid]
        model, w, h, prm = cams[cid]
        fx, fy, cx, cy = (prm[0], prm[0], prm[1], prm[2]) if model == 0 else prm[:4]
        kp = kp[kp["id"] >= 0]
        X = np.array([pts[i] for i in kp["id"] if i in pts])
        kp = kp[[i in pts for i in kp["id"]]]
        Xc = X @ qvec2rotmat(q).T + t
        u, v = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        obs = np.stack([kp["x"], kp["y"]], 1)
        err_k.append(np.linalg.norm(np.stack([fx * u + cx, fy * v + cy], 1) - obs, axis=1))
        err_c.append(np.linalg.norm(np.stack([fx * u + w / 2, fy * v + h / 2], 1) - obs, axis=1))
        offs.append((cx - w / 2, cy - h / 2))
    ek, ec = np.concatenate(err_k), np.concatenate(err_c)
    o = np.mean(offs, 0)
    print(f"{len(ids)} images, {len(ek)} observations, image {w}x{h}")
    print(f"reprojection with COLMAP K:       mean {ek.mean():.3f} px  median {np.median(ek):.3f} px")
    print(f"reprojection with centred c (T2S): mean {ec.mean():.3f} px  median {np.median(ec):.3f} px")
    print(f"principal point offset from centre: ({o[0]:+.2f}, {o[1]:+.2f}) px")


if __name__ == "__main__":
    main()
