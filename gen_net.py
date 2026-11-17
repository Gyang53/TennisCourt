"""Generate a tournament-spec tennis net (mesh grid + white top band) as OBJ.

Spec used:
  posts 1.07 m, centre strap 0.914 m  -> 0.156 m sag, parabolic between posts
  white top band 0.06 m (2 3/8")
  mesh openings 0.05 m, threads 8 mm wide / 4 mm thick

Writes meshes/net_grid.obj and meshes/net_band.obj.
Net plane is x = 0, spans y in [-6.4, 6.4], z up.
"""
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")
HALF_Y = 6.4
POST_H = 1.07
CENTRE_H = 0.914
BAND_H = 0.06
PITCH = 0.05
THREAD_W = 0.008
THREAD_T = 0.004


def top_z(y):
    return POST_H - (POST_H - CENTRE_H) * (1 - (y / HALF_Y) ** 2)


class Mesher:
    def __init__(self):
        self.verts = []
        self.tris = []

    def box(self, x0, x1, y0, y1, z0, z1):
        base = len(self.verts)
        self.verts += [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                 (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        for a, b, c, d in quads:
            self.tris += [(base + a, base + b, base + c), (base + a, base + c, base + d)]

    def write(self, path):
        with open(path, "w") as f:
            f.write(f"# {len(self.verts)} verts {len(self.tris)} tris\n")
            f.writelines(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n" for v in self.verts)
            f.writelines(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n" for t in self.tris)
        print(f"wrote {path}: {len(self.verts)} verts, {len(self.tris)} tris")


grid = Mesher()
band = Mesher()

# vertical threads, from the ground up to the underside of the top band
n_v = int(round(2 * HALF_Y / PITCH)) + 1
for i in range(n_v):
    y = -HALF_Y + i * PITCH
    z_top = top_z(y) - BAND_H + 0.002
    if z_top <= 0:
        continue
    grid.box(-THREAD_T / 2, THREAD_T / 2, y - THREAD_W / 2, y + THREAD_W / 2, 0.0, z_top)

# horizontal threads: clipped by the sagging top edge. Because the sag dips in the
# middle, a high thread exists in TWO runs (one per post) - never as one straight run
# across the whole net, which would poke above the white band.
z = PITCH
while z < POST_H - BAND_H:
    ys = np.linspace(-HALF_Y, HALF_Y, 2001)
    mask = top_z(ys) - BAND_H > z
    if mask.any():
        idx = np.flatnonzero(mask)
        splits = np.flatnonzero(np.diff(idx) > 1)
        runs = np.split(idx, splits + 1)
        for run in runs:
            y0, y1 = float(ys[run[0]]), float(ys[run[-1]])
            parts = max(1, int(round((y1 - y0) / (4 * PITCH))))
            for k in range(parts):
                a = y0 + (y1 - y0) * k / parts
                b = y0 + (y1 - y0) * (k + 1) / parts
                grid.box(-THREAD_T / 2, THREAD_T / 2, a, b, z - THREAD_W / 2, z + THREAD_W / 2)
    z += PITCH

# white top band: one continuous strip following the sag (2 faces, no seams)
segs = 240
ys = np.linspace(-HALF_Y, HALF_Y, segs + 1)
zt = top_z(ys)
zb = zt - BAND_H
half_t = 0.005
base = len(band.verts)
for side in (-half_t, half_t):
    for k in range(len(ys)):
        band.verts.append((side, float(ys[k]), float(zb[k])))
        band.verts.append((side, float(ys[k]), float(zt[k])))
for face in (0, 1):
    off = face * 2 * len(ys)
    for k in range(len(ys) - 1):
        a0 = base + off + 2 * k          # bottom k
        a1 = a0 + 1                      # top k
        b0 = a0 + 2                      # bottom k+1
        b1 = a0 + 3                      # top k+1
        if face == 0:
            band.tris += [(a0, b0, b1), (a0, b1, a1)]
        else:
            band.tris += [(a0, b1, b0), (a0, a1, b1)]

os.makedirs(OUT, exist_ok=True)
grid.write(os.path.join(OUT, "net_grid.obj"))
band.write(os.path.join(OUT, "net_band.obj"))
print("post height", POST_H, "centre", CENTRE_H, "band", BAND_H)
