"""Measurements taken from an SWC reconstruction."""
import math
import statistics
from collections import defaultdict

DENDRITE = (3, 4)          # SWC types: basal and apical dendrite


def read_swc(path):
    """Points of an SWC file as {id: (type, x, y, z, radius, parent)}."""
    pts = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 7 or p[0].startswith("#"):
                continue
            pts[int(p[0])] = (int(float(p[1])), float(p[2]), float(p[3]), float(p[4]), float(p[5]), int(float(p[6])))
    return pts


def dendritic_segments(pts):
    """Segments whose two end points are both dendritic: (length, radius at parent, radius at child, parent id)."""
    for typ, x, y, z, r, par in pts.values():
        if typ not in DENDRITE or par not in pts or pts[par][0] not in DENDRITE:
            continue
        _, px, py, pz, pr, _ = pts[par]
        yield math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2), pr, r, par


def caliber(pts):
    """Dendritic caliber (µm): length-weighted mean diameter of the dendritic segments.
    The mean radius is kept to six decimals."""
    length = weighted = 0.0
    for h, r1, r2, _ in dendritic_segments(pts):
        weighted += (r1 + r2) / 2 * h
        length += h
    return 2 * round(weighted / length, 6) if length > 0 else float("nan")


def point_radius(pts):
    """Mean and median radius (µm) of the dendritic sample points: the two alternative caliber definitions.
    Both are kept to six decimals, as the mean radius of the caliber."""
    r = sorted(p[4] for p in pts.values() if p[0] in DENDRITE)
    n = len(r)
    if n == 0:
        return float("nan"), float("nan")
    return round(statistics.mean(r), 6), round(statistics.median(r), 6)


def path_distance(pts):
    """Path distance of every point from the root of its tree."""
    children, roots = defaultdict(list), []
    for i, p in pts.items():
        (children[p[5]].append(i) if p[5] in pts else roots.append(i))
    dist, stack = {}, [(r, 0.0) for r in roots]
    while stack:
        i, d = stack.pop()
        dist[i] = d
        x, y, z = pts[i][1:4]
        for c in children[i]:
            cx, cy, cz = pts[c][1:4]
            stack.append((c, d + math.sqrt((cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2)))
    return dist


def electrotonic_extent(pts, Rm, Ra, window):
    """Sum of segment length / space constant over the dendritic segments that start within `window` µm
    (path distance) of the soma. Rm in ohm cm2, Ra in ohm cm."""
    dist = path_distance(pts)
    total = 0.0
    for h, r1, r2, par in dendritic_segments(pts):
        start = dist.get(par, 0.0)
        if start >= window:
            continue
        diameter_cm = max(r1 + r2, 0.05) * 1e-4
        space_constant = math.sqrt(Rm * diameter_cm / (4 * Ra)) * 1e4
        total += min(h, window - start) / max(space_constant, 1e-6)
    return total
