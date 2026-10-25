from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


def _inside_polygon(points, polygon):
    """Vectorised ray crossing test for 2-D points."""
    x, y = points[:, 0], points[:, 1]; px, py = polygon[:, 0], polygon[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        hit = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-12) + px[i])
        inside ^= hit; j = i
    return inside


def _basis(points):
    c = points.mean(0); _, _, vh = np.linalg.svd(points - c, full_matrices=False)
    return c, vh[0], vh[1], vh[2]


def _ear_clip(xy: np.ndarray, loop: np.ndarray, eps: float = 1e-10):
    """Triangulate a simple 2-D polygon while preserving its boundary edges."""
    if len(xy) < 3:
        return []
    area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
    order = list(range(len(xy))) if area >= 0 else list(range(len(xy) - 1, -1, -1))
    out = []

    def cross(a, b, c):
        ab = xy[b] - xy[a]; ac = xy[c] - xy[a]
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def inside(p, a, b, c):
        c1 = cross(a, b, p); c2 = cross(b, c, p); c3 = cross(c, a, p)
        return c1 >= -eps and c2 >= -eps and c3 >= -eps

    guard = 0
    while len(order) > 3 and guard < len(xy) * len(xy):
        guard += 1; clipped = False
        for pos, cur in enumerate(order):
            prev, nxt = order[pos - 1], order[(pos + 1) % len(order)]
            if cross(prev, cur, nxt) <= eps:
                continue
            if any(inside(other, prev, cur, nxt) for other in order if other not in (prev, cur, nxt)):
                continue
            out.append((prev, cur, nxt)); del order[pos]; clipped = True; break
        if not clipped:
            return []
    if len(order) == 3 and abs(cross(order[0], order[1], order[2])) > eps:
        out.append(tuple(order))
    return [np.asarray(loop[list(t)], dtype=np.int64) for t in out]


def planar_delaunay_or_fan(vertices: np.ndarray, loop: np.ndarray, planarity: float,
                           threshold: float = 0.08, fan_offset: float = 0.0):
    points = vertices[loop]; c, u, v, n = _basis(points)
    xy = np.stack([(points - c) @ u, (points - c) @ v], axis=1)
    faces = []
    extended_vertices = vertices.copy()
    method = "planar_delaunay"
    if planarity <= threshold and len(points) >= 3:
        try:
            # Prefer a constrained Delaunay triangulation when the optional
            # ``triangle`` backend is available.  It preserves every boundary
            # segment and avoids the long skinny ears produced by a purely
            # polygonal ear clip on high-resolution PartNet loops.
            try:
                import triangle as _triangle
                # The native backend is reliable for ordinary loops; very
                # large boundary polygons are routed through the Python
                # fallback to avoid excessive native memory use.
                safe_boundary = (len(xy) <= 80 and np.all(np.isfinite(xy)) and
                                 len(np.unique(np.round(xy, decimals=10), axis=0)) == len(xy) and
                                 np.all(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1) > 1e-10))
                if safe_boundary:
                    seg = np.column_stack([np.arange(len(xy)), np.roll(np.arange(len(xy)), -1)])
                    # A modest minimum-angle constraint gives the patch an
                    # interior Steiner mesh instead of triangulating every
                    # high-resolution boundary vertex directly into a long,
                    # skinny ear.  The explicit cap keeps this local backend
                    # bounded on irregular PartNet assemblies.
                    mesh = _triangle.triangulate({"vertices": xy, "segments": seg}, "pq20S512Q")
                    tri_s = np.asarray(mesh.get("triangles", []), dtype=np.int64)
                    if len(tri_s):
                        out_xy = np.asarray(mesh["vertices"], dtype=float)
                        extra_xy = out_xy[len(xy):]
                        if len(extra_xy):
                            extra_3d = c + extra_xy[:, :1] * u + extra_xy[:, 1:2] * v
                            extended_vertices = np.vstack([vertices, extra_3d])
                        else:
                            extended_vertices = vertices.copy()
                        remap = np.concatenate([loop, np.arange(len(vertices), len(extended_vertices), dtype=np.int64)])
                        faces = [np.asarray(remap[t], dtype=np.int64) for t in tri_s]
                    else:
                        extended_vertices = vertices.copy()
            except Exception:
                faces = []
                extended_vertices = vertices.copy()
            # Ear clipping gives a valid triangulation for concave loops; use
            # polygon clipping as a fallback for numerical/pathological input.
            if not faces:
                faces = _ear_clip(xy, loop)
            if not faces:
                tri = Delaunay(xy)
                for t in tri.simplices:
                    if _inside_polygon(xy[np.asarray([xy[t].mean(0)])], xy)[0]:
                        faces.append(np.asarray(loop[t], dtype=np.int64))
            oriented = []
            for f in faces:
                tri_points = extended_vertices[np.asarray(f, dtype=np.int64)]
                if np.dot(np.cross(tri_points[1] - tri_points[0], tri_points[2] - tri_points[0]), n) < 0:
                    f = f[[0, 2, 1]]
                oriented.append(f)
            faces = oriented
            if not faces:
                raise ValueError("empty triangulation")
        except Exception:
            faces = []
    if not faces:
        method = "advancing_front_fan"
        center_idx = len(vertices)
        edge_scale = float(np.median(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1))) if len(points) > 2 else 1.0
        center = points.mean(0) + float(fan_offset) * max(edge_scale, 1e-8) * n
        faces = [np.array([loop[i], loop[(i + 1) % len(loop)], center_idx], dtype=np.int64) for i in range(len(loop))]
        return np.vstack([vertices, center]), np.asarray(faces, dtype=np.int64), method, center_idx
    return extended_vertices, np.asarray(faces, dtype=np.int64), method, None
