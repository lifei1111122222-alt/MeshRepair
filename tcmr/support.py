from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def adaptive_support(vertices: np.ndarray, loop: np.ndarray, scale: float, planarity: float, curvature_std: float,
                     eta: float = 3.0, a: float = 2.0, b: float = 1.0,
                     rings: int | None = None, faces: np.ndarray | None = None):
    if rings is not None:
        # k-ring approximation is supplied by nearest vertices when connectivity is unavailable.
        radius = eta * scale
    else:
        # ``curvature`` is measured in inverse length.  Multiplying it by the
        # local edge scale gives a dimensionless variation; using the raw
        # value makes ordinary PartNet loops (whose edge lengths are often
        # 1e-2) expand the support to the whole object.
        dimensionless_curvature = float(np.clip(curvature_std * max(scale, 1e-12), 0.0, 2.0))
        radius = eta * scale * (1.0 + a * float(max(planarity, 0.0)) + b * dimensionless_curvature)
    # Prefer a true face-connectivity collar when retained topology is
    # available.  Euclidean radius alone can jump across nearby but separate
    # PartNet components and violates compact-support intent.
    if faces is not None and rings is not None and len(loop):
        ff = np.asarray(faces, dtype=np.int64)
        n = min(len(vertices), int(ff.max()) + 1) if len(ff) else len(vertices)
        adjacency = [[] for _ in range(n)]
        for face in ff:
            if np.any(face < 0) or np.any(face >= n):
                continue
            a0, a1, a2 = (int(face[0]), int(face[1]), int(face[2]))
            adjacency[a0].extend((a1, a2)); adjacency[a1].extend((a0, a2)); adjacency[a2].extend((a0, a1))
        visited = set(int(v) for v in np.asarray(loop, dtype=np.int64) if 0 <= int(v) < n)
        frontier = set(visited)
        for _ in range(max(0, int(rings))):
            nxt = set()
            for vtx in frontier:
                nxt.update(adjacency[vtx])
            nxt.difference_update(visited)
            visited.update(nxt); frontier = nxt
            if not frontier:
                break
        collar = np.asarray(sorted(visited), dtype=np.int64)
        dist = np.zeros(len(vertices), dtype=float)
    else:
        tree = cKDTree(vertices[loop])
        dist, _ = tree.query(vertices, k=1)
        collar = np.flatnonzero(dist <= radius)
    support = collar.copy()
    return support, collar, float(radius), dist
