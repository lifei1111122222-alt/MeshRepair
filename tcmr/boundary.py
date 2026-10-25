from __future__ import annotations

import numpy as np


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    undirected = np.sort(edges, axis=1)
    unique, counts = np.unique(undirected, axis=0, return_counts=True)
    return unique[counts == 1]


def extract_boundary_loops(vertices: np.ndarray, faces: np.ndarray, min_length: int = 3) -> list[np.ndarray]:
    edges = boundary_edges(faces)
    if len(edges) == 0:
        return []
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b)); adj.setdefault(int(b), []).append(int(a))
    for vertex in adj:
        adj[vertex].sort()
    unseen = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    loops = []
    while unseen:
        # Stable seeds and neighbour order make loop IDs reproducible across
        # Python processes, which is required for paired selector/reranking
        # evaluation and exact experiment replay.
        e = min(unseen); start, nxt = e
        path = [start, nxt]; unseen.discard(e); prev, cur = start, nxt
        while cur != start:
            candidates = [x for x in adj[cur] if tuple(sorted((cur, x))) in unseen and x != prev]
            if not candidates:
                # Open/non-manifold chains are retained as a loop-like candidate for diagnostics.
                break
            nn = candidates[0]; unseen.discard(tuple(sorted((cur, nn)))); path.append(nn); prev, cur = cur, nn
            if len(path) > len(edges) + 2:
                break
        if path[-1] == start:
            path = path[:-1]
        if len(path) >= min_length:
            loops.append(np.asarray(path, dtype=np.int64))
    return loops


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex unit normals once for a mesh.

    Boundary feature construction may evaluate many loops on the same mesh.
    Keeping this operation separate lets callers cache it without changing the
    feature definition.
    """
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    normals = np.zeros_like(vertices)
    # Vectorised accumulation avoids a Python loop over every triangle. This
    # matters for the 2k-object selector probe where several meshes contain
    # tens of thousands of faces.
    np.add.at(normals, faces[:, 0], fn)
    np.add.at(normals, faces[:, 1], fn)
    np.add.at(normals, faces[:, 2], fn)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals


def loop_geometry(vertices: np.ndarray, faces: np.ndarray, loop: np.ndarray, cached_normals: np.ndarray | None = None):
    p = vertices[loop]
    seg = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(seg, axis=1)
    tangents = seg / np.maximum(lengths[:, None], 1e-12)
    # Vertex normals from incident faces; callers can cache them per mesh.
    normals = cached_normals if cached_normals is not None else vertex_normals(vertices, faces)
    n = normals[loop]
    prev_t = np.roll(tangents, 1, axis=0)
    curvature = np.linalg.norm(tangents - prev_t, axis=1) / np.maximum((lengths + np.roll(lengths, 1)) * 0.5, 1e-12)
    centered = p - p.mean(axis=0)
    _, sv, vh = np.linalg.svd(centered, full_matrices=False)
    planarity = float(sv[-1] / max(sv[0], 1e-12)) if len(sv) >= 3 else 0.0
    return {"points": p, "lengths": lengths, "tangents": tangents, "normals": n, "curvature": curvature,
            "planarity": planarity, "scale": float(np.median(lengths))}
