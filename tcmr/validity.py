from __future__ import annotations

import numpy as np
import trimesh


def orient_patch_to_boundary(patch_faces: np.ndarray, retained_faces: np.ndarray,
                             boundary_loop: np.ndarray | None) -> np.ndarray:
    """Orient a candidate patch consistently with the retained seam.

    The PCA basis used by the initializer has an arbitrary sign, so a patch
    can otherwise be inserted with all of its triangles reversed.  For every
    shared boundary edge we compare the directed edge in the retained face
    and in the patch.  A consistently oriented two-manifold has opposite
    directions on the two sides of a shared edge; when the majority of the
    observed edges use the same direction, reverse every patch face.  This
    operation only changes winding, never geometry or candidate ranking.
    """
    pf = np.asarray(patch_faces, dtype=np.int64)
    if pf.ndim != 2 or pf.shape[1] != 3 or len(pf) == 0:
        return pf.copy()
    rf = np.asarray(retained_faces, dtype=np.int64)
    loop = np.asarray(boundary_loop, dtype=np.int64) if boundary_loop is not None else np.empty(0, dtype=np.int64)
    if len(rf) == 0 or len(loop) < 3:
        return pf.copy()

    def edge_sign(face: np.ndarray, a: int, b: int) -> int:
        for x, y in ((int(face[0]), int(face[1])),
                     (int(face[1]), int(face[2])),
                     (int(face[2]), int(face[0]))):
            if x == a and y == b:
                return 1
            if x == b and y == a:
                return -1
        return 0

    retained_map: dict[tuple[int, int], np.ndarray] = {}
    for face in rf:
        for e in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(e[0]), int(e[1]))))
            retained_map.setdefault(key, face)
    patch_map: dict[tuple[int, int], np.ndarray] = {}
    for face in pf:
        for e in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(e[0]), int(e[1]))))
            patch_map.setdefault(key, face)
    same = opposite = 0
    for a, b in zip(loop, np.roll(loop, -1)):
        key = tuple(sorted((int(a), int(b))))
        if key not in retained_map or key not in patch_map:
            continue
        rs = edge_sign(retained_map[key], int(a), int(b))
        ps = edge_sign(patch_map[key], int(a), int(b))
        if rs == 0 or ps == 0:
            continue
        if rs == ps:
            same += 1
        else:
            opposite += 1
    # A single malformed edge should not reverse a well-formed patch.  In the
    # tie case preserve the initializer's winding for deterministic behaviour.
    if same > opposite and same >= 2:
        return pf[:, [0, 2, 1]].copy()
    return pf.copy()


def _triangles_intersect(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> bool:
    """Separating-axis test for two non-degenerate 3-D triangles."""
    ea = a[[1, 2, 0]] - a
    eb = b[[1, 2, 0]] - b
    na = np.cross(ea[0], ea[1]); nb = np.cross(eb[0], eb[1])
    if np.linalg.norm(na) <= eps or np.linalg.norm(nb) <= eps:
        return False
    # The 3-D edge-cross-edge axes collapse for coplanar triangles. In that
    # case project to the dominant coordinate plane and run a complete 2-D
    # triangle SAT; otherwise disjoint coplanar triangles with overlapping
    # AABBs are incorrectly reported as intersecting.
    if np.linalg.norm(np.cross(na, nb)) <= eps * max(np.linalg.norm(na), np.linalg.norm(nb)):
        na_u = na / np.linalg.norm(na)
        if abs(float(np.dot(b[0] - a[0], na_u))) > eps:
            return False
        drop = int(np.argmax(np.abs(na_u)))
        keep = [i for i in range(3) if i != drop]
        aa, bb = a[:, keep], b[:, keep]
        axes2 = []
        for tri in (aa, bb):
            for k in range(3):
                e = tri[(k + 1) % 3] - tri[k]
                axes2.append(np.array([-e[1], e[0]], dtype=float))
        for axis in axes2:
            norm = np.linalg.norm(axis)
            if norm <= eps:
                continue
            axis /= norm
            pa, pb = aa @ axis, bb @ axis
            if pa.max() <= pb.min() + eps or pb.max() <= pa.min() + eps:
                return False
        return True
    axes = [na, nb]
    axes.extend(np.cross(x, y) for x in ea for y in eb)
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm <= eps:
            continue
        axis = axis / norm
        pa, pb = a @ axis, b @ axis
        if pa.max() < pb.min() - eps or pb.max() < pa.min() - eps:
            return False
    return True


def _local_self_intersections(vertices: np.ndarray, faces: np.ndarray,
                              target_loop: np.ndarray | None, max_faces: int = 6000) -> int | None:
    # Restrict the expensive test to faces incident to the candidate seam.
    if target_loop is None or len(faces) == 0:
        return None
    seam_vertices = np.asarray(target_loop, dtype=np.int64)
    ids = np.flatnonzero(np.any(np.isin(faces, seam_vertices), axis=1))
    if len(ids) == 0:
        return 0
    if len(ids) > max_faces:
        return None
    tri = vertices[faces[ids]]
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    order = np.argsort(lo[:, 0], kind="stable")
    count = 0
    for oi, i in enumerate(order):
        j = oi + 1
        while j < len(order) and lo[order[j], 0] <= hi[i, 0] + 1e-9:
            k = order[j]
            if np.all(lo[i] <= hi[k] + 1e-9) and np.all(lo[k] <= hi[i] + 1e-9):
                # Adjacent triangles meet by design and are not self-intersections.
                if not np.intersect1d(faces[ids[i]], faces[ids[k]]).size and _triangles_intersect(tri[i], tri[k]):
                    count += 1
            j += 1
    return int(count)


def patch_retained_intersections(vertices: np.ndarray, retained_faces: np.ndarray,
                                 patch_faces: np.ndarray, target_loop: np.ndarray | None,
                                 max_faces: int = 6000) -> int | None:
    """Count seam-local retained/patch triangle crossings.

    This deliberately ignores pairs sharing a vertex, since seam contact is
    expected there.  It is used as a label-free initialization/optimization
    guard and therefore never inspects the removed-part geometry.
    """
    if target_loop is None or len(retained_faces) == 0 or len(patch_faces) == 0:
        return None
    seam = np.asarray(target_loop, dtype=np.int64)
    rf = np.asarray(retained_faces, dtype=np.int64)
    pf = np.asarray(patch_faces, dtype=np.int64)
    rids = np.flatnonzero(np.any(np.isin(rf, seam), axis=1))
    if len(rids) == 0 or len(rids) > max_faces:
        return None
    # Patch faces are already local; keep the test bounded for pathological
    # loops while retaining all ordinary candidates.
    if len(pf) > max_faces:
        return None
    rt = np.asarray(vertices, dtype=float)[rf[rids]]
    pt = np.asarray(vertices, dtype=float)[pf]
    rlo, rhi = rt.min(axis=1), rt.max(axis=1)
    plo, phi = pt.min(axis=1), pt.max(axis=1)
    order = np.argsort(plo[:, 0], kind="stable")
    retained_order = np.argsort(rlo[:, 0], kind="stable")
    count = 0
    for oi, pi in enumerate(order):
        j = 0
        while j < len(retained_order) and rlo[retained_order[j], 0] <= phi[pi, 0] + 1e-9:
            ri = retained_order[j]
            if np.all(plo[pi] <= rhi[ri] + 1e-9) and np.all(rlo[ri] <= phi[pi] + 1e-9):
                if not np.intersect1d(pf[pi], rf[rids[ri]]).size and _triangles_intersect(pt[pi], rt[ri]):
                    count += 1
            j += 1
    return int(count)


def check_mesh_validity(vertices: np.ndarray, faces: np.ndarray, target_loop: np.ndarray | None = None):
    """Report mesh defects, preserving defects already present in the input."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    ff = np.asarray(faces, dtype=np.int64)
    if len(ff):
        edges = np.concatenate([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        nonmanifold = int(np.sum(counts > 2))
        edge_counts = {tuple(e): int(c) for e, c in zip(unique_edges, counts)}
    else:
        edges = np.empty((0, 2), dtype=np.int64); edge_counts = {}; nonmanifold = 0
    degenerate = int(np.sum(mesh.area_faces <= 1e-12))
    inverted = int(np.sum(~np.isfinite(mesh.area_faces)))
    closure = 0.0
    if target_loop is not None:
        residual = []
        for a, b in zip(target_loop, np.roll(target_loop, -1)):
            if edge_counts.get(tuple(sorted((int(a), int(b)))), 0) < 2:
                residual.append(np.linalg.norm(vertices[int(a)] - vertices[int(b)]))
        closure = float(np.sum(residual))
    self_intersections = _local_self_intersections(vertices, ff, target_loop)
    return {"non_manifold_edges": nonmanifold, "degenerate_faces": degenerate,
            "inverted_faces": inverted, "self_intersections": self_intersections,
            "watertight": bool(mesh.is_watertight), "closure": closure,
            "valid": nonmanifold == 0 and degenerate == 0 and inverted == 0}
