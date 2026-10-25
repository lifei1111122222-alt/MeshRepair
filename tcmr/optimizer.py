from __future__ import annotations

"""Differentiable compact-support patch optimization."""

import numpy as np
import torch


def triangle_quality(vertices, faces):
    p = vertices[faces]
    e = np.stack([np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
                  np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
                  np.linalg.norm(p[:, 0] - p[:, 2], axis=1)], axis=1)
    area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) * 0.5
    return 4 * np.sqrt(3) * area / np.maximum((e ** 2).sum(1), 1e-12)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    return np.unique(np.sort(edges, axis=1), axis=0)


def _seam_pairs(base_faces: np.ndarray, patch_faces: np.ndarray, boundary_loop: np.ndarray):
    if len(base_faces) == 0 or len(patch_faces) == 0 or len(boundary_loop) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    # Add edges face-by-face.  The previous implementation enumerated a
    # concatenated three-block edge array and used ``ei % n_faces``; that
    # associates the second and third edge blocks with unrelated faces and
    # corrupts the seam normal target on most boundaries.
    for fi, face in enumerate(np.asarray(base_faces, dtype=np.int64)):
        for edge in (face[[0, 1]], face[[1, 2]], face[[2, 0]]):
            edge_to_faces.setdefault(tuple(np.sort(edge).tolist()), []).append(fi)
    patch_map: dict[tuple[int, int], int] = {}
    for fi, face in enumerate(patch_faces):
        for edge in (face[[0, 1]], face[[1, 2]], face[[2, 0]]):
            patch_map.setdefault(tuple(np.sort(edge).tolist()), fi)
    retained, patch = [], []
    for a, b in zip(boundary_loop, np.roll(boundary_loop, -1)):
        key = tuple(sorted((int(a), int(b))))
        if key in patch_map and key in edge_to_faces:
            retained.append(edge_to_faces[key][0]); patch.append(patch_map[key])
    return np.asarray(retained, dtype=np.int64), np.asarray(patch, dtype=np.int64)


def optimize_local(vertices, faces, patch_faces, patch_vertices, collar_vertices, boundary_loop,
                   weights=None, steps=80, lr=0.03, device="cpu",
                   curvature_mode="residual_variation"):
    """Optimize a patch while freezing vertices outside compact support."""
    weights = weights or {"fair": 1.0, "lap": 0.1, "fid": 10.0,
                          "normal": 1.0, "curv": 0.1, "quality": 1.0}
    vertices = np.asarray(vertices, dtype=np.float64); faces = np.asarray(faces, dtype=np.int64)
    patch_faces = np.asarray(patch_faces, dtype=np.int64)
    patch_vertices = np.unique(np.asarray(patch_vertices, dtype=np.int64)); collar_vertices = np.unique(np.asarray(collar_vertices, dtype=np.int64))
    var_idx = np.unique(np.concatenate([patch_vertices, collar_vertices])).astype(np.int64)
    if len(var_idx) == 0 or len(patch_faces) == 0:
        return vertices.copy(), {k: 0.0 for k in ("fair", "lap", "fid", "normal", "curv", "quality")}, 0.0
    x0 = torch.tensor(vertices, dtype=torch.float32, device=device); xi = torch.nn.Parameter(x0[var_idx].clone())
    local_map = {int(v): i for i, v in enumerate(var_idx)}; pf = torch.tensor(patch_faces, dtype=torch.long, device=device)
    coll_pos = torch.tensor([local_map[int(v)] for v in collar_vertices if int(v) in local_map], dtype=torch.long, device=device)
    patch_pos = torch.tensor([local_map[int(v)] for v in patch_vertices if int(v) in local_map], dtype=torch.long, device=device)
    baseline = xi.detach().clone()
    p_edges = _unique_edges(patch_faces)
    edge_pos = torch.tensor([[local_map[int(a)], local_map[int(b)]] for a, b in p_edges if int(a) in local_map and int(b) in local_map], dtype=torch.long, device=device)
    seam_ret, seam_patch = _seam_pairs(faces, patch_faces, np.asarray(boundary_loop, dtype=np.int64))
    seam_patch_t = torch.tensor(seam_patch, dtype=torch.long, device=device) if len(seam_patch) else None
    bp = vertices[faces] if len(faces) else np.zeros((0, 3, 3)); bn = np.cross(bp[:, 1] - bp[:, 0], bp[:, 2] - bp[:, 0]) if len(faces) else np.zeros((0, 3))
    base_face_normals = bn / np.maximum(np.linalg.norm(bn, axis=1, keepdims=True), 1e-8)
    seam_target = torch.tensor(base_face_normals[seam_ret], dtype=torch.float32, device=device) if len(seam_ret) else None

    def assemble():
        xx = x0.clone(); xx[var_idx] = xi; return xx

    def energy_terms():
        xx = assemble(); tri = xx[pf]
        if len(edge_pos):
            a, b = edge_pos[:, 0], edge_pos[:, 1]
            lap_sum = torch.zeros((len(var_idx), 3), device=device); lap_count = torch.zeros(len(var_idx), device=device)
            lap_sum.index_add_(0, a, xi[b] - xi[a]); lap_sum.index_add_(0, b, xi[a] - xi[b]); lap_count.index_add_(0, a, torch.ones(len(a), device=device)); lap_count.index_add_(0, b, torch.ones(len(b), device=device))
            fair = ((lap_sum / torch.clamp(lap_count[:, None], min=1.0))[patch_pos] ** 2).mean() if len(patch_pos) else torch.zeros((), device=device)
            base_edge = baseline[b] - baseline[a]; lap = (((xi[b] - xi[a]) - base_edge) ** 2).mean()
        else: fair = torch.zeros((), device=device); lap = torch.zeros((), device=device)
        fid = ((xi[coll_pos] - baseline[coll_pos]) ** 2).mean() if len(coll_pos) else torch.zeros((), device=device)
        fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1); fn_norm = torch.linalg.norm(fn, dim=1, keepdim=True).clamp_min(1e-8); fn_unit = fn / fn_norm
        if seam_patch_t is not None and seam_target is not None:
            dots0 = (fn_unit.detach()[seam_patch_t] * seam_target).sum(1); orient = torch.where(dots0 >= 0, torch.ones_like(dots0), -torch.ones_like(dots0)); seam_delta = orient[:, None] * fn_unit[seam_patch_t] - seam_target
            aligned_patch = orient[:, None] * fn_unit[seam_patch_t]
            normal = (seam_delta ** 2).mean()
            if len(seam_delta) <= 1:
                curv = torch.zeros((), device=device)
            elif curvature_mode == "residual_variation":
                # Legacy term: smooth the normal-mismatch residual along the
                # seam.  It is retained for an explicit ablation because it
                # does not directly compare the two sides' curvature.
                curv = ((seam_delta[1:] - seam_delta[:-1]) ** 2).mean()
            elif curvature_mode == "normal_variation":
                # Discrete seam-curvature proxy.  Consecutive seam-face
                # normals approximate normal variation along the boundary.
                # Keep it dimensionless: dividing by short PartNet boundary
                # edges makes this term dominate the objective and changes
                # the intended trade-off with fidelity and seam-normal loss.
                patch_k = torch.linalg.norm(torch.roll(aligned_patch, shifts=-1, dims=0) - aligned_patch, dim=1)
                retained_k = torch.linalg.norm(torch.roll(seam_target, shifts=-1, dims=0) - seam_target, dim=1)
                curv = (patch_k - retained_k).square().mean()
            else:
                raise ValueError(f"Unknown curvature_mode: {curvature_mode}")
        else: normal = torch.zeros((), device=device); curv = torch.zeros((), device=device)
        e0 = torch.linalg.norm(tri[:, 1] - tri[:, 0], dim=1); e1 = torch.linalg.norm(tri[:, 2] - tri[:, 1], dim=1); e2 = torch.linalg.norm(tri[:, 0] - tri[:, 2], dim=1); quality_value = 2.0 * np.sqrt(3.0) * fn_norm[:, 0] / (e0.square() + e1.square() + e2.square()).clamp_min(1e-10); quality = torch.relu(0.20 - quality_value).square().mean()
        return {"fair": fair, "lap": lap, "fid": fid, "normal": normal, "curv": curv, "quality": quality}

    def closure():
        opt.zero_grad(); terms = energy_terms(); loss = sum(float(weights.get(k, 0.0)) * v for k, v in terms.items())
        if not torch.isfinite(loss): return torch.zeros((), device=device, requires_grad=True)
        loss.backward(); return loss
    opt = torch.optim.LBFGS([xi], lr=lr, max_iter=max(1, int(steps)), line_search_fn="strong_wolfe")
    try: opt.step(closure)
    except (RuntimeError, ValueError, FloatingPointError): pass
    with torch.no_grad():
        optimized = assemble().cpu().numpy()
        # Preserve the input dtype/coordinates outside compact support exactly.
        # The optimization tensor is float32 for speed, so returning it
        # wholesale would introduce artificial 1e-8 displacements everywhere.
        out = vertices.copy()
        out[var_idx] = optimized[var_idx].astype(out.dtype, copy=False)
        terms_t = energy_terms()
        terms = {k: float(torch.nan_to_num(v).detach().cpu()) for k, v in terms_t.items()}
    disp = np.linalg.norm(out[collar_vertices] - vertices[collar_vertices], axis=1) if len(collar_vertices) else np.zeros(1)
    bp_loop = vertices[np.asarray(boundary_loop, dtype=np.int64)]; seg = np.linalg.norm(np.roll(bp_loop, -1, axis=0) - bp_loop, axis=1) if len(bp_loop) > 2 else np.ones(1); scale = max(float(np.median(seg)), 1e-8)
    displacement_cost = float(np.mean((disp / scale) ** 2)); cost = float(sum(float(weights.get(k, 0.0)) * terms[k] for k in terms) + displacement_cost)
    return out, terms, cost
