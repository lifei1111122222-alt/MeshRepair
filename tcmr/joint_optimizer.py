"""Normalized float64 optimization of a patch and a bounded retained collar."""
from __future__ import annotations

import numpy as np
import torch

from .optimizer import _seam_pairs, _unique_edges


DEFAULT_WEIGHTS = {"normal": 10.0, "normal_variation": 0.2, "patch_fair": 0.01,
                   "patch_edges": 0.1, "retained_laplacian": 1.0,
                   "retained_edges": 1.0, "retained_fidelity": 10.0,
                   "retained_normals": 0.2, "quality": 0.1, "orientation": 1.0}


def retained_collar(faces, boundary_loop, rings=2):
    edges = _unique_edges(np.asarray(faces, dtype=np.int64))
    chosen = np.unique(np.asarray(boundary_loop, dtype=np.int64))
    for _ in range(rings):
        incident = np.isin(edges[:, 0], chosen) | np.isin(edges[:, 1], chosen)
        chosen = np.unique(np.r_[chosen, edges[incident].reshape(-1)])
    return chosen


def optimize_joint_collar(vertices, retained_faces, patch_faces, boundary_loop,
                          retained_count, mode="joint", steps=10, learning_rate=0.02,
                          retained_displacement_budget=0.02, rings=2, weights=None,
                          seam_target_override=None, feature_guidance=False,
                          feasible_steps=False, initial_native_clear=None,
                          start_vertices=None, feasible_fallback="unfiltered",
                          fixed_patch_vertices=None):
    """Return coordinates and diagnostics; all terms use median-edge units.

    Patch-only fixes all retained vertices. Joint optimization also varies the
    selected boundary's retained 2-ring. Each Adam update is projected onto a
    per-retained-vertex displacement ball, measured from ``vertices``. An
    optional safe starting iterate never changes those reference coordinates,
    the normalization, or the objective's reference edge vectors and normals.
    Explicit feature-curve vertices may be held fixed, including their rim
    endpoints; their coordinates stay equal to the supplied reference mesh.
    """
    if mode not in ("joint", "patch_only"):
        raise ValueError("mode must be joint or patch_only")
    if feasible_fallback not in ("unfiltered", "initializer"):
        raise ValueError("feasible_fallback must be unfiltered or initializer")
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    vertices = np.asarray(vertices, dtype=np.float64)
    retained_faces = np.asarray(retained_faces, dtype=np.int64)
    patch_faces = np.asarray(patch_faces, dtype=np.int64)
    loop = np.asarray(boundary_loop, dtype=np.int64)
    scale = float(np.median(np.linalg.norm(vertices[loop] - vertices[np.roll(loop, -1)], axis=1)))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Boundary median edge must be finite and positive")
    origin = vertices[loop].mean(axis=0)
    collar = retained_collar(retained_faces, loop, rings=rings)
    collar = collar[collar < retained_count]
    new_patch = np.unique(patch_faces)
    new_patch = new_patch[new_patch >= retained_count]
    variable = np.unique(np.r_[new_patch, collar] if mode == "joint" else new_patch)
    fixed_feature = np.unique(np.asarray([] if fixed_patch_vertices is None else
                                        fixed_patch_vertices, dtype=np.int64))
    if not np.isin(fixed_feature, np.unique(patch_faces)).all():
        raise ValueError("Fixed feature vertices must belong to the patch")
    variable = np.setdiff1d(variable, fixed_feature)
    start = vertices if start_vertices is None else np.asarray(start_vertices, dtype=np.float64)
    if start.shape != vertices.shape or not np.isfinite(start).all():
        raise ValueError("Starting iterate must be finite with the reference topology")
    outside = np.setdiff1d(np.arange(len(vertices)), variable)
    if not np.array_equal(start[outside], vertices[outside]):
        raise ValueError("Starting iterate changed coordinates outside optimization support")
    start_retained_motion = np.linalg.norm(start[:retained_count] - vertices[:retained_count], axis=1) / scale
    if start_retained_motion.max(initial=0) > retained_displacement_budget + 1e-10:
        raise ValueError("Starting iterate exceeds the original retained-displacement budget")
    affected_mask = np.any(np.isin(retained_faces, collar), axis=1)
    affected_ids = np.flatnonzero(affected_mask)
    affected_faces = retained_faces[affected_ids]
    all_retained_edges = _unique_edges(retained_faces)
    retained_edges = all_retained_edges[np.isin(all_retained_edges, collar).any(axis=1)]
    patch_edges = _unique_edges(patch_faces)
    local_ids = np.unique(np.r_[patch_faces.reshape(-1), affected_faces.reshape(-1), retained_edges.reshape(-1), variable])
    inverse = np.full(len(vertices), -1, dtype=np.int64)
    inverse[local_ids] = np.arange(len(local_ids))
    dtype = torch.float64
    baseline = torch.tensor((vertices[local_ids] - origin) / scale, dtype=dtype)
    var_local = torch.tensor(inverse[variable], dtype=torch.long)
    xi = torch.nn.Parameter(baseline[var_local].clone() if start_vertices is None else
                            torch.tensor((start[variable] - origin) / scale, dtype=dtype))
    collar_local = torch.tensor(inverse[collar], dtype=torch.long)
    new_local = torch.tensor(inverse[new_patch], dtype=torch.long)
    patch_t = torch.tensor(inverse[patch_faces], dtype=torch.long)
    retained_t = torch.tensor(inverse[affected_faces], dtype=torch.long)
    pedges = torch.tensor(inverse[patch_edges], dtype=torch.long)
    redges = torch.tensor(inverse[retained_edges], dtype=torch.long)
    retained_var_positions = torch.tensor(np.flatnonzero(variable < retained_count), dtype=torch.long)
    nonboundary_var_positions = torch.tensor(np.flatnonzero((variable < retained_count) & ~np.isin(variable, loop)), dtype=torch.long)
    seam_retained, seam_patch = _seam_pairs(retained_faces, patch_faces, loop)
    face_lookup = np.full(len(retained_faces), -1, dtype=np.int64)
    face_lookup[affected_ids] = np.arange(len(affected_ids))
    seam_ret_t = torch.tensor(face_lookup[seam_retained], dtype=torch.long)
    seam_patch_t = torch.tensor(seam_patch, dtype=torch.long)
    zero = torch.zeros((), dtype=dtype)
    guidance = None
    apply_guidance = False
    if feature_guidance:
        if feature_guidance == "observed_neighborhood":
            from .feature_guidance import build_observable_seam_guidance
            guidance = build_observable_seam_guidance(vertices, retained_faces, patch_faces, loop)
        elif feature_guidance is True:
            from .feature_guidance import build_seam_guidance
            guidance = build_seam_guidance(vertices, retained_faces, patch_faces, loop)
        else:
            raise ValueError("Unknown feature guidance mode")
        if not (np.array_equal(guidance["seam_retained"], seam_retained)
                and np.array_equal(guidance["seam_patch"], seam_patch)):
            raise ValueError("Feature guidance uses different seam correspondences")
        delta_target = torch.tensor(guidance["delta_target"], dtype=dtype)
        normal_weight = torch.tensor(guidance["normal_weight"], dtype=dtype)
        variation_weight = torch.tensor(guidance["variation_weight"], dtype=dtype)
        apply_guidance = (feature_guidance is True or bool(np.any(guidance["sharp_mask"])))
    guard = None
    guard_fallback = None
    accepted_full = start.copy()
    hold_initializer = False
    if feasible_steps:
        from .feasible_step import FeasibleStepGuard
        guard = FeasibleStepGuard(start, np.vstack([retained_faces, patch_faces]), variable,
                                  initial_native_clear=initial_native_clear)
        if not guard.diagnostics()["enabled"]:
            hold_initializer = feasible_fallback == "initializer"
            guard_fallback = {**guard.diagnostics(), "unfiltered_fallback": not hold_initializer,
                              "fallback": ("preserve_native_verified_starting_iterate" if hold_initializer
                                           else "original_optimizer_and_final_native_verification")}
            guard.close()
            guard = None

    def normals(xx, faces):
        tri = xx[faces]
        cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
        length = torch.linalg.vector_norm(cross, dim=1).clamp_min(1e-15)
        return cross / length[:, None], length

    def laplacian(xx, edges, selected):
        total = torch.zeros_like(xx)
        degree = torch.zeros(len(xx), dtype=dtype)
        total.index_add_(0, edges[:, 0], xx[edges[:, 1]] - xx[edges[:, 0]])
        total.index_add_(0, edges[:, 1], xx[edges[:, 0]] - xx[edges[:, 1]])
        ones = torch.ones(len(edges), dtype=dtype)
        degree.index_add_(0, edges[:, 0], ones)
        degree.index_add_(0, edges[:, 1], ones)
        return total[selected] / degree[selected, None].clamp_min(1)

    with torch.no_grad():
        patch_norm0, _ = normals(baseline, patch_t)
        retained_norm0, _ = normals(baseline, retained_t)
        orient = torch.where((patch_norm0[seam_patch_t] * retained_norm0[seam_ret_t]).sum(dim=1) >= 0, 1.0, -1.0)
        if seam_target_override is not None:
            target_np = np.asarray(seam_target_override, dtype=np.float64)
            if target_np.shape != (len(seam_patch), 3):
                raise ValueError("seam_target_override must have one normal per seam edge")
            target_np = target_np / np.maximum(np.linalg.norm(target_np, axis=1, keepdims=True), 1e-12)
            seam_target = torch.tensor(target_np, dtype=dtype)
        else:
            seam_target = None
        retained_lap0 = laplacian(baseline, redges, collar_local)
        pedge0 = baseline[pedges[:, 1]] - baseline[pedges[:, 0]]
        redge0 = baseline[redges[:, 1]] - baseline[redges[:, 0]]

    def assembled():
        xx = baseline.clone()
        xx[var_local] = xi
        return xx

    def terms():
        xx = assembled()
        pn, double_area = normals(xx, patch_t)
        rn, _ = normals(xx, retained_t)
        p_seam = orient[:, None] * pn[seam_patch_t]
        r_seam = seam_target if seam_target is not None else rn[seam_ret_t]
        delta = p_seam - r_seam
        normal = delta.square().mean() if len(delta) else zero
        normal_variation = ((torch.roll(p_seam, -1, 0) - p_seam) -
                            (torch.roll(r_seam, -1, 0) - r_seam)).square().mean() if len(delta) > 1 else zero
        if apply_guidance:
            residual = delta - delta_target
            normal = (normal_weight[:, None] * residual.square()).mean() if len(delta) else zero
            normal_variation = (variation_weight[:, None] *
                                (torch.roll(residual, -1, 0) - residual).square()).mean() if len(delta) > 1 else zero
        tri = xx[patch_t]
        edge_squared = sum((tri[:, (i + 1) % 3] - tri[:, i]).square().sum(dim=1) for i in range(3))
        quality = 2 * np.sqrt(3) * double_area / edge_squared.clamp_min(1e-15)
        return {"normal": normal, "normal_variation": normal_variation,
                "patch_fair": laplacian(xx, pedges, new_local).square().mean() if len(new_local) else zero,
                "patch_edges": ((xx[pedges[:, 1]] - xx[pedges[:, 0]]) - pedge0).square().mean(),
                "retained_laplacian": (laplacian(xx, redges, collar_local) - retained_lap0).square().mean(),
                "retained_edges": ((xx[redges[:, 1]] - xx[redges[:, 0]]) - redge0).square().mean(),
                "retained_fidelity": (xx[collar_local] - baseline[collar_local]).square().mean(),
                "retained_normals": (rn - retained_norm0).square().mean(),
                "quality": torch.relu(0.20 - quality).square().mean(),
                "orientation": torch.relu(0.05 - (pn * patch_norm0).sum(1)).square().mean() +
                               torch.relu(0.05 - (rn * retained_norm0).sum(1)).square().mean()}

    initial_terms = terms()
    initial_loss = sum(weights[k] * v for k, v in initial_terms.items())
    gradient_info = {"nonboundary_collar_normal_gradient_l2": 0.0, "nonboundary_collar_normal_gradient_nonzero_vertices": 0}
    if len(variable) and len(nonboundary_var_positions):
        initial_gradient = torch.autograd.grad(initial_terms["normal"], xi, retain_graph=True)[0][nonboundary_var_positions]
        gradient_info = {"nonboundary_collar_normal_gradient_l2": float(torch.linalg.vector_norm(initial_gradient)),
                         "nonboundary_collar_normal_gradient_nonzero_vertices": int((torch.linalg.vector_norm(initial_gradient, dim=1) > 1e-15).sum())}
    initial_numeric = {k: float(v.detach()) for k, v in initial_terms.items()}
    if len(variable) and not hold_initializer:
        optimizer = torch.optim.Adam([xi], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            values = terms()
            loss = sum(weights[k] * v for k, v in values.items())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite normalized objective")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if len(retained_var_positions):
                    displacement = xi[retained_var_positions] - baseline[var_local[retained_var_positions]]
                    factor = (retained_displacement_budget / torch.linalg.vector_norm(displacement, dim=1).clamp_min(1e-15)).clamp_max(1.0)
                    xi[retained_var_positions] = baseline[var_local[retained_var_positions]] + displacement * factor[:, None]
                if guard is not None:
                    proposed = vertices.copy()
                    proposed[variable] = assembled()[var_local].numpy() * scale + origin
                    accepted_full, _ = guard.filter(accepted_full, proposed)
                    xi.copy_(torch.tensor((accepted_full[variable] - origin) / scale, dtype=dtype))
    with torch.no_grad():
        final_terms = {k: float(v) for k, v in terms().items()}
        out = vertices.copy()
        out[variable] = assembled()[var_local].numpy() * scale + origin
        if guard is not None or hold_initializer:
            out = accepted_full.copy()
    if not np.array_equal(out[outside], vertices[outside]):
        raise AssertionError("Coordinates outside support changed")
    normalized_retained_disp = np.linalg.norm(out[:retained_count] - vertices[:retained_count], axis=1) / scale
    if normalized_retained_disp.max(initial=0) > retained_displacement_budget + 1e-10:
        raise AssertionError("Hard retained-displacement constraint violated")
    info = {"mode": mode, "steps": steps, "dtype": "float64", "boundary_median_edge": scale,
            "retained_displacement_budget_normalized": retained_displacement_budget,
            "initial_terms": initial_numeric, "final_terms": final_terms,
            "initial_objective": float(initial_loss.detach()),
            "final_objective": sum(weights[k] * v for k, v in final_terms.items()),
            "weights": weights, "variable_vertices_n": len(variable), "collar_vertices_n": len(collar),
            "affected_retained_faces_in_objective": len(affected_faces), "outside_support_bitwise_equal": True,
            "seam_target_override": seam_target_override is not None,
            "warm_start": start_vertices is not None,
            "retained_displacement_reference": "original_reference_vertices",
            "executed_steps": 0 if hold_initializer else steps,
            "fixed_feature_vertices_n": len(fixed_feature),
            "fixed_feature_vertices_bitwise_equal": bool(np.array_equal(out[fixed_feature], vertices[fixed_feature])),
            **gradient_info}
    if guidance is not None:
        info["feature_guidance"] = guidance["summary"]
    if guard is not None:
        info["feasible_step"] = guard.diagnostics()
        guard.close()
    elif guard_fallback is not None:
        info["feasible_step"] = guard_fallback
    return out, info
