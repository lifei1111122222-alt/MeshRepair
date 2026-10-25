"""Movable-neighborhood comparators using the existing prescribed targets.

The formal comparator (--cgal) calls official CGAL fair on patch+two-ring
vertices, using its default C1 bi-Laplacian solve with an unchanged exterior.

The earlier independent Calatroni smooth-joint adaptation is diagnostic only:
equations (6), (17)--(28), M_E=0, an unknown patch+two-ring mask, and Dirichlet
exterior. Its numerical instability excludes it from manuscript method ranking.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
# Clarabel's current Windows abi3 binary needs the bundled modern Python runtime.
# Its isolated SciPy wheel must not shadow the older experiment interpreter.
if sys.version_info >= (3, 12):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external_baselines/convex_fairing"))
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "experiments/formal_movable_neighborhood"
FORMAL = ROOT / "experiments/formal_collar_mechanism/results.json"
DEVELOPMENT = ROOT / "experiments/joint_collar_complete201_v4/frozen_queue201.json"
CONVEX_PYTHON = Path(os.environ.get("TCMR_CONVEX_PYTHON", sys.executable))
MESHLAB_PYTHON = Path(os.environ.get("TCMR_MESHLAB_PYTHON", sys.executable))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collar_vertices(faces, loop):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges = np.unique(edges, axis=0)
    chosen = np.unique(loop)
    for _ in range(2):
        chosen = np.unique(np.r_[chosen, edges[np.isin(edges, chosen).any(axis=1)].ravel()])
    return chosen


def normals(vertices, faces):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(cross, axis=1)
    return cross / np.maximum(length[:, None], 1e-15), length


def adjacency(faces):
    edge_to_faces = {}
    for i, face in enumerate(faces):
        for j in range(3):
            edge = tuple(sorted((int(face[j]), int(face[(j + 1) % 3]))))
            edge_to_faces.setdefault(edge, []).append(i)
    entries = [(edge, pair) for edge, pair in edge_to_faces.items() if len(pair) == 2]
    return np.asarray([x[0] for x in entries], dtype=np.int64), np.asarray([x[1] for x in entries], dtype=np.int64)


def optimize(initial, retained, patch, loop, retained_n, beta, gradient_check=False):
    """Smooth branch ADMM with three Armijo GD updates in each V subproblem."""
    start = time.perf_counter()
    h = float(np.median(np.linalg.norm(initial[loop] - initial[np.roll(loop, -1)], axis=1)))
    origin = initial[loop].mean(axis=0)
    vertices = (initial.copy() - origin) / h
    variable = np.unique(np.r_[collar_vertices(retained, loop), np.unique(patch)[np.unique(patch) >= retained_n]])
    faces = np.vstack([retained, patch])
    affected = np.flatnonzero(np.isin(faces, variable).any(axis=1))
    edges, face_pairs = adjacency(faces)
    keep = np.isin(face_pairs, affected).any(axis=1)
    edges, face_pairs = edges[keep], face_pairs[keep]
    face_ids = np.unique(face_pairs)
    # Isolated affected triangles do not contribute any two-face normal term.
    face_ids = np.unique(np.r_[face_ids, affected])
    face_map = np.full(len(faces), -1, dtype=np.int64)
    face_map[face_ids] = np.arange(len(face_ids))
    local_faces = faces[face_ids]
    pairs = face_map[face_pairs]
    free = face_map[affected]
    fixed = np.setdiff1d(np.arange(len(face_ids)), free)
    actual, _ = normals(vertices, local_faces)
    field = actual.copy()
    rho1 = np.zeros((len(edges), 3))
    rho2 = np.zeros_like(actual)
    all_rows = np.repeat(np.arange(len(edges)), 2)
    all_cols = pairs.ravel()
    signs = np.tile([1., -1.], len(edges))
    affected_faces = faces[affected]
    history = []
    gradcheck = None
    initial_actual = actual.copy()
    initial_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    initial_energy = float(np.square(initial_lengths[:, None] * (actual[pairs[:, 0]] - actual[pairs[:, 1]])).sum())
    reason = "iteration_cap"
    failures = 0
    for iteration in range(200):
        old_vertices = vertices[variable].copy()
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        D = sparse.csr_matrix((np.repeat(lengths, 2) * signs, (all_rows, all_cols)), shape=(len(edges), len(face_ids)))
        # Eq. (22), M_E=0: exact quadratic proximal update.
        t = (beta * (D @ field) + rho1) / (beta + 2.)
        Dfree = D[:, free]
        constant = D[:, fixed] @ actual[fixed]
        # Eq. (25) with fixed exterior normal rows eliminated.
        rhs = actual[free] + rho2[free] / beta + Dfree.T @ (t - rho1 / beta - constant)
        A = (Dfree.T @ Dfree + sparse.eye(len(free), format="csr")).tocsc()
        field[free] = spsolve(A, rhs)
        field[free] /= np.maximum(np.linalg.norm(field[free], axis=1, keepdims=True), 1e-15)
        field[fixed] = actual[fixed]
        z = rho2[free] - beta * field[free]

        def value_grad(points, need_grad=True):
            n, area2 = normals(points, affected_faces)
            value = float(np.sum(z * n))  # omitted constants of Eq. (26)
            if not need_grad:
                return value
            projected = (z - np.sum(z * n, axis=1, keepdims=True) * n) / np.maximum(area2[:, None], 1e-15)
            tri = points[affected_faces]
            grad = np.zeros_like(points)
            for j in range(3):
                g = np.cross(projected, tri[:, (j + 2) % 3] - tri[:, (j + 1) % 3])
                np.add.at(grad, affected_faces[:, j], g)
            return value, grad[variable]

        if gradient_check and iteration == 0:
            _, gradient = value_grad(vertices)
            errors = []
            for position in np.linspace(0, len(variable) - 1, min(5, len(variable)), dtype=int):
                for coordinate in range(3):
                    shifted = vertices.copy()
                    shifted[variable[position], coordinate] += 1e-6
                    plus = value_grad(shifted, False)
                    shifted[variable[position], coordinate] -= 2e-6
                    minus = value_grad(shifted, False)
                    numeric = (plus - minus) / 2e-6
                    exact = gradient[position, coordinate]
                    errors.append(abs(numeric - exact) / max(1., abs(exact), abs(numeric)))
            gradcheck = max(errors)
            if gradcheck > 1e-5:
                raise ValueError(f"Eq28 analytic gradient finite-difference error {gradcheck}")

        for _ in range(3):
            value, grad = value_grad(vertices)
            norm2 = float(np.square(grad).sum())
            if norm2 < 1e-24:
                break
            alpha = 1.
            accepted = False
            for _backtrack in range(40):
                proposal = vertices.copy()
                proposal[variable] -= alpha * grad
                new_value = value_grad(proposal, False)
                if np.isfinite(new_value) and new_value <= value - 1e-4 * alpha * norm2:
                    vertices = proposal
                    accepted = True
                    break
                alpha *= .5
            if not accepted:
                failures += 1
                break
        actual, _ = normals(vertices, local_faces)
        rho1 -= beta * (t - D @ field)
        rho2[free] -= beta * (field[free] - actual[free])
        relative = float(np.linalg.norm(vertices[variable] - old_vertices) / max(np.linalg.norm(old_vertices), 1e-15))
        lengths_new = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        energy = float(np.square(lengths_new[:, None] * (actual[pairs[:, 0]] - actual[pairs[:, 1]])).sum())
        residual = float(np.sqrt(np.mean(np.square(field[free] - actual[free]))))
        if iteration == 0 or (iteration + 1) % 20 == 0 or relative < 1e-6:
            history.append({"iteration": iteration + 1, "objective": energy, "relative_step": relative, "normal_primal_rms": residual})
        if relative < 1e-6:
            reason = "relative_step_tolerance"
            break
        if time.perf_counter() - start > 120:
            reason = "optimization_time_cap"
            break
    result = initial.copy()
    result[variable] = vertices[variable] * h + origin
    return result, {"method": "Calatroni2023_smooth_joint_local_adaptation", "beta1": beta, "beta2": beta,
                    "outer_iterations": iteration + 1, "inner_GD_per_outer": 3, "stop_reason": reason,
                    "initial_objective": initial_energy, "final_objective": energy,
                    "normal_primal_rms": residual, "relative_step": relative,
                    "line_search_failures": failures, "variable_vertices_n": len(variable),
                    "normal_field_variable_faces_n": len(free), "gradient_check_relative_error": gradcheck,
                    "hard_retained_displacement_bound": None, "history": history,
                    "optimizer_seconds": time.perf_counter() - start}


def attachment_solve(initial, retained, patch, loop, retained_n, gradient_check=False,
                     *, outer_max=200, defer_relative_stop=False, checkpoints=None):
    """Independent Eq6 context-completion implementation, soft observed fidelity."""
    from scripts.cgal_local_hole_adapter import _target_component, _validate_boundary
    start = time.perf_counter()
    component = _target_component(retained, _validate_boundary(initial[:retained_n], retained, loop))
    full_faces = np.vstack([component, patch])
    ids = np.unique(full_faces)
    inverse = np.full(len(initial), -1, dtype=np.int64)
    inverse[ids] = np.arange(len(ids))
    faces, rim = inverse[full_faces], inverse[loop]
    h = float(np.median(np.linalg.norm(initial[loop]-initial[np.roll(loop, -1)], axis=1)))
    origin = initial[loop].mean(axis=0)
    vertices = (initial[ids].copy()-origin)/h
    base = vertices.copy()
    edges, pairs = adjacency(faces)
    strip = collar_vertices(faces, rim)
    fidelity = np.ones(len(vertices))
    fidelity[strip] = 0.
    actual, _ = normals(vertices, faces)
    field = actual.copy()
    angles = np.arccos(np.clip(np.sum(actual[pairs[:, 0]]*actual[pairs[:, 1]], axis=1), -1., 1.))
    feature = (angles > np.deg2rad(30.)) & ~np.isin(edges, strip).any(axis=1)
    beta1, beta2, lam = 2., 100., 100.
    rho1 = np.zeros((len(edges), 3))
    rho2 = np.zeros_like(actual)
    rows, cols = np.repeat(np.arange(len(edges)), 2), pairs.ravel()
    signs = np.tile([1., -1.], len(edges))
    history, errors, failures = [], [], 0

    def objective(x):
        n, _ = normals(x, faces)
        lengths = np.linalg.norm(x[edges[:, 0]]-x[edges[:, 1]], axis=1)
        u = np.linalg.norm(lengths[:, None]*(n[pairs[:, 0]]-n[pairs[:, 1]]), axis=1)
        mc = np.where(u < np.sqrt(2.), -.5*u*u+np.sqrt(2.)*u, 1.)
        reg = float(np.where(feature, mc, u*u).sum())
        fid = float(.5*lam*np.sum(fidelity[:, None]*(x-base)**2))
        return reg+fid

    initial_energy = objective(vertices)
    reason = "iteration_cap"
    first_relative_stop = None
    for iteration in range(outer_max):
        old = vertices.copy()
        lengths = np.linalg.norm(vertices[edges[:, 0]]-vertices[edges[:, 1]], axis=1)
        D = sparse.csr_matrix((np.repeat(lengths, 2)*signs, (rows, cols)), shape=(len(edges), len(faces)))
        r = (beta1*(D@field)+rho1)/(beta1+2.*(~feature)[:, None])
        shrink = np.clip(2.-np.sqrt(2.)/np.maximum(np.linalg.norm(r, axis=1), 1e-30), 0., 1.)
        t = r.copy()
        t[feature] *= shrink[feature, None]
        A = (D.T@D+(beta2/beta1)*sparse.eye(len(faces), format="csr")).tocsc()
        field = spsolve(A, (beta2/beta1)*actual+rho2/beta1+D.T@(t-rho1/beta1))
        field /= np.maximum(np.linalg.norm(field, axis=1, keepdims=True), 1e-15)
        z = rho2-beta2*field

        def value_grad(x, need_grad=True):
            n, area2 = normals(x, faces)
            delta = x-base
            # Remove a coordinate-independent constant before the Armijo and
            # finite-difference evaluations to avoid cancellation on big meshes.
            value = float(.5*lam*np.sum(fidelity[:, None]*delta*delta)+np.sum(z*(n-actual)))
            if not need_grad:
                return value
            proj = (z-np.sum(z*n, axis=1, keepdims=True)*n)/np.maximum(area2[:, None], 1e-15)
            tri = x[faces]
            grad = lam*fidelity[:, None]*delta
            for j in range(3):
                np.add.at(grad, faces[:, j], np.cross(proj, tri[:, (j+2)%3]-tri[:, (j+1)%3]))
            return value, grad

        if gradient_check and iteration == 0:
            _, g = value_grad(vertices)
            for pos in np.linspace(0, len(vertices)-1, 8, dtype=int):
                for coord in range(3):
                    shift = vertices.copy()
                    shift[pos, coord] += 1e-6
                    plus = value_grad(shift, False)
                    shift[pos, coord] -= 2e-6
                    numeric = (plus-value_grad(shift, False))/2e-6
                    errors.append(abs(numeric-g[pos, coord])/max(1., abs(numeric), abs(g[pos, coord])))
        for inner in range(3):
            val, g = value_grad(vertices)
            norm2 = float(np.square(g).sum())
            if norm2 < 1e-24:
                break
            step, accepted = 1., False
            for backtrack in range(40):
                proposal = vertices-step*g
                nv = value_grad(proposal, False)
                if np.isfinite(nv) and nv <= val-1e-4*step*norm2:
                    vertices, accepted = proposal, True
                    break
                step *= .5
            if not accepted:
                failures += 1
                break
        actual, _ = normals(vertices, faces)
        rho1 -= beta1*(t-D@field)
        rho2 -= beta2*(field-actual)
        relative = float(np.linalg.norm(vertices-old)/max(np.linalg.norm(old), 1e-15))
        residual = float(np.sqrt(np.mean(np.square(field-actual))))
        energy = objective(vertices)
        if relative < 1e-6 and first_relative_stop is None:
            first_relative_stop = iteration+1
        if checkpoints is not None and (iteration+1 in (200, 400) or iteration+1 == first_relative_stop):
            checkpoint = initial.copy()
            checkpoint[ids] = vertices*h+origin
            checkpoints[iteration+1] = {"vertices": checkpoint, "objective": energy,
                "relative_step": relative, "normal_primal_rms": residual}
        if iteration == 0 or (iteration+1)%20 == 0 or relative < 1e-6:
            history.append({"iteration": iteration+1, "objective": energy, "relative_step": relative, "normal_primal_rms": residual})
        if relative < 1e-6 and not defer_relative_stop:
            reason = "relative_step_tolerance"
            break
        if time.perf_counter()-start > 300:
            reason = "optimization_time_cap"
            break
    after = initial.copy()
    after[ids] = vertices*h+origin
    passed = (np.isfinite(after).all() and not failures and energy <= initial_energy*(1+1e-8)
              and residual <= .05 and reason != "optimization_time_cap" and max(errors or [0.]) < 1e-5)
    fidelity_energy = float(.5*lam*np.sum(fidelity[:, None]*(vertices-base)**2))
    return after, {"method": "calatroni_attachment", "outer_iterations": iteration+1, "stop_reason": reason,
                   "initial_objective": initial_energy, "final_objective": energy, "objective_ratio": energy/initial_energy,
                   "initial_fidelity_energy": 0., "initial_normal_regularization_energy": initial_energy,
                   "final_fidelity_energy": fidelity_energy, "final_normal_regularization_energy": energy-fidelity_energy,
                   "normal_primal_rms": residual, "relative_step": relative, "line_search_failures": failures,
                   "gradient_check_relative_error": max(errors) if errors else None, "variable_vertices_n": len(ids),
                   "zero_fidelity_strip_vertices_n": len(strip), "fidelity_vertices_n": int(fidelity.sum()),
                   "feature_edges_n": int(feature.sum()), "development_numerically_accepted": bool(passed),
                   "first_relative_step_tolerance_iteration": first_relative_stop,
                   "history": history, "optimizer_seconds": time.perf_counter()-start}


def attachment_adequacy():
    """Read all saved formal traces; one fixed, development-only budget diagnostic.

    No formal candidate is recomputed or replaced. Deferring the relative-step
    stop is explicit: all three original development cases otherwise terminate
    before 200, so raising only the iteration ceiling would be uninformative.
    The ADMM updates, masks, scaling, parameters and 300 s timeout are unchanged.
    """
    from scripts.evaluate_external_benchmark import geometry_metrics

    destination = OUT / "attachment_adequacy.json"
    if destination.exists() and load(destination).get("complete"):
        print("Existing complete adequacy evidence retained", flush=True)
        return

    def stats(values):
        v = np.asarray(values, float)
        return {"n": len(v), "min": float(np.min(v)), "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "max": float(np.max(v))} if len(v) else {"n": 0}

    formal = load(OUT / "attachment_results.json")
    trace_rows = []
    for row in formal["rows"]:
        info = row["methods"]["calatroni_attachment"].get("optimizer")
        if info is None:
            continue
        hist = info["history"]
        final = hist[-1]
        prev = hist[-2] if len(hist) > 1 else None
        trace_rows.append({"dataset": row["dataset"], "index": row["index"],
            "source_model_id": row["source_model_id"], "stop_reason": info["stop_reason"],
            "outer_iterations": info["outer_iterations"], "last": final,
            "previous": prev,
            "final_window_objective_relative_reduction": ((prev["objective"]-final["objective"])/max(abs(prev["objective"]), 1e-15)) if prev else None,
            "final_window_span_iterations": final["iteration"]-prev["iteration"] if prev else None,
            "initial_to_final_objective_ratio": info["objective_ratio"],
            "last_window_geometry_saved": False})
    caps = [r for r in trace_rows if r["stop_reason"] == "iteration_cap"]
    result = {"complete": False, "protocol": {
        "formal_source": str(OUT / "attachment_results.json"),
        "formal_source_sha256": digest(OUT / "attachment_results.json"),
        "formal_inference_recomputed": False, "formal_output_replaced": False,
        "development_queue": str(DEVELOPMENT), "development_positions": [0, 1, 2],
        "development_extension_trials": 1, "outer_max": 400,
        "relative_step_stop_deferred_for_diagnostic_only": True,
        "reason": "All original three development runs stop before 200; recording first stop and continuing unchanged updates permits a meaningful 200/400 comparison.",
        "optimization_parameters_changed": False, "original_timeout_seconds": 300,
        "no_optimality_claim": "Objective reduction and small field-normal residual are numerical diagnostics, not stationarity or global convergence certificates.",
        "geometry_scoring": "Unchanged unified geometry_metrics; no native safety rerun or formal eligibility decision in this development-only diagnostic."},
        "formal_trace_summary": {"returned_n": len(trace_rows), "iteration_cap_n": len(caps),
            "relative_step_stop_n": sum(r["stop_reason"] == "relative_step_tolerance" for r in trace_rows),
            "first_iteration_stop_n": sum(r["outer_iterations"] == 1 for r in trace_rows),
            "zero_final_relative_step_n": sum(r["last"]["relative_step"] == 0 for r in trace_rows),
            "cap_180_to_200_objective_relative_reduction": stats([r["final_window_objective_relative_reduction"] for r in caps]),
            "cap_180_to_200_objective_increase_n": sum(r["final_window_objective_relative_reduction"] < 0 for r in caps),
            "cap_final_relative_step": stats([r["last"]["relative_step"] for r in caps]),
            "cap_final_normal_primal_rms": stats([r["last"]["normal_primal_rms"] for r in caps]),
            "cap_normal_residual_increase_last_window_n": sum(r["last"]["normal_primal_rms"] > r["previous"]["normal_primal_rms"] for r in caps),
            "geometry_window_limitation": "Saved formal logs contain energy, relative coordinate step and normal-field consistency every 20 iterations, not saved late-window coordinates. These cannot establish stable seam or shape metrics."},
        "formal_trace_rows": trace_rows, "development_rows": []}
    save(destination, result)
    records = load(DEVELOPMENT)["records"]
    original_dev = {r["position"]: r for r in load(OUT / "attachment_development.json")["rows"]}
    for position in [0, 1, 2]:
        before, retained, loop, initial, patch = arrays(records[position])
        checkpoints = {}
        after, info = attachment_solve(initial, retained, patch, loop, len(before),
            outer_max=400, defer_relative_stop=True, checkpoints=checkpoints)
        raw_path = OUT / f"attachment_adequacy_dev{position:03d}.npz"
        np.savez_compressed(raw_path, patch_faces=patch,
            **{f"vertices_{k}": c["vertices"] for k, c in checkpoints.items()})
        measured = {}
        for k, c in checkpoints.items():
            measured[str(k)] = {name: value for name, value in c.items() if name != "vertices"}
            measured[str(k)]["geometry"] = geometry_metrics(before, retained, c["vertices"], patch, loop)
        comparison = None
        if 200 in checkpoints and 400 in checkpoints:
            left, right = checkpoints[200], checkpoints[400]
            h = float(np.median(np.linalg.norm(initial[loop]-initial[np.roll(loop, -1)], axis=1)))
            delta = np.linalg.norm(right["vertices"]-left["vertices"], axis=1)/max(h, 1e-15)
            h_in = measured["200"]["geometry"]["median_input_edge"]
            delta_input = delta*h/max(h_in, 1e-15)
            g0, g1 = measured["200"]["geometry"], measured["400"]["geometry"]
            comparison = {"objective_relative_reduction": (left["objective"]-right["objective"])/max(abs(left["objective"]), 1e-15),
                "normal_rms_difference": right["normal_primal_rms"]-left["normal_primal_rms"],
                "full_vertex_rms_change_over_rim_edge": float(np.sqrt(np.mean(delta*delta))),
                "maximum_vertex_change_over_rim_edge": float(delta.max()),
                "retained_rms_change_over_rim_edge": float(np.sqrt(np.mean(delta[:len(before)]**2))),
                "new_patch_vertex_rms_change_over_rim_edge": float(np.sqrt(np.mean(delta[len(before):]**2))) if len(delta)>len(before) else 0.,
                "full_vertex_rms_change_over_input_edge": float(np.sqrt(np.mean(delta_input**2))),
                "maximum_vertex_change_over_input_edge": float(delta_input.max()),
                "retained_rms_change_over_input_edge": float(np.sqrt(np.mean(delta_input[:len(before)]**2))),
                "new_patch_vertex_rms_change_over_input_edge": float(np.sqrt(np.mean(delta_input[len(before):]**2))) if len(delta_input)>len(before) else 0.,
                "seam_mean_difference": g1["seam_normal_absolute_mean"]-g0["seam_normal_absolute_mean"],
                "seam_p95_difference": g1["seam_normal_absolute_p95"]-g0["seam_normal_absolute_p95"],
                "patch_quality_mean_difference": g1["triangle_quality"]["mean"]-g0["triangle_quality"]["mean"],
                "patch_quality_p05_difference": g1["triangle_quality"]["p05"]-g0["triangle_quality"]["p05"],
                "patch_quality_min_difference": g1["triangle_quality"]["min"]-g0["triangle_quality"]["min"]}
        first = info["first_relative_step_tolerance_iteration"]
        old = original_dev[position]["optimizer"]
        checkpoint_first = measured.get(str(first))
        result["development_rows"].append({"position": position,
            "sample_id": records[position]["sample_id"], "source_model_id": records[position]["source_model_id"],
            "raw_checkpoints": str(raw_path), "optimizer": info, "checkpoints": measured,
            "first_stop_matches_original_iteration": first == old["outer_iterations"],
            "first_stop_objective_minus_original": checkpoint_first["objective"]-old["final_objective"] if checkpoint_first else None,
            "comparison_200_to_400": comparison})
        save(destination, result)
        print(json.dumps({"position": position, "iterations": info["outer_iterations"], "first_stop": first,
                          "comparison_200_to_400": comparison}), flush=True)
    result["complete"] = True
    result["interpretation"] = {"formal_status": "Budget-limited independent adaptation, not demonstrated fully converged author performance.",
        "development_status": "Three fixed development examples only; continued iterates are not used to select parameters or replace formal results.",
        "residual_scope": "Normal primal RMS measures field/geometric-normal consistency only. The .05 numerical acceptance gate is not an ADMM optimality or stopping tolerance.",
        "required_claim_scope": "Compare the predeclared independent adaptation at its stated budget; retain the closest-method model and implementation limitations."}
    save(destination, result)


def attachment_protocol():
    return {"method": "Calatroni2023 context smooth-attachment independent adaptation", "author_implementation": False,
            "reference_doi": "10.1007/s10851-022-01107-w",
            "primary_paper": "https://link.springer.com/article/10.1007/s10851-022-01107-w",
            "author_publications": "https://sites.google.com/view/lucacalatroni/papers",
            "author_code_search": "Publisher and author publication page do not link author code; exact title and author repository searches found no trusted implementation.",
            "equations": "Eq6 spatial MC/quadratic mask, Eq17-28 ADMM, three warm-start Armijo GD per V subproblem, Eq35 stopping",
            "domain": "Entire retained target connected component plus identical saved CGAL initializer patch; every component coordinate and face normal is free; other components unchanged",
            "fidelity_mask": "Zero on two edge-ring strip of COMBINED retained+patch mesh around rim; one on remote retained AND initialized patch template interior. Template-only mask extension is an explicit fidelity-anchoring adaptation, not a reported author default.",
            "objective": "lambda/2 times sum_i M_V[i]*||V[i]-V0[i]||^2 + sum_e M_E[e]*MC(||D(V)n(V)[e]||;a) + (1-M_E[e])*||D(V)n(V)[e]||^2; Eq6 with current edge lengths in D as Eq4",
            "edge_mask": "Smooth attachment: M_E=0 on edges with any strip endpoint. Else initial oriented dihedral>30 degrees sets M_E=1. Fixed MC mask, a=1.",
            "parameters": {"lambda": 100., "a": 1., "beta1": 2., "beta2": 100., "coordinate_scale": "median target rim edge"},
            "parameter_provenance": "Paper gives no numerical lambda/beta defaults or mandatory coordinate scaling. One predeclared configuration, no tuning: beta1=2>a ensures MC proximal convexity; beta2=100 and lambda=100 emphasize geometric-normal consistency and template/observed fidelity at unit boundary-edge scale. These are independent adaptation choices, not author defaults.",
            "adaptations": ["Same CGAL initializer acts as prepositioned template; no known missing-surface template or reference geometry.",
                            "No hard displacement bound, remeshing, quality/orientation/seam-only penalty, TCMR projection, or verification-dependent optimization.",
                            "Smooth attachment configuration only; no re-opening sharp-feature recovery development."],
            "value_evaluation": "Eq26 omits the coordinate-independent sum z*n_current for numerical cancellation control; gradient unchanged",
            "optimization": {"outer_max": 200, "relative_step_tolerance": 1e-6, "inner_GD": 3, "Armijo_c1": 1e-4,
                             "backtracking_factor": .5, "initial_step": 1., "backtracking_max": 40, "timeout_seconds": 300, "dtype": "float64"},
            "development": {"queue": str(DEVELOPMENT), "positions": [0, 1, 2], "parameter_trials": 1,
                            "acceptance": "All3 finite; Eq26 gradient relative error<1e-5; zero Armijo failures; final original Eq6 objective<=initial*(1+1e-8); final normal RMS<=.05; no timeout. If any fail, do not run/rank on formal data and do not tune."},
            "formal_queue": str(FORMAL), "formal_queue_sha256": digest(FORMAL), "planned_n": 110,
            "parallel_workers": 2, "math_threads_per_worker": 2,
            "statistics": "Separate50/60; 20000 paired source bootstrap seed20261025; all-source validity denominator; common-native-valid geometry; same original three fallback alternatives"}


def attachment_develop():
    OUT.mkdir(parents=True, exist_ok=True)
    p = attachment_protocol()
    path = OUT / "attachment_protocol.json"
    prior = load(OUT / "attachment_development.json") if (OUT / "attachment_development.json").exists() else None
    if path.exists() and load(path) != p:
        old = load(path)
        old["value_evaluation"] = p["value_evaluation"]
        if old != p:
            raise ValueError("Frozen attachment protocol differs")
    save(path, p)
    queue, rows = load(DEVELOPMENT)["records"], []
    for position in [0, 1, 2]:
        before, retained, loop, initial, patch = arrays(queue[position])
        _, info = attachment_solve(initial, retained, patch, loop, len(before), gradient_check=True)
        rows.append({"position": position, "sample_id": queue[position]["sample_id"],
                     "source_model_id": queue[position]["source_model_id"], "optimizer": info})
        save(OUT / "attachment_development.json", {"complete": len(rows) == 3,
             "passed": len(rows) == 3 and all(r["optimizer"]["development_numerically_accepted"] for r in rows), "rows": rows,
             "uncentered_value_gradient_diagnostic": prior.get("uncentered_value_gradient_diagnostic", prior) if prior else None})
        print(json.dumps(rows[-1]), flush=True)


def attachment_generate(position):
    record = load(FORMAL)["rows"][position]
    before, retained, loop, initial, patch = arrays(record)
    after, info = attachment_solve(initial, retained, patch, loop, len(before))
    destination = OUT / f"attachment_{record['dataset']}_{record['index']:03d}.npz"
    np.savez_compressed(destination, vertices=after, patch_faces=patch)
    print("RESULT="+json.dumps(info), flush=True)


def attachment_worker(position):
    record = load(FORMAL)["rows"][position]
    key = f"attachment_{record['dataset']}_{record['index']:03d}"
    destination = OUT / f"{key}.npz"
    verification_path = OUT / f"{key}_verification.tmp.json"
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    row = {k: record[k] for k in ("dataset", "index", "sample_id", "source_model_id", "category", "input_path")}
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--attachment-generate", str(position)],
                                 cwd=ROOT, capture_output=True, text=True, timeout=360, env=env,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if process.returncode:
            raise RuntimeError(f"Attachment generation failed:{process.returncode} {process.stderr[-1000:]}")
        info = json.loads(next(line[7:] for line in process.stdout.splitlines() if line.startswith("RESULT=")))
        started = time.perf_counter()
        process = subprocess.run([sys.executable, str(ROOT/"scripts/cgal_changed_surface_audit.py"),
                                 "--input", record["input_path"], "--patch", str(destination), "--output", str(verification_path)],
                                 cwd=ROOT, capture_output=True, text=True, timeout=100,
                                 env=dict(env, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if process.returncode or not verification_path.exists():
            raise RuntimeError(f"Attachment verifier failed:{process.returncode} {process.stderr[-1000:]}")
        native = load(verification_path)
        verify_seconds = time.perf_counter()-started
        from scripts.evaluate_collar_controls import score_saved_candidate
        item = score_saved_candidate(Path(record["input_path"]), destination, native, "calatroni_attachment")
        item.update(optimizer=info, optimizer_seconds=info["optimizer_seconds"], verification_seconds=verify_seconds)
        verification_path.unlink()
    except Exception as exc:
        item = {"method": "calatroni_attachment", "status": "candidate_failure", "valid": False, "safe_eligible": False, "error": str(exc)}
    row["methods"] = {"calatroni_attachment": item}
    return position, row


def run_attachment():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    p = attachment_protocol()
    if not load(OUT/"attachment_development.json").get("passed"):
        raise ValueError("Three-source attachment numerical development must pass")
    prior = load(OUT/"attachment_protocol.json")
    # Resource-count metadata was fixed before the first formal inference.
    prior.update(parallel_workers=2, math_threads_per_worker=2)
    if prior != p:
        raise ValueError("Frozen numerical attachment protocol differs")
    save(OUT/"attachment_protocol.json", p)
    queue = load(FORMAL)["rows"]
    destination = OUT/"attachment_results.json"
    old = load(destination) if destination.exists() else {"rows": []}
    positions = {(r["dataset"], r["index"]): i for i, r in enumerate(queue)}
    finished = {positions[(r["dataset"], r["index"])]: r for r in old["rows"]}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attachment_worker, i) for i in range(len(queue)) if i not in finished]
        for future in as_completed(futures):
            position, row = future.result()
            finished[position] = row
            rows = [finished[i] for i in sorted(finished)]
            result = {"complete": len(rows)==len(queue), "protocol": p,
                      "elapsed_seconds": time.perf_counter()-started, "rows": rows}
            if result["complete"]:
                result["summary"] = summarize(rows, "calatroni_attachment")
                result["fallback"] = two_step_fallback(rows, "calatroni_attachment", "calatroni_attachment")
                infos = [r["methods"]["calatroni_attachment"]["optimizer"] for r in rows if "optimizer" in r["methods"]["calatroni_attachment"]]
                result["numerical_summary"] = {"returned_n": len(infos),
                    "objective_increase_n": sum(v["final_objective"]>v["initial_objective"]*(1+1e-8) for v in infos),
                    "normal_rms_above_development_gate_n": sum(v["normal_primal_rms"]>.05 for v in infos),
                    "normal_primal_rms_max": max(v["normal_primal_rms"] for v in infos),
                    "line_search_failures": sum(v["line_search_failures"] for v in infos)}
            save(destination, result)
            item = row["methods"]["calatroni_attachment"]
            print(json.dumps({"completed": len(rows), "planned": len(queue), "case": position,
                              "status": item["status"], "valid": item.get("valid"), "elapsed_seconds": result["elapsed_seconds"],
                              "error": item.get("error")}), flush=True)


def arrays(record):
    with np.load(record["input_path"], allow_pickle=False) as z:
        before, retained, loop = z["vertices"].astype(float), z["faces"].astype(np.int64), z["selected_loop"].astype(np.int64)
    path = record.get("initializer_path") or record["methods"]["initializer"]["patch_path"]
    with np.load(path, allow_pickle=False) as z:
        initial, patch = z["vertices"].astype(float), z["patch_faces"].astype(np.int64)
    return before, retained, loop, initial, patch


def protocol():
    return {"method": "Calatroni et al., JMIV 65:82-98 (2023), smooth-joint local adaptation",
            "doi": "10.1007/s10851-022-01107-w", "author_code": False,
            "equations": "Eq6 with M_E=0; exact quadratic t Eq22; normal solve Eq25 and row normalization; vertex Eq26/28, three Armijo GD steps; multipliers Eq20/21",
            "objective": "sum_e || l_e(V) [n_left(V)-n_right(V)] ||_2^2 over two-face edges adjacent to affected faces; original Eq6 smooth branch; fidelity is zero on unknown patch+collar and exterior coordinates are fixed",
            "normal_variables": "Affected faces incident to new patch vertices or retained two-ring vertices. Adjacent exterior faces supply fixed geometric normals. Solve Eq25 after eliminating fixed normal rows, then normalize each free row to unit length.",
            "update_details": "Update edge lengths D at each outer iteration; quadratic t=(beta1*D*N+rho1)/(beta1+2). Eliminate exterior rows in sparse normal solve. Three warm-start Armijo GD updates use Eq28 analytic derivative of normal map. Update duals with the current outer-iteration D and new vertex normals.",
            "scope": "Hole-filling mask expanded to an interface narrow band; not the template-interior fidelity configuration and not the complete adaptive feature-aware author system.",
            "author_code_search": "Publisher article and first-author publication list link paper/preprint but no implementation; title/author repository search found no author implementation.",
            "adaptations": ["Same existing CGAL initializer and connectivity as paired controls.",
                            "Unknown mask is all new patch vertices plus retained boundary two-ring.",
                            "Vertices and normals exterior to the local active domain are Dirichlet fixed instead of exterior soft fidelity.",
                            "No TCMR displacement projection, quality term, orientation penalty, seam-only loss, or verification-dependent optimization.",
                            "Smooth-joint branch M_E=0 throughout active domain; does not reproduce spatially adaptive MC feature-preserving branch.",
                            "Coordinates centered at boundary centroid and divided by median boundary edge; penalties selected on three development inputs only."],
            "development_selection": {"queue": str(DEVELOPMENT), "positions": [0, 1, 2], "beta_candidates": [.5, 2., 8.],
                                      "rule": "Minimize median final/initial smooth objective among finite runs; ties choose lower beta. No evaluation of final 110 test outcomes is used."},
            "optimization": {"outer_max": 200, "relative_step_tolerance": 1e-6, "inner_GD": 3,
                             "Armijo_c1": 1e-4, "backtracking_factor": .5, "initial_step": 1., "backtracking_max": 40,
                             "timeout_seconds": 120, "dtype": "float64"},
            "formal_queue": str(FORMAL), "formal_queue_sha256": digest(FORMAL), "planned_n": 110,
            "statistics": "Separate 50/60 cohorts; 20000 paired source bootstrap draws seed20261025; all sources in validity denominator; geometry on common-valid outputs.",
            "verification": "Same changed-surface CGAL checker in isolated subprocess, once per returned output; no rejection of raw output prior to scoring.",
            "parallel_workers": 1, "math_threads": 2}


def develop():
    OUT.mkdir(parents=True, exist_ok=True)
    p = protocol()
    if (OUT / "protocol.json").exists() and load(OUT / "protocol.json") != p:
        raise ValueError("Existing protocol differs")
    save(OUT / "protocol.json", p)
    queue = load(DEVELOPMENT)["records"]
    rows = []
    for i in [0, 1, 2]:
        before, retained, loop, initial, patch = arrays(queue[i])
        for beta in [.5, 2., 8.]:
            _, info = optimize(initial, retained, patch, loop, len(before), beta, gradient_check=(i == 0 and beta == .5))
            row = {"position": i, "sample_id": queue[i]["sample_id"], "source_model_id": queue[i]["source_model_id"],
                   "beta": beta, "objective_ratio": info["final_objective"] / info["initial_objective"], "optimizer": info}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("position", "beta", "objective_ratio")}), flush=True)
    scores = {str(b): float(np.median([r["objective_ratio"] for r in rows if r["beta"] == b])) for b in [.5, 2., 8.]}
    chosen = min([.5, 2., 8.], key=lambda b: (scores[str(b)], b))
    save(OUT / "development.json", {"complete": True, "rows": rows, "median_objective_ratio": scores, "selected_beta": chosen})
    print(json.dumps({"selected_beta": chosen, "scores": scores}), flush=True)


def worker(position):
    from scripts.evaluate_collar_controls import external_measure
    record = load(FORMAL)["rows"][position]
    beta = load(OUT / "development.json")["selected_beta"]
    before, retained, loop, initial, patch = arrays(record)
    after, info = optimize(initial, retained, patch, loop, len(before), beta)
    key = f"{record['dataset']}_{record['index']:03d}"
    destination = OUT / f"{key}.npz"
    np.savez_compressed(destination, vertices=after, patch_faces=patch)
    verification_path = OUT / f"{key}_verification.json"
    start = time.perf_counter()
    native = subprocess.run([sys.executable, str(ROOT / "scripts/cgal_changed_surface_audit.py"),
                             "--input", record["input_path"], "--patch", str(destination), "--output", str(verification_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=100,
                            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if native.returncode or not verification_path.exists():
        raise RuntimeError(f"CGAL subprocess failed: {native.returncode} {native.stderr[-300:]}")
    item = external_measure(after, before, retained, patch, loop, load(verification_path), destination)
    item.update(optimizer=info, optimizer_seconds=info["optimizer_seconds"], verification_seconds=time.perf_counter() - start)
    row = {k: record[k] for k in ("dataset", "index", "sample_id", "source_model_id", "category", "input_path")}
    row["methods"] = {"calatroni_smooth": item}
    save(OUT / f"{key}.json", row)
    print("RESULT=" + json.dumps(row, allow_nan=False), flush=True)


def summarize(rows, method_key="calatroni_smooth"):
    from scripts.evaluate_collar_controls import paired_statistics, external_value
    refs = {(r["dataset"], r["index"]): r for r in load(FORMAL)["rows"]}
    metrics = ["seam_normal_absolute_mean", "seam_normal_absolute_p95", "triangle_quality.mean", "triangle_quality.p05",
               "retained_displacement_rms_over_edge", "retained_displacement_max_over_edge", "retained_rms_over_boundary_edge"]
    output = {}
    for name in ("holdout50", "cross_category60", "pooled110"):
        chosen = [r for r in rows if r["dataset"] == name or name == "pooled110"]
        if not chosen:
            continue
        items = [r["methods"][method_key] for r in chosen]
        group = {"n": len(chosen), "returned_n": sum(i.get("status") == "success" for i in items),
                 "valid_n": sum(i.get("valid") is True for i in items),
                 "intersection_n": sum(i.get("verification", {}).get("introduced_intersection") is True for i in items),
                 "unknown_or_no_output_n": sum(i.get("verification", {}).get("introduced_intersection") is None for i in items)}
        finite = [i for i in items if i.get("status") == "success"]
        if finite:
            group["max_retained_motion_over_boundary_edge"] = max(i["retained_max_over_boundary_edge"] for i in finite)
            group["mean_retained_max_motion_over_boundary_edge"] = float(np.mean([i["retained_max_over_boundary_edge"] for i in finite]))
            group["median_optimizer_seconds"] = float(np.median([i["optimizer_seconds"] for i in finite]))
            group["mean_optimizer_seconds"] = float(np.mean([i["optimizer_seconds"] for i in finite]))
            group["stop_reason_counts"] = {s: sum(i["optimizer"]["stop_reason"] == s for i in finite) for s in set(i["optimizer"]["stop_reason"] for i in finite)}
        group["paired_joint_minus_baseline"] = {}
        for reference in ("initializer", "patch_only", "joint"):
            common = [r for r in chosen if r["methods"][method_key].get("valid") is True and refs[(r["dataset"], r["index"])]["methods"][reference].get("valid") is True]
            stats = {m: paired_statistics([external_value(refs[(r["dataset"], r["index"])]["methods"][reference], m) for r in common],
                                         [external_value(r["methods"][method_key], m) for r in common], seed=20261025) for m in metrics}
            group["paired_joint_minus_baseline"][reference + "_minus_" + method_key] = stats
        group["valid_joint_minus_baseline"] = paired_statistics([float(refs[(r["dataset"], r["index"])]["methods"]["joint"].get("valid") is True) for r in chosen], [float(r["methods"][method_key].get("valid") is True) for r in chosen], seed=20261025)
        output[name] = group
    return output


def cgal_protocol():
    return {"method": "CGAL movable-neighborhood bi-Laplacian fairing", "author_implementation": "Official CGAL 6.0.1 SWIG distribution",
            "api": "CGAL::Polygon_mesh_processing::fair(polyhedron, free_vertices, fairing_continuity=1)",
            "official_documentation": "https://doc.cgal.org/6.0.1/Polygon_mesh_processing/group__PMP__meshing__grp.html",
            "algorithm_reference": "Botsch and Sorkine, On Linear Variational Surface Deformation Methods, IEEE TVCG 14(1):213-230, 2008",
            "initialization": "Exactly the saved CGAL refine-and-fair initializer used by fixed-collar and bounded-joint controls",
            "free_vertices": "All new patch vertices and retained boundary two-ring; identical coordinate degrees of freedom to bounded joint",
            "fixed_vertices": "Every other vertex, bitwise unchanged; no remeshing or vertex insertion/deletion",
            "displacement_bound": None, "parameters": "CGAL defaults, C1 fairing; no tuning or additional regularization",
            "native_context": "Whole edge-connected retained component incident to selected seam plus existing patch; other components preserved unchanged in output",
            "solver": "Official linear bi-Laplacian system with boundary constraints; no iterative parameter sweep",
            "formal_queue": str(FORMAL), "formal_queue_sha256": digest(FORMAL), "planned_n": 110,
            "statistics": "Separate 50/60 cohorts; 20000 paired source bootstrap draws seed20261025; all sources in validity denominator; geometry on common-valid outputs",
            "resource_limits": "One candidate worker, at most two math threads, CGAL generation and verification isolated from PyTorch; 120 second generation cap, 100 second verification cap",
            "timing_scope": "Descriptive generation/verification only; old frozen methods are reused, so this is not a matched end-to-end timing experiment"}


def cgal_generate(position):
    # This subprocess must never import the torch-dependent shared evaluator.
    from scripts.cgal_local_hole_adapter import _imports, _validate_boundary, _target_component, _polyhedron
    record = load(FORMAL)["rows"][position]
    before, retained, loop, initial, patch = arrays(record)
    start = time.perf_counter()
    incident = _validate_boundary(before, retained, loop)
    component = _target_component(retained, incident)
    faces = np.vstack([component, patch])
    ids = np.unique(faces)
    inverse = np.full(len(initial), -1, dtype=np.int64)
    inverse[ids] = np.arange(len(ids))
    Point, Polyhedron, Modifier, pmp = _imports()
    poly = _polyhedron(initial[ids], inverse[faces], Polyhedron, Modifier, Point)
    variable = np.unique(np.r_[collar_vertices(retained, loop), np.unique(patch)[np.unique(patch) >= len(before)]])
    if (inverse[variable] < 0).any():
        raise ValueError("Free vertices extend outside the native target component")
    handles = list(poly.vertices())
    solver_start = time.perf_counter()
    solver_return = pmp.fair(poly, [handles[int(inverse[i])] for i in variable], 1)
    solver_seconds = time.perf_counter() - solver_start
    after = initial.copy()
    coordinates = np.asarray([[v.point().x(), v.point().y(), v.point().z()] for v in handles])
    outside_ids = np.setdiff1d(np.arange(len(initial)), variable)
    after[ids] = coordinates
    if not np.array_equal(after[outside_ids], initial[outside_ids]):
        raise ValueError("Official fairing changed vertices outside supplied range")
    if poly.size_of_vertices() != len(ids) or poly.size_of_facets() != len(faces):
        raise ValueError("Official fairing changed topology")
    if not np.isfinite(after).all():
        raise ValueError("Official fairing returned nonfinite coordinates")
    key = f"cgal_{record['dataset']}_{record['index']:03d}"
    destination = OUT / f"{key}.npz"
    np.savez_compressed(destination, vertices=after, patch_faces=patch)
    info = {"method": "cgal_movable_fair", "official_solver_return": bool(solver_return) if solver_return is not None else None,
            "solver_return_type": type(solver_return).__name__, "stop_reason": "direct_linear_solve",
            "fairing_continuity": 1, "variable_vertices_n": len(variable), "outside_support_bitwise_unchanged": True,
            "topology_unchanged": True, "native_solver_seconds": solver_seconds, "optimizer_seconds": time.perf_counter() - start}
    save(OUT / f"{key}_generation.json", info)


def cgal_worker(position):
    record = load(FORMAL)["rows"][position]
    key = f"cgal_{record['dataset']}_{record['index']:03d}"
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    native = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--cgal-generate", str(position)],
                            cwd=ROOT, capture_output=True, text=True, timeout=120, env=env,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if native.returncode:
        raise RuntimeError(f"CGAL fair generation failed: {native.returncode} {native.stderr[-500:]}")
    destination = OUT / f"{key}.npz"
    verification_path = OUT / f"{key}_verification.json"
    start = time.perf_counter()
    verification_process = subprocess.run([sys.executable, str(ROOT / "scripts/cgal_changed_surface_audit.py"),
                             "--input", record["input_path"], "--patch", str(destination), "--output", str(verification_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=100,
                            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if verification_process.returncode or not verification_path.exists():
        raise RuntimeError(f"CGAL fair verification failed: {verification_process.returncode} {verification_process.stderr[-500:]}")
    verify_seconds = time.perf_counter() - start
    from scripts.evaluate_collar_controls import external_measure
    before, retained, loop, initial, patch = arrays(record)
    with np.load(destination, allow_pickle=False) as z:
        after = z["vertices"].astype(float)
    item = external_measure(after, before, retained, patch, loop, load(verification_path), destination)
    info = load(OUT / f"{key}_generation.json")
    item.update(optimizer=info, optimizer_seconds=info["optimizer_seconds"], verification_seconds=verify_seconds)
    row = {k: record[k] for k in ("dataset", "index", "sample_id", "source_model_id", "category", "input_path")}
    row["methods"] = {"cgal_movable_fair": item}
    save(OUT / f"{key}.json", row)


def run_cgal():
    OUT.mkdir(parents=True, exist_ok=True)
    p = cgal_protocol()
    dest_protocol = OUT / "cgal_fair_protocol.json"
    if dest_protocol.exists() and load(dest_protocol) != p:
        raise ValueError("CGAL fair protocol differs")
    save(dest_protocol, p)
    queue = load(FORMAL)["rows"]
    rows = []
    start = time.perf_counter()
    for position, record in enumerate(queue):
        key = f"cgal_{record['dataset']}_{record['index']:03d}"
        destination = OUT / f"{key}.json"
        if destination.exists():
            row = load(destination)
        else:
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--cgal-worker", str(position)],
                                      cwd=ROOT, capture_output=True, text=True, timeout=250,
                                      env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"),
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if proc.returncode or not destination.exists():
                    raise RuntimeError(f"worker={proc.returncode}: {proc.stderr[-1000:]}")
                row = load(destination)
            except Exception as exc:
                row = {k: record[k] for k in ("dataset", "index", "sample_id", "source_model_id", "category", "input_path")}
                row["methods"] = {"cgal_movable_fair": {"status": "candidate_failure", "valid": False, "error": str(exc)}}
                save(destination, row)
        rows.append(row)
        result = {"complete": len(rows) == len(queue), "protocol": p, "elapsed_seconds": time.perf_counter() - start, "rows": rows}
        if result["complete"]:
            result["summary"] = summarize(rows, "cgal_movable_fair")
        save(OUT / "cgal_fair_results.json", result)
        item = row["methods"]["cgal_movable_fair"]
        print(json.dumps({"completed": len(rows), "planned": len(queue), "case": key, "status": item["status"], "valid": item.get("valid")}), flush=True)


def cotangent_bending_system(initial, retained, patch, loop, retained_n):
    """Frozen FEM stiffness and lumped-area bending form K.T M^-1 K."""
    h = float(np.median(np.linalg.norm(initial[loop] - initial[np.roll(loop, -1)], axis=1)))
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Invalid boundary edge scale")
    origin = initial[loop].mean(axis=0)
    points = (initial - origin) / h
    faces = np.vstack([retained, patch])
    variable = np.unique(np.r_[collar_vertices(retained, loop), np.unique(patch)[np.unique(patch) >= retained_n]])
    tri = points[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    relevant_rows = np.unique(faces[np.isin(faces, variable).any(axis=1)])
    relevant_faces = np.isin(faces, relevant_rows).any(axis=1)
    if np.any(area2[relevant_faces] <= 1e-14):
        raise ValueError("Degenerate triangle in bending support; no mesh cleaning")
    mass = np.zeros(len(points))
    for k in range(3):
        np.add.at(mass, faces[:, k], area2 / 6.)
    rr, cc, ww = [], [], []
    for k in range(3):
        u, v = tri[:, (k + 1) % 3] - tri[:, k], tri[:, (k + 2) % 3] - tri[:, k]
        weight = .5 * np.sum(u * v, axis=1) / np.maximum(area2, 1e-14)
        weight[area2 <= 1e-14] = 0.  # Irrelevant disconnected degenerate faces only.
        i, j = faces[:, (k + 1) % 3], faces[:, (k + 2) % 3]
        rr.extend([i, j, i, j]); cc.extend([i, j, j, i]); ww.extend([weight, weight, -weight, -weight])
    K = sparse.coo_matrix((np.concatenate(ww), (np.concatenate(rr), np.concatenate(cc))), shape=(len(points), len(points))).tocsr()
    rows = np.unique(K[:, variable].nonzero()[0])
    if np.any(mass[rows] <= 0):
        raise ValueError("Nonpositive lumped mass in bending support")
    B = K[rows][:, variable].tocsc()
    constant = K[rows] @ points
    inverse_mass = 1. / mass[rows]
    weighted_B = sparse.diags(inverse_mass) @ B
    Q = (B.T @ weighted_B).tocsc()
    Q = ((Q + Q.T) * .5).tocsc()
    g = B.T @ (inverse_mass[:, None] * constant)
    return points, variable, B, constant, inverse_mass, Q, g, h, origin


def solve_bounded_fair(initial, retained, patch, loop, retained_n, check_unconstrained=False):
    import clarabel
    started = time.perf_counter()
    points, variable, B, constant, inverse_mass, Q, g, h, origin = cotangent_bending_system(initial, retained, patch, loop, retained_n)
    objective_scale = float(max(np.max(np.abs(Q.diagonal())), 1e-30))
    Pfull = 2. * sparse.kron(Q / objective_scale, sparse.eye(3), format="csc")
    P = sparse.triu(Pfull, format="csc")
    q = 2. * np.asarray(g).ravel() / objective_scale
    retained_positions = np.flatnonzero(variable < retained_n)
    conerows = np.concatenate([np.arange(4 * i + 1, 4 * i + 4) for i in range(len(retained_positions))])
    conecols = np.concatenate([np.arange(3 * j, 3 * j + 3) for j in retained_positions])
    A = sparse.csc_matrix((-np.ones(len(conerows)), (conerows, conecols)), shape=(4 * len(retained_positions), len(q)))
    b = np.zeros(A.shape[0]); b[::4] = .02
    cones = [clarabel.SecondOrderConeT(4) for _ in retained_positions]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    settings.time_limit = 60.
    settings.max_threads = 2
    settings.tol_gap_abs = 1e-8
    settings.tol_gap_rel = 1e-8
    settings.tol_feas = 1e-8
    solver = clarabel.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()
    delta = np.asarray(solution.x).reshape(-1, 3)
    objective_initial = float(np.sum(inverse_mass[:, None] * constant ** 2))
    objective_final = float(np.sum(inverse_mass[:, None] * (constant + B @ delta) ** 2))
    retained_norms = np.linalg.norm(delta[retained_positions], axis=1)
    max_displacement = float(retained_norms.max(initial=0.))
    primal_residual = float(np.linalg.norm(A @ delta.ravel() + np.asarray(solution.s) - b))
    dual_residual = float(np.linalg.norm(Pfull @ delta.ravel() + q + A.T @ np.asarray(solution.z)))
    gap = float(abs(solution.obj_val - solution.obj_val_dual))
    relative_gap = gap / max(1., abs(solution.obj_val), abs(solution.obj_val_dual))
    accepted = (str(solution.status) == "Solved" and max_displacement <= .02 + 1e-8
                and objective_final <= objective_initial + 1e-8 * max(1., objective_initial)
                and float(solution.r_prim) <= 1e-8 and float(solution.r_dual) <= 1e-8 and relative_gap <= 1e-8)
    info = {"method": "bounded_cotangent_biharmonic", "solver": "Clarabel", "solver_version": clarabel.__version__,
            "solver_status": str(solution.status), "solution_accepted": bool(accepted), "stop_reason": str(solution.status),
            "iterations": int(solution.iterations), "solver_seconds": float(solution.solve_time),
            "objective_initial": objective_initial, "objective_final": objective_final, "objective_ratio": objective_final / max(objective_initial, 1e-30),
            "objective_scale": objective_scale, "primal_residual_reported": float(solution.r_prim), "dual_residual_reported": float(solution.r_dual),
            "primal_residual_absolute": primal_residual, "dual_residual_absolute": dual_residual,
            "relative_duality_gap": relative_gap, "absolute_duality_gap_scaled_objective": gap,
            "max_retained_displacement_over_boundary_edge": max_displacement, "retained_bound": .02,
            "variable_vertices_n": len(variable), "bounded_retained_vertices_n": len(retained_positions),
            "unbounded_new_patch_vertices_n": int(np.sum(variable >= retained_n)), "clipping_or_projection": False}
    if check_unconstrained:
        direct = spsolve(Q, -g)
        unbounded_solver = clarabel.DefaultSolver(P, q, sparse.csc_matrix((0, len(q))), np.empty(0), [], settings)
        unconstrained = unbounded_solver.solve()
        free_delta = np.asarray(unconstrained.x).reshape(-1, 3)
        equality_error = float(np.linalg.norm(direct - free_delta) / max(1., np.linalg.norm(direct)))
        residual = float(np.linalg.norm(Q @ direct + g) / max(1., np.linalg.norm(g)))
        info["unconstrained_check"] = {"clarabel_status": str(unconstrained.status), "relative_solution_error_against_sparse_linear_solve": equality_error,
                                       "linear_stationarity_relative_residual": residual,
                                       "objective_unconstrained": float(np.sum(inverse_mass[:, None] * (constant + B @ direct) ** 2))}
        if str(unconstrained.status) != "Solved" or equality_error > 1e-5 or residual > 1e-8:
            raise ValueError(f"Unconstrained operator check failed: {info['unconstrained_check']}")
    after = initial.copy()
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite conic solution")
    after[variable] = (points[variable] + delta) * h + origin
    info["outside_support_bitwise_unchanged"] = bool(np.array_equal(after[np.setdiff1d(np.arange(len(initial)), variable)], initial[np.setdiff1d(np.arange(len(initial)), variable)]))
    info["optimizer_seconds"] = time.perf_counter() - started
    return after, info


def bounded_protocol():
    return {"method": "Displacement-constrained cotangent biharmonic fairing", "method_key": "bounded_biharmonic_fair",
            "objective": "min tr(V^T K^T M^-1 K V), with initial cotangent FEM stiffness K and barycentric lumped area M frozen",
            "stiffness": "Each face opposite-angle cotangent contributes 0.5*cot(theta) to symmetric edge weight; negative cotangents are not clamped; K_ii=sum_j w_ij, K_ij=-w_ij",
            "mass": "M_ii=sum_{incident faces} area(face)/3, positive barycentric lumped mass; fixed from the CGAL initializer",
            "algorithm_reference": "Botsch and Sorkine, On Linear Variational Surface Deformation Methods, IEEE TVCG 14(1):213-230, 2008",
            "implementation_relation": "Independent standard FEM bending discretization. CGAL 6.0.1 fair was inspected: it uses secure clamped cotangents and Voronoi areas in a recursive Laplacian. This constrained comparator does not claim identical CGAL discretization or author code.",
            "source_inspected": ["https://raw.githubusercontent.com/CGAL/cgal/v6.0.1/Polygon_mesh_processing/include/CGAL/Polygon_mesh_processing/fair.h", "https://raw.githubusercontent.com/CGAL/cgal/v6.0.1/Polygon_mesh_processing/include/CGAL/Polygon_mesh_processing/internal/fair_impl.h", "https://raw.githubusercontent.com/CGAL/cgal/v6.0.1/Weights/include/CGAL/Weights/cotangent_weights.h"],
            "initialization": "Same saved CGAL refine-and-fair mesh, same target and topology as fixed/movable collar controls",
            "free_vertices": "All new patch vertices plus retained boundary two-ring; all exterior vertices exactly fixed",
            "constraints": "Each retained free vertex has Euclidean displacement <=0.02 median boundary-edge lengths; each new patch vertex is unbounded",
            "solver": {"name": "Clarabel", "version": "0.11.1", "type": "Convex quadratic objective and one second-order cone per retained free vertex",
                       "max_iterations": 200, "time_limit_seconds": 60, "max_threads": 2, "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8,
                       "acceptance": "Solved status, constraint excess<=1e-8, objective nonincrease tolerance1e-8, normalized primal/dual residual and relative gap<=1e-8; no clipping/projection"},
            "numerics": "Coordinates centered at boundary centroid and normalized by median boundary edge. Positive objective scaling by maximum diagonal of Q does not change the minimizer. Only rows of K affected by variable coordinates contribute nonconstant terms.",
            "degeneracy": "A triangle with normalized double-area<=1e-14 within operator support produces a retained failure, without mesh cleaning",
            "development": "Exactly first three saved development sources; one constrained solve each and one unconstrained Clarabel versus sparse linear solution check; no parameter search",
            "formal_queue": str(FORMAL), "formal_queue_sha256": digest(FORMAL), "planned_n": 110,
            "statistics": "Separate 50/60 cohorts; 20000 paired source bootstrap replicates seed20261025; all sources in validity denominator; geometry on common-valid outputs",
            "verification": "Existing isolated native changed-surface CGAL check, same metrics as earlier direct collar comparison",
            "workers": 1, "math_threads": 2}


def bounded_develop():
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "bounded_fair_protocol.json", bounded_protocol())
    queue = load(DEVELOPMENT)["records"]
    rows = []
    for position in [0, 1, 2]:
        before, retained, loop, initial, patch = arrays(queue[position])
        _, info = solve_bounded_fair(initial, retained, patch, loop, len(before), check_unconstrained=True)
        if not info["solution_accepted"]:
            raise ValueError(f"Development numerical verification failed: {info}")
        rows.append({"position": position, "source_model_id": queue[position]["source_model_id"], "optimizer": info})
        print(json.dumps(rows[-1]), flush=True)
    save(OUT / "bounded_fair_development.json", {"complete": True, "passed": True, "rows": rows})


def bounded_generate(position):
    record = load(FORMAL)["rows"][position]
    before, retained, loop, initial, patch = arrays(record)
    after, info = solve_bounded_fair(initial, retained, patch, loop, len(before))
    key = f"bounded_{record['dataset']}_{record['index']:03d}"
    save(OUT / f"{key}_generation.json", info)
    if not info["solution_accepted"]:
        raise RuntimeError("Convex solver did not meet frozen numerical acceptance conditions")
    np.savez_compressed(OUT / f"{key}.npz", vertices=after, patch_faces=patch)


def bounded_worker(position):
    record = load(FORMAL)["rows"][position]
    key = f"bounded_{record['dataset']}_{record['index']:03d}"
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    proc = subprocess.run([str(CONVEX_PYTHON), str(Path(__file__).resolve()), "--bounded-generate", str(position)],
                          cwd=ROOT, capture_output=True, text=True, timeout=120, env=env,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if proc.returncode:
        raise RuntimeError(f"Constrained fairing failed: {proc.returncode} {proc.stderr[-800:]}")
    destination = OUT / f"{key}.npz"
    verification_path = OUT / f"{key}_verification.json"
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/cgal_changed_surface_audit.py"),
                           "--input", record["input_path"], "--patch", str(destination), "--output", str(verification_path)],
                          cwd=ROOT, capture_output=True, text=True, timeout=100,
                          env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if proc.returncode or not verification_path.exists():
        raise RuntimeError(f"Constrained fairing native check failed: {proc.returncode} {proc.stderr[-500:]}")
    verify_seconds = time.perf_counter() - start
    from scripts.evaluate_collar_controls import external_measure
    before, retained, loop, initial, patch = arrays(record)
    with np.load(destination, allow_pickle=False) as z:
        after = z["vertices"].astype(float)
    item = external_measure(after, before, retained, patch, loop, load(verification_path), destination)
    info = load(OUT / f"{key}_generation.json")
    item.update(optimizer=info, optimizer_seconds=info["optimizer_seconds"], verification_seconds=verify_seconds)
    row = {k: record[k] for k in ("dataset", "index", "sample_id", "source_model_id", "category", "input_path")}
    row["methods"] = {"bounded_biharmonic_fair": item}
    save(OUT / f"{key}.json", row)


def run_bounded():
    if not (OUT / "bounded_fair_development.json").exists() or not load(OUT / "bounded_fair_development.json").get("passed"):
        raise ValueError("Three-source numerical development checks must pass before formal evaluation")
    if load(OUT / "bounded_fair_protocol.json") != bounded_protocol():
        raise ValueError("Bounded-fairing frozen protocol differs")
    queue = load(FORMAL)["rows"]
    rows = []
    start = time.perf_counter()
    for position, record in enumerate(queue):
        key = f"bounded_{record['dataset']}_{record['index']:03d}"
        destination = OUT / f"{key}.json"
        if destination.exists():
            row = load(destination)
        else:
            try:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--bounded-worker", str(position)],
                                      cwd=ROOT, capture_output=True, text=True, timeout=250,
                                      env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"),
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if proc.returncode or not destination.exists():
                    raise RuntimeError(f"worker={proc.returncode}: {proc.stderr[-1000:]}")
                row = load(destination)
            except Exception as exc:
                row = {k: record[k] for k in ("dataset", "index", "sample_id", "source_model_id", "category", "input_path")}
                item = {"status": "candidate_failure", "valid": False, "error": str(exc)}
                gen = OUT / f"{key}_generation.json"
                if gen.exists(): item["optimizer"] = load(gen)
                row["methods"] = {"bounded_biharmonic_fair": item}
                save(destination, row)
        rows.append(row)
        result = {"complete": len(rows) == len(queue), "protocol": bounded_protocol(), "elapsed_seconds": time.perf_counter() - start, "rows": rows}
        if result["complete"]:
            result["summary"] = summarize(rows, "bounded_biharmonic_fair")
        save(OUT / "bounded_fair_results.json", result)
        item = row["methods"]["bounded_biharmonic_fair"]
        print(json.dumps({"completed": len(rows), "planned": len(queue), "case": key, "status": item["status"], "valid": item.get("valid")}), flush=True)


def two_step_protocol():
    return {"method": "Belyaev--Ohtake normal-domain local smoothing using official MeshLab/VCG",
            "method_key": "two_step_smoothing", "implementation": "PyMeshLab 2021.10 twostep_smooth official binary",
            "reference": "Belyaev and Ohtake, A Comparison of Mesh Smoothing Methods, 2003, pp.83-87",
            "official_source": "https://github.com/cnr-isti-vclab/meshlab/blob/Meshlab-2021.10/src/meshlabplugins/filter_unsharp/filter_unsharp.cpp",
            "official_documentation": "https://pymeshlab.readthedocs.io/en/latest/filter_list.html#apply-coord-two-steps-smoothing",
            "relation": "Established normal-filtering and vertex-fitting refinement of an already initialized mesh. Not Calatroni2023 or a hole-construction algorithm.",
            "initialization": "Same CGAL refine-fair initializer, prescribed target and connectivity as all existing direct collar controls",
            "requested_free_vertices": "All new patch vertices and retained target boundary two-ring; this is the seed support, not a strict actual-support constraint",
            "selection": "Select every face incident to a requested free vertex using official conditional_face_selection; official twostep_smooth converts faces to vertices with VertexFromFaceStrict. Accept this deterministic native selection closure, recording all additional vertex IDs and counts.",
            "comparison_identity": "Official normal-domain local method control; actual support may exceed the joint collar, and no common displacement bound is imposed. The existing bounded-fairing comparator remains the strict same-support control.",
            "preformal_protocol_adjustment": "Before any formal scoring, strict requested-ROI equality failed on development position1 with five additional vertices. The API-native selection closure replaces that incompatible requirement; the original diagnostic is retained in the development record. No quality outcome or parameter tuning determines the adjustment.",
            "normal_context": "All referenced faces of the initialized mesh supply the official normal filter; only selected vertices move during official FastFitMesh",
            "fixed_exterior": "Unselected coordinates must remain bitwise identical; no post-hoc coordinate restoration or clipping",
            "unreferenced_vertices": "Only vertices referenced by input triangles enter MeshLab, with explicit original index map; unused original coordinates stay outside the native mesh",
            "parameters": {"stepsmoothnum": 3, "normalthr": 60., "stepnormalnum": 20, "stepfitnum": 20, "selected": True},
            "displacement_bound": None, "parameter_search": False,
            "development": {"queue": str(DEVELOPMENT), "positions": [0, 1, 2], "purpose": "Finite coordinates, unchanged topology, exact ROI and fixed exterior; no performance tuning"},
            "formal_queue": str(FORMAL), "formal_queue_sha256": digest(FORMAL), "planned_n": 110,
            "verification": "Same isolated CGAL changed-surface verification and existing unified geometry evaluator",
            "statistics": "20000 paired source bootstrap draws seed20261025, separate 50/60 cohorts, all sources in validity denominators",
            "limits": {"generation_seconds": 120, "verification_seconds": 100, "workers": 1, "math_threads": 2},
            "timing_scope": "Descriptive filter generation and verification timing, excluding shared input loading; no matched end-to-end speed claim"}


def solve_two_step(initial, retained, patch, loop, retained_n):
    import importlib.metadata
    import pymeshlab
    started = time.perf_counter()
    if importlib.metadata.version("pymeshlab") != "2021.10":
        raise ValueError("Frozen PyMeshLab 2021.10 version required")
    faces = np.vstack([retained, patch])
    ids = np.unique(faces)
    inverse = np.full(len(initial), -1, dtype=np.int64)
    inverse[ids] = np.arange(len(ids))
    local_faces = inverse[faces].astype(np.int32)
    variable = np.unique(np.r_[collar_vertices(retained, loop), np.unique(patch)[np.unique(patch) >= retained_n]])
    selected_faces = np.isin(faces, variable).any(axis=1)
    incident = np.bincount(local_faces.ravel(), minlength=len(ids))
    selected_incident = np.bincount(local_faces[selected_faces].ravel(), minlength=len(ids))
    official_variable = ids[(incident > 0) & (incident == selected_incident)]
    requested_variable = variable
    extra_variable = np.setdiff1d(official_variable, requested_variable)
    if len(np.setdiff1d(requested_variable, official_variable)):
        raise ValueError("Official face-selection closure omits a requested vertex")
    variable = official_variable
    mesh = pymeshlab.Mesh(vertex_matrix=initial[ids].copy(), face_matrix=local_faces,
                         f_quality_array=selected_faces.astype(float))
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(mesh)
    if not np.array_equal(meshset.current_mesh().vertex_matrix(), initial[ids]):
        raise ValueError("Native import changes initial coordinate precision")
    meshset.apply_filter("conditional_face_selection", condselect="fq > 0.5")
    if not np.array_equal(meshset.current_mesh().face_selection_array(), selected_faces):
        raise ValueError("Official conditional face selection did not preserve supplied ROI")
    filter_started = time.perf_counter()
    meshset.apply_filter("twostep_smooth", **two_step_protocol()["parameters"])
    filter_seconds = time.perf_counter() - filter_started
    output = meshset.current_mesh()
    if not np.array_equal(output.face_matrix(), local_faces) or len(output.vertex_matrix()) != len(ids):
        raise ValueError("Official filter changed vertex indices or triangle connectivity")
    if not np.array_equal(ids[np.flatnonzero(output.vertex_selection_array())], variable):
        raise ValueError("Official filter's selected vertices differ from the computed native selection closure")
    coordinates = output.vertex_matrix()
    if not np.isfinite(coordinates).all():
        raise ValueError("Official filter returned nonfinite coordinates")
    outside = np.setdiff1d(np.arange(len(initial)), variable)
    local_outside = ~np.isin(ids, variable)
    if not np.array_equal(coordinates[local_outside], initial[ids[local_outside]]):
        raise ValueError("Official filter moved an unselected exterior vertex")
    after = initial.copy()
    after[ids] = coordinates  # Complete native output, without restoring any processed vertex.
    info = {"method": "two_step_smoothing", "stop_reason": "official_default_fixed_iterations",
            "pymeshlab_version": "2021.10", "parameters": two_step_protocol()["parameters"],
            "requested_variable_vertices_n": len(requested_variable), "variable_vertices_n": len(variable),
            "extra_variable_vertices_n": len(extra_variable), "extra_variable_vertex_ids": extra_variable.tolist(),
            "native_vertices_n": len(ids), "selected_faces_n": int(selected_faces.sum()),
            "native_roi_equals_requested": bool(np.array_equal(variable, requested_variable)), "topology_unchanged": True,
            "outside_support_bitwise_unchanged": bool(np.array_equal(after[outside], initial[outside])),
            "native_filter_seconds": filter_seconds, "optimizer_seconds": time.perf_counter() - started,
            "retained_displacement_bound": None, "posthoc_coordinate_restoration": False}
    return after, info


def two_step_develop():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "two_step_protocol.json"
    p = two_step_protocol()
    development_path = OUT / "two_step_development.json"
    prior_development = load(development_path) if development_path.exists() else None
    original_diagnostic = None
    if prior_development:
        original_diagnostic = prior_development.get("preformal_strict_roi_diagnostic")
        if original_diagnostic is None and prior_development.get("passed") is False:
            original_diagnostic = prior_development
    if protocol_path.exists() and load(protocol_path) != p and (OUT / "two_step_results.json").exists():
        raise ValueError("Cannot change two-step protocol after formal scoring")
    save(protocol_path, p)
    rows = []
    queue = load(DEVELOPMENT)["records"]
    for position in p["development"]["positions"]:
        before, retained, loop, initial, patch = arrays(queue[position])
        try:
            _, info = solve_two_step(initial, retained, patch, loop, len(before))
        except Exception as exc:
            rows.append({"position": position, "source_model_id": queue[position]["source_model_id"],
                         "passed": False, "error": str(exc)})
            save(OUT / "two_step_development.json", {"complete": False, "passed": False,
                 "stopped_on_failure": True, "rows": rows,
                 "preformal_strict_roi_diagnostic": original_diagnostic})
            raise
        rows.append({"position": position, "source_model_id": queue[position]["source_model_id"], "optimizer": info})
        print(json.dumps(rows[-1]), flush=True)
    save(OUT / "two_step_development.json", {"complete": True, "passed": True, "rows": rows,
         "preformal_strict_roi_diagnostic": original_diagnostic})


def two_step_generate(position):
    record = load(FORMAL)["rows"][position]
    before, retained, loop, initial, patch = arrays(record)
    after, info = solve_two_step(initial, retained, patch, loop, len(before))
    destination = OUT / f"two_step_{record['dataset']}_{record['index']:03d}.npz"
    np.savez_compressed(destination, vertices=after, patch_faces=patch)
    print("RESULT=" + json.dumps(info), flush=True)


def two_step_fallback(rows, slot="two_step_smoothing", policy="two_step"):
    from scripts.evaluate_collar_controls import paired_statistics, external_value, source_mean_interval
    source = load(ROOT / "experiments/formal_collar_mechanism/fallback_results.json")
    originals = {(r["dataset"], r["index"], r["source_model_id"]): r for r in source["rows"]}
    shared = ("cgal_refine_fair", "cgal_polyline", "center_fan")
    policies_rows = []
    for row in rows:
        original = originals[(row["dataset"], row["index"], row["source_model_id"])]
        candidates = {k: v for k, v in original["candidates"].items() if k in (*shared, "joint")}
        candidates[slot] = row["methods"][slot]
        policies = {}
        for name, candidate_slot in (("joint_collar", "joint"), (policy, slot)):
            allowed = (*shared, candidate_slot)
            eligible = [candidates[k] for k in allowed if candidates[k].get("safe_eligible") is True]
            chosen = min(eligible, key=lambda x: (x["selection_seam_unoriented"], allowed.index(x["method"]))) if eligible else None
            policies[name] = ({**chosen, "accepted": True, "used_shared_fallback": chosen["method"] in shared} if chosen else
                              {"accepted": False, "valid": False, "method": None, "used_shared_fallback": False})
        prior = original["policies"]["movable_collar"]
        if any(policies["joint_collar"].get(k) != prior.get(k) for k in ("method", "accepted", "valid", "selection_seam_unoriented")):
            raise ValueError("Frozen joint-collar fallback decision did not reproduce")
        policies_rows.append({**{k: row[k] for k in ("dataset", "index", "source_model_id", "sample_id")}, "policies": policies})
    metrics = ("seam_normal_absolute_mean", "seam_normal_absolute_p95", "triangle_quality.mean", "triangle_quality.p05",
               "retained_displacement_rms_over_edge", "retained_displacement_max_over_edge", "changed_surface_area_ratio")
    summary = {}
    for cohort in ("holdout50", "cross_category60", "pooled110"):
        selected = [r for r in policies_rows if cohort == "pooled110" or r["dataset"] == cohort]
        common = [r for r in selected if all(r["policies"][p].get("valid") is True for p in ("joint_collar", policy))]
        summary[cohort] = {"n": len(selected), "policies": {p: {
            "valid_n": sum(r["policies"][p].get("valid") is True for r in selected),
            "abstain_n": sum(not r["policies"][p]["accepted"] for r in selected),
            "selected_candidates": {m: sum(r["policies"][p].get("method") == m for r in selected) for m in (*shared, "joint", slot)},
            "valid_rate": source_mean_interval([float(r["policies"][p].get("valid") is True) for r in selected])}
            for p in ("joint_collar", policy)}, "common_valid_n": len(common),
            "joint_minus_"+policy: {m: paired_statistics([external_value(r["policies"]["joint_collar"], m) for r in common],
                                                       [external_value(r["policies"][policy], m) for r in common], seed=20261025) for m in metrics},
            "valid_rate_difference_all_sources": paired_statistics([float(r["policies"]["joint_collar"].get("valid") is True) for r in selected],
                                                                    [float(r["policies"][policy].get("valid") is True) for r in selected], seed=20261025)}
    return {"prescribed_target": True, "end_to_end_target_selection": False,
            "shared_candidates": list(shared), "fourth_slots": {"joint_collar": "joint", policy: slot},
            "selection": "Same successful native closure/no-new-intersection and finite minimum unoriented seam score; shared candidates win exact ties in input order",
            "new_generation_in_replay": 0, "new_verification_in_replay": 0, "summary": summary, "rows": policies_rows}


def run_two_step():
    from scripts.evaluate_collar_controls import score_saved_candidate
    if not load(OUT / "two_step_development.json").get("passed") or load(OUT / "two_step_protocol.json") != two_step_protocol():
        raise ValueError("Frozen three-source two-step development must pass")
    queue = load(FORMAL)["rows"]
    destination_results = OUT / "two_step_results.json"
    rows = load(destination_results)["rows"] if destination_results.exists() else []
    started = time.perf_counter()
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for position in range(len(rows), len(queue)):
        record = queue[position]
        key = f"two_step_{record['dataset']}_{record['index']:03d}"
        destination = OUT / f"{key}.npz"
        row = {k: record[k] for k in ("dataset", "index", "sample_id", "source_model_id", "category", "input_path")}
        try:
            proc = subprocess.run([str(MESHLAB_PYTHON), str(Path(__file__).resolve()), "--two-step-generate", str(position)],
                                  cwd=ROOT, capture_output=True, text=True, timeout=120, env=env,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if proc.returncode:
                raise RuntimeError(f"Official two-step generation failed: {proc.returncode}: {proc.stderr[-1000:]}")
            info = json.loads(next(line[7:] for line in proc.stdout.splitlines() if line.startswith("RESULT=")))
            audit_path = OUT / "two_step_verification.tmp.json"
            if audit_path.exists():
                audit_path.unlink()
            verify_started = time.perf_counter()
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/cgal_changed_surface_audit.py"), "--input", record["input_path"],
                                  "--patch", str(destination), "--output", str(audit_path)], cwd=ROOT, capture_output=True,
                                  text=True, timeout=100, env=dict(env, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if proc.returncode or not audit_path.exists():
                raise RuntimeError(f"Native verifier failed: {proc.returncode}: {proc.stderr[-1000:]}")
            audit = load(audit_path)
            audit_path.unlink()
            item = score_saved_candidate(Path(record["input_path"]), destination, audit, "two_step_smoothing")
            item.update(optimizer=info, optimizer_seconds=info["optimizer_seconds"], verification_seconds=time.perf_counter() - verify_started)
        except Exception as exc:
            item = {"method": "two_step_smoothing", "status": "candidate_failure", "valid": False, "safe_eligible": False, "error": str(exc)}
        row["methods"] = {"two_step_smoothing": item}
        rows.append(row)
        result = {"complete": len(rows) == len(queue), "protocol": two_step_protocol(), "elapsed_seconds": time.perf_counter() - started, "rows": rows}
        if result["complete"]:
            result["summary"] = summarize(rows, "two_step_smoothing")
            result["fallback"] = two_step_fallback(rows)
        save(destination_results, result)
        print(json.dumps({"completed": len(rows), "planned": len(queue), "case": key, "status": item["status"], "valid": item.get("valid"),
                          "elapsed_seconds": result["elapsed_seconds"], "error": item.get("error")}), flush=True)


def run():
    if not (OUT / "development.json").exists():
        develop()
    queue = load(FORMAL)["rows"]
    rows = []
    start = time.perf_counter()
    for position, record in enumerate(queue):
        key = f"{record['dataset']}_{record['index']:03d}"
        dest = OUT / f"{key}.json"
        if dest.exists():
            row = load(dest)
        else:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(position)],
                                     cwd=ROOT, capture_output=True, text=True, timeout=250,
                                     env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"),
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if process.returncode or not dest.exists():
                row = {k: record[k] for k in ("dataset", "index", "sample_id", "source_model_id", "category", "input_path")}
                row["methods"] = {"calatroni_smooth": {"status": "candidate_failure", "valid": False, "error": process.stderr[-1000:], "returncode": process.returncode}}
                save(dest, row)
            else:
                row = load(dest)
        rows.append(row)
        result = {"complete": len(rows) == len(queue), "protocol": load(OUT / "protocol.json"), "selected_beta": load(OUT / "development.json")["selected_beta"],
                  "elapsed_seconds": time.perf_counter() - start, "rows": rows}
        if result["complete"]:
            result["summary"] = summarize(rows)
        save(OUT / "results.json", result)
        item = row["methods"]["calatroni_smooth"]
        print(json.dumps({"completed": len(rows), "planned": len(queue), "case": key, "status": item["status"], "valid": item.get("valid")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attachment-develop", action="store_true")
    parser.add_argument("--attachment-adequacy", action="store_true")
    parser.add_argument("--attachment-generate", type=int)
    parser.add_argument("--attachment", action="store_true")
    parser.add_argument("--develop", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--cgal", action="store_true")
    parser.add_argument("--calatroni-diagnostic", action="store_true")
    parser.add_argument("--cgal-generate", type=int)
    parser.add_argument("--cgal-worker", type=int)
    parser.add_argument("--bounded", action="store_true")
    parser.add_argument("--bounded-develop", action="store_true")
    parser.add_argument("--bounded-generate", type=int)
    parser.add_argument("--bounded-worker", type=int)
    parser.add_argument("--two-step", action="store_true")
    parser.add_argument("--two-step-develop", action="store_true")
    parser.add_argument("--two-step-generate", type=int)
    args = parser.parse_args()
    if args.attachment_adequacy:
        attachment_adequacy()
    elif args.attachment_develop:
        attachment_develop()
    elif args.attachment_generate is not None:
        attachment_generate(args.attachment_generate)
    elif args.attachment:
        run_attachment()
    elif args.two_step_develop:
        two_step_develop()
    elif args.two_step_generate is not None:
        two_step_generate(args.two_step_generate)
    elif args.two_step:
        run_two_step()
    elif args.bounded_develop:
        bounded_develop()
    elif args.bounded_generate is not None:
        bounded_generate(args.bounded_generate)
    elif args.bounded_worker is not None:
        bounded_worker(args.bounded_worker)
    elif args.bounded:
        run_bounded()
    elif args.cgal_generate is not None:
        cgal_generate(args.cgal_generate)
    elif args.cgal_worker is not None:
        cgal_worker(args.cgal_worker)
    elif args.cgal:
        run_cgal()
    elif args.worker is not None:
        worker(args.worker)
    elif args.develop:
        develop()
    elif args.summarize:
        d = load(OUT / "results.json")
        d["summary"] = summarize(d["rows"])
        save(OUT / "results.json", d)
    elif args.calatroni_diagnostic:
        run()
    else:
        run_cgal()
