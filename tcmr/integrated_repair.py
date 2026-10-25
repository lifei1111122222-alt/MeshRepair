"""One integrated, bounded target-locked repair method and fixed ablations.

Inference reads damaged geometry and frozen prior probabilities only. Each of
the two prior candidate loops receives the same four repair methods. Geometry
cost chooses identity without collision results; safety only accepts or rejects
repairs of that locked identity. This module also provides live prior inference.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL = ("cgal_refine_fair", "cgal_polyline", "center_fan", "bounded_joint10")
LEGACY_WEIGHTS = {"fair": 1.0, "lap": 0.1, "fid": 1000.0, "normal": 1.0, "curv": 0.1, "quality": 1.0}
POLICIES = ("prior_locked_cgal_safe", "prior_locked_pool_without_joint",
            "prior_locked_pool_with_joint", "cost_target_locked_pool_with_joint")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def protocol_definition():
    return {
        "candidate_pool": list(POOL), "prior": "frozen seed09 top2, stable original candidate order on ties",
        "steps": 10, "joint_optimizer": "existing joint_optimizer float64 projected Adam, default fixed weights, lr0.02, retained two-ring, retained per-vertex budget0.02 median boundary edge",
        "legacy_energy_weights": LEGACY_WEIGHTS,
        "energy_reference": "For each fixed initializer candidate, its own output is its initializer. For bounded_joint10, CGAL refine/fair is the initializer. Consequently energy has initializer-reference dependence; it is not a reference-free objective or the joint optimizer internal objective.",
        "dimensionless_energy": "Evaluate legacy fair/lap/fid/normal/residual_variation/quality terms in coordinates normalized by original median boundary edge h; same six fixed legacy weights. Retained reference normals use initializer coordinates. Legacy evaluator and collar RMS use three retained connectivity rings, while joint movable support uses two rings.",
        "loop_representative": "Minimum finite measured seam candidate, regardless of collision, closure, or safety state; fixed POOL order breaks exact ties",
        "loop_geometry_penalty": "seam/(pi/2) + min(three-ring retained RMS displacement/h,1) + E/(1+E)",
        "identity_score": "prior_logit - loop_geometry_penalty; no safety/validity term. Missing geometry has no score; if every loop lacks finite geometry return null",
        "target_lock": "After choosing identity, pick minimum seam explicitly no-new-intersection AND strictly closed candidate only within that identity; if none clear return null. Safety never substitutes another loop.",
        "controls": list(POLICIES), "timeout_seconds_per_candidate": 120,
        "candidate_workers_max": 2, "unknown_policy": "failure/timeout/unknown is never safe",
        "target_labels_read": False, "parameter_sweep": False,
    }


def content_key(vertices, faces, loop):
    digest = hashlib.sha256()
    for array, dtype in ((vertices, np.float64), (faces, np.int64), (loop, np.int64)):
        value = np.ascontiguousarray(array, dtype=dtype)
        digest.update(np.asarray(value.shape, np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _unit_normals(vertices, faces):
    p = vertices[faces]
    value = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-15)


def legacy_dimensionless_energy(vertices, initial, retained, patch, loop, retained_count):
    """One shared six-term evaluator, independent of the generating method."""
    from .joint_optimizer import retained_collar
    from .optimizer import _seam_pairs, _unique_edges
    h = float(np.median(np.linalg.norm(initial[loop] - initial[np.roll(loop, -1)], axis=1)))
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Invalid boundary scale")
    origin = initial[loop].mean(axis=0)
    x, x0 = (vertices - origin) / h, (initial - origin) / h
    collar = retained_collar(retained, loop, rings=3)
    collar = collar[collar < retained_count]
    pv, edges = np.unique(patch), _unique_edges(patch)
    lap = np.zeros_like(x)
    degree = np.zeros(len(x))
    np.add.at(lap, edges[:, 0], x[edges[:, 1]] - x[edges[:, 0]])
    np.add.at(lap, edges[:, 1], x[edges[:, 0]] - x[edges[:, 1]])
    np.add.at(degree, edges[:, 0], 1)
    np.add.at(degree, edges[:, 1], 1)
    fair = float(np.mean((lap[pv] / np.maximum(degree[pv, None], 1)) ** 2))
    lap_term = float(np.mean(((x[edges[:, 1]] - x[edges[:, 0]]) - (x0[edges[:, 1]] - x0[edges[:, 0]])) ** 2))
    fid = float(np.mean((x[collar] - x0[collar]) ** 2)) if len(collar) else 0.0
    ri, pi = _seam_pairs(retained, patch, loop)
    retained_normals = _unit_normals(x0, retained[ri])
    patch_normals = _unit_normals(x, patch[pi])
    signs = np.where(np.sum(retained_normals * patch_normals, axis=1) >= 0, 1., -1.)
    delta = patch_normals * signs[:, None] - retained_normals
    normal = float(np.mean(delta ** 2)) if len(delta) else 0.0
    curv = float(np.mean((delta[1:] - delta[:-1]) ** 2)) if len(delta) > 1 else 0.0
    tri = x[patch]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    edge2 = sum(np.sum((tri[:, (j + 1) % 3] - tri[:, j]) ** 2, axis=1) for j in range(3))
    q = 2 * np.sqrt(3) * area2 / np.maximum(edge2, 1e-15)
    terms = {"fair": fair, "lap": lap_term, "fid": fid, "normal": normal,
             "curv": curv, "quality": float(np.mean(np.maximum(.20 - q, 0) ** 2))}
    energy = sum(LEGACY_WEIGHTS[k] * v for k, v in terms.items())
    rms = float(np.sqrt(np.mean(np.sum((x[collar] - x0[collar]) ** 2, axis=1)))) if len(collar) else 0.0
    return {"dimensionless_legacy_energy": energy, "dimensionless_legacy_terms": terms,
            "collar_rms_over_boundary_edge": rms, "median_boundary_edge": h,
            "patch_triangle_quality_mean": float(q.mean()), "patch_triangle_quality_p05": float(np.quantile(q, .05)),
            "patch_triangle_quality_min": float(q.min())}


def candidate_worker(spec, output):
    """Generate and measure one actual candidate; parent enforces120s including verification."""
    import torch
    from scripts.cgal_local_hole_adapter import repair
    from scripts.evaluate_geometry_comparison import _seam_normal_error
    from scripts.local_repair_topology import inspect_local_patch
    from .joint_optimizer import optimize_joint_collar, retained_collar
    from .validity import orient_patch_to_boundary
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    output = Path(output)
    with np.load(spec["input_path"], allow_pickle=False) as z:
        before, retained, loop = np.asarray(z["vertices"], float), np.asarray(z["faces"], np.int64), np.asarray(z["selected_loop"], np.int64)
    method = spec["method"]
    info = None
    if method in ("cgal_refine_fair", "cgal_polyline"):
        initial, patch, native = repair(before, retained, loop, method)
        vertices = initial
    elif method == "center_fan":
        initial = np.vstack((before, before[loop].mean(axis=0)))
        patch = np.column_stack((loop, np.roll(loop, -1), np.full(len(loop), len(before)))).astype(np.int64)
        vertices, native = initial, None
    elif method == "bounded_joint10":
        with np.load(spec["initializer_path"], allow_pickle=False) as z:
            initial, patch = np.asarray(z["vertices"], float), np.asarray(z["patch_faces"], np.int64)
        if not np.array_equal(initial[:len(before)], before):
            raise ValueError("Refine initializer moved retained vertices")
        vertices, info = optimize_joint_collar(initial, retained, patch, loop, len(before), mode="joint", steps=10)
        native = None
    else:
        raise ValueError(method)
    patch = orient_patch_to_boundary(patch, retained, loop)
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite geometry")
    metrics = legacy_dimensionless_energy(vertices, initial, retained, patch, loop, len(before))
    seam = _seam_normal_error(vertices, retained, patch, loop)
    if seam is None or not math.isfinite(seam) or not math.isfinite(metrics["dimensionless_legacy_energy"]):
        raise ValueError("Undefined cost features")
    topology = inspect_local_patch(retained, loop, patch)
    patch_path = output.with_suffix(".npz")
    np.savez_compressed(patch_path, vertices=vertices, patch_faces=patch)
    displacement = np.linalg.norm(vertices[:len(before)] - before, axis=1) / metrics["median_boundary_edge"]
    collar = retained_collar(retained, loop, rings=2) if method == "bounded_joint10" else np.empty(0, np.int64)
    outside = np.ones(len(before), dtype=bool)
    outside[collar] = False
    result = {"method": method, "status": "success", "patch_path": str(patch_path.resolve()),
              "input_geometry_key": spec["input_geometry_key"], "seam_normal_error": seam, **metrics,
              "strict_closure_generation": bool(topology["strict_target_closure"]),
              "patch_disk_topology": bool(topology["patch_disk_topology"]),
              "retained_max_displacement_over_edge": float(displacement.max(initial=0)),
              "retained_rms_displacement_over_edge": float(np.sqrt(np.mean(displacement ** 2))),
              "outside_support_bitwise_fixed": bool(np.array_equal(vertices[:len(before)][outside], before[outside])),
              "bounded_joint_diagnostics": info, "native_details": native,
              "geometry_seconds": time.perf_counter() - started,
              "verification_status": "pending", "introduced_intersection": None,
              "strict_selected_loop_closure": False, "safe_eligible": False}
    # Geometry is persisted before verification; safety never enters identity cost.
    write(output, result)
    verification_path = output.with_name(output.stem + "_verification.json")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.run([sys.executable, str(ROOT / "scripts/cgal_changed_surface_audit.py"), "--input", spec["input_path"],
                        "--patch", str(patch_path), "--output", str(verification_path)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=max(1, 112 - (time.perf_counter() - started)),
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    verification = json.loads(verification_path.read_text("utf-8")) if p.returncode == 0 and verification_path.exists() else {"status": "native_failure"}
    result.update(verification_status=verification.get("status"), introduced_intersection=verification.get("introduced_intersection"),
                  strict_selected_loop_closure=verification.get("strict_selected_loop_closure") is True,
                  verification_path=str(verification_path), worker_seconds=time.perf_counter() - started)
    result["safe_eligible"] = bool(result["verification_status"] == "success" and result["introduced_intersection"] is False and result["strict_selected_loop_closure"])
    write(output, result)
    return result


def _run_candidate(spec, destination):
    spec_path = destination.with_name(destination.stem + "_spec.json")
    write(spec_path, spec)
    start = time.perf_counter()
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
    process = subprocess.Popen([sys.executable, "-m", "tcmr.integrated_repair", str(spec_path), str(destination)],
                               cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        stdout, stderr = process.communicate(timeout=120)
        result = json.loads(destination.read_text("utf-8")) if destination.exists() else {"method": spec["method"], "status": "candidate_failure"}
        if process.returncode:
            result.update(verification_status="candidate_failure", introduced_intersection=None, safe_eligible=False,
                          error=stderr[-1500:], returncode=process.returncode)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            process.kill()
        process.communicate()
        result = json.loads(destination.read_text("utf-8")) if destination.exists() else {"method": spec["method"], "status": "candidate_timeout"}
        result.update(verification_status="timeout", introduced_intersection=None, safe_eligible=False, candidate_timeout=True)
    result.update(inclusive_candidate_seconds=time.perf_counter() - start, input_geometry_key=spec["input_geometry_key"])
    write(destination, result)
    return result


def _repair_loop(vertices, faces, prior, work):
    work.mkdir(parents=True, exist_ok=False)
    loop = np.asarray(prior["loop"], np.int64)
    key = content_key(vertices, faces, loop)
    inp = work / "damaged_input.npz"
    np.savez_compressed(inp, vertices=vertices, faces=faces, selected_loop=loop)
    candidates = []
    for method in POOL:
        spec = {"method": method, "input_path": str(inp.resolve()), "input_geometry_key": key}
        if method == "bounded_joint10":
            refine = candidates[0]
            if not refine.get("patch_path") or not Path(refine["patch_path"]).is_file():
                candidates.append({"method": method, "status": "initializer_unavailable", "safe_eligible": False,
                                   "introduced_intersection": None, "verification_status": "unavailable"})
                continue
            spec["initializer_path"] = refine["patch_path"]
        candidates.append(_run_candidate(spec, work / (method + ".json")))
    finite = [c for c in candidates if c.get("seam_normal_error") is not None and
              c.get("dimensionless_legacy_energy") is not None and math.isfinite(c["seam_normal_error"]) and math.isfinite(c["dimensionless_legacy_energy"])]
    representative = min(finite, key=lambda c: (c["seam_normal_error"], POOL.index(c["method"]))) if finite else None
    penalty = None
    if representative:
        energy = representative["dimensionless_legacy_energy"]
        penalty = representative["seam_normal_error"] / (np.pi / 2) + min(representative["collar_rms_over_boundary_edge"], 1) + energy / (1 + energy)
    return {"loop_id": int(prior["loop_id"]), "prior_probability": prior["prior_probability"], "prior_logit": prior["prior_logit"],
            "input_geometry_key": key, "input_path": str(inp.resolve()), "candidates": candidates,
            "geometry_representative": representative["method"] if representative else None,
            "fixed_geometry_penalty": penalty, "identity_score": prior["prior_logit"] - penalty if penalty is not None else None}


def _cue_geometry_gate(loop_results, identity, cue):
    """Resolve a low-confidence prior/geometry disagreement without labels.

    The gate uses only the frozen cue and quantities already produced during
    candidate generation.  It is deliberately conservative: a cue switch is
    made only when the prior margin is small, the selected geometry is
    strongly penalized, or the cue is substantially closer to the alternative
    loop.  Target labels are never consulted.
    """
    if identity is None or cue is None or len(loop_results) < 2:
        return identity
    cue = np.asarray(cue, dtype=float).reshape(-1)[:3]
    if cue.size != 3 or not np.isfinite(cue).all():
        return identity
    distances = []
    for row in loop_results:
        try:
            with np.load(row["input_path"], allow_pickle=False) as z:
                vertices = np.asarray(z["vertices"], dtype=float)
                loop = np.asarray(z["selected_loop"], dtype=np.int64)
            distances.append((float(np.linalg.norm(vertices[loop].mean(axis=0) - cue)), row))
        except Exception:
            return identity
    cue_row = min(distances, key=lambda x: (x[0], x[1]["loop_id"]))[1]
    if int(cue_row["loop_id"]) == int(identity["loop_id"]):
        return identity
    probs = [float(r.get("prior_probability", 0.0)) for r in loop_results]
    margin = abs(probs[0] - probs[1])
    penalty = float(identity.get("fixed_geometry_penalty") or 0.0)
    max_distance = max(d for d, _ in distances)
    cue_ratio = min(d for d, r in distances if int(r["loop_id"]) == int(cue_row["loop_id"])) / max(max_distance, 1e-12)
    if margin < 0.2 or penalty >= 0.55 or cue_ratio < 0.5:
        return cue_row
    return identity


def select_policies(loop_results, cue=None):
    """Pure inference decisions. Safety has no role in selecting loop identity."""
    prior = loop_results[0]
    finite = [r for r in loop_results if r["identity_score"] is not None]
    cost = max(finite, key=lambda r: r["identity_score"]) if finite else None
    prior = _cue_geometry_gate(loop_results, prior, cue)
    cost = _cue_geometry_gate(loop_results, cost, cue)
    result = {}
    for name in POLICIES:
        identity = cost if name == "cost_target_locked_pool_with_joint" else prior
        allowed = ("cgal_refine_fair",) if name == "prior_locked_cgal_safe" else POOL[:-1] if name == "prior_locked_pool_without_joint" else POOL
        safe = [c for c in identity["candidates"] if c["method"] in allowed and c.get("safe_eligible") is True] if identity else []
        chosen = min(safe, key=lambda c: (c["seam_normal_error"], POOL.index(c["method"]))) if safe else None
        result[name] = {"locked_loop_id": identity["loop_id"] if identity else None,
                        "selected_loop_id": identity["loop_id"] if chosen else None,
                        "decision": "accept" if chosen else "null", "chosen_method": chosen["method"] if chosen else None,
                        "patch_path": chosen["patch_path"] if chosen else None,
                        "output_path": chosen["patch_path"] if chosen else identity["input_path"] if identity else loop_results[0]["input_path"],
                        "metrics": {k: chosen.get(k) for k in ("seam_normal_error", "patch_triangle_quality_mean", "patch_triangle_quality_p05", "patch_triangle_quality_min", "retained_max_displacement_over_edge", "retained_rms_displacement_over_edge", "outside_support_bitwise_fixed", "strict_selected_loop_closure", "introduced_intersection")} if chosen else None}
    return result


def repair_from_top_two(vertices, faces, top_two, work_dir, workers=2, cue=None):
    """Repair up to two prior candidates; one-loop inputs use the same policy.

    The original name remains for compatibility with the frozen multi-loop
    experiment. No geometry or selection rule changes for that experiment.
    """
    if not 1 <= len(top_two) <= 2 or workers not in (1, 2):
        raise ValueError("Require one or two distinct loops and one or two workers")
    if len({r['loop_id'] for r in top_two}) != len(top_two):
        raise ValueError("Top2 loops must be distinct")
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_repair_loop, np.asarray(vertices, float), np.asarray(faces, np.int64), prior,
                                   work_dir / f"loop{prior['loop_id']}") for prior in top_two]
        rows = [future.result() for future in futures]
    result = {"target_labels_read": False, "loops": rows, "decisions": select_policies(rows, cue=cue),
              "wall_seconds": time.perf_counter() - started,
              "candidate_inclusive_seconds_sum": sum(c.get("inclusive_candidate_seconds", 0) for r in rows for c in r["candidates"])}
    write(work_dir / "inference.json", result)
    return result


def infer(vertices, faces, cue, work_dir, ranker_path=None, cbe_path=None, workers=2):
    """Live full inference API: damaged vertices/faces/cue -> selected patch or null."""
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.train_cue_geometry_ranker import cbe_logits, loop_features
    from .boundary import extract_boundary_loops, vertex_normals
    from .features import construct_loop_features
    from scripts.local_repair_topology import inspect_target_cycle
    loops = extract_boundary_loops(vertices, faces)
    candidates = [i for i, loop in enumerate(loops) if inspect_target_cycle(faces, loop)["input_simple_closed_boundary"]]
    if not candidates:
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=False)
        unchanged = work_dir / 'unchanged_input.npz'
        np.savez_compressed(unchanged, vertices=vertices, faces=faces)
        decisions = {name: {'locked_loop_id': None, 'selected_loop_id': None,
                           'decision': 'null', 'chosen_method': None, 'patch_path': None,
                           'output_path': str(unchanged), 'metrics': None} for name in POLICIES}
        result = {'target_labels_read': False, 'loops': [], 'decisions': decisions,
                  'abstention_reason': 'no_valid_boundary',
                  'candidate_inclusive_seconds_sum': 0., 'wall_seconds': 0.}
        write(work_dir / 'inference.json', result)
        return result
    normals = vertex_normals(vertices, faces)
    features = [construct_loop_features(vertices, faces, loops[i], cue, cached_normals=normals) for i in candidates]
    tokens = np.asarray([r[0] for r in features], np.float32)
    scalars = np.asarray([r[1]["scalars"] for r in features], np.float32)
    lg = cbe_logits(tokens, scalars, cbe_path or ROOT / "models/cbe.pt")
    model = joblib.load(ranker_path or ROOT / "models/ranker.joblib")
    with threadpool_limits(limits=2):
        probabilities = model.predict_proba(loop_features(tokens, scalars, lg))[:, 1]
    order = np.argsort(-probabilities, kind="stable")[:2]
    top_two = [{"loop_id": int(candidates[i]), "loop": np.asarray(loops[candidates[i]], np.int64).tolist(),
                "prior_probability": float(probabilities[i]), "prior_logit": float(np.log(np.clip(probabilities[i], 1e-7, 1 - 1e-7) / (1 - np.clip(probabilities[i], 1e-7, 1 - 1e-7))))} for i in order]
    return repair_from_top_two(vertices, faces, top_two, work_dir, workers, cue=cue)


if __name__ == "__main__":
    spec_path, output_path = map(Path, sys.argv[1:3])
    try:
        candidate_worker(json.loads(spec_path.read_text("utf-8")), output_path)
    except Exception as exc:
        # Preserve finite geometry written before a verification exception.
        result = json.loads(output_path.read_text("utf-8")) if output_path.exists() else {}
        result.update(method=json.loads(spec_path.read_text("utf-8"))["method"], status=result.get("status", "candidate_failure"),
                      verification_status="candidate_failure", introduced_intersection=None, safe_eligible=False,
                      error=f"{type(exc).__name__}: {exc}")
        write(output_path, result)
