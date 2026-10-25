"""Unified, label-blind generation and label-aware scoring for the 79-case benchmark.

The fixed TCMR and CGAL predictions are generated before this script opens the
target labels.  The score stage then applies one protocol to all available
methods.  Target-only safety requires a selected target, a closed selected
loop, a still-open distractor loop, and no newly introduced geometric defect.
Unknown full-mesh intersection or degeneration status is never counted safe.

The SeMIGCN output is an explicitly adapted target-only track.  Its learned
vertices are retained, while only the cue-selected cap is recovered by exact
face membership and append offsets from the common initializer.  Unselected
caps are removed without reading target labels.  A separate global diagnostic
keeps the original all-cap output visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCH = ROOT / "experiments" / "formal_coseg_external_benchmark"
PROTOCOL_PATH = BENCH / "protocol.json"
PRED_DIR = BENCH / "predictions"
MESHFIX_PATH = BENCH / "meshfix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def verify_local_output(before_v, before_f, after_v, patch_f, selected_loop, destination):
    """Materialize and verify an output without consulting target annotations."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    full_f = np.vstack([before_f, patch_f])
    output = destination / "returned_mesh.npz"
    np.savez_compressed(output, vertices=after_v, faces=full_f, before_vertices=before_v,
                        retained_faces=before_f, patch_faces=patch_f,
                        selected_loop=np.asarray(selected_loop if selected_loop is not None else [], np.int64))
    started = time.perf_counter()
    report_path = destination / "runtime_verification.json"
    # Isolate native CGAL from Torch's Windows DLLs, as in candidate generation.
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            "--verify-local-mesh", str(output), "--verification-report", str(report_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
            env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        verification = load(report_path, {}) if process.returncode == 0 else {}
        if not verification:
            verification = {"status": "native_failure", "introduced_intersection": None,
                            "strict_selected_loop_closure": False, "returncode": process.returncode,
                            "error": process.stderr[-1500:]}
    except subprocess.TimeoutExpired:
        verification = {"status": "timeout", "introduced_intersection": None, "strict_selected_loop_closure": False}
    verification["verification_seconds"] = time.perf_counter() - started
    verification["output_path"] = str(output.resolve())
    verification["target_labels_read"] = False
    dump(destination / "runtime_verification.json", verification)
    return verification


def verify_local_mesh(path, report_path):
    from scripts.cgal_changed_surface_audit import audit_changed_surface
    with np.load(path, allow_pickle=False) as z:
        loop = z["selected_loop"]
        value = audit_changed_surface(z["before_vertices"], z["retained_faces"], z["vertices"],
                                      z["patch_faces"], loop if len(loop) else None)
        value["output_topology"] = topology(z["vertices"], z["faces"])
        value["input_topology"] = topology(z["before_vertices"], z["retained_faces"])
    dump(report_path, value)


def verify_learning_case(index: int, stage: str, base: Path):
    # Read only label-free input identity and generation records from protocol.
    entry = next(c for c in load(PROTOCOL_PATH)["cases"] if int(c["index"]) == index)
    case = {k: entry[k] for k in ("index", "sample_id", "input_path", "input_sha256", "cue")}
    root = base / "cases" / f"case{index:03d}"
    destination = root / stage / root.name
    started = time.perf_counter()
    manifest = load(root / "prepared_manifest.json")
    mapping = semigcn_selected_faces(case, find_prediction(case), destination, manifest)
    if not mapping or mapping.get("error"):
        raise RuntimeError(f"Cannot recover cue-selected cap: {mapping}")
    with np.load(destination / "restore_mapping.npz", allow_pickle=False) as z:
        before_v, before_f, loop = z["vertices"], z["faces"], z["selected_loop"]
    with np.load(destination / "restored_patch.npz", allow_pickle=False) as z:
        after_v = z["vertices"]
    verification = verify_local_output(before_v, before_f, after_v, mapping["patch_faces"], loop, destination)
    verification["adapter_export_verification_seconds"] = time.perf_counter() - started
    dump(destination / "runtime_verification.json", verification)
    # Seed the normal scorer with this measured, complete native verification.
    fingerprint = hashlib.sha256(b"".join(np.ascontiguousarray(x).tobytes() for x in
        (before_v, before_f, after_v, mapping["patch_faces"], loop))).hexdigest()
    verification.update(audit_input="selected_cap_only_full_changed_surface", input_sha256=fingerprint)
    dump(destination / "selected_changed_surface_audit.json", verification)
    print(json.dumps({"verified_case": index, "verification_status": verification.get("status"),
                      "introduced_intersection": verification.get("introduced_intersection")}), flush=True)
    return verification


def measure_runtime_case(index: int, method_name: str):
    """Fresh one-case measurement; no cached geometry enters the timed path."""
    protocol = load(PROTOCOL_PATH)
    entry = next(c for c in protocol["cases"] if int(c["index"]) == index)
    case = {k: entry[k] for k in ("index", "sample_id", "input_path", "input_sha256", "cue")}
    destination = BENCH / "runtime" / f"case{index:03d}" / method_name
    result_path = destination / "result.json"
    if result_path.exists():
        old = load(result_path)
        if old.get("input_sha256") != case["input_sha256"]:
            raise RuntimeError("Runtime record belongs to a different input")
        return old
    recovery = destination.exists()
    work_directory = destination / ("recovery_work" if recovery else "work")
    if work_directory.exists():
        raise RuntimeError(f"Repeated incomplete timed run needs diagnosis: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if sha(Path(case["input_path"])) != case["input_sha256"]:
        raise RuntimeError("Frozen input changed")
    started = time.perf_counter()
    try:
        before_v, before_f = load_input(case)
        if method_name == "meshfix":
            import pymeshfix
            from scripts.cgal_triangle_soup_audit import intersections
            after_v, full_f = pymeshfix.clean_from_arrays(before_v, before_f, verbose=False)
            after_v, full_f = np.asarray(after_v, float), np.asarray(full_f, np.int64)
            np.savez_compressed(destination / "returned_mesh.npz", vertices=after_v, faces=full_f)
            generated = time.perf_counter()
            verification = intersections(after_v, full_f, 0)
            verification["output_topology"] = topology(after_v, full_f)
            dump(destination / "runtime_verification.json", verification)
            prediction = None
        else:
            from tcmr import integrated_repair as method
            if method_name == "tcmr":
                ranker = ROOT / "models/ranker.joblib"
                cbe = ROOT / "models/cbe.pt"
                prediction = method.infer(before_v, before_f, np.asarray(case["cue"], float),
                    work_directory, ranker_path=ranker, cbe_path=cbe, workers=1)
            elif method_name == "cgal_pool":
                from tcmr.boundary import extract_boundary_loops
                from scripts.local_repair_topology import inspect_target_cycle
                loops = extract_boundary_loops(before_v, before_f)
                choices = [(float(np.linalg.norm(before_v[loop].mean(0) - np.asarray(case["cue"][:3]))), i, loop)
                           for i, loop in enumerate(loops)
                           if inspect_target_cycle(before_f, loop)["input_simple_closed_boundary"]]
                loop_rows = []
                decision = {"decision": "null", "selected_loop_id": None, "chosen_method": None}
                if choices:
                    _, loop_id, loop = min(choices, key=lambda x: (x[0], x[1]))
                    inp = destination / "damaged_input.npz"
                    np.savez_compressed(inp, vertices=before_v, faces=before_f, selected_loop=loop)
                    candidates = [method._run_candidate({"method": name, "input_path": str(inp.resolve()),
                        "input_geometry_key": method.content_key(before_v, before_f, loop)}, destination / f"{name}.json")
                        for name in ("cgal_refine_fair", "cgal_polyline")]
                    safe = [c for c in candidates if c.get("safe_eligible") is True and c.get("seam_normal_error") is not None]
                    chosen = min(safe, key=lambda c: (c["seam_normal_error"], c["method"])) if safe else None
                    decision = {"decision": "accept" if chosen else "null", "selected_loop_id": loop_id if chosen else None,
                                "chosen_method": chosen["method"] if chosen else None}
                    loop_rows = [{"loop_id": loop_id, "input_path": str(inp.resolve()), "candidates": candidates}]
                prediction = {"loops": loop_rows, "decisions": {"prior_locked_pool_with_joint": decision}}
            else:
                raise ValueError(method_name)
            dump(destination / "prediction.json", prediction)
            decision, loop_row, candidate = candidate_for(prediction, "tcmr", case)
            after_v, patch_f, loop = before_v, np.empty((0, 3), np.int64), None
            if decision.get("decision") == "accept" and candidate and candidate.get("patch_path"):
                with np.load(candidate["patch_path"], allow_pickle=False) as z:
                    after_v, patch_f = z["vertices"], z["patch_faces"]
                with np.load(loop_row["input_path"], allow_pickle=False) as z:
                    loop = z["selected_loop"]
            generated = time.perf_counter()
            verification = verify_local_output(before_v, before_f, after_v, patch_f, loop, destination)
        finished = time.perf_counter()
        result = {"status": "complete", "method": method_name, "index": index, "sample_id": case["sample_id"],
                  "input_sha256": case["input_sha256"], "input_to_verified_output_wall_seconds": finished-started,
                  "generation_seconds": generated-started, "export_and_verification_seconds": finished-generated,
                  "input_vertices": len(before_v), "input_faces": len(before_f), "target_labels_read_in_timing": False,
                  "workers": 1, "prediction_path": str(destination / "prediction.json") if prediction else None,
                  "recovered_after_interrupted_attempt": recovery,
                  "output_path": str(destination / "returned_mesh.npz"),
                  "scope": "NPZ input read through returned mesh export and native geometry verification; startup/import inside method included; evaluation labels and metric calculations excluded"}
    except Exception as exc:
        result = {"status": "failed", "method": method_name, "index": index, "sample_id": case["sample_id"],
                  "input_sha256": case["input_sha256"], "input_to_verified_output_wall_seconds": time.perf_counter()-started,
                  "error": repr(exc), "target_labels_read_in_timing": False}
    dump(result_path, result)
    print(json.dumps(result), flush=True)
    return result


def measure_runtime(case_indices=None):
    cases = load(PROTOCOL_PATH)["cases"]
    if case_indices is not None:
        cases = [c for c in cases if int(c["index"]) in case_indices]
    # A fresh process for every method/input includes per-input model setup.
    for case in cases:
        names = ["tcmr", "cgal_pool", "meshfix"]
        shift = int(case["index"]) % len(names)
        for name in names[shift:] + names[:shift]:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--benchmark-root", str(BENCH),
                            "--measure-runtime-case", str(case["index"]), "--runtime-method", name], check=True,
                           cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"))


def score_annotations(case):
    if "target_loop_vertices" in case and "distractor_loop_vertices" in case:
        return case
    ledger = load(BENCH / "score_labels.json", {"rows": []})
    annotation = next(x for x in ledger["rows"] if x["sample_id"] == case["sample_id"])
    return {**case, **{k: annotation[k] for k in ("target_loop_vertices", "distractor_loop_vertices")}}


def selected_decision_row(case, prediction, decision):
    modified = dict(prediction)
    modified["decisions"] = {"prior_locked_pool_with_joint": decision}
    return unified_method_row(case, modified, "tcmr")


def runtime_summary(rows, stage, base):
    raw = []
    for row in rows:
        index = int(row["index"])
        item = {"index": index, "sample_id": row["sample_id"], "methods": {}}
        for name in ("tcmr", "cgal_pool", "meshfix"):
            record = load(BENCH / "runtime" / f"case{index:03d}" / name / "result.json", {})
            if record:
                item["methods"][name] = record
        if base:
            path = base / "cases" / f"case{index:03d}" / stage / f"case{index:03d}" / "stage_output_manifest.json"
            status = load(path, {})
            measured = status.get("runtime_measurement", {})
            seconds = measured.get("raw_input_to_verified_output_wall_seconds")
            if seconds is not None:
                item["methods"]["semigcn"] = {"status": "complete" if measured.get("runtime_verification_returncode") == 0 else "failed",
                    "input_to_verified_output_wall_seconds": seconds, "details": measured}
        raw.append(item)
    summary = {}
    for name in ("tcmr", "cgal_pool", "meshfix", "semigcn"):
        records = [r["methods"][name] for r in raw if name in r["methods"]]
        values = [r["input_to_verified_output_wall_seconds"] for r in records if r.get("status") == "complete"]
        summary[name] = {"terminal_n": len(records), "failed_n": sum(r.get("status") == "failed" for r in records),
                         "completed_wall_seconds": _stability_distribution(values),
                         "failure_wall_seconds": _stability_distribution([r["input_to_verified_output_wall_seconds"]
                             for r in records if r.get("status") == "failed"])}
    paired = {}
    for name in ("cgal_pool", "meshfix", "semigcn"):
        pairs = [(r["methods"].get("tcmr"), r["methods"].get(name)) for r in raw]
        pairs = [(a["input_to_verified_output_wall_seconds"], b["input_to_verified_output_wall_seconds"])
                 for a, b in pairs if a and b and a.get("status") == b.get("status") == "complete"]
        x, y = [a for a, _ in pairs], [b for _, b in pairs]
        paired[f"tcmr_minus_{name}"] = {"n": len(pairs), "mean_difference_seconds": float(np.mean(np.asarray(x)-y)) if pairs else None,
                                         "bootstrap_ci95_seconds": bootstrap(x, y)}
    return {"scope": "Fresh per-input generation, returned mesh export, and uncached native geometry verification; per-input initialization included; scoring against target labels excluded. One active method/job, two native math threads; TCMR candidate workers=1 and learning workers=1. Shared process startup is excluded; method-specific helper launches are included.",
            "hardware_note": "Same host, method-native CPU/CUDA backends; no claim of identical arithmetic hardware or exclusive system ownership.",
            "complete": all(summary[k]["terminal_n"] == len(rows) for k in summary),
            "summary": summary, "paired": paired, "rows": raw}


def edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    ff = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(ff):
        return {}
    ee = np.sort(np.concatenate([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]], axis=0), axis=1)
    unique, counts = np.unique(ee, axis=0, return_counts=True)
    return {tuple(map(int, e)): int(n) for e, n in zip(unique, counts)}


def area_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    ff = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(ff):
        return np.empty(0, dtype=float)
    tri = np.asarray(vertices, dtype=float)[ff]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def topology(vertices: np.ndarray, faces: np.ndarray) -> dict:
    ff = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    finite = bool(np.isfinite(vertices).all() and np.isfinite(ff).all())
    areas = area_faces(vertices, ff)
    edges = edge_counts(ff)
    return {
        "finite": finite,
        "degenerate_faces": int(np.sum(areas <= 1e-12)) if finite else None,
        "nonmanifold_edges": int(sum(n > 2 for n in edges.values())) if finite else None,
        "boundary_edges": int(sum(n == 1 for n in edges.values())) if finite else None,
    }


def loop_edges(loop) -> list[tuple[int, int]]:
    ll = np.asarray(loop, dtype=np.int64).reshape(-1)
    return [tuple(sorted((int(a), int(b)))) for a, b in zip(ll, np.roll(ll, -1))]


def loop_edge_counts(faces: np.ndarray, loop) -> list[int]:
    counts = edge_counts(faces)
    return [counts.get(e, 0) for e in loop_edges(loop)]


def loop_state(faces: np.ndarray, loop) -> dict:
    ll = np.asarray(loop, dtype=np.int64).reshape(-1)
    cc = loop_edge_counts(faces, ll)
    return {"n_vertices": int(len(ll)), "unique": bool(len(np.unique(ll)) == len(ll)),
            "edge_counts": cc, "closed": bool(len(ll) >= 3 and len(np.unique(ll)) == len(ll) and all(n >= 2 for n in cc)),
            "open": bool(len(ll) >= 3 and len(np.unique(ll)) == len(ll) and all(n == 1 for n in cc))}


def seam_normal_metrics(vertices: np.ndarray, retained_faces: np.ndarray,
                        patch_faces: np.ndarray, loop) -> dict:
    """Per-edge oriented dihedral: absolute angles are the error endpoint.

    The loop direction supplies the edge tangent.  This deliberately does not
    take ``abs(dot(n_retained, n_patch))``; legacy unoriented angles remain in
    the historical reports only.
    """
    rf = np.asarray(retained_faces, dtype=np.int64).reshape(-1, 3)
    pf = np.asarray(patch_faces, dtype=np.int64).reshape(-1, 3)
    ll = np.asarray(loop, dtype=np.int64).reshape(-1)
    if not len(rf) or not len(pf) or len(ll) < 3:
        return {"n": 0, "signed_mean": None, "absolute_mean": None, "absolute_p95": None}

    def normal(face):
        q = np.asarray(vertices, dtype=float)[face]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        d = np.linalg.norm(n)
        return n / d if d > 1e-12 else None

    def oriented(face, a, b):
        f = np.asarray(face, dtype=np.int64)
        for x, y in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if int(x) == int(a) and int(y) == int(b):
                return 1
            if int(x) == int(b) and int(y) == int(a):
                return -1
        return 0

    rm, pm = {}, {}
    for f in rf:
        for e in (f[[0, 1]], f[[1, 2]], f[[2, 0]]):
            rm.setdefault(tuple(sorted(map(int, e))), f)
    for f in pf:
        for e in (f[[0, 1]], f[[1, 2]], f[[2, 0]]):
            pm.setdefault(tuple(sorted(map(int, e))), f)
    angles = []
    for a, b in zip(ll, np.roll(ll, -1)):
        key = tuple(sorted((int(a), int(b))))
        if key not in rm or key not in pm:
            continue
        nr, npatch = normal(rm[key]), normal(pm[key])
        edge = np.asarray(vertices, dtype=float)[int(b)] - np.asarray(vertices, dtype=float)[int(a)]
        en = np.linalg.norm(edge)
        if nr is None or npatch is None or en <= 1e-12:
            continue
        tangent = edge / en
        # Correct the face normals to the common seam orientation before
        # measuring the dihedral.  This preserves a signed winding measure
        # while avoiding an unoriented absolute cosine.
        sr, sp = oriented(rm[key], a, b), oriented(pm[key], a, b)
        if sr == 0 or sp == 0:
            continue
        nr = nr * sr
        npatch = npatch * (-sp)
        cross = np.cross(nr, npatch)
        angles.append(float(np.arctan2(np.dot(cross, tangent), np.dot(nr, npatch))))
    absolute = np.abs(np.asarray(angles, dtype=float))
    return {"n": len(angles), "signed_mean": float(np.mean(angles)) if angles else None,
            "absolute_mean": float(np.mean(absolute)) if angles else None,
            "absolute_p95": float(np.percentile(absolute, 95)) if angles else None}


def triangle_quality_stats(vertices: np.ndarray, patch_faces: np.ndarray) -> dict:
    ff = np.asarray(patch_faces, dtype=np.int64).reshape(-1, 3)
    if not len(ff):
        return {"mean": None, "p05": None, "min": None}
    q = np.asarray(vertices, dtype=float)[ff]
    lengths = np.stack([np.linalg.norm(q[:, 1] - q[:, 0], axis=1),
                        np.linalg.norm(q[:, 2] - q[:, 1], axis=1),
                        np.linalg.norm(q[:, 0] - q[:, 2], axis=1)], axis=1)
    # 4*sqrt(3)*area / sum(edge^2), equal to one for an equilateral triangle.
    ar = area_faces(vertices, ff)
    quality = 4.0 * np.sqrt(3.0) * ar / np.maximum(np.sum(lengths * lengths, axis=1), 1e-12)
    quality = quality[np.isfinite(quality)]
    if not len(quality):
        return {"mean": None, "p05": None, "min": None}
    return {"mean": float(np.mean(quality)), "p05": float(np.percentile(quality, 5)),
            "min": float(np.min(quality))}


def geometry_metrics(before_v, before_f, after_v, patch_f, selected_loop) -> dict:
    before_v = np.asarray(before_v, dtype=float)
    before_f = np.asarray(before_f, dtype=np.int64)
    after_v = np.asarray(after_v, dtype=float)
    patch_f = np.asarray(patch_f, dtype=np.int64).reshape(-1, 3)
    if len(after_v) < len(before_v) or not np.isfinite(after_v).all():
        return {"status": "unknown", "error": "invalid retained-prefix output"}
    full_f = np.vstack([before_f, patch_f])
    bt, at = topology(before_v, before_f), topology(after_v, full_f)
    moved = np.linalg.norm(after_v[:len(before_v)] - before_v, axis=1)
    edges = np.concatenate([before_f[:, [0, 1]], before_f[:, [1, 2]], before_f[:, [2, 0]]])
    lengths = np.linalg.norm(before_v[edges[:, 0]] - before_v[edges[:, 1]], axis=1)
    h = float(np.median(lengths)) if len(lengths) else 1.0
    h = max(h, 1e-12)
    collar_vertices = set(np.asarray(selected_loop, dtype=np.int64).tolist())
    # The 2-ring support is used only for reporting a fixed support-area
    # denominator; generation never uses labels.
    for _ in range(2):
        current = set(collar_vertices)
        neighbors = {int(x) for f in before_f if any(int(v) in current for v in f) for x in f}
        collar_vertices.update(neighbors)
    support = np.flatnonzero(np.any(np.isin(before_f, list(collar_vertices)), axis=1))
    support_area = float(np.sum(area_faces(before_v, before_f[support]))) if len(support) else 0.0
    patch_area = float(np.sum(area_faces(after_v, patch_f))) if len(patch_f) else 0.0
    changed_faces = np.flatnonzero(np.any(moved[before_f] > 0.0, axis=1))
    changed_area = float(np.sum(area_faces(before_v, before_f[changed_faces]))) if len(changed_faces) else 0.0
    denom = float(np.sum(area_faces(before_v, before_f)))
    q = triangle_quality_stats(after_v, patch_f)
    # Existing defects are tolerated only when they do not increase.  A
    # changed-surface audit with unknown result is never safe.
    no_new = (bt["finite"] and at["finite"] and
              at["degenerate_faces"] <= bt["degenerate_faces"] and
              at["nonmanifold_edges"] <= bt["nonmanifold_edges"])
    seam = seam_normal_metrics(after_v, before_f, patch_f, selected_loop)
    return {
        "status": "success", "before_topology": bt, "after_topology": at,
        "introduced_degenerate": bool(at["degenerate_faces"] > bt["degenerate_faces"]),
        "introduced_nonmanifold": bool(at["nonmanifold_edges"] > bt["nonmanifold_edges"]),
        "geometry_no_new_topology_defect": bool(no_new),
        "signed_seam_normal_mean": seam["signed_mean"],
        "seam_normal_absolute_mean": seam["absolute_mean"],
        "seam_normal_absolute_p95": seam["absolute_p95"],
        "seam_measured_edges": seam["n"],
        "triangle_quality": q,
        "retained_displacement_mean_over_edge": float(np.mean(moved) / h),
        "retained_displacement_rms_over_edge": float(np.sqrt(np.mean(moved * moved)) / h),
        "retained_displacement_p95_over_edge": float(np.percentile(moved, 95) / h),
        "retained_displacement_max_over_edge": float(np.max(moved, initial=0.0) / h),
        "retained_outside_two_ring_bitwise_fixed": bool(np.array_equal(after_v[:len(before_v)][~np.isin(np.arange(len(before_v)), list(collar_vertices))],
                                                                      before_v[~np.isin(np.arange(len(before_v)), list(collar_vertices))])),
        "patch_area": patch_area, "moved_retained_area": changed_area,
        "changed_surface_area_ratio": float((patch_area + changed_area) / denom) if denom > 0 else None,
        "patch_to_two_ring_support_area_ratio": float(patch_area / support_area) if support_area > 0 else None,
        "median_input_edge": h, "patch_faces": int(len(patch_f)),
    }


def load_input(case: dict):
    with np.load(case["input_path"], allow_pickle=False) as z:
        return np.asarray(z["vertices"], float), np.asarray(z["faces"], np.int64)


def find_prediction(case: dict) -> dict:
    return load(PRED_DIR / f"{int(case['index']):03d}.json", {})


def candidate_for(pred: dict, method: str, case: dict):
    if method == "tcmr":
        decision = pred.get("decisions", {}).get("prior_locked_pool_with_joint", {})
        sid = decision.get("selected_loop_id")
        chosen = decision.get("chosen_method")
        row = next((x for x in pred.get("loops", []) if x.get("loop_id") == sid), None)
        cand = next((x for x in (row or {}).get("candidates", []) if x.get("method") == chosen), None)
        return decision, row, cand
    cue = np.asarray(case.get("cue", [])[:3], float)
    choices = []
    for row in pred.get("loops", []):
        cand = next((x for x in row.get("candidates", []) if x.get("method") == "cgal_refine_fair"), None)
        if cand is None:
            continue
        try:
            with np.load(row["input_path"], allow_pickle=False) as z:
                vv, ll = np.asarray(z["vertices"], float), np.asarray(z["selected_loop"], np.int64)
            choices.append((float(np.linalg.norm(vv[ll].mean(axis=0) - cue)), int(row["loop_id"]), row, cand))
        except Exception:
            continue
    if not choices:
        return {"decision": "null"}, None, None
    _, _, row, cand = min(choices, key=lambda x: (x[0], x[1]))
    safe = cand.get("safe_eligible") is True
    return {"decision": "accept" if safe else "null", "chosen_method": "cgal_refine_fair"}, row, cand


def audit_status(cand: dict) -> dict:
    path = Path(cand.get("verification_path", "")) if cand else Path()
    value = load(path, None) if path else None
    if isinstance(value, dict) and value.get("status") == "success":
        return value
    return {"status": "unknown", "introduced_intersection": None,
            "strict_selected_loop_closure": False}


def audit_semigcn_changed_surface(root: Path, before_v, before_f, after_v,
                                  patch_f, selected_loop) -> dict:
    """Run/cache the same full changed-surface CGAL primitive for learning output."""
    cache = root / "selected_changed_surface_audit.json"
    fingerprint = hashlib.sha256(b"".join(np.ascontiguousarray(x).tobytes()
                                        for x in (before_v, before_f, after_v, patch_f, selected_loop))).hexdigest()
    if cache.exists():
        value = load(cache, {})
        if (value.get("audit_input") == "selected_cap_only_full_changed_surface" and
                value.get("input_sha256") == fingerprint):
            return value
    from scripts.cgal_changed_surface_audit import audit_changed_surface
    value = audit_changed_surface(before_v, before_f, after_v, patch_f, selected_loop)
    value["audit_input"] = "selected_cap_only_full_changed_surface"
    value["input_sha256"] = fingerprint
    value["stage_output"] = str(root / "restored_patch.npz")
    dump(cache, value)
    return value


def unified_method_row(case, pred, method: str) -> dict:
    before_v, before_f = load_input(case)
    decision, loop_row, cand = candidate_for(pred, method, case)
    if not loop_row or not cand or not cand.get("patch_path"):
        terminal_null = bool(pred and decision.get("decision") == "null")
        return {"method": method, "decision": "null", "status": "complete" if terminal_null else "pending", "correct_safe": False,
                "wrong_accepted": False, "selected_correct": False, "distractor_open": False,
                "geometry": {"status": "unknown"}}
    with np.load(loop_row["input_path"], allow_pickle=False) as z:
        selected_loop = np.asarray(z["selected_loop"], dtype=np.int64)
    with np.load(cand["patch_path"], allow_pickle=False) as z:
        after_v = np.asarray(z["vertices"], float)
        patch_f = np.asarray(z["patch_faces"], np.int64)
    geometry = geometry_metrics(before_v, before_f, after_v, patch_f, selected_loop)
    audit = audit_status(cand)
    full_f = np.vstack([before_f, patch_f])
    target = np.asarray(case["target_loop_vertices"], dtype=np.int64)
    distractor = np.asarray(case["distractor_loop_vertices"], dtype=np.int64)
    selected_correct = bool(set(map(int, selected_loop)) == set(map(int, target)))
    distractor_open = loop_state(full_f, distractor)["open"]
    accepted = decision.get("decision") == "accept"
    strict = bool(audit.get("strict_selected_loop_closure") is True)
    no_intersection = audit.get("introduced_intersection") is False
    safe = bool(accepted and strict and no_intersection and geometry.get("geometry_no_new_topology_defect") is True)
    correct_safe = bool(safe and selected_correct and distractor_open)
    return {"method": method, "decision": decision.get("decision"), "status": "complete",
            "selected_loop_id": int(loop_row.get("loop_id")), "chosen_method": cand.get("method"),
            "accepted": accepted, "selected_correct": selected_correct, "distractor_open": distractor_open,
            "selected_closed": strict, "target_closed": loop_state(full_f, target)["closed"],
            "no_new_intersection": no_intersection,
            "correct_safe": correct_safe, "wrong_accepted": bool(accepted and not correct_safe),
            "audit": {k: audit.get(k) for k in ("status", "introduced_intersection", "strict_selected_loop_closure",
                                                  "after_degenerate_faces", "introduced_intersection_nondegenerate")},
            "geometry": geometry,
            "candidate_seconds": cand.get("inclusive_candidate_seconds"),
            "inference_wall_seconds": pred.get("wall_seconds") if method == "tcmr" else None}


def semigcn_selected_faces(case, pred, root: Path, manifest: dict):
    """Recover only the cue-selected cap from the common initializer.

    This repeats the label-free append mapping used by the adapter and never
    reads target/distractor annotation fields.
    """
    mapping = np.load(root / "restore_mapping.npz", allow_pickle=False)
    vin = np.asarray(mapping["vertices"], float)
    all_faces = np.asarray(mapping["patch_faces"], np.int64)
    selected_id = int(manifest["selected_loop_id"])
    choices = []
    for loop_id in manifest["filled_loop_ids"]:
        row = next((r for r in pred.get("loops", []) if int(r.get("loop_id", -1)) == int(loop_id)), None)
        if row is None:
            return {"error": "initializer loop membership missing"}
        if int(row.get("loop_id", -1)) != selected_id:
            continue
        cand = next((x for x in row.get("candidates", []) if x.get("method") == "cgal_refine_fair"), None)
        if cand is None:
            continue
        choices.append((row, cand))
    if not choices:
        return None
    selected_cand = choices[0][1]
    # Locate this exact cap by reproducing append offsets.  Face membership is
    # exact; no target label enters the mapping.
    offset = 0
    selected_segment = None
    ordered = []
    for loop_id in manifest["filled_loop_ids"]:
        row = next((r for r in pred.get("loops", []) if int(r.get("loop_id", -1)) == int(loop_id)), None)
        if row is None:
            return {"error": "initializer loop membership missing"}
        cand = next((x for x in row.get("candidates", []) if x.get("method") == "cgal_refine_fair"), None)
        if cand is None:
            continue
        with np.load(cand["patch_path"], allow_pickle=False) as z:
            local = np.asarray(z["patch_faces"], np.int64)
            pv = np.asarray(z["vertices"], float)
        adjusted = local.copy()
        adjusted[adjusted >= len(vin)] += offset
        ordered.append((int(row.get("loop_id")), adjusted))
        if int(row.get("loop_id")) == selected_id:
            selected_segment = adjusted
        offset += len(pv) - len(vin)
    if selected_segment is None:
        return None
    # Verify membership against the stored common initializer without relying
    # on labels.  This guards stale stage output and face-offset drift.
    flat = np.vstack([x[1] for x in ordered]) if ordered else np.empty((0, 3), int)
    if not np.array_equal(flat, all_faces):
        return {"error": "initializer face membership mismatch"}
    return {"selected_loop_id": selected_id, "patch_faces": selected_segment,
            "all_patch_faces": all_faces, "selected_cand": selected_cand}


def semigcn_row(case, pred, stage: str, base: Path | None = None,
                status_file: Path | None = None) -> dict:
    base = base or BENCH / ("semigcn" if stage == "external" else f"semigcn_{stage}")
    if not base.exists():
        return {"method": f"semigcn_{stage}", "status": "pending", "correct_safe": False,
                "wrong_accepted": False, "selected_correct": False, "distractor_open": False}
    root = base / "cases" / f"case{int(case['index']):03d}"
    run_status = (load(status_file, None) if status_file else None) or load(base / "batch_status.json", None) or load(base / "run_status.json", {"rows": []})
    batch = next((x for x in run_status.get("rows", [])
                  if x.get("sample_id") == case["sample_id"] or x.get("case") == root.name), {})
    destination = root / stage / root.name
    # The historical fit root used an external directory. Preserve that default.
    if stage == "fit" and not destination.exists():
        destination = root / "external" / root.name
    status = load(destination / "stage_output_manifest.json", {})
    if (run_status.get("stage") not in (None, stage) or
            (status.get("iterations") is not None and
             run_status.get("iterations", status.get("iterations")) != status.get("iterations"))):
        batch = {}
    output = destination / "restored_patch.npz"
    manifest = load(root / "prepared_manifest.json", {})
    if not manifest:
        manifest = load(root / "manifest.json", {})
    pred_map = semigcn_selected_faces(case, pred, destination, manifest) if status.get("status") == "completed" and output.exists() else None
    if not pred_map or pred_map.get("error"):
        failed = bool(batch.get("status") == "failed" or pred_map or status.get("status") == "completed")
        return {"method": f"semigcn_{stage}", "status": "failed" if failed else "pending", "correct_safe": False,
                "wrong_accepted": False, "selected_correct": False, "distractor_open": False,
                "error": (pred_map or {}).get("error", batch.get("status", "missing"))}
    mapping = np.load(destination / "restore_mapping.npz", allow_pickle=False)
    before_v, before_f = np.asarray(mapping["vertices"], float), np.asarray(mapping["faces"], np.int64)
    with np.load(output, allow_pickle=False) as z:
        after_v = np.asarray(z["vertices"], float)
    patch_f = np.asarray(pred_map["patch_faces"], np.int64)
    selected_loop = np.asarray(mapping["selected_loop"], np.int64)
    geometry = geometry_metrics(before_v, before_f, after_v, patch_f, selected_loop)
    audit = audit_semigcn_changed_surface(destination, before_v, before_f, after_v,
                                          patch_f, selected_loop)
    full_f = np.vstack([before_f, patch_f])
    target = np.asarray(case["target_loop_vertices"], np.int64)
    distractor = np.asarray(case["distractor_loop_vertices"], np.int64)
    selected_correct = bool(set(map(int, selected_loop)) == set(map(int, target)))
    distractor_open = loop_state(full_f, distractor)["open"]
    no_intersection = audit.get("introduced_intersection") is False
    strict = bool(audit.get("strict_selected_loop_closure") is True)
    safe = bool(strict and no_intersection and geometry.get("geometry_no_new_topology_defect") is True)
    correct_safe = bool(safe and selected_correct and distractor_open)
    return {"method": f"semigcn_{stage}", "status": "complete", "decision": "accept",
            "accepted": True, "selected_loop_id": int(pred_map["selected_loop_id"]),
            "selected_correct": selected_correct, "distractor_open": distractor_open,
            "selected_closed": strict, "target_closed": loop_state(full_f, target)["closed"],
            "no_new_intersection": no_intersection,
            "correct_safe": correct_safe, "wrong_accepted": bool(not correct_safe),
            "audit": {k: audit.get(k) for k in ("status", "introduced_intersection", "strict_selected_loop_closure",
                                                  "after_degenerate_faces", "introduced_intersection_nondegenerate")},
            "geometry": geometry, "learning_stage_startup_inclusive_seconds": batch.get("elapsed_seconds"),
            "fit_refine_seconds": status.get("elapsed_seconds"), "iterations": status.get("iterations"),
            "output_manifest": str(destination / "stage_output_manifest.json"),
            "source_commit": status.get("source_commit"),
            "device_environment": status.get("device_environment"),
            "global_diagnostic": {"all_cap_faces": int(len(pred_map["all_patch_faces"])),
                                  "selected_cap_faces": int(len(patch_f))}}


def ci(k, n):
    if not n:
        return [None, None]
    x = binomtest(int(k), int(n)).proportion_ci(confidence_level=.95, method="exact")
    return [float(x.low), float(x.high)]


def bootstrap(x, y, seed=20261011, draws=20000):
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    differences = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    ix = rng.integers(0, len(x), (draws, len(x)))
    vals = np.mean(differences[ix], axis=1)
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def meshfix_rows(cases):
    data = load(MESHFIX_PATH, {"rows": []})
    by = {x.get("sample_id"): x for x in data.get("rows", [])}
    rows = []
    for case in cases:
        x = by.get(case["sample_id"])
        # The development case000 timing was only an interface check alongside
        # live GPU jobs. Never substitute it into the frozen budget comparisons.
        timed = (load(BENCH / "runtime" / f"case{int(case['index']):03d}" / "meshfix" / "result.json", {})
                 if BENCH.name != "formal_coseg_external_benchmark" else {})
        if timed.get("status") == "complete":
            from scripts.formal_coseg_external_benchmark import mapped_loop
            vin, fin = load_input(case)
            with np.load(timed["output_path"], allow_pickle=False) as z:
                vout, fout = z["vertices"], z["faces"]
            tolerance = max(1e-5 * float(np.linalg.norm(np.ptp(vin, axis=0))), 1e-8)
            x = {"sample_id": case["sample_id"], "output_path": timed["output_path"],
                 "target": mapped_loop(vin, case["target_loop_vertices"], vout, fout, tolerance),
                 "distractor": mapped_loop(vin, case["distractor_loop_vertices"], vout, fout, tolerance),
                 "global_watertight": bool(len(fout) and all(n == 2 for n in edge_counts(fout).values())),
                 "runtime_seconds": timed["input_to_verified_output_wall_seconds"]}
        elif timed.get("status") == "failed":
            x = {"sample_id": case["sample_id"], "error": timed.get("error"),
                 "runtime_seconds": timed["input_to_verified_output_wall_seconds"]}
        if not x:
            rows.append({"method": "meshfix", "status": "pending", "correct_safe": False,
                         "wrong_accepted": False, "selected_correct": False, "distractor_open": False})
            continue
        target, distractor = x.get("target", {}), x.get("distractor", {})
        accepted = not x.get("failed") and not x.get("error")
        structural = bool(accepted and target.get("closed") and distractor.get("open"))
        verification = {"status": "not_needed_for_failed_structural_endpoint", "introduced_intersection": None}
        if structural and x.get("output_path") and Path(x["output_path"]).exists():
            from scripts.scan_edit_benchmark import meshfix_geometry_verification
            before_v, before_f = load_input(case)
            with np.load(x["output_path"], allow_pickle=False) as z:
                verification = meshfix_geometry_verification(before_v, before_f, z["vertices"], z["faces"])
        strict = structural and all(int(n) == 2 for n in target.get("edge_counts", [])) and bool(target.get("edge_counts"))
        correct = bool(strict and verification.get("introduced_intersection") is False and
                       verification.get("geometry_no_new_topology_defect") is True)
        rows.append({"method": "meshfix", "status": "complete" if accepted else "failed", "decision": "accept" if accepted else "null",
                     "accepted": accepted, "selected_correct": structural,
                     "target_closed": bool(target.get("closed")),
                     "distractor_open": bool(distractor.get("open")),
                     "correct_safe": correct, "wrong_accepted": bool(accepted and not correct),
                     "audit": verification,
                     "global_watertight": x.get("global_watertight"),
                     "runtime_seconds": x.get("runtime_seconds"),
                     "geometry": {"status": "unavailable", "global_watertight": x.get("global_watertight")}})
    return rows


def method_summary(rows):
    n = len(rows)
    accepted = sum(bool(x.get("accepted", x.get("decision") == "accept")) for x in rows)
    correct = sum(bool(x.get("correct_safe")) for x in rows)
    wrong = sum(bool(x.get("wrong_accepted")) for x in rows)
    complete = sum(x.get("status") == "complete" for x in rows)
    failed = sum(x.get("status") == "failed" for x in rows)
    evaluated = complete + failed
    selected_correct = sum(bool(x.get("selected_correct")) for x in rows)
    distractor_open = sum(bool(x.get("distractor_open")) for x in rows)
    # Geometry quality is reported only for returned/accepted repairs.  A
    # rejected CGAL candidate remains in the raw row for auditability but is
    # not silently counted as an accepted output.
    geom = [x.get("geometry", {}) for x in rows
            if x.get("status") == "complete" and x.get("accepted", True) and
            x.get("geometry", {}).get("status") == "success"]
    def vals(path):
        out = []
        for g in geom:
            v = g
            for part in path.split("."):
                v = v.get(part) if isinstance(v, dict) else None
            if isinstance(v, (int, float)) and np.isfinite(v):
                out.append(float(v))
        return out
    def stats(v):
        return {"n": len(v), "mean": float(np.mean(v)) if v else None,
                "median": float(np.median(v)) if v else None,
                "p05": float(np.percentile(v, 5)) if v else None,
                "p95": float(np.percentile(v, 95)) if v else None,
                "max": float(np.max(v)) if v else None}
    inf = [x.get("inference_wall_seconds") for x in rows if isinstance(x.get("inference_wall_seconds"), (int, float))]
    cand = [x.get("candidate_seconds") for x in rows if isinstance(x.get("candidate_seconds"), (int, float))]
    run = [x.get("runtime_seconds") for x in rows if isinstance(x.get("runtime_seconds"), (int, float))]
    fit = [x.get("fit_refine_seconds") for x in rows if isinstance(x.get("fit_refine_seconds"), (int, float))]
    startup = [x.get("learning_stage_startup_inclusive_seconds") for x in rows
               if isinstance(x.get("learning_stage_startup_inclusive_seconds"), (int, float))]
    return {"n": n, "complete_n": complete, "accepted_n": accepted, "correct_safe_n": correct,
            "selected_correct_n": selected_correct, "distractor_open_n": distractor_open,
            "selected_closed_n": sum(bool(x.get("selected_closed")) for x in rows),
            "target_closed_n": sum(bool(x.get("target_closed")) for x in rows),
            "no_new_intersection_n": sum(bool(x.get("no_new_intersection")) for x in rows),
            "introduced_intersection_n": sum(x.get("audit", {}).get("introduced_intersection") is True for x in rows),
            "intersection_unknown_n": sum(x.get("status") == "complete" and x.get("audit", {}).get("introduced_intersection") is None for x in rows),
            "wrong_accepted_n": wrong, "failed_n": failed, "evaluated_n": evaluated, "pending_n": n - evaluated,
            "rate_denominator": evaluated, "formal_full_denominator": evaluated == n,
            "correct_safe_rate": float(correct / evaluated) if evaluated else None, "correct_safe_rate_ci95": ci(correct, evaluated),
            "accepted_precision": float(correct / accepted) if accepted else None,
            "accepted_precision_ci95": ci(correct, accepted), "risk_among_accepted": float(wrong / accepted) if accepted else None,
            "signed_seam_normal_mean": stats(vals("signed_seam_normal_mean")),
            "seam_normal_absolute_mean": stats(vals("seam_normal_absolute_mean")),
            "seam_normal_absolute_p95": stats(vals("seam_normal_absolute_p95")),
            "triangle_quality_mean": stats(vals("triangle_quality.mean")),
            "triangle_quality_p05": stats(vals("triangle_quality.p05")),
            "triangle_quality_min": stats(vals("triangle_quality.min")),
            "retained_displacement_mean_over_edge": stats(vals("retained_displacement_mean_over_edge")),
            "retained_displacement_rms_over_edge": stats(vals("retained_displacement_rms_over_edge")),
            "retained_displacement_p95_over_edge": stats(vals("retained_displacement_p95_over_edge")),
            "retained_displacement_max_over_edge": stats(vals("retained_displacement_max_over_edge")),
            "changed_surface_area_ratio": stats(vals("changed_surface_area_ratio")),
            "outside_two_ring_fixed_rate": float(np.mean([bool(g.get("retained_outside_two_ring_bitwise_fixed")) for g in geom])) if geom else None,
            "inference_wall_seconds": stats([float(x) for x in inf]),
            "fit_refine_seconds": stats([float(x) for x in fit]),
            "learning_stage_startup_inclusive_seconds": stats([float(x) for x in startup]),
            "candidate_seconds": stats([float(x) for x in cand]),
            "runtime_seconds": stats([float(x) for x in run])}


def paired_summary(rows, left: str, right: str) -> dict:
    """Paired source-distinct comparisons; pending cases are not failures."""
    terminal = [(r[left], r[right]) for r in rows
                if r[left].get("status") in ("complete", "failed") and
                r[right].get("status") in ("complete", "failed")]
    common = [(a, b) for a, b in terminal
              if a.get("correct_safe") and b.get("correct_safe") and
              a.get("selected_loop_id") == b.get("selected_loop_id")]
    def metric(path, geometry=True):
        x, y = [], []
        for a, b in common:
            va, vb = (a.get("geometry", {}), b.get("geometry", {})) if geometry else (a, b)
            for part in path.split("."):
                va = va.get(part) if isinstance(va, dict) else None
                vb = vb.get(part) if isinstance(vb, dict) else None
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and np.isfinite(va) and np.isfinite(vb):
                x.append(float(va)); y.append(float(vb))
        mean = float(np.mean(np.asarray(x) - np.asarray(y))) if x else None
        result = {"n": len(x), "mean_difference": mean,
                  f"{left}_minus_{right}_mean": mean, "bootstrap_ci95": bootstrap(x, y)}
        return result
    t = np.asarray([int(a.get("correct_safe", False)) for a, _ in terminal])
    c = np.asarray([int(b.get("correct_safe", False)) for _, b in terminal])
    da, db = int(np.sum((t == 1) & (c == 0))), int(np.sum((t == 0) & (c == 1)))
    endpoint_ci = bootstrap(t, c)
    def terminal_quantity(key):
        values = [(a.get(key), b.get(key)) for a, b in terminal]
        values = [(float(a), float(b)) for a, b in values
                  if isinstance(a, (int, float)) and isinstance(b, (int, float)) and np.isfinite(a) and np.isfinite(b)]
        x, y = ([a for a, _ in values], [b for _, b in values])
        return {"n": len(values), "mean_difference": float(np.mean(np.asarray(x) - np.asarray(y))) if values else None,
                "bootstrap_ci95": bootstrap(x, y)}
    outside_left = np.asarray([int(a.get("geometry", {}).get("retained_outside_two_ring_bitwise_fixed", False))
                               for a, _ in terminal])
    outside_right = np.asarray([int(b.get("geometry", {}).get("retained_outside_two_ring_bitwise_fixed", False))
                                for _, b in terminal])
    outside_ci = bootstrap(outside_left, outside_right)
    return {"left": left, "right": right, "n_terminal_pairs": len(terminal),
            "pending_pair_n": len(rows) - len(terminal),
            "n_common_correct_safe_same_target": len(common),
            "seam_normal_absolute_mean": metric("seam_normal_absolute_mean"),
            "seam_normal_absolute_p95": metric("seam_normal_absolute_p95"),
            "signed_seam_normal": metric("signed_seam_normal_mean"),
            "triangle_quality_mean": metric("triangle_quality.mean"),
            "triangle_quality_p05": metric("triangle_quality.p05"),
            "retained_displacement_rms_over_edge": metric("retained_displacement_rms_over_edge"),
            "retained_displacement_max_over_edge": metric("retained_displacement_max_over_edge"),
            "changed_surface_area_ratio": metric("changed_surface_area_ratio"),
            "inference_wall_seconds": metric("inference_wall_seconds", geometry=False),
            "fit_refine_seconds_terminal_pairs": terminal_quantity("fit_refine_seconds"),
            "outside_two_ring_fixed_endpoint": {
                "n": len(terminal), "difference_pp": float(100 * np.mean(outside_left - outside_right)) if len(terminal) else None,
                "bootstrap_ci95_pp": [100 * x if x is not None else None for x in outside_ci]},
            "correct_safe_endpoint": {f"{left}_only": da, f"{right}_only": db,
                                      "difference_pp": float(100 * np.mean(t - c)) if len(t) else None,
                                      "bootstrap_ci95_pp": [100 * x if x is not None else None for x in endpoint_ci],
                                      "mcnemar_exact_two_sided_p": float(binomtest(min(da, db), da + db).pvalue) if da + db else 1.0}}


def evaluate(semigcn_stage: str, semigcn_root: Path | None = None,
             case_indices: list[int] | None = None, output_stem: str = "unified_evaluation",
             baseline_stage: str | None = None, semigcn_status_file: Path | None = None):
    protocol = load(PROTOCOL_PATH)
    cases = [score_annotations(c) for c in protocol["cases"]]
    if case_indices is not None:
        wanted = set(map(int, case_indices))
        if len(wanted) != len(case_indices) or not wanted.issubset({int(c["index"]) for c in cases}):
            raise ValueError("case indices must be unique members of the original protocol")
        cases = [c for c in cases if int(c["index"]) in wanted]
    if Path(output_stem).name != output_stem:
        raise ValueError("output stem must be a simple file stem")
    rows = []
    for case in cases:
        pred = find_prediction(case)
        t = unified_method_row(case, pred, "tcmr")
        c = unified_method_row(case, pred, "cgal")
        s = semigcn_row(case, pred, semigcn_stage, semigcn_root, semigcn_status_file)
        rows.append({"index": case["index"], "sample_id": case["sample_id"], "category": case["category"],
                     "tcmr": t, "cgal": c, "semigcn": s})
        for field in ("source_model_id", "cue_fraction", "cue_stratum", "perimeter_stratum", "stratum", "evaluation_cohort"):
            if field in case:
                rows[-1][field] = case[field]
        for name in (("tcmr", "cgal_pool") if BENCH.name != "formal_coseg_external_benchmark" else ()):
            measured = load(BENCH / "runtime" / f"case{int(case['index']):03d}" / name / "result.json", {})
            if measured.get("status") == "complete":
                fresh_pred = load(Path(measured["prediction_path"]))
                rows[-1][name] = unified_method_row(case, fresh_pred, "tcmr")
                rows[-1][name]["method"] = name
                rows[-1][name]["input_to_verified_output_wall_seconds"] = measured["input_to_verified_output_wall_seconds"]
            elif measured.get("status") == "failed":
                rows[-1][name] = {"method": name, "status": "failed", "correct_safe": False, "wrong_accepted": False,
                                 "error": measured.get("error")}
        if "cgal_pool" not in rows[-1] and "cgal_library_decision" in pred:
            rows[-1]["cgal_pool"] = selected_decision_row(case, pred, pred["cgal_library_decision"])
        for name, decision in pred.get("ablations", {}).items():
            rows[-1][f"ablation_{name}"] = selected_decision_row(case, pred, decision)
        if baseline_stage:
            rows[-1]["semigcn_baseline"] = semigcn_row(case, pred, baseline_stage, semigcn_root)
        print(json.dumps({"scored": len(rows), "n": len(cases), "sample_id": case["sample_id"],
                          "semigcn_status": s["status"], "semigcn_correct_safe": s.get("correct_safe")}), flush=True)
    methods = {"tcmr": [r["tcmr"] for r in rows], "cgal": [r["cgal"] for r in rows],
               "meshfix": meshfix_rows(cases), "semigcn": [r["semigcn"] for r in rows]}
    extra_methods = sorted({k for r in rows for k in r if k == "cgal_pool" or k.startswith("ablation_")})
    for name in extra_methods:
        for row in rows:
            row.setdefault(name, {"status": "pending", "correct_safe": False})
        methods[name] = [r[name] for r in rows]
    if baseline_stage:
        methods["semigcn_baseline"] = [r["semigcn_baseline"] for r in rows]
    summary = {k: method_summary(v) for k, v in methods.items()}
    summary["meshfix"].update({
        "target_closed_n": int(sum(bool(x.get("target_closed")) for x in methods["meshfix"])),
        "distractor_open_n": int(sum(bool(x.get("distractor_open")) for x in methods["meshfix"])),
        "global_watertight_n": int(sum(bool(x.get("global_watertight")) for x in methods["meshfix"])),
    })
    pairs = [("tcmr", "cgal"), ("tcmr", "semigcn"), ("cgal", "semigcn")]
    pairs.extend(("tcmr", name) for name in extra_methods)
    if baseline_stage:
        pairs.append(("semigcn_baseline", "semigcn"))
    paired_comparisons = {f"{a}_vs_{b}": paired_summary(rows, a, b) for a, b in pairs}
    paired = paired_comparisons["tcmr_vs_cgal"]
    pending = sum(any(methods[k][i].get("status") == "pending" for k in methods) for i in range(len(rows)))
    replay = BENCH.name in ("formal_coseg_external_benchmark", "coseg_external_expansion",
                           "formal_coseg_external_expansion")
    report = {"schema": "unified_external_benchmark_evaluation", "complete": pending == 0,
              "n": len(rows), "pending_n": pending,
              "protocol_sha256": sha(PROTOCOL_PATH), "semigcn_stage": semigcn_stage,
              "baseline_stage": baseline_stage, "original_case_indices": [int(c["index"]) for c in cases],
              "semigcn_root": str(semigcn_root) if semigcn_root else None,
              "evidence_role": "development replay after observed failures" if replay else "see frozen source protocol",
              "metric_policy": {"unknown_intersection_not_safe": True, "target_only_requires_distractor_open": True,
                                "pending_cases": "reported as pending, excluded from temporary rate and paired denominators; never failures",
                                "main_seam_endpoint": "mean absolute per-edge oriented atan2 dihedral, radians; signed mean is diagnostic only",
                                "displacement_normalization": "median of all input retained-face edge lengths; same denominator across methods",
                                "outside_support": "two frozen-frontier vertex rings from selected loop",
                                "geometry_pairing": "common correct-safe same-target cases only",
                                "changed_surface_area_ratio": "patch area plus moved retained-face area divided by input retained surface area",
                                "signed_seam_normal": "oriented atan2 dihedral; no absolute normal dot",
                                "inference_wall_seconds": "TCMR full candidate-pool prediction time only",
                                "candidate_seconds": "CGAL repair plus verification; not a full learned selector runtime",
                                "fit_refine_seconds": "SeMIGCN per-mesh fitting plus refinement author-main stage; excludes common CGAL initialization and smoothing; GPU with three concurrent workers",
                                "learning_stage_startup_inclusive_seconds": "batch child launch through completed output; excludes earlier common CGAL initializer preparation; missing manual-case timers remain unavailable"},
              "summary": summary, "paired": paired, "paired_comparisons": paired_comparisons, "rows": rows}
    if (BENCH.name != "formal_coseg_external_benchmark" and (BENCH / "runtime").exists()) or any(r["semigcn"].get("output_manifest") and
            load(Path(r["semigcn"]["output_manifest"]), {}).get("runtime_measurement") for r in rows):
        report["unified_runtime"] = runtime_summary(rows, semigcn_stage, semigcn_root)
    strata = {}
    for key in ("category", "cue_fraction", "cue_stratum", "perimeter_stratum", "stratum", "evaluation_cohort"):
        values = sorted({str(r[key]) for r in rows if key in r})
        if values:
            strata[key] = {value: {"n": sum(str(r.get(key)) == value for r in rows),
                "summary": {name: method_summary([methods[name][i] for i, r in enumerate(rows) if str(r.get(key)) == value]) for name in methods}}
                for value in values}
    report["prespecified_strata"] = strata
    if any(r.get("evaluation_cohort") == "primary" for r in rows):
        primary = [r for r in rows if r.get("evaluation_cohort") == "primary"]
        report["primary_cohort"] = {"n": len(primary), "definition": "Prespecified cue fractions <=0.4; severe0.6 stress cases separately reported",
            "paired_comparisons": {f"{a}_vs_{b}": paired_summary(primary, a, b) for a, b in pairs}}
    out = BENCH / f"{output_stem}.json"
    dump(out, report)
    lines = [f"# Unified {len(rows)}-case external benchmark", "", f"Complete: `{report['complete']}`; pending SeMIGCN cases: `{report['pending_n']}`.", "",
             "Evidence role: " + report["evidence_role"] + ". The cue-geometry gate was designed after inspecting the 79/104 development failures; only the separately frozen 50-case holdout is confirmatory for that gate.", "",
             "|Method|Complete|Accepted|Correct-safe|Wrong|Pending|Rate|95% exact CI|Risk among accepted|Median inference s|Median candidate s|Median runtime s|", "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for name, x in summary.items():
        def med(key):
            v = x[key].get("median") if isinstance(x.get(key), dict) else None
            return "---" if v is None else f"{v:.3f}"
        rate = "---" if x['correct_safe_rate'] is None else f"{x['correct_safe_rate']:.1%}"
        interval = "---" if x['correct_safe_rate_ci95'][0] is None else f"[{x['correct_safe_rate_ci95'][0]:.1%}, {x['correct_safe_rate_ci95'][1]:.1%}]"
        lines.append(f"|{name}|{x['complete_n']}|{x['accepted_n']}|{x['correct_safe_n']}|{x['wrong_accepted_n']}|{x['pending_n']}|{rate}|{interval}|{x['risk_among_accepted'] if x['risk_among_accepted'] is not None else '---'}|{med('inference_wall_seconds')}|{med('candidate_seconds')}|{med('runtime_seconds')}|")
    p = paired["correct_safe_endpoint"]
    lines += ["", f"TCMR minus CGAL correct-safe difference: {p['difference_pp']:.2f} pp; bootstrap 95% CI [{p['bootstrap_ci95_pp'][0]:.2f}, {p['bootstrap_ci95_pp'][1]:.2f}] pp; exact McNemar p={p['mcnemar_exact_two_sided_p']:.3g}.", "", f"Geometry pairing uses {paired['n_common_correct_safe_same_target']} common correct-safe same-target cases.", "", "Learning output is target-only adapted from cue-selected cap membership; global all-cap output is retained only as a diagnostic."]
    lines += ["", "## Returned-output geometry and fitting cost", "",
              "|Method|Mean absolute seam rad|Mean per-case seam P95 rad|Mean triangle quality|Retained RMS / median input edge|Outside two-ring fixed|Median fit/refine s|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, x in summary.items():
        def value(key, field="mean"):
            v = x.get(key, {}).get(field)
            return "---" if v is None else f"{v:.6f}"
        outside = x.get("outside_two_ring_fixed_rate")
        lines.append(f"|{name}|{value('seam_normal_absolute_mean')}|{value('seam_normal_absolute_p95')}|{value('triangle_quality_mean')}|{value('retained_displacement_rms_over_edge')}|{'---' if outside is None else f'{outside:.1%}'}|{value('fit_refine_seconds', 'median')}|")
    lines += ["", "Geometry rows describe accepted returned outputs, including unsafe outputs; each mesh has equal weight, and the P95 column averages within-mesh absolute seam P95 values. Comparisons below use common correct-safe same-target cases. Signed averages are diagnostics and cannot be called seam error. Two-ring support uses a frozen frontier at each step.", "",
              "## Paired comparisons", "", "|Left minus right|Terminal pairs|Common safe pairs|Correct-safe difference pp [95% CI]|Absolute seam difference rad [95% CI]|Quality difference [95% CI]|",
              "|---|---:|---:|---|---|---|"]
    def pair_value(x, endpoint=False):
        mean = x.get("difference_pp" if endpoint else "mean_difference")
        interval = x.get("bootstrap_ci95_pp" if endpoint else "bootstrap_ci95")
        return "---" if mean is None else f"{mean:.6f} [{interval[0]:.6f}, {interval[1]:.6f}]"
    for name, x in paired_comparisons.items():
        lines.append(f"|{x['left']} - {x['right']}|{x['n_terminal_pairs']}|{x['n_common_correct_safe_same_target']}|{pair_value(x['correct_safe_endpoint'], True)}|{pair_value(x['seam_normal_absolute_mean'])}|{pair_value(x['triangle_quality_mean'])}|")
    lines += ["", "Unknown native intersection status is unsafe. Pending outputs are unfinished observations, never failures. Timing scopes differ: SeMIGCN uses per-mesh fitting + refinement author-main stage, excluding common CGAL initialization and smoothing; batch startup-inclusive timers are separate and missing manual timers remain unavailable. CGAL candidate time includes repair + verification; TCMR wall time includes its complete candidate pool. End-to-end speedups are not inferred from these columns."]
    if baseline_stage:
        lines += ["", f"Budget comparison: semigcn_baseline = {baseline_stage}; semigcn = {semigcn_stage}. Cases retain their original frozen protocol indices; subset membership was fixed without reading outcomes."]
    if "unified_runtime" in report:
        runtime = report["unified_runtime"]
        lines += ["", "## Fresh end-to-end runtime", "", runtime["scope"], "", runtime["hardware_note"], "",
                  "|Method|Completed timing n|Failed n|Mean s|Median s|P95 s|", "|---|---:|---:|---:|---:|---:|"]
        for name, measured in runtime["summary"].items():
            stats = measured["completed_wall_seconds"]
            fmt = lambda key: "---" if stats.get(key) is None else f"{stats[key]:.3f}"
            lines.append(f"|{name}|{stats['n']}|{measured['failed_n']}|{fmt('mean')}|{fmt('median')}|{fmt('p95')}|")
        lines += ["", "Paired source-level mean-time differences and 95% bootstrap intervals, failure durations, and every raw timer are saved in the JSON. Missing timings are not replaced with historical candidate or fitting durations."]
    if "evaluation_cohort" in strata:
        lines += ["", "## Prespecified primary and severe-cue cohorts", "",
                  "The pooled table above describes all fixed inputs. The primary cohort excludes the separately prespecified cue0.6 ambiguity stress condition; no cases are removed based on method results.", "",
                  "|Cohort|N|Method|Correct-safe|Wrong accepted|", "|---|---:|---|---:|---:|"]
        for group, values in strata["evaluation_cohort"].items():
            for name, stats in values["summary"].items():
                lines.append(f"|{group}|{values['n']}|{name}|{stats['correct_safe_n']}|{stats['wrong_accepted_n']}|")
        lines += ["", "Primary-cohort paired 95% confidence intervals and category/interface-size/cue-level breakdowns are saved in the JSON."]
    (BENCH / f"{output_stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"complete": report["complete"], "pending_n": report["pending_n"], "summary": summary, "paired": paired}, ensure_ascii=False, indent=2))
    return report


STABILITY_GEOMETRY_PATHS = {
    "seam_normal_absolute_mean": "seam_normal_absolute_mean",
    "seam_normal_absolute_p95": "seam_normal_absolute_p95",
    "triangle_quality_mean": "triangle_quality.mean",
    "triangle_quality_p05": "triangle_quality.p05",
    "triangle_quality_min": "triangle_quality.min",
    "retained_displacement_rms_over_edge": "retained_displacement_rms_over_edge",
    "retained_displacement_max_over_edge": "retained_displacement_max_over_edge",
    "changed_surface_area_ratio": "changed_surface_area_ratio",
}


def _stability_value(row: dict, path: str):
    value = row.get("geometry", {})
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _stability_distribution(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(np.mean(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "min": float(np.min(values)) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None}


def _stability_safety_state(row: dict) -> str:
    if row.get("status") != "complete":
        return row.get("status", "pending")
    if not row.get("accepted", row.get("decision") == "accept"):
        return "null"
    if row.get("audit", {}).get("introduced_intersection") is None:
        return "intersection_unknown"
    if row.get("audit", {}).get("introduced_intersection") is True:
        return "introduced_intersection"
    if not row.get("selected_correct"):
        return "wrong_target"
    if not row.get("distractor_open"):
        return "distractor_closed"
    if not row.get("selected_closed"):
        return "selected_loop_not_closed"
    if row.get("geometry", {}).get("geometry_no_new_topology_defect") is False:
        return "introduced_topology_defect"
    return "correct_safe" if row.get("correct_safe") else "other_unsafe"


def _stability_statistics(rows: list[dict], seeds: list[int]) -> dict:
    """Treat seeds as repeated measurements of each source mesh, never new meshes."""
    keys = [str(seed) for seed in seeds]
    terminal = [r for r in rows if all(r["seeds"][key].get("status") in ("complete", "failed")
                                     for key in keys)]
    paired_seed_keys = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    source_rows = []
    for row in terminal:
        outputs = [row["seeds"][key] for key in keys]
        correct = [int(x.get("correct_safe", False)) for x in outputs]
        states = [_stability_safety_state(x) for x in outputs]
        source = {"index": int(row["index"]), "sample_id": row["sample_id"],
                  "seed_correct_safe": dict(zip(keys, correct)),
                  "mean_seed_correct_safe": float(np.mean(correct)),
                  "mean_seed_returned_outside_two_ring_fixed": float(np.mean([
                      x.get("status") == "complete" and x.get("geometry", {}).get("retained_outside_two_ring_bitwise_fixed", False)
                      for x in outputs])),
                  "safety_states": dict(zip(keys, states)),
                  "correct_safe_pair_disagreement": float(np.mean([
                      row["seeds"][a].get("correct_safe", False) != row["seeds"][b].get("correct_safe", False)
                      for a, b in paired_seed_keys])),
                  "safety_state_pair_disagreement": float(np.mean([
                      _stability_safety_state(row["seeds"][a]) != _stability_safety_state(row["seeds"][b])
                      for a, b in paired_seed_keys])), "geometry": {}}
        complete = all(x.get("status") == "complete" for x in outputs)
        same_target = complete and len({x.get("selected_loop_id") for x in outputs}) == 1
        source["selected_loop_adapter_agreement"] = same_target if complete else None
        source["all_seeds_correct_safe_same_target"] = bool(all(correct) and same_target)
        for name, path in STABILITY_GEOMETRY_PATHS.items():
            values = [_stability_value(x, path) for x in outputs]
            if complete and same_target and all(v is not None for v in values):
                source["geometry"][name] = {"seed_values": dict(zip(keys, values)),
                                            "seed_mean": float(np.mean(values)),
                                            "seed_sd": float(np.std(values, ddof=1)),
                                            "seed_range": float(np.max(values) - np.min(values))}
        source_rows.append(source)

    def source_estimate(values):
        return {"source_n": len(values), "mean": float(np.mean(values)) if values else None,
                "source_bootstrap_ci95": bootstrap(values, np.zeros(len(values))) if values else [None, None]}

    correct_values = [r["mean_seed_correct_safe"] for r in source_rows]
    correct_endpoint = source_estimate(correct_values)
    correct_endpoint.update({"always_safe_source_n": sum(r["mean_seed_correct_safe"] == 1 for r in source_rows),
                             "sometimes_unsafe_source_n": sum(0 < r["mean_seed_correct_safe"] < 1 for r in source_rows),
                             "never_safe_source_n": sum(r["mean_seed_correct_safe"] == 0 for r in source_rows)})
    common_safe = [r for r in source_rows if r["all_seeds_correct_safe_same_target"]]
    geometry = {}
    for name in STABILITY_GEOMETRY_PATHS:
        all_rows = [r["geometry"][name] for r in source_rows if name in r["geometry"]]
        safe_rows = [r["geometry"][name] for r in common_safe if name in r["geometry"]]
        geometry[name] = {
            "all_three_returned_same_target": {
                "seed_mean": _stability_distribution([r["seed_mean"] for r in all_rows]),
                "within_source_seed_sd": _stability_distribution([r["seed_sd"] for r in all_rows]),
                "within_source_seed_range": _stability_distribution([r["seed_range"] for r in all_rows])},
            "all_three_correct_safe_same_target": {
                "seed_mean": source_estimate([r["seed_mean"] for r in safe_rows]),
                "within_source_seed_sd": _stability_distribution([r["seed_sd"] for r in safe_rows]),
                "within_source_seed_range": _stability_distribution([r["seed_range"] for r in safe_rows])}}

    paired_sources = [(r, s) for r, s in zip(terminal, source_rows)
                      if r["tcmr"].get("status") in ("complete", "failed")]
    t = [int(r["tcmr"].get("correct_safe", False)) for r, _ in paired_sources]
    s = [x["mean_seed_correct_safe"] for _, x in paired_sources]
    overall_pair = {"source_n": len(t),
                    "correct_safe_difference_pp": float(100 * np.mean(np.asarray(t) - np.asarray(s))) if t else None,
                    "source_bootstrap_ci95_pp": [None if v is None else 100 * v for v in bootstrap(t, s)],
                    "geometry": {}}
    for name, path in STABILITY_GEOMETRY_PATHS.items():
        pairs = [(r, x) for r, x in paired_sources if x["all_seeds_correct_safe_same_target"]
                 and r["tcmr"].get("correct_safe")
                 and r["tcmr"].get("selected_loop_id") == r["seeds"][keys[0]].get("selected_loop_id")
                 and name in x["geometry"] and _stability_value(r["tcmr"], path) is not None]
        a = [_stability_value(r["tcmr"], path) for r, _ in pairs]
        b = [x["geometry"][name]["seed_mean"] for _, x in pairs]
        overall_pair["geometry"][name] = {"common_safe_source_n": len(a),
            "tcmr_minus_seed_mean": float(np.mean(np.asarray(a) - np.asarray(b))) if a else None,
            "source_bootstrap_ci95": bootstrap(a, b)}
    pairwise = {}
    for a, b in paired_seed_keys:
        pair_rows = [{"left": r["seeds"][a], "right": r["seeds"][b]} for r in rows]
        pairwise[f"seed{a}_vs_seed{b}"] = paired_summary(pair_rows, "left", "right")
    adapter = [r["selected_loop_adapter_agreement"] for r in source_rows
               if r["selected_loop_adapter_agreement"] is not None]
    return {"independent_source_n": len(rows), "terminal_all_seed_source_n": len(terminal),
            "pending_source_n": len(rows) - len(terminal), "correct_safe": correct_endpoint,
            "correct_safe_pair_disagreement": source_estimate([r["correct_safe_pair_disagreement"] for r in source_rows]),
            "safety_state_pair_disagreement": source_estimate([r["safety_state_pair_disagreement"] for r in source_rows]),
            "returned_outside_two_ring_fixed": source_estimate([
                r["mean_seed_returned_outside_two_ring_fixed"] for r in source_rows]),
            "selected_loop_adapter_control": {"n_all_three_returned": len(adapter),
                "all_seed_agree_n": sum(adapter),
                "interpretation": "Cue-nearest cap selection is fixed by the adapter; this is an input/mapping control, not learned target-selection stability."},
            "all_three_correct_safe_same_target_source_n": len(common_safe),
            "geometry": geometry, "tcmr_vs_seed_mean": overall_pair,
            "seed_pairwise": pairwise, "source_rows": source_rows}


def evaluate_stability(protocol_path: Path, semigcn_root: Path | None = None,
                       output_stem: str = "stability_evaluation") -> dict:
    protocol_path = protocol_path.resolve()
    protocol = load(protocol_path)
    if not protocol:
        raise FileNotFoundError(protocol_path)
    if Path(output_stem).name != output_stem:
        raise ValueError("output stem must be a simple file stem")
    base = semigcn_root or protocol_path.parent
    source_protocol = load(PROTOCOL_PATH)
    expected_sha = protocol.get("source_protocol_sha256")
    if expected_sha and expected_sha != sha(PROTOCOL_PATH):
        raise RuntimeError("source protocol changed after the stability cohort was frozen")
    by_index = {int(c["index"]): c for c in source_protocol["cases"]}
    frozen_cases = protocol["cases"]
    indices = [int(c["index"]) for c in frozen_cases]
    if len(indices) != len(set(indices)) or not set(indices).issubset(by_index):
        raise ValueError("stability cases must be unique original protocol indices")
    cases = [by_index[index] for index in indices]
    for frozen, case in zip(frozen_cases, cases):
        for key in ("sample_id", "category", "input_sha256"):
            if key in frozen and frozen[key] != case[key]:
                raise RuntimeError(f"frozen stability case {case['index']} changed {key}")
    if len({c["source_model_id"] for c in cases}) != len(cases):
        raise ValueError("stability source meshes must be distinct")
    seed_entries = protocol["seeds"]
    seeds = [int(entry["seed"]) for entry in seed_entries]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("stability protocol requires three distinct seeds")
    iterations = int(protocol.get("iterations", protocol.get("fitting_iterations", 300)))
    baseline_seed = int(protocol.get("baseline_seed", 314))
    baseline_stage = protocol.get("baseline_stage", "gpu100")
    budget_status_path = base / "budget_status.json"
    budget_status = load(budget_status_path, {})
    budget_times = {r.get("case"): r for r in budget_status.get("rows", [])}
    rows = []
    for case in cases:
        pred = find_prediction(case)
        row = {"index": int(case["index"]), "sample_id": case["sample_id"],
               "source_model_id": case["source_model_id"], "category": case["category"],
               "tcmr": unified_method_row(case, pred, "tcmr"), "seeds": {}}
        for entry in seed_entries:
            seed, stage = int(entry["seed"]), entry["stage"]
            status_path = Path(entry.get("status_file", base / f"stability_seed{seed}_status.json"))
            if not status_path.is_absolute():
                status_path = ROOT / status_path
            scored = semigcn_row(case, pred, stage, base, status_path)
            scored["requested_seed"] = seed
            scored["stage"] = stage
            stage_status = load(status_path, {})
            if scored.get("status") == "failed":
                timer = next((x for x in stage_status.get("rows", [])
                              if x.get("case") == f"case{int(case['index']):03d}"
                              or x.get("sample_id") == case["sample_id"]), {})
                if (stage_status.get("stage") == stage and timer.get("status") == "failed"):
                    scored["learning_stage_startup_inclusive_seconds"] = timer.get("elapsed_seconds")
            if scored.get("status") == "complete":
                manifest = load(Path(scored["output_manifest"]), {})
                recorded_seed = int(manifest.get("seed", 314))
                if manifest.get("iterations") != iterations or (
                        recorded_seed != seed):
                    raise RuntimeError(f"wrong fitting budget or seed for {case['sample_id']}/{stage}")
                scored["recorded_seed"] = recorded_seed
                scored["seed_provenance"] = ("explicit stage manifest" if "seed" in manifest else
                    "historical wrapper fixed seed314; stage predates explicit seed metadata")
                timer = budget_times.get(f"case{int(case['index']):03d}", {})
                if (seed == baseline_seed and scored.get("learning_stage_startup_inclusive_seconds") is None
                        and budget_status.get("stage") == stage
                        and budget_status.get("iterations") == iterations
                        and timer.get("status") == "completed"):
                    scored["learning_stage_startup_inclusive_seconds"] = timer.get("elapsed_seconds")
                    scored["startup_timer_origin"] = str(budget_status_path)
            row["seeds"][str(seed)] = scored
        row["baseline100"] = semigcn_row(case, pred, baseline_stage, base)
        rows.append(row)
        print(json.dumps({"stability_scored_source_n": len(rows), "source_n": len(cases),
                          "original_index": int(case["index"]),
                          "seed_status": {key: x["status"] for key, x in row["seeds"].items()}}), flush=True)
    summary = {str(seed): method_summary([r["seeds"][str(seed)] for r in rows]) for seed in seeds}
    for seed in seeds:
        outputs = [r["seeds"][str(seed)] for r in rows]
        summary[str(seed)]["selected_loop_closure_failure_n"] = sum(
            x.get("status") == "complete" and x.get("selected_closed") is False for x in outputs)
        summary[str(seed)]["introduced_topology_defect_n"] = sum(
            x.get("status") == "complete" and x.get("geometry", {}).get("geometry_no_new_topology_defect") is False
            for x in outputs)
        summary[str(seed)]["wrong_target_n"] = sum(
            x.get("status") == "complete" and x.get("selected_correct") is False for x in outputs)
        summary[str(seed)]["null_n"] = sum(x["seeds"][str(seed)].get("status") == "complete" and
            not x["seeds"][str(seed)].get("accepted", False) for x in rows)
        for key in ("fit_refine_seconds", "learning_stage_startup_inclusive_seconds"):
            times = [r["seeds"][str(seed)].get(key) for r in rows]
            times = [float(x) for x in times if isinstance(x, (int, float)) and np.isfinite(x)]
            summary[str(seed)][key].update(_stability_distribution(times))
            summary[str(seed)][key]["sum"] = float(np.sum(times)) if times else None
    pending = sum(x.get("status") == "pending" for row in rows for x in row["seeds"].values())
    seed314 = str(baseline_seed)
    if seed314 not in {str(x) for x in seeds}:
        raise ValueError("the author-default comparison seed must be part of the stability grid")
    budget_rows = [{"extended": r["seeds"][seed314], "default": r["baseline100"]} for r in rows]
    report = {"schema": "semigcn_source_cluster_seed_stability", "complete": pending == 0,
              "stability_protocol": str(protocol_path), "stability_protocol_sha256": sha(protocol_path),
              "source_protocol_sha256": sha(PROTOCOL_PATH), "independent_source_n": len(rows),
              "planned_fitting_runs_n": len(rows) * len(seeds), "pending_runs_n": pending,
              "iterations": iterations, "seeds": seeds, "original_case_indices": indices,
              "evidence_role": "Balanced development-prefix fitting-budget and stochastic-stability experiment; not independent confirmation.",
              "budget_selection": protocol.get("budget_selection", protocol.get("budget_choice", "Prespecified enlarged operational budget; no target labels or relative method superiority used.")),
              "metric_policy": {
                  "sample_unit": "source mesh; the three seeds are repeated measurements, not independent new meshes",
                  "interval": "95% percentile paired bootstrap over source meshes, 20000 draws; retain all seed outputs together",
                  "per_seed_endpoint_ci": "exact binomial CI over source meshes; no binomial CI on 60 correlated fitting runs",
                  "pending": "unfinished, never failures; temporary rates identify their actual terminal denominator",
                  "failed": "counted as unsuccessful endpoint observations after becoming terminal",
                  "safety": "correct selected target, strict closure, open distractor, no introduced intersection or topology defect; unknown is unsafe",
                  "geometry": "within-source seed SD uses ddof1; range is max minus min; common-safe source counts accompany quality comparisons",
                  "seam": "absolute oriented dihedral in radians; signed means are diagnostics",
                  "displacement": "normalized by median edge length of retained input faces, shared by all methods and seeds",
                  "timing": "fit/refine author stage and child startup-inclusive timers are separate, exclude common CGAL preparation; concurrent workers preclude dedicated-device speed claims",
                  "convergence": "fixed iteration count and low late loss/geometry change do not prove optimization convergence"},
              "per_seed": summary, "stability": _stability_statistics(rows, seeds),
              "expanded_default100_vs_selected_budget": {
                  "evidence_role": "Expanded balanced prefix supplements the original fixed 12-case budget cohort; all original indices retained.",
                  "default_stage": baseline_stage, "extended_iterations": iterations,
                  "seed": baseline_seed, "extended_minus_default": paired_summary(budget_rows, "extended", "default")},
              "rows": rows}
    dump(BENCH / f"{output_stem}.json", report)
    def number(value, precision=4):
        return "---" if value is None else f"{value:.{precision}f}"
    def interval(values, percent=False):
        return "---" if not values or values[0] is None else (
            f"[{values[0]:.1%}, {values[1]:.1%}]" if percent else f"[{values[0]:.4f}, {values[1]:.4f}]")
    lines = ["# SeMIGCN fitting budget and three-seed stability", "",
             f"{len(rows)} independent source meshes; {len(seeds)} repeated seeds; {iterations} iterations per fit. Complete: `{report['complete']}`; unfinished fitting runs: {pending}/{report['planned_fitting_runs_n']}.", "",
             report["evidence_role"], "", "## Per-seed safety and cost", "",
             "|Seed|Terminal / planned|Correct-safe|Exact 95% CI|Known intersections|Unknown intersections|Failures|Median fit/refine s|Median child pipeline s|",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        x = summary[str(seed)]
        lines.append(f"|{seed}|{x['evaluated_n']}/{x['n']}|{x['correct_safe_n']}/{x['rate_denominator']}|{interval(x['correct_safe_rate_ci95'], True)}|{x['introduced_intersection_n']}|{x['intersection_unknown_n']}|{x['failed_n']}|{number(x['fit_refine_seconds']['median'], 1)}|{number(x['learning_stage_startup_inclusive_seconds']['median'], 1)}|")
    s = report["stability"]
    endpoint = s["correct_safe"]
    lines += ["", "## Source-cluster stability", "",
              f"All-seed terminal sources: {s['terminal_all_seed_source_n']}/{len(rows)}. Seed-average correct-safe rate: {number(endpoint['mean'])}; source bootstrap 95% CI {interval(endpoint['source_bootstrap_ci95'], True)}.", "",
              f"Always safe: {endpoint['always_safe_source_n']}; sometimes unsafe: {endpoint['sometimes_unsafe_source_n']}; never safe: {endpoint['never_safe_source_n']}. Mean pairwise correct-safe disagreement: {number(s['correct_safe_pair_disagreement']['mean'])}; mean detailed safety-state disagreement: {number(s['safety_state_pair_disagreement']['mean'])}.", "",
              "The independent sample size is the number of source meshes. Three fits of one mesh remain together in every bootstrap resample. Cue-nearest cap selection is deterministic in the adapter; its agreement is an input/mapping control, not evidence of learned selection stability.", "",
              "## Geometry variation", "",
              "|Metric|All-three returned sources|Mean within-source seed SD|Mean within-source range|All-three safe sources|Mean safe-source seed SD|",
              "|---|---:|---:|---:|---:|---:|"]
    for name, x in s["geometry"].items():
        a, b = x["all_three_returned_same_target"], x["all_three_correct_safe_same_target"]
        lines.append(f"|{name}|{a['seed_mean']['n']}|{number(a['within_source_seed_sd']['mean'])}|{number(a['within_source_seed_range']['mean'])}|{b['seed_mean']['source_n']}|{number(b['within_source_seed_sd']['mean'])}|")
    b = report["expanded_default100_vs_selected_budget"]["extended_minus_default"]
    e = b["correct_safe_endpoint"]
    lines += ["", "## Expanded same-seed budget comparison", "",
              f"Seed{baseline_seed}, {iterations} minus100 iterations: {b['n_terminal_pairs']}/{len(rows)} terminal source pairs; correct-safe difference {number(e['difference_pp'], 2)} percentage points, paired source bootstrap 95% CI {interval(e['bootstrap_ci95_pp'])}; exact McNemar p={number(e['mcnemar_exact_two_sided_p'])}.", "",
              "This balanced prefix supplements the original frozen12-case budget comparison. Both budgets remain visible regardless of which yields a better endpoint. Same-seed reruns may have CUDA numeric differences; the original budget report separately records100-checkpoint replay discrepancies, dummy-mask equality, and late fitting trajectories.", "",
              "Runtime columns have separate scopes and exclude common CGAL preparation. Missing timer values remain unavailable. Fits run with concurrent GPU workers; these columns do not establish dedicated-device speedups. Fixed300 iterations do not establish convergence. Full per-seed quality/cost distributions, source rows, paired geometry intervals and raw outcomes are retained in the JSON."]
    (BENCH / f"{output_stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"complete": report["complete"], "source_n": len(rows), "pending_runs_n": pending,
                      "per_seed_correct_safe": {str(k): summary[str(k)]["correct_safe_n"] for k in seeds},
                      "source_cluster_correct_safe": endpoint}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--semigcn-stage", default="external")
    parser.add_argument("--semigcn-root", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--case-indices", type=int, nargs="+")
    parser.add_argument("--output-stem", default="unified_evaluation")
    parser.add_argument("--baseline-stage")
    parser.add_argument("--semigcn-status-file", type=Path)
    parser.add_argument("--stability-protocol", type=Path)
    parser.add_argument("--verify-learning-case", type=int)
    parser.add_argument("--measure-runtime", action="store_true")
    parser.add_argument("--measure-runtime-case", type=int)
    parser.add_argument("--runtime-method", choices=("tcmr", "cgal_pool", "meshfix"))
    parser.add_argument("--verify-local-mesh", type=Path)
    parser.add_argument("--verification-report", type=Path)
    args = parser.parse_args()
    if args.benchmark_root:
        BENCH = args.benchmark_root.resolve()
        PROTOCOL_PATH = BENCH / "protocol.json"
        PRED_DIR = BENCH / "predictions"
        MESHFIX_PATH = BENCH / "meshfix.json"
    if args.verify_local_mesh:
        if not args.verification_report:
            parser.error("--verify-local-mesh requires --verification-report")
        verify_local_mesh(args.verify_local_mesh, args.verification_report)
    elif args.verify_learning_case is not None:
        if not args.semigcn_root:
            parser.error("--verify-learning-case requires --semigcn-root")
        verify_learning_case(args.verify_learning_case, args.semigcn_stage, args.semigcn_root.resolve())
    elif args.measure_runtime:
        measure_runtime(args.case_indices)
    elif args.measure_runtime_case is not None:
        if not args.runtime_method:
            parser.error("--measure-runtime-case requires --runtime-method")
        measure_runtime_case(args.measure_runtime_case, args.runtime_method)
    elif args.stability_protocol:
        evaluate_stability(args.stability_protocol,
                           args.semigcn_root.resolve() if args.semigcn_root else None,
                           "stability_evaluation" if args.output_stem == "unified_evaluation" else args.output_stem)
    else:
        evaluate(args.semigcn_stage, args.semigcn_root.resolve() if args.semigcn_root else None,
                 args.case_indices, args.output_stem, args.baseline_stage, args.semigcn_status_file)
