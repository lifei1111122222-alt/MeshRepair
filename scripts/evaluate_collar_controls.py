"""Evaluate collar controls, common-pool fallback, budgets, and seam ablations.

The frozen optimizer uses the recorded candidate and native-verification limits.
Historical data locations remain intact for existing per-source evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from tcmr.joint_optimizer import DEFAULT_WEIGHTS

DATA = ROOT / "data/partnet_tcmr_prospective_v2"
OPTIMIZER = ROOT / "tcmr/joint_optimizer.py"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def paired_statistics(first, second, seed=20261011, replicates=20000):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if not len(first):
        return {"n": 0, "mean_first": None, "mean_second": None, "mean_first_minus_second": None,
                "bootstrap_95ci_first_minus_second": None}
    difference = first - second
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(replicates, len(difference)))
    means = difference[draws].mean(axis=1)
    return {"n": len(difference), "mean_first": float(first.mean()), "mean_second": float(second.mean()),
            "mean_first_minus_second": float(difference.mean()),
            "bootstrap_95ci_first_minus_second": np.quantile(means, [.025, .975]).tolist(),
            "first_greater_n": int((difference > 0).sum()), "equal_n": int((difference == 0).sum()),
            "first_lower_n": int((difference < 0).sum()), "bootstrap_replicates": replicates,
            "bootstrap_seed": seed, "independent_unit": "one source_model_id"}


EXTERNAL_DATASETS = {
    "holdout50": ROOT / "experiments/formal_coseg_external_holdout",
    "cross_category60": ROOT / "experiments/formal_coseg_cross_category",
}


def external_record(dataset, index):
    root = EXTERNAL_DATASETS[dataset]
    case = next(c for c in load(root / "protocol.json")["cases"] if c["index"] == index)
    targets = []
    for folder in (root / "work" / f"case{index:03d}").glob("loop*"):
        with np.load(folder / "damaged_input.npz", allow_pickle=False) as data:
            if set(data["selected_loop"].tolist()) == set(case["target_loop_vertices"]):
                targets.append(folder)
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one prescribed target for {dataset}/{index}")
    return case, targets[0]


def external_measure(vertices, before, retained, patch, loop, audit, patch_path):
    from scripts.evaluate_external_benchmark import geometry_metrics
    from tcmr.joint_optimizer import retained_collar
    geom = geometry_metrics(before, retained, vertices, patch, loop)
    h = float(np.median(np.linalg.norm(before[loop] - before[np.roll(loop, -1)], axis=1)))
    collar = retained_collar(retained, loop, rings=2)
    nonboundary = np.setdiff1d(collar, loop)
    disp = np.linalg.norm(vertices[:len(before)] - before, axis=1) / h
    valid = (audit.get("status") == "success"
             and audit.get("strict_selected_loop_closure") is True
             and audit.get("introduced_intersection") is False
             and geom.get("geometry_no_new_topology_defect") is True)
    return {"status": "success", "patch_path": str(patch_path.resolve()),
            "valid": bool(valid), "geometry": geom,
            "retained_rms_over_boundary_edge": float(np.sqrt(np.mean(disp ** 2))),
            "retained_max_over_boundary_edge": float(disp.max(initial=0)),
            "nonboundary_collar_rms_over_boundary_edge": float(np.sqrt(np.mean(disp[nonboundary] ** 2))) if len(nonboundary) else 0.,
            "verification": {k: audit.get(k) for k in (
                "status", "strict_selected_loop_closure", "introduced_intersection",
                "introduced_intersection_nondegenerate", "introduced_pair_count",
                "introduced_groups", "before_degenerate_faces", "after_degenerate_faces")}}


def external_worker(dataset, index, output_dir):
    """One new patch-only output, with the same frozen objective and native check."""
    import os
    import subprocess
    import torch
    from tcmr.joint_optimizer import optimize_joint_collar
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    case, folder = external_record(dataset, index)
    with np.load(folder / "damaged_input.npz", allow_pickle=False) as data:
        before = data["vertices"].astype(np.float64)
        retained = data["faces"].astype(np.int64)
        loop = data["selected_loop"].astype(np.int64)
    row = {"dataset": dataset, "index": index, "sample_id": case["sample_id"],
           "source_model_id": case["source_model_id"], "category": case["category"],
           "input_path": str((folder / "damaged_input.npz").resolve()), "methods": {}}
    initializer_path = folder / "cgal_refine_fair.npz"
    initializer_record = load(folder / "cgal_refine_fair.json")
    if initializer_record.get("status") != "success" or not initializer_path.exists():
        row["methods"] = {m: {"status": "no_initializer", "valid": False} for m in ("initializer", "patch_only", "joint")}
        print("EXTERNAL_RESULT=" + json.dumps(row), flush=True)
        return
    with np.load(initializer_path, allow_pickle=False) as data:
        initial, patch = data["vertices"].astype(np.float64), data["patch_faces"].astype(np.int64)
    if not np.array_equal(initial[:len(before)], before):
        raise ValueError("CGAL initializer changed retained vertices")
    initial_audit = load(folder / "cgal_refine_fair_verification.json")
    row["methods"]["initializer"] = external_measure(initial, before, retained, patch, loop, initial_audit, initializer_path)
    started = time.perf_counter()
    after, info = optimize_joint_collar(initial, retained, patch, loop, len(before), mode="patch_only",
                                       steps=10, learning_rate=.02, retained_displacement_budget=.02, rings=2)
    optimize_seconds = time.perf_counter() - started
    destination = output_dir / f"{dataset}_{index:03d}_patch_only.npz"
    np.savez_compressed(destination, vertices=after, patch_faces=patch)
    started = time.perf_counter()
    # CGAL's Windows native bindings must be isolated from the PyTorch process.
    verification_path = output_dir / f"{dataset}_{index:03d}_verification.json"
    native = subprocess.run([sys.executable, str(ROOT / "scripts/cgal_changed_surface_audit.py"),
                             "--input", str(folder / "damaged_input.npz"), "--patch", str(destination),
                             "--output", str(verification_path)], cwd=ROOT, capture_output=True, text=True,
                            timeout=100, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if native.returncode or not verification_path.exists():
        raise RuntimeError(f"Isolated native verification failed: {native.returncode} {native.stderr[-500:]}")
    verification = load(verification_path)
    verify_seconds = time.perf_counter() - started
    item = external_measure(after, before, retained, patch, loop, verification, destination)
    item.update(optimizer=info, optimizer_seconds=optimize_seconds, verification_seconds=verify_seconds)
    row["methods"]["patch_only"] = item
    joint_path = folder / "bounded_joint10.npz"
    joint_record = load(folder / "bounded_joint10.json")
    spec = load(folder / "bounded_joint10_spec.json")
    if Path(spec["initializer_path"]).resolve() != initializer_path.resolve():
        raise ValueError("Saved joint output used a different initializer")
    if joint_record.get("status") == "success" and joint_path.exists():
        with np.load(joint_path, allow_pickle=False) as data:
            joint, joint_patch = data["vertices"].astype(np.float64), data["patch_faces"].astype(np.int64)
        if not np.array_equal(joint_patch, patch) or len(joint) != len(initial):
            raise ValueError("Joint/patch-only topology differs")
        diagnostics = joint_record["bounded_joint_diagnostics"]
        if diagnostics["mode"] != "joint" or diagnostics["steps"] != 10 or diagnostics["dtype"] != "float64" or diagnostics["weights"] != DEFAULT_WEIGHTS or diagnostics["retained_displacement_budget_normalized"] != .02:
            raise ValueError("Saved joint settings differ from frozen objective")
        if any(not np.isclose(info["initial_terms"][k], value, rtol=1e-10, atol=1e-12) for k, value in diagnostics["initial_terms"].items()):
            raise ValueError("Saved joint initializer objective differs")
        audit = load(folder / "bounded_joint10_verification.json")
        item = external_measure(joint, before, retained, patch, loop, audit, joint_path)
        item.update(optimizer=diagnostics, reused_frozen_output=True, identical_initializer_objective_and_topology=True)
        row["methods"]["joint"] = item
    else:
        row["methods"]["joint"] = {"status": joint_record.get("status", "missing"), "valid": False}
    print("EXTERNAL_RESULT=" + json.dumps(row, allow_nan=False), flush=True)


def external_value(item, metric):
    if metric == "retained_rms_over_boundary_edge":
        return item[metric]
    geom = item["geometry"]
    return geom["triangle_quality"][metric.split(".")[1]] if metric.startswith("triangle_quality.") else geom[metric]


def external_summary(rows):
    names = ("initializer", "patch_only", "joint")
    metrics = ("seam_normal_absolute_mean", "seam_normal_absolute_p95", "triangle_quality.mean",
               "triangle_quality.p05", "retained_displacement_rms_over_edge",
               "retained_displacement_max_over_edge", "retained_rms_over_boundary_edge")
    methods = {}
    for method in names:
        items = [r["methods"][method] for r in rows]
        returned = [r for r in items if r.get("geometry", {}).get("status") == "success"]
        methods[method] = {"denominator": len(rows), "returned_n": len(returned),
                           "valid_n": sum(r.get("valid") is True for r in items),
                           "closure_n": sum(r.get("verification", {}).get("strict_selected_loop_closure") is True for r in items),
                           "intersection_n": sum(r.get("verification", {}).get("introduced_intersection") is True for r in items),
                           "unknown_or_no_output_n": sum(r.get("verification", {}).get("introduced_intersection") is None for r in items),
                           "outside_two_ring_fixed_n": sum(r["geometry"].get("retained_outside_two_ring_bitwise_fixed") is True for r in returned),
                           "max_retained_displacement_over_boundary_edge": max((r["retained_max_over_boundary_edge"] for r in returned), default=None)}
    cohorts = {}
    for name, chosen in (
        ("all_returned", [r for r in rows if all(r["methods"][m].get("geometry", {}).get("status") == "success" for m in names)]),
        ("common_valid", [r for r in rows if all(r["methods"][m].get("valid") is True for m in names)])):
        cohorts[name] = {"n": len(chosen), "source_ids": [r["source_model_id"] for r in chosen], "method_means": {}, "paired": {}}
        for m in names:
            cohorts[name]["method_means"][m] = {k: float(np.mean([external_value(r["methods"][m], k) for r in chosen])) if chosen else None for k in metrics}
        for first, second in (("initializer", "patch_only"), ("patch_only", "joint"), ("initializer", "joint")):
            cohorts[name]["paired"][first + "_minus_" + second] = {
                k: paired_statistics([external_value(r["methods"][first], k) for r in chosen],
                                     [external_value(r["methods"][second], k) for r in chosen], seed=20261025)
                for k in metrics}
    valid_pairs = paired_statistics([float(r["methods"]["joint"].get("valid") is True) for r in rows],
                                    [float(r["methods"]["patch_only"].get("valid") is True) for r in rows], seed=20261025)
    free_vertex_strata = {}
    for label, has_free in (("no_new_patch_vertices", False), ("with_new_patch_vertices", True)):
        group = [r for r in rows if all(r["methods"][m].get("valid") is True for m in names)
                 and (r["methods"]["patch_only"]["optimizer"]["variable_vertices_n"] > 0) == has_free]
        free_vertex_strata[label] = paired_statistics(
            [r["methods"]["patch_only"]["geometry"]["seam_normal_absolute_mean"] for r in group],
            [r["methods"]["joint"]["geometry"]["seam_normal_absolute_mean"] for r in group], seed=20261025)
    return {"source_n": len(rows), "methods": methods, "cohorts": cohorts, "valid_joint_minus_patch_only": valid_pairs,
            "common_valid_patch_free_vertex_strata": free_vertex_strata}


def run_external(output_dir):
    import os
    import subprocess
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_sha = sha(OPTIMIZER)
    protocol = {"schema": "formal_fixed_vs_movable_collar", "prescribed_target_mechanism_experiment": True,
                "end_to_end_selector_evaluation": False, "planned_n": 110,
                "datasets": {name: str(path) for name, path in EXTERNAL_DATASETS.items()},
                "methods": ["initializer", "patch_only", "joint"], "steps": 10, "learning_rate": .02,
                "weights": DEFAULT_WEIGHTS, "dtype": "float64", "collar_rings": 2, "retained_budget_over_boundary_edge": .02,
                "optimizer_sha256": optimizer_sha, "new_outputs": "patch_only; initializer and joint reused from frozen same-topology records after checking objective and settings",
                "selection": "Every prescribed target in both complete cohorts, independent of method acceptance and intersection outcome",
                "metrics": "same oriented absolute seam evaluator as manuscript; per-source mean and P95; triangle quality mean/P05; retained RMS and maximum",
                "statistics": "20000 paired bootstrap replicates over independent source meshes; seed20261025; no vertex or edge replication",
                "timing": "Descriptive optimization and verification only for new patch-only outputs; not a matched runtime comparison"}
    results_path = output_dir / "results.json"
    previous = load(results_path) if results_path.exists() else {}
    if previous and previous["protocol"] != protocol:
        raise ValueError("Protocol differs from existing formal results")
    completed = {(r["dataset"], r["index"]): r for r in previous.get("rows", []) if "error" not in r}
    queue = [(name, c["index"]) for name, path in EXTERNAL_DATASETS.items() for c in load(path / "protocol.json")["cases"]]
    def job(key):
        dataset, index = key
        started = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--external-worker", dataset,
                                   "--external-index", str(index), "--output-dir", str(output_dir)],
                                  cwd=ROOT, capture_output=True, text=True, timeout=120,
                                  env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"),
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            lines = [line[len("EXTERNAL_RESULT="):] for line in proc.stdout.splitlines() if line.startswith("EXTERNAL_RESULT=")]
            if proc.returncode or len(lines) != 1:
                raise RuntimeError(f"worker returned {proc.returncode}: {proc.stderr[-1500:]}")
            row = json.loads(lines[0])
            row["new_measurement_inclusive_seconds"] = time.perf_counter() - started
            return row
        except Exception as exc:
            # Keep every prescribed source; diagnose infrastructure failures before any manuscript inference.
            case, _ = external_record(dataset, index)
            return {"dataset": dataset, "index": index, "sample_id": case["sample_id"],
                    "source_model_id": case["source_model_id"], "error": str(exc),
                    "methods": {m: {"status": "measurement_failure", "valid": False} for m in protocol["methods"]}}
    pending = [key for key in queue if key not in completed]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(job, key): key for key in pending}
        for future in as_completed(futures):
            key = futures[future]
            row = future.result()
            completed[key] = row
            save(results_path, {"complete": False, "protocol": protocol, "rows": [completed[k] for k in queue if k in completed]})
            print(json.dumps({"completed": len(completed), "total": len(queue), "dataset": key[0], "index": key[1],
                              "valid": {m: row["methods"][m]["valid"] for m in protocol["methods"]}, "error": row.get("error")}), flush=True)
    rows = [completed[key] for key in queue]
    canonical_ids = [r["source_model_id"].removeprefix("holdout_").removeprefix("cross_") for r in rows]
    if len(set(canonical_ids)) != len(rows):
        raise ValueError("Source identities are not independent")
    if sha(OPTIMIZER) != optimizer_sha:
        raise ValueError("Frozen optimizer changed")
    summaries = {name: external_summary([r for r in rows if r["dataset"] == name]) for name in EXTERNAL_DATASETS}
    summaries["pooled110"] = external_summary(rows)
    result = {"complete": True, "protocol": protocol, "canonical_source_ids": canonical_ids,
              "canonical_source_unique_n": len(set(canonical_ids)), "summary": summaries, "rows": rows}
    save(results_path, result)
    print(json.dumps({"complete": True, "summary": summaries}, indent=2), flush=True)


def budget_protocol(output_dir):
    """Freeze a small development-only budget check before reading outcomes."""
    source_path = ROOT / "experiments/joint_collar_complete201_v4/frozen_queue201.json"
    records = []
    seen = set()
    for record in load(source_path)["records"]:
        if record["source_model_id"] in seen or record.get("initializer_status") != "success":
            continue
        if not Path(record["initializer_path"]).exists():
            continue
        records.append(record)
        seen.add(record["source_model_id"])
        if len(records) == 30:
            break
    if len(records) != 30:
        raise ValueError("Expected 30 source-distinct saved development initializers")
    protocol = {
        "schema": "limited_optimization_budget", "development_only": True,
        "selection": "First 30 source-distinct records with a saved successful CGAL initializer in the frozen development201 queue; no geometric or safety outcome used for selection",
        "source_queue": str(source_path), "source_queue_sha256": sha(source_path),
        "sample_n": 30, "records": records, "methods": ["patch_only", "joint"],
        "budgets": [10, 30, 100], "learning_rate": .02, "retained_budget_over_boundary_edge": .02,
        "collar_rings": 2, "dtype": "float64", "weights": DEFAULT_WEIGHTS,
        "optimizer_sha256": sha(OPTIMIZER), "candidate_workers": 1, "torch_cpu_threads": 2,
        "verification": "Reuse exact saved ten-step native records; one isolated changed-surface check for each new 30/100-step output, including all moved retained faces",
        "reuse": "Ten-step outputs only after matching settings, topology, and initializer objective; no optimizer modifications or tuning",
        "statistics": "20000 paired source bootstrap replicates, seed20261025; all 30 sources in validity denominators; geometry on declared common-valid subsets",
        "timing": "Optimization and verification wall time descriptive for newly computed 30/100-step outputs in a serial run; historical ten-step timing is separately labeled and not an equal-CPU comparison",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "budget_protocol.json"
    if path.exists() and load(path) != protocol:
        raise ValueError("Existing limited-budget protocol differs")
    save(path, protocol)
    return protocol


def budget_worker(position, output_dir):
    import os
    import subprocess
    import torch
    from tcmr.joint_optimizer import optimize_joint_collar
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    protocol = load(output_dir / "budget_protocol.json")
    record = protocol["records"][position]
    previous = load(ROOT / "experiments/joint_collar_complete201_v4/results.json")
    historical_settings = previous["protocol"]["optimizer_settings"]
    if historical_settings["learning_rate_in_median_boundary_edge_units"] != .02 or historical_settings["collar_retained_graph_rings"] != 2:
        raise ValueError("Historical learning rate or collar support differs")
    old = next(r for r in previous["rows"] if r["sample_id"] == record["sample_id"])
    with np.load(record["input_path"], allow_pickle=False) as data:
        before, retained, loop = data["vertices"].astype(float), data["faces"].astype(np.int64), data["selected_loop"].astype(np.int64)
    with np.load(record["initializer_path"], allow_pickle=False) as data:
        initial, patch = data["vertices"].astype(float), data["patch_faces"].astype(np.int64)
    if not np.array_equal(initial[:len(before)], before):
        raise ValueError("Development initializer changes retained vertices")
    row = {"position": position, "source_position": record["position"], "sample_id": record["sample_id"],
           "source_model_id": record["source_model_id"], "input_path": record["input_path"],
           "initializer_path": record["initializer_path"], "methods": {}}
    destination = output_dir / "budget_meshes"
    destination.mkdir(exist_ok=True)
    for steps in protocol["budgets"]:
        for mode in protocol["methods"]:
            label = f"{mode}_{steps}"
            if steps == 10:
                cached = old["methods"][mode]
                info = cached["optimizer"]
                if info["mode"] != mode or info["steps"] != steps or info["dtype"] != "float64" or info["weights"] != DEFAULT_WEIGHTS or info["retained_displacement_budget_normalized"] != .02:
                    raise ValueError("Historical ten-step settings differ")
                patch_path = Path(cached["patch_path"]).resolve()
                with np.load(patch_path, allow_pickle=False) as data:
                    vertices, actual_patch = data["vertices"].astype(float), data["patch_faces"].astype(np.int64)
                if len(vertices) != len(initial) or not np.array_equal(actual_patch, patch):
                    raise ValueError("Historical ten-step topology differs")
                _, initial_info = optimize_joint_collar(initial, retained, patch, loop, len(before), mode=mode, steps=0,
                                                        learning_rate=.02, retained_displacement_budget=.02, rings=2)
                if any(not np.isclose(value, initial_info["initial_terms"][key], rtol=1e-10, atol=1e-12)
                       for key, value in info["initial_terms"].items()):
                    raise ValueError("Historical ten-step initializer objective differs")
                audit_path = Path(cached["audit_path"]).resolve()
                audit = load(audit_path)
                item = external_measure(vertices, before, retained, actual_patch, loop, audit, patch_path)
                item.update(optimizer=info, optimizer_seconds=cached.get("optimizer_seconds"),
                            verification_seconds=cached.get("audit_seconds"), verification_path=str(audit_path),
                            reused_exact_output=True, timing_scope="historical run; not matched timing")
            else:
                started = time.perf_counter()
                vertices, info = optimize_joint_collar(initial, retained, patch, loop, len(before), mode=mode, steps=steps,
                                                       learning_rate=.02, retained_displacement_budget=.02, rings=2)
                optimize_seconds = time.perf_counter() - started
                patch_path = destination / f"{position:03d}_{label}.npz"
                np.savez_compressed(patch_path, vertices=vertices, patch_faces=patch)
                audit_path = destination / f"{position:03d}_{label}_verification.json"
                started = time.perf_counter()
                try:
                    process = subprocess.run([sys.executable, str(ROOT / "scripts/cgal_changed_surface_audit.py"),
                        "--input", record["input_path"], "--patch", str(patch_path), "--output", str(audit_path)],
                        cwd=ROOT, capture_output=True, text=True, timeout=120,
                        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    if process.returncode or not audit_path.exists():
                        raise RuntimeError(f"Native verifier returned {process.returncode}: {process.stderr[-1000:]}")
                    audit = load(audit_path)
                except Exception as exc:
                    audit = {"status": "verification_failure", "error": repr(exc), "introduced_intersection": None,
                             "strict_selected_loop_closure": False}
                    save(audit_path, audit)
                verify_seconds = time.perf_counter() - started
                item = external_measure(vertices, before, retained, patch, loop, audit, patch_path)
                item.update(optimizer=info, optimizer_seconds=optimize_seconds, verification_seconds=verify_seconds,
                            verification_path=str(audit_path), reused_exact_output=False,
                            timing_scope="current serial run; optimization and native verification separately")
            row["methods"][label] = item
    print("BUDGET_RESULT=" + json.dumps(row, allow_nan=False), flush=True)


def budget_summary(rows):
    names = tuple(f"{mode}_{steps}" for steps in (10, 30, 100) for mode in ("patch_only", "joint"))
    metrics = ("seam_normal_absolute_mean", "seam_normal_absolute_p95", "triangle_quality.mean", "triangle_quality.p05",
               "retained_rms_over_boundary_edge", "retained_displacement_max_over_edge")
    summary = {"source_n": len(rows), "methods": {}, "common_valid_all_budgets": {}, "paired_by_budget": {}, "budget_changes": {}}
    def mean_interval(values):
        values = np.asarray(values, float)
        if not len(values):
            return {"n": 0, "mean": None, "bootstrap_95ci_mean": None}
        rng = np.random.default_rng(20261025)
        means = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
        return {"n": len(values), "mean": float(values.mean()),
                "bootstrap_95ci_mean": np.quantile(means, [.025, .975]).tolist(),
                "bootstrap_seed": 20261025, "bootstrap_replicates": 20000, "independent_unit": "one source_model_id"}
    for name in names:
        items = [r["methods"][name] for r in rows]
        returned = [x for x in items if x.get("geometry", {}).get("status") == "success"]
        summary["methods"][name] = {
            "denominator": len(rows), "returned_n": len(returned), "valid_n": sum(x.get("valid") is True for x in items),
            "intersection_n": sum(x.get("verification", {}).get("introduced_intersection") is True for x in items),
            "unknown_n": sum(x.get("verification", {}).get("introduced_intersection") is None for x in items),
            "mean_on_all_returned": {k: float(np.mean([external_value(x, k) for x in returned])) if returned else None for k in metrics},
            "all_returned_mean_intervals": {k: mean_interval([external_value(x, k) for x in returned]) for k in metrics},
            "all_returned_objective": {k: mean_interval([x["optimizer"][k] for x in returned])
                                       for k in ("initial_objective", "final_objective")},
            "optimization_seconds_mean": float(np.mean([x["optimizer_seconds"] for x in returned])) if returned else None,
            "optimization_seconds_median": float(np.median([x["optimizer_seconds"] for x in returned])) if returned else None,
            "verification_seconds_mean": float(np.mean([x["verification_seconds"] for x in returned])) if returned else None,
            "timing_scope": "historical reused run" if name.endswith("_10") else "new serial budget run",
        }
    common = [r for r in rows if all(r["methods"][m].get("valid") is True for m in names)]
    summary["common_valid_all_budgets"] = {"n": len(common), "source_ids": [r["source_model_id"] for r in common],
        "means": {m: {k: float(np.mean([external_value(r["methods"][m], k) for r in common])) if common else None for k in metrics} for m in names},
        "mean_intervals": {m: {k: mean_interval([external_value(r["methods"][m], k) for r in common]) for k in metrics} for m in names},
        "objective": {m: {k: mean_interval([r["methods"][m]["optimizer"][k] for r in common])
                           for k in ("initial_objective", "final_objective")} for m in names}}
    def compare(group, first, second):
        return {k: paired_statistics([external_value(r["methods"][first], k) for r in group],
                                     [external_value(r["methods"][second], k) for r in group], seed=20261025) for k in metrics}
    for steps in (10, 30, 100):
        first, second = f"patch_only_{steps}", f"joint_{steps}"
        group = [r for r in rows if r["methods"][first].get("valid") is True and r["methods"][second].get("valid") is True]
        summary["paired_by_budget"][str(steps)] = {"n_common_valid": len(group), "patch_only_minus_joint": compare(group, first, second),
                                                  "all_budgets_common_valid": compare(common, first, second)}
    for mode in ("patch_only", "joint"):
        for steps in (30, 100):
            first, second = f"{mode}_{steps}", f"{mode}_10"
            group = [r for r in rows if r["methods"][first].get("valid") is True and r["methods"][second].get("valid") is True]
            summary["budget_changes"][first + "_minus_10"] = {"n_common_valid": len(group), "paired": compare(group, first, second),
                                                               "all_budgets_common_valid": compare(common, first, second)}
    summary["fixed100_minus_joint10_all_budgets_common_valid"] = {
        "n": len(common), "paired": compare(common, "patch_only_100", "joint_10"),
        "final_objective": paired_statistics(
            [r["methods"]["patch_only_100"]["optimizer"]["final_objective"] for r in common],
            [r["methods"]["joint_10"]["optimizer"]["final_objective"] for r in common], seed=20261025)}
    return summary


def run_budget(output_dir):
    import os
    import subprocess
    protocol = budget_protocol(output_dir)
    result_path = output_dir / "budget_results.json"
    prior = load(result_path) if result_path.exists() else {}
    if prior and prior["protocol"] != protocol:
        raise ValueError("Budget protocol changed")
    rows = {r["position"]: r for r in prior.get("rows", [])}
    for position in range(30):
        if position in rows:
            continue
        started = time.perf_counter()
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--budget-worker", str(position),
                                  "--output-dir", str(output_dir)], cwd=ROOT, capture_output=True, text=True, timeout=900,
                                 env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"),
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        lines = [s[len("BUDGET_RESULT="):] for s in process.stdout.splitlines() if s.startswith("BUDGET_RESULT=")]
        if process.returncode or len(lines) != 1:
            raise RuntimeError(f"Budget infrastructure failure at position {position}: {process.returncode} {process.stderr[-2000:]}")
        row = json.loads(lines[0])
        row["source_inclusive_seconds"] = time.perf_counter() - started
        rows[position] = row
        save(result_path, {"complete": False, "protocol": protocol, "rows": [rows[i] for i in sorted(rows)]})
        print(json.dumps({"budget_completed": len(rows), "planned": 30, "sample_id": row["sample_id"],
                          "valid": {k: v["valid"] for k, v in row["methods"].items()}}), flush=True)
    if sha(OPTIMIZER) != protocol["optimizer_sha256"]:
        raise ValueError("Frozen optimizer changed during budget comparison")
    ordered = [rows[i] for i in range(30)]
    result = {"complete": True, "protocol": protocol, "rows": ordered, "summary": budget_summary(ordered)}
    save(result_path, result)
    print(json.dumps({"complete": True, "budget_summary": result["summary"]}), flush=True)


def fallback_summary(rows):
    common = [r for r in rows if all(r["policies"][m].get("valid") is True for m in ("fixed_collar", "movable_collar"))]
    summary = {"source_n": len(rows), "categories": {}, "policies": {}, "paired_common_valid": {},
               "matched_predicted_target_n": sum(r["matches_frozen_predicted_target"] for r in rows)}
    for category in ("safe_improvement", "successful_fallback", "no_gain", "no_feasible_output"):
        summary["categories"][category] = sum(r["category"] == category for r in rows)
    for policy in ("fixed_collar", "movable_collar"):
        summary["policies"][policy] = {"accepted_n": sum(r["policies"][policy]["accepted"] for r in rows),
            "valid_n": sum(r["policies"][policy].get("valid") is True for r in rows),
            "selected_candidates": {name: sum(r["policies"][policy].get("method") == name for r in rows)
                                    for name in ("cgal_refine_fair", "cgal_polyline", "center_fan", "patch_only", "joint")}}
    for metric in ("seam_normal_absolute_mean", "seam_normal_absolute_p95", "triangle_quality.mean", "triangle_quality.p05", "retained_rms_over_boundary_edge"):
        summary["paired_common_valid"][metric] = paired_statistics(
            [external_value(r["policies"]["fixed_collar"], metric) for r in common],
            [external_value(r["policies"]["movable_collar"], metric) for r in common], seed=20261025)
    summary["valid_movable_minus_fixed"] = paired_statistics(
        [float(r["policies"]["movable_collar"].get("valid") is True) for r in rows],
        [float(r["policies"]["fixed_collar"].get("valid") is True) for r in rows], seed=20261025)
    summary["worse_oriented_seam_n_common_valid"] = sum(r["oriented_seam_fixed_minus_movable"] < -1e-12 for r in common)
    summary["equal_oriented_seam_n_common_valid"] = sum(abs(r["oriented_seam_fixed_minus_movable"]) <= 1e-12 for r in common)
    summary["joint_unsafe_n"] = sum(not r["candidates"]["joint"]["safe_eligible"] for r in rows)
    summary["joint_unsafe_fallback_valid_n"] = sum(not r["candidates"]["joint"]["safe_eligible"] and r["policies"]["movable_collar"].get("valid") is True for r in rows)
    return summary


def run_fallback(output_dir):
    """Replay only candidate selection on the prescribed target; never rerank."""
    from scripts.evaluate_geometry_comparison import _seam_normal_error
    source_path = output_dir / "results.json"
    source = load(source_path)
    if not source.get("complete") or len(source["rows"]) != 110:
        raise ValueError("Requires completed 110-source collar comparison")
    shared = ("cgal_refine_fair", "cgal_polyline", "center_fan")
    protocol = {"schema": "same_target_safe_fallback_replay", "source_results": str(source_path.resolve()),
                "source_results_sha256": sha(source_path), "prescribed_target_diagnostic": True,
                "end_to_end_selector_evaluation": False, "planned_n": 110,
                "shared_candidates": list(shared), "replaced_slot": "patch_only versus joint, both frozen ten-step outputs",
                "selection": "Same explicitly clear and strictly closed eligibility as frozen policy; select lowest unoriented seam score with fixed candidate order",
                "reporting": "Oriented absolute seam under the unified evaluator; all 110 source denominators; geometry on common-valid accepted outputs",
                "net_categories": "Mutually exclusive: no feasible output; successful fallback when joint ineligible and common alternative valid; safe improvement against valid fixed-collar policy; otherwise no gain. Any worsened oriented seam reported separately",
                "statistics": "20000 paired source bootstrap replicates, seed20261025",
                "predicted_target_subset": "Supplemental conditional replay only where original frozen locked target equals prescribed target; not an estimate of full end-to-end counterfactual accuracy",
                "new_optimization_runs": 0, "new_native_verifications": 0}
    rows = []
    for original in source["rows"]:
        dataset, index = original["dataset"], original["index"]
        case, folder = external_record(dataset, index)
        with np.load(folder / "damaged_input.npz", allow_pickle=False) as data:
            before, retained, loop = data["vertices"].astype(float), data["faces"].astype(np.int64), data["selected_loop"].astype(np.int64)
        candidates = {}
        for name in shared:
            record = load(folder / f"{name}.json")
            path = folder / f"{name}.npz"
            if record.get("status") != "success" or not path.exists():
                candidates[name] = {"method": name, "status": record.get("status", "missing"), "safe_eligible": False, "valid": False}
                continue
            with np.load(path, allow_pickle=False) as data:
                vertices, patch = data["vertices"].astype(float), data["patch_faces"].astype(np.int64)
            audit = load(folder / f"{name}_verification.json")
            item = external_measure(vertices, before, retained, patch, loop, audit, path)
            score = _seam_normal_error(vertices, retained, patch, loop)
            item.update(method=name, selection_seam_unoriented=score,
                        safe_eligible=bool(audit.get("status") == "success" and audit.get("strict_selected_loop_closure") is True
                                           and audit.get("introduced_intersection") is False and score is not None and np.isfinite(score)))
            if item["safe_eligible"] != record.get("safe_eligible"):
                raise ValueError("Replayed shared-candidate eligibility differs from frozen inference")
            candidates[name] = item
        for name in ("patch_only", "joint"):
            item = dict(original["methods"][name])
            with np.load(item["patch_path"], allow_pickle=False) as data:
                score = _seam_normal_error(data["vertices"], retained, data["patch_faces"], loop)
            audit = item["verification"]
            item.update(method=name, selection_seam_unoriented=score,
                        safe_eligible=bool(audit.get("status") == "success" and audit.get("strict_selected_loop_closure") is True
                                           and audit.get("introduced_intersection") is False and score is not None and np.isfinite(score)))
            candidates[name] = item
        policies = {}
        for policy, variant in (("fixed_collar", "patch_only"), ("movable_collar", "joint")):
            allowed = (*shared, variant)
            eligible = [candidates[m] for m in allowed if candidates[m]["safe_eligible"]]
            chosen = min(eligible, key=lambda x: (x["selection_seam_unoriented"], allowed.index(x["method"]))) if eligible else None
            policies[policy] = {**chosen, "accepted": True} if chosen else {"accepted": False, "valid": False, "method": None}
        fixed, movable = policies["fixed_collar"], policies["movable_collar"]
        delta = (fixed["geometry"]["seam_normal_absolute_mean"] - movable["geometry"]["seam_normal_absolute_mean"]
                 if fixed.get("valid") and movable.get("valid") else None)
        if not movable["accepted"]:
            category = "no_feasible_output"
        elif not candidates["joint"]["safe_eligible"] and movable.get("valid"):
            category = "successful_fallback"
        elif delta is not None and delta > 1e-12:
            category = "safe_improvement"
        else:
            category = "no_gain"
        inference = load(EXTERNAL_DATASETS[dataset] / "work" / f"case{index:03d}" / "inference.json")
        locked = inference["decisions"]["prior_locked_pool_with_joint"]["locked_loop_id"]
        matches = locked == int(folder.name.removeprefix("loop"))
        row = {"dataset": dataset, "index": index, "sample_id": case["sample_id"], "source_model_id": case["source_model_id"],
               "category": category, "candidates": candidates, "policies": policies,
               "oriented_seam_fixed_minus_movable": delta, "matches_frozen_predicted_target": matches}
        rows.append(row)
    summaries = {name: fallback_summary([r for r in rows if r["dataset"] == name]) for name in EXTERNAL_DATASETS}
    summaries["pooled110"] = fallback_summary(rows)
    summaries["conditional_matched_predicted_targets"] = fallback_summary([r for r in rows if r["matches_frozen_predicted_target"]])
    result = {"complete": True, "protocol": protocol, "summary": summaries, "rows": rows}
    save(output_dir / "fallback_results.json", result)
    print(json.dumps({"complete": True, "fallback_summary": summaries}), flush=True)


def source_mean_interval(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0, "mean": None, "bootstrap_95ci_mean": None}
    rng = np.random.default_rng(20261025)
    means = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return {"n": len(values), "mean": float(values.mean()), "bootstrap_95ci_mean": np.quantile(means, [.025, .975]).tolist(),
            "bootstrap_replicates": 20000, "bootstrap_seed": 20261025, "independent_unit": "one source_model_id"}


FAIRING_FALLBACK_SLOTS = {"fixed_collar": "patch_only", "bounded_fair": "bounded_biharmonic_fair", "joint_collar": "joint"}
FAIRING_FALLBACK_METRICS = (
    "seam_normal_absolute_mean", "seam_normal_absolute_p95", "triangle_quality.mean", "triangle_quality.p05",
    "retained_displacement_rms_over_edge", "retained_displacement_max_over_edge", "retained_rms_over_boundary_edge",
    "changed_surface_area_ratio", "patch_area", "moved_retained_area")


def fairing_fallback_summary(rows):
    names = tuple(FAIRING_FALLBACK_SLOTS)
    common = [r for r in rows if all(r["policies"][m].get("valid") is True for m in names)]
    identity = lambda r: {k: r[k] for k in ("dataset", "index", "source_model_id")}
    summary = {"source_n": len(rows), "policies": {}, "common_valid_all_three_n": len(common),
               "common_valid_all_three_sources": [identity(r) for r in common],
               "all_three_common_means": {}, "paired": {}}
    for name, variant in FAIRING_FALLBACK_SLOTS.items():
        items = [r["policies"][name] for r in rows]
        bad = [r for r in rows if not r["candidates"][variant]["safe_eligible"]]
        rescued = [r for r in bad if r["policies"][name].get("valid") is True]
        summary["policies"][name] = {
            "accepted_n": sum(x["accepted"] for x in items),
            "valid_n": sum(x.get("valid") is True for x in items),
            "abstain_n": sum(not x["accepted"] for x in items),
            "accepted_invalid_n": sum(x["accepted"] and x.get("valid") is not True for x in items),
            "valid_rate": source_mean_interval([float(x.get("valid") is True) for x in items]),
            "selected_candidates": {candidate: sum(x.get("method") == candidate for x in items)
                                    for candidate in ("cgal_refine_fair", "cgal_polyline", "center_fan", variant)},
            "slot_ineligible_n": len(bad), "ineligible_slot_rescued_n": len(rescued),
            "rescued_sources": [identity(r) for r in rescued],
            "unresolved_sources": [identity(r) for r in bad if r["policies"][name].get("valid") is not True],
            "abstained_sources": [identity(r) for r in rows if not r["policies"][name]["accepted"]]}
        summary["all_three_common_means"][name] = {
            m: source_mean_interval([external_value(r["policies"][name], m) for r in common]) for m in FAIRING_FALLBACK_METRICS}
    for first, second in (("joint_collar", "fixed_collar"), ("joint_collar", "bounded_fair"), ("bounded_fair", "fixed_collar")):
        pair = [r for r in rows if all(r["policies"][name].get("valid") is True for name in (first, second))]
        def compare(group):
            return {m: paired_statistics([external_value(r["policies"][first], m) for r in group],
                                         [external_value(r["policies"][second], m) for r in group], seed=20261025)
                    for m in FAIRING_FALLBACK_METRICS}
        deltas = {r["source_model_id"]: external_value(r["policies"][first], "seam_normal_absolute_mean")
                  - external_value(r["policies"][second], "seam_normal_absolute_mean") for r in pair}
        summary["paired"][first + "_minus_" + second] = {
            "common_valid_n": len(pair), "common_valid_sources": [identity(r) for r in pair],
            "common_valid_metrics": compare(pair), "all_three_common_metrics": compare(common),
            "valid_rate_difference_all_sources": paired_statistics(
                [float(r["policies"][first].get("valid") is True) for r in rows],
                [float(r["policies"][second].get("valid") is True) for r in rows], seed=20261025),
            "seam_improved_sources": [identity(r) for r in pair if deltas[r["source_model_id"]] < -1e-12],
            "seam_tied_sources": [identity(r) for r in pair if abs(deltas[r["source_model_id"]]) <= 1e-12],
            "seam_worsened_sources": [identity(r) for r in pair if deltas[r["source_model_id"]] > 1e-12],
            "first_only_valid_sources": [identity(r) for r in rows if r["policies"][first].get("valid") is True and r["policies"][second].get("valid") is not True],
            "second_only_valid_sources": [identity(r) for r in rows if r["policies"][second].get("valid") is True and r["policies"][first].get("valid") is not True]}
    return summary


def run_fairing_fallback(output_dir):
    """Three equally sized prescribed-target policies, using cached formal outputs only."""
    from scripts.evaluate_geometry_comparison import _seam_normal_error
    output_dir = output_dir.resolve()
    source_path = output_dir / "fallback_results.json"
    bounded_path = ROOT / "experiments/formal_movable_neighborhood/bounded_fair_results.json"
    source, bounded = load(source_path), load(bounded_path)
    if not all(x.get("complete") and len(x["rows"]) == 110 for x in (source, bounded)):
        raise ValueError("Requires both completed 110-source cached comparisons")
    source_keys = {(r["dataset"], r["index"], r["source_model_id"]) for r in source["rows"]}
    fair_rows = {(r["dataset"], r["index"], r["source_model_id"]): r for r in bounded["rows"]}
    if source_keys != set(fair_rows) or len(source_keys) != 110:
        raise ValueError("The two cached comparisons do not have identical source identities")
    for dataset, expected in (("holdout50", 50), ("cross_category60", 60)):
        if {r["index"] for r in source["rows"] if r["dataset"] == dataset} != set(range(expected)):
            raise ValueError("Formal source cohort is incomplete")
    if len({r["source_model_id"] for r in source["rows"]}) != 110:
        raise ValueError("Bootstrap requires one row per source")
    shared = ("cgal_refine_fair", "cgal_polyline", "center_fan")
    protocol = {"schema": "fixed_bounded_fair_joint_same_pool_replay", "planned_n": 110,
                "prescribed_target": True, "end_to_end_target_selection": False,
                "source_files": {str(p.relative_to(ROOT)): sha(p) for p in (source_path, bounded_path)},
                "shared_candidates_in_tie_order": list(shared), "fourth_slots": FAIRING_FALLBACK_SLOTS,
                "selection": "Four candidates per policy; explicit successful native closure and no new intersection; finite minimum unoriented seam score; fixed input order breaks ties",
                "reporting": "Cached oriented absolute seam, triangle quality, retained motion and changed area; validity on all sources, geometry on pairwise and all-three common-valid prescribed targets",
                "statistics": "20000 paired source bootstrap draws, seed20261025; pointwise intervals; same source index draws for every endpoint",
                "normalization": "RMS and max displacement over median retained-input edge h_in; retained_rms_over_boundary_edge uses original selected-rim h; changed area over retained-input area; raw patch/moved area remain in mesh units",
                "scope": "Additional controlled analysis of existing formal sources; no new independent data, training, parameter selection or production changes",
                "new_optimization_runs": 0, "new_native_verifications": 0, "new_mesh_exports": 0,
                "existing_fallback_reproduction": "Fixed and joint selected candidates must exactly match the prior fixed-target replay"}
    rows = []
    keep = ("method", "status", "patch_path", "valid", "geometry", "verification", "selection_seam_unoriented", "safe_eligible",
            "retained_rms_over_boundary_edge", "retained_max_over_boundary_edge", "nonboundary_collar_rms_over_boundary_edge")
    for original in source["rows"]:
        key = original["dataset"], original["index"], original["source_model_id"]
        fair_row = fair_rows[key]
        if original["sample_id"] != fair_row["sample_id"]:
            raise ValueError("Formal sample identities differ")
        case, folder = external_record(original["dataset"], original["index"])
        if Path(fair_row["input_path"]).resolve() != (folder / "damaged_input.npz").resolve():
            raise ValueError("Fairing output was generated for a different prescribed target")
        candidates = {name: {k: v for k, v in item.items() if k in keep} for name, item in original["candidates"].items()}
        item = {k: v for k, v in fair_row["methods"]["bounded_biharmonic_fair"].items() if k in keep}
        item["method"] = "bounded_biharmonic_fair"
        with np.load(folder / "damaged_input.npz", allow_pickle=False) as data:
            retained, loop = data["faces"], data["selected_loop"]
        with np.load(item["patch_path"], allow_pickle=False) as data:
            item["selection_seam_unoriented"] = _seam_normal_error(data["vertices"], retained, data["patch_faces"], loop)
        candidates[item["method"]] = item
        for name, candidate in candidates.items():
            audit = candidate.get("verification", {})
            score = candidate.get("selection_seam_unoriented")
            eligible = bool(audit.get("status") == "success" and audit.get("strict_selected_loop_closure") is True
                            and audit.get("introduced_intersection") is False and score is not None and np.isfinite(score))
            if name != "bounded_biharmonic_fair" and eligible != candidate["safe_eligible"]:
                raise ValueError("Cached candidate eligibility no longer agrees with explicit verification")
            candidate["safe_eligible"] = eligible
            candidate["eligibility_reason"] = ("eligible" if eligible else "no_output_or_geometry" if candidate.get("status") != "success"
                else "native_unknown" if audit.get("status") != "success" or audit.get("introduced_intersection") is None
                else "new_intersection" if audit.get("introduced_intersection") is True
                else "not_closed" if audit.get("strict_selected_loop_closure") is not True else "nonfinite_selection_score")
        policies = {}
        for policy, variant in FAIRING_FALLBACK_SLOTS.items():
            allowed = (*shared, variant)
            eligible = [candidates[name] for name in allowed if candidates[name]["safe_eligible"]]
            chosen = min(eligible, key=lambda x: (x["selection_seam_unoriented"], allowed.index(x["method"]))) if eligible else None
            policies[policy] = ({**chosen, "accepted": True, "used_shared_fallback": chosen["method"] in shared,
                                 "slot_ineligible": not candidates[variant]["safe_eligible"]} if chosen else
                                {"accepted": False, "valid": False, "method": None, "used_shared_fallback": False,
                                 "slot_ineligible": not candidates[variant]["safe_eligible"]})
        for current, prior in (("fixed_collar", "fixed_collar"), ("joint_collar", "movable_collar")):
            a, b = policies[current], original["policies"][prior]
            if any(a.get(k) != b.get(k) for k in ("method", "accepted", "valid", "selection_seam_unoriented")):
                raise ValueError("Three-policy replay changed a previously frozen fixed/joint decision")
        rows.append({k: original[k] for k in ("dataset", "index", "sample_id", "source_model_id")})
        rows[-1].update(shape_category=fair_row["category"], candidates=candidates, policies=policies)
    summary = {dataset: fairing_fallback_summary([r for r in rows if r["dataset"] == dataset]) for dataset in EXTERNAL_DATASETS}
    summary["pooled110"] = fairing_fallback_summary(rows)
    result = {"complete": True, "protocol": protocol, "summary": summary, "rows": rows,
              "checks": {"source_ids_unique": 110, "same_target_inputs": 110, "fixed_joint_decisions_reproduced": 220}}
    destination = output_dir / "fairing_fallback_results.json"
    save(destination, result)
    print(json.dumps({"complete": True, "output": str(destination), "cohorts": {
        name: {"sources": s["source_n"], "common_all_three": s["common_valid_all_three_n"],
               "valid": {p: v["valid_n"] for p, v in s["policies"].items()},
               "joint_minus_fair_seam": s["paired"]["joint_collar_minus_bounded_fair"]["common_valid_metrics"]["seam_normal_absolute_mean"]}
        for name, s in summary.items()}}), flush=True)


def score_saved_candidate(input_path, patch_path, verification, method):
    from scripts.evaluate_geometry_comparison import _seam_normal_error
    with np.load(input_path, allow_pickle=False) as data:
        before, retained, loop = data["vertices"].astype(float), data["faces"].astype(np.int64), data["selected_loop"].astype(np.int64)
    with np.load(patch_path, allow_pickle=False) as data:
        vertices, patch = data["vertices"].astype(float), data["patch_faces"].astype(np.int64)
    item = external_measure(vertices, before, retained, patch, loop, verification, Path(patch_path))
    score = _seam_normal_error(vertices, retained, patch, loop)
    item.update(method=method, selection_seam_unoriented=score,
                safe_eligible=bool(verification.get("status") == "success" and verification.get("strict_selected_loop_closure") is True
                                   and verification.get("introduced_intersection") is False and score is not None and np.isfinite(score)))
    return item


def run_budget_fallback(output_dir):
    """Same three cached alternatives at all budgets; no geometry generation."""
    from scripts.evaluate_geometry_comparison import _seam_normal_error
    from tcmr.boundary import boundary_edges
    source_path = output_dir / "budget_results.json"
    source = load(source_path)
    if not source.get("complete") or len(source["rows"]) != 30:
        raise ValueError("Completed fixed 30-source budget experiment is required")
    oracle = load(DATA / "prospective_oracle_development201_v1.json")
    polyline = load(DATA / "prospective_oracle_cgal_polyline_development201_v1.json")
    simple = load(DATA / "prospective_simple_baselines_development201_v2.json")
    shared = ("cgal_refine_fair", "cgal_polyline", "center_fan")
    generations = {
        "cgal_refine_fair": {r["sample_id"]: r for r in oracle["methods"]["cgal_refine_fair"]["rows"]},
        "cgal_polyline": {r["sample_id"]: r for r in polyline["rows"]},
        "center_fan": {r["sample_id"]: r for r in simple["methods"]["center_fan"]["rows"]}}
    audit_paths = {"cgal_refine_fair": DATA / "prospective_oracle_development201_changed_audit_v1.json",
                   "cgal_polyline": DATA / "prospective_oracle_cgal_polyline_development201_changed_audit_v1.json",
                   "center_fan": DATA / "prospective_simple_baselines_development201_changed_audit_v1.json"}
    audits = {m: {r["sample_id"]: r for r in load(path)["rows"] if r.get("method") == m} for m, path in audit_paths.items()}
    protocol = {"schema": "budget_safe_fallback", "development_only": True, "prescribed_target": True,
                "end_to_end_target_selection": False, "source_budget_results": str(source_path.resolve()),
                "source_budget_sha256": sha(source_path), "source_n": 30, "budgets": [10, 30, 100],
                "shared_candidates": list(shared), "variant": "one fixed-collar or movable-collar output at the stated budget",
                "selection": "Frozen strict-closure and explicit no-new-intersection eligibility, then minimum unoriented seam with fixed candidate order",
                "target_correctness": "The original construction target is prescribed. Correct-safe additionally requires original non-target boundary edges to remain open and no new topology defect",
                "statistics": "20000 source-bootstrap draws, seed20261025; every source remains in endpoint denominators; geometry on stated common-correct-safe populations",
                "new_optimization_runs": 0, "new_native_checks": 0,
                "timing": "Historical heterogeneous times excluded; no matched end-to-end runtime claim"}
    names = [f"{mode}_{steps}" for steps in (10, 30, 100) for mode in ("patch_only", "joint")]
    rows = []
    for original in source["rows"]:
        sid = original["sample_id"]
        input_path = Path(original["input_path"])
        with np.load(input_path, allow_pickle=False) as data:
            before, retained, loop = data["vertices"].astype(float), data["faces"].astype(np.int64), data["selected_loop"].astype(np.int64)
        original_boundaries = {tuple(map(int, e)) for e in boundary_edges(retained)}
        target_edges = {tuple(sorted((int(a), int(b)))) for a, b in zip(loop, np.roll(loop, -1))}
        other_boundary_edges = original_boundaries - target_edges
        candidates = {}
        for method in shared:
            record = generations[method][sid]
            audit = audits[method][sid]
            patch_path = Path(record.get("patch_path", ""))
            if record.get("status") != "success" or not patch_path.is_file():
                candidates[method] = {"method": method, "valid": False, "safe_eligible": False, "status": record.get("status", "missing")}
            else:
                candidates[method] = score_saved_candidate(input_path, patch_path, audit, method)
        for name in names:
            item = dict(original["methods"][name])
            with np.load(item["patch_path"], allow_pickle=False) as data:
                score = _seam_normal_error(data["vertices"], retained, data["patch_faces"], loop)
            audit = item["verification"]
            item.update(method=name, selection_seam_unoriented=score,
                        safe_eligible=bool(audit.get("status") == "success" and audit.get("strict_selected_loop_closure") is True
                                           and audit.get("introduced_intersection") is False and score is not None and np.isfinite(score)))
            candidates[name] = item
        for item in candidates.values():
            if not item.get("patch_path"):
                item["prescribed_target_correct_safe"] = False
                continue
            with np.load(item["patch_path"], allow_pickle=False) as data:
                pf = data["patch_faces"].astype(np.int64)
            patch_edges = {tuple(map(int, e)) for e in np.sort(np.vstack((pf[:, [0, 1]], pf[:, [1, 2]], pf[:, [2, 0]])), axis=1)}
            item["other_boundaries_preserved"] = not bool(other_boundary_edges & patch_edges)
            item["prescribed_target_correct_safe"] = bool(item["valid"] and item["other_boundaries_preserved"])
        policies = {}
        for name in names:
            allowed = (*shared, name)
            eligible = [candidates[m] for m in allowed if candidates[m]["safe_eligible"]]
            chosen = min(eligible, key=lambda x: (x["selection_seam_unoriented"], allowed.index(x["method"]))) if eligible else None
            policies[name] = ({**chosen, "accepted": True, "used_shared_fallback": chosen["method"] in shared,
                               "variant_unsafe": not candidates[name]["safe_eligible"]} if chosen else
                              {"accepted": False, "valid": False, "prescribed_target_correct_safe": False,
                               "method": None, "variant_unsafe": True, "used_shared_fallback": False,
                               "unknown_among_candidates": any(candidates[m].get("verification", {}).get("introduced_intersection") is None for m in allowed)})
        rows.append({"sample_id": sid, "source_model_id": original["source_model_id"], "candidates": candidates, "policies": policies})
        print(json.dumps({"budget_fallback_scored": len(rows), "total": 30}), flush=True)
    metrics = ("seam_normal_absolute_mean", "seam_normal_absolute_p95", "triangle_quality.mean", "triangle_quality.p05", "retained_rms_over_boundary_edge")
    common = [r for r in rows if all(r["policies"][name]["prescribed_target_correct_safe"] for name in names)]
    summary = {"source_n": 30, "methods": {}, "common_correct_safe_all_six_n": len(common),
               "common_correct_safe_source_ids": [r["source_model_id"] for r in common], "common_means": {}, "paired_by_budget": {}, "paired_budget_changes": {}}
    for name in names:
        items = [r["policies"][name] for r in rows]
        correct = [x for x in items if x["prescribed_target_correct_safe"]]
        summary["methods"][name] = {
            "denominator": 30, "accepted_n": sum(x["accepted"] for x in items), "correct_safe_n": len(correct),
            "wrong_accepted_n": sum(x["accepted"] and not x["prescribed_target_correct_safe"] for x in items),
            "abstain_n": sum(not x["accepted"] for x in items),
            "abstention_with_unknown_n": sum(not x["accepted"] and x["unknown_among_candidates"] for x in items),
            "fallback_selected_n": sum(x["used_shared_fallback"] for x in items),
            "unsafe_variant_rescued_n": sum(x["variant_unsafe"] and x["prescribed_target_correct_safe"] for x in items),
            "coverage": source_mean_interval([float(x["accepted"]) for x in items]),
            "prescribed_target_correct_safe_rate": source_mean_interval([float(x["prescribed_target_correct_safe"]) for x in items]),
            "correct_safe_means": {k: source_mean_interval([external_value(x, k) for x in correct]) for k in metrics},
            "selected_counts": {m: sum(x.get("method") == m for x in items) for m in (*shared, name)}}
        summary["common_means"][name] = {k: source_mean_interval([external_value(r["policies"][name], k) for r in common]) for k in metrics}
    def compare(group, first, second):
        return {k: paired_statistics([external_value(r["policies"][first], k) for r in group],
                                     [external_value(r["policies"][second], k) for r in group], seed=20261025) for k in metrics}
    for steps in (10, 30, 100):
        first, second = f"patch_only_{steps}", f"joint_{steps}"
        pair = [r for r in rows if r["policies"][first]["prescribed_target_correct_safe"] and r["policies"][second]["prescribed_target_correct_safe"]]
        summary["paired_by_budget"][str(steps)] = {"common_pair_n": len(pair), "fixed_minus_movable": compare(pair, first, second),
            "all_six_common": compare(common, first, second), "correct_safe_movable_minus_fixed": paired_statistics(
                [float(r["policies"][second]["prescribed_target_correct_safe"]) for r in rows],
                [float(r["policies"][first]["prescribed_target_correct_safe"]) for r in rows], seed=20261025)}
    for mode in ("patch_only", "joint"):
        for steps in (30, 100):
            summary["paired_budget_changes"][f"{mode}_{steps}_minus_10"] = compare(common, f"{mode}_{steps}", f"{mode}_10")
    summary["fixed100_minus_joint10_all_six_common"] = compare(common, "patch_only_100", "joint_10")
    result = {"complete": True, "protocol": protocol, "summary": summary, "rows": rows}
    save(output_dir / "budget_fallback_results.json", result)
    print(json.dumps({"complete": True, "budget_fallback_summary": summary}), flush=True)


def fixed_inference_record(input_path, measured, verification_path):
    from tcmr.integrated_repair import content_key, legacy_dimensionless_energy
    from scripts.evaluate_geometry_comparison import _seam_normal_error
    with np.load(input_path, allow_pickle=False) as data:
        before, retained, loop = data["vertices"].astype(float), data["faces"].astype(np.int64), data["selected_loop"].astype(np.int64)
    with np.load(Path(input_path).parent / "cgal_refine_fair.npz", allow_pickle=False) as data:
        initial, patch = data["vertices"].astype(float), data["patch_faces"].astype(np.int64)
    with np.load(measured["patch_path"], allow_pickle=False) as data:
        vertices, actual_patch = data["vertices"].astype(float), data["patch_faces"].astype(np.int64)
    if not np.array_equal(actual_patch, patch) or not np.array_equal(vertices[:len(before)], before):
        raise ValueError("Fixed-collar candidate changed retained coordinates or initializer topology")
    energy = legacy_dimensionless_energy(vertices, initial, retained, patch, loop, len(before))
    score = _seam_normal_error(vertices, retained, patch, loop)
    verification = measured["verification"]
    return {"method": "bounded_joint10", "slot_variant": "fixed_collar", "status": measured["status"],
            "patch_path": measured["patch_path"], "input_geometry_key": content_key(before, retained, loop),
            "seam_normal_error": score, **energy, "verification_path": str(Path(verification_path).resolve()),
            "verification_status": verification["status"], "introduced_intersection": verification["introduced_intersection"],
            "strict_selected_loop_closure": verification["strict_selected_loop_closure"],
            "safe_eligible": bool(verification["status"] == "success" and verification["strict_selected_loop_closure"] is True
                                  and verification["introduced_intersection"] is False)}


def fixed_loop_worker(input_path, destination):
    import os
    import subprocess
    import torch
    from tcmr.joint_optimizer import optimize_joint_collar
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with np.load(input_path, allow_pickle=False) as data:
        before, retained, loop = data["vertices"].astype(float), data["faces"].astype(np.int64), data["selected_loop"].astype(np.int64)
    with np.load(input_path.parent / "cgal_refine_fair.npz", allow_pickle=False) as data:
        initial, patch = data["vertices"].astype(float), data["patch_faces"].astype(np.int64)
    started = time.perf_counter()
    vertices, info = optimize_joint_collar(initial, retained, patch, loop, len(before), mode="patch_only", steps=10,
                                         learning_rate=.02, retained_displacement_budget=.02, rings=2)
    optimizer_seconds = time.perf_counter() - started
    patch_path = destination.with_suffix(".npz")
    np.savez_compressed(patch_path, vertices=vertices, patch_faces=patch)
    verification_path = destination.with_name(destination.stem + "_verification.json")
    process = subprocess.run([sys.executable, str(ROOT / "scripts/cgal_changed_surface_audit.py"), "--input", str(input_path),
                              "--patch", str(patch_path), "--output", str(verification_path)], cwd=ROOT,
                             capture_output=True, text=True, timeout=100,
                             env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if process.returncode or not verification_path.exists():
        raise RuntimeError(f"Native verification infrastructure failed: {process.returncode} {process.stderr[-1000:]}")
    measured = external_measure(vertices, before, retained, patch, loop, load(verification_path), patch_path)
    measured.update(optimizer=info, optimizer_seconds=optimizer_seconds, verification_path=str(verification_path.resolve()))
    result = {"input_path": str(input_path.resolve()), "metric_record": measured,
              "inference_candidate": fixed_inference_record(input_path, measured, verification_path)}
    save(destination, result)
    print(json.dumps({"fixed_candidate_completed": str(destination), "valid": measured["valid"]}), flush=True)


def replay_candidate_pool(prediction, fixed_candidates, strategy, cue):
    import copy
    import math
    from tcmr.integrated_repair import POOL, select_policies
    loops = copy.deepcopy(prediction["loops"])
    for row in loops:
        if strategy == "fixed_collar":
            row["candidates"][-1] = fixed_candidates[str(Path(row["input_path"]).resolve())]
        finite = [c for c in row["candidates"] if c.get("seam_normal_error") is not None and c.get("dimensionless_legacy_energy") is not None
                  and math.isfinite(c["seam_normal_error"]) and math.isfinite(c["dimensionless_legacy_energy"])]
        representative = min(finite, key=lambda c: (c["seam_normal_error"], POOL.index(c["method"]))) if finite else None
        penalty = None
        if representative:
            energy = representative["dimensionless_legacy_energy"]
            penalty = representative["seam_normal_error"] / (np.pi / 2) + min(representative["collar_rms_over_boundary_edge"], 1) + energy / (1 + energy)
        row.update(geometry_representative=representative["method"] if representative else None,
                   fixed_geometry_penalty=penalty, identity_score=row["prior_logit"] - penalty if penalty is not None else None)
    return {"loops": loops, "decisions": select_policies(loops, cue=np.asarray(cue, float)), "target_labels_read": False}


def predicted_fallback_summary(rows):
    names = ("fixed_collar", "movable_collar")
    summary = {"source_n": len(rows), "methods": {}, "target_lock_changed_n": 0, "same_target_common_correct_safe_n": 0}
    for name in names:
        items = [r["scored"][name] for r in rows]
        summary["methods"][name] = {"correct_safe_n": sum(x["correct_safe"] for x in items),
            "wrong_accepted_n": sum(x["wrong_accepted"] for x in items), "abstain_n": sum(x["decision"] == "null" for x in items),
            "correct_safe_rate": source_mean_interval([float(x["correct_safe"]) for x in items])}
    common = []
    transitions = []
    for row in rows:
        left, right = [row["predictions"][n]["decisions"]["prior_locked_pool_with_joint"] for n in names]
        if left["locked_loop_id"] != right["locked_loop_id"]:
            summary["target_lock_changed_n"] += 1
            transitions.append({"sample_id": row["sample_id"], "fixed_locked": left["locked_loop_id"], "movable_locked": right["locked_loop_id"],
                                "fixed_correct_safe": row["scored"][names[0]]["correct_safe"], "movable_correct_safe": row["scored"][names[1]]["correct_safe"]})
        if all(row["scored"][n]["correct_safe"] for n in names) and left["selected_loop_id"] == right["selected_loop_id"]:
            common.append(row)
    summary["target_changes"] = transitions
    summary["same_target_common_correct_safe_n"] = len(common)
    summary["paired_correct_safe_movable_minus_fixed"] = paired_statistics([float(r["scored"][names[1]]["correct_safe"]) for r in rows],
        [float(r["scored"][names[0]]["correct_safe"]) for r in rows], seed=20261025)
    metrics = ("seam_normal_absolute_mean", "seam_normal_absolute_p95", "retained_displacement_rms_over_edge")
    summary["paired_same_target_fixed_minus_movable"] = {k: paired_statistics(
        [r["scored"][names[0]]["geometry"][k] for r in common], [r["scored"][names[1]]["geometry"][k] for r in common], seed=20261025) for k in metrics}
    for k in ("mean", "p05"):
        summary["paired_same_target_fixed_minus_movable"]["triangle_quality." + k] = paired_statistics(
            [r["scored"][names[0]]["geometry"]["triangle_quality"][k] for r in common],
            [r["scored"][names[1]]["geometry"]["triangle_quality"][k] for r in common], seed=20261025)
    return summary


def run_predicted_fallback(output_dir):
    import os
    import subprocess
    from scripts.evaluate_external_benchmark import unified_method_row
    source_path = output_dir / "results.json"
    source = load(source_path)
    fixed_candidates = {}
    for row in source["rows"]:
        key = str(Path(row["input_path"]).resolve())
        verification_path = output_dir / f"{row['dataset']}_{row['index']:03d}_verification.json"
        fixed_candidates[key] = fixed_inference_record(Path(key), row["methods"]["patch_only"], verification_path)
    queue = []
    score_cases = {}
    for dataset, root in EXTERNAL_DATASETS.items():
        for original in load(root / "protocol.json")["cases"]:
            index = int(original["index"])
            score_cases[(dataset, index)] = original
            case = {k: original[k] for k in ("index", "sample_id", "source_model_id", "category", "input_path", "cue")}
            prediction = load(root / "work" / f"case{index:03d}" / "inference.json")
            queue.append((dataset, case, prediction))
    protocol = {"schema": "full_target_policy_replay", "source_n": 110, "shortlisted_loops": sum(len(p["loops"]) for _, _, p in queue),
                "source_frozen_collar_results": str(source_path.resolve()), "source_frozen_collar_sha256": sha(source_path),
                "strategies": ["fixed_collar", "movable_collar"], "frozen_prior": "Exactly cached ranked top-two loop IDs, probabilities and logits; input-only encoder and ranker unchanged",
                "candidate_pool": ["cgal_refine_fair", "cgal_polyline", "center_fan", "ten_step_refinement"],
                "identity": "Recompute each strategy's finite minimum-seam representative and geometry penalty, then invoke the frozen cue-geometry gate and select_policies; no prescribed-target lock",
                "fixed_slot_internal_name": "bounded_joint10 retained solely for frozen function compatibility; slot_variant identifies fixed_collar",
                "ground_truth": "Target and distractor annotations enter only after both strategies' predictions have been generated and saved",
                "new_work": "Only missing fixed-collar ten-step outputs on shortlisted loops; same initialization, topology, objective, lr.02, float64 and native changed-surface validation",
                "workers": 1, "torch_cpu_threads": 2, "optimizer_sha256": sha(OPTIMIZER),
                "statistics": "20000 paired source bootstraps, seed20261025; complete endpoint denominators50/60; geometry common-correct-safe same selected target"}
    protocol_path = output_dir / "predicted_fallback_protocol.json"
    if protocol_path.exists() and load(protocol_path) != protocol:
        raise ValueError("Predicted replay protocol differs")
    save(protocol_path, protocol)
    destination = output_dir / "shortlist_fixed_collar"
    destination.mkdir(exist_ok=True)
    completed = 0
    for dataset, case, prediction in queue:
        for row in prediction["loops"]:
            key = str(Path(row["input_path"]).resolve())
            if key in fixed_candidates:
                continue
            path = destination / f"{dataset}_{case['index']:03d}_loop{row['loop_id']}.json"
            if not path.exists():
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--fixed-loop-input", key,
                    "--fixed-loop-output", str(path)], cwd=ROOT, capture_output=True, text=True, timeout=120,
                    env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if process.returncode or not path.exists():
                    raise RuntimeError(f"Missing-loop candidate infrastructure failure: {process.returncode} {process.stderr[-1500:]}")
            value = load(path)
            if Path(value["input_path"]).resolve() != Path(key):
                raise ValueError("Saved fixed candidate belongs to another loop")
            fixed_candidates[key] = value["inference_candidate"]
            completed += 1
            print(json.dumps({"other_loop_fixed_completed": completed, "planned": 110}), flush=True)
    rows = []
    for dataset, case, prediction in queue:
        predictions = {name: replay_candidate_pool(prediction, fixed_candidates, name, case["cue"]) for name in ("fixed_collar", "movable_collar")}
        actual, cached = predictions["movable_collar"]["decisions"]["prior_locked_pool_with_joint"], prediction["decisions"]["prior_locked_pool_with_joint"]
        if any(actual[k] != cached[k] for k in ("locked_loop_id", "selected_loop_id", "decision", "chosen_method")):
            raise ValueError("Frozen movable-collar replay fails to reproduce original decision")
        rows.append({"dataset": dataset, "index": case["index"], "sample_id": case["sample_id"], "source_model_id": case["source_model_id"],
                     "predictions": predictions})
    result_path = output_dir / "predicted_fallback_results.json"
    save(result_path, {"complete": False, "predictions_complete_before_scoring": True, "protocol": protocol, "rows": rows})
    for row in rows:
        labels = score_cases[(row["dataset"], row["index"])]
        row["scored"] = {name: unified_method_row(labels, pred, "tcmr") for name, pred in row["predictions"].items()}
    summaries = {dataset: predicted_fallback_summary([r for r in rows if r["dataset"] == dataset]) for dataset in EXTERNAL_DATASETS}
    summaries["pooled110"] = predicted_fallback_summary(rows)
    save(result_path, {"complete": True, "predictions_complete_before_scoring": True, "protocol": protocol, "summary": summaries, "rows": rows})
    print(json.dumps({"complete": True, "predicted_fallback_summary": summaries}), flush=True)


def seam_ablation_protocol(output_dir):
    """One joint objective intervention on the existing development cohort."""
    path = output_dir / "budget_protocol.json"
    existing = load(path)
    weights = dict(DEFAULT_WEIGHTS, normal=0., normal_variation=0.)
    protocol = {"schema": "paired_seam_objective_ablation", "development_only": True,
        "prescribed_target": True, "end_to_end_evaluation": False,
        "source_protocol": str(path.resolve()), "source_protocol_sha256": sha(path),
        "records": existing["records"], "source_n": 30,
        "arms": {"full": DEFAULT_WEIGHTS, "without_seam_terms": weights},
        "intervention": "Only normal and normal_variation weights are zeroed through the frozen optimizer's explicit weights argument; all eight remaining terms unchanged",
        "steps": 10, "learning_rate": .02, "retained_budget_over_boundary_edge": .02,
        "collar_rings": 2, "dtype": "float64", "optimizer_sha256": sha(OPTIMIZER),
        "full_reuse": "Cached joint_10 geometry and native verification; first fixed development source recomputed once with explicit full weights to confirm exact reproduction",
        "candidate_pool": ["cgal_refine_fair", "cgal_polyline", "center_fan", "arm"],
        "pool_selection": "Frozen native eligibility then minimum unsigned seam; prescribed target unchanged",
        "statistics": "20000 paired source bootstrap draws, seed20261025; all30 in validity denominators; geometry on stated common-valid/correct-safe sources",
        "runs": "Exactly one no-seam output and isolated native verification per source; no parameter search; one source worker at a time",
        "optimizer_modified": False, "new_full_native_verifications": 0}
    if len(existing["records"]) != 30 or len({r["source_model_id"] for r in existing["records"]}) != 30:
        raise ValueError("A fixed 30-source distinct development cohort is required")
    destination = output_dir / "seam_ablation_protocol.json"
    if destination.exists() and load(destination) != protocol:
        raise ValueError("Seam ablation protocol changed")
    save(destination, protocol)
    return protocol


def seam_ablation_worker(position, output_dir):
    import os
    import subprocess
    import torch
    from tcmr.joint_optimizer import optimize_joint_collar
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    protocol = load(output_dir / "seam_ablation_protocol.json")
    record = protocol["records"][position]
    source = load(output_dir / "budget_results.json")
    cached = next(r for r in source["rows"] if r["sample_id"] == record["sample_id"])["methods"]["joint_10"]
    with np.load(record["input_path"], allow_pickle=False) as data:
        before, retained, loop = data["vertices"].astype(float), data["faces"].astype(np.int64), data["selected_loop"].astype(np.int64)
    with np.load(record["initializer_path"], allow_pickle=False) as data:
        initial, patch = data["vertices"].astype(float), data["patch_faces"].astype(np.int64)
    reproduction = None
    if position == 0:
        reproduced, info = optimize_joint_collar(initial, retained, patch, loop, len(before), mode="joint", steps=10,
            learning_rate=.02, retained_displacement_budget=.02, rings=2, weights=protocol["arms"]["full"])
        with np.load(cached["patch_path"], allow_pickle=False) as data:
            reference = data["vertices"].astype(float)
            same_faces = np.array_equal(data["patch_faces"], patch)
        reproduction = {"bitwise_equal_vertices": bool(np.array_equal(reproduced, reference)),
            "max_coordinate_absolute_difference": float(np.max(np.abs(reproduced-reference))),
            "same_patch_connectivity": bool(same_faces), "explicit_weights": info["weights"],
            "new_verification": False}
        if not reproduction["bitwise_equal_vertices"] or not same_faces:
            raise ValueError("Explicit full objective did not exactly reproduce the frozen cached geometry")
    started = time.perf_counter()
    vertices, info = optimize_joint_collar(initial, retained, patch, loop, len(before), mode="joint", steps=10,
        learning_rate=.02, retained_displacement_budget=.02, rings=2, weights=protocol["arms"]["without_seam_terms"])
    optimize_seconds = time.perf_counter() - started
    destination = output_dir / "seam_ablation_meshes"
    destination.mkdir(exist_ok=True)
    patch_path = destination / f"{position:03d}_without_seam_terms.npz"
    audit_path = destination / f"{position:03d}_verification.json"
    np.savez_compressed(patch_path, vertices=vertices, patch_faces=patch)
    started = time.perf_counter()
    try:
        process = subprocess.run([sys.executable, str(ROOT / "scripts/cgal_changed_surface_audit.py"),
            "--input", record["input_path"], "--patch", str(patch_path), "--output", str(audit_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=100,
            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if process.returncode or not audit_path.exists():
            raise RuntimeError(f"Native verifier returned {process.returncode}: {process.stderr[-1000:]}")
        audit = load(audit_path)
    except Exception as exc:
        audit = {"status": "verification_failure", "error": repr(exc), "introduced_intersection": None,
                 "strict_selected_loop_closure": False}
        save(audit_path, audit)
    item = external_measure(vertices, before, retained, patch, loop, audit, patch_path)
    item.update(optimizer=info, optimizer_seconds=optimize_seconds, verification_seconds=time.perf_counter()-started,
                verification_path=str(audit_path), reused_exact_output=False)
    row = {"position": position, "sample_id": record["sample_id"], "source_model_id": record["source_model_id"],
        "input_path": record["input_path"], "initializer_path": record["initializer_path"],
        "full_reproduction": reproduction, "methods": {"full": cached, "without_seam_terms": item}}
    print("SEAM_ABLATION_RESULT=" + json.dumps(row, allow_nan=False), flush=True)


def run_seam_ablation(output_dir):
    import os
    import subprocess
    from tcmr.boundary import boundary_edges
    from scripts.evaluate_geometry_comparison import _seam_normal_error
    protocol = seam_ablation_protocol(output_dir)
    result_path = output_dir / "seam_ablation_results.json"
    prior = load(result_path) if result_path.exists() else {}
    if prior and prior["protocol"] != protocol:
        raise ValueError("Saved seam ablation protocol differs")
    rows = {r["position"]: r for r in prior.get("rows", [])}
    for position in range(30):
        if position in rows:
            continue
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--seam-ablation-worker", str(position),
            "--output-dir", str(output_dir)], cwd=ROOT, capture_output=True, text=True, timeout=160,
            env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = [s.removeprefix("SEAM_ABLATION_RESULT=") for s in process.stdout.splitlines() if s.startswith("SEAM_ABLATION_RESULT=")]
        if process.returncode or len(lines) != 1:
            raise RuntimeError(f"Seam ablation infrastructure failure at {position}: {process.returncode} {process.stderr[-1500:]}")
        rows[position] = json.loads(lines[0])
        save(result_path, {"complete": False, "protocol": protocol, "rows": [rows[i] for i in sorted(rows)]})
        print(json.dumps({"seam_ablation_completed": len(rows), "planned": 30,
            "valid": {m: v["valid"] for m,v in rows[position]["methods"].items()}}), flush=True)
    ordered = [rows[i] for i in range(30)]
    cached_fallback = {r["sample_id"]: r for r in load(output_dir / "budget_fallback_results.json")["rows"]}
    shared = ("cgal_refine_fair", "cgal_polyline", "center_fan")
    for row in ordered:
        with np.load(row["input_path"], allow_pickle=False) as data:
            retained, loop = data["faces"].astype(np.int64), data["selected_loop"].astype(np.int64)
        target_edges = {tuple(sorted(map(int, e))) for e in zip(loop, np.roll(loop,-1))}
        other_edges = {tuple(map(int, e)) for e in boundary_edges(retained)} - target_edges
        candidates = {m: cached_fallback[row["sample_id"]]["candidates"][m] for m in shared}
        for name, item in row["methods"].items():
            with np.load(item["patch_path"], allow_pickle=False) as data:
                vertices, patch = data["vertices"], data["patch_faces"].astype(np.int64)
            score = _seam_normal_error(vertices, retained, patch, loop)
            audit = item["verification"]
            patch_edges = {tuple(map(int,e)) for e in np.sort(np.vstack((patch[:,[0,1]],patch[:,[1,2]],patch[:,[2,0]])),axis=1)}
            item.update(method=name, selection_seam_unoriented=score,
                safe_eligible=bool(audit.get("status")=="success" and audit.get("strict_selected_loop_closure") is True
                    and audit.get("introduced_intersection") is False and score is not None and np.isfinite(score)),
                other_boundaries_preserved=not bool(other_edges & patch_edges),
                prescribed_target_correct_safe=bool(item["valid"] and not (other_edges & patch_edges)))
            candidates[name] = item
        row["policies"] = {}
        for name in row["methods"]:
            allowed = (*shared, name)
            eligible = [candidates[m] for m in allowed if candidates[m]["safe_eligible"]]
            chosen = min(eligible, key=lambda x: (x["selection_seam_unoriented"], allowed.index(x["method"]))) if eligible else None
            row["policies"][name] = ({**chosen, "accepted": True, "fallback": chosen["method"] in shared} if chosen else
                {"accepted": False, "valid": False, "prescribed_target_correct_safe": False, "method": None, "fallback": False})
        old_choice = cached_fallback[row["sample_id"]]["policies"]["joint_10"]
        new_choice = row["policies"]["full"]
        expected = "full" if old_choice.get("method")=="joint_10" else old_choice.get("method")
        if new_choice["method"] != expected or new_choice["accepted"] != old_choice["accepted"]:
            raise ValueError("Full objective policy failed to reproduce frozen budget fallback")
    metrics = ("seam_normal_absolute_mean", "seam_normal_absolute_p95", "triangle_quality.mean", "triangle_quality.p05", "retained_rms_over_boundary_edge")
    summary = {"source_n": 30, "single_candidate": {}, "same_pool": {}}
    for scope, key in (("single_candidate", "methods"), ("same_pool", "policies")):
        summary[scope] = {"arms": {}}
        correct_key = "prescribed_target_correct_safe"
        for name in ("full", "without_seam_terms"):
            items = [r[key][name] for r in ordered]
            summary[scope]["arms"][name] = {"denominator": 30,
                "valid_n": sum(x.get(correct_key) is True for x in items),
                "intersection_n": sum(x.get("verification",{}).get("introduced_intersection") is True for x in items),
                "unknown_n": sum(bool(x.get("verification")) and x["verification"].get("introduced_intersection") is None for x in items),
                "abstain_n": sum(x.get("accepted") is False for x in items),
                "fallback_n": sum(x.get("fallback") is True for x in items)}
        common = [r for r in ordered if all(r[key][n].get(correct_key) is True for n in ("full","without_seam_terms"))]
        summary[scope].update(common_correct_safe_n=len(common),
            common_source_ids=[r["source_model_id"] for r in common],
            full_minus_without_seam_terms={metric: paired_statistics([external_value(r[key]["full"],metric) for r in common],
                [external_value(r[key]["without_seam_terms"],metric) for r in common],seed=20261025) for metric in metrics},
            paired_valid_full_minus_without=paired_statistics([float(r[key]["full"].get(correct_key) is True) for r in ordered],
                [float(r[key]["without_seam_terms"].get(correct_key) is True) for r in ordered], seed=20261025))
    if sha(OPTIMIZER) != protocol["optimizer_sha256"]:
        raise ValueError("Frozen production optimizer changed")
    save(result_path, {"complete": True, "protocol": protocol, "summary": summary, "rows": ordered})
    print(json.dumps({"complete": True, "seam_ablation_summary": summary}), flush=True)


def seam_edge_diagnostic(vertices, retained, patch, loop):
    """Same edge orientation convention as unified seam_normal_metrics."""
    from scripts.evaluate_geometry_comparison import _unit_face_normals, _seam_normal_error
    loop_edges = {tuple(sorted(map(int,e))) for e in zip(loop,np.roll(loop,-1))}
    maps = []
    for faces in (retained, patch):
        mapping = {}
        for fi, face in enumerate(faces):
            for a,b in zip(face,np.roll(face,-1)):
                edge=tuple(sorted((int(a),int(b))))
                if edge in loop_edges:
                    mapping.setdefault(edge,(fi,int(a),int(b)))
        maps.append(mapping)
    rn,pn = _unit_face_normals(vertices,retained),_unit_face_normals(vertices,patch)
    angles=[]
    for a,b in zip(loop,np.roll(loop,-1)):
        edge=tuple(sorted((int(a),int(b))))
        if edge not in maps[0] or edge not in maps[1]:
            continue
        r,ra,rb=maps[0][edge]
        p,pa,pb=maps[1][edge]
        if np.linalg.norm(rn[r]) < .99 or np.linalg.norm(pn[p]) < .99:
            continue
        nr=rn[r]*(1 if (ra,rb)==(int(a),int(b)) else -1)
        npatch=pn[p]*(-1 if (pa,pb)==(int(a),int(b)) else 1)
        tangent=vertices[int(b)]-vertices[int(a)]
        en=np.linalg.norm(tangent)
        if en<=1e-12:
            continue
        angles.append(abs(float(np.arctan2(np.dot(np.cross(nr,npatch),tangent/en),np.dot(nr,npatch)))))
    values=np.asarray(angles,float)
    return {"seam_edge_n": len(values), "oriented_mean": float(values.mean()) if len(values) else None,
        "oriented_max": float(values.max()) if len(values) else None,
        "edges_above_pi_over_2_n": int((values>np.pi/2).sum()),
        "edge_fraction_above_pi_over_2": float(np.mean(values>np.pi/2)) if len(values) else None,
        "unsigned_mean": _seam_normal_error(vertices,retained,patch,loop)}


def run_selection_score_diagnostic(output_dir):
    """Cache-only, prescribed-target intervention; frozen deployment unchanged."""
    source_path=output_dir / "fallback_results.json"
    source=load(source_path)
    if not source.get("complete") or len(source["rows"])!=110:
        raise ValueError("The complete110 prescribed-target candidate cache is required")
    protocol={"schema":"selection_score_diagnostic", "source_results":str(source_path.resolve()),
        "source_sha256":sha(source_path), "source_n":110, "prescribed_target":True,
        "end_to_end_evaluation":False, "deployed_method_modified":False,
        "pool":["cgal_refine_fair","cgal_polyline","center_fan","joint"],
        "intervention":"Frozen eligible candidates and target; replace minimum acos(abs(dot)) by minimum unified oriented-absolute mean only, tie order unchanged",
        "no_new_geometry_or_native_checks":True,
        "statistics":"20000 paired source bootstrap draws, seed20261025; cohorts separately; common valid accepted geometry",
        "interpretation":"Conditional candidate selection diagnostic. Agreement with the frozen predicted target is recorded; no target reranking or end-to-end counterfactual accuracy claim"}
    path=output_dir/"selection_score_diagnostic.json"
    rows=[]
    allowed=protocol["pool"]
    for original in source["rows"]:
        case,folder=external_record(original["dataset"],original["index"])
        with np.load(folder/"damaged_input.npz",allow_pickle=False) as data:
            retained,loop=data["faces"].astype(np.int64),data["selected_loop"].astype(np.int64)
        candidates={}
        for name in allowed:
            item=original["candidates"][name]
            entry={"method":name,"safe_eligible":item["safe_eligible"],"valid":item.get("valid",False)}
            if item.get("patch_path"):
                with np.load(item["patch_path"],allow_pickle=False) as data:
                    diagnostic=seam_edge_diagnostic(data["vertices"].astype(float),retained,data["patch_faces"].astype(np.int64),loop)
                cached_mean=external_value(item,"seam_normal_absolute_mean")
                if not np.isclose(diagnostic["oriented_mean"],cached_mean,rtol=1e-8,atol=1e-10):
                    raise ValueError("Per-edge diagnostic disagrees with cached unified oriented mean")
                if not np.isclose(diagnostic["unsigned_mean"],item["selection_seam_unoriented"],rtol=1e-8,atol=1e-10):
                    raise ValueError("Unsigned diagnostic disagrees with frozen cached score")
                entry.update(diagnostic, geometry=item["geometry"], retained_rms_over_boundary_edge=item["retained_rms_over_boundary_edge"])
            candidates[name]=entry
        eligible=[candidates[n] for n in allowed if candidates[n]["safe_eligible"]]
        policies={}
        ranks={}
        for name,score in (("unsigned","unsigned_mean"),("oriented","oriented_mean")):
            ranked=sorted(eligible,key=lambda x:(x[score],allowed.index(x["method"])))
            ranks[name]=[x["method"] for x in ranked]
            policies[name]={**ranked[0],"accepted":True} if ranked else {"method":None,"accepted":False,"valid":False}
        if policies["unsigned"]["method"] != original["policies"]["movable_collar"]["method"]:
            raise ValueError("Cached unsigned policy was not reproduced")
        rows.append({k:original[k] for k in ("dataset","index","sample_id","source_model_id","matches_frozen_predicted_target")})
        rows[-1].update(candidates=candidates,policies=policies,rankings=ranks,
            selected_changed=policies["unsigned"]["method"]!=policies["oriented"]["method"],
            full_rank_changed=ranks["unsigned"]!=ranks["oriented"])
    def summarize(group):
        common=[r for r in group if all(r["policies"][n].get("valid") is True for n in ("unsigned","oriented"))]
        summary={"source_n":len(group),"rank_changed_n":sum(r["full_rank_changed"] for r in group),
            "chosen_changed_n":sum(r["selected_changed"] for r in group),"common_valid_n":len(common),
            "policies":{},"paired_oriented_minus_unsigned":{}}
        for name in ("unsigned","oriented"):
            items=[r["policies"][name] for r in group]
            accepted=[x for x in items if x["accepted"]]
            summary["policies"][name]={"accepted_n":len(accepted),"valid_n":sum(x.get("valid") is True for x in items),
                "selected_mean_above_pi_over_2_n":sum(x["oriented_mean"]>np.pi/2 for x in accepted),
                "selected_has_edge_above_pi_over_2_n":sum(x["edges_above_pi_over_2_n"]>0 for x in accepted),
                "selected_mean_edge_fraction_above_pi_over_2":float(np.mean([x["edge_fraction_above_pi_over_2"] for x in accepted])) if accepted else None}
        for metric in ("seam_normal_absolute_mean","seam_normal_absolute_p95","triangle_quality.mean","triangle_quality.p05","retained_rms_over_boundary_edge"):
            summary["paired_oriented_minus_unsigned"][metric]=paired_statistics(
                [external_value(r["policies"]["oriented"],metric) for r in common],
                [external_value(r["policies"]["unsigned"],metric) for r in common],seed=20261025)
        summary["changed_sources"]=[{k:r[k] for k in ("dataset","index","sample_id","source_model_id")} for r in group if r["selected_changed"]]
        safe=[c for r in group for c in r["candidates"].values() if c["safe_eligible"]]
        summary["eligible_candidate_n"]=len(safe)
        summary["eligible_candidate_mean_above_pi_over_2_n"]=sum(c["oriented_mean"]>np.pi/2 for c in safe)
        summary["eligible_candidate_with_edge_above_pi_over_2_n"]=sum(c["edges_above_pi_over_2_n"]>0 for c in safe)
        return summary
    summary={name:summarize([r for r in rows if r["dataset"]==name]) for name in EXTERNAL_DATASETS}
    summary["pooled110"]=summarize(rows)
    summary["conditional_matches_frozen_target"]=summarize([r for r in rows if r["matches_frozen_predicted_target"]])
    save(path,{"complete":True,"protocol":protocol,"summary":summary,"rows":rows})
    print(json.dumps({"complete":True,"selection_score_summary":summary}),flush=True)


def run_optimizer_timing(output_dir):
    """One serial, matched optimization pass; no repeated export or verification."""
    import torch
    from tcmr.joint_optimizer import optimize_joint_collar
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source_path = output_dir / "budget_results.json"
    source = load(source_path)
    if not source.get("complete") or len(source["rows"]) != 30:
        raise ValueError("Timing requires the completed unchanged budget outputs")
    names = [f"{mode}_{steps}" for steps in (10, 30, 100) for mode in ("patch_only", "joint")]
    protocol = {"schema": "matched_optimizer_only_timing", "source_n": 30, "calls_per_source": 6,
                "source_budget_results": str(source_path.resolve()), "source_budget_sha256": sha(source_path),
                "optimizer_sha256": sha(OPTIMIZER), "workers": 1, "torch_cpu_threads": 2,
                "sequence": "Fixed six-condition order cyclically rotated by source position; one call per source-condition",
                "scope": "Wall time inside optimize_joint_collar, including local data assembly, objective setup and Adam iterations; excludes module import, input reading, output comparison/export, candidate alternatives, native validation and target selection",
                "one_time_setup": "Construct one unused scalar Adam instance before timing to initialize the optimizer module; no source geometry optimized during setup",
                "output_equivalence": "Require exact coordinate equality to reuse stored validity; report mismatches without silently treating them as verified",
                "new_native_checks": 0, "duplicate_mesh_exports": 0,
                "statistics": "20000 source-bootstrap draws, seed20261025; matched source pairs for timing differences"}
    path = output_dir / "optimizer_timing_results.json"
    if path.exists():
        raise ValueError("Matched timing already exists; do not silently repeat timing measurements")
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))], lr=.02)
    rows = []
    for position, original in enumerate(source["rows"]):
        with np.load(original["input_path"], allow_pickle=False) as data:
            before, retained, loop = data["vertices"].astype(float), data["faces"].astype(np.int64), data["selected_loop"].astype(np.int64)
        with np.load(original["initializer_path"], allow_pickle=False) as data:
            initial, patch = data["vertices"].astype(float), data["patch_faces"].astype(np.int64)
        h = float(np.median(np.linalg.norm(initial[loop] - initial[np.roll(loop, -1)], axis=1)))
        order = names[position % len(names):] + names[:position % len(names)]
        row = {"sample_id": original["sample_id"], "source_model_id": original["source_model_id"], "order": order, "methods": {}}
        for name in order:
            mode, count = name.rsplit("_", 1)
            started = time.perf_counter()
            vertices, info = optimize_joint_collar(initial, retained, patch, loop, len(before), mode=mode, steps=int(count),
                                                 learning_rate=.02, retained_displacement_budget=.02, rings=2)
            elapsed = time.perf_counter() - started
            cached = original["methods"][name]
            with np.load(cached["patch_path"], allow_pickle=False) as data:
                exact = bool(np.array_equal(vertices, data["vertices"]) and np.array_equal(patch, data["patch_faces"]))
                max_difference = float(np.linalg.norm(vertices - data["vertices"], axis=1).max(initial=0) / h)
            row["methods"][name] = {"optimizer_seconds": elapsed, "coordinates_and_topology_exact": exact,
                                     "max_coordinate_difference_over_boundary_edge": max_difference,
                                     "reused_validity": cached["valid"] if exact else None,
                                     "final_objective": info["final_objective"]}
        rows.append(row)
        save(path, {"complete": False, "protocol": protocol, "rows": rows})
        print(json.dumps({"optimizer_timing_sources": len(rows), "total": 30,
                          "exact_outputs_this_source": sum(v["coordinates_and_topology_exact"] for v in row["methods"].values())}), flush=True)
    summary = {"source_n": 30, "methods": {}, "timing_movable_minus_fixed": {},
               "exact_outputs_n": sum(x["coordinates_and_topology_exact"] for r in rows for x in r["methods"].values())}
    for name in names:
        values = [r["methods"][name]["optimizer_seconds"] for r in rows]
        summary["methods"][name] = {**source_mean_interval(values), "median_seconds": float(np.median(values)),
                                     "p95_seconds": float(np.quantile(values, .95)),
                                     "exact_outputs_n": sum(r["methods"][name]["coordinates_and_topology_exact"] for r in rows)}
    for steps in (10, 30, 100):
        summary["timing_movable_minus_fixed"][str(steps)] = paired_statistics(
            [r["methods"][f"joint_{steps}"]["optimizer_seconds"] for r in rows],
            [r["methods"][f"patch_only_{steps}"]["optimizer_seconds"] for r in rows], seed=20261025)
    summary["fixed100_minus_joint10_seconds"] = paired_statistics(
        [r["methods"]["patch_only_100"]["optimizer_seconds"] for r in rows],
        [r["methods"]["joint_10"]["optimizer_seconds"] for r in rows], seed=20261025)
    save(path, {"complete": True, "protocol": protocol, "rows": rows, "summary": summary})
    print(json.dumps({"complete": True, "optimizer_timing_summary": summary}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=ROOT / "experiments/formal_collar_mechanism")
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--external-worker", choices=tuple(EXTERNAL_DATASETS))
    ap.add_argument("--external-index", type=int)
    ap.add_argument("--prepare-budget", action="store_true")
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--budget-worker", type=int)
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--fairing-fallback", action="store_true")
    ap.add_argument("--budget-fallback", action="store_true")
    ap.add_argument("--predicted-fallback", action="store_true")
    ap.add_argument("--fixed-loop-input", type=Path)
    ap.add_argument("--fixed-loop-output", type=Path)
    ap.add_argument("--optimizer-timing", action="store_true")
    ap.add_argument("--prepare-seam-ablation", action="store_true")
    ap.add_argument("--seam-ablation", action="store_true")
    ap.add_argument("--seam-ablation-worker", type=int)
    ap.add_argument("--selection-score-diagnostic", action="store_true")
    args = ap.parse_args()
    if args.prepare_seam_ablation:
        print(json.dumps(seam_ablation_protocol(args.output_dir),indent=2))
    elif args.seam_ablation_worker is not None:
        seam_ablation_worker(args.seam_ablation_worker,args.output_dir)
    elif args.seam_ablation:
        run_seam_ablation(args.output_dir)
    elif args.selection_score_diagnostic:
        run_selection_score_diagnostic(args.output_dir)
    elif args.fairing_fallback:
        run_fairing_fallback(args.output_dir)
    elif args.optimizer_timing:
        run_optimizer_timing(args.output_dir)
    elif args.budget_fallback:
        run_budget_fallback(args.output_dir)
    elif args.fixed_loop_input:
        fixed_loop_worker(args.fixed_loop_input, args.fixed_loop_output)
    elif args.predicted_fallback:
        run_predicted_fallback(args.output_dir)
    elif args.prepare_budget:
        print(json.dumps(budget_protocol(args.output_dir), indent=2))
    elif args.budget_worker is not None:
        budget_worker(args.budget_worker, args.output_dir)
    elif args.budget:
        run_budget(args.output_dir)
    elif args.fallback:
        run_fallback(args.output_dir)
    elif args.external_worker:
        external_worker(args.external_worker, args.external_index, args.output_dir)
    elif args.external:
        run_external(args.output_dir)
    else:
        ap.error("Choose a formal control or scoring mode; use --help for available actions")
