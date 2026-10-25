"""Run and score fixed COSEG cohorts with the frozen repair method.

The method under test and the cue-nearest CGAL comparator receive exactly the
same damaged vertices, faces, and cue.  Target and distractor labels are only
opened by the score stage.  MeshFix receives damaged vertices and faces only.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "data" / "coseg_external_benchmark"
OUT = ROOT / "experiments" / "formal_coseg_external_benchmark"
RANKER = ROOT / "models" / "ranker.joblib"
CBE = ROOT / "models" / "cbe.pt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def load_protocol() -> dict:
    return json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    src = load_protocol()
    cases = []
    for index, row in enumerate(src["cases"]):
        cases.append({k: row[k] for k in (
            "sample_id", "category", "source_model_id", "input_path", "input_sha256",
            "target_loop_vertices", "distractor_loop_vertices", "target_perimeter",
            "distractor_perimeter", "cue") } | {"index": index})
    protocol = {
        "schema": "formal_coseg_external_benchmark",
        "sealed_test": True,
        "source_protocol_sha256": sha256(BASE / "protocol.json"),
        "n_cases": len(cases),
        "selection": src["selection"],
        "source": src["source"],
        "policy": "prior_locked_pool_with_joint",
        "ranker_sha256": sha256(RANKER),
        "cbe_sha256": sha256(CBE),
        "implementation_sha256": sha256(ROOT / "tcmr" / "integrated_repair.py"),
        "predictions_saved_before_labels": True,
        "target_labels_read": False,
        "cases": cases,
    }
    if src.get("schema") == "coseg_independent_cross_category_semantic_target":
        protocol.update(schema=src["schema"], n_sources=src["n_sources"],
                        independent_unit="source_model_id", cue_stress=src["cue_stress"],
                        primary_cohort="45 sources with cue_fraction <= 0.4; 15 sources with cue_fraction == 0.6 are a separate severe-ambiguity stress cohort",
                        ablation_definitions=ablation_definitions(),
                        candidate_budget="Both input boundaries receive the identical four-candidate pool; selection ablations reuse these outputs")
        for case, source in zip(cases, src["cases"]):
            for key in ("source_mesh_sha256", "raw_mesh", "target_kind", "distractor_kind",
                        "target_size_stratum", "target_to_distractor_perimeter_ratio",
                        "cue_offset_fraction_of_center_separation"):
                case[key] = source[key]
            case["cue_fraction"] = source["cue_offset_fraction_of_center_separation"]
            case["stratum"] = source["target_size_stratum"]
            case["evaluation_cohort"] = "severe_cue_stress" if case["cue_fraction"] > .4 else "primary"
    label_rows = [{k: row[k] for k in (
        "sample_id", "category", "source_model_id", "target_loop_vertices", "distractor_loop_vertices")}
        for row in cases]
    dump(OUT / "protocol.json", protocol)
    dump(OUT / "score_labels.json", {
        "target_labels_read": False,
        "rows": label_rows,
    })


def ablation_definitions(selector_mode=False) -> dict:
    if selector_mode:
        return {
            "cue_only": "Cue-nearest identity in the same generated candidate set; identical safety-eligible pool",
            "full": "Validated full contact competition locks identity before repair; no cue override; frozen safety/local optimization",
            "no_safety_gate": "Keep contact-selected identity; minimum finite seam without safety filtering; supplementary failure analysis",
            "cgal_library": "Cue-nearest identity in the same generated candidate set; safe CGAL refine/fair and polyline pool",
        }
    return {
        "cue_only": "Cue-nearest identity; minimum seam among safety-eligible candidates",
        "prior_only": "Maximum frozen learned prior identity; minimum seam among the identical safety-eligible pool",
        "prior_geometry": "Maximum prior-logit minus fixed geometry penalty identity; identical safety-eligible pool; no cue switch",
        "full": "Frozen TCMR prior_locked_pool_with_joint, including existing cue/geometry gate and safety-filtered pool",
        "no_safety_gate": "Keep full-method locked identity; minimum finite seam candidate without safety filtering; supplementary failure analysis only",
        "cgal_library": "Cue-nearest identity; minimum seam among safe CGAL refine/fair and polyline candidates",
    }


def comparison_decisions(result: dict, cue) -> dict:
    """Frozen same-budget controls. This function never sees target labels."""
    from tcmr.integrated_repair import POOL
    loops = result.get("loops", [])
    definitions = ablation_definitions(selector_mode=bool(result.get("target_selector")))
    def choose(identity, safety=True, allowed=POOL):
        candidates = [c for c in (identity or {}).get("candidates", [])
                      if c.get("method") in allowed and isinstance(c.get("seam_normal_error"), (int, float))
                      and math.isfinite(c["seam_normal_error"]) and c.get("patch_path")
                      and (not safety or c.get("safe_eligible") is True)]
        candidate = min(candidates, key=lambda c: (c["seam_normal_error"], POOL.index(c["method"]))) if candidates else None
        return {"locked_loop_id": (identity or {}).get("loop_id"),
                "selected_loop_id": identity["loop_id"] if candidate else None,
                "decision": "accept" if candidate else "null",
                "chosen_method": candidate["method"] if candidate else None,
                "patch_path": candidate.get("patch_path") if candidate else None,
                "output_path": candidate.get("patch_path") if candidate else (identity or {}).get("input_path"),
                "metrics": {k: candidate.get(k) for k in ("seam_normal_error", "patch_triangle_quality_mean", "strict_selected_loop_closure", "introduced_intersection")} if candidate else None,
                "target_labels_read": False}
    if not loops:
        return {name: choose(None) for name in definitions}
    distances = []
    for row in loops:
        with np.load(row["input_path"], allow_pickle=False) as data:
            center = data["vertices"][data["selected_loop"]].mean(axis=0)
        distances.append((float(np.linalg.norm(center-np.asarray(cue)[:3])), row["loop_id"], row))
    cue_identity = min(distances, key=lambda x: (x[0], x[1]))[2]
    prior = max(loops, key=lambda r: r.get("prior_probability", -math.inf))
    finite = [r for r in loops if r.get("identity_score") is not None and math.isfinite(r["identity_score"])]
    geometric = max(finite, key=lambda r: r["identity_score"]) if finite else None
    full = result["decisions"]["prior_locked_pool_with_joint"]
    full_identity = next((r for r in loops if r["loop_id"] == full.get("locked_loop_id")), None)
    decisions = {"cue_only": choose(cue_identity), "prior_only": choose(prior),
            "prior_geometry": choose(geometric), "no_safety_gate": choose(full_identity, False),
            "full": result["decisions"]["prior_locked_pool_with_joint"],
            "cgal_library": choose(cue_identity, True, ("cgal_refine_fair", "cgal_polyline"))}
    return {name: decisions[name] for name in definitions}


def infer() -> None:
    from tcmr import integrated_repair as v4

    import joblib  # noqa: F401 - imported before threaded inference on Windows
    import torch  # noqa: F401

    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("target_selector"):
        raise ValueError("This release runs the frozen prior; contact-selector development protocols are unsupported")
    assert sha256(RANKER) == protocol["ranker_sha256"]
    assert sha256(CBE) == protocol["cbe_sha256"]
    assert sha256(ROOT / "tcmr" / "integrated_repair.py") == protocol["implementation_sha256"]
    prediction_dir = OUT / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for done, case in enumerate(protocol["cases"], 1):
        target = prediction_dir / f"{case['index']:03d}.json"
        if target.exists():
            print(json.dumps({"skipped": done, "total": len(protocol["cases"]),
                              "sample_id": case["sample_id"]}), flush=True)
            continue
        with np.load(case["input_path"], allow_pickle=False) as z:
            vertices = np.asarray(z["vertices"], dtype=float)
            faces = np.asarray(z["faces"], dtype=np.int64)
            cue = np.asarray(z["cue"], dtype=float)
        work = OUT / "work" / f"case{case['index']:03d}"
        case_started = time.perf_counter()
        result = v4.infer(vertices, faces, cue, work, ranker_path=RANKER, cbe_path=CBE, workers=2)
        inference_seconds = time.perf_counter() - case_started
        prediction = {
            "index": case["index"], "sample_id": case["sample_id"],
            "source_model_id": case["source_model_id"], "category": case["category"],
            "decisions": result["decisions"], "loops": result["loops"],
            "target_labels_read": False, "wall_seconds": result.get("wall_seconds"),
            "inference_including_prior_seconds": inference_seconds,
            "timing_scope": "Live prior features/loading plus both candidate pools and their validation; excludes damaged-input file I/O and final report scoring",
        }
        if protocol.get("ablation_definitions"):
            decisions = comparison_decisions(result, cue)
            prediction.update(ablations=decisions, cgal_library_decision=decisions["cgal_library"])
        dump(target, prediction)
        print(json.dumps({"predicted": done, "total": len(protocol["cases"]),
                          "sample_id": case["sample_id"],
                          "elapsed_seconds": time.perf_counter() - start}), flush=True)
    dump(OUT / "predictions_manifest.json", {
        "complete": True, "target_labels_read": False, "n": len(protocol["cases"]),
        "wall_seconds": time.perf_counter() - start,
        "protocol_sha256": sha256(OUT / "protocol.json"),
    })


def edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    edges = np.sort(np.concatenate([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
    ]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return {tuple(map(int, e)): int(c) for e, c in zip(unique, counts)}


def mapped_loop(vertices_in: np.ndarray, loop: list[int], vertices_out: np.ndarray,
                faces_out: np.ndarray, tolerance: float) -> dict:
    ids = np.asarray(loop, dtype=np.int64)
    if len(ids) < 3 or len(vertices_out) == 0:
        return {"retained": False, "closed": False, "open": False,
                "max_distance": None, "unique": False}
    distances, mapped = cKDTree(vertices_out).query(vertices_in[ids], k=1)
    mapped = np.asarray(mapped, dtype=np.int64)
    counts = edge_counts(faces_out)
    seam_counts = [counts.get(tuple(sorted((int(a), int(b)))), 0)
                   for a, b in zip(mapped, np.roll(mapped, -1))]
    retained = bool(np.all(np.isfinite(distances)) and np.max(distances) <= tolerance)
    unique = bool(len(np.unique(mapped)) == len(mapped))
    closed = bool(retained and unique and all(c >= 2 for c in seam_counts))
    open_boundary = bool(retained and unique and all(c == 1 for c in seam_counts))
    return {"retained": retained, "closed": closed, "open": open_boundary,
            "max_distance": float(np.max(distances)), "mean_distance": float(np.mean(distances)),
            "unique": unique, "edge_counts": seam_counts}


def method_row(pred: dict, case: dict, policy: str) -> dict:
    target = set(case["target_loop_vertices"])
    decision = pred["decisions"][policy]
    metrics = decision.get("metrics") or {}
    selected = decision.get("selected_loop_id")
    loops = pred.get("loops", [])
    selected_row = next((row for row in loops if row.get("loop_id") == selected), None)
    selected_vertices = None
    if selected_row is not None:
        chosen_candidate = next((x for x in selected_row.get("candidates", [])
                                 if x.get("method") == decision.get("chosen_method")), None)
        if chosen_candidate is not None:
            metrics = dict(metrics)
            for key in ("inclusive_candidate_seconds", "retained_max_displacement_over_edge",
                        "retained_rms_displacement_over_edge", "outside_support_bitwise_fixed",
                        "seam_normal_error", "patch_triangle_quality_mean"):
                if key not in metrics or metrics.get(key) is None:
                    metrics[key] = chosen_candidate.get(key)
        if selected_row.get("input_path"):
            try:
                with np.load(selected_row["input_path"], allow_pickle=False) as z:
                    selected_vertices = set(np.asarray(z["selected_loop"], dtype=np.int64).tolist())
            except Exception:
                selected_vertices = None
    correct = bool(decision.get("decision") == "accept" and selected_vertices == target and
                   metrics.get("strict_selected_loop_closure") is True and
                   metrics.get("introduced_intersection") is False)
    return {
        "decision": decision.get("decision"), "selected_loop_id": selected,
        "chosen_method": decision.get("chosen_method"), "correct_safe": correct,
        "wrong_accepted": bool(decision.get("decision") == "accept" and not correct),
        "metrics": metrics,
    }


def cgal_cue_nearest(pred: dict, case: dict) -> dict:
    """CGAL refine/fair selected by the same cue-only nearest-loop rule."""
    cue = np.asarray(case["cue"][:3], dtype=float)
    best = None
    for loop_row in pred.get("loops", []):
        candidates = loop_row.get("candidates", [])
        cgal = next((x for x in candidates if x.get("method") == "cgal_refine_fair"), None)
        if cgal is None:
            continue
        try:
            with np.load(loop_row["input_path"], allow_pickle=False) as z:
                loop_vertices = np.asarray(z["selected_loop"], dtype=np.int64)
                vertices = np.asarray(z["vertices"], dtype=float)
            distance = float(np.linalg.norm(vertices[loop_vertices].mean(axis=0) - cue))
        except Exception:
            continue
        key = (distance, int(loop_row["loop_id"]))
        if best is None or key < best[0]:
            best = (key, loop_row, cgal)
    if best is None:
        return {"decision": "null", "correct_safe": False, "wrong_accepted": False,
                "selected_loop_id": None, "chosen_method": None, "metrics": {}}
    _, loop_row, cgal = best
    safe = cgal.get("safe_eligible") is True
    target = set(case["target_loop_vertices"])
    selected = set()
    if loop_row.get("input_path"):
        try:
            with np.load(loop_row["input_path"], allow_pickle=False) as z:
                selected = set(np.asarray(z["selected_loop"], dtype=np.int64).tolist())
        except Exception:
            selected = set()
    metrics = {k: cgal.get(k) for k in (
        "seam_normal_error", "patch_triangle_quality_mean", "strict_selected_loop_closure",
        "introduced_intersection", "retained_max_displacement_over_edge",
        "retained_rms_displacement_over_edge", "outside_support_bitwise_fixed",
        "inclusive_candidate_seconds")}
    correct = bool(safe and selected == target and metrics["strict_selected_loop_closure"] is True
                   and metrics["introduced_intersection"] is False)
    return {"decision": "accept" if safe else "null", "selected_loop_id": int(loop_row["loop_id"]),
            "chosen_method": "cgal_refine_fair", "correct_safe": correct,
            "wrong_accepted": bool(safe and not correct), "metrics": metrics}


def confidence_interval(successes: int, n: int) -> list[float]:
    if n == 0:
        return [None, None]
    interval = binomtest(successes, n).proportion_ci(confidence_level=0.95, method="exact")
    return [float(interval.low), float(interval.high)]


def bootstrap_difference(x: np.ndarray, y: np.ndarray, seed: int = 20261010,
                         draws: int = 20000) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    values = np.empty(draws, dtype=float)
    for i in range(draws):
        ids = rng.integers(0, n, size=n)
        values[i] = float(np.mean(x[ids]) - np.mean(y[ids]))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def score() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    labels = json.loads((OUT / "score_labels.json").read_text(encoding="utf-8"))
    if labels.get("target_labels_read") not in (False, True):
        raise RuntimeError("invalid score-label state")
    cases = protocol["cases"]
    if protocol.get("target_selector"):
        label_map = {row["sample_id"]: row for row in labels["rows"]}
        cases = [dict(case, **{key: value for key, value in label_map[case["sample_id"]].items()
                              if key not in ("sample_id", "category", "source_model_id")}) for case in cases]
    predictions = [json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted((OUT / "predictions").glob("*.json"))]
    if len(predictions) != len(protocol["cases"]):
        raise RuntimeError("incomplete predictions")
    by_id = {row["sample_id"]: row for row in predictions}
    rows = []
    methods = ("tcmr", "cgal_cue_nearest")
    for case in cases:
        pred = by_id[case["sample_id"]]
        tcmr = method_row(pred, case, "prior_locked_pool_with_joint")
        cgal = cgal_cue_nearest(pred, case)
        rows.append({"index": case["index"], "sample_id": case["sample_id"],
                     "category": case["category"], "source_model_id": case["source_model_id"],
                     "tcmr": tcmr, "cgal_cue_nearest": cgal,
                     "target_labels_read": True})
    summary = {}
    for method in methods:
        values = np.asarray([int(row[method]["correct_safe"]) for row in rows], dtype=np.int8)
        accepted = np.asarray([int(row[method]["decision"] == "accept") for row in rows], dtype=np.int8)
        wrong = np.asarray([int(row[method]["wrong_accepted"]) for row in rows], dtype=np.int8)
        def metric_mean(name):
            vals = [row[method]["metrics"].get(name) for row in rows
                    if row[method]["decision"] == "accept" and
                    isinstance(row[method]["metrics"].get(name), (int, float)) and
                    np.isfinite(row[method]["metrics"].get(name))]
            return float(np.mean(vals)) if vals else None
        def metric_rate(name):
            vals = [row[method]["metrics"].get(name) for row in rows
                    if row[method]["decision"] == "accept" and
                    row[method]["metrics"].get(name) is not None]
            return float(np.mean(np.asarray(vals, dtype=float))) if vals else None
        summary[method] = {
            "n": len(rows), "accepted_n": int(accepted.sum()),
            "correct_safe_n": int(values.sum()), "wrong_accepted_n": int(wrong.sum()),
            "null_n": int(len(rows) - accepted.sum()),
            "correct_safe_rate": float(values.mean()),
            "correct_safe_rate_ci95": confidence_interval(int(values.sum()), len(values)),
            "risk_among_accepted": float(wrong.sum() / accepted.sum()) if accepted.sum() else None,
            "accepted_correct_precision_ci95": confidence_interval(
                int(values.sum()), int(accepted.sum())) if accepted.sum() else [None, None],
            "mean_seam_normal_error": float(np.mean([row[method]["metrics"].get("seam_normal_error")
                                                      for row in rows
                                                      if row[method]["decision"] == "accept" and
                                                      row[method]["metrics"].get("seam_normal_error") is not None]))
            if any(row[method]["decision"] == "accept" and row[method]["metrics"].get("seam_normal_error") is not None for row in rows) else None,
            "mean_patch_triangle_quality": float(np.mean([row[method]["metrics"].get("patch_triangle_quality_mean")
                                                           for row in rows
                                                           if row[method]["decision"] == "accept" and
                                                           row[method]["metrics"].get("patch_triangle_quality_mean") is not None]))
            if any(row[method]["decision"] == "accept" and row[method]["metrics"].get("patch_triangle_quality_mean") is not None for row in rows) else None,
            "mean_retained_max_displacement_over_edge": metric_mean("retained_max_displacement_over_edge"),
            "mean_retained_rms_displacement_over_edge": metric_mean("retained_rms_displacement_over_edge"),
            "outside_support_bitwise_fixed_rate": metric_rate("outside_support_bitwise_fixed"),
            "mean_candidate_seconds": metric_mean("inclusive_candidate_seconds"),
        }
    t = np.asarray([int(row["tcmr"]["correct_safe"]) for row in rows], dtype=np.int8)
    c = np.asarray([int(row["cgal_cue_nearest"]["correct_safe"]) for row in rows], dtype=np.int8)
    b = int(np.sum((t == 1) & (c == 0)))
    cc = int(np.sum((t == 0) & (c == 1)))
    summary["paired_tcmr_minus_cgal"] = {
        "tcmr_only_correct": b, "cgal_only_correct": cc,
        "difference_pp": float(100 * (t.mean() - c.mean())),
        "bootstrap_ci95_pp": [100 * x for x in bootstrap_difference(t, c)],
        "mcnemar_exact_two_sided_p": float(binomtest(min(b, cc), b + cc, p=0.5).pvalue)
        if b + cc else 1.0,
    }
    common = [row for row in rows if row["tcmr"]["decision"] == "accept" and
              row["cgal_cue_nearest"]["decision"] == "accept" and
              isinstance(row["tcmr"]["metrics"].get("seam_normal_error"), (int, float)) and
              isinstance(row["cgal_cue_nearest"]["metrics"].get("seam_normal_error"), (int, float))]
    if common:
        t_seam = np.asarray([row["tcmr"]["metrics"]["seam_normal_error"] for row in common], dtype=float)
        c_seam = np.asarray([row["cgal_cue_nearest"]["metrics"]["seam_normal_error"] for row in common], dtype=float)
        t_quality = np.asarray([row["tcmr"]["metrics"].get("patch_triangle_quality_mean") for row in common], dtype=float)
        c_quality = np.asarray([row["cgal_cue_nearest"]["metrics"].get("patch_triangle_quality_mean") for row in common], dtype=float)
        summary["paired_geometry_common_accepts"] = {
            "n": len(common),
            "seam_difference_tcmr_minus_cgal_mean": float(np.mean(t_seam - c_seam)),
            "seam_difference_bootstrap_ci95": bootstrap_difference(t_seam, c_seam),
            "quality_difference_tcmr_minus_cgal_mean": float(np.mean(t_quality - c_quality)),
            "quality_difference_bootstrap_ci95": bootstrap_difference(t_quality, c_quality),
        }
    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        ids = [i for i, row in enumerate(rows) if row["category"] == category]
        by_category[category] = {
            method: {"n": len(ids), "correct_safe_n": int(sum(int(rows[i][method]["correct_safe"])
                                                                 for i in ids)),
                     "accepted_n": int(sum(int(rows[i][method]["decision"] == "accept")
                                            for i in ids))}
            for method in methods
        }
    report = {"schema": "formal_coseg_external_benchmark_report", "complete": True,
              "protocol_sha256": sha256(OUT / "protocol.json"), "source": protocol["source"],
              "target_labels_read_only_in_score": True, "summary": summary,
              "by_category": by_category, "rows": rows}
    dump(OUT / "report.json", report)
    lines = ["# Formal 79-case external COSEG benchmark", "",
             "All methods use the fixed damaged mesh and cue protocol. Target and distractor labels are opened only in scoring.", "",
             "|Method|Accepted|Correct-safe|Wrong|Null|Correct-safe rate|95% CI|Risk|Mean seam|Mean quality|RMS disp./edge|Outside fixed|Candidate sec.|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        x = summary[method]
        risk = "---" if x["risk_among_accepted"] is None else f"{x['risk_among_accepted']:.1%}"
        seam = "---" if x["mean_seam_normal_error"] is None else f"{x['mean_seam_normal_error']:.3f}"
        quality = "---" if x["mean_patch_triangle_quality"] is None else f"{x['mean_patch_triangle_quality']:.3f}"
        rms = "---" if x["mean_retained_rms_displacement_over_edge"] is None else f"{x['mean_retained_rms_displacement_over_edge']:.4f}"
        fixed = "---" if x["outside_support_bitwise_fixed_rate"] is None else f"{x['outside_support_bitwise_fixed_rate']:.1%}"
        seconds = "---" if x["mean_candidate_seconds"] is None else f"{x['mean_candidate_seconds']:.2f}"
        lines.append(f"|{method}|{x['accepted_n']}|{x['correct_safe_n']}|{x['wrong_accepted_n']}|{x['null_n']}|{x['correct_safe_rate']:.1%}|[{x['correct_safe_rate_ci95'][0]:.1%}, {x['correct_safe_rate_ci95'][1]:.1%}]|{risk}|{seam}|{quality}|{rms}|{fixed}|{seconds}|")
    p = summary["paired_tcmr_minus_cgal"]
    lines += ["", f"Paired TCMR minus cue-nearest CGAL difference: {p['difference_pp']:.2f} percentage points; bootstrap 95% CI [{p['bootstrap_ci95_pp'][0]:.2f}, {p['bootstrap_ci95_pp'][1]:.2f}] pp; exact McNemar p={p['mcnemar_exact_two_sided_p']:.3g}.", "", "## Category breakdown", "", "|Category|TCMR correct-safe|TCMR n|CGAL correct-safe|CGAL n|", "|---|---:|---:|---:|---:|"]
    for category, values in by_category.items():
        lines.append(f"|{category}|{values['tcmr']['correct_safe_n']}|{values['tcmr']['n']}|{values['cgal_cue_nearest']['correct_safe_n']}|{values['cgal_cue_nearest']['n']}|")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    label_state = {"target_labels_read": True, "rows": rows}
    if protocol.get("target_selector"):
        # Preserve the separated original labels so a repeated scoring pass
        # never needs to move labels back into the inference protocol.
        label_state.update(rows=labels["rows"], scored_rows=rows)
    (OUT / "score_labels.json").write_text(json.dumps(label_state, indent=2,
                                                         ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def meshfix() -> None:
    """Run the cue-free MeshFix control on every fixed damaged input."""
    import pymeshfix

    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    cases = protocol["cases"]
    if protocol.get("target_selector"):
        labels = json.loads((OUT / "score_labels.json").read_text(encoding="utf-8"))
        label_map = {row["sample_id"]: row for row in labels["rows"]}
        cases = [dict(case, **{key: value for key, value in label_map[case["sample_id"]].items()
                              if key not in ("sample_id", "category", "source_model_id")}) for case in cases]
    output = OUT / "meshfix.json"
    existing = {}
    if output.exists():
        existing = {row["sample_id"]: row for row in json.loads(output.read_text(encoding="utf-8")).get("rows", [])}
    rows = []
    started_all = time.perf_counter()
    for done, case in enumerate(cases, 1):
        if case["sample_id"] in existing:
            rows.append(existing[case["sample_id"]])
            continue
        with np.load(case["input_path"], allow_pickle=False) as z:
            vertices = np.asarray(z["vertices"], dtype=float)
            faces = np.asarray(z["faces"], dtype=np.int64)
        diag = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-8)
        tolerance = max(1e-5 * diag, 1e-8)
        t0 = time.perf_counter()
        try:
            vout, fout = pymeshfix.clean_from_arrays(vertices, faces, verbose=False)
            vout = np.asarray(vout, dtype=float)
            fout = np.asarray(fout, dtype=np.int64)
            error = None
        except Exception as exc:  # pragma: no cover - external tool failure path
            vout = np.empty((0, 3), dtype=float)
            fout = np.empty((0, 3), dtype=np.int64)
            error = repr(exc)
        repair_seconds = time.perf_counter() - t0
        raw_output = OUT / "meshfix_outputs" / f"{int(case['index']):03d}.npz"
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_output, vertices=vout, faces=fout)
        target = mapped_loop(vertices, case["target_loop_vertices"], vout, fout, tolerance) if len(vout) else {}
        distractor = mapped_loop(vertices, case["distractor_loop_vertices"], vout, fout, tolerance) if len(vout) else {}
        rows.append({
            "sample_id": case["sample_id"], "category": case["category"],
            "runtime_seconds": repair_seconds, "failed": error,
            "timing_scope": "MeshFix clean_from_arrays only; excludes input/output I/O and independent label scoring",
            "output_path": str(raw_output.resolve()), "output_sha256": sha256(raw_output),
            "n_vertices_in": int(len(vertices)), "n_faces_in": int(len(faces)),
            "n_vertices_out": int(len(vout)), "n_faces_out": int(len(fout)),
            "map_tolerance": tolerance, "target": target, "distractor": distractor,
            "global_watertight": bool(len(vout) and len(fout) and
                                       all(v == 2 for v in edge_counts(fout).values())),
        })
        dump(output, {"schema": "meshfix_external_control", "complete": False,
                      "n": len(protocol["cases"]), "rows": rows,
                      "protocol_sha256": sha256(OUT / "protocol.json")})
        print(json.dumps({"meshfix": done, "total": len(protocol["cases"]),
                          "sample_id": case["sample_id"]}), flush=True)
    successful = [row for row in rows if not row["failed"]]
    target_specific = [row for row in successful if row["target"].get("closed") and
                       row["distractor"].get("open")]
    report = {
        "schema": "meshfix_external_control", "complete": True,
        "protocol_sha256": sha256(OUT / "protocol.json"), "n": len(rows),
        "successful_calls": len(successful), "failed_calls": len(rows) - len(successful),
        "target_closure_n": sum(bool(row["target"].get("closed")) for row in successful),
        "target_specific_n": len(target_specific),
        "distractor_preserved_open_n": sum(bool(row["distractor"].get("open")) for row in successful),
        "global_watertight_n": sum(bool(row["global_watertight"]) for row in successful),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows])) if rows else None,
        "wall_seconds": time.perf_counter() - started_all, "rows": rows,
    }
    dump(output, report)
    print(json.dumps({k: report[k] for k in ("n", "successful_calls", "target_closure_n",
                                             "target_specific_n", "distractor_preserved_open_n",
                                             "global_watertight_n", "mean_runtime_seconds")},
                     ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "infer", "score", "meshfix"))
    parser.add_argument("--data", type=Path, default=BASE,
                        help="fixed damaged-input dataset with protocol.json")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="formal experiment directory")
    args = parser.parse_args()
    BASE, OUT = args.data.resolve(), args.out.resolve()
    if args.stage in ("prepare", "infer"):
        {"prepare": prepare, "infer": infer}[args.stage]()
    else:
        {"score": score, "meshfix": meshfix}[args.stage]()
