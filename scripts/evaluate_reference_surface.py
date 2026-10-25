"""Reference-surface measurements for the frozen 110-source collar comparison.

Reconstruct the exact removed target triangles from construction metadata.
Surface samples approximate area integrals; every distance is evaluated against
triangles, not a sampled reference point cloud. Sources are the statistical unit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "2"

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_coseg_external_benchmark import read_off

OUT = ROOT / "experiments/formal_collar_mechanism"
DATA = {"holdout50": ROOT / "data/coseg_external_holdout/protocol.json",
        "cross_category60": ROOT / "data/coseg_cross_category/protocol.json"}
CORE_METHODS = ("initializer", "patch_only", "joint")
METRICS = ("patch.surface_chamfer_l1_over_h", "patch.sampled_hausdorff_over_h",
           "patch.distance_p95_over_h", "patch.normal_angle_degrees",
           "patch.oriented_normal_angle_degrees", "patch.relative_area_error",
           "collar.surface_chamfer_l1_over_h", "collar.normal_angle_degrees")
SEAM_METRICS = ("reference_seam.mean_absolute_error_rad", "reference_seam.p95_absolute_error_rad")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def triangle_info(vertices, faces):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(twice_area[:, None], np.finfo(float).tiny)
    return twice_area / 2, normals


def surface(vertices, faces):
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return pv.PolyData(np.asarray(vertices, dtype=np.float64),
                       np.column_stack([np.full(len(faces), 3), faces]).ravel())


def sample_surface(vertices, faces, count, seed):
    areas, normals = triangle_info(vertices, faces)
    if not len(faces) or not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Empty, nonfinite, or zero-area evaluation surface")
    rng = np.random.default_rng(seed)
    ids = np.searchsorted(np.cumsum(areas) / areas.sum(), rng.random(count), side="right")
    ids = np.minimum(ids, len(faces) - 1)
    random = rng.random((count, 2))
    root = np.sqrt(random[:, 0])
    weights = np.column_stack([1 - root, root * (1 - random[:, 1]), root * random[:, 1]])
    points = np.einsum("ij,ijk->ik", weights, vertices[faces[ids]])
    return points, normals[ids], float(areas.sum())


def closest(surface_mesh, points, face_normals):
    ids, projections = surface_mesh.find_closest_cell(points, return_closest_point=True)
    ids = np.asarray(ids, dtype=np.int64)
    distances = np.linalg.norm(points - projections, axis=1)
    return distances, face_normals[ids]


def compare_surface(vertices, faces, reference_v, reference_f, h, count, seed):
    source_points, source_normals, area = sample_surface(vertices, faces, count, seed)
    target_points, target_normals, target_area = sample_surface(reference_v, reference_f, count, seed + 1)
    _, source_face_normals = triangle_info(vertices, faces)
    _, reference_face_normals = triangle_info(reference_v, reference_f)
    forward, closest_reference_normals = closest(surface(reference_v, reference_f), source_points, reference_face_normals)
    backward, closest_source_normals = closest(surface(vertices, faces), target_points, source_face_normals)
    dot = np.concatenate([np.einsum("ij,ij->i", source_normals, closest_reference_normals),
                          np.einsum("ij,ij->i", target_normals, closest_source_normals)])
    dot = np.clip(dot, -1, 1)
    distances = np.concatenate([forward, backward]) / h
    return {"surface_chamfer_l1_over_h": float(distances.mean()),
            "surface_rms_over_h": float(np.sqrt(np.mean(distances ** 2))),
            "sampled_hausdorff_over_h": float(distances.max()),
            "distance_p95_over_h": float(np.percentile(distances, 95)),
            "forward_mean_over_h": float(forward.mean() / h),
            "backward_mean_over_h": float(backward.mean() / h),
            "normal_angle_degrees": float(np.degrees(np.arccos(np.abs(dot))).mean()),
            "oriented_normal_angle_degrees": float(np.degrees(np.arccos(dot)).mean()),
            "relative_area_error": float(abs(area - target_area) / target_area),
            "output_area": area, "reference_area": target_area,
            "samples_per_direction": count}


def collar_vertices(faces, loop, rings=2):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    chosen = np.unique(loop)
    for _ in range(rings):
        adjacent = np.isin(edges, chosen).any(axis=1)
        chosen = np.unique(np.r_[chosen, edges[adjacent].ravel()])
    return chosen


def prepare_reference(row, construction):
    with np.load(row["input_path"], allow_pickle=False) as z:
        before = z["vertices"].astype(float)
        retained = z["faces"].astype(np.int64)
        loop = z["selected_loop"].astype(np.int64)
    full_v, full_f = read_off(Path(construction["raw_mesh"]))
    labels = np.fromstring(Path(construction["raw_segmentation"]).read_text(encoding="utf-8"), sep=" ", dtype=np.int64)
    if len(labels) != len(full_f):
        raise ValueError("Source labels do not match source triangle count")
    if "removed_target_face_ids" in construction:
        target_ids = np.asarray(construction["removed_target_face_ids"], np.int64)
        distractor_ids = np.asarray(construction["removed_distractor_face_ids"], np.int64)
        source_ids = np.asarray(construction["retained_original_vertex_ids"], np.int64)
        if not np.array_equal(target_ids, np.flatnonzero(labels == construction["target_label"])):
            raise ValueError("Recorded target deletion and source labels differ")
    else:
        target_ids = np.flatnonzero(labels == construction["target_label"])
        distractor_ids = np.flatnonzero(labels == construction["distractor_label"])
        keep = np.ones(len(full_f), bool)
        keep[np.r_[target_ids, distractor_ids]] = False
        source_ids = np.unique(full_f[keep])
    keep = np.ones(len(full_f), bool)
    keep[np.r_[target_ids, distractor_ids]] = False
    inverse = np.full(len(full_v), -1, np.int64)
    inverse[source_ids] = np.arange(len(source_ids))
    if not np.array_equal(before, full_v[source_ids]):
        raise ValueError("Original source and damaged input have different coordinate frames")
    if not np.array_equal(retained, inverse[full_f[keep]]):
        raise ValueError("Recorded deletion does not reconstruct input triangles")
    if set(loop) != set(construction["target_loop_vertices"]):
        raise ValueError("Wrong target boundary")
    edges = np.concatenate([retained[:, [0, 1]], retained[:, [1, 2]], retained[:, [2, 0]]])
    h = float(np.median(np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)))
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Invalid input edge scale")
    target_f = full_f[target_ids]
    edge_to_target = {}
    for i, tri in enumerate(target_f):
        for a, b in zip(tri, np.roll(tri, -1)):
            edge_to_target.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    edge_to_retained = {}
    for i, tri in enumerate(full_f[keep]):
        for a, b in zip(tri, np.roll(tri, -1)):
            edge_to_retained.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    _, tn = triangle_info(full_v, target_f)
    _, rn = triangle_info(before, retained)
    original_loop = source_ids[loop]
    angles, oriented_angles, lengths = [], [], []
    for a, b in zip(original_loop, np.roll(original_loop, -1)):
        edge = tuple(sorted((int(a), int(b))))
        if len(edge_to_target.get(edge, [])) != 1 or len(edge_to_retained.get(edge, [])) != 1:
            raise ValueError("Target seam has no unique reference/retained face pair")
        dot = float(np.clip(tn[edge_to_target[edge][0]] @ rn[edge_to_retained[edge][0]], -1, 1))
        angles.append(float(np.degrees(np.arccos(abs(dot)))))
        oriented_angles.append(float(np.degrees(np.arccos(dot))))
        lengths.append(float(np.linalg.norm(full_v[a] - full_v[b])))
    mean_angle = float(np.average(angles, weights=lengths))
    collar_ids = collar_vertices(retained, loop)
    collar_f = retained[np.isin(retained, collar_ids).any(axis=1)]
    metadata = {"raw_mesh": construction["raw_mesh"], "raw_segmentation": construction["raw_segmentation"],
                "coordinate_alignment": "exact retained vertex and face reconstruction; no registration",
                "target_faces_n": len(target_f), "target_label": int(construction["target_label"]),
                "retained_vertices_n": len(before), "collar_incident_faces_n": len(collar_f),
                "h_in": h, "reference_bbox_diagonal": float(np.linalg.norm(np.ptp(full_v, axis=0))),
                "reference_seam_length_weighted_angle_degrees": mean_angle,
                "reference_seam_oriented_length_weighted_angle_degrees": float(np.average(oriented_angles, weights=lengths)),
                "reference_seam_sharp_edge_length_fraction": float(np.average(np.asarray(angles) >= 30, weights=lengths)),
                "reference_shape_stratum": "sharp_interface" if mean_angle >= 30 else "smooth_interface",
                "seam_input_vertex_edges": np.column_stack([loop, np.roll(loop, -1)]).tolist(),
                "seam_original_vertex_edges": np.column_stack([original_loop, np.roll(original_loop, -1)]).tolist(),
                "seam_reference_angles_rad": np.radians(angles).tolist()}
    return before, retained, loop, full_v, target_f, collar_f, h, metadata


def reference_seam_error(vertices, retained, patch, reference):
    """Compare the same rim-edge face pairs, with orientation-invariant angles."""
    edges = [tuple(sorted(edge)) for edge in reference["seam_input_vertex_edges"]]
    wanted = set(edges)
    adjacent = []
    for faces in (retained, patch):
        lookup = {edge: [] for edge in wanted}
        for face_id, tri in enumerate(faces):
            for a, b in zip(tri, np.roll(tri, -1)):
                edge = tuple(sorted((int(a), int(b))))
                if edge in lookup:
                    lookup[edge].append(face_id)
        if any(len(lookup[edge]) != 1 for edge in edges):
            raise ValueError("Output lacks a unique retained/patch pair on an original target rim edge")
        _, normals = triangle_info(vertices, faces)
        adjacent.append(normals[[lookup[edge][0] for edge in edges]])
    dot = np.clip(np.abs(np.einsum("ij,ij->i", adjacent[0], adjacent[1])), 0, 1)
    angles = np.arccos(dot)
    reference_angles = np.asarray(reference["seam_reference_angles_rad"], float)
    errors = np.abs(angles - reference_angles)
    return {"mean_absolute_error_rad": float(errors.mean()),
            "p95_absolute_error_rad": float(np.percentile(errors, 95)),
            "mean_output_angle_rad": float(angles.mean()),
            "mean_reference_angle_rad": float(reference_angles.mean()),
            "boundary_edges_n": len(edges), "output_angles_rad": angles.tolist(),
            "absolute_errors_rad": errors.tolist()}


def score_row(row, construction, external, samples, collar_samples, seed, previous=None):
    before, retained, loop, reference_v, reference_f, collar_f, h, metadata = prepare_reference(row, construction)
    methods = dict(row["methods"])
    methods.update(external)
    result = {k: row[k] for k in ("dataset", "index", "sample_id", "source_model_id", "category", "input_path")}
    result.update(reference=metadata, methods={})
    for name, item in methods.items():
        prior = (previous or {}).get("methods", {}).get(name, {})
        reuse_surface = (prior.get("status") == "success" and prior.get("patch_path") == item.get("patch_path")
                         and "patch" in prior and "collar" in prior)
        output = {"valid": item.get("valid") is True, "generation_status": item.get("status"),
                  "patch_path": item.get("patch_path"), "status": "no_geometry"}
        if reuse_surface:
            output.update(patch=prior["patch"], collar=prior["collar"], surface_evaluation_reused=True)
            if "reference_seam" in prior:
                output.update(reference_seam=prior["reference_seam"], status="success")
                result["methods"][name] = output
                continue
        if item.get("patch_path") and Path(item["patch_path"]).is_file():
            try:
                with np.load(item["patch_path"], allow_pickle=False) as z:
                    vertices, faces = z["vertices"].astype(float), z["patch_faces"].astype(np.int64)
                if len(vertices) < len(before) or not np.isfinite(vertices).all():
                    raise ValueError("Output lost retained prefix or contains nonfinite geometry")
                if not reuse_surface:
                    output["patch"] = compare_surface(vertices, faces, reference_v, reference_f, h, samples, seed)
                    output["collar"] = compare_surface(vertices, collar_f, before, collar_f, h, collar_samples, seed + 2)
                    output["surface_evaluation_reused"] = False
                output["reference_seam"] = reference_seam_error(vertices, retained, faces, metadata)
                output["status"] = "success"
            except Exception as exc:
                output.update(status="measurement_failure", error=f"{type(exc).__name__}: {exc}")
        result["methods"][name] = output
    return result


def value(row, method, metric):
    section, key = metric.split(".")
    return row["methods"][method][section][key]


def interval(values, rng):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0, "mean": None, "ci95": None}
    draws = rng.integers(0, len(values), size=(20000, len(values)))
    means = values[draws].mean(axis=1)
    return {"n": len(values), "mean": float(values.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist()}


def summarize_group(rows, methods, seed):
    rng = np.random.default_rng(seed)
    out = {"n": len(rows), "source_ids": [r["source_model_id"] for r in rows], "methods": {}, "paired": {}}
    for method in methods:
        out["methods"][method] = {metric: interval([value(r, method, metric) for r in rows], rng) for metric in METRICS}
    for other in methods:
        if other == "joint":
            continue
        out["paired"]["joint_minus_" + other] = {
            metric: interval([value(r, "joint", metric) - value(r, other, metric) for r in rows], rng)
            for metric in METRICS}
    # Keep the existing surface bootstrap streams unchanged when adding the
    # reference-seam endpoint. Both absolute values and paired effects are kept.
    seam_rng = np.random.default_rng(seed + 397)
    for method in methods:
        out["methods"][method].update({metric: interval([value(r, method, metric) for r in rows], seam_rng)
                                       for metric in SEAM_METRICS})
    for other in methods:
        if other != "joint":
            out["paired"]["joint_minus_" + other].update({
                metric: interval([value(r, "joint", metric) - value(r, other, metric) for r in rows], seam_rng)
                for metric in SEAM_METRICS})
    return out


def summary(rows, methods, seed):
    generated = [r for r in rows if all(r["methods"].get(m, {}).get("status") == "success" for m in methods)]
    valid = [r for r in generated if all(r["methods"][m]["valid"] for m in methods)]
    core_generated = [r for r in rows if all(r["methods"].get(m, {}).get("status") == "success" for m in CORE_METHODS)]
    core_valid = [r for r in core_generated if all(r["methods"][m]["valid"] for m in CORE_METHODS)]
    out = {"source_n": len(rows), "methods": {},
           "all_generated": summarize_group(generated, methods, seed),
           "common_valid": summarize_group(valid, methods, seed), "reference_strata_common_valid": {},
           "core_all_generated": summarize_group(core_generated, CORE_METHODS, seed),
           "core_common_valid": summarize_group(core_valid, CORE_METHODS, seed),
           "reference_strata_core_common_valid": {},
           "pairwise_valid": {}}
    for m in methods:
        out["methods"][m] = {"generated_n": sum(r["methods"].get(m, {}).get("status") == "success" for r in rows),
                             "valid_n": sum(r["methods"].get(m, {}).get("valid") is True for r in rows)}
    for stratum in ("smooth_interface", "sharp_interface"):
        chosen = [r for r in valid if r["reference"]["reference_shape_stratum"] == stratum]
        out["reference_strata_common_valid"][stratum] = summarize_group(chosen, methods, seed)
        core_chosen = [r for r in core_valid if r["reference"]["reference_shape_stratum"] == stratum]
        out["reference_strata_core_common_valid"][stratum] = summarize_group(core_chosen, CORE_METHODS, seed)
    for other in methods:
        if other == "joint":
            continue
        chosen = [r for r in rows if all(r["methods"].get(m, {}).get("status") == "success" and r["methods"][m]["valid"] for m in ("joint", other))]
        out["pairwise_valid"]["joint_minus_" + other] = summarize_group(chosen, ("joint", other), seed)
    if "bounded_biharmonic_fair" in methods:
        compared = ("patch_only", "joint", "bounded_biharmonic_fair")
        chosen = [r for r in rows if all(r["methods"].get(m, {}).get("status") == "success"
                  and r["methods"][m]["valid"] for m in compared)]
        out["bounded_comparison_common_valid"] = summarize_group(chosen, compared, seed)
        out["reference_strata_bounded_common_valid"] = {
            stratum: summarize_group([r for r in chosen if r["reference"]["reference_shape_stratum"] == stratum], compared, seed)
            for stratum in ("smooth_interface", "sharp_interface")}
    return out


def visualization_cases(rows, records):
    """One source per reference stratum, chosen without favoring improvements."""
    chosen = {}
    eligible = [r for r in rows if all(r["methods"].get(m, {}).get("status") == "success"
                and r["methods"][m]["valid"] for m in CORE_METHODS)]
    for stratum in ("smooth_interface", "sharp_interface"):
        group = [r for r in eligible if r["reference"]["reference_shape_stratum"] == stratum]
        if not group:
            continue
        def effect(row):
            return value(row, "joint", SEAM_METRICS[0]) - value(row, "patch_only", SEAM_METRICS[0])
        median = float(np.median([effect(r) for r in group]))
        row = min(group, key=lambda r: (abs(effect(r) - median), r["dataset"], r["index"]))
        construction = records[row["dataset"]][row["sample_id"]]
        labels = np.fromstring(Path(construction["raw_segmentation"]).read_text(encoding="utf-8"), sep=" ", dtype=np.int64)
        chosen[stratum] = {"dataset": row["dataset"], "index": row["index"], "source_model_id": row["source_model_id"],
            "sample_id": row["sample_id"], "eligible_sources_in_stratum": len(group),
            "rule": "Among pooled core-common-valid sources in this original-reference stratum, closest to the median joint-minus-fixed reference-seam MAE; ties use dataset then index; no improvement filter",
            "stratum_median_joint_minus_fixed_reference_seam_mae_rad": median,
            "chosen_joint_minus_fixed_reference_seam_mae_rad": effect(row),
            "input_path": row["input_path"],
            "construction_protocol": str(DATA[row["dataset"]]),
            "raw_mesh": construction["raw_mesh"], "raw_segmentation": construction["raw_segmentation"],
            "gt_target_source_face_ids": np.flatnonzero(labels == construction["target_label"]).tolist(),
            "reference_loading": "read_off(raw_mesh) returns original coordinates and triangles; index triangles with gt_target_source_face_ids; no alignment transform",
            "method_paths": {m: item.get("patch_path") for m, item in row["methods"].items()},
            "method_validity": {m: item.get("valid") is True for m, item in row["methods"].items()},
            "method_reference_seam_mae_rad": {m: item.get("reference_seam", {}).get("mean_absolute_error_rad")
                                               for m, item in row["methods"].items()},
            "original_reference_seam_edges": row["reference"]["seam_original_vertex_edges"]}
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="One input coordinate and exact-distance numerical check; no method scores")
    parser.add_argument("--external-results", type=Path, action="append", default=[])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--collar-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--output", type=Path, default=OUT / "reference_geometry.json")
    args = parser.parse_args()
    raw = read(OUT / "results.json")
    if not raw.get("complete") or len(raw["rows"]) != 110:
        raise ValueError("The frozen 110-source collar comparison must be complete")
    records = {name: {r["sample_id"]: r for r in read(path)["cases"]} for name, path in DATA.items()}
    if args.check_only:
        row = raw["rows"][0]
        data = prepare_reference(row, records[row["dataset"]][row["sample_id"]])
        _, _, _, vertices, faces, _, h, metadata = data
        points, _, _ = sample_surface(vertices, faces, 128, args.seed)
        _, normals = triangle_info(vertices, faces)
        distances, _ = closest(surface(vertices, faces), points, normals)
        if distances.max() / h > 1e-10:
            raise ValueError("Exact triangle distance self-check failed")
        print(json.dumps({"checked_source": row["source_model_id"], "reference": metadata,
                          "sample_to_same_surface_max_over_h": float(distances.max() / h),
                          "method_quality_not_scored": True}, indent=2), flush=True)
        return
    external = {}
    external_methods = []
    for path in args.external_results:
        document = read(path)
        if not document.get("complete"):
            raise ValueError(f"External comparison incomplete: {path}")
        for row in document["rows"]:
            methods = {k: v for k, v in row["methods"].items() if k not in CORE_METHODS}
            external.setdefault((row["dataset"], row["index"]), {}).update(methods)
            external_methods += [m for m in methods if m not in external_methods]
    methods = list(CORE_METHODS) + external_methods
    protocol = {"schema": "exact_removed_target_surface_reference", "source_n": 110,
                "methods": methods, "patch_samples_per_direction": args.samples,
                "retained_collar_samples_per_direction": args.collar_samples, "sampling_seed": args.seed,
                "per_source_seed": "sampling_seed + 10 * position in frozen110 row order; reverse direction +1; collar +2",
                "sampling": "Independent area-uniform barycentric samples in each direction; identical random streams across methods; no reference-based fitting",
                "distance": "Euclidean sample-to-triangle nearest points using VTK cell locator; symmetric average of the two directional means",
                "hausdorff": "Sampled bidirectional maximum; an approximation of surface Hausdorff, not an exact continuous maximum",
                "normal": "Mean nearest-triangle angular discrepancy in degrees; primary uses absolute dot (unoriented); oriented variant also retained",
                "normalization": "h_in = median length of all triangle edges of the damaged retained input, including edge multiplicities; shared across methods",
                "reference": "Exact original target-label surface and saved target deletion; distractor deleted surface excluded; no registration or neighborhood proxy",
                "collar": "Original retained triangles incident on any vertex of the target boundary two-ring; unchanged connectivity and original reference coordinates",
                "sharp_interface": "Original reference length-weighted mean unoriented interface dihedral >=30 degrees; threshold fixed before method scores",
                "reference_seam": "Original target rim edge IDs; original and output use arccos(abs(dot(retained normal, target/patch normal))) in [0,pi/2]; per-edge absolute angle error; arithmetic edge mean and P95 in radians; no reference used in inference",
                "reference_seam_bootstrap_seed": args.seed + 397,
                "statistics": "20000 source-paired bootstrap replicates; seed20261025; 50 and 60 cohorts separate; point samples are not independent observations",
                "validity": "Existing closure, topology and new-intersection result; geometry scored for invalid outputs as well; report all-generated, common-valid, and pairwise-valid",
                "external_results": [str(p.resolve()) for p in args.external_results],
                "inference_rerun": False, "math_threads": 2, "workers": 1}
    previous = read(args.output) if args.output.exists() else {}
    if previous:
        flexible = {"methods", "external_results", "reference_seam", "reference_seam_bootstrap_seed"}
        if any(previous["protocol"].get(k) != v for k, v in protocol.items() if k not in flexible):
            raise ValueError("Existing surface measurement settings differ; incremental reuse refused")
        if not set(previous["protocol"]["methods"]).issubset(methods):
            raise ValueError("Incremental scoring must retain previously evaluated methods")
        if not set(previous["protocol"]["external_results"]).issubset(protocol["external_results"]):
            raise ValueError("Incremental scoring must retain previous external result sources")
        for key in ("reference_seam", "reference_seam_bootstrap_seed"):
            if key in previous["protocol"] and previous["protocol"][key] != protocol[key]:
                raise ValueError("Existing reference-seam measurement settings differ")
    completed = {(r["dataset"], r["index"]): r for r in previous.get("rows", [])}
    started = time.perf_counter()
    for position, row in enumerate(raw["rows"]):
        key = (row["dataset"], row["index"])
        completed[key] = score_row(row, records[row["dataset"]][row["sample_id"]], external.get(key, {}),
                                   args.samples, args.collar_samples, args.seed + 10 * position, previous=completed.get(key))
        ordered = [completed[(r["dataset"], r["index"])] for r in raw["rows"] if (r["dataset"], r["index"]) in completed]
        save(args.output, {"complete": False, "protocol": protocol, "rows": ordered})
        print(json.dumps({"completed": position + 1, "total": 110, "source": row["source_model_id"],
                          "states": {m: v["status"] for m, v in completed[key]["methods"].items()}}), flush=True)
    rows = [completed[(r["dataset"], r["index"])] for r in raw["rows"]]
    summaries = {name: summary([r for r in rows if r["dataset"] == name], methods, args.seed) for name in DATA}
    elapsed = time.perf_counter() - started
    save(args.output, {"complete": True, "protocol": protocol, "rows": rows, "summary": summaries,
                       "visualization_cases": visualization_cases(rows, records),
                       "measurement_wall_seconds": previous.get("measurement_wall_seconds", 0) + elapsed,
                       "last_incremental_wall_seconds": elapsed})
    print(json.dumps({"complete": True, "output": str(args.output), "source_n": len(rows), "methods": methods}), flush=True)


if __name__ == "__main__":
    main()
