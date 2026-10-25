"""Locked geometry comparison for predicted versus oracle target repair."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _unit_face_normals(vertices, faces):
    """Return unit normals for a face array, with safe handling of degeneracy."""
    ff = np.asarray(faces, dtype=np.int64)
    if len(ff) == 0:
        return np.empty((0, 3), dtype=float)
    tri = np.asarray(vertices, dtype=float)[ff]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    d = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(d, 1e-12)


def _seam_normal_error(vertices, retained_faces, patch_faces, loop):
    """Mean seam normal angle (radians) for retained/patch faces sharing loop edges."""
    loop = np.asarray(loop, dtype=np.int64)
    if len(loop) == 0 or len(retained_faces) == 0 or len(patch_faces) == 0:
        return None
    base_map = {}
    for fi, face in enumerate(np.asarray(retained_faces, dtype=np.int64)):
        for edge in (face[[0, 1]], face[[1, 2]], face[[2, 0]]):
            base_map.setdefault(tuple(np.sort(edge).tolist()), fi)
    patch_map = {}
    for fi, face in enumerate(np.asarray(patch_faces, dtype=np.int64)):
        for edge in (face[[0, 1]], face[[1, 2]], face[[2, 0]]):
            patch_map.setdefault(tuple(np.sort(edge).tolist()), fi)
    bn = _unit_face_normals(vertices, retained_faces)
    pn = _unit_face_normals(vertices, patch_faces)
    angles = []
    for a, b in zip(loop, np.roll(loop, -1)):
        key = tuple(sorted((int(a), int(b))))
        if key not in base_map or key not in patch_map:
            continue
        # Face orientation can differ at the seam, so compare the unoriented
        # normals.  The resulting angle is invariant to triangle winding.
        dot = float(np.clip(abs(np.dot(bn[base_map[key]], pn[patch_map[key]])), 0.0, 1.0))
        angles.append(float(np.arccos(dot)))
    return float(np.mean(angles)) if angles else None


def _stats(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": None, "p05": None, "p95": None, "max": None}
    arr = np.asarray(vals, dtype=float)
    return {"n": int(len(arr)), "mean": float(np.mean(arr)),
            "p05": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr))}


def target_closure(faces, target_loop):
    """Closure of annotated target edges, independent of selected candidate.

    Damaged PartNet assemblies can contain pre-existing duplicate/coincident
    faces on a target edge.  A minimal-change repair should not be penalized
    for preserving such an inherited incidence count, so closure requires at
    least two incident faces per target edge.  Any newly introduced excess
    incidence remains visible through the separate introduced-nonmanifold
    metric.
    """
    if target_loop is None or len(target_loop) < 3:
        return None
    ff = np.asarray(faces, dtype=np.int64)
    edges = np.sort(np.concatenate([ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    incidence = {tuple(e): int(n) for e, n in zip(unique, counts)}
    return all(incidence.get(tuple(sorted((int(a), int(b)))), 0) >= 2
               for a, b in zip(target_loop, np.roll(target_loop, -1)))


def mapped_target_loop_closure(faces, loops, annotated_target_loop):
    """Evaluate closure on the extracted loop corresponding to the annotation.

    PartNet annotations can contain a denser or slightly different sampling
    of the same interface than the damaged-mesh boundary extractor.  Using
    annotation vertex edges directly can therefore report a false failure even
    when the selected extracted boundary is fully closed.  Target identity is
    still determined by the annotation overlap; only the closure geometry is
    measured on the extracted loop used by the repair pipeline.
    """
    if annotated_target_loop is None or not loops:
        return None
    from tcmr.evaluation import target_index_from_loop
    target_idx = target_index_from_loop(loops, annotated_target_loop)
    if target_idx is None or int(target_idx) >= len(loops):
        return False
    return target_closure(faces, loops[int(target_idx)])


def _selected_geometry(prediction, before_faces):
    """Extract selected candidate validity and patch/seam diagnostics.

    Candidate validity is already computed by ``TCMRSystem.repair``.  Reusing
    it here avoids a second self-intersection pass during the full evaluation.
    A null decision is represented by the damaged-input baseline and does not
    count as an introduced defect.
    """
    diagnostics = prediction.get("diagnostics", {})
    selected = prediction.get("selected")
    candidates = diagnostics.get("candidates", {})
    candidate = candidates.get(str(selected), candidates.get(selected)) if selected is not None else None
    validity = dict(candidate.get("validity", {})) if candidate else dict(diagnostics.get("baseline_validity", {}))
    baseline = diagnostics.get("baseline_validity", {})
    if candidate:
        final_nm = int(validity.get("non_manifold_edges", 0))
        final_deg = int(validity.get("degenerate_faces", 0))
        final_inv = int(validity.get("inverted_faces", 0))
        validity["introduced_nonmanifold"] = final_nm > int(baseline.get("non_manifold_edges", 0))
        validity["introduced_degenerate"] = final_deg > int(baseline.get("degenerate_faces", 0))
        validity["introduced_inverted"] = final_inv > int(baseline.get("inverted_faces", 0))
        validity["introduced_self_intersections"] = bool(
            validity.get("introduced_self_intersections", False)
        )
        validity["introduced_defect"] = bool(validity["introduced_nonmanifold"] or
                                              validity["introduced_degenerate"] or
                                              validity["introduced_inverted"] or
                                              validity["introduced_self_intersections"])
    else:
        validity.update({"introduced_nonmanifold": False, "introduced_degenerate": False,
                         "introduced_inverted": False, "introduced_self_intersections": False,
                         "introduced_defect": False})

    vertices, faces = prediction["mesh"]
    patch_faces = np.asarray(faces, dtype=np.int64)[len(before_faces):]
    from tcmr.optimizer import triangle_quality
    quality = triangle_quality(vertices, patch_faces) if len(patch_faces) else np.empty(0, dtype=float)
    loop = prediction.get("loops", [])[int(selected)] if selected is not None and int(selected) < len(prediction.get("loops", [])) else None
    seam_error = _seam_normal_error(vertices, before_faces, patch_faces, loop) if loop is not None else None
    energy = candidate.get("energy", {}) if candidate else {}
    initial_collision = candidate.get("initial_patch_retained_intersections") if candidate else None
    optimized_collision = candidate.get("optimized_patch_retained_intersections") if candidate else None
    # The optimizer's normal/curvature terms are squared residuals.  Their
    # square roots are retained as compact diagnostics alongside the direct
    # seam angle above.
    return {"validity": validity, "quality": quality, "seam_normal_error": seam_error,
            "seam_normal_rmse": float(np.sqrt(max(float(energy.get("normal", 0.0)), 0.0))),
            "seam_curvature_rmse": float(np.sqrt(max(float(energy.get("curv", 0.0)), 0.0))),
            "initial_patch_retained_intersections": initial_collision,
            "optimized_patch_retained_intersections": optimized_collision,
            "selected_candidate": bool(candidate),
            "candidate_validities": [v.get("validity", {}) for v in candidates.values()]}


def _aggregate_geometry(rows):
    """Aggregate selected validity, attempted-candidate validity and quality."""
    selected = [r["geometry"] for r in rows]
    validities = [x["validity"] for x in selected]
    def mean_key(key):
        vals = [v.get(key) for v in validities if v.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    known_self = [v.get("self_intersections") for v in validities if v.get("self_intersections") is not None]
    closure = [x['validity'].get('closure') for x in selected
               if x['selected_candidate'] and x['validity'].get('closure') is not None]
    validity_summary = {
        "n": len(validities),
        "mean_non_manifold_edges": mean_key("non_manifold_edges"),
        "mean_degenerate_faces": mean_key("degenerate_faces"),
        "mean_inverted_faces": mean_key("inverted_faces"),
        "mean_self_intersections_known": float(np.mean(known_self)) if known_self else None,
        "self_intersection_known_rate": float(len(known_self) / max(len(validities), 1)),
        "self_intersection_positive_rate": float(np.mean([x > 0 for x in known_self])) if known_self else None,
        "mean_closure": float(np.mean(closure)) if closure else None,
        "closure_success_rate": float(np.mean([x <= 1e-8 for x in closure])) if closure else None,
        "closure_evaluated_repairs": len(closure),
        "watertight_rate": float(np.mean([bool(v.get("watertight", False)) for v in validities])) if validities else None,
        "valid_rate": float(np.mean([bool(v.get("valid", False)) for v in validities])) if validities else None,
        "introduced_defect_rate": float(np.mean([bool(v.get("introduced_defect", False)) for v in validities])) if validities else None,
        "introduced_nonmanifold_rate": float(np.mean([bool(v.get("introduced_nonmanifold", False)) for v in validities])) if validities else None,
        "introduced_degenerate_rate": float(np.mean([bool(v.get("introduced_degenerate", False)) for v in validities])) if validities else None,
        "introduced_inverted_rate": float(np.mean([bool(v.get("introduced_inverted", False)) for v in validities])) if validities else None,
        "introduced_self_intersection_rate": float(np.mean([bool(v.get("introduced_self_intersections", False)) for v in validities])) if validities else None,
    }
    attempted = [v for r in selected for v in r["candidate_validities"]]
    attempted_summary = {
        "n": len(attempted),
        "introduced_defect_rate": float(np.mean([bool(v.get("introduced_defect", False)) for v in attempted])) if attempted else None,
        "introduced_self_intersection_rate": float(np.mean([bool(v.get("introduced_self_intersections", False)) for v in attempted])) if attempted else None,
        "mean_non_manifold_edges": float(np.mean([v.get("non_manifold_edges", 0) for v in attempted])) if attempted else None,
        "mean_degenerate_faces": float(np.mean([v.get("degenerate_faces", 0) for v in attempted])) if attempted else None,
        "mean_inverted_faces": float(np.mean([v.get("inverted_faces", 0) for v in attempted])) if attempted else None,
    }
    quality = np.concatenate([x["quality"] for x in selected if len(x["quality"])]) if any(len(x["quality"]) for x in selected) else np.empty(0)
    init_collisions = [x.get("initial_patch_retained_intersections") for x in selected if x.get("initial_patch_retained_intersections") is not None]
    opt_collisions = [x.get("optimized_patch_retained_intersections") for x in selected if x.get("optimized_patch_retained_intersections") is not None]
    return {"selected": validity_summary, "attempted_candidates": attempted_summary,
            "patch_triangle_quality": _stats(quality),
            "seam_normal_error_radians": _stats([x["seam_normal_error"] for x in selected]),
            "seam_normal_rmse": _stats([x["seam_normal_rmse"] for x in selected]),
            "seam_curvature_rmse": _stats([x["seam_curvature_rmse"] for x in selected]),
            "patch_retained_intersections_initial": _stats(init_collisions),
            "patch_retained_intersections_optimized": _stats(opt_collisions)}

def run(index, checkpoint, split, steps, oracle, top_k=1, limit=None, gate_path=None, planarity_threshold=0.08, device=None, strict_validity=False, fast_oracle=False, conservative_collision=False, reject_initial_collisions=None, reject_collision_increase=None, safety_rerank=False, new_self_intersection_weight=0.0, revert_patch_self_intersection=False, normal_weight=None, initializer_safety=False, initializer_safety_max_loop=400):
    import torch
    from tcmr.mesh_io import load_mesh_npz
    from tcmr.boundary import extract_boundary_loops
    from tcmr.pipeline import TCMRSystem
    from tcmr.evaluation import evaluate_prediction, target_index_from_loop
    idx = json.loads(Path(index).read_text(encoding='utf-8'))
    gate = json.loads(Path(gate_path).read_text(encoding='utf-8')) if gate_path else None
    if reject_initial_collisions is None:
        reject_initial_collisions = bool(conservative_collision)
    if reject_collision_increase is None:
        reject_collision_increase = bool(conservative_collision)
    weights = None
    if normal_weight is not None:
        weights = {"fair": 1.0, "lap": 0.1, "fid": 1000.0,
                   "normal": float(normal_weight), "curv": 0.1,
                   "quality": 1.0}
    cfg = __import__('tcmr.pipeline', fromlist=['TCMRConfig']).TCMRConfig(top_k=top_k, optimization_steps=steps, planarity_threshold=planarity_threshold, initializer_safety_selector=bool(initializer_safety), initializer_safety_max_loop=int(initializer_safety_max_loop), weights=weights or {"fair": 1.0, "lap": 0.1, "fid": 1000.0, "normal": 1.0, "curv": 0.1, "quality": 1.0}, selector_gate=gate and {'weights':gate['weights'],'bias':gate['bias'],'mean':gate['mean'],'std':gate['std'],'threshold':gate['best_validation']['threshold']}, retry_fan_on_self_intersection=bool(strict_validity), reject_new_self_intersections=bool(strict_validity), reject_invalid_selected=bool(strict_validity), collision_retry=not (bool(fast_oracle) and bool(oracle)), revert_if_worse_self_intersection=not (bool(fast_oracle) and bool(oracle)), revert_if_worse_patch_self_intersection=bool(revert_patch_self_intersection), reject_initial_collisions=bool(reject_initial_collisions), reject_collision_increase=bool(reject_collision_increase), safety_rerank=bool(safety_rerank), new_self_intersection_weight=float(new_self_intersection_weight))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    system = TCMRSystem(cfg, device=device)
    system.config.optimization_steps = steps
    system.config.top_k = top_k
    if checkpoint:
        system.encoder.load_state_dict(torch.load(checkpoint, map_location=system.device))
        system.encoder.eval()
    rows=[]; started=time.perf_counter()
    count = 0
    for rec in idx['records']:
        if rec.get('split') != split: continue
        if limit is not None and count >= limit: break
        count += 1
        d=load_mesh_npz(rec['path'])
        target=None
        if oracle:
            # Oracle evaluation needs only the annotated loop index.  Do not
            # run a full non-oracle repair just to recover the loops: that
            # would optimize up to ``top_k`` extra candidates per sample and
            # make the oracle protocol unnecessarily expensive.
            loops = extract_boundary_loops(d['vertices'], d['faces'])
            target=target_index_from_loop(loops, d.get('target_loop'))
        pred=system.repair(d['vertices'], d['faces'], d['cue'], oracle_target=target)
        row=evaluate_prediction(d['vertices'], d['faces'], pred, d.get('target_loop'))
        # Keep the detailed validity/quality record in memory only; the locked
        # evaluator writes one aggregate JSON and does not retain meshes.
        row['geometry'] = _selected_geometry(pred, d['faces'])
        row['repair_executed'] = bool(pred.get('selected') is not None and len(pred['mesh'][1]) > len(d['faces']))
        row['annotation_edge_closed_legacy'] = target_closure(pred['mesh'][1], d.get('target_loop'))
        row['annotated_target_closed'] = mapped_target_loop_closure(
            pred['mesh'][1], pred.get('loops', []), d.get('target_loop'))
        row['target_repair_success'] = bool(row['repair_executed'] and row['annotated_target_closed'])
        row['sample_id']=rec['sample_id']; rows.append(row)
        print(f"{count}: {rec['sample_id']} completed", flush=True)
    def mean(key, subset):
        vals=[r[key] for r in subset if isinstance(r.get(key), (int,float,bool))]
        return float(np.mean(vals)) if vals else None
    multi=[r for r in rows if r['num_loops']>1]
    geometry = _aggregate_geometry(rows)
    return {'n':len(rows), 'multi_loop_n':len(multi), 'device': device,
            'target_accuracy':mean('target_correct', rows),
            'multi_target_accuracy':mean('target_correct', multi),
            'repair_rate':mean('repair_executed', rows),
            'null_rate':mean('null_selected', rows),
            'annotated_target_closure_rate':mean('annotated_target_closed', rows),
            'annotation_edge_closure_rate_legacy':mean('annotation_edge_closed_legacy', rows),
            'target_repair_success_rate':mean('target_repair_success', rows),
            'config':vars(cfg),
            'sample_outcomes':[{k:r[k] for k in ('sample_id', 'target_correct', 'repair_executed', 'null_selected', 'annotated_target_closed', 'target_repair_success')} for r in rows],
            'mean_displacement':float(np.mean([r['displacement']['mean'] for r in rows])) if rows else None,
            'p95_displacement':float(np.percentile([r['displacement']['mean'] for r in rows],95)) if rows else None,
            'mean_changed_ratio':float(np.mean([r['displacement']['changed_ratio'] for r in rows])) if rows else None,
            'geometry_metrics': geometry,
            'elapsed_seconds':time.perf_counter()-started}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--index',default='data/partnet_tcmr_2000/dataset_index.json'); p.add_argument('--checkpoint',default='data/partnet_tcmr_2000/cbe.pt'); p.add_argument('--split',default='test'); p.add_argument('--steps',type=int,default=20); p.add_argument('--top-k',type=int,default=1); p.add_argument('--limit',type=int,default=None); p.add_argument('--planarity-threshold',type=float,default=0.08); p.add_argument('--normal-weight',type=float,default=None, help='Override local optimizer seam-normal loss weight.'); p.add_argument('--device',default=None, choices=['cpu','cuda']); p.add_argument('--strict-validity',action='store_true'); p.add_argument('--predicted-only',action='store_true'); p.add_argument('--conservative-collision',action='store_true'); p.add_argument('--reject-initial-collisions',action='store_true'); p.add_argument('--reject-collision-increase',action='store_true'); p.add_argument('--safety-rerank',action='store_true'); p.add_argument('--new-self-intersection-weight',type=float,default=0.0); p.add_argument('--revert-patch-self-intersection',action='store_true'); p.add_argument('--initializer-safety',action='store_true', help='Compare fan and forced planar initializers by seam-local self intersections on small loops.'); p.add_argument('--initializer-safety-max-loop',type=int,default=400); p.add_argument('--output',default='data/partnet_tcmr_2000/geometry_comparison.json'); a=p.parse_args()
    tag = 'sanity subset' if a.limit is not None else 'full locked split'
    conservative = bool(a.conservative_collision)
    ri = True if a.reject_initial_collisions else (conservative if conservative else None)
    rc = True if a.reject_collision_increase else (conservative if conservative else None)
    out={'protocol':f'{tag}; damaged mesh and cue only for predicted mode; oracle target only for diagnostic separation; top-k candidate optimization; no intermediate meshes retained.', 'planarity_threshold': a.planarity_threshold, 'normal_weight': a.normal_weight, 'strict_validity': bool(a.strict_validity), 'safety_rerank': bool(a.safety_rerank), 'new_self_intersection_weight': float(a.new_self_intersection_weight), 'revert_patch_self_intersection': bool(a.revert_patch_self_intersection), 'initializer_safety': bool(a.initializer_safety), 'initializer_safety_max_loop': int(a.initializer_safety_max_loop), 'reject_initial_collisions': bool(ri) if ri is not None else False, 'reject_collision_increase': bool(rc) if rc is not None else False, 'predicted':run(a.index,a.checkpoint,a.split,a.steps,False,a.top_k,a.limit,None,a.planarity_threshold,a.device,a.strict_validity,reject_initial_collisions=ri,reject_collision_increase=rc,safety_rerank=a.safety_rerank,new_self_intersection_weight=a.new_self_intersection_weight,revert_patch_self_intersection=a.revert_patch_self_intersection,normal_weight=a.normal_weight,initializer_safety=a.initializer_safety,initializer_safety_max_loop=a.initializer_safety_max_loop), 'oracle':None if a.predicted_only else run(a.index,a.checkpoint,a.split,a.steps,True,a.top_k,a.limit,None,a.planarity_threshold,a.device,a.strict_validity,reject_initial_collisions=ri,reject_collision_increase=rc,safety_rerank=a.safety_rerank,new_self_intersection_weight=a.new_self_intersection_weight,revert_patch_self_intersection=a.revert_patch_self_intersection,normal_weight=a.normal_weight,initializer_safety=a.initializer_safety,initializer_safety_max_loop=a.initializer_safety_max_loop)}
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
