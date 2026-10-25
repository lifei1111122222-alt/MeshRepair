from __future__ import annotations

import numpy as np


def target_index_from_loop(loops, target_loop):
    if target_loop is None or len(loops) == 0:
        return None
    target = set(np.asarray(target_loop, dtype=np.int64).tolist())
    scores = [len(target.intersection(set(l.tolist()))) / max(len(target.union(set(l.tolist()))), 1) for l in loops]
    return int(np.argmax(scores)) if max(scores) > 0 else None


def vertex_displacement(before, after):
    n = min(len(before), len(after)); d = np.linalg.norm(after[:n] - before[:n], axis=1)
    return {"mean": float(np.mean(d)) if len(d) else 0.0, "median": float(np.median(d)) if len(d) else 0.0,
            "p95": float(np.percentile(d, 95)) if len(d) else 0.0, "max": float(np.max(d)) if len(d) else 0.0,
            "changed_ratio": float(np.mean(d > 1e-7)) if len(d) else 0.0, "values": d}


def evaluate_prediction(before_v, before_f, prediction, target_loop=None):
    loops = prediction.get("loops", [])
    gt_idx = target_index_from_loop(loops, target_loop)
    selected = prediction.get("selected")
    d = vertex_displacement(before_v, prediction["mesh"][0])
    diagnostics = prediction.get("diagnostics", {})
    rank_map = diagnostics.get("target_rank", {})
    prune_map = diagnostics.get("prune_rank", {})
    joint_map = diagnostics.get("joint_rank", {})
    selected_rank = rank_map.get(str(selected), rank_map.get(selected)) if selected is not None else None
    selected_prune_rank = prune_map.get(str(selected), prune_map.get(selected)) if selected is not None else None
    selected_joint_rank = joint_map.get(str(selected), joint_map.get(selected)) if selected is not None else None
    gt_rank = rank_map.get(str(gt_idx), rank_map.get(gt_idx)) if gt_idx is not None else None
    gt_prune_rank = prune_map.get(str(gt_idx), prune_map.get(gt_idx)) if gt_idx is not None else None
    gt_joint_rank = joint_map.get(str(gt_idx), joint_map.get(gt_idx)) if gt_idx is not None else None
    candidate_indices = [int(i) for i in diagnostics.get("candidate_indices", diagnostics.get("candidates", {}).keys())]
    candidate_rank = {int(i): j + 1 for j, i in enumerate(candidate_indices)}
    target_candidate_rank = candidate_rank.get(gt_idx) if gt_idx is not None else None
    top_k = len(candidate_indices)
    return {"target_index": gt_idx, "selected": selected, "target_correct": bool(gt_idx is not None and selected == gt_idx),
            "selected_rank": selected_rank, "selected_prune_rank": selected_prune_rank,
            "selected_joint_rank": selected_joint_rank, "target_rank": gt_rank,
            "target_prune_rank": gt_prune_rank, "target_joint_rank": gt_joint_rank,
            "candidate_rank": target_candidate_rank,
            "top_k_recall": bool(target_candidate_rank is not None and target_candidate_rank <= top_k), "displacement": {k: v for k, v in d.items() if k != "values"},
            "num_loops": len(loops), "null_selected": selected is None, "decision": diagnostics.get("decision", {}),
            "candidates": diagnostics.get("candidates", {})}
