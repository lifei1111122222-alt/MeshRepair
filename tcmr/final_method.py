"""Frozen final TCMR inference entry point."""
from __future__ import annotations
from . import integrated_repair

FINAL_POLICY = "prior_locked_pool_with_joint"

def infer(vertices, faces, cue, work_dir, *, ranker_path=None, cbe_path=None, workers=2):
    result = integrated_repair.infer(vertices, faces, cue, work_dir,
                                        ranker_path=ranker_path, cbe_path=cbe_path,
                                        workers=workers)
    selected = result["decisions"][FINAL_POLICY]
    return {"policy": FINAL_POLICY, "decision": selected,
            "target_labels_read": False, "all_decisions": result["decisions"],
            "loops": result["loops"], "wall_seconds": result.get("wall_seconds")}
