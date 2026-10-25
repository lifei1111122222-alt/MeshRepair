from __future__ import annotations
import numpy as np


def fit_score_normalizer(rows, keys=("neglog_prior", "cue_cost", "repair_cost", "validity_penalty"), robust=True):
    """Fit validation-only location/scale statistics for joint-score terms.

    ``rows`` is an iterable of dictionaries containing candidate term values. The
    returned object is JSON serialisable and can be applied to unseen samples.
    A median/MAD scale is used by default; the fallback scale is one to avoid
    amplifying nearly constant terms (for example a binary validity penalty).
    """
    out = {}
    for key in keys:
        vals = np.asarray([float(r[key]) for r in rows if key in r and np.isfinite(r[key])], dtype=float)
        if vals.size == 0:
            out[key] = {"center": 0.0, "scale": 1.0, "method": "identity"}
            continue
        if robust:
            center = float(np.median(vals)); scale = float(1.4826 * np.median(np.abs(vals - center)))
            method = "median_mad"
        else:
            center = float(np.mean(vals)); scale = float(np.std(vals)); method = "mean_std"
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0; method = "identity" if np.ptp(vals) < 1e-8 else method
        out[key] = {"center": center, "scale": scale, "method": method}
    return out


def normalize_score_terms(values, normalizer):
    """Apply a fitted normalizer to a mapping or structured numpy arrays."""
    if isinstance(values, dict):
        return {k: (float(v) - normalizer.get(k, {}).get("center", 0.0)) /
                max(normalizer.get(k, {}).get("scale", 1.0), 1e-8) for k, v in values.items()}
    arr = np.asarray(values, dtype=float)
    return arr


def cue_cost(vertices, loop, cue):
    p = vertices[loop]; center, scale = cue[:3], max(float(cue[3]), 1e-8)
    z = np.linalg.norm(p - center, axis=1) / scale
    return float(np.mean(np.minimum(z * z, 9.0)))


def joint_decision(priors, cue_costs, repair_costs, validity_penalties, alpha=1.0, beta=1.0, gamma=10.0,
                   null_prior=None, null_tau=0.0, normalizer=None):
    priors = np.asarray(priors, dtype=float)
    terms = {"neglog_prior": -np.log(priors + 1e-8), "cue_cost": np.asarray(cue_costs, dtype=float),
             "repair_cost": np.asarray(repair_costs, dtype=float), "validity_penalty": np.asarray(validity_penalties, dtype=float)}
    if normalizer:
        terms = {k: (v - float(normalizer.get(k, {}).get("center", 0.0))) /
                max(float(normalizer.get(k, {}).get("scale", 1.0)), 1e-8) for k, v in terms.items()}
    scores = terms["neglog_prior"] + alpha * terms["cue_cost"] + beta * terms["repair_cost"] + gamma * terms["validity_penalty"]
    null_raw = -np.log(float(null_prior if null_prior is not None else 1.0 / (len(priors) + 1)) + 1e-8)
    if normalizer:
        null_raw = (null_raw - float(normalizer.get("neglog_prior", {}).get("center", 0.0))) / max(float(normalizer.get("neglog_prior", {}).get("scale", 1.0)), 1e-8)
    null_score = null_raw + null_tau
    all_scores = np.concatenate([[null_score], scores])
    idx = int(np.argmin(all_scores))
    return {"selected": None if idx == 0 else idx - 1, "scores": scores, "null_score": null_score,
            "margin": float(np.partition(all_scores, 1)[1] - all_scores.min()) if len(all_scores) > 1 else float("inf")}
