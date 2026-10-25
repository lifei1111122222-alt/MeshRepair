"""Train a candidate ranker on historical train labels, select on historical val.

Frozen CBE logits, cue distances and loop geometry form input-only features.
No prospective or controlled-surface target labels are read during training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tcmr.encoder import CueConditionedCyclicBoundaryEncoder


def loop_features(tokens: np.ndarray, scalars: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Candidate-relative rows from one damaged mesh (no target information)."""
    tok = np.asarray(tokens, np.float32)
    sc = np.asarray(scalars, np.float32)
    lg = np.asarray(logits, np.float32)
    n = len(tok)
    dist = tok[:, :, 11]
    rel = tok[:, :, :3]
    lengths = tok[:, :, 10]
    curv = tok[:, :, 9]
    normals = tok[:, :, 6:9]
    normal_concentration = np.linalg.norm(normals.mean(axis=1), axis=1)
    base = np.column_stack((
        np.log1p(np.maximum(dist.mean(axis=1), 0)),
        np.log1p(np.maximum(dist.min(axis=1), 0)),
        np.log1p(np.maximum(np.linalg.norm(rel.mean(axis=1), axis=1), 0)),
        np.log1p(np.maximum(sc[:, 0], 0)),
        sc[:, 1], sc[:, 2], sc[:, 3],
        np.log1p(np.maximum(sc[:, 4], 0)),
        lengths.mean(axis=1), lengths.std(axis=1),
        curv.mean(axis=1), curv.std(axis=1),
        normal_concentration,
        lg,
    )).astype(np.float32)
    cols = [base]
    # Relative values make a score depend on the competing holes, not only
    # each hole's absolute size or the scale of its source object.
    for j in (0, 1, 2, 3, 4, 13):
        col = base[:, j]
        order = np.argsort(col, kind="stable")
        rank = np.empty(n, dtype=np.float32)
        rank[order] = np.arange(n, dtype=np.float32) / max(n-1, 1)
        cols.append(np.column_stack((col-col.mean(), col-col.min(), col-col.max(), rank)))
    cols.append(np.full((n, 1), np.log1p(n), dtype=np.float32))
    result = np.concatenate(cols, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite ranker features")
    return result


def cbe_logits(tokens, scalars, checkpoint):
    torch.set_num_threads(2)
    model = CueConditionedCyclicBoundaryEncoder()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    chunks = []
    with torch.no_grad():
        for lo in range(0, len(tokens), 1024):
            hi = min(len(tokens), lo+1024)
            chunks.append(model.loop_logits(torch.from_numpy(tokens[lo:hi]),
                                             torch.from_numpy(scalars[lo:hi])).numpy())
    return np.concatenate(chunks)


def train(cache_path, checkpoint, output_dir, seed):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    with np.load(cache_path, allow_pickle=False) as z:
        tokens, scalars = z["tokens"], z["scalars"]
        offsets, targets, splits = z["offsets"], z["targets"], z["splits"]
    logits = cbe_logits(tokens, scalars, checkpoint)
    x_train, y_train, w_train, validation = [], [], [], []
    for i in range(len(targets)):
        a, b = int(offsets[i]), int(offsets[i+1])
        n = b-a
        if n < 2 or targets[i] < 0 or splits[i] not in (0, 1):
            continue
        x = loop_features(tokens[a:b], scalars[a:b], logits[a:b])
        if splits[i] == 0:
            x_train.append(x)
            y_train.extend((np.arange(n) == targets[i]).astype(np.int8))
            w_train.extend(np.where(np.arange(n) == targets[i], 0.5, 0.5/max(n-1,1)))
        else:
            validation.append((x, int(targets[i]), int(np.argmax(logits[a:b])),
                               int(np.argmin(tokens[a:b, :, 11].mean(axis=1)))))
    x_train = np.concatenate(x_train)
    y_train = np.asarray(y_train)
    w_train = np.asarray(w_train)
    trials = []
    for leaves in (15, 31):
        model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=leaves,
                                               learning_rate=0.06, min_samples_leaf=30,
                                               l2_regularization=1.0, random_state=seed)
        model.fit(x_train, y_train, sample_weight=w_train)
        ranker = [int(np.argmax(model.predict_proba(x)[:,1])) for x, _, _, _ in validation]
        accuracy = float(np.mean([p==t for p, (_,t,_,_) in zip(ranker, validation)]))
        trials.append((accuracy, leaves, model))
    trials.sort(key=lambda x: (-x[0], x[1]))
    best_acc, best_leaves, best = trials[0]
    joblib.dump(best, output_dir/"ranker.joblib", compress=3)
    summary = {"schema": "cue_geometry_ranker_v1", "training": "historical cache train multi-loop only",
               "n_train_candidates": len(x_train), "n_validation_multiloop": len(validation),
               "model_selection": "maximum historical validation multi-loop accuracy; smaller tree on tie",
               "trials": [{"max_leaf_nodes": leaves, "val_accuracy": acc} for acc, leaves, _ in trials],
               "selected_max_leaf_nodes": best_leaves, "validation_accuracy": best_acc,
               "random_seed": seed,
               "validation_cbe_accuracy": float(np.mean([c==t for _,t,c,_ in validation])),
               "validation_cue_nearest_accuracy": float(np.mean([c==t for _,t,_,c in validation])),
               "prospective_or_controlled_labels_used_for_training": False,
               "checkpoint": str(checkpoint.resolve()), "feature_dim": x_train.shape[1]}
    (output_dir/"manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT/"data/partnet_tcmr_90pct/selector_features_cache.npz")
    ap.add_argument("--checkpoint", type=Path, default=ROOT/"models/cbe.pt")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()
    train(args.cache, args.checkpoint, args.output_dir, args.seed)
