"""Precompute selector features once for a dataset index.

The cache contains only damaged-mesh/cue-derived loop features and the split
metadata needed by the selector trainer.  Variable numbers of boundary loops
are packed with offsets, so one compressed file is sufficient for large
PartNet runs and no mesh needs to be decoded during every training epoch.
Target indices are retained for train/validation supervision and audit; the
training loader never uses test targets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcmr.boundary import extract_boundary_loops, vertex_normals
from tcmr.evaluation import target_index_from_loop
from tcmr.features import construct_loop_features
from tcmr.mesh_io import load_mesh_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="dataset_index.json")
    parser.add_argument("--output", required=True, help="single packed .npz cache")
    parser.add_argument("--q", type=int, default=64, help="resampled boundary length")
    args = parser.parse_args()

    index_path = Path(args.index)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    token_rows: list[np.ndarray] = []
    scalar_rows: list[np.ndarray] = []
    offsets = [0]
    targets: list[int] = []
    splits: list[int] = []
    sample_ids: list[str] = []
    model_ids: list[str] = []
    categories: list[str] = []
    kept_records: list[dict] = []
    split_code = {"train": 0, "val": 1, "test": 2}

    for rec in index.get("records", []):
        try:
            data = load_mesh_npz(rec["path"])
            loops = extract_boundary_loops(data["vertices"], data["faces"])
            if not loops:
                continue
            normals = vertex_normals(data["vertices"], data["faces"])
            features = [
                construct_loop_features(
                    data["vertices"], data["faces"], loop, data["cue"],
                    q=args.q, cached_normals=normals,
                )
                for loop in loops
            ]
            tokens = np.stack([item[0] for item in features]).astype(np.float32)
            scalars = np.stack([item[1]["scalars"] for item in features]).astype(np.float32)
            target = target_index_from_loop(loops, data.get("target_loop"))
        except Exception as exc:  # keep a single bad mesh from aborting a large run
            print(f"skip {rec.get('sample_id', rec.get('path'))}: {exc}", file=sys.stderr)
            continue

        token_rows.append(tokens)
        scalar_rows.append(scalars)
        offsets.append(offsets[-1] + len(features))
        targets.append(-1 if target is None else int(target))
        splits.append(split_code.get(rec.get("split", "train"), 0))
        sample_ids.append(str(rec.get("sample_id", Path(rec["path"]).stem)))
        model_ids.append(str(rec.get("model_id", "")))
        categories.append(str(rec.get("category", "")))
        kept_records.append(rec)

    if not token_rows:
        raise RuntimeError("No valid records found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        tokens=np.concatenate(token_rows, axis=0),
        scalars=np.concatenate(scalar_rows, axis=0),
        offsets=np.asarray(offsets, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int64),
        splits=np.asarray(splits, dtype=np.int8),
        sample_ids=np.asarray(sample_ids),
        model_ids=np.asarray(model_ids),
        categories=np.asarray(categories),
        index_path=np.asarray(str(index_path)),
        q=np.asarray(int(args.q), dtype=np.int32),
    )
    print(json.dumps({
        "records": len(token_rows),
        "loops": int(offsets[-1]),
        "train": int(np.sum(np.asarray(splits) == 0)),
        "val": int(np.sum(np.asarray(splits) == 1)),
        "test": int(np.sum(np.asarray(splits) == 2)),
        "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
