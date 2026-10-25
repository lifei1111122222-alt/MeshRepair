from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from .mesh_io import load_obj, merge_labeled_parts, save_mesh_npz
from .boundary import extract_boundary_loops


def _leaf_parts(tree):
    out = []
    def rec(node):
        children = node.get("children", []) if isinstance(node, dict) else []
        if children:
            for c in children: rec(c)
        elif node.get("objs"):
            out.append((node.get("name", "part"), node["objs"][0]))
    rec(tree)
    return out


def build_partnet_dataset(partnet_root: str | Path, output_dir: str | Path, categories=None, max_samples=30,
                          seed=0, min_boundary=3, split=(0.7, 0.15, 0.15),
                          exclude_model_ids=None, max_per_category=None):
    rng = np.random.default_rng(seed); root, out = Path(partnet_root), Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    records, skipped = [], []
    # Sort before shuffling so the same seed gives the same sample order on
    # filesystems whose directory enumeration order is not stable.
    ids = sorted((p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists()), key=lambda p: p.name)
    rng.shuffle(ids)
    counts = {}
    excluded = set(str(x) for x in (exclude_model_ids or []))
    for model_dir in ids:
        if max_samples is not None and len(records) >= max_samples: break
        if model_dir.name in excluded: continue
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8")); cat = meta.get("model_cat", "unknown")
        if categories and cat not in categories: continue
        if max_samples is not None and counts.get(cat, 0) >= max_samples: continue
        if max_per_category is not None and counts.get(cat, 0) >= int(max_per_category): continue
        result_path = model_dir / "result.json"
        if not result_path.exists(): continue
        tree = json.loads(result_path.read_text(encoding="utf-8")); leaves = _leaf_parts(tree[0] if isinstance(tree, list) else tree)
        if len(leaves) < 2: continue
        remove_name, remove_obj = leaves[int(rng.integers(len(leaves)))]
        parts = []
        for li, (_, obj) in enumerate(leaves):
            p = model_dir / "objs" / f"{obj}.obj"
            if not p.exists(): continue
            v, f = load_obj(p); parts.append((v, f, li))
        if len(parts) < 2: continue
        v_all, f_all, labels = merge_labeled_parts(parts)
        remove_label = next((li for li, (_, obj) in enumerate(leaves) if obj == remove_obj), None)
        if remove_label is None: continue
        keep = labels != remove_label; damaged_faces = f_all[keep]
        used = np.unique(damaged_faces); remap = -np.ones(len(v_all), dtype=np.int64); remap[used] = np.arange(len(used))
        vd, fd = v_all[used], remap[damaged_faces]
        removed_vertices = v_all[np.unique(f_all[labels == remove_label])]
        bmin, bmax = removed_vertices.min(0), removed_vertices.max(0); cue = np.concatenate([(bmin + bmax) / 2, [np.linalg.norm(bmax - bmin) + 1e-8]])
        loops = extract_boundary_loops(vd, fd, min_boundary)
        if not loops: skipped.append({"model_id": model_dir.name, "reason": "no_boundary"}); continue
        # Independent interface annotation: maximize proximity of loop vertices to removed geometry.
        d = [np.min(((vd[l][:, None, :] - removed_vertices[None, :, :]) ** 2).sum(-1), axis=1).mean() for l in loops]
        target = int(np.argmin(d))
        sample_id = f"{model_dir.name}_{remove_name}"
        path = out / f"{sample_id}.npz"; save_mesh_npz(path, vd, fd, cue, loops[target], model_id=model_dir.name, category=cat, removed_part=remove_name, target_loop_index=target)
        records.append({"sample_id": sample_id, "path": str(path), "model_id": model_dir.name, "category": cat, "removed_part": remove_name, "target_loop_index": target, "num_loops": len(loops)})
        counts[cat] = counts.get(cat, 0) + 1
    # Object-level split: one model_id can never occur in multiple partitions.
    model_ids = sorted({r["model_id"] for r in records}); rng.shuffle(model_ids)
    n = len(model_ids); n_train = int(split[0] * n); n_val = int(split[1] * n)
    train_ids, val_ids = set(model_ids[:n_train]), set(model_ids[n_train:n_train + n_val])
    for r in records:
        r["split"] = "train" if r["model_id"] in train_ids else ("val" if r["model_id"] in val_ids else "test")
    result = {
        "protocol": {
            "seed": int(seed),
            "split_fractions": [float(x) for x in split],
            "object_level_split": True,
            "min_boundary_length": int(min_boundary),
            "excluded_model_count": len(excluded),
            "one_sample_per_model": True,
            "target_annotation": "nearest removed-part geometry; labels retained for training/evaluation only",
        },
        "records": records,
        "skipped": skipped,
        "splits": {"train": sum(r["split"] == "train" for r in records), "val": sum(r["split"] == "val" for r in records), "test": sum(r["split"] == "test" for r in records)},
    }
    (out / "dataset_index.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
